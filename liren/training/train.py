import os
import sys
import time
import atexit

import numpy as np
import jax
import jax.numpy as jnp
import flax
import tensorflow as tf
# TensorFlow configuration
tf.config.set_visible_devices([], 'GPU')
tf.config.threading.set_inter_op_parallelism_threads(1)
import tqdm
import wandb

from typing import Mapping
from plotly import graph_objects as go

from jax.experimental.compilation_cache import compilation_cache

from agentlace.data.rlds_writer import RLDSWriter
from agentlace.data.tf_agents_episode_buffer import EpisodicTFDataStore
from agentlace.trainer import TrainerServer

from dlimp.dataset import DLataset

from liren.data.load_data import (
    dataset_postprocess,
    setup_datasets,
    dataset_preprocess,
)
from liren.model.utils.timer_utils import Timer
from liren.training.agent import Agent
from liren.utils.trainer_bridge_common import (
    task_data_format,
    make_trainer_config,
)
from liren.utils.utils import average_dict, average_dicts

from orbax.checkpoint import (
    CheckpointManager,
    CheckpointManagerOptions,
    PyTreeCheckpointer,
)

from absl import app, flags, logging as absl_logging
from ml_collections import config_flags, ConfigDict

from tpu_utils import prevent_cross_region



# JAX compilation cache initialization
compilation_cache.initialize_cache("/tmp/jax_cc_lydia")

# Device information
device_list = jax.devices()
num_devices = len(device_list)
FLAGS = flags.FLAGS

# Global variables  
WAYPOINT_SPACING = 0.25
ANGLE_SCALE = 1 # np.pi / 9
X_OFFSET = -1 
MIN_LENGTH = 3

end_stats = {
    "crash": 0,
    "reach": 0,
    "timeout": 0,
    "total": 0,
}

def main(_):

    global end_stats
    tf.get_logger().setLevel("WARNING")
    absl_logging.set_verbosity("WARNING")

    model_config: ConfigDict = FLAGS.model_config
    online_data_config: ConfigDict = FLAGS.online_data_config
    offline_data_config: ConfigDict = FLAGS.offline_data_config

    # Make sure data accesses are valid 
    if FLAGS.data_dir is not None:
        prevent_cross_region(FLAGS.data_dir)
    if FLAGS.checkpoint_load_dir is not None:
        prevent_cross_region(FLAGS.checkpoint_load_dir)
    if FLAGS.checkpoint_load_dir is not None:
        prevent_cross_region(FLAGS.checkpoint_load_dir)

    def train_step(batch, agent, update_actor):
        if "bc" in model_config.agent_name:
            return agent.update(batch, pmap_axis="num_devices")
        else:
            return agent.update(batch, pmap_axis="num_devices", networks_to_update={"actor", "critic"} if update_actor else {"critic"})
    pmap_train_step = jax.pmap(train_step, axis_name="num_devices", devices=device_list, static_broadcasted_argnums=(2,))
    
    def batch_for_devices(dataset: DLataset, total_batch_size: int, num_devices: int):
        dataset = dataset.batch(total_batch_size // num_devices, drop_remainder=True, num_parallel_calls=None)
        dataset = dataset.batch(num_devices, drop_remainder=True, num_parallel_calls=None)
        return dataset
    
    # Load data 
    if FLAGS.data_mix is not None:
        train_dataset, val_datasets = setup_datasets(
            FLAGS.data_mix,
            FLAGS.data_dir,
            **offline_data_config,
            discount = model_config.discount,
            history_len = model_config.agent_config.history_len,
            validate = model_config.validate,
            prioritize_space= FLAGS.prioritize_space,
            train_buffer_size=model_config.train_buffer_size,
        )

        val_datasets = {
            k: batch_for_devices(v, model_config.batch_size, num_devices)
            for k, v in val_datasets.items()
        }
    
    else:
        train_dataset = None
        val_datasets = None
        print("Not using any offline training data")
    print(f"Dataset set up, using {len(val_datasets.keys()) if val_datasets is not None else 0} val_datasets")

    # Load Agent
    agent_type = model_config.agent_name
    agent = Agent(model_config, FLAGS.seed)

    if FLAGS.checkpoint_load_dir is not None:
        agent.load_checkpoint(FLAGS.checkpoint_load_dir, FLAGS.checkpoint_load_step)
        print(f"Loaded step {FLAGS.checkpoint_load_step}")
    else:
        print("Training from scratch")
    
    agent.replicate()
    print("Agent set up!")

    # Saving & Metrics 
    if FLAGS.wandb_name is not None:
        datetime_string = time.strftime("%Y_%m_%d_%H_%M_%S")
        wandb.init(
            project=model_config.wandb_proj,
            name=f"{FLAGS.wandb_name}_{datetime_string}",
            config=model_config.to_dict() | online_data_config.to_dict() | offline_data_config.to_dict(),
            dir=FLAGS.wandb_dir,
        )
        print("Wandb set up")
        save_wandb = True
    else:
        save_wandb = False
        print("Not saving to weights & biases")

    if FLAGS.checkpoint_save_dir is not None:
        if save_wandb:
            checkpoint_manager = CheckpointManager(
                directory=tf.io.gfile.join(FLAGS.checkpoint_save_dir, wandb.run.name),
                checkpointers=PyTreeCheckpointer(),
                options=CheckpointManagerOptions(
                    save_interval_steps=FLAGS.checkpoint_interval, 
                    max_to_keep=FLAGS.checkpoint_max_keep,
                ),
            )
        else: 
            checkpoint_manager = CheckpointManager(
                directory=FLAGS.checkpoint_save_dir,
                checkpointers=PyTreeCheckpointer(),
                options=CheckpointManagerOptions(
                    save_interval_steps=FLAGS.checkpoint_interval, 
                    max_to_keep=FLAGS.checkpoint_max_keep,
                ),
            )
        print("Checkpoint manager set up")
    else:
        checkpoint_manager = None
        print("Not saving checkpoints")
    
    # Online Setup 
    if FLAGS.online_training:
        # Data Collection 
        data_spec = task_data_format()

        # Saving online data 
        if FLAGS.data_save_dir:
            version = FLAGS.data_save_version
            datastore_path = tf.io.gfile.join(FLAGS.data_save_dir, version)
            if not tf.io.gfile.exists(datastore_path):
                tf.io.gfile.makedirs(datastore_path)
            prevent_cross_region(datastore_path)

            writer = RLDSWriter(
                dataset_name="test",
                data_spec = data_spec,
                data_directory = datastore_path,
                version = version,
                max_episodes_per_file = FLAGS.data_max_eps_file,
            )
            atexit.register(writer.close)

            online_dataset_datastore = EpisodicTFDataStore(
                capacity=10000,
                data_spec= data_spec,
                rlds_logger = writer
            )

            print("Datastore set up")
        
        # Not saving online data 
        else:
            online_dataset_datastore = EpisodicTFDataStore(
                capacity=10000,
                data_spec = data_spec,
            )
            
            print("Online data not saved")

        # Server setup 
    
    
        def request_callback(_type, _payload):
            if _type == "send-stats":
                global end_stats

                for key, value in _payload.items(): # reach, timeout, crash
                    end_stats[key] += int(value)

                end_stats["total"] += 1

            elif _type == "get-model-config":
                return model_config # .agent_config.to_dict() 
            else:
                raise NotImplementedError(f"Unknown request type {_type}")

        train_server = TrainerServer(
            config=make_trainer_config(),
            request_callback=request_callback,
        )
        train_server.register_data_store("online_data", online_dataset_datastore)
        train_server.start(threaded=True)
        print("Train server started")

        # Get initial data 
        samples_to_wait_for = FLAGS.wait_data
        pbar = tqdm.tqdm(total=samples_to_wait_for, desc="Waiting for data")
        while online_dataset_datastore.size < samples_to_wait_for:
            pbar.update(online_dataset_datastore.size - pbar.n)
            # Make sure actor gets model params 
            train_server.publish_network(
                {
                    "params": jax.tree_map(
                        np.asarray, flax.jax_utils.unreplicate(agent.actor.state.params)
                    )
                }
            )
        print("Initial data collected")
    
        # # Pause server while model loads & first batch processed 
        # train_server.stop()

        online_dataset = dataset_preprocess(
            online_dataset_datastore.as_dataset().ignore_errors(log_warning=True, name="online_data"),
            waypoint_spacing = WAYPOINT_SPACING,
            x_offset = X_OFFSET,
            angle_scale = ANGLE_SCALE,
            assign_goal=True,
            end_is_crash = False,
            min_length= MIN_LENGTH,
            action_key="action",
            has_goal = True,
            discount = model_config.discount,
            history_len = model_config.agent_config.history_len,
            **online_data_config,
        )
        online_dataset = dataset_postprocess(online_dataset, image_size = model_config.image_size, 
            history = True, buffer_size = model_config.train_buffer_size, prioritize_space = FLAGS.prioritize_space)

    else:
        train_server = None
        online_dataset = None

    # Set up training data 
    if online_dataset is None:
        if train_dataset is None:
            ValueError("Cannot train model on no data")
        else:
            train_data = batch_for_devices(train_dataset, model_config.batch_size, num_devices)
            train_data = train_data.iterator()
            training_data_prefetch = flax.jax_utils.prefetch_to_device(train_data, 2)
            print("Training on offline data only")
    else:
        if train_dataset is None:
            print("Training on online data only")
        else:
            if FLAGS.taper_data:
                if FLAGS.taper_step is None:
                    ValueError("Cannot taper data mix when taper step is not set")
                else: 
                    dataset_mixes = [DLataset.sample_from_datasets(
                            [online_dataset, train_dataset], weights = [split, 1 - split])
                        for split in [0.1, 0.2, 0.3, 0.4, 0.5]]
                    dataset_mixes = [batch_for_devices(
                                        dataset, model_config.batch_size, num_devices
                                    ) for dataset in dataset_mixes] 
                    dataset_mixes = [dataset.iterator() for dataset in dataset_mixes]
                    curr_data_mix = 0
            else: 
                train_data = DLataset.sample_from_datasets(
                    [online_dataset, train_dataset], weights = [0.5, 0.5]
                )
                train_data = batch_for_devices(train_data, model_config.batch_size, num_devices)
                train_data = train_data.iterator()
                training_data_prefetch = flax.jax_utils.prefetch_to_device(train_data, 2)
           
            print("Training on blend of online / offline data")

    # Set up validation data
    if model_config.validate and val_datasets is not None:
        val_data_prefetch = {
            k: flax.jax_utils.prefetch_to_device(DLataset.iterator(v), 2)
            for k, v in val_datasets.items()
        }
        print("Validation prefetch setup")
    else:
        val_data_prefetch = None
        print("Not performing validation")


    timer = Timer()
    print("Starting Training Loop")
    pbar = tqdm.trange(model_config.train_steps, dynamic_ncols=True)
    for step in pbar:

        # Handle special online training processing 
        if FLAGS.online_training:
            # if step == 3:
            #     print("Starting server back up ...")
            #     train_server.start(threaded=True)
            #     print("... started!")
            pbar.set_postfix({"online data": online_dataset_datastore.size})

            # Update data mixture 
            if FLAGS.taper_data and step % FLAGS.taper_step == 0:
                with timer.context("dataset_switching"):
                    if curr_data_mix < len(dataset_mixes): # that's all we have!
                        training_data_prefetch = flax.jax_utils.prefetch_to_device(dataset_mixes[curr_data_mix], 2)
                        print("Got prefetch for data mix", curr_data_mix - 1)
                        curr_data_mix += 1
        # Get data batch 
        with timer.context("sample_buffer"):
            batch = None
            attempts = 0
            while batch is None and attempts < 12:
                try:
                    attempts += 1
                    batch = next(training_data_prefetch)
                except Exception as e:
                    print(f"Error processing batch at step {step}: {e}")
            if attempts >= 12:
                print("Could not successfully get next batch!")
                sys.exit()

        # Do training step 
        with timer.context("train_step"):
            agent.actor, update_info = pmap_train_step(batch, agent.actor, step % 3 == 0)  
            update_info = average_dict(update_info)  # compress from 8 dicts for 8 devices
            update_info = {f"train/{key}": value for key, value in update_info.items()}
            update_info["data_stats/reached_goal_frac"] = np.mean(batch["reached"])
            update_info["data_stats/original_goals"] = np.mean(batch["resample_type"] == 0)
            update_info["data_stats/positive_goals"] = np.mean(batch["resample_type"] == 1)
            update_info["data_stats/negative_goals"] = np.mean(batch["resample_type"] == 2)
            update_info["data_stats/crash_frac"] = np.mean(batch["crashed"])

        if save_wandb and step % FLAGS.wandb_interval == 0:
            all_updates = update_info
            action_info = {
                "action_0": wandb.Histogram(batch["actions"][..., 0].flatten()),
                "action_1": wandb.Histogram(batch["actions"][..., 1].flatten()),
            }
            all_updates = all_updates | action_info 

            if FLAGS.online_training:
                online_info = {
                    "online_size": online_dataset_datastore.size,
                    "online_reach_percent": end_stats["reach"] / (end_stats["total"] + 1e-3),
                    "online_crash_percent": end_stats["crash"] / (end_stats["total"] + 1e-3),
                    "online_timeout_percent": end_stats["timeout"] / (end_stats["total"] + 1e-3),
                    "online_reach_num": end_stats["reach"],
                    "online_crash_num": end_stats["crash"],
                    "online_timeout_num": end_stats["timeout"],
                }
                all_updates = all_updates | online_info
            
            timer_info = {f"timer/{k}": v for k, v in timer.get_average_times().items()}
            all_updates = all_updates | timer_info

            wandb.log(all_updates, step=step)

        with timer.context("val_step"):
            if val_data_prefetch is not None and step % model_config.val_steps == 0:  # validation!
                val_info_all = {}
                for single_dataset_name, single_val_data in val_data_prefetch.items():
                    val_metrics = []
                    for _ in range(100): 
                        val_batch = next(single_val_data)
                        _, update_info = pmap_train_step(val_batch, agent.actor, True)
                        update_info = average_dict(update_info)
                        update_info = {
                            f"val/{key}": value for key, value in update_info.items()
                        }
                        val_metrics.append(update_info)

                    val_info_all[single_dataset_name] = average_dicts(val_metrics)
                if save_wandb:
                    wandb.log(val_info_all, step=step)

        with timer.context("log_save"):
            if checkpoint_manager is not None and step % FLAGS.checkpoint_interval == 0:
                checkpoint_manager.save(
                    step,
                    items=jax.device_get(flax.jax_utils.unreplicate(agent.actor)),
                )
            
            if val_data_prefetch is not None and step % FLAGS.viz_interval == 0 and save_wandb:
                viz_info = {}
                for single_dataset_name, single_val_data in val_data_prefetch.items():
                    # Sample actions on train data
                    batch = next(single_val_data)
                    sampled_actions = jax.pmap(
                        lambda batch, actor: actor.sample_actions(
                            batch["observations"], batch["goals"], seed=actor.state.rng
                        ),
                        axis_name="num_devices",
                        devices=device_list,
                    )(batch, agent.actor)
                    mode_actions = jax.pmap(
                        lambda batch, actor: actor.sample_actions(
                            batch["observations"], batch["goals"], argmax=True
                        ),
                        axis_name="num_devices",
                        devices=device_list,
                    )(batch, agent.actor)
                    sampled_actions = jax.device_get(sampled_actions)
                    dataset_actions = jax.device_get(batch["actions"])

                    # Scatter plot
                    plot = go.Figure()
                    plot.add_trace(
                        go.Scatter(
                            x=sampled_actions[..., 0].flatten(),
                            y=sampled_actions[..., 1].flatten(),
                            mode="markers",
                            name="sampled",
                        )
                    )
                    plot.add_trace(
                        go.Scatter(
                            x=mode_actions[..., 0].flatten(),
                            y=mode_actions[..., 1].flatten(),
                            mode="markers",
                            name="modes",
                        )
                    )
                    plot.add_trace(
                        go.Scatter(
                            x=dataset_actions[..., 0].flatten(),
                            y=dataset_actions[..., 1].flatten(),
                            mode="markers",
                            name="dataset",
                        )
                    )
                    viz_info[f"viz/{single_dataset_name}"] = wandb.Plotly(plot)

                wandb.log(viz_info, step=step)

            # Update weights 
            if FLAGS.online_training and step % FLAGS.model_update_interval == 0:
                train_server.publish_network(
                    {
                        "params": jax.tree_map(
                            np.asarray, flax.jax_utils.unreplicate(agent.actor.state.params)
                        )
                    }
                )

    wandb.finish()


if __name__ == "__main__":
    import os

    # Model Config 
    config_flags.DEFINE_config_file(
        "model_config",
        os.path.join(os.path.dirname(__file__), "model_config.py:gc_bc"),
        "Configuration for the agent",
    )

    # Data Config 
    config_flags.DEFINE_config_file(
        "offline_data_config",
        os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/data_config.py:gnm')),
        "Configuration for the agent",
    )
    config_flags.DEFINE_config_file(
        "online_data_config",
        os.path.abspath(os.path.join(os.path.dirname( __file__ ), '..', 'data/data_config.py:create')),
        "Configuration for the agent",
    )
    flags.DEFINE_bool('prioritize_space', False, 'If true, process raw images.')

    # Misc 
    flags.DEFINE_integer("seed", 42, "Seed for training")
    
    # Saving 
    flags.DEFINE_string("checkpoint_save_dir", None, "Where to store checkpoints")
    flags.DEFINE_integer("checkpoint_interval", 2000, 
                         "Interval between checkpoint saves")
    flags.DEFINE_string("wandb_name", None, help="Name of run on W&B")
    flags.DEFINE_string("wandb_dir", "/tmp/wandb", 
                        "Where to store temporary W&B data to sync to cloud")
    flags.DEFINE_integer("wandb_interval", 10, "Interval between calls to wandb.log")
    flags.DEFINE_integer("viz_interval", 5000, "Interval between visualizations")

    # Starting Checkpoint 
    flags.DEFINE_string("checkpoint_load_dir", None, "Where to load checkpoints")
    flags.DEFINE_integer("checkpoint_load_step", None, "Which step to load checkpoints")
    flags.DEFINE_integer("checkpoint_max_keep", None, "Up to how many checkpoints to save")

    # Offline Data Info 
    flags.DEFINE_string("data_dir", None, help="Dataset directory")
    flags.DEFINE_string("data_mix", None, help="Dataset mix")
    
    # Online only
    flags.DEFINE_bool("online_training", False, 'If True, do online finetuning')
    flags.DEFINE_bool("taper_data", False, 'If True, gradually increase online data percent mix')
    flags.DEFINE_integer("taper_step", 2500, 'If True, gradually increase online data percent mix')
    flags.DEFINE_integer("wait_data", 1000, "how many data points to wait for")
    flags.DEFINE_string("data_save_dir", None, "Where to save collected data")
    flags.DEFINE_string("data_save_version", "0.0.1", "Where to save collected data")
    flags.DEFINE_integer("data_max_eps_file", 100, "Where to save collected data")
    flags.DEFINE_integer("model_update_interval", 25, "Where to save collected data")

    app.run(main)