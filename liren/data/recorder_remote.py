import time
import sys
import os 
import tensorflow as tf
import atexit

from absl import app, flags, logging as absl_logging
from ml_collections import config_flags, ConfigDict
from tpu_utils import prevent_cross_region

from agentlace.data.rlds_writer import RLDSWriter
from agentlace.data.tf_agents_episode_buffer import EpisodicTFDataStore
from agentlace.trainer import TrainerServer

from liren.utils.trainer_bridge_common import (
    task_data_format,
    make_trainer_config,
)

FLAGS = flags.FLAGS

def main(_):

    tf.get_logger().setLevel("WARNING")
    absl_logging.set_verbosity("WARNING")

    start_time = time.time()

    data_spec = task_data_format()
    data_save_dir = FLAGS.data_save_dir
    # existing_folders = [0] + [int(folder.split('.')[-1]) for folder in os.listdir(data_dir)]
    # latest_version = max(existing_folders)

    # version= f"0.0.{1 + latest_version}"
    # datastore_path = f"{data_dir}/{version}"
    # os.makedirs(datastore_path)
    version = "0.0.1"
    datastore_path = tf.io.gfile.join(data_save_dir, version)
    if not tf.io.gfile.exists(datastore_path):
        tf.io.gfile.makedirs(datastore_path)
    prevent_cross_region(datastore_path)

    writer = RLDSWriter(
        dataset_name="test",
        data_spec = data_spec,
        data_directory = datastore_path,
        version = version,
        max_episodes_per_file = 100,
    )
    atexit.register(writer.close) # save on exit 
    online_dataset_datastore = EpisodicTFDataStore(
        capacity=10000,
        data_spec= task_data_format(),
        rlds_logger = writer
    )
    print("Datastore set up")

    def request_callback(_type, _payload):
        if _type == "send-stats":
            print("Not Implemented!")
        elif _type == "get-model-config":
            return None
        else:
            raise NotImplementedError(f"Unknown request type {_type}")

    train_server = TrainerServer(
        config=make_trainer_config(),
        request_callback=request_callback,
    )
    train_server.register_data_store("online_data", online_dataset_datastore)
    train_server.start(threaded=True)

    while True:
        if FLAGS.max_time is not None and time.time() - start_time > FLAGS.max_time:
            print(f"Killing recorder after {time.time() - start_time} seconds.")
            sys.exit()


if __name__ == "__main__":
    import os

    flags.DEFINE_string("data_save_dir", "/home/lydia/data/create_data/deployment/trash", "Where to save collected data")
    flags.DEFINE_integer("max_time", 3000, "Interval between visualizations")

    app.run(main)