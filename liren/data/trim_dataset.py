from liren.utils.trainer_bridge_common import (
    task_data_format,
    observation_format
)
from agentlace.data.rlds_writer import RLDSWriter
from agentlace.data.tf_agents_episode_buffer import EpisodicTFDataStore
from collections import deque
import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from liren.data.load_data import load_dataset, setup_datasets, dataset_postprocess, DATASETS, dataset_preprocess

from liren.utils.utils import flatten_dict_list
import tensorflow_datasets as tfds
import dlimp
from dlimp.dataset import DLataset
from functools import partial
import tensorflow as tf
# No visible GPU! don't let the dataset clog up our GPU space! 
tf.config.set_visible_devices([], 'GPU')
os.environ['JAX_PLATFORMS'] = 'cuda'

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])


def main(dataset_name,
         dataset_dir,
         save_dir,
         version,
         count,
         shard_max,
         min_length,
         ):

    dataset_builder = tfds.builder(dataset_name, data_dir=dataset_dir)
    dataset = (
        DLataset.from_rlds(dataset_builder)
        .filter(lambda traj: traj["_len"][0] >= min_length)
    ).ignore_errors(log_warning=True, name=f"ignore_errors")

    # COPY SMALL BIT: CORRECTLY FLATTENED
    datastore_path = os.path.join(save_dir, version)
    if not os.path.exists(datastore_path):
        print(f"Did not find directory at {datastore_path}, creating." )
        os.makedirs(datastore_path)

    dataset_small = dataset.take(count)
    data_spec = task_data_format()

    writer = RLDSWriter(
        dataset_name="test",
        data_spec=data_spec,
        data_directory=datastore_path,
        version=version,
        max_episodes_per_file=shard_max,
    )

    online_dataset_datastore = EpisodicTFDataStore(
        capacity=10000,
        data_spec=task_data_format(),
        rlds_logger=writer
    )

    data_spec = task_data_format()
    obs_spec = observation_format()

    for one_step in dataset_small:
        traj_len = one_step["_len"][0]
        flattened = []

        # FLATTEN MANUALLY
        for i in range(traj_len):
            flattened.append(flatten_dict_list(i, data_spec, one_step))
        
        # INSERT IN ORDER
        for i in range(traj_len):
            online_dataset_datastore.insert(flattened[i])

    writer.close()


if __name__ == "__main__":
    import argparse
    import sys
    import tqdm
    import yaml
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_name", "-n", required=True,
                        type=str, help="Name of the dataset, including version. Example format: biggest_loop:0.0.3")
    parser.add_argument("--dataset_dir", "-d", required=True,
                        type=str, help="Directory to load dataset from")
    parser.add_argument("--save_dir", "-s", required=True,
                        type=str, help="Where to save generated tfds dataset")
    parser.add_argument("--save_version", "-v", default="0.0.1",
                        type=str, help="Where to save generated tfds dataset")

    parser.add_argument("--count", "-c", default=10,
                        type=int, help="Number of trajectories to copy over")
    parser.add_argument("--shard_max", "-m", default=100,
                        type=int, help="Max number of trajectories to save per shard")
    parser.add_argument("--min_length", "-l", default=20,
                        type=int, help="Min number of steps in saved trajectory")

    args = parser.parse_args()
    main(
        args.dataset_name,
        args.dataset_dir,
        args.save_dir,
        args.save_version,
        args.count,
        args.shard_max,
        args.min_length,
    )
