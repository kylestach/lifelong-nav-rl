import tensorflow_datasets as tfds
from dlimp.dataset import DLataset
import numpy as np
import tensorflow as tf
import cv2


def main(dataset,
         data_dir,
         save_dir,
         step_size):
    
    dataset_builder = tfds.builder(dataset, data_dir=data_dir)
    dataset = (
        DLataset.from_rlds(dataset_builder)
        .filter(lambda traj: traj["_len"][0] >= 30)
    )

    # first trajectory is the long goal loop
    dataset_iter = iter(dataset)
    one_step = next(dataset_iter)

    save_data = {}
    save_data["data/position"] = one_step["observation"]["position"][::step_size]
    save_data["data/orientation"] = one_step["observation"]["orientation"][::step_size]
    save_data["data/image"] = [tf.io.decode_image(pic, expand_animations=False)
                               for pic in one_step["observation"]["image"]][::step_size]
    save_data["data/image"] = [cv2.resize(np.array(img), dsize=(
        64, 64), interpolation=cv2.INTER_CUBIC) for img in save_data["data/image"]]

    np.savez(save_dir, **save_data)


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
    parser.add_argument("--save_name", "-s", required=True,
                        type=str, help="Where to save generated .npz")
    parser.add_argument("--step", "-p", default=1, type=int)

    args = parser.parse_args()
    main(
        args.dataset_name,
        args.dataset_dir,
        args.save_name,
        args.step,
    )
