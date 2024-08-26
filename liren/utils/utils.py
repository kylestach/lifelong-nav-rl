import collections
import pprint
import time
from typing import Any, Dict, Sequence
import uuid
import tempfile
import os
from copy import copy
from socket import gethostname
import chex
import cloudpickle as pickle
import einops
import jax
import jax.numpy as jnp
import jaxlib
import numpy as np
import io

import tensorflow as tf

import absl.flags
from absl import logging
from ml_collections import ConfigDict
from ml_collections.config_flags import config_flags
from ml_collections.config_dict import config_dict

import wandb

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])

# BATCH MANIPULATION
def split_batch_pmap(batch, num_devices):
    for key in batch.keys():
        batch[key] = batch[key].reshape(
            (num_devices, -1, *batch[key].shape[1:]))
    return batch


def get_batch_pattern(*batch_dims):
    batch_str = " ".join([f"b{i}" for i in range(len(batch_dims))])
    batch_dict = {f"b{i}": batch_dims[i] for i in range(len(batch_dims))}
    return batch_str, batch_dict


def flatten(x: jax.Array, start_dim: int, end_dim: int) -> jax.Array:
    leading_dims = x.shape[:start_dim]
    trailing_dims = x.shape[end_dim:]
    return jnp.reshape(x, leading_dims + (-1,) + trailing_dims)


def unflatten(x: jax.Array, dim: int, new_shape: Sequence[int]) -> jax.Array:
    leading_dims = x.shape[:dim]
    trailing_dims = x.shape[dim + 1:]
    return jnp.reshape(x, leading_dims + tuple(new_shape) + trailing_dims)


def multi_vmap(function, num_vmap_dims: int):
    """
    Collapse all batch dims into dimension 0, apply a vmap'd function, and unflatten
    """

    @jax.jit
    def result(*args, **kwargs):
        arg_inputs = (args, kwargs)
        batch_dims = jnp.shape(jax.tree_util.tree_leaves(arg_inputs)[0])[
            :num_vmap_dims]
        chex.assert_tree_shape_prefix(arg_inputs, batch_dims)
        batch_str, batch_dict = get_batch_pattern(*batch_dims)

        def _flatten(x: jax.Array):
            return einops.rearrange(
                x, f"{batch_str} ... -> ({batch_str}) ...", **batch_dict
            )

        def _unflatten(x: jax.Array):
            return einops.rearrange(
                x, f"({batch_str}) ... -> {batch_str} ...", **batch_dict
            )

        flattened_args, flattened_kwargs = jax.tree_map(_flatten, arg_inputs)
        flattened_output = jax.vmap(function)(
            *flattened_args, **flattened_kwargs)
        return jax.tree_map(_unflatten, flattened_output)

    return result


# DICTIONARY MANIPULATION

def average_dict(some_dict):
    """
    Average a dictionary where each value is a list
    """
    new_dict = {}
    for key, value in some_dict.items():
        if isinstance(value, dict):
            new_dict[key] = average_dict(value)
        elif isinstance(value, jax.Array):
            new_dict[key] = np.mean(value.tolist())
        elif isinstance(value, list): ## NOT JUST LIST
            new_dict[key] = np.mean(value)
        else:
            new_dict[key] = value
    return new_dict

def average_dicts(some_dicts):
    """
    Average a list of dictionaries with the same keys
    """
    averaged = {}
    for key in some_dicts[0].keys():
        val_type = type(some_dicts[0][key])
        if val_type == dict:
            averaged[key] = average_dicts([i[key] for i in some_dicts])
        elif val_type == np.float64:  # they're numbers!
            averaged[key] = np.mean([i[key] for i in some_dicts])
        else:
            print("val type", val_type, "not recognized")
    return averaged


def print_dict_keys(d, indent=0):
    for key, value in d.items():
        print(" " * indent + f"{key}")
        if isinstance(value, dict):
            print_dict_keys(value, indent + 1)

def print_dict_types(my_dict, level = 0):
    for k in my_dict.keys():
        val = my_dict[k]
        print("\t" * level + k, f"has type {type(val)}",  f"{val.shape if isinstance(val, tf.Tensor) else ''}")
        if isinstance(val, dict):
            print_dict_types(val, level + 1)
    

def flatten_dict_list(idx, data_format, data):
    """Extracts out the idx element for every single key in data_format from data"""
    ret = {}
    for k, v in data_format.items():
        if type(v) is dict:
            ret[k] = flatten_dict_list(idx, v, data[k])
        else:
            ret[k] = data[k][idx]
    return ret


def update(d, u):
    for k, v in u.items():
        if isinstance(v, collections.abc.Mapping):
            d[k] = update(d.get(k, {}), v)
        else:
            d[k] = v
    return d


def dictionary_merge(*dicts):
    result = {}

    for d in dicts:
        result = update(result, d)

    return result


def flatten_for_wandb(info: Dict[str, Any]):
    flat_data = {}

    def _recursive_flatten(data, prefix="", depth=0):
        nonlocal flat_data
        for k, v in data.items():
            if isinstance(v, dict):
                sep = "/" if depth < 2 else "."
                _recursive_flatten(
                    v, prefix=f"{prefix}{k}{sep}", depth=depth + 1)
            else:
                flat_data[f"{prefix}{k}"] = v

    _recursive_flatten(info)

    return flat_data


# MISC 
def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize the image to be between 0 and 1
    """
    return (tf.cast(image, tf.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


# from dataset_transforms.py 
def get_relative_position(base_position, base_yaw, goal_position, is_negative):
    goal_dist_threshold = 0.2
    if base_yaw.ndim > 0:
        base_yaw = tf.squeeze(base_yaw, axis=-1)
    goal_position = (goal_position - base_position) / WAYPOINT_SPACING
    rotation_matrix = tf.stack(
        [
            tf.stack([tf.cos(base_yaw), tf.sin(base_yaw)], axis=0),
            tf.stack([-tf.sin(base_yaw), tf.cos(base_yaw)], axis=0),
        ],
        axis=0,
    )
    goal_vector = tf.matmul(rotation_matrix, goal_position[..., None])[..., 0]
    random_goal_vectors = tf.random.normal(
        goal_vector.shape, mean=0, stddev=10, dtype=goal_vector.dtype
    )

    goal_vector = tf.where(is_negative, random_goal_vectors, goal_vector)

    goal_vector_norm = tf.maximum(
        tf.norm(goal_vector, axis=-1, keepdims=True), goal_dist_threshold
    )
    goal_vector_magdir = tf.concat(
        [
            goal_vector / goal_vector_norm,
            1 / goal_vector_norm,
        ],
        axis=-1,
    )
    return goal_vector, goal_vector_magdir

# from goal_task.py
def _yaw(quat):
    return np.arctan2(
        2.0 * (quat[3] * quat[2] + quat[0] * quat[1]),
        1.0 - 2.0 * (quat[1] ** 2 + quat[2] ** 2),
    )


def pose_distance(
    start_pose: np.ndarray, # pose, quaternion 
    goal_pose: np.ndarray, # pose, quaternion 
    orientation_weight: float = 1.0,
):
    position = np.array(start_pose[:3])
    quaternion = np.array(start_pose[3:])
    goal_position = np.array(goal_pose[:3])
    goal_quaternion = np.array(goal_pose[3:])

    # Compute quaternion distance
    # q1 = quaternion / np.linalg.norm(quaternion, axis=-1, keepdims=True)
    # q2 = goal_quaternion / np.linalg.norm(goal_quaternion, axis=-1, keepdims=True)
    # d_quat = 2 * np.arccos(np.abs(np.sum(q1 * q2, axis=-1)))
    d_quat = 0
    # Compute position distance
    d_pos = np.linalg.norm(position - goal_position, axis=-1)
    return d_pos + orientation_weight * d_quat


class Timer(object):
    def __init__(self):
        self._time = None

    def __enter__(self):
        self._start_time = time.time()
        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        self._time = time.time() - self._start_time

    def __call__(self):
        return self._time


class WandBLogger(object):
    @staticmethod
    def get_default_config(updates=None):
        config = ConfigDict()
        config.mode = "online"
        config.prefix = "MultinavRL"
        config.project = "debug"
        config.output_dir = "/tmp/MultinavRL"
        config.random_delay = 0.0
        config.experiment_id = config_dict.placeholder(str)
        config.anonymous = config_dict.placeholder(str)
        config.entity = config_dict.placeholder(str)
        config.notes = config_dict.placeholder(str)

        if updates is not None:
            config.update(ConfigDict(updates).copy_and_resolve_references())
        return config

    def __init__(self, config, variant):
        self.config = self.get_default_config(config)

        if self.config.experiment_id is None:
            self.config.experiment_id = uuid.uuid4().hex

        if self.config.prefix != "":
            self.config.project = "{}--{}".format(
                self.config.prefix, self.config.project
            )

        if self.config.output_dir == "":
            self.config.output_dir = tempfile.mkdtemp()
        else:
            self.config.output_dir = os.path.join(
                self.config.output_dir, self.config.experiment_id
            )
            os.makedirs(self.config.output_dir, exist_ok=True)

        self._variant = copy(variant)

        if "hostname" not in self._variant:
            self._variant["hostname"] = gethostname()

        if self.config.random_delay > 0:
            time.sleep(np.random.uniform(0, self.config.random_delay))

        self.run = wandb.init(
            reinit=True,
            config=self._variant,
            project=self.config.project,
            dir=self.config.output_dir,
            id=self.config.experiment_id,
            anonymous=self.config.anonymous,
            entity=self.config.entity,
            notes=self.config.notes,
            settings=wandb.Settings(
                start_method="thread",
                _disable_stats=True,
            ),
            mode=self.config.mode,
        )

    def log(self, *args, **kwargs):
        self.run.log(*args, **kwargs)

    def save_pickle(self, obj, filename):
        with open(os.path.join(self.config.output_dir, filename), "wb") as fout:
            pickle.dump(obj, fout)
        # os.remove(os.path.join(self.config.output_dir, ))

    def tpu_save_pickle(self, obj, filepath, filename):
        import tensorflow as tf

        file_path = os.path.join(filepath, filename)
        tf.io.gfile.makedirs(filepath)

        # Pickle to a bytes object first
        buffer = io.BytesIO()
        pickle.dump(obj, buffer)
        buffer.seek(0)  # ensure the read pointer is at the start of the buffer

        # Then write that out using tf.io.gfile
        with tf.io.gfile.GFile(file_path, "wb") as fout:
            fout.write(buffer.read())

    @property
    def experiment_id(self):
        return self.config.experiment_id

    @property
    def variant(self):
        return self.config.variant

    @property
    def output_dir(self):
        return self.config.output_dir


def define_flags_with_default(**kwargs):
    for key, val in kwargs.items():
        if isinstance(val, ConfigDict):
            config_flags.DEFINE_config_dict(key, val)
        elif isinstance(val, bool):
            # Note that True and False are instances of int.
            absl.flags.DEFINE_bool(key, val, "automatically defined flag")
        elif isinstance(val, int):
            absl.flags.DEFINE_integer(key, val, "automatically defined flag")
        elif isinstance(val, float):
            absl.flags.DEFINE_float(key, val, "automatically defined flag")
        elif isinstance(val, str):
            absl.flags.DEFINE_string(key, val, "automatically defined flag")
        else:
            breakpoint()
            raise ValueError("Incorrect value type")
    return kwargs


def print_flags(flags, flags_def):
    logging.info(
        "Running training with hyperparameters: \n{}".format(
            pprint.pformat(
                [
                    "{}: {}".format(key, val)
                    for key, val in get_user_flags(flags, flags_def).items()
                ]
            )
        )
    )


def get_user_flags(flags, flags_def):
    output = {}
    for key in flags_def:
        val = getattr(flags, key)
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=key))
        else:
            output[key] = val

    return output


def flatten_config_dict(config, prefix=None):
    output = {}
    for key, val in config.items():
        if prefix is not None:
            next_prefix = "{}.{}".format(prefix, key)
        else:
            next_prefix = key
        if isinstance(val, ConfigDict):
            output.update(flatten_config_dict(val, prefix=next_prefix))
        else:
            output[next_prefix] = val
    return output


def prefix_metrics(metrics, prefix):
    return {"{}/{}".format(prefix, key): value for key, value in metrics.items()}


def average_metrics(metrics):
    averaged = {}
    for key in metrics[0].keys():
        averaged[key] = np.mean([m[key] for m in metrics])
    return averaged