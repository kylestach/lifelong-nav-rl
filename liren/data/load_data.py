from tensorflow_graphics.geometry.transformation.euler import from_quaternion
import liren.data.dataset_transforms as mnav_transforms
import copy
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

from matplotlib.pyplot import yscale
import tensorflow as tf
import json
import tensorflow_datasets as tfds
import dlimp
from dlimp.dataset import DLataset
from dlimp.transforms.common import selective_tree_map
import sys
from functools import partial
import numpy as np
from scipy.stats import norm

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])
CRASH_PENALTY = -5


# discretization constants
X_PARTITIONS = 3
ANGLE_PARTITIONS = 9
X_BOUNDS = [
    norm.ppf(q=i / X_PARTITIONS, loc=0, scale=1 / 3) for i in range(1, X_PARTITIONS)
]
ANGLE_BOUNDS = [
    norm.ppf(q=i / ANGLE_PARTITIONS, loc=0, scale=1 / 3)
    for i in range(1, ANGLE_PARTITIONS)
]


DATASETS = {
    "recon": {
        "end_is_crash": True,
        "skip_0_x": False,
        "waypoint_spacing": 0.25,
        "angle_scale": np.pi / 9,
        "action_key": "action_angle",
        "dataset_size": 599072,
        "has_goal": False,
        "x_offset": -1,
    },
    "sacson": {
        "end_is_crash": True,
        "skip_0_x": False,
        "waypoint_spacing": 0.255,
        "angle_scale": 1,
        "action_key": "action_angle",
        "dataset_size": 238103,
        "has_goal": False,
        "x_offset": -1,
    },
    "scand": {
        "end_is_crash": False,
        "skip_0_x": False,
        "waypoint_spacing": 0.38,
        "angle_scale": np.pi / 15,
        "action_key": "action_angle",
        "dataset_size": 31970,
        "has_goal": False,
        "x_offset": -1,
    },
    "seattle": {
        "end_is_crash": False,
        "skip_0_x": False,
        "waypoint_spacing": 0.35,
        "angle_scale": np.pi / 15,
        "action_key": "action_angle",
        "dataset_size": 7439,
        "has_goal": False,
        "x_offset": -1,
    },
    "tartan_drive": {
        "end_is_crash": True,
        "skip_0_x": False,
        "waypoint_spacing": 0.72,
        "angle_scale": np.pi / 24,
        "action_key": "action_angle",
        "dataset_size": 17239,
        "has_goal": False,
        "x_offset": -1,
    },
    "cory_hall": {
        "end_is_crash": True,
        "skip_0_x": False,
        "waypoint_spacing": 0.06,
        "angle_scale": np.pi / 15,
        "action_key": "action_angle",
        "dataset_size": 148680,
        "has_goal": False,
        "x_offset": -1,
    },
    "go_stanford": {
        "end_is_crash": True,
        "skip_0_x": False,
        "waypoint_spacing": 0.12,
        "angle_scale": np.pi / 6,
        "action_key": "action_angle",
        "dataset_size": 194429,
        "has_goal": False,
        "x_offset": -1,
    },
}

def skip_last(trajectory: dict, amount: int):
    return tf.nest.map_structure(lambda x: x[: -1 * amount], trajectory)


def skip_by(trajectory: dict, amount: int):

    start_idx = tf.random.uniform(
        shape=(), minval=0, maxval=amount, dtype=tf.int32)

    traj_length = tf.shape(trajectory["_len"])[0] - start_idx
    target_length = (traj_length + amount - 1) // amount
    pad_length = amount * target_length - traj_length

    def reduce(x):
        result = tf.concat(
            [
                x[start_idx:],
                tf.repeat(x[-1:], pad_length, axis=0),
            ],
            axis=0,
        )

        result = tf.reshape(result, (target_length, amount, *result.shape[1:]))
        if result.dtype == tf.bool:
            result = tf.reduce_any(result, axis=1)
        else:
            result = result[:, 0]
        return result

    trajectory = tf.nest.map_structure(reduce, trajectory)
    trajectory["_len"] = tf.fill((target_length,), target_length)
    trajectory["_frame_index"] = tf.range(target_length, dtype=tf.int32)
    return trajectory


def image_to_01(
    x: Dict[str, Any], match: Union[str, Sequence[str]] = "image"
) -> Dict[str, Any]:
    if isinstance(match, str):
        match = [match]

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match]),
        lambda image: tf.cast(image, tf.float32) / 255.0,
    )


def normalize_image_imagenet(
    x: Dict[str, Any],
    match: Union[str, Sequence[str]] = "image",
) -> Dict[str, Any]:
    """
    Can operate on nested dicts. Normalizes any leaves that have `match` anywhere in their path.
    Takes uint8 images as input and returns float images in range [0, 1].
    """
    if isinstance(match, str):
        match = [match]

    def normalize_image(image: tf.Tensor) -> tf.Tensor:
        """
        Normalize the image to have mean 0 and std 1.
        """
        return (image - IMAGENET_MEAN) / IMAGENET_STD

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match]),
        normalize_image,
    )


def normalize_actions(
    trajectory,
    *,
    scale: float,
    x_offset: float,
    action_key: str = "action",
    num_spatial_actions: int = 2,
):
    action = trajectory[action_key]

    action_offset = tf.constant(
        [x_offset] + [0] * (action.shape[-1] - 1), dtype=action.dtype
    )

    trajectory[action_key] = (
        tf.concat(
            [
                action[..., :num_spatial_actions] /
                scale,  # scaling x to be around 0
                action[..., num_spatial_actions:],
            ],
            axis=-1,
        )
        + action_offset
    )
    return trajectory


def zero_x_action(trajectory, action_key: str = "action"):
    action = trajectory[action_key]
    num_spatial_actions = 1 if action.shape[-1] == 2 else 2

    trajectory[action_key] = tf.concat(
        [
            tf.constant([0], shape=action[..., :1].shape, dtype=action.dtype),
            action[..., num_spatial_actions:],
        ],
        axis=-1,
    )
    return trajectory


def fix_dtypes(sample: Dict[str, Any]) -> Dict[str, Any]:
    def _recursive_remove_str(d: dict):
        result = {}
        for k, v in d.items():
            if isinstance(v, dict):
                result[k] = _recursive_remove_str(v)
            elif tf.is_tensor(v) and v.dtype == tf.string:
                continue
            else:
                result[k] = v

        return result

    def _cast_types(x: tf.Tensor):
        if isinstance(x, tf.Tensor) and x.dtype == tf.float64:
            return tf.cast(x, tf.float32)
        elif isinstance(x, (np.int32,np.float32)):
            return tf.cast(x, tf.float32)
        elif x.dtype == tf.bool:
            return tf.cast(x, tf.float32)
        else:
            return x

    return tf.nest.map_structure(_cast_types, _recursive_remove_str(sample))


# helper functions for loading discrete dataset
def smallest_index(arr, input):
    right_ans = -1
    found = False
    for i, val in enumerate(arr):
        if input < val and not found: 
            right_ans = i 
            found = True
    if right_ans == -1 and not found:
        right_ans = len(arr)
    return right_ans  # last one!


def discretize(sample):
    x_loc = smallest_index(X_BOUNDS, sample["actions"][0])
    a_loc = smallest_index(ANGLE_BOUNDS, sample["actions"][1])
    disc_index = x_loc + X_PARTITIONS * a_loc
    sample["actions"] = disc_index
    return sample


def load_dataset(
    dataset_name: str,
    data_dir: str,
    discount: float,
    action_type: str,
    min_length: Optional[int] = None,
    skip_crash=False,
    discrete=False,
    truncate_goal=False,
    negative_probability=0.25,
    reward_type="sparse",
    split=None,
    waypoint_num_skip=4,
    num_frame_skip=1,
    history_len=None,
    prioritize_space=False,
):
    dataset_builder = tfds.builder(
        f"{dataset_name}",
        data_dir=data_dir,
    )

    if split:
        dataset = DLataset.from_rlds(dataset_builder, split=split)
    else:
        dataset = DLataset.from_rlds(dataset_builder)
    dataset = dataset.ignore_errors(
        log_warning=True, name=f"ignore_errors_{dataset_name}").repeat()

    dataset_config = copy.deepcopy(DATASETS[dataset_name])
    size = dataset_config.pop("dataset_size", None)
    return dataset_preprocess(
        dataset,
        assign_goal=True,
        discount=discount,
        min_length=min_length,
        skip_crash=skip_crash,
        discrete=discrete,
        truncate_goal=truncate_goal,
        negative_probability=negative_probability,
        reward_type=reward_type,
        action_type=action_type,
        waypoint_num_skip=waypoint_num_skip,
        num_frame_skip=num_frame_skip,
        history_len=history_len,
        prioritize_space=prioritize_space,
        **dataset_config,
    )


def select_action_key(trajectory, action_key: str):
    trajectory["action"] = trajectory[action_key]
    return trajectory


def squeeze_proprio(data):
    def _squeeze_proprio_obs(obs):
        if "proprio" in obs:
            obs["proprio"] = tf.squeeze(obs["proprio"], axis= 0)

    _squeeze_proprio_obs(data["observations"])
    _squeeze_proprio_obs(data["next_observations"])
    return data

def add_proprio(data, key: str):
    def _add_proprio_to_obs(obs):
        new_proprio = tf.cast(obs[key], tf.float32)
        if "proprio" in obs:
            obs["proprio"] = tf.cast(
                tf.concat([obs["proprio"], new_proprio], axis=-1), tf.float32)
        else:
            obs["proprio"] = new_proprio

    _add_proprio_to_obs(data["observations"])
    _add_proprio_to_obs(data["next_observations"])
    return data


def add_proprio_from_data(data, key: str):
    def _add_proprio_to_obs(obs):
        new_proprio = data[key]
        if "proprio" in obs:
            obs["proprio"] = tf.cast(
                tf.concat([obs["proprio"], new_proprio], axis=-1), tf.float32)
        else:
            obs["proprio"] = tf.cast(new_proprio, tf.float32)

    _add_proprio_to_obs(data["observations"])
    _add_proprio_to_obs(data["next_observations"])
    return data


def map_transforms(transforms):
    def _inner(x):
        for transform in transforms:
            x = transform(x)
        return x

    return _inner


def normalize_y_actions(trajectory, *, scale: float):
    action = trajectory["action"]
    # Divide action[..., 1] by scale
    trajectory["action"] = tf.concat(
        [
            action[..., :1],
            action[..., 1:2] / scale,
            action[..., 2:],
        ],
        axis=-1,
    )
    return trajectory


def dataset_preprocess(
    dataset: DLataset,
    *,
    waypoint_spacing: float,
    x_offset: float,
    angle_scale: float,
    assign_goal: bool,
    end_is_crash: bool,
    discount: float,
    negative_probability=0.25,
    min_length: Optional[int] = None,
    skip_crash=False,
    discrete=False,
    skip_0_x=False,
    truncate_goal=False,
    action_key="action_angle",  # "action" for ros_2
    has_goal=False,
    reward_type="sparse",
    action_type: Literal["twist", "waypoint"],
    y_scale=1,
    waypoint_num_skip=4,
    num_frame_skip=1,
    relabel_probability=0.5,  # only relevant when has_goals
    history_len=None,
    prioritize_space=True,  # decode AFTER shuffle buffers
) -> DLataset:
    # only keep trajectories that are long enough
    if action_type == "waypoint":
        min_length = max(min_length or 2, waypoint_num_skip + 1)

    if min_length is None:
        min_length = 1

    dataset: DLataset = dataset.filter(
        lambda x: tf.shape(x["_len"])[0] > num_frame_skip *
        min_length 
    )

    if action_type == "twist":
        action_transforms = [
            partial(select_action_key, action_key=action_key),
            partial(
                normalize_actions,
                scale=waypoint_spacing,
                x_offset=x_offset,
                num_spatial_actions=1,
            ),
            partial(mnav_transforms.normalize_angles, scale=angle_scale),
            partial(mnav_transforms.remove_y_action),
        ]
    else:
        action_transforms = [
            partial(mnav_transforms.waypoint_actions,
                    num_skip=waypoint_num_skip),
            partial(
                normalize_actions,
                scale=waypoint_spacing * waypoint_num_skip,
                x_offset=x_offset,
                num_spatial_actions=2,
            ),
        ]

    if history_len:
        if prioritize_space:
            history_dep_transforms = []
        else:  # move up decoding, multiplying from postprocessing function to not redo work.
            history_dep_transforms = [
                partial(mnav_transforms.batch_decode_images, match=[
                        "image"]),  
                partial(mnav_transforms.batch_duplicate_channels,
                        match=["image"]),
            ]
    else:
        history_dep_transforms = []
    # Make sure observations are of the proper format (x y position, jpg image, yaw, crash)
    dataset = dataset.map(
        map_transforms(
            action_transforms
            +
            [
                partial(mnav_transforms.fix_obs_type, has_goal=has_goal),
                partial(skip_by, amount=num_frame_skip),
            ] + history_dep_transforms + [

                partial(
                    mnav_transforms.relabel,
                    end_is_crash=end_is_crash,
                    has_goal=has_goal,
                    assign_goal=assign_goal,
                    discount=discount,
                    truncate_goal=truncate_goal,
                    # we want goal distance to be picked by real time not skipped time
                    goal_sample_steps=20/num_frame_skip,
                    crash_penalty=(
                        -1 / (1 - discount) if reward_type == "sparse" else 0
                    )
                    + CRASH_PENALTY,
                    recompute_original_rewards=True,
                    reward_type=reward_type,
                    waypoint_spacing=waypoint_spacing,
                    relabel_probability=relabel_probability,
                    history_len=history_len,
                    prioritize_space=prioritize_space,
                ),
                mnav_transforms.prepare_flatten_trajectory,
            ]
        ),
    )
    # Flatten the dataset into individual frames
    # Skip the RL frames with current observation crash (because of relabeling)
    dataset = dataset.flatten(num_parallel_calls=None)
    # Skip samples that are not moving
    if skip_0_x:
        dataset = dataset.filter(
            # not moving forward enough - skip!
            lambda x: x["actions"][0] > 0.00001
        )

    # Resample negatives
    if negative_probability > 0:
        dataset = dataset.shuffle(1000)  
        dataset = dataset.batch(10)  
        dataset = dataset.map(
            partial(
                mnav_transforms.resample_negatives,
                negative_probability=negative_probability,
                crash_penalty=-1 / (1 - discount) + CRASH_PENALTY,
                discount=discount,
                reward_type=reward_type,
            ),
        )
        dataset = dataset.unbatch()

    # Housekeeping, move things around to get them in the right format for jaxrl
    dataset = dataset.map(
        map_transforms(
            [
                mnav_transforms.to_rl_format,
                partial(
                    mnav_transforms.make_positions_relative,
                    waypoint_spacing=waypoint_spacing,
                ),
                partial(add_proprio, key="goal_vector_magdir"),
                partial(add_proprio, key="prev_action"),
            ]
            + ([discretize] if discrete else [])
        ),
    )

    return dataset

def squeeze_goals(x: Dict[str, Any]) -> Dict[str, Any]: 
    x["goals"]["image"] = tf.squeeze(x["goals"]["image"], axis= 0)
    return x


def flip_left_right(x: Dict[str, Any]) -> Dict[str, Any]:
    should_flip = tf.random.uniform(shape=x["_frame_index"].shape) > 0.5

    flip1 = tf.where(should_flip, -1.0, 1.0)
    ones = tf.ones_like(should_flip, dtype=tf.float32)
    flip_position = tf.stack([ones, flip1], axis=-1)
    flip_proprio = tf.stack([ones, flip1, ones], axis=-1)
    flip_proprio_w_action = tf.stack([tf.stack([ones, flip1, ones, ones, flip1], axis=-1)])
    
    flip_proprio_use = flip_proprio_w_action

    if "proprio" in x["observations"]:
        x["observations"]["proprio"] = x["observations"]["proprio"] * tf.cast(
            flip_proprio_use, x["observations"]["proprio"].dtype
        )
        x["next_observations"]["proprio"] = x["next_observations"]["proprio"] * tf.cast(
            flip_proprio_use, x["next_observations"]["proprio"].dtype
        )
    x["observations"]["position"] = x["observations"]["position"] * tf.cast(
        flip_position, x["observations"]["position"].dtype
    )
    x["next_observations"]["position"] = x["next_observations"]["position"] * tf.cast(
        flip_position, x["observations"]["position"].dtype
    )
    x["actions"] = x["actions"] * tf.cast(flip_position, x["actions"].dtype)
    x["observations"]["image"] = mnav_transforms._broadcasted_where(
        should_flip,
        x["observations"]["image"][..., ::-1, :],
        x["observations"]["image"],
    )
    x["next_observations"]["image"] = mnav_transforms._broadcasted_where(
        should_flip,
        x["next_observations"]["image"][..., ::-1, :],
        x["next_observations"]["image"],
    )
    x["goals"]["image"] = mnav_transforms._broadcasted_where(
        should_flip,
        x["goals"]["image"][..., ::-1, :],
        x["goals"]["image"],
    )

    return x


def dataset_postprocess(dataset: DLataset, image_size: int, history: bool, buffer_size: int, prioritize_space=True) -> DLataset:
    from dlimp.transforms.frame_transforms import augment

    # dataset = dataset.shuffle(25000)
    dataset = dataset.shuffle(buffer_size)
    transforms_to_apply = []
    if history:
        if prioritize_space:
            history_dep_transforms = [
                partial(mnav_transforms.batch_decode_images, match=["image"]),
                partial(mnav_transforms.batch_duplicate_channels, match = ["image"]),
                squeeze_goals,
            ]
        else:
            history_dep_transforms = []
    else:
        history_dep_transforms = [
            partial(dlimp.transforms.decode_images, match=["image"]),
            partial(mnav_transforms.duplicate_channels, match=["image"]),
        ]

    dataset = dataset.map(
        map_transforms(
            history_dep_transforms + [
                partial(dlimp.transforms.resize_images, match=[
                        "image"], size=(image_size, image_size)),
                partial(image_to_01, match=["image"]),
                partial(
                    augment,
                    keys_identical=True,
                    traj_identical=False,
                    augment_kwargs={
                        "augment_order": [
                            "random_brightness",
                            "random_contrast",
                            "random_hue",
                        ],
                        "random_brightness": [0.1],
                        "random_contrast": [0.9, 1.1],
                        "random_hue": [0.1],
                    },
                ),
                flip_left_right,
                partial(normalize_image_imagenet, match=["image"]),
                fix_dtypes,
                squeeze_proprio, # when dim = 5 
            ]
        ),
    )

    return dataset


DATA_MIXES = {
    "gnm": [
        ("cory_hall", 1.0),
        ("recon", 1.0),
        ("sacson", 1.0),
        ("scand", 1.0),
        ("seattle", 1.0),
        ("tartan_drive", 1.0),
        ("go_stanford", 1.0),
    ],
    "indoor_only": [
        ("cory_hall", 1.0),
        ("sacson", 1.0),
    ], 
}

for step in range(0, 310, 10):
    DATA_MIXES[f"create_dep_{step}k"] = [(f"dep_{step}k", 1.0)]
    DATA_MIXES[f"create_fine_{step}k"] = [(f"fine_{step}k", 1.0)]


def setup_datasets(
    data_mix: str,
    data_dir: str,
    discount: float,
    skip_crash: bool = False,
    num_frame_skip=1,
    discrete=False,
    truncate_goal=False,
    validate=None,
    negative_probability=0.25,
    reward_type: Literal["sparse", "dense"] = "sparse",
    action_type: Literal["twist", "waypoint"] = "twist",
    image_size=64,
    history_len=None,
    prioritize_space=True,
    train_buffer_size=25000,
    val_buffer_size=1000,
) -> Tuple[DLataset, Dict[str, DLataset]]:
    dataset_names_and_weights = DATA_MIXES[data_mix]
    dataset_sizes = [
        DATASETS[name]["dataset_size"] for name, _ in dataset_names_and_weights
    ]
    dataset_names = [name for name, _ in dataset_names_and_weights]
    dataset_sample_weights = [
        size * weight
        for size, (_, weight) in zip(dataset_sizes, dataset_names_and_weights)
    ]
    dataset_sample_weights = [
        weight / sum(dataset_sample_weights) for weight in dataset_sample_weights
    ]

    if validate is None:
        train_split = "train"
    else:
        valid_pct = int(100.0 * validate)
        train_pct = 100 - valid_pct
        train_split = f"train[:{train_pct}%]"
        valid_split = f"train[{train_pct}%:]"

    train_datasets = [
        load_dataset(
            dataset_name=name,
            data_dir=data_dir,
            discount=discount,
            min_length=3,
            skip_crash=skip_crash,
            discrete=discrete,
            truncate_goal=truncate_goal,
            split=train_split,
            reward_type=reward_type,
            negative_probability=negative_probability,
            action_type=action_type,
            num_frame_skip=num_frame_skip,
            history_len=history_len,
            prioritize_space=prioritize_space,
        )
        for name in dataset_names
    ]

    train_dataset = DLataset.sample_from_datasets(
        train_datasets, dataset_sample_weights
    )
    train_dataset = dataset_postprocess(
        train_dataset,
        image_size=image_size,
        history=history_len,
        buffer_size=train_buffer_size,
        prioritize_space=prioritize_space
    )

    if validate is None:
        valid_datasets = {}
    else:
        valid_datasets = {
            dataset: dataset_postprocess(
                load_dataset(
                    dataset_name=dataset,
                    data_dir=data_dir,
                    discount=discount,
                    min_length=3,
                    skip_crash=skip_crash,
                    discrete=discrete,
                    truncate_goal=truncate_goal,
                    split=valid_split,
                    reward_type=reward_type,
                    action_type=action_type,
                    num_frame_skip=num_frame_skip,
                    history_len=history_len,
                    prioritize_space=prioritize_space,
                ),
                image_size=image_size,
                history=history_len,
                buffer_size=val_buffer_size,
                prioritize_space=prioritize_space,
            )
            for dataset in dataset_names
        }

    return train_dataset, valid_datasets
