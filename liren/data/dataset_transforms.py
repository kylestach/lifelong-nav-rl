from tkinter import NE
from typing import Dict, List, Union
import numpy as np
from sklearn.utils import resample
import tensorflow as tf

from dlimp.transforms.common import selective_tree_map
from functools import partial

from tensorflow_graphics.geometry.transformation.euler import from_quaternion
from typing import Any, Callable, Dict, Sequence, Tuple, Union

ORIGINAL_GOAL = 0
POSITIVE_GOAL = 1
NEGATIVE_GOAL = 2 


def duplicate_channels(x: Dict[str, Any], match: Union[str, Sequence[str]] = "image") -> Dict[str, Any]:
    """Duplicate 1-channel images to 3-channel images."""
    if isinstance(match, str):
        match = [match]

    def process_image(value):
        # Check if the value is a 1-channel image (dtype=tf.uint8 or tf.float32, shape=[height, width, 1])
        if value.dtype in [tf.uint8, tf.float32] and value.shape[-1] == 1:
            duplicated_image = tf.tile(value, multiples=[1, 1, 3])
            return duplicated_image
        else:
            return value

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match]),
        process_image
    )

def batch_duplicate_channels(x: Dict[str, Any], match: Union[str, Sequence[str]] = "image") -> Dict[str, Any]:
    """Duplicate 1-channel images to 3-channel images."""
    if isinstance(match, str):
        match = [match]

    def process_image(value):
        # Check if the value is a 1-channel image (dtype=tf.uint8 or tf.float32, shape=[height, width, 1])
        if value.dtype in [tf.uint8, tf.float32] and value.shape[-1] == 1:
            duplicated_image = tf.tile(value, multiples=[1, 1, 3])
            return duplicated_image
        else:
            return value

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match]),
        process_image
    )


def expand_dims(x: Dict[str, Any], match: Union[str, Sequence[str]] = "image") -> Dict[str, Any]:
    """Duplicate 1-channel images to 3-channel images."""
    if isinstance(match, str):
        match = [match]

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match]),
        partial(tf.expand_dims, axis = 1)
    )

def decode_images_tensor(
        x: Dict[str, Any], match: Union[str, Sequence[str]] = "image"
) -> Dict[str, Any]:
    """ When I have a tensor of shape (history_len , ) turn it into (history_len, [img shape], 3)"""
    if isinstance(match, str):
        match = [match]

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match])
        and value.dtype == tf.string,
        partial(
            tf.vectorized_map, partial(tf.io.decode_image, expand_animations=False)
        ),
    )
    

def batch_decode_images(
    x: Dict[str, Any], match: Union[str, Sequence[str]] = "image"
) -> Dict[str, Any]:
    """Can operate on nested dicts. Decodes any leaves that have `match` anywhere in their path."""
    if isinstance(match, str):
        match = [match]

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match])
        and value.dtype == tf.string,
        partial(
            tf.vectorized_map, partial(tf.io.decode_image, expand_animations=False)
        ),
    )

def decode_images(
    x: Dict[str, Any], match: Union[str, Sequence[str]] = "image"
) -> Dict[str, Any]:
    """Can operate on nested dicts. Decodes any leaves that have `match` anywhere in their path."""
    if isinstance(match, str):
        match = [match]

    return selective_tree_map(
        x,
        lambda keypath, value: any([s in keypath for s in match])
        and value.dtype == tf.string,
        # partial(tf.io.decode_image, expand_animations=False),
        partial(tf.io.decode_jpeg, channels=3),
    )

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


def normalize_angles(trajectory, scale, action_key: str = "action"):
    num_spatial_actions = 1 if trajectory[action_key].shape[-1] == 2 else 2

    angles = trajectory[action_key][..., num_spatial_actions:]
    between_pi = (
        tf.math.floormod(angles + np.pi, 2 * np.pi) - np.pi
    )

    trajectory[action_key] = tf.concat(
        [
            trajectory[action_key][
                ..., :num_spatial_actions
            ],
            tf.clip_by_value(
                between_pi / scale, -1, 1
            ),
        ],
        axis=-1,
    )
    return trajectory


def _broadcasted_where(condition, x, y):
    # Broadcast condition to (N, 1, 1, ...) based on shapes of x and y
    assert x.shape == y.shape
    while len(condition.shape) < len(x.shape):
        condition = condition[..., None]
    return tf.where(condition, x, y)


def compute_rewards(
    reward_type,
    did_crash,
    time_to_goal,
    crash_penalty,
    obs,
    goal_obs,
    discount,
    waypoint_spacing,
):
    reached = tf.cast(time_to_goal == 0.0, tf.float32)
    if reward_type == "sparse":
        reward = reached - 1.0 + crash_penalty * tf.cast(did_crash, tf.float32)
        mc_returns = -(1 - discount**time_to_goal) / (1 - discount)
    elif reward_type == "dense":
        # Reward is (distance to goal) - (next distance to goal)
        # Reward for last step is 0 (this will be deleted later anyway)
        distance_to_goal = tf.cast(
            tf.norm(obs["position"][:-1] - goal_obs["position"][:-1], axis=-1),
            tf.float32,
        )
        next_distance_to_goal = tf.cast(
            tf.norm(obs["position"][1:] - goal_obs["position"][:-1], axis=-1),
            tf.float32,
        )
        reward = (
            distance_to_goal - next_distance_to_goal
        ) / waypoint_spacing + crash_penalty * tf.cast(did_crash[:-1], tf.float32)

        mc_returns = tf.where(
            time_to_goal[:-1] == np.inf, 0.0, 0.5 / (1 - discount))

        reward = tf.where(
            time_to_goal == 0,
            tf.fill(tf.shape(time_to_goal), 1 / (1 - discount)),
            tf.concat([reward, tf.zeros_like(reward[-1:])], axis=0),
        )
        mc_returns = tf.concat(
            [mc_returns, tf.zeros_like(mc_returns[-1:])], axis=0)

    return reward, mc_returns


def compute_rewards_negatives(reward_type, did_crash, crash_penalty, discount):
    if reward_type == "sparse":
        reward = tf.cast(did_crash, tf.float32) * crash_penalty - 1
        mc_returns = -tf.fill(tf.shape(did_crash), value=1 / (1 - discount))
    elif reward_type == "dense":
        reward = crash_penalty * tf.cast(did_crash, tf.float32)
        mc_returns = tf.zeros_like(did_crash, dtype=tf.float32)

    return reward, mc_returns

def add_history(observations, history_len, traj_len):
    # converts traj_len tensor into history_len x traj_len tensor
    def tile_context(tensor, history_len, traj_len):
        row_indices = tf.reshape(
            tf.range(traj_len, dtype=tf.int32), (traj_len, 1))
        col_indices = tf.reshape(
            tf.range(history_len, dtype=tf.int32), (1, history_len))
        row_indices = tf.tile(row_indices, [1, history_len])
        col_indices = tf.tile(col_indices, [traj_len, 1])
        idxs = tf.where(col_indices < history_len - 1 - row_indices,
                        0, col_indices - (history_len - 1 - row_indices))
        return tf.gather(tensor, idxs)

    return selective_tree_map(
        observations,
        lambda keypath, value: any([s in keypath for s in ["image"]])
        and value.dtype == tf.uint8 or value.dtype == tf.string,
        partial(tile_context, history_len=history_len, traj_len=traj_len)
    )

# Primarily computes goals
def relabel(
    trajectory: dict,
    end_is_crash: bool,
    crash_penalty: float,
    has_goal: bool,
    recompute_original_rewards: bool,
    assign_goal: bool,
    goal_sample_steps: float,
    discount: float,
    truncate_goal: bool,
    waypoint_spacing: float,
    reward_type: str, 
    history_len = None,
    relabel_probability=0.15, 
    prioritize_space= False,
): 
    """
    Relabel the trajectory with computed goals and rewards
    """

    traj_len = trajectory["_len"][0]

    if "crash" in trajectory["observation"]:
        if "keepout" in trajectory["observation"]:
            did_crash = trajectory["observation"]["crash"] | trajectory["observation"]["keepout"]
            del trajectory["observation"]["keepout"]
        else:
            did_crash = trajectory["observation"]["crash"]

        del trajectory["observation"]["crash"]
    else:
        did_crash = trajectory["is_last"] & end_is_crash

    # Crash is true if this timestep crashes OR next timestep crashes
    did_crash = did_crash | tf.concat(
        [
            did_crash[1:],
            tf.zeros_like(did_crash[:1]),
        ],
        axis=0,
    )

    # First, compute the index for relabeled positives
    # Sample a goal offset
    goal_offset = tf.cast(
        tf.random.gamma(
            shape=tf.shape(trajectory["_frame_index"]),
            alpha=1,
            beta=1 / goal_sample_steps,
        ),
        tf.int32,
    )
    if truncate_goal:  # wrap around to frame index
        goal_idx = trajectory["_frame_index"] + tf.math.floormod(
            goal_offset, (traj_len - trajectory["_frame_index"])
        )
    else:  # wrap around to beginning
        goal_idx = tf.math.floormod(
            trajectory["_frame_index"] + goal_offset, traj_len)

    # Resampled (positive) goals
    resampled_positive_goal_obs: Dict[str, tf.Tensor] = tf.nest.map_structure(
        lambda obs: tf.gather(obs, goal_idx), trajectory["observation"]
    )
    time_to_reach_positives = tf.cast(
        goal_idx - trajectory["_frame_index"], tf.float32)
    time_to_reach_positives = tf.where(
        time_to_reach_positives >= 0.0, time_to_reach_positives, np.inf)
    positives_reached = time_to_reach_positives == 0.0

    reward_positives, mc_returns_positives = compute_rewards(
        reward_type,
        did_crash,
        time_to_reach_positives,
        crash_penalty,
        trajectory["observation"],
        resampled_positive_goal_obs,
        discount,
        waypoint_spacing,
    )

    if history_len and prioritize_space:
        goal_imgs = tf.expand_dims(resampled_positive_goal_obs["image"], 1)
    else:
        goal_imgs = resampled_positive_goal_obs["image"]
    
    postive_goals = {
        "observation": {
            "position": resampled_positive_goal_obs["position"],
            "yaw": resampled_positive_goal_obs["yaw"],
            "image": goal_imgs, 
        },
        "is_terminal": did_crash | positives_reached,
        "reached": positives_reached,
        "time_to_goal": time_to_reach_positives,
        "reward": reward_positives,
        "mc_returns": mc_returns_positives,
        "type": tf.fill(tf.shape(trajectory["_len"]), POSITIVE_GOAL),
    }

    # Compute partial resampling if we have a goal already
    if has_goal:
        # We can find out if the goal is eventually reached by looking at the last reward
        assert (
            "reached" in trajectory
        ), f"Expected `reached_goal` in trajectory, got {trajectory.keys()}"

        # Assume the goal is at the end
        original_reached = trajectory["reached"][-1]

        # The time until the next `reached=True` is given by...
        # [0 0 0 1 0 0 1 0 0 0]
        # [3 2 1 0 2 1 0 inf inf inf]
        # Compute as:

        if "time_to_goal" in trajectory:
            time_to_original_goal = trajectory["time_to_goal"]
        else:
            time_to_original_goal = tf.scan(
                lambda prev_time, reached: tf.where(
                    reached,
                    0.,
                    prev_time + 1,
                ),
                trajectory["reached"][::-1],
                initializer=np.inf,
            )[::-1]

        if recompute_original_rewards or "reward" not in trajectory:
            original_reward, original_mc_returns = compute_rewards(
                reward_type,
                did_crash,
                time_to_original_goal,
                crash_penalty,
                trajectory["observation"],
                trajectory["observation"]["goal"],
                discount,
                waypoint_spacing=waypoint_spacing,
            )
        else:
            original_reward = trajectory["reward"]
            original_mc_returns = trajectory["mc_returns"]

        if history_len and prioritize_space:
            goal_imgs = tf.expand_dims(trajectory["observation"]["goal"]["image"], 1)
        else:
            goal_imgs = trajectory["observation"]["goal"]["image"]

        original_goal = {
            "observation": {
                "position": trajectory["observation"]["goal"]["position"],
                "yaw": trajectory["observation"]["goal"]["yaw"],
                "image": goal_imgs,
            },
            "is_terminal": trajectory["is_terminal"],
            "reached": trajectory["reached"],
            "time_to_goal": time_to_original_goal,
            "reward": original_reward,
            "mc_returns": original_mc_returns,
            "type": tf.fill(tf.shape(trajectory["_len"]), ORIGINAL_GOAL),
        }

        del trajectory["observation"]["goal"]

        should_resample_goals = tf.random.uniform(
            [traj_len]) < relabel_probability

        goals = tf.nest.map_structure(
            lambda x, y: _broadcasted_where(
                should_resample_goals,
                x,
                y,
            ),
            postive_goals,
            original_goal,
        )
    else:
        goals = postive_goals

    if history_len:
        obs_with_context = add_history(trajectory["observation"], history_len, traj_len)
    else: 
        obs_with_context = trajectory["observation"]

    return {
        # No terminal or rewards, these correspond to goals
        "observation": obs_with_context,
        "action": trajectory["action"],
        "is_first": trajectory["is_first"],
        "is_last": trajectory["is_last"],
        "crash": did_crash,
        "goal": goals,
        "_len": trajectory["_len"],
        "_frame_index": trajectory["_frame_index"],
    }


def prepare_flatten_trajectory(trajectory: dict):
    # SKIP FIRST
    trajectory["observation"] = tf.nest.map_structure(
            lambda x: x[1:], trajectory["observation"])
    trajectory["observation"]["prev_action"] = trajectory["action"][:-1]

    result = {
        # Real
        "observations": tf.nest.map_structure(
            lambda x: x[:-1], trajectory["observation"]
        ),
        "next_observations": tf.nest.map_structure(
            lambda x: x[1:], trajectory["observation"]
        ),
        "crash": trajectory["crash"][2:],
        "actions": trajectory["action"][1:-1],
        # Possibly relabeled
        "goals": tf.nest.map_structure(lambda x: x[1:-1], trajectory["goal"]),
        "_frame_index": trajectory["_frame_index"][1:-1],
        "_len": trajectory["_len"][1:-1] - 1,
    }

    return result


def resample_negatives(
    batch_of_rl_data, negative_probability, crash_penalty, discount, reward_type
):
    batch_shape = tf.shape(batch_of_rl_data["_len"])
    batch_size = batch_shape[0]

    # Resample negatives only with P=should_resample
    should_resample = tf.random.uniform(
        shape=batch_shape) < negative_probability
    resample_idcs = tf.where(
        should_resample,
        tf.random.uniform(
            shape=batch_shape, minval=1, maxval=batch_size, dtype=tf.int32
        ),
        0,
    ) + tf.range(batch_size)
    resample_idcs = tf.math.floormod(resample_idcs, batch_size)

    # Construct the negative goals
    did_crash = batch_of_rl_data["crash"]

    negative_rewards, negative_mc_returns = compute_rewards_negatives(
        reward_type,
        did_crash,
        crash_penalty,
        discount,
    )

    negative_time_to_goal = tf.fill(batch_shape, value=np.inf)
    negative_goal = {
        "observation": tf.nest.map_structure(
            lambda x: tf.gather(x, resample_idcs, axis=0),
            batch_of_rl_data["goals"]["observation"],
        ),
        "is_terminal": did_crash,
        "reached": tf.fill(batch_shape, False),
        "time_to_goal": negative_time_to_goal,
        "reward": negative_rewards,
        "mc_returns": negative_mc_returns,
        "type": tf.fill(batch_shape, NEGATIVE_GOAL),
    }

    batch_of_rl_data["goals"] = tf.nest.map_structure(
        lambda x, y: _broadcasted_where(should_resample, x, y),
        negative_goal,
        batch_of_rl_data["goals"],
    )
    return batch_of_rl_data


def to_rl_format(data):
    return {
        "observations": data["observations"],
        "next_observations": data["next_observations"],
        "actions": data["actions"],
        "goals": data["goals"]["observation"],
        "rewards": data["goals"]["reward"],
        "is_terminal": data["goals"]["is_terminal"],
        "masks": ~data["goals"]["is_terminal"],
        "mc_returns": data["goals"]["mc_returns"],
        "reached": data["goals"]["reached"],
        "crashed": data["crash"],
        "time_to_goal": data["goals"]["time_to_goal"],
        "resample_type": data["goals"]["type"],
        "_frame_index": data["_frame_index"],
    }


def remove_y_action(trajectory: dict, action_key: str = "action"):
    trajectory[action_key] = (
        tf.stack(  # only keep FIRST and LAST (should correspond to x, and yaw)
            [trajectory[action_key][..., 0], trajectory[action_key][..., -1]], axis=-1
        )
    )
    return trajectory


obs_desired_keys = {
    "position": 2,  # just x, y, drop z if that's included
    "image": str,  # should always be a string
    "yaw": 1,  # should always be a string
}


def fix_obs_type(trajectory: dict, has_goal):
    # make sure it's a fully formed trajectory
    tensors = tf.nest.flatten(trajectory)
    lengths = [tf.shape(tensor)[0] for tensor in tensors]
    traj_len = tf.reduce_min(lengths)  # tf.reduce_min(trajectory["_len"])
    trajectory = tf.nest.map_structure(lambda x: x[:traj_len], trajectory)

    # use relative positions for trajectory loading
    if (
        "odom_pose" in trajectory["observation"].keys()
    ):  # we have a different relative & absolute position to think about
        if (
            has_goal and "goal" in trajectory["observation"].keys()
        ):  # recorder.py records fake goals to fit the format: # stick with 'absolute' position because that's the coords the goal uses
            new_pos = trajectory["observation"]["position"][:, :2]
            new_yaw = tf.expand_dims(
                from_quaternion(trajectory["observation"]["orientation"])[
                    :, 2], axis=1
            )
        else:
            new_pos = trajectory["observation"]["odom_pose"][:, :2]
            new_yaw = trajectory["observation"]["odom_pose"][:, -1]
    else:  # just have one position
        new_pos = (
            trajectory["observation"]["position"]
            if trajectory["observation"]["position"].shape[1] == 2
            else trajectory["observation"]["position"][:, :2]
        )
        new_yaw = (
            trajectory["observation"]["yaw"]
            if "yaw" in trajectory["observation"].keys()
            else tf.expand_dims(
                from_quaternion(trajectory["observation"]["orientation"])[
                    :, 2], axis=1
            )
        )

    new_obs = {
        "position": new_pos,
        "yaw": new_yaw,
        "image": trajectory["observation"][
            "image"
        ],
    }

    if "crash" in trajectory["observation"].keys():
        new_obs["crash"] = trajectory["observation"]["crash"]

    # KEEP GOAL if it was in there
    if (
        has_goal and "goal" in trajectory["observation"].keys()
    ): 
        new_obs["goal"] = {}

        goal_yaw = (
            trajectory["observation"]["goal"]["yaw"]
            if "yaw" in trajectory["observation"]["goal"].keys()
            else tf.expand_dims(
                from_quaternion(
                    trajectory["observation"]["goal"]["orientation"])[:, 2],
                axis=1,
            )
        )

        new_obs["goal"]["yaw"] = goal_yaw
        new_obs["goal"]["position"] = trajectory["observation"]["goal"]["position"][
            :, :2
        ]
        new_obs["goal"]["image"] = trajectory["observation"]["goal"]["image"]
        new_obs["goal"]["reached"] = trajectory["observation"]["goal"]["reached"]

        trajectory["reached"] = new_obs["goal"]["reached"]

    trajectory["observation"] = new_obs

    return trajectory


def waypoint_actions(trajectory, num_skip):
    current_position = trajectory["observation"]["position"]
    current_yaw = trajectory["observation"]["yaw"]
    next_position = trajectory["observation"]["position"][num_skip:]
    next_position = tf.concat([
        next_position,
        tf.repeat(next_position[-1:], num_skip, axis=0),
    ], axis=0)

    while current_yaw.ndim > 1:
        current_yaw = tf.squeeze(current_yaw, axis=-1)

    delta = next_position - current_position
    rot_mat = tf.stack([
        tf.stack([tf.math.cos(current_yaw), tf.math.sin(current_yaw)], axis=1),
        tf.stack([-tf.math.sin(current_yaw), tf.math.cos(current_yaw)], axis=1),
    ], axis=1)

    delta_rotated = tf.matmul(rot_mat, delta[..., None])[..., 0]

    trajectory["action"] = delta_rotated

    return trajectory


def make_positions_relative(sample, waypoint_spacing, goal_dist_threshold=0.2):
    def get_relative_position(base_position, base_yaw, goal_position, is_negative):
        if base_yaw.ndim > 0:
            base_yaw = tf.squeeze(base_yaw, axis=-1)
        goal_position = (goal_position - base_position) / waypoint_spacing
        rotation_matrix = tf.stack(
            [
                tf.stack([tf.cos(base_yaw), tf.sin(base_yaw)], axis=0),
                tf.stack([-tf.sin(base_yaw), tf.cos(base_yaw)], axis=0),
            ],
            axis=0,
        )
        goal_vector = tf.matmul(
            rotation_matrix, goal_position[..., None])[..., 0]
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

    (
        sample["observations"]["goal_vector"],
        sample["observations"]["goal_vector_magdir"],
    ) = get_relative_position(
        sample["observations"]["position"],
        sample["observations"]["yaw"],
        sample["goals"]["position"],
        sample["resample_type"] == NEGATIVE_GOAL,
    )
    (
        sample["next_observations"]["goal_vector"],
        sample["next_observations"]["goal_vector_magdir"],
    ) = get_relative_position(
        sample["next_observations"]["position"],
        sample["next_observations"]["yaw"],
        sample["goals"]["position"],
        sample["resample_type"] == NEGATIVE_GOAL,
    )

    return sample
