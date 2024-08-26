from agentlace.action import ActionConfig
from agentlace.trainer import TrainerConfig
import tensorflow as tf
import numpy as np


def make_trainer_config():
    return TrainerConfig(
        port_number=5488,
        broadcast_port=5489,
        request_types=["send-stats", "get-model-config"],
    )

def observation_format(observation_key_type):
    if observation_key_type == "generic":
        return    {
            "image": tf.TensorSpec((), tf.string, name="image"),

            "action_state_source": tf.TensorSpec((), tf.string, name="action_state_source"),
            "last_action_linear": tf.TensorSpec((3,), tf.float32, name="last_action_linear"),
            "last_action_angular": tf.TensorSpec((3,), tf.float32, name="last_action_angular"),
        }
    
    elif observation_key_type == "create": 
        return    {
            # Raw sensor
            "image": tf.TensorSpec((), tf.string, name="image"),
            "imu_accel": tf.TensorSpec((3,), tf.float32, name="imu_accel"),
            "imu_gyro": tf.TensorSpec((3,), tf.float32, name="imu_gyro"),
            "odom_pose": tf.TensorSpec((3,), tf.float32, name="odom_pose"),
            "linear_velocity": tf.TensorSpec((3,), tf.float32, name="linear_velocity"),
            "angular_velocity": tf.TensorSpec((3,), tf.float32, name="angular_velocity"),

            # Hazards from IRobot
            "cliff": tf.TensorSpec((), tf.bool, name="cliff"),
            "crash": tf.TensorSpec((), tf.bool, name="crash"),
            "crash_left": tf.TensorSpec((), tf.bool, name="crash_left"),
            "crash_right": tf.TensorSpec((), tf.bool, name="crash_right"),
            "crash_center": tf.TensorSpec((), tf.bool, name="crash_center"),
            "stall": tf.TensorSpec((), tf.bool, name="stall"),
            "keepout": tf.TensorSpec((), tf.bool, name="keepout"),
            "battery_charge": tf.TensorSpec((), tf.float32, name="battery_charge"), 
            "battery_charging": tf.TensorSpec((), tf.bool, name="battery_charging"), 

            # Estimator
            "position": tf.TensorSpec((3,), tf.float32, name="position"),
            "orientation": tf.TensorSpec((4,), tf.float32, name="orientation"),
            "pose_std": tf.TensorSpec((6,), tf.float32, name="pose_std"),

            # State machine and action
            "action_state_source": tf.TensorSpec((), tf.string, name="action_state_source"),
            "last_action_linear": tf.TensorSpec((3,), tf.float32, name="last_action_linear"),
            "last_action_angular": tf.TensorSpec((3,), tf.float32, name="last_action_angular"),
        }
    else: 
        raise ValueError(f"Unknown observation config type {observation_key_type}")

def rlds_data_format(observation_key_type):
    obs_format = observation_format(observation_key_type)
    # del obs_format["action_state_source"] 
    return {
        "observation": obs_format,
        "is_first": tf.TensorSpec((), tf.bool, name="is_first"),
        "is_last": tf.TensorSpec((), tf.bool, name="is_last"),
        "is_terminal": tf.TensorSpec((), tf.bool, name="is_terminal"),
}

def task_data_format(observation_key_type):
    return {
        "observation": {
            **observation_format(observation_key_type),
            "goal": {
                "image": tf.TensorSpec((), tf.string, name="image"),
                "position": tf.TensorSpec((3,), tf.float32, name="position"),
                "orientation": tf.TensorSpec((4,), tf.float32, name="orientation"),
                "reached": tf.TensorSpec((), tf.bool, name="reached"),
                "sample_info": {
                    "position": tf.TensorSpec((3,), tf.float32, name="position"),
                    "orientation": tf.TensorSpec((4,), tf.float32, name="orientation"),
                    "offset": tf.TensorSpec((), tf.float32, name="offset"),
                }
            },
        },
        "action": tf.TensorSpec((6,), tf.float32, name="action"),
        "is_first": tf.TensorSpec((), tf.bool, name="is_first"),
        "is_last": tf.TensorSpec((), tf.bool, name="is_last"),
        "is_terminal": tf.TensorSpec((), tf.bool, name="is_terminal"),
    }

def observation_keys(observation_key_type):
    keys = list(observation_format(observation_key_type).keys())
    return keys 

def make_action_config(action_config_type):
    if action_config_type == "generic":
        return ActionConfig(
            port_number=1111,
            action_keys=["action_vw", "new_goal"],
            observation_keys=list(observation_format(action_config_type).keys()),
        )
    elif action_config_type == "create":
        return ActionConfig(
            port_number=1111,
            action_keys=["action_vw", "action_nav2", "reset", "dock", "undock", "enable_reflexes", "disable_reflexes", "new_goal", "q_vals"],
            observation_keys=list(observation_format(action_config_type).keys()),
        )
    else:
        raise ValueError(f"Unknown action config type {action_config_type}")