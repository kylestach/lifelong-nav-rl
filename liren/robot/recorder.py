## general recorder of just observations from the robot action server and nothing else
# NO GOAL ! since that's not getting sent by robot action server... this is JUST Robot action server info. 
# based on model_deployment.py (also rosless)

# generic imports
import time
import numpy as np
from PIL import Image
import os
import sys
import io
from dataclasses import dataclass
from typing import Optional
import tensorflow as tf
import time 
import atexit
import logging

import argparse

# ros imports
from liren.utils.trainer_bridge_common import (
    make_action_config,
    task_data_format,
)

# custom imports
from agentlace.action import ActionClient
from agentlace.data.rlds_writer import RLDSWriter
from agentlace.data.tf_agents_episode_buffer import EpisodicTFDataStore

# data loading
from absl import app, flags, logging as absl_logging


@dataclass
class RobotConfig:
    image_topic: str
    imu_topic: str
    pose_topic: str
    action_topic: str

    is_simulation: bool = False
    gazebo_model_name: Optional[str] = None

IMAGENET_MEAN = tf.constant([0.485, 0.456, 0.406])
IMAGENET_STD = tf.constant([0.229, 0.224, 0.225])

MAX_TRAJ_LEN = 100 # 3 times a second * 30 seconds = 90 long
STEPS_TRY = 60
GOAL_DIST = STEPS_TRY // 2 # 4 - 10 # normal around this with 5 std 

def normalize_image(image: tf.Tensor) -> tf.Tensor:
    """
    Normalize the image to be between 0 and 1
    """
    return (tf.cast(image, tf.float32) / 255.0 - IMAGENET_MEAN) / IMAGENET_STD


class Recorder():
    def __init__(self, server_ip: str, 
                 save_dir, 
                 max_time, 
                 handle_crash):
        self.handle_crash = handle_crash.lower() == "true" or handle_crash.lower() == "t"

        self.max_time = max_time
        self.start_time = time.time()
        self.tick_rate = 3

        data_dir = save_dir
        if not os.path.exists(data_dir):
            os.makedirs(data_dir)
        existing_folders = [0] + [int(folder.split('.')[-1]) for folder in os.listdir(data_dir)]
        latest_version = max(existing_folders)

        self.version= f"0.0.{1 + latest_version}"
        self.datastore_path = f"{data_dir}/{self.version}"
        os.makedirs(self.datastore_path)

        self.action_client = ActionClient(
            server_ip,
            make_action_config("create"),
        )
        
        # setting up rlds writer
        self.image_size = (64, 64)
        data_spec = task_data_format("create")
    
        self.writer = RLDSWriter(
            dataset_name="test",
            data_spec = data_spec,
            data_directory = self.datastore_path,
            version = self.version,
            max_episodes_per_file = 100,
        )

        atexit.register(self.writer.close) 

        self.data_store = EpisodicTFDataStore(
            capacity=1000,
            data_spec= data_spec,
            rlds_logger = self.writer
        )
        print("Datastore set up")
            
    def run(self):
        loop_time = 1 / self.tick_rate
        start_time = time.time()

        self.first = True
        self.last = False
        self.terminal = False

        self.just_crashed = False
        self.traj_len = 0
        self.curr_goal = None 

        while True:
            new_start_time = time.time()
            elapsed = new_start_time - start_time
            if elapsed < loop_time:
                time.sleep(loop_time - elapsed)
            start_time = time.time()

            self.tick()

    def int_image(self, img):
        return np.asarray((img * IMAGENET_STD + IMAGENET_MEAN) * 255, dtype = np.uint8)

    def save(self, obs):
        obs["image"] = tf.convert_to_tensor(obs["image"]) 
        obs["action_state_source"] = tf.convert_to_tensor(obs["action_state_source"]) 

        if self.handle_crash:
            if obs["crash"] or obs["keepout"]:
                self.last = True
                self.terminal = True
            else:
                self.last = False
                self.terminal = False
        else:
            obs["crash"] = False 
            self.last = False
            self.terminal = False

        sample_info = {
            "position": tf.convert_to_tensor(np.array([0.0, 0.0,0.0]), dtype = tf.float32),
            "orientation":  tf.convert_to_tensor(np.array([0.0, 0.0, 0.0, 0.0]), dtype=tf.float32),
            "offset": np.float32(0),
        }
        
        obs["goal"] = {
            "image": obs["image"], 
            "position":  tf.convert_to_tensor(np.array([0.0, 0.0,0.0]), dtype = tf.float32), 
            "orientation": tf.convert_to_tensor(np.array([0.0, 0.0, 0.0, 0.0]), dtype=tf.float32), 
            "reached": False,
            "sample_info": sample_info,
        }

        # del obs["action_state_source"]
        formatted_obs = {
            "observation": obs,
            "action": tf.concat([obs["last_action_linear"], obs["last_action_angular"]], axis = 0),
            "is_first": self.first, 
            "is_last": self.last, 
            "is_terminal": self.terminal, 
        }

        if self.first:
            self.first = False

        self.data_store.insert(formatted_obs)

    def tick(self): 
        obs = self.action_client.obs() 
        if obs is not None:
            self.save(obs)
            
        if self.max_time is not None:
            if time.time() - self.start_time > self.max_time:
                print(f"Killing recording after {time.time() - self.start_time} seconds")
                sys.exit()
            
if __name__ == "__main__":

    tf.get_logger().setLevel("WARNING")
    logging.basicConfig(level=logging.WARNING)
    absl_logging.set_verbosity("WARNING")

    parser = argparse.ArgumentParser(description='My Python Script')
    parser.add_argument('--data_save_dir', type=str, help='Where to save collected data')
    parser.add_argument('--max_time', type=int, help='How long to run for')
    parser.add_argument('--server_ip', type=str, help='What IP to connect to a robot action server on')
    parser.add_argument('--handle_crash', type=str, help='What IP to connect to a robot action server on')
    args = parser.parse_args()

    Recorder(server_ip= args.server_ip,  
          save_dir = args.data_save_dir, 
          max_time = args.max_time,
          handle_crash = args.handle_crash,
          ).run() 