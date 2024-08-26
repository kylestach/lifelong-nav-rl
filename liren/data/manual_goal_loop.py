import argparse

from agentlace.action import ActionServer, ActionConfig, ActionClient
import pickle
from liren.utils.trainer_bridge_common import (
    make_action_config,
    task_data_format,
)
import time
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import sys, select
import cv2

def main(server_ip,
        img_type, 
        save_path,
        img_size,
        action_config_type,
        ): 
    
    print("Connecting to action server...")
    action_config = make_action_config(action_config_type)
    action_client = ActionClient(
            server_ip,
            action_config,
        )
    print("... connected!")
    
    print("Collecting goal loop. Your input optinos are: \n\t'n': take a picture \n\t'd':delete the latest observation, \n\t's': save completed loop")
    images = []
    while True:
        i, o, e = select.select([sys.stdin], [], [], 0.001)
        if i:
            received_input = sys.stdin.readline().strip()

            if 'n' in received_input:
                obs = action_client.obs()
                pic = obs["image"]
                if img_type == "str":
                    pic = tf.io.decode_image(pic, expand_animations=False)
                images.append(pic)
                print(f"Image collected, goal trajectory currently has {len(images)} steps.")
            elif 'd' in received_input:
                images.pop()
                print(f"Latest image removed, now goal trajectory has {len(images)} steps.")
            elif 's' in received_input:
                save_data = {}
                save_data["data/position"] = [[0, 0, 0]for i in range(len(images))]
                save_data["data/orientation"] = [[ 0, 0, 0, 0] for i in range(len(images))]
                save_data["data/image"] = images
                save_data["data/image"] = [cv2.resize(np.array(img), dsize=(img_size, img_size), interpolation=cv2.INTER_CUBIC) for img in save_data["data/image"]]

                np.savez(save_path, **save_data)


                print(f"Saved .npz with {len(images)} images to {save_path}.")
                return

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Goal Loop Collection')
    
    parser.add_argument('--robot_ip', type=str, default="localhost", 
                        help='IP address to connect to robot server')
    parser.add_argument('--obs_type', type=str, default = "generic",
                        help="Path to save the goal loop to (including the .npz at the end)")
    parser.add_argument('--img_type', type=str, default = "str",
                        help='Maximum run time')
    parser.add_argument('--img_size', type=int, default = 64,
                        help='Maximum run time')
    parser.add_argument('--save_path', type=str,
                        help="Path to save the goal loop to (including the .npz at the end)")
    args = parser.parse_args()

    main(args.robot_ip, args.img_type, args.save_path, args.img_size, args.obs_type)
