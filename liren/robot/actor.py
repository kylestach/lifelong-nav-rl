# Generic Imports
import tty
import termios
from liren.utils.trainer_bridge_common import (
    make_action_config,
    make_trainer_config,
    task_data_format,
)
from liren.utils.utils import normalize_image, _yaw, pose_distance
from liren.robot.robot_config import get_config as get_robot_config
from liren.robot.tasks.goal_task import TrainingTask
from liren.training.agent import Agent, RandomAgent
from liren.training.model_config import get_config as get_agent_config
from agentlace.data.rlds_writer import RLDSWriter
from agentlace.trainer import TrainerClient
from agentlace.action import ActionClient

from absl import logging as absl_logging
from collections import deque
import os
import sys

import jax
import time

import atexit
import pickle
import logging
import argparse
import numpy as np
from PIL import Image
from io import BytesIO
import tensorflow as tf
import tensorflow_datasets as tfds

# Handling blocking / non-blocking input
input_settings = termios.tcgetattr(sys.stdin)
def set_blocking_mode():
    tty.setcbreak(sys.stdin)
def set_non_blocking_mode():
    termios.tcsetattr(sys.stdin, termios.TCSADRAIN, input_settings)

# Entering tuple over command line 
def pose_type(input_str):
    try:
        input_str = input_str.strip('()')
        return tuple(map(float, input_str.split(',')))
    except ValueError:
        raise argparse.ArgumentTypeError(
            "Input must be a tuple of integers separated by commas.")


class Actor():
    # SETUP
    def __init__(self,
                 robot_ip: str,
                 obs_type: str,
                 robot_type: str,
                 save_data_info: dict,
                 load_actor_info: dict,
                 goal_info: dict,
                 recovery_info: dict,
                 max_time: int,
                 ):

        # General Setup 
        self.max_time = max_time
        self.start_time = time.time()
        self.obs_type = obs_type
        self.tick_rate = 3
        self.robot_config = get_robot_config(robot_type)

        # Recovery
        self.recovery_info = recovery_info

        if self.recovery_info["check_battery"]:
            self.battery_check_rate = 5  # check every 5 seconds
            self.battery_threshold = 0.20
            self.last_battery_check = time.time()
            self.battery_low = False

        if self.recovery_info["check_stuck"]:
            self.crash_history = deque(maxlen=10)

        # Robot
        self.robot = ActionClient(
            robot_ip,
            make_action_config(self.obs_type),
        )

        # Data Saving
        self.remote_server = None
        self.save_data_info = save_data_info
        if self.save_data_info["data_save_location"] == "local":
            self.datastore = self.set_up_local_datastore()
            print("Saving locally \n")
            if self.datastore is None:
                raise ValueError("Could not set up local datastore")
        elif self.save_data_info["data_save_location"] == "remote":
            self.datastore = self.set_up_remote_datastore()
            print("Saving remotely \n")
            if self.datastore is None:
                raise ValueError("Could not set up local datastore")
        else:
            self.datastore = None
            print("Not saving \n")

        # Agent
        print("Setting up agent...")
        self.load_actor_info = load_actor_info
        self.action_type = load_actor_info["action_type"]

        if self.action_type == "random":
            self.agent = RandomAgent()
        elif self.action_type == "teleop":
            self.agent = None
        elif self.action_type == "inplace":
            self.agent = None
        elif self.action_type == "forward_jerk":
            self.agent = None
            self.action_step = 15
        elif self.action_type.endswith("local"):
            self.agent = self.set_up_local_agent()
        elif self.action_type.endswith("remote"):
            self.agent = self.set_up_remote_agent()
        else:
            raise ValueError(f"Unknown agent type {self.action_type}")

        print("...agent set up! \n")

        # Data History 
        if self.agent_config and hasattr(self.agent_config, 'image_size'):
            self.image_size = self.agent_config.image_size
        else:
            self.image_size = 64

        if self.agent_config and hasattr(self.agent_config.agent_config, 'history_len'):
            self.obs_buffer = deque(
                maxlen=self.agent_config.agent_config["history_len"])
            print(f"Using observation history of size {self.agent_config.agent_config['history_len']}. \n")
        else:
            self.obs_buffer = None
            print("Not using observation history. \n")

        # Metrics 
        self.ends = {"reach": 0, "timeout": 0, "crash": 0}
        atexit.register(self.exit)

        # Goals 
        self.goal_info = goal_info
        self.task = TrainingTask(
            self.goal_info["goal_npz"],
            self.goal_info["step_by_one"],
            self.goal_info["manually_advance"],
        )
        print("Goals set up \n")

    # RUNNING

    def run(self):
        # Get intial params, if running remote model
        if self.action_type.endswith("remote"):
            self.get_initial_params()

        # Running Params
        loop_time = 1 / self.tick_rate
        start_time = time.time()

        # Initialization
        self.curr_goal = None
        self.action_counter = 0  # used for forward_jerk action type
        self.rng = jax.random.PRNGKey(seed=42)

        # Running Loop
        while True:
            new_start_time = time.time()
            elapsed = new_start_time - start_time
            if elapsed < loop_time:
                time.sleep(loop_time - elapsed)
            start_time = time.time()
            self.tick()

    def tick(self):
        # Get observation
        obs = self.robot.obs()
        if obs is None:
            return

        # Check battery status
        if self.recovery_info["check_battery"]:
            self.check_battery(obs)

        # Make observation image usable
        # Decode jpeg
        if isinstance(obs["image"], (str, bytes, np.str_)):
            self.curr_obs_image = tf.io.decode_image(
                obs["image"], expand_animations=False)
        else:  # need it compressed to send & save
            self.curr_obs_image = obs["image"]
            img = BytesIO()
            Image.fromarray(obs["image"]).save(img, format="JPEG")
            obs["image"] = tf.constant(img.getvalue(), dtype=tf.string)
        # Handle grayscale
        if self.curr_obs_image.shape[-1] != 3:
            self.curr_obs_image = tf.concat(
                [self.curr_obs_image, self.curr_obs_image, self.curr_obs_image], axis=-1)
        # Normalize
        self.curr_obs_image = normalize_image(self.curr_obs_image)
        # Make Proper Shape
        self.curr_obs_image = np.array(self.curr_obs_image)
        if self.curr_obs_image.shape != (self.image_size, self.image_size, 3):
            self.curr_obs_image = np.array(
                tf.image.resize(self.curr_obs_image, [self.image_size, self.image_size]))

        # Make sure we have a goal
        if self.curr_goal is None:
            if self.goal_info["manually_advance"]:
                self.reset_goal(None, None, False)
            else:
                self.reset_goal(obs["position"], obs["orientation"], False)

        # Add to buffer
        if self.agent_config and hasattr(self.agent_config.agent_config, 'history_len'):
            self.obs_buffer.append(self.curr_obs_image)

        # Update Task
        if self.goal_info["manually_advance"]:
            self.task_result = self.task.update()
        elif self.recovery_info["handle_keepouts"]:
            self.task_result = self.task.update(
                obs["position"], obs["orientation"], obs["crash"] or obs["keepout"])
        else:
            self.task_result = self.task.update(
                obs["position"], obs["orientation"], obs["crash"])

        self.prev_goal = self.curr_goal
        # Handle End of Trajectory
        if self.task_result["reached_goal"] or self.task_result["timeout"] or self.task_result["crash"]:
            action = self.handle_traj_end(obs)
        else:
            action = self.take_action(obs)

        # Save
        if self.datastore is not None:
            self.save(obs, action)

        # Timeout
        if self.max_time is not None and time.time() - self.start_time > self.max_time:
            sys.exit()

    def exit(self):
        self.robot.act("action_vw", np.array([0, 0]))
        print(
            f"Killing model deployment after {time.time() - self.start_time} seconds.")
        print(
            f"Reached {self.ends['reach']} trajectories, crashed {self.ends['crash']} times, and had {self.ends['timeout']} timeouts.")

    ## HELPER FUNCTIONS ##

    # SAVING
    def set_up_local_datastore(self):
        # set up directory
        data_dir = self.save_data_info["data_save_dir"]

        existing_folders = [0] + [int(folder.split('.')[-1])
                                  for folder in os.listdir(data_dir)]
        latest_version = max(existing_folders)

        version = f"0.0.{1 + latest_version}"
        datastore_path = f"{data_dir}/{version}"
        print(f"Saving to {datastore_path}")
        os.makedirs(datastore_path)

        # get data format
        data_spec = task_data_format(self.obs_type)

        # set up writer
        writer = RLDSWriter(
            dataset_name="test",
            data_spec=data_spec,
            data_directory=datastore_path,
            version=version,
            max_episodes_per_file=100,
        )
        atexit.register(writer.close)  # so it SAVES on exit

        from agentlace.data.tf_agents_episode_buffer import EpisodicTFDataStore
        return EpisodicTFDataStore(
            capacity=1000,
            data_spec=data_spec,
            rlds_logger=writer
        )

    def set_up_remote_datastore(self):
        from agentlace.data.data_store import QueuedDataStore
        local_data_store = QueuedDataStore(capacity=10)

        self.remote_server = TrainerClient(
            "online_data",
            self.save_data_info["saver_ip"],
            make_trainer_config(),
            local_data_store,
            wait_for_server=True,
        )

        return local_data_store

    def save(self, obs, action):
        if obs["action_state_source"] == "DoResetState":
            return

        obs["image"] = tf.convert_to_tensor(
            obs["image"])  # get it as raw byte array

        obs["action_state_source"] = tf.convert_to_tensor(
            obs["action_state_source"])

        obs["goal"] = {
            "image": self.prev_goal["img_bytes"],
            "position": self.prev_goal["position"],
            "orientation": self.prev_goal["orientation"],
            "reached": self.task_result["reached_goal"],
            "sample_info": self.prev_goal["sample_info"],
        }

        if self.action_type == "teleop":
            taken_action = tf.concat(
                [obs["last_action_linear"], obs["last_action_angular"]], axis=0)
        else:
            taken_action = tf.convert_to_tensor(
                np.array([action[0], 0, 0, 0, 0, action[1]], dtype=np.float32))

        formatted_obs = {
            "observation": obs,
            "action": taken_action,
            "is_first": self.task_result["is_first"],
            "is_last": self.task_result["is_last"],
            "is_terminal": self.task_result["is_terminal"],
        }
        self.datastore.insert(formatted_obs)

    # AGENT

    def set_up_local_agent(self):
        self.agent_config = get_agent_config(
            self.action_type[:-6])  # trim "_local"
        agent = Agent(self.agent_config, 42)
        agent.load_checkpoint(
            self.load_actor_info["checkpoint_dir"],
            self.load_actor_info["checkpoint_step"],
        )
        return agent

    def set_up_remote_agent(self):
        if not self.remote_server:  # wasn't set up for the datastore

            self.remote_server = TrainerClient(
                "online_data",
                self.save_data_info["saver_ip"],
                make_trainer_config(),
                None,  # no datastore to connect
                wait_for_server=True,
            )

        self.agent_config = self.remote_server.request(
            "get-model-config", {})
        agent = Agent(self.agent_config, 42)
        return agent

    def get_initial_params(self):
        received_params = False

        def _update_actor(data):
            nonlocal received_params
            received_params = True
            self.agent.update_params(data["params"])

        self.remote_server.recv_network_callback(_update_actor)
        self.remote_server.start_async_update(interval=1)

        while not received_params:
            time.sleep(1.0)
            print("Waiting for initial params...")

    # MOVING

    def take_action(self, obs):

        self.action_counter += 1

        if self.action_type == "random":
            action = self.agent.rand_action()
        elif self.action_type == "teleop":
            pass
        elif self.action_type == "inplace":
            action = np.array([0, np.random.uniform(-1.5, 1.5)])
        elif self.action_type == "forward_jerk":
            if self.action_counter % self.action_step == 0:
                action = np.array([1, 0])
            else:
                action = np.array([0, 0])
        elif self.action_type.endswith("local") or self.action_type.endswith("remote"):
            # Prepare Input Image
            curr_img = np.array(self.curr_obs_image)
            goal_img = np.array(self.curr_goal["image"])

            assert curr_img.shape == (
                self.image_size, self.image_size, 3), f"actual obs img size {curr_img.shape}"
            assert goal_img.shape == (
                self.image_size, self.image_size, 3), f"actual goal img size {goal_img.shape}"

            if self.agent_config and hasattr(self.agent_config.agent_config, 'history_len'):
                obs_list = list(self.obs_buffer)
                while len(obs_list) < self.agent_config.agent_config.history_len:
                    obs_list.insert(0, obs_list[0])
                obs_list = np.array(obs_list)
                action = self.agent.predict(
                    obs_image=obs_list, goal_image=goal_img, random=not self.load_actor_info["deterministic"])
            else:
                action = self.agent.predict(
                    obs_image=curr_img, goal_image=goal_img, random=not self.load_actor_info["deterministic"])

            action = np.array(
                [self.robot_config.waypoint_spacing * (action[0] - self.robot_config.x_offset),
                 action[1] * self.robot_config.angle_scale])
        else:
            raise ValueError(f"Unknown agent type {self.action_type}")

        # Make sure action is safe, and take it
        action[0] = max(self.robot_config.min_linear_vel, min(
            self.robot_config.max_linear_vel, action[0]))
        action[1] = max(self.robot_config.min_angular_vel, min(
            self.robot_config.max_angular_vel, action[1]))

        if not self.action_type == "teleop":
            self.robot.act("action_vw", action)

        return action

    # ENDING

    def reset_goal(self, pos, quat, reached):
        position, orientation = self.task.reset(pos, quat, reached)
        self.curr_goal = self.task.get_goal()
        if self.curr_goal is None:
            print("Goal loop completed!")

            # HANDLE EXIT 
            if self.action_type.endswith("remote"):
                self.remote_server.request("send-stats",
                    {"reach": self.task_result["reached_goal"],
                    "timeout": self.task_result["timeout"],
                    "crash": self.task_result["crash"]})

            self.ends["reach"] += self.task_result["reached_goal"]
            self.ends["timeout"] += self.task_result["timeout"]
            self.ends["crash"] += self.task_result["crash"]

            # Save
            obs = self.robot.obs()
            while obs is None:
                obs = self.robot.obs()

            if self.datastore is not None:
                self.save(obs, np.array([0, 0]))
            sys.exit()

        if type(self.curr_goal["image"]) == str:  # need to decode it!
            self.curr_goal["image"] = tf.io.decode_image(
                self.curr_goal["image"], expand_animations=False)

        if self.curr_obs_image.shape != (self.image_size, self.image_size, 3):
            resized_obs_img = np.array(
                tf.image.resize(self.curr_obs_image, [self.image_size, self.image_size]))

        self.curr_goal["int_image"] = self.curr_goal["image"]
        self.curr_goal["image"] = normalize_image(self.curr_goal["image"])
        self.curr_goal["yaw"] = _yaw(self.curr_goal["orientation"])

        goal_jpeg = BytesIO()
        Image.fromarray(self.curr_goal["int_image"]).save(
            goal_jpeg, format="JPEG")
        goal_img_bytes_pose = np.frombuffer(
            goal_jpeg.getvalue(), dtype=np.uint8)
        self.curr_goal["img_bytes"] = tf.constant(
            goal_jpeg.getvalue(), dtype=tf.string)

        goal_pose = {
            "position": self.curr_goal["position"].astype(float),
            "orientation": self.curr_goal["orientation"].astype(float),
            "image": goal_img_bytes_pose
        }

        self.robot.act("new_goal", goal_pose)

    def handle_traj_end(self, obs):
        reset_str = f"\nTrajectory End! Reached {self.task_result['reached_goal']} timeout {self.task_result['timeout']} crash {self.task_result['crash']}"
        if self.recovery_info["handle_keepouts"]:
            reset_str += f" keepout {obs['keepout']}"
        print(reset_str)

        if self.recovery_info["check_stuck"]:
            if self.task_result["timeout"]:
                self.crash_history.append("timeout")
            elif self.task_result["crash"]:
                self.crash_history.append("crash")
            elif self.task_result["reached_goal"]:
                self.crash_history.append("reach")

            # Check if we have crashed too many times recently and need help
            if self.task_result["timeout"] or self.task_result["crash"]:
                if self.crash_history.count("crash") + self.crash_history.count("keepout") > 8:
                    self.robot.act("action_vw", np.array([0.0, 0.0]))
                    self.send_slack_message(
                        f"I'm stuck at {obs['position']}, please come help!")
                    set_blocking_mode()
                    input(
                        "Crashed too many times, fix the robot and hit ENTER to continue:")
                    set_non_blocking_mode()
                    self.crash_history.clear()

                elif self.task_result["crash"]:
                    self.reset(obs)

        obs = self.robot.obs()
        while obs is None:
            obs = self.robot.obs()

        if self.goal_info["manually_advance"]:
            self.reset_goal(None, None, self.task_result["reached_goal"])
        else:
            self.reset_goal(obs["position"], obs["orientation"],
                            self.task_result["reached_goal"])

        if self.action_type.endswith("remote"):
            self.remote_server.request("send-stats",
                                       {"reach": self.task_result["reached_goal"],
                                        "timeout": self.task_result["timeout"],
                                        "crash": self.task_result["crash"]})

        self.ends["reach"] += self.task_result["reached_goal"]
        self.ends["timeout"] += self.task_result["timeout"]
        self.ends["crash"] += self.task_result["crash"]

        self.task.reset_timer()

        return np.array([0, 0])

    # RECOVERY

    def send_slack_message(self, text):
        from slack_sdk.errors import SlackApiError
        try:
            result = self.recovery_info["slack_client"].chat_postMessage(
                channel=self.recovery_info["slack_channel_id"],
                text=text
            )
        except SlackApiError as e:
            print(f"Slack Error: {e}")

    def reset(self, obs):
        if obs["keepout"]:
            twists = [np.array([-0.3, 0]), np.array([0.0, -0.5])]
            time_per_twist = [3.0, 1.2]
        # elif self.task_result["timeout"]:
        #     twists = [np.array([0.0, np.random.choice([-1, 1])])]
        #     time_per_twist = [np.random.uniform(0.25, 2)]
        elif obs["crash_left"]:
            twists = [np.array([-0.2, 0]), np.array([0.0, -0.5])]
            time_per_twist = [1.0, 1.0]
        elif obs["crash_right"]:
            twists = [np.array([-0.2, 0]), np.array([0.0, 0.5])]
            time_per_twist = [1.0, 1.0]
        elif obs["crash_center"]:
            twists = [np.array([-0.2, 0]),
                      np.random.choice([-1, 1]) * np.array([0.0, 0.5])]
            time_per_twist = [1.0, 1.0]

        res = {"running": True, "reason": "starting"}

        while res is None or res["running"]:
            res = self.robot.act(
                "reset", {"twists": twists, "time_per_twist": time_per_twist})
            time.sleep(0.5)

    def check_battery(self, obs):
        if time.time() - self.last_battery_check < self.battery_check_rate:
            return

        if obs["battery_charge"] < self.battery_threshold and not self.battery_low:
            self.battery_low = True

            self.send_slack_message(
                f"Battery has dropped below {self.battery_threshold}, I might need help soon. I am at {obs['position']} and heading home.")

            docking_attempts = 5
            for i in range(docking_attempts):
                self.return_to_dock()

                obs = None
                while obs is None:
                    obs = self.robot.obs()

                # successfully docked!
                if obs["battery_charging"]:
                    print("Battery Charging Successfully")
                    break
            self.send_slack_message("Docked! Charging up.")

            # Check every 5 minutes to see if you've charged enough
            while obs["battery_charge"] < self.battery_threshold:
                time.sleep(300)

                obs = None
                while obs is None:
                    obs = self.robot.obs()

            self.send_slack_message("Charged! Resuming operation.")

            # Back up from dock
            twists = [np.array([-0.2, 0]),
                      np.random.choice([-1, 1]) * np.array([0.0, 0.5])]
            time_per_twist = [3.0, 1.0]
            res = {"running": True, "reason": "starting"}
            while res is None or res["running"]:
                res = self.robot.act(
                    "reset", {"twists": twists, "time_per_twist": time_per_twist})
                time.sleep(0.3)
        else:
            self.battery_low = False
        self.last_battery_check = time.time()

    def return_to_dock(self):
        # RECOVERY MECHANISM

        # Get Closest Starting Map Point
        obs = None
        while obs is None:
            obs = self.robot.obs()
            curr_point = tuple(
                list(obs["position"]) + list(obs["orientation"]))
            point = min(self.recovery_info["path_to_dock"].keys(
            ), key=lambda x: pose_distance(curr_point, x, 0))

        # Navigate to be near dock
        start_time = time.time()
        res = {}
        while point != "DONE":
            goal_pose = point
            res["running"] = True
            while res["running"]:
                res = self.robot.act("action_nav2", {"goal_pose": goal_pose})
            point = self.recovery_info["path_to_dock"][point]

        print("Computed all Nav2 Waypoints")

        # For up to 15 minutes, wait to get close enough to dock
        def keep_waiting(start_time):
            return time.time() - start_time < 15 * 60 * 60

        def far_from_dock():
            obs = None
            while obs is None:
                obs = self.robot.obs()
            curr_point = tuple(
                list(obs["position"]) + list(obs["orientation"]))
            return pose_distance(curr_point, self.recovery_info["dock_pose"], 0) > 1.5

        while keep_waiting(start_time):
            if far_from_dock():
                pass  # keep trying
            else:
                break  # we got close enough!

        print("Either timed out or close enough to dock")
        time.sleep(5)  # make sure nav2 is done

        obs = None
        while obs is None:
            obs = self.robot.obs()

        if not keep_waiting(start_time):
            print("Timed Out")
            # didn't get close enough in 15 minutes, must be stuck! giving up!
            self.send_slack_message(
                f"Timed out on the way to dock. I am at {obs['position']}")
            pass

        # Try to dock!
        twists = [np.array([-0.2, 0]),
                  np.random.choice([-1, 1]) * np.array([0.0, 0.5])]
        time_per_twist = [0.5, 0.5]
        docking_attempts = 10
        for i in range(docking_attempts):
            res["running"] = True
            print(f"Attempting to dock {i}")

            while res is None or res["running"]:
                res = self.robot.act("dock")

            time.sleep(2)
            obs = None
            while obs is None:
                obs = self.robot.obs()
            # successfully docked!
            if obs["battery_charging"]:
                print("Battery Charging")
                break
            else:
                print("resetting position")
                res = {"running": True, "reason": "starting"}
                while not obs["battery_charging"] and (res is None or res["running"]):
                    res = self.robot.act(
                        "reset", {"twists": twists, "time_per_twist": time_per_twist})
                    obs = None
                    while obs is None:
                        obs = self.robot.obs()

                if obs["battery_charging"]:
                    print("battery charging during reset!")
                    break

        print("completed")


if __name__ == "__main__":

    tf.get_logger().setLevel("WARNING")
    logging.basicConfig(level=logging.WARNING)
    absl_logging.set_verbosity("WARNING")

    # ROBOT
    parser = argparse.ArgumentParser(description='Actor')
    parser.add_argument('--robot_ip', type=str, default="localhost",
                        help='IP address to connect to robot server')
    parser.add_argument('--obs_type', type=str, default="generic",
                        help='Observation type (generic or create)')
    parser.add_argument('--robot_type', type=str,
                        help='Robot type (jackal or create)')
    parser.add_argument('--max_time', type=int,
                        help='Maximum run time')

    # DATA
    parser.add_argument('--data_save_location', type=str,
                        help="Possible data save types: local, remote, none")
    parser.add_argument('--data_save_dir', type=str,
                        help='Where to save collected data')
    parser.add_argument('--saver_ip', type=str,
                        help='Where to save collected data and where to load remote model from')

    # ACTOR
    parser.add_argument('--action_type', type=str,
                        help="Possible action types: gc_cql_local/remote, gc_bc_local/remote, random, inplace, teleop, forward_jerk")
    parser.add_argument('--checkpoint_load_dir', type=str,
                        help='Where to load model checkpoint from')
    parser.add_argument('--checkpoint_load_step', type=int,
                        help='Which checkpoint to load')
    parser.add_argument('--deterministic', action="store_true",
                        help="Take deterministic actions from agent")

    # GOAL
    parser.add_argument('--goal_npz', type=str,
                        help="npz to load goals from ")
    parser.add_argument('--step_by_one', action="store_true",
                        help="If the goal should be selected as NEXT in loop")
    parser.add_argument('--manually_advance', action="store_true",
                        help="Advance goal graph manually by pressing r for reached, c for crashed!")

    # RECOVERY
    parser.add_argument('--check_battery', action="store_true",
                        help="Check robot battery status")
    parser.add_argument('--check_stuck', action="store_true",
                        help="Check if robot crashes or times out too much")
    parser.add_argument('--handle_keepouts', action="store_true",
                        help="Treat keepout zone like crashes")
    parser.add_argument('--dock_pose', type=pose_type,
                        help='The pose from which it is reasonable to dock')
    parser.add_argument('--dock_dir', type=str,
                        help='Where to load waypoints for docking')

    args = parser.parse_args()

    # Save Data Info
    save_data_info = {}
    save_data_info["data_save_location"] = args.data_save_location

    if args.data_save_location == "remote":
        if not args.saver_ip:
            raise ValueError(f"Must have saver IP specified to save remotely.")
        save_data_info["saver_ip"] = args.saver_ip
    elif args.data_save_location == "local":
        if not args.data_save_dir:
            raise ValueError(
                f"Must have save directory path specified to save locally.")
        save_data_info["data_save_dir"] = args.data_save_dir

    # Load Actor Info
    load_actor_info = {}
    load_actor_info["action_type"] = args.action_type
    if args.action_type.endswith("local") or args.action_type.endswith("remote"):
        load_actor_info["deterministic"] = args.deterministic

    if args.action_type.endswith("local"):
        if not args.checkpoint_load_dir:
            raise ValueError(
                f"Must have checkpoint load directory specified to load locally.")
        load_actor_info["checkpoint_dir"] = args.checkpoint_load_dir

        if not args.checkpoint_load_step:
            raise ValueError(
                f"Must have checkpoint load step specified to load locally.")
        load_actor_info["checkpoint_step"] = args.checkpoint_load_step
    elif args.action_type.endswith("remote"):
        if not args.saver_ip:
            raise ValueError(
                f"Must have saver IP specified to load agent remotely.")
        save_data_info["saver_ip"] = args.saver_ip

    # Goal Info
    goal_info = {}
    goal_info["goal_npz"] = args.goal_npz
    goal_info["step_by_one"] = args.step_by_one
    goal_info["manually_advance"] = args.manually_advance

    # Recovery Info
    recovery_info = {}
    
    recovery_info["check_battery"] = args.check_battery
    recovery_info["check_stuck"] = args.check_stuck
    recovery_info["handle_keepouts"] = args.handle_keepouts

    if args.check_battery or args.check_stuck:
        from slack_sdk import WebClient
        recovery_info["slack_client"] = WebClient(
            token=os.environ.get("SLACK_BOT_TOKEN"))
        recovery_info["slack_channel_id"] = "C07AWRX5C02"

    if recovery_info["check_battery"]:
        recovery_info["dock_pose"] = args.dock_pose
        if len(recovery_info["dock_pose"]) != 7:
            raise ValueError(
                f"Must have saver IP specified to load agent remotely.")

        with open(args.dock_dir, 'rb') as handle:
            recovery_info["path_to_dock"] = pickle.load(handle)

    Actor(robot_ip=args.robot_ip,
          obs_type=args.obs_type,
          robot_type=args.robot_type,
          save_data_info=save_data_info,
          load_actor_info=load_actor_info,
          goal_info=goal_info,
          recovery_info=recovery_info,
          max_time=args.max_time,
          ).run()
