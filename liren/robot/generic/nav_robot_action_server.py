from abc import ABC
import argparse
import logging
from typing import Any, Optional, Tuple, Set, Callable
from agentlace.action import ActionServer, ActionConfig
import numpy as np
import tensorflow as tf

from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy
from rclpy.action import ActionClient as RosActionClient
import rclpy
from rclpy.time import Time as RosTime
from rclpy.node import Node
from rcl_interfaces.srv import SetParameters

import geometry_msgs.msg as gm
import nav_msgs.msg as nm
import sensor_msgs.msg as sm
import std_msgs.msg as stdm
from nav2_msgs.action import NavigateToPose
from tf2_ros import Buffer as TransformBuffer, TransformListener, TransformBroadcaster, LookupException, ConnectivityException, ExtrapolationException
import tf_transformations
from tf2_geometry_msgs import do_transform_pose_stamped

import cv2
from cv_bridge import CvBridge, CvBridgeError

from liren.robot.generic import state_machine
from liren.utils.trainer_bridge_common import make_action_config

class NavRobotActionServer(Node):
    def __init__(self, server_ip_address, raw_img):
        super().__init__("nav_action_server")

        # Empty Obs
        self._latest_obs = {
            "image": np.array(b"", dtype=bytes),

            "action_state_source": np.zeros((), dtype=str),
            "last_action_linear": np.zeros((3,), dtype=str),
            "last_action_angular": np.zeros((3,), dtype=str),
        }
        print("Observation type set up")

        # ROS parameters
        self.tick_rate = self.declare_parameter("tick_rate", 10)
        self.keepout_requested = False

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1,
        )
        transient_local_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )
        print("QoS Profiles Loaded")

        self.state_machine = state_machine.StateMachine(self)
        self.state_publisher = self.create_publisher(stdm.String, "state_machine_state", 10)
        self.twist_publisher = self.create_publisher(gm.Twist, "/j100_0166/cmd_vel", 10)
        self.goal_publisher = self.create_publisher(gm.PoseStamped, "goal_pose_viz", transient_local_qos)
        self.goal_image_publisher = self.create_publisher(sm.CompressedImage, "/goal_img/compressed", transient_local_qos)
        print("Publishers set up")

        # Sensor subscriptions
        if raw_img:
            self.bridge = CvBridge()
            self.image_sub = self.create_subscription(
                sm.Image,
                "/image_raw",
                self.image_callback_raw,
                10,
            )
        else: 
            self.image_sub = self.create_subscription(
                sm.CompressedImage,
                "/image_raw/compressed",
                self.image_callback_compressed,
                10,
            )

        self.action_server = ActionServer(
            config=make_action_config("generic"),
            obs_callback=self.agentlace_obs_callback,
            act_callback=self.agentlace_act_callback,
        )

        self.state_machine_timer = self.create_timer(
            1 / self.tick_rate.value, self.tick
        )
        self.last_agentlace_action_key = None

        # Start running
        self.action_server.start(threaded=True)
        print("action server started")
    
    def image_callback_raw(self, image:sm.Image): # image: sm.CompressedImage): # LYDIA FIX 
        try:
            cv_image = self.bridge.imgmsg_to_cv2(image, desired_encoding='rgb8')
            rgb_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2RGB)
            img_np = np.array(cv_image)
            self._latest_obs["image"] = img_np
            
        except CvBridgeError as e:
            print("couldn't convert image")

    def image_callback_compressed(self, image:sm.CompressedImage): # image: sm.CompressedImage): # LYDIA FIX 
        self._latest_obs["image"] = bytes(image.data)

    def agentlace_obs_callback(self, keys: Set[str]):
        return {k: self._latest_obs[k] for k in keys}

    def agentlace_act_callback(self, key: str, payload: Any):
        if key == "action_vw":
            result = self.receive_vw_action_callback(payload)
        elif key == "new_goal":
            result = self.update_goal_pose(payload)
        else:
            result = {"running": False, "reason": f"Unknown key {key}"}
        return result

    def update_goal_pose(self, pose: dict):
        # get an array with 7 elements: 3 for point, 4 for quaternion. 
        new_goal = gm.PoseStamped()
        new_goal.header.stamp = self.get_clock().now().to_msg()
        new_goal.header.frame_id = "map" # in map coords 

        new_goal.pose.position.x = pose["position"][0]
        new_goal.pose.position.y = pose["position"][1]
        new_goal.pose.position.z = pose["position"][2]

        new_goal.pose.orientation.x = pose["orientation"][0]
        new_goal.pose.orientation.y = pose["orientation"][1]
        new_goal.pose.orientation.z = pose["orientation"][2]
        new_goal.pose.orientation.w = pose["orientation"][3]

        self.goal_publisher.publish(new_goal)

        if "image" in pose.keys():
            my_img = sm.CompressedImage()
            my_img.header.stamp = self.get_clock().now().to_msg()
            my_img.format = "jpeg"  # Set the appropriate format (jpeg, png, etc.)
            my_img.data = pose["image"].tobytes()

            self.goal_image_publisher.publish(my_img)

        return {"running": False, "reason": f"completed"}
        
    def receive_vw_action_callback(self, command: np.ndarray):
        if self.state_machine.try_update(state_machine.TwistTaskState, twist=command):
            return {"running": True, "reason": "already running"}
        else:
            accepted = self.state_machine.accept_state(
                state_machine.TwistTaskState(
                    self.get_clock().now(),
                    twist=command,
                )
            )
            if accepted:
                return {"running": True, "reason": "started action"}
            else:
                return {"running": False, "reason": f"current state is {self.state_machine.current_state}"}

    def tick(self):
        self.state_machine.tick(self._latest_obs)
        self.republish()
        self.state_publisher.publish(stdm.String(data=type(self.state_machine.current_state).__name__))

    def republish(self):
        self._latest_obs["action_state_source"] = type(self.state_machine.current_state).__name__
        self._latest_obs["last_action_linear"] = np.array([self.state_machine.current_state.twist[0], 0.0, 0.0], dtype=np.float32)
        self._latest_obs["last_action_angular"] = np.array([0.0, 0.0, self.state_machine.current_state.twist[1]], dtype=np.float32)

        twist_msg = gm.Twist()
        twist_msg.linear.x = float(self.state_machine.current_state.twist[0])
        twist_msg.angular.z = float(self.state_machine.current_state.twist[1])
        self.twist_publisher.publish(twist_msg)


if __name__ == "__main__":
    rclpy.init()

    import logging
    logging.basicConfig(level=logging.WARNING)

    parser = argparse.ArgumentParser(description='Robot Action Server')
    parser.add_argument('--raw_img', action='store_true')
    args = parser.parse_args()

    node = NavRobotActionServer("127.0.0.1", raw_img = args.raw_img)

    rclpy.spin(node)
