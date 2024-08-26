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
from irobot_create_msgs.msg import HazardDetectionVector, HazardDetection
from irobot_create_msgs.action import Undock, Dock
from tf2_ros import Buffer as TransformBuffer, TransformListener, TransformBroadcaster, LookupException, ConnectivityException, ExtrapolationException
import tf_transformations
from tf2_geometry_msgs import do_transform_pose_stamped

import cv2
from cv_bridge import CvBridge, CvBridgeError

from liren.robot.create import state_machine
from liren.utils.trainer_bridge_common import make_action_config

def transform_odometry_to_map(odometry_msg: nm.Odometry, tf_buffer):
    try:
        # Create a PoseStamped object from the Odometry message
        pose_stamped = gm.PoseStamped()
        pose_stamped.header = odometry_msg.header
        pose_stamped.pose = odometry_msg.pose.pose

        # Lookup the latest transform from the "odom" frame to the "map" frame
        transform = tf_buffer.lookup_transform('map', odometry_msg.header.frame_id, RosTime())

        # Transform the PoseStamped object to the "map" frame
        transformed_pose = do_transform_pose_stamped(pose_stamped, transform)

        # Create a new Odometry message for the transformed pose
        transformed_odometry = nm.Odometry()
        transformed_odometry.header = transformed_pose.header
        transformed_odometry.child_frame_id = 'base_link'  # or whichever frame your robot's odometry refers to
        transformed_odometry.pose.pose = transformed_pose.pose
        transformed_odometry.twist = odometry_msg.twist  # assuming twist remains in the original frame

        return transformed_odometry

    except (LookupException, ConnectivityException, ExtrapolationException) as e:
        print(f"Transform error: {e}")
        return None

class NavRobotActionServer(Node):
    def __init__(self, server_ip_address, raw_img):
        super().__init__("nav_action_server")

        # TODO: Pull config from server
        self._latest_obs = {
            "image": np.array(b"", dtype=bytes),
            "imu_accel": np.zeros((3,), dtype=np.float32),
            "imu_gyro": np.zeros((3,), dtype=np.float32),
            "odom_pose": np.zeros((3,), dtype=np.float32),
            "linear_velocity": np.zeros((3,), dtype=np.float32),
            "angular_velocity": np.zeros((3,), dtype=np.float32),

            "cliff": np.zeros((), dtype=bool),
            "crash": np.zeros((), dtype=bool),
            "crash_left": np.zeros((), dtype=bool),
            "crash_right": np.zeros((), dtype=bool),
            "crash_center": np.zeros((), dtype=bool),
            "battery_charge": np.zeros((), dtype=np.float32),

            "stall": np.zeros((), dtype=bool),
            "keepout": np.zeros((), dtype=bool),

            "position": np.zeros((3,), dtype=np.float32),
            "orientation": np.array([0, 0, 0, 1], dtype=np.float32),
            "pose_std": np.zeros((6,), dtype=np.float32),

            "action_state_source": np.zeros((), dtype=str),
            "last_action_linear": np.zeros((3,), dtype=str),
            "last_action_angular": np.zeros((3,), dtype=str),
        }

        # ROS parameters
        self.tick_rate = self.declare_parameter("tick_rate", 10)
        self.keepout_requested = False

        self.nav2_action_client = RosActionClient(
            self, NavigateToPose, "navigate_to_pose"
        )
        self.dock_action_client = RosActionClient(self, Dock, "dock")
        self.undock_action_client = RosActionClient(self, Undock, "undock")

        self.tf_buffer = TransformBuffer()
        self.tf_broadcaster = TransformBroadcaster(self)
        self.reset_frame = gm.TransformStamped()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        best_effort_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            depth=1,
        )

        transient_local_qos = QoSProfile(
            depth=10,
            durability=DurabilityPolicy.TRANSIENT_LOCAL
        )

        self.state_machine = state_machine.StateMachine(self)
        self.state_publisher = self.create_publisher(stdm.String, "state_machine_state", 10)
        self.twist_publisher = self.create_publisher(gm.Twist, "cmd_vel", 10)
        self.goal_publisher = self.create_publisher(gm.PoseStamped, "goal_pose_viz", transient_local_qos)
        self.goal_image_publisher = self.create_publisher(sm.CompressedImage, "/front/goal_img/compressed", transient_local_qos)
        self.q_val_publisher = self.create_publisher(stdm.Float32MultiArray, "/q_vals", transient_local_qos)


        self.motion_param_client = self.create_client(SetParameters, 'motion_control/set_parameters')
        while not self.motion_param_client.wait_for_service(timeout_sec=1.0):
                    self.get_logger().info('Waiting for motion control parameter service...')


        self.teleop_twist_sub = self.create_subscription(
            gm.Twist,
            "teleop_vel",
            self.receive_teleop_twist_callback,
            10,
        )
        self.nav2_twist_sub = self.create_subscription(
            gm.Twist,
            "nav2_vel",
            self.receive_nav2_twist_callback,
            10,
        )
        self.hazard_sub = self.create_subscription(
            HazardDetectionVector, 
            "/hazard_detection", 
            self.hazard_callback, 
            best_effort_qos,
        )
        self.battery_sub = self.create_subscription(
            sm.BatteryState, 
            "/battery_state", 
            self.battery_callback, 
            best_effort_qos,
        )

        # COPIED IN FROM bairvFKeepout Map .yaml, could read instead somehow
        self.keepout_grid = {}
        self.keepout_grid["grid"] = None # need to get it loaded 
        self.keepout_grid["origin_x"] = -11.2
        self.keepout_grid["origin_y"] = -37.2
        self.keepout_grid["resolution"] = 0.06
        self.keepout_grid["occupied_thresh"] = 0.65

        self.keepout_sub = self.create_subscription(
            nm.OccupancyGrid,
            '/keepout_filter_mask',
            self.keepout_callback,
            10,
        )

        # Sensor subscriptions
        if raw_img:
            self.bridge = CvBridge()
            self.image_sub = self.create_subscription(
                sm.Image,
                "/front/image_raw",
                self.image_callback_raw,
                10,
            )
        else: # compressed image
            self.image_sub = self.create_subscription(
                sm.CompressedImage,
                # "/front/image_raw/compressed", # GRAY SCALE
                "/front/image_raw/compressed_rgb",
                self.image_callback_compressed,
                10,
            )
        self.amcl_pose_sub = self.create_subscription(
            gm.PoseWithCovarianceStamped,
            "/amcl_pose",
            self.amcl_pose_callback,
            10,
        )
        self.odom_sub = self.create_subscription(
            nm.Odometry,
            "/odom",
            self.odom_callback,
            best_effort_qos,
        )
        self.imu_sub = self.create_subscription(
            sm.Imu,
            "/imu",
            self.imu_callback,
            best_effort_qos,
        )

        self.action_server = ActionServer(
            config=make_action_config("create"),
            obs_callback=self.agentlace_obs_callback,
            act_callback=self.agentlace_act_callback,
        )

        self.state_machine_timer = self.create_timer(
            1 / self.tick_rate.value, self.tick
        )

        self.last_agentlace_action_key = None

        # Start running
        self.action_server.start(threaded=True)
    
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
    
    def amcl_pose_callback(self, pose: gm.PoseWithCovarianceStamped):
        self._latest_obs["pose_std"] = np.sqrt(np.diag(np.array(pose.pose.covariance).reshape((6, 6)))).astype(np.float32)

    def hazard_callback(self, hazard: HazardDetectionVector):
        self._latest_obs["crash"] = False
        self._latest_obs["crash_left"] = False
        self._latest_obs["crash_right"] =  False
        self._latest_obs["crash_center"] = False
        self._latest_obs["cliff"] = False
        self._latest_obs["stall"] = False

        for d in hazard.detections:
            if d.type == HazardDetection.BUMP:
                self._latest_obs["crash"] = True
                if "right" in d.header.frame_id:
                    self._latest_obs["crash_right"] = True
                if "left" in d.header.frame_id:
                    self._latest_obs["crash_left"] = True
                if "center" in d.header.frame_id:
                    self._latest_obs["crash_center"] = True
    
            elif d.type == HazardDetection.CLIFF:
                self._latest_obs["cliff"] = True

            elif d.type == HazardDetection.STALL:
                self._latest_obs["stall"] = True

    
    def battery_callback(self, battery_state: sm.BatteryState):
        self._latest_obs["battery_charging"] = battery_state.current > 0 
        self._latest_obs["battery_charge"] =  tf.cast(battery_state.percentage, tf.float32)
        


    def odom_callback(self, odom: nm.Odometry):
        self._latest_obs["odom_pose"] = np.array([
            odom.pose.pose.position.x,
            odom.pose.pose.position.y,
            tf_transformations.euler_from_quaternion([
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            ])[-1],
        ], dtype=np.float32)

        self._latest_obs["linear_velocity"] = np.array([
            odom.twist.twist.linear.x,
            odom.twist.twist.linear.y,
            odom.twist.twist.linear.z,
        ], dtype=np.float32)

        self._latest_obs["angular_velocity"] = np.array([
            odom.twist.twist.angular.x,
            odom.twist.twist.angular.y,
            odom.twist.twist.angular.z,
        ], dtype=np.float32)

        odom_map = transform_odometry_to_map(odom, self.tf_buffer)

        if odom_map is not None:
            self._latest_obs["position"] = np.array(
                [
                    odom_map.pose.pose.position.x,
                    odom_map.pose.pose.position.y,
                    odom_map.pose.pose.position.z,
                ],
                dtype=np.float32
            )
            self._latest_obs["orientation"] = np.array(
                [
                    odom_map.pose.pose.orientation.x,
                    odom_map.pose.pose.orientation.y,
                    odom_map.pose.pose.orientation.z,
                    odom_map.pose.pose.orientation.w,
                ],
                dtype=np.float32
            )

            if self.in_keepout_zone(odom_map.pose.pose):
                self._latest_obs["keepout"] = True
                # print("NEW POSITION IN KEEPOUT ZONE")
            else:
                self._latest_obs["keepout"] = False



    def imu_callback(self, imu: sm.Imu):
        self._latest_obs["imu_accel"] = np.array([
            imu.linear_acceleration.x,
            imu.linear_acceleration.y,
            imu.linear_acceleration.z,
        ], dtype=np.float32)

        self._latest_obs["imu_gyro"] = np.array([
            imu.angular_velocity.x,
            imu.angular_velocity.y,
            imu.angular_velocity.z,
        ], dtype=np.float32)

    def keepout_callback(self, grid: nm.OccupancyGrid):
        self.keepout_grid["grid"] = grid
        self.keepout_grid["origin_x"] = grid.info.origin.position.x
        self.keepout_grid["origin_y"] = grid.info.origin.position.y
        self.keepout_grid["width"] = grid.info.width
        self.keepout_grid["height"] = grid.info.height
        self.keepout_grid["resolution"] = grid.info.resolution

    def in_keepout_zone(self, pose: gm.Pose):
        if self.keepout_grid["grid"] is None:
            print("NO GRID MAP FOUND")
            return False
        
        map_x = int((pose.position.x - self.keepout_grid["origin_x"]) / self.keepout_grid["resolution"])
        map_y = int((pose.position.y - self.keepout_grid["origin_y"]) / self.keepout_grid["resolution"])
        
        # in the occupancy grid 
        if map_x < 0 or map_x >= self.keepout_grid["width"] or map_y < 0 or map_y >= self.keepout_grid["height"]:
            print("OUT OF BOUNDS")
            return False
        
        index = map_y * self.keepout_grid["width"]  + map_x
        grid_value = self.keepout_grid["grid"].data[index]
        
        if grid_value > self.keepout_grid["occupied_thresh"]:
            return True
        else:
            return False

    def agentlace_obs_callback(self, keys: Set[str]):
        return {k: self._latest_obs[k] for k in keys}

    def agentlace_act_callback(self, key: str, payload: Any):
        if key == "action_vw":
            result = self.receive_vw_action_callback(payload)
        elif key == "action_nav2":
            result = self.start_nav2_action(payload)
        elif key == "reset":
            result = self.start_reset_callback(payload)
        elif key == "dock":
            result = self.start_dock_callback()
        elif key == "undock":
            result = self.start_undock_callback()
        elif key == "enable_reflexes":
            result = self.enable_reflexes()
        elif key == "disable_reflexes":
            result = self.disable_reflexes()
        elif key == "new_goal":
            result = self.update_goal_pose(payload)
        elif key == "q_vals":
            result =  self.receive_q_vals(payload)
        else:
            result = {"running": False, "reason": f"Unknown key {key}"}
        
        self.last_agentlace_action_key = key


        return result

    def receive_teleop_twist_callback(self, command: state_machine.TwistType):
        if not self.state_machine.try_update(state_machine.TeleopState, twist=command):
            return self.state_machine.accept_state(
                state_machine.TeleopState(
                    self.get_clock().now(),
                    twist=command,
                )
            )

    def receive_nav2_twist_callback(self, command: state_machine.TwistType):
        if not self.state_machine.try_update(
            state_machine.Nav2ActionState, twist=command
        ):
            logging.info("Nav2 command received while not in Nav2ActionState")

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
        
    def receive_q_vals(self, q_vals: np.ndarray):
        # just get [x q, theta q]
        msg = stdm.Float32MultiArray()
        msg.data = [float(q_vals[0]), float(q_vals[1])]
        self.q_val_publisher.publish(msg)
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

    def _do_start_action(self, make_state_machine_state: Callable):
        """
        Return True if a new reset was accepted or there is an existing reset in progress.
        Return False if there is an existing reset that has completed.
        """
        accepted = self.state_machine.accept_state(make_state_machine_state())
        if accepted:
            return {"running": True, "reason": "started action"}
        else:
            return {"running": False, "reason": f"current state is {self.state_machine.current_state}"}

    def start_reset_callback(self, payload):
        # COMPUTE TWISTS 
        if "twists"  in payload.keys():
            twists = payload["twists"]
            time_per_twist = payload["time_per_twist"]
        else:
            twists = [[-0.1, np.random.choice([-1, 1]) * np.random.uniform(np.pi / 12, np.pi / 6)]]
            time_per_twist = [2.0]
        
        # CALL RESET
        if isinstance(self.state_machine.current_state, state_machine.DoResetState):
            if self.state_machine.current_state.expired(self.get_clock().now()):
                return {"running": False, "reason": "completed reset"}
            else:
                return {"running": True, "reason": "still resetting"}
        elif self.last_agentlace_action_key == "reset":
            return {"running": False, "reason": "already reset, idle"}
        else: # trigger it! 
            return self._do_start_action(
                lambda: state_machine.DoResetState(
                    self.get_clock().now(),
                    twists,
                    time_per_twist,
                )
            )

    def start_nav2_action(self, payload):
        return self._do_start_action(
            lambda: state_machine.IRobotNavState(
                self.get_clock().now(),
                self.nav2_action_client,
                payload["goal_pose"],
                self.get_clock(),
            )
        )   

    def start_dock_callback(self):
        if isinstance(self.state_machine.current_state, state_machine.IRobotDockState):
            if self.state_machine.current_state.expired(self.get_clock().now()):
                return {"running": False, "reason": "completed dock"}
            else:
                return {"running": True, "reason": "still docking"}
        else: 
            return self._do_start_action(
                lambda: state_machine.IRobotDockState(
                    self.get_clock().now(),
                    self.dock_action_client,
                    self.get_clock(),
                )
            )

    def start_undock_callback(self):
        if isinstance(self.state_machine.current_state, state_machine.IRobotDockState):
            if self.state_machine.current_state.expired(self.get_clock().now()):
                return {"running": False, "reason": "completed dock"}
            else:
                return {"running": True, "reason": "still docking"}
        else: # trigger it! 
            return self._do_start_action(
                lambda: state_machine.IRobotUndockState(
                    self.get_clock().now(),
                    self.undock_action_client,
                    self.get_clock(),
                )
            )

    def enable_reflexes(self):
        return self._do_start_action(
            lambda: state_machine.RosParamState(
                self.get_clock().now(),
                self.get_clock(),
                self.motion_param_client,
                "reflexes_enabled",
                True,
                bool,
            )
        )

    def disable_reflexes(self):
        return self._do_start_action(
            lambda: state_machine.RosParamState(
                self.get_clock().now(),
                self.get_clock(),
                self.motion_param_client,
                "reflexes_enabled",
                False,
                bool,
            )
        )

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
