import launch 
from launch_ros.actions import Node, SetRemap
from launch.actions import IncludeLaunchDescription, GroupAction
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch.conditions import UnlessCondition, IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_teleop_joy = get_package_share_directory('teleop_twist_joy')
    teleop_launch_file = PathJoinSubstitution(
    [pkg_teleop_joy, 'launch', 'teleop-launch.py'])
    
    teleop_joy = GroupAction(
        actions =[
            SetRemap(src="/cmd_vel", dst="/teleop_vel"),
            IncludeLaunchDescription(
                PythonLaunchDescriptionSource([teleop_launch_file]),
                launch_arguments={
                        'joy_config': 'xbox',
                    }.items())
            ]  
        )
    pkg_create3_common_bringup = get_package_share_directory('irobot_create_common_bringup')
    robot_description_launch_file = PathJoinSubstitution(
    [pkg_create3_common_bringup, 'launch', 'robot_description.launch.py'])
    robot_description = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([robot_description_launch_file])
    )

    return launch.LaunchDescription([
        Node(
            package='rplidar_ros',
            executable='rplidar_node',
            name='rplidar_node',
            parameters=[{'channel_type':'serial',
                         'serial_port': '/dev/rplidar',
                         'serial_baudrate': 256000,
                         'frame_id': 'laser_link',
                         
                         'inverted': False,
                         'angle_compensate': True
                         }],
            output='screen'),
        Node(
            package='usb_cam',
            namespace='front',
            executable='usb_cam_node_exe',
            name='usb_cam',
            parameters=['/home/create/create_ws/src/deployment/config/camera_front_new.yaml']
        ),
        robot_description,
        teleop_joy,
    ])