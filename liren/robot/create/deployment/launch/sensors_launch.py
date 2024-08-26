import launch 
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.substitutions import Command, LaunchConfiguration, PathJoinSubstitution
from launch.conditions import UnlessCondition, IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
import launch_ros
import os
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    pkg_irobot_create_description = get_package_share_directory('irobot_create_description')
    pkg_create3_common_bringup = get_package_share_directory('irobot_create_common_bringup')
    pkg_create3_gazebo_bringup = get_package_share_directory('irobot_create_gazebo_bringup')
    robot_description_launch_file = PathJoinSubstitution(
    [pkg_create3_common_bringup, 'launch', 'robot_description.launch.py'])
    
    # Create 3 robot model and description
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
            arguments=['--params-file', '/opt/ros/humble/share/usb_cam/config/params.yaml']

        ),
        robot_description,
    ])
