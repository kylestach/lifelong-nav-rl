LIREN_DIR=~/lifelong-nav-rl
ROBOT_DIR=$LIREN_DIR/liren/robot/create

# Create a new tmux session
session_name="nav_fallback"
tmux new-session -d -s $session_name

# Split the window into 5 panes
tmux selectp -t 0  
tmux splitw -v -p 50
tmux selectp -t 1    
tmux splitw -h -p 50
tmux selectp -t 0  
tmux splitw -h -p 50 
tmux selectp -t 0
tmux splitw -h -p 50 

# Launch sensors and robot description
tmux select-pane -t 0
tmux send-keys "ros2 service call /reset_pose irobot_create_msgs/ResetPose" Enter
tmux send-keys "ros2 param set motion_control reflexes_enabled False" Enter
tmux send-keys "ros2 param set motion_control reflexes.REFLEX_BUMP True" Enter
tmux send-keys "ros2 param set motion_control reflexes.REFLEX_CLIFF True" Enter
tmux send-keys "ros2 param set motion_control reflexes.REFLEX_DOCK_AVOID True" Enter
tmux send-keys "ros2 param set motion_control reflexes.REFLEX_GYRO_CAL True" Enter
tmux send-keys "ros2 param set motion_control reflexes.REFLEX_PANIC True" Enter
tmux send-keys "ros2 param set motion_control reflexes.REFLEX_PROXIMITY_SLOWDOWN True" Enter
tmux send-keys "ros2 param set motion_control reflexes.REFLEX_STUCK True" Enter
tmux send-keys "ros2 param set motion_control reflexes.REFLEX_VIRTUAL_WALL True" Enter
tmux send-keys "ros2 param set motion_control reflexes.REFLEX_WHEEL_DROP True" Enter
tmux send-keys "ros2 launch deployment robot_launch.py" Enter

# Run the navigation stack
tmux select-pane -t 1
tmux send-keys "ros2 launch nav2_bringup bringup_launch.py params_file:=$ROBOT_DIR/deployment/config/nav2_params_keepout.yaml map:=$ROBOT_DIR/deployment/maps/map_example.yaml" Enter

# Launch RGB image compression node
tmux select-pane -t 4
tmux send-keys "conda activate liren-fine" Enter
tmux send-keys "python $ROBOT_DIR/deployment/deployment/image_compressor.py" Enter

# robot action server
tmux select-pane -t 2
tmux send-keys "conda activate liren-fine" Enter
tmux send-keys "python $ROBOT_DIR/nav_robot_action_server.py" Enter # --raw_img

# Launch foxglove
tmux select-pane -t 3
tmux send-keys "ros2 launch foxglove_bridge foxglove_bridge_launch.xml" Enter