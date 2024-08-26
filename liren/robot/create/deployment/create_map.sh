LIREN_DIR=~/lifelong-nav-rl
ROBOT_DIR=$LIREN_DIR/liren/robot/create
DEP_DIR=$ROBOT_DIR/deployment

# Create a new tmux session
session_name="create_map"
tmux new-session -d -s $session_name

# Split the window into panes
tmux selectp -t 0    
tmux splitw -v -p 50 
tmux selectp -t 0    
tmux splitw -h -p 50 
tmux selectp -t 0 
tmux splitw -v -p 50 
tmux selectp -t 2 
tmux splitw -v -p 50 
tmux selectp -t 4    
tmux splitw -h -p 50 
tmux selectp -t 0    

# Launch sensors and robot description
tmux select-pane -t 0
tmux send-keys "ros2 launch foxglove_bridge foxglove_bridge_launch.xml" Enter

# Launch sensors and robot description
tmux select-pane -t 1
tmux send-keys "ros2 launch deployment sensors_launch.py" Enter

# Run the teleop_launch.py script in the second pane
tmux select-pane -t 2
tmux send-keys "ros2 launch deployment teleop_launch.py" Enter

# Run the navigation stack
tmux select-pane -t 3
tmux send-keys "ros2 launch nav2_bringup navigation_launch.py params_file:=$DEP_DIR/config/nav2_params.yaml" Enter

# Set up the keys to run the slam node
tmux select-pane -t 4
tmux send-keys "ros2 launch slam_toolbox online_async_launch.py slam_params_file:=$DEP_DIR/config/slam_toolbox_mapping.yaml" 

# Have a pane ready to save the map
tmux select-pane -t 5
tmux send-keys "ros2 service call /slam_toolbox/serialize_map slam_toolbox/SerializePoseGraph $DEP_DIR/maps/$1" 


# Attach to the tmux session
tmux -2 attach-session -t $session_name