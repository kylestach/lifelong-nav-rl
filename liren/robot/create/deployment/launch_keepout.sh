LIREN_DIR=~/lifelong-nav-rl
ROBOT_DIR=$LIREN_DIR/liren/robot/create
DEP_DIR=$ROBOT_DIR/deployment

session_name="nav_keepout_nodes"
if tmux has-session -t $session_name 2>/dev/null; then
	tmux kill-session -t $session_name
fi
tmux new-session -d -s $session_name

# Split the window into four panes
tmux selectp -t 0    # select the first (0) pane
tmux splitw -v -p 50 # split it into two halves
tmux selectp -t 1    # go back to the first pane
tmux splitw -h -p 50 # split it into two halves
tmux selectp -t 0   # go back to the first pane
tmux splitw -h -p 50 # split it into two halves

# # Filter Mask
tmux select-pane -t 0
tmux send-keys "ros2 run nav2_map_server map_server --ros-args -r __node:=filter_mask_server --params-file $DEP_DIR/config/nav2_params_keepout.yaml" Enter

# CostMap
tmux select-pane -t 1
tmux send-keys "ros2 run nav2_map_server costmap_filter_info_server --ros-args -r __node:=costmap_filter_info_server --params-file $DEP_DIR/config/nav2_params_keepout.yaml" Enter

# Lifecycle Manager
tmux select-pane -t 2
tmux send-keys "ros2 run nav2_lifecycle_manager lifecycle_manager --ros-args -r __node:=lifecycle_manager_costmap_filters -p \"use_sim_time:=True\" -p \"autostart:=True\" -p \"node_names:=['filter_mask_server', 'costmap_filter_info_server']\"" Enter


tmux -2 attach-session -t $session_name
