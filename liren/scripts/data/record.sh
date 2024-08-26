conda deactivate
source /home/lydia/jaxrl_env/bin/activate
source /opt/ros/humble/setup.bash
#  \ /home/lydia/real_data/goal_loops/rail_desks 
# /home/lydia/real_data/vint/phonebooths 

python /home/lydia/repos/multinav-rl/multinav/robot/recorder.py \
    --data_save_dir  /home/lydia/data/create_data/deployment/vint/chai_corner_6 \
    --max_time 1200 \
    --server_ip 10.41.196.233 \
    --handle_crash True