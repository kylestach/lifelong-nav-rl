conda deactivate
source /home/lydia/jaxrl_env/bin/activate
source /opt/ros/humble/setup.bash
#  \ /home/lydia/real_data/goal_loops/rail_desks 
# /home/lydia/real_data/vint/phonebooths 

python /home/lydia/repos/multinav-rl/multinav/data/tfds_to_pkl_npz.py \
    --dataset_name  rail_corner_debug:0.0.3 \
    --dataset_dir /home/lydia/data/goal_loops/raw/ \
    --save_name /home/lydia/data/goal_loops/npz/rail_corner_debug.npz \
    --step 1