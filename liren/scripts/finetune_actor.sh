#!/bin/bash

# GENERAL PARAMS
ROBOT_IP=localhost
CODE_DIR=/home/create/repos
DATA_DIR=/home/create/misc/data 

# PARAMS FOR AGENT, SAVING
ACTION_TYPE=gc_cql_remote
DATA_SAVE_LOCATION=remote
SAVER_IP=100.95.228.62

# PARAMS FOR GOAL LOOP
IMG_SIZE=64
DEP_ENV=bair_loop
GOAL_LOOP_DIR=$DATA_DIR/goal_loops 
GOAL_NPZ=$GOAL_LOOP_DIR/npz/img_$IMG_SIZE/$DEP_ENV.npz

# RECOVERY INFO
DOCK_POSE="(-0.2763309,-0.31755227,0.00217097,-0.00100256,-0.01681328,-0.0863167,0.99612534)"
DOCK_DIR=/home/create/misc/data/docking/rail_docking.pkl

# FULL THING: pick and chose local / remote data / model combo! 
python $CODE_DIR/multinav-rl/multinav/robot/actor.py \
    --robot_ip $ROBOT_IP \
    --obs_type create \
    --robot_type create \
    --action_type $ACTION_TYPE \
    --goal_npz $GOAL_NPZ \
    --data_save_location $DATA_SAVE_LOCATION \
    --saver_ip $SAVER_IP \
    --check_battery \
    --check_stuck \
    --handle_keepouts \
    --dock_pose $DOCK_POSE \
    --dock_dir $DOCK_DIR