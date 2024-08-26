#!/bin/bash

# GENERAL PARAMS
ROBOT_IP=localhost
MAX_TIME=600

# LOCAL MODEL INFO
MODEL_TYPE=gc_bc
MODEL=gc_bc_waypoint_2024_05_31_07_46_27
MODELS_FOLDER=bc_models
FINETUNING=/
IMG_SIZE=64
DEP_STEP=dep_495k
DEP_ENV=outdoor_morning_6
CODE_DIR=/home/jackal/repos # /home/lydia/repos/
DATA_DIR=/home/jackal/data # /home/create/misc/data 

# PARAMS FOR SAVING
DATA_SAVE_LOCATION=none # local, remote, none 

# PARAMS FOR AGENT
ACTION_TYPE=gc_bc_local # gc_cql_local/remote, gc_bc_local/remote, random, inplace, teleop, forward_jerk
CHECKPOINT_BASE_DIR=$DATA_DIR/checkpoints # /home/lydia/data/checkpoints
CHECKPOINT_DIR=$CHECKPOINT_BASE_DIR/$MODELS_FOLDER/$FINETUNING$MODEL
CHECKPOINT_STEP=450000

# PARAMS FOR GOAL 
GOAL_LOOP_BASE=$DATA_DIR/goal_loops # /home/lydia/data/goal_loops
GOAL_NPZ=$GOAL_LOOP_BASE/npz/img_$IMG_SIZE/$DEP_ENV.npz # $CODE_DIR/multinav-rl/multinav/data/bair_loop.npz 

# FULL THING: pick and chose local / remote data / model combo! 
python $CODE_DIR/multinav-rl/multinav/robot/actor.py \
    --robot_ip $ROBOT_IP \
    --obs_type generic \
    --data_save_location $DATA_SAVE_LOCATION \
    --action_type $ACTION_TYPE \
    --goal_npz $GOAL_NPZ \
    --checkpoint_load_dir $CHECKPOINT_DIR \
    --checkpoint_load_step $CHECKPOINT_STEP \
    --deterministic \
    --step_by_one \
    --manually_advance \
    --max_time $MAX_TIME \