#!/bin/bash

# GENERAL PARAMS
ROBOT_IP=localhost
MAX_TIME=600

# LOCAL MODEL INFO
MODEL_TYPE=gc_cql
MODEL=gc_cql_twist_dense_alpha1.0_proprioTrue_hist1_2024_07_19_22_33_54
MODELS_FOLDER=cql_models
FINETUNING=/
IMG_SIZE=64
DEP_STEP=dep_495k
DEP_ENV=bair_loop
CODE_DIR=/home/create/repos
DATA_DIR=/home/create/misc/data

# PARAMS FOR SAVING
DATA_SAVE_LOCATION=local # local, remote, none 
DATA_SAVE_DIR_BASE=$DATA_DIR/deployment
DATA_SAVE_DIR=$DATA_SAVE_DIR_BASE/$MODELS_FOLDER/$MODEL/$DEP_STEP/$DEP_ENV

# PARAMS FOR AGENT
ACTION_TYPE=gc_cql_local # gc_cql_local/remote, gc_bc_local/remote, random, inplace, teleop, forward_jerk
CHECKPOINT_BASE_DIR=$DATA_DIR/checkpoints # /home/lydia/data/checkpoints
CHECKPOINT_DIR=$CHECKPOINT_BASE_DIR/$MODELS_FOLDER/$FINETUNING$MODEL
CHECKPOINT_STEP=495000

# PARAMS FOR GOAL 
GOAL_LOOP_BASE=$DATA_DIR/goal_loops # /home/lydia/data/goal_loops
GOAL_NPZ=$GOAL_LOOP_BASE/npz/img_$IMG_SIZE/$DEP_ENV.npz # $CODE_DIR/multinav-rl/multinav/data/bair_loop.npz 

# RECOVERY
DOCK_POSE="(-0.2763309,-0.31755227,0.00217097,-0.00100256,-0.01681328,-0.0863167,0.99612534)"
DOCK_DIR=/home/create/misc/data/docking/rail_docking.pkl

python $CODE_DIR/multinav-rl/multinav/robot/actor_general.py \
    --robot_ip $ROBOT_IP \
    --obs_type create \
    --data_save_location $DATA_SAVE_LOCATION \
    --data_save_dir $DATA_SAVE_DIR \
    --action_type $ACTION_TYPE \
    --goal_npz $GOAL_NPZ \
    --check_battery \
    --dock_pose $DOCK_POSE \
    --dock_dir $DOCK_DIR \
    --check_stuck \
    --handle_keepouts \
    --checkpoint_load_dir $CHECKPOINT_DIR \
    --checkpoint_load_step $CHECKPOINT_STEP \