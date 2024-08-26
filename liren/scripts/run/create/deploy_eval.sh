#!/bin/bash

# GENERAL PARAMS
# ROBOT_IP=localhost
ROBOT_IP=10.41.196.233
MAX_TIME=600

# LOCAL MODEL INFO
MODEL_TYPE=gc_cql

MODEL=gc_cql_twist_dense_alpha1.0_proprioTrue_hist1_2024_07_19_22_33_54-fine_495k-11_2024_08_13_03_56_47
FINETUNING=finetuned/
DEP_STEP=dep_25k
CHECKPOINT_STEP=25000

# MODEL=gc_cql_twist_dense_alpha1.0_proprioTrue_hist1_2024_07_19_22_33_54
# FINETUNING=/
# DEP_STEP=dep_495k
# CHECKPOINT_STEP=495000


MODELS_FOLDER=cql_models
IMG_SIZE=64
DEP_ENV=phonebooth_bend_6
CODE_DIR=/home/lydia/repos
DATA_DIR=/home/lydia/data
# CODE_DIR=/home/create/repos
# DATA_DIR=/home/create/misc/data

# PARAMS FOR SAVING
DATA_SAVE_LOCATION=local # local, remote, none 
DATA_SAVE_DIR_BASE=$DATA_DIR/create_data/deployment
DATA_SAVE_DIR=$DATA_SAVE_DIR_BASE/$MODELS_FOLDER/$FINETUNING$MODEL/$DEP_STEP/$DEP_ENV

# PARAMS FOR AGENT
ACTION_TYPE=gc_cql_local 
CHECKPOINT_BASE_DIR=$DATA_DIR/checkpoints 
CHECKPOINT_DIR=$CHECKPOINT_BASE_DIR/$MODELS_FOLDER/$FINETUNING$MODEL

# PARAMS FOR GOAL 
GOAL_LOOP_BASE=$DATA_DIR/goal_loops 
GOAL_NPZ=$GOAL_LOOP_BASE/npz/img_$IMG_SIZE/$DEP_ENV.npz

python $CODE_DIR/multinav-rl/multinav/robot/actor.py \
    --robot_ip $ROBOT_IP \
    --obs_type create \
    --data_save_location $DATA_SAVE_LOCATION \
    --data_save_dir $DATA_SAVE_DIR \
    --action_type $ACTION_TYPE \
    --goal_npz $GOAL_NPZ \
    --checkpoint_load_dir $CHECKPOINT_DIR \
    --checkpoint_load_step $CHECKPOINT_STEP \
    --deterministic \
    --step_by_one \

# # FULL THING: pick and chose local / remote data / model combo! 
# python $CODE_DIR/multinav-rl/multinav/robot/agent_general.py \
#     --robot_ip $ROBOT_IP \ 
#     --max_time $MAX_TIME \ 
#     # Data
#     --data_save_location $DATA_SAVE_LOCATION \ 
#     --data_save_dir $DATA_SAVE_DIR \ 
#     --saver_ip $SAVER_IP \ 
#     # Agent
#     --action_type $ACTION_TYPE \
#     --checkpoint_load_dir $CHECKPOINT_DIR \ 
#     --checkpoint_load_step $CHECKPOINT_STEP \ 
#     --deterministic \ 
#     # Goal
#     --goal_npz $GOAL_NPZ \ 
#     --step_by_one \
#     # Recovery
#     --check_battery \
#     --check_stuck \
#     --handle_keepouts \