#!/bin/bash

# GENERAL PARAMS
ROBOT_IP=localhost
MAX_TIME=600

# LOCAL MODEL INFO
MODEL_TYPE=gc_cql
MODEL=gc_cql_twist_dense_alpha1.0_proprioTrue_hist1_2024_07_19_22_33_54-fine_495k-6
MODELS_FOLDER=cql_models
FINETUNING=finetuned/
IMG_SIZE=64
DEP_STEP=dep_495k
DEP_ENV=bair_loop
CODE_DIR=/home/lydia/repos/
DATA_DIR=/home/lydia/data

# PARAMS FOR SAVING
DATA_SAVE_LOCATION=remote # local, remote, none 
DATA_SAVE_DIR_BASE=$DATA_DIR/deployment # /home/lydia/data/create_data/deployment
# DATA_SAVE_DIR=$DATA_SAVE_DIR_BASE/$MODELS_FOLDER/$FINETUNING$MODEL/$DEP_STEP/$DEP_ENV
DATA_SAVE_DIR=$DATA_DIR/goal_loops/raw/hallway
SAVER_IP=100.87.243.97

# PARAMS FOR AGENT
ACTION_TYPE=gc_cql_remote # gc_cql_local/remote, gc_bc_local/remote, random, inplace, teleop, forward_jerk
CHECKPOINT_BASE_DIR=$DATA_DIR/checkpoints # /home/lydia/data/checkpoints
CHECKPOINT_DIR=$CHECKPOINT_BASE_DIR/$MODELS_FOLDER/$FINETUNING$MODEL
CHECKPOINT_STEP=495000

# PARAMS FOR GOAL 
GOAL_LOOP_BASE=$DATA_DIR/goal_loops # /home/lydia/data/goal_loops
GOAL_NPZ=$GOAL_LOOP_BASE/npz/img_$IMG_SIZE/$DEP_ENV.npz # $CODE_DIR/multinav-rl/multinav/data/bair_loop.npz 


## ADD MANUAL ADVANCE 
MKDIR=FALSE
if [ $MKDIR = "TRUE" ]
then
    mkdir $DATA_SAVE_DIR_BASE/$MODELS_FOLDER/$FINETUNING$MODEL
    mkdir $DATA_SAVE_DIR_BASE/$MODELS_FOLDER/$FINETUNING$MODEL/$DEP_STEP
    mkdir $DATA_SAVE_DIR_BASE/$MODELS_FOLDER/$FINETUNING$MODEL/$DEP_STEP/$DEP_ENV
fi


# FULL THING: pick and chose local / remote data / model combo! 
python $CODE_DIR/multinav-rl/multinav/robot/actor_general.py \
    --robot_ip $ROBOT_IP \
    --data_save_location $DATA_SAVE_LOCATION \
    --data_save_dir $DATA_SAVE_DIR \
    --action_type $ACTION_TYPE \
    --goal_npz $GOAL_NPZ \
    --saver_ip $SAVER_IP \
    --check_battery \
    --check_stuck \
    --handle_keepouts \
    # --checkpoint_load_dir $CHECKPOINT_DIR \
    # --checkpoint_load_step $CHECKPOINT_STEP \
    # --deterministic \
    # --step_by_one \
    # --manually_advance \
    # --max_time $MAX_TIME \

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