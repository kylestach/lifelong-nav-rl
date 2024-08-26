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
DEP_ENV=hallway_3
CODE_DIR=/home/jackal/repos # /home/lydia/repos/
DATA_DIR=/home/jackal/data # /home/create/misc/data 

# PARAMS FOR SAVING
DATA_SAVE_LOCATION=none # local, remote, none 

# PARAMS FOR AGENT
ACTION_TYPE=gc_cql_local # gc_cql_local/remote, gc_bc_local/remote, random, inplace, teleop, forward_jerk
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
python $CODE_DIR/multinav-rl/multinav/robot/actor.py \
    --robot_ip $ROBOT_IP \
    --robot_type jackal \
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