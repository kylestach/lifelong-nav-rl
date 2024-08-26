#!/bin/bash

ROBOT_IP=localhost
IMG_TYPE=numpy
IMG_SIZE=64
SAVE_DIR=/home/jackal/data/goal_loops/npz/img_$IMG_SIZE
ENV_NAME=hallway_3
SAVE_PATH=$SAVE_DIR/$ENV_NAME.npz

python /home/jackal/repos/multinav-rl/multinav/data/manual_goal_loop.py \
    --robot_ip $ROBOT_IP \
    --img_type $IMG_TYPE \
    --img_size $IMG_SIZE \
    --save_path $SAVE_PATH \