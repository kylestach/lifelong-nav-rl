MODEL=gc_cql_twist_dense_alpha1.0_proprioTrue_hist1_2024_07_19_22_33_54
MODEL_TYPE=gc_cql
MODEL_FOLDER=cql_models/history
FOLDER=fine_495k
STEP=495000
COUNT=11

CODE_DIR=/nfs/nfs2/users/lydia/repos 
DATA_SAVE_DIR=gs://gnm-data-c2/create/finetuning/$MODEL/$FOLDER/run_$COUNT
DATA_LOAD_DIR=gs://gnm-data-c2/gnm_dataset
CHECKPOINT_LOAD_DIR=gs://gnm-checkpoints-c2/$MODEL_FOLDER/$MODEL/
CHECKPOINT_SAVE_DIR=gs://gnm-checkpoints-c2/$MODEL_FOLDER/finetuned/


XLA_PYTHON_CLIENT_MEM_FRACTION=0.9  \
    python $CODE_DIR/multinav-rl/multinav/training/train.py \
    --wait_data 20 \
    --data_mix gnm \
    --data_dir $DATA_LOAD_DIR \
    --data_save_dir $DATA_SAVE_DIR \
    --wandb_name $MODEL-$FOLDER-$COUNT \
    --checkpoint_load_dir $CHECKPOINT_LOAD_DIR \
    --checkpoint_load_step $STEP \
    --checkpoint_save_dir $CHECKPOINT_SAVE_DIR \
    --checkpoint_interval 500 \
    --model_config $CODE_DIR/multinav-rl/multinav/training/model_config.py:$MODEL_TYPE \
    --model_config.wandb_proj cql_models_finetuning \
    --offline_data_config.reward_type dense \
    --online_data_config.reward_type dense \
    --model_config.agent_config.cql_alpha 0.3 \
    --model_config.train_buffer_size 10000 \
    --model_config.agent_config.critic_use_proprio \
    --model_config.agent_config.critic_feed_actions \
    --prioritize_space \
    --online_training \


# In another window, run 
# on create: XLA_PYTHON_CLIENT_PREALLOCATE=false python /home/create/repos/multinav-rl/multinav/robot/train_actor.py --trainer_ip 100.80.219.34 --robot_ip localhost --goal_dir  /home/create/misc/data/goal_loops/bair_loop.npz --seed 65 --inplace True
# on oppenehiemer: XLA_PYTHON_CLIENT_PREALLOCATE=false python /home/lydia/repos/multinav-rl/multinav/robot/train_actor.py --trainer_ip localhost --robot_ip 192.168.68.57 --goal_dir  /home/lydia/real_data/goal_npzs/bair_loop.npz --seed 65 --inplace True
# TPU: JAX_PLATFORMS='' XLA_PYTHON_CLIENT_PREALLOCATE=false python /nfs/nfs2/users/lydia/repos/multinav-rl/multinav/robot/train_actor.py --trainer_ip localhost --robot_ip 100.121.96.41 --goal_dir  /nfs/nfs2/users/lydia/repos/multinav-rl/multinav/data/bair_loop.npz --seed 65