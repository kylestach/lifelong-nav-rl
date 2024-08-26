MODEL_TYPE=gc_cql

CODE_DIR=/nfs/nfs2/users/lydia/repos 
DATA_LOAD_DIR=gs://gnm-data-c2/gnm_dataset
CHECKPOINT_SAVE_DIR=gs://gnm-checkpoints-c2/cql_models/history/

REWARD=dense
ACTION=twist
ALPHA=0.001
HISTORY=5
IMG=64

# source $CODE_DIR/multinav-rl/multinav/scripts/tpu_helpers/tpu_12.sh

XLA_PYTHON_CLIENT_MEM_FRACTION=0.95 \
    python $CODE_DIR/multinav-rl/multinav/training/train.py \
    --data_mix gnm \
    --data_dir $DATA_LOAD_DIR \
    --checkpoint_save_dir $CHECKPOINT_SAVE_DIR \
    --checkpoint_interval 5000 \
    --wandb_name $MODEL_TYPE-$ACTION-$REWARD-alpha$ALPHA-hist$HISTORY\
    --model_config $CODE_DIR/multinav-rl/multinav/training/model_config.py:$MODEL_TYPE \
    --model_config.batch_size 256 \
    --offline_data_config.reward_type $REWARD \
    --offline_data_config.action_type $ACTION \
    --model_config.agent_config.cql_alpha $ALPHA \
    --model_config.agent_config.history_len $HISTORY \
    --model_config.image_size $IMG \
    --model_config.agent_config.gamma 0 \
    --model_config.agent_config.critic_use_proprio \
    --model_config.agent_config.critic_feed_actions \
    # --model_config.validate "0.05"