CODE_DIR=/home/jackal/repos/
GOAL_DIR=/home/jackal/data/goal_loops/npz/img_64/outdoor_morning_6.npz
SERVER_IP=localhost

python /home/jackal/visualnav-transformer/vint_model_deployment.py \
    --max_time 450 \
    --server_ip $SERVER_IP \
    --goal_dir $GOAL_DIR \