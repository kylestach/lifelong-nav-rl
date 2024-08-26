#!/bin/bash
# GENERIC
MODEL=gc_cql_twist_dense_alpha1.0_proprioTrue_hist1_2024_07_19_22_33_54-fine_100k-2
MODEL_TYPE=cql_models

BASE_DIR=gs://gnm-checkpoints-c2/$MODEL_TYPE/history/finetuned

MKDIR=TRUE
if [ $MKDIR = "TRUE" ]
then
    mkdir /home/lydia/data/checkpoints/$MODEL_TYPE/finetuned/$MODEL # CHANGE WHEN NEEDED 
    mkdir /home/lydia/data/create_data/deployment/finetuned/$MODEL_TYPE
    mkdir /home/lydia/data/create_data/deployment/finetuned/$MODEL_TYPE/$MODEL
    mkdir /home/lydia/data/create_data/deployment/finetuned/$MODEL_TYPE/$MODEL/dep_25k
fi


gsutil -m cp -r \
 $BASE_DIR/$MODEL/25000 \
 /home/lydia/data/checkpoints/$MODEL_TYPE/finetuned/$MODEL

# gsutil -m cp -r \
#  $BASE_DIR/$MODEL/102000 \
#  /home/lydia/data/checkpoints/$MODEL_TYPE/finetuned/$MODEL

#  gsutil -m cp -r \
#  $BASE_DIR/$MODEL/150000 \
#  /home/lydia/data/checkpoints/$MODEL_TYPE/finetuned/$MODEL