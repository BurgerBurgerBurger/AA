#!/bin/bash
TYPE=$1
TEACHER_DIR=$2
LAMBDA=$3
SEED=$4

NAME=${TYPE}_lambda${LAMBDA}_seed${SEED}
python run.py --do_train \
--data_dir dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--train_file train_revised.json \
--dev_file dev_revised.json \
--teacher_sig_path ${TEACHER_DIR} \
--save_path ${NAME} \
--train_batch_size 4 \
--test_batch_size 8 \
--gradient_accumulation_steps 1 \
--num_labels 4 \
--lr_transformer 3e-5 \
--max_grad_norm 1.0 \
--evi_thresh 0.2 \
--attn_lambda ${LAMBDA} \
--warmup_ratio 0.06 \
--num_train_epochs 30.0 \
--seed ${SEED} \
--num_class 97 \
--use_graph
