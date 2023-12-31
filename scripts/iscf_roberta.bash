MODEL_DIR=$1
MODEL_EVI_DIR=$2
SPLIT=$3

python run.py --data_dir dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--load_path ${MODEL_EVI_DIR} \
--eval_mode single \
--test_file ${SPLIT}.json \
--test_batch_size 4 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class 97 \
--attn_heads 1 \
--gcn_layers 1 \
--iters 2 \
--use_graph

python run.py --data_dir dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--load_path ${MODEL_DIR} \
--results_path ${MODEL_EVI_DIR} \
--eval_mode fushion \
--test_file ${SPLIT}.json \
--test_batch_size 32 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class 97 \
--attn_heads 1 \
--gcn_layers 1 \
--iters 2 \
--use_graph