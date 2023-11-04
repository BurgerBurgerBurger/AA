LOAD_DIR=$1

python run.py --data_dir dataset/docred \
--transformer_type roberta \
--model_name_or_path roberta-large \
--load_path ${LOAD_DIR} \
--eval_mode single \
--test_file train_revised.json \
--test_batch_size 4 \
--num_labels 4 \
--evi_thresh 0.2 \
--num_class 97 \
--use_graph \
--save_attn
