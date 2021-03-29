#!/usr/bin/env bash
export BERT_BASE_DIR=./chinese_L-12_H-768_A-12
export MY_DATASET=./data/XQ

case $1 in
train)
  python3 run_classifier.py \
    --task_name=nlpccxq \
    --do_train=true \
    --data_dir=$MY_DATASET \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=60 \
    --train_batch_size=128 \
    --learning_rate=5e-5 \
    --num_train_epochs=10.0 \
    --output_dir=./xq_output/ >>./run_xq_train.log 2>&1 
;;
predict)
  python3 run_classifier.py \
    --task_name=nlpccxq \
    --do_predict=true \
    --data_dir=$MY_DATASET \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=60 \
    --train_batch_size=128 \
    --learning_rate=5e-5 \
    --num_train_epochs=3.0 \
    --output_dir=./xq_output/ >>./run_xq_predict.log 2>&1 
;;
eval)
  python3 run_classifier.py \
    --task_name=nlpccxq \
    --do_eval=true \
    --data_dir=$MY_DATASET \
    --vocab_file=$BERT_BASE_DIR/vocab.txt \
    --bert_config_file=$BERT_BASE_DIR/bert_config.json \
    --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
    --max_seq_length=60 \
    --train_batch_size=128 \
    --learning_rate=5e-5 \
    --num_train_epochs=3.0 \
    --output_dir=./xq_output/ >>./run_xq_eval.log 2>&1 
;;
*)
    echo "Usage: ./run_xq.sh train|predict|eval"
;;
esac