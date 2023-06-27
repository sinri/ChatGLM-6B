#!/usr/bin/env bash

PRE_SEQ_LEN=128
LR=2e-2

#train_base_model=THUDM/chatglm-6b
train_base_model=/mnt/e/OneDrive/Leqee/ai/repo/THUDM/chatglm-6b

train_file=/mnt/e/sinri/ChatGLM-6B/workspace/train/plan_1/AdvertiseGen/train.json
validation_file=/mnt/e/sinri/ChatGLM-6B/workspace/train/plan_1/AdvertiseGen/dev.json

# output/adgen-chatglm-6b-pt-$PRE_SEQ_LEN-$LR
output_dir=/mnt/e/sinri/ChatGLM-6B/workspace/train/plan_1/output

prompt_column=content
response_column=summary

CUDA_VISIBLE_DEVICES=0 python3 /mnt/e/sinri/ChatGLM-6B/ptuning/main.py \
    --do_train \
    --train_file $train_file \
    --validation_file $validation_file \
    --prompt_column $prompt_column \
    --response_column $response_column \
    --overwrite_cache \
    --model_name_or_path $train_base_model \
    --output_dir $output_dir \
    --overwrite_output_dir \
    --max_source_length 64 \
    --max_target_length 64 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 16 \
    --predict_with_generate \
    --max_steps 3000 \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate $LR \
    --pre_seq_len $PRE_SEQ_LEN
#    --quantization_bit 4

