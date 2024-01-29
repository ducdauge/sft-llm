#!/bin/bash

NUM_SHOTS=0
TASK_NAME=mmlu
DEV_AVAILABLE=true

source parse_eval_args.sh

SAVE_DIR=${SFT_EXPERIMENT_DIR}/mmlu/${EXPERIMENT_NAME}
mkdir -p $SAVE_DIR
LOG_FILE="$SAVE_DIR/log.txt"

python -m eval.mmlu.run_eval \
    $EVAL_DEV_ARG \
    --ntrain $NUM_SHOTS \
    --data_dir ../data/eval/mmlu \
    --save_dir $SAVE_DIR \
    --model_name_or_path $MODEL_NAME \
    --tokenizer_name_or_path $MODEL_NAME \
    $PEFT_ARG \
    $QPEFT_ARG \
    $MERGE_ARG \
    --torch_dtype bfloat16 \
    --eval_batch_size 4 >> $LOG_FILE 2>&1
