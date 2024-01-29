#!/bin/bash

NUM_SHOTS=1
TASK_NAME=tydiqa
DEV_AVAILABLE=false

source scripts/parse_eval_args.sh

SAVE_DIR=${SFT_EXPERIMENT_DIR}/tydiqa/${EXPERIMENT_NAME}
mkdir -p $SAVE_DIR
LOG_FILE="$SAVE_DIR/log.txt"

python -m eval.tydiqa.run_eval \
    --data_dir ../data/eval/tydiqa/ \
    --n_shot $NUM_SHOTS \
    --max_num_examples_per_lang 100 \
    --max_context_length 1024 \
    --save_dir $SAVE_DIR \
    --model $MODEL_NAME \
    --tokenizer $MODEL_NAME \
    $PEFT_ARG \
    $QPEFT_ARG \
    $MERGE_ARG \
    --torch_dtype bfloat16 \
    --eval_batch_size 5 >> $LOG_FILE 2>&1
