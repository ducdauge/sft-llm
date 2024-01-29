#!/bin/bash

NUM_SHOTS=3
TASK_NAME=bbh
DEV_AVAILABLE=false

source scripts/parse_eval_args.sh

SAVE_DIR=${SFT_EXPERIMENT_DIR}/bbh/${EXPERIMENT_NAME}
mkdir -p $SAVE_DIR
LOG_FILE="$SAVE_DIR/log.txt"

python -m eval.bbh.run_eval \
    --data_dir ../data/eval/bbh \
    --save_dir ${SAVE_DIR}_cot \
    --model $MODEL_NAME \
    --tokenizer $MODEL_NAME \
    $PEFT_ARG \
    $QPEFT_ARG \
    $MERGE_ARG \
    --max_num_examples_per_task 40 \
    --torch_dtype bfloat16 >> $LOG_FILE 2>&1
