#!/bin/bash

NUM_SHOTS=8
TASK_NAME=gsm
DEV_AVAILABLE=false

source scripts/parse_eval_args.sh

SAVE_DIR=${SFT_EXPERIMENT_DIR}/gsm/${EXPERIMENT_NAME}_cot
mkdir -p $SAVE_DIR
LOG_FILE="$SAVE_DIR/log.txt"

python -m eval.gsm.run_eval \
    --data_dir data/eval/gsm \
    --max_num_examples 200 \
    --save_dir ${SAVE_DIR} \
    --model $MODEL_NAME \
    --tokenizer $MODEL_NAME \
    $PEFT_ARG \
    $QPEFT_ARG \
    $MERGE_ARG \
    --n_shot $NUM_SHOTS \
    --torch_dtype bfloat16 >> ${LOG_FILE} 2>&1
