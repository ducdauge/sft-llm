#!/bin/bash

NUM_SHOTS=0
TASK_NAME=codex
DEV_AVAILABLE=false

source parse_eval_args.sh

SAVE_DIR=${SFT_EXPERIMENT_DIR}/codex/${EXPERIMENT_NAME}
mkdir -p $SAVE_DIR
LOG_FILE="$SAVE_DIR/log.txt"

export TOKENIZERS_PARALLELISM=false
python -m eval.codex_humaneval.run_eval \
    --data_file ../data/eval/codex_humaneval/HumanEval.jsonl.gz \
    --eval_pass_at_ks 1 \
    --unbiased_sampling_size_n 20 \
    --temperature 0.1 \
    --save_dir $SAVE_DIR \
    --model $MODEL_NAME \
    --tokenizer $MODEL_NAME \
    $PEFT_ARG \
    $QPEFT_ARG \
    $MERGE_ARG \
    --torch_dtype bfloat16 >> $LOG_FILE 2>&1
