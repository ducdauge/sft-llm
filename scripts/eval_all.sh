#!/bin/bash

MODEL_SIZE=13b
MODEL_NAME=meta-llama/Llama-2-${MODEL_SIZE}-hf

TASKS_AND_SHOTS=("mmlu 5" "tydiqa 1" "bbh 3" "gsm 8" "codex 0")
NUM_JOBS=${#TASKS_AND_SHOTS[@]}

PEFT=tulu_v2_13b_sft_rigl_r-64_lr-1e-5_wd-30

if [[ "$PEFT" != "" ]]; then
    if [[ ! -f "$SFT_EXPERIMENT_DIR/$PEFT/adapter_model.bin" ]]; then
        echo "No PEFT found at $PEFT" >&2
        exit 1
    fi
    PEFT_ARG="--peft $SFT_EXPERIMENT_DIR/$PEFT"
    NAME="$PEFT"
else
    PEFT_ARG=""
    NAME="${MODEL_SIZE}_vanilla"
fi

QPEFT=false

if [[ "$QPEFT" == true ]]; then
    QPEFT_ARG="--qpeft"
    NO_MERGE_ARG="--no_merge"
else
    QPEFT_ARG=""
    NO_MERGE_ARG=""
fi


for ((i=0;i<NUM_JOBS;i++)); do
    TS=(${TASKS_AND_SHOTS[i]})
    sbatch $SFT_SLURM_ARGS ./scripts/eval_${TS[0]}.sh \
        --experiment_name ${NAME}_${TS[1]}shot_test \
        --model_name $MODEL_NAME \
        $PEFT_ARG \
        $QPEFT_ARG \
        $NO_MERGE_ARG \
        --num_shots ${TS[1]}
done
