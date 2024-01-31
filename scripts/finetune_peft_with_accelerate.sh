#!/bin/bash

MODEL_SIZES=( 7b 13b )
MODEL_SIZE_IDX=1
MODEL_SIZE=${MODEL_SIZES[MODEL_SIZE_IDX]}
PARAMS_PER_RANK_BY_MODEL_SIZE=( 2498560 3911680 )
PARAMS_PER_RANK=${PARAMS_PER_RANK_BY_MODEL_SIZE[MODEL_SIZE_IDX]}

RANKS=( 8 16 32 64 )
RANK_IDX=3
RANK=${RANKS[RANK_IDX]}
NUM_TUNABLE_WEIGHTS=$((RANK * PARAMS_PER_RANK))

WEIGHT_DECAYS=( 1 3 10 30 )
WEIGHT_DECAY_IDX=3
WEIGHT_DECAY=${WEIGHT_DECAYS[WEIGHT_DECAY_IDX]}

PREFIX_LENGTHS=( 10 20 40 80 )
PREFIX_LENGTH_IDX=3
PREFIX_LENGTH=${PREFIX_LENGTHS[PREFIX_LENGTH_IDX]}

LEARNING_RATE=1e-5

DATASET="tulu_v2"
if [[ "$DATASET" == "tulu_v2" ]]; then
    DATASET_ARG="--dataset_name allenai/tulu-v2-sft-mixture"
elif [[ "$DATASET" == "flan_v2" ]]; then
    DATASET_ARG="--dataset_name ostapeno/tulu_v2_flan_v2_subset"
elif [[ "$DATASET" == "gpt4_alpaca" ]]; then
    DATASET_ARG="--train_file data/processed/gpt4_alpaca/gpt4_alpaca_data.jsonl"
else
    echo "Unsupported dataset \"$DATASET\"" >&2
    exit 1
fi

PEFT_METHOD="sft"
SFT_METHOD="rigl"
if [[ "$PEFT_METHOD" == "sft" ]]; then
    METHOD_NAME="sft_${SFT_METHOD}"
    HP_STRING="r-${RANK}_lr-${LEARNING_RATE}_wd-${WEIGHT_DECAY}"
elif [[ "$PEFT_METHOD" == "lora" ]]; then
    WEIGHT_DECAY=0
    METHOD_NAME="lora"
    HP_STRING="r-${RANK}_lr-${LEARNING_RATE}"
elif [[ "$PEFT_METHOD" == "ia3" ]]; then
    WEIGHT_DECAY=0
    METHOD_NAME="ia3"
    HP_STRING="lr-${LEARNING_RATE}"
elif [[ "$PEFT_METHOD" == "pt" ]]; then
    WEIGHT_DECAY=0
    METHOD_NAME="pt"
    HP_STRING="lr-${LEARNING_RATE}_plength-${PREFIX_LENGTH}"
else
    echo "Unsupported PEFT method $PEFT_METHOD" >&2
    exit 1
fi

QPEFT=false
if [[ "$QPEFT" == true ]]; then
    QPEFT_ARG="--use_qpeft"
    PEFT_DTYPE="bfloat16"
    METHOD_NAME="q${METHOD_NAME}"
else
    QPEFT_ARG=""
    PEFT_DTYPE="float32"
fi

OUTPUT_DIR=${SFT_EXPERIMENT_DIR}/${DATASET}_${MODEL_SIZE}_${METHOD_NAME}_${HP_STRING}
mkdir -p $OUTPUT_DIR

NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps" | tee -a $OUTPUT_DIR/log.txt

accelerate launch \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    finetune/finetune.py \
    --model_name_or_path meta-llama/Llama-2-${MODEL_SIZE}-hf \
    $DATASET_ARG \
    --resume_from_checkpoint yes \
    --use_flash_attn \
    --seed 42 \
    --peft_tuner $PEFT_METHOD \
    $QPEFT_ARG \
    --sft_num_tunable_weights ${NUM_TUNABLE_WEIGHTS} \
    --peft_dtype $PEFT_DTYPE \
    --sft_selection $SFT_METHOD \
    --lora_rank $RANK \
    --lora_alpha 16 \
    --lora_dropout 0.1 \
    --pt_num_virtual_tokens $PREFIX_LENGTH \
    --tokenizer_name meta-llama/Llama-2-${MODEL_SIZE}-hf \
    --use_slow_tokenizer \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --checkpointing_steps 50 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate ${LEARNING_RATE} \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay $WEIGHT_DECAY \
    --num_train_epochs 2 \
    --output_dir $OUTPUT_DIR \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 2>&1 | tee -a $OUTPUT_DIR/log.txt
