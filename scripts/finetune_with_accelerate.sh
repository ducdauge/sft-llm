export CUDA_VISIBLE_DEVICES=0

MODEL_SIZE=7b

LEARNING_RATE=2e-5

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

HP_STRING="lr-${LEARNING_RATE}"

NUM_GPUS=1
BATCH_SIZE_PER_GPU=1
TOTAL_BATCH_SIZE=128
GRADIENT_ACC_STEPS=$(($TOTAL_BATCH_SIZE/$NUM_GPUS/$BATCH_SIZE_PER_GPU))
echo "Training llama model ${MODEL_SIZE} using $NUM_GPUS GPUs, $BATCH_SIZE_PER_GPU batch size per GPU, $GRADIENT_ACC_STEPS gradient accumulation steps"

OUTPUT_DIR=${SFT_EXPERIMENT_DIR}/${DATASET}_${MODEL_SIZE}_vanilla_${HP_STRING}
mkdir -p $OUTPUT_DIR

accelerate launch \
    --mixed_precision bf16 \
    --num_machines 1 \
    --num_processes $NUM_GPUS \
    --use_deepspeed \
    --deepspeed_config_file ds_configs/stage3_offloading_accelerate.conf \
    finetune/finetune.py \
    --model_name_or_path meta-llama/Llama-2-${MODEL_SIZE}-hf \
    --use_flash_attn \
    --tokenizer_name meta-llama/Llama-2-${MODEL_SIZE}-hf \
    --use_slow_tokenizer \
    --train_file $DATASET_ARG \
    --max_seq_length 2048 \
    --preprocessing_num_workers 16 \
    --per_device_train_batch_size $BATCH_SIZE_PER_GPU \
    --gradient_accumulation_steps $GRADIENT_ACC_STEPS \
    --learning_rate $LEARNING_RATE \
    --lr_scheduler_type linear \
    --warmup_ratio 0.03 \
    --weight_decay 0. \
    --num_train_epochs 2 \
    --output_dir ${SFT_EXPERIMENT_DIR}/flanv2_50K_${MODEL_SIZE}_fullft \
    --with_tracking \
    --report_to tensorboard \
    --logging_steps 1 >> $OUTPUT_DIR/log.txt 2>&1