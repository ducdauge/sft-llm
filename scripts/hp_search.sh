#!/bin/bash

EVAL_DEV_ARG="--eval_dev"
TIME="15:00"

dispatch() {
    peft_name=$1
    peft_path=${SFT_EXPERIMENT_DIR}/${peft_name}
    if [[ -f "$peft_path/adapter_model.bin" ]]; then
        for num_shots in 5; do
            exp_name=${peft_name}_${num_shots}shot_${EVAL_SET}
            echo $exp_name
            sbatch $SFT_SLURM_ARGS --time=$TIME ./eval_mmlu.sh \
                --experiment_name $exp_name \
                --model_name meta-llama/Llama-2-${MODEL_SIZE}-hf \
                $EVAL_DEV_ARG \
                --peft $peft_path \
                --num_shots $num_shots
        done
    fi
}

for MODEL_SIZE in 7b 13b; do

    # SFT-AG and LoRA
    for lr in 1e-4 3e-5 1e-5 3e-6; do
        for rank in 8 16 32 64; do
            for method in sft_rigl lora; do
                for wd in 1 3 10 30; do
                    dispatch flanv2_${MODEL_SIZE}_${method}_r-${rank}_lr-${lr}_wd-${wd}
                done
            done
        done
    done

    # SFT-MA
    for lr in 2e-3 1e-3 7e-4 4e-4; do
        for rank in 8 16 32 64; do
            for wd in 1 3 10 30; do
                dispatch flanv2_${MODEL_SIZE}_sft_sm3_r-${rank}_lr-${lr}_wd-${wd}
            done
        done
    done

    # (IA)^3
    for lr in 3e-4 1e-4 3e-5 1e-5; do
        dispatch flanv2_${MODEL_SIZE}_ia3_lr-${lr}
    done

    # Prefix Tuning
    for lr in 1e-2 3e-2 1e-1 3e-1; do
        for plength in 10 20 40 80; do
            dispatch flanv2_${MODEL_SIZE}_pt_lr-${lr}_plength-${plength}
        done
    done
done
