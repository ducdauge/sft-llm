# Scaling Sparse Fine-Tuning to Large Language Models

This is the code to replicate the instruction tuning experiments in the paper "Scaling Sparse Fine-Tuning to Large Language Models".

For our Sparse Fine-Tuning (SFT) implementation based on the Hugging Face library, please visit [peft](https://github.com/AlanAnsell/peft).

## Setup
```bash
pip install -r requirements.txt
git submodule update --init --recursive
export SFT_EXPERIMENT_DIR=<path>
```

```bash
./scripts/prepare_train_data.sh
./scripts/prepare_eval_data.sh
```

## Train

```bash
./script/finetune_peft_with_accelerate.sh
```

## Eval

```bash
./script/eval_all.sh
```

## Acknowledgements
Our code and setup for the instruction tuning experiments builds on [open-instruct](https://github.com/allenai/open-instruct).