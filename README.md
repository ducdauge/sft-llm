# Scaling Sparse Fine-Tuning to Large Language Models

This is the code to replicate the instruction tuning experiments in the paper *Scaling Sparse Fine-Tuning to Large Language Models*.

For our Sparse Fine-Tuning (SFT) implementation based on the Hugging Face library, please visit [peft](https://github.com/AlanAnsell/peft).

## Setup
```bash
pip install -r requirements.txt
git submodule update --init --recursive
export SFT_EXPERIMENT_DIR=./results
```

Next, prepare train and eval data. 

Note that our original experiments were run based on [Flan v2 50K sub-mixture](https://beaker.org/api/v3/datasets/01HBS0N5ZSDF5AECA9VMB1RKXQ/files/flan_v2_resampled_50k.jsonl) sourced from Beaker, which now requires authorisation. Hence, we now rely on an unofficial snapshot from Hugging Face Hub.

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