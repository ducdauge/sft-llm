echo "Downloading the flan_v2 collection for tulu v2 with 50K subsampled examples."
mkdir -p data/raw_train/flan_v2/
wget -O data/raw_train/flan_v2/tulu_v2_resampled_flan_50k.jsonl https://beaker.org/api/v3/datasets/01HBS0N5ZSDF5AECA9VMB1RKXQ/files/flan_v2_resampled_50k.jsonl

echo "Downloading the gpt4-llm dataset..."
wget -P data/raw_train/gpt4_alpaca/ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data.json

echo "Processing datasets..."
python scripts/reformat_datasets.py --raw_data_dir data/raw_train/ --output_dir data/processed/
