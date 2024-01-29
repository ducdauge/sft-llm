echo "Downloading the gpt4-llm dataset..."
wget -P data/raw_train/gpt4_alpaca/ https://github.com/Instruction-Tuning-with-GPT-4/GPT-4-LLM/raw/main/data/alpaca_gpt4_data.json

echo "Processing datasets..."
python scripts/reformat_datasets.py --raw_data_dir data/raw_train/ --output_dir data/processed/ --dataset gpt4_alpaca
