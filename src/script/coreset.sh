python data/raw_code_collection/main.py \
    --data_path input_example.jsonl \
    --save_path data/seed.jsonl \
    --batch_size 4096 \
    --seed 42 \
    --model_name sentence-transformers/all-roberta-large-v1 \
    --coreset_size 2