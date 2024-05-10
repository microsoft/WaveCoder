MODEL=microsoft/wavecoder-ds-6.7b
python  eval/mbpp/generate.py \
    --model_path $MODEL \
    --save_path eval/mbpp_generation.json \
    --max_new_tokens 512 \
    --temperature 0.0 \
    --n_samples 1 \
    --batch_size 1 \
    --top_p 1.0 \
