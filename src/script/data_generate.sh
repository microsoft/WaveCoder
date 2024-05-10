set -ex

python data/llm_gen_dis/main.py \
    --source_data_path data/seed.jsonl \
    --gen_prompt_path data/llm_gen_dis/prompt/generator.txt \
    --dis_prompt_path data/llm_gen_dis/prompt/discriminator.txt \
    --good_case_path data/llm_gen_dis/fewshot_case/good_case \
    --bad_case_path data/llm_gen_dis/fewshot_case/bad_case \
    --data_stream_path data/result.txt \
    --save_json \
    --gen_max_token 800 \
    --sample_size 1 \
    --output_data_path data/result.json \
    --openai_key your_key \
    --openai_url url \
    --openai_model your_model_name \