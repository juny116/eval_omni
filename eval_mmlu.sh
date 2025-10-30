lm_eval \
    --model local-chat-completions \
    --model_args '{"base_url":"http://host:port/v1/chat/completions","model":"Qwen/Qwen2.5-Omni-7B","tokenizer_backend":"huggingface","num_concurrent":8,"temperature":0,"max_length":8192,"max_gen_toks":2048,"trust_remote_code":true}' \
    --include_path lm_eval_tasks/ \
    --tasks mmlu \
    --num_fewshot 5 \
    --batch_size auto \
    --output_path /path/to/output \
    --log_samples \
    --apply_chat_template True 