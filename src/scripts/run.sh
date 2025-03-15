cd src/r1-v

export PYTHONPATH=$PYTHONPATH:/workspace/R1-V/src/r1-v/src
export DEBUG_MODE="true" # Enable Debug if you want to see the rollout of model during RL
export LOG_PATH="./debug_log_2b.txt"
export PYTORCH_CUDA_ALLOC_CONF="max_split_size_mb:64"

python src/open_r1/grpo.py \
    --output_dir /workspace/R1-V/output \
    --model_name_or_path Qwen/Qwen2-VL-2B-Instruct \
    --dataset_name leonardPKU/GEOQA_R1V_Train_8K \
    --deepspeed local_scripts/zero2.json \
    --max_prompt_length 512 \
    --max_completion_length 512 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --logging_steps 1 \
    --fp16 \
    --report_to wandb \
    --gradient_checkpointing false \
    --attn_implementation flash_attention_2 \
    --max_pixels 401408 \
    --num_train_epochs 2 \
    --run_name Qwen2-VL-2B-GRPO-GEOQA_R1V_Train_8K \
    --save_steps 100 \
    --save_only_model true \
    --num_generations 1  # number of outputs G in grpo, reduce it would lead to faster training and smaller memory cost but higher variance
