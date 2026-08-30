#!/bin/bash

# export API_KEY=
export WANDB_PROJECT=skbench
export WANDB_MODE=offline

MODEL_NAME="openrouter/openai/gpt-5.4-mini:nitro"
BASE_URL="https://openrouter.ai/api/v1"
TASKS_FILE="./subset.txt"

uv run python -m eval_litellm \
    --model-name $MODEL_NAME \
    --base-url $BASE_URL \
    --tasks-file $TASKS_FILE \
    --save-details \
    --max-tokens 8192 \
    --reasoning-effort low \
    --wandb \
