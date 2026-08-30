#!/bin/bash
set -euo pipefail
# Runs evaluation in the container environment.
# Called directly or from a SLURM script.
# Usage: run-eval.sh <model-name> [additional-args...]
#
# Run from the repository root, not from the container directory.
# Requires a prepared ~/apphome directory with a container image
# and an installed Python virtual environment.
#
# Override the container runtime by setting CONTAINER_CMD before invoking.

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model-name> [additional-args...]" >&2
    exit 1
fi

SCRIPT_DIR=$(dirname "$(realpath "$0")")
source "$SCRIPT_DIR/container-env.sh"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=1200

rm -r .litellm_cache || true

MODEL_NAME="$1"
OTHER_ARGS=("${@:2}")
CHOME="${APPHOME:-/mnt/data/home/$USER/apphome}"
echo "Using CHOME=$CHOME"
echo "Running evaluation for model: $MODEL_NAME"
echo "Other arguments: ${OTHER_ARGS[*]}"

# Optional passthrough env vars; only forward them when explicitly set.
ENV_ARGS=()
if [[ -n "${VLLM_TOKENIZER_MODE:-}" ]]; then
    ENV_ARGS+=(--env "VLLM_TOKENIZER_MODE=$VLLM_TOKENIZER_MODE")
fi
if [[ -n "${HF_TOKEN:-}" ]]; then
    ENV_ARGS+=(--env "HF_TOKEN=$HF_TOKEN")
fi

"$CONTAINER_CMD" exec --nv \
    --home "$CHOME" \
    --bind "$(pwd):/home/sk-bench" \
    --pwd /home/sk-bench \
    --env VIRTUAL_ENV="$CHOME/venv" \
    --env PATH="/usr/local/cuda/bin:/usr/local/cuda/lib64:$CHOME/.local/bin:$CHOME/venv/bin:$PATH" \
    "${ENV_ARGS[@]}" \
    "$CHOME/cuda.sif" \
    bash container/eval_entry.sh "$MODEL_NAME" "${OTHER_ARGS[@]}"
