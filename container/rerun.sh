#!/bin/bash
set -euo pipefail
# Runs evaluation in the container environment, re-running from saved details.
# Called directly or from a SLURM script.
#
# Run from the repository root, not from the container directory.
# Requires a prepared ~/apphome directory with a container image
# and an installed Python virtual environment.
#
# Override the container runtime by setting CONTAINER_CMD before invoking.
#
# Usage:
#   ./rerun.sh <details-path> [options]
#
# Arguments:
#   details-path  Path to the evaluation details directory (required)
#   options       Additional arguments passed to rerun_from_details module
#
# Examples:
#   ./rerun.sh ./outputs/details/mistralai/Ministral-3-14B-Reasoning-2512
#   ./rerun.sh ./outputs/details/mistralai/Ministral-3-14B-Reasoning-2512 --workers 1 --thinking-mode --reasoning-tags "[('[THINK]', '[/THINK]')]"

SCRIPT_DIR=$(dirname "$(realpath "$0")")
source "$SCRIPT_DIR/container-env.sh"

export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

rm -r .litellm_cache || true

CHOME="${APPHOME:-$HOME/apphome}"
echo "Using CHOME=$CHOME"

# Check if details path is provided
if [[ $# -lt 1 ]]; then
    echo "Error: Missing required argument: details-path" >&2
    echo "Usage: $0 <details-path> [options]" >&2
    exit 1
fi

DETAILS_PATH="$1"
shift

# Build the command with all remaining arguments
CONTAINER_CMD="${CONTAINER_CMD:-}"
BASE_CMD="uv run python -m scripts.rerun_from_details ${DETAILS_PATH} $@"

if [[ -n "$CONTAINER_CMD" ]]; then
    # Use custom container runtime
    "$CONTAINER_CMD" exec --nv \
        --home "$CHOME" \
        --bind "$(pwd):/home/sk-bench" \
        --pwd /home/sk-bench \
        --env VIRTUAL_ENV="$CHOME/venv" \
        --env HF_DATASETS_OFFLINE=1 \
        --env PATH="/usr/local/cuda/bin:/usr/local/cuda/lib64:$CHOME/.local/bin:$CHOME/venv/bin:$PATH" \
        "$CHOME/cuda.sif" \
        bash -c "$BASE_CMD"
else
    # Default: run without container wrapper
    echo "Warning: CONTAINER_CMD not set, running directly" >&2
    eval "$BASE_CMD"
fi
