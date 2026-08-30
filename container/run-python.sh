#!/bin/bash
set -euo pipefail
# Runs a Python script inside the Apptainer/Singularity container via `uv run`.
# Mirrors container/run-eval.sh and container/run-hf.sh.
#
# Usage: bash container/run-python.sh <script.py> [args...]
# Example: bash container/run-python.sh scripts/analyze_chat_template.py org/model
#
# Run from the repository root. Requires a prepared ~/apphome directory
# with a container image and an installed Python virtual environment.

SCRIPT_DIR=$(dirname "$(realpath "$0")")
source "$SCRIPT_DIR/container-env.sh"

CHOME="${APPHOME:-/mnt/data/home/$USER/apphome}"
echo "Using CHOME=$CHOME"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <script.py> [args...]" >&2
    exit 1
fi

"$CONTAINER_CMD" exec --nv \
    --home "$CHOME" \
    --bind "$(pwd):/home/sk-bench" \
    --pwd /home/sk-bench \
    --env VIRTUAL_ENV="$CHOME/venv" \
    --env PATH="/usr/local/cuda/bin:/usr/local/cuda/lib64:$CHOME/.local/bin:$CHOME/venv/bin:$PATH" \
    --env HF_TOKEN="${HF_TOKEN:-}" \
    "$CHOME/cuda.sif" \
    uv run python "$@"
