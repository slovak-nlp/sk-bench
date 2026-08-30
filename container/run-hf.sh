#!/bin/bash
set -euo pipefail
# Runs an `hf` command inside the Apptainer/Singularity container.
# Mirrors container/run-eval.sh but for arbitrary `hf` subcommands.
#
# Usage: bash container/run-hf.sh <hf-subcommand> [args...]
# Example: bash container/run-hf.sh download meta-llama/Llama-2-7b-hf
#
# Run from the repository root. Requires a prepared ~/apphome directory
# with a container image and an installed Python virtual environment.

SCRIPT_DIR=$(dirname "$(realpath "$0")")
source "$SCRIPT_DIR/container-env.sh"

CHOME="${APPHOME:-/mnt/data/home/$USER/apphome}"
echo "Using CHOME=$CHOME"

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <hf-subcommand> [args...]" >&2
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
    hf "$@"
