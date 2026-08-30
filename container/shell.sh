#!/bin/bash
set -euo pipefail
# Opens an interactive shell in the container environment.
#
# Run from the repository root, not from the container directory.
# Requires a prepared ~/apphome directory with a container image
# and an installed Python virtual environment.
#
# Source container-env.sh or set CONTAINER_CMD to override the runtime.

SCRIPT_DIR=$(dirname "$(realpath "$0")")
source "$SCRIPT_DIR/container-env.sh"

CHOME="${APPHOME:-/mnt/data/home/$USER/apphome}"
echo "Using CHOME=$CHOME"

"$CONTAINER_CMD" shell --nv \
    --home "$CHOME" \
    --bind "$(pwd):/home/sk-bench" \
    --pwd /home/sk-bench \
    --env VIRTUAL_ENV="$CHOME/venv" \
    --env PATH="$CHOME/.local/bin:$CHOME/venv/bin:$PATH" \
    "$CHOME/cuda.sif"
