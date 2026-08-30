#!/bin/bash
set -euo pipefail
# Script to install dependencies inside a container.
#
# Installs uv, vllm, and lighteval to the container local home directory.
# Requires VIRTUAL_ENV to be set by the container runtime.
#
# Result: uv install + a venv directory with Python dependencies in the container home directory.
# https://github.com/vllm-project/vllm/issues/31018

SCRIPT_DIR=$(dirname "$(realpath "$0")")
source "$SCRIPT_DIR/container-env.sh"

curl -LsSf https://astral.sh/uv/install.sh | sh
#source ~/.local/bin/env

echo "VIRTUAL_ENV=$VIRTUAL_ENV"
uv venv "$VIRTUAL_ENV" --python 3.12.3
source "$VIRTUAL_ENV/bin/activate"

uv sync --active

# Download required NLTK corpora for METEOR metric
python -c "import nltk; nltk.download('punkt_tab'); nltk.download('wordnet'); nltk.download('omw-1.4')"

echo "Inside ${CONTAINER_NAME} shell, run:"
echo "  wandb login"
echo "  hf auth login"
