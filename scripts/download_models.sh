#!/bin/bash
set -euo pipefail

# Download all models listed in a file to warm up the local Hugging Face cache.
MODELS_FILE="${1:-models_list.txt}"

if [[ ! -f "$MODELS_FILE" ]]; then
    echo "Models file '$MODELS_FILE' not found!" >&2
    exit 1
fi

while IFS= read -r MODEL; do
    [[ -z "$MODEL" || "$MODEL" =~ ^[[:space:]]*# ]] && continue

    echo "Loading model: $MODEL"
    hf download "$MODEL"
done < "$MODELS_FILE"
