#!/bin/bash
set -euo pipefail
# Entry point executed INSIDE the container.
# Called by run-eval.sh — do not run directly.
#
# Activates the virtual environment and runs eval_vllm.py
# with W&B logging enabled (offline mode).

export WANDB_PROJECT=skbench
export WANDB_MODE=offline
export GIT_PYTHON_REFRESH=quiet

if [[ $# -lt 1 ]]; then
    echo "Usage: $0 <model-name> [additional-args...]" >&2
    exit 1
fi

MODEL_NAME="$1"
OTHER_ARGS=("${@:2}")

nvcc --version || true

# Ensure Mistral-derived tokenizers (e.g. Step-3.7-Flash) use the corrected
# regex in every process, including vLLM's spawned tensor/data-parallel workers.
uv run python scripts/patch_transformers_mistral_regex.py

CU13_LIB=$(uv run python - <<'PY'
import sysconfig, pathlib
site_packages = sysconfig.get_paths()["purelib"]
cu13 = pathlib.Path(site_packages) / "nvidia" / "cu13" / "lib"
print(cu13 if cu13.is_dir() else "")
PY
)
if [[ -n "$CU13_LIB" ]]; then
    export LD_LIBRARY_PATH="${CU13_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

uv run python eval_vllm.py \
    --model-name "$MODEL_NAME" \
    --wandb \
    --save-details \
    --trust-remote-code \
    "${OTHER_ARGS[@]}"
