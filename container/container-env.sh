#!/bin/bash
# Shared container runtime detection.
# Source this file in scripts that need to run apptainer or singularity.
#
# Sets:
#   CONTAINER_CMD    "apptainer" or "singularity" (prefers apptainer)
#   CONTAINER_NAME   human-readable name for echo messages
#
# Override by setting CONTAINER_CMD before sourcing:
#   CONTAINER_CMD=singularity source container/container-env.sh

set -euo pipefail

if [[ -z "${CONTAINER_CMD:-}" ]]; then
    if command -v apptainer &>/dev/null; then
        CONTAINER_CMD="apptainer"
    elif command -v singularity &>/dev/null; then
        CONTAINER_CMD="singularity"
    else
        # Try loading the module on HPC clusters
        module load singularity 2>/dev/null || true
        if command -v singularity &>/dev/null; then
            CONTAINER_CMD="singularity"
        else
            echo "ERROR: Neither apptainer nor singularity found." >&2
            echo "Install one of them or set CONTAINER_CMD to the full path." >&2
            exit 1
        fi
    fi
fi

export CONTAINER_CMD
CONTAINER_NAME="${CONTAINER_CMD}"  # "apptainer" or "singularity"

echo "Using container runtime: ${CONTAINER_NAME}"
