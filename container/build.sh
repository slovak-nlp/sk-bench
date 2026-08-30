#!/bin/bash
set -euo pipefail
SCRIPT_DIR=$(dirname "$(realpath "$0")")
# Prepares a directory with the container image.
# The directory later serves as the container home directory.
#
# Source container-env.sh or set CONTAINER_CMD to override the runtime.

source "$SCRIPT_DIR/container-env.sh"

mkdir -p ~/apphome
cd ~/apphome
echo "Building from definition: $SCRIPT_DIR/mycontainer.def"
"$CONTAINER_CMD" build --fakeroot cuda.sif "$SCRIPT_DIR/mycontainer.def"
