#!/bin/bash
# Pre-flight for the job arrays: exercises env forwarding, the vendored
# entrypoint, and overlay task registration at 32 envs / 3 iterations.
set -euo pipefail
cd "$UPSTREAM"
echo "### forwarded vars: TASK=$TASK RUN_NAME=$RUN_NAME SEED=$SEED ###"
exec bash "$REPO/slurm/inner/train.sh"
