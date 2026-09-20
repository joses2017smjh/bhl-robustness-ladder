#!/bin/bash
set -euo pipefail
stage=${1:?Approach Corridor Full}; iterations=${2:?iterations}; previous=${3:-}
index=${SLURM_ARRAY_TASK_ID:?array index missing}
arms=(Blind Lidar Stereo Both)
arm=${arms[$((index % 4))]}
seed=$((index / 4))
exec bash "$REPO/slurm/inner/weekend_maze.sh" train "$stage" "$arm" "$seed" 1024 "$iterations" "$previous"
