#!/bin/bash
#SBATCH --job-name=yaw-probe
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Two 400-iteration probes on CubeToShelf-Blind, seed 0, 1024 envs — the same
# cell and budget as 21191526, which went 9.1 -> 5.0 -> 5.0 with fall 1.000
# because plant_feet was planting a robot lying on its side.
#
#   _0  new yaw, plant_feet off     — does standing up at spawn hold a gait?
#   _1  new yaw, BHL_PLANT_FEET=1   — does planting help once the robot is upright?
#
# Not the 18-cell re-run. One cell, two flags. The 18 follow only if _0 holds.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$WORKSPACE/overrides"

export TASK="TaskV2-BHL-CubeToShelf-Blind-v0"
export SEED=0 MAX_ITER=400 NUM_ENVS=1024

if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
    export RUN_NAME="yaw-cubetoshelf-blind-s0"
    unset BHL_PLANT_FEET || true
    LABEL="new yaw, plant_feet off"
else
    export RUN_NAME="yawplant-cubetoshelf-blind-s0"
    export BHL_PLANT_FEET=1
    LABEL="new yaw, BHL_PLANT_FEET=1"
fi

export OVERRIDE_FILE="$WORKSPACE/overrides/yaw-probe-${SLURM_ARRAY_TASK_ID}.txt"
: > "$OVERRIDE_FILE"

echo "=== node: $(hostname) | $RUN_NAME | $LABEL | envs=$NUM_ENVS iters=$MAX_ITER | $(date) ==="
echo "    BHL_PLANT_FEET=${BHL_PLANT_FEET:-<unset>}  BHL_LEGACY_YAW=${BHL_LEGACY_YAW:-<unset>}"
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
