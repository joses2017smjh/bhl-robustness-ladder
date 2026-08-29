#!/bin/bash
#SBATCH --job-name=v2-trainsmoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Three iterations through the REAL training path, one per vision condition.
#
# `86_task_v2_smoke` builds environments and never touches scripts/train.py, so
# it passed 9/9 through all three training failures: the import order, the wrong
# rsl-rl version, and the wrong runner-config schema. Every one of those lives
# in the training path, and a gate that does not execute the thing it is gating
# is decoration.
#
# Three iterations at 64 envs is a couple of minutes and exercises the whole
# chain -- shim, library, runner config, first optimiser step. If this passes,
# an 8,000-iteration arm fails for a reason worth learning about.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$WORKSPACE/overrides"

# Arm 3 is the one-robot ball control, smoked alongside the vision conditions
# because it is a different action-manager shape and could break on its own.
VIS=(Blind Depth Rgb)
if [ "$SLURM_ARRAY_TASK_ID" -eq 3 ]; then
    export TASK="TaskV2-BHL-BallToNetSolo-Blind-v0" RUN_NAME="_smoke-v2-solo"
else
    V=${VIS[$SLURM_ARRAY_TASK_ID]}
    export TASK="TaskV2-BHL-CubeToShelf-${V}-v0" RUN_NAME="_smoke-v2-${V,,}"
fi
export SEED=0 MAX_ITER=3 NUM_ENVS=64
export OVERRIDE_FILE="$WORKSPACE/overrides/v2-smoke-${SLURM_ARRAY_TASK_ID}.txt"
: > "$OVERRIDE_FILE"

echo "=== node: $(hostname) | $TASK | 3 iters at $NUM_ENVS envs | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
