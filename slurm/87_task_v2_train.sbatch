#!/bin/bash
#SBATCH --job-name=v2-train
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --array=0-8
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# The nine redesigned cells: three tasks x blind/depth/rgb, seed 0.
#
# Seed 0 only, deliberately. Twenty-seven jobs launched at once would all fail
# the same way if anything is wrong end-to-end, and this pipeline has been
# through five Isaac Lab 3.x breakages in one afternoon. Seeds 1 and 2 follow
# once these produce a curve.
#
# NUM_ENVS is 1024 for every cell and must stay that way. RGB carries two
# rendered cameras per env and is the heaviest arm; matching the others to it is
# what makes the vision comparison a comparison. If RGB OOMs, drop ALL NINE to
# 512 and rerun the lot -- never mix.

set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$WORKSPACE/overrides"

IDX=$SLURM_ARRAY_TASK_ID
TASKS=(CubeToShelf BallToNet PlankToWall)
VIS=(Blind Depth Rgb)
T=${TASKS[$(( IDX / 3 ))]}
V=${VIS[$(( IDX % 3 ))]}

export TASK="TaskV2-BHL-${T}-${V}-v0"
export RUN_NAME="v2-${T,,}-${V,,}-s${SEED:-0}"
export SEED=${SEED:-0} MAX_ITER=${MAX_ITER:-8000} NUM_ENVS=${NUM_ENVS:-1024}
export OVERRIDE_FILE="$WORKSPACE/overrides/v2-${IDX}.txt"
: > "$OVERRIDE_FILE"

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
