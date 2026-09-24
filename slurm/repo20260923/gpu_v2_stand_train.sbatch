#!/bin/bash
#SBATCH --job-name=v2-stand-train
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# COOP-17 follow-up (2026-09-24, user-authorized): one cell, seed 0, blind,
# of the standing-height cube task. The 87_task_v2_train recipe (v60, cameras
# on for stack parity, 1024 envs, 8000 iterations).
#
# Predeclared before the job started (SLURM_JOBS.md):
#   kill    at model_999: mean episode length < 25 steps (1 s) over the last
#           100 iterations, or Episode_Termination/fallen > 0.9 -- the spawn
#           is unusable under the policy as it was under PD hold.
#   result  Episode_Termination/success averaged over the last 200 iterations
#           >= 0.10 = "learned placement (standing variant)"; else NEGATIVE.
#   scope   CubeToShelfStand only; never read as a CubeToShelf number.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$WORKSPACE/overrides"
export TASK="TaskV2-BHL-CubeToShelfStand-Blind-v0"
export SEED=${SEED:-0} MAX_ITER=${MAX_ITER:-8000} NUM_ENVS=${NUM_ENVS:-1024}
export RUN_NAME="v2-cubetoshelfstand-blind-s${SEED}"
export OVERRIDE_FILE="$WORKSPACE/overrides/v2-stand-${SLURM_JOB_ID}.txt"
: > "$OVERRIDE_FILE"
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
