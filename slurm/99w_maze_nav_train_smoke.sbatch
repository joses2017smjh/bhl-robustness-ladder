#!/bin/bash
#SBATCH --job-name=mazenav-smoke
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

# Three iterations through the real training path, one per navigation arm.
# Env smoke 21353199 passed construct/reset/step; v2 taught that train.py is
# a different gate (mean episode length < 2 is a broken task).
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"
source "$REPO/slurm/inner/v60_boot_gate.sh"; v60_boot_gate

ARMS=(Blind Lidar Stereo Both)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
export TASK="Velocity-BHL-Maze-${A}-v0"
export RUN_NAME="_smoke-mazenav-${A,,}"
export SEED=0 MAX_ITER=3 NUM_ENVS=64
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | 3 iters at $NUM_ENVS | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
