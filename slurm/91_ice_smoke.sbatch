#!/bin/bash
#SBATCH --job-name=ice-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-2
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Three iterations through the real training path, one per ice arm. The v2
# tasks taught this the hard way: an env smoke that never calls train.py passed
# 9/9 through four separate training failures.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
mkdir -p "$WORKSPACE/overrides"
IDS=(Velocity-BHL-Biped-Ice-v0 Velocity-BHL-Biped-Ice-Depth-v0 Velocity-BHL-Biped-IceVisible-v0)
export TASK=${IDS[$SLURM_ARRAY_TASK_ID]}
export RUN_NAME="_smoke-ice-${SLURM_ARRAY_TASK_ID}"
export SEED=0 MAX_ITER=3 NUM_ENVS=64
export OVERRIDE_FILE="$WORKSPACE/overrides/ice-smoke-${SLURM_ARRAY_TASK_ID}.txt"
: > "$OVERRIDE_FILE"
echo "=== node: $(hostname) | $TASK | 3 iters at $NUM_ENVS | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
