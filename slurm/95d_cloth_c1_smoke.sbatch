#!/bin/bash
#SBATCH --job-name=cloth-c1-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Three iterations through scripts/train.py on the rigid proxy. Not 8,000.
# Same lesson as 90_v2_train_smoke: construct-and-step can pass while the
# runner config is wrong.
set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"
export TASK=ClothSort-BHL-Rigid-Oracle-v0
export RUN_NAME=_smoke-cloth-c1
export SEED=0 NUM_ENVS=64 MAX_ITER=3
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
