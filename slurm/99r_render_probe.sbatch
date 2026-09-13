#!/bin/bash
#SBATCH --job-name=render-probe
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Why Isaac clips show no robot: viewport vs camera sensor, fabric vs USD, and
# where the body actually is. Three tasks x two modes. No policy, no checkpoint.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"
bhl_exec "$REPO/slurm/inner/render_probe.sh"
echo "=== done: $(date) ==="
