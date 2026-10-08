#!/bin/bash
#SBATCH --job-name=mazenav-video
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Seed-0 mazenav clips (blind / lidar / stereo / both) through a following
# camera sensor. dgxh-1 has no Vulkan for Isaac Sim 6.0. Old maze-* clips stay.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
export BHL_VIEW_EYE="1.8:1.8:1.1" BHL_VIEW_LOOKAT="0:0:0.3"
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"
source "$REPO/slurm/inner/v60_boot_gate.sh"; v60_boot_gate
echo "=== node: $(hostname) | mazenav seed-0 clips | $(date) ==="
bhl_exec "$REPO/slurm/inner/mazenav_video.sh"
echo "=== done: $(date) ==="
