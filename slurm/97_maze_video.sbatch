#!/bin/bash
#SBATCH --job-name=maze-video
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# B5 clips from the seed-0 checkpoints: blind, lidar, stereo at pool 4, both at
# pool 16. Isaac 6.0 with RTX; the camera follows each robot (BHL_VIEW_FOLLOW).
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
export BHL_VIEW_FOLLOW=1 BHL_VIEW_EYE="2.2:2.2:1.8" BHL_VIEW_LOOKAT="0:0:0.3"
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"
bhl_exec "$REPO/slurm/inner/maze_video.sh"
echo "=== done: $(date) ==="
