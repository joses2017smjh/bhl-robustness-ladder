#!/bin/bash
#SBATCH --job-name=maze-fixclip
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Isaac clip of the corrected stereo maze policy with before/after sensor panels.
# v60 renders; dgxh-1 has no Vulkan for 6.0, so it is excluded.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
export BHL_VIEW_EYE="1.8:1.8:1.1" BHL_VIEW_LOOKAT="0:0:0.3"
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"
source "$REPO/slurm/inner/v60_boot_gate.sh"; v60_boot_gate
echo "=== node: $(hostname) | maze stereo fixed clip | $(date) ==="
bhl_exec "$REPO/slurm/inner/maze_fix_clip.sh"
echo "=== done: $(date) ==="
