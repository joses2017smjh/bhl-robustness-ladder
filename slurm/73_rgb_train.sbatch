#!/bin/bash
#SBATCH --job-name=rgb60
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# The first rendered-camera training in this project.
#
# Section 6's whole shape came from one fact: Isaac Sim 5.1's RTX renderer
# segfaults on this cluster, so depth came from Warp ray-casting and colour was
# not available at all. On the 6.0 stack RTX renders -- measured, rgb std 24.4
# across 1,902 unique colours with depth correct to the unit -- and 14 of 15
# overlay modules import unchanged once a removed noise alias is shimmed.
#
# Two arms, both on rough terrain, both otherwise identical to the terrain rung:
#   0  rendered RGB      the modality this repo has never had
#   1  rendered depth    the same sensor as section 6 but rendered rather than
#                        ray-cast, which is the comparison section 6 could not
#                        run: exact geometric depth against depth that has been
#                        through a renderer's near-plane, aliasing and noise
#
# Results from here do NOT belong in the same table as a 5.1 number. isaaclab
# 3.0.0b2 is a beta and the simulator underneath changed major version; this is
# a separate stack producing separate numbers.

set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

# TiledCamera needs the camera-enabled Kit experience, which starts RTX.
export ENABLE_CAMERAS=1
export SEED=0 MAX_ITER=6000 NUM_ENVS=512

case $SLURM_ARRAY_TASK_ID in
  0) export TASK=Velocity-BHL-Biped-Rgb-v0   RUN_NAME=rgb60-bumpy-s0 ;;
  1) export TASK=Velocity-BHL-Biped-Depth-v0 RUN_NAME=rendered-depth60-bumpy-s0 ;;
  *) echo "bad array index"; exit 1 ;;
esac

echo "=== node: $(hostname) | $RUN_NAME | stack=$BHL_STACK | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
