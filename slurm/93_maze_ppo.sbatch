#!/bin/bash
#SBATCH --job-name=maze-ppo
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-6
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# B5: corridor navigation with an RPLIDAR C1 and a global-shutter stereo pair.
#
# Four arms, one variable moving: blind, lidar, stereo, both. The blind arm is
# the control, and without it a lidar number is a fact about this maze rather
# than about lidar -- which is the mistake G-B2 made when it measured its own
# iteration budget and called it a terrain verdict.
#
# v60 because the stereo pair needs cameras enabled. Both sensors are ray-cast
# rather than RTX, so neither depends on the renderer that segfaults on 5.1 --
# that was the reason for choosing ray-cast over an RGB camera in the design.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

ARMS=(Blind Lidar Stereo Both StereoP8 StereoP16 BothP16)
A=${ARMS[$SLURM_ARRAY_TASK_ID]}
export TASK="Velocity-BHL-Maze-${A}-v0"
export RUN_NAME="${RUN_PREFIX:-maze}-${A,,}-s${SEED:-0}"
export SEED=${SEED:-0} MAX_ITER=${MAX_ITER:-6000}
# Identical across all four arms and it has to stay that way: the stereo arms
# carry two ray-cast cameras and the blind arm carries none, so letting the
# cheap arm run wider would make throughput the variable instead of sensing.
export NUM_ENVS=${NUM_ENVS:-2048}

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
