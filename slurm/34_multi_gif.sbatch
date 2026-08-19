#!/bin/bash
#SBATCH --job-name=multi-gif
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Four policies, one MuJoCo world. GPU is for EGL offscreen rendering only.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export CACHE_DIR=$WORKSPACE/mjcf_cache
mkdir -p "$REPO/docs/gifs"

bhl_exec "$REPO/slurm/inner/render_multi.sh"
echo "=== done $(date) ==="
