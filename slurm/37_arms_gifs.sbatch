#!/bin/bash
#SBATCH --job-name=arms-gif
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# 22-DoF pair clips for the same three README comparisons as the biped:
# DR strafe, 0.5 m/s shove, terrain d=0.80. GPU is EGL offscreen only.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export CACHE_DIR=$WORKSPACE/mjcf_cache
export VARIANT=humanoid EPISODE_S=10 N_SEEDS=1
mkdir -p "$REPO/docs/gifs" "$REPO/results/demo_arms"

bhl_exec "$REPO/slurm/inner/arms_gifs.sh"
echo "=== done $(date) ==="
