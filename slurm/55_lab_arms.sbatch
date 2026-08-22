#!/bin/bash
#SBATCH --job-name=lab-arms
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# `share`, and no --constraint. This job is MuJoCo and ONNX Runtime end to end:
# the physics is CPU, the policies are CPU, and the only GPU work is MuJoCo's
# offscreen EGL context, which any card can do. Naming a GPU architecture is how
# the earlier render jobs ended up queued behind training on the RTX 8000s.
#
# The one gres is not optional even so: this cluster's QOS rejects a GPU-eligible
# partition request that asks for no GRES (QOSMinGRES), and the CPU-only nodes in
# `share` have no DRI device for EGL. The osmesa fallback below covers the case
# where it lands somewhere without one anyway.
#
# Rebuilds the lab-floor clip on the 22-DoF machine, with the robot's own
# egocentric depth along the bottom and the policy that clears the course in the
# orange/black livery, and prints the traversal table both variants are scored on.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export OMP_NUM_THREADS=8
export CACHE_DIR=$WORKSPACE/mjcf_cache
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS VX TRAVERSE_S"
export VX=${VX:-0.40} EPISODE_S=${EPISODE_S:-16} TRAVERSE_S=${TRAVERSE_S:-24}
mkdir -p "$REPO/docs/gifs" "$REPO/results"

# EGL needs a device; on a plain `share` node there is none. Fall back rather
# than fail, because the whole point of dropping --gres was to run anywhere.
if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ ! -e /dev/dri/renderD128 ]; then
    echo "note: no GPU or DRI node on $(hostname); rendering through osmesa"
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
fi

bhl_exec "$REPO/slurm/inner/lab_arms.sh"

ls -la "$REPO/docs/gifs/multi_lab."*
echo "=== done: $(date) ==="
