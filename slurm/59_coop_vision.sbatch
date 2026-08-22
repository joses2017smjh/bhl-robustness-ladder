#!/bin/bash
#SBATCH --job-name=coop-vision
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# The answer to "does vision in the RL loop help cooperative lifting", scored on
# the MuJoCo side rather than off PhysX training curves.
#
# Submit this with --dependency=afterok:<coop-depth jobid>. It reads checkpoints
# that job produces, so on its own it would score a baseline against nothing.
#
# A gres is requested for the EGL context: unlike 55 and 56 this job renders
# depth every control step for every robot, because the policies under test read
# it. That is two 64x64 offscreen passes per robot per step, which wants a real
# GPU context rather than the osmesa fallback -- though the fallback below still
# works, just slowly.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export OMP_NUM_THREADS=8
export CACHE_DIR=$WORKSPACE/mjcf_cache
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS CARRY_STEPS"
export N_SEEDS=${N_SEEDS:-8} CARRY_STEPS=${CARRY_STEPS:-300}
mkdir -p "$REPO/docs/gifs" "$REPO/results"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ ! -e /dev/dri/renderD128 ]; then
    echo "note: no GPU or DRI node on $(hostname); rendering through osmesa"
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
fi

bhl_exec "$REPO/slurm/inner/coop_vision.sh"

echo "=== done: $(date) ==="
