#!/bin/bash
#SBATCH --job-name=pov-gifs
#SBATCH --account=eecs
#SBATCH --partition=eecs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Same shape as 55, and same reason for the single unconstrained gres. The work
# is MuJoCo physics, a numpy policy forward pass, and an ffmpeg pipe; the coop
# actor is 194 -> 256 -> 256 -> 128 -> 44, which is cheaper in numpy than the
# cost of moving it to a device.
#
# Closes §5's missing sim2sim column and its missing clip in one job. The scores
# come first; the 2/3/4-robot clips are rendered from the arm the scores pick.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=${MUJOCO_GL:-egl} PYOPENGL_PLATFORM=${PYOPENGL_PLATFORM:-egl}
export OMP_NUM_THREADS=8
export CACHE_DIR=$WORKSPACE/mjcf_cache
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS CARRY_STEPS CLIP_RUN"
export N_SEEDS=${N_SEEDS:-8} CARRY_STEPS=${CARRY_STEPS:-300}
mkdir -p "$REPO/docs/gifs" "$REPO/results"

if [ -z "${CUDA_VISIBLE_DEVICES:-}" ] && [ ! -e /dev/dri/renderD128 ]; then
    echo "note: no GPU or DRI node on $(hostname); rendering through osmesa"
    export MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa
fi

bhl_exec "$REPO/slurm/inner/pov_gifs.sh"

echo "=== done: $(date) ==="
