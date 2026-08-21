#!/bin/bash
#SBATCH --job-name=depth-loco
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Depth-conditioned rough-terrain locomotion. The vision experiment this repo
# has been calling blocked.
#
# It was blocked on the wrong thing. Isaac Sim 5.1's RTX renderer does segfault
# here, which rules out `TiledCamera` and every rendered annotator -- but
# `RayCasterCamera` produces `distance_to_image_plane` by intersecting a pinhole
# ray bundle with the scene mesh in warp, and never creates a Hydra engine at
# all. Measured on this node: 4,096 envs with a 48x48 depth camera runs at
# 21,490 env-steps/s against 21,844 for physics alone. The depth costs 1.6%.
#
# What is given up versus rendered depth is material and lighting: specular
# dropout, IR-pattern failure on dark surfaces, edge fattening. What is given up
# is therefore artefacts that clean rendered depth does not have either -- both
# paths need a degradation model on top, and geometric depth is the cleaner
# starting point for one.
#
# 64x64 average-pooled to 16x16 before it reaches the MLP. Raw, the image would
# be 4,096 numbers against 45 of proprioception and the first layer would be
# almost entirely depth weights. Pooling to 256 is the cheapest honest encoder;
# a CNN trunk is the real answer and is not what rsl-rl's default actor is.
#
# The comparison is `terrain-bumpy` at the same seed: same terrain, same
# randomization, depth as the only added variable.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export TASK=Velocity-BHL-Biped-Depth-v0
export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME="depth-bumpy-s${SEED}"
export MAX_ITER=6000

echo "=== node: $(hostname) | $RUN_NAME | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
