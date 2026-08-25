#!/bin/bash
#SBATCH --job-name=build60
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Build the Isaac Sim 6.0 stack beside the 5.1 one, and prove the renderer works.
#
# Why this is worth doing at all: everything vision-shaped in this repo is
# shaped by one fact -- Isaac Sim 5.1's RTX renderer segfaults on this cluster
# inside omni.usd.create_hydra_engine, so there is no RGB, no rendered depth,
# and no material or lighting effects. Depth here comes from Warp ray-casting,
# which returns geometry and nothing else. A probe already showed 6.0.1 getting
# past the call that kills 5.1, so the renderer is a version problem rather than
# a hardware one.
#
# The exact pairing matters and is not the obvious one. isaaclab 3.0.0b2 pins
# isaacsim[all,extscache]==6.0.0.1 -- not 6.0.1.0, which is what the original
# probe installed -- and requires Python >= 3.12 against 5.1's 3.11. So this is
# a three-way version move, which is precisely why it gets its own venv instead
# of an upgrade in place.
#
# What this job does NOT claim: that the overlays work on it. Isaac Lab went
# 2.x -> 3.x and the task configs, the mdp imports and the manager API will have
# moved. This builds the stack and answers the one question that decides whether
# porting is worth anyone's time: does RTX actually render here.

set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

echo "=== node: $(hostname) | target: $UV_PROJECT_ENVIRONMENT | $(date) ==="
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader || true

bhl_exec "$REPO/slurm/inner/build_isaac60.sh"
echo "=== done: $(date) ==="
