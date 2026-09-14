#!/bin/bash
#SBATCH --job-name=marl-tgate
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# G-B4t, the gate for Tier 1's MARL rows (slurm/89b_marl_terrain.sbatch):
# {MAPPO, IPPO} on the 12-DoF biped split left leg | right leg, plus the limb1
# single-agent control, on ice / stairs / slippery / rough with ray-cast depth.
# Two real iterations each at 4,096 envs, so an OOM at the grid's width shows
# here and not six hours into the array. Passes only on what the constructed
# env and trainer report about themselves -- see marl_terrain_check.py.
#
# v51, per the work order. Ray-cast depth needs no renderer and no cameras flag.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export NUM_ENVS=${NUM_ENVS:-4096}
echo "=== node: $(hostname) | G-B4t | $(date) ==="
bhl_exec "$REPO/slurm/inner/marl_terrain_gate.sh"
echo "=== done: $(date) ==="
