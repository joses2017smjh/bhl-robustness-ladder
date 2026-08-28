#!/bin/bash
#SBATCH --job-name=marl-gate
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# G-B4. The offline half (partition coverage, round-trip, joint order) already
# passes and needs no GPU; this runs the online half -- does the wrapped env
# construct, reset and step, with each agent's action slice the declared width.
#
# v51, per the work order: everything in the MARL block is 5.1 and the 6.0 stack
# does not get pulled in.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
mkdir -p "$REPO/results"
export NUM_ENVS=${NUM_ENVS:-8}
bhl_exec "$REPO/slurm/inner/marl_gate.sh"
echo "=== done: $(date) ==="
