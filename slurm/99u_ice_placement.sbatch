#!/bin/bash
#SBATCH --job-name=ice-place
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Where the B3 ice patches are relative to the robots, on the built scene at the
# training width (4,096 envs, v51 -- the stack B3 trained on). Decides whether
# the six held ice rows of the MARL grid run as they are.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export NUM_ENVS=${NUM_ENVS:-4096}
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
echo "=== node: $(hostname) | ice placement probe | $(date) ==="
bhl_exec "$REPO/slurm/inner/ice_placement.sh"
echo "=== done: $(date) ==="
