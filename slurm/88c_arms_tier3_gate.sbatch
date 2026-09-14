#!/bin/bash
#SBATCH --job-name=tier3-gate
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# G-T3, the gate for Tier 3 (slurm/89c_arms_tier3.sbatch): PPO, MAPPO over four
# limb agents, and the limb1 control on the 22-DoF robot, stairs, depth on, arm
# deviation ablated in the task. Two iterations each at full width. v51.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export NUM_ENVS=${NUM_ENVS:-4096}
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
echo "=== node: $(hostname) | G-T3 | $(date) ==="
bhl_exec "$REPO/slurm/inner/arms_tier3_gate.sh"
echo "=== done: $(date) ==="
