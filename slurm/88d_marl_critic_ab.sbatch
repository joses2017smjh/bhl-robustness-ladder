#!/bin/bash
#SBATCH --job-name=marl-critic-ab
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --array=0-2
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Was the MARL plateau the critic? limb1 + privileged MAPPO critic, limb1 + the
# policy-observation critic the grid rows had, and legs2 + privileged, 1,500
# iterations each on stairs with depth. v51.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
case $SLURM_ARRAY_TASK_ID in
  0) export BHL_PARTITION=limb1 BHL_ALGO=mappo RUN_NAME=ab-limb1-mappo-privileged-stairs ;;  # PPO-like if the critic was the gap
  1) export BHL_PARTITION=limb1 BHL_ALGO=ippo  RUN_NAME=ab-limb1-ippo-policy-stairs ;;       # the critic every grid row had
  2) export BHL_PARTITION=legs2 BHL_ALGO=mappo RUN_NAME=ab-legs2-mappo-privileged-stairs ;;  # the split, fixed critic
esac
export MAX_ITER=${MAX_ITER:-1500}
echo "=== node: $(hostname) | critic A/B $RUN_NAME | $(date) ==="
bhl_exec "$REPO/slurm/inner/marl_critic_ab.sh"
echo "=== done: $(date) ==="
