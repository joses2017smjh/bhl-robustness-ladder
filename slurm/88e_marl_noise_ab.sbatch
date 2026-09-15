#!/bin/bash
#SBATCH --job-name=marl-noise-ab
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

# The critic was not the plateau (21330372/21330392): with PPO's privileged critic
# a one-agent skrl MAPPO still tracks at 0.58 against rsl-rl PPO's 1.40 at 1,500
# iterations. Its exploration noise collapsed (std 0.13 vs 0.42) after skrl's
# KL-adaptive LR hit its 1e-2 ceiling in ~30 iterations. These three arms split
# the two suspects -- how the std is parameterised and whether the LR adapts --
# on the one-agent control, 1,500 iterations on stairs with depth. v51.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
case $SLURM_ARRAY_TASK_ID in
  0) export RUN_NAME=ab2-limb1-mappo-privileged-scalar-kl-stairs ;;       # rsl-rl's std parameterisation, skrl's LR schedule
  1) export RUN_NAME=ab2-limb1-mappo-privileged-scalar-fixedlr-stairs ;;  # rsl-rl's std, no schedule
  2) export RUN_NAME=ab2-limb1-mappo-privileged-log-fixedlr-stairs ;;     # skrl's log-std, no schedule
esac
export BHL_PARTITION=limb1 BHL_ALGO=mappo
export MAX_ITER=${MAX_ITER:-1500}
echo "=== node: $(hostname) | critic A/B $RUN_NAME | $(date) ==="
bhl_exec "$REPO/slurm/inner/marl_critic_ab.sh"
echo "=== done: $(date) ==="
