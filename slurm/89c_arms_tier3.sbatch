#!/bin/bash
#SBATCH --job-name=tier3-arms
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-2
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Tier 3: do the arms matter where perturbations are? The 22-DoF robot on
# stairs with ray-cast depth, `joint_deviation_shoulder` / `_elbow` ablated in
# the task itself (Velocity-BHL-Arms-Stairs-Depth-v0), so every row has it off:
#
#   0  PPO, rsl-rl                 the work order's single-agent row
#   1  MAPPO, limb4 (5/5/6/6)      the work order's limb-agent row
#   2  IPPO, limb1 (one agent)     the skrl control -- same trainer and settings
#                                  as row 1 with the split switched off
#
# Ice waits on B3's patches reaching the robots. NUM_ENVS 4096, 6,000 iterations,
# skrl rows on the rsl-rl settings (--hparams rsl, chosen by run name).
#
# DO NOT SUBMIT without G-T3 (88c_arms_tier3_gate.sbatch) passing first.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
mkdir -p "$WORKSPACE/overrides"

case $SLURM_ARRAY_TASK_ID in
  0) ROW=ppoarms;   export BHL_PARTITION=none  BHL_ALGO=ppo   ;;
  1) ROW=mappoarms; export BHL_PARTITION=limb4 BHL_ALGO=mappo ;;
  2) ROW=limb1arms; export BHL_PARTITION=limb1 BHL_ALGO=ippo  ;;
  *) echo "bad array index" >&2; exit 1 ;;
esac
export TASK=Velocity-BHL-Arms-Stairs-Depth-v0
export SEED=${SEED:-0} MAX_ITER=${MAX_ITER:-6000} NUM_ENVS=${NUM_ENVS:-4096}
export RUN_NAME="${ROW}-stairs-depth-s${SEED}"
# The ablation is in the task; the trainer flag would find nothing to clear and refuse.
export BHL_ABLATE_ARM_DEV=0
export OVERRIDE_FILE="$WORKSPACE/overrides/tier3-${SLURM_ARRAY_TASK_ID}-s${SEED}.txt"
: > "$OVERRIDE_FILE"

source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | $(date) ==="
bhl_exec "$REPO/slurm/inner/marl_train.sh"
echo "=== done: $(date) ==="
