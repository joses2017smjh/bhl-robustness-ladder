#!/bin/bash
#SBATCH --job-name=marl-train
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Tier 1, first block: limb factorisation against the single-agent baseline.
#
#   0  limb4 + MAPPO   the headline -- one agent per limb, centralised critic
#   1  limb4 + IPPO    prices the centralised critic, holding the split fixed
#   2  limb2 + MAPPO   prices the split, holding the algorithm fixed
#   3  PPO             the control, single agent, 22 actions
#
# One variable moves per row against row 0, which is the only arrangement that
# lets a difference be attributed. NUM_ENVS is identical across all four and
# must stay that way; if MARL OOMs, drop every row to 2048 including the PPO
# control, never a mix.
#
# `joint_deviation_arms` is ablated in every row including the control -- it
# penalises the arms for leaving their default pose, which under a per-limb
# agent is a penalty for doing anything, and section 5 already found it
# fighting a squat-and-pinch. Ablating it only in the MARL rows would confound
# the factorisation with the reward.
#
# DO NOT SUBMIT until G-B4 passes (`slurm/88_marl_gate.sbatch`). If the four
# action slices do not reassemble into exactly the single-agent vector, these
# rows measure the partition rather than the algorithm.

set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
mkdir -p "$WORKSPACE/overrides"

IDX=$SLURM_ARRAY_TASK_ID
case $IDX in
  0) export BHL_PARTITION=limb4 BHL_ALGO=mappo ;;
  1) export BHL_PARTITION=limb4 BHL_ALGO=ippo  ;;
  2) export BHL_PARTITION=limb2 BHL_ALGO=mappo ;;
  3) export BHL_PARTITION=none  BHL_ALGO=ppo   ;;
  *) echo "bad array index $IDX" >&2; exit 1 ;;
esac

export TASK=${TASK:-Velocity-BHL-Arms-Bumpy-v0}
export RUN_NAME="marl-${BHL_PARTITION}-${BHL_ALGO}-s${SEED:-0}"
export SEED=${SEED:-0} MAX_ITER=${MAX_ITER:-6000} NUM_ENVS=${NUM_ENVS:-4096}
export BHL_ABLATE_ARM_DEV=1
export OVERRIDE_FILE="$WORKSPACE/overrides/marl-${IDX}.txt"
: > "$OVERRIDE_FILE"

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | $(date) ==="
bhl_exec "$REPO/slurm/inner/marl_train.sh"
echo "=== done: $(date) ==="
