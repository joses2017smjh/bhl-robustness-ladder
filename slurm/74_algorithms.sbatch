#!/bin/bash
#SBATCH --job-name=algo
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Two algorithm changes, each aimed at a specific thing the data says is wrong.
# Not a survey: rsl-rl ships exactly these beyond vanilla PPO, and both have a
# reason to be here rather than being tried because they exist.
#
# 0-1  RND exploration bonus on the cube lift.
#      Height never leaves the 4 cm curriculum floor. That has now survived
#      fourteen reward ablations, three recipe eras, three seeds, and a
#      16,000-iteration run. A plateau that robust against changes to the
#      reward looks less like mis-specified reward and more like a policy that
#      stopped exploring -- it found a stable 4 cm hover and never sampled the
#      states where a real lift begins. RND pays for visiting novel states,
#      which is exactly the missing pressure. Two weights, because the useful
#      range is unknown and one point would tell us nothing.
#
# 2-3  Recurrent (LSTM) actor on the occluded lift, blind and sighted.
#      The occluded task withholds the object's pose and spawns it +/- 8 cm
#      away, so at any instant the observation does not contain where the
#      payload is. A memoryless policy cannot integrate what it saw a moment
#      ago; that is the textbook case for recurrence, and it is the same
#      argument section 3 already used for the distilled scan student. The
#      blind arm is the control: if recurrence helps *without* a camera, the
#      gain is memory rather than vision.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

export SEED=0 MAX_ITER=8000
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/algo-${SLURM_ARRAY_TASK_ID}.txt"
echo "env.stage_lift_on_pinch=true" > "$OVERRIDE_FILE"

case $SLURM_ARRAY_TASK_ID in
  0) export TASK=CoopLift-BHL-Cube-v0 RUN_NAME=coop-rnd003-s0 NUM_ENVS=1024 BHL_RND=0.003 ;;
  1) export TASK=CoopLift-BHL-Cube-v0 RUN_NAME=coop-rnd010-s0 NUM_ENVS=1024 BHL_RND=0.010 ;;
  2) export TASK=CoopLift-BHL-Cube-Occluded-v0 RUN_NAME=coop-occ-lstm-s0 \
            NUM_ENVS=1024 BHL_POLICY=recurrent ;;
  3) export TASK=CoopLift-BHL-Cube-Occluded-Depth-v0 RUN_NAME=coop-occd-lstm-s0 \
            NUM_ENVS=512 BHL_POLICY=recurrent ;;
  *) echo "bad array index"; exit 1 ;;
esac

echo "=== node: $(hostname) | $RUN_NAME | rnd=${BHL_RND:-off} policy=${BHL_POLICY:-mlp} | $(date) ==="
cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
