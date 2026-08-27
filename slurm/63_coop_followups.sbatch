#!/bin/bash
#SBATCH --job-name=coop-next
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=30:00:00
#SBATCH --array=0-7
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# The five follow-ups the cooperative-lift data actually asks for.
#
# 0-2  staged, 16,000 iterations, three seeds.
#      The single best-supported item, and not a guess: the 4,000-iteration
#      staged run went 12.31 -> 18.46 reward over its last 400 iterations. It
#      was cut off mid-climb, not converged. Everything below is uninterpretable
#      until this one has actually finished rising, and at three seeds, because
#      section 5's previous headline died to exactly that check.
#
# 3-4  tilt weight 0.5x and 0x against the default.
#      `notilt` reaches a peak lift of 15.9 cm -- height is physically there --
#      while falling 81% of episodes. Every other arm sits pinned at the 4 cm
#      curriculum floor. That brackets the answer: the tilt penalty is what caps
#      height, and somewhere between 0 and 1x there may be height without the
#      falls. Two points is enough to see the direction.
#
# 5    randomised payload mass and friction.
#      Every run so far lifted the same 0.5 kg block with the same friction, and
#      the sim2sim harness replays that same block, so it cannot tell a policy
#      that generalises from one that memorised a mass. A non-prehensile pinch
#      is a friction bet; this is the term that says so.
#
# 6-7  the occluded pair: blind and sighted, object pose withheld.
#      This is the only fair test of vision in this project. Adding depth to a
#      policy already handed the exact object pose made it worse -- pinch 0.30
#      to 0.00 -- because depth was substituting a pooled image for a quantity
#      it was being given exactly. Withhold the pose and randomise the payload's
#      size, and depth becomes the only route to knowing where the object is and
#      how big it is. Arm 6 is the control that arm 7 has to beat; 6 is expected
#      to do badly and that is the point.
#
# RGB is absent, and not by preference. Colour needs the RTX renderer, which is
# the exact thing that segfaults on this cluster under Isaac Sim 5.1 (section 6).
# Ray-casting returns geometry and nothing else -- there is no colour to read out
# of a Warp mesh query. So an RGB arm is gated behind the Isaac Sim 6.0.1
# migration, which clears the crash but is its own piece of work. Depth was
# never a stylistic choice here; it was the only modality the cluster could
# produce, and it still is.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

IDX=$SLURM_ARRAY_TASK_ID
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/coop-next-${IDX}.txt"
# Defaulted, not hardcoded, so a rerun can vary the seed from the sbatch line.
export TASK=CoopLift-BHL-Cube-v0 SEED=${SEED:-0}
export MAX_ITER=${MAX_ITER:-16000} NUM_ENVS=${NUM_ENVS:-1024}
echo "env.stage_lift_on_pinch=true" > "$OVERRIDE_FILE"

case $IDX in
  0|1|2) export SEED=$IDX RUN_NAME="coop-staged16k-s${IDX}" ;;
  3) export RUN_NAME=coop-tilt050-s0
     echo "env.rewards.object_tilt.weight=-0.75" >> "$OVERRIDE_FILE" ;;
  4) export RUN_NAME=coop-tilt000-s0
     echo "env.rewards.object_tilt.weight=0.0" >> "$OVERRIDE_FILE" ;;
  5) export TASK=CoopLift-BHL-Cube-Random-v0 RUN_NAME="coop-randpayload-s${SEED}" ;;
  6) export TASK=CoopLift-BHL-Cube-Occluded-v0 RUN_NAME="coop-occluded-blind-s${SEED}" ;;
  7) export TASK=CoopLift-BHL-Cube-Occluded-Depth-v0 RUN_NAME="coop-occluded-depth-s${SEED}"
     export NUM_ENVS=512 ;;
  *) echo "bad array index $IDX"; exit 1 ;;
esac

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | iters=$MAX_ITER | $(date) ==="
cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
