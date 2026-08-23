#!/bin/bash
#SBATCH --job-name=obj-staged
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# The ladder and the yoga ball, under the recipe that actually lifts.
#
# Section 5 reports three objects, but only the cube ever saw the fix. The
# ladder and ball were last trained on 2026-08-19 under standing spawn and the
# single-sigma pinch kernel -- the era that produced a toss with pinch
# identically zero. Everything learned since (squat spawn, two-scale reach,
# opposing clamp, and above all staging the lift weights behind a formed pinch)
# has only ever been applied to the cube, so "cube works, ladder and ball do
# not" is a statement about when they were run rather than about the objects.
#
# `staged` is the arm to carry over rather than the control: on the cube it is
# the only arm that gets a pinch, a clamp and a nonzero lift bonus at once,
# and it holds the payload 4.2 cm up while falling 19% of the time.
#
# The two objects are the interesting contrast for a non-prehensile side-lift.
# The ladder is long and light with a graspable rung geometry, so it rewards
# reaching wide; the ball is round and offers no flat face at all, so a pinch
# has to be a true opposing-force clamp or the object squirts out. If staging
# transfers to the cube's geometry only, that is a finding about the recipe.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

IDX=$SLURM_ARRAY_TASK_ID
case $IDX in
  0) export TASK=CoopLift-BHL-Ladder-v0 RUN_NAME=coop-ladder-staged-s0 ;;
  1) export TASK=CoopLift-BHL-Ball-v0   RUN_NAME=coop-ball-staged-s0 ;;
  *) echo "bad array index $IDX"; exit 1 ;;
esac
export SEED=0 MAX_ITER=4000 NUM_ENVS=1024

mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/obj-staged-${IDX}.txt"
cat > "$OVERRIDE_FILE" <<'OVR'
env.stage_lift_on_pinch=true
OVR

echo "=== node: $(hostname) | $RUN_NAME | $(date) ==="
cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
