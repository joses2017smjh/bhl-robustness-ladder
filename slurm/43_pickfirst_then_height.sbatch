#!/bin/bash
#SBATCH --job-name=coop-stage
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# pickfirst THEN height -- the run section 5 already names as the next one worth
# a GPU, and the third of the three dead arms from 39.
#
# `pickfirst` reaches pinch 0.397 and clamp 0.496 without ever falling, by never
# paying for height at all. `stage_lift_on_pinch` is the same start with an exit:
# lift weights stay at zero until the batch-mean pinch kernel clears 0.40, then
# turn on and stay on. Two bugs made the overnight `staged` arm meaningless --
# hydra wrote the flag after __post_init__ consumed it, and a missing pinch
# distance counted as fully pinched so the latch fired at step 0. Both are fixed.
#
# The question is a real fork. If height is learnable from a formed pinch, the
# section closes with a lift. If pinch collapses the moment height turns on, then
# height competes with formation no matter when it is introduced, and that is the
# finding -- a stronger one than "we never got it to lift".

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

export TASK=CoopLift-BHL-Cube-v0 SEED=0 MAX_ITER=4000 NUM_ENVS=1024
export RUN_NAME=coop-cube-staged-r2-s0
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/coop-staged-r2.txt"
cat > "$OVERRIDE_FILE" <<'OVR'
env.stage_lift_on_pinch=true
OVR

echo "=== node: $(hostname) | $RUN_NAME | $(date) ==="
cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
