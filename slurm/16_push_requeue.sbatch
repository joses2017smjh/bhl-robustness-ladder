#!/bin/bash
#SBATCH --job-name=push-requeue
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Replaces push-sweep ceiling 0.4 seed 1, which diverged at iteration 3,982 with
#   RuntimeError: normal expects all elements of std >= 0.0
#
# That is PPO's action-noise std going negative. rsl-rl defaults
# noise_std_type="scalar" -- a raw learnable parameter with nothing preventing
# it from crossing zero -- and BHL never overrides it. The "log" option
# parameterises std as exp(log_std) and cannot go negative.
#
# This run deliberately does NOT switch to "log": ~20 policies are already
# trained under the scalar parameterisation, and changing it now would make
# this seed incomparable with the rest of its own sweep. It is a fresh seed of
# the identical configuration, replacing a diverged sample. The instability is
# documented rather than silently patched.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

CEIL=0.4
SEED=2
export TASK=Velocity-BHL-Biped-PushCurriculum-v0
export RUN_NAME="push-c${CEIL}-s${SEED}"
export SEED MAX_ITER=6000

mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/push-c${CEIL}-s${SEED}.txt"
cat > "$OVERRIDE_FILE" <<OVR
env.curriculum.push_levels.params.end_magnitude=$CEIL
env.curriculum.push_levels.params.start_step=24000
env.curriculum.push_levels.params.full_step=120000
env.events.push_robot.interval_range_s=[5.0,9.0]
OVR

echo "=== replacing diverged run | ceiling $CEIL | seed $SEED | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
