#!/bin/bash
#SBATCH --job-name=push-sweep
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Push recovery, pass 2: sweep the curriculum ceiling instead of guessing it.
#
# Pass 1 ramped to 1.5 m/s and destroyed the policy. The reward trace is
# unambiguous: it rose to ~24 by iteration 1600 (push ~0.3 m/s) and then
# collapsed to ~3 as the ramp continued past ~0.7 m/s, ending with 92% of
# episodes terminating on bad orientation. The fixed-magnitude arm never
# learned at all (reward ~2.5 throughout), which does confirm the curriculum
# rationale -- but the ceiling was far beyond what an 11.3 kg robot with 6 Nm
# joint limits can reject.
#
# So the ceiling becomes the swept variable. "How much disturbance can this
# machine learn to reject" is a better result than a single arbitrary setting,
# and it is the same shape of answer as the DR ladder.
#
# The ramp is also stretched (full strength at iteration 5000 rather than 4000)
# and pushes spaced further apart, so the policy has time to adapt at each level.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

CEILINGS=(0.2 0.4 0.6)
CEIL=${CEILINGS[$((SLURM_ARRAY_TASK_ID / 2))]}
SEED=$((SLURM_ARRAY_TASK_ID % 2))

export TASK=Velocity-BHL-Biped-PushCurriculum-v0
export RUN_NAME="push-c${CEIL}-s${SEED}"
export SEED
export MAX_ITER=6000

mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/push-c${CEIL}-s${SEED}.txt"
cat > "$OVERRIDE_FILE" <<OVR
env.curriculum.push_levels.params.end_magnitude=$CEIL
env.curriculum.push_levels.params.start_step=24000
env.curriculum.push_levels.params.full_step=120000
env.events.push_robot.interval_range_s=[5.0,9.0]
OVR

echo "=== node: $(hostname) | push ceiling: $CEIL m/s | seed: $SEED | $(date) ==="
cat "$OVERRIDE_FILE"

bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
