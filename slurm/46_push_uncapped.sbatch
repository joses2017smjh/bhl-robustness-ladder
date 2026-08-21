#!/bin/bash
#SBATCH --job-name=push-uncap
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# sm_75 or newer AND >=32 GB. The el9 tag is not enough: preempt mixes in
# GTX 1080 (sm_61) and V100 (sm_70), both below the pinned torch's cubin floor,
# and the 11 GB cards cannot hold this run's env count.

# Raise the adaptive curriculum's own safety cap from 1.0 to 2.0 m/s.
#
# The published 0.87 m/s is a lower bound, not a measurement: the arm pinned
# against m_max = 1.0 (peak 1.000), so what was reported is where the cap was,
# not where the robot stopped coping. "How hard a shove can an 11.3 kg robot
# with 6 Nm joints learn to reject" is arguably the most quotable number in this
# repo, and right now it is quoted with an asterisk.
#
# 2.0 m/s is chosen to be comfortably unreachable rather than to be plausible --
# the point is that the ceiling stops being the binding constraint. The rule
# demotes as readily as it promotes, so an unreachable cap costs nothing if the
# true ceiling is 0.9.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export TASK=Velocity-BHL-Biped-PushAdaptive-v0
export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME="push-adaptive-uncapped-s${SEED}"
export MAX_ITER=6000

mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/push-uncapped.txt"
cat > "$OVERRIDE_FILE" <<'OVR'
env.curriculum.push_levels.params.max_magnitude=2.0
OVR

echo "=== node: $(hostname) | $RUN_NAME | $(date) ==="
cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
