#!/bin/bash
#SBATCH --job-name=push-adapt
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Push recovery, third method: magnitude gated on measured fall rate.
#
# Pass 1's fixed ramp advanced on iteration count regardless of whether the
# policy was coping, and drove a working gait (reward 24.7) down to ~3. This arm
# raises the push only while the fall rate is under target and backs it off when
# it is not, so the curriculum cannot outrun the policy.
#
# It also reports something the swept-ceiling runs cannot: the magnitude it
# settles at IS an estimate of the largest shove this robot can learn to reject.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export TASK=Velocity-BHL-Biped-PushAdaptive-v0
export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME="push-adaptive-s${SEED}"
export MAX_ITER=6000

echo "=== node: $(hostname) | adaptive push | seed: $SEED | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
