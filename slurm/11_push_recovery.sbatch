#!/bin/bash
#SBATCH --job-name=push-rec
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Experiment 1: push recovery / disturbance rejection.
#
# Two arms x 3 seeds:
#   curriculum - push ramped 0 -> 1.5 m/s over training (the method)
#   fixed      - push at 1.5 m/s from step zero (ablation)
#
# The ablation is what makes this a result rather than a demo. Without it a
# win only shows "pushes help", which is already known; with it, the curriculum
# itself is the isolated variable.
#
# The no-push control is the `dr-default` rung of the ladder array, so it is
# not duplicated here.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

ARMS=(curriculum fixed)
ARM=${ARMS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))

case "$ARM" in
    curriculum) export TASK=Velocity-BHL-Biped-PushCurriculum-v0 ;;
    fixed)      export TASK=Velocity-BHL-Biped-Push-v0 ;;
esac

export RUN_NAME="push-${ARM}-s${SEED}"
export SEED
export MAX_ITER=6000

echo "=== node: $(hostname) | arm: $ARM | seed: $SEED | task: $TASK | started: $(date) ==="

bhl_exec "$REPO/slurm/inner/train.sh"

echo "=== done: $(date) ==="
