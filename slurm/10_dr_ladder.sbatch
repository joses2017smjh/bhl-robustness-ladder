#!/bin/bash
#SBATCH --job-name=dr-ladder
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-8
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Experiment 2 (headline): domain-randomization fidelity ladder.
# 3 rungs x 3 seeds = 9 runs, one GPU each.
#
# The `default` rung doubles as the control for the push-recovery and
# rough-terrain experiments, so this array is the backbone of the whole project.
#
# Seeds are not optional here: the deliverable is a retention *curve*, and a
# curve through three unreplicated points cannot be distinguished from noise.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
source "$REPO/slurm/dr_levels.sh"
setup_node_cache

LEVELS=(off default aggressive)
LEVEL=${LEVELS[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))

export TASK=Velocity-Berkeley-Humanoid-Lite-Biped-v0
export RUN_NAME="dr-${LEVEL}-s${SEED}"
export SEED
export MAX_ITER=6000

# Overrides go through a file: Apptainer's --env splits on commas and would
# mangle every [lo,hi] range.
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/dr-${LEVEL}-s${SEED}.txt"
dr_overrides "$LEVEL" > "$OVERRIDE_FILE"

echo "=== node: $(hostname) | rung: $LEVEL | seed: $SEED | started: $(date) ==="
cat "$OVERRIDE_FILE"

bhl_exec "$REPO/slurm/inner/train.sh"

echo "=== done: $(date) ==="
