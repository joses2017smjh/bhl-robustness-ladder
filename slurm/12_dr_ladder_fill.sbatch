#!/bin/bash
#SBATCH --job-name=dr-fill
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-5
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Fills in the DR ladder between the rungs already trained.
#
# Pass 1 trained s=0, s=1, s=2 and found s=2 unlearnable (reward 4.6 vs 48 at
# s=0, 69% of episodes terminating on bad orientation). A 3-point curve whose
# top point is a training failure cannot show a tradeoff -- it only shows a
# cliff with nothing between. These two rungs turn it into a 5-point curve and
# bracket where the cliff actually is.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
source "$REPO/slurm/dr_levels.sh"
setup_node_cache

SCALES=(0.5 1.5)
SCALE=${SCALES[$((SLURM_ARRAY_TASK_ID / 3))]}
SEED=$((SLURM_ARRAY_TASK_ID % 3))

export TASK=Velocity-Berkeley-Humanoid-Lite-Biped-v0
export RUN_NAME="dr-s${SCALE}-s${SEED}"
export SEED
export MAX_ITER=6000

mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/dr-s${SCALE}-s${SEED}.txt"
dr_overrides_scale "$SCALE" > "$OVERRIDE_FILE"

echo "=== node: $(hostname) | DR scale: $SCALE | seed: $SEED | $(date) ==="
cat "$OVERRIDE_FILE"

bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
