#!/bin/bash
#SBATCH --job-name=arms
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=0-7
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# 22-DoF (arms) counterparts of the three biped experiments.
#
# Four conditions x 2 seeds, chosen to be exactly the policies the three
# README comparison clips need, rather than a full re-run of every rung:
#
#   dr s=0.0        failure side of the randomization pair
#   dr s=1.0        success side of that pair, AND the flat-trained baseline
#                   that the push and terrain pairs are measured against
#   push-adaptive   success side of the push pair
#   terrain-bumpy   success side of the terrain pair
#
# Arms are not cosmetic here. A humanoid rejects a lateral shove partly by
# swinging its arms to shed angular momentum, which a 12-DoF biped cannot do.
# Upstream also penalises arm deviation, so this may measure that penalty
# suppressing the very strategy the arms would provide.
#
# Logs land in logs/rsl_rl/humanoid/ (experiment_name="humanoid"), separate
# from the biped runs.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
source "$REPO/slurm/dr_levels.sh"
setup_node_cache

IDX=$SLURM_ARRAY_TASK_ID
SEED=$((IDX % 2))
COND=$((IDX / 2))
export SEED MAX_ITER=6000
mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE=""

case $COND in
  0) export TASK=Velocity-Berkeley-Humanoid-Lite-v0 RUN_NAME="arms-dr0.0-s${SEED}"
     export OVERRIDE_FILE="$WORKSPACE/overrides/arms-dr0.0-s${SEED}.txt"
     dr_overrides_scale 0.0 > "$OVERRIDE_FILE" ;;
  1) export TASK=Velocity-Berkeley-Humanoid-Lite-v0 RUN_NAME="arms-dr1.0-s${SEED}"
     export OVERRIDE_FILE="$WORKSPACE/overrides/arms-dr1.0-s${SEED}.txt"
     dr_overrides_scale 1.0 > "$OVERRIDE_FILE" ;;
  2) export TASK=Velocity-BHL-Arms-PushAdaptive-v0 RUN_NAME="arms-push-s${SEED}" ;;
  3) export TASK=Velocity-BHL-Arms-Bumpy-v0        RUN_NAME="arms-terrain-s${SEED}" ;;
esac

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | $(date) ==="
[ -n "$OVERRIDE_FILE" ] && cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
