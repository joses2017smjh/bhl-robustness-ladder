#!/bin/bash
#SBATCH --job-name=terr-flatfill
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# The obstacle ablation, done correctly.
#
# `terrain-smooth` removed the discrete obstacles and redistributed their 20%
# share into rough ground and slope (rough 0.40 -> 0.50, each slope 0.20 ->
# 0.25). So the "no obstacles" arm is also *rougher on average*, and the section
# flags its own result as unquotable. `FLATFILL_TERRAINS_CFG` holds every other
# proportion fixed and fills the obstacle share with flat tiles, so the only
# difference from the bumpy menu is that a fifth of the tiles have nothing to
# step over.
#
# What this settles: whether obstacles were the limiter on the terrain level, or
# whether removing them just made the menu harder in a different way. The
# confounded version said obstacles were not the limiter; this one can say it.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export TASK=Velocity-BHL-Biped-FlatFill-v0
export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME="terrain-flatfill-s${SEED}"
export MAX_ITER=6000

echo "=== node: $(hostname) | $RUN_NAME | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
