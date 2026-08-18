#!/bin/bash
#SBATCH --job-name=terrain
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-4
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Experiment 3: rough terrain + level curriculum.
#
#   bumpy  (3 seeds) - noise + slopes + low discrete obstacles
#   smooth (2 seeds) - the same menu WITHOUT obstacles
#
# The smooth arm is the ablation that makes the obstacles measurable. Discrete
# obstacles are the component most likely to be unlearnable at 6 Nm, so if the
# bumpy arm stalls, comparing terrain levels against the smooth arm says whether
# the obstacles caused it or the roughness did.
#
# No stairs, by design: a step tall enough to count as a stair is probably past
# what this robot can lift a foot over, and it would poison every terrain level.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

if [ "$SLURM_ARRAY_TASK_ID" -lt 3 ]; then
    ARM=bumpy;  export TASK=Velocity-BHL-Biped-Bumpy-v0;  SEED=$SLURM_ARRAY_TASK_ID
else
    ARM=smooth; export TASK=Velocity-BHL-Biped-Smooth-v0; SEED=$((SLURM_ARRAY_TASK_ID - 3))
fi
export RUN_NAME="terrain-${ARM}-s${SEED}"
export SEED MAX_ITER=6000

echo "=== node: $(hostname) | arm: $ARM | seed: $SEED | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
