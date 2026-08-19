#!/bin/bash
#SBATCH --job-name=coll-mesh
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Collision-representation ablation: identical policies, convex-decomposition
# mesh collision instead of the primitive boxes/cylinders upstream ships.
# Two seeds, matched to the s=1.0 biped baseline (already trained) so the
# only variable is the colliding geometry.
#
# Submit with --dependency=afterok:<convert-job> so the USD exists first.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export BHL_CONVEX_USD=$WORKSPACE/assets/bhl_biped_convex/berkeley_humanoid_lite_biped.usd
[ -f "$BHL_CONVEX_USD" ] || { echo "missing $BHL_CONVEX_USD"; exit 2; }

export TASK=Velocity-BHL-Biped-ConvexCollision-v0
export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME="coll-convex-s${SEED}"
export MAX_ITER=6000
export OVERRIDE_FILE=""

echo "=== node: $(hostname) | $RUN_NAME | usd=$BHL_CONVEX_USD | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
