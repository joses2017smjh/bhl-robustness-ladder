#!/bin/bash
#SBATCH --job-name=convex-usd
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Re-convert the biped URDF with convex-decomposition collision from the
# visual meshes. Upstream's converter keeps the primitive geoms; this is the
# other setting. The USD lands outside the git tree so external/ stays clean.
#
# Must finish before slurm/33_collision_train.sbatch (use --dependency=afterok).

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export BHL_CONVEX_USD_DIR=$WORKSPACE/assets/bhl_biped_convex
export BENCH_OUT=$WORKSPACE/assets/convex_usd.txt
mkdir -p "$BHL_CONVEX_USD_DIR"
rm -f "$BENCH_OUT"

echo "=== convert convex USD  $(date)  $(hostname) ==="
bhl_exec "$REPO/slurm/inner/convert_convex.sh"
echo "=== marker ==="
cat "$BENCH_OUT"
ls -lh "$BHL_CONVEX_USD_DIR"
echo "=== done $(date) ==="
