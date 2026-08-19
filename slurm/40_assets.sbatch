#!/bin/bash
#SBATCH --job-name=assets
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:20:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# CPU-only: three-way URDF/USD/MJCF audit + scripted OpenUSD stages.
# No GPU, no Isaac Sim. pxr is loaded from the bundled omni.usd.libs wheel.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_pxr

mkdir -p "$REPO/results/audit" "$REPO/results/usd"
export CACHE_DIR=$WORKSPACE/mjcf_cache

bhl_exec "$REPO/slurm/inner/assets.sh"
echo "=== done $(date) ==="
