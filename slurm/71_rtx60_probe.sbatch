#!/bin/bash
#SBATCH --job-name=rtx60
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Re-probe RTX on the 6.0 stack with a shaded scene.
#
# The first probe reported "no image" on a frame whose depth was correct to the
# centimetre -- 270 units to a cube face 270 units away. That was a bare
# UsdGeom.Cube with no material and a default-oriented light: correct geometry,
# black shading. The renderer was working and the scene was not.

set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export BENCH_OUT=$REPO/results/rtx60_probe.txt
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS BENCH_OUT"
mkdir -p "$REPO/results"

echo "=== node: $(hostname) | $(date) ==="
bhl_exec "$REPO/slurm/inner/rtx60.sh"
echo "=== done: $(date) ==="
