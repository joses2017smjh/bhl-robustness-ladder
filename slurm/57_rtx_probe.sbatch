#!/bin/bash
#SBATCH --job-name=rtx-probe
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgxh
#SBATCH --constraint="rtx8000|a40|l40s|h100|h200"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=06:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# RT cores are a hard requirement and the constraint says so. §6 already ruled
# out "the hardware is wrong" as the cause of the 5.1 segfault by running it on
# the Quadro RTX 8000s, so a probe that landed on a V100 (sm_70, no RT cores)
# would produce a failure that means nothing and reads like a result.
#
# This is the question §6 left open: 5.1's renderer dies here, 6.0.1 survives
# the call that kills it, and nobody has checked whether 6.0.1 returns an actual
# image or what it would cost to keep one in the training loop.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS WORKSPACE RTX_COUNTS RTX_RES"
export RTX_COUNTS=${RTX_COUNTS:-"1 4 16 64"}
export RTX_RES=${RTX_RES:-"64 128"}
mkdir -p "$REPO/results"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "Isaac Sim 6.0 wants >= 580.95.05; 5.1's own driver-requirements.json"
echo "recommends 535.161.7, which is the mismatch S6 blamed for the segfault."
echo

bhl_exec "$REPO/slurm/inner/rtx_probe.sh"

echo "=== done: $(date) ==="
