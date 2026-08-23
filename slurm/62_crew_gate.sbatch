#!/bin/bash
#SBATCH --job-name=crew-gate
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Build and step all four crew variants before any of them gets a training slot.
# 61_crew_lift depends on this with afterok, so a broken config costs an hour
# rather than four 24-hour allocations producing numbers about the wrong scene.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OMP_NUM_THREADS=8
export GATE_OUT=$REPO/results/crew_gate.txt
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS GATE_OUT WORKSPACE"
mkdir -p "$REPO/results"; rm -f "$GATE_OUT"

echo "=== node: $(hostname) | $(date) ==="
bhl_exec "$REPO/slurm/inner/crew_validate.sh"
echo "=== done: $(date) ==="
