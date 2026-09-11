#!/bin/bash
#SBATCH --job-name=cloth-c0
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --time=00:20:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Kinematic C0/C1/C5. No GPU, no Isaac. The planner either sorts rectangles
# or it does not; that answer is due before anyone pays for Newton cloth.
set -euo pipefail
export REPO=/nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder
export PY=python3
bash "$REPO/slurm/inner/cloth_c0.sh"
echo "=== done: $(date) ==="
