#!/bin/bash
#SBATCH --job-name=bhl-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Gate check: does Isaac Sim actually launch headless on a Turing RTX 8000
# behind Apptainer, and does the biped task step?
#
# Small on purpose (64 envs, 10 iterations). This is not a training run; it is
# a test that the renderer, PhysX GPU pipeline, and task registration all work.
# First launch is slow (shader compile + extension registry), budget ~20 min.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

echo "=== node: $(hostname)  gpu: $CUDA_VISIBLE_DEVICES  started: $(date) ==="
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

chmod +x "$REPO"/slurm/inner/*.sh
bhl_exec "$REPO/slurm/inner/verify.sh"
bhl_exec "$REPO/slurm/inner/smoke_train.sh"

echo "=== done: $(date) ==="
