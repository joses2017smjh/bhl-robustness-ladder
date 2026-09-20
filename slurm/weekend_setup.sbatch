#!/bin/bash
#SBATCH --job-name=wknd-setup
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
set -euo pipefail
repo=/nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder
# Batch directives in the reusable launcher are comments when called by bash;
# this job allocates CPUs only, while preserving its container/environment setup.
exec bash "$repo/slurm/weekend_gpu.sbatch" "$@"
