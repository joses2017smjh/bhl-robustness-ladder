#!/bin/bash
#SBATCH --job-name=wknd-setup
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
# Supply cluster account/partition through sbatch flags.
set -euo pipefail
repo=${REPO:-${SLURM_SUBMIT_DIR:-$PWD}}
# Batch directives in the reusable launcher are comments when called by bash;
# this job allocates CPUs only, while preserving its container/environment setup.
exec bash "$repo/slurm/weekend_gpu.sbatch" "$@"
