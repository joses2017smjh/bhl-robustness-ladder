#!/bin/bash
#SBATCH --job-name=pytest-full
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# Full test suite on the working tree (the interactive allocation's 6 GB cgroup OOM-kills it).
cd /nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
export PYTHONPATH=$PWD/src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
echo "HEAD $(git rev-parse --short HEAD) on $(hostname) $(date)"
/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python -m pytest -q tests -p no:cacheprovider
echo "pytest exit $?"
