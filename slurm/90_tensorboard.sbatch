#!/bin/bash
#SBATCH --job-name=tboard
#SBATCH --partition=share
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=2-00:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# TensorBoard as a long-lived job.
#
# Deliberately on `share`, not `gpu`: TensorBoard needs no GPU, and the gpu
# partition caps this user at 8 GPUs which the training arrays are already
# saturating. Running it here costs nothing from the training budget.
#
# Runs inside the container because TensorBoard's data server needs GLIBC_2.29
# and the login node has 2.28.

set -euo pipefail
echo "=== TensorBoard on $(hostname -s), port 6006 ==="
echo
echo "To view, from your laptop:"
echo "  ssh -N -L 6006:$(hostname -s):6006 $USER@submit-b.hpc.engr.oregonstate.edu"
echo "  then open http://localhost:6006"
echo
bash /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/scripts/tensorboard.sh 6006
