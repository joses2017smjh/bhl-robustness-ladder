#!/bin/bash
#SBATCH --job-name=cloth-sort-bench
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Stage 2 throughput. NOT a training job. Cost-gated in the inner script.
# Do not submit until C0 kinematic has passed and someone is ready to measure
# low-res cloth, not to train it.
set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export NUM_ENVS=${NUM_ENVS:-8}
export BENCH_STEPS=${BENCH_STEPS:-40}
bhl_exec "$REPO/slurm/inner/cloth_sort_bench.sh"
echo "=== done: $(date) ==="
