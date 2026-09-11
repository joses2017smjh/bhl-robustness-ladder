#!/bin/bash
#SBATCH --job-name=cloth-deform-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Construct/reset/step 8×8 Newton cloth at 8 envs. Cost-gated in the smoke.
# Not five live cloths. Not a training job.
set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$REPO/results/cloth"
export NUM_ENVS=8 BENCH_STEPS=6 INCLUDE_DEFORMABLE=1 CLOTH_RES=8
export BENCH_OUT="$REPO/results/cloth/isaac_deform_smoke.json"
bhl_exec "$REPO/slurm/inner/cloth_sort_smoke.sh"
echo "=== done: $(date) ==="
