#!/bin/bash
#SBATCH --job-name=spawn-quat
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# CubeToShelf spawn quaternion candidates on v60. After 21342562, robot
# init_state.rot goes through native_quat of the documented yaw. This re-probe
# fails unless as_configured STANDING. No re-train until it does.
set -euo pipefail
export BHL_STACK=v60
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$REPO/results"
export NUM_ENVS=${NUM_ENVS:-4}
export BENCH_OUT="${BENCH_OUT:-$REPO/results/spawn_quat_probe.json}"
source "$REPO/slurm/inner/v60_boot_gate.sh"; v60_boot_gate
echo "=== node: $(hostname) | spawn quat probe | $(date) ==="
bhl_exec "$REPO/slurm/inner/spawn_quat.sh"
echo "=== done: $(date) ==="
