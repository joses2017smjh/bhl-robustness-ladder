#!/bin/bash
#SBATCH --job-name=yaw-grip
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Finding 10 on the upright spawn. Same 400-iter / 1024-env / CubeToShelf-Blind
# cell as 21192861_0, only the two hand DoF restored. The 6.3 -> 427.7 step
# gain was measured on the rolled+buried spawn; this is whether it survives
# once the robots stand up. plant_feet stays off -- one variable.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$WORKSPACE/overrides"

export TASK="TaskV2-BHL-CubeToShelfGrip-Blind-v0"
export RUN_NAME="yaw-cubetoshelfgrip-blind-s0"
export SEED=0 MAX_ITER=400 NUM_ENVS=1024
unset BHL_PLANT_FEET || true
export OVERRIDE_FILE="$WORKSPACE/overrides/yaw-grip.txt"
: > "$OVERRIDE_FILE"

echo "=== node: $(hostname) | $RUN_NAME | new yaw, grippers, plant_feet off | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
