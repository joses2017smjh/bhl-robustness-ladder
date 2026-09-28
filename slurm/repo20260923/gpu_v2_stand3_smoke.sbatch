#!/bin/bash
#SBATCH --job-name=v2-stand3-smoke
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# CubeToShelfStand3 (2026-09-27, "side deck"): does the task construct, reset
# and step in blind / depth / rgb with its terms active (deck_progress,
# Stand2's lift terms, action_rate_clipped, placed reading the success
# termination, over_deck / cube_abs_x diagnostics) and Stand2's removed terms
# absent (carry, object_xy, action_rate, termination_penalty)? Gate for
# gpu_v2_stand3_train.sbatch (afterok). Same v60 + cameras recipe as
# gpu_v2_stand2_smoke.sbatch. CubeToShelfStand3 is a DIFFERENT, EASIER task
# than CubeToShelf and than CubeToShelfStand2 and is never compared with them.
# Verdict: the SMOKE-VERDICT line, computed from the printed table and the
# manager tables in the log (inner_v2_stand3_smoke.sh), not from an exit code.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$REPO/results/repo-gpu-20260923/stand3_2026-09-27"
export NUM_ENVS=${NUM_ENVS:-4}
echo "=== node: $(hostname) | CubeToShelfStand3 smoke | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
bhl_exec "$REPO/slurm/repo20260923/inner_v2_stand3_smoke.sh"
echo "=== done: $(date) ==="
