#!/bin/bash
#SBATCH --job-name=waiter-asset
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# Waiter phase 1 (docs/WAITER_PROGRAM.md): build the 24-DoF + palm-collider URDF, convert it to USD on the v51 stack
# (the stack every gait in this project trained on) and check it against the shipped USD (slurm/inner/waiter_asset.sh).
# Output: assets/waiter/usd/ (workspace) and results/waiter-20261005/asset/asset_check.json.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
export WAITER_RES=$REPO/results/waiter-20261005/asset
export BHL_FORWARD_VARS="${BHL_FORWARD_VARS:-} WAITER_RES"
echo "=== waiter asset | job ${SLURM_JOB_ID:-local} | $(hostname) | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date)"
bhl_exec "$REPO/slurm/inner/waiter_asset.sh" 2>&1 | grep -vE "pthread_setaffinity|EGLError|^\[.*Warning" || true
grep -q '"upright_like_shipped"' "$WAITER_RES/asset_check.json" 2>/dev/null || { echo "WAITER ASSET CHECK: FAIL (no asset_check.json)"; exit 1; }
echo "=== done $(date)"
