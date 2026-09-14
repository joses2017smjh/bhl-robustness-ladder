#!/bin/bash
#SBATCH --job-name=stereo-pitch
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=00:40:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Where the maze stereo pair looks on v60, before and after `native_quat`.
# Gates the corrected stereo re-runs (afterok): a PASS means the corrected pair
# pitches down and returns terrain, and the raw tuple the old runs used pitched
# up -- i.e. the bug is real and the fix is the fix.
#
# Same stack and flags as 93_maze_ppo.sbatch, or it is not measuring those runs.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$REPO/results"
export NUM_ENVS=${NUM_ENVS:-8}
source "$REPO/slurm/inner/v60_boot_gate.sh"; v60_boot_gate
echo "=== node: $(hostname) | stereo pitch probe | $(date) ==="
bhl_exec "$REPO/slurm/inner/stereo_pitch.sh"
echo "=== done: $(date) ==="
