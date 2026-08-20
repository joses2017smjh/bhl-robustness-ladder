#!/bin/bash
#SBATCH --job-name=coop-lift
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Two 22-DoF humanoids, one object. Standing-spawn / σ=0.15 / ungated-lift
# finished: cube and ball died in 0.2 s, ladder height was a toss (reach=0).
# This requeue is squat-pinch spawn, two-scale reach, lift gated on pinch.
# One seed each — seeds multiply only after a pinch appears. Submitted even
# if QOSGrpGRES is full; it waits. Do not route to `eecs`.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
# Two Kit processes on one node deadlock on a shared ov-cache (CUDA 999 +
# "another kit process is locking it"). Give every array task its own dir.
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

IDX=$SLURM_ARRAY_TASK_ID
export SEED=0 MAX_ITER=4000
# eecs nodes include 11 GB 2080 Tis; 1024 dual-humanoid envs OOM there.
# The gpu-partition RTX 8000s (48 GB) take 1024.
gpu_mib=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | awk 'NR==1{print int($1)}')
if [ "${gpu_mib:-48000}" -lt 20000 ]; then
  export NUM_ENVS=256
else
  export NUM_ENVS=1024
fi

case $IDX in
  0) export TASK=CoopLift-BHL-Cube-v0   RUN_NAME=coop-cube-pinch-s0 ;;
  1) export TASK=CoopLift-BHL-Ladder-v0 RUN_NAME=coop-ladder-pinch-s0 ;;
  2) export TASK=CoopLift-BHL-Ball-v0   RUN_NAME=coop-ball-pinch-s0 ;;
  *) echo "bad array index $IDX"; exit 1 ;;
esac

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
