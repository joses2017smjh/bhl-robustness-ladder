#!/bin/bash
#SBATCH --job-name=plank-rerun
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --array=0-5
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# The six plank cells, re-run on a scene that does not eject its own payload.
#
# The originals trained with the robots at x = +/-0.85, where their hands begin
# 18.7 cm inside a plank spanning +/-0.75. The contact solver threw the payload
# 22 cm upward before the policy acted, so every one of those cells measured a
# scene rather than a policy -- and their gripper arms showed none of the
# survival effect the cube and ball arms did (episode length 11.6 against 428).
#
# Stand-off is 1.05 m now, verified by a zero-action rollout holding the plank
# at z = 0.300 for forty steps where it previously reached 0.52.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$WORKSPACE/overrides"

IDX=$SLURM_ARRAY_TASK_ID
VIS=(Blind Depth Rgb)
V=${VIS[$(( IDX % 3 ))]}
if [ $(( IDX / 3 )) -eq 0 ]; then
    export TASK="TaskV2-BHL-PlankToWall-${V}-v0"     RUN_NAME="v2b-planktowall-${V,,}-s0"
else
    export TASK="TaskV2-BHL-PlankToWallGrip-${V}-v0" RUN_NAME="v2b-planktowallgrip-${V,,}-s0"
fi
export SEED=0 MAX_ITER=${MAX_ITER:-8000} NUM_ENVS=${NUM_ENVS:-1024}
export OVERRIDE_FILE="$WORKSPACE/overrides/plank-rerun-${IDX}.txt"
: > "$OVERRIDE_FILE"

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
