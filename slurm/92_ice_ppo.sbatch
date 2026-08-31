#!/bin/bash
#SBATCH --job-name=ice-ppo
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-5
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# B3: patchy friction on flat ground. Blind against depth, two seeds each, plus
# the visible-ice arm.
#
# The depth arm is the strongest negative control in the project: the patches
# are flush, ice_gate casts the boundary to prove it, so the ray-caster returns
# the same height field either way. If depth helps here it is not because it
# sees the ice -- which is the question section 6 left open when depth beat
# blind 2.5x on uniform low friction and inverted its own prediction.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
mkdir -p "$WORKSPACE/overrides"
IDX=$SLURM_ARRAY_TASK_ID
CASES=(Ice-v0:blind Ice-Depth-v0:depth IceVisible-v0:visible)
C=${CASES[$(( IDX / 2 ))]}
export TASK="Velocity-BHL-Biped-${C%%:*}"
export SEED=$(( IDX % 2 ))
export RUN_NAME="ppo-ice-${C##*:}-s${SEED}"
export MAX_ITER=${MAX_ITER:-6000} NUM_ENVS=${NUM_ENVS:-4096}
export OVERRIDE_FILE="$WORKSPACE/overrides/ice-${IDX}.txt"
: > "$OVERRIDE_FILE"
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
