#!/bin/bash
#SBATCH --job-name=crew-lift
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Crews of three and four on ONE payload, blind and sighted.
#
# The existing "3 robot" and "4 robot" clips are not crews. The policy controls
# exactly two robots, so a crew of N was N/2 independent pairs each holding its
# own crate -- which the replay scores prove, since crew 2, 3 and 4 return the
# same peak lift to two decimal places. Pairs that never interact cannot be
# evidence about cooperation at scale.
#
# `coop_crew_env_cfg` builds the real thing: robots on a ring around a single
# object, every per-robot term generated rather than typed. Two reward terms
# had to change and one is new:
#
#   reach          same form, contact points now N on a ring (identical at N=2)
#   force closure  replaces the pairwise dot product; 1 - |mean inward unit
#                  vector|, which is algebraically the old clamp at N=2
#   crew spread    new, and only bites above N=2: force closure is satisfied by
#                  a balanced but loose crew, so without a variance penalty
#                  three robots carry and the fourth trails
#
# Payload mass scales with the crew and the edge with its cube root, so the
# per-robot share stays near the pair's 0.25 kg. Holding mass fixed would make
# the task monotonically easier with N and the comparison worthless.
#
# 2x2 design: crew size crossed with vision, so "does a bigger crew help" and
# "does depth help" are separable rather than confounded. All four arms use the
# staged recipe, the only one that lifts on the cube pair.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

IDX=$SLURM_ARRAY_TASK_ID
case $IDX in
  0) N=3; VIS="";       TAG=crew3-blind ;;
  1) N=4; VIS="";       TAG=crew4-blind ;;
  2) N=3; VIS="-Depth"; TAG=crew3-vision ;;
  3) N=4; VIS="-Depth"; TAG=crew4-vision ;;
  *) echo "bad array index $IDX"; exit 1 ;;
esac

export TASK="CoopLift-BHL-Cube-Crew${N}${VIS}-v0"
export RUN_NAME="coop-${TAG}-s0"
export SEED=0 MAX_ITER=4000
# Four articulations plus per-robot cameras will not hold 1,024 envs on a 46 GB
# card. Halving envs leaves num_steps_per_env alone so the batch stays
# comparable, and the pair runs are the control for absolute reward anyway.
if [ "$N" -ge 4 ] || [ -n "$VIS" ]; then export NUM_ENVS=512; else export NUM_ENVS=1024; fi

mkdir -p "$WORKSPACE/overrides"
export OVERRIDE_FILE="$WORKSPACE/overrides/crew-${IDX}.txt"
cat > "$OVERRIDE_FILE" <<'OVR'
env.stage_lift_on_pinch=true
OVR

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | $(date) ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
cat "$OVERRIDE_FILE"
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
