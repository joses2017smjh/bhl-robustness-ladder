#!/bin/bash
#SBATCH --job-name=terr-ppo
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --array=0-7
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# The single-agent rows of the grid: PPO on the two new terrains, depth and
# blind, two seeds. These do not wait on the MARL port -- they are the controls
# every MARL row will be compared against, and they are also, on their own, the
# per-terrain depth claim.
#
# The pairing is the point. `slippery` shares its geometry with `bumpy` exactly
# and differs only in contact friction; `stairs` is nothing but geometry. A
# ray-cast camera returns geometry and nothing else, so the pre-registered
# prediction is that depth pays on stairs and shows no gain on slippery. Running
# blind and sighted on both is what makes that falsifiable rather than asserted:
# without the blind arms there is no delta to report.
#
# `rough` is deliberately absent. Section 3 has blind (1.443 / 1.446 / 1.417) and
# section 6 has depth (1.601 / 1.598) at these exact settings; re-running it
# would spend eight GPU-days confirming a number already in the README.
#
# NUM_ENVS is fixed at 4096 across every arm here and every arm of the grid.
# Batch is envs x steps, so letting it drift between terrains or algorithms
# would confound the comparison with sample budget.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export OV_CACHE=/scratch/$USER/ov-cache-${SLURM_JOB_ID}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${SLURM_JOB_ID}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH"

IDX=$SLURM_ARRAY_TASK_ID
TERRAINS=(slippery stairs)
VIS=(depth blind)
T=${TERRAINS[$(( (IDX / 4) % 2 ))]}
V=${VIS[$(( (IDX / 2) % 2 ))]}
# SEED_OFFSET lets a third seed reuse the same index->arm mapping instead of
# renumbering it: --array=0,2,4,6 --export=ALL,SEED_OFFSET=2 is one seed of
# each of the four cells. Changing the arithmetic instead would silently
# repoint every index in this file, including the ones already published.
S=$(( IDX % 2 + ${SEED_OFFSET:-0} ))

case "$T-$V" in
  slippery-depth) export TASK=Velocity-BHL-Biped-Slippery-Depth-v0 ;;
  slippery-blind) export TASK=Velocity-BHL-Biped-Slippery-v0 ;;
  stairs-depth)   export TASK=Velocity-BHL-Biped-Stairs-Depth-v0 ;;
  stairs-blind)   export TASK=Velocity-BHL-Biped-Stairs-v0 ;;
esac

# Run-name convention the analysis script parses: {algo}-{terrain}-{vision}-s{seed}
export RUN_NAME="ppo-${T}-${V}-s${S}"
export SEED=$S MAX_ITER=6000 NUM_ENVS=4096 OVERRIDE_FILE=""

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
