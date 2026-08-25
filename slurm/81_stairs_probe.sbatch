#!/bin/bash
#SBATCH --job-name=stair-probe
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# G-B2: 300 iterations of single-agent PPO with depth on the stairs.
#
# The question is not "does it learn" but "can it enter the terrain at all". A
# curriculum pinned at level 0 means the first riser is a wall, and every stairs
# row in the grid would then measure the wall rather than the algorithm. Section
# 8 already found this robot falling on a 2.5 cm cable, so a 5 cm riser is not
# obviously enterable and the assumption is worth 300 iterations rather than
# 6,000 x 8 arms.
#
# Pass: terrain level moves off 0 within 300 iterations.
# Fail: swap STAIRS_TERRAINS_CFG for STAIRS_LOW_TERRAINS_CFG (3 cm) and re-probe.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export TASK=Velocity-BHL-Biped-Stairs-Depth-v0
export RUN_NAME=_probe-stairs SEED=0 MAX_ITER=300 NUM_ENVS=1024
export OVERRIDE_FILE=""

echo "=== node: $(hostname) | 300-iteration stairs entry probe | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh" 2>&1 | tail -40

LOG=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/stair-probe-${SLURM_JOB_ID}.out
LVL=$(grep -oE "Curriculum/terrain_levels: [0-9.]+" "$LOG" | tail -1 | awk '{print $NF}')
echo "=== final terrain level: ${LVL:-none} ==="
awk -v l="${LVL:-0}" 'BEGIN{
  if (l+0 > 0.05) print "G-B2 PASS | the robot enters the stairs (level " l ")";
  else {print "G-B2 FAIL | curriculum pinned at " l " -- drop step_height_range to (0.0, 0.03)"; exit 1}
}'
