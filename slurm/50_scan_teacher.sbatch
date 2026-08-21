#!/bin/bash
#SBATCH --job-name=scan-teach
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=16:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Privileged height-scan teacher: is the terrain plateau sensing, or torque?
#
# Section 3 reports the terrain curriculum stalling at level 1.4 of 9 and
# attributes it to "6 Nm joints with no height sensing". That is two hypotheses
# in one sentence and they are separable. This run gives the policy perfect
# terrain knowledge -- an 11 x 7 height grid at 10 cm spacing, yaw-aligned --
# and changes nothing else: same terrain menu, same level curriculum, same
# s = 1.0 randomization, same PPO config, same 6,000 iterations.
#
# The scan is a warp ray-cast, the same mechanism as the depth camera and the
# same reason neither needs the RTX renderer that segfaults on this cluster.
#
# Three outcomes, all of them results:
#   teacher climbs past 1.4  -> the plateau is a sensing limit, and this
#                               quantifies what putting a sensor on BHL buys.
#   teacher also stalls at 1.4 -> the plateau is torque. Even with perfect
#                               terrain knowledge an 11.3 kg robot with 6 Nm
#                               joints cannot do better. Nobody has published
#                               that for this platform.
#   (the third branch needs the student -- see 51_scan_distill.)
#
# Two seeds, because the number being compared is a curriculum level and
# section 3's existing level claim is a two-seed mean.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export TASK=Velocity-BHL-Biped-Scan-v0
export SEED=$SLURM_ARRAY_TASK_ID
export RUN_NAME="scan-teacher-s${SEED}"
export MAX_ITER=6000

echo "=== node: $(hostname) | $RUN_NAME | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
