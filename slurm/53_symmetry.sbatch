#!/bin/bash
#SBATCH --job-name=sym-aug
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# Symmetry augmentation at s = 1.0: one knob, scored on the existing protocol.
#
# BHL is bilaterally symmetric and nothing upstream exploits that. rsl-rl 3.x
# will mirror each batch, but only if it is handed a robot-specific mirror map;
# `bhl_robust.tasks.symmetry` is that map -- sagittal reflection, so the y
# component of a vector and the roll/yaw components of a rotation flip, pitch
# does not, and the two leg blocks swap.
#
# Why this robot specifically: a symmetric gait is more torque-efficient than an
# asymmetric one, and torque is precisely what an 11.3 kg machine with 6 Nm
# joints does not have. The hypothesis that symmetry augmentation buys more on a
# weak robot than on a strong one is testable on the ladder that already exists
# here, and it slots in as a fourth rung with no new infrastructure.
#
# Arm 0 is data augmentation, arm 1 adds the mirror loss, which is the ablation
# Mittal et al. report as behaving differently -- augmentation changes the data
# distribution, the loss changes the objective.
#
# The control is the already-trained dr-default rung; this deliberately does not
# retrain it.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

IDX=$SLURM_ARRAY_TASK_ID
export TASK=Velocity-Berkeley-Humanoid-Lite-Biped-v0
export SEED=0 MAX_ITER=6000

case $IDX in
  0) export BHL_SYMMETRY=aug        RUN_NAME=sym-aug-s0 ;;
  1) export BHL_SYMMETRY=aug+loss   BHL_MIRROR_COEFF=1.0 RUN_NAME=sym-augloss-s0 ;;
  *) echo "bad array index $IDX"; exit 1 ;;
esac

echo "=== node: $(hostname) | $RUN_NAME | symmetry=$BHL_SYMMETRY | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
