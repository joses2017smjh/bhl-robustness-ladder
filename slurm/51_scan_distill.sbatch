#!/bin/bash
#SBATCH --job-name=scan-distill
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=10:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Distil the privileged height-scan teacher into a blind, recurrent student.
#
# Submit with --dependency=afterok:<50_scan_teacher job id>.
#
# This is the third branch of the plateau question. If the teacher climbs past
# level 1.4 and the student cannot follow, the plateau is a sensing limit and
# this measures what a sensor would buy. If the student matches the teacher,
# then proprioception already carried the terrain information -- which is what
# section 3's "randomization alone reaches d = 0.4" was the first sign of.
#
# The student is recurrent and the teacher is not, deliberately. The scan sees
# ground the feet have not reached; the information the student is missing lives
# in its own past, so a memoryless student cannot reproduce the teacher even in
# principle and a feedforward result would be an artefact of the architecture.
#
# The student's observation set is `policy` -- the same 45 dims every other
# policy here deploys with -- so the distilled policy scores through the
# existing sim2sim harness with no changes.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

LOGROOT=$UPSTREAM/logs/rsl_rl/biped
TEACHER=$(ls -d "$LOGROOT"/*_scan-teacher-s0 2>/dev/null | head -1)
[ -n "$TEACHER" ] || { echo "no scan-teacher-s0 run to distil from"; exit 2; }
ls "$TEACHER"/model_*.pt >/dev/null 2>&1 || { echo "teacher has no checkpoint"; exit 2; }

export TASK=Velocity-BHL-Biped-Scan-v0
export SEED=0
export RUN_NAME="scan-student-s0"
export MAX_ITER=3000
export TRAIN_SCRIPT=$REPO/scripts/train_distill.py
export LOAD_RUN=$(basename "$TEACHER")
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS LOAD_RUN"

echo "=== node: $(hostname) | $RUN_NAME | teacher=$LOAD_RUN | $(date) ==="
bhl_exec "$REPO/slurm/inner/distill.sh"
echo "=== done: $(date) ==="
