#!/bin/bash
#SBATCH --job-name=ice-control
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --array=0-3%2
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out

# LOC-11 (2026-09-24, user-authorized): the no-ice control for the placed-ice
# rung. Same launcher shape as slurm/92_ice_ppo.sbatch (v51, 6000 iterations,
# 4096 envs, the author's v51 boot lock), on the two IceControl tasks: the
# ice_* patches keep their placement and lose their low friction.
#
#   0 blind s0   1 blind s1   2 depth s0   3 depth s1
#
# Compared afterwards with the placed runs 2026-09-16_*_ppo-ice-placed-*
# (job 21344928): blind terrain level 2.5763 / 2.6055, depth 2.8541 / 2.9837.
# Predeclared reading: if the depth-minus-blind gap on the control is as large
# as on ice, the ice gap was never about friction.
#
# The parent cfgs (terrain_env_cfg.py, depth_env_cfg.py, terrains/ice.py,
# terrains/bumpy.py) carry the author's uncommitted edits; this job records
# their working-tree diff next to the results so the run stays attributable.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
mkdir -p "$WORKSPACE/overrides" "$REPO/results/repo-gpu-20260923/ice-control"
IDX=$SLURM_ARRAY_TASK_ID
CASES=(IceControl-v0:blind IceControl-Depth-v0:depth)
C=${CASES[$(( IDX / 2 ))]}
export TASK="Velocity-BHL-Biped-${C%%:*}"
export SEED=$(( IDX % 2 ))
export RUN_NAME="ppo-ice-control-${C##*:}-s${SEED}"
export MAX_ITER=${MAX_ITER:-6000} NUM_ENVS=${NUM_ENVS:-4096}
export OVERRIDE_FILE="$WORKSPACE/overrides/ice-control-${IDX}.txt"
: > "$OVERRIDE_FILE"
# Working-tree provenance of the author's files, measured when the job boots.
cd "$REPO"
DIFF="$REPO/results/repo-gpu-20260923/ice-control/worktree-${SLURM_ARRAY_JOB_ID}_${IDX}.diff"
git diff HEAD -- src/bhl_robust/terrains/ice.py src/bhl_robust/terrains/bumpy.py \
    src/bhl_robust/tasks/depth_env_cfg.py src/bhl_robust/tasks/terrain_env_cfg.py > "$DIFF"
HASH=$(sha256sum "$DIFF" | cut -c1-16)
echo "worktree diff hash $HASH (submitted against a532c90b6848c643) -> $DIFF"
[ "$HASH" = "a532c90b6848c643" ] || echo "WARNING: the author's ice files changed since submission; results carry hash $HASH"
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | HEAD $(git rev-parse --short HEAD) | $(date) ==="
bhl_exec "$REPO/slurm/inner/train.sh"
echo "=== done: $(date) ==="
