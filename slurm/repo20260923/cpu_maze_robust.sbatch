#!/bin/bash
#SBATCH --job-name=maze-robust
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-5
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# Randomized-maze robustness (2026-09-26): the published 24/24 table
# (results/maze-explore-20260924, job 21412035) under three perturbations,
# same maze seeds 0-11, same controller, no source change.
#   0/1  sensor dropout 35 %  base 5x5 / hard 6x6   (reactive_dropout)
#   2/3  sensor dropout 70 %  base 5x5 / hard 6x6   (exploratory, not gated)
#   4    brake off (record mode)         base + hard
#   5    random initial heading          base + hard
# Predeclared (SLURM_JOBS.md, 2026-09-26): dropout 35 % PASS per configuration
# = >= 10/12 reached, 0 falls. Brake-off: clean >= 11/12 in both = the brake is
# not load-bearing for the clean result; any <= 9/12 = load-bearing; 10 =
# inconclusive. Random heading PASS = >= 10/12 reached, 0 falls per config.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
CACHE=${TMPDIR:-/tmp}/bhl-maze-robust-$SLURM_JOB_ID; mkdir -p "$CACHE"
OUT=$REPO/results/maze-robust-20260926
# Run from a snapshot of a pinned commit, not the live working tree: other work
# edits scripts/bench/maze_explore.py while these jobs may be running.
PIN=${MAZE_ROBUST_COMMIT:-$(git -C "$REPO" rev-parse HEAD)}
WT=$CACHE/wt
git -C "$REPO" worktree add --detach "$WT" "$PIN" >/dev/null
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true; rm -rf "$CACHE"' EXIT
cd "$WT"
export PYTHONPATH="$WT/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
run() {   # tag n m openings time_limit extra-args...
    local tag=$1 n=$2 m=$3 ex=$4 tl=$5; shift 5
    echo "--- $tag"
    "$PY" scripts/bench/maze_explore.py --upstream "$U" --cache-dir "$CACHE/$tag" --seeds 12 --n "$n" --m "$m" \
        --extra-openings "$ex" --time-limit "$tl" --out-dir "$OUT/$tag" "$@" 2>&1 | grep -v "^\["
}
echo "=== node: $(hostname) | pinned commit $PIN | task $SLURM_ARRAY_TASK_ID | $(date) ==="
case $SLURM_ARRAY_TASK_ID in
  0) run dropout35-base-5x5 5 5 2 240 --sensor-mode reactive_dropout --dropout-probability 0.35 ;;
  1) run dropout35-hard-6x6 6 6 1 240 --sensor-mode reactive_dropout --dropout-probability 0.35 ;;
  2) run dropout70-base-5x5 5 5 2 240 --sensor-mode reactive_dropout --dropout-probability 0.70 ;;
  3) run dropout70-hard-6x6 6 6 1 240 --sensor-mode reactive_dropout --dropout-probability 0.70 ;;
  4) run brakeoff-base-5x5 5 5 2 150 --sensor-mode record
     run brakeoff-hard-6x6 6 6 1 180 --sensor-mode record ;;
  5) run heading-base-5x5 5 5 2 150 --random-heading
     run heading-hard-6x6 6 6 1 180 --random-heading ;;
esac
echo "=== done: $(date) ==="
