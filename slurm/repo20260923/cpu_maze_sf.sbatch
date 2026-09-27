#!/bin/bash
#SBATCH --job-name=maze-sf
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --array=0-14
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# Randomized-maze sensor-fusion stress (2026-09-26): the published hard 6x6
# configuration of results/maze-explore-20260924 (--n 6 --m 6 --extra-openings 1
# --time-limit 180, maze seeds 0-11, same gait/planner/controller) with the
# maze_explore.py stress options. One array task per configuration, 12 seeds each.
#   0  yaw0            heading error 0 deg: the control (= default flags; also
#                      re-runs the published hard-6x6 table at the pinned commit)
#   1  yaw1            heading error 1 deg
#   2  yaw3            heading error 3 deg
#   3  yaw10           heading error 10 deg
#   4  yaw3-brakeoff   heading 3 deg, speed brake off (record mode)  -- exploratory, not gated
#   5  yaw10-brakeoff  heading 10 deg, speed brake off               -- exploratory, not gated
#   6  imu-est0        attitude filter in the loop, 0 ms: the IMU control
#   7-10 imu-est20/30/40/60   same filter, IMU delivery delay 20/30/40/60 ms
#   11-13 bias0.05/0.15/0.30  constant position bias (random direction per seed)
#   14 noise0.10       white position noise 0.10 m per policy step
# Pose error (SF-02 conventions of inspection_maze.py: default_rng(10_000+seed),
# bias direction drawn first) enters the ESTIMATED pose used by the lidar map,
# the A* planner and the turn-then-walk controller; the judge keeps the TRUE pose.
# IMU arm: inspection_maze.EstimatedAttitude imported unchanged -- Mahony
# (kp 1.0, ki 0.1) at 200 Hz from the physics substep hook, accelerometer
# alignment gate, NO added sensor noise (as in the 2026-09-25 pilot: 20 ms 2/2,
# 40 ms 1/2, 60 ms 0/2); this isolates delay, whereas SF-03b (humanoid,
# inspection maze) swept delay at L1 noise. The navigation heading stays oracle.
#
# PREDECLARED (before submission; SLURM_JOBS.md and the report quote this block):
#   Primary metric: reached with no fall, out of 12 (a fall ends the episode).
#   A level PASSES with >= 10/12 reached and 0 falls.
#   Heading tolerance = the largest heading error (1/3/10 deg, brake on) that
#     passes; VALID only if the 0 deg control reproduces 12/12. A smaller level
#     failing while a larger passes is flagged non_monotone, not smoothed over.
#   IMU budget = the largest delay (20/30/40/60 ms) that passes; VALID only if
#     the 0 ms estimated control is >= 11/12.
#   Position bias PASS per level (0.05/0.15/0.30 m) = >= 10/12 reached and
#     0 falls; noise 0.10 m is judged by the same per-level rule.
#   stuck (true position within 0.2 m for the final 20 s) and
#     arrived_not_judged (controller claimed the goal on its estimated pose,
#     the judge's true pose was outside the 0.30 m radius) are reported
#     separately per configuration and never counted as reached.
#   Brake-off twins are reported beside yaw3/yaw10, not gated.
#   Nothing is re-tuned after the sweep. Verdicts come from the per-seed JSONs
#   (maze_explore.py sf-verdict, the function sf_sweep_verdict), never exit codes.
#
# Usage (from the repo, MAZE_SF_COMMIT = the commit that holds these options):
#   MAZE_SF_COMMIT=<sha> sbatch --export=ALL,MAZE_SF_COMMIT=<sha> slurm/repo20260923/cpu_maze_sf.sbatch
#   after the array:  MAZE_SF_COMMIT=<sha> bash slurm/repo20260923/cpu_maze_sf.sbatch verdict
#     -> results/maze-sf-20260926/verdict.json
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
OUT=$REPO/results/maze-sf-20260926
# Run from a snapshot of a pinned commit, not the live working tree. No HEAD
# fallback: the options must be committed first, and the pin must hold them.
PIN=${MAZE_SF_COMMIT:?set MAZE_SF_COMMIT to the commit holding the maze_explore.py stress options}
if ! git -C "$REPO" show "$PIN:scripts/bench/maze_explore.py" 2>/dev/null | grep -c -- "--pose-yaw-deg" >/dev/null; then
    echo "pinned commit $PIN has no maze_explore.py stress options; refusing to run" >&2; exit 2
fi
JOB=${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local$$}}
CACHE=${TMPDIR:-/tmp}/bhl-maze-sf-$JOB-${SLURM_ARRAY_TASK_ID:-verdict}; mkdir -p "$CACHE"
WT=$CACHE/wt
git -C "$REPO" worktree add --detach "$WT" "$PIN" >/dev/null
trap 'git -C "$REPO" worktree remove --force "$WT" >/dev/null 2>&1 || true; rm -rf "$CACHE"' EXIT
cd "$WT"
export PYTHONPATH="$WT/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1

if [ "${1:-}" = verdict ]; then
    "$PY" scripts/bench/maze_explore.py sf-verdict "$OUT"
    exit 0
fi

run() {   # tag extra-args...   (hard 6x6, seeds 0-11)
    local tag=$1; shift
    echo "--- $tag"
    "$PY" scripts/bench/maze_explore.py --upstream "$U" --cache-dir "$CACHE/$tag" --seeds 12 \
        --n 6 --m 6 --extra-openings 1 --time-limit 180 --out-dir "$OUT/$tag" "$@" 2>&1 | grep -v "^\["
}
IMU=(--imu-source estimated --imu-filter mahony --imu-rate-hz 200)
echo "=== node: $(hostname) | pinned commit $PIN | task $SLURM_ARRAY_TASK_ID | $(date) ==="
case $SLURM_ARRAY_TASK_ID in
  0) run yaw0 --pose-yaw-deg 0 ;;
  1) run yaw1 --pose-yaw-deg 1 ;;
  2) run yaw3 --pose-yaw-deg 3 ;;
  3) run yaw10 --pose-yaw-deg 10 ;;
  4) run yaw3-brakeoff --pose-yaw-deg 3 --sensor-mode record ;;
  5) run yaw10-brakeoff --pose-yaw-deg 10 --sensor-mode record ;;
  6) run imu-est0 "${IMU[@]}" --imu-delay-ms 0 ;;
  7) run imu-est20 "${IMU[@]}" --imu-delay-ms 20 ;;
  8) run imu-est30 "${IMU[@]}" --imu-delay-ms 30 ;;
  9) run imu-est40 "${IMU[@]}" --imu-delay-ms 40 ;;
  10) run imu-est60 "${IMU[@]}" --imu-delay-ms 60 ;;
  11) run bias0.05 --pose-bias-m 0.05 ;;
  12) run bias0.15 --pose-bias-m 0.15 ;;
  13) run bias0.30 --pose-bias-m 0.30 ;;
  14) run noise0.10 --pose-noise-m 0.10 ;;
  *) echo "unknown task $SLURM_ARRAY_TASK_ID"; exit 2 ;;
esac
echo "=== done: $(date) ==="
