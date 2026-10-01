#!/bin/bash
#SBATCH --job-name=navgym-capture-pose
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-10-01: N0 of docs/SOLUTIONS_2026-10-01.md. maze_explore.py --policy integrates each lidar packet into the learned
# ego map at the loop-top pose of the step AFTER capture (as scored in 21484212); --policy-capture-pose (opt-in)
# integrates it at the estimated pose it was captured at, as the humanoid branch already does for its grid. In a gym
# re-creation this removed armV4-s5's extra physics misses (investigator report, not independently verified), and
# on one physics training maze (9001) s5 went from stuck for 180 s to the goal in 50.1 s.
# What runs: MuJoCo 3.3.5, the frozen LEARNED biped gait (dr-default-s0) driven by the LEARNED NavGym v4 actors' (vx, wz)
# commands (no planner; the team_sensors speed brake still filters them). ORACLE: pose and goal coordinate.
#
# PREDECLARED (2026-10-01, before any of these episodes has run; rules also in scripts/bench/navgym_capture_pose.py):
#   Actors: the FINAL actors (actor.onnx) of the two gate-passing seeds, armV4-s5 and armV4-s6.
#   SCREEN: each actor with and without --policy-capture-pose on training-range hard 6x6 maze seeds 9120-9143
#     (1 extra opening; never run by any physics or gym job), 180 s per episode.
#     PASS iff armV4-s5 nets >= +4 goals with the flag (won minus lost, paired by maze) with <= 1 lost, armV4-s6 nets
#     >= -1, each actor has 0 falls with the flag, and each actor's total wall-contact steps with the flag are <= its
#     total without. McNemar exact p reported, not gated (the gym estimate is +5/24, so +4 is at the edge).
#     Otherwise NEGATIVE; INCOMPLETE if any arm is missing an episode or a pairing check fails.
#   FRESH (run only if the screen PASSES): both actors WITH --policy-capture-pose on never-used hard 6x6 maze seeds
#     60000-60011, 180 s. PASS iff EACH actor reaches >= 10/12 goals with 0 falls (21484212's rule, unchanged);
#     otherwise NEGATIVE; INCOMPLETE if any summary is missing or not 12 episodes. A* on the same mazes reported, not gated.
#   Read from the per-seed JSONs by scripts/bench/navgym_capture_pose.py, never from exit codes. Seeds 40000-40047 and
#   50000-50011 are not run.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
V4=$REPO/results/navgym-v4-20260928
OUT=${NAVGYM_CAPTURE_POSE_OUT:-$REPO/results/navgym-v4-capture-pose-20261001}
CACHE=${TMPDIR:-/tmp}/bhl-navgym-capture-pose-${SLURM_JOB_ID:-local}; mkdir -p "$CACHE"
for f in "$OUT/screen/verdict.json" "$OUT/fresh/verdict.json"; do
    [ -e "$f" ] && { echo "REFUSING: $f exists (never overwrite a result)"; exit 3; }
done
mkdir -p "$OUT/screen" "$OUT/fresh"
COMMON=(--upstream "$U" --n 6 --m 6 --extra-openings 1 --time-limit 180 --no-overwrite)
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | NavGym v4 capture-pose map integration | $(date) ==="
sha256sum "$V4/armV4-s5/actor.onnx" "$V4/armV4-s6/actor.onnx"
run() {   # run <log> <maze_explore args...>; the verdict comes from the per-seed JSONs, never from this exit code
  local log=$1; shift
  set +e
  "$PY" scripts/bench/maze_explore.py "$@" > "$log" 2>&1
  local rc=$?
  set -e
  echo "maze_explore exit code $rc (informational): $log"
}
echo "--- SCREEN | maze seeds 9120-9143 | 4 arms in parallel"
for a in armV4-s5 armV4-s6; do
  run "$OUT/screen/$a-looptop.log" "${COMMON[@]}" --seeds 24 --seed-start 9120 --cache-dir "$CACHE/$a-looptop" \
      --policy "$V4/$a/actor.onnx" --out-dir "$OUT/screen/$a-looptop" &
  run "$OUT/screen/$a-capture.log" "${COMMON[@]}" --seeds 24 --seed-start 9120 --cache-dir "$CACHE/$a-capture" \
      --policy "$V4/$a/actor.onnx" --policy-capture-pose --out-dir "$OUT/screen/$a-capture" &
done
wait
"$PY" scripts/bench/navgym_capture_pose.py screen "$OUT/screen"
SCREEN=$("$PY" -c "import json,sys; from pathlib import Path; p=Path(sys.argv[1]); print(json.loads(p.read_text())['verdict'] if p.is_file() else 'INCOMPLETE')" "$OUT/screen/verdict.json")
if [ "$SCREEN" != "PASS" ]; then
  echo "FRESH stage not run: the screen verdict is $SCREEN (predeclared: fresh runs only after a screen PASS)"
else
  echo "--- FRESH | maze seeds 60000-60011 | both actors with the flag + A* reference, in parallel"
  for a in armV4-s5 armV4-s6; do
    run "$OUT/fresh/$a-capture.log" "${COMMON[@]}" --seeds 12 --seed-start 60000 --cache-dir "$CACHE/fresh-$a" \
        --policy "$V4/$a/actor.onnx" --policy-capture-pose --out-dir "$OUT/fresh/$a-capture" &
  done
  run "$OUT/fresh/astar.log" "${COMMON[@]}" --seeds 12 --seed-start 60000 --cache-dir "$CACHE/fresh-astar" --out-dir "$OUT/fresh/astar" &
  wait
  "$PY" scripts/bench/navgym_capture_pose.py fresh "$OUT/fresh"
fi
rm -rf "$CACHE"
echo "=== done: $(date) ==="
