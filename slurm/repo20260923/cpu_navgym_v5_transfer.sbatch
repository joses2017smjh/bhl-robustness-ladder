#!/bin/bash
#SBATCH --job-name=navgym-v5-transfer
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=4
#SBATCH --mem=12G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-10-02: physics transfer of the NavGym v5 actors (N2 of docs/SOLUTIONS_2026-10-01.md) that PASSED their gym gate
# (slurm/repo20260923/cpu_navgym_v5.sbatch). It reads the FINAL gym verdict (results/navgym-v5-20261002/verdict_v5.json)
# and runs nothing unless v5 PASSED: then only the gate-passing seeds' FINAL actors (actor.onnx; no best checkpoint, no
# post-hoc choice), each with --policy-capture-pose, plus the A* reference on the same mazes.
# What runs: scripts/bench/maze_explore.py --policy <actor.onnx> --policy-capture-pose -- MuJoCo 3.3.5, the frozen LEARNED
# biped gait (dr-default-s0) driven by the LEARNED NavGym v5 policy's (vx, wz) commands (no planner, no scripted controller;
# the team_sensors speed brake still filters the commands, with its range, stereo-depth and IMU terms -- the gym modelled
# the range term only, see the "brake" note in cpu_navgym_v5.sbatch). Its observation (lidar, near, map_visit,
# map_coarse, goal) is built by the gym's own build_obs from the robot's own 108 lidar rays (10 Hz packets, each
# integrated into the ego map at its capture pose) and a visitation memory stepped once per control step. ORACLE: pose
# and goal coordinate.
# Reference (SCRIPTED, not gated): A* on the lidar map + turn-then-walk.
#
# PREDECLARED (2026-10-02, FROZEN before any v5 training or episode):
#   - Gym gate: V4_RULE's clauses unchanged (5x5 success >= 0.80, 6x6 success >= 0.70, 6x6 collision <= 0.15, n = 48
#     per size, the run reached its last step (num_timesteps >= 30 000 000), and it is the declared v5 configuration),
#     judged on each seed's FINAL actor (deterministic) on the NEVER-USED set maze seeds 61000-61047 (dynamics
#     reset(seed = 61000 + k)). v5 PASSES iff >= 2 of the 3 seeds (8, 9, 10) meet every clause; otherwise NEGATIVE.
#     Seeds 10000-10047 and the best checkpoint are reported, not gated.
#   - Physics transfer, only if the gym gate PASSES, only for gate-passing final actors: MuJoCo 3.3.5, the frozen learned
#     biped gait, LEARNED v5 (vx, wz) commands, ORACLE pose and goal, --policy-capture-pose; hard 6x6, 1 extra opening,
#     maze seeds 63000-63011, 180 s. PASS iff EACH gate-passing final actor reaches >= 10/12 goals with 0 falls; otherwise
#     NEGATIVE; INCOMPLETE if a summary is missing or not 12 episodes. A* on the same mazes reported, not gated.
#   - Reported, not gated: yaw flips per second and |wz| saturation share in the gym evaluation. Maze seeds 40000-40047
#     and 50000-50011 are scored seeds of earlier tests: never run them.
#   Validity (INCOMPLETE, never PASS): an actor's episodes must be exactly maze seeds 63000-63011, each run by that actor's
#   final actor.onnx with capture-pose integration and the v5 visitation memory recorded, on a hard 6x6 maze with 1 extra
#   opening, and its summary.json must agree (180 s, capture pose, success count). Code: navgym_train.py
#   V5_TRANSFER_RULE / v5_gate_passing_actors / judge_v5_transfer, read from the per-seed JSONs and summary.json files,
#   never from exit codes; verdict.json (or, when the gate did not pass, not_run.json) is never overwritten.
#   Only a FINAL gym verdict is read (a non-final one stops the launcher, exit 4, nothing written), and verdict.json is
#   written only by the --final verdict call below, after every episode run has exited. A transfer run that did not finish
#   (node failure, time limit, cancel: $OUT holds files but no verdict.json) is never run again on top of its files: the
#   launcher refuses (exit 3) before running anything. Moving $OUT aside and resubmitting would be a second touch of maze
#   seeds 63000-63011 (the coordinator's decision); recording the partial run as INCOMPLETE is
#   `navgym_train.py verdict-v5-transfer $OUT --gym-verdict $GV --final`.
#   Maze seeds 63000-63011: checked 2026-10-02 by grep (ledger, docs, scripts, slurm, results, logs) -- never used.
#   Labels: LEARNED gait + LEARNED NavGym v5 (vx, wz) commands; ORACLE pose + goal; A* reference SCRIPTED.
#
# Submit after the gym verdict job (cpu_navgym_v5.sbatch, NAVGYM_MODE=verdict):
#   sbatch --dependency=afterok:<verdict job> cpu_navgym_v5_transfer.sbatch
# Exit codes: 0 = ran, or did not run because the gym gate did not pass (not_run.json says why); 3 = refused: a verdict.json
# or not_run.json exists, or $OUT holds the files of a run that did not finish; 4 = no FINAL gym verdict file (run the
# verdict step first).
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
V5=${NAVGYM_V5_ROOT:-$REPO/results/navgym-v5-20261002}
OUT=${NAVGYM_V5_TRANSFER_OUT:-$REPO/results/navgym-v5-transfer-20261002}
GV=$V5/verdict_v5.json
for f in "$OUT/verdict.json" "$OUT/not_run.json"; do
  [ -e "$f" ] && { echo "REFUSING: $f exists (never overwrite a result)"; exit 3; }
done
if [ -e "$OUT" ] && [ -n "$(find "$OUT" -mindepth 1 -print -quit)" ]; then
  echo "REFUSING: $OUT is not empty but holds no verdict.json or not_run.json: an earlier transfer run did not finish."
  echo "  Nothing was run. maze_explore --no-overwrite would stop at once and the verdict would record a permanent INCOMPLETE."
  echo "  Coordinator: move $OUT aside and resubmit (a second touch of the transfer's scored mazes), or record the partial run"
  echo "  as INCOMPLETE: $PY scripts/bench/navgym_train.py verdict-v5-transfer $OUT --gym-verdict $GV --final"
  exit 3
fi
[ -f "$GV" ] || { echo "NO GYM VERDICT: $GV is missing; run cpu_navgym_v5.sbatch with NAVGYM_MODE=verdict first"; exit 4; }
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | NavGym v5 physics transfer | gym verdict $GV | $(date) ==="
# a gym verdict that is not final makes this exit 4 (set -e), before anything is written
LIST=$("$PY" scripts/bench/navgym_train.py verdict-v5-transfer "$OUT" --gym-verdict "$GV" --list-actors)
ACTORS=()
for a in $LIST; do ACTORS+=("$a"); done
mkdir -p "$OUT"
if [ ${#ACTORS[@]} -eq 0 ]; then
  "$PY" - "$GV" "$OUT/not_run.json" <<'PY'
import json, sys
from pathlib import Path
gv = json.loads(Path(sys.argv[1]).read_text())
rec = {"status": "NOT RUN", "reason": "the physics transfer runs only after a FINAL gym PASS (predeclared); the gym verdict is "
       f"{gv.get('verdict')} (final={gv.get('final')}, seeds passing {gv.get('seeds_passing')})",
       "gym_verdict": {"path": sys.argv[1], "verdict": gv.get("verdict"), "final": gv.get("final"),
                       "gate_passing_seeds": gv.get("gate_passing_seeds")}}
out = Path(sys.argv[2])
if out.exists():
    sys.exit(f"REFUSING: {out} exists")
out.write_text(json.dumps(rec, indent=2) + "\n")
print("NAVGYM-V5-TRANSFER: not run by the predeclared rule -- " + rec["reason"])
PY
  exit 0
fi
echo "gate-passing final actors: ${ACTORS[*]} | mujoco $("$PY" -c 'import mujoco; print(mujoco.__version__)')"
for a in "${ACTORS[@]}"; do sha256sum "$V5/$a/actor.onnx"; done
CACHE=${TMPDIR:-/tmp}/bhl-navgym-v5-transfer-${SLURM_JOB_ID:-local}; mkdir -p "$CACHE"
COMMON=(--upstream "$U" --seeds 12 --seed-start 63000 --n 6 --m 6 --extra-openings 1 --time-limit 180 --no-overwrite)
run() {   # run <log> <maze_explore args...>; the verdict comes from the per-seed JSONs, never from this exit code
  local log=$1; shift
  set +e
  "$PY" scripts/bench/maze_explore.py "$@" > "$log" 2>&1
  local rc=$?
  set -e
  echo "maze_explore exit code $rc (informational): $log"
}
echo "--- maze seeds 63000-63011 | LEARNED v5 final actors (capture pose) + SCRIPTED A* reference, in parallel"
for a in "${ACTORS[@]}"; do
  run "$OUT/$a.log" "${COMMON[@]}" --cache-dir "$CACHE/$a" --policy "$V5/$a/actor.onnx" --policy-capture-pose --out-dir "$OUT/$a" &
done
run "$OUT/astar.log" "${COMMON[@]}" --cache-dir "$CACHE/astar" --out-dir "$OUT/astar" &
wait
rm -rf "$CACHE"
"$PY" scripts/bench/navgym_train.py verdict-v5-transfer "$OUT" --gym-verdict "$GV" --final
echo "=== done: $(date) ==="
