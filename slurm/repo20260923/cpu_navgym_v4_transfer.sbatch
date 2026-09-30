#!/bin/bash
#SBATCH --job-name=navgym-v4-transfer
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-30: physics transfer of the NavGym v4 actors that PASSED their gym gate (job 21463690/21463691:
# s5 and s6 met every clause on the never-used maze seeds 40000-40047; s7 missed).
# What runs: scripts/bench/maze_explore.py --policy <actor.onnx> -- MuJoCo 3.3.5, the frozen learned
# biped gait (dr-default-s0) driven by the LEARNED NavGym v4 policy's (vx, wz) commands (no planner, no
# scripted controller; the team_sensors speed brake still filters the commands); its observation is the
# 36 lidar sector minima and a 0.2 m egocentric log-odds map from the robot's own 108 lidar rays.
# ORACLE: pose and goal coordinate. Reference: the SCRIPTED A* + turn-then-walk driver on the same mazes.
#
# PREDECLARED (2026-09-30, before any of these episodes has run):
#   Actors: the FINAL actors (actor.onnx) of the two gate-passing seeds, armV4-s5 and armV4-s6 -- the actors
#   the gym gate scored; no best-checkpoint or post-hoc choice.
#   Mazes: hard 6x6, 1 extra opening, maze seeds 50000-50011 (never used by the maze harness or any gym
#   training/evaluation: gym training < 10000, gym eval 10000-10047 / 20000-20047 / 40000-40047, earlier
#   transfer 0-11 / 30000-30011), 180 s per episode.
#   PASS iff EACH of the two actors reaches the goal in >= 10 of 12 mazes with 0 falls. Otherwise NEGATIVE;
#   INCOMPLETE if any summary is missing or not 12 episodes. A* is reported beside it, not gated.
#   Read from the per-actor summary.json files, never from exit codes.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
V4=$REPO/results/navgym-v4-20260928
OUT=${NAVGYM_V4_TRANSFER_OUT:-$REPO/results/navgym-v4-transfer-20260930}
CACHE=${TMPDIR:-/tmp}/bhl-navgym-v4-transfer-${SLURM_JOB_ID:-local}; mkdir -p "$CACHE"
if [ -e "$OUT/verdict.json" ]; then echo "REFUSING: $OUT/verdict.json exists (never overwrite a result)"; exit 3; fi
mkdir -p "$OUT"
declare -a NAMES=(armV4-s5 armV4-s6)
COMMON=(--upstream "$U" --seeds 12 --seed-start 50000 --n 6 --m 6 --extra-openings 1 --time-limit 180 --no-overwrite)
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | NavGym v4 physics transfer | $(date) ==="
sha256sum "$V4/armV4-s5/actor.onnx" "$V4/armV4-s6/actor.onnx"
run() {
  set +e
  "$PY" scripts/bench/maze_explore.py "$@" 2>&1 | grep -v "^\["
  local rc=${PIPESTATUS[0]}
  set -e
  echo "maze_explore exit code $rc (informational; the verdict is computed from summary.json)"
}
for a in "${NAMES[@]}"; do
  echo "--- LEARNED NavGym v4 $a (final actor) | maze seeds 50000-50011"
  run "${COMMON[@]}" --cache-dir "$CACHE/$a" --policy "$V4/$a/actor.onnx" --out-dir "$OUT/$a"
done
echo "--- A* reference (SCRIPTED: A* on the lidar map + turn-then-walk) | maze seeds 50000-50011"
run "${COMMON[@]}" --cache-dir "$CACHE/astar" --out-dir "$OUT/astar"
rm -rf "$CACHE"
"$PY" - "$OUT" "$REPO" <<'PY'
import hashlib, json, sys
from pathlib import Path
out, repo = Path(sys.argv[1]), Path(sys.argv[2])
names = ["armV4-s5", "armV4-s6"]

def row(p):
    try:
        s = json.loads(p.read_text())
    except Exception as exc:  # noqa: BLE001
        return {"missing": repr(exc)}
    return {k: s.get(k) for k in ("n", "success", "clean_success", "falls", "time_outs", "stuck", "median_completion_s")}

learned = {a: row(out / a / "summary.json") for a in names}
astar = row(out / "astar" / "summary.json")
complete = all("missing" not in r and r.get("n") == 12 for r in learned.values())
ok = {a: complete and r["success"] >= 10 and r["falls"] == 0 for a, r in learned.items()}
verdict = "INCOMPLETE" if not complete else ("PASS" if all(ok.values()) else "NEGATIVE")
res = {"verdict": verdict, "rule": "PASS iff EACH gate-passing final actor (armV4-s5, armV4-s6) reaches >= 10/12 goals "
       "with 0 falls on hard 6x6 maze seeds 50000-50011 (180 s); A* reported, not gated",
       "labels": {"learned": "biped gait (frozen PPO) + NavGym v4 (vx, wz) commands", "scripted": "A* reference only",
                  "oracle": "pose and goal coordinate"},
       "learned": {a: {**learned[a], "meets": ok[a],
                       "onnx_sha256": hashlib.sha256((repo / f"results/navgym-v4-20260928/{a}/actor.onnx").read_bytes()).hexdigest()}
                   for a in names},
       "astar_reference": astar}
(out / "verdict.json").write_text(json.dumps(res, indent=2) + "\n")
line = lambda r: (f"{r['success']}/12 (clean {r['clean_success']}, falls {r['falls']}, time-outs {r['time_outs']}, "
                  f"median {r['median_completion_s']} s)") if "missing" not in r else "MISSING"
print(f"NAVGYM-V4-TRANSFER VERDICT: {verdict} | " + " | ".join(f"{a}: {line(learned[a])}" for a in names)
      + f" | A* reference: {line(astar)} | learned gait + learned NavGym v4 commands; oracle pose + goal")
PY
echo "=== done: $(date) ==="
