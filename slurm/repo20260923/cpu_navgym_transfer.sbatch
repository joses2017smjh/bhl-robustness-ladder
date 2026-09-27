#!/bin/bash
#SBATCH --job-name=navgym-transfer
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-27: NavGym v2 actors on the PHYSICS robot -- EXPLORATORY.
# The v2 arm did NOT pass its predeclared gym bar (seed 0 missed 5x5 by 0.03), so the declared
# transfer never became due; this run is exploratory and every output says so.
#
# What runs: scripts/bench/maze_explore.py --policy <actor.onnx> -- MuJoCo 3.3.5, the frozen learned
# biped gait (dr-default-s0), the robot's own 108-ray lidar feeding the SAME 36 sector minima + 0.2 m
# egocentric log-odds map the gym builds (bhl_robust.navgym.env.build_obs), the actor's deterministic
# mean as the (vx, wz) command. LEARNED: gait and navigation commands. ORACLE: pose and goal
# coordinate. Not in the gym: lidar arrives in 10 Hz packets (the gym scans every 0.04 s step) and the
# team_sensors speed brake still filters the commands (as for the A* table).
#
# Actors (results/navgym-v2-20260926):  armA-s1 actor_best.onnx (4 M steps), armA-s1 actor.onnx (final),
#                                         armA-s0 actor_best.onnx (14 M steps).
# Set 1 "hard-6x6" = the A* table's settings (results/maze-explore-20260924/hard-6x6, job 21412035):
#   --n 6 --m 6 --extra-openings 1 --time-limit 180, maze seeds 0-11, initial heading +x (no
#   --random-heading), reactive sensors. CAVEAT: maze seeds 0-11 are inside the gym's training-seed
#   range (< 10 000, same generator, same extra_openings), so these layouts were probably trained on.
# Set 2 "fresh-6x6": the same settings on maze seeds 30000-30011 (never used by any gym training or
#   evaluation; the v3 scored set is 20000-20047), for the three actors AND the A* baseline.
# Output: results/navgym-transfer-20260927/<actor>/ (set 1), .../fresh-6x6/<actor|astar>/ (set 2),
#   .../transfer_table.json with the summary lines beside the A* table (A* 12/12 on set 1).
# Submit the render job with --dependency=afterany:<this job>; it checks transfer_table.json itself.
# Render gate (declared now, applied by gpu_navgym_transfer_render.sbatch from transfer_table.json):
#   the actor with the most set-1 successes (ties: more clean successes, then lower median completion
#   time, then the order above) is rendered only if it reaches >= 8/12 on set 1; the episode is its
#   median-time successful set-1 seed (summary.json "median_seed_by_time").
# Login-node pilot (2026-09-27, no render): armA-s1 actor_best.onnx, seeds 0 and 1: both goals, clean,
#   75.4 s / 59.8 s (A* 88.1 / 73.1 s); armA-s1 actor.onnx, seed 1: goal in 161.2 s (median forward
#   command 0, |wz| command 1.0 -- the saturated final actor dithers here too).
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
V2=$REPO/results/navgym-v2-20260926
OUT=${NAVGYM_TRANSFER_OUT:-$REPO/results/navgym-transfer-20260927}
CACHE=${TMPDIR:-/tmp}/bhl-navgym-transfer-${SLURM_JOB_ID:-local}; mkdir -p "$CACHE"
if [ -e "$OUT/transfer_table.json" ]; then echo "REFUSING: $OUT/transfer_table.json exists (never overwrite a result)"; exit 3; fi
mkdir -p "$OUT"
declare -a NAMES=(armA-s1-actor_best armA-s1-actor armA-s0-actor_best)
declare -A ONNX=([armA-s1-actor_best]=$V2/armA-s1/actor_best.onnx [armA-s1-actor]=$V2/armA-s1/actor.onnx [armA-s0-actor_best]=$V2/armA-s0/actor_best.onnx)
COMMON=(--upstream "$U" --seeds 12 --n 6 --m 6 --extra-openings 1 --time-limit 180)
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | EXPLORATORY NavGym transfer | $(date) ==="
sha256sum "${ONNX[@]}"
# one run = one maze_explore.py call; a failing run must not stop the others or the table (its summary is then missing)
run() {
  set +e
  "$PY" scripts/bench/maze_explore.py "$@" 2>&1 | grep -v "^\["
  local rc=${PIPESTATUS[0]}
  set -e
  echo "maze_explore exit code $rc (informational; the table is computed from summary.json)"
}
for a in "${NAMES[@]}"; do
  echo "--- set 1 hard-6x6 (maze seeds 0-11) | LEARNED $a"
  run "${COMMON[@]}" --seed-start 0 --cache-dir "$CACHE/h-$a" --policy "${ONNX[$a]}" --out-dir "$OUT/$a"
done
for a in "${NAMES[@]}"; do
  echo "--- set 2 fresh-6x6 (maze seeds 30000-30011) | LEARNED $a"
  run "${COMMON[@]}" --seed-start 30000 --cache-dir "$CACHE/f-$a" --policy "${ONNX[$a]}" --out-dir "$OUT/fresh-6x6/$a"
done
echo "--- set 2 fresh-6x6 (maze seeds 30000-30011) | A* baseline (SCRIPTED: A* on the lidar map + turn-then-walk)"
run "${COMMON[@]}" --seed-start 30000 --cache-dir "$CACHE/f-astar" --out-dir "$OUT/fresh-6x6/astar"
rm -rf "$CACHE"
# The table (from the per-set summary.json files, never from exit codes)
"$PY" - "$OUT" "$REPO" "${NAMES[@]}" <<'PYEOF'
import hashlib, json, os, sys
from pathlib import Path
out, repo, names = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3:]
astar = json.loads((repo / "results/maze-explore-20260924/hard-6x6/summary.json").read_text())
onnx = {"armA-s1-actor_best": "results/navgym-v2-20260926/armA-s1/actor_best.onnx", "armA-s1-actor": "results/navgym-v2-20260926/armA-s1/actor.onnx",
        "armA-s0-actor_best": "results/navgym-v2-20260926/armA-s0/actor_best.onnx"}

def row(p, label):
    s = json.loads(p.read_text()) if p.is_file() else None
    if s is None:
        return {"label": label, "missing": str(p)}
    ok = s["n"] == 12 and s["seeds"] == sorted(s["seeds"]) and len(s["seeds"]) == 12
    return {"label": label, "complete": ok, "n": s["n"], "success": s["success"], "clean_success": s["clean_success"], "falls": s["falls"],
            "time_outs": s["time_outs"], "stuck": s.get("stuck"), "median_completion_s": s["median_completion_s"], "completion_s": s["completion_s"],
            "median_seed_by_time": s["median_seed_by_time"], "seeds": s["seeds"], "summary": os.path.relpath(p, repo)}

def line(r):
    if "missing" in r:
        return f"{r['label']}: NO SUMMARY ({r['missing']})"
    return (f"{r['label']}: {r['success']}/{r['n']} (clean {r['clean_success']}, falls {r['falls']}, time-outs {r['time_outs']}, "
            f"median {r['median_completion_s']} s)")

t = {"status": "EXPLORATORY -- the v2 arm did not pass its gym bar, so no transfer was due; nothing here is gated",
     "labels": {"learned": "biped gait (frozen PPO) + navigation commands (NavGym v2 PPO actor, deterministic mean)", "scripted": "A* baseline only",
                "oracle": "pose and goal coordinate", "not_in_gym": "10 Hz lidar packets; team_sensors speed brake on the commands"},
     "settings": {"n": 6, "m": 6, "extra_openings": 1, "time_limit_s": 180, "heading": "+x", "sensor_mode": "reactive"},
     "hard_6x6": {"maze_seeds": "0-11 (the A* table's; INSIDE the gym's training-seed range < 10 000)",
                  "astar_published": {"label": "A* (scripted) 21412035", "success": astar["success"], "clean_success": astar["clean_success"],
                                      "n": astar["n"], "median_completion_s": astar["median_completion_s"], "completion_s": astar["completion_s"]},
                  "learned": {a: {**row(out / a / "summary.json", f"LEARNED {a}"), "onnx": onnx[a],
                                  "onnx_sha256": hashlib.sha256((repo / onnx[a]).read_bytes()).hexdigest()} for a in names}},
     "fresh_6x6": {"maze_seeds": "30000-30011 (never used by any gym training or evaluation)",
                   "astar": row(out / "fresh-6x6/astar/summary.json", "A* (scripted)"),
                   "learned": {a: row(out / "fresh-6x6" / a / "summary.json", f"LEARNED {a}") for a in names}}}
# the render gate declared in the launcher header
cands = [(a, t["hard_6x6"]["learned"][a]) for a in names if "missing" not in t["hard_6x6"]["learned"][a] and t["hard_6x6"]["learned"][a]["complete"]]
order = {a: i for i, a in enumerate(names)}
cands.sort(key=lambda x: (-x[1]["success"], -x[1]["clean_success"], x[1]["median_completion_s"] if x[1]["median_completion_s"] is not None else 1e9, order[x[0]]))
if cands:
    a, r = cands[0]
    t["render"] = {"rule": "most set-1 successes (ties: clean, then median time, then declared order); render iff >= 8/12; seed = median_seed_by_time",
                   "actor": a, "onnx": onnx[a], "success": r["success"], "eligible": r["success"] >= 8, "seed": r["median_seed_by_time"],
                   "table_completion_s": None}
    if t["render"]["eligible"]:
        ep = json.loads((out / a / f"seed{r['median_seed_by_time']}.json").read_text())
        t["render"]["table_completion_s"] = ep["completion_s"]
else:
    t["render"] = {"eligible": False, "reason": "no complete learned set-1 run"}
(out / "transfer_table.json").write_text(json.dumps(t, indent=2) + "\n")
t = json.loads((out / "transfer_table.json").read_text())
h = t["hard_6x6"]
print("NAVGYM TRANSFER (EXPLORATORY; oracle pose + goal) hard-6x6 seeds 0-11 [training-range mazes] | "
      + f"A* (scripted, 21412035): {h['astar_published']['success']}/12 (clean {h['astar_published']['clean_success']}, median {h['astar_published']['median_completion_s']} s) | "
      + " | ".join(line(r) for r in h["learned"].values()))
f = t["fresh_6x6"]
print("NAVGYM TRANSFER (EXPLORATORY) fresh-6x6 seeds 30000-30011 | " + line(f["astar"]) + " | " + " | ".join(line(r) for r in f["learned"].values()))
print("NAVGYM TRANSFER render gate: " + json.dumps(t["render"]))
PYEOF
echo "=== done: $(date) ==="
