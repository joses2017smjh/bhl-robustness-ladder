#!/bin/bash
# SF-04 predeclared evaluation: BothRobust Full s0 (recurrent) vs the published
# Both / Lidar / Stereo Full s0 (MLP), same settings matrix, one boot per arm.
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
OUT="$REPO/results/repo-gpu-20260923/sf04-eval"; mkdir -p "$OUT"
PUB="$REPO/results/weekend-20260919"
run_arm() { arm=$1; ckpt=$2; policy=$3
  json="$OUT/${arm}-s0.json"; log="$OUT/${arm}-s0.log"
  echo "=== $arm | $ckpt | policy $policy | $(date) ==="; set +e
  BHL_POLICY="$policy" timeout 2400 "$PY" "$REPO/scripts/bench/maze_recovery_probe.py" --headless --enable_cameras \
      --task "Velocity-BHL-MazeRecovery-Full-${arm}-v0" --num-envs 32 --steps 600 --seed 100 \
      --checkpoint "$ckpt" --keep-corruption --output "$json" --settings '[{"name":"baseline"},{"name":"lidar_off","zero_terms":["lidar"]},{"name":"stereo_off","zero_terms":["stereo_l","stereo_r"]},{"name":"both_off","zero_terms":["lidar","stereo_l","stereo_r"]},{"name":"delay1","imu_delay_steps":1},{"name":"delay2","imu_delay_steps":2},{"name":"gyro0.10","gyro_std":0.10},{"name":"grav0.10","gravity_std":0.10}]' > "$log" 2>&1
  rc=$?; set -e; grep -E '^\{"name"|ERROR' "$log" | cut -c1-200 || true; echo "=== $arm exit $rc ==="; }
ck_robust=$("$PY" "$REPO/scripts/bench/maze_checkpoint.py" resolve --log-root "$UPSTREAM/logs/rsl_rl/biped" --run-name wknd-full-bothrobust-s0 --newer-than 0)
run_arm BothRobust "$ck_robust" recurrent
for arm in Both Lidar Stereo; do
  ck=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint"])' "$PUB/maze-eval-Full-$arm-s0.json")
  run_arm "$arm" "$ck" ""
done
"$PY" - <<'PY2'
import json, os, glob
out = os.environ["REPO"] + "/results/repo-gpu-20260923/sf04-eval"
rows = {}
for f in sorted(glob.glob(out + "/*-s0.json")):
    d = json.load(open(f)); rows[os.path.basename(f)[:-8]] = {s["name"]: (round(s["first_episode_success_rate"] * 32) if "error" not in s else None) for s in d["settings"]}
names = ["baseline", "lidar_off", "stereo_off", "both_off", "delay1", "delay2", "gyro0.10", "grav0.10"]
print("%-11s" % "arm" + "".join("%11s" % n for n in names))
for a, r in rows.items(): print("%-11s" % a + "".join("%11s" % (("%d/32" % r[n]) if r.get(n) is not None else "ERR") for n in names))
json.dump(rows, open(out + "/summary.json", "w"), indent=2)
br, bo = rows.get("BothRobust", {}), rows.get("Both", {})
ok = all(br.get(n) is not None for n in names) and br.get("baseline", 0) >= bo.get("baseline", 99) and br.get("stereo_off", 0) >= rows.get("Lidar", {}).get("baseline", 99) and br.get("lidar_off", 0) >= rows.get("Stereo", {}).get("baseline", 99)
print("SF04-EVAL RESULT:", "PASS (predeclared gate met)" if ok else "FAIL or NEGATIVE (see table)")
PY2
