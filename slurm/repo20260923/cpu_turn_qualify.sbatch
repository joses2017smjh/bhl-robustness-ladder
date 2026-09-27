#!/bin/bash
#SBATCH --job-name=turn-qualify
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# 2026-09-27: single-checkpoint qualification of a turning gait (CPU MuJoCo,
# LEARNED policy replayed; nothing scripted). Default checkpoint
# arms-turn-turnboth-s0 -- the only one of 12 turning-arm seeds that turns
# under turn_test v2 (5/6) -- as a candidate gait for Mission 7, the cooperative
# carry and a README clip. It is ONE checkpoint (1 of 12 seeds), NOT a reliable
# recipe, and every output says so.
#   sbatch slurm/repo20260923/cpu_turn_qualify.sbatch
#   TURNQ_RUN=<run name> sbatch --export=ALL,TURNQ_RUN=<run name> ...   (another exported 22-DoF run)
#
# (i)  turn_test.py --protocol v2x: reset seeds 10 11 12 13 14 x (0,0,+0.6) and
#      (0,0,-0.6) rad/s for 6 s after a 3.0 s zero-command settle (10 turn runs),
#      plus the walk run (0.35,0,0), 1.0 s warm-up, 6 s, on reset seeds 10 11 12.
# (ii) matched-push regression, the protocol of cpu_eval.sbatch push /
#      cpu_push_rescore.sbatch (run_eval, 0.5 m/s shoves every 3 s, 12 s episodes,
#      terrain 0, --variant humanoid) with --seeds 10: the harness's six commands
#      x eval seeds 0-9 = n = 60 episodes. Reference: arms-dr1.0-s0 fell 3/30 =
#      0.10 under this protocol (arms-dr1.0 s0+s1: 7/60 = 0.117).
#
# PREDECLARED (2026-09-27, before any of these runs exist):
#   QUALIFIED iff ALL of
#     - >= 9 of the 10 turn runs reach >= 150 deg IN THE COMMANDED DIRECTION with no fall;
#     - |yaw drift| <= 15 deg with no fall on >= 2 of the 3 walk seeds;
#     - push fall rate <= 0.15, i.e. <= 9 falls in exactly 60 episodes.
#   NOT QUALIFIED if any clause fails; INCOMPLETE if the v2x JSON is missing /
#   not protocol v2x with the declared seeds and bars, or the CSV does not hold
#   exactly 60 rows. Read from the JSON and CSV only, never from exit codes.
# Data overlap, stated openly: TurnBoth-s0 has been run on reset seeds 0-2 (turn
# test v1/v2 and turn_diagnose.py mujoco, 2026-09-24..27) and walk seed 0. The
# scored seeds here (10-14 turn, 10-12 walk) are disjoint from those and have
# never been run on it; it has never been push-evaluated. runner.reset(rng)
# perturbs the 22 joint positions by N(0, 0.02 rad), so reset seeds do differ.
#
# Outputs (new directory; nothing published is overwritten):
#   results/repo-gpu-20260923/turn-20260927/qualify/<run>__v2x.json
#   results/repo-gpu-20260923/turn-20260927/qualify/<run>__p0.5_n60.csv
#   results/repo-gpu-20260923/turn-20260927/qualify/<run>__qualify.json   (the verdict)
# Resubmit-safe: a complete v2x JSON / 60-row CSV is reused, not recomputed; an
# existing verdict JSON is never overwritten (the job re-prints it and exits).
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
JOBTAG=${SLURM_JOB_ID:-local$$}
CACHE=${TMPDIR:-/tmp}/bhl-turnq-$JOBTAG; mkdir -p "$CACHE"
RUN=${TURNQ_RUN:-arms-turn-turnboth-s0}
OUT=${TURNQ_OUT_DIR:-$REPO/results/repo-gpu-20260923/turn-20260927/qualify}
mkdir -p "$OUT"
echo "=== turn qualify $RUN | $(hostname) | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
"$PY" -c "import mujoco, onnxruntime; print('preflight OK', mujoco.__version__)"

VERDICT=$OUT/${RUN}__qualify.json
if [ -f "$VERDICT" ]; then
    echo "verdict already recorded, not overwritten: $VERDICT"
    "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); print('TURN-QUALIFY RESULT:', d['verdict'], '--', d['detail'])" "$VERDICT"
    exit 0
fi

mapfile -t dirs < <(ls -d "$U"/logs/rsl_rl/humanoid/*_"$RUN" 2>/dev/null | sort)
if [ "${#dirs[@]}" -ne 1 ]; then echo "TURN-QUALIFY RESULT: INCOMPLETE -- expected exactly one run dir for $RUN, found ${#dirs[@]}"; exit 1; fi
DEPLOY=${dirs[0]}/exported/deploy.yaml
[ -f "$DEPLOY" ] || { echo "TURN-QUALIFY RESULT: INCOMPLETE -- no exported deploy.yaml: $DEPLOY"; exit 1; }
echo "deploy: $DEPLOY"

# (i) v2x turn test -> fresh temp JSON, moved into place only when written
V2X=$OUT/${RUN}__v2x.json
if [ -f "$V2X" ] && "$PY" -c "import json,sys; d=json.load(open(sys.argv[1])); assert d['protocol']=='v2x' and len(d['turns'])==10 and len(d['walks'])==3" "$V2X" 2>/dev/null; then
    echo "have $V2X"
else
    tmp=$OUT/.${RUN}__v2x.$JOBTAG.json; rm -f "$tmp"
    "$PY" scripts/bench/turn_test.py --protocol v2x --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" \
        --variant humanoid --turn-seeds 10 11 12 13 14 --walk-seeds 10 11 12 --min-turn-ok 9 --min-walk-ok 2 \
        --turn-min-deg 150 --drift-max-deg 15 --seconds 6 --out "$tmp" 2>&1 | grep -v "^\[" || true
    [ -f "$tmp" ] && mv -f "$tmp" "$V2X"
fi

# (ii) matched push regression, n = 60
CSV=$OUT/${RUN}__p0.5_n60.csv
if [ -f "$CSV" ] && [ "$(tail -n +2 "$CSV" | wc -l)" -eq 60 ]; then
    echo "have $CSV"
else
    tmpc=$OUT/.${RUN}__p0.5_n60.$JOBTAG.csv; rm -f "$tmpc"
    echo "=== push regression $RUN $(date) ==="
    "$PY" -m bhl_robust.eval.run_eval --deploy-cfg "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" \
        --out "$tmpc" --label "$RUN" --variant humanoid \
        --episode-s 12 --seeds 10 --push-speed 0.5 --terrain-difficulty 0 || echo "EVAL FAILED: $RUN"
    [ -f "$tmpc" ] && mv -f "$tmpc" "$CSV"
fi

echo "=== verdict $(date) ==="
"$PY" - "$V2X" "$CSV" "$VERDICT" "$RUN" "$DEPLOY" <<'EOF'
import csv, json, sys
from pathlib import Path
v2x, csvp, verdict_path, run, deploy = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4], sys.argv[5]
RULE = {"turn_min_deg": 150.0, "drift_max_deg": 15.0, "turn_seeds": [10, 11, 12, 13, 14], "walk_seeds": [10, 11, 12],
        "min_turn_ok": 9, "min_walk_ok": 2, "turn_wz": 0.6, "turn_warm_s": 3.0, "seconds": 6.0}
MAX_FALLS, N_PUSH = 9, 60
problems, turn = [], None
try:
    d = json.loads(v2x.read_text())
    if d.get("protocol") != "v2x":
        problems.append("v2x JSON is not protocol v2x")
    bad = {k: (d["rule"].get(k), v) for k, v in RULE.items() if d["rule"].get(k) != v}
    if bad:
        problems.append(f"v2x JSON rule differs from the declared one: {bad}")
    if len(d["turns"]) != 10 or len(d["walks"]) != 3:
        problems.append(f"v2x JSON has {len(d['turns'])} turn / {len(d['walks'])} walk runs (need 10 / 3)")
    turn = d
except Exception as exc:  # noqa: BLE001
    problems.append(f"no readable v2x JSON ({exc!r})")
push = None
try:
    rows = list(csv.DictReader(csvp.open()))
    falls = sum(r["fell"].strip().lower() == "true" for r in rows)
    push = {"falls": falls, "n": len(rows), "rate": round(falls / len(rows), 4) if rows else None}
    if len(rows) != N_PUSH:
        problems.append(f"push CSV has {len(rows)} rows (need {N_PUSH})")
except Exception as exc:  # noqa: BLE001
    problems.append(f"no readable push CSV ({exc!r})")
clauses = {}
if turn is not None and not problems:
    n_t = sum(bool(r["fell_at_s"] is None and r["yaw_deg"] * (1 if r["cmd"][2] > 0 else -1) >= 150.0) for r in turn["turns"])
    n_w = sum(bool(w["fell_at_s"] is None and abs(w["yaw_deg"]) <= 15.0) for w in turn["walks"])
    clauses = {"turn": {"ok_runs": n_t, "of": 10, "need": 9, "pass": n_t >= 9,
                        "yaw_deg": [r["yaw_deg"] for r in turn["turns"]]},
               "walk": {"ok_runs": n_w, "of": 3, "need": 2, "pass": n_w >= 2,
                        "drift_deg": [w["yaw_deg"] for w in turn["walks"]]},
               "push": {**push, "max_falls": MAX_FALLS, "pass": push["falls"] <= MAX_FALLS}}
if problems:
    verdict, detail = "INCOMPLETE", "; ".join(problems)
else:
    verdict = "QUALIFIED" if all(c["pass"] for c in clauses.values()) else "NOT QUALIFIED"
    detail = (f"turn {clauses['turn']['ok_runs']}/10 (need 9), walk {clauses['walk']['ok_runs']}/3 (need 2), "
              f"push {push['falls']}/{push['n']} = {push['rate']} (need <= 0.15)")
detail += (f"; {run} is a single checkpoint" + (" (1 of 12 turning-arm seeds)" if run == "arms-turn-turnboth-s0" else "")
           + ", not a reliable recipe")
out = {"run": run, "deploy": deploy, "verdict": verdict, "detail": detail, "clauses": clauses,
       "label": "LEARNED policy, MuJoCo replay (CPU); single checkpoint, not a reliable recipe",
       "predeclared": "QUALIFIED iff >= 9/10 turn runs (reset seeds 10-14 x +/-0.6 rad/s, 3 s settle, 6 s) reach >= 150 deg in "
                      "the commanded direction with no fall, |drift| <= 15 deg with no fall on >= 2/3 walk seeds (10-12), and "
                      "push fall rate <= 0.15 (<= 9/60; run_eval 0.5 m/s every 3 s, 12 s, --seeds 10, --variant humanoid)",
       "reference": "arms-dr1.0-s0 push 3/30 = 0.10; arms-dr1.0 s0+s1 7/60 = 0.117 (results/push-matched-20260923)",
       "overlap": "scored seeds (turn 10-14, walk 10-12, push eval seeds 0-9) disjoint from every run previously inspected on "
                  "this checkpoint (reset seeds 0-2; never push-evaluated)",
       "inputs": {"v2x_json": str(v2x), "push_csv": str(csvp)}}
if verdict != "INCOMPLETE":
    verdict_path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {verdict_path}")
print(f"TURN-QUALIFY RESULT: {verdict} -- {detail}")
EOF
echo "=== done: $(date) ==="
