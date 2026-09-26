#!/bin/bash
#SBATCH --job-name=scripted-carry
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-26, user request: honestly labelled MuJoCo clips of 2 and 4 BHL
# humanoids lifting a 0.28 m / 0.5 kg cube together and carrying it, to replace
# the failed learned-policy carry clips. This job SCORES; it renders nothing.
#
#   label: learned gait (frozen) + scripted arms + oracle cube pose
#   learned : 22-DoF locomotion gait arms-dr1.0-s0 (legs + each robot's outer arm), frozen
#   scripted: each robot's grasping arm (joint keyframes reach/squeeze/lift/hold)
#   oracle  : cube pose (carry gate + stop) and robot base poses (pair sync)
#   modelling choices: box hand pads = hand-mesh AABB (upstream hands are visual-only);
#             grasping-arm PD kp 30 (deploy 10), 4 Nm arm effort cap unchanged; no welds.
#
# PREDECLARED RULE (fixed before any scored seed ran; src/bhl_robust/eval/scripted_carry.py
# SUCCESS_RULE; do not relax): an episode (30 s) succeeds iff the cube rises
# >= 0.10 m above its plinth rest height, stays >= 0.05 m up for >= 3.0 s
# contiguous, is carried >= 1.0 m horizontally from rest while >= 0.05 m up, no
# robot of the pair tilts > 0.78 rad at any policy step, and the cube never
# touches the floor (checked every physics step). Crew of 2 PASS iff >= 8/10 of
# reset-jitter seeds 0-9 succeed; crew of 4 (two pairs, two cubes, one world) is
# reported and gated the same way per pair: crew-4 PASS iff EACH pair
# independently has >= 8/10 successes. The render job (gpu_scripted_carry_render
# .sbatch) renders, independently per crew, the median-by-completion-time seed
# among episodes in which every pair succeeded, ONLY for a crew whose verdict
# here is PASS (crew 2 can be rendered while crew 4 is not, and vice versa).
# Only a full seeds-0-9, full-30-s run is marked scored_run; a partial re-run
# (e.g. one missing seed) is a pipeline check by construction -- re-score all 10.
#
# Login-node exploration used seeds >= 100 only. Frozen-config numbers there:
# seed 100 lifted 21 cm and held 7.6 s, seed 101 16.8 cm / 2.8 s; the carry
# stalls (0.24-0.28 m) and the cube slips -- the expected verdict is NEGATIVE.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-scripted-carry-$SLURM_JOB_ID; mkdir -p "$CACHE"
OUT=$REPO/results/scripted-carry-20260926; mkdir -p "$OUT"
DEPLOY=$U/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml
[ -f "$DEPLOY" ] || { echo "SCRIPTED-CARRY RESULT: FAIL (deploy config missing: $DEPLOY)"; exit 1; }
for n in 2 4; do
    if [ -e "$OUT/score_crew$n.json" ]; then
        echo "SCRIPTED-CARRY RESULT: REFUSED ($OUT/score_crew$n.json exists; published results are never overwritten)"
        exit 1
    fi
done
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date) ==="
for n in 2 4; do
    echo "--- crew $n, seeds 0-9"
    "$PY" scripts/bench/coop_scripted_carry.py --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" \
        --crew "$n" --seeds 0-9 --out "$OUT/score_crew$n.json" 2>&1 | grep -v "^\[" || true
done
rm -rf "$CACHE"
# Verdicts from the JSON, never from exit codes.
for n in 2 4; do
    "$PY" - "$OUT/score_crew$n.json" "$n" <<'EOF'
import json, sys
from pathlib import Path
path, crew = Path(sys.argv[1]), sys.argv[2]
if not path.is_file():
    print(f"SCRIPTED-CARRY crew={crew} VERDICT: INCOMPLETE (no JSON)"); sys.exit(0)
d = json.loads(path.read_text())
s = d["summary"]
pp = " ".join(f"pair{p['pair']}={p['successes']}/{p['episodes']} (first failed: {p['first_failed_check_counts']}, "
              f"median lift {p['lift_peak_m_median']:.3f} m, median hold {p['lift_hold_s_median']:.2f} s, "
              f"median carry {p['carry_m_median']:.3f} m, max carry {p['carry_m_max']:.3f} m)"
              for p in s["per_pair"] if p["episodes"])
ok = d.get("scored_run") and s["complete"]
verdict = s["verdict"] if ok else "INCOMPLETE"
print(f"SCRIPTED-CARRY crew={crew} VERDICT: {verdict} | {pp} | median_seed={s['median_seed']} | {d['label']}")
EOF
done
echo "=== done: $(date) ==="
