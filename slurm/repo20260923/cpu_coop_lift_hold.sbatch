#!/bin/bash
#SBATCH --job-name=coop-lift-hold
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-26: cooperative LIFT-AND-HOLD, a separate experiment predeclared in
# SLURM_JOBS.md before any scored seed of the carry run 21434982 existed. This
# job SCORES; it renders nothing (gpu_coop_lift_hold_render.sbatch does, on PASS).
#
#   label: learned gait (frozen) + scripted arms + oracle cube pose
#   note : cooperative lift and hold — carry not achieved
#   learned : 22-DoF locomotion gait arms-dr1.0-s0 (legs + each robot's outer arm), frozen
#   scripted: each robot's grasping arm (joint keyframes reach/squeeze/lift, then held)
#   oracle  : cube pose, used only to score; every velocity command is zero (no carry)
#   modelling choices: box hand pads = hand-mesh AABB (upstream hands are visual-only);
#             grasping-arm PD kp 30 (deploy 10), 4 Nm arm effort cap unchanged; no welds.
#   Same harness, layout, hand pads, arm script and label as cpu_scripted_carry.sbatch
#   (coop_scripted_carry.py --protocol lift_hold): the carry phase is removed, so after
#   the lift the pair stands and holds for the rest of a 20 s episode.
#
# PREDECLARED RULE (src/bhl_robust/eval/scripted_carry.py LIFT_HOLD_RULE; do not
# relax): an episode (20 s) succeeds iff the cube rises >= 0.10 m above its plinth
# rest height, stays >= 0.05 m up for >= 5.0 s continuously, no robot of the pair
# tilts > 0.78 rad at any policy step, and the cube never touches the floor (checked
# every physics step). Crew 2 PASS iff >= 8/10 of reset-jitter seeds 0-9 succeed;
# crew 4 (two pairs, two cubes, one world) PASS iff EACH pair has >= 8/10. Only a
# full seeds-0-9, full-20-s run is marked scored_run. The carry rule and its verdict
# (score_crew{2,4}.json) stand unchanged beside this one.
#
# Login-node pilot (seeds 100-102 only, never the scored seeds), crew 2: lift
# 18.3 / 19.9 / 20.5 cm, longest hold 8.16 / 5.84 / 10.40 s, max tilt <= 0.18 rad
# (no falls), but in all three the cube later slipped out and touched the floor
# (first failed check no_floor_contact) -- the floor clause is the expected failure.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-coop-lift-hold-$SLURM_JOB_ID; mkdir -p "$CACHE"
OUT=$REPO/results/scripted-carry-20260926; mkdir -p "$OUT"
DEPLOY=$U/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml
[ -f "$DEPLOY" ] || { echo "COOP-LIFT-HOLD RESULT: FAIL (deploy config missing: $DEPLOY)"; exit 1; }
for n in 2 4; do
    if [ -e "$OUT/score_lifthold_crew$n.json" ]; then
        echo "COOP-LIFT-HOLD RESULT: REFUSED ($OUT/score_lifthold_crew$n.json exists; published results are never overwritten)"
        exit 1
    fi
done
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date) ==="
for n in 2 4; do
    echo "--- crew $n, seeds 0-9, protocol lift_hold (20 s)"
    "$PY" scripts/bench/coop_scripted_carry.py --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" \
        --crew "$n" --protocol lift_hold --seeds 0-9 --out "$OUT/score_lifthold_crew$n.json" 2>&1 | grep -v "^\[" || true
done
rm -rf "$CACHE"
# Verdicts from the JSON (re-summarised under LIFT_HOLD_RULE), never from exit codes.
for n in 2 4; do
    "$PY" - "$OUT/score_lifthold_crew$n.json" "$n" <<'PYEOF'
import json, sys
from pathlib import Path
from bhl_robust.eval import scripted_carry as sc
path, crew = Path(sys.argv[1]), sys.argv[2]
if not path.is_file():
    print(f"COOP-LIFT-HOLD crew {crew}: INCOMPLETE (no JSON) | {sc.LIFT_HOLD_NOTE} | {sc.LABEL}"); sys.exit(0)
print(sc.lift_hold_verdict_line(json.loads(path.read_text()), crew))
PYEOF
done
echo "=== done: $(date) ==="
