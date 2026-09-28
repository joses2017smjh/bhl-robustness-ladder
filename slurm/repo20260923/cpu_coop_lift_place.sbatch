#!/bin/bash
#SBATCH --job-name=coop-lift-place
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:30:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-27: cooperative LIFT, HOLD and PLACE, a third, separate experiment of the
# scripted-arm harness (coop_scripted_carry.py --protocol lift_place). This job SCORES;
# it renders nothing (gpu_coop_lift_place_render.sbatch does, on PASS only).
#
#   label: learned gait (frozen) + scripted arms + oracle cube pose
#   note : cooperative lift, hold and place — scripted arms, frozen learned gait; carry not achieved
#   learned : 22-DoF locomotion gait arms-dr1.0-s0 (legs + each robot's outer arm), frozen
#   scripted: each robot's grasping arm: reach/squeeze/lift keyframes (unchanged), hold,
#             lower, open, rest (scripted_carry.PlaceScript / PlaceParams)
#   oracle  : cube pose, used only to score; every velocity command is zero (station keeping OFF)
#   modelling choices: box hand pads = hand-mesh AABB; grasping-arm PD kp 30 (deploy 10), 4 Nm
#             arm cap unchanged; cube 0.28 m / 0.5 kg and plinth unchanged; no welds. Lowering
#             9.2 -> 10.7 s (before the earliest lift_hold grip slip, 11.36 s, on exploration seeds
#             100-119); lowering target = the lift keyframe with its shoulder pitch reversed past
#             the squeeze pitch (-0.5 -> +0.6; elbow 1.0 and roll 1.1 held), then open (roll -> the
#             reach keyframe's -0.2) and hang (rest keyframe); the pair stands to 20 s.
#
# PREDECLARED RULE (src/bhl_robust/eval/scripted_carry.py LIFT_PLACE_RULE; do not relax):
# a 20 s episode succeeds iff the cube rises >= 0.10 m above its plinth rest height, stays
# >= 0.05 m up for >= 3.0 s continuously, and at the final step rests on its plinth: centre
# height within 0.03 m of rest, linear speed < 0.05 m/s, centre within the plinth top
# footprint (|dx|, |dy| <= 0.09 m from the plinth centre) and no robot geom touching it
# (released); the cube never touches the floor (every physics step) and no robot of the pair
# tilts > 0.78 rad (every policy step). Crew 2 PASS iff >= 8/10 of reset-jitter seeds 10-19
# succeed; crew 4 (two pairs, two cubes, one world) PASS iff EACH pair has >= 8/10 of seeds
# 10-19. Seeds 0-9 (scored for the carry and lift_hold rules) are never run: the script
# refuses them. Only a full seeds-10-19, full-20-s run is marked scored_run.
#
# Disclosed pilot (exploration seeds >= 100 only; no seed 0-19 of this protocol was ever run):
# crew 2 seeds 100-109: 1/10 succeed (2/10 end placed on the plinth, one of them with a hand
# still on the cube); crew 4 seeds 100-104: 0/5 on each pair (pair 1 placed one cube but a
# hand stayed on it; robot falls 1/5 and 4/5). Dominant failure:
# the cube rolls ~90 deg in the hands during the lift, so a hand ends on its top face; opening
# that hand drags the cube off the plinth (floor contact during release/retract) and in 5/10
# crew-2 pilot episodes the dragging robot falls. EXPECTED NEGATIVE.
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-coop-lift-place-$SLURM_JOB_ID; mkdir -p "$CACHE"
OUT=$REPO/results/scripted-carry-20260927; mkdir -p "$OUT"
DEPLOY=$U/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml
[ -f "$DEPLOY" ] || { echo "COOP-LIFT-PLACE RESULT: FAIL (deploy config missing: $DEPLOY)"; exit 1; }
for n in 2 4; do
    if [ -e "$OUT/score_liftplace_crew$n.json" ]; then
        echo "COOP-LIFT-PLACE RESULT: REFUSED ($OUT/score_liftplace_crew$n.json exists; published results are never overwritten)"
        exit 1
    fi
done
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date) ==="
for n in 2 4; do
    echo "--- crew $n, seeds 10-19, protocol lift_place (20 s)"
    "$PY" scripts/bench/coop_scripted_carry.py --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" \
        --crew "$n" --protocol lift_place --seeds 10-19 --out "$OUT/score_liftplace_crew$n.json" 2>&1 | grep -v "^\[" || true
done
rm -rf "$CACHE"
# Verdicts from the JSON (re-summarised under LIFT_PLACE_RULE), never from exit codes. A JSON
# is only complete when every seed 10-19 ran the full 20 s (steps == 500 per episode) or
# stopped on a recorded non-finite state (a scored failure, `failed` set).
for n in 2 4; do
    "$PY" - "$OUT/score_liftplace_crew$n.json" "$n" <<'PYEOF'
import json, sys
from pathlib import Path
from bhl_robust.eval import scripted_carry as sc
path, crew = Path(sys.argv[1]), sys.argv[2]
if not path.is_file():
    print(f"COOP-LIFT-PLACE crew {crew}: INCOMPLETE (no JSON) | {sc.LIFT_PLACE_NOTE} | {sc.LABEL}"); sys.exit(0)
d = json.loads(path.read_text())
line = sc.lift_place_verdict_line(d, crew)
n_full = int(round(float(d["rule"]["episode_s"]) / float(d["policy_dt"])))     # 500 = 20 s / 0.04 s
short = [e["seed"] for e in d.get("episodes", []) if e.get("steps") != n_full and e.get("failed") is None]
if short:
    line = line.replace(": PASS |", ": INCOMPLETE |").replace(": NEGATIVE |", ": INCOMPLETE |")
    line += f" | episodes that did not reach 20 s: seeds {short}"
print(line)
PYEOF
done
echo "=== done: $(date) ==="
