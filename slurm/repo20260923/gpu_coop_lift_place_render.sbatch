#!/bin/bash
#SBATCH --job-name=coop-lift-place-render
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-27: render the cooperative LIFT, HOLD and PLACE, GATED on cpu_coop_lift_place.sbatch.
# Submit after the scoring job: sbatch --dependency=afterany:<cpu jobid> this-file.
#
#   label: learned gait (frozen) + scripted arms + oracle cube pose
#   note : cooperative lift, hold and place — scripted arms, frozen learned gait; carry not achieved
#   (hand pads added; grasping arm scripted at kp 30; zero velocity command; details in the sidecar)
#
# For each crew (2, 4) independently: if results/scripted-carry-20260927/
# score_liftplace_crew<N>.json re-summarises (scripted_carry.lift_place_verdict_line, from
# its episodes, never its own summary) to PASS under the predeclared LIFT_PLACE_RULE
# (>= 0.10 m lift, >= 0.05 m for >= 3.0 s continuously, final cube on its plinth within
# 0.03 m of rest height, < 0.05 m/s, centre on the plinth top, released, no floor contact,
# no tilt > 0.78 rad, 20 s; each pair >= 8/10 of seeds 10-19), re-simulate its
# median-by-hold successful seed (an episode's hold = its weakest pair's; lower median on
# an even count) with an EGL renderer and write NEW files docs/gifs/coop_lift_place_scripted_<N>.gif
# (+ .json sidecar: source hashes, label, note, learned/scripted/oracle, metrics, playback
# speeds; <= 5 MiB via panels.write_gif, which shrinks colours/fps/width until it fits).
# Nothing is overwritten: an existing GIF, sidecar, mp4 or render JSON makes the script refuse.
# A GIF over the budget is deleted (never left in docs/gifs) and the crew reports FAIL.
#
# TRUE PLAYBACK SPEED: stride 1 records all 25 policy frames per sim second and the mp4 is
# encoded at 25 fps (playback_speed_mp4 = 1.0); --gif-speed 1.0 keeps the GIF at real time
# (badge "GIF 1.00x", sidecar sim_seconds_per_gif_second = 1.0). The GIF is sampled at 8 fps.
# Size assumption: ~172 GIF frames (21.5 s incl. the 1.5 s end card) at 860 px; the two
# 860 px / 8 fps sidecars in docs/gifs are 1.0-1.2 MB for comparable scenes, so ~2-4 MB is
# expected; write_gif steps down (96/64/48 colours, 6/5 fps, 0.85/0.75 width) if not.
# If the verdict is not PASS, the script prints COOP_LIFT_PLACE_RENDER=SKIPPED and writes no clip.
set -euo pipefail
export MUJOCO_GL=egl
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-coop-lift-place-render-$SLURM_JOB_ID; mkdir -p "$CACHE"
SCORES=$REPO/results/scripted-carry-20260927
OUT=$SCORES/render_liftplace; mkdir -p "$OUT"
DEPLOY=$U/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml
[ -f "$DEPLOY" ] || { echo "COOP-LIFT-PLACE-RENDER RESULT: FAIL (deploy config missing: $DEPLOY)"; exit 1; }
"$PY" -c "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'), 64, 64); r.render(); r.close(); print('EGL render preflight OK', mujoco.__version__)"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date) ==="
for n in 2 4; do
    SCORE=$SCORES/score_liftplace_crew$n.json
    GIF=$REPO/docs/gifs/coop_lift_place_scripted_$n.gif
    if [ ! -f "$SCORE" ]; then
        echo "COOP-LIFT-PLACE-RENDER crew=$n: SKIPPED (no score JSON $SCORE)"; continue
    fi
    if [ -e "$GIF" ] || [ -e "${GIF%.gif}.json" ]; then
        echo "COOP-LIFT-PLACE-RENDER crew=$n: REFUSED ($GIF or its sidecar exists; never overwritten)"; continue
    fi
    if [ "$n" = 2 ]; then DIST=3.4; else DIST=4.6; fi
    "$PY" scripts/bench/coop_scripted_carry.py --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" \
        --crew "$n" --protocol lift_place --render-from "$SCORE" --out-dir "$OUT" --frames-root "$CACHE/frames" \
        --width 960 --height 540 --distance "$DIST" --azimuth 135 --elevation -38 --stride 1 \
        --gif "$GIF" --gif-speed 1.0 --gif-fps 8 --gif-width 860 2>&1 | grep -v "^\[" || true
    rm -rf "$CACHE/frames"
    # Verdict from the JSONs, never from exit codes.
    "$PY" - "$SCORE" "$OUT/coop_lift_place_scripted_$n.json" "$GIF" "$n" <<'PYEOF'
import json, sys
from pathlib import Path
from bhl_robust.eval import scripted_carry as sc
score, res, gif, crew = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
line = sc.lift_place_verdict_line(json.loads(score.read_text()), crew)
if not line.startswith(f"COOP-LIFT-PLACE crew {crew}: PASS"):
    print(f"COOP-LIFT-PLACE-RENDER crew={crew}: NOT RENDERED (score: {line.split(' | ')[0]})"); sys.exit(0)
if not res.is_file():
    print(f"COOP-LIFT-PLACE-RENDER crew={crew}: FAIL (no render JSON)"); sys.exit(0)
r = json.loads(res.read_text())
side = gif.with_suffix(".json")
g = r.get("gif") or {}
side_ok = False
if side.is_file():
    sd = json.loads(side.read_text())
    side_ok = (sd.get("note") == sc.LIFT_PLACE_NOTE and sd.get("label") == sc.LABEL
               and sd.get("sim_seconds_per_gif_second") == 1.0 and sd.get("playback_speed_mp4") == 1.0)
ok = bool(r.get("reproduced_success") and gif.is_file() and side.is_file() and side_ok
          and g.get("within_budget") and not r.get("gif_deleted_over_budget"))
if r.get("gif_deleted_over_budget"):
    why = " (GIF over the 5 MiB budget: deleted)"
elif not r.get("reproduced_success"):
    why = " (re-simulated episode did not succeed)"
elif not side_ok:
    why = " (sidecar missing, mislabelled or not at true playback speed)"
else:
    why = ""
print(f"COOP-LIFT-PLACE-RENDER crew={crew}: {'PASS' if ok else 'FAIL'}{why} seed={r.get('seed')} "
      f"reproduced={r.get('reproduced_success')} gif={gif if gif.is_file() else None} "
      f"mb={g.get('mb')} badge={g.get('badge')} | {sc.LIFT_PLACE_NOTE} | {sc.LABEL}")
PYEOF
done
rm -rf "$CACHE"
echo "=== done: $(date) ==="
