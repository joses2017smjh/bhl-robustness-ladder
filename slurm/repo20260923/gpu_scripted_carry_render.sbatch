#!/bin/bash
#SBATCH --job-name=scripted-carry-render
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-26: render the scripted-arm cube carry, GATED on cpu_scripted_carry.sbatch.
# Submit after the scoring job: sbatch --dependency=afterany:<cpu jobid> this-file.
#
#   label: learned gait (frozen) + scripted arms + oracle cube pose
#   (hand pads added; grasping arm scripted at kp 30; details in the sidecar)
#
# For each crew (2, 4) independently: if results/scripted-carry-20260926/
# score_crew<N>.json has verdict PASS under the predeclared rule (>= 0.10 m lift,
# >= 0.05 m for >= 3 s, >= 1.0 m carried while lifted, no tilt > 0.78 rad, no
# floor contact, 30 s; each pair >= 8/10 of seeds 0-9), re-simulate its median-
# by-completion-time seed among episodes where every pair succeeded with an EGL renderer and write NEW files
# docs/gifs/carry_scripted_<N>.gif (+ .json sidecar: source hashes, labels,
# success metrics, playback speed; <= 5 MiB via panels.write_gif). If the verdict
# is not PASS, the script prints SCRIPTED_CARRY_RENDER=SKIPPED and writes no
# clip: a NEGATIVE result gets no "success" GIF.
set -euo pipefail
export MUJOCO_GL=egl
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-scripted-carry-render-$SLURM_JOB_ID; mkdir -p "$CACHE"
SCORES=$REPO/results/scripted-carry-20260926
OUT=$SCORES/render; mkdir -p "$OUT"
DEPLOY=$U/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml
[ -f "$DEPLOY" ] || { echo "SCRIPTED-CARRY-RENDER RESULT: FAIL (deploy config missing: $DEPLOY)"; exit 1; }
"$PY" -c "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'), 64, 64); r.render(); r.close(); print('EGL render preflight OK', mujoco.__version__)"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date) ==="
for n in 2 4; do
    SCORE=$SCORES/score_crew$n.json
    GIF=$REPO/docs/gifs/carry_scripted_$n.gif
    if [ ! -f "$SCORE" ]; then
        echo "SCRIPTED-CARRY-RENDER crew=$n: SKIPPED (no score JSON $SCORE)"; continue
    fi
    if [ -e "$GIF" ]; then
        echo "SCRIPTED-CARRY-RENDER crew=$n: REFUSED ($GIF exists; never overwritten)"; continue
    fi
    if [ "$n" = 2 ]; then DIST=3.4; else DIST=4.6; fi
    "$PY" scripts/bench/coop_scripted_carry.py --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" \
        --crew "$n" --render-from "$SCORE" --out-dir "$OUT" --frames-root "$CACHE/frames" \
        --width 960 --height 540 --distance "$DIST" --azimuth 135 --elevation -38 --stride 2 \
        --gif "$GIF" --gif-speed 2.0 --gif-fps 8 --gif-width 860 2>&1 | grep -v "^\[" || true
    rm -rf "$CACHE/frames"
    # Verdict from the JSONs, never from exit codes.
    "$PY" - "$SCORE" "$OUT/carry_scripted_$n.json" "$GIF" "$n" <<'EOF'
import json, sys
from pathlib import Path
score, res, gif, crew = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3]), sys.argv[4]
s = json.loads(score.read_text())["summary"]
if s.get("verdict") != "PASS":
    print(f"SCRIPTED-CARRY-RENDER crew={crew}: NOT RENDERED (score verdict {s.get('verdict')})"); sys.exit(0)
if not res.is_file():
    print(f"SCRIPTED-CARRY-RENDER crew={crew}: FAIL (no render JSON)"); sys.exit(0)
r = json.loads(res.read_text())
side = gif.with_suffix(".json")
ok = bool(r.get("reproduced_success") and gif.is_file() and side.is_file()
          and r.get("gif", {}).get("within_budget"))
print(f"SCRIPTED-CARRY-RENDER crew={crew}: {'PASS' if ok else 'FAIL'} seed={r.get('seed')} "
      f"reproduced={r.get('reproduced_success')} gif={gif if gif.is_file() else None} "
      f"mb={r.get('gif', {}).get('mb') if r.get('gif') else None}")
EOF
done
rm -rf "$CACHE"
echo "=== done: $(date) ==="
