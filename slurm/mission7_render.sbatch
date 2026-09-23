#!/bin/bash
#SBATCH --job-name=m7-render
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=01:00:00
set -euo pipefail
repo=${1:?repo}; shift 1
cd "$repo"
PY="${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}"
export PYTHONPATH="$repo/src:$repo/scripts" MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
command -v ffmpeg >/dev/null || { echo "ffmpeg missing on $(hostname)"; exit 2; }
"$PY" -c "import mujoco; m=mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1 1 .1\" type=\"box\"/></worldbody></mujoco>'); mujoco.Renderer(m, 64, 64); print('EGL OK on', __import__('socket').gethostname())"
C=results/mission7-campaign-20260923/media
"$PY" scripts/render_mission7.py --repo "$repo" --out "$C/doors1-baseline-stall.mp4" --stage doors --index 1 --max-seconds 70 \
  --caption "Doors/1 baseline: guarded stage, then a 134 s stall (timeout at 180 s)" --evidence results/mission7-campaign-20260923/campaign-a-doors-00-07/result.json
"$PY" scripts/render_mission7.py --repo "$repo" --out "$C/doors1-crossing-fix-success.mp4" --stage doors --index 1 \
  --stage-activate --stage-wait-open 0 --pre-point 0.45 --cross-clear 0.35 --cross-kick --rejoin-advance --exit-ramp 1.2 --exit-ramp-center \
  --caption "Doors/1 with the crossing fix (kick, cross-until-clear, centre-aimed ramp): success at 77.7 s -- single layout, replay gate not passed" \
  --evidence results/mission7-campaign-20260923/local-stage-v2/doors-1-kick-center/result.json
"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
from make_gifs import pair_gif
C = Path("results/mission7-campaign-20260923/media")
pair_gif(C / "doors1-baseline-stall.mp4", C / "doors1-crossing-fix-success.mp4", Path("docs/gifs/mission7-doors1-stall-vs-crossing-fix.gif"),
         "baseline: stalls after the plate", "crossing fix: reaches the goal", outline_right=True)
print("gif written")
PYEOF
echo MISSION7_RENDER_COMPLETE
