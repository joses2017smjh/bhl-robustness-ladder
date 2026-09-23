#!/bin/bash
#SBATCH --job-name=m7-render-approach
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=12G
#SBATCH --time=00:30:00
set -euo pipefail
repo=${1:?repo}; shift 1
cd "$repo"
PY="${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}"
export PYTHONPATH="$repo/src:$repo/scripts" MUJOCO_GL=egl OMP_NUM_THREADS=1 PYTHONUNBUFFERED=1
C=results/mission7-campaign-20260923/media
EV=results/mission7-campaign-20260923/approach-matrix-center-bias/result.json
"$PY" scripts/render_mission7_approach.py --repo "$repo" --out "$C/approach-negx14-baseline.mp4" --controller recovery --index 14 --replay-order=-1,+0 --caption "Approach, world -x, layout 14: 0.50 m/s recovery translation -- hits goal_post_-1 on the first stride" --evidence results/mission7-campaign-20260923/approach-negx-recovery/result.json
"$PY" scripts/render_mission7_approach.py --repo "$repo" --out "$C/approach-negx14-centered.mp4" --controller center_bias --index 14 --replay-order=-1,+0 --caption "Approach, world -x, layout 14: gap-midline centring with +0.015 m pre-bias -- through the posts to the goal (gate 62/64)" --evidence "$EV"
"$PY" -c "import json,sys; b=json.load(open('$C/approach-negx14-baseline.json'))['outcome_of_this_run']; c=json.load(open('$C/approach-negx14-centered.json'))['outcome_of_this_run']; assert not b['success'] and c['success'], (b, c); print('outcomes reproduce the evaluation:', b, c)"
"$PY" - <<'PYEOF'
import sys; sys.path.insert(0, "scripts")
from pathlib import Path
from make_gifs import pair_gif
C = Path("results/mission7-campaign-20260923/media")
pair_gif(C / "approach-negx14-baseline.mp4", C / "approach-negx14-centered.mp4", Path("docs/gifs/mission7-approach-negx-centering.gif"), "baseline: post collision", "centred: through the gap", max_s=12.0, outline_right=True)
print("gif written")
PYEOF
echo MISSION7_RENDER_COMPLETE
