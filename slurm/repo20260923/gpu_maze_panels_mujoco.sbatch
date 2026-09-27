#!/bin/bash
#SBATCH --job-name=maze-panels-mj
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# 2026-09-24, user request: a maze-completion GIF with a top view, the robot's
# own stereo (ray depth) window and a lidar window. MuJoCo inspection maze,
# the README's episode (seed 0, ordered route, reactive sensor brake) re-run
# through the same harness with a per-step recorder; EGL render on a GPU node.
# The login-node pipeline check reproduced the README recording's 17.36 s.
set -euo pipefail
export MUJOCO_GL=egl
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-maze-panels-$SLURM_JOB_ID; mkdir -p "$CACHE"
OUT=$REPO/results/repo-gpu-20260923/maze-panels; mkdir -p "$OUT"
DEPLOY=$U/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml
[ -f "$DEPLOY" ] || { echo "MAZE-PANELS RESULT: FAIL (deploy config missing: $DEPLOY)"; exit 1; }
"$PY" -c "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'), 64, 64); r.render(); r.close(); print('EGL render preflight OK', mujoco.__version__)"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date) ==="
"$PY" scripts/bench/maze_panels.py --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" \
    --seed 0 --route ordered --sensor-mode reactive --seconds 60 --speed 0.4 \
    --out-dir "$OUT" --name inspection-maze-panels --frames-root "$CACHE/frames" \
    --width 960 --height 540 --elevation -66 --azimuth 90 --distance 6.4 --lookat 2.4:0.2:0.0 \
    --gif "$REPO/docs/gifs/inspection-maze-panels.gif" --gif-speed 2.0 --gif-fps 8 --gif-width 860 2>&1 | grep -v "^\[" 
rm -rf "$CACHE/frames"
ls -la "$OUT"/inspection-maze-panels.* "$REPO"/docs/gifs/inspection-maze-panels.* 2>/dev/null
echo "=== done: $(date) ==="
