#!/bin/bash
#SBATCH --job-name=maze-explore-gif
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:45:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# Three-panel clip of one randomized-maze episode (EGL on a GPU node): the seed
# and configuration come from the predeclared rule applied to the table job.
#   MAZE_CFG=hard|base  MAZE_SEED=<k>  (set by the submitter from summary.json)
set -euo pipefail
export MUJOCO_GL=egl
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-maze-gif-$SLURM_JOB_ID; mkdir -p "$CACHE"
CFG=${MAZE_CFG:?}; SEED=${MAZE_SEED:?}
case "$CFG" in
  hard) N=6; M=6; EXTRA=1; TL=180; OUT=$REPO/results/maze-explore-20260924/hard-6x6 ;;
  base) N=5; M=5; EXTRA=2; TL=150; OUT=$REPO/results/maze-explore-20260924/base-5x5 ;;
  *) echo "unknown MAZE_CFG $CFG"; exit 2 ;;
esac
GIF=$REPO/docs/gifs/random-maze-explore.gif
"$PY" -c "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'), 64, 64); r.render(); r.close(); print('EGL render preflight OK', mujoco.__version__)"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | cfg $CFG seed $SEED | $(date) ==="
"$PY" scripts/bench/maze_explore.py --upstream "$U" --cache-dir "$CACHE" --seeds 1 --seed-start "$SEED" --n "$N" --m "$M" --extra-openings "$EXTRA" \
    --time-limit "$TL" --out-dir "$OUT" --tag "_render" --render --frames-root "$CACHE/frames" --stride 2 \
    --gif "$GIF" --gif-speed "${MAZE_GIF_SPEED:-4.0}" --gif-fps 8 --gif-width 860 2>&1 | grep -v "^\["
rm -rf "$CACHE"
ls -la "$GIF" "${GIF%.gif}.json" "$OUT"/seed${SEED}_render.* 2>/dev/null
echo "=== done: $(date) ==="
