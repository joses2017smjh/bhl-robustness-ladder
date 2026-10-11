#!/bin/bash
#SBATCH --job-name=waiter-demos
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=03:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# Waiter phase 2 demonstrations (SLURM_JOBS.md 2026-10-09): scripts/waiter/record_demos.py over DEMO_SEED0..DEMO_SEED1
# (range 3000-3999) in DEMO_PROCS processes sharing DEMO_OUT (an existing episode is skipped, so a rerun resumes; the
# manifest is rebuilt from every episode JSON by each process at its end). EGL renders the simulated head camera.
# dgxh-1 is excluded: its MIG GPU slices have no EGL and MuJoCo would render in software (different images).
#   sbatch --export=ALL,DEMO_OUT=<dir> slurm/repo20260923/waiter_record_demos.sbatch
set -uo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
DEPLOY=${DEMO_DEPLOY:-$REPO/results/waiter-20261005/wbc1c/training/waiter-wbc1c-s7.deploy.yaml}
OUT=${DEMO_OUT:?set DEMO_OUT}
SEED0=${DEMO_SEED0:-3000}; SEED1=${DEMO_SEED1:-3299}; PROCS=${DEMO_PROCS:-8}
export TMPDIR=/tmp MUJOCO_GL=egl PYOPENGL_PLATFORM=egl MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0} \
  PYTHONPATH="$REPO/src:$REPO/scripts/bench:$REPO/scripts/waiter" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$REPO"
echo "=== waiter demos | $(hostname) | HEAD $(git rev-parse --short HEAD) | seeds $SEED0-$SEED1 | out $OUT | $(date)"
mkdir -p "$OUT"
# stop here if this node does not render the head camera like the first demonstrations (render_check.py)
"$PY" scripts/waiter/render_check.py || { echo "WAITER DEMOS: REFUSED (this node renders differently)"; exit 1; }
for i in $(seq 0 $((PROCS - 1))); do
  S=$(seq "$SEED0" "$SEED1" | awk -v i="$i" -v n="$PROCS" '(NR - 1) % n == i' | paste -sd, -)
  "$PY" scripts/waiter/record_demos.py --deploy "$DEPLOY" --upstream "$U" --cache-dir "/tmp/waiter-demos-${SLURM_JOB_ID:-$$}-$i" \
    --out "$OUT" --seeds "$S" --range demo > "$OUT/worker_$i.log" 2>&1 &
done
wait
"$PY" - "$OUT" <<'PYEOF'
import json, sys
from pathlib import Path
m = json.loads((Path(sys.argv[1]) / "manifest.json").read_text())
size = sum(p.stat().st_size for p in Path(sys.argv[1]).glob("ep_*.npz"))
print(f"WAITER DEMOS: {m['n_success']}/{m['n']} successful episodes, {size / 1e9:.2f} GB of npz -> {sys.argv[1]}")
PYEOF
