#!/bin/bash
#SBATCH --job-name=waiter-expert-gate
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
# Waiter phase 2 scored expert gate (rule predeclared in SLURM_JOBS.md): scripts/waiter/expert_gate.py over the gate
# seeds in GATE_PROCS parallel processes (no rendering; a GPU node only because it starts sooner), then the verdict
# (gate.json, never overwritten). GATE_FREE=1 runs the reported free-base baseline (stance_lock=False) instead.
#   sbatch --export=ALL,GATE_OUT=<dir> slurm/repo20260923/waiter_expert_gate.sbatch
set -uo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
DEPLOY=${GATE_DEPLOY:-$REPO/results/waiter-20261005/wbc1c/training/waiter-wbc1c-s7.deploy.yaml}
OUT=${GATE_OUT:?set GATE_OUT}
SEED0=${GATE_SEED0:-2000}; SEED1=${GATE_SEED1:-2099}; PROCS=${GATE_PROCS:-8}; MIN=${GATE_MIN_SUCCESS:-80}
FREE=(); [ "${GATE_FREE:-0}" = 1 ] && FREE=(--free-base)
[ "${GATE_START_ONLY:-0}" = 1 ] && FREE=(--lock-start-only)       # the lock-removal test (base held for the settle only)
RANGE=(); [ -n "${GATE_RANGE:-}" ] && RANGE=(--range "$GATE_RANGE")
export TMPDIR=/tmp PYTHONPATH="$REPO/src:$REPO/scripts/bench:$REPO/scripts/waiter" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
cd "$REPO"
echo "=== waiter expert gate | $(hostname) | HEAD $(git rev-parse --short HEAD) | dirty: $(git status --porcelain -- \
  src/bhl_robust/eval/waiter_env.py src/bhl_robust/eval/waiter_scene.py scripts/waiter/expert.py scripts/waiter/expert_gate.py \
  | wc -l) | $(date)"
echo "deploy $DEPLOY | out $OUT | seeds $SEED0-$SEED1 | ${FREE[*]:-stance lock}"
mkdir -p "$OUT"
for i in $(seq 0 $((PROCS - 1))); do
  S=$(seq "$SEED0" "$SEED1" | awk -v i="$i" -v n="$PROCS" '(NR - 1) % n == i' | paste -sd, -)
  "$PY" scripts/waiter/expert_gate.py --deploy "$DEPLOY" --upstream "$U" --cache-dir "/tmp/waiter-gate-${SLURM_JOB_ID:-$$}-$i" \
    --out "$OUT" --seeds "$S" "${FREE[@]}" "${RANGE[@]}" > "$OUT/worker_$i.log" 2>&1 &
done
wait
grep -h "^GATE: seed" "$OUT"/worker_*.log | wc -l | xargs echo "episodes this job:"
"$PY" scripts/waiter/expert_gate.py --out "$OUT" --seeds "$SEED0-$SEED1" --summarize --min-success "$MIN" "${RANGE[@]}"
