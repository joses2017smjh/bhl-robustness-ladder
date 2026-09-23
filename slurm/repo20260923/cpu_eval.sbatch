#!/bin/bash
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:00:00
# Repo-wide campaign (2026-09-23): CPU MuJoCo evaluations of existing exported
# checkpoints.  Usage: sbatch cpu_eval.sbatch <mode> [args]
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
CACHE=${TMPDIR:-/tmp}/bhl-cpu-eval-$SLURM_JOB_ID; mkdir -p "$CACHE"
"$PY" -c "import mujoco, onnxruntime; print('preflight OK', mujoco.__version__)"
mode=${1:?mode}; shift
case "$mode" in
  push)   # label pattern list; matched protocol: 0.5 m/s shoves every 3 s, 12 s episodes, 5 seeds x the harness's command set
    for pat in "$@"; do
      for d in "$U"/logs/rsl_rl/biped/*_"$pat"; do
        [ -f "$d/exported/deploy.yaml" ] || { echo "skip (not exported): $d"; continue; }
        label=$(basename "$d" | sed 's/^[0-9-]*_[0-9-]*_//')
        "$PY" -m bhl_robust.eval.run_eval --deploy-cfg "$d/exported/deploy.yaml" --upstream "$U" --cache-dir "$CACHE" \
          --out "results/push-matched-20260923/${label}__p0.5.csv" --label "$label" --episode-s 12 --seeds 5 --push-speed 0.5 --terrain-difficulty 0
      done
    done ;;
  lab)    # variant seed...
    variant=$1; shift
    if [ "$variant" = humanoid ]; then B=$U/logs/rsl_rl/humanoid; runs=(arms-dr1.0-s0 arms-dr0.0-s0 arms-push-s0 arms-terrain-s0)
    else B=$U/logs/rsl_rl/biped; runs=(dr-default-s0 dr-off-s0 push-adaptive-s0 terrain-bumpy-s0); fi
    deps=(); for r in "${runs[@]}"; do deps+=("$(ls -d "$B"/*_"$r" | head -1)/exported/deploy.yaml"); done
    for seed in "$@"; do
      echo "=== lab traverse $variant seed $seed ==="
      "$PY" scripts/bench/lab_traverse.py --deploy "${deps[@]}" --labels randomized no_randomization push_trained terrain_trained \
        --upstream "$U" --cache-dir "$CACHE" --variant "$variant" --seconds 24 --vx 0.40 --seed "$seed"
    done ;;
  coop)   # run dirs (relative to coop_lift logs)
    dirs=(); for r in "$@"; do dirs+=("$U/logs/rsl_rl/coop_lift/$r"); done
    "$PY" scripts/bench/coop_sim2sim.py --run-dir "${dirs[@]}" --upstream "$U" --cache-dir "$CACHE" --seeds 8 --crews 2 3 \
      --csv results/coop-occluded-20260923/coop_occluded_sim2sim.csv ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
echo REPO_CPU_EVAL_COMPLETE
