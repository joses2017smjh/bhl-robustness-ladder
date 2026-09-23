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
  sf03)   # attitude filter in the loop (docs/SENSOR_FUSION.md SF-03): predeclared levels, seeds 0-2, ordered route, reactive sensors
    D=$(ls -d "$U"/logs/rsl_rl/humanoid/*_arms-dr1.0-s0 | head -1)
    O=results/sensor-fusion-20260923
    common=(--deploy "$D/exported/deploy.yaml" --upstream "$U" --cache-dir "$CACHE" --seeds 3 --seconds 60 --sensor-modes reactive --routes ordered)
    "$PY" scripts/bench/inspection_maze.py "${common[@]}" --output "$O/sf03-L0-truth.json" --imu-source truth
    # tag filter gyro_std accel_std gyro_bias delay_steps kp   (L1/L2/L3 predeclared; F* isolate one factor at a time from L1; G lowers the Mahony gain at L2)
    for spec in "L1 mahony 0.02 0.3 0.02 0 1.0" "L1 madgwick 0.02 0.3 0.02 0 1.0" "L2 mahony 0.05 1.0 0.05 1 1.0" "L2 madgwick 0.05 1.0 0.05 1 1.0" "L3 mahony 0.10 2.0 0.10 2 1.0" \
                "Faccel mahony 0.02 1.0 0.02 0 1.0" "Fgyro mahony 0.05 0.3 0.05 0 1.0" "Fdelay mahony 0.02 0.3 0.02 1 1.0" "Gkp0.3-L2 mahony 0.05 1.0 0.05 1 0.3"; do
      set -- $spec
      "$PY" scripts/bench/inspection_maze.py "${common[@]}" --output "$O/sf03-$1-$2.json" --imu-source estimated --imu-filter "$2" \
        --imu-gyro-std "$3" --imu-accel-std "$4" --imu-gyro-bias "$5" --imu-delay-steps "$6" --imu-kp "$7"
    done ;;
  sf03b)  # attitude filter at 200 Hz from the physics substep hook; IMU delay in ms (SF-03b latency boundary), L1 noise, seeds 0-2
    D=$(ls -d "$U"/logs/rsl_rl/humanoid/*_arms-dr1.0-s0 | head -1); O=results/sensor-fusion-20260923
    common=(--deploy "$D/exported/deploy.yaml" --upstream "$U" --cache-dir "$CACHE" --seeds 3 --seconds 60 --sensor-modes reactive --routes ordered
            --imu-source estimated --imu-filter mahony --imu-rate-hz 200 --imu-gyro-std 0.02 --imu-accel-std 0.3 --imu-gyro-bias 0.02)
    for ms in 0 5 10 20 30 40 60; do
      "$PY" scripts/bench/inspection_maze.py "${common[@]}" --imu-delay-ms "$ms" --output "$O/sf03b-delay${ms}ms.json"
    done ;;
  sf02)   # localization error fed to the route controller (SF-02, MuJoCo side), truth attitude, seeds 0-2
    D=$(ls -d "$U"/logs/rsl_rl/humanoid/*_arms-dr1.0-s0 | head -1); O=results/sensor-fusion-20260923
    common=(--deploy "$D/exported/deploy.yaml" --upstream "$U" --cache-dir "$CACHE" --seeds 3 --seconds 60 --sensor-modes reactive --routes ordered)
    for spec in "bias0.05 --pose-bias-m 0.05" "bias0.10 --pose-bias-m 0.10" "bias0.15 --pose-bias-m 0.15" "bias0.20 --pose-bias-m 0.20" \
                "noise0.05 --pose-noise-m 0.05" "noise0.10 --pose-noise-m 0.10" "yaw3 --pose-yaw-deg 3" "yaw10 --pose-yaw-deg 10" "yaw20 --pose-yaw-deg 20" \
                "drift0.01 --pose-drift-mps 0.01" "drift0.03 --pose-drift-mps 0.03"; do
      set -- $spec; tag=$1; shift
      "$PY" scripts/bench/inspection_maze.py "${common[@]}" "$@" --output "$O/sf02-$tag.json"
    done ;;
  *) echo "unknown mode $mode"; exit 2 ;;
esac
echo REPO_CPU_EVAL_COMPLETE
