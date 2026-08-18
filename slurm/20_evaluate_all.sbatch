#!/bin/bash
#SBATCH --job-name=eval-all
#SBATCH --account=eecs
#SBATCH --partition=gpu
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --time=08:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Export every finished policy to ONNX, then score it through the MuJoCo
# harness and render clips.
#
# Needs a GPU for two different reasons: Isaac Sim for the export step, and EGL
# for offscreen rendering. The scoring itself is CPU-only.
#
# Policies trained with a push curriculum are additionally scored under the push
# protocol, so the push videos actually show a shove.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=4
export CACHE_DIR=$WORKSPACE/mjcf_cache
export VIDEO_DIR=$REPO/results/video
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS RUN_DIR"

LOGROOT=$UPSTREAM/logs/rsl_rl/biped
mkdir -p "$REPO/results/raw" "$VIDEO_DIR"

for run in "$LOGROOT"/*/; do
    name=$(basename "$run")
    [ "$name" = "isaaclab" ] && continue
    ls "$run"/model_*.pt >/dev/null 2>&1 || { echo "skip (no checkpoint): $name"; continue; }

    # Label from the run_name suffix that train.py appends.
    label=$(echo "$name" | sed 's/^[0-9-]*_[0-9-]*_//')

    if [ ! -f "$run/exported/policy.onnx" ]; then
        echo "=== export $label ==="
        export RUN_DIR=$name
        bhl_exec "$REPO/slurm/inner/export_one.sh" >/dev/null 2>&1 || true
    fi
    [ -f "$run/exported/deploy.yaml" ] || { echo "  export FAILED: $label"; continue; }

    # Push-trained policies get scored under disturbance; everything else on
    # flat ground with no push, so the DR ladder stays a clean comparison.
    case "$label" in
        push-*) PUSH=0.4 ;;
        *)      PUSH=0 ;;
    esac

    echo "=== evaluate $label (push=$PUSH) ==="
    export DEPLOY_CFG=$run/exported/deploy.yaml
    export OUT_CSV=$REPO/results/raw/$label.csv
    export LABEL=$label EPISODE_S=10 N_SEEDS=5 PUSH_SPEED=$PUSH
    bhl_exec "$REPO/slurm/inner/eval_one.sh" 2>&1 | grep -vE "pthread_setaffinity|EGLError" || true
done

echo "=== summary ==="
ls -1 "$REPO/results/raw"/*.csv 2>/dev/null | wc -l | sed 's/^/csv files: /'
ls -1 "$VIDEO_DIR"/*.mp4 2>/dev/null | wc -l | sed 's/^/videos:    /'
