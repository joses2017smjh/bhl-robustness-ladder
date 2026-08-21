#!/bin/bash
#SBATCH --job-name=coll-eval
#SBATCH --account=eecs
#SBATCH --partition=gpu,eecs,dgxh,ampere,preempt
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# Score the two convex-mesh policies in MuJoCo. Section 4 currently ends on
# "these two policies have not been scored in MuJoCo yet", which leaves its own
# question half answered: swapping the colliding geometry does not move training
# reward, but training reward is exactly the quantity section 1 showed to be
# misleading about transfer.
#
# The comparison is against the primitive-collision s=1.0 rung already scored by
# 20_evaluate_all (dr-default-s0..2, sim2sim fall 0.000), same protocol, same
# terrain seed. A null result here is not a wasted run: it makes section 4 a
# clean two-part negative -- the assets agree AND the colliding geometry does not
# matter -- which is a stronger claim than the half it can make now.
#
# A GPU is requested only because the ONNX export needs SimulationApp; the
# scoring itself is CPU MuJoCo.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=8
export CACHE_DIR=$WORKSPACE/mjcf_cache
mkdir -p "$REPO/results/collision" "$REPO/results/video"

LOGROOT=$UPSTREAM/logs/rsl_rl/biped

for run in "$LOGROOT"/*coll-convex-s*/; do
    [ -d "$run" ] || continue
    name=$(basename "$run")
    label=$(echo "$name" | sed 's/^[0-9-]*_[0-9-]*_//')
    ls "$run"/model_*.pt >/dev/null 2>&1 || { echo "no checkpoint: $label"; continue; }

    if [ ! -f "$run/exported/policy.onnx" ]; then
        export RUN_DIR=$name
        bhl_exec "$REPO/slurm/inner/export_one.sh" 2>&1 | tail -3
    fi
    [ -f "$run/exported/deploy.yaml" ] || { echo "no export: $label"; continue; }

    export DEPLOY_CFG=$run/exported/deploy.yaml
    export LABEL=$label EPISODE_S=10 N_SEEDS=5 PUSH_SPEED=0

    # Flat, the protocol every DR rung was scored on, so the row drops straight
    # into the section 1 table.
    export TERRAIN_D=0.0
    export OUT_CSV=$REPO/results/collision/${label}__flat.csv
    export VIDEO_DIR=$REPO/results/video
    echo "=== $label @ flat ==="
    bhl_exec "$REPO/slurm/inner/eval_one.sh" 2>&1 | grep -E "falls|video ->" || true

    # Rough ground too: primitives vs meshes is a contact-geometry question, and
    # contact geometry is exactly what a bumpy surface exercises. If the swap
    # matters anywhere it should matter here.
    export VIDEO_DIR=""
    for d in 0.20 0.60; do
        export TERRAIN_D=$d
        export OUT_CSV=$REPO/results/collision/${label}__d${d}.csv
        echo "=== $label @ terrain d=$d ==="
        bhl_exec "$REPO/slurm/inner/eval_one.sh" 2>&1 | grep -E "falls" || true
    done
done

echo "=== collision eval done: $(date) ==="
ls -1 "$REPO/results/collision"/*.csv 2>/dev/null | wc -l | sed 's/^/csv files: /'
