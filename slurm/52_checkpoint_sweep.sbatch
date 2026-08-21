#!/bin/bash
#SBATCH --job-name=ckpt-sweep
#SBATCH --account=eecs
#SBATCH --partition=gpu,eecs,dgxh,ampere,preempt
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g|rtx6000|rtx2080|t4"
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out

# sm_75 or newer, any VRAM -- eval starts SimulationApp briefly then runs on CPU.
# Excludes GTX 1080 (sm_61) and V100 (sm_70), which el9 alone would let through.

# Does training reward invert against transfer *within* a run, not just across
# policies?
#
# Finding 1 is that the best-training-reward policy is the worst-transferring
# one. The obvious follow-up is whether the same holds along a single training
# curve -- because if it does, then "take the last checkpoint" is the wrong
# deployment criterion, and training reward cannot tell you that. That turns
# section 1 from an observation into a recommendation.
#
# Six checkpoints x four policies, chosen to span the ladder rather than to be
# exhaustive: the un-randomized rung that already inverts, the repo default that
# transfers best, the aggressive rung whose reward collapses, and the terrain
# arm. No training at all -- this is the cheapest item on the list per unit of
# claim, and it reuses the harness verbatim.
#
# The prediction worth writing down before the numbers arrive: at high s I
# expect the best-transferring checkpoint NOT to be the last one, because the
# late stages of a run at s = 1.5-2.0 are spending their gradient on surviving
# randomization extremes that MuJoCo never presents.

set -euo pipefail
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl OMP_NUM_THREADS=8
export CACHE_DIR=$WORKSPACE/mjcf_cache
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS TERRAIN_D RUN_DIR CKPT EXP"
export EXP=biped
mkdir -p "$REPO/results/checkpoints"

LOGROOT=$UPSTREAM/logs/rsl_rl/biped
ITERS="1000 2000 3000 4000 5000 5999"
WANT="dr-off-s0 dr-default-s0 dr-aggressive-s0 terrain-bumpy-s0"

for want in $WANT; do
    run=$(ls -d "$LOGROOT"/*_"$want" 2>/dev/null | head -1)
    [ -n "$run" ] || { echo "no run for $want"; continue; }
    name=$(basename "$run")

    for it in $ITERS; do
        ckpt="model_${it}.pt"
        [ -f "$run/$ckpt" ] || { echo "no $ckpt in $want"; continue; }
        export RUN_DIR=$name CKPT=$ckpt
        bhl_exec "$REPO/slurm/inner/export_ckpt.sh" 2>&1 | tail -2

        cfg="$run/exported_iter${it}/deploy.yaml"
        [ -f "$cfg" ] || continue
        export DEPLOY_CFG=$cfg LABEL="${want}@${it}"
        export EPISODE_S=10 N_SEEDS=5 PUSH_SPEED=0 TERRAIN_D=0.0 VIDEO_DIR=""
        export OUT_CSV=$REPO/results/checkpoints/${want}__iter${it}.csv
        [ -f "$OUT_CSV" ] && continue
        echo "=== $want @ iter $it ==="
        bhl_exec "$REPO/slurm/inner/eval_one.sh" 2>&1 | grep -E "falls" || true
    done
done

echo "=== checkpoint sweep done: $(date) ==="
ls -1 "$REPO/results/checkpoints"/*.csv 2>/dev/null | wc -l | sed 's/^/csv files: /'
