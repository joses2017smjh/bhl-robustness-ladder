#!/bin/bash
#SBATCH --job-name=navgym-v3
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --array=2-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-09-27: NavGym v3 -- CONFIRMATORY run, 3 FRESH seeds (2, 3, 4; seeds 0 and 1 were v2).
#
# v3 = the v2 environment (MazeNavEnv(version=2), unchanged) + `--ppo v3` = the v2 PPO settings
# plus ONE change: a policy std CEILING of 1.0 (log_std clamped to [log 0.2, log 1.0] after
# every update, in the v2 std floor's own hook). Nothing else differs from v2 Arm A: 30 M steps,
# 16 envs, same curriculum, same periodic evaluation (held-out 10000-10023, 5x5 + 6x6, every 1 M)
# and best-checkpoint selection.
#
# Why (scripts/bench/navgym_diagnose.py on results/navgym-v2-20260926/armA-s{0,1}, 5x5 maze
# seeds 10000-10047 -- the published v2 eval seeds, diagnosis only):
#   * the pre-clamp std grew monotonically (ent_coef 0.01, actions clipped to [-1, 1]) to 10.0 / 4.6
#     (s0, vx / wz) and 8.0 / 4.5 (s1); the means followed: median |mean| 25 (vx) and 6-12 (wz),
#     median 91-96 % of wz means and >= 99 % of vx means outside [-1, 1] -> the deterministic
#     actor is bang-bang, wz sign flips 15-20 per second even on successful episodes;
#   * the 15 5x5 time-outs of the final actors (s0 7, s1 8): 11 are a dither in place by the
#     declared classes (stalled / looping, < 3 cm/s over the last 20 s, never near a wall); the
#     other 4 move but wander (net geodesic progress 0.04-0.07 m/s over the episode vs the ~0.10
#     m/s the limit needs). Forward command at zero: median 72 % of the steps of a
#     time-out vs 3-6 % of a success. 0 of 15 reaches the goal with twice the time, so the
#     route-scaled time limit is not the cause;
#   * across the 15 checkpoints of each seed, std vs wz saturation: Spearman 0.97 / 0.96; the
#     checkpoints at std <= 1.15 (2 M and 4 M steps, both seeds) had wz saturation 0.10-0.23,
#     1.4-2.7 flips/s and periodic 5x5 time-outs 0.00-0.125 (mean 0.06) vs 0.24 / 0.28 at higher
#     std (but also more collisions, 0.13-0.29; confounded with training time);
#   * float-level differences (torch vs ONNX vs the batched pool, <= 1e-5) move the final s0 5x5
#     success between 0.771 and 0.812 on the same 48 episodes: saturated actors make the outcome
#     of single episodes chaotic, which the one-clause miss of v2 (0.771 vs 0.80) sits inside.
#
# PREDECLARED (2026-09-27, before any v3 training; no data from maze seeds >= 20000 has been
# looked at by anyone):
#   v3 PASSES iff >= 2 of the 3 seeds (2, 3, 4) meet EVERY clause, judged on the FINAL actor's
#   evaluation on the FRESH held-out set (maze seeds 20000-20047, dynamics reset(seed=20000+k),
#   48 per size, deterministic, v2 time limit; summary.json "final_eval_fresh_48"):
#     5x5 success >= 0.80, 6x6 success >= 0.70, 6x6 collision <= 0.15, and the run reached its
#     last step (summary.json num_timesteps >= 30 000 000; the summary is written only after
#     model.learn returns). Anything else is NEGATIVE.
#   Also reported, NOT gated: the same actors on seeds 10000-10047 (comparability with v2), the
#   best checkpoint (selected on 10000-10023) on both sets, std and clamp counts.
#   Overlap statement: the v3 design used diagnosis on 10000-10047 (and the v2 periodic evals on
#   10000-10023); none of it touched 20000-20047.
#   The verdict is computed from summary.json (navgym_train.py verdict-v3), never an exit code.
#
# Modes (NAVGYM_MODE):
#   train   (default) one array task = one seed, output results/navgym-v3-20260927/armV3-s<seed>/
#   verdict the final arm verdict after every task has ended (missing summary = a miss):
#           sbatch --dependency=afterany:<array job> --array=0 --cpus-per-task=1 --mem=2G --time=00:10:00 \
#                  --export=ALL,NAVGYM_MODE=verdict cpu_navgym_v3.sbatch
#
# Wall time: v2 Arm A took 9.2-9.3 h for 30 M steps on 16 CPUs of el9 share nodes; v3 adds the
# fresh-set evaluations of the final and best actors (2 x 144 episodes, ~10 min) -> 24 h (x2.5).
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
ROOT=${NAVGYM_ROOT:-$REPO/results/navgym-v3-20260927}
MODE=${NAVGYM_MODE:-train}
mkdir -p "$ROOT"
if [ "$MODE" = verdict ]; then
  echo "=== NavGym v3 final verdict | HEAD $(git rev-parse --short HEAD) | $(date) ==="
  "$PY" scripts/bench/navgym_train.py verdict-v3 "$ROOT" --seeds 2 3 4 --final --out "$ROOT/verdict_v3.json"
  exit 0
fi
[ "$MODE" = train ] || { echo "NAVGYM_MODE must be train or verdict, got '$MODE'"; exit 2; }
SEED=${NAVGYM_SEED:-${SLURM_ARRAY_TASK_ID:?run as an array task (2-4) or set NAVGYM_SEED}}
case "$SEED" in 2|3|4) ;; *) echo "v3 seeds are 2, 3, 4 (0 and 1 were v2), got '$SEED'"; exit 2 ;; esac
STEPS=30000000
OUT=$ROOT/armV3-s${SEED}
if [ -e "$OUT/summary.json" ]; then
  echo "REFUSING: $OUT/summary.json exists (never overwrite a result)"; exit 3
fi
mkdir -p "$OUT"
echo "=== node: $(hostname) | HEAD $(git rev-parse --short HEAD) | NavGym v3 (env v2, ppo v3: std ceiling 1.0) | seed $SEED | steps $STEPS | $(date) ==="
set +e
"$PY" scripts/bench/navgym_train.py --out "$OUT" --seed "$SEED" --total-steps "$STEPS" --env-version 2 --ppo v3 \
    --n-envs 16 --eval-every 1000000 --final-eval-fresh-base 20000 2>&1 | grep --line-buffered -vE "Warning|warnings.warn"
RC=${PIPESTATUS[0]}
set -e
echo "trainer exit code $RC (informational; the verdict is computed from summary.json)"
# this seed's checks plus the arm status so far (PENDING while other tasks run); the final verdict is the verdict mode
"$PY" scripts/bench/navgym_train.py verdict-v3 "$ROOT" --seeds 2 3 4 --out "$OUT/verdict_v3_at_end_of_s${SEED}.json"
echo "=== done: $(date) ==="
