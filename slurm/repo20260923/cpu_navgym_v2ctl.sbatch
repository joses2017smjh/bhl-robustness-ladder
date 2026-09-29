#!/bin/bash
#SBATCH --job-name=navgym-v2ctl
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9&skylake
#SBATCH --array=2-4
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-09-28: NavGym v2 MATCHED-SEED CONTROL for v3 ("v2ctl") -- REPORT-ONLY, no gate.
#
# v2ctl = the v2 recipe exactly (MazeNavEnv(version=2), `--ppo v2`: target_kl 0.02, LR 3e-4 -> 3e-5,
# ent 0.01, log_std_init -0.5, std floor 0.2, gamma 0.998, NO std ceiling) on training seeds 2, 3, 4 --
# the seeds of v3 (results/navgym-v3-20260927/armV3-s{2,3,4}). Everything else as cpu_navgym_v3.sbatch:
# 30 M steps, 16 envs, same curriculum, periodic evaluation on held-out 10000-10023 (5x5 + 6x6, every 1 M),
# best-checkpoint selection, final + best actors on 10000-10047. The one other difference: the final and
# best actors are ALSO evaluated on the never-used set (maze seeds 40000-40047, dynamics reset(seed=40000+k))
# instead of v3's 20000-set, which is spent. Training is unaffected by the evaluation sets.
# The navgym training code is unchanged since both v2 Arm A (d291263) and v3 (8d00f24) ran (`git diff`
# of env.py / navgym_train.py: only held-out-set helpers and the opt-in v3 ceiling were added; the opt-in
# v4 --idle-cost of 2026-09-28 is off here, default paths proven identical by a before/after fingerprint), so the
# v2ctl runs and v2 Arm A s0/s1 share one training code path.
#
# Why: v3 (v2 + std ceiling 1.0) was NEGATIVE 0/3 and WORSE than v2 (s2 collapsed; s3/s4 collided more),
# but v2 was only ever run on seeds 0-1 and v3 on 2-4, so seed and ceiling are confounded. The
# diagnosis (wf-verify-0928/diag-navgym-v3-regression) found: the ceiling bound (std pinned 1.00 from
# 3.4-4.5 M steps) but the MEANS grew (turn saturation 0.04 -> 0.87); a commanded STOP (forward command
# zero on 72 % of time-out steps vs 3-6 % on successes) separates time-outs from successes; stalling is
# cheap (~ -1 discounted vs -5 for a crash, because SB3 bootstraps truncations); and s2's collapse began
# BEFORE any clamp (reward 7.44 -> -0.08 by 2.75 M; first clamp at 4.48 M; it never reached stage 3).
#
# PREDECLARED REPORT-ONLY READING (2026-09-28, before any v2ctl run; not a gate):
#   (a) determinism check -- if each v2ctl run's rollout/ep_rew_mean matches its v3 twin (relative
#       tolerance 1e-6) at every point logged at a step strictly below that seed's first logged
#       std-ceiling clamp (TensorBoard train/std_ceiling_clamps > 0 first at 4 481 024 / 3 645 440 /
#       3 362 816 steps for s2 / s3 / s4), the pair is a SINGLE-FACTOR comparison, else a SEED-MATCHED
#       REPLICATE (the first divergence step is reported);
#   (b) s2: v2ctl-s2 FINAL 5x5 success on the 10000-set < 0.30 -> the collapse is not caused by the
#       ceiling; >= 0.60 -> the ceiling caused or deepened it; else inconclusive;
#   (c) on the 10000-set (the only set both twins are evaluated on by identical code): >= 2/3 v2ctl seeds
#       with 6x6 collision at least 0.10 lower than the v3 twin -> the ceiling raised collisions;
#       >= 2/3 within 0.05 -> it did not; else inconclusive.
#   The v3 twins' 10000-set numbers (from their summary.json, already published): 5x5 success 0.104 /
#   0.729 / 0.562; 6x6 collision 0.146 / 0.458 / 0.271 (s2 / s3 / s4).
#   Also reported, NOT read: every final actor on the NEVER-USED set, maze seeds 40000-40047
#   (checked 2026-09-28 with grep: no maze seed 40000-40049 appears anywhere in results/; the 20000-set is
#   spent by v3, 30000-30011 by the physics transfer); a 5-seed v2 estimate (Arm A s0, s1 + v2ctl s2-s4,
#   10000-set, final actors). The 40000-set is shared with cpu_navgym_v4.sbatch (its gate set); no design
#   step of either launcher read it.
#   The reading is computed from summary.json + TensorBoard (navgym_train.py reading-v2ctl), never from an
#   exit code, and is written only under this launcher's result root.
#   Labels: learned actors (deterministic Gaussian mean); pose and goal coordinate are oracle inputs.
#
# Determinism note: `--constraint=el9&skylake` matches the v3 twins' hardware (cn-e02/03/04, skylake,
# avx512). On other el9 share nodes (broadwell, epyc) float results can differ, which would turn (a) into
# a seed-matched replicate. Dropping `&skylake` only affects (a).
#
# Modes (NAVGYM_MODE):
#   train   (default) one array task = one seed, output results/navgym-v2ctl-20260928/armV2C-s<seed>/
#   reading the final report-only reading after every task has ended:
#           sbatch --dependency=afterany:<array job> --array=0 --cpus-per-task=1 --mem=4G --time=00:20:00 \
#                  --constraint=el9 --export=ALL,NAVGYM_MODE=reading cpu_navgym_v2ctl.sbatch
#
# Wall time: v3 took 9.1-9.3 h and v2 Arm A 9.2-9.3 h for 30 M steps on 16 CPUs -> 18 h (2x; cut from 24 h on 2026-09-28 so a run fits in the 40 h gap between the 09-29 and 10-01 maintenance reservations).
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
ROOT=${NAVGYM_ROOT:-$REPO/results/navgym-v2ctl-20260928}
V3ROOT=$REPO/results/navgym-v3-20260927
MODE=${NAVGYM_MODE:-train}
mkdir -p "$ROOT"
if [ "$MODE" = reading ]; then
  echo "=== NavGym v2ctl final reading (report-only) | HEAD $(git rev-parse --short HEAD) | $(date) ==="
  "$PY" scripts/bench/navgym_train.py reading-v2ctl "$ROOT" --v3-root "$V3ROOT" --out "$ROOT/reading_v2ctl.json"
  exit 0
fi
[ "$MODE" = train ] || { echo "NAVGYM_MODE must be train or reading, got '$MODE'"; exit 2; }
SEED=${NAVGYM_SEED:-${SLURM_ARRAY_TASK_ID:?run as an array task (2-4) or set NAVGYM_SEED}}
case "$SEED" in 2|3|4) ;; *) echo "v2ctl seeds are 2, 3, 4 (the v3 twins), got '$SEED'"; exit 2 ;; esac
[ -f "$V3ROOT/armV3-s${SEED}/summary.json" ] || { echo "the v3 twin $V3ROOT/armV3-s${SEED} is missing"; exit 2; }
STEPS=30000000
OUT=$ROOT/armV2C-s${SEED}
if [ -e "$OUT/summary.json" ]; then
  echo "REFUSING: $OUT/summary.json exists (never overwrite a result)"; exit 3
fi
mkdir -p "$OUT"
echo "=== node: $(hostname) | cpu: $(lscpu | sed -n 's/^Model name: *//p') | avx512f: $(grep -qw avx512f /proc/cpuinfo && echo yes || echo no) | HEAD $(git rev-parse --short HEAD) | NavGym v2ctl (env v2, ppo v2, no ceiling) | seed $SEED | steps $STEPS | $(date) ==="
echo "v3 twin ran on: $(grep -m1 -o 'node: [^ ]*' /nfs/hpc/share/sanchej7/Humanoid_Lite/logs/navgym-v3-21442348_${SEED}.out 2>/dev/null || echo unknown)"
set +e
"$PY" scripts/bench/navgym_train.py --out "$OUT" --seed "$SEED" --total-steps "$STEPS" --env-version 2 --ppo v2 \
    --n-envs 16 --eval-every 1000000 --final-eval-fresh-base 40000 2>&1 | grep --line-buffered -vE "Warning|warnings.warn"
RC=${PIPESTATUS[0]}
set -e
echo "trainer exit code $RC (informational; the reading is computed from summary.json + TensorBoard)"
# the reading so far (seeds still running show as "no summary"); the final reading is the reading mode
"$PY" scripts/bench/navgym_train.py reading-v2ctl "$ROOT" --v3-root "$V3ROOT" --out "$OUT/reading_at_end_of_s${SEED}.json"
echo "=== done: $(date) ==="
