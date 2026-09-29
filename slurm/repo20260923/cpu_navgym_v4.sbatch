#!/bin/bash
#SBATCH --job-name=navgym-v4
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --array=5-7
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=18:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-09-28: NavGym v4 -- CONFIRMATORY run, 3 NEW training seeds (5, 6, 7; 0-1 were v2, 2-4 v3 and v2ctl).
#
# v4 = v2 exactly (MazeNavEnv(version=2), `--ppo v2`: target_kl 0.02, LR 3e-4 -> 3e-5, ent 0.01,
# log_std_init -0.5, std floor 0.2, gamma 0.998, NO std ceiling) + ONE change that prices stalling:
#   `--idle-cost 0.008` -- in the TRAINING envs, an extra -0.008 on every step whose commanded forward
#   action a[0] <= -0.9 (v_cmd <= 5 % of V_MAX, "forward command ~0") while the straight-line goal
#   distance at the start of the step is > 0.5 m. With the unchanged -0.002 step cost an endless idle
#   stall is worth (-0.002 - 0.008) / (1 - 0.998) = -5.0 at the learner's gamma, the crash penalty,
#   instead of -1.0 in v2 (SB3 bootstraps truncations: on_policy_algorithm.py adds gamma * V(s_T) when
#   TimeLimit.truncated). Moving steps, the crash (-5), the goal (+5) and the progress shaping are
#   unchanged. Held-out evaluations are unaffected (the idle cost changes no outcome). Everything else as
#   cpu_navgym_v3.sbatch: 30 M steps, 16 envs, same curriculum, periodic evaluation on 10000-10023.
#
# Why (wf-verify-0928/diag-navgym-v3-regression): v2's and v3's time-outs are a COMMANDED STOP -- forward
# command zero on 72 % of time-out steps vs 3-6 % on successes, never near a wall -- and stalling is
# cheap: ~ -1 discounted vs -5 for a crash. v3's std ceiling bound but did not remove it (NEGATIVE 0/3).
#
# Lever chosen by a PILOT on TRAINING-RANGE mazes only (maze seeds < 10 000, asserted on every episode),
# rule written before any pilot episode ran (scratch wf-next-0928/navgym-v2ctl-v4/PILOT_RULE.txt), between
#   (i)  raising the step cost to -0.010 (an endless stall = -5.0), and
#   (ii) the idle cost above (an endless idle stall = -5.0); both constants fixed by "stall = crash",
#        not tuned. Rule: at each time-out's stall onset (500 steps before the limit), dG = how much the
#        lever raises the value of moving on over stalling, dD = how much it changes the value of moving
#        on over crashing (crash deterrent), both vs v2; choose (ii) iff its median dG >= (i)'s AND its
#        median dD >= (i)'s; then a route check on the 5 longest training 6x6 routes.
#   Pilot data: learned actors v2 Arm A final s0, s1 and v3 final s2, s4, deterministic, on maze seeds
#   9000-9023 (5x5 and 6x6, dynamics reset(seed=9000 + k)): 192 episodes, 62 time-outs. Longest routes:
#   BFS scan of 6x6 maze seeds 0-9999, fine geodesic on the top 20; the 5 longest are seeds 412, 8838,
#   9303, 3124, 3466 (32.6-35.8 m; the 4500-step cap binds on all), each run by the scripted A* +
#   turn-then-walk driver (3/5 reach the goal, in 3610-4201 steps).
#   Result: pooled medians dG (i) +0.24, (ii) +2.08; dD (i) -3.75, (ii) -1.84 -> (ii). (On the two v2
#   Arm A actors alone: dG +0.65 vs +3.90, dD -3.34 vs -0.09.) A raised step cost is paid by movers and
#   stallers alike, so it barely separates them over a 500-step discount horizon, and crashing ends the
#   step-cost stream, so it weakens the crash deterrent; the idle cost is paid only by idle steps.
#   The idle cost covers a median 100 % of the time-out windows (v2 Arm A s1: 37 % -- its time-outs are
#   partly slow wandering, which (ii) does not price) and 3.5 % of the steps of successful episodes
#   (median change of a success's discounted return: (i) -3.58, (ii) -0.20).
#   Route check (PASS): a uniform-pace traverse of each of the 5 longest routes that uses the WHOLE
#   4500-step limit, with the worst-case idle fraction 0.43-0.48 (every step not needed at full speed
#   idle), is worth 5.69-6.25 more than an endless stall and than a crash at the start (need >= 1.0);
#   undiscounted +11.2 to +16.3. The scripted driver's successful traversals: undiscounted +22 to +30
#   under (ii) (v2: +30 to +33; under (i) they would have been -2.3 to +4.0).
#   FROZEN before this launcher was written: lever (ii), idle cost 0.008, threshold a[0] <= -0.9,
#   far = 0.5 m (straight line, start of the step).
#   Known limitation: the price is on the COMMAND, so a policy can dodge it by creeping or circling with
#   a[0] just above -0.9 (1 of 18 fully stalled pilot windows already stalled that way).
#
# PREDECLARED GATE (2026-09-28, before any v4 training; no data from maze seeds 40000-40047 has been
# looked at by anyone):
#   v4 PASSES iff >= 2 of the 3 seeds (5, 6, 7) meet EVERY clause, judged on the FINAL actor's
#   evaluation on the NEVER-USED set (maze seeds 40000-40047, dynamics reset(seed=40000 + k), 48 per
#   size, deterministic, v2 time limit; summary.json "final_eval_fresh_48"):
#     5x5 success >= 0.80, 6x6 success >= 0.70, 6x6 collision <= 0.15, n = 48 per size, and the run
#     reached its last step (num_timesteps >= 30 000 000), and the run is the declared v4 configuration
#     (ppo v2, env v2, no std ceiling, summary.json stall_price == the frozen idle cost). Anything else
#     is NEGATIVE.
#   Also reported, NOT gated: the same actors on 10000-10047 (comparability with v2 / v3 / v2ctl), the
#   best checkpoint (selected on 10000-10023) on both sets, stall/idle_charged_frac (TensorBoard) and the
#   training idle fraction.
#   Never-used set: checked 2026-09-28 with grep -- no maze seed 40000-40049 appears anywhere in
#   results/ (20000-20047 is spent by v3, 30000-30011 by the physics transfer). The same set is the
#   report-only never-used set of cpu_navgym_v2ctl.sbatch; no design step of either launcher read it.
#   The verdict is computed from summary.json (navgym_train.py verdict-v4), never an exit code.
#   Labels: learned actors (deterministic Gaussian mean); pose and goal coordinate are oracle inputs.
#
# Modes (NAVGYM_MODE):
#   train   (default) one array task = one seed, output results/navgym-v4-20260928/armV4-s<seed>/
#   verdict the final arm verdict after every task has ended (missing summary = a miss):
#           sbatch --dependency=afterany:<array job> --array=0 --cpus-per-task=1 --mem=2G --time=00:10:00 \
#                  --export=ALL,NAVGYM_MODE=verdict cpu_navgym_v4.sbatch
#
# Wall time: v3 took 9.1-9.3 h and v2 Arm A 9.2-9.3 h for 30 M steps on 16 CPUs -> 18 h (2x; cut from 24 h on 2026-09-28 so a run fits in the 40 h gap between the 09-29 and 10-01 maintenance reservations).
set -euo pipefail
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
cd "$REPO"
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
ROOT=${NAVGYM_ROOT:-$REPO/results/navgym-v4-20260928}
MODE=${NAVGYM_MODE:-train}
mkdir -p "$ROOT"
if [ "$MODE" = verdict ]; then
  echo "=== NavGym v4 final verdict | HEAD $(git rev-parse --short HEAD) | $(date) ==="
  "$PY" scripts/bench/navgym_train.py verdict-v4 "$ROOT" --seeds 5 6 7 --final --out "$ROOT/verdict_v4.json"
  exit 0
fi
[ "$MODE" = train ] || { echo "NAVGYM_MODE must be train or verdict, got '$MODE'"; exit 2; }
SEED=${NAVGYM_SEED:-${SLURM_ARRAY_TASK_ID:?run as an array task (5-7) or set NAVGYM_SEED}}
case "$SEED" in 5|6|7) ;; *) echo "v4 seeds are 5, 6, 7 (0-4 were v2 / v3 / v2ctl), got '$SEED'"; exit 2 ;; esac
STEPS=30000000
OUT=$ROOT/armV4-s${SEED}
if [ -e "$OUT/summary.json" ]; then
  echo "REFUSING: $OUT/summary.json exists (never overwrite a result)"; exit 3
fi
mkdir -p "$OUT"
echo "=== node: $(hostname) | cpu: $(lscpu | sed -n 's/^Model name: *//p') | HEAD $(git rev-parse --short HEAD) | NavGym v4 (env v2, ppo v2, idle cost 0.008) | seed $SEED | steps $STEPS | $(date) ==="
set +e
"$PY" scripts/bench/navgym_train.py --out "$OUT" --seed "$SEED" --total-steps "$STEPS" --env-version 2 --ppo v2 --idle-cost 0.008 \
    --n-envs 16 --eval-every 1000000 --final-eval-fresh-base 40000 2>&1 | grep --line-buffered -vE "Warning|warnings.warn"
RC=${PIPESTATUS[0]}
set -e
echo "trainer exit code $RC (informational; the verdict is computed from summary.json)"
# this seed's checks plus the arm status so far (PENDING while other tasks run); the final verdict is the verdict mode
"$PY" scripts/bench/navgym_train.py verdict-v4 "$ROOT" --seeds 5 6 7 --out "$OUT/verdict_v4_at_end_of_s${SEED}.json"
echo "=== done: $(date) ==="
