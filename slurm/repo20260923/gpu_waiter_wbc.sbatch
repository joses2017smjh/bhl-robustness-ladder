#!/bin/bash
#SBATCH --job-name=waiter-wbc
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# Waiter program phase 1 (docs/WAITER_PROGRAM.md; frozen 2026-10-05, SLURM_JOBS.md "Predeclared now ... the Waiter
# program"): train Velocity-BHL-Waiter-WBC-v0 FROM SCRATCH (seeds 0-2 = array index, 6000 iterations, the task's env
# count, R1's DR overrides at scale 1.0), export (train_play.py via export_one_humanoid.sh, under the shared export
# lock), check the export (24 joints, 12 leg actions, 83 actor observations in the ONNX) and stamp the gait_clock +
# waiter_wbc blocks (bhl_robust.eval.waiter_wbc stamp). The MuJoCo gates Q1 / Q2 run in a separate CPU job.
# Smoke: WAITER_SMOKE=1 (3 iterations, 64 envs, outputs under a directory ending in -smoke): the same steps.
# usage: S=$(slurm_clean sbatch --parsable --array=0 --export=ALL,WAITER_SMOKE=1 slurm/repo20260923/gpu_waiter_wbc.sbatch) && \
#        slurm_clean sbatch --dependency=afterok:${S%%;*} --export=ALL,WAITER_SMOKE=0 slurm/repo20260923/gpu_waiter_wbc.sbatch
# WAITER_VARIANT=1c (2026-10-07; SLURM_JOBS.md 'Predeclared now ... Waiter phase 1c'): task
# Velocity-BHL-Waiter-WBC-Curriculum-v0 (= 1b + the disturbance curriculum: arm goals, hand forces and hand payload
# scaled 0 before iteration 4000, linear to 1 at 7000, then 1), seeds 6-8, 10000 iterations, std <= 1.0; submit with
# --time=16:00:00. Its smoke compresses the ramp to iterations 0-2 (Hydra override) so the log shows the term move;
# its full run must log Curriculum/disturbance 0 first and 1.0 last, with the frozen ramp values in params/env.yaml.
# WAITER_VARIANT=1d (2026-10-08; SLURM_JOBS.md 'Predeclared now ... Waiter phase 1d'): task
# Velocity-BHL-Waiter-WBC-Stance-v0 (= 1c + the trained stance: standing envs' gait clock held at phase 0.5 in the
# observations and feet_gait, rel_standing_envs 0.25), seeds 9-11, otherwise as 1c (the same curriculum checks) plus
# the stance check in params/env.yaml (also done by the stamp); submit with --time=16:00:00.
# WAITER_VARIANT=1e (2026-10-10; SLURM_JOBS.md 'Predeclared now ... Waiter phase 1e'): task
# Velocity-BHL-Waiter-WBC-Stand-v0 (= 1d + no pushes / hand forces on standing envs + the standing penalties
# stand_anchor and stand_tilt), seeds 12-14, otherwise as 1d (the same curriculum checks) plus the stand check in
# params/env.yaml (also done by the stamp); submit with --time=16:00:00.
# WAITER_VARIANT=1f (2026-10-11; SLURM_JOBS.md 'Predeclared now ... Waiter phase 1f'): task
# Velocity-BHL-Waiter-WBC-StandPush-v0 (= 1d + the standing penalties; standing envs keep their pushes and hand forces;
# a new anchor 1 s after each push), seeds 15-17, otherwise as 1d (the same curriculum checks) plus the stand-push
# check in params/env.yaml (also done by the stamp); submit with --time=16:00:00.
# WAITER_VARIANT=1g (2026-10-11; SLURM_JOBS.md 'Predeclared now ... Waiter phase 1g'): task
# Velocity-BHL-Waiter-WBC-StandBonus-v0 (= 1f with the two standing penalties replaced by bounded positive rewards of
# the same error measures: stand_anchor_bonus +2.0, stand_tilt_bonus +1.0), seeds 18-20, otherwise as 1f (the same
# curriculum and std checks) plus the stand-bonus check in params/env.yaml (also done by the stamp); submit with
# --time=16:00:00.
# WAITER_VARIANT=1h (2026-10-12; SLURM_JOBS.md 'Predeclared now ... Waiter phase 1h'): task
# Velocity-BHL-Waiter-WBC-StandStill-v0 (= 1d + Isaac Lab's stand-still term on the leg joints, weight -0.4, while the
# whole command is near zero + a walk-first curriculum on the standing share, 0.02 -> 0.25 over iterations 4000-7000),
# seeds 21-23, otherwise as 1d (the same curriculum and std checks) plus the stand-still check in params/env.yaml (also
# done by the stamp) and the standing-share log check; submit with --time=16:00:00.
# WAITER_VARIANT=1i (2026-10-13; SLURM_JOBS.md 'Predeclared now ... Waiter phase 1i'): task
# Velocity-BHL-Waiter-WBC-StandOnly-v0, a STANDING-ONLY WBC (every env standing; 1g's bounded standing rewards and 1h's
# stand-still term; pushes with re-anchoring), seeds 24-26, otherwise as 1d (the same curriculum and std checks) plus
# the stand-only check in params/env.yaml (also done by the stamp). It is gated as a pair with s7
# (cpu_waiter_gates.sbatch WAITER_GATES_DUAL_WALK); submit with --time=16:00:00.
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
PYH=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid
JOBTAG=${SLURM_JOB_ID:-local$$}
SMOKE=${WAITER_SMOKE:-0}
TASK_ID=Velocity-BHL-Waiter-WBC-v0
VARIANT=${WAITER_VARIANT:-1}         # 1 = phase 1 as frozen; 1b = + action-noise std bounded to <= 1.0, seeds 3-5
FULL_ITERS=6000
case "$VARIANT" in
    1)  PREFIX=waiter-wbc;   SEED0=0; RES_NAME=wbc ;;
    1b) PREFIX=waiter-wbc1b; SEED0=3; RES_NAME=wbc1b; export BHL_STD_MAX=1.0 ;;
    1c) PREFIX=waiter-wbc1c; SEED0=6; RES_NAME=wbc1c; export BHL_STD_MAX=1.0
        TASK_ID=Velocity-BHL-Waiter-WBC-Curriculum-v0; FULL_ITERS=10000 ;;
    1d) PREFIX=waiter-wbc1d; SEED0=9; RES_NAME=wbc1d; export BHL_STD_MAX=1.0
        TASK_ID=Velocity-BHL-Waiter-WBC-Stance-v0; FULL_ITERS=10000 ;;
    1e) PREFIX=waiter-wbc1e; SEED0=12; RES_NAME=wbc1e; export BHL_STD_MAX=1.0
        TASK_ID=Velocity-BHL-Waiter-WBC-Stand-v0; FULL_ITERS=10000 ;;
    1f) PREFIX=waiter-wbc1f; SEED0=15; RES_NAME=wbc1f; export BHL_STD_MAX=1.0
        TASK_ID=Velocity-BHL-Waiter-WBC-StandPush-v0; FULL_ITERS=10000 ;;
    1g) PREFIX=waiter-wbc1g; SEED0=18; RES_NAME=wbc1g; export BHL_STD_MAX=1.0
        TASK_ID=Velocity-BHL-Waiter-WBC-StandBonus-v0; FULL_ITERS=10000 ;;
    1h) PREFIX=waiter-wbc1h; SEED0=21; RES_NAME=wbc1h; export BHL_STD_MAX=1.0
        TASK_ID=Velocity-BHL-Waiter-WBC-StandStill-v0; FULL_ITERS=10000 ;;
    1i) PREFIX=waiter-wbc1i; SEED0=24; RES_NAME=wbc1i; export BHL_STD_MAX=1.0
        TASK_ID=Velocity-BHL-Waiter-WBC-StandOnly-v0; FULL_ITERS=10000 ;;
    *)  echo "WAITER-WBC RESULT: FAIL (unknown WAITER_VARIANT $VARIANT)"; exit 2 ;;
esac
OBS_POLICY=83
if [ "$SMOKE" = 1 ]; then
    RES=${WAITER_OUT_DIR:-/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/campaign-waiter/$RES_NAME-smoke}
    case "$RES" in *-smoke) ;; *) echo "SMOKE: FAIL (smoke outputs must go to a directory ending in -smoke)"; exit 1;; esac
else
    RES=$REPO/results/waiter-20261005/$RES_NAME
fi
IDX=${SLURM_ARRAY_TASK_ID:-0}
case $IDX in 0|1|2) SEED=$((SEED0 + IDX));; *) echo "WAITER-WBC RESULT: FAIL (array index $IDX outside 0-2)"; exit 2;; esac
if [ "$SMOKE" = 1 ]; then
    export MAX_ITER=3 NUM_ENVS=64 RUN_NAME="$PREFIX-s${SEED}-smoke-$JOBTAG"
else
    export MAX_ITER=$FULL_ITERS RUN_NAME="$PREFIX-s${SEED}"
    unset NUM_ENVS
fi
FINAL_CKPT="model_$((MAX_ITER - 1)).pt"
export TASK="$TASK_ID" SEED
mkdir -p "$RES"/training
unset BHL_POLICY BHL_RND BHL_SYMMETRY
[ "$VARIANT" = 1 ] && unset BHL_STD_MAX
source "$REPO/slurm/dr_levels.sh"
setup_node_cache
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate
EXPORT_LOCK="$WORKSPACE/.turngait-r12-export.lock"      # the shared lock: export_one_humanoid.sh rewrites configs/policy_latest.yaml
_release() { [ "$(cut -d' ' -f1 "$EXPORT_LOCK/holder" 2>/dev/null)" = "$JOBTAG" ] && rm -rf "$EXPORT_LOCK"; return 0; }
trap "_v60_release '$WORKSPACE/.v51-boot.lock'; _release" EXIT
export_lock() {
    local waited=0
    while ! mkdir "$EXPORT_LOCK" 2>/dev/null; do
        if [ -n "$(find "$EXPORT_LOCK" -maxdepth 0 -mmin +20 2>/dev/null)" ]; then rm -rf "$EXPORT_LOCK"; continue; fi
        sleep $(( 10 + RANDOM % 10 )); waited=$(( waited + 15 ))
        [ $waited -gt 3600 ] && return 1
    done
    echo "$JOBTAG $(hostname) $(date +%s)" > "$EXPORT_LOCK/holder"
}
if [ "$SMOKE" = 1 ]; then OVR_DIR="$RES/overrides"; else OVR_DIR="$WORKSPACE/overrides"; fi
mkdir -p "$OVR_DIR"
export OVERRIDE_FILE="$OVR_DIR/${RUN_NAME}.txt"
dr_overrides_scale 1.0 > "$OVERRIDE_FILE"
if { [ "$VARIANT" = 1c ] || [ "$VARIANT" = 1d ] || [ "$VARIANT" = 1e ] || [ "$VARIANT" = 1f ] || [ "$VARIANT" = 1g ] || [ "$VARIANT" = 1h ] || [ "$VARIANT" = 1i ]; } && [ "$SMOKE" = 1 ]; then
    # smoke only: compress the ramp into the smoke's 3 iterations, so its log shows the term move
    printf ' env.curriculum.disturbance.params.start_iter=0 env.curriculum.disturbance.params.end_iter=2' >> "$OVERRIDE_FILE"
    # phase 1h: its standing-share ramp is compressed the same way
    [ "$VARIANT" = 1h ] && printf ' env.curriculum.standing_share.params.start_iter=0 env.curriculum.standing_share.params.end_iter=2' >> "$OVERRIDE_FILE"
fi
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS RUN_DIR BHL_STD_MAX"
find_run() { ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | sort | tail -1; }
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | smoke=$SMOKE | from scratch, $MAX_ITER iters -> $FINAL_CKPT | RES $RES | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
git -C "$REPO" status --porcelain -- src/bhl_robust/waiter_asset.py src/bhl_robust/tasks/waiter_env_cfg.py \
    src/bhl_robust/tasks/waiter_wbc_mdp.py src/bhl_robust/tasks/waiter_stand_mdp.py src/bhl_robust/eval/waiter_wbc.py \
    src/bhl_robust/tasks/__init__.py "$0" || true
TLOG="${TMPDIR:-/tmp}/waiter-wbc-$JOBTAG.log"; : > "$TLOG"
existing=$(find_run || true)
if [ "$SMOKE" != 1 ] && [ -n "$existing" ] && [ -f "$existing/$FINAL_CKPT" ]; then
    echo "already trained: $existing (reused)"
    # WAITER_TLOG_REUSE=<the run's original training log>: the log-based checks below (std bound, curriculum) are
    # skipped when the log is empty; with this they run on a reused run as well (2026-10-10, phase 1e provisional path)
    if [ -n "${WAITER_TLOG_REUSE:-}" ]; then
        grep -aq "| $RUN_NAME | task=$TASK |" "$WAITER_TLOG_REUSE" || { echo "WAITER-WBC RESULT: FAIL ($RUN_NAME: WAITER_TLOG_REUSE is not this run's log)"; exit 1; }
        cp "$WAITER_TLOG_REUSE" "$TLOG"
        echo "log-based checks run on $WAITER_TLOG_REUSE"
    fi
else
    bhl_exec "$REPO/slurm/inner/train.sh" 2>&1 | tee "$TLOG" || true
    echo "=== training done: $(date) ==="
fi
run=$(find_run || true)
why=""
[ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ] || why="$why no-$FINAL_CKPT"
grep -q "waiter-wbc id NOT registered" "$TLOG" && why="$why task-not-registered"
[ "$VARIANT" = 1e ] && grep -q "waiter-wbc-stand id NOT registered" "$TLOG" && why="$why stand-task-not-registered"
[ "$VARIANT" = 1f ] && grep -q "waiter-wbc-standpush id NOT registered" "$TLOG" && why="$why standpush-task-not-registered"
[ "$VARIANT" = 1g ] && grep -q "waiter-wbc-standbonus id NOT registered" "$TLOG" && why="$why standbonus-task-not-registered"
[ "$VARIANT" = 1h ] && grep -q "waiter-wbc-standstill id NOT registered" "$TLOG" && why="$why standstill-task-not-registered"
[ "$VARIANT" = 1i ] && grep -q "waiter-wbc-standonly id NOT registered" "$TLOG" && why="$why standonly-task-not-registered"
if { [ "$VARIANT" = 1b ] || [ "$VARIANT" = 1c ] || [ "$VARIANT" = 1d ] || [ "$VARIANT" = 1e ] || [ "$VARIANT" = 1f ] || [ "$VARIANT" = 1g ] || [ "$VARIANT" = 1h ] || [ "$VARIANT" = 1i ]; } && [ "$(ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | wc -l)" -ge 1 ] && [ -s "$TLOG" ]; then
    grep -q "\[overlay\] action-noise std bounded to \[0.001, 1.0\]" "$TLOG" || why="$why std-bound-overlay-missing"
    # WAITER_STD_LOG_TOL (default 1e-6 = the check as predeclared): the log line shows the std parameter after the
    # iteration's last optimizer step, before the next clamp, so it can read up to 1.01 while every use is <= 1.0
    # (SLURM_JOBS.md 2026-10-10 "phase 1e training fails its validity check as written"). A larger tolerance is a
    # PROVISIONAL reading and is recorded in the run's training JSON.
    "$PYH" - "$TLOG" "${WAITER_STD_LOG_TOL:-1e-6}" <<'PY' || why="$why noise-std-above-1.0"
import re, sys
v = [float(x) for x in re.findall(r"Mean action noise std:\s*([0-9.]+)", open(sys.argv[1], errors="ignore").read())]
tol = float(sys.argv[2])
over = [i for i, x in enumerate(v) if x > 1.0 + 1e-6]
print(f"std check: max logged {max(v) if v else None}; {len(over)} of {len(v)} iterations above 1.0 (iterations {over[:8]}); "
      f"tolerance {tol}{'' if tol <= 1e-6 else ' (PROVISIONAL: above the predeclared 1e-6)'}")
sys.exit(0 if v and max(v) <= 1.0 + tol else 1)
PY
fi
if { [ "$VARIANT" = 1c ] || [ "$VARIANT" = 1d ] || [ "$VARIANT" = 1e ] || [ "$VARIANT" = 1f ] || [ "$VARIANT" = 1g ] || [ "$VARIANT" = 1h ] || [ "$VARIANT" = 1i ]; } && [ -n "$run" ] && [ -s "$TLOG" ]; then
    # phase 1c: the curriculum ran (logged 0 first, and 1.0 last in a full run), the PPO rollout is the 24 steps the
    # iteration count assumes, and a full run carries the frozen ramp values
    (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" - "$TLOG" "$run" "$SMOKE" <<'PY') || why="$why curriculum-check"
import re, sys
from pathlib import Path
from bhl_robust.eval.gait_clock import load_env_yaml
tlog, run, smoke = sys.argv[1], Path(sys.argv[2]), sys.argv[3] == "1"
v = [float(x) for x in re.findall(r"Curriculum/disturbance:\s*([0-9.]+)", open(tlog, errors="ignore").read())]
agent = (run / "params" / "agent.yaml").read_text()
ok = bool(v) and v == sorted(v) and re.search(r"^num_steps_per_env: 24$", agent, re.M) is not None
if smoke:
    # the compressed ramp (iterations 0-2) starts inside the first iteration's average (its last step is iteration 1),
    # so the smoke shows the term rising from near 0 to 1 (smoke 21657423 logged 0.0208, 0.5208, 1.0)
    ok = ok and v[0] < 0.5 and max(v) >= 0.999
else:
    p = load_env_yaml(run / "params" / "env.yaml")["curriculum"]["disturbance"]["params"]
    ok = ok and v[0] == 0.0 and v[-1] >= 0.999 and (p["start_iter"], p["end_iter"], p["steps_per_iter"]) == (4000, 7000, 24) \
        and list(p["force_range"]) == [-6.0, 6.0] and list(p["payload_range"]) == [0.0, 0.8]
print(f"curriculum check: {'OK' if ok else 'FAIL'} (logged {len(v)} values, first {v[0] if v else None}, "
      f"last {v[-1] if v else None}, max {max(v) if v else None})")
sys.exit(0 if ok else 1)
PY
fi
if [ "$VARIANT" = 1d ] && [ -n "$run" ] && [ -f "$run/params/env.yaml" ]; then
    # phase 1d: the stance terms and the standing-env share are in the run's own env.yaml
    s=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -c "
import sys
from bhl_robust.eval.gait_clock import load_env_yaml
from bhl_robust.eval.waiter_wbc import check_stance
p = check_stance(load_env_yaml(sys.argv[1]))
print('stance check: ' + ('OK' if not p else 'FAIL ' + '; '.join(p)))" "$run/params/env.yaml") 2>&1 | tail -1)
    echo "$s"
    case "$s" in "stance check: OK") ;; *) why="$why stance-check";; esac
fi
if [ "$VARIANT" = 1f ] && [ -n "$run" ] && [ -f "$run/params/env.yaml" ]; then
    # phase 1f: 1d's stance terms + the standing penalties + pushes (push_and_reanchor) and hand forces on every env
    s=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -c "
import sys
from bhl_robust.eval.gait_clock import load_env_yaml
from bhl_robust.eval.waiter_wbc import check_stand_push
p = check_stand_push(load_env_yaml(sys.argv[1]))
print('stand-push check: ' + ('OK' if not p else 'FAIL ' + '; '.join(p)))" "$run/params/env.yaml") 2>&1 | tail -1)
    echo "$s"
    case "$s" in "stand-push check: OK") ;; *) why="$why stand-push-check";; esac
    # the std bound must also show in the log now that the overlay clamps right after each PPO update
    grep -q "\[overlay\] action-noise std also bounded after each PPO update" "$TLOG" || [ ! -s "$TLOG" ] || why="$why std-clamp-after-update-missing"
fi
if [ "$VARIANT" = 1g ] && [ -n "$run" ] && [ -f "$run/params/env.yaml" ]; then
    # phase 1g: 1d's stance terms + pushes (push_and_reanchor) and hand forces on every env + the two bounded positive
    # standing rewards, and no standing penalty term
    s=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -c "
import sys
from bhl_robust.eval.gait_clock import load_env_yaml
from bhl_robust.eval.waiter_wbc import check_stand_bonus
p = check_stand_bonus(load_env_yaml(sys.argv[1]))
print('stand-bonus check: ' + ('OK' if not p else 'FAIL ' + '; '.join(p)))" "$run/params/env.yaml") 2>&1 | tail -1)
    echo "$s"
    case "$s" in "stand-bonus check: OK") ;; *) why="$why stand-bonus-check";; esac
    # as in 1f: the logged std is the bounded one only if the overlay clamps right after each PPO update
    grep -q "\[overlay\] action-noise std also bounded after each PPO update" "$TLOG" || [ ! -s "$TLOG" ] || why="$why std-clamp-after-update-missing"
fi
if [ "$VARIANT" = 1h ] && [ -n "$run" ] && [ -f "$run/params/env.yaml" ]; then
    # phase 1h: the held-clock stance terms, the stand-still term, the standing-share curriculum and no 1e-1g term
    s=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -c "
import sys
from bhl_robust.eval.gait_clock import load_env_yaml
from bhl_robust.eval.waiter_wbc import check_stand_still
p = check_stand_still(load_env_yaml(sys.argv[1]), window=sys.argv[2] != '1')   # a smoke compresses the window
print('stand-still check: ' + ('OK' if not p else 'FAIL ' + '; '.join(p)))" "$run/params/env.yaml" "$SMOKE") 2>&1 | tail -1)
    echo "$s"
    case "$s" in "stand-still check: OK") ;; *) why="$why stand-still-check";; esac
    grep -q "\[overlay\] action-noise std also bounded after each PPO update" "$TLOG" || [ ! -s "$TLOG" ] || why="$why std-clamp-after-update-missing"
    if [ -s "$TLOG" ]; then
        # the standing share was logged rising from 0.02 to 0.25 (a smoke: within its 3 iterations)
        "$PYH" - "$TLOG" <<'PY' || why="$why standing-share-check"
import re, sys
v = [float(x) for x in re.findall(r"Curriculum/standing_share:\s*([0-9.]+)", open(sys.argv[1], errors="ignore").read())]
ok = bool(v) and v == sorted(v) and v[0] < 0.25 and abs(max(v) - 0.25) < 1e-6 and min(v) >= 0.02 - 1e-6
print(f"standing-share check: {'OK' if ok else 'FAIL'} (logged {len(v)} values, first {v[0] if v else None}, last {v[-1] if v else None})")
sys.exit(0 if ok else 1)
PY
    fi
fi
if [ "$VARIANT" = 1i ] && [ -n "$run" ] && [ -f "$run/params/env.yaml" ]; then
    # phase 1i: every env standing, the held-clock stance terms, 1g's bonuses + 1h's stand-still term, no penalty
    s=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -c "
import sys
from bhl_robust.eval.gait_clock import load_env_yaml
from bhl_robust.eval.waiter_wbc import check_stand_only
p = check_stand_only(load_env_yaml(sys.argv[1]))
print('stand-only check: ' + ('OK' if not p else 'FAIL ' + '; '.join(p)))" "$run/params/env.yaml") 2>&1 | tail -1)
    echo "$s"
    case "$s" in "stand-only check: OK") ;; *) why="$why stand-only-check";; esac
    grep -q "\[overlay\] action-noise std also bounded after each PPO update" "$TLOG" || [ ! -s "$TLOG" ] || why="$why std-clamp-after-update-missing"
fi
if [ "$VARIANT" = 1e ] && [ -n "$run" ] && [ -f "$run/params/env.yaml" ]; then
    # phase 1e: 1d's stance terms + no disturbances on standing envs + the standing penalties, in the run's env.yaml
    s=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -c "
import sys
from bhl_robust.eval.gait_clock import load_env_yaml
from bhl_robust.eval.waiter_wbc import check_stand
p = check_stand(load_env_yaml(sys.argv[1]))
print('stand check: ' + ('OK' if not p else 'FAIL ' + '; '.join(p)))" "$run/params/env.yaml") 2>&1 | tail -1)
    echo "$s"
    case "$s" in "stand check: OK") ;; *) why="$why stand-check";; esac
fi
if [ -z "$why" ]; then
    "$PYH" - "$RES/training/$RUN_NAME.json" "$RUN_NAME" "$TASK_ID" "$SEED" "$FINAL_CKPT" "$run" "$JOBTAG" "$TLOG" <<'PY'
import json, os, re, sys
out, run_name, task, seed, ckpt, run, job, tlog = sys.argv[1:]
log = open(tlog, errors="ignore").read()
its = [int(m) for m in re.findall(r"Learning iteration (\d+)/", log)]
std = [float(x) for x in re.findall(r"Mean action noise std:\s*([0-9.]+)", log)]
tol = float(os.environ.get("WAITER_STD_LOG_TOL", "1e-6"))
json.dump({"run": run_name, "task": task, "seed": int(seed), "final_ckpt": ckpt, "run_dir": run, "job": job,
           "logged_iterations": (max(its) + 1) if its else None, "max_logged_std": max(std) if std else None,
           "std_log_tol": tol, "provisional": tol > 1e-6, "training_log_reused": os.environ.get("WAITER_TLOG_REUSE")},
          open(out, "w"), indent=1)
PY
    cp "$TLOG" "$RES/training/$RUN_NAME.$JOBTAG.log" 2>/dev/null || true
    export RUN_DIR=$(basename "$run")
    cfg="$run/exported/deploy.yaml"
    if [ ! -f "$cfg" ]; then
        if export_lock; then
            echo "=== export $RUN_DIR $(date) ==="
            TASK="$TASK_ID" bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh" 2>&1 | grep -vE "pthread_setaffinity|EGLError" || true
            _release
        else why="$why export-lock"; fi
    fi
    if [ -f "$cfg" ] && [ -f "$run/exported/policy.onnx" ]; then
        w=$("$PYH" -c "import onnxruntime as o, sys; s=o.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']); print(s.get_inputs()[0].shape[-1], s.get_outputs()[0].shape[-1])" "$run/exported/policy.onnx" 2>/dev/null || echo "?")
        [ "$w" = "$OBS_POLICY 12" ] || why="$why onnx-io-$w"
        s=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -m bhl_robust.eval.waiter_wbc stamp --deploy "$cfg" --env-yaml "$run/params/env.yaml" --task "$TASK_ID" 2>&1 | tail -1) || true)
        echo "$s"
        case "$s" in "WAITER-WBC: stamped"*|"WAITER-WBC: already stamped"*) ;; *) why="$why stamp";; esac
        cp "$cfg" "$RES/training/$RUN_NAME.deploy.yaml"
    else
        why="$why no-deploy.yaml-or-policy.onnx"
    fi
fi
if [ -n "$why" ]; then
    echo "WAITER-WBC RESULT: FAIL ($RUN_NAME:$why)"
    [ "$SMOKE" = 1 ] && echo "WAITER-WBC SMOKE: FAIL"
    exit 1
fi
echo "WAITER-WBC RESULT: TRAINED + EXPORTED ($RUN_NAME, $run)"
[ "$SMOKE" = 1 ] && echo "WAITER-WBC SMOKE: PASS ($RUN_NAME: $MAX_ITER iterations, export 83 -> 12, stamped)"
echo "=== done $(date) ==="
