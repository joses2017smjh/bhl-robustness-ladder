#!/bin/bash
#SBATCH --job-name=turngait-hold
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=0-2%3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-10-02: turning follow-up R1H for the 22-DoF humanoid with arms (SLURM_JOBS.md "(C') revised design",
# 2026-10-02 14:44, frozen before any implementation; the user's approval of 2026-10-02 14:15: "keep the clock and
# add a heading-hold reward"). Built from gpu_turngait_r12.sbatch, this arm only.
# Why: R1 (Velocity-BHL-Arms-TurnGaitClock-v0, gait clock in the actor) turns in place on every seed (18/18 v2 turns)
# but FAILS by the v5 joint rule on straight-walk heading drift (1/3 seeds count; results/repo-gpu-20260923/
# turngait-r12-20261001/). R1 trains in upstream's heading mode in EVERY env (heading_command=True,
# rel_heading_envs=1.0: wz = clip(0.5 * wrap(heading_target - yaw), +/-1.5), recomputed every step), so it never
# trains on the open-loop wz = 0 that the gate's walk test commands ((0.35, 0, 0), 1 s warm-up + 6 s).
#
# ONE NEW task, Velocity-BHL-Arms-TurnGaitClockHold-v0 (arms_env_cfg.HumanoidTurnGaitClockHoldCfg), trained FROM
# SCRATCH (no parent checkpoint; never a fine-tune of R1), 6000 iterations, v51, R1's DR (scale 1.0 via dr_levels.sh)
# and R1's runner. R1H = R1 EXACTLY + these two changes ONLY, both chosen, not tuned:
#   (i)  command mix (turn_command.TurnHoldMixVelocityCommand, the repo's TurnMixVelocityCommand DIRECT mechanism):
#        per resample, 30 % of envs leave heading control (is_heading_env cleared) and take EXPLICIT commands: vx and
#        vy from R1's own lin_vel ranges (-1.0, 1.0) / (-0.5, 0.5); wz = 0 EXACTLY with probability 0.5, otherwise
#        uniform over R1's own ang_vel_z range (-1.5, 1.5). The other 70 % keep R1's heading mode unchanged. No
#        PURE_TURN envs. The 2 % standing behaviour stays as upstream (a standing env is zeroed whatever its mode).
#   (ii) reward heading_hold (gait_clock_mdp.heading_hold), weight 1.0: active only in those explicit envs while
#        |wz_cmd| < 0.05; reward = exp(-(dpsi / 0.2 rad)^2), dpsi = wrap(base yaw - psi_ref), psi_ref = that env's
#        base yaw at its last command resample (and at episode reset). The resample is detected by the command term
#        itself: TurnHoldMixVelocityCommand._resample_command, which CommandTerm._resample calls for the time-out
#        resample and for the episode-reset resample (after the reset events wrote the new root pose), stores
#        heading_w as yaw_at_resample; the reward reads it. Zero elsewhere.
#        Scale: R1's track_ang_vel_z_exp has weight 2.0 (std 0.25 rad/s), so heading_hold's ceiling (1.0 per step) is
#        half the yaw-rate term's, and it is active in about 16 % of envs (0.30 x 0.53).
#   No other change: rewards, observations (the 77-obs clock), events (the fixed pushes), DR and runner are R1's.
# ARM TABLE (array index -> arm, task, run names, actor observations):
#   0-2  R1H  Velocity-BHL-Arms-TurnGaitClockHold-v0  arms-turngait-hold-s{0,1,2}  actor 77 = 75 + sin/cos(2 pi phase_left), critic 80
# Per seed, in the job: training -> export with the arm's OWN task (77 wide) -> the gait_clock block stamped into
# deploy.yaml from the run's own env.yaml -> turn_test v2 (gating) -> the non-gating MuJoCo diagnosis ->
# cpu_turn_qualify.sbatch (unchanged; its v2x and run_eval paths build the clock-aware controller through
# bhl_robust.eval.gait_clock) -> the arm verdict (scripts/bench/turngait_hold_verdict.py).
#
# PREDECLARED RULE (frozen 2026-10-02, before any R1H run exists; v5's joint rule verbatim, SLURM_JOBS.md line 3055):
#   PASS iff >= 2/3 seeds both PASS turn_test v2 AND are QUALIFIED by cpu_turn_qualify's unchanged rule (v2x >= 9/10
#   on reset seeds 10-14, walk <= 15 deg on >= 2/3, push <= 9/60); else FAIL; INCOMPLETE if any JSON is missing. A
#   seed that fails v2 does not count even if it qualifies through v2x.
#   Training counts only if the run dir holds model_5999.pt and the training log shows the arm's task id, the
#   feet_gait term, the heading_hold term, the explicit-command mix and the push event.
#   A new task, never presented as a fine-tune of R1.
#   Labels: LEARNED gait; MuJoCo gates.
# Disclosure: the arm changes the command mix (the training distribution), so it tests explicit straight-walk
# commands + the hold reward together, not the reward alone.
# Disclosed priors: R1's drift is not a constant bias (signs and sizes vary with the reset seed; s1 -37 to -54 deg
# over 7 s); the 0.2 rad kernel, the 0.05 threshold, the weight 1.0, the 30 % / 0.5 mix are chosen, not tuned, and
# the gate (turn_test v2, cpu_turn_qualify) is unchanged.
#
# Verdicts come from JSON, never exit codes (Isaac exits 0 on a crash). Per seed:
#   <RES>/training/<run>.json        written once from the FRESH training log (frozen clause, six flags)
#   <RES>/turn-test-v2/<run>.json    turn_test.py --protocol v2 (never overwritten)
#   <RES>/diagnose/<run>.json        turn_diagnose.py mujoco (reported, not gated)
#   <RES>/qualify/<run>__qualify.json  cpu_turn_qualify.sbatch (never overwritten)
#   <RES>/verdict/R1H.json           written once, by the arm's last seed (O_EXCL)
# RES = results/repo-gpu-20260923/turngait-hold-20261002. An existing run dir with model_5999.pt is reused, never
# retrained or overwritten; an existing verdict is re-printed and the job exits.
# Wrong-arm guard on the fresh training log (beyond the frozen clause): the policy / critic observation widths (77 /
# 80), the command class TurnHoldMixVelocityCommand (and its frozen config line), the heading_hold (1.0),
# feet_swing_height (-20.0) and feet_air_time (0.0) rows, no push_levels curriculum, the gait_clock_mdp terms and
# heading_hold having run, and the run's env.yaml = R1's (arms-turngait-clock-s0's params/env.yaml) + the two
# declared changes only (turngait_hold_verdict.py check-recipe). Its reasons are stored in the training record
# (wrong_arm_guard). Before any export or gate, every non-smoke job -- fresh, or a resubmit that reuses the trained
# run dir and so has no fresh log -- re-reads that record and runs no gate unless it is this run dir's, holds the
# frozen clause's six flags and has an empty wrong_arm_guard. (The verdict's training clause is unchanged.)
#
# Modes (env vars):
#   TURNGAIT_SMOKE=1  MAX_ITER=3, NUM_ENVS=64, run arms-turngait-hold-s<seed>-smoke, outputs under
#                     /nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/campaign-turning-hold/turngait-hold-smoke
#                     (the dir must end in -smoke). Trains, checks the log and env.yaml (record check included),
#                     runs an Isaac probe on a fresh 64-env R1H env (heading_hold in the reward manager with weight
#                     1.0; the command term is the mix over >= 19,200 forced resamples: about 30 % explicit envs,
#                     about half of them at wz = 0, is_heading_env cleared exactly on them; psi_ref = the yaw at reset
#                     and at a resample; the live term = the formula), exports with the arm's task, checks the
#                     deploy (77 + stamped clock), then a short clock-aware MuJoCo rollout (python -m
#                     bhl_robust.eval.gait_clock smoke) and short runs of turn_test v1, turn_diagnose mujoco and
#                     run_eval through the same deploy (plumbing: readable output required, verdicts ignored), and
#                     the verdict script. Every smoke rollout runs on exploration seed 100, never on a scored seed
#                     (v2 reset 0-2 / walk 0, v2x 10-14 / 10-12, push eval 0-9).
#                     One "SMOKE: PASS|FAIL (...)" line; exit 0 iff PASS (chain afterok).
# Side effects outside RES, as for every humanoid training and export: the run dir <ts>_<run> in
# $UPSTREAM/logs/rsl_rl/humanoid (a smoke run's name ends in -smoke), Isaac's log
# $UPSTREAM/logs/rsl_rl/humanoid/isaaclab/isaaclab_<ts>.log, and the export's rewrite of
# $UPSTREAM/configs/policy_latest.yaml (serialised by the export lock shared with gpu_turngait_r12.sbatch and
# gpu_platecross.sbatch).
#   Submit (coordinator) in ONE command: a fresh smoke of the exact code being submitted, with the real array
#   chained afterok on it. Never chain on an older smoke id: slurmctld forgets a finished job after ~5 min
#   (MinJobAge 300 s), and the code may have changed.
#     cd /nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder && source slurm/_env.sh && \
#     S=$(slurm_clean sbatch --parsable --job-name=turngait-hold-smoke --array=0 --time=01:00:00 \
#         --export=ALL,TURNGAIT_SMOKE=1 slurm/repo20260923/gpu_turngait_hold.sbatch) && \
#     slurm_clean sbatch --dependency=afterok:${S%%;*} --export=ALL,TURNGAIT_SMOKE=0 \
#         slurm/repo20260923/gpu_turngait_hold.sbatch
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
PYH=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid
JOBTAG=${SLURM_JOB_ID:-local$$}
SMOKE=${TURNGAIT_SMOKE:-0}
FULL_ITERS=6000
SMOKE_RES=/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/campaign-turning-hold/turngait-hold-smoke
if [ "$SMOKE" = 1 ]; then
    RES=${TURNGAIT_OUT_DIR:-$SMOKE_RES}
    case "$RES" in *-smoke) ;; *) echo "SMOKE: FAIL (smoke outputs must go to a directory ending in -smoke, not $RES)"; exit 1;; esac
else
    RES=$REPO/results/repo-gpu-20260923/turngait-hold-20261002
fi
IDX=${SLURM_ARRAY_TASK_ID:-0}
case $IDX in
    0|1|2) ARM=R1H; ARM_TASK=Velocity-BHL-Arms-TurnGaitClockHold-v0; PREFIX=arms-turngait-hold; OBS_POLICY=77; OBS_CRITIC=80; EXPECT=clock;;
    *) echo "TURNGAIT-HOLD RESULT: FAIL (array index $IDX outside 0-2)"; exit 2;;
esac
SEED=$((IDX % 3))
if [ "$SMOKE" = 1 ]; then
    export MAX_ITER=3 NUM_ENVS=64 RUN_NAME="$PREFIX-s${SEED}-smoke"
else
    export MAX_ITER=$FULL_ITERS RUN_NAME="$PREFIX-s${SEED}"
    unset NUM_ENVS                                  # the task's own env count, as R1 trained
fi
FINAL_CKPT="model_$((MAX_ITER - 1)).pt"
export TASK="$ARM_TASK" SEED
mkdir -p "$RES"/{training,turn-test-v2,diagnose,qualify,verdict}
CACHE="${TMPDIR:-/tmp}/turngait-hold-$JOBTAG"
# The recipe reference: R1's own trained run (the -s0-smoke dirs do not match this glob).
REF_RUN=$(ls -d "$LOGROOT"/*_arms-turngait-clock-s0 2>/dev/null | sort | tail -1 || true)

if [ "$SMOKE" != 1 ] && [ -f "$RES/verdict/$ARM.json" ]; then
    echo "verdict already recorded, not overwritten: $RES/verdict/$ARM.json"
    "$PYH" -c "import json,sys; d=json.load(open(sys.argv[1])); print('ARM-HOLD', d['arm'], d['verdict'], d['n_counted'], '/ 3')" "$RES/verdict/$ARM.json"
    exit 0
fi

# ---------------------------------------------------------------- GPU from here
# From scratch: a recurrent / RND / symmetry overlay forwarded from the submitting shell
# (BHL_FORWARD_VARS) would change the network and is not part of the recipe.
unset BHL_POLICY BHL_RND BHL_SYMMETRY
source "$REPO/slurm/dr_levels.sh"
setup_node_cache
CACHE="${TMPDIR:-/tmp}/turngait-hold-$JOBTAG"
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate

# Exports serialise on one lock, shared with gpu_turngait_r12.sbatch and gpu_platecross.sbatch:
# export_one_humanoid.sh copies the shared $UPSTREAM/configs/policy_latest.yaml, which two jobs exporting
# together would race on (mkdir is atomic on Lustre/NFS; stale after 20 min > the 16-min export cap).
EXPORT_LOCK="$WORKSPACE/.turngait-r12-export.lock"
_hold_export_release() {
    [ "$(cut -d' ' -f1 "$EXPORT_LOCK/holder" 2>/dev/null)" = "$JOBTAG" ] && rm -rf "$EXPORT_LOCK"
    return 0
}
trap "_v60_release '$WORKSPACE/.v51-boot.lock'; _hold_export_release" EXIT
export_lock() {
    local waited=0
    while ! mkdir "$EXPORT_LOCK" 2>/dev/null; do
        if [ -n "$(find "$EXPORT_LOCK" -maxdepth 0 -mmin +20 2>/dev/null)" ]; then
            echo "export lock: clearing stale lock ($(cat "$EXPORT_LOCK/holder" 2>/dev/null || true))"
            rm -rf "$EXPORT_LOCK"; continue
        fi
        sleep $(( 10 + RANDOM % 10 )); waited=$(( waited + 15 ))
        [ $waited -gt 3600 ] && { echo "export lock: not acquired in an hour"; return 1; }
    done
    echo "$JOBTAG $(hostname) $(date +%s)" > "$EXPORT_LOCK/holder"
    echo "export lock: acquired after ~${waited}s"
}

if [ "$SMOKE" = 1 ]; then OVR_DIR="$RES/overrides"; else OVR_DIR="$WORKSPACE/overrides"; fi
mkdir -p "$OVR_DIR"
export OVERRIDE_FILE="$OVR_DIR/${RUN_NAME}.txt"
dr_overrides_scale 1.0 > "$OVERRIDE_FILE"
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS RUN_DIR"
find_run() { ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | sort | tail -1; }
echo "=== node: $(hostname) | $ARM $RUN_NAME | task=$TASK | smoke=$SMOKE | from scratch, $MAX_ITER iters -> $FINAL_CKPT | RES $RES | ref R1 ${REF_RUN:-none} | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
cat "$OVERRIDE_FILE"
TLOG="${TMPDIR:-/tmp}/turngait-hold-$JOBTAG.log"; : > "$TLOG"
existing=$(find_run || true)
FRESH=0
if [ "$SMOKE" != 1 ] && [ -n "$existing" ] && [ -f "$existing/$FINAL_CKPT" ]; then
    echo "already trained: $existing (reused; no fresh training log in this job)"
else
    [ -n "$existing" ] && echo "WARNING: an incomplete run dir exists ($existing); training anew. cpu_turn_qualify needs exactly one run dir per name, so move the partial one aside before qualifying."
    bhl_exec "$REPO/slurm/inner/train.sh" 2>&1 | tee "$TLOG" || true
    FRESH=1
    echo "=== training done: $(date) ==="
fi
run=$(find_run || true)

# ---------------------------------------------------------------- training evidence
# The frozen clause (training JSON, six flags) and the wrong-arm guard, both from the FRESH log.
TASK_RE="^task=$ARM_TASK run=$RUN_NAME seed=$SEED "
GAIT_RE="\| +[0-9]+ +\| +feet_gait +\| +0\.5 +\|"
HOLD_RE="\| +[0-9]+ +\| +heading_hold +\| +1\.0 +\|"
MIXROW_RE="\| +base_velocity +\| +TurnHoldMixVelocityCommand +\|"
MIX_RE="^\[turn_command\] TurnHoldMixVelocityCommand: explicit 0\.3 zero_wz_p 0\.5 pure 0\.0 vx \(-1\.0, 1\.0\) vy \(-0\.5, 0\.5\) wz \(-1\.5, 1\.5\) heading_command True rel_heading 1\.0 stiffness 0\.5 rel_standing 0\.02$"
PUSH_RE="\| +[0-9]+ +\| +push_robot +\| +\(5\.0, 9\.0\) +\|"
arm_guard() {   # <log>: reasons the log is NOT this arm's recipe (empty = ok)
    grep -qE "Group: 'policy' \(shape: \($OBS_POLICY,\)\)" "$1" || printf ' policy-obs-not-%s' "$OBS_POLICY"
    grep -qE "Group: 'critic' \(shape: \($OBS_CRITIC,\)\)" "$1" || printf ' critic-obs-not-%s' "$OBS_CRITIC"
    grep -qE "\| +[0-9]+ +\| +feet_swing_height +\| +-20\.0 +\|" "$1" || printf ' no-feet_swing_height--20.0-row'
    grep -qE "\| +[0-9]+ +\| +feet_air_time +\| +0\.0 +\|" "$1" || printf ' feet_air_time-weight-not-0.0'
    grep -qE "$MIXROW_RE" "$1" || printf ' command-class-is-not-TurnHoldMixVelocityCommand'
    grep -qE "\| +base_velocity +\| +UniformVelocityCommand +\|" "$1" && printf ' command-class-is-UniformVelocityCommand'
    grep -qE "^\[turn_command\] TurnHoldMix first resample: n [0-9]+ .* heading_cleared_iff_explicit True " "$1" \
        || printf ' command-mix-did-not-resample-as-declared'
    grep -q "push_levels" "$1" && printf ' unexpected-push_levels-curriculum'
    grep -qE "^\[gait_clock_mdp\] feet_gait bodies \['[^']*left[^']*', '[^']*right[^']*'\] offsets \[0\.0, 0\.5\]" "$1" || printf ' feet_gait-did-not-run-on-(left,right)'
    grep -qE "^\[gait_clock_mdp\] feet_swing_height bodies" "$1" || printf ' feet_swing_height-did-not-run'
    grep -qE "^\[gait_clock_mdp\] heading_hold command base_velocity \(TurnHoldMixVelocityCommand\) std 0\.2 wz_threshold 0\.05: active in [0-9]+ of [0-9]+ envs" "$1" \
        || printf ' heading_hold-did-not-run'
    grep -qE "train.sh: reached [0-9]+ logged iterations" "$1" || printf ' no-logged-iterations'
    return 0
}
# training_record_problems <record JSON> <run dir>: reasons the stored training record does not
# let this seed's gates run (empty = ok): it must be this run's and this run dir's, name
# $FINAL_CKPT, hold the frozen clause's six flags and an EMPTY wrong_arm_guard list. A missing,
# unreadable or crashing check is a reason too. Always returns 0 (safe in $(...) under set -e).
training_record_problems() {
    "$PYH" - "$1" "$2" "$RUN_NAME" "$ARM_TASK" "$FINAL_CKPT" <<'EOF' || printf ' training-record-check-crashed'
import json, sys
path, run_dir, run, task, ckpt = sys.argv[1:6]
try:
    with open(path) as f:
        t = json.load(f)
    assert isinstance(t, dict), "not a JSON object"
except Exception as exc:  # noqa: BLE001
    print(f" unreadable-training-record({type(exc).__name__})")
    sys.exit(0)
why = []
if t.get("run") != run: why.append(f"record-run-not-{run}")
if t.get("task") != task: why.append(f"record-task-not-{task}")
if t.get("run_dir") != run_dir: why.append("record-is-for-another-run-dir")
if t.get("final_ckpt") != ckpt: why.append(f"record-final_ckpt-not-{ckpt}")
for k in ("final_ckpt_present", "task_id_in_log", "feet_gait_in_log", "heading_hold_in_log", "command_mix_in_log",
          "push_robot_in_log"):
    if t.get(k) is not True: why.append(f"record-{k}-not-true")
g = t.get("wrong_arm_guard")
if not isinstance(g, list): why.append("record-has-no-wrong_arm_guard-list")
elif g: why.append("record-wrong_arm_guard:" + ",".join(str(x) for x in g))
print("".join(" " + w for w in why))
EOF
    return 0
}
TJSON="$RES/training/$RUN_NAME.json"
if [ "$SMOKE" = 1 ]; then TJSON="$RES/training/$RUN_NAME.$JOBTAG.json"; fi   # each smoke writes and re-checks its own
why_train=""
# The run's own env.yaml = R1's (arms-turngait-clock-s0's params/env.yaml) + the two declared changes only.
rline="HOLD RECIPE: not checked (no env.yaml or no arms-turngait-clock-s0 reference)"
if [ -n "$run" ] && [ -f "$run/params/env.yaml" ] && [ -n "$REF_RUN" ] && [ -f "$REF_RUN/params/env.yaml" ]; then
    rline=$( (cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 "$PYH" scripts/bench/turngait_hold_verdict.py check-recipe \
        --env-yaml "$run/params/env.yaml" --ref-env-yaml "$REF_RUN/params/env.yaml" 2>&1 | tail -1) || true)
fi
echo "$rline"
case "$rline" in "HOLD RECIPE: OK"*) ;; *) why_train="$why_train recipe-check-failed";; esac
if [ "$FRESH" = 1 ]; then
    ckpt_ok=false; { [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; } && ckpt_ok=true
    task_ok=false; grep -qE "$TASK_RE" "$TLOG" && task_ok=true
    gait_ok=false; grep -qE "$GAIT_RE" "$TLOG" && gait_ok=true
    hold_ok=false; grep -qE "$HOLD_RE" "$TLOG" && hold_ok=true
    mix_ok=false; grep -qE "$MIXROW_RE" "$TLOG" && grep -qE "$MIX_RE" "$TLOG" && mix_ok=true
    push_ok=false; grep -qE "$PUSH_RE" "$TLOG" && push_ok=true
    why_train="$why_train$(arm_guard "$TLOG")"
    [ "$ckpt_ok" = true ] || why_train="$why_train no-$FINAL_CKPT"
    [ "$task_ok" = true ] || why_train="$why_train task-id-not-in-log"
    [ "$gait_ok" = true ] || why_train="$why_train no-feet_gait-0.5-row"
    [ "$hold_ok" = true ] || why_train="$why_train no-heading_hold-1.0-row"
    [ "$mix_ok" = true ] || why_train="$why_train no-explicit-command-mix-in-log"
    [ "$push_ok" = true ] || why_train="$why_train no-push_robot-(5.0,9.0)-row"
    { head -n 400 "$TLOG"; echo "[... cut ...]"; tail -n 60 "$TLOG"; } > "$RES/training/$RUN_NAME.$JOBTAG.log-excerpt.txt"
    grep -hE "^\[gait_clock_mdp\]|^\[turn_command\]" "$TLOG" | head -6 || true
    if [ "$ckpt_ok" = true ] && [ ! -e "$TJSON" ]; then
        "$PYH" - "$TJSON" "$RUN_NAME" "$ARM" "$ARM_TASK" "$SEED" "$FINAL_CKPT" "$run" "$JOBTAG" \
            "$ckpt_ok" "$task_ok" "$gait_ok" "$hold_ok" "$mix_ok" "$push_ok" "$why_train" "$rline" <<'EOF'
import json, os, sys
(out, run, arm, task, seed, ckpt, run_dir, job, ckpt_ok, task_ok, gait_ok, hold_ok, mix_ok, push_ok, why,
 recipe) = sys.argv[1:17]
b = lambda s: s == "true"  # noqa: E731
rec = {"run": run, "arm": arm, "task": task, "seed": int(seed), "final_ckpt": ckpt, "final_ckpt_present": b(ckpt_ok),
       "task_id_in_log": b(task_ok), "feet_gait_in_log": b(gait_ok), "heading_hold_in_log": b(hold_ok),
       "command_mix_in_log": b(mix_ok), "push_robot_in_log": b(push_ok),
       "run_dir": run_dir, "job": job, "wrong_arm_guard": why.split(), "recipe_check": recipe,
       "clause": "Training counts only if the run dir holds model_5999.pt and the training log shows the arm's task id, "
                 "the feet_gait term, the heading_hold term, the explicit-command mix and the push event.",
       "note": "from scratch (no parent checkpoint); a new task, never a fine-tune of R1; written once from this "
               "job's fresh training log"}
tmp = f"{out}.{job}.tmp"
with open(tmp, "w") as f:
    f.write(json.dumps(rec, indent=2) + "\n")
try:
    os.link(tmp, out)                    # never overwrite an existing training record
    print(f"wrote {out}")
except FileExistsError:
    print(f"training record already exists, not overwritten: {out}")
os.unlink(tmp)
EOF
    elif [ "$ckpt_ok" != true ]; then
        echo "training incomplete: no training record written (it is written only for a run that reached $FINAL_CKPT)"
    fi
fi

if [ "$SMOKE" != 1 ]; then
    { [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; } || {
        echo "TURNGAIT-HOLD RESULT: FAIL (training incomplete: no $FINAL_CKPT for $RUN_NAME; seed stays INCOMPLETE)"; exit 1; }
    if [ "$FRESH" = 1 ] && [ -n "${why_train// /}" ]; then
        echo "TURNGAIT-HOLD RESULT: FAIL (training log / env.yaml is not the $ARM recipe:$why_train; no gates run, seed stays INCOMPLETE)"; exit 1
    fi
    if [ "$FRESH" != 1 ]; then
        [ -f "$TJSON" ] || { echo "TURNGAIT-HOLD RESULT: FAIL (already trained but no training record $TJSON and no fresh log; seed stays INCOMPLETE)"; exit 1; }
        case "$why_train" in *recipe-check-failed*) echo "TURNGAIT-HOLD RESULT: FAIL ($why_train; no gates run, seed stays INCOMPLETE)"; exit 1;; esac
    fi
    # Fresh or resubmitted (a resubmit has no fresh log): the stored record must be this run
    # dir's, hold the frozen clause's six flags and an empty wrong_arm_guard, or no gate runs.
    why_rec=$(training_record_problems "$TJSON" "$run")
    if [ -n "${why_rec// /}" ]; then
        echo "TURNGAIT-HOLD RESULT: FAIL (training record $TJSON:$why_rec; no gates run, seed stays INCOMPLETE)"; exit 1
    fi
fi

# ---------------------------------------------------------------- smoke: Isaac probe of the mix and heading_hold
# A fresh process with ONE env (a second ManagerBasedRLEnv in one process hangs Isaac Sim; see
# scripts/bench/crew_validate.py). The probe source is written under the -smoke results dir; its verdict is the
# JSON it writes before SimulationApp.close() (which hard-exits the interpreter), never its exit code.
probe_why=""
if [ "$SMOKE" = 1 ]; then
    PDIR="$RES/isaac-probe"; mkdir -p "$PDIR"
    PROBE_PY="$PDIR/hold_probe.$JOBTAG.py"; PROBE_SH="$PDIR/hold_probe.$JOBTAG.sh"; PROBE_OUT="$PDIR/hold_probe.$JOBTAG.json"
    cat > "$PROBE_PY" <<'EOF'
"""R1H smoke probe (Isaac, written by gpu_turngait_hold.sbatch): the command mix and heading_hold on a live env."""
import argparse
import json
import math
from isaaclab.app import AppLauncher

p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--num_envs", type=int, default=64)
p.add_argument("--draws", type=int, default=300)
p.add_argument("--out", required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args()
a.headless = True
app = AppLauncher(a).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import berkeley_humanoid_lite.tasks  # noqa: F401,E402
import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.tasks import gait_clock_mdp as gm  # noqa: E402
from bhl_robust.tasks.turn_command import DIRECT, PURE_TURN  # noqa: E402

R1_TERMS = ["track_lin_vel_xy_exp", "track_ang_vel_z_exp", "termination_penalty", "lin_vel_z_l2", "ang_vel_xy_l2",
            "flat_orientation_l2", "action_rate_l2", "dof_torques_l2", "dof_acc_l2", "dof_pos_limits", "feet_air_time",
            "feet_slide", "undesired_contacts", "joint_deviation_hip", "joint_deviation_ankle_roll",
            "joint_deviation_shoulder", "joint_deviation_elbow", "feet_gait", "feet_swing_height"]
checks, rec = [], {"task": a.task, "num_envs": a.num_envs, "draws": a.draws}


def check(ok, msg):
    checks.append({"ok": bool(ok), "msg": msg})
    print(("  PASS " if ok else "  FAIL ") + msg, flush=True)


try:
    cfg = parse_env_cfg(a.task, device="cuda:0", num_envs=a.num_envs)
    env = gym.make(a.task, cfg=cfg)
    u = env.unwrapped
    dev = u.device
    n = u.num_envs
    env.reset()
    term = u.command_manager.get_term("base_velocity")
    robot = term.robot
    check(type(term).__name__ == "TurnHoldMixVelocityCommand", f"command term class {type(term).__name__}")
    rm = u.reward_manager
    names = list(rm.active_terms)
    check(names == R1_TERMS + ["heading_hold"], f"reward terms = R1's 19 in order + heading_hold last: {names}")
    hc = rm.get_term_cfg("heading_hold")
    check(float(hc.weight) == 1.0, f"heading_hold weight {hc.weight}")
    check(dict(hc.params) == {"command_name": "base_velocity", "std": 0.2, "wz_threshold": 0.05},
          f"heading_hold params {dict(hc.params)}")
    check(getattr(hc.func, "__name__", "") == "heading_hold", f"heading_hold func {hc.func}")
    tav = rm.get_term_cfg("track_ang_vel_z_exp")
    rec["track_ang_vel_z_exp"] = {"weight": float(tav.weight), "std": float(tav.params.get("std"))}
    check(float(tav.weight) == 2.0 and float(tav.params.get("std")) == 0.25,
          f"R1's track_ang_vel_z_exp unchanged: weight {tav.weight} std {tav.params.get('std')} (heading_hold ceiling = half)")
    c = term.cfg
    check(float(c.rel_direct_envs) == 0.3 and float(c.rel_pure_turn_envs) == 0.0 and float(c.direct_zero_wz_prob) == 0.5
          and bool(c.heading_command) and float(c.rel_heading_envs) == 1.0 and float(c.rel_standing_envs) == 0.02,
          f"command cfg: explicit {c.rel_direct_envs} pure {c.rel_pure_turn_envs} zero-wz p {c.direct_zero_wz_prob} "
          f"heading_command {c.heading_command} rel_heading {c.rel_heading_envs} standing {c.rel_standing_envs}")
    check(tuple(c.direct_lin_vel_x) == tuple(c.ranges.lin_vel_x) and tuple(c.direct_lin_vel_y) == tuple(c.ranges.lin_vel_y)
          and tuple(c.direct_ang_vel_z) == tuple(c.ranges.ang_vel_z),
          f"explicit ranges = the task's own: vx {c.direct_lin_vel_x} vy {c.direct_lin_vel_y} wz {c.direct_ang_vel_z}")
    # psi_ref = the yaw at the episode-reset resample (env.reset() reset every env)
    err = float((term.yaw_at_resample - robot.data.heading_w).abs().max())
    check(err <= 1e-5, f"after env.reset(): psi_ref == heading_w in every env (max |diff| {err:.2e})")
    act = torch.zeros(env.action_space.shape, device=dev)
    for _ in range(5):
        env.step(act)
    ids = torch.arange(0, n, 4, device=dev)
    yaw_before = robot.data.heading_w[ids].clone()
    u._reset_idx(ids)                               # what ManagerBasedRLEnv.step does for terminated envs
    yaw_after = robot.data.heading_w[ids]
    err = float((term.yaw_at_resample[ids] - yaw_after).abs().max())
    moved = int(((yaw_after - yaw_before).abs() > 1e-3).sum())
    rec["reset_yaw_changed"] = f"{moved}/{ids.numel()}"
    check(err <= 1e-5, f"after a partial episode reset: psi_ref == the new episode's heading_w (max |diff| {err:.2e}; "
                       f"yaw changed by the reset in {moved}/{ids.numel()} envs)")
    env.step(act)
    ids2 = torch.arange(1, n, 4, device=dev)
    cnt = term.command_counter[ids2].clone()
    term._resample(ids2)                            # what CommandTerm.compute does at a time-out
    err = float((term.yaw_at_resample[ids2] - robot.data.heading_w[ids2]).abs().max())
    check(err <= 1e-5 and bool((term.command_counter[ids2] == cnt + 1).all()),
          f"after a time-out resample: psi_ref == heading_w (max |diff| {err:.2e}), command_counter + 1")
    # the command mix over many forced resamples of every env
    all_ids = torch.arange(n, device=dev)
    E = Z = S = N = 0
    bad = set()
    wz_seen = []
    for _ in range(a.draws):
        term._resample(all_ids)
        mode, head, v = term.turn_mode, term.is_heading_env, term.vel_command_b
        zero, stand = term.zero_wz_env, term.is_standing_env
        direct = mode == DIRECT
        if bool((mode == PURE_TURN).any()):
            bad.add("a PURE_TURN env was drawn")
        if not torch.equal(~head, direct):
            bad.add("is_heading_env not cleared exactly on the explicit envs")
        if not torch.equal(term.explicit_env, direct):
            bad.add("explicit_env != the DIRECT envs")
        if bool((zero & ~direct).any()):
            bad.add("a zero-wz flag on a non-explicit env")
        if not bool((v[zero, 2] == 0.0).all()):
            bad.add("a zero-wz env with wz != 0")
        if not bool(((v[direct, 0].abs() <= 1.0) & (v[direct, 1].abs() <= 0.5) & (v[direct, 2].abs() <= 1.5)).all()):
            bad.add("an explicit command outside R1's ranges")
        if not bool((term.yaw_at_resample == robot.data.heading_w).all()):
            bad.add("psi_ref != heading_w after a resample")
        wz_seen.append(v[direct & ~zero, 2].detach().cpu())
        E += int(direct.sum()); Z += int(zero.sum()); S += int(stand.sum()); N += n
        before = v.clone()
        term._update_command()                      # the per-step update: explicit commands survive, standing zeroed
        keep = direct & ~stand
        if not torch.equal(term.vel_command_b[keep], before[keep]):
            bad.add("_update_command changed an explicit command")
        if not bool((term.vel_command_b[stand] == 0.0).all()):
            bad.add("a standing env was not zeroed")
    wz_all = torch.cat(wz_seen) if wz_seen else torch.zeros(0)
    rec.update(explicit_share=E / N, zero_share_of_explicit=(Z / E if E else None), standing_share=S / N, n_draws=N,
               wz_nonzero_min=float(wz_all.min()) if wz_all.numel() else None,
               wz_nonzero_max=float(wz_all.max()) if wz_all.numel() else None)
    check(not bad, f"every draw: {sorted(bad) if bad else 'no PURE_TURN; heading cleared exactly on explicit envs; wz = 0 exactly on the zero-coin ones; R1 ranges; psi_ref = yaw'}")
    check(abs(E / N - 0.30) <= 0.02, f"explicit share {E / N:.4f} of {N} draws (0.30 +/- 0.02)")
    check(E > 0 and abs(Z / E - 0.5) <= 0.03, f"wz = 0 share of explicit {Z / max(E, 1):.4f} (0.50 +/- 0.03)")
    check(abs(S / N - 0.02) <= 0.01, f"standing share {S / N:.4f} (0.02 +/- 0.01)")
    check(wz_all.numel() > 0 and float(wz_all.min()) < -1.4 and float(wz_all.max()) > 1.4,
          f"explicit nonzero wz spans R1's range: [{rec['wz_nonzero_min']}, {rec['wz_nonzero_max']}]")
    # heading_hold, live: a fresh draw, three steps, then the manager's own value against the formula
    for attempt in range(20):
        term._resample(all_ids)
        ex, wz = term.explicit_env, term.vel_command_b[:, 2]
        if bool((ex & (wz == 0.0) & ~term.is_standing_env).any()) and bool((ex & (wz.abs() >= 0.05)).any()) \
                and bool((~ex).any()):
            break
    for _ in range(3):
        env.step(act)
    idx = names.index("heading_hold")
    live = rm._step_reward[:, idx]                  # = func * weight of the last step
    want = gm.heading_hold_reward(robot.data.heading_w, term.yaw_at_resample, term.vel_command_b[:, 2],
                                  term.explicit_env, 0.2, 0.05)
    keep = ~u.reset_buf.bool()                     # envs reset in that step got a new psi_ref after the reward
    err = float((live[keep] - want[keep]).abs().max()) if bool(keep.any()) else float("nan")
    ex, wz = term.explicit_env, term.vel_command_b[:, 2]
    active = ex & (wz.abs() < 0.05)
    rec.update(live_active=int((active & keep).sum()), live_kept=int(keep.sum()))
    check(bool(keep.any()) and err <= 1e-5, f"reward manager's heading_hold == formula on {int(keep.sum())} non-reset envs "
                                            f"(max |diff| {err:.2e}; active {int((active & keep).sum())})")
    check(bool((live[keep & ~ex] == 0).all()) and bool((live[keep & ex & (wz.abs() >= 0.05)] == 0).all()),
          "heading_hold = 0 on non-explicit envs and on explicit envs with |wz_cmd| >= 0.05")
    vals = live[keep & active]
    check(vals.numel() > 0 and bool(((vals > 0) & (vals <= 1)).all()),
          f"heading_hold in (0, 1] on the active envs: {[round(float(x), 4) for x in vals[:8]]}")
    direct_val = hc.func(u, **hc.params)
    check(float((direct_val - gm.heading_hold_reward(robot.data.heading_w, term.yaw_at_resample, wz, ex, 0.2, 0.05))
                .abs().max()) <= 1e-6, "the term function on the live env == the formula (all envs)")
    k = int(torch.nonzero(active)[0]) if bool(active.any()) else None
    if k is not None:
        saved = term.yaw_at_resample[k].clone()
        term.yaw_at_resample[k] = robot.data.heading_w[k] - 2.0 * math.pi + 0.1     # dpsi = 2 pi - 0.1 -> -0.1
        r = float(hc.func(u, **hc.params)[k])
        term.yaw_at_resample[k] = saved
        check(abs(r - math.exp(-0.25)) <= 1e-4, f"wrap: dpsi = 2 pi - 0.1 pays exp(-(0.1/0.2)^2) = {math.exp(-0.25):.4f}: {r:.4f}")
    else:
        check(False, "no active env for the wrap check")
    env.close()
except Exception as exc:  # noqa: BLE001
    import traceback
    traceback.print_exc()
    check(False, f"probe crashed: {type(exc).__name__}: {exc}")

rec["checks"] = checks
rec["verdict"] = "PASS" if checks and all(c["ok"] for c in checks) else "FAIL"
with open(a.out, "w") as f:
    f.write(json.dumps(rec, indent=2) + "\n")
print(f"R1H-PROBE: {rec['verdict']} ({sum(c['ok'] for c in checks)}/{len(checks)} checks)", flush=True)
app.close()
EOF
    cat > "$PROBE_SH" <<'EOF'
set -uo pipefail
cd "$UPSTREAM"
export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" "$1" --task "$2" --num_envs 64 --draws 300 --out "$3"
EOF
    PROBE_LOG="$PDIR/hold_probe.$JOBTAG.log"
    rm -f "$PROBE_OUT"
    echo "=== Isaac probe of the command mix and heading_hold $(date) ==="
    # full output to a file (no early-closing pipe on a live Isaac process), then the check lines
    bhl_exec "$PROBE_SH" "$PROBE_PY" "$ARM_TASK" "$PROBE_OUT" > "$PROBE_LOG" 2>&1 || true
    grep -E "^  (PASS|FAIL) |^R1H-PROBE" "$PROBE_LOG" || true
    grep -A20 "Traceback" "$PROBE_LOG" | tail -25 || true
    probe_json_line() {   # <probe JSON>: one "R1H-PROBE JSON: PASS|FAIL (...)" line, read from the JSON only
        "$PYH" - "$1" <<'EOF' 2>&1 | tail -1
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    bad = [c["msg"] for c in d.get("checks", []) if not c.get("ok")]
    print(f"R1H-PROBE JSON: {d.get('verdict')} (explicit {d.get('explicit_share')}, zero-wz of explicit "
          f"{d.get('zero_share_of_explicit')}, standing {d.get('standing_share')}, draws {d.get('n_draws')}; "
          f"failed {bad[:3]})")
except Exception as exc:  # noqa: BLE001
    print(f"R1H-PROBE JSON: FAIL (no readable probe JSON: {exc!r})")
EOF
        return 0
    }
    pline=$(probe_json_line "$PROBE_OUT")
    echo "$pline"
    case "$pline" in "R1H-PROBE JSON: PASS"*) ;; *) probe_why=" isaac-probe";; esac
fi

# ---------------------------------------------------------------- export (arm's own task)
cfg="$run/exported/deploy.yaml"
why_export=""
if [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; then
    export RUN_DIR=$(basename "$run")
    if [ ! -f "$cfg" ]; then
        if export_lock; then
            echo "=== export $RUN_DIR with $ARM_TASK $(date) ==="
            TASK="$ARM_TASK" bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh" 2>&1 | grep -vE "pthread_setaffinity|EGLError" || true
            _hold_export_release
        else
            why_export="$why_export export-lock"
        fi
    fi
    norm() { awk '/^policy_checkpoint_path:/{next} /^command_velocity:/{skip=1;next} skip&&/^- /{next} {skip=0; print}' "$1"; }
    norm_r12() { norm "$1" | awk '/^num_observations:/{next} /^gait_clock:/{skip=1;next} skip&&/^  /{next} {skip=0; print}'; }
    ref=$(ls -d "$LOGROOT"/*_arms-dr1.0-s0 | sort | tail -1)/exported/deploy.yaml
    if [ -f "$cfg" ] && [ -f "$run/exported/policy.onnx" ]; then
        grep -q "^policy_checkpoint_path:.*$RUN_DIR/exported/policy.onnx" "$cfg" || why_export="$why_export policy-path"
        grep -qE "^num_joints: 22$" "$cfg" || why_export="$why_export num_joints"
        grep -qE "^num_observations: $OBS_POLICY$" "$cfg" || why_export="$why_export num_observations-not-$OBS_POLICY"
        w=$("$PYH" -c "import onnxruntime as o, sys; print(o.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']).get_inputs()[0].shape[-1])" \
            "$run/exported/policy.onnx" 2>/dev/null || echo "?")
        [ "$w" = "$OBS_POLICY" ] || why_export="$why_export onnx-input-width-$w-not-$OBS_POLICY"
        if [ -z "$why_export" ]; then
            sline=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -m bhl_robust.eval.gait_clock stamp --deploy "$cfg" \
                --env-yaml "$run/params/env.yaml" 2>&1 | tail -1) || true)
            echo "$sline"
            case "$sline" in "GAIT-CLOCK: stamped"*|"GAIT-CLOCK: already stamped"*) ;; *) why_export="$why_export stamp";; esac
            grep -q "^gait_clock:" "$cfg" || why_export="$why_export no-gait_clock-block"
        fi
        diff <(norm_r12 "$ref") <(norm_r12 "$cfg") || why_export="$why_export deploy.yaml-differs-from-arms-dr1.0-s0"
    else
        why_export="$why_export no-deploy.yaml-or-policy.onnx"
    fi
else
    why_export="$why_export no-$FINAL_CKPT"
fi
if [ -z "$why_export" ]; then echo "EXPORT: OK $cfg ($ARM, $OBS_POLICY obs)"; else echo "EXPORT: FAIL ($RUN_NAME:$why_export)"; fi

# ---------------------------------------------------------------- smoke: deploy-path checks, then exit
if [ "$SMOKE" = 1 ]; then
    why="$why_train$probe_why$why_export"
    # this smoke's own training record passes the check every real job applies before its gates
    srec=$(training_record_problems "$TJSON" "$run")
    [ -z "${srec// /}" ] || why="$why training-record:${srec# }"
    if [ -z "$why_export" ]; then
        mkdir -p "$RES/gait-clock-smoke" "$RES/turn-test-v1" "$RES/run-eval"
        kline=$( (cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 "$PYH" -m bhl_robust.eval.gait_clock smoke \
            --deploy "$cfg" --upstream "$UPSTREAM" --cache-dir "$CACHE" --expect "$EXPECT" --steps 30 --seed 100 \
            --out "$RES/gait-clock-smoke/$RUN_NAME.$JOBTAG.json" 2>&1 | grep -v "^\[" | tail -1) || true)
        echo "$kline"
        case "$kline" in "GAIT-CLOCK SMOKE: PASS"*) ;; *) why="$why gait-clock-rollout";; esac
        # plumbing through the gate entry points with this deploy (readable output required; verdicts ignored)
        t1="$RES/turn-test-v1/$RUN_NAME.$JOBTAG.json"
        ( cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 "$PYH" scripts/bench/turn_test.py --deploy "$cfg" \
            --upstream "$UPSTREAM" --cache-dir "$CACHE" --variant humanoid --seconds 1 --warm 0.5 --seed 100 \
            --out "$t1" 2>&1 | grep -v "^\[" | tail -1 ) || true
        "$PYH" -c "import json,sys; d=json.load(open(sys.argv[1])); assert set(d['results'])=={'turn','walk','arc'}" "$t1" 2>/dev/null \
            || why="$why turn_test-plumbing"
        d1="$RES/diagnose/$RUN_NAME.$JOBTAG.json"
        ( cd "$REPO" && PYTHONPATH="$REPO/src:$REPO/scripts/bench" OMP_NUM_THREADS=2 "$PYH" scripts/bench/turn_diagnose.py mujoco \
            --deploy "$cfg" --labels "$RUN_NAME" --upstream "$UPSTREAM" --cache-dir "$CACHE" --seeds 100 --warm 0.5 --seconds 1 \
            --out "$d1" 2>&1 | grep -E "^MUJOCO" | head -2 ) || true
        [ -s "$d1" ] || why="$why turn_diagnose-plumbing"
        # run_eval (the qualify gate's push entry point) unchanged, on exploration seed 100: its six
        # commands x 1 seed with 0.5 m/s pushes; PASS needs a 6-row CSV whose seed column is all 100.
        e1="$RES/run-eval/$RUN_NAME.$JOBTAG.csv"
        eline=$( (cd "$REPO" && PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=2 "$PYH" -m bhl_robust.eval.gait_clock run-eval \
            --deploy "$cfg" --upstream "$UPSTREAM" --cache-dir "$CACHE" --out "$e1" --label "$RUN_NAME" --episode-s 2 \
            --push-speed 0.5 --seed0 100 2>&1 | grep "^GAIT-CLOCK" | tail -1) || true)
        echo "$eline"
        case "$eline" in "GAIT-CLOCK RUN-EVAL: PASS"*) ;; *) why="$why run_eval-plumbing";; esac
        ( cd "$REPO" && "$PYH" scripts/bench/turngait_hold_verdict.py verdict --res "$RES" 2>&1 | tail -1 ) | grep -q "^ARM-HOLD $ARM" \
            || why="$why verdict-script-plumbing"
    fi
    grep -hE "^\[gait_clock_mdp\]|^\[turn_command\]|Group: '(policy|critic)'|\| +(feet_gait|feet_swing_height|feet_air_time|heading_hold|push_robot|base_velocity) +\|" "$TLOG" | head -12 || true
    echo "=== done: $(date) ==="
    if [ -z "${why// /}" ]; then echo "SMOKE: PASS ($ARM $RUN_NAME, $run)"; exit 0; else echo "SMOKE: FAIL ($ARM $RUN_NAME:$why)"; exit 1; fi
fi
[ -z "$why_export" ] || { echo "TURNGAIT-HOLD RESULT: FAIL (export:$why_export; seed stays INCOMPLETE)"; exit 1; }

# ---------------------------------------------------------------- gates (CPU MuJoCo, outside the container)
OUT_DIR=$RES/turn-test-v2
# v2_test <deploy.yaml> <label>: turn_test v2 into a fresh temp JSON, one verdict line read from
# THAT JSON, then moved to $OUT_DIR/<label>.json, or to <label>.$JOBTAG.json if that exists.
v2_test() {
    local cfg=$1 label=$2 tmp
    tmp="$OUT_DIR/.$label.$JOBTAG.json"
    rm -f "$tmp"
    ( cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 "$PYH" scripts/bench/turn_test.py --protocol v2 \
        --deploy "$cfg" --upstream "$UPSTREAM" --cache-dir "$CACHE" --variant humanoid \
        --out "$tmp" 2>&1 | grep -v "^\[" | tail -8 ) || true
    "$PYH" - "$tmp" "$label" <<'EOF'
import json, sys
path, label = sys.argv[1:3]
try:
    d = json.load(open(path))
    assert d.get("protocol") == "v2", "not a v2 JSON"
    turns = ", ".join(f"{r['name']} {r['yaw_deg']:+.1f}" for r in d["turns"])
    print(f"TURN-V2 {label}: {d['verdict']} (turns ok {d['n_turn_ok']}/{d['n_turn']}: {turns}; "
          f"walk drift {d['walk']['yaw_deg']:+.1f} deg, fell {d['walk']['fell_at_s']})")
except Exception as exc:  # noqa: BLE001
    print(f"TURN-V2 {label}: FAIL (no readable v2 JSON: {exc!r})")
EOF
    if [ -f "$tmp" ]; then
        local dst="$OUT_DIR/$label.json"
        [ -e "$dst" ] && dst="$OUT_DIR/$label.$JOBTAG.json"
        mv "$tmp" "$dst"
    fi
}
diag() {
    local out="$RES/diagnose/$2.json"
    [ -e "$out" ] && out="$RES/diagnose/$2.$JOBTAG.json"
    ( cd "$REPO" && PYTHONPATH="$REPO/src:$REPO/scripts/bench" OMP_NUM_THREADS=2 "$PYH" scripts/bench/turn_diagnose.py mujoco \
        --deploy "$1" --labels "$2" --upstream "$UPSTREAM" --cache-dir "$CACHE" --seeds 0 1 2 \
        --out "$out" 2>&1 | grep -E "^MUJOCO|lift-offs L|fell before" | sed 's/^/  diag /' ) || true
    [ -s "$out" ] || echo "  diag $2: no JSON written (non-gating)"
}
VLOG="${TMPDIR:-/tmp}/turngait-hold-v2-$JOBTAG.log"
v2_test "$cfg" "$RUN_NAME" | tee "$VLOG"
diag "$cfg" "$RUN_NAME"
line=$(grep "^TURN-V2 $RUN_NAME:" "$VLOG" | tail -1 || true)
case "$line" in
    *": PASS ("*) echo "TURNGAIT-HOLD RESULT: v2 PASS ($RUN_NAME; $ARM, new task trained from scratch) [LEARNED gait; MuJoCo gates]";;
    *) echo "TURNGAIT-HOLD RESULT: v2 NEGATIVE ($RUN_NAME: ${line#*: }) [LEARNED gait; MuJoCo gates]";;
esac
echo "=== qualification (cpu_turn_qualify.sbatch, unchanged rule) $(date) ==="
TURNQ_RUN="$RUN_NAME" TURNQ_OUT_DIR="$RES/qualify" bash "$REPO/slurm/repo20260923/cpu_turn_qualify.sbatch" 2>&1 \
    | grep -E "TURN-QUALIFY|verdict|FAIL|INCOMPLETE" | tail -4 || true
echo "=== arm verdict $(date) ==="
( cd "$REPO" && "$PYH" scripts/bench/turngait_hold_verdict.py verdict --res "$RES" ) || true
echo "=== done: $(date) ==="
