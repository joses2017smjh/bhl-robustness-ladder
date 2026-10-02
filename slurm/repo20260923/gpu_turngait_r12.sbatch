#!/bin/bash
#SBATCH --job-name=turngait-r12
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=20:00:00
#SBATCH --array=0-5%3
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# 2026-10-01: turning gait recipe R1 / R2 for the 22-DoF humanoid with arms
# (docs/SOLUTIONS_2026-10-01.md section 2). Verified finding: the arm checkpoints
# step only under their own training noise; the noise-free gates and the robot see
# a standing mean policy, and nothing pays stepping at zero command (feet_air_time
# pays only in single stance and only when ||cmd[:2]|| > 0.1). Evidence:
# results/solutions-20260930/turning/ and verify-turning/.
#
# Two NEW tasks, each trained FROM SCRATCH (no parent checkpoint; never a fine-tune
# of arms-turn-turnboth-s0), 6000 iterations, v51, the same DR as TurnBoth (scale 1.0
# via dr_levels.sh) and the same runner. Each = Velocity-BHL-Arms-TurnBoth-v0's
# rewards and command mix with these changes ONLY (FROZEN; constants in
# src/bhl_robust/tasks/gait_clock_mdp.py, classes in arms_env_cfg.py):
#   + feet_gait contact schedule: per foot, phase = ((episode_length_buf * step_dt)
#     mod 0.8) / 0.8 + offset, offsets [0.0, 0.5] (left, right), stance = (phase mod 1)
#     < 0.55; reward = sum over the two feet of XNOR(stance, in_contact); weight 0.5;
#     paid at EVERY command including zero (no command gating). unitree_rl_lab's G1
#     form and constants; paying at zero command follows unitree_rl_gym.
#   + feet_swing_height: weight -20 x sum over feet in swing (||F|| <= 1 N) of
#     (z_foot - h)^2 (unitree_rl_gym G1 form); h = 0.05 m (the upstream env defines no
#     foot-clearance target). DISCLOSED INTERPRETATION: z_foot is read as the sole
#     height, the .*_ankle_roll link origin z minus 0.060 m (the origin's height above
#     the lowest point of the foot collision box at the default joint pose with an
#     upright base, measured in the MuJoCo MJCF of the same URDF before any training).
#     Found after freezing, NOT used to change it (2026-10-02): that pose tilts the
#     foot, so 0.060 is a box corner; the origin sits 0.050 m above a FLAT sole (URDF
#     geometry), TurnBoth-s0 standing in MuJoCo keeps it at a median 0.051 m (seeds
#     100-102), and Isaac printed 0.072 m at the first reward call after a reset (not
#     a settled stance). The swing target is therefore an origin height of 0.11 m,
#     about 6 cm of flat-sole clearance rather than the nominal 5 cm. Read literally
#     (link origin, offset 0), h = 0.05 m is the origin's resting height and the term
#     would penalise lifting a foot. The training log prints Isaac's first reading.
#   = feet_air_time weight 0.
#   + push_robot interval event from iteration 0: velocity_range x and y in [-0.5, 0.5]
#     m/s at a FIXED magnitude (no push curriculum), every 5-9 s (ArmsPushEventsCfg's
#     interval, PUSH_INTERVAL_S).
#   = default PPO noise (no std clamp); same runner as TurnBoth.
# ARM TABLE (array index -> arm, task, run names, actor observations):
#   0-2  R1  Velocity-BHL-Arms-TurnGaitClock-v0   arms-turngait-clock-s{0,1,2}   actor 77 = 75 + sin/cos(2 pi phase_left), critic 80
#   3-5  R2  Velocity-BHL-Arms-TurnGaitCritic-v0  arms-turngait-critic-s{0,1,2}  actor 75 (clock in the critic only), critic 80
# Per seed, in the job: training -> export with the ARM's OWN task (R1's actor is 77 wide;
# both critics are 80 wide, so neither loads into the stock task's runner) -> R1 only:
# the gait_clock block stamped into deploy.yaml from the run's own env.yaml -> turn_test
# v2 (gating) -> the non-gating MuJoCo diagnosis -> cpu_turn_qualify.sbatch (unchanged;
# its v2x and run_eval paths build the controller through bhl_robust.eval.gait_clock,
# which returns upstream's RlController itself for a deploy.yaml without a gait_clock
# block) -> the arm's verdict (scripts/bench/turngait_r12_verdict.py).
#
# PREDECLARED RULE (frozen 2026-10-01, before any R1/R2 run exists; v5's joint rule
# verbatim, SLURM_JOBS.md line 3055), per arm, each arm judged separately:
#   PASS iff >= 2/3 seeds both PASS turn_test v2 AND are QUALIFIED by cpu_turn_qualify's
#   unchanged rule (v2x >= 9/10 on reset seeds 10-14, walk <= 15 deg on >= 2/3, push <=
#   9/60); else FAIL; INCOMPLETE if any JSON is missing. A seed that fails v2 does not
#   count even if it qualifies through v2x.
#   Training counts only if the run dir holds model_5999.pt and the training log shows
#   the arm's task id, the feet_gait term and the push event.
#   The two arms are new tasks, never presented as fine-tunes of TurnBoth-s0.
#   Labels: LEARNED gait; MuJoCo gates.
# Disclosed priors: pushes from iteration 0 may cost the turn (the v5 PUSH arm 21443377,
# TurnRest + adaptive pushes, kept push falls <= 0.10 but turned 0/10 on fresh reset
# seeds on all three seeds); 22-DoF from-scratch push training showed no detectable
# benefit at four seeds. The period (0.8 s) and weight (0.5) are the unitree_rl_lab G1
# precedent, not tuned here; no Unitree repo has tested a clockless actor (R2) that
# marches at zero command.
#
# Verdicts come from JSON, never exit codes (Isaac exits 0 on a crash). Per seed:
#   <RES>/training/<run>.json        written once from the FRESH training log (frozen clause)
#   <RES>/turn-test-v2/<run>.json    turn_test.py --protocol v2 (never overwritten)
#   <RES>/diagnose/<run>.json        turn_diagnose.py mujoco (reported, not gated)
#   <RES>/qualify/<run>__qualify.json  cpu_turn_qualify.sbatch (never overwritten)
#   <RES>/verdict/<R1|R2>.json       written once, by the arm's last seed (O_EXCL)
# RES = results/repo-gpu-20260923/turngait-r12-20261001. An existing run dir with
# model_5999.pt is reused, never retrained or overwritten; an existing verdict is re-printed.
# Wrong-arm guard on the fresh training log (beyond the frozen clause): the policy /
# critic observation widths of the arm, the feet_swing_height (-20.0) and feet_air_time
# (0.0) rows, no push_levels curriculum, both gait_clock_mdp terms having run, and the
# run's env.yaml = TurnBoth-s0's + the declared changes only (gait_clock check-recipe).
# Its reasons are stored in the training record (wrong_arm_guard). Before any export or
# gate, every non-smoke job -- fresh, or a resubmit that reuses the trained run dir and so
# has no fresh log -- re-reads that record and runs no gate unless it is this run dir's,
# holds the frozen clause's flags and has an empty wrong_arm_guard: a seed whose first job
# failed the guard stays INCOMPLETE on a resubmit. (The verdict's training clause is unchanged.)
#
# Modes (env vars):
#   TURNGAIT_SMOKE=1  MAX_ITER=3, NUM_ENVS=64, run <prefix>-s<seed>-smoke, outputs under
#                     /nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/campaign-turning/turngait-r12-smoke
#                     (the dir must end in -smoke). Trains, checks the log and env.yaml,
#                     exports with the arm's task, checks the deploy (R1: 77 + stamped
#                     clock; R2: 75, no clock), then a short clock-aware (R1) / plain (R2)
#                     MuJoCo rollout (python -m bhl_robust.eval.gait_clock smoke) and short
#                     runs of turn_test v1, turn_diagnose mujoco and run_eval through the
#                     same deploy (plumbing: readable output required, verdicts ignored).
#                     Every smoke rollout runs on exploration seed 100, never on a scored
#                     seed (v2 reset 0-2 / walk 0, v2x 10-14 / 10-12, push eval 0-9): run_eval
#                     goes through `python -m bhl_robust.eval.gait_clock run-eval`, which runs
#                     it unchanged with its eval seeds moved to 100+ and requires a 6-row CSV
#                     all on seed 100. The smoke also writes its own training record and
#                     passes it through the record check every real job applies.
#                     One "SMOKE: PASS|FAIL (...)" line; exit 0 iff PASS (chain afterok).
# Side effects outside RES, as for every humanoid training and export: the run dir
# <ts>_<run> in $UPSTREAM/logs/rsl_rl/humanoid (a smoke run's name ends in -smoke), Isaac's
# log $UPSTREAM/logs/rsl_rl/humanoid/isaaclab/isaaclab_<ts>.log, and the export's rewrite
# of $UPSTREAM/configs/policy_latest.yaml (serialised by the export lock below).
#   Submit (coordinator) in ONE command: a fresh smoke of the exact code being submitted,
#   with the real array chained afterok on it. Never chain on an older smoke id: slurmctld
#   forgets a finished job after ~5 min (MinJobAge 300 s), and the code may have changed.
#     cd /nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder && source slurm/_env.sh && \
#     S=$(slurm_clean sbatch --parsable --job-name=turngait-r12-smoke --array=0,3 --time=01:00:00 \
#         --export=ALL,TURNGAIT_SMOKE=1 slurm/repo20260923/gpu_turngait_r12.sbatch) && \
#     slurm_clean sbatch --dependency=afterok:${S%%;*} --export=ALL,TURNGAIT_SMOKE=0 \
#         slurm/repo20260923/gpu_turngait_r12.sbatch
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
PYH=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid
JOBTAG=${SLURM_JOB_ID:-local$$}
SMOKE=${TURNGAIT_SMOKE:-0}
FULL_ITERS=6000
SMOKE_RES=/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/campaign-turning/turngait-r12-smoke
if [ "$SMOKE" = 1 ]; then
    RES=${TURNGAIT_OUT_DIR:-$SMOKE_RES}
    case "$RES" in *-smoke) ;; *) echo "SMOKE: FAIL (smoke outputs must go to a directory ending in -smoke, not $RES)"; exit 1;; esac
else
    RES=$REPO/results/repo-gpu-20260923/turngait-r12-20261001
fi
IDX=${SLURM_ARRAY_TASK_ID:-0}
case $IDX in
    0|1|2) ARM=R1; ARM_TASK=Velocity-BHL-Arms-TurnGaitClock-v0;  PREFIX=arms-turngait-clock;  OBS_POLICY=77; OBS_CRITIC=80; EXPECT=clock;;
    3|4|5) ARM=R2; ARM_TASK=Velocity-BHL-Arms-TurnGaitCritic-v0; PREFIX=arms-turngait-critic; OBS_POLICY=75; OBS_CRITIC=80; EXPECT=plain;;
    *) echo "TURNGAIT-R12 RESULT: FAIL (array index $IDX outside 0-5)"; exit 2;;
esac
SEED=$((IDX % 3))
if [ "$SMOKE" = 1 ]; then
    export MAX_ITER=3 NUM_ENVS=64 RUN_NAME="$PREFIX-s${SEED}-smoke"
else
    export MAX_ITER=$FULL_ITERS RUN_NAME="$PREFIX-s${SEED}"
    unset NUM_ENVS                                  # the task's own env count, as TurnBoth trained
fi
FINAL_CKPT="model_$((MAX_ITER - 1)).pt"
export TASK="$ARM_TASK" SEED
mkdir -p "$RES"/{training,turn-test-v2,diagnose,qualify,verdict}
CACHE="${TMPDIR:-/tmp}/turngait-r12-$JOBTAG"
REF_RUN=$(ls -d "$LOGROOT"/*_arms-turn-turnboth-s0 2>/dev/null | sort | tail -1 || true)

if [ "$SMOKE" != 1 ] && [ -f "$RES/verdict/$ARM.json" ]; then
    echo "verdict already recorded, not overwritten: $RES/verdict/$ARM.json"
    "$PYH" -c "import json,sys; d=json.load(open(sys.argv[1])); print('ARM-R12', d['arm'], d['verdict'], d['n_counted'], '/ 3')" "$RES/verdict/$ARM.json"
    exit 0
fi

# ---------------------------------------------------------------- GPU from here
# From scratch: a recurrent / RND / symmetry overlay forwarded from the submitting shell
# (BHL_FORWARD_VARS) would change the network and is not part of the recipe.
unset BHL_POLICY BHL_RND BHL_SYMMETRY
source "$REPO/slurm/dr_levels.sh"
setup_node_cache
CACHE="${TMPDIR:-/tmp}/turngait-r12-$JOBTAG"
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate

# Exports serialise on one lock: export_one_humanoid.sh copies the shared
# $UPSTREAM/configs/policy_latest.yaml, which two array tasks exporting together
# would race on (mkdir is atomic on Lustre/NFS; stale after 20 min > the 16-min export cap).
EXPORT_LOCK="$WORKSPACE/.turngait-r12-export.lock"
_r12_export_release() {
    [ "$(cut -d' ' -f1 "$EXPORT_LOCK/holder" 2>/dev/null)" = "$JOBTAG" ] && rm -rf "$EXPORT_LOCK"
    return 0
}
trap "_v60_release '$WORKSPACE/.v51-boot.lock'; _r12_export_release" EXIT
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
echo "=== node: $(hostname) | $ARM $RUN_NAME | task=$TASK | smoke=$SMOKE | from scratch, $MAX_ITER iters -> $FINAL_CKPT | RES $RES | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
cat "$OVERRIDE_FILE"
TLOG="${TMPDIR:-/tmp}/turngait-r12-$JOBTAG.log"; : > "$TLOG"
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
# The frozen clause (training JSON) and the wrong-arm guard, both from the FRESH log.
TASK_RE="^task=$ARM_TASK run=$RUN_NAME seed=$SEED "
GAIT_RE="\| +[0-9]+ +\| +feet_gait +\| +0\.5 +\|"
PUSH_RE="\| +[0-9]+ +\| +push_robot +\| +\(5\.0, 9\.0\) +\|"
arm_guard() {   # <log>: reasons the log is NOT this arm's recipe (empty = ok)
    grep -qE "Group: 'policy' \(shape: \($OBS_POLICY,\)\)" "$1" || printf ' policy-obs-not-%s' "$OBS_POLICY"
    grep -qE "Group: 'critic' \(shape: \($OBS_CRITIC,\)\)" "$1" || printf ' critic-obs-not-%s' "$OBS_CRITIC"
    grep -qE "\| +[0-9]+ +\| +feet_swing_height +\| +-20\.0 +\|" "$1" || printf ' no-feet_swing_height--20.0-row'
    grep -qE "\| +[0-9]+ +\| +feet_air_time +\| +0\.0 +\|" "$1" || printf ' feet_air_time-weight-not-0.0'
    grep -qE "\| +base_velocity +\| +UniformVelocityCommand +\|" "$1" || printf ' command-class-is-not-UniformVelocityCommand'
    grep -q "push_levels" "$1" && printf ' unexpected-push_levels-curriculum'
    grep -qE "^\[gait_clock_mdp\] feet_gait bodies \['[^']*left[^']*', '[^']*right[^']*'\] offsets \[0\.0, 0\.5\]" "$1" || printf ' feet_gait-did-not-run-on-(left,right)'
    grep -qE "^\[gait_clock_mdp\] feet_swing_height bodies" "$1" || printf ' feet_swing_height-did-not-run'
    grep -qE "train.sh: reached [0-9]+ logged iterations" "$1" || printf ' no-logged-iterations'
    return 0
}
# training_record_problems <record JSON> <run dir>: reasons the stored training record does not
# let this seed's gates run (empty = ok): it must be this run's and this run dir's, name
# $FINAL_CKPT, hold the frozen clause's four flags and an EMPTY wrong_arm_guard list. A missing,
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
for k in ("final_ckpt_present", "task_id_in_log", "feet_gait_in_log", "push_robot_in_log"):
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
# The run's own env.yaml = TurnBoth-s0's + the declared recipe changes only (both arms).
rline="GAIT-CLOCK RECIPE: not checked (no env.yaml or no TurnBoth-s0 reference)"
if [ -n "$run" ] && [ -f "$run/params/env.yaml" ] && [ -n "$REF_RUN" ]; then
    rline=$( (cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 "$PYH" -m bhl_robust.eval.gait_clock check-recipe \
        --env-yaml "$run/params/env.yaml" --arm "$ARM" --ref-env-yaml "$REF_RUN/params/env.yaml" 2>&1 | tail -1) || true)
fi
echo "$rline"
case "$rline" in "GAIT-CLOCK RECIPE: OK"*) ;; *) why_train="$why_train recipe-check-failed";; esac
if [ "$FRESH" = 1 ]; then
    ckpt_ok=false; { [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; } && ckpt_ok=true
    task_ok=false; grep -qE "$TASK_RE" "$TLOG" && task_ok=true
    gait_ok=false; grep -qE "$GAIT_RE" "$TLOG" && gait_ok=true
    push_ok=false; grep -qE "$PUSH_RE" "$TLOG" && push_ok=true
    why_train="$why_train$(arm_guard "$TLOG")"
    [ "$ckpt_ok" = true ] || why_train="$why_train no-$FINAL_CKPT"
    [ "$task_ok" = true ] || why_train="$why_train task-id-not-in-log"
    [ "$gait_ok" = true ] || why_train="$why_train no-feet_gait-0.5-row"
    [ "$push_ok" = true ] || why_train="$why_train no-push_robot-(5.0,9.0)-row"
    { head -n 400 "$TLOG"; echo "[... cut ...]"; tail -n 60 "$TLOG"; } > "$RES/training/$RUN_NAME.$JOBTAG.log-excerpt.txt"
    grep -hE "^\[gait_clock_mdp\]" "$TLOG" | head -4 || true
    if [ "$ckpt_ok" = true ] && [ ! -e "$TJSON" ]; then
        "$PYH" - "$TJSON" "$RUN_NAME" "$ARM" "$ARM_TASK" "$SEED" "$FINAL_CKPT" "$run" "$JOBTAG" \
            "$ckpt_ok" "$task_ok" "$gait_ok" "$push_ok" "$why_train" "$rline" <<'EOF'
import json, os, sys
(out, run, arm, task, seed, ckpt, run_dir, job, ckpt_ok, task_ok, gait_ok, push_ok, why, recipe) = sys.argv[1:15]
b = lambda s: s == "true"  # noqa: E731
rec = {"run": run, "arm": arm, "task": task, "seed": int(seed), "final_ckpt": ckpt, "final_ckpt_present": b(ckpt_ok),
       "task_id_in_log": b(task_ok), "feet_gait_in_log": b(gait_ok), "push_robot_in_log": b(push_ok),
       "run_dir": run_dir, "job": job, "wrong_arm_guard": why.split(), "recipe_check": recipe,
       "clause": "Training counts only if the run dir holds model_5999.pt and the training log shows the arm's task id, "
                 "the feet_gait term and the push event.",
       "note": "from scratch (no parent checkpoint); written once from this job's fresh training log"}
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
        echo "TURNGAIT-R12 RESULT: FAIL (training incomplete: no $FINAL_CKPT for $RUN_NAME; seed stays INCOMPLETE)"; exit 1; }
    if [ "$FRESH" = 1 ] && [ -n "${why_train// /}" ]; then
        echo "TURNGAIT-R12 RESULT: FAIL (training log / env.yaml is not the $ARM recipe:$why_train; no gates run, seed stays INCOMPLETE)"; exit 1
    fi
    if [ "$FRESH" != 1 ]; then
        [ -f "$TJSON" ] || { echo "TURNGAIT-R12 RESULT: FAIL (already trained but no training record $TJSON and no fresh log; seed stays INCOMPLETE)"; exit 1; }
        case "$why_train" in *recipe-check-failed*) echo "TURNGAIT-R12 RESULT: FAIL ($why_train; no gates run, seed stays INCOMPLETE)"; exit 1;; esac
    fi
    # Fresh or resubmitted (a resubmit has no fresh log): the stored record must be this run
    # dir's, hold the frozen clause's flags and an empty wrong_arm_guard, or no gate runs.
    why_rec=$(training_record_problems "$TJSON" "$run")
    if [ -n "${why_rec// /}" ]; then
        echo "TURNGAIT-R12 RESULT: FAIL (training record $TJSON:$why_rec; no gates run, seed stays INCOMPLETE)"; exit 1
    fi
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
            _r12_export_release
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
        if [ -z "$why_export" ] && [ "$ARM" = R1 ]; then
            sline=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -m bhl_robust.eval.gait_clock stamp --deploy "$cfg" \
                --env-yaml "$run/params/env.yaml" 2>&1 | tail -1) || true)
            echo "$sline"
            case "$sline" in "GAIT-CLOCK: stamped"*|"GAIT-CLOCK: already stamped"*) ;; *) why_export="$why_export stamp";; esac
            grep -q "^gait_clock:" "$cfg" || why_export="$why_export no-gait_clock-block"
        fi
        if [ "$ARM" = R2 ] && grep -q "^gait_clock:" "$cfg"; then why_export="$why_export R2-deploy-has-a-gait_clock-block"; fi
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
    why="$why_train$why_export"
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
        ( cd "$REPO" && "$PYH" scripts/bench/turngait_r12_verdict.py --arm "$ARM" --res "$RES" 2>&1 | tail -1 ) | grep -q "^ARM-R12 $ARM" \
            || why="$why verdict-script-plumbing"
    fi
    grep -hE "^\[gait_clock_mdp\]|Group: '(policy|critic)'|\| +(feet_gait|feet_swing_height|feet_air_time|push_robot) +\|" "$TLOG" | head -8 || true
    echo "=== done: $(date) ==="
    if [ -z "${why// /}" ]; then echo "SMOKE: PASS ($ARM $RUN_NAME, $run)"; exit 0; else echo "SMOKE: FAIL ($ARM $RUN_NAME:$why)"; exit 1; fi
fi
[ -z "$why_export" ] || { echo "TURNGAIT-R12 RESULT: FAIL (export:$why_export; seed stays INCOMPLETE)"; exit 1; }

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
VLOG="${TMPDIR:-/tmp}/turngait-r12-v2-$JOBTAG.log"
v2_test "$cfg" "$RUN_NAME" | tee "$VLOG"
diag "$cfg" "$RUN_NAME"
line=$(grep "^TURN-V2 $RUN_NAME:" "$VLOG" | tail -1 || true)
case "$line" in
    *": PASS ("*) echo "TURNGAIT-R12 RESULT: v2 PASS ($RUN_NAME; $ARM, new task trained from scratch) [LEARNED gait; MuJoCo gates]";;
    *) echo "TURNGAIT-R12 RESULT: v2 NEGATIVE ($RUN_NAME: ${line#*: }) [LEARNED gait; MuJoCo gates]";;
esac
echo "=== qualification (cpu_turn_qualify.sbatch, unchanged rule) $(date) ==="
TURNQ_RUN="$RUN_NAME" TURNQ_OUT_DIR="$RES/qualify" bash "$REPO/slurm/repo20260923/cpu_turn_qualify.sbatch" 2>&1 \
    | grep -E "TURN-QUALIFY|verdict|FAIL|INCOMPLETE" | tail -4 || true
echo "=== arm verdict $(date) ==="
( cd "$REPO" && "$PYH" scripts/bench/turngait_r12_verdict.py --arm "$ARM" --res "$RES" ) || true
echo "=== done: $(date) ==="
