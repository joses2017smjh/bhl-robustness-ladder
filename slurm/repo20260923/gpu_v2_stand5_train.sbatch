#!/bin/bash
#SBATCH --job-name=v2-stand5-train
#SBATCH --account=eecs
#SBATCH --partition=gpu,dgxh,ampere
#SBATCH --constraint="a40|rtx8000|l40s|h100|h200|h200-70g"
#SBATCH --exclude=dgxh-1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=40:00:00
#SBATCH --array=0-1
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%A_%a.out
# CubeToShelfStand5 (2026-10-03; SLURM_JOBS.md "User approval recorded
# 2026-10-03 09:55", item (S)): blind, seeds 0 and 1 (array index = seed).
# Stand5 = CubeToShelfStand4 + exactly two changes; a different configuration
# from Stand4, judged only by its own predeclared rule below, and never compared
# with CubeToShelfStand3 or CubeToShelf numbers. Recipe as
# gpu_v2_stand4_train.sbatch: v60, cameras on for stack parity, 1024 envs, 8000
# iterations, Stand3's runner (TaskV2Stand3PPORunnerCfg), Stand4's resources.
# Submit afterok on gpu_v2_stand5_smoke.sbatch, passing its job id
# (--export=ALL,STAND5_SMOKE_JOB=<smoke>; BYTES-CHECK below). Every term,
# constant and reason: src/bhl_robust/tasks/stand5_mdp.py.
#
# FROZEN DESIGN (verbatim, SLURM_JOBS.md "User approval recorded 2026-10-03
# 09:55", item (S); stand5_mdp.DESIGN_S):
#   **(S) Stand5.** Stand4 + two changes.
#   (i) The actor and critic observe the cube's orientation in each robot's
#   root frame (its z axis; ORACLE, like `object_pos_a/b`).
#   (ii) The lift curriculum promotes on Stand4's roll-proof lift condition
#   (lowest corner ≥ 2 cm above every support and tilt ≤ 15°) instead of
#   centre height + pinch.
#   Stand4's rule verbatim: the kill rule at model_1000; COMPLETE at 7999;
#   PASS iff ≥ 1 of 2 seeds is COMPLETE, not killed, with last-200 success
#   ≥ 0.10. 2 seeds × 8000 iterations, Stand4's resources.
#   Labels: LEARNED crew policy (blind) + ORACLE privileged observations (now
#   including cube orientation).
#
# Context, not a reason the design states: Stand4 21506758 was NEGATIVE by its
# rule (last-200 success 0.0064 / 0.0009; the crews stand but neither lift the
# cube clear nor seat it).
# Changes from Stand4, these ONLY (Stand4, Stand3 and every other task untouched):
#   (i)  observation  object_zaxis_a / object_zaxis_b: the cube's own z axis in
#                     robot_a's / robot_b's root frame, R_robot^T (R_cube e_z), from
#                     the simulator quaternions read through quat_order.unpack_wxyz
#                     ((x, y, z, w) on v60); appended last to BOTH the actor's
#                     (policy) and the critic's groups: 194 -> 200 and 206 -> 212
#                     inputs. ORACLE like object_pos_a/b: no noise, no scale;
#                     clipped +/-100 like every observation term since Stand3 (it
#                     never binds: every component is in [-1, 1]).
#   (ii) curriculum   lift_height keeps its name, place, parameters (step 0.02,
#                     0.04 -> 0.06 cap, target 0.35, demotion below 0.175, term
#                     lifting_object) and its write into minimal_height; an env
#                     counts as competent iff the cube's lowest corner is >= 0.02 m
#                     above every support under it AND its tilt (own z axis vs
#                     world up) is <= 15 deg (stand4_mdp.roll_proof_mask), instead
#                     of centre > spawn + level AND pinch distance < 0.20 m.
# Unchanged: rewards (lifting_object still pays centre > spawn + minimal_height x
# roll-proof gate x pinch kernel x upright gate), success (Stand4's termination:
# seated, tilt <= 8 deg, released, held 12 steps; placed x 5000 reads it),
# terminations, scene, spawn, hand colliders (the overlay), actuators, actions,
# events, runner, 8000 iterations, seeds 0 and 1, resources.
#
# PREDECLARED RULE: Stand4's rule, verbatim (stand4_mdp.PREDECLARED_RULE):
#   "PREDECLARED RULE (frozen; Stand3's structure): the kill rule at model_1000
#   exactly as Stand3's launcher applies it (copy its condition and window
#   verbatim); a seed is COMPLETE iff it reaches iteration 7999; Stand4 PASSES
#   iff >= 1 of 2 seeds has last-200-iteration success >= 0.10, where success is
#   the Stand4 termination above (seated, tilt <= 8 deg, released). Otherwise
#   NEGATIVE; INCOMPLETE if a seed is neither complete nor stopped by the kill
#   rule. Stand4 is a different task from Stand3 and CubeToShelf and is never
#   compared with their numbers. Labels: LEARNED crew policies; ORACLE
#   privileged observations exactly as Stand3 uses them (say which); MODIFIED
#   hand colliders (the overlay) relative to Stand3."
# Applied to Stand5 as written, by Stand4's functions: Stand5's success IS the
# Stand4 termination (unchanged), and the verdict is Stand5's ("PASS: learned
# roll-proof placement (Stand5)"). Stand5's labels are the frozen design's (they
# replace the rule's last sentence, which describes Stand4's observations).
#
# The clauses as applied (constants and code: stand5_mdp.py, which calls
# stand4_mdp.py):
#   kill      Stand3's condition and window, copied verbatim from
#             gpu_v2_stand3_train.sbatch (as Stand4 did) and evaluated by the
#             same function (stand_mdp.evaluate_kill3 via stand5_mdp.evaluate_kill):
#   kill      Stand2's rule: at model_1000, over iterations 901-1000,
#             mean episode length < 100 steps (4 s), OR mean
#             (time_out + success) < 0.10 -> "cannot stand"; training is
#             stopped by the watcher below. A killed seed is decided, not placed.
#             Amended 2026-09-27 on review, before any Stand3 run
#             (stand_mdp.kill_verdict3): the length clause is skipped when the
#             mean success over 901-1000 is >= 0.10, because success ends the
#             episode and a quick placement is short -- a placing seed is never
#             killed for short episodes.
#   complete  Stand3's completeness check, as Stand4 applied it (clause (a) of
#             Stand4's predeclaration): a seed is COMPLETE iff its events reach
#             iteration 7999 (8000 iterations) AND every Loss/value, success and
#             time_out value in its last 200 iterations is finite AND that window
#             holds a Loss/value value (stand4_mdp.evaluate_seed4 via
#             stand5_mdp.evaluate_seed5). A seed that logged to 7999 but went
#             non-finite in its last 200 iterations is INCOMPLETE: neither a PASS
#             nor a NEGATIVE. The Loss/value instability count (> 1000 or
#             non-finite, any iteration) is reported, not gating. Exit codes are
#             never read.
#   tilt      Stand4's, for the success clause (<= 8 deg) and for the lift and
#             curriculum clauses (<= 15 deg): the angle between the cube's OWN z
#             axis and world up (a cube rolled 90 deg onto another face reads 90
#             deg and never seats).
#   curriculum  the lift curriculum's competence is the share of the resetting
#             envs whose cube's lowest corner is >= 0.02 m above every support
#             under it AND whose tilt is <= 15 deg (stand4_mdp.roll_proof_mask);
#             no centre-height and no pinch clause; promote above 0.35, demote
#             below 0.175, level 0.04 -> 0.06 in steps of 0.02, written into
#             lifting_object's minimal_height
#             (stand5_mdp.stand5_lift_height_curriculum). Observed in the smoke,
#             a property of the frozen design (not tuned): a cube clear and flat
#             in the air counts without a grasp.
#   result    success = last-200-iteration mean Episode_Termination/success, the
#             Stand4 termination (seated, tilt <= 8 deg, released, held 12
#             steps). PASS iff >= 1 of the 2 seeds is COMPLETE, not killed, and
#             >= 0.10 = "PASS: learned roll-proof placement (Stand5)". Otherwise
#             NEGATIVE if both seeds are decided (complete or killed); INCOMPLETE
#             (no verdict) if a seed is neither complete nor stopped by the kill
#             rule. Precedence (Stand4's): PASS if any seed passes whatever the
#             other did, then INCOMPLETE, then NEGATIVE (stand4_mdp.pair_verdict4,
#             relabelled by stand5_mdp.pair_verdict5).
#   labels    LEARNED crew policy (one PPO actor drives both robots, blind);
#             ORACLE privileged observations, now including cube orientation:
#             the actor and the critic both observe object_pos_a/b (simulator
#             cube position in each robot's root frame) and object_zaxis_a/b (the
#             simulator cube's own z axis in each robot's root frame); the critic
#             also base_lin_vel_a/b, object_lin_vel, object_ang_vel; MODIFIED hand
#             colliders (the overlay) relative to Stand3, inherited from Stand4.
#   scope     CubeToShelfStand5 only; judged by this rule alone, never compared
#             with CubeToShelfStand3 or CubeToShelf numbers. Both seeds are fresh
#             runs; no seed is added or rerun after seeing a result: this launcher
#             refuses to start if a result JSON for its seed or a run dir with its
#             run name exists.
#
# Also checked at start (RUNNER-CHECK, stand5_mdp.config_check5 -- the same check
# the smoke ran on its own run): every clause of Stand4's config_check
# (entropy_coef 0.001 and std_type log; lifting_object = stand4_object_is_lifted
# x 15.0 with clearance 0.02 and tilt_max_deg 15.0; placed = success_bonus x
# 5000.0; action_rate_clipped; success = stand4_cube_placed with 12 steps, 8.0
# deg, 1.0 N, cube_contact; both robots on the overlay USD; cube contact
# reporting; the 26 sensor filters; the curriculum terms; the 0.06 lift cap) plus
# Stand5's: both observation groups are Stand4's terms followed by
# object_zaxis_a, object_zaxis_b (stand5_mdp:object_zaxis_in_root, robot_a /
# robot_b, no noise, no scale, clip +/-100), and lift_height =
# stand5_mdp:stand5_lift_height_curriculum with Stand4's parameters plus
# clearance 0.02 and tilt_max_deg 15.0. Otherwise training is stopped and the
# seed is INCOMPLETE (bad_config).
#
# BYTES-CHECK (Stand4's, pinned to the Stand5 sources), before anything trains:
# the training runs on the bytes its chained smoke validated. Submit with
# --dependency=afterok:<smoke> --export=ALL,STAND5_SMOKE_JOB=<smoke>. The job
# refuses to start (exit 1, no run dir, nothing trains) unless
# results/repo-gpu-20260923/stand5_2026-10-03-smoke/verdict_<smoke>.json reads
# PASS and every file in that smoke's sha256_<smoke>.txt matches this working
# tree (sha256sum -c --strict), except src/bhl_robust/tasks/__init__.py (shared
# with other workstreams; what the Stand5 registration resolves to -- the env cfg
# and runner the run trains with -- is what RUNNER-CHECK verifies on
# params/*.yaml) and the smoke's own two launcher scripts (not used here). The
# checked set must contain stand5_mdp.py, stand4_mdp.py, task_v2_env_cfg.py,
# stand_mdp.py, coop_lift_mdp.py, coop_lift_env_cfg.py, quat_order.py and the
# overlay USD. Output: bytes_check_s<seed>_<job>.txt.
#
# Verdict lines (BYTES-CHECK, RUNNER-CHECK, KILL-RULE, RESULT) come from JSON
# written under results/repo-gpu-20260923/stand5_2026-10-03/, never from exit codes.
set -euo pipefail
export BHL_STACK=v60 ENABLE_CAMERAS=1
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
setup_node_cache
JOBTAG="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}_${SLURM_ARRAY_TASK_ID:-0}"
export OV_CACHE=/scratch/$USER/ov-cache-${JOBTAG}
export CUDA_CACHE_PATH=/scratch/$USER/nv-computecache-${JOBTAG}
mkdir -p "$OV_CACHE" "$CUDA_CACHE_PATH" "$WORKSPACE/overrides"
source "$REPO/slurm/inner/v60_boot_gate.sh"
export TASK="TaskV2-BHL-CubeToShelfStand5-Blind-v0"
# 8000 is the predeclared budget. MAX_ITER can be lowered for a dry run, but the
# completion check always demands iteration 7999, so a shorter run can never be
# read as a result.
export SEED=${SLURM_ARRAY_TASK_ID:-${SEED:-0}} MAX_ITER=${MAX_ITER:-8000} NUM_ENVS=${NUM_ENVS:-1024}
case "$SEED" in 0|1) ;; *) echo "STAND5: seed $SEED is not a predeclared Stand5 seed (0, 1); refused"; exit 1;; esac
export RUN_NAME="v2-cubetoshelfstand5-blind-s${SEED}"
export OVERRIDE_FILE="$WORKSPACE/overrides/v2-stand5-${JOBTAG}.txt"
: > "$OVERRIDE_FILE"
unset TRAIN_SCRIPT BHL_LEGACY_YAW

KILL_AT=1000
LOGROOT=$UPSTREAM/logs/rsl_rl/task_v2
OUTDIR=$REPO/results/repo-gpu-20260923/stand5_2026-10-03
SMOKE_OUT=$REPO/results/repo-gpu-20260923/stand5_2026-10-03-smoke
# BYTES-CHECK: files that must be in the smoke's sha256 list and match
BYTES_REQUIRED="src/bhl_robust/tasks/stand5_mdp.py src/bhl_robust/tasks/stand4_mdp.py src/bhl_robust/tasks/task_v2_env_cfg.py src/bhl_robust/tasks/stand_mdp.py src/bhl_robust/tasks/coop_lift_mdp.py src/bhl_robust/tasks/coop_lift_env_cfg.py src/bhl_robust/quat_order.py assets/cloth/berkeley_humanoid_lite_hand_colliders.usda"
HOSTPY=$WORKSPACE/venv/bin/python          # host interpreter with tensorboard
mkdir -p "$OUTDIR"
# No rerun after a result, no overwrite: an earlier result for this seed or a run
# dir with this run name stops the job before anything trains.
prior=$(ls "$OUTDIR"/result_s"${SEED}"_*.json 2>/dev/null || true)
if [ -n "$prior" ]; then
    echo "STAND5 $RUN_NAME: refusing to start -- a result for seed $SEED exists: $prior"; exit 1
fi
prior=$(find "$LOGROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" 2>/dev/null || true)
if [ -n "$prior" ]; then
    echo "STAND5 $RUN_NAME: refusing to start -- run dir(s) with this run name exist: $prior"; exit 1
fi
# Same filesystem as the run dirs (NFS), so -newer is not a cross-host clock
# comparison. Removed at the end (the boot gate owns the EXIT trap).
STAMP=$(mktemp "$OUTDIR/.start-${JOBTAG}-XXXXXX")
KILLED_FLAG="$OUTDIR/killed_s${SEED}_${JOBTAG}.flag"
BADCFG_FLAG="$OUTDIR/badcfg_s${SEED}_${JOBTAG}.flag"

find_run() {  # this job's run dir: newest *_RUN_NAME created after job start
    find "$LOGROOT" -maxdepth 1 -type d -name "*_${RUN_NAME}" -newer "$STAMP" 2>/dev/null | sort | tail -1 || true
}

# stand5_eval MODE RUN_DIR OUT_JSON [KILLED] [BADCFG]: MODE kill|seed. Prints
# nothing; the JSON is the verdict. Rules come from stand5_mdp.py (loaded by
# path: the bhl_robust.tasks package needs a simulator to import), which calls
# stand4_mdp.py's. A seed JSON is never overwritten.
stand5_eval() {
    REPO="$REPO" OMP_NUM_THREADS=2 "$HOSTPY" - "$@" <<'PY'
import importlib.util, json, os, sys
mode, run, out = sys.argv[1], sys.argv[2], sys.argv[3]
killed = len(sys.argv) > 4 and sys.argv[4] == "1"
bad = len(sys.argv) > 5 and sys.argv[5] == "1"
spec = importlib.util.spec_from_file_location(
    "stand5_mdp", os.environ["REPO"] + "/src/bhl_robust/tasks/stand5_mdp.py")
s5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s5)
if mode != "kill" and os.path.exists(out):
    sys.exit(f"refusing to overwrite {out}")
sc = s5.sm.read_scalars(run)
last = max([s for s, _ in sc.get(s5.s4.TAG_LEN, [])] or [-1])
if mode == "kill":
    res = s5.evaluate_kill(sc)
    res["ready"] = last >= s5.sm.KILL_AT_ITER
else:
    res = s5.evaluate_seed5(sc, killed=killed, bad_config=bad)
res.update({"mode": mode, "run_dir": run, "latest_iter_logged": last,
            "task": "CubeToShelfStand5 (blind)", "note": s5.TASK5_NOTE,
            "labels": s5.LABELS_NOTE5, "design": list(s5.DESIGN_S),
            "rule": s5.PREDECLARED_RULE5, "stand4_rule": s5.STAND4_RULE})
json.dump(res, open(out, "w"), indent=1, default=str)
PY
}
jget() { "$HOSTPY" -c "import json,sys; print(json.load(open(sys.argv[1])).get(sys.argv[2]))" "$1" "$2"; }

# check_smoke_bytes SMOKE_JOB: 0 iff that smoke's verdict JSON reads PASS, its
# sha256 list holds every BYTES_REQUIRED file, and every file in the list except
# __init__.py and the smoke's two launcher scripts matches this working tree.
# Prints the reason or sha256sum -c's lines.
check_smoke_bytes() {
    local job="$1" v h f list
    if [ -z "$job" ]; then echo "no smoke job id: STAND5_SMOKE_JOB is not set"; return 1; fi
    v="$SMOKE_OUT/verdict_${job}.json"; h="$SMOKE_OUT/sha256_${job}.txt"
    if [ ! -f "$v" ] || [ ! -f "$h" ]; then echo "smoke $job: $v or $h is missing"; return 1; fi
    if [ "$(jget "$v" verdict 2>/dev/null)" != "PASS" ]; then echo "smoke $job: verdict is not PASS ($v)"; return 1; fi
    list=$(grep -v -E '  (src/bhl_robust/tasks/__init__\.py|slurm/repo20260923/gpu_v2_stand5_smoke\.sbatch|slurm/inner/inner_v2_stand5_smoke\.sh)$' "$h" || true)
    for f in $BYTES_REQUIRED; do
        if ! printf '%s\n' "$list" | awk -v f="$f" '$2 == f {found = 1} END {exit !found}'; then
            echo "smoke $job: $f is not in $h"; return 1
        fi
    done
    ( cd "$REPO" && printf '%s\n' "$list" | sha256sum -c --strict - )
}

kill_tree() {  # TERM then KILL the training subtree (Isaac can ignore TERM)
    local pids p kids
    pids="$1"; kids="$1"
    while [ -n "$kids" ]; do
        kids=$(for p in $kids; do pgrep -P "$p" || true; done | tr '\n' ' ' | xargs || true)
        pids="$pids $kids"
    done
    kill -TERM $pids 2>/dev/null || true
    sleep 60
    kill -KILL $pids 2>/dev/null || true
}

check_runner() {  # the run trains Stand5 with Stand3's runner, or it stops
    local run="$1" i out js
    js="$OUTDIR/runner_check_s${SEED}_${JOBTAG}.json"
    for i in $(seq 1 60); do
        [ -f "$run/params/agent.yaml" ] && [ -f "$run/params/env.yaml" ] && break
        sleep 30
    done
    out=$(REPO="$REPO" OMP_NUM_THREADS=2 "$HOSTPY" - "$run" "$js" <<'PY'
import importlib.util, json, os, sys
run, out = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location(
    "stand5_mdp", os.environ["REPO"] + "/src/bhl_robust/tasks/stand5_mdp.py")
s5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s5)
try:
    r = s5.config_check5(open(os.path.join(run, "params", "agent.yaml")).read(),
                         open(os.path.join(run, "params", "env.yaml")).read())
except OSError as exc:
    r = {"ok": False, "failed": [f"unreadable params: {exc}"], "checks": {}}
if not os.path.exists(out):
    json.dump(dict(r, run_dir=run), open(out, "w"), indent=1)
print("PASS" if r["ok"] else "FAIL " + ",".join(r["failed"]))
PY
) || out="FAIL (checker error)"
    if [ "$out" = "PASS" ]; then
        echo "RUNNER-CHECK $RUN_NAME: PASS (stand5_mdp.config_check5: Stand3 runner, Stand4 terms, object_zaxis_a/b last in policy and critic, roll-proof lift curriculum; $js)"
        return 0
    fi
    echo "RUNNER-CHECK $RUN_NAME: $out -- params/*.yaml are not Stand5's; stopping ($(date))"
    echo "$run" > "$BADCFG_FLAG"
    return 1
}

watch_kill() {  # runner check, then the predeclared kill rule at model_1000
    local run js tmp v i checked=0
    while kill -0 "$TRAIN_PID" 2>/dev/null; do
        run=$(find_run)
        if [ -n "$run" ] && [ "$checked" = 0 ]; then
            checked=1
            if ! check_runner "$run"; then kill_tree "$TRAIN_PID"; return 0; fi
        fi
        if [ -n "$run" ] && [ -f "$run/model_${KILL_AT}.pt" ]; then
            js="$OUTDIR/kill_s${SEED}_${JOBTAG}.json"
            tmp="$js.pending"
            # the event writer flushes on its own clock: wait (<= 30 min) until
            # iteration 1000 is in the events file
            for i in $(seq 1 60); do
                stand5_eval kill "$run" "$tmp" || true
                [ -f "$tmp" ] && [ "$(jget "$tmp" ready)" = "True" ] && break
                sleep 30
            done
            [ -f "$tmp" ] && [ ! -e "$js" ] && mv "$tmp" "$js"
            v=$( [ -f "$js" ] && jget "$js" verdict || echo UNKNOWN)
            if [ ! -f "$js" ] || [ "$(jget "$js" ready 2>/dev/null)" != "True" ]; then
                echo "KILL-RULE $RUN_NAME: UNEVALUATED -- iteration $KILL_AT never reached the events file; training continues"
                return 0
            fi
            echo "KILL-RULE $RUN_NAME @model_${KILL_AT}: $v ep_len=$(jget "$js" ep_len) time_out=$(jget "$js" time_out) success=$(jget "$js" success) fallen=$(jget "$js" fallen) reasons=$(jget "$js" reasons) length_clause_skipped=$(jget "$js" length_clause_skipped) ($js)"
            if [ "$v" = "KILL" ]; then
                echo "$js" > "$KILLED_FLAG"
                echo "KILL-RULE $RUN_NAME: stopping training ($(date))"
                kill_tree "$TRAIN_PID"
            fi
            return 0
        fi
        sleep 120
    done
}

echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | envs=$NUM_ENVS | iters=$MAX_ITER | HEAD $(git -C "$REPO" rev-parse --short HEAD 2>/dev/null || echo '?') | $(date) ==="
( cd "$REPO" && sha256sum src/bhl_robust/tasks/stand5_mdp.py src/bhl_robust/tasks/stand4_mdp.py \
      src/bhl_robust/tasks/task_v2_env_cfg.py src/bhl_robust/tasks/__init__.py \
      src/bhl_robust/tasks/stand_mdp.py src/bhl_robust/tasks/coop_lift_mdp.py \
      src/bhl_robust/tasks/coop_lift_env_cfg.py src/bhl_robust/quat_order.py \
      assets/cloth/berkeley_humanoid_lite_hand_colliders.usda ) | sed 's/^/sha256 /'
# BYTES-CHECK (header): the bytes the chained smoke validated, or no training.
BYTES_LOG="$OUTDIR/bytes_check_s${SEED}_${JOBTAG}.txt"
if check_smoke_bytes "${STAND5_SMOKE_JOB:-}" > "$BYTES_LOG" 2>&1; then
    sed 's/^/bytes /' "$BYTES_LOG"
    echo "BYTES-CHECK $RUN_NAME: PASS (smoke $STAND5_SMOKE_JOB verdict PASS; the Stand5 files match its sha256 list; $BYTES_LOG)"
else
    sed 's/^/bytes /' "$BYTES_LOG"
    echo "BYTES-CHECK $RUN_NAME: FAIL -- refusing to start: smoke ${STAND5_SMOKE_JOB:-<unset>} did not PASS or the files differ from what it validated ($BYTES_LOG)"
    rm -f "$STAMP"
    exit 1
fi
v60_boot_gate
bhl_exec "$REPO/slurm/inner/train.sh" &
TRAIN_PID=$!
watch_kill &
WATCH_PID=$!
train_rc=0
wait "$TRAIN_PID" || train_rc=$?
# After a rule kill or a failed runner check, let the watcher finish kill_tree.
if [ -f "$KILLED_FLAG" ] || [ -f "$BADCFG_FLAG" ]; then :; else kill "$WATCH_PID" 2>/dev/null || true; fi
wait "$WATCH_PID" 2>/dev/null || true
echo "=== training ended rc=$train_rc: $(date) ==="

killed=0; [ -f "$KILLED_FLAG" ] && killed=1
bad=0; [ -f "$BADCFG_FLAG" ] && bad=1
run=$(find_run)
rm -f "$STAMP"
if [ -z "$run" ]; then
    echo "RESULT $RUN_NAME: INCOMPLETE -- no run directory *_${RUN_NAME} under $LOGROOT"
    exit 1
fi
ARRAY_ID="${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID}}"
js="$OUTDIR/result_s${SEED}_${ARRAY_ID}.json"
if stand5_eval seed "$run" "$js" "$killed" "$bad" && [ -f "$js" ]; then
    status=$( [ "$(jget "$js" decided)" = "True" ] && echo DECIDED || echo "INCOMPLETE (not a result)")
    echo "RESULT $RUN_NAME: $status complete=$(jget "$js" complete) passes=$(jget "$js" passes)" \
         "killed=$(jget "$js" killed) bad_config=$(jget "$js" bad_config)" \
         "window_finite=$(jget "$js" window_finite) has_value=$(jget "$js" has_value)" \
         "success_last200=$(jget "$js" success) time_out_last200=$(jget "$js" time_out)" \
         "ep_len_last200=$(jget "$js" ep_len) fallen_last200=$(jget "$js" fallen)" \
         "lift_level_last200=$(jget "$js" lift_level) upright_gate_last200=$(jget "$js" upright_gate)" \
         "last_iter=$(jget "$js" last_iter) ($js)"
else
    echo "RESULT $RUN_NAME: INCOMPLETE -- could not read $run (or $js already existed)"
    exit 1
fi

# Both seeds of THIS array in: the predeclared two-seed verdict (whichever task
# finishes last writes it; exclusive create, never overwritten). Results are
# tagged by array id, so a resubmitted seed is never pooled with an earlier
# attempt of the other.
if [ -f "$OUTDIR/result_s0_${ARRAY_ID}.json" ] && [ -f "$OUTDIR/result_s1_${ARRAY_ID}.json" ]; then
    REPO="$REPO" OMP_NUM_THREADS=2 "$HOSTPY" - "$OUTDIR" "$ARRAY_ID" <<'PY'
import importlib.util, json, os, sys
d, aid = sys.argv[1], sys.argv[2]
spec = importlib.util.spec_from_file_location(
    "stand5_mdp", os.environ["REPO"] + "/src/bhl_robust/tasks/stand5_mdp.py")
s5 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s5)
seeds = [json.load(open(os.path.join(d, f"result_s{s}_{aid}.json"))) for s in s5.STAND5_SEEDS]
verdict = s5.pair_verdict5(seeds)
out = os.path.join(d, f"result_pair_{aid}.json")
try:
    with open(out, "x") as f:
        json.dump({"verdict": verdict, "seeds": seeds, "task": "CubeToShelfStand5 (blind)",
                   "design": list(s5.DESIGN_S), "rule": s5.PREDECLARED_RULE5,
                   "stand4_rule": s5.STAND4_RULE, "as_applied": s5.AS_APPLIED5,
                   "precedence": s5.PRECEDENCE_NOTE, "complete_def": s5.COMPLETE_NOTE,
                   "tilt_def": s5.TILT_NOTE, "labels": s5.LABELS_NOTE5, "note": s5.TASK5_NOTE},
                  f, indent=1)
except FileExistsError:
    verdict = json.load(open(out))["verdict"]
print(f"RESULT CubeToShelfStand5 blind (2 seeds): {verdict} ({out})")
PY
fi
# A run stopped by its predeclared kill rule is an outcome, not a job failure.
[ "$killed" = 1 ] && exit 0
[ "$bad" = 1 ] && exit 1
exit "$train_rc"
