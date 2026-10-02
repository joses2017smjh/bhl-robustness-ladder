#!/bin/bash
#SBATCH --job-name=platecross
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
# 2026-10-02: Mission 7 learned crossing (workstream m7-platecross). Frozen design: SLURM_JOBS.md, 'User approval
# recorded 2026-10-02 14:15', item (B). Both scripted crossing plans failed Mission 7's plate bench (M2 8/64,
# M3 16/41). This launcher FINE-TUNES the qualified turning checkpoint arms-turngait-clock-s2 (task
# Velocity-BHL-Arms-TurnGaitClock-v0 = R1: 77-observation actor with the gait clock; QUALIFIED by
# cpu_turn_qualify: turn 10/10, walk 2/3, push 9/60) from its model_5999.pt for 3000 iterations (->
# model_8998.pt), seeds 0-2, on flat ground scattered with Mission 7's own plates, then runs the unchanged turn
# qualification per seed and the selection after the last seed. Resources: the turning launchers' (gpu_turngait_v5
# fine-tunes the same 3000 iterations in 10 h; 12 h here because triangle-mesh contacts cost more than a plane).
#
# TASK Velocity-BHL-Arms-PlateCross-v0 (src/bhl_robust/tasks/platecross_env_cfg.py) = R1's env
# (arms_env_cfg.HumanoidTurnGaitClockCfg: rewards, command mix, observations including the 77-obs clock, events
# including the fixed pushes, domain randomization) UNCHANGED except the terrain (scene.terrain only:
# terrain_type generator, the plates generator, spawn on every tile, no MDL visual material).
# Plates exactly as Mission 7's (bhl_robust.mission.layout.world_xml, src/bhl_robust/mission/layout.py:152-153):
# round disc r 0.24 m and square 0.48 x 0.48 m, both 0.03 m high, resting on the floor, squares axis-aligned.
# DECLARED field (2026-10-02, before any training; chosen, not tuned; src/bhl_robust/tasks/platecross_terrain.py):
# square lattice, pitch 1.2 m; 8.4 m tiles (7 x 7 lattice points, centred); round / square in a checkerboard
# inside each tile; the tile-centre point left empty (the spawn point; nearest plate edge 0.96 m away); 24 round +
# 24 square per tile = 0.68 plates/m^2, 14.0 % of the floor (a Mission 7 door cell: 0.69-0.89 plates/m^2,
# 14-18 %); round plate = a 64-gon with its vertices on r = 0.24 m; 10 x 10 identical tiles inside the flat 20 m
# border; generator seed 0 (all seeds train on the same field). The plates are baked into the static terrain
# collision mesh (/World/ground/terrain/mesh: CollisionAPI, no RigidBodyAPI, i.e. a PhysX static rigid body) with
# the ground's own material (friction 1.0, restitution 0). Blind policy: the 77 actor observations are R1's.
# Disclosed baseline: clock-s2 drifts on straight walks (qualify walk drift 19.8 / 10.8 / 2.4 deg); the
# fine-tunes inherit it. Disclosed: R1's feet_swing_height reward (unchanged) measures foot height in absolute
# world z (gait_clock_mdp.feet_swing_height: flat plane, ground z = 0), so over a plate its swing-clearance
# target is 0.03 m lower relative to the plate top; feet in contact are excluded, and it is the only absolute-z
# term among R1's 19 reward and 2 termination terms. Not covered by the smoke's training (64 envs): training
# 4096 envs on the plates mesh; the smoke's scene probe builds and steps the task at 4096 envs (no learning).
#
# PREDECLARED RULE (frozen 2026-10-02, before any PlateCross run exists; the task's text verbatim):
#   the three fine-tuned final checkpoints go through the unchanged turn qualification (turn_test v2 +
#   cpu_turn_qualify); the qualified seed with the lowest push-fall rate (tie: lowest seed index) is the
#   SINGLE stage gait run on bench v2 under bench v2's rule (N = 41; >= 39/41 clears, 0 falls, >= 11/10/9/7
#   per heading) with the generic stage-gait override; no qualified seed -> NEGATIVE (no bench).
#   Labels: LEARNED gait (fine-tuned from clock-s2 on plates); MuJoCo gates; bench v2's 180-deg timing caveat
#   applies.
# READING, stated before any run (the frozen rule's "unchanged turn qualification (turn_test v2 +
#   cpu_turn_qualify)" counted per seed as R1 / v5 count it): A seed is qualified iff it both PASSES turn_test v2
#   (its <run>.json verdict, under the unchanged v2 protocol) AND is QUALIFIED by cpu_turn_qualify's unchanged rule
#   (its <run>__qualify.json verdict), as the R1 / v5 joint rule counts a seed (turngait_r12_verdict.seed_counts);
#   a seed that fails v2 does not count even if cpu_turn_qualify says QUALIFIED. The push-fall rate is falls / n
#   of the qualify JSON's push clause (n = 60). Every seed needs its training record, v2 JSON (with the unchanged
#   v2 protocol's rule block) and qualify JSON, all for the same exported deploy.yaml; otherwise the selection is
#   INCOMPLETE and nothing is written.
# Training counts only if the run dir holds the final checkpoint and the fresh training log shows the PlateCross
# task id, the parent checkpoint being loaded, the plates terrain marker, the feet_gait term and the push event,
# with an empty guard list and both env.yaml checks OK (R1's recipe via `bhl_robust.eval.gait_clock
# check-recipe --arm R1` against TurnBoth-s0, as R12; and the whole env.yaml = arms-turngait-clock-s2's except
# the training seed and the declared terrain fields, via platecross_select.py check-env). No gate runs otherwise.
# The bench v2 run of the selected gait is the COORDINATOR's later submission (it needs workstream A's generic
# stage-gait override); this launcher never builds or runs it.
#
# Verdicts come from JSON, never exit codes (Isaac exits 0 on a crash). Per seed, all written once:
#   <RES>/training/<run>.json          the training record (from the FRESH log; hard link, never overwritten)
#   <RES>/turn-test-v2/<run>.json      turn_test.py --protocol v2 (gates with the qualify verdict; never overwritten)
#   <RES>/qualify/<run>__qualify.json  cpu_turn_qualify.sbatch, unchanged (+ __v2x.json, __p0.5_n60.csv)
#   <RES>/selection.json               written ONCE (O_EXCL) by whichever seed's job completes the set:
#                                      SELECTED (run, export path, sha256s) or NEGATIVE (no bench);
#                                      INCOMPLETE is printed, never written. An existing one is re-printed.
# RES = results/repo-gpu-20260923/platecross-20261002; runs arms-platecross-clocks2-s{0,1,2}.
# Never overwritten or retrained: a run dir that holds the final checkpoint is reused (its stored training record
# must pass the same check); a run dir WITHOUT it makes the job refuse (move it aside to retrain:
# cpu_turn_qualify needs exactly one run dir per name). Scored seeds touched only by the real array: turn_test v2
# reset seeds 0-2 / walk 0, v2x reset 10-14 / walk 10-12, push eval seeds 0-9.
#
# Modes (env vars):
#   PLATECROSS_SMOKE=1  MAX_ITER=3, NUM_ENVS=64, run <prefix>-s<seed>-smoke (resumed from clock-s2 -> model_6001.pt),
#                       outputs under
#                       /nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/campaign-m7-platecross/platecross-smoke
#                       (the dir must end in -smoke; the job name must contain 'smoke'; its time limit must be
#                       <= 1 h). Checks: the training log (task id, parent loaded, plates marker, R1 terms, 77 / 80
#                       obs), both env.yaml checks and the training record; the SCENE PROBE (platecross_select.py
#                       probe-scene in the container at 4096 envs: plate count, shape, radius / side, height and
#                       positions measured in the imported terrain mesh, the static collider, spawn clearance, 77
#                       obs; 64 robots placed on plates must stand 0.03 m (+/- 0.01) higher than 64 placed on the
#                       floor, i.e. the plates carry the feet; 50 zero-action steps with no link below the floor
#                       and no PhysX overflow); the 77-obs export (stamped clock, ONNX width,
#                       the ONNX = the final checkpoint's actor, deploy = clock-s2's but the policy path); short
#                       clock-aware MuJoCo plumbing rollouts on
#                       exploration seed 100 only (gait_clock smoke, turn_test v1, run_eval via gait_clock run-eval);
#                       and the selection code on the smoke outputs (INCOMPLETE, nothing written), on three
#                       stand-in seeds built from R1's real qualify / v2 JSONs (must select stand-in seed 2, write
#                       once, refuse a second write), and on a SYNTHETIC variant whose stand-in seed 1 keeps its
#                       real v2 FAIL with its qualify JSON relabelled QUALIFIED 0/60 (must still select seed 2: a
#                       seed that fails v2 never counts). One "SMOKE: PASS|FAIL (...)" line; exit 0 iff PASS.
# Side effects outside RES, as for every humanoid fine-tune and export: the run dir <ts>_<run> in
# $UPSTREAM/logs/rsl_rl/humanoid, Isaac's log there, the override file, and the export's rewrite of
# $UPSTREAM/configs/policy_latest.yaml (serialised by the export lock shared with gpu_turngait_r12.sbatch).
#   Submit (coordinator) in ONE command: a fresh smoke of the exact code being submitted, with the real array
#   chained afterok on it (slurmctld forgets a finished job after ~5 min, and the code may have changed):
#     cd /nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder && source slurm/_env.sh && \
#     S=$(slurm_clean sbatch --parsable --job-name=platecross-smoke --array=0 --time=01:00:00 \
#         --export=ALL,PLATECROSS_SMOKE=1 slurm/repo20260923/gpu_platecross.sbatch) && \
#     slurm_clean sbatch --dependency=afterok:${S%%;*} --export=ALL,PLATECROSS_SMOKE=0 \
#         slurm/repo20260923/gpu_platecross.sbatch
set -euo pipefail
export BHL_STACK=v51
source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh
PYH=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
LOGROOT=$UPSTREAM/logs/rsl_rl/humanoid
JOBTAG=${SLURM_JOB_ID:-local$$}
SMOKE=${PLATECROSS_SMOKE:-0}
TASK_ID=Velocity-BHL-Arms-PlateCross-v0
PREFIX=arms-platecross-clocks2
PARENT_RUN=arms-turngait-clock-s2
PARENT_ITER=5999
FT_ITERS=3000
OBS_POLICY=77
OBS_CRITIC=80
PROBE_NUM_ENVS=4096
SELECT=$REPO/scripts/bench/platecross_select.py
R1_RES=$REPO/results/repo-gpu-20260923/turngait-r12-20261001
SMOKE_RES=/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/campaign-m7-platecross/platecross-smoke
if [ "$SMOKE" = 1 ]; then
    RES=${PLATECROSS_OUT_DIR:-$SMOKE_RES}
    case "$RES" in *-smoke) ;; *) echo "SMOKE: FAIL (smoke outputs must go to a directory ending in -smoke, not $RES)"; exit 1;; esac
else
    RES=$REPO/results/repo-gpu-20260923/platecross-20261002
fi
IDX=${SLURM_ARRAY_TASK_ID:-0}
case $IDX in
    0|1|2) SEED=$IDX;;
    *) echo "PLATECROSS RESULT: FAIL (array index $IDX outside 0-2)"; exit 2;;
esac
if [ "$SMOKE" = 1 ]; then
    export MAX_ITER=3 NUM_ENVS=64 RUN_NAME="$PREFIX-s${SEED}-smoke"
else
    export MAX_ITER=$FT_ITERS RUN_NAME="$PREFIX-s${SEED}"
    unset NUM_ENVS                                  # the task's own env count, as R1 trained
fi
FINAL_CKPT="model_$((PARENT_ITER + MAX_ITER - 1)).pt"
export TASK="$TASK_ID" SEED
mkdir -p "$RES"/{training,turn-test-v2,qualify}

# secs_of_limit <Slurm %l>: D-HH:MM:SS, HH:MM:SS or MM:SS -> seconds (empty if unparseable)
secs_of_limit() {
    echo "$1" | awk -F'[-:]' '/^[0-9]+(-[0-9]+)?(:[0-9]+){0,2}$/ {
        if ($0 ~ /-/) { s = $1*86400 + $2*3600 + $3*60 + $4 } else if (NF == 3) { s = $1*3600 + $2*60 + $3 }
        else if (NF == 2) { s = $1*60 + $2 } else { s = $1*60 }
        print s }'
}
if [ "$SMOKE" = 1 ]; then
    case "${SLURM_JOB_NAME:-}" in *smoke*) ;; *) echo "SMOKE: FAIL (a smoke job's name must contain 'smoke', not '${SLURM_JOB_NAME:-unset}')"; exit 1;; esac
    tl=$(squeue -h -j "${SLURM_JOB_ID:-0}" -o %l 2>/dev/null | head -1 || true)
    tls=$(secs_of_limit "$tl")
    if [ -n "$tls" ] && [ "$tls" -gt 3600 ]; then echo "SMOKE: FAIL (smoke time limit $tl is over 1 h)"; exit 1; fi
    [ -n "$tls" ] || echo "note: smoke time limit not read from squeue ('$tl')"
fi

if [ "$SMOKE" != 1 ] && [ -f "$RES/selection.json" ]; then
    echo "selection already recorded, not overwritten: $RES/selection.json"
    ( cd "$REPO" && "$PYH" "$SELECT" select --res "$RES" ) || true
    exit 0
fi

# ---------------------------------------------------------------- parent, run dir
mapfile -t parents < <(ls -d "$LOGROOT"/*_"$PARENT_RUN" 2>/dev/null | sort)
if [ "${#parents[@]}" -ne 1 ] || [ ! -f "${parents[0]}/model_$PARENT_ITER.pt" ] || [ ! -f "${parents[0]}/params/env.yaml" ] \
    || [ ! -f "${parents[0]}/exported/deploy.yaml" ]; then
    echo "PLATECROSS RESULT: FAIL (parent: expected exactly one $PARENT_RUN dir with model_$PARENT_ITER.pt, params/env.yaml and exported/deploy.yaml; found ${#parents[@]} dirs)"
    [ "$SMOKE" = 1 ] && echo "SMOKE: FAIL (parent)"
    exit 1
fi
PARENT_PATH=${parents[0]}
PARENT_DIR=$(basename "$PARENT_PATH")
PARENT_SHA=$(sha256sum "$PARENT_PATH/model_$PARENT_ITER.pt" | cut -d' ' -f1)
REF_RUN=$(ls -d "$LOGROOT"/*_arms-turn-turnboth-s0 2>/dev/null | sort | tail -1 || true)
find_run() { ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null | sort | tail -1; }
existing=$(find_run || true)
if [ "$SMOKE" != 1 ]; then
    # `|| true` inside: with no run dir yet, ls exits 2, and under pipefail the bare pipeline killed every real task
    # in 4 s with an empty log (array 21517362, 2026-10-02); the smoke never runs this block.
    n_runs=$( (ls -d "$LOGROOT"/*_"$RUN_NAME" 2>/dev/null || true) | wc -l)
    if [ "$n_runs" -gt 1 ]; then
        echo "PLATECROSS RESULT: FAIL ($n_runs run dirs named $RUN_NAME; cpu_turn_qualify needs exactly one; seed stays INCOMPLETE)"; exit 1
    fi
    if [ -n "$existing" ] && [ ! -f "$existing/$FINAL_CKPT" ]; then
        echo "PLATECROSS RESULT: FAIL (an incomplete run dir exists: $existing has no $FINAL_CKPT; it is never retrained or overwritten -- move it aside to retrain; seed stays INCOMPLETE)"; exit 1
    fi
fi

# ---------------------------------------------------------------- GPU from here
# A fine-tune of an MLP [256,128,128] actor-critic: a recurrent / RND / symmetry overlay forwarded from the
# submitting shell (BHL_FORWARD_VARS) would make runner.load fail on the state_dict and Isaac still exit 0.
unset BHL_POLICY BHL_RND BHL_SYMMETRY
source "$REPO/slurm/dr_levels.sh"
setup_node_cache
CACHE="${TMPDIR:-/tmp}/platecross-$JOBTAG"
source "$REPO/slurm/inner/v60_boot_gate.sh"
V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate

# Exports serialise on the lock gpu_turngait_r12.sbatch uses: export_one_humanoid.sh copies the shared
# $UPSTREAM/configs/policy_latest.yaml, which two exports together would race on (mkdir is atomic on Lustre/NFS;
# stale after 20 min > the 16-min export cap).
EXPORT_LOCK="$WORKSPACE/.turngait-r12-export.lock"
_pc_export_release() {
    [ "$(cut -d' ' -f1 "$EXPORT_LOCK/holder" 2>/dev/null)" = "$JOBTAG" ] && rm -rf "$EXPORT_LOCK"
    return 0
}
trap "_v60_release '$WORKSPACE/.v51-boot.lock'; _pc_export_release" EXIT
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
{ dr_overrides_scale 1.0
  printf 'agent.resume=true\nagent.load_run=%s\nagent.load_checkpoint=model_%s.pt\n' "$PARENT_DIR" "$PARENT_ITER"; } > "$OVERRIDE_FILE"
export BHL_FORWARD_VARS="$BHL_FORWARD_VARS RUN_DIR"
echo "=== node: $(hostname) | $RUN_NAME | task=$TASK | smoke=$SMOKE | +$MAX_ITER iters from $PARENT_DIR/model_$PARENT_ITER.pt (sha256 ${PARENT_SHA:0:12}) -> $FINAL_CKPT | RES $RES | HEAD $(git -C "$REPO" rev-parse --short HEAD) | $(date) ==="
cat "$OVERRIDE_FILE"
TLOG="${TMPDIR:-/tmp}/platecross-$JOBTAG.log"; : > "$TLOG"
FRESH=0
if [ "$SMOKE" != 1 ] && [ -n "$existing" ] && [ -f "$existing/$FINAL_CKPT" ]; then
    echo "already trained: $existing (reused; no fresh training log in this job)"
else
    bhl_exec "$REPO/slurm/inner/train.sh" 2>&1 | tee "$TLOG" || true
    FRESH=1
    echo "=== training done: $(date) ==="
fi
run=$(find_run || true)

# ---------------------------------------------------------------- training evidence
TASK_RE="^task=$TASK_ID run=$RUN_NAME seed=$SEED "
PARENT_RE="Loading model checkpoint from: .*${PARENT_DIR}/model_${PARENT_ITER}\.pt"
PLATES_RE='^\[platecross_terrain\] plates tile 8\.4 x 8\.4 m: 24 round \(r 0\.24 m, 64-gon\) \+ 24 square \(0\.48 m\), height 0\.03 m, pitch 1\.2 m, centre clear'
GAIT_RE="\| +[0-9]+ +\| +feet_gait +\| +0\.5 +\|"
PUSH_RE="\| +[0-9]+ +\| +push_robot +\| +\(5\.0, 9\.0\) +\|"
guard() {   # <log>: reasons the log is NOT this task's fine-tune (empty = ok)
    grep -qE "Group: 'policy' \(shape: \($OBS_POLICY,\)\)" "$1" || printf ' policy-obs-not-%s' "$OBS_POLICY"
    grep -qE "Group: 'critic' \(shape: \($OBS_CRITIC,\)\)" "$1" || printf ' critic-obs-not-%s' "$OBS_CRITIC"
    grep -qE "\| +[0-9]+ +\| +feet_swing_height +\| +-20\.0 +\|" "$1" || printf ' no-feet_swing_height--20.0-row'
    grep -qE "\| +[0-9]+ +\| +feet_air_time +\| +0\.0 +\|" "$1" || printf ' feet_air_time-weight-not-0.0'
    grep -qE "\| +base_velocity +\| +UniformVelocityCommand +\|" "$1" || printf ' command-class-is-not-UniformVelocityCommand'
    grep -q "push_levels" "$1" && printf ' unexpected-push_levels-curriculum'
    grep -qE "^\[gait_clock_mdp\] feet_gait bodies \['[^']*left[^']*', '[^']*right[^']*'\] offsets \[0\.0, 0\.5\]" "$1" || printf ' feet_gait-did-not-run-on-(left,right)'
    grep -qE "^\[gait_clock_mdp\] feet_swing_height bodies" "$1" || printf ' feet_swing_height-did-not-run'
    grep -q "m7-platecross id NOT registered" "$1" && printf ' platecross-id-not-registered'
    grep -qE "train.sh: reached [0-9]+ logged iterations" "$1" || printf ' no-logged-iterations'
    return 0
}
TJSON="$RES/training/$RUN_NAME.json"
if [ "$SMOKE" = 1 ]; then TJSON="$RES/training/$RUN_NAME.$JOBTAG.json"; fi   # each smoke writes and re-checks its own
why_train=""
rline="GAIT-CLOCK RECIPE: not checked (no env.yaml or no TurnBoth-s0 reference)"
eline="PLATECROSS ENV: not checked (no env.yaml)"
if [ -n "$run" ] && [ -f "$run/params/env.yaml" ]; then
    # R1's recipe (the R12 check, against TurnBoth-s0), and the whole env.yaml against the parent's
    [ -n "$REF_RUN" ] && rline=$( (cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 "$PYH" -m bhl_robust.eval.gait_clock \
        check-recipe --env-yaml "$run/params/env.yaml" --arm R1 --ref-env-yaml "$REF_RUN/params/env.yaml" 2>&1 | tail -1) || true)
    smoke_flag=""; [ "$SMOKE" = 1 ] && smoke_flag="--smoke"
    eline=$( (cd "$REPO" && OMP_NUM_THREADS=1 "$PYH" "$SELECT" check-env --env-yaml "$run/params/env.yaml" \
        --parent-env-yaml "$PARENT_PATH/params/env.yaml" $smoke_flag 2>&1 | tail -1) || true)
fi
echo "$rline"
echo "$eline"
case "$rline" in "GAIT-CLOCK RECIPE: OK"*) ;; *) why_train="$why_train recipe-check-failed";; esac
case "$eline" in "PLATECROSS ENV: OK"*) ;; *) why_train="$why_train env-check-failed";; esac
if [ "$FRESH" = 1 ]; then
    ckpt_ok=false; { [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; } && ckpt_ok=true
    task_ok=false; grep -qE "$TASK_RE" "$TLOG" && task_ok=true
    parent_ok=false; grep -qE "$PARENT_RE" "$TLOG" && parent_ok=true
    plates_ok=false; grep -qE "$PLATES_RE" "$TLOG" && plates_ok=true
    gait_ok=false; grep -qE "$GAIT_RE" "$TLOG" && gait_ok=true
    push_ok=false; grep -qE "$PUSH_RE" "$TLOG" && push_ok=true
    why_train="$why_train$(guard "$TLOG")"
    [ "$ckpt_ok" = true ] || why_train="$why_train no-$FINAL_CKPT"
    [ "$task_ok" = true ] || why_train="$why_train task-id-not-in-log"
    [ "$parent_ok" = true ] || why_train="$why_train parent-checkpoint-not-loaded"
    [ "$plates_ok" = true ] || why_train="$why_train no-plates-terrain-marker"
    [ "$gait_ok" = true ] || why_train="$why_train no-feet_gait-0.5-row"
    [ "$push_ok" = true ] || why_train="$why_train no-push_robot-(5.0,9.0)-row"
    { head -n 400 "$TLOG"; echo "[... cut ...]"; tail -n 60 "$TLOG"; } > "$RES/training/$RUN_NAME.$JOBTAG.log-excerpt.txt"
    grep -hE "^\[platecross_terrain\]|^\[gait_clock_mdp\]|Loading model checkpoint" "$TLOG" | head -4 || true
    if [ "$ckpt_ok" = true ]; then
        ( cd "$REPO" && "$PYH" "$SELECT" write-record --out "$TJSON" --run "$RUN_NAME" --task "$TASK_ID" --seed "$SEED" \
            --final-ckpt "$FINAL_CKPT" --run-dir "$run" --job "$JOBTAG" --parent-dir "$PARENT_PATH" \
            --parent-ckpt "model_$PARENT_ITER.pt" --parent-sha256 "$PARENT_SHA" --ckpt-ok "$ckpt_ok" --task-ok "$task_ok" \
            --parent-ok "$parent_ok" --plates-ok "$plates_ok" --gait-ok "$gait_ok" --push-ok "$push_ok" \
            --guard="$why_train" --env-check="$eline" --recipe-check="$rline" ) || true
    else
        echo "training incomplete: no training record written (it is written only for a run that reached $FINAL_CKPT)"
    fi
fi
record_problems() {   # <record> <run dir>: reasons the stored record lets no gate run (empty = ok)
    ( cd "$REPO" && "$PYH" "$SELECT" check-record --record "$1" --run "$RUN_NAME" --final-ckpt "$FINAL_CKPT" \
        --run-dir "$2" 2>&1 | tail -1 ) || printf ' training-record-check-crashed'
    return 0
}

if [ "$SMOKE" != 1 ]; then
    { [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; } || {
        echo "PLATECROSS RESULT: FAIL (training incomplete: no $FINAL_CKPT for $RUN_NAME; seed stays INCOMPLETE)"; exit 1; }
    if [ "$FRESH" = 1 ] && [ -n "${why_train// /}" ]; then
        echo "PLATECROSS RESULT: FAIL (training log / env.yaml is not the PlateCross fine-tune:$why_train; no gates run, seed stays INCOMPLETE)"; exit 1
    fi
    if [ "$FRESH" != 1 ]; then
        [ -f "$TJSON" ] || { echo "PLATECROSS RESULT: FAIL (already trained but no training record $TJSON and no fresh log; seed stays INCOMPLETE)"; exit 1; }
        case "$why_train" in *check-failed*) echo "PLATECROSS RESULT: FAIL ($why_train; no gates run, seed stays INCOMPLETE)"; exit 1;; esac
    fi
    why_rec=$(record_problems "$TJSON" "$run")
    if [ -n "${why_rec// /}" ]; then
        echo "PLATECROSS RESULT: FAIL (training record $TJSON: $why_rec; no gates run, seed stays INCOMPLETE)"; exit 1
    fi
fi

# ---------------------------------------------------------------- smoke: the scene probe (Isaac, in the container)
if [ "$SMOKE" = 1 ]; then
    mkdir -p "$RES/probe"
    PROBE_JSON="$RES/probe/$RUN_NAME.$JOBTAG.json"; PROBE_LOG="$RES/probe/$RUN_NAME.$JOBTAG.log"
    rm -f "$PROBE_JSON"
    export BHL_FORWARD_VARS="$BHL_FORWARD_VARS PROBE_JSON PROBE_LOG PROBE_ENVS PROBE_TMO"
    INNER_PROBE='set -uo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
echo "=== probe-scene $TASK ($PROBE_ENVS envs) | timeout ${PROBE_TMO}s | $(date) ==="
timeout -k 60 "$PROBE_TMO" "$PY" "$REPO/scripts/bench/platecross_select.py" probe-scene --headless --task "$TASK" \
    --num-envs "$PROBE_ENVS" --out "$PROBE_JSON" > "$PROBE_LOG" 2>&1
rc=$?
grep -E "^PLATECROSS PROBE|PROBE-ERROR|^(ValueError|RuntimeError|TypeError|KeyError|AttributeError|ImportError)" "$PROBE_LOG" | cut -c1-300 || true
echo "=== probe exit $rc (informational; the verdict is the JSON) ==="'
    V60_BOOT_LOCK="$WORKSPACE/.v51-boot.lock" v60_boot_gate                    # serialise this boot too
    trap "_v60_release '$WORKSPACE/.v51-boot.lock'; _pc_export_release" EXIT     # (the gate replaced the trap)
    PROBE_JSON="$PROBE_JSON" PROBE_LOG="$PROBE_LOG" PROBE_ENVS=$PROBE_NUM_ENVS PROBE_TMO=900 bhl_exec -c "$INNER_PROBE" || true
fi

# ---------------------------------------------------------------- export (the PlateCross task: 77 obs, clock-stamped)
cfg="$run/exported/deploy.yaml"
why_export=""
if [ -n "$run" ] && [ -f "$run/$FINAL_CKPT" ]; then
    export RUN_DIR=$(basename "$run")
    if [ ! -f "$cfg" ]; then
        if export_lock; then
            echo "=== export $RUN_DIR with $TASK_ID $(date) ==="
            TASK="$TASK_ID" bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh" 2>&1 | grep -vE "pthread_setaffinity|EGLError" || true
            _pc_export_release
        else
            why_export="$why_export export-lock"
        fi
    fi
    # the parent's deploy.yaml, but the policy path, the unused command_velocity and the clock block's source line
    normp() { awk '/^policy_checkpoint_path:/{next} /^command_velocity:/{skip=1;next} skip&&/^- /{next} /^  source: /{next} {skip=0; print}' "$1"; }
    if [ -f "$cfg" ] && [ -f "$run/exported/policy.onnx" ]; then
        grep -q "^policy_checkpoint_path:.*$RUN_DIR/exported/policy.onnx" "$cfg" || why_export="$why_export policy-path"
        grep -qE "^num_joints: 22$" "$cfg" || why_export="$why_export num_joints"
        grep -qE "^num_observations: $OBS_POLICY$" "$cfg" || why_export="$why_export num_observations-not-$OBS_POLICY"
        w=$("$PYH" -c "import onnxruntime as o, sys; print(o.InferenceSession(sys.argv[1], providers=['CPUExecutionProvider']).get_inputs()[0].shape[-1])" \
            "$run/exported/policy.onnx" 2>/dev/null || echo "?")
        [ "$w" = "$OBS_POLICY" ] || why_export="$why_export onnx-input-width-$w-not-$OBS_POLICY"
        # the exported policy IS the final checkpoint's actor (not an earlier checkpoint, not the parent's)
        xline=$( (cd "$REPO" && OMP_NUM_THREADS=1 "$PYH" "$SELECT" check-export --onnx "$run/exported/policy.onnx" \
            --ckpt "$run/$FINAL_CKPT" --parent "$PARENT_PATH/model_$PARENT_ITER.pt" 2>&1 | tail -1) || true)
        echo "$xline"
        case "$xline" in "PLATECROSS EXPORT: OK"*) ;; *) why_export="$why_export export-is-not-$FINAL_CKPT";; esac
        if [ -z "$why_export" ]; then
            sline=$( (cd "$REPO" && PYTHONPATH="$REPO/src" "$PYH" -m bhl_robust.eval.gait_clock stamp --deploy "$cfg" \
                --env-yaml "$run/params/env.yaml" 2>&1 | tail -1) || true)
            echo "$sline"
            case "$sline" in "GAIT-CLOCK: stamped"*|"GAIT-CLOCK: already stamped"*) ;; *) why_export="$why_export stamp";; esac
        fi
        grep -q "^gait_clock:" "$cfg" || why_export="$why_export no-gait_clock-block"
        diff <(normp "$PARENT_PATH/exported/deploy.yaml") <(normp "$cfg") || why_export="$why_export deploy.yaml-differs-from-$PARENT_RUN"
    else
        why_export="$why_export no-deploy.yaml-or-policy.onnx"
    fi
else
    why_export="$why_export no-$FINAL_CKPT"
fi
if [ -z "$why_export" ]; then echo "EXPORT: OK $cfg ($TASK_ID, $OBS_POLICY obs, clock-stamped)"; else echo "EXPORT: FAIL ($RUN_NAME:$why_export)"; fi

# ---------------------------------------------------------------- smoke: plumbing checks, then exit
if [ "$SMOKE" = 1 ]; then
    why="$why_train$why_export"
    srec=$(record_problems "$TJSON" "$run")
    [ -z "${srec// /}" ] || why="$why training-record:${srec# }"
    # the scene probe's verdict, from its JSON
    pline=$("$PYH" - "$PROBE_JSON" <<'EOF' 2>/dev/null || echo "PLATECROSS PROBE: FAIL (no readable probe JSON)"
import json, sys
d = json.load(open(sys.argv[1]))
f = (d.get("plates") or {}).get("found") or {}
print(f"PLATECROSS PROBE: {d['verdict']} (round {f.get('round')}, square {f.get('square')}, other {f.get('other')}; "
      f"collider {(d.get('collider') or {}).get('static')}; obs {d.get('obs_dims')}; spawn clearance "
      f"{(d.get('spawn') or {}).get('min_plate_clearance_chebyshev_m')}; plate step "
      f"{(d.get('plate_drop') or {}).get('plate_step_m')}; problems: {'; '.join(d['problems'][:6]) or 'none'})")
EOF
)
    echo "$pline"
    case "$pline" in "PLATECROSS PROBE: PASS"*) ;; *) why="$why scene-probe";; esac
    if grep -qiE "buffer overflow|overflow detected" "$PROBE_LOG" 2>/dev/null; then why="$why physx-overflow-in-probe"; fi
    if [ -z "$why_export" ]; then
        mkdir -p "$RES/gait-clock-smoke" "$RES/turn-test-v1" "$RES/run-eval"
        kline=$( (cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 "$PYH" -m bhl_robust.eval.gait_clock smoke \
            --deploy "$cfg" --upstream "$UPSTREAM" --cache-dir "$CACHE" --expect clock --steps 30 --seed 100 \
            --out "$RES/gait-clock-smoke/$RUN_NAME.$JOBTAG.json" 2>&1 | grep -v "^\[" | tail -1) || true)
        echo "$kline"
        case "$kline" in "GAIT-CLOCK SMOKE: PASS"*) ;; *) why="$why gait-clock-rollout";; esac
        t1="$RES/turn-test-v1/$RUN_NAME.$JOBTAG.json"
        ( cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 "$PYH" scripts/bench/turn_test.py --deploy "$cfg" \
            --upstream "$UPSTREAM" --cache-dir "$CACHE" --variant humanoid --seconds 1 --warm 0.5 --seed 100 \
            --out "$t1" 2>&1 | grep -v "^\[" | tail -1 ) || true
        "$PYH" -c "import json,sys; d=json.load(open(sys.argv[1])); assert set(d['results'])=={'turn','walk','arc'}" "$t1" 2>/dev/null \
            || why="$why turn_test-plumbing"
        e1="$RES/run-eval/$RUN_NAME.$JOBTAG.csv"
        eline2=$( (cd "$REPO" && PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=2 "$PYH" -m bhl_robust.eval.gait_clock run-eval \
            --deploy "$cfg" --upstream "$UPSTREAM" --cache-dir "$CACHE" --out "$e1" --label "$RUN_NAME" --episode-s 2 \
            --push-speed 0.5 --seed0 100 2>&1 | grep "^GAIT-CLOCK" | tail -1) || true)
        echo "$eline2"
        case "$eline2" in "GAIT-CLOCK RUN-EVAL: PASS"*) ;; *) why="$why run_eval-plumbing";; esac
    fi
    # the selection code on the smoke outputs: one smoke seed and no gate JSONs -> INCOMPLETE, nothing written
    sline1=$( (cd "$REPO" && "$PYH" "$SELECT" select --res "$RES" --suffix=-smoke --iters "$MAX_ITER" 2>&1 | tail -1) || true)
    echo "selection on the smoke outputs: $sline1"
    case "$sline1" in "PLATECROSS SELECTION: INCOMPLETE"*) ;; *) why="$why selection-on-smoke-outputs";; esac
    [ ! -e "$RES/selection.json" ] || why="$why selection-json-written-for-an-incomplete-set"
    # ... and on three stand-in seeds (R1's real qualify / v2 JSONs, re-pointed at this smoke's run): seed 2 is the
    # only QUALIFIED one (9/60), so the rule must select it, write once and refuse a second write
    SDIR="$RES/selection-standin-$JOBTAG"
    if [ -f "$TJSON" ] && [ -z "$why_export" ]; then
        ( cd "$REPO" && "$PYH" "$SELECT" smoke-standins --dest "$SDIR" --source-res "$R1_RES" --training-record "$TJSON" 2>&1 | tail -1 ) || true
        s2=$( (cd "$REPO" && "$PYH" "$SELECT" select --res "$SDIR" --suffix=-smoke --iters "$MAX_ITER" 2>&1 | tail -1) || true)
        echo "selection on the stand-ins: $s2"
        case "$s2" in "PLATECROSS SELECTION: SELECTED $PREFIX-s2-smoke "*) ;; *) why="$why selection-on-standins";; esac
        "$PYH" - "$SDIR/selection.json" "$cfg" <<'EOF' 2>/dev/null || why="$why selection-json-content"
import hashlib, json, sys
d = json.load(open(sys.argv[1]))
s = d["selected"]
assert d["verdict"] == "SELECTED" and s["seed"] == 2 and s["deploy_yaml"] == sys.argv[2], d
assert s["sha256"]["deploy_yaml"] == hashlib.sha256(open(sys.argv[2], "rb").read()).hexdigest()
assert s["gait_clock_block"] is True and s["num_observations_line"] == "num_observations: 77"
EOF
        s3=$( (cd "$REPO" && "$PYH" "$SELECT" select --res "$SDIR" --suffix=-smoke --iters "$MAX_ITER" 2>&1 | head -1) || true)
        case "$s3" in "selection already recorded, not overwritten"*) ;; *) why="$why selection-rewrite-not-refused";; esac
        # ... and on a SYNTHETIC variant: stand-in s1 keeps its real v2 FAIL, its qualify JSON relabelled QUALIFIED
        # 0/60; counting the qualify verdict alone would select s1, the joint reading must still select s2
        JDIR="$RES/selection-standin-joint-$JOBTAG"
        ( cd "$REPO" && "$PYH" "$SELECT" smoke-standins --joint-check --dest "$JDIR" --source-res "$R1_RES" \
            --training-record "$TJSON" 2>&1 | tail -1 ) || true
        s4=$( (cd "$REPO" && "$PYH" "$SELECT" select --res "$JDIR" --suffix=-smoke --iters "$MAX_ITER" 2>&1 | tail -1) || true)
        echo "selection on the joint-check stand-ins: $s4"
        case "$s4" in "PLATECROSS SELECTION: SELECTED $PREFIX-s2-smoke "*) ;; *) why="$why selection-joint-check";; esac
        "$PYH" - "$JDIR/selection.json" <<'EOF' 2>/dev/null || why="$why selection-joint-check-content"
import json, sys
d = json.load(open(sys.argv[1]))
s1 = d["per_seed"]["s1"]
assert d["verdict"] == "SELECTED" and d["selected"]["seed"] == 2, d
assert (s1["v2_verdict"], s1["qualify_verdict"], s1["qualified"], s1["falls"]) == ("FAIL", "QUALIFIED", False, 0), s1
EOF
    else
        why="$why selection-standins-skipped(no-training-record-or-export)"
    fi
    grep -hE "^\[platecross_terrain\]|^\[gait_clock_mdp\]|Loading model checkpoint|Group: '(policy|critic)'|\| +(feet_gait|feet_swing_height|feet_air_time|push_robot) +\|" "$TLOG" | head -9 || true
    echo "=== done: $(date) ==="
    if [ -z "${why// /}" ]; then echo "SMOKE: PASS ($RUN_NAME, $run)"; exit 0; else echo "SMOKE: FAIL ($RUN_NAME:$why)"; exit 1; fi
fi
[ -z "$why_export" ] || { echo "PLATECROSS RESULT: FAIL (export:$why_export; seed stays INCOMPLETE)"; exit 1; }

# ---------------------------------------------------------------- gates (CPU MuJoCo, outside the container)
OUT_DIR=$RES/turn-test-v2
# v2_test <deploy.yaml> <label>: turn_test v2 into a fresh temp JSON, one line read from THAT JSON, then moved to
# $OUT_DIR/<label>.json, or to <label>.$JOBTAG.json if that exists (never overwrite). A resubmit that finds
# $OUT_DIR/<label>.json already there re-reads it instead of running the test again (v2 is deterministic).
v2_test() {
    local cfg=$1 label=$2 tmp
    tmp="$OUT_DIR/.$label.$JOBTAG.json"
    rm -f "$tmp"
    if [ -s "$OUT_DIR/$label.json" ]; then
        echo "have $OUT_DIR/$label.json (recorded earlier; not re-run, never overwritten)"
        tmp="$OUT_DIR/$label.json"
    else
        ( cd "$REPO" && PYTHONPATH="$REPO/src" OMP_NUM_THREADS=2 "$PYH" scripts/bench/turn_test.py --protocol v2 \
            --deploy "$cfg" --upstream "$UPSTREAM" --cache-dir "$CACHE" --variant humanoid \
            --out "$tmp" 2>&1 | grep -v "^\[" | tail -8 ) || true
    fi
    "$PYH" - "$tmp" "$label" <<'EOF'
import json, sys
path, label = sys.argv[1:3]
try:
    d = json.load(open(path))
    assert d.get("protocol") == "v2", "not a v2 JSON"
    turns = ", ".join(f"{r['name']} {r['yaw_deg']:+.1f}" for r in d["turns"])
    print(f"TURN-V2 {label}: {d['verdict']} (turns ok {d['n_turn_ok']}/{d['n_turn']}: {turns}; "
          f"walk drift {d['walk']['yaw_deg']:+.1f} deg, fell {d['walk']['fell_at_s']}) [gates the selection with the "
          f"qualify verdict: v2 PASS AND QUALIFIED]")
except Exception as exc:  # noqa: BLE001
    print(f"TURN-V2 {label}: FAIL (no readable v2 JSON: {exc!r})")
EOF
    if [ "$tmp" = "$OUT_DIR/.$label.$JOBTAG.json" ] && [ -f "$tmp" ]; then
        local dst="$OUT_DIR/$label.json"
        [ -e "$dst" ] && dst="$OUT_DIR/$label.$JOBTAG.json"
        mv "$tmp" "$dst"
    fi
}
v2_test "$cfg" "$RUN_NAME"
echo "=== qualification (cpu_turn_qualify.sbatch, unchanged rule) $(date) ==="
TURNQ_RUN="$RUN_NAME" TURNQ_OUT_DIR="$RES/qualify" bash "$REPO/slurm/repo20260923/cpu_turn_qualify.sbatch" 2>&1 \
    | grep -E "TURN-QUALIFY|verdict|FAIL|INCOMPLETE" | tail -4 || true
echo "=== selection (written once, by the job that completes the set) $(date) ==="
( cd "$REPO" && "$PYH" "$SELECT" select --res "$RES" ) || true
echo "=== done: $(date) ==="
