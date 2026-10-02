#!/bin/bash
#SBATCH --job-name=coop-flushpad
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# 2026-10-02: C1 of docs/SOLUTIONS_2026-10-01.md (section 4): FLUSH-PAD WRIST MECHANISM PROBE for the scripted
# cooperative lift (scripts/bench/coop_scripted_carry.py --flushpad-stage ...; code in the opt-in flush-pad section
# of src/bhl_robust/eval/scripted_carry.py). This job SCORES; it renders nothing.
#
#   MODIFIED END-EFFECTOR -- NOT THE STOCK ROBOT: each hand's box collision pad (the hand-mesh AABB, stock size and
#             centre) is re-oriented in the hand-link frame so its faces are parallel to the cube faces at the squeeze
#             pose (squeeze keyframe, shoulder roll stopped where the pad's outer face reaches the cube face plane;
#             kinematics only, no tuned number); pad contact condim 4, torsional friction 0.04 m.
#   HARNESS CHANGE: elliptic friction cone + impratio 10. These are global MuJoCo options, so they also change the
#             foot-floor contact of the frozen gait; the robot-fall clauses are kept unchanged.
#   labels  : LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + ORACLE cube pose (scoring only)
#   learned : 22-DoF locomotion gait arms-dr1.0-s0 (legs + each robot's outer arm), frozen, zero velocity command
#   scripted: each robot's grasping arm: reach/squeeze/lift keyframes (unchanged), then held (lift-hold) or hold,
#             lower, open, rest (lift-place); grasping-arm PD kp 30 (deploy 10), 4 Nm arm cap unchanged; no welds
#   oracle  : cube pose from the simulator, used only to score (cube tilt, lift, placement)
#   lowering (lift-place): the reverse keyframe, lift -> squeeze (FlushPadPlaceParams.lower_to = None), FROZEN
#             2026-10-02 before any probe or scored episode of this variant.
#
# Usage (job name must contain "smoke" for the smoke; the run refuses without a passing smoke of the same bytes):
#   smoke: sbatch --job-name=coop-flushpad-smoke slurm/repo20260923/cpu_coop_flushpad.sbatch smoke
#   run  : sbatch --dependency=afterok:<SMOKE> --export=ALL,FLUSHPAD_SMOKE_JOB=<SMOKE> \
#                 slurm/repo20260923/cpu_coop_flushpad.sbatch run
#   Always submit a fresh smoke and chain the run on it with afterok; never reuse an old smoke id. The run also
#   refuses unless that smoke's smoke_verdict.json reads SMOKE_PASS (which needs its unit tests and both smoke
#   episodes recorded with exit status 0 in step_status.json), records its unit tests as exit 0, and the code, the
#   test file and this launcher have the sha256 the smoke ran.
#
# PREDECLARED RULES (FROZEN 2026-10-02; verbatim; scripted_carry.FLUSHPAD_PROBE_RULE_TEXT and
# FLUSHPAD_SCORED_RULE_TEXT hold the same words; do not relax):
# - Probe on exploration seeds 120-124 (state the grep evidence that they were never used). PROCEED iff the cube
#   tilt stays <= 0.35 rad throughout lift-hold on >= 4/5 seeds, with LIFT_HOLD_RULE's robot-fall clauses
#   unchanged. Otherwise STOP: NEGATIVE, the flush pad does not stop the roll.
# - Scored, only after PROCEED (state the evidence that seeds 20-39 were never used): lift-hold on seeds 20-29 and
#   lift-place on seeds 30-39, each with its UNCHANGED rule (LIFT_HOLD_RULE / LIFT_PLACE_RULE, pass_min 8) PLUS
#   cube tilt <= 0.35 rad (hold: throughout the hold; place: at release and when seated), >= 8/10 per crew or pair
#   as the existing rules define them. PASS iff every crew/pair meets it; otherwise NEGATIVE; INCOMPLETE if any
#   episode is missing.
#
# CLAUSES AS APPLIED (decided 2026-10-02 before any episode; each stricter than or equal to the text; the verdict
# JSONs repeat them from scripted_carry.FLUSHPAD_CLAUSES_AS_APPLIED):
# * cube tilt = the angle between the cube's own z axis and world z (a cube rolled 90 deg reads 1.571 rad), at every
#   policy-step state (t = i * 0.04 s, i = 0..500), full float precision; <= 0.35 rad passes.
# * probe "throughout lift-hold" = every policy-step state of the 20 s lift_hold episode, t = 0 to 20 s inclusive
#   (stricter than the hold alone). A probe seed passes iff that tilt clause, LIFT_HOLD_RULE's robot-fall clause
#   unchanged (no robot of the pair tilts > 0.78 rad at any policy step) and the validity clauses (finite state,
#   no MuJoCo auto-reset, full 20 s) hold; crew 2 (one pair); PROCEED iff >= 4 of seeds 120-124 pass.
# * probe disclosure: the rule names only LIFT_HOLD_RULE's robot-fall clauses, so its lift, hold and floor clauses
#   are not gated in the probe (reported per seed as diagnostics). A probe episode that never lifts the cube
#   therefore passes the probe's tilt clause; the scored stage keeps every LIFT_HOLD_RULE clause.
# * hold "throughout the hold" = every policy-step state of the hold LIFT_HOLD_RULE scores (its longest contiguous
#   >= 0.05 m run, hold_start_s inclusive to hold_end_s exclusive); no hold = the clause fails.
# * place "at release" = every policy-step state from the end of the lowering (t_end['unload'] = 10.7 s: state 268,
#   t = 10.72 s) through the hands fully open (t_end['release'] = 11.7 s: state 293, t = 11.72 s), inclusive;
#   "when seated" = the final state (t = 20 s), where LIFT_PLACE_RULE checks the seated cube.
# * crews as the existing rules and launchers define them: crew 2 (one pair) and crew 4 (two pairs, two cubes, one
#   world), >= 8/10 per pair; PASS iff all six pairs (lift-hold crew 2, crew 4 pair 0, crew 4 pair 1; lift-place
#   the same) meet it.
# * an episode stopped by a non-finite state is a scored failure; its non-finite numbers are written as null so the
#   score JSON holds it. A numerical blow-up that MuJoCo catches itself is a scored failure too, in every stage
#   (check no_sim_reset): MuJoCo 3.3.5 resets the state automatically (qpos0, time 0) when qpos, qvel or qacc holds
#   a NaN or a value beyond 1e10, so the harness's own non-finite check never sees it; it is detected after every
#   policy step from the simulated time and MuJoCo's bad-qpos/qvel/qacc warning counters. An episode that is
#   absent, shorter than 20 s without a recorded failure, or has no tilt log or no auto-reset record is MISSING
#   (INCOMPLETE). (This reading was added 2026-10-02 after review, before any probe or scored episode.)
# * INVALID (never PROCEED, never PASS): a score JSON that is not this variant's or stage's, not under the frozen
#   rule, parameters, keyframes or lowering, not a scored run, or whose compiled model does not show the designed
#   flush pads with condim 4, torsional 0.04 m, elliptic cone and impratio 10.
#
# SEED EVIDENCE (grep, 2026-10-02; repeated in every verdict JSON):
# * results/scripted-carry-20260926/score_{crew2,crew4,lifthold_crew2,lifthold_crew4}.json hold seeds 0-9 only;
#   results/scripted-carry-20260927/score_liftplace_crew{2,4}.json hold seeds 10-19 only.
# * Slurm logs of this harness: scripted-carry-21434982 and coop-lift-hold-21435079 ran seeds 0-9,
#   coop-lift-place-21443282 seeds 10-19; the render jobs 21435080 and 21443283 skipped (no PASS).
# * Exploration: the harness docstring and lift-place notes name seeds 100-119 only; the 2026-09-30 investigators'
#   grip/place mechanism JSONs (solutions-20260930/coop, verify-coop) hold seeds 100-104 only.
# * The ledger names seeds 0-9, 10-19 and exploration seeds >= 100 for this harness. No score JSON, Slurm log,
#   ledger entry or investigator file of this harness contains seed 120-124 or any seed 20-39.
# * The smoke runs only declared tuning seed 119 (the script refuses 120-124 and 20-39 in smoke mode).
#
# DISCLOSED PRIORS (exploration seeds 100-104 only, stock pads): pad torsional friction 0.04 m + elliptic cone +
# impratio 10 made lift-hold 3/3, but the cube was held rolled 0.63-0.93 rad; at the squeeze keyframe the stock pad is
# 26.2 deg off-face. Kinematics: the hand rotates ~0.99 rad about the pinch axis between the squeeze contact and the
# lift, so a rigid flush grip is EXPECTED to still roll the cube (~1 rad): the probe is expected to read NEGATIVE.
#
# Outputs. run (refused if the run dir exists; nothing is ever overwritten):
#   results/coop-flushpad-20261002/probe/score_probe_crew2.json + probe/verdict.json
#   results/coop-flushpad-20261002/scored/score_{hold,place}_crew{2,4}.json + scored/verdict.json
#   (scored/verdict.json reads NOT_RUN unless the probe verdict JSON reads PROCEED)
# smoke: /nfs/hpc/share/sanchej7/Humanoid_Lite/logs/coop-flushpad-<jobid>-smoke/ : unit tests, one full 20 s
#   lift_hold crew-2 episode and one full lift_place crew-4 episode on tuning seed 119, step_status.json (each
#   step's exit status, unit tests included), smoke_verdict.json (SMOKE_PASS needs every step_status entry 0),
#   code_sha256.txt (code + test file), launcher_sha256.txt. Exits non-zero unless the smoke verdict JSON reads
#   SMOKE_PASS and every step exited 0. Verdicts are always read from JSON, never from exit codes.
set -uo pipefail
MODE=${1:-}
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
LOGS=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
DEPLOY=$U/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml
CODE=(src/bhl_robust/eval/scripted_carry.py scripts/bench/coop_scripted_carry.py tests/test_coop_flushpad.py)
LABEL="MODIFIED END-EFFECTOR (flush hand pads) — not the stock robot | HARNESS CHANGE: elliptic cone + impratio 10 | LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + ORACLE cube pose (scoring only)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
JOB=${SLURM_JOB_ID:-nojob}
LAUNCHER_SHA=$(sha256sum "$0" | awk '{print $1}')

case "$MODE" in
    smoke|run) ;;
    *) echo "COOP-FLUSHPAD RESULT: REFUSED (usage: cpu_coop_flushpad.sbatch smoke|run)"; exit 1 ;;
esac
[ -f "$DEPLOY" ] || { echo "COOP-FLUSHPAD RESULT: FAIL (deploy config missing: $DEPLOY)"; exit 1; }

# verdicts are read from the JSON (MISSING if it cannot be read), never from an exit status
verdict_of() {
    "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["verdict"])' "$1" 2>/dev/null || echo MISSING
}
# the smoke's unit-test exit status as its smoke verdict JSON records it (MISSING if absent)
pytest_status_of() {
    "$PY" -c 'import json, sys; print(json.load(open(sys.argv[1]))["step_status"]["pytest"])' "$1" 2>/dev/null \
        || echo MISSING
}
write_verdict() {
    "$PY" -c 'import sys; sys.path.insert(0, "scripts/bench"); import coop_scripted_carry as m; sys.exit(m.flushpad_verdict_cli(sys.argv[1:]))' "$@"
}
score() {
    "$PY" scripts/bench/coop_scripted_carry.py --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" "$@" 2>&1 \
        | grep --line-buffered -v '^\['
    return "${PIPESTATUS[0]}"
}

echo "=== coop-flushpad $MODE | job $JOB | node $(hostname) | HEAD $(git rev-parse --short HEAD) | $(date) ==="
echo "--- code sha256 (launcher = the script this job runs):"
sha256sum "${CODE[@]}"
echo "$LAUNCHER_SHA  launcher"
echo "--- git status of the C1 files (empty = committed):"
git status --porcelain -- "${CODE[@]}" slurm/repo20260923/cpu_coop_flushpad.sbatch
echo "$LABEL"

if [ "$MODE" = smoke ]; then
    case "${SLURM_JOB_NAME:-}" in
        *smoke*) ;;
        *) echo "COOP-FLUSHPAD SMOKE: REFUSED (the smoke's job name must contain 'smoke')"; exit 1 ;;
    esac
    SM=$LOGS/coop-flushpad-$JOB-smoke
    if [ -e "$SM" ]; then echo "COOP-FLUSHPAD SMOKE: REFUSED ($SM exists; nothing is overwritten)"; exit 1; fi
    mkdir -p "$SM" || exit 1
    CACHE=${TMPDIR:-/tmp}/bhl-coop-flushpad-$JOB
    mkdir -p "$CACHE" || exit 1
    sha256sum "${CODE[@]}" > "$SM/code_sha256.txt"
    echo "$LAUNCHER_SHA" > "$SM/launcher_sha256.txt"
    rc=0
    echo "--- unit tests: tests/test_coop_flushpad.py tests/test_scripted_carry.py"
    "$PY" -m pytest -q -p no:cacheprovider tests/test_coop_flushpad.py tests/test_scripted_carry.py 2>&1 | tail -40
    t=${PIPESTATUS[0]}
    echo "pytest exit $t"
    [ "$t" -eq 0 ] || rc=1
    echo "--- smoke episode 1: crew 2, lift_hold (the probe's path), declared tuning seed 119, full 20 s"
    score --flushpad-stage smoke --protocol lift_hold --crew 2 --seeds 119 --out "$SM/smoke_hold_crew2.json"
    s1=$?
    echo "exit $s1"
    [ "$s1" -eq 0 ] || rc=1
    echo "--- smoke episode 2: crew 4, lift_place (stage 2's place and crew-4 paths), tuning seed 119, full 20 s"
    score --flushpad-stage smoke --protocol lift_place --crew 4 --seeds 119 --out "$SM/smoke_place_crew4.json"
    s2=$?
    echo "exit $s2"
    [ "$s2" -eq 0 ] || rc=1
    # every step's exit status goes into the smoke verdict JSON (SMOKE_PASS needs all of them 0), so a smoke whose
    # unit tests failed never reads SMOKE_PASS, even when its id is reused for a run without afterok
    printf '{"pytest": %d, "smoke_hold_crew2": %d, "smoke_place_crew4": %d}\n' "$t" "$s1" "$s2" \
        > "$SM/step_status.json" || rc=1
    write_verdict smoke "$SM/smoke_verdict.json" step_status="$SM/step_status.json" \
        hold_crew2="$SM/smoke_hold_crew2.json" place_crew4="$SM/smoke_place_crew4.json" || rc=1
    rm -rf "$CACHE"
    V=$(verdict_of "$SM/smoke_verdict.json")
    if [ "$V" = SMOKE_PASS ] && [ "$rc" -eq 0 ]; then
        echo "COOP-FLUSHPAD SMOKE: PASS (smoke_verdict.json reads $V; every step exited 0) | $SM | $LABEL"
        exit 0
    fi
    echo "COOP-FLUSHPAD SMOKE: FAIL (smoke_verdict.json reads $V; step status $rc) | $SM | $LABEL"
    exit 1
fi

# ---------------------------------------------------------------- run: bytes check against a passing smoke
SMOKE_JOB=${FLUSHPAD_SMOKE_JOB:-}
if [ -z "$SMOKE_JOB" ]; then
    echo "COOP-FLUSHPAD RESULT: REFUSED (run mode needs --export=ALL,FLUSHPAD_SMOKE_JOB=<smoke job id>)"
    exit 1
fi
SM=$LOGS/coop-flushpad-$SMOKE_JOB-smoke
SV=$(verdict_of "$SM/smoke_verdict.json")
if [ "$SV" != SMOKE_PASS ]; then
    echo "COOP-FLUSHPAD RESULT: REFUSED (smoke $SMOKE_JOB: $SM/smoke_verdict.json reads $SV, not SMOKE_PASS)"
    exit 1
fi
PT=$(pytest_status_of "$SM/smoke_verdict.json")
if [ "$PT" != 0 ]; then
    echo "COOP-FLUSHPAD RESULT: REFUSED (smoke $SMOKE_JOB: $SM/smoke_verdict.json records its unit tests as exit $PT, not 0)"
    exit 1
fi
if ! sha256sum -c --quiet "$SM/code_sha256.txt"; then
    echo "COOP-FLUSHPAD RESULT: REFUSED (the code differs from the bytes smoke $SMOKE_JOB ran)"
    exit 1
fi
if [ "$(cat "$SM/launcher_sha256.txt" 2>/dev/null)" != "$LAUNCHER_SHA" ]; then
    echo "COOP-FLUSHPAD RESULT: REFUSED (this launcher differs from the one smoke $SMOKE_JOB ran)"
    exit 1
fi
echo "BYTES-CHECK: PASS (smoke $SMOKE_JOB reads SMOKE_PASS, unit tests exit 0; code, test file and launcher sha256 match)"

RUN=$REPO/results/coop-flushpad-20261002
if [ -e "$RUN" ]; then
    echo "COOP-FLUSHPAD RESULT: REFUSED ($RUN exists; results are never overwritten)"
    exit 1
fi
mkdir -p "$RUN/probe" "$RUN/scored" || exit 1
CACHE=${TMPDIR:-/tmp}/bhl-coop-flushpad-$JOB
mkdir -p "$CACHE" || exit 1
rc=0

echo "=== STAGE 1: probe | crew 2 | lift_hold (20 s) | seeds 120-124"
score --flushpad-stage probe --protocol lift_hold --crew 2 --seeds 120-124 --out "$RUN/probe/score_probe_crew2.json"
s=$?
echo "exit $s"
[ "$s" -eq 0 ] || rc=1
write_verdict probe "$RUN/probe/verdict.json" probe="$RUN/probe/score_probe_crew2.json" || rc=1
PV=$(verdict_of "$RUN/probe/verdict.json")
echo "STAGE 1 verdict (read from $RUN/probe/verdict.json): $PV"

if [ "$PV" = PROCEED ]; then
    echo "=== STAGE 2: scored | lift-hold seeds 20-29 and lift-place seeds 30-39 | crews 2 and 4"
    for n in 2 4; do
        echo "--- lift-hold crew $n, seeds 20-29 (20 s)"
        score --flushpad-stage hold --protocol lift_hold --crew "$n" --seeds 20-29 \
            --out "$RUN/scored/score_hold_crew$n.json"
        s=$?
        echo "exit $s"
        [ "$s" -eq 0 ] || rc=1
    done
    for n in 2 4; do
        echo "--- lift-place crew $n, seeds 30-39 (20 s)"
        score --flushpad-stage place --protocol lift_place --crew "$n" --seeds 30-39 \
            --out "$RUN/scored/score_place_crew$n.json"
        s=$?
        echo "exit $s"
        [ "$s" -eq 0 ] || rc=1
    done
    write_verdict scored "$RUN/scored/verdict.json" \
        hold_crew2="$RUN/scored/score_hold_crew2.json" hold_crew4="$RUN/scored/score_hold_crew4.json" \
        place_crew2="$RUN/scored/score_place_crew2.json" place_crew4="$RUN/scored/score_place_crew4.json" || rc=1
else
    echo "STAGE 2 NOT RUN (the probe verdict JSON reads $PV, not PROCEED): seeds 20-39 stay unused"
    write_verdict scored_not_run "$RUN/scored/verdict.json" probe_verdict="$RUN/probe/verdict.json" || rc=1
fi
SV2=$(verdict_of "$RUN/scored/verdict.json")
rm -rf "$CACHE"
echo "COOP-FLUSHPAD RESULT: probe $PV | scored $SV2 | $RUN/probe/verdict.json $RUN/scored/verdict.json | $LABEL"
echo "=== done: $(date) (step status $rc) ==="
[ "$rc" -eq 0 ] || exit 3
exit 0
