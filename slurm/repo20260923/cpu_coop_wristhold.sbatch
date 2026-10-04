#!/bin/bash
#SBATCH --job-name=coop-wristhold
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# 2026-10-03: design (W) of the user approval recorded in SLURM_JOBS.md on 2026-10-03 09:55: COOPERATIVE SCRIPTED LIFT
# THAT HOLDS WRIST ORIENTATION (code: the opt-in coop-wristhold sections of src/bhl_robust/eval/scripted_carry.py and
# scripts/bench/coop_scripted_carry.py, entry `wristhold_main`). This job SCORES; it renders nothing.
#
#   The flush-pad variant EXACTLY as predeclared on 2026-10-02 04:59, unchanged:
#   MODIFIED END-EFFECTOR -- NOT THE STOCK ROBOT: each hand's box collision pad (the hand-mesh AABB, stock size and
#             centre) is re-oriented in the hand-link frame so its faces are parallel to the cube faces at the squeeze
#             pose (kinematics only); pad contact condim 4, torsional friction 0.04 m.
#   HARNESS CHANGE: elliptic friction cone + impratio 10. These are global MuJoCo options, so they also change the
#             foot-floor contact of the frozen gait; the robot-fall clauses are kept unchanged.
#   lowering (lift-place): the reverse keyframe, lift -> squeeze (FlushPadPlaceParams.lower_to = None), as frozen
#             2026-10-02.
#   PLUS the KINEMATIC WRIST-ORIENTATION HOLD (WristHoldParams, frozen 2026-10-03 before any episode):
#   * wrist   : the BHL arm has no wrist joint (shoulder pitch -> shoulder roll -> shoulder yaw -> elbow pitch -> elbow
#               roll -> hand link, the hand link welded to the elbow-roll body), so the wrist is the ELBOW ROLL, the
#               joint the hand hangs from. Shoulder yaw is a shoulder joint (it translates the pad 12 cm); not used.
#   * held    : each grasping hand's rotation about the pinch axis (world x, a constant of the scripted layout; never
#               read from the cube) at its value at grasp time, in the world frame (twist of the swing-twist split).
#   * grasp   : the first policy-step state at or after the end of the squeeze (4.5 s): state 113, t = 4.52 s.
#   * when    : every control step (0.04 s, when the harness recomputes every arm target) from state 113: lift-hold to
#               the end of the episode; lift-place through the release (t < 11.7 s), then the last correction is
#               withdrawn over the retract segment (11.7 -> 12.7 s) with the script's smooth profile (no target jump).
#   * how     : from the robot's own measured state (base attitude + arm joint angles from the simulator; never the
#               cube), the elbow-roll angle that brings the hand's twist back to its grasp value, or as close as the
#               joint range (+/-0.785 rad) allows: damped Gauss-Newton, clamped to the range, with the exact twist
#               derivative (the hand's MuJoCo rotational Jacobian column chained through the swing-twist split) and a
#               step accepted only if it strictly reduces |twist| (never worse than the measured state).
#   * unchanged: arm keyframes and timeline, grasping-arm PD kp 30 and kd 2, the 4 Nm arm cap (the stock PD loop
#               computes and clips every torque; the hold only replaces the wrist entry of the target), gait, scoring.
#   labels  : LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms (with a kinematic wrist-orientation hold) + ORACLE cube
#             pose (scoring only); MODIFIED END-EFFECTOR (flush pads); HARNESS CHANGE as the flush-pad variant
#   oracle  : cube pose from the simulator, used only to score (cube tilt, lift, placement)
#
# Usage (job name must contain "smoke" for the smoke; the run refuses without a passing smoke of the same bytes):
#   smoke: sbatch --job-name=coop-wristhold-smoke slurm/repo20260923/cpu_coop_wristhold.sbatch smoke
#   run  : sbatch --dependency=afterok:<SMOKE> --export=ALL,WRISTHOLD_SMOKE_JOB=<SMOKE> \
#                 slurm/repo20260923/cpu_coop_wristhold.sbatch run
#   Always submit a fresh smoke and chain the run on it with afterok; never reuse an old smoke id. The run also
#   refuses unless that smoke's smoke_verdict.json reads SMOKE_PASS (which needs its unit tests and all three smoke
#   episodes recorded with exit status 0 in step_status.json), records its unit tests as exit 0, and the code, the
#   test files and this launcher have the sha256 the smoke ran.
#
# PREDECLARED RULES (user approval 2026-10-03 09:55: the flush-pad probe rule verbatim except the seeds, and the
# flush-pad scored rule verbatim; scripted_carry.WRISTHOLD_PROBE_RULE_TEXT and WRISTHOLD_SCORED_RULE_TEXT hold the
# same words; do not relax):
# - Probe on exploration seeds 125-129 (state the grep evidence that they were never used). PROCEED iff the cube
#   tilt stays <= 0.35 rad throughout lift-hold on >= 4/5 seeds, with LIFT_HOLD_RULE's robot-fall clauses
#   unchanged. Otherwise STOP: NEGATIVE, the flush pad does not stop the roll.
# - Scored, only after PROCEED (state the evidence that seeds 20-39 were never used): lift-hold on seeds 20-29 and
#   lift-place on seeds 30-39, each with its UNCHANGED rule (LIFT_HOLD_RULE / LIFT_PLACE_RULE, pass_min 8) PLUS
#   cube tilt <= 0.35 rad (hold: throughout the hold; place: at release and when seated), >= 8/10 per crew or pair
#   as the existing rules define them. PASS iff every crew/pair meets it; otherwise NEGATIVE; INCOMPLETE if any
#   episode is missing.
#
# CLAUSES AS APPLIED (the flush-pad clauses as applied, 2026-10-02, with the W seeds and the hold's record; decided
# 2026-10-03 before any episode; each stricter than or equal to the text; the verdict JSONs repeat them from
# scripted_carry.WRISTHOLD_CLAUSES_AS_APPLIED):
# * cube tilt = the angle between the cube's own z axis and world z (a cube rolled 90 deg reads 1.571 rad), at every
#   policy-step state (t = i * 0.04 s, i = 0..500), full float precision; <= 0.35 rad passes.
# * probe "throughout lift-hold" = every policy-step state of the 20 s lift_hold episode, t = 0 to 20 s inclusive
#   (stricter than the hold alone). A probe seed passes iff that tilt clause, LIFT_HOLD_RULE's robot-fall clause
#   unchanged (no robot of the pair tilts > 0.78 rad at any policy step) and the validity clauses (finite state,
#   no MuJoCo auto-reset, full 20 s) hold; crew 2 (one pair); PROCEED iff >= 4 of seeds 125-129 pass.
# * probe disclosure: the rule names only LIFT_HOLD_RULE's robot-fall clauses, so its lift, hold and floor clauses
#   are not gated in the probe (reported per seed as diagnostics). A probe episode that never lifts the cube
#   therefore passes the probe's tilt clause (so would a cube the hold drops that lands flat); the scored stage keeps
#   every LIFT_HOLD_RULE clause.
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
#   (check no_sim_reset), detected after every policy step from the simulated time and MuJoCo's bad-qpos/qvel/qacc
#   warning counters. An episode that is absent, shorter than 20 s without a recorded failure, or has no tilt log or
#   no auto-reset record is MISSING (INCOMPLETE).
# * the wrist-orientation hold adds measurements (hand twist and swing, wrist angle against its range, PD demand
#   against the 4 Nm cap), never a gate clause. An episode without a well-formed hold record (hold installed after
#   policy step 0, engaged at the grasp state 113, the twist logged at every state) is MISSING (INCOMPLETE).
# * INVALID (never PROCEED, never PASS): a score JSON that is not this variant's or stage's, not under the frozen
#   rule, parameters (WristHoldParams), keyframes or lowering, not a scored run, whose compiled model does not show
#   the designed flush pads with condim 4, torsional 0.04 m, elliptic cone and impratio 10, or whose wrist-hold design
#   record does not name the frozen wrist (elbow roll).
#
# SEED EVIDENCE (grep, 2026-10-03; repeated in every verdict JSON):
# * grep -rhoE '"seed": *[0-9]+' over results/scripted-carry-20260926, results/scripted-carry-20260927,
#   results/coop-flushpad-20261002 and every logs/coop-flushpad-*-smoke directory finds seeds 0-19, 119 and 120-124
#   only: no episode of this harness ran any seed 20-39 or 125-129.
# * Slurm logs of this harness: scripted-carry-21434982 and coop-lift-hold-21435079 ran seeds 0-9,
#   coop-lift-place-21443282 seeds 10-19, the flush-pad smokes seed 119, the flush-pad probe 21507881 seeds 120-124;
#   its scored stage did not run (results/coop-flushpad-20261002/scored/verdict.json: NOT_RUN, seeds 20-39 unused).
# * A seed-context scan of the 255 coop / carry / crew / cube / flush files under results/ and logs/, the
#   investigators' and flush-pad workstream's files and SLURM_JOBS.md finds seed-like integers <= 1000 only in 0-39,
#   99 (a MARL smoke), 100-129 and 300-319; 125-129 appear only in the ledger's 2026-10-03 design (W), 20-39 only as
#   reserved, unused scored seeds.
# * Smoke seeds 140-144 (throw-away): in no seed context anywhere; the smoke runs seed 140 only (the CLI refuses any
#   other seed in smoke mode, and any seed outside its stage).
#
# KINEMATIC PREDICTION (pre-episode; scripted_carry.wristhold_design, recorded in every output): the cube stops the
# shoulder roll near 0 (0.01-0.07 rad), so at the contact pose the arm hangs and the forearm points forward. Joint
# axis . pinch axis at the squeeze / lift contact poses: shoulder pitch 1.00 / 1.00, elbow pitch 0.96 / 0.95,
# shoulder yaw 0.27 / 0.31, elbow roll 0.24 / 0.17, shoulder roll 0.00: only the two LIFT joints turn the hand about
# the pinch axis. The lift keyframe turns the hand ~0.98 rad about it; clamped at +/-0.785 rad the elbow roll removes
# ~0.17-0.18 rad (18-19 %) and swings the pad ~0.9 rad off the cube face; no admissible wrist set holds the twist
# (shoulder yaw + elbow roll: 31 %, residual 0.67 rad > 0.35). Expected: the hold does NOT stop the roll; the probe is
# expected to read NEGATIVE.
# DISCLOSED DEVELOPMENT RUNS (throw-away seed 140, before this launcher's smoke; no frozen choice changed after them):
# a dev check (crew 2 lift_hold, 8 s) and CLI dry runs of the smoke's three episodes. The first dry run exposed an IK
# flaw: the first implementation used the projected Jacobian (pinch axis . wrist axis) as the twist derivative and
# accepted every step; with a large swing that is not the derivative, and at the release the iteration walked to the
# opposite range bound and made the twist worse (one hand -0.003 -> -0.15 rad; target 0 -> -0.785 rad; a 23.6 Nm PD
# demand). Fixed: exact derivative, a step accepted only if |twist| decreases, start projected onto the range. The
# numbers below are from the fixed code. Crew 2 lift_hold with the hold: the wrist reached its range limit at 5.44 /
# 5.56 s; hand twist at the lift end 0.39 / 0.78 rad vs 0.80 / 0.85 without the hold; swing up to 1.71 rad vs 0.42;
# PD demand at most 2.5 Nm (never above the 4 Nm cap); the cube DROPPED to the floor (held >= 5 cm for 1.3 s vs 14.3 s
# without the hold); cube tilt max 1.61 rad vs 0.76. Crew 4 lift_place: both pairs dropped the cube; the cap bound on
# 2 and 7 policy steps of the two left hands during the release (target moves up to 0.20 rad per step, PD demand up
# to 6.1 Nm, applied torque clipped at 4 Nm).
#
# Outputs. run (refused if the run dir exists; nothing is ever overwritten):
#   results/coop-wristhold-20261003/probe/score_probe_crew2.json + probe/verdict.json
#   results/coop-wristhold-20261003/scored/score_{hold,place}_crew{2,4}.json + scored/verdict.json
#   (scored/verdict.json reads NOT_RUN unless the probe verdict JSON reads PROCEED)
# smoke: /nfs/hpc/share/sanchej7/Humanoid_Lite/logs/coop-wristhold-<jobid>-smoke/ : unit tests, one full 20 s
#   lift_hold crew-2 episode with the hold, one full lift_place crew-4 episode with the hold and one full lift_hold
#   crew-2 NO-HOLD baseline (the flush-pad variant, measurement only), all on throw-away seed 140; step_status.json
#   (each step's exit status, unit tests included), smoke_verdict.json (SMOKE_PASS needs every step_status entry 0;
#   its "measurement" says how much of the hand rotation the hold removed, the wrist range and cap use, and the cube
#   tilt with and without the hold), code_sha256.txt (code + test files), launcher_sha256.txt. Exits non-zero unless
#   the smoke verdict JSON reads SMOKE_PASS and every step exited 0. Verdicts are always read from JSON, never from
#   exit codes.
set -uo pipefail
MODE=${1:-}
REPO=/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
LOGS=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs
PY=/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python
U=$REPO/external/Berkeley-Humanoid-Lite
DEPLOY=$U/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml
CODE=(src/bhl_robust/eval/scripted_carry.py scripts/bench/coop_scripted_carry.py tests/test_coop_wristhold.py tests/test_coop_flushpad.py tests/test_scripted_carry.py)
LABEL="MODIFIED END-EFFECTOR (flush hand pads) — not the stock robot | HARNESS CHANGE: elliptic cone + impratio 10 | LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms (with a kinematic wrist-orientation hold) + ORACLE cube pose (scoring only)"
cd "$REPO" || exit 1
export PYTHONPATH="$REPO/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
JOB=${SLURM_JOB_ID:-nojob}
LAUNCHER_SHA=$(sha256sum "$0" | awk '{print $1}')

case "$MODE" in
    smoke|run) ;;
    *) echo "COOP-WRISTHOLD RESULT: REFUSED (usage: cpu_coop_wristhold.sbatch smoke|run)"; exit 1 ;;
esac
[ -f "$DEPLOY" ] || { echo "COOP-WRISTHOLD RESULT: FAIL (deploy config missing: $DEPLOY)"; exit 1; }

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
    "$PY" -c 'import sys; sys.path.insert(0, "scripts/bench"); import coop_scripted_carry as m; sys.exit(m.wristhold_verdict_cli(sys.argv[1:]))' "$@"
}
score() {
    "$PY" -c 'import sys; sys.path.insert(0, "scripts/bench"); import coop_scripted_carry as m; sys.exit(m.wristhold_main(sys.argv[1:]))' \
        --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" "$@" 2>&1 | grep --line-buffered -v '^\['
    return "${PIPESTATUS[0]}"
}

echo "=== coop-wristhold $MODE | job $JOB | node $(hostname) | HEAD $(git rev-parse --short HEAD 2>/dev/null) | $(date) ==="
echo "--- code sha256 (launcher = the script this job runs):"
sha256sum "${CODE[@]}"
echo "$LAUNCHER_SHA  launcher"
echo "--- git status of the W files (empty = committed):"
git status --porcelain -- "${CODE[@]}" slurm/repo20260923/cpu_coop_wristhold.sbatch 2>/dev/null
echo "$LABEL"

if [ "$MODE" = smoke ]; then
    case "${SLURM_JOB_NAME:-}" in
        *smoke*) ;;
        *) echo "COOP-WRISTHOLD SMOKE: REFUSED (the smoke's job name must contain 'smoke')"; exit 1 ;;
    esac
    SM=$LOGS/coop-wristhold-$JOB-smoke
    if [ -e "$SM" ]; then echo "COOP-WRISTHOLD SMOKE: REFUSED ($SM exists; nothing is overwritten)"; exit 1; fi
    mkdir -p "$SM" || exit 1
    CACHE=${TMPDIR:-/tmp}/bhl-coop-wristhold-$JOB
    mkdir -p "$CACHE" || exit 1
    sha256sum "${CODE[@]}" > "$SM/code_sha256.txt"
    echo "$LAUNCHER_SHA" > "$SM/launcher_sha256.txt"
    rc=0
    echo "--- unit tests: tests/test_coop_wristhold.py tests/test_coop_flushpad.py tests/test_scripted_carry.py"
    "$PY" -m pytest -q -p no:cacheprovider tests/test_coop_wristhold.py tests/test_coop_flushpad.py \
        tests/test_scripted_carry.py 2>&1 | tail -40
    t=${PIPESTATUS[0]}
    echo "pytest exit $t"
    [ "$t" -eq 0 ] || rc=1
    echo "--- smoke episode 1: crew 2, lift_hold WITH the hold (the probe's path), throw-away seed 140, full 20 s"
    score --wristhold-stage smoke --protocol lift_hold --crew 2 --seeds 140 --out "$SM/smoke_hold_crew2.json"
    s1=$?
    echo "exit $s1"
    [ "$s1" -eq 0 ] || rc=1
    echo "--- smoke episode 2: crew 4, lift_place WITH the hold (stage 2's place and crew-4 paths), seed 140, full 20 s"
    score --wristhold-stage smoke --protocol lift_place --crew 4 --seeds 140 --out "$SM/smoke_place_crew4.json"
    s2=$?
    echo "exit $s2"
    [ "$s2" -eq 0 ] || rc=1
    echo "--- smoke episode 3: crew 2, lift_hold NO-HOLD baseline (flush-pad variant, measured only), seed 140, 20 s"
    score --wristhold-stage smoke_nohold --protocol lift_hold --crew 2 --seeds 140 --out "$SM/smoke_nohold_crew2.json"
    s3=$?
    echo "exit $s3"
    [ "$s3" -eq 0 ] || rc=1
    # every step's exit status goes into the smoke verdict JSON (SMOKE_PASS needs all of them 0), so a smoke whose
    # unit tests failed never reads SMOKE_PASS, even when its id is reused for a run without afterok
    printf '{"pytest": %d, "smoke_hold_crew2": %d, "smoke_place_crew4": %d, "smoke_nohold_crew2": %d}\n' \
        "$t" "$s1" "$s2" "$s3" > "$SM/step_status.json" || rc=1
    write_verdict smoke "$SM/smoke_verdict.json" step_status="$SM/step_status.json" \
        hold_crew2="$SM/smoke_hold_crew2.json" place_crew4="$SM/smoke_place_crew4.json" \
        nohold_crew2="$SM/smoke_nohold_crew2.json" || rc=1
    rm -rf "$CACHE"
    "$PY" -c 'import json, sys; m = json.load(open(sys.argv[1])).get("measurement") or {}; print("MEASUREMENT (diagnostic, throw-away seed; never a verdict):", json.dumps({"hands": m.get("hands"), "cube": m.get("cube")}))' \
        "$SM/smoke_verdict.json" 2>/dev/null || echo "MEASUREMENT: MISSING"
    V=$(verdict_of "$SM/smoke_verdict.json")
    if [ "$V" = SMOKE_PASS ] && [ "$rc" -eq 0 ]; then
        echo "COOP-WRISTHOLD SMOKE: PASS (smoke_verdict.json reads $V; every step exited 0) | $SM | $LABEL"
        exit 0
    fi
    echo "COOP-WRISTHOLD SMOKE: FAIL (smoke_verdict.json reads $V; step status $rc) | $SM | $LABEL"
    exit 1
fi

# ---------------------------------------------------------------- run: bytes check against a passing smoke
SMOKE_JOB=${WRISTHOLD_SMOKE_JOB:-}
if [ -z "$SMOKE_JOB" ]; then
    echo "COOP-WRISTHOLD RESULT: REFUSED (run mode needs --export=ALL,WRISTHOLD_SMOKE_JOB=<smoke job id>)"
    exit 1
fi
SM=$LOGS/coop-wristhold-$SMOKE_JOB-smoke
SV=$(verdict_of "$SM/smoke_verdict.json")
if [ "$SV" != SMOKE_PASS ]; then
    echo "COOP-WRISTHOLD RESULT: REFUSED (smoke $SMOKE_JOB: $SM/smoke_verdict.json reads $SV, not SMOKE_PASS)"
    exit 1
fi
PT=$(pytest_status_of "$SM/smoke_verdict.json")
if [ "$PT" != 0 ]; then
    echo "COOP-WRISTHOLD RESULT: REFUSED (smoke $SMOKE_JOB: $SM/smoke_verdict.json records its unit tests as exit $PT, not 0)"
    exit 1
fi
if ! sha256sum -c --quiet "$SM/code_sha256.txt"; then
    echo "COOP-WRISTHOLD RESULT: REFUSED (the code differs from the bytes smoke $SMOKE_JOB ran)"
    exit 1
fi
if [ "$(cat "$SM/launcher_sha256.txt" 2>/dev/null)" != "$LAUNCHER_SHA" ]; then
    echo "COOP-WRISTHOLD RESULT: REFUSED (this launcher differs from the one smoke $SMOKE_JOB ran)"
    exit 1
fi
echo "BYTES-CHECK: PASS (smoke $SMOKE_JOB reads SMOKE_PASS, unit tests exit 0; code, test files and launcher sha256 match)"

RUN=$REPO/results/coop-wristhold-20261003
if [ -e "$RUN" ]; then
    echo "COOP-WRISTHOLD RESULT: REFUSED ($RUN exists; results are never overwritten)"
    exit 1
fi
mkdir -p "$RUN/probe" "$RUN/scored" || exit 1
CACHE=${TMPDIR:-/tmp}/bhl-coop-wristhold-$JOB
mkdir -p "$CACHE" || exit 1
rc=0

echo "=== STAGE 1: probe | crew 2 | lift_hold (20 s) | seeds 125-129"
score --wristhold-stage probe --protocol lift_hold --crew 2 --seeds 125-129 --out "$RUN/probe/score_probe_crew2.json"
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
        score --wristhold-stage hold --protocol lift_hold --crew "$n" --seeds 20-29 \
            --out "$RUN/scored/score_hold_crew$n.json"
        s=$?
        echo "exit $s"
        [ "$s" -eq 0 ] || rc=1
    done
    for n in 2 4; do
        echo "--- lift-place crew $n, seeds 30-39 (20 s)"
        score --wristhold-stage place --protocol lift_place --crew "$n" --seeds 30-39 \
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
echo "COOP-WRISTHOLD RESULT: probe $PV | scored $SV2 | $RUN/probe/verdict.json $RUN/scored/verdict.json | $LABEL"
echo "=== done: $(date) (step status $rc) ==="
[ "$rc" -eq 0 ] || exit 3
exit 0
