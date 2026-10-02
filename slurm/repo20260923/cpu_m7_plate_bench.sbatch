#!/bin/bash
#SBATCH --job-name=m7-plate-bench
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# Mission 7 plate bench (M2 of docs/SOLUTIONS_2026-10-01.md section 3), declared 2026-10-02 before any bench,
# replay or route episode ran with the TurnBoth-s0 stage gait.  CPU MuJoCo, no GPU, no render.
#   LEARNED gaits: the shipped arms-dr1.0-s0 gait outside the stage, TurnBoth-s0 (arms-turn-turnboth-s0, the one
#   qualified turning checkpoint -- one checkpoint of 12 turning-arm seeds, not a recipe) during the stage |
#   SCRIPTED stage: scripts/mission7_plate_stage.py PlateStage with --stage-gait turnboth | ORACLE layout and
#   plate pose (as the existing stage uses them).
#
# Budget: the Mission 7 line of 106 episodes the user authorized on 2026-10-01 ("ok do 106 episodes";
# "mission 7 dont worry about the budget implement"; SLURM_JOBS.md "User authorizations recorded 2026-10-01"):
# 64 bench crossings (this job) + 10 exact-replay episodes + 32 route-gate episodes.  The older line (docs
# "101 of 512", ledger "~53 over") stays unreconciled; the user set it aside and the report flags it.  A smoke
# (mode `smoke`) is NOT part of the 106 episodes: it runs two bench crossings on exploration train layouts >= 32,
# the replay's own --smoke layout (validation 0) with the pre-M2 stage, this stage and --stage-gait turnboth, and
# a route-controller integration run on exploration train layout 252 -- never a scored bench or route layout.
#
# PREDECLARED RULES (frozen; a chain, each step only after the previous PASSES):
# - Bench: PASS iff >= 62/64 crossings clear (base along >= +0.35 m past the plate centre along the door
#   direction, within 10 s of takeover, by mission7_gates' clear definition), 0 falls (tilt >= 0.78), and
#   >= 15/16 clears per heading. Otherwise FAIL; INCOMPLETE if any of the 64 is missing.
# - Then the unchanged exact ten-fall replay once with --stage-gait turnboth, through the existing replay-gate
#   machinery (complete, 10 episodes, 10 upright, 0 falls).
# - Then the route gate as coded by M1: Doors and Transport each >= 16/16 successes on validation layouts 0-15
#   (32 episodes).
# - M3 (turn while stepping with the shipped gait + a stall watchdog) is predeclared as conditional on a bench
#   FAIL; it is NOT implemented and would be its own request of the same size.
# The verdict is scripts/mission7_plate_bench.py bench_verdict(), written to <out>/verdict.json and read back
# from that JSON (never from an exit code).  Together the clauses allow at most 2 misses and at most 1 per
# heading (hence at most 1 per cell).  Nothing below is tuned on a bench, replay or route result.
#
# Grid (scripts/mission7_plate_bench.py grid()): one crossing per TRAINING-split layout L = 0..31
# (bhl_robust.mission.layout.generate("train", L), seeds 0..31) and door d = 0/1.  Entry heading relative to
# the door direction 0, +90, -90, 180 deg for L mod 4 = 0, 1, 2, 3; plate round (correct side) for
# floor(L/4) mod 2 = 0, square (wrong side) for 1; entry standstill for d XOR (floor(L/8) mod 2) = 0, walking for
# 1: 16 cells x 4 crossings, 2 door-0 + 2 door-1 per cell, 16 per heading (asserted in
# tests/test_mission7_plate_bench.py).  Layout sets checked 2026-10-02: the replay is the ten retained falls of
# results/mission7-replay-smoke-20260921 (validation 0, 1, 4, 5, 7, 9, 12, 13, 14, 15; seeds 10000+), the route
# gate validation 0-15 (seeds 10000-10015); the bench's train seeds 0-31 are disjoint from both.  Train layouts
# were used before only for the Approach stage (a held-out -x replication, 2026-09-23), never for door crossings.
#
# Stage gait (scripts/mission7_plate_stage.py, declared constants): at takeover env.controller.policy :=
# TurnBoth-s0 (policy.onnx sha256 562ceed7...5c95, deploy.yaml = the shipped one except the policy path and the
# unused command_velocity, both checked at load) and prev_actions := 0; approach to the pre-point and the 0.40 s
# settle unchanged; turn in place to the door direction at 0.40 rad/s until |error| < 0.15 rad (bound 2 pi/0.4 =
# 15.7 s); cross straight forward at 0.30 m/s x max(0, cos error), heading hold clip(1.2 x error, +-0.40), no
# lateral command, until the base is 0.35 m past the plate centre (= mission7_gates.CLEAR_ALONG_M; --cross-clear
# unset) or 4.0 s pass; turn back to the takeover heading the same way; at hand-back the shipped policy and
# prev_actions := 0.  The turn rate is DECLARED, not tuned: 0.40 rad/s is the largest yaw rate Mission 7's
# command interface admits (tanh x 0.4); TurnBoth-s0's scored maze controller (TurnWalkController) turns at
# 0.6, which that interface cannot command.  Disclosed risks, accepted before any scored run: TurnBoth-s0 was
# qualified at +-0.6 rad/s and its turn at 0.4 rad/s was unmeasured before the protocol was declared; smoke showed
# 2/3 turns timing out (below).  A 180-degree entry needs 0.40 s settle + (pi - 0.15)/0.40 = 7.48 s of turning +
# 0.65 m / 0.30 m/s = 2.17 s from the pre-point to +0.35 m = 10.05 s at nominal tracking, over the 10 s window,
# so the 180-degree heading can meet its >= 15/16 clause only if TurnBoth-s0 turns faster than commanded.
#
# Smoke observations, disclosed before the bench (smokes 21506958, 21507072, 21507112 on the declared protocol,
# identical outcomes, and 21507687 on the review-fixed code, identical to 21507112 sample for sample; exploration
# train layouts 250-252 and the replay's smoke layout only; not scored; no parameter, constant or protocol choice
# was changed in response):
#   - bench L250 d0 (-90 deg, round, walking): turn 2.60 s at ~0.57 rad/s, no time-out; the crossing then stalled
#     at along -0.353 m: no clear, no fall.
#   - bench L251 d1 (180 deg, round, standstill): the turn stalled (yaw flat near 0.97 rad at 12.0-14.0 s and near
#     2.18 rad at 16.8-25.0 s under the 0.40 rad/s command) and timed out after 15.8 s with 0.68 rad left; the
#     crossing timed out at along +0.187 m; the first clear came during turn_back, 20.7 s after takeover (outside
#     the 10 s window): no clear, no fall.
#   - exact replay, validation layout 0, --stage-gait turnboth: the turn timed out after 15.7 s with 0.85 rad
#     left; the crossing ended at along -0.347 m; no fall.
#   - route-controller integration (exploration train layout 252, early handoff): takeover at 18.8 s, then a fall
#     at 20.9 s in the stage's approach phase on TurnBoth-s0.
#   So 2/3 TurnBoth-s0 turns timed out and 0 of 2 bench smoke crossings cleared inside 10 s, which predicts a
#   bench FAIL (and with it M3's conditional request); the rule and the protocol stand as declared.
# Coverage: the bench takes over at the pre-point (within 0.13 m of it), so a bench PASS does not cover
# TurnBoth-s0's capture-to-pre-point approach (capture 0.78 m from the plate, the stage's unchanged world-frame
# approach law, which can command lateral or backward steps); the exact replay (10 episodes) exercises that
# phase first, before the route gate.
#
# Bench protocol (module docstring of scripts/mission7_plate_bench.py, constants there): per crossing a fresh
# DebugEnv(split train, seed 7000 + 2L + d); base placed at route[k-1]'s centre with the entry yaw; shipped gait
# 3.0 s at zero command; run-up to the pre-point (plate centre - 0.30 m along the door direction) with the stage's
# own approach law (0.30 m/s world frame) and the route's heading hold on the entry yaw; arrival within 0.13 m;
# walking -> takeover at arrival; standstill -> 3.0 s at zero command, then takeover.  The stage is polled at
# 5 Hz with the route's action transform; physics as in the exact replay (runner.step, doors closed, no
# MissionState termination); falls at any phase end the crossing and count; wall contacts and plate presses
# reported, not gated (the route gate applies Mission 7's own collision rule); a run-up not arrived in 20 s or a
# stage not handed back 60 s after takeover is a miss.  Run-ups from route[k-1] follow the route's geometry (27
# straight and 37 turned door cells).  In 17 of the 64 crossings the straight run-up line passes 0.114-0.131 m
# from the centre of the door's OPPOSITE plate (radius / half-side 0.24 m), i.e. over it, on the shipped gait
# before the takeover: L3 d0, L3 d1, L4 d1, L5 d0, L5 d1, L7 d0, L7 d1, L13 d0, L16 d0, L18 d1, L22 d1, L25 d0,
# L26 d0, L28 d1, L29 d0, L29 d1, L30 d0 (by heading 0: 3, +90: 6, -90: 4, 180: 4).  Over all 64 run-ups the
# body walks forward in 15, sideways in 33 and backward in 16; no run-up line passes within 0.24 m of the target
# plate's centre.  A fall there would come from the shipped gait in the run-up, not from the stage, but it
# counts as a bench fall and FAILS the bench (falls at any phase count).  Each crossing records both plates'
# distances from its run-up line, and verdict.json lists the 17 and the phase of every fall (reported, not
# gated).  Not adopted: a run-up routed around the plate.  The plates straddle the door cell's centreline (along
# -0.36..+0.12 m at lateral +-0.42 m), so a detour must pass in front of a plate (0.59 m to the door face in a
# 1.5 m cell) or behind it (0.35 m to a back wall where there is one) with a 0.316 m half-body: a new protocol
# with wall and door contact risk and no smoke coverage.  The route itself would normally stage the near plate
# (capture within 0.78 m) rather than walk over it.
#
# Chain wiring: after a scored run this job reads <out>/verdict.json.  PASS -> it runs
#   bash slurm/repo20260923/submit_m7_replay_gates.sh --only m2-turnboth --submit
# which queues the exact ten-fall replay (scripts/submit_mission7_plate_stage.py --stage-gait turnboth, pinned
# to cn-c22, source-frozen) and its follow-up (m7_replay_gate_followup.sbatch), which releases the route gate
# (Doors 16 + Transport 16, --stage-gait turnboth, lock slot 1) only on a 10/10.  FAIL or INCOMPLETE -> nothing
# is submitted and the chain stops.  DRY_RUN=1 prints the release instead of running it.  The release script
# itself refuses m2-turnboth (RELEASE_BLOCKED, exit 3, nothing queued) unless M7_BENCH_VERDICT (this job passes
# <out>/verdict.json; default results/mission7-campaign-20260923/m2-plate-bench/verdict.json) is a scored PASS
# and every source the replay snapshot copies still has the sha256 recorded in <out>/provenance.json, so neither
# a hand-run release nor a stage edit committed while the bench ran can queue a replay the bench did not validate.
# The bench refuses to start unless every physics source (scripts/mission7*.py, src/bhl_robust/mission, and the
# eval and sensor_io modules the replay snapshot freezes) matches HEAD; provenance.json records their sha256.
#
#   usage (from anywhere):
#     sbatch --job-name=m7-plate-bench-smoke --time=01:00:00 slurm/repo20260923/cpu_m7_plate_bench.sbatch smoke
#     sbatch --dependency=afterok:<smoke> slurm/repo20260923/cpu_m7_plate_bench.sbatch bench
#   outputs: bench -> results/mission7-campaign-20260923/m2-plate-bench/ (refused if it exists; scored mode also
#   refuses uncommitted bench sources); smoke -> results/mission7-campaign-20260923/m2-bench-smoke/job-<id>/.
#   The smoke exits non-zero on any machinery failure; it is not scored and never releases anything.  Smoke mode
#   refuses to run (exit 2) unless the Slurm job name contains "smoke" and the job's own time limit is <= 1:00:00.
set -euo pipefail
mode=${1:?usage: cpu_m7_plate_bench.sbatch bench|smoke}
REPO=${M7_REPO:-/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder}
PY=${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}
CAMPAIGN=results/mission7-campaign-20260923
REPLAY_SOURCE=results/mission7-replay-smoke-20260921
REPLAY_BASELINE=results/mission7-approach-followup-20260922/replay-diagnose-cn-c22
RELEASE=${M7_BENCH_RELEASE:-slurm/repo20260923/submit_m7_replay_gates.sh}
DRY_RUN=${DRY_RUN:-0}
# The pre-M2 stage (the file every replay gate up to 2026-09-24 ran; sha256 recorded in their submission.json).
REFERENCE_COMMIT=094797e
REFERENCE_STAGE_SHA256=fdde7a559bc92d047cc10f6c66775eb24e444db7b211404378bd0e65d3555576
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "m7 plate bench: mode=$mode host=$(hostname) job=${SLURM_JOB_ID:-local} head=$(git rev-parse --short HEAD) dry_run=$DRY_RUN"

# Release the exact replay only on a bench PASS read from verdict.json.
release_if_pass() {
  local out=$1 verdict abs_out release_tmp rc=0
  verdict=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/verdict.json")
  case "$out" in /*) abs_out=$out ;; *) abs_out=$REPO/$out ;; esac
  case "$verdict" in
    PASS)
      echo "M7 BENCH PASS -> releasing the exact ten-fall replay with --stage-gait turnboth (tag m2-turnboth)"
      if [ "$DRY_RUN" != 0 ]; then
        echo "DRY_RUN: would run: M7_BENCH_VERDICT=$abs_out/verdict.json bash $RELEASE --only m2-turnboth --submit"
        return 0
      fi
      # the release script proves its sbatch shim directory executes before it queues anything; give it one on
      # the project filesystem rather than whatever TMPDIR this node provides
      release_tmp="$abs_out/.release-tmp"
      mkdir -p "$release_tmp"
      # The release script re-reads this verdict and compares the sources it will snapshot with this bench's
      # provenance.json (m2_bench_gate); exit 3 = RELEASE_BLOCKED, nothing queued.
      M7_BENCH_VERDICT="$abs_out/verdict.json" TMPDIR="$release_tmp" bash "$RELEASE" --only m2-turnboth --submit || rc=$?
      if [ "$rc" = 3 ]; then
        echo "RELEASE_BLOCKED: the release script refused m2-turnboth (not a scored PASS, or a source the replay would"
        echo "  snapshot differs from $out/provenance.json); nothing was queued; the verdict stands in $out/verdict.json."
        return 1
      elif [ "$rc" != 0 ]; then
        echo "M7 BENCH RELEASE FAILED (exit $rc): the PASS stands in $out/verdict.json; run by hand once the cause is fixed:"
        echo "  cd $REPO && M7_BENCH_VERDICT=$abs_out/verdict.json bash $RELEASE --only m2-turnboth --submit"
        return 1
      fi ;;
    FAIL)
      echo "M7 BENCH FAIL -> the chain stops: the replay and the route gate are NOT submitted."
      echo "M3 (turn while stepping with the shipped gait + a stall watchdog) is predeclared as conditional on this FAIL;"
      echo "it is not implemented and needs its own request (same size, 106 episodes)." ;;
    *)
      echo "M7 BENCH $verdict -> nothing released (a bench that is not complete is not a verdict)." ;;
  esac
}

if [ "$mode" = bench ]; then
  OUT=${M7_BENCH_OUT:-$CAMPAIGN/m2-plate-bench}
  [ ! -e "$OUT" ] || { echo "refusing: $OUT exists (a bench result is never overwritten)" >&2; exit 2; }
  # the physics sources: the replay snapshot's set (scripts/submit_mission7_plate_stage.py) minus its sbatch, plus
  # this launcher; scripts/mission7_plate_bench.py BENCH_SOURCES lists the same files for provenance.json
  sources=(scripts/mission7*.py src/bhl_robust/mission src/bhl_robust/__init__.py src/bhl_robust/sensor_io.py
           src/bhl_robust/eval/__init__.py src/bhl_robust/eval/multi_robot.py src/bhl_robust/eval/mjcf_assets.py
           src/bhl_robust/eval/livery.py src/bhl_robust/eval/team_sensors.py slurm/repo20260923/cpu_m7_plate_bench.sbatch)
  for f in scripts/mission7_plate_bench.py slurm/repo20260923/cpu_m7_plate_bench.sbatch; do
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "refusing: $f is not committed" >&2; exit 2; }
  done
  if ! git diff --quiet HEAD -- "${sources[@]}"; then
    echo "refusing: bench sources differ from HEAD; commit them first:" >&2; git status --short -- "${sources[@]}" >&2; exit 2
  fi
  "$PY" scripts/mission7_plate_bench.py --repo "$REPO" --out "$OUT" --mode scored --preflight
  "$PY" scripts/mission7_plate_bench.py --repo "$REPO" --out "$OUT" --mode scored
  release_if_pass "$OUT"
  echo "M7_PLATE_BENCH_COMPLETE"
  exit 0
fi

[ "$mode" = smoke ] || { echo "unknown mode $mode (bench|smoke)" >&2; exit 2; }

# Smoke limits (fail closed): a Slurm job whose name contains "smoke" and whose own time limit is <= 1:00:00.
smoke_limits_ok() {
  local name=${SLURM_JOB_NAME:-} id=${SLURM_JOB_ID:-} limit a b c seconds
  case "$name" in *smoke*) ;; *) echo "refusing: smoke mode needs a Slurm job name containing 'smoke' (got '$name')" >&2; return 1 ;; esac
  [ -n "$id" ] || { echo "refusing: smoke mode runs only as a Slurm job (no SLURM_JOB_ID)" >&2; return 1; }
  for _ in 1 2 3; do
    limit=$(squeue -h -j "$id" -o %l 2>/dev/null | head -n 1 | tr -d '[:space:]') || limit=""
    [ -z "$limit" ] || break
    sleep 10
  done
  # squeue prints M:SS below an hour, H:MM:SS below a day, D-HH:MM:SS above; anything else is refused
  case "$limit" in
    *[!0-9:]*|""|:*|*:|*::*|*:*:*:*) echo "refusing: smoke time limit '$limit' is not <= 1:00:00" >&2; return 1 ;;
  esac
  IFS=: read -r a b c <<<"$limit"
  if [ -n "$c" ]; then seconds=$(( 10#$a * 3600 + 10#$b * 60 + 10#$c ))
  elif [ -n "$b" ]; then seconds=$(( 10#$a * 60 + 10#$b ))
  else echo "refusing: smoke time limit '$limit' is not <= 1:00:00" >&2; return 1; fi
  [ "$seconds" -le 3600 ] || { echo "refusing: smoke time limit $limit is over 1:00:00" >&2; return 1; }
  echo "smoke limits: job $id '$name', time limit $limit"
}
smoke_limits_ok || exit 2
SMOKE=${M7_SMOKE_DIR:-$CAMPAIGN/m2-bench-smoke/job-${SLURM_JOB_ID:-local}}
case "$SMOKE" in *-smoke/*|*-smoke) ;; *) echo "refusing: smoke output must sit under a *-smoke directory" >&2; exit 2 ;; esac
[ ! -e "$SMOKE" ] || { echo "refusing: $SMOKE exists" >&2; exit 2; }
mkdir -p "$SMOKE/reference"
SMOKE_CROSSINGS=251:1,250:0   # exploration train layouts: 251 d1 = 180 deg round standstill, 250 d0 = -90 deg round walking
start=$(date +%s)

# 1. preflights (imports, TurnBoth-s0 hash + deploy.yaml equivalence, argument parsing)
"$PY" scripts/mission7_plate_bench.py --repo "$REPO" --out "$SMOKE/bench" --mode smoke --crossings "$SMOKE_CROSSINGS" --preflight
"$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/replay-turnboth" --smoke --stage-gait turnboth --preflight

# 2. default-path identity in MuJoCo: the pre-M2 stage file and this one, shipped gait, replay layout 0, same node
git show "$REFERENCE_COMMIT:scripts/mission7_plate_stage.py" > "$SMOKE/reference/mission7_plate_stage.py"
echo "$REFERENCE_STAGE_SHA256  $SMOKE/reference/mission7_plate_stage.py" | sha256sum -c -
"$PY" "$SMOKE/reference/mission7_plate_stage.py" --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/default-reference" --smoke
"$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/default-new" --smoke
cmp "$SMOKE/default-reference/episodes.json" "$SMOKE/default-new/episodes.json"
cmp "$SMOKE/default-reference/result.json" "$SMOKE/default-new/result.json"
echo "SMOKE default path: episodes.json and result.json byte-identical to the pre-M2 stage"

# 3. the exact replay machinery with --stage-gait turnboth on one layout (the replay's own --smoke selection)
"$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/replay-turnboth" --smoke --stage-gait turnboth

# 4. two bench crossings end to end (one standstill, one walking), exploration layouts only
"$PY" scripts/mission7_plate_bench.py --repo "$REPO" --out "$SMOKE/bench" --mode smoke --crossings "$SMOKE_CROSSINGS"

# 4b. the route-gate controller with --stage-gait turnboth on an EXPLORATION train layout (never a route-gate
#     layout): RouteHandoffController with the follow-up's route composition, Doors stage, until the first
#     hand-back + 3 s or 120 s.  Integration only: nothing here is scored.
"$PY" - "$SMOKE/route-integration" "$REPO" <<'PYEOF'
import json, pathlib, sys
import numpy as np
from bhl_robust.mission.approach_debug import DebugEnv
from mission7_route_handoff_probe import RouteHandoffController
out, repo = pathlib.Path(sys.argv[1]), sys.argv[2]
out.mkdir(parents=True)
env = DebugEnv(repo, out / "cache", stage="doors", split="train", seed=4252)
env.reset(252)
shipped = env.controller.policy
controller = RouteHandoffController(env, handoff_mode="early", stage_activate=True, exit_ramp_s=1.2,
                                    stage_wait_open_s=0.0, rejoin_advance=True, exit_ramp_center=True,
                                    stage_gait="turnboth")
done = False
while not done and env.runner.d.time < 120.:
    env.runner.contact_trace = []
    action = controller.action()
    _, _, done, _ = env.step(action)
    controller.observe_contacts(env.runner.contact_trace)
    controller.observe_step(action)
    events = controller.stage.gait_events
    if events and events[-1]["event"] == "swap_to_shipped" and env.runner.d.time > events[-1]["time_s"] + 3.:
        break
summary = {"layout": ["train", 252], "time_s": float(env.runner.d.time), "done": bool(done),
           "failure": env.state.failure, "open": list(env.state.open), "crossed": list(env.state.crossed),
           "stage_activations": controller.stage_activation_times, "phases": [h["phase"] for h in controller.stage.history],
           "history": controller.stage.history,
           "gait_events": controller.stage.gait_events, "exit_ramp_events": controller.exit_ramp_events,
           "rejoin_advance_events": controller.rejoin_advance_events,
           "policy_restored": env.controller.policy is shipped if controller.stage.phase == "recorded" else None}
(out / "route_integration.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
print("ROUTE_INTEGRATION", json.dumps({k: summary[k] for k in ("time_s", "done", "failure", "phases", "policy_restored")}))
PYEOF

# 5. machinery checks; any failure exits non-zero
"$PY" - "$SMOKE" "$start" <<'PYEOF'
import gzip, json, pathlib, sys, time
import mission7_gates as gates
smoke, start = pathlib.Path(sys.argv[1]), float(sys.argv[2])
checks, problems = {}, []

def check(name, ok, detail=None):
    checks[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        problems.append(name)

def gait_sequence_ok(history, events):
    """Every takeover swaps in TurnBoth-s0 with prev_actions zeroed; every hand-back swaps it out the same way."""
    takeovers = [h["time_s"] for h in history if h["phase"] == "approach"]
    handbacks = [h["time_s"] for h in history if h["phase"] == "recorded"]
    swaps_in = [e for e in events if e["event"] == "swap_to_turnboth"]
    swaps_out = [e for e in events if e["event"] == "swap_to_shipped"]
    return ([e["time_s"] for e in swaps_in] == takeovers and [e["time_s"] for e in swaps_out] == handbacks
            and all(e["prev_actions_after_norm"] == 0.0 for e in events)
            and [e["event"] for e in events] == (["swap_to_turnboth", "swap_to_shipped"] * len(takeovers))[:len(events)])

result = json.loads((smoke / "replay-turnboth/result.json").read_text())
rows = result["episode_comparison"]
check("replay_turnboth_complete", result.get("complete") and result.get("status") == "COMPLETED_SMOKE"
      and result.get("episodes") == 1 and result.get("stage_gait") == "turnboth", result.get("status"))
row = rows[0] if rows else {}
check("replay_turnboth_gait_events", rows and gait_sequence_ok(row.get("stage_history", []), row.get("gait_events", [])),
      {"phases": [h["phase"] for h in row.get("stage_history", [])], "gait_events": row.get("gait_events")})
try:
    traced = gates.replay_gate_crossings(gates.replay_trace_path(smoke / "replay-turnboth"), rows)
    check("replay_turnboth_real_clears_readable", True, {k: traced[k] for k in ("crossings", "real_clears")})
except Exception as exc:
    check("replay_turnboth_real_clears_readable", False, f"{type(exc).__name__}: {exc}")
check("replay_turnboth_status_reported", True, {"fall": row.get("fall"), "first_fall_s": row.get("first_fall_s")})

verdict = json.loads((smoke / "bench/verdict.json").read_text())
check("bench_smoke_verdict_never_pass", verdict["verdict"] == "SMOKE_INCOMPLETE", verdict["verdict"])
crossings = sorted((smoke / "bench/crossings").glob("*.json.gz"))
check("bench_two_crossings_written", len(crossings) == 2, [p.name for p in crossings])
entries = set()
for path in crossings:
    with gzip.open(path, "rt") as stream:
        c = json.load(stream)
    entries.add(c["entry"])
    name = path.name.split(".")[0]
    check(f"{name}_complete", c["complete"] and c["end_reason"] in ("complete", "fall", "runup_timeout", "stage_timeout"),
          c["end_reason"])
    check(f"{name}_gait_events", gait_sequence_ok(c["stage_history"], c["gait_events"]), c["gait_events"])
    check(f"{name}_outcome_reported", True, {k: c[k] for k in ("heading_deg", "plate", "entry", "end_reason", "takeover_s",
          "handback_s", "clear", "real_clear", "clear_after_takeover_s", "fell", "fall_phase", "maximum_tilt")}
          | {"takeover": next((e for e in c["events"] if e["event"] == "takeover"), None),
             "phases": [h["phase"] for h in c["stage_history"]], "wall_contacts": c["wall_contacts"],
             "plate_activation": c["plate_activation"], "samples": len(c["samples"])})
check("bench_one_standstill_one_walking", entries == {"standstill", "walking"}, sorted(entries))
route = json.loads((smoke / "route-integration/route_integration.json").read_text())
check("route_integration_gait_events", gait_sequence_ok(route["history"], route["gait_events"])
      and route["policy_restored"] in (True, None), route)
# informational, never a failure: whether the route reached a plate in its 120 s is the route's own outcome
check("route_integration_stage_exercised_reported", True,
      "the stage took over" if route["phases"] else "not exercised (no takeover before the route ended or 120 s)")
summary = {"ok": not problems, "problems": problems, "wall_s": time.time() - start, "checks": checks}
(smoke / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
print(json.dumps(summary, indent=2, default=str))
sys.exit(0 if not problems else 1)
PYEOF
echo "M7_PLATE_BENCH_SMOKE_PASS (machinery only; not scored; $(( $(date +%s) - start )) s)"
