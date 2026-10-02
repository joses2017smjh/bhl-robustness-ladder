#!/bin/bash
#SBATCH --job-name=m7-plate-bench-v2-clocks2
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# Mission 7 plate bench v2 with arms-turngait-clock-s2 as the stage gait (workstream m7-clocks2; frozen design:
# SLURM_JOBS.md, "User approval recorded 2026-10-02 14:15", item (A)), declared 2026-10-02 before any bench-v2, replay
# or route episode ran with --stage-gait clocks2.  CPU MuJoCo, no GPU, no render.  The bench, its grid, its protocol
# and its rule are bench v2's (scripts/mission7_plate_bench_v2.py, slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch,
# whose header stays the M3 predeclaration of record); this launcher changes the stage gait only.
#   LEARNED gaits: the shipped arms-dr1.0-s0 gait outside the stage + the LEARNED clock-s2 stage gait
#   (arms-turngait-clock-s2, Turning R1's QUALIFIED seed, a 77-observation gait-clock policy) during the stage |
#   SCRIPTED stage: scripts/mission7_plate_stage.py PlateStage with --stage-gait clocks2 | ORACLE layout and plate pose
#   (as the existing stage uses them).
#
# PREDECLARED RULE (frozen before any episode; bench v2's rule verbatim, unchanged; the chain as M3's, each step only
# after the previous PASSES):
# - Bench v2: with N crossings run (64 minus the dropped set) and N_h per heading, bench v2 PASSES iff clears >=
#   N - 2, 0 falls, and clears >= N_h - 1 for every heading. Otherwise FAIL; INCOMPLETE if any declared crossing is
#   missing. The clear definition is unchanged: mission7_gates' real clear (base along >= +0.35 m past the plate
#   centre along the door direction), within 10 s of takeover.
# - Then the unchanged exact ten-fall replay once with --stage-gait clocks2, through the existing replay-gate
#   machinery (complete, 10 episodes, 10 upright, 0 falls; release arm m7-clocks2).
# - Then the route gate as coded by M1: Doors and Transport each >= 16/16 successes on validation layouts 0-15
#   (32 episodes).
# Here N = 41 (0 deg: 12, +90: 11, -90: 10, 180: 8), so PASS needs >= 39/41 clears, 0 falls, and >= 11, 10, 9, 7 clears
# at 0, +90, -90, 180 deg.  The verdict is scripts/mission7_plate_bench_v2.py bench_v2_verdict(), unchanged, written to
# <out>/verdict.json and read back from that JSON (never from an exit code).  Falls (tilt >= 0.78) count in EVERY
# phase after the spawn: settle, walk, stage, post.  Nothing below is tuned on a bench, replay or route result, and no
# parameter was chosen from a result.
#
# Budget: 41 bench-v2 crossings (this job) + 10 exact-replay episodes + 32 route-gate episodes = 83 more Mission 7
# episodes (the frozen design's line; flagged in the report).  The smoke (mode `smoke`) is NOT part of the line: it
# runs on exploration train layouts >= 250 only and never on a validation layout or a train layout 0-31.
#
# The stage gait (scripts/mission7_plate_stage.py, EXPORT_STAGE_GAITS / CLOCKS2_*): at takeover the stage swaps the
#   CONTROLLER, not only the policy: env.controller := a fresh bhl_robust.eval.gait_clock.make_controller(clock-s2's
#   deploy.yaml) = a GaitClockRlController (77 observations: upstream's 75 + [sin, cos] of the gait clock, period
#   0.8 s) running clock-s2's policy; prev_actions = 0, the observation buffer is zero and the gait-clock phase starts
#   at 0, as at an Isaac episode start.  At hand-back env.controller := the shipped controller object it replaced and
#   prev_actions := 0.  MissionEnv and the route probe keep building their RlController directly; only PlateStage
#   swaps (env.controller is read at every gait update by MissionEnv.step, the replay and the benches).
#   Export: external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/2026-10-02_03-58-04_arms-turngait-clock-s2/exported,
#   policy.onnx sha256 c1862f1e4429c1e6ac2e4fae1391f3f40ccc353382c8bbf4b88efee4c6c8d8de (pinned, checked at load);
#   its deploy.yaml must equal the shipped gait's in every key but policy_checkpoint_path, command_velocity,
#   num_observations and gait_clock (checked at load: the runner's PD gains, effort limits, default pose and dt come
#   from the shipped cfg for the whole episode), 77 observations with the frozen 0.8 s / 0.0 clock, ONNX input 77.
#   Qualification (results/repo-gpu-20260923/turngait-r12-20261001/qualify/arms-turngait-clock-s2__qualify.json):
#   turn 10/10 at +-0.6 rad/s, walk 2/3, push 9/60 = 0.15 (at the limit); one checkpoint, not a recipe.
# The stage LAW is M2's turnboth law, unchanged (the same PlateStage._turnboth_command and TURNBOTH_* constants), to
#   isolate the gait factor: after the unchanged approach and settle, turn in place at [0, 0, +-0.40 rad/s] until
#   |error| < 0.15 rad (bound 2 pi/0.40 = 15.7 s); cross straight at [0.30 m/s x max(0, cos error), 0,
#   clip(1.2 x error, +-0.40)] (no lateral command) until the base is 0.35 m past the plate centre (=
#   mission7_gates.CLEAR_ALONG_M) or the stage's 4.0 s cross bound; turn back to the takeover heading the same way;
#   hand back.  0.40 rad/s is the largest yaw rate Mission 7's command interface admits (tanh x 0.4).
#
# Disclosed before any scored episode:
# - clock-s2 was qualified at +-0.6 rad/s and is UNTESTED at 0.40 rad/s.
# - 180-deg timing: a 180-deg entry needs (pi - 0.15)/0.40 = 7.48 s (about 7.5 s) of turning + the settle (0.40 or
#   0.60 s as polled) + 0.65 m / 0.30 m/s = 2.17 s (about 2.2 s) of crossing = 10.05-10.25 s (about 10.1-10.3 s)
#   against the 10 s clear window, so the 180-deg clause (>= 7/8) is likely to fail on timing alone unless the gait
#   turns faster than commanded.  That clause and window are an interface question for the user, not something a gait
#   fixes; they are kept as declared.
# - The stage acts after the unchanged approach and settle, not at takeover (M2's and M3's pattern): the controller
#   swap happens at takeover, so clock-s2 also stands through the settle.  Timing as polled (inherited, not changed):
#   the settle comparison (`now < phase_until`) has no epsilon, so at the bench's 0.2 s polling the declared 0.40 s
#   settle lasts 0.40 or 0.60 s, and the cross bound 4.0 or 4.2 s.
# - Bench v2 was introduced after the M2 bench FAIL (21507962) revealed the run-up confound; M2's FAIL (8/64) and M3's
#   FAIL on bench v2 (21514946: 16/41, 5 falls) stand; v1, bench v2 and their launchers are unchanged.
# - Route probe: its chain trace (--chain-trace) and the prev_actions-reset delivery check wrap the SHIPPED
#   controller's update; while clock-s2's controller is swapped in they record nothing.  The route gate composition
#   (follow-up step 3) uses neither (--rejoin-fix none, no --chain-trace).
# - Replay snapshot: with --stage-gait clocks2 both snapshot submitters also copy src/bhl_robust/eval/gait_clock.py
#   (the snapshot jobs import only the snapshot); the default snapshot set is unchanged.
# - The replay CLI with --stage-gait clocks2 runs no episode before the chain reaches it (its smoke layout, validation
#   0, is scored): it is exercised through its preflight, and its per-update loop reads env.controller as both benches
#   and MissionEnv.step do.
#
# Smoke observations, disclosed before any scored episode (smoke 21516422, cn-c21, 5 min, PASS; exploration train
# layouts 250-253 only; not scored; no constant, parameter or protocol choice changed in response).  It is a MACHINERY
# smoke of the uncommitted working tree (HEAD 1bf3e3a plus uncommitted edits), not of the code a scored run uses.
# Edited after it, before any scored episode: in this launcher comments only (this paragraph, the replay-CLI bullet
# above, the Coverage paragraph below and the route-gate snapshot note under "Chain wiring"; with exactly those lines
# removed the file has the sha256 that the smoke's bench provenance recorded for it,
# f4e9a21a31d55fe283f511670c6f42b5de30294c1393f951e32e7539a39c0b8e); scripts/mission7_route_handoff_probe.py run()
# (the export check moved before the first episode; run() is not exercised by the smoke); the release script, the
# follow-up and the tests.  So the committed code is smoked again (mode smoke) and the bench runs afterok on it:
#   - machinery: the turnboth and m3 bench-v2 records (L253 d1), the shipped stage path (L253 d1), the v1 bench (L250 d0)
#     and the route probe's default path (train 252) are byte-identical to the committed M3 code's (e33e44f); every
#     swap-in was a fresh GaitClockRlController, 77 wide, at clock step 0, and at every hand-back its clock step equalled
#     the gait updates since takeover (280, 525, 325), with the shipped controller back.
#   - bench v2 L251 d1 (180 deg, round, standstill): no turn time-out; the crossing began 0.60 m before the plate centre
#     (P is 0.30 m before it) and reached +0.19 m within the 4.0 s cross bound: no clear, no fall, no wall contact.
#   - bench v2 L250 d0 (+90 deg, round, walking, explicit labels): the crossing began 0.87 m before the plate centre and
#     reached -0.71 m: no clear, no fall; wall contact in 47 stage samples.
#   - bench v2 L253 d1 (+90 deg, square, standstill): a clear 9.20 s after takeover, no fall; wall contact in 6 samples.
#   - route integration (train 252, early handoff): takeover at 18.8 s; the route episode ended with Mission 7's
#     excessive_collision at 28.9 s, during the clock-s2 turn.
#   So 1 of 3 smoke bench crossings cleared inside 10 s, with 0 falls, and in two of them the in-place turn moved the base
#   away from the plate (0.30-0.57 m) before the crossing began.  A bench-v2 FAIL is likely; the rule and the protocol
#   stand as declared.
#
# Coverage (disclosed before any scored episode; M2's coverage note, same pattern): the bench takes over at the
#   pre-point (within 0.13 m of it; in all three smoke crossings the approach ended at the takeover instant), so a bench
#   PASS does not cover clock-s2 driving the stage's capture-to-pre-point approach.  In the exact replay and the route
#   gate, PlateStage._start swaps the controller in and opens the approach in the same call, at capture (0.78 m from
#   the plate), and the stage's unchanged world-frame approach law (0.30 m/s straight toward the pre-point, turned into
#   body-frame vx/vy; yaw command 0, --align-yaw being refused with clocks2) can command lateral or backward steps.
#   Smoke route run (train 252): 6.8 s of approach on clock-s2 (18.8 -> 25.6 s, maximum tilt 0.18 rad, no fall), then
#   the settle, the turn and Mission 7's excessive_collision at 28.9 s during the turn.  The exact replay (10 episodes)
#   exercises that phase first, before the route gate.
#
# Grid (identical to v1, mission7_plate_bench.grid()): one crossing per TRAINING-split layout L = 0..31 (seeds 0..31)
# and door d = 0/1.  Entry heading relative to the door direction 0, +90, -90, 180 deg for L mod 4 = 0, 1, 2, 3; plate
# round (correct side) for floor(L/4) mod 2 = 0, square (wrong side) for 1; entry standstill for d XOR
# (floor(L/8) mod 2) = 0, walking for 1.  Disjoint from the replay (validation 0, 1, 4, 5, 7, 9, 12-15) and the route
# gate (validation 0-15).
#
# Entry protocol (scripts/mission7_plate_bench_v2.py; NO run-up):
#   P = plate centre - 0.30 m along the door direction (PlateStage's pre-point); entry yaw = door yaw + heading.
#   Spawn = the env's own standing pose (DebugEnv(stage doors, split train, seed 7000 + 2L + d) reset to L: default
#   joints + N(0, 0.02) rad, base at the model's qpos0 height, which puts the lowest foot geom 0.039 m above the floor
#   and 0.009 m above a plate top), moved in x, y and yaw only, velocity zeroed (as v1); then 3.0 s at zero command on
#   the shipped gait (the settle).
#   standstill: spawn at P, settle 3.0 s, take over.
#   walking: spawn at S = P - 0.30 m along the entry-yaw unit vector, settle 3.0 s, walk FORWARD at 0.30 m/s with the
#   route's heading hold clip(1.2 x yaw error, +-0.35) on the entry yaw for the nominal 1.0 s, take over at arrival
#   (within 0.13 m of P, as v1); not arrived after 3.0 s of walking = a miss.
#   Takeover = PlateStage._start(d, side) on the bench's plate, PlateStage.command until hand-back, then 2.0 s at zero
#   command; polled every 5 gait updates (0.2 s) through the route's action transform; runner.step physics, doors
#   closed, no MissionState termination; a stage not handed back 60 s after takeover is a miss.  Wall/door contacts,
#   spawn contacts and plate presses are recorded per crossing and reported, not gated.
# Drop rule (walking entries only; layout geometry alone, before any episode): a walking crossing runs only if no wall
#   or closed door lies within the robot's planar radius 0.316 m (measured envelope 0.632 m elbow to elbow; FK of the
#   default pose 0.295 m) of the straight segment S -> P, and no foot box (FK of the default pose: x -0.080..+0.143 m,
#   |y| 0.021..0.093 m in the base frame) lies on any plate at S.  DROPPED (23 of the 32 walking crossings; all 8
#   walking 180-deg crossings drop by construction, S being the plate centre):
#     L3 d1, L4 d1, L5 d1, L6 d1, L7 d1, L9 d0, L10 d0, L11 d0, L12 d0, L13 d0, L14 d0, L15 d0, L16 d1, L19 d1, L21 d1,
#     L22 d1, L23 d1, L24 d0, L26 d0, L27 d0, L29 d0, L30 d0, L31 d0
#   Run: 32 standstill + 9 walking (L0 d1, L1 d1, L2 d1, L8 d0, L17 d1, L18 d1, L20 d1, L25 d0, L28 d0) = 41.
# Spec tension, disclosed (not dropped, not shifted: the spec places every standstill base directly at P): in 6
#   standstill crossings, all in 1.5 m cells, the spawn pose starts with an elbow inside a wall (mj_forward at the spawn
#   instant, dist < 0): L4 d0 6.6 mm, L5 d0 6.9, L11 d1 1.3, L22 d0 4.2, L24 d1 7.0, L30 d1 9.3 mm (P lies 0.29 m from
#   the wall face against the 0.294 m elbow reach).  Every standstill spawn at P also has a foot over the plate edge
#   (P is 0.06 m from the plate edge; the feet reach 0.08-0.14 m from the base), the replay's own settle point.
#
# Chain wiring: after a scored run this job reads <out>/verdict.json.  PASS -> it runs
#   M7_CLOCKS2_BENCH_VERDICT=<out>/verdict.json bash slurm/repo20260923/submit_m7_replay_gates.sh --only m7-clocks2 --submit
# (the exact ten-fall replay with --stage-gait clocks2, pinned to cn-c22, source-frozen with gait_clock.py in the
# snapshot, then its follow-up, which releases the route gate with --stage-gait clocks2 only on a 10/10).  The release
# script refuses (RELEASE_BLOCKED, exit 3, nothing queued) unless that verdict is a scored bench-v2 clocks2 PASS whose
# provenance.json records clock-s2's pinned policy sha256 and every replay-snapshot source (gait_clock.py included)
# still has the sha256 recorded there.  FAIL or INCOMPLETE -> nothing is submitted; the chain stops.  DRY_RUN=1 prints
# the release.  The bench refuses an existing output directory and any physics source that differs from HEAD.
#   Route-gate snapshot (a gap inherited from M2/M3; disclosed, not closed; the chain stays M3's): the follow-up releases
#   the route gate through scripts/submit_mission7_route_handoff.py, which snapshots the LIVE tree at that moment
#   (scripts/mission7*.py, src/bhl_robust/mission/*.py, the listed src/bhl_robust/eval files and, for clocks2,
#   src/bhl_robust/eval/gait_clock.py) and records their sha256 in its submission.json; nothing compares them with the
#   bench's provenance or the replay's snapshot before that release.  None of those files is to be edited between this
#   bench and the route-gate submission; the route gate's submission.json source_sha256 is compared with the replay
#   gate's after the fact.
#
# Smoke (machinery only, never a verdict, never a scored layout; exits non-zero on any failure):
#   1. preflights: bench v2 --stage-gait clocks2; the replay CLI with --stage-gait clocks2 and with --stage-gait export
#      --stage-gait-export <clock-s2's export> (preflight only: no replay episode, validation layouts are scored); the
#      route probe with --stage-gait clocks2 (preflight only).
#   2. default-path identity against the committed M3 code (e33e44f; stage, bench-v2 and probe sha256 pinned below),
#      exploration layouts only, byte-for-byte: the shipped stage path (bench-v2 run_crossing, stage_gait shipped), the
#      turnboth and m3 paths (bench-v2 smoke crossing), the v1 bench (its L250 d0 smoke crossing) and the route probe's
#      default path (train layout 252, the route gate's composition, until the first hand-back + 3 s or 60 s).
#   3. three bench-v2 crossings with --stage-gait clocks2: L251 d1 (180 deg, round, standstill), L250 d0 with explicit
#      labels (+90 deg, round, walking) and L253 d1 (+90 deg, square, standstill).
#   4. the route probe's controller with --stage-gait clocks2 on exploration train layout 252 (MissionEnv.step reads
#      the swapped controller): until the first hand-back + 3 s or 120 s.
#   5. machinery checks: the swap-in event is a GaitClockRlController, 77 wide, clock step 0, zero buffer and
#      prev_actions; at hand-back the stage controller's clock step equals the gait updates since takeover
#      ((hand-back - takeover) / 0.04 s, i.e. its phase started at 0 at takeover) and its last frame carries that
#      step's clock; the shipped controller (75 wide, prev_actions 0) is back; provenance and labels name clock-s2.
#
#   usage (from anywhere):
#     sbatch --job-name=m7-plate-bench-v2-clocks2-smoke --time=01:00:00 slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch smoke
#     sbatch --dependency=afterok:<smoke> slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch bench
#   outputs: bench -> results/mission7-campaign-20260923/clocks2-plate-bench-v2/; smoke ->
#   results/mission7-campaign-20260923/clocks2-bench-v2-smoke/job-<id>/.  The smoke refuses to run (exit 2) unless the
#   job name contains "smoke" and the job's own time limit is <= 1:00:00.
set -euo pipefail
mode=${1:?usage: cpu_m7_plate_bench_v2_clocks2.sbatch bench|smoke}
REPO=${M7_REPO:-/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder}
PY=${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}
CAMPAIGN=results/mission7-campaign-20260923
REPLAY_SOURCE=results/mission7-replay-smoke-20260921
REPLAY_BASELINE=results/mission7-approach-followup-20260922/replay-diagnose-cn-c22
RELEASE=${M7_BENCH_RELEASE:-slurm/repo20260923/submit_m7_replay_gates.sh}
DRY_RUN=${DRY_RUN:-0}
STAGE_GAIT=clocks2   # the stage gait predeclared for this launcher
ARM=m7-clocks2
CLOCKS2_EXPORT=external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/2026-10-02_03-58-04_arms-turngait-clock-s2/exported
CLOCKS2_POLICY_SHA256=c1862f1e4429c1e6ac2e4fae1391f3f40ccc353382c8bbf4b88efee4c6c8d8de
# The committed M3 code (bench v2 + stage + probe, = the m3 bench's provenance) for the identity checks.
REFERENCE_COMMIT=e33e44f
REFERENCE_STAGE_SHA256=22418834da39658ba4ff2120044bcab6f60cc5f6b5be04e34061f9f8997e7deb
REFERENCE_BENCH_V2_SHA256=8a1b08987f277fa5b56b88cb382b7a4cd4c309607eeb55032bd3ae1879cf5e72
REFERENCE_PROBE_SHA256=396bdbca86aa73ebdb01379feffee9df6d8e3305ad30c59c94c720f050da3056
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "m7 plate bench v2 (clocks2): mode=$mode gait=$STAGE_GAIT host=$(hostname) job=${SLURM_JOB_ID:-local} head=$(git rev-parse --short HEAD) dry_run=$DRY_RUN"

# Release the exact replay (--stage-gait clocks2) only on a bench-v2 PASS read from verdict.json.
release_if_pass() {
  local out=$1 verdict abs_out release_tmp rc=0
  verdict=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/verdict.json")
  case "$out" in /*) abs_out=$out ;; *) abs_out=$REPO/$out ;; esac
  case "$verdict" in
    PASS)
      echo "M7 BENCH V2 CLOCKS2 PASS -> releasing the exact ten-fall replay with --stage-gait clocks2 (arm m7-clocks2)"
      if [ "$DRY_RUN" != 0 ]; then
        echo "DRY_RUN: would run: M7_CLOCKS2_BENCH_VERDICT=$abs_out/verdict.json bash $RELEASE --only m7-clocks2 --submit"
        return 0
      fi
      release_tmp="$abs_out/.release-tmp"
      mkdir -p "$release_tmp"
      M7_CLOCKS2_BENCH_VERDICT="$abs_out/verdict.json" TMPDIR="$release_tmp" bash "$RELEASE" --only m7-clocks2 --submit || rc=$?
      if [ "$rc" = 3 ]; then
        echo "RELEASE_BLOCKED: the release script refused m7-clocks2 (not a scored bench-v2 clocks2 PASS, the pinned clock-s2"
        echo "  weights differ, or a source the replay would snapshot differs from $out/provenance.json); nothing was queued;"
        echo "  the verdict stands in $out/verdict.json."
        return 1
      elif [ "$rc" != 0 ]; then
        echo "M7 BENCH V2 CLOCKS2 RELEASE FAILED (exit $rc): the PASS stands in $out/verdict.json; run by hand once the cause is fixed:"
        echo "  cd $REPO && M7_CLOCKS2_BENCH_VERDICT=$abs_out/verdict.json bash $RELEASE --only m7-clocks2 --submit"
        return 1
      fi ;;
    FAIL)
      echo "M7 BENCH V2 CLOCKS2 FAIL -> the chain stops: the replay and the route gate are NOT submitted." ;;
    *)
      echo "M7 BENCH V2 CLOCKS2 $verdict -> nothing released (a bench that is not complete is not a verdict)." ;;
  esac
}

if [ "$mode" = bench ]; then
  OUT=${M7_BENCH_V2_CLOCKS2_OUT:-$CAMPAIGN/clocks2-plate-bench-v2}
  [ ! -e "$OUT" ] || { echo "refusing: $OUT exists (a bench result is never overwritten)" >&2; exit 2; }
  # the physics sources: bench v2's set (the replay snapshot's files minus its sbatch, plus both bench launchers) plus
  # the clock controller module and this launcher; scripts/mission7_plate_bench_v2.py source_files(repo, "clocks2")
  # lists the same files for provenance.json
  sources=(scripts/mission7*.py src/bhl_robust/mission src/bhl_robust/__init__.py src/bhl_robust/sensor_io.py
           src/bhl_robust/eval/__init__.py src/bhl_robust/eval/multi_robot.py src/bhl_robust/eval/mjcf_assets.py
           src/bhl_robust/eval/livery.py src/bhl_robust/eval/team_sensors.py src/bhl_robust/eval/gait_clock.py
           slurm/repo20260923/cpu_m7_plate_bench.sbatch slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch
           slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch)
  for f in scripts/mission7_plate_bench_v2.py scripts/mission7_plate_stage.py src/bhl_robust/eval/gait_clock.py \
           slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch; do
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "refusing: $f is not committed" >&2; exit 2; }
  done
  if ! git diff --quiet HEAD -- "${sources[@]}"; then
    echo "refusing: bench sources differ from HEAD; commit them first:" >&2; git status --short -- "${sources[@]}" >&2; exit 2
  fi
  echo "$CLOCKS2_POLICY_SHA256  $CLOCKS2_EXPORT/policy.onnx" | sha256sum -c -
  "$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$OUT" --mode scored --stage-gait "$STAGE_GAIT" --preflight
  "$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$OUT" --mode scored --stage-gait "$STAGE_GAIT"
  release_if_pass "$OUT"
  echo "M7_PLATE_BENCH_V2_CLOCKS2_COMPLETE"
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
SMOKE=${M7_SMOKE_DIR:-$CAMPAIGN/clocks2-bench-v2-smoke/job-${SLURM_JOB_ID:-local}}
case "$SMOKE" in *-smoke/*|*-smoke) ;; *) echo "refusing: smoke output must sit under a *-smoke directory" >&2; exit 2 ;; esac
[ ! -e "$SMOKE" ] || { echo "refusing: $SMOKE exists" >&2; exit 2; }
mkdir -p "$SMOKE/ref-tree"
# exploration train layouts only: 251 d1 = grid 180 deg round standstill; 253 d1 = grid +90 deg square standstill;
# 250 d0 with explicit labels +90 deg round walking (no grid-labelled walking crossing on 250-255 passes the drop rule)
SMOKE_CROSSINGS=251:1,250:0:90:round:walking,253:1
IDENTITY_CROSSING=253:1   # one standstill +90 deg crossing per identity path
V1_SMOKE_CROSSING=250:0   # the v1 bench's own smoke crossing (M2's)
ROUTE_LAYOUT=252          # exploration train layout of the route-probe runs
start=$(date +%s)
REF_PATH="$REPO/src:$SMOKE/ref-tree/scripts:$REPO/scripts"

# 1. preflights (imports, argument parsing, the export check); no episode
"$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$SMOKE/bench" --mode smoke --stage-gait "$STAGE_GAIT" \
      --crossings "$SMOKE_CROSSINGS" --preflight | tee "$SMOKE/bench-preflight.txt"
"$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/replay-clocks2" --smoke --stage-gait clocks2 --preflight | tee "$SMOKE/replay-clocks2-preflight.txt"
"$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/replay-export" --smoke --stage-gait export --stage-gait-export "$CLOCKS2_EXPORT" \
      --stage-gait-export-sha256 "$CLOCKS2_POLICY_SHA256" --preflight | tee "$SMOKE/replay-export-preflight.txt"
"$PY" scripts/mission7_route_handoff_probe.py --repo "$REPO" --out "$SMOKE/probe-clocks2" --stage doors --indices 0 \
      --handoff early --stage-gait clocks2 --preflight | tee "$SMOKE/probe-clocks2-preflight.txt"

# 2. default-path identity against the committed M3 code, exploration layouts only
git ls-tree --name-only "$REFERENCE_COMMIT" scripts/ | grep -E '^scripts/mission7[^/]*\.py$' > "$SMOKE/ref-tree.files"
git archive "$REFERENCE_COMMIT" $(cat "$SMOKE/ref-tree.files") | tar -x -C "$SMOKE/ref-tree"
sha256sum -c - <<EOF
$REFERENCE_STAGE_SHA256  $SMOKE/ref-tree/scripts/mission7_plate_stage.py
$REFERENCE_BENCH_V2_SHA256  $SMOKE/ref-tree/scripts/mission7_plate_bench_v2.py
$REFERENCE_PROBE_SHA256  $SMOKE/ref-tree/scripts/mission7_route_handoff_probe.py
EOF
#  2a. turnboth and m3 through the bench-v2 CLI, one crossing each, decompressed records compared
for gait in turnboth m3; do
  PYTHONPATH="$REF_PATH" "$PY" "$SMOKE/ref-tree/scripts/mission7_plate_bench_v2.py" --repo "$REPO" \
      --out "$SMOKE/identity-$gait-ref" --mode smoke --stage-gait "$gait" --crossings "$IDENTITY_CROSSING"
  "$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$SMOKE/identity-$gait-new" --mode smoke \
      --stage-gait "$gait" --crossings "$IDENTITY_CROSSING"
  for side in ref new; do
    zcat "$SMOKE/identity-$gait-$side/crossings/L253-d1.json.gz" > "$SMOKE/identity-$gait-$side.L253-d1.json"
  done
  cmp "$SMOKE/identity-$gait-ref.L253-d1.json" "$SMOKE/identity-$gait-new.L253-d1.json"
  rm -f "$SMOKE/identity-$gait-ref.L253-d1.json" "$SMOKE/identity-$gait-new.L253-d1.json"
  echo "SMOKE identity $gait: the L253 d1 bench-v2 record is byte-identical to the committed M3 code's"
done
#  2b. the shipped (default) stage path: bench-v2 run_crossing with stage_gait shipped (the CLI offers only stage gaits)
for side in ref new; do
  path="$PYTHONPATH"; [ "$side" = new ] || path="$REF_PATH"
  PYTHONPATH="$path" "$PY" - "$SMOKE/identity-shipped-$side" "$REPO" "$IDENTITY_CROSSING" <<'PYEOF'
import json, pathlib, sys
import mission7_plate_bench_v2 as v2
out, repo, crossing = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
out.mkdir(parents=True)
[spec] = v2.parse_smoke_specs(crossing)
result = v2.run_crossing(repo, out / "cache", spec, "shipped")
(out / "crossing.json").write_text(json.dumps(result, sort_keys=True) + "\n")
print("IDENTITY_SHIPPED", v2.__file__, result["end_reason"], len(result["samples"]), [h["phase"] for h in result["stage_history"]])
PYEOF
done
cmp "$SMOKE/identity-shipped-ref/crossing.json" "$SMOKE/identity-shipped-new/crossing.json"
gzip "$SMOKE/identity-shipped-ref/crossing.json" "$SMOKE/identity-shipped-new/crossing.json"
echo "SMOKE identity shipped: the default stage path's L253 d1 record is byte-identical to the committed M3 code's"
#  2c. the v1 bench (M2's): its own smoke crossing with the committed and the working scripts
PYTHONPATH="$REF_PATH" "$PY" "$SMOKE/ref-tree/scripts/mission7_plate_bench.py" --repo "$REPO" \
    --out "$SMOKE/identity-v1-ref" --mode smoke --crossings "$V1_SMOKE_CROSSING"
"$PY" scripts/mission7_plate_bench.py --repo "$REPO" --out "$SMOKE/identity-v1-new" --mode smoke --crossings "$V1_SMOKE_CROSSING"
for side in ref new; do
  zcat "$SMOKE/identity-v1-$side/crossings/L250-d0.json.gz" > "$SMOKE/identity-v1-$side.L250-d0.json"
done
cmp "$SMOKE/identity-v1-ref.L250-d0.json" "$SMOKE/identity-v1-new.L250-d0.json"
rm -f "$SMOKE/identity-v1-ref.L250-d0.json" "$SMOKE/identity-v1-new.L250-d0.json"
echo "SMOKE identity v1: the L250 d0 v1-bench record is byte-identical to the committed M3 code's"

# The route-probe driver (steps 2d and 4): RouteHandoffController with the follow-up's route composition, Doors stage,
# exploration train layout $ROUTE_LAYOUT, until the first hand-back + 3 s or the time limit.  Integration only.
route_run() {   # $1 = out dir, $2 = stage gait, $3 = time limit (s), $4 = PYTHONPATH
  PYTHONPATH="$4" "$PY" - "$1" "$REPO" "$2" "$3" "$ROUTE_LAYOUT" <<'PYEOF'
import json, pathlib, sys
from bhl_robust.mission.approach_debug import DebugEnv
import mission7_route_handoff_probe as probe
out, repo, gait, limit_s, layout = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4]), int(sys.argv[5])
out.mkdir(parents=True)
env = DebugEnv(repo, out / "cache", stage="doors", split="train", seed=4000 + layout)
env.reset(layout)
shipped = env.controller
controller = probe.RouteHandoffController(env, handoff_mode="early", stage_activate=True, exit_ramp_s=1.2,
                                          stage_wait_open_s=0.0, rejoin_advance=True, exit_ramp_center=True,
                                          stage_gait=gait)
done, steps = False, []
while not done and env.runner.d.time < limit_s:
    env.runner.contact_trace = []
    action = controller.action()
    _, _, done, _ = env.step(action)
    controller.observe_contacts(env.runner.contact_trace)
    controller.observe_step(action)
    active = env.controller
    steps.append({"time_s": float(env.runner.d.time), "xy": env.runner.d.xpos[env.slot.body_id, :2].tolist(),
                  "tilt": float(env.runner.tilt(0)), "stage_phase": controller.stage.phase,
                  "action": [float(a) for a in action], "controller": type(active).__name__,
                  "shipped_object": active is shipped,
                  "traced_update": getattr(active.update, "__name__", "") == "traced_update",
                  "obs_width": int(active.policy_observations.shape[-1])})
    handbacks = [h for h in controller.stage.history if h["phase"] == "recorded"]
    if handbacks and env.runner.d.time > handbacks[-1]["time_s"] + 3.:
        break
summary = {"layout": ["train", layout], "stage_gait": gait, "time_s": float(env.runner.d.time), "done": bool(done),
           "failure": env.state.failure, "open": list(env.state.open), "crossed": list(env.state.crossed),
           "stage_activations": controller.stage_activation_times, "history": controller.stage.history,
           "gait_events": getattr(controller.stage, "gait_events", None), "steps": steps}
(out / "route.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
print("ROUTE_RUN", probe.__file__, json.dumps({k: summary[k] for k in ("stage_gait", "time_s", "done", "failure")}),
      [h["phase"] for h in summary["history"]])
PYEOF
}
#  2d. the route probe's default path (stage gait shipped), committed vs working scripts
route_run "$SMOKE/identity-route-ref" shipped 60 "$REF_PATH"
route_run "$SMOKE/identity-route-new" shipped 60 "$PYTHONPATH"
cmp "$SMOKE/identity-route-ref/route.json" "$SMOKE/identity-route-new/route.json"
gzip "$SMOKE/identity-route-ref/route.json" "$SMOKE/identity-route-new/route.json"
echo "SMOKE identity route probe: the default path's train-$ROUTE_LAYOUT run is byte-identical to the committed M3 code's"

# 3. bench-v2 crossings end to end with --stage-gait clocks2 (standstill 180, walking +90, standstill +90), exploration only
"$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$SMOKE/bench" --mode smoke --stage-gait "$STAGE_GAIT" \
      --crossings "$SMOKE_CROSSINGS"

# 4. the route probe's controller with --stage-gait clocks2 (the swap through MissionEnv.step), exploration only
route_run "$SMOKE/route-clocks2" clocks2 120 "$PYTHONPATH"

# 5. machinery checks; any failure exits non-zero
"$PY" - "$SMOKE" "$start" "$CLOCKS2_POLICY_SHA256" <<'PYEOF'
import gzip, json, pathlib, sys, time
import numpy as np
from bhl_robust.eval.gait_clock import clock_features
smoke, start, pinned = pathlib.Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
checks, problems = {}, []
DT, PERIOD = 0.04, 0.8
FREE = ["command_velocity", "gait_clock", "num_observations", "policy_checkpoint_path"]


def check(name, ok, detail=None):
    checks[name] = {"ok": bool(ok), "detail": detail}
    if not ok:
        problems.append(name)


def last_json(path):
    lines = [line for line in path.read_text().splitlines() if line.strip()]
    try:
        return json.loads(lines[-1]) if lines else {}
    except ValueError:
        return {}


for name in ("bench-preflight", "replay-clocks2-preflight", "replay-export-preflight", "probe-clocks2-preflight"):
    pre = last_json(smoke / f"{name}.txt")
    c = pre.get("stage_gait_check") if isinstance(pre.get("stage_gait_check"), dict) else {}
    check(f"{name}_export_checked", pre.get("status") == "PREFLIGHT_OK"
          and pre.get("stage_gait") == ("export" if "export" in name else "clocks2")
          and c.get("policy_sha256") == pinned and c.get("num_observations") == 77 and c.get("onnx_input_width") == 77
          and c.get("controller") == "GaitClockRlController" and c.get("deploy_keys_differing") == FREE
          and c.get("gait_clock") == {"period_s": 0.8, "phase_offset": 0.0}
          and (c.get("stage_law") or {}).get("turn_rate_rps") == 0.4, {k: c.get(k) for k in (
              "policy_sha256", "num_observations", "controller", "deploy_keys_differing", "gait_clock")})
check("no_replay_or_probe_episode", not any((smoke / d).exists() for d in ("replay-clocks2", "replay-export",
                                                                            "probe-clocks2")),
      "preflight only: validation layouts are scored")

verdict = json.loads((smoke / "bench/verdict.json").read_text())
check("bench_smoke_verdict_never_pass", verdict["verdict"] in ("SMOKE_RUN", "SMOKE_INCOMPLETE")
      and verdict.get("bench") == "v2" and verdict.get("stage_gait") == "clocks2"
      and "m7-clocks2" in verdict.get("chain", "") and "clock-s2" in verdict["labels"]["gaits"]
      and "10.05-10.25 s" in verdict.get("disclosure_stage_gait", ""), verdict["verdict"])
prov = json.loads((smoke / "bench/provenance.json").read_text())
srcs = prov.get("sources_sha256", {})
check("bench_provenance", prov.get("bench") == "v2" and prov.get("stage_gait") == "clocks2"
      and prov.get("weights_sha256", {}).get("clocks2") == pinned
      and {"src/bhl_robust/eval/gait_clock.py", "scripts/mission7_plate_stage.py", "scripts/mission7_plate_bench_v2.py",
           "slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch",
           "slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch"} <= set(srcs)
      and prov.get("stage_constants", {}).get("policy_sha256") == pinned
      and "m7-clocks2" in prov.get("chain", ""), {"weights": prov.get("weights_sha256"), "n_sources": len(srcs)})
crossings = sorted((smoke / "bench/crossings").glob("*.json.gz"))
check("bench_three_crossings_written", len(crossings) == 3, [p.name for p in crossings])
entries, headings, swapped = set(), set(), 0
PHASES = {"approach", "settle", "turn", "cross", "turn_back", "recorded", "wait_open", "cross_kick"}
for path in crossings:
    with gzip.open(path, "rt") as stream:
        c = json.load(stream)
    entries.add(c["entry"]); headings.add(c["heading_deg"])
    name = path.name.split(".")[0]
    check(f"{name}_complete", c["complete"] and c["end_reason"] in ("complete", "fall", "walk_timeout", "stage_timeout"),
          c["end_reason"])
    check(f"{name}_phases_valid", set(h["phase"] for h in c["stage_history"]) <= PHASES,
          [h["phase"] for h in c["stage_history"]])
    events = c.get("gait_events") or []
    if c["takeover_s"] is None:
        check(f"{name}_no_swap_without_takeover", events == [] and c["policy_is_shipped_at_end"] is True, events)
        continue
    swapped += 1
    first = events[0] if events else {}
    check(f"{name}_swap_in_is_a_fresh_clock_controller",
          first.get("event") == "swap_to_clocks2" and first.get("controller") == "GaitClockRlController"
          and first.get("obs_width") == 77 and first.get("clock_step") == 0 and first.get("policy_observations_zero") is True
          and first.get("prev_actions_after_norm") == 0.0 and abs(first.get("time_s", -1) - c["takeover_s"]) < 1e-9,
          first)
    if c["handback_s"] is not None:
        back = events[1] if len(events) > 1 else {}
        updates = int(round((c["handback_s"] - c["takeover_s"]) / DT))
        expected_clock = clock_features(updates - 1, DT, PERIOD).astype(float).tolist() if updates > 0 else None
        check(f"{name}_phase_started_at_takeover_and_swapped_back",
              len(events) == 2 and back.get("event") == "swap_to_shipped" and back.get("controller") == "RlController"
              and back.get("obs_width") == 75 and back.get("prev_actions_after_norm") == 0.0
              and back.get("stage_controller") == "GaitClockRlController" and back.get("stage_controller_obs_width") == 77
              and back.get("stage_controller_clock_step") == updates
              and np.allclose(back.get("stage_controller_last_clock") or [9, 9], expected_clock, atol=1e-6)
              and c["policy_is_shipped_at_end"] is True,
              {"updates_since_takeover": updates, "expected_last_clock": expected_clock, "swap_back": back})
    else:   # a fall or a time-out inside the stage: the clock controller is still in charge, which is reported
        check(f"{name}_stage_ended_without_handback_reported", c["policy_is_shipped_at_end"] is False and len(events) == 1,
              {"end_reason": c["end_reason"], "events": events})
    check(f"{name}_outcome_reported", True, {k: c[k] for k in (
        "heading_deg", "plate", "entry", "end_reason", "takeover_s", "handback_s", "clear", "real_clear",
        "clear_after_takeover_s", "fell", "fall_phase", "maximum_tilt")}
        | {"phases": [h["phase"] for h in c["stage_history"]], "gait_events": events, "events": c["events"],
           "wall_contacts": c["wall_contacts"], "plate_activation": c["plate_activation"], "samples": len(c["samples"])})
check("bench_swap_exercised", swapped >= 1, swapped)
check("bench_standstill_and_walking", entries == {"standstill", "walking"}, sorted(entries))
check("bench_has_90_or_180", bool(headings & {90, -90, 180}), sorted(headings))

route = json.loads((smoke / "route-clocks2/route.json").read_text())
steps, events = route["steps"], route.get("gait_events") or []
took_over = any(e.get("event") == "swap_to_clocks2" for e in events)
check("route_clocks2_runs_without_error", route["stage_gait"] == "clocks2" and steps, route["failure"])
if took_over:
    during = [s for s in steps if s["controller"] == "GaitClockRlController"]
    check("route_clocks2_swaps_through_missionenv_step", during and all(s["obs_width"] == 77 and not s["shipped_object"]
                                                                        for s in during), len(during))
    handed_back = any(e.get("event") == "swap_to_shipped" for e in events)
    if handed_back:
        after = [s for s in steps if s["time_s"] > max(e["time_s"] for e in events if e["event"] == "swap_to_shipped")]
        check("route_clocks2_restores_the_traced_shipped_controller", after and all(
            s["shipped_object"] and s["traced_update"] and s["obs_width"] == 75 for s in after), len(after))
    check("route_clocks2_outcome_reported", True, {k: route[k] for k in ("time_s", "done", "failure", "open")}
          | {"phases": [h["phase"] for h in route["history"]], "gait_events": events})
else:   # informational: whether the route reached a plate in its 120 s is the route's own outcome
    check("route_clocks2_stage_not_exercised_reported", True, {k: route[k] for k in ("time_s", "done", "failure")})
summary = {"ok": not problems, "problems": problems, "wall_s": time.time() - start, "checks": checks}
(smoke / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
print(json.dumps(summary, indent=2, default=str))
sys.exit(0 if not problems else 1)
PYEOF
gzip "$SMOKE/route-clocks2/route.json"
echo "M7_PLATE_BENCH_V2_CLOCKS2_SMOKE_PASS (machinery only; not scored; $(( $(date +%s) - start )) s)"
