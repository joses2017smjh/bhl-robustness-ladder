#!/bin/bash
#SBATCH --job-name=m7-plate-bench-v2-fix
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=haswell&el8
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# Mission 7 plate bench v2 with the crossing-fix options F2 "cross budget" and F3 "yaw cap 0.60" (workstream
# m7-fix-opt; frozen design: SLURM_JOBS.md, "User approval recorded 2026-10-03 09:55", item (F)), declared 2026-10-03
# before any bench-v2, replay or route episode ran with --cross-budget or --yaw-cap.  CPU MuJoCo, no GPU, no render.
# The bench, its grid, its protocol and its rule are bench v2's (scripts/mission7_plate_bench_v2.py;
# slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch, whose header stays the M3 predeclaration of record); this launcher
# adds F2 + F3 (always both) and chooses the stage gait.
#   LEARNED gaits: the shipped arms-dr1.0-s0 gait outside the stage + the LEARNED stage gait: clock-s2
#   (arms-turngait-clock-s2, Turning R1's QUALIFIED seed) by default, or the export the coordinator passes in
#   M7_FIX_STAGE_GAIT_EXPORT with its pinned policy sha256 in M7_FIX_STAGE_GAIT_EXPORT_SHA256 (the design's decision
#   rule: F1's selected PlateCross v2 seed if F1 ran and selected one, else clock-s2) | SCRIPTED stage:
#   scripts/mission7_plate_stage.py PlateStage with --stage-gait clocks2|export --cross-budget window --yaw-cap 0.6 |
#   ORACLE layout and plate pose (as the existing stage uses them).
#
# PREDECLARED RULE (frozen before any episode; bench v2's rule verbatim, unchanged; the chain as M3's, each step only
# after the previous PASSES, every step with the same stage gait and the same two options):
# - Bench v2: with N crossings run (64 minus the dropped set) and N_h per heading, bench v2 PASSES iff clears >=
#   N - 2, 0 falls, and clears >= N_h - 1 for every heading. Otherwise FAIL; INCOMPLETE if any declared crossing is
#   missing. The clear definition is unchanged: mission7_gates' real clear (base along >= +0.35 m past the plate
#   centre along the door direction), within 10 s of takeover.
# - Then the unchanged exact ten-fall replay once with the same stage gait and --cross-budget window --yaw-cap 0.6,
#   through the existing replay-gate machinery (complete, 10 episodes, 10 upright, 0 falls; release arm m7-fix).
# - Then the route gate as coded by M1 with the same stage gait and options: Doors and Transport each >= 16/16
#   successes on validation layouts 0-15 (32 episodes).
# Here N = 41 (0 deg: 12, +90: 11, -90: 10, 180: 8), so PASS needs >= 39/41 clears, 0 falls, and >= 11, 10, 9, 7 clears
# at 0, +90, -90, 180 deg.  The verdict is scripts/mission7_plate_bench_v2.py bench_v2_verdict(), unchanged, written to
# <out>/verdict.json and read back from that JSON (never from an exit code).  Falls (tilt >= 0.78) count in EVERY
# phase after the spawn: settle, walk, stage, post.  Nothing below is tuned on a bench, replay or route result, and no
# parameter was chosen from a result.  ONE run: the output directory is fixed whatever the stage gait, and refused if
# it exists.
#
# Budget: 41 bench-v2 crossings (this job) + 10 exact-replay episodes + 32 route-gate episodes = 83 more Mission 7
# episodes (the frozen design's line; flagged in the report).  The smoke (mode `smoke`) is NOT part of the line: it
# runs on exploration train layouts 101-116 only and never on a validation layout or a train layout 0-31.
#
# The options (scripts/mission7_plate_stage.py FIX_*; opt-in, --stage-gait clocks2 | export only; every other path
# byte-identical to 3a67bc2, checked by the smoke):
# - F2 "cross budget" (--cross-budget window): the stage's crossing ends at the clear point (base along >= 0.35 m past
#   the plate centre along the door direction = mission7_gates.CLEAR_ALONG_M: the stage law's own check, unchanged) or
#   when 0.2 s of the 10 s clear window counted from takeover remains, whichever comes first, instead of at the fixed
#   cross_max_s = 4.0 s.  The clear definition is unchanged.  Deterministic: the crossing ends at the first stage poll
#   with takeover + 10.0 - now <= 0.2 s (+1e-9), so at the bench's 0.2 s polling its last command runs to at most
#   takeover + 10.0 s.  --cross-clear does not compose with it.
#   Takeover = PlateStage._start.  On the bench it IS the bench's takeover (the window the clear is scored in).  In the
#   exact replay and the route gate (no bench window) the same expression counts from the stage's own takeover, its
#   capture at 0.78 m from the plate (PlateStage.approach_radius): there the approach to the pre-point, the settle and
#   the turn run inside the same 9.8 s, so the crossing can get less than the default 4.0 s when they take longer than
#   5.8 s, and more when they are quick.
# - F3 "yaw cap 0.60" (--yaw-cap 0.6), a declared interface change: Mission 7's yaw command scale becomes 0.60 rad/s
#   (vx and vy scales unchanged) consistently in the bench transform (capped_command: tanh(arctanh(clip(c / [0.4,
#   0.35, 0.6], +-0.999999))) x [0.4, 0.35, 0.6], bitwise v1's _route_transform at the 0.40 scale), the route probe
#   (physical_to_action with the 0.60 scale and MissionEnv(yaw_scale=0.6)) and the exact replay (capped_command on the
#   stage's own commands; the recorded route commands stay as replayed).  The stage's in-place turn (turn and turn
#   back) becomes 0.60 rad/s, the gaits' qualified rate; the crossing heading hold stays clip(1.2 x error, +-0.40).
#   Implementation reading, disclosed: CAP semantics -- every command below the cap is delivered unchanged.  The route
#   controller's own actions are encoded at the 0.40 scale by approach_debug.command_action (outside this change), so
#   the route probe re-encodes their yaw channel only and they keep their physical values (heading hold
#   clip(-1.2 x yaw, +-0.35) stays +-0.35 rad/s, as the replay's recorded commands do).  A turn command of 0.60 is
#   delivered as 0.6 x tanh(arctanh(0.999999)) = 0.5999994 rad/s, as 0.40 is delivered as 0.3999996 today.  The turn
#   bound keeps its definition, a full revolution at the commanded rate: 2 pi / 0.60 = 10.47 s (15.71 s at 0.40), so
#   a turn that never converges ends sooner, and the turn back too.
#
# Disclosed before any scored episode (the frozen design's three, then this implementation's):
# - Bundling means a PASS is not attributable to one change.
# - Faster turns in narrow cells add wall-contact and fall risk (6 of 8 180-deg turns already touched walls at 0.40);
#   the clocks2 bench's only fall (21517668: L22 d0, -90 deg, square, standstill) was during the in-place turn.
# - Crossing until clear produced a replay fall before (9/10): the 2026-09-24 replay gates that crossed until clear
#   (--cross-clear 0.35), v2-align 21405537 and v4-settle080 21405541, were 9/10 each.
# - 180-deg timing at 0.60 rad/s, a projection, not a measurement: (pi - 0.15)/0.60 = 4.99 s of turning + the settle
#   (0.40 or 0.60 s as polled) leaves 4.2-4.4 s of the F2 budget for the crossing (0.65 m at 0.30 m/s = 2.17 s
#   nominal), if the gait tracks 0.60 rad/s in place inside the cell.
# - Coverage (M2's and clocks2's note, same pattern): the bench takes over at the pre-point, so a bench PASS does not
#   cover the stage gait driving the capture-to-pre-point approach; the exact replay exercises that phase first.
# - Replay coverage (clocks2's note, same pattern): the replay CLI with the options runs no episode before the chain
#   reaches it (its layouts, validation, are scored): the replay's F3 transform of the stage's own commands and its F2
#   budget are exercised by tests/test_mission7_fix.py on a stub runner and by the replay preflight in the smoke; the
#   first physics episodes of the replay with the options are the chain's scored exact replay.
# - Route probe diagnostics: with --yaw-cap the probe's diagnostic_trace "command" field (DebugEnv.active_command,
#   approach_debug.py, outside this change) records wz at the 0.40 scale, so a delivered 0.60 reads 0.40 there; the
#   route gate's verdict reads only each episode's success, and the fall time and phase come from the trace's tilt and
#   phase, not its command.  The chain trace and the prev_actions-reset check wrap the SHIPPED controller only (as
#   clocks2's header says); the route gate composition uses neither.
# - Bench v2 was introduced after the M2 bench FAIL (21507962) revealed the run-up confound; M2's FAIL (8/64), M3's FAIL
#   (21514946: 16/41, 5 falls) and clocks2's FAIL (21517668: 18/41, 1 fall; 180 deg 0/8 on timing) stand; v1, bench v2
#   and their launchers are unchanged.
# - Smoke layouts, a forced deviation from the instruction "exploration layouts >= 290": train layouts >= 256 do not
#   exist (bhl_robust.mission.layout.SPLITS["train"] = (0, 256)).  The crossing diagnosis (D) runs on train layouts
#   170-249 (the coordinator's correction of 2026-10-03; its declared 250-289 does not exist) and its smoke on 32, 33 and
#   38, so every smoke of these options is restricted to train layouts 32-169 except 32, 33 and 38 (guard:
#   mission7_plate_bench_v2.parse_fix_smoke_specs) and never runs a crossing on 170-249.  This smoke uses exploration
#   train layouts 101-116 (no Mission 7 crossing record existed for 32-249 before these smokes).  By geometry alone,
#   before any episode: the lowest layouts >= 100 with a grid 180-deg
#   standstill crossing whose pre-point is >= 0.316 m from every wall (103 d0), a grid walking crossing with a turn that
#   passes the drop rule (113 d1) and a grid +90-deg standstill crossing for the identity checks (101 d0); the v1
#   identity crossing is 104 d0.  For machinery coverage, from development smokes of the uncommitted tree on
#   exploration layouts (21531935: this launcher's smoke; 21532094: candidate crossings 115-124 and candidate route
#   layouts 108-117 whose first door does not face +x, so that the stage must turn in place): 116 d0 (grid 0-deg square
#   standstill), the first candidate crossing whose crossing ran beyond 4.0 s before it cleared (5.8 s there), and route
#   layout 112, the only candidate whose route run reached the stage's hand-back (106, the first pick, ended with
#   Mission 7's excessive_collision one step into the turn).  Machinery only: never scored, nothing tuned on them.
#   The bench-v2 CLI's own smoke parser (source-pinned) admits 250-255 only, so these options carry their own smoke
#   parser (parse_fix_smoke_specs: 32-169 except 32, 33, 38).
#
# Grid, entry protocol and drop rule: bench v2's, unchanged (cpu_m7_plate_bench_v2.sbatch's header).  DROPPED (23 of
# the 32 walking crossings): L3 d1, L4 d1, L5 d1, L6 d1, L7 d1, L9 d0, L10 d0, L11 d0, L12 d0, L13 d0, L14 d0, L15 d0,
# L16 d1, L19 d1, L21 d1, L22 d1, L23 d1, L24 d0, L26 d0, L27 d0, L29 d0, L30 d0, L31 d0.  Run: 32 standstill + 9 walking
# = 41.  Every command passes through the bench transform (here capped_command with the 0.60 yaw scale), polled every 5
# gait updates (0.2 s); each sample also records the command the active controller received (policy_command).
#
# Chain wiring: after a scored run this job reads <out>/verdict.json.  PASS -> it runs
#   M7_FIX_BENCH_VERDICT=<out>/verdict.json M7_FIX_STAGE_GAIT_EXPORT=<export or empty> \
#   M7_FIX_STAGE_GAIT_EXPORT_SHA256=<sha256 or empty> bash slurm/repo20260923/submit_m7_replay_gates.sh --only m7-fix --submit
# (the exact ten-fall replay with the same stage gait + --cross-budget window --yaw-cap 0.6, pinned to cn-c22,
# source-frozen with gait_clock.py and env.py in the snapshot, then its follow-up, which releases the route gate with the
# same gait and options only on a 10/10).  The release script refuses (RELEASE_BLOCKED, exit 3, nothing queued) unless
# that verdict is a scored bench-v2 PASS with that stage gait whose verdict.json and provenance.json record both
# options, whose provenance records the pinned stage-gait weights (and, for an export, the same export directory), the
# policy on disk still has them, and every replay-snapshot source still has the sha256 recorded there.  FAIL or
# INCOMPLETE -> nothing is submitted; the chain stops.  DRY_RUN=1 prints the release.  The bench refuses an existing
# output directory and any physics source that differs from HEAD.
#   Route-gate snapshot (a gap inherited from M2/M3/clocks2; disclosed, not closed): the follow-up releases the route
#   gate through scripts/submit_mission7_route_handoff.py, which snapshots the LIVE tree at that moment and records its
#   sha256 in its submission.json; nothing compares it with the bench's provenance before that release.  None of those
#   files is to be edited between this bench and the route-gate submission; the route gate's source_sha256 is compared
#   with the replay gate's after the fact.
#
# Smoke (machinery only, never a verdict, never a scored layout; exits non-zero on any failure):
#   1. preflights with both options: bench v2, the replay CLI and the route probe with the bench's stage gait, and bench
#      v2 and the replay CLI with --stage-gait export (the bench's export, or clock-s2's export standing in for an F1
#      export) (preflight only: validation layouts are scored).
#   2. default-path identity against the committed code 3a67bc2 (stage, bench v2, probe, env.py and v1 sha256 pinned
#      below; the reference tree carries 3a67bc2's src/bhl_robust and scripts/mission7*.py), exploration layouts only,
#      byte-for-byte: bench-v2 run_crossing with the shipped, turnboth, m3 and clocks2 stage gaits (IDENTITY_CROSSING),
#      the v1 bench's smoke crossing, the route probe's default path and its clocks2 path (ROUTE_LAYOUT, until the
#      first hand-back + 3 s or 60 s), and the replay and probe CLIs' default and clocks2 preflight lines.
#   3. three bench-v2 crossings with the bench's stage gait (clock-s2 unless an export is passed) + F2 + F3 through the
#      CLI's smoke mode (FIX_SMOKE_CROSSINGS).
#   4. the route probe's controller with the same stage gait + F2 + F3 through MissionEnv.step (DebugEnv(yaw_scale=0.6))
#      on ROUTE_LAYOUT, until the first hand-back + 3 s or 120 s.
#   5. machinery checks: (a) the controller receives |wz| = 0.60 (to 1e-5) in every in-place turn sample on the bench
#      and in every in-place turn step of the route run, the crossing's |wz| stays <= 0.40, and the route's own steps
#      deliver the route's physical command (to 1e-9); (b) every crossing that started ends at the clear point or at
#      the budget (window left <= 0.2 s), and the evidence of a crossing running beyond 4.0 s while not yet clear is
#      reported; the controller swap as clocks2's smoke checks it; provenance, labels, chain and options recorded;
#      (c) each crossing's fix line in verdict.json (and the record's fix_summary) equals the one recomputed from the
#      crossing record's own samples and history, the |wz| maxima included.
#
#   usage (from anywhere):
#     sbatch --job-name=m7-plate-bench-v2-fix-smoke --time=01:00:00 slurm/repo20260923/cpu_m7_plate_bench_v2_fix.sbatch smoke
#     sbatch --dependency=afterok:<smoke> slurm/repo20260923/cpu_m7_plate_bench_v2_fix.sbatch bench
#   with an F1 export: --export=ALL,M7_FIX_STAGE_GAIT_EXPORT=<exported dir>,M7_FIX_STAGE_GAIT_EXPORT_SHA256=<sha256> on
#   both (the smoke checks the export's preflight; the bench runs it).
#   outputs: bench -> results/mission7-campaign-20260923/fix-plate-bench-v2/; smoke ->
#   results/mission7-campaign-20260923/fix-bench-v2-smoke/job-<id>/.  The smoke refuses to run (exit 2) unless the job
#   name contains "smoke" and the job's own time limit is <= 1:00:00.
set -euo pipefail
mode=${1:?usage: cpu_m7_plate_bench_v2_fix.sbatch bench|smoke}
REPO=${M7_REPO:-/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder}
PY=${PYTHON:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}
CAMPAIGN=results/mission7-campaign-20260923
REPLAY_SOURCE=results/mission7-replay-smoke-20260921
REPLAY_BASELINE=results/mission7-approach-followup-20260922/replay-diagnose-cn-c22
RELEASE=${M7_BENCH_RELEASE:-slurm/repo20260923/submit_m7_replay_gates.sh}
DRY_RUN=${DRY_RUN:-0}
ARM=m7-fix
FIX_ARGS=(--cross-budget window --yaw-cap 0.6)   # F2 + F3, always both
CLOCKS2_EXPORT=external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid/2026-10-02_03-58-04_arms-turngait-clock-s2/exported
CLOCKS2_POLICY_SHA256=c1862f1e4429c1e6ac2e4fae1391f3f40ccc353382c8bbf4b88efee4c6c8d8de
# The stage gait: clock-s2 unless the coordinator passes an export (with its pinned sha256; both or neither).
STAGE_GAIT_EXPORT=${M7_FIX_STAGE_GAIT_EXPORT:-}
STAGE_GAIT_EXPORT_SHA256=${M7_FIX_STAGE_GAIT_EXPORT_SHA256:-}
if [ -n "$STAGE_GAIT_EXPORT" ] || [ -n "$STAGE_GAIT_EXPORT_SHA256" ]; then
  [ -n "$STAGE_GAIT_EXPORT" ] && [ -n "$STAGE_GAIT_EXPORT_SHA256" ] || {
    echo "refusing: M7_FIX_STAGE_GAIT_EXPORT and M7_FIX_STAGE_GAIT_EXPORT_SHA256 go together" >&2; exit 2; }
  [[ "$STAGE_GAIT_EXPORT_SHA256" =~ ^[0-9a-f]{64}$ ]] || {
    echo "refusing: M7_FIX_STAGE_GAIT_EXPORT_SHA256 is not a sha256: $STAGE_GAIT_EXPORT_SHA256" >&2; exit 2; }
  case "$STAGE_GAIT_EXPORT" in *[[:space:]]*)
    echo "refusing: M7_FIX_STAGE_GAIT_EXPORT may not contain whitespace" >&2; exit 2 ;; esac
  STAGE_GAIT=export
  GAIT_ARGS=(--stage-gait export --stage-gait-export "$STAGE_GAIT_EXPORT" --stage-gait-export-sha256 "$STAGE_GAIT_EXPORT_SHA256")
  STAGE_POLICY=$STAGE_GAIT_EXPORT/policy.onnx; STAGE_POLICY_SHA256=$STAGE_GAIT_EXPORT_SHA256
else
  STAGE_GAIT=clocks2
  GAIT_ARGS=(--stage-gait clocks2)
  STAGE_POLICY=$CLOCKS2_EXPORT/policy.onnx; STAGE_POLICY_SHA256=$CLOCKS2_POLICY_SHA256
fi
# The committed code before this workstream (bench v2 + stage + probe + MissionEnv) for the identity checks.
REFERENCE_COMMIT=3a67bc2
REFERENCE_STAGE_SHA256=100ce2147dc4bbb541b412579a72be46ca2b467628c05f40a11454d136b3944b
REFERENCE_BENCH_V2_SHA256=4e4a9dcb5f63ef6d67b2bbef505c2d9f1f74b71cd93b4931fc033e0c5bed2025
REFERENCE_PROBE_SHA256=cf77fab82f17328562fb5d2827b2f4498108bda707ef7d73508cc5afa156bf5e
REFERENCE_ENV_SHA256=99cd21167c01d1aa7fd014e2a40b32f13ad06a04bccc1048433d20749f959f9b
REFERENCE_V1_SHA256=3b98e47114eed2ceb82f5b437917d7135a5b6e4e0ded30dca106405f78683182
cd "$REPO"
export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
echo "m7 plate bench v2 (fix): mode=$mode gait=$STAGE_GAIT options=${FIX_ARGS[*]} host=$(hostname) job=${SLURM_JOB_ID:-local} head=$(git rev-parse --short HEAD) dry_run=$DRY_RUN"

# Release the exact replay (same stage gait + both options) only on a bench-v2 PASS read from verdict.json.
release_if_pass() {
  local out=$1 verdict abs_out release_tmp rc=0
  verdict=$("$PY" -c 'import json,sys; print(json.load(open(sys.argv[1]))["verdict"])' "$out/verdict.json")
  case "$out" in /*) abs_out=$out ;; *) abs_out=$REPO/$out ;; esac
  case "$verdict" in
    PASS)
      echo "M7 BENCH V2 FIX PASS -> releasing the exact ten-fall replay with --stage-gait $STAGE_GAIT ${FIX_ARGS[*]} (arm m7-fix)"
      if [ "$DRY_RUN" != 0 ]; then
        echo "DRY_RUN: would run: M7_FIX_BENCH_VERDICT=$abs_out/verdict.json M7_FIX_STAGE_GAIT_EXPORT=$STAGE_GAIT_EXPORT M7_FIX_STAGE_GAIT_EXPORT_SHA256=$STAGE_GAIT_EXPORT_SHA256 bash $RELEASE --only m7-fix --submit"
        return 0
      fi
      release_tmp="$abs_out/.release-tmp"
      mkdir -p "$release_tmp"
      M7_FIX_BENCH_VERDICT="$abs_out/verdict.json" M7_FIX_STAGE_GAIT_EXPORT="$STAGE_GAIT_EXPORT" \
        M7_FIX_STAGE_GAIT_EXPORT_SHA256="$STAGE_GAIT_EXPORT_SHA256" TMPDIR="$release_tmp" \
        bash "$RELEASE" --only m7-fix --submit || rc=$?
      if [ "$rc" = 3 ]; then
        echo "RELEASE_BLOCKED: the release script refused m7-fix (not a scored bench-v2 PASS with this stage gait and both"
        echo "  options, the pinned stage-gait weights differ, or a source the replay would snapshot differs from"
        echo "  $out/provenance.json); nothing was queued; the verdict stands in $out/verdict.json."
        return 1
      elif [ "$rc" != 0 ]; then
        echo "M7 BENCH V2 FIX RELEASE FAILED (exit $rc): the PASS stands in $out/verdict.json; run by hand once the cause is fixed:"
        echo "  cd $REPO && M7_FIX_BENCH_VERDICT=$abs_out/verdict.json M7_FIX_STAGE_GAIT_EXPORT=$STAGE_GAIT_EXPORT M7_FIX_STAGE_GAIT_EXPORT_SHA256=$STAGE_GAIT_EXPORT_SHA256 bash $RELEASE --only m7-fix --submit"
        return 1
      fi ;;
    FAIL)
      echo "M7 BENCH V2 FIX FAIL -> the chain stops: the replay and the route gate are NOT submitted." ;;
    *)
      echo "M7 BENCH V2 FIX $verdict -> nothing released (a bench that is not complete is not a verdict)." ;;
  esac
}

if [ "$mode" = bench ]; then
  OUT=${M7_BENCH_V2_FIX_OUT:-$CAMPAIGN/fix-plate-bench-v2}
  [ ! -e "$OUT" ] || { echo "refusing: $OUT exists (a bench result is never overwritten; ONE fix bench run)" >&2; exit 2; }
  # the physics sources: bench v2's set for an export stage gait plus this launcher; scripts/mission7_plate_bench_v2.py
  # source_files(repo, gait, fix=True) lists the same files for provenance.json
  sources=(scripts/mission7*.py src/bhl_robust/mission src/bhl_robust/__init__.py src/bhl_robust/sensor_io.py
           src/bhl_robust/eval/__init__.py src/bhl_robust/eval/multi_robot.py src/bhl_robust/eval/mjcf_assets.py
           src/bhl_robust/eval/livery.py src/bhl_robust/eval/team_sensors.py src/bhl_robust/eval/gait_clock.py
           slurm/repo20260923/cpu_m7_plate_bench.sbatch slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch
           slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch slurm/repo20260923/cpu_m7_plate_bench_v2_fix.sbatch)
  for f in scripts/mission7_plate_bench_v2.py scripts/mission7_plate_stage.py src/bhl_robust/mission/env.py \
           src/bhl_robust/eval/gait_clock.py slurm/repo20260923/cpu_m7_plate_bench_v2_fix.sbatch; do
    git ls-files --error-unmatch "$f" >/dev/null 2>&1 || { echo "refusing: $f is not committed" >&2; exit 2; }
  done
  if ! git diff --quiet HEAD -- "${sources[@]}"; then
    echo "refusing: bench sources differ from HEAD; commit them first:" >&2; git status --short -- "${sources[@]}" >&2; exit 2
  fi
  [ -f "$STAGE_POLICY" ] || { echo "refusing: no stage-gait policy at $STAGE_POLICY" >&2; exit 2; }
  echo "$STAGE_POLICY_SHA256  $STAGE_POLICY" | sha256sum -c -
  "$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$OUT" --mode scored "${GAIT_ARGS[@]}" "${FIX_ARGS[@]}" --preflight
  "$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$OUT" --mode scored "${GAIT_ARGS[@]}" "${FIX_ARGS[@]}"
  release_if_pass "$OUT"
  echo "M7_PLATE_BENCH_V2_FIX_COMPLETE"
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
SMOKE=${M7_SMOKE_DIR:-$CAMPAIGN/fix-bench-v2-smoke/job-${SLURM_JOB_ID:-local}}
case "$SMOKE" in *-smoke/*|*-smoke) ;; *) echo "refusing: smoke output must sit under a *-smoke directory" >&2; exit 2 ;; esac
[ ! -e "$SMOKE" ] || { echo "refusing: $SMOKE exists" >&2; exit 2; }
mkdir -p "$SMOKE/ref-tree"
# exploration train layouts 101-116 only (header: the forced deviation and how they were chosen): 103 d0 = grid 180 deg
# square standstill; 113 d1 = grid +90 deg round walking (passes the drop rule); 116 d0 = grid 0 deg square standstill
# (its crossing ran beyond 4.0 s before clearing in development smoke 21532094)
FIX_SMOKE_CROSSINGS=103:0,113:1,116:0
IDENTITY_CROSSING=101:0   # grid +90 deg square standstill, one per identity path
V1_SMOKE_CROSSING=104:0   # a v1-bench smoke crossing (v1's own grid labels: 0 deg round walking)
ROUTE_LAYOUT=112          # exploration train layout of the route-probe runs (Doors stage; its first door faces +y)
start=$(date +%s)
REF_PATH="$SMOKE/ref-tree/src:$SMOKE/ref-tree/scripts"
STAGE_EXPORT_ARGS=(--stage-gait export --stage-gait-export "$CLOCKS2_EXPORT" --stage-gait-export-sha256 "$CLOCKS2_POLICY_SHA256")
if [ "$STAGE_GAIT" = export ]; then
  STAGE_EXPORT_ARGS=("${GAIT_ARGS[@]}")   # the export the bench will run
fi

# 1. preflights (imports, argument parsing, the export check, the options); no episode.  The bench's own stage gait
#    (GAIT_ARGS) everywhere, plus the export path (the bench's export, or clock-s2's export standing in for one).
"$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$SMOKE/bench" --mode smoke "${GAIT_ARGS[@]}" \
      "${FIX_ARGS[@]}" --crossings "$FIX_SMOKE_CROSSINGS" --preflight | tee "$SMOKE/bench-preflight.txt"
"$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$SMOKE/bench-export" --mode smoke \
      "${STAGE_EXPORT_ARGS[@]}" "${FIX_ARGS[@]}" --crossings "$FIX_SMOKE_CROSSINGS" --preflight | tee "$SMOKE/bench-export-preflight.txt"
"$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/replay-fix" --smoke "${GAIT_ARGS[@]}" "${FIX_ARGS[@]}" --preflight | tee "$SMOKE/replay-fix-preflight.txt"
"$PY" scripts/mission7_plate_stage.py --repo "$REPO" --campaign "$REPLAY_SOURCE" --baseline "$REPLAY_BASELINE" \
      --out "$SMOKE/replay-fix-export" --smoke "${STAGE_EXPORT_ARGS[@]}" "${FIX_ARGS[@]}" --preflight \
      | tee "$SMOKE/replay-fix-export-preflight.txt"
"$PY" scripts/mission7_route_handoff_probe.py --repo "$REPO" --out "$SMOKE/probe-fix" --stage doors --indices 0 \
      --handoff early "${GAIT_ARGS[@]}" "${FIX_ARGS[@]}" --preflight | tee "$SMOKE/probe-fix-preflight.txt"

# 2. default-path identity against the committed code before this workstream, exploration layouts only
{ echo src/bhl_robust; git ls-tree --name-only "$REFERENCE_COMMIT" scripts/ | grep -E '^scripts/mission7[^/]*\.py$' || true; } \
    > "$SMOKE/ref-tree.files"
[ "$(wc -l < "$SMOKE/ref-tree.files")" -gt 10 ] || { echo "reference file list is short: $SMOKE/ref-tree.files" >&2; exit 1; }
git archive "$REFERENCE_COMMIT" $(cat "$SMOKE/ref-tree.files") | tar -x -C "$SMOKE/ref-tree"
sha256sum -c - <<EOF
$REFERENCE_STAGE_SHA256  $SMOKE/ref-tree/scripts/mission7_plate_stage.py
$REFERENCE_BENCH_V2_SHA256  $SMOKE/ref-tree/scripts/mission7_plate_bench_v2.py
$REFERENCE_PROBE_SHA256  $SMOKE/ref-tree/scripts/mission7_route_handoff_probe.py
$REFERENCE_ENV_SHA256  $SMOKE/ref-tree/src/bhl_robust/mission/env.py
$REFERENCE_V1_SHA256  $SMOKE/ref-tree/scripts/mission7_plate_bench.py
EOF
#  2a. bench-v2 run_crossing with every pre-existing stage gait (the CLI's smoke parser admits 250-255 only), one crossing
for gait in shipped turnboth m3 clocks2; do
  for side in ref new; do
    path="$PYTHONPATH"; [ "$side" = new ] || path="$REF_PATH"
    PYTHONPATH="$path" "$PY" - "$SMOKE/identity-$gait-$side" "$REPO" "$IDENTITY_CROSSING" "$gait" <<'PYEOF'
import json, pathlib, sys
import mission7_plate_bench as v1
import mission7_plate_bench_v2 as v2
out, repo, crossing, gait = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3], sys.argv[4]
out.mkdir(parents=True)
layout_index, door = (int(x) for x in crossing.split(":"))
heading, plate, entry = v1.cell_of(layout_index, door)
spec = {"layout": layout_index, "door": door, "heading_deg": heading, "plate": plate, "entry": entry}
result = v2.run_crossing(repo, out / "cache", spec, gait)
(out / "crossing.json").write_text(json.dumps(result, sort_keys=True) + "\n")
print("IDENTITY", gait, v2.__file__, result["end_reason"], len(result["samples"]), [h["phase"] for h in result["stage_history"]])
PYEOF
  done
  cmp "$SMOKE/identity-$gait-ref/crossing.json" "$SMOKE/identity-$gait-new/crossing.json"
  gzip "$SMOKE/identity-$gait-ref/crossing.json" "$SMOKE/identity-$gait-new/crossing.json"
  echo "SMOKE identity $gait: the bench-v2 L${IDENTITY_CROSSING%%:*} d${IDENTITY_CROSSING##*:} record is byte-identical to $REFERENCE_COMMIT's"
done
#  2b. the v1 bench (M2's): its smoke crossing with the committed and the working code
PYTHONPATH="$REF_PATH" "$PY" "$SMOKE/ref-tree/scripts/mission7_plate_bench.py" --repo "$REPO" \
    --out "$SMOKE/identity-v1-ref" --mode smoke --crossings "$V1_SMOKE_CROSSING"
"$PY" scripts/mission7_plate_bench.py --repo "$REPO" --out "$SMOKE/identity-v1-new" --mode smoke --crossings "$V1_SMOKE_CROSSING"
v1_record=$(printf 'L%03d-d%d.json.gz' "${V1_SMOKE_CROSSING%%:*}" "${V1_SMOKE_CROSSING##*:}")
for side in ref new; do
  zcat "$SMOKE/identity-v1-$side/crossings/$v1_record" > "$SMOKE/identity-v1-$side.json"
done
cmp "$SMOKE/identity-v1-ref.json" "$SMOKE/identity-v1-new.json"
rm -f "$SMOKE/identity-v1-ref.json" "$SMOKE/identity-v1-new.json"
echo "SMOKE identity v1: the $v1_record v1-bench record is byte-identical to $REFERENCE_COMMIT's"

# The route-probe driver (steps 2c and 4): RouteHandoffController with the follow-up's route composition, Doors stage,
# exploration train layout $ROUTE_LAYOUT, until the first hand-back + 3 s or the time limit.  Integration only.  With
# fix=1 the env decodes with yaw_scale 0.6 and the stage gets both options; each step records the command the active
# controller received and the route's own (physical) command.
route_run() {   # $1 = out dir, $2 = stage gait, $3 = time limit (s), $4 = PYTHONPATH, $5 = fix (0|1)[, $6 $7 = export, sha256]
  PYTHONPATH="$4" "$PY" - "$1" "$REPO" "$2" "$3" "$ROUTE_LAYOUT" "$5" "${6:-}" "${7:-}" <<'PYEOF'
import json, pathlib, sys
import numpy as np
from bhl_robust.mission.approach_debug import DebugEnv
import mission7_route_handoff_probe as probe
out, repo, gait, limit_s, layout, fix = (pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3], float(sys.argv[4]),
                                         int(sys.argv[5]), sys.argv[6] == "1")
if gait == "export":   # the export the bench runs, pinned (as the CLIs select it)
    import mission7_plate_stage
    mission7_plate_stage.select_export_gait(sys.argv[7], sys.argv[8])
out.mkdir(parents=True)
env_kwargs, ctrl_kwargs = ({"yaw_scale": .6}, {"cross_budget": "window", "yaw_cap": .6}) if fix else ({}, {})
env = DebugEnv(repo, out / "cache", stage="doors", split="train", seed=4000 + layout, **env_kwargs)
env.reset(layout)
shipped = env.controller
controller = probe.RouteHandoffController(env, handoff_mode="early", stage_activate=True, exit_ramp_s=1.2,
                                          stage_wait_open_s=0.0, rejoin_advance=True, exit_ramp_center=True,
                                          stage_gait=gait, **ctrl_kwargs)
route_action, seen = controller.route.action, {}


def traced_route_action():   # observes the route's own action; returns it unchanged
    action = route_action()
    seen["base"] = np.asarray(action, dtype=float).copy()
    return action


controller.route.action = traced_route_action
yaw_scale = getattr(env, "yaw_scale", None)
done, steps = False, []
while not done and env.runner.d.time < limit_s:
    env.runner.contact_trace = []
    action = controller.action()
    _, _, done, _ = env.step(action)
    controller.observe_contacts(env.runner.contact_trace)
    controller.observe_step(action)
    active = env.controller
    scales = [.4, .35, .4 if yaw_scale is None else yaw_scale]
    steps.append({"time_s": float(env.runner.d.time), "xy": env.runner.d.xpos[env.slot.body_id, :2].tolist(),
                  "tilt": float(env.runner.tilt(0)), "stage_phase": controller.stage.phase, "env_phase": env.phase,
                  "action": [float(a) for a in action], "controller": type(active).__name__,
                  "shipped_object": active is shipped,
                  "traced_update": getattr(active.update, "__name__", "") == "traced_update",
                  "obs_width": int(active.policy_observations.shape[-1]),
                  "received": active.policy_observations[0, :3].astype(float).tolist(),
                  "delivered": (np.tanh(np.asarray(action[:3], dtype=float)) * scales).tolist(),
                  "route_physical": (np.tanh(seen["base"][:3]) * [.4, .35, .4]).tolist()})
    handbacks = [h for h in controller.stage.history if h["phase"] == "recorded"]
    if handbacks and env.runner.d.time > handbacks[-1]["time_s"] + 3.:
        break
summary = {"layout": ["train", layout], "stage_gait": gait, "fix": ctrl_kwargs, "env_yaw_scale": yaw_scale,
           "time_s": float(env.runner.d.time), "done": bool(done), "failure": env.state.failure,
           "open": list(env.state.open), "crossed": list(env.state.crossed),
           "stage_activations": controller.stage_activation_times, "history": controller.stage.history,
           "gait_events": getattr(controller.stage, "gait_events", None), "steps": steps}
(out / "route.json").write_text(json.dumps(summary, indent=1, default=str) + "\n")
print("ROUTE_RUN", probe.__file__, json.dumps({k: summary[k] for k in ("stage_gait", "fix", "time_s", "done", "failure")}),
      [h["phase"] for h in summary["history"]])
PYEOF
}
#  2c. the route probe's default path (stage gait shipped) and its clocks2 path, committed vs working code
for gait in shipped clocks2; do
  route_run "$SMOKE/identity-route-$gait-ref" "$gait" 60 "$REF_PATH" 0
  route_run "$SMOKE/identity-route-$gait-new" "$gait" 60 "$PYTHONPATH" 0
  cmp "$SMOKE/identity-route-$gait-ref/route.json" "$SMOKE/identity-route-$gait-new/route.json"
  gzip "$SMOKE/identity-route-$gait-ref/route.json" "$SMOKE/identity-route-$gait-new/route.json"
  echo "SMOKE identity route probe $gait: the train-$ROUTE_LAYOUT run is byte-identical to $REFERENCE_COMMIT's"
done
#  2d. the replay and probe CLIs' default and clocks2 preflight lines, committed vs working code
for gait in shipped clocks2; do
  extra=(); [ "$gait" = shipped ] || extra=(--stage-gait "$gait")
  for side in ref new; do
    path="$PYTHONPATH"; scripts=scripts; [ "$side" = new ] || { path="$REF_PATH"; scripts="$SMOKE/ref-tree/scripts"; }
    PYTHONPATH="$path" "$PY" "$scripts/mission7_plate_stage.py" --repo "$REPO" --campaign "$REPLAY_SOURCE" \
        --baseline "$REPLAY_BASELINE" --out "$SMOKE/never-replay" --smoke ${extra[@]+"${extra[@]}"} --preflight \
        > "$SMOKE/preflight-replay-$gait-$side.txt"
    PYTHONPATH="$path" "$PY" "$scripts/mission7_route_handoff_probe.py" --repo "$REPO" --out "$SMOKE/never-probe" \
        --stage doors --indices 0 --handoff early ${extra[@]+"${extra[@]}"} --preflight > "$SMOKE/preflight-probe-$gait-$side.txt"
  done
  cmp "$SMOKE/preflight-replay-$gait-ref.txt" "$SMOKE/preflight-replay-$gait-new.txt"
  cmp "$SMOKE/preflight-probe-$gait-ref.txt" "$SMOKE/preflight-probe-$gait-new.txt"
  echo "SMOKE identity preflights $gait: the replay and probe preflight lines are byte-identical to $REFERENCE_COMMIT's"
done

# 3. bench-v2 crossings end to end with the bench's stage gait + F2 + F3 (180 deg standstill, +90 walking, -90 standstill)
"$PY" scripts/mission7_plate_bench_v2.py --repo "$REPO" --out "$SMOKE/bench" --mode smoke "${GAIT_ARGS[@]}" \
      "${FIX_ARGS[@]}" --crossings "$FIX_SMOKE_CROSSINGS"

# 4. the route probe's controller with the bench's stage gait + F2 + F3 through MissionEnv.step, exploration only
route_run "$SMOKE/route-fix" "$STAGE_GAIT" 120 "$PYTHONPATH" 1 "$STAGE_GAIT_EXPORT" "$STAGE_GAIT_EXPORT_SHA256"

# 5. machinery checks; any failure exits non-zero
"$PY" - "$SMOKE" "$start" "$STAGE_POLICY_SHA256" "$STAGE_GAIT" <<'PYEOF'
import gzip, json, pathlib, sys, time
import numpy as np
from bhl_robust.eval.gait_clock import clock_features
smoke, start, pinned, gait = pathlib.Path(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
checks, problems = {}, []
DT, PERIOD = 0.04, 0.8
CAP, HOLD, TOL = .60, .40, 1e-5
OPTIONS = {"cross_budget": "window", "yaw_cap_rps": .6}


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


for name in ("bench-preflight", "bench-export-preflight", "replay-fix-preflight", "replay-fix-export-preflight",
             "probe-fix-preflight"):
    pre = last_json(smoke / f"{name}.txt")
    fix = pre.get("stage_fix") or {}
    c = pre.get("stage_gait_check") if isinstance(pre.get("stage_gait_check"), dict) else {}
    want_gait = "export" if "export" in name else gait
    check(f"{name}_options_and_export_checked", pre.get("status") == "PREFLIGHT_OK"
          and pre.get("stage_gait") == want_gait and {k: fix.get(k) for k in OPTIONS} == OPTIONS
          and fix.get("turn_rate_rps") == CAP and fix.get("wz_walk_rps") == HOLD
          and fix.get("command_scales") == [.4, .35, .6] and abs(fix.get("turn_max_s", 0) - 2 * np.pi / CAP) < 1e-9
          and c.get("policy_sha256") == pinned, {"stage_fix": fix, "policy_sha256": c.get("policy_sha256")})
# the controller the bench's stage gait swaps in (its export check): GaitClockRlController for a 77-observation clock
# export (clock-s2, an F1 PlateCross v2 seed), upstream's RlController for a 75-observation one
gait_check = last_json(smoke / "bench-preflight.txt").get("stage_gait_check") or {}
CTRL, WIDTH = gait_check.get("controller"), gait_check.get("num_observations")
CLOCK = CTRL == "GaitClockRlController"
check("stage_gait_controller_known", CTRL in ("GaitClockRlController", "RlController") and WIDTH in (75, 77),
      {"controller": CTRL, "num_observations": WIDTH})
check("no_replay_or_probe_episode", not any((smoke / d).exists() for d in (
    "replay-fix", "replay-fix-export", "probe-fix", "never-replay", "never-probe", "bench-export")),
      "preflight only: validation layouts are scored")

verdict = json.loads((smoke / "bench/verdict.json").read_text())
check("bench_smoke_verdict_never_pass_and_records_the_options",
      verdict["verdict"] in ("SMOKE_RUN", "SMOKE_INCOMPLETE") and verdict.get("bench") == "v2"
      and verdict.get("stage_gait") == gait and "m7-fix" in verdict.get("chain", "")
      and {k: (verdict.get("stage_fix") or {}).get(k) for k in OPTIONS} == OPTIONS
      and "Bundling" in verdict.get("disclosure_fix", "")
      and ("clock-s2" if gait == "clocks2" else "--stage-gait-export") in verdict["labels"]["gaits"]
      and "cross_budget=window" in verdict["labels"]["stage"] and "yaw_cap=0.6" in verdict["labels"]["stage"],
      verdict["verdict"])
prov = json.loads((smoke / "bench/provenance.json").read_text())
srcs = prov.get("sources_sha256", {})
check("bench_provenance", prov.get("bench") == "v2" and prov.get("stage_gait") == gait
      and prov.get("weights_sha256", {}).get(gait) == pinned
      and {k: (prov.get("stage_fix") or {}).get(k) for k in OPTIONS} == OPTIONS
      and {"src/bhl_robust/eval/gait_clock.py", "src/bhl_robust/mission/env.py", "scripts/mission7_plate_stage.py",
           "scripts/mission7_plate_bench_v2.py", "slurm/repo20260923/cpu_m7_plate_bench_v2_fix.sbatch"} <= set(srcs)
      and "m7-fix" in prov.get("chain", ""), {"weights": prov.get("weights_sha256"), "n_sources": len(srcs)})
crossings = sorted((smoke / "bench/crossings").glob("*.json.gz"))
check("bench_three_crossings_written", len(crossings) == 3, [p.name for p in crossings])
entries, headings, turned, crossed, beyond_4s = set(), set(), 0, 0, []
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
    check(f"{name}_records_the_options", c.get("stage_kwargs") == {"stage_gait": gait, "cross_budget": "window",
                                                                    "yaw_cap": .6}, c.get("stage_kwargs"))
    samples = c["samples"]
    turn = [s for s in samples if s["phase"] in ("stage_turn", "stage_turn_back")]
    cross = [s for s in samples if s["phase"] == "stage_cross"]
    if turn:
        turned += 1
        check(f"{name}_turn_wz_sent_and_received_is_0.60",
              all(abs(abs(s["command"][2]) - CAP) < TOL and abs(abs(s["policy_command"][2]) - CAP) < TOL
                  and s["command"][0] == 0. and s["command"][1] == 0. for s in turn),
              {"samples": len(turn), "sent": sorted({round(abs(s["command"][2]), 7) for s in turn}),
               "received": sorted({round(abs(s["policy_command"][2]), 7) for s in turn})})
    check(f"{name}_crossing_wz_within_0.40", all(abs(s["command"][2]) <= HOLD + 1e-9 for s in cross),
          max((abs(s["command"][2]) for s in cross), default=None))
    check(f"{name}_every_sample_received_what_was_sent",
          all(np.allclose(s["policy_command"], s["command"], atol=1e-6) for s in samples), len(samples))
    # the verdict's per-crossing fix line (written from v1.read_results' sample-less results) = the one recomputed here
    # from this crossing's own record, the |wz| maxima included
    record_back = next((h for h in c["stage_history"] if h["phase"] == "turn_back"), {})
    record_fix = {"cross_s": record_back.get("cross_s"), "cross_ended_by": record_back.get("cross_ended_by"),
                  "window_left_s": record_back.get("window_left_s"),
                  "turn_wz_sent_max": max((abs(s["command"][2]) for s in turn), default=None),
                  "turn_wz_received_max": max((abs(s["policy_command"][2]) for s in turn), default=None),
                  "cross_wz_sent_max": max((abs(s["command"][2]) for s in cross), default=None)}
    verdict_fix = next((x.get("fix") for x in verdict["crossings"] if [x["layout"], x["door"]] == [c["layout"], c["door"]]),
                       None)
    check(f"{name}_verdict_fix_line_matches_the_record", verdict_fix == record_fix and c.get("fix_summary") == record_fix,
          {"verdict": verdict_fix, "record": record_fix})
    hist = c["stage_history"]
    start_cross = next((h for h in hist if h["phase"] == "cross"), None)
    back = next((h for h in hist if h["phase"] == "turn_back"), None)
    if start_cross is not None and back is not None:
        crossed += 1
        ended = back.get("cross_ended_by")
        ok = (ended == "clear" and back["along_m"] >= .35) or (ended == "budget" and back["along_m"] < .35
                                                                 and back["window_left_s"] <= .2 + 1e-9)
        check(f"{name}_crossing_ends_at_clear_or_budget", ok and abs(back["cross_s"] - (back["time_s"] - start_cross["time_s"])) < 1e-9,
              {k: back.get(k) for k in ("cross_ended_by", "along_m", "window_left_s", "cross_s", "time_s")})
        if back["cross_s"] > 4.0 + 1e-9:
            beyond_4s.append({"crossing": name, "cross_s": back["cross_s"], "ended_by": ended,
                              "window_left_s": back["window_left_s"]})
    events = c.get("gait_events") or []
    if c["takeover_s"] is None:
        check(f"{name}_no_swap_without_takeover", events == [] and c["policy_is_shipped_at_end"] is True, events)
        continue
    first = events[0] if events else {}
    check(f"{name}_swap_in_is_a_fresh_stage_controller",
          first.get("event") == f"swap_to_{gait}" and first.get("controller") == CTRL
          and first.get("obs_width") == WIDTH and first.get("clock_step") == (0 if CLOCK else None)
          and first.get("policy_observations_zero") is True
          and first.get("prev_actions_after_norm") == 0.0 and abs(first.get("time_s", -1) - c["takeover_s"]) < 1e-9,
          first)
    if c["handback_s"] is not None:
        back_event = events[1] if len(events) > 1 else {}
        updates = int(round((c["handback_s"] - c["takeover_s"]) / DT))
        expected_clock = clock_features(updates - 1, DT, PERIOD).astype(float).tolist() if updates > 0 else None
        clock_ok = (back_event.get("stage_controller_clock_step") == updates
                    and np.allclose(back_event.get("stage_controller_last_clock") or [9, 9], expected_clock, atol=1e-6)
                    ) if CLOCK else back_event.get("stage_controller_last_clock") is None
        check(f"{name}_phase_started_at_takeover_and_swapped_back",
              len(events) == 2 and back_event.get("event") == "swap_to_shipped" and back_event.get("controller") == "RlController"
              and back_event.get("obs_width") == 75 and back_event.get("prev_actions_after_norm") == 0.0
              and back_event.get("stage_controller") == CTRL and back_event.get("stage_controller_obs_width") == WIDTH
              and clock_ok and c["policy_is_shipped_at_end"] is True,
              {"updates_since_takeover": updates, "swap_back": back_event})
    else:
        check(f"{name}_stage_ended_without_handback_reported", c["policy_is_shipped_at_end"] is False and len(events) == 1,
              {"end_reason": c["end_reason"], "events": events})
    check(f"{name}_outcome_reported", True, {k: c[k] for k in (
        "heading_deg", "plate", "entry", "end_reason", "takeover_s", "handback_s", "clear", "real_clear",
        "clear_after_takeover_s", "fell", "fall_phase", "maximum_tilt")}
        | {"phases": [h["phase"] for h in hist], "events": c["events"], "wall_contacts": c["wall_contacts"],
           "fix": next((x["fix"] for x in verdict["crossings"] if [x["layout"], x["door"]] == [c["layout"], c["door"]]), None)})
check("bench_turns_exercised", turned >= 1, turned)
check("bench_crossings_exercised", crossed >= 1, crossed)
check("bench_standstill_and_walking", entries == {"standstill", "walking"}, sorted(entries))
check("bench_has_180", 180 in headings, sorted(headings))
check("f2_crossing_beyond_4s_reported", True, beyond_4s or "no smoke crossing ran beyond 4.0 s (not exercised)")

route = json.loads((smoke / "route-fix/route.json").read_text())
steps, events = route["steps"], route.get("gait_events") or []
check("route_fix_runs_through_missionenv_with_yaw_scale_0.6", route["stage_gait"] == gait and steps
      and route["env_yaw_scale"] == .6 and route["fix"] == {"cross_budget": "window", "yaw_cap": .6}, route["failure"])
passthrough = [s for s in steps if s["stage_phase"] == "recorded" and not str(s["env_phase"]).startswith(
    ("stage_", "exit_ramp", "rejoin_recover"))]
check("route_fix_route_steps_deliver_the_routes_physical_command",
      passthrough and all(np.allclose(s["delivered"], s["route_physical"], atol=1e-9)
                          and np.allclose(s["received"], s["route_physical"], atol=1e-6) for s in passthrough),
      {"steps": len(passthrough), "max_abs_route_wz": max((abs(s["route_physical"][2]) for s in passthrough), default=None)})
turn_steps = [s for s in steps if s["stage_phase"] in ("turn", "turn_back") and str(s["env_phase"]) in (
    "stage_turn", "stage_turn_back")]
took_over = any(e.get("event") == f"swap_to_{gait}" for e in events)
if turn_steps:
    check("route_fix_controller_receives_0.60_in_the_turns",
          all(abs(abs(s["received"][2]) - CAP) < TOL and abs(abs(s["delivered"][2]) - CAP) < TOL for s in turn_steps),
          {"steps": len(turn_steps), "received": sorted({round(abs(s["received"][2]), 7) for s in turn_steps})})
else:
    check("route_fix_turn_not_exercised_reported", True, {"took_over": took_over, "failure": route["failure"],
                                                         "time_s": route["time_s"]})
if took_over:
    during = [s for s in steps if not s["shipped_object"]]
    check("route_fix_swaps_through_missionenv_step", during and all(s["controller"] == CTRL and s["obs_width"] == WIDTH
                                                                    for s in during), len(during))
    backs = [e for e in route["history"] if e.get("phase") == "turn_back"]
    check("route_fix_crossing_ends_at_clear_or_budget", all(
        (b.get("cross_ended_by") == "clear" and b["along_m"] >= .35)
        or (b.get("cross_ended_by") == "budget" and b["window_left_s"] <= .2 + 1e-9) for b in backs),
        [{k: b.get(k) for k in ("cross_ended_by", "along_m", "window_left_s", "cross_s")} for b in backs])
check("route_fix_outcome_reported", True, {k: route[k] for k in ("time_s", "done", "failure", "open")}
      | {"phases": [h["phase"] for h in route["history"]], "gait_events": events})
check("launcher_stage_gait", gait in ("clocks2", "export"), gait)
summary = {"ok": not problems, "problems": problems, "wall_s": time.time() - start, "checks": checks}
(smoke / "smoke_summary.json").write_text(json.dumps(summary, indent=2, default=str) + "\n")
print(json.dumps(summary, indent=2, default=str))
sys.exit(0 if not problems else 1)
PYEOF
gzip "$SMOKE/route-fix/route.json"
echo "M7_PLATE_BENCH_V2_FIX_SMOKE_PASS (machinery only; not scored; $(( $(date +%s) - start )) s)"
