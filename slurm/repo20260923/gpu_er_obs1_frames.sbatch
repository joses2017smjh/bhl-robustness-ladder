#!/bin/bash
#SBATCH --job-name=er-obs1-frames
#SBATCH --account=eecs
#SBATCH --partition=gpu,ampere,dgx2,share,eecs
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# 2026-10-03: ER-OBS-1 (design G) FRAMES AND ORACLE LABELS. Renders; makes NO API call (cpu_er_obs1_calls.sbatch does).
#
#   labels  : LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + EXTERNAL VLM observer (Gemini Robotics-ER 2 preview, rendered RGB, offline)
#   harness : the UNCHANGED scripted lift-hold-place harness (scripted_carry.run_place_episode, stock pads, crew 2,
#             LIFT_PLACE_RULE's 20 s timeline, PlaceParams()), entered only through its frame_hook; checked equal to
#             the 2026-09-27 scored lift-place run (rule, parameters, lowering, keyframes, gait checkpoint).
#   observer: one fixed-camera RGB frame per simulated second + a segmentation pass + the C1 frame (cube invisible)
#             + the ORACLE labels of the same snapshot (scripts/bench/er_obs.py frames; bhl_robust.eval.er_obs).
#
# DESIGN (verbatim; SLURM_JOBS.md "User approval recorded 2026-10-03 09:55", item G; er_obs.DESIGN_TEXT):
#   (G) ER-OBS-1 (Gemini Robotics-ER 2 as an observer). Run the unchanged scripted lift-hold-place harness (stock
#   pads, crew 2) on seeds 300–319 (with evidence they were never used). Render one fixed RGB frame per second,
#   20 frames × 20 episodes, with a segmentation pass and ORACLE labels at the same step. Call
#   `gemini-robotics-er-2-preview` (thinking low, paid tier) with a fixed prompt and JSON schema, whose sha256 is
#   recorded before any call. The labels are 4 booleans (lifted clear, cube on floor, robot contact > 1 N, seated
#   flat) and a cube point. ACCURACY PASS iff balanced accuracy ≥ 0.90 on all 4 booleans, the point hit rate is
#   ≥ 0.95 on cube-visible frames, and control C1 holds (same frames with the cube rendered invisible must score
#   ≤ 0.60 balanced accuracy). INCOMPLETE if any class has fewer than 20 frames. IN-LOOP-ELIGIBLE is a separate
#   verdict: p95 latency ≤ 1.0 s over 50 calls. Labels: LEARNED gait + SCRIPTED arms + EXTERNAL VLM observer
#   (offline). It needs the user's paid-tier API key, kept outside the repo (never logged or committed). Build and
#   mock dry run now; scored calls only once the key exists.
#
# CLAUSES AS APPLIED (decided 2026-10-03 before any scored frame or call; er_obs.CLAUSES_AS_APPLIED, in every output):
# * frames: one frame at the end of every simulated second (policy step 25k-1, k = 1..20, sim time k s); frame 20 is
#   the final state LIFT_PLACE_RULE scores. Snapshot: after a policy step MuJoCo's poses, contacts and contact forces
#   all describe the start of its last 0.5 ms substep; frame, segmentation, C1 and labels come from that one
#   snapshot; nothing calls mj_forward on the simulation's data.
# * camera: fixed free camera, lookat (0, 0.10, 0.30) m, distance 2.6 m, azimuth -110 deg, elevation -30 deg (in
#   front of the pair, 20 deg to robot a's side), 960x540, fovy 45 deg; shadows ON; floor reflection OFF (the floor's
#   0.12 reflectance mirrors the cube onto the floor under the plinth in every frame, a phantom "cube on the floor").
# * (1) lifted_clear: lowest cube corner >= 0.02 m above every support it could rest on: the floor always, the
#   plinth top (0.19 m) when the cube's world-xy corner AABB strictly overlaps the 0.18 x 0.18 m top
#   (stand4_mdp.support_clearance in numpy; no tilt clause).
# * (2) on_floor: a cube-floor contact in MuJoCo's contact list (the harness's own cube_floor test).
# * (3) robot_contact: summed normal force of every robot geom of both robots on the cube (CarryRunner.cube_forces,
#   a + b) > 1.0 N (strict).
# * (4) seated_flat: cube-plinth contact AND centre over the plinth top (|dx|, |dy| <= 0.09 m) AND
#   arccos(max_i |R[2,i]|) <= 8 deg (any face down); no speed clause; frames where Stand4's strict body-z tilt
#   (arccos R[2,2] <= 8 deg, same contact and footprint clauses) disagrees are reported separately.
# * C1: a render-only deep copy of the compiled model with the cube's geom rgba alpha 0 (MuJoCo leaves the geom out
#   of the scene: no body, shadow, reflection or segmentation). Every C1 frame must pass two gates: (exact) the cube
#   geom is absent from the C1 scene; (pixels) the C1 frame lies within the renderer's own repeat noise of the main
#   renderer's frame with the cube's scene geom moved 100 m away: max abs difference <= 2 levels and <= 100
#   differing pixels (GPU renders of an identical scene differ by 1 level in <= 20 px, job 21532252; the measured
#   shadow leak of alpha 0 set on the scene geom instead was 400-4,700 px at up to 84 levels, render probe 1, so that
#   variant is not used). The reference's own repeat noise is recorded per frame. Physics model and data are never
#   modified.
#   Disclosure: the first frames smoke (21532217) FAILED by design on an earlier exact pixel-identity C1 check (7 of
#   28 frames differed by 1 level in 1-20 px on an RTX 2080 Ti); diagnostic 21532252 showed the same noise when one
#   context renders an identical scene twice; the check was changed to the two gates above on 2026-10-03, on
#   throw-away seeds, before any call.
# * C2 (status quo, reported, never gating): lifted = centre >= 0.05 m above the harness's rest height
#   (LIFT_PLACE_RULE lift_hold_m); placed = |centre - rest| <= 0.03 m AND |dx|, |dy| <= 0.09 m (placed_on_plinth;
#   no orientation clause). The rest height is the harness's own, mirrored and checked to 0.0 on every step.
# * cube-visible: >= 50 cube pixels in the main segmentation. Rendering: EGL, node-dependent: NVIDIA hardware (A40,
#   RTX 2080 Ti seen) or a Mesa llvmpipe fallback (seen on dgxh-1 under MIG, with shadow-acne streaks), so dgxh is
#   left out of the partition list; the GL vendor/renderer strings are recorded in the manifest.
# * The run is valid only if every episode's rendered rollout equals a read-only rollout of the same seed (qpos
#   bitwise at all 20 capture steps and the same episode dict), every C1 frame passes both C1 gates, no PNG
#   carries metadata chunks, and the manifest holds 20 x 20 = 400 frames (FRAMES_OK). The manifest
#   also prints the class counts: if any of the 8 classes has fewer than 20 frames the accuracy verdict will read
#   INCOMPLETE whatever the calls return.
#
# SEED EVIDENCE (grep, 2026-10-03 10:08; er_obs.SEED_EVIDENCE, in every output):
# * 409 coop/scripted-carry/stand/team JSON files under results/ and solutions-20260930/ hold no seed in 130-999.
# * "seed(s)" + 300-319 or 900-901 (case-insensitive) over the ledger, docs/, slurm/, scripts/, src/, tests/,
#   results/, logs/, solutions-20260930/: only the ER-OBS-1 design entries (SLURM_JOBS.md:3720, its copy in the
#   m7-crossdiag worktree, er_coop_assessment_20261003.txt:76).
# * Reserved elsewhere: 0-39 and 100-124 (scripted_carry.py), 125-129 (workstream W). The smoke runs only 900-901;
#   the script refuses 0-39 and 100-129 always and 300-319 outside the scored run.
#
# Usage (job name must contain "smoke" for the smoke; the run refuses without a passing smoke of the same bytes):
#   smoke: sbatch --job-name=er-obs1-frames-smoke slurm/repo20260923/gpu_er_obs1_frames.sbatch smoke
#   run  : sbatch --dependency=afterok:<SMOKE> --export=ALL,ER_OBS_FRAMES_SMOKE_JOB=<SMOKE> \
#                 slurm/repo20260923/gpu_er_obs1_frames.sbatch run
#   Always submit a fresh smoke and chain the run on it with afterok; never reuse an old smoke id.
#   Overrides (env): ER_OBS_CODE (code root; default the repo), ER_OBS_DATA (outputs; default
#   /nfs/hpc/share/sanchej7/Humanoid_Lite/er-obs1-data, outside the repo: PNG frames are not gitignored),
#   ER_OBS_PY (interpreter).
#
# Outputs (nothing is ever overwritten):
#   smoke: $ER_OBS_DATA/frames-smoke/<jobid>/ : unit tests (tests/test_er_obs.py), frames on seeds 900,901 with 14 s
#          episodes (28 frames), step_status.json, smoke_verdict.json (SMOKE_PASS needs every step 0 and FRAMES_OK),
#          code_sha256.txt, launcher_sha256.txt. Exits non-zero unless SMOKE_PASS.
#   run  : $ER_OBS_DATA/frames-run/ : manifest.json (labels, C2, hashes, checks) + frames/s<seed>/t<k>_{rgb,c1,
#          cube_mask}.png (400 frames). Exits 0 only if the manifest reads FRAMES_OK.
#
# RESUME AFTER A CRASH (the run is not resumable in place; it refuses to write into an existing frames-run/):
#   The manifest is written once, at the end. A run that died (time limit, node failure, crash) leaves frames-run/
#   with PNGs and NO manifest.json. Every episode is bitwise deterministic (each rendered rollout is checked equal to
#   a read-only rollout of the same seed), so a re-run reproduces the same frames. Procedure: (1) confirm no frames
#   run job is running (squeue); (2) move the partial directory aside, never delete it:
#   mv $ER_OBS_DATA/frames-run $ER_OBS_DATA/frames-run.partial-<dead job id>; (3) record both in the ledger;
#   (4) resubmit the same run line with the same ER_OBS_FRAMES_SMOKE_JOB (the bytes check still binds the run to that
#   smoke's code). A frames-run/ WITH a manifest.json is a finished run (FRAMES_OK or FRAMES_FAIL), not a crash: it
#   is never moved or re-run without a new ledger entry.
#
# POST-REVIEW FIXES (disclosure): the independent review of 2026-10-03 cleared the frames part. The fixes made after
# it, before any real call, are in the calls path (client.py, verdict.py, scripts/bench/er_obs.py calls/latency/
# verdict, the calls launcher; er_obs.POST_REVIEW_FIXES) plus this resume procedure (fix 7). The frames code path
# (render.py, labels.py, cmd_frames) is unchanged; this launcher's bytes check covers all ER-OBS-1 code files, so a
# fresh smoke is required.
set -uo pipefail
MODE=${1:-}
CODE=${ER_OBS_CODE:-/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder}
DATA=${ER_OBS_DATA:-/nfs/hpc/share/sanchej7/Humanoid_Lite/er-obs1-data}
PY=${ER_OBS_PY:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}
U=$CODE/external/Berkeley-Humanoid-Lite
DEPLOY=$U/logs/rsl_rl/humanoid/2026-08-18_20-57-50_arms-dr1.0-s0/exported/deploy.yaml
CODEFILES=(src/bhl_robust/eval/er_obs/__init__.py src/bhl_robust/eval/er_obs/labels.py
           src/bhl_robust/eval/er_obs/render.py src/bhl_robust/eval/er_obs/client.py
           src/bhl_robust/eval/er_obs/verdict.py scripts/bench/er_obs.py tests/test_er_obs.py
           src/bhl_robust/eval/scripted_carry.py src/bhl_robust/eval/coop_replay.py src/bhl_robust/eval/livery.py
           src/bhl_robust/eval/mjcf_assets.py src/bhl_robust/eval/multi_robot.py scripts/bench/team_airlock.py
           src/bhl_robust/eval/team_airlock.py)
LABEL="LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + EXTERNAL VLM observer (Gemini Robotics-ER 2 preview, rendered RGB, offline)"
cd "$CODE" || exit 1
export PYTHONPATH="$CODE/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1 MUJOCO_GL=egl LP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-4}
case "$(hostname)" in dgx2-*) export MUJOCO_EGL_DEVICE_ID=${MUJOCO_EGL_DEVICE_ID:-0} ;; esac
JOB=${SLURM_JOB_ID:-nojob}
LAUNCHER_SHA=$(sha256sum "$0" | awk '{print $1}')

case "$MODE" in
    smoke|run) ;;
    *) echo "ER-OBS-1 FRAMES RESULT: REFUSED (usage: gpu_er_obs1_frames.sbatch smoke|run)"; exit 1 ;;
esac

# verdicts are read from the JSON (MISSING if it cannot be read), never from an exit status
json_field() {
    "$PY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    d = d[k]
print(d)' "$@" 2>/dev/null || echo MISSING
}
erobs() {
    "$PY" "$CODE/scripts/bench/er_obs.py" "$@" 2>&1 | grep --line-buffered -v -E '^\[|libEGL|pci id for fd|kmsro'
    return "${PIPESTATUS[0]}"
}

egl_preflight() {
    "$PY" -c "import mujoco; r = mujoco.Renderer(mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom size=\"1\"/></worldbody></mujoco>'), 64, 64); r.render(); r.close(); print('EGL render preflight OK', mujoco.__version__)" 2>&1 | grep -v -E 'libEGL|pci id for fd|kmsro'
    return "${PIPESTATUS[0]}"
}

echo "=== er-obs1 frames $MODE | job $JOB | node $(hostname) | code $CODE | HEAD $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null) | $(date) ==="
echo "--- code sha256 (launcher = the script this job runs):"
sha256sum "${CODEFILES[@]}"
echo "$LAUNCHER_SHA  launcher"
echo "--- git status of the code files (empty = committed):"
git -C "$CODE" status --porcelain -- "${CODEFILES[@]}" slurm/repo20260923/gpu_er_obs1_frames.sbatch 2>/dev/null
echo "$LABEL"

if [ "$MODE" = smoke ]; then
    case "${SLURM_JOB_NAME:-}" in
        *smoke*) ;;
        *) echo "ER-OBS-1 FRAMES SMOKE: REFUSED (the smoke's job name must contain 'smoke')"; exit 1 ;;
    esac
    SM=$DATA/frames-smoke/$JOB
    if [ -e "$SM" ]; then echo "ER-OBS-1 FRAMES SMOKE: REFUSED ($SM exists; nothing is overwritten)"; exit 1; fi
    [ -f "$DEPLOY" ] || { echo "ER-OBS-1 FRAMES SMOKE: FAIL (deploy config missing: $DEPLOY)"; exit 1; }
    mkdir -p "$SM" || exit 1
    CACHE=${TMPDIR:-/tmp}/bhl-er-obs1-frames-$JOB
    mkdir -p "$CACHE" || exit 1
    sha256sum "${CODEFILES[@]}" > "$SM/code_sha256.txt"
    echo "$LAUNCHER_SHA" > "$SM/launcher_sha256.txt"
    rc=0
    egl_preflight
    e=$?
    echo "EGL preflight exit $e"
    [ "$e" -eq 0 ] || rc=1
    echo "--- unit tests: tests/test_er_obs.py"
    "$PY" -m pytest -q -p no:cacheprovider tests/test_er_obs.py 2>&1 | tail -40
    t=${PIPESTATUS[0]}
    echo "pytest exit $t"
    [ "$t" -eq 0 ] || rc=1
    echo "--- smoke frames: throw-away seeds 900,901, 14 s episodes (pipeline check, never scored)"
    erobs frames --mode smoke --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" --seeds 900,901 --seconds 14 \
        --out "$SM/frames"
    f=$?
    echo "frames exit $f"
    [ "$f" -eq 0 ] || rc=1
    printf '{"egl_preflight": %d, "pytest": %d, "frames": %d}\n' "$e" "$t" "$f" > "$SM/step_status.json" || rc=1
    erobs smoke-verdict frames --step-status "$SM/step_status.json" --frames "$SM/frames" \
        --out "$SM/smoke_verdict.json" || rc=1
    rm -rf "$CACHE"
    V=$(json_field "$SM/smoke_verdict.json" verdict)
    if [ "$V" = SMOKE_PASS ] && [ "$rc" -eq 0 ]; then
        echo "ER-OBS-1 FRAMES SMOKE: PASS (smoke_verdict.json reads $V; every step exited 0) | $SM | $LABEL"
        exit 0
    fi
    echo "ER-OBS-1 FRAMES SMOKE: FAIL (smoke_verdict.json reads $V; step status $rc) | $SM | $LABEL"
    exit 1
fi

# ---------------------------------------------------------------- run: bytes check against a passing smoke
SMOKE_JOB=${ER_OBS_FRAMES_SMOKE_JOB:-}
if [ -z "$SMOKE_JOB" ]; then
    echo "ER-OBS-1 FRAMES RESULT: REFUSED (run mode needs --export=ALL,ER_OBS_FRAMES_SMOKE_JOB=<smoke job id>)"
    exit 1
fi
SM=$DATA/frames-smoke/$SMOKE_JOB
SV=$(json_field "$SM/smoke_verdict.json" verdict)
if [ "$SV" != SMOKE_PASS ]; then
    echo "ER-OBS-1 FRAMES RESULT: REFUSED (smoke $SMOKE_JOB: $SM/smoke_verdict.json reads $SV, not SMOKE_PASS)"
    exit 1
fi
PT=$(json_field "$SM/smoke_verdict.json" step_status pytest)
if [ "$PT" != 0 ]; then
    echo "ER-OBS-1 FRAMES RESULT: REFUSED (smoke $SMOKE_JOB records its unit tests as exit $PT, not 0)"
    exit 1
fi
if ! sha256sum -c --quiet "$SM/code_sha256.txt"; then
    echo "ER-OBS-1 FRAMES RESULT: REFUSED (the code differs from the bytes smoke $SMOKE_JOB ran)"
    exit 1
fi
if [ "$(cat "$SM/launcher_sha256.txt" 2>/dev/null)" != "$LAUNCHER_SHA" ]; then
    echo "ER-OBS-1 FRAMES RESULT: REFUSED (this launcher differs from the one smoke $SMOKE_JOB ran)"
    exit 1
fi
echo "BYTES-CHECK: PASS (smoke $SMOKE_JOB reads SMOKE_PASS, unit tests exit 0; code, tests and launcher sha256 match)"
[ -f "$DEPLOY" ] || { echo "ER-OBS-1 FRAMES RESULT: FAIL (deploy config missing: $DEPLOY)"; exit 1; }

RUN=$DATA/frames-run
if [ -e "$RUN" ]; then
    if [ -e "$RUN/manifest.json" ]; then
        echo "ER-OBS-1 FRAMES RESULT: REFUSED ($RUN exists with a manifest: a finished run; results are never overwritten)"
    else
        echo "ER-OBS-1 FRAMES RESULT: REFUSED ($RUN exists without a manifest: a run died; results are never overwritten; follow RESUME AFTER A CRASH in this header)"
    fi
    exit 1
fi
if ! egl_preflight; then echo "ER-OBS-1 FRAMES RESULT: FAIL (EGL render preflight)"; exit 1; fi
mkdir -p "$DATA" || exit 1
CACHE=${TMPDIR:-/tmp}/bhl-er-obs1-frames-$JOB
mkdir -p "$CACHE" || exit 1
echo "=== RUN: seeds 300-319 | full 20 s episodes | 400 frames + segmentation + C1 + ORACLE labels"
erobs frames --mode run --deploy "$DEPLOY" --upstream "$U" --cache-dir "$CACHE" --seeds 300-319 --out "$RUN"
s=$?
echo "frames exit $s"
rm -rf "$CACHE"
CK=$(json_field "$RUN/manifest.json" check)
echo "ER-OBS-1 FRAMES RESULT: $CK (read from $RUN/manifest.json; frames exit $s) | $LABEL"
echo "=== done: $(date) ==="
[ "$CK" = FRAMES_OK ] && [ "$s" -eq 0 ] && exit 0
exit 3
