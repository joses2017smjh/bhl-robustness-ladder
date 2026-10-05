#!/bin/bash
#SBATCH --job-name=er-obs1-calls
#SBATCH --account=eecs
#SBATCH --partition=share
#SBATCH --constraint=el9
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out
#
# 2026-10-03: ER-OBS-1 (design G) OBSERVER CALLS AND VERDICTS (gemini-robotics-er-2-preview, offline, rendered RGB).
#
#   labels  : LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + EXTERNAL VLM observer (Gemini Robotics-ER 2 preview, rendered RGB, offline)
#   frames  : from gpu_er_obs1_frames.sbatch (manifest FRAMES_OK; the scored run = seeds 300-319, 400 frames).
#   model   : gemini-robotics-er-2-preview (the standard endpoint, not -streaming), in the documented ER 2 REST shape
#             (saved docs aidev_robotics-overview.txt:211-235, aidev_robotics-spatial.txt:44-64): POST
#             https://generativelanguage.googleapis.com/v1beta/interactions, body {model, input: {parts: [inlineData
#             PNG, text]}, ...} as first frozen; AMENDED 2026-10-05 (see REQUEST_SHA256 below). The JSON schema travels in
#             the text (prompt, then "JSON schema of the answer: <schema>"); the answer is the model_output text. Temperature,
#             top_p, the output-token cap, media resolution, safety settings, system instruction and storage are at
#             the API defaults; no tools. The request format is UNTESTED against the live service (no key yet): a
#             preflight call on a smoke frame (throw-away seed) must pass its format check before any scored call.
#   sent    : ONLY the PNG frame and the fixed text (with the fixed generation config): no repo data, no paths, no
#             seeds, no user data. The API key goes only in the x-goog-api-key header.
#   PROMPT+SCHEMA SHA256 (the exact text part sent + the schema; record in the ledger before any real call; run and
#   latency refuse any other):
#     PROMPT_SHA256=873d16aed67f87f36fff88dbe7deb2ad6ec55abd0ce71c1a9e581f059191c20b
#   request template sha256 (endpoint, body without the image, answer field, timeouts, pacing and re-send policy;
#   record it too; run and latency refuse any other):
#     REQUEST_SHA256=ddc1f3d1cc08ba9ad97399145b119c1566a6f1d6e02789668e69ea356c8d20b2
#     (amended 2026-10-05 before any scored call; was 6e12d1a1..., then 915d324d...: live preflights 21564188 and
#     21564286 refused the saved robotics REST example's shape with HTTP 400; the body now follows the live
#     Interactions API reference: input [{type: image, data, mime_type}, {type: text, text}], generation_config
#     {thinking_level: low}; the answer is the model_output step's text; er_obs.REQUEST_AMENDMENT)
#     (python scripts/bench/er_obs.py hashes prints both and the template; key and image never appear)
#
# DESIGN (verbatim; SLURM_JOBS.md "User approval recorded 2026-10-03 09:55", item G; er_obs.DESIGN_TEXT):
#   Call `gemini-robotics-er-2-preview` (thinking low, paid tier) with a fixed prompt and JSON schema, whose sha256 is
#   recorded before any call. The labels are 4 booleans (lifted clear, cube on floor, robot contact > 1 N, seated
#   flat) and a cube point. ACCURACY PASS iff balanced accuracy ≥ 0.90 on all 4 booleans, the point hit rate is
#   ≥ 0.95 on cube-visible frames, and control C1 holds (same frames with the cube rendered invisible must score
#   ≤ 0.60 balanced accuracy). INCOMPLETE if any class has fewer than 20 frames. IN-LOOP-ELIGIBLE is a separate
#   verdict: p95 latency ≤ 1.0 s over 50 calls. It needs the user's paid-tier API key, kept outside the repo (never
#   logged or committed). Build and mock dry run now; scored calls only once the key exists.
#
# CLAUSES AS APPLIED (decided 2026-10-03 before any call; er_obs.CLAUSES_AS_APPLIED, in every output):
# * MAIN frames: no answer (no model_output text), a non-JSON answer, JSON failing the schema check (object; cube_point =
#   two finite numbers in [0, 1000]; the four fields JSON booleans), a final HTTP error, a transport error, or a wall
#   time over 60 s counts as WRONG: each boolean is scored as the negation of its label and the point as a miss.
#   the answer text counts as JSON iff, trimmed, it is one JSON value or exactly one ``` / ```json code block holding one.
# * C1 frames: the four booleans are parsed on their own, whatever cube_point holds (the point is never scored on C1;
#   [-1, -1] is an honest answer when no cube is visible). A C1 call without four parsed booleans counts as CORRECT on
#   every boolean (on a control that must score LOW, "wrong" would help it pass); the C1 failure rate and the C1 BA
#   over answered frames only are reported, never gating.
# * balanced accuracy = (TPR + TNR) / 2 per boolean over every main frame (C1: every C1 frame). ACCURACY PASS iff all
#   four main BAs >= 0.90 AND point hit rate >= 0.95 AND each of the four C1 BAs <= 0.60. INCOMPLETE takes precedence:
#   any of the 8 label classes (4 booleans x true/false) with fewer than 20 main frames, or any predeclared main or
#   C1 frame without a recorded call outcome. Otherwise NEGATIVE.
# * point hit: row = min(floor(y * 540 / 1000), 539), col = min(floor(x * 960 / 1000), 959); a hit iff that pixel
#   is a cube pixel of the main segmentation; denominator = main frames with >= 50 cube pixels.
# * IN-LOOP-ELIGIBLE iff nearest-rank p95 (the 48th of 50 sorted wall times, POST to body received, keep-alive
#   session) <= 1.0 s; failed call = +inf; INCOMPLETE if fewer than 50 outcomes. Latency calls are never re-sent.
# * PACING AND RE-SENDS: a request is sent no earlier than 1.0 s after the previous request ended (every request).
#   A request is re-sent ONLY after HTTP 429, HTTP 503 or a connection error with no HTTP response (the model
#   produced no answer; a read timeout is final), after max(Retry-After, google.rpc.RetryInfo retryDelay) when given,
#   else 5, 15, 45 s (never < 1.0 s), at most 3 re-sends per frame and variant within one job. A 429/503/connection
#   error is never a scored outcome: when the re-sends are used up, or the service asks for a wait over 120 s, the
#   caller stops (exit 4) with no final outcome for that frame, and resubmitting resumes (the stop-not-score choice
#   is the implementer's, for the coordinator to confirm). Every attempt is recorded (cache/<frame>__<variant>.
#   attempts.jsonl and the final record). Every other outcome is final at its first attempt and every final error or
#   timeout counts as a failed call. The caller also stops after 5 consecutive final transport-level failures.
# * CACHE: each frame and variant is claimed by an exclusive create (cache/<frame>__<variant>.claim, O_EXCL, naming
#   the job, host, pid and time) before its first request, and its outcome is published once (atomic no-overwrite
#   hard link), so overlapping resumes never call a frame twice. ORPHANED CLAIM (a .claim without a .json: a job died
#   mid-call): never re-sent automatically; the run stays INCOMPLETE (exit 3) until the coordinator, with NO calls
#   job running, records it in the ledger and moves that .claim aside (mv x.claim x.claim.orphan-<date>); the next
#   resume then sends that frame (its earlier answer, if any, was never received, so nothing is selected).
# * PREFLIGHT: one call on a smoke frame must return HTTP 200, an answer text that passes the schema check and a
#   usage object (any top-level field containing "usage"; its thought-token counts are recorded to show the thinking
#   level applied), else exit 5 BEFORE any scored or measured call, naming the failing layer (request shape, key,
#   response shape, answer format, service unavailable); the raw response (redacted, 64 KB) is kept.
# * C2 (status-quo rules vs the strict labels) and the body-z disagreement frames are reported in the accuracy
#   verdict, never gating.
#
# POST-REVIEW FIXES (disclosure): made 2026-10-03 (evening) after the independent review and the coordinator's fix
# list, BEFORE ANY REAL CALL (no key exists; nothing was ever sent to a Google endpoint): (1, blocking) failed C1
# calls are scored CORRECT, not as the negation of the label (which had helped C1 hold), C1 booleans parsed without
# the point, C1 failure rate reported; (2) 1.0 s pacing and re-sends only after 429/503/connection errors; (3) the
# documented interactions REST shape replaces models/...:generateContent + responseJsonSchema + thinkingLevel (so
# both hashes changed: the old b4ba018d.../fb7e6eb2... are void), with the preflight format check and usage record;
# (4) O_EXCL claims and atomic publication; (5) refusal when a class has < 20 frames; (6) ER_OBS_PAID_TIER_CONFIRMED;
# (7, frames launcher) the frames-run resume procedure. er_obs.POST_REVIEW_FIXES is in every output.
#
# KEY: read only from the file named by $GEMINI_API_KEY_FILE (default ~/.config/bhl/gemini_api_key); run and
# latency refuse unless it is a regular file (not a symlink) owned by this user with mode 600 and lies outside the
# code root, the repo, the frames and the outputs. It is never printed, logged, stored, or put in argv or the
# environment. Mock never opens it and cannot reach the network (sockets disabled in-process).
# The key MUST be (a) a PAID-tier key: on the free tier, inputs are "used to improve our products" (pricing page),
# and the design says paid tier; and (b) API-restricted in AI Studio: the docs say the Gemini API answers requests
# from unrestricted keys with 403 Forbidden. Run and latency refuse unless ER_OBS_PAID_TIER_CONFIRMED=1 is exported:
# the operator's confirmation of both, made before any paid call.
# TIER AMENDMENT (2026-10-05, before any real call; er_obs.TIER_AMENDMENT): the operator may instead declare
# ER_OBS_TIER=free (the user's choice: the inputs are only rendered frames of a simulated scene and the fixed prompt).
# Run then proceeds unchanged (a 429 is never scored, so a free-tier quota only delays it; resubmit to resume), the
# tier is recorded in every output, and latency refuses on the free tier (it would measure the rate limit).
#   run (free tier): sbatch --time=04:00:00 --export=ALL,ER_OBS_TIER=free,ER_OBS_CALLS_SMOKE_JOB=<MOCK>,ER_OBS_PREFLIGHT_FRAMES=<smoke frames dir> \
#              slurm/repo20260923/cpu_er_obs1_calls.sbatch run
# PAID CALLS also refuse when the run manifest's summary.classes_below_20 is not [] (the accuracy verdict would be
# INCOMPLETE whatever the answers). Expected cost about $6 (about 852 calls at $1/M input, $5/M output incl.
# thinking); no output cap is set (documented limit 65,536 tokens, at most about $0.33 per call).
#
# Usage (job name must contain "smoke" for mock; latency and run refuse without a passing mock smoke of the same bytes):
#   mock   : sbatch --job-name=er-obs1-calls-mock-smoke \
#              --export=ALL,ER_OBS_MOCK_FRAMES=<a frames dir, e.g. $ER_OBS_DATA/frames-smoke/<id>/frames> \
#              slurm/repo20260923/cpu_er_obs1_calls.sbatch mock
#   latency: sbatch --export=ALL,ER_OBS_PAID_TIER_CONFIRMED=1,ER_OBS_CALLS_SMOKE_JOB=<MOCK>,ER_OBS_PREFLIGHT_FRAMES=<smoke frames dir> \
#              slurm/repo20260923/cpu_er_obs1_calls.sbatch latency
#   run    : sbatch --time=04:00:00 --export=ALL,ER_OBS_PAID_TIER_CONFIRMED=1,ER_OBS_CALLS_SMOKE_JOB=<MOCK>,ER_OBS_PREFLIGHT_FRAMES=<smoke frames dir> \
#              slurm/repo20260923/cpu_er_obs1_calls.sbatch run          (resubmit the same line to resume)
#   Never run two calls jobs at once on purpose: claims make it safe, but the second job only waits on the first.
#   Overrides (env): ER_OBS_CODE, ER_OBS_DATA (default /nfs/hpc/share/sanchej7/Humanoid_Lite/er-obs1-data),
#   ER_OBS_PY, ER_OBS_FRAMES_RUN (default $ER_OBS_DATA/frames-run), GEMINI_API_KEY_FILE.
#
# Outputs (nothing is ever overwritten):
#   mock   : $ER_OBS_DATA/calls-mock-smoke/<jobid>/ : unit tests, calls/ (cache + calls.json), verdict_accuracy.json,
#            latency/latency.json, verdict_latency.json (all PIPELINE_CHECK: synthetic responses, not a result),
#            step_status.json, smoke_verdict.json (MOCK_SMOKE_PASS), code_sha256.txt, launcher_sha256.txt.
#   run    : $ER_OBS_DATA/calls-run/ : preflight-*.json, cache/<frame>__<variant>.{json,attempts.jsonl[,claim]},
#            calls.json (when complete), verdict_accuracy.json (written once, when complete).
#   latency: $ER_OBS_DATA/latency-run/ : preflight-*.json, latency.json, verdict_latency.json.
set -uo pipefail
MODE=${1:-}
CODE=${ER_OBS_CODE:-/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder}
DATA=${ER_OBS_DATA:-/nfs/hpc/share/sanchej7/Humanoid_Lite/er-obs1-data}
PY=${ER_OBS_PY:-/nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python}
PROMPT_SHA256=873d16aed67f87f36fff88dbe7deb2ad6ec55abd0ce71c1a9e581f059191c20b
REQUEST_SHA256=ddc1f3d1cc08ba9ad97399145b119c1566a6f1d6e02789668e69ea356c8d20b2
CODEFILES=(src/bhl_robust/eval/er_obs/__init__.py src/bhl_robust/eval/er_obs/labels.py
           src/bhl_robust/eval/er_obs/render.py src/bhl_robust/eval/er_obs/client.py
           src/bhl_robust/eval/er_obs/verdict.py scripts/bench/er_obs.py tests/test_er_obs.py)
LABEL="LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + EXTERNAL VLM observer (Gemini Robotics-ER 2 preview, rendered RGB, offline)"
cd "$CODE" || exit 1
export PYTHONPATH="$CODE/src" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
export PYTHONDONTWRITEBYTECODE=1
JOB=${SLURM_JOB_ID:-nojob}
LAUNCHER_SHA=$(sha256sum "$0" | awk '{print $1}')

case "$MODE" in
    mock|latency|run) ;;
    *) echo "ER-OBS-1 CALLS RESULT: REFUSED (usage: cpu_er_obs1_calls.sbatch mock|latency|run)"; exit 1 ;;
esac

json_field() {
    "$PY" -c 'import json, sys
d = json.load(open(sys.argv[1]))
for k in sys.argv[2:]:
    d = d[k]
print(d)' "$@" 2>/dev/null || echo MISSING
}
erobs() {
    "$PY" "$CODE/scripts/bench/er_obs.py" "$@"
}

echo "=== er-obs1 calls $MODE | job $JOB | node $(hostname) | code $CODE | HEAD $(git -C "$CODE" rev-parse --short HEAD 2>/dev/null) | $(date) ==="
echo "--- code sha256 (launcher = the script this job runs):"
sha256sum "${CODEFILES[@]}"
echo "$LAUNCHER_SHA  launcher"
echo "--- git status of the code files (empty = committed):"
git -C "$CODE" status --porcelain -- "${CODEFILES[@]}" slurm/repo20260923/cpu_er_obs1_calls.sbatch 2>/dev/null
echo "$LABEL"

if [ "$MODE" = mock ]; then
    case "${SLURM_JOB_NAME:-}" in
        *smoke*) ;;
        *) echo "ER-OBS-1 CALLS MOCK: REFUSED (the mock smoke's job name must contain 'smoke')"; exit 1 ;;
    esac
    FR=${ER_OBS_MOCK_FRAMES:-}
    if [ -z "$FR" ] || [ ! -f "$FR/manifest.json" ]; then
        echo "ER-OBS-1 CALLS MOCK: REFUSED (needs --export=ALL,ER_OBS_MOCK_FRAMES=<a frames dir with manifest.json>)"
        exit 1
    fi
    SM=$DATA/calls-mock-smoke/$JOB
    if [ -e "$SM" ]; then echo "ER-OBS-1 CALLS MOCK: REFUSED ($SM exists; nothing is overwritten)"; exit 1; fi
    mkdir -p "$SM" || exit 1
    sha256sum "${CODEFILES[@]}" > "$SM/code_sha256.txt"
    echo "$LAUNCHER_SHA" > "$SM/launcher_sha256.txt"
    rc=0
    echo "--- unit tests: tests/test_er_obs.py"
    "$PY" -m pytest -q -p no:cacheprovider tests/test_er_obs.py > "$SM/pytest.log" 2>&1
    t=$?
    tail -40 "$SM/pytest.log"
    echo "pytest exit $t"
    [ "$t" -eq 0 ] || rc=1
    echo "--- mock calls (virtual clock; malformed/prose/missing answers, schema violations, HTTP errors, timeouts, re-sent 429/503/connection errors)"
    erobs calls --mode mock --frames "$FR" --preflight-frames "$FR" --out "$SM/calls"
    c=$?
    echo "calls exit $c"
    [ "$c" -eq 0 ] || rc=1
    erobs verdict accuracy --frames "$FR" --calls "$SM/calls/calls.json" --out "$SM/verdict_accuracy.json"
    va=$?
    [ "$va" -eq 0 ] || rc=1
    echo "--- mock latency (50 calls, synthetic wall times on a virtual clock, 2 forced timeouts)"
    erobs latency --mode mock --frames "$FR" --preflight-frames "$FR" --out "$SM/latency"
    l=$?
    echo "latency exit $l"
    [ "$l" -eq 0 ] || rc=1
    erobs verdict latency --latency "$SM/latency/latency.json" --out "$SM/verdict_latency.json"
    vl=$?
    [ "$vl" -eq 0 ] || rc=1
    printf '{"pytest": %d, "calls": %d, "verdict_accuracy": %d, "latency": %d, "verdict_latency": %d}\n' \
        "$t" "$c" "$va" "$l" "$vl" > "$SM/step_status.json" || rc=1
    erobs smoke-verdict calls --step-status "$SM/step_status.json" --frames "$FR" --calls "$SM/calls/calls.json" \
        --verdict-accuracy "$SM/verdict_accuracy.json" --verdict-latency "$SM/verdict_latency.json" \
        --out "$SM/smoke_verdict.json" || rc=1
    V=$(json_field "$SM/smoke_verdict.json" verdict)
    if [ "$V" = MOCK_SMOKE_PASS ] && [ "$rc" -eq 0 ]; then
        echo "ER-OBS-1 CALLS MOCK: PASS (smoke_verdict.json reads $V; every step exited 0) | $SM | $LABEL"
        exit 0
    fi
    echo "ER-OBS-1 CALLS MOCK: FAIL (smoke_verdict.json reads $V; step status $rc) | $SM | $LABEL"
    exit 1
fi

# ---------------------------------------------------------------- latency / run: bytes check against a passing mock
SMOKE_JOB=${ER_OBS_CALLS_SMOKE_JOB:-}
if [ -z "$SMOKE_JOB" ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED ($MODE needs --export=ALL,ER_OBS_CALLS_SMOKE_JOB=<mock smoke job id>)"
    exit 1
fi
SM=$DATA/calls-mock-smoke/$SMOKE_JOB
SV=$(json_field "$SM/smoke_verdict.json" verdict)
if [ "$SV" != MOCK_SMOKE_PASS ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED (mock smoke $SMOKE_JOB: $SM/smoke_verdict.json reads $SV, not MOCK_SMOKE_PASS)"
    exit 1
fi
PT=$(json_field "$SM/smoke_verdict.json" step_status pytest)
if [ "$PT" != 0 ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED (mock smoke $SMOKE_JOB records its unit tests as exit $PT, not 0)"
    exit 1
fi
if ! sha256sum -c --quiet "$SM/code_sha256.txt"; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED (the code differs from the bytes mock smoke $SMOKE_JOB ran)"
    exit 1
fi
if [ "$(cat "$SM/launcher_sha256.txt" 2>/dev/null)" != "$LAUNCHER_SHA" ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED (this launcher differs from the one mock smoke $SMOKE_JOB ran)"
    exit 1
fi
echo "BYTES-CHECK: PASS (mock smoke $SMOKE_JOB reads MOCK_SMOKE_PASS, unit tests exit 0; code, tests and launcher sha256 match)"
FR=${ER_OBS_FRAMES_RUN:-$DATA/frames-run}
PRE=${ER_OBS_PREFLIGHT_FRAMES:-}
if [ "$(json_field "$FR/manifest.json" check)" != FRAMES_OK ] || [ "$(json_field "$FR/manifest.json" mode)" != run ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED ($FR/manifest.json is not a FRAMES_OK run manifest)"
    exit 1
fi
SHORT=$(json_field "$FR/manifest.json" summary classes_below_20)
if [ "$SHORT" != "[]" ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED ($FR/manifest.json summary.classes_below_20 reads $SHORT, not []: the accuracy verdict would be INCOMPLETE whatever the answers; no paid call is made)"
    exit 1
fi
if [ -z "$PRE" ] || [ "$(json_field "$PRE/manifest.json" mode)" != smoke ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED (needs --export=ALL,ER_OBS_PREFLIGHT_FRAMES=<a smoke frames dir>)"
    exit 1
fi
HASHES=$("$PY" "$CODE/scripts/bench/er_obs.py" hashes 2>/dev/null)
if [ "$(printf '%s' "$HASHES" | json_field /dev/stdin prompt_schema_sha256)" != "$PROMPT_SHA256" ] \
   || [ "$(printf '%s' "$HASHES" | json_field /dev/stdin request_sha256)" != "$REQUEST_SHA256" ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED (the code's prompt+schema or request sha256 differs from this launcher's $PROMPT_SHA256 / $REQUEST_SHA256)"
    exit 1
fi
TIER=""
case "${ER_OBS_TIER:-}" in
    "") [ "${ER_OBS_PAID_TIER_CONFIRMED:-}" = 1 ] && TIER=paid ;;
    paid) TIER=paid ;;
    free) [ "${ER_OBS_PAID_TIER_CONFIRMED:-}" = 1 ] || TIER=free ;;
esac
if [ -z "$TIER" ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED ($MODE needs the key's tier declared: --export=ALL,ER_OBS_PAID_TIER_CONFIRMED=1 (or ER_OBS_TIER=paid) for a PAID-tier key, or ER_OBS_TIER=free, never both; the key must be API-restricted; nothing was sent)"
    exit 1
fi
if [ "$MODE" = latency ] && [ "$TIER" = free ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED (latency on the free tier would measure the rate limit, not the model: it needs a paid key; nothing was sent)"
    exit 1
fi
echo "TIER: $TIER (declared by the operator; recorded in every output)"
if [ "$MODE" = run ]; then OUT=$DATA/calls-run; else OUT=$DATA/latency-run; fi
if ! erobs keycheck --forbid "$FR" "$PRE" "$OUT"; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED (API key file; see above; nothing was sent)"
    exit 1
fi

if [ "$MODE" = latency ]; then
    if [ -e "$OUT/latency.json" ]; then
        echo "ER-OBS-1 CALLS RESULT: REFUSED ($OUT/latency.json exists; never overwritten)"; exit 1
    fi
    echo "=== LATENCY: 50 synchronous calls on the first 50 scored main frames | prompt+schema $PROMPT_SHA256 | request $REQUEST_SHA256"
    erobs latency --mode run --frames "$FR" --preflight-frames "$PRE" --out "$OUT" \
        --expect-prompt-sha256 "$PROMPT_SHA256" --expect-request-sha256 "$REQUEST_SHA256"
    s=$?
    echo "latency exit $s"
    if [ "$s" -ne 0 ]; then echo "ER-OBS-1 LATENCY RESULT: NOT MEASURED (exit $s: 5 = the preflight failed its format check; see above)"; exit 3; fi
    erobs verdict latency --latency "$OUT/latency.json" --out "$OUT/verdict_latency.json" || exit 3
    echo "ER-OBS-1 LATENCY RESULT: $(json_field "$OUT/verdict_latency.json" verdict) (read from $OUT/verdict_latency.json) | $LABEL"
    exit 0
fi

if [ -e "$OUT/verdict_accuracy.json" ] || [ -e "$OUT/calls.json" ]; then
    echo "ER-OBS-1 CALLS RESULT: REFUSED ($OUT already holds calls.json or verdict_accuracy.json; never overwritten)"
    exit 1
fi
echo "=== RUN: 400 main + 400 C1 calls (resumable; published outcomes are final) | prompt+schema $PROMPT_SHA256 | request $REQUEST_SHA256"
erobs calls --mode run --frames "$FR" --preflight-frames "$PRE" --out "$OUT" \
    --expect-prompt-sha256 "$PROMPT_SHA256" --expect-request-sha256 "$REQUEST_SHA256"
s=$?
echo "calls exit $s"
if [ "$s" -ne 0 ]; then
    echo "ER-OBS-1 CALLS RESULT: INCOMPLETE OR STOPPED (exit $s: 3 = incomplete (frames missing or claimed without an outcome), 4 = stopped (re-sends used up, a long Retry-After, or 5 transport failures in a row), 5 = the preflight failed its format check); resubmit the same line to resume"
    exit 3
fi
erobs verdict accuracy --frames "$FR" --calls "$OUT/calls.json" --out "$OUT/verdict_accuracy.json" || exit 3
echo "ER-OBS-1 ACCURACY RESULT: $(json_field "$OUT/verdict_accuracy.json" verdict) (read from $OUT/verdict_accuracy.json) | $LABEL"
echo "=== done: $(date) ==="
exit 0
