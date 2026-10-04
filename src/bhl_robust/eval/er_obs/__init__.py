"""ER-OBS-1: Gemini Robotics-ER 2 as an EXTERNAL observer of the scripted cooperative lift (design G, frozen).

Question: can ER 2 reproduce, from rendered pixels alone, the ORACLE state the cooperative harness scores on?

Design (SLURM_JOBS.md, "User approval recorded 2026-10-03 09:55", item G; spec: solutions-20260930/
campaign-m7-180-gemini/er_coop_assessment_20261003.txt, "ER-OBS-1"). The words of the design are in
`DESIGN_TEXT` and `RULE_TEXT`; how each open clause is applied (decided before any frame or call) is in
`CLAUSES_AS_APPLIED`. Nothing here is tuned on results.

Pipeline (scripts/bench/er_obs.py; launchers slurm/repo20260923/gpu_er_obs1_frames.sbatch and
cpu_er_obs1_calls.sbatch):

1. frames (GPU node, EGL): the UNCHANGED scripted lift-hold-place harness (`scripted_carry.run_place_episode`,
   stock pads, crew 2, `LIFT_PLACE_RULE`'s 20 s timeline) through its `frame_hook`; one fixed-camera RGB frame per
   simulated second, its segmentation pass, the C1 frame (cube invisible) and the ORACLE labels of the same state
   (`render.py`, `labels.py`).
2. calls (CPU node): one request per frame and variant to `gemini-robotics-er-2-preview` (the documented ER 2 REST
   shape, POST /v1beta/interactions) with ONE fixed prompt and JSON schema (`client.py`); paced; re-sent only after
   HTTP 429/503 or a connection error with no response; every final outcome cached once; MOCK mode exercises
   everything without network.
3. verdicts (`verdict.py`): pure functions; written once, read back from JSON.

Labels on every output: `LABEL`. The fixes applied after the independent review of 2026-10-03, before any real call,
are listed in `POST_REVIEW_FIXES`.
"""

from __future__ import annotations

LABEL = ("LEARNED gait (frozen arms-dr1.0-s0) + SCRIPTED arms + EXTERNAL VLM observer "
         "(Gemini Robotics-ER 2 preview, rendered RGB, offline)")
TASK = "er_obs1_v1"

#: Verbatim: the ledger's item G (SLURM_JOBS.md, user approval recorded 2026-10-03 09:55).
DESIGN_TEXT = (
    "(G) ER-OBS-1 (Gemini Robotics-ER 2 as an observer). "
    "Run the unchanged scripted lift-hold-place harness (stock pads, crew 2) on seeds 300–319 (with evidence they "
    "were never used). Render one fixed RGB frame per second, 20 frames × 20 episodes, with a segmentation pass and "
    "ORACLE labels at the same step. "
    "Call `gemini-robotics-er-2-preview` (thinking low, paid tier) with a fixed prompt and JSON schema, whose sha256 "
    "is recorded before any call. The labels are 4 booleans (lifted clear, cube on floor, robot contact > 1 N, seated "
    "flat) and a cube point. "
    "ACCURACY PASS iff balanced accuracy ≥ 0.90 on all 4 booleans, the point hit rate is ≥ 0.95 on cube-visible "
    "frames, and control C1 holds (same frames with the cube rendered invisible must score ≤ 0.60 balanced accuracy). "
    "INCOMPLETE if any class has fewer than 20 frames. IN-LOOP-ELIGIBLE is a separate verdict: p95 latency ≤ 1.0 s "
    "over 50 calls. "
    "Labels: LEARNED gait + SCRIPTED arms + EXTERNAL VLM observer (offline). "
    "It needs the user's paid-tier API key, kept outside the repo (never logged or committed). Build and mock dry run "
    "now; scored calls only once the key exists."
)

#: Verbatim: the workstream task's rule and label definitions (2026-10-03).
RULE_TEXT = (
    "ORACLE labels at the same step: (1) lifted clear = lowest cube corner >= 2 cm above every support (Stand4's "
    "support_clearance definition, reimplemented for MuJoCo); (2) cube on floor; (3) robot-cube contact > 1 N; "
    "(4) seated flat on the plinth = the minimum over the cube's six body-axis directions of the angle to world up "
    "<= 8 deg (any face down), with Stand4's strict body-z disagreement frames reported separately. C1: the same frames "
    "with the cube rendered invisible (rgba alpha 0, render-only, physics untouched). "
    "ACCURACY PASS iff balanced accuracy >= 0.90 on all 4 booleans AND the point hit rate (share of cube-visible "
    "frames, >= 50 cube pixels in the segmentation, where the point lands on a cube pixel) >= 0.95 AND C1 holds "
    "(cube-invisible frames score <= 0.60 balanced accuracy); otherwise NEGATIVE; INCOMPLETE if any class has fewer "
    "than 20 frames. IN-LOOP-ELIGIBLE (separate): p95 latency <= 1.0 s over the 50 calls. Also report C2 (the "
    "status-quo rules: centre-height lifted test; place rule without the orientation clause) scored against the same "
    "strict labels. A non-JSON answer, refusal, error or timeout counts as wrong."
)

# ------------------------------------------------------------------ frozen numbers
SCORED_SEEDS = tuple(range(300, 320))
#: Throw-away smoke seeds (grep evidence in SEED_EVIDENCE); never 0-39, 100-129 or 300-319.
SMOKE_SEEDS = (900, 901)
SMOKE_SECONDS = 14.0
RESERVED_SEEDS = frozenset(range(0, 40)) | frozenset(range(100, 130))
EPISODE_S = 20.0
FRAMES_PER_EPISODE = 20
EXPECTED_RUN_FRAMES = len(SCORED_SEEDS) * FRAMES_PER_EPISODE      # 400

LIFT_CLEARANCE_M = 0.02          # (1) lowest corner >= 2 cm above every support
SEAT_TILT_MAX_DEG = 8.0          # (4) any face within 8 deg of level
CONTACT_FORCE_N = 1.0            # (3) robot normal force on the cube > 1 N (strict)
PLINTH_HALF_M = 0.09             # LIFT_PLACE_RULE plinth_half_m (centre over the plinth top)
C2_LIFT_CENTRE_M = 0.05          # LIFT_PLACE_RULE lift_hold_m: the harness's per-step "up" test
C2_PLACE_HEIGHT_TOL_M = 0.03     # LIFT_PLACE_RULE final_height_tol_m
VISIBLE_MIN_PIXELS = 50          # cube-visible: >= 50 cube pixels in the segmentation pass
BA_PASS_MIN = 0.90
POINT_HIT_PASS_MIN = 0.95
C1_BA_MAX = 0.60
MIN_CLASS_FRAMES = 20
LATENCY_CALLS = 50
LATENCY_P95_MAX_S = 1.0
#: C1 pixel check (secondary gate): within the renderer's own repeat noise of the cube-removed render. GPU renders
#: of an identical scene differ by 1 level in <= 20 px (job 21532252); the measured shadow leak of scene-level
#: alpha 0 was 400-4,700 px at up to 84 levels (render probe 1).
C1_NOISE_MAX_LEVELS = 2
C1_NOISE_MAX_PX = 100
BOOLEANS = ("lifted_clear", "on_floor", "robot_contact", "seated_flat")
VARIANTS = ("main", "c1")

#: How each clause is applied (decided 2026-10-03 before any scored frame and before any call; repeated in the
#: launchers' headers and in every output).
CLAUSES_AS_APPLIED = (
    "frames: the unchanged lift-hold-place harness (scripted_carry.run_place_episode with CarryParams() = stock pads, "
    "PlaceParams(), LIFT_PLACE_RULE's 20 s timeline, crew 2, gait arms-dr1.0-s0), entered only through its frame_hook; "
    "one frame at the end of every simulated second (policy step 25k-1, k = 1..20, simulated time k s), so frame 20 is "
    "the final state LIFT_PLACE_RULE scores",
    "snapshot: after a policy step, MuJoCo's derived quantities (body and geom poses, contacts, contact forces) all "
    "describe the state at the start of its last 0.5 ms physics substep; the RGB frame, the segmentation, the C1 frame "
    "and every label are taken from that one snapshot; nothing calls mj_forward on the simulation's data",
    "camera: fixed free camera, lookat (0, 0.10, 0.30) m, distance 2.6 m, azimuth -110 deg, elevation -30 deg (in "
    "front of the pair, 20 deg to robot a's side of their facing axis), 960x540 px, MuJoCo default fovy 45 deg; "
    "shadows ON; floor reflection OFF (the floor's 0.12 reflectance mirrors the cube onto the floor under the plinth in "
    "every frame, a phantom 'cube on the floor'); chosen before any call",
    "(1) lifted_clear: the cube's lowest corner is >= 0.02 m above every support it could rest on: the floor (z = 0) "
    "always, and the plinth top (z = 0.19 m) when the world-xy AABB of the cube's 8 corners strictly overlaps the "
    "plinth's 0.18 x 0.18 m top (stand4_mdp.support_clearance, in numpy; no tilt clause)",
    "(2) on_floor: a cube-floor contact is present in MuJoCo's contact list (the harness's own cube_floor test)",
    "(3) robot_contact: the summed normal force of every robot geom of both robots on the cube "
    "(CarryRunner.cube_forces: robot a + robot b) is > 1.0 N (strict)",
    "(4) seated_flat: a cube-plinth contact is present AND the cube centre lies over the plinth top (|dx|, |dy| <= "
    "0.09 m from the plinth centre, LIFT_PLACE_RULE plinth_half_m) AND min over the cube's six body-axis directions "
    "of the angle to world up, arccos(max_i |R[2,i]|), is <= 8 deg (any face down); speed is not a clause (one frame "
    "cannot show it); frames where Stand4's strict body-z tilt (arccos R[2,2] <= 8 deg with the same contact and "
    "footprint clauses) disagrees are reported separately (the uniform-cube blind spot)",
    "C1: the same frames rendered by a render-only deep copy of the compiled model whose cube geom has rgba alpha 0 "
    "(MuJoCo then leaves the cube out of the scene: no body, shadow, reflection or segmentation); every C1 frame "
    "must pass two gates: (exact) the cube geom is absent from the C1 scene, and (pixels) the C1 frame lies within "
    "the renderer's own repeat noise of the main renderer's frame with the cube's scene geom moved 100 m away: max "
    "abs difference <= 2 levels and <= 100 differing pixels (GPU renders of an identical scene differ by 1 level in "
    "<= 20 px, job 21532252; the measured shadow leak of alpha 0 set on the scene geom was 400-4,700 px at up to "
    "84 levels, so that variant is not used); the per-frame repeat noise of the reference is recorded; the physics "
    "model and data are never modified",
    "C2 (status quo; reported, never gating): lifted = cube centre >= 0.05 m above the harness's rest height "
    "(LIFT_PLACE_RULE lift_hold_m, the harness's per-step up test); placed = |centre - rest| <= 0.03 m AND |dx|, |dy| "
    "<= 0.09 m (summarize_lift_place's placed_on_plinth; no orientation clause); each scored by balanced accuracy "
    "against the strict labels (1) and (4) on the same frames; rest height = the harness's own (mirrored and checked "
    "against the lift it passes to the hook on every step)",
    "cube-visible: >= 50 pixels of the cube geom in the main frame's segmentation pass; point hit: row = "
    "min(floor(y * 540 / 1000), 539), col = min(floor(x * 960 / 1000), 959) for the answer's [y, x]; a hit iff that "
    "pixel is a cube pixel; the hit rate's denominator is every cube-visible main frame",
    "request (pre-call; changed after the review): the documented ER 2 REST shape (saved docs aidev_robotics-"
    "overview.txt:211-235, aidev_robotics-spatial.txt:44-64): POST https://generativelanguage.googleapis.com/v1beta/"
    "interactions, the key in the x-goog-api-key header, body {model: gemini-robotics-er-2-preview, input: {parts: "
    "[{inlineData: {mimeType: image/png, data: <the PNG, base64>}}, {text: <the prompt, then 'JSON schema of the "
    "answer: ' and the schema as JSON>}]}, generation_config: {thinking_config: {thinking_level: low}}}; where the docs "
    "differ (the SDK examples' flat generation_config.thinking_level and typed input list) the documented REST form "
    "is used; no response-schema field is documented for this endpoint, so the schema travels in the prompt text (the "
    "docs request JSON formats in the prompt) and PROMPT_SCHEMA_SHA256 covers that exact text and the schema; nothing "
    "else is set: temperature, top_p, the output-token cap (documented output limit 65,536; the earlier 8192 cap is "
    "dropped), media resolution, safety settings, system instruction and interaction storage stay at the API "
    "defaults; single turn, no tools",
    "answer (pre-call; changed after the review): the response's output_text (the field all four documented SDKs "
    "expose); it is JSON iff, after trimming whitespace, it is one JSON value or exactly one markdown code block (an "
    "opening ``` or ```json line, the value, a closing ``` line) with nothing outside it; anything else is non-JSON",
    "main frames: no answer (no output_text), a non-JSON answer, JSON that fails the schema check (an object with "
    "cube_point = two finite numbers in [0, 1000] and the four fields as JSON booleans), a final HTTP error, a "
    "transport error, or a wall time over 60 s counts as WRONG: each boolean is scored as the negation of its label "
    "and the point as a miss",
    "C1 frames (changed after the review, before any call): the four booleans are parsed on their own, whatever "
    "cube_point holds (the point is never scored on C1, and [-1, -1] is an honest answer when no cube is visible); a "
    "C1 call without four parsed booleans counts as CORRECT on every boolean (on a control that must score LOW, "
    "'wrong' would help it pass; scored correct, failures can only make C1 harder to hold); the C1 failure rate and "
    "the C1 BA over answered C1 frames only are reported, never gating",
    "balanced accuracy = (TPR + TNR) / 2 per boolean over every main frame (C1: every C1 frame, failed calls scored "
    "correct). ACCURACY PASS iff all four main BAs >= 0.90 AND the point hit rate >= 0.95 AND each of the four C1 BAs "
    "<= 0.60; INCOMPLETE takes precedence: any of the 8 label classes (4 booleans x true/false) with fewer than 20 "
    "main frames, or any predeclared main or C1 frame without a recorded call outcome; otherwise NEGATIVE",
    "IN-LOOP-ELIGIBLE iff the nearest-rank p95 (the 48th of the 50 sorted wall times) of 50 synchronous calls is "
    "<= 1.0 s; a failed call's time is +inf; INCOMPLETE if fewer than 50 calls have a recorded outcome; latency calls "
    "are never re-sent (a 429/503 or connection error there is a failed call)",
    "pacing and re-sends (pre-call; changed after the review): a request is sent no earlier than 1.0 s after the "
    "previous request ended (every request: preflight, scored, re-sends, latency); a request is re-sent ONLY after "
    "HTTP 429, HTTP 503 or a connection error with no HTTP response (the model produced no answer; a read timeout is "
    "final), after max(Retry-After, google.rpc.RetryInfo retryDelay) when the response gives one, else 5, 15, 45 s, "
    "never less than 1.0 s; at most 3 re-sends per frame and variant within one job; a 429/503/connection error is "
    "never a scored outcome: when the re-sends are used up, or the service asks for a wait over 120 s, the caller "
    "stops (exit 4) with no final outcome for that frame and a resubmission resumes (this stop-not-score choice is "
    "the implementer's, for the coordinator to confirm); every attempt is recorded (the frame's attempts log and its "
    "final record); every other outcome is final at its first attempt, and every final error or timeout still counts "
    "as a failed call",
    "cache (pre-call; changed after the review): each frame and variant is claimed by an exclusive create (O_EXCL, a "
    ".claim file naming the job, host, pid and time) before its first request, and its final outcome is published "
    "with an atomic no-overwrite hard link, so overlapping resumes never call a frame twice; a claim without an "
    "outcome (a job that died mid-call) is never re-sent automatically: the run stays INCOMPLETE until the "
    "coordinator, with no calls job running, moves that claim aside (recorded in the ledger)",
    "preflight: one call on a SMOKE frame (throw-away seed) must return HTTP 200, an output_text that passes the "
    "schema check and a usage object (any top-level field whose name contains 'usage'; every thought-token count in it "
    "is recorded to show the thinking level applied) before any scored or measured call; otherwise the caller stops "
    "(exit 5) and names the failing layer (request shape, key, response shape, answer format, or service "
    "unavailable); the raw response (redacted, first 64 KB) is kept; the caller stops after 5 consecutive final "
    "transport-level failures",
    "paid calls (run and latency) refuse before any request unless ER_OBS_PAID_TIER_CONFIRMED=1 (the key must be a "
    "PAID-tier key, since free-tier inputs are used to improve Google's products, and API-restricted, since the docs "
    "say unrestricted keys get HTTP 403) and the frames manifest has no label class below 20 frames "
    "(summary.classes_below_20 empty and recomputed empty: otherwise the accuracy verdict is INCOMPLETE whatever the "
    "answers)",
)

#: Applied 2026-10-03 (evening) after the independent review (REVIEW_2026-10-03.md) and the coordinator's fix list,
#: before ANY real call (no key exists; nothing has ever been sent to a Google endpoint). Repeated in every output.
POST_REVIEW_FIXES = (
    "1 (blocking) C1 scoring: a failed C1 call is scored CORRECT (was: the negation of the label, which helped C1 "
    "hold); C1 booleans parsed independently of cube_point; the C1 failure rate is reported",
    "2 pacing and transient errors: 1.0 s minimum gap between requests; re-sends only after HTTP 429/503 or a "
    "connection error with no response, honouring Retry-After/RetryInfo, at most 3 per frame per job, every attempt "
    "recorded; a 429/503/connection error is never scored (the caller stops instead)",
    "3 request format: the documented ER 2 REST shape (POST /v1beta/interactions, input.parts, generation_config."
    "thinking_config.thinking_level low; was models/...:generateContent with responseJsonSchema and thinkingLevel); the "
    "JSON schema moved into the prompt text; the answer is output_text; the preflight records the raw response and "
    "the token usage and fails closed on any format mismatch",
    "4 cache: per-frame O_EXCL claims and atomic no-overwrite publication (overlapping resumes never call a frame "
    "twice)",
    "5 the calls launcher and the run/latency modes refuse before any paid call when a label class has fewer than 20 "
    "frames",
    "6 paid tier: ER_OBS_PAID_TIER_CONFIRMED=1 required in run and latency modes; header guidance on paid-tier and "
    "API-restricted keys",
    "7 frames run: the resume procedure after a crash is documented in the frames launcher header",
)

#: Seed-use evidence (grep, 2026-10-03 10:08), repeated in every output.
SEED_EVIDENCE = (
    "results/ and solutions-20260930/: 409 coop/scripted-carry/stand/team JSON files (path matching coop|carry|lift|"
    "crew|flush|wrist|stand|team|place|hold) parsed for seed/seeds/seeds_run values: none in 130-999, so none of "
    "300-319 or 900-901",
    "text grep (case-insensitive) of 'seed(s)' followed within 20 non-digit characters by 300-319 or 900-901 over the "
    "ledger, docs/, slurm/, scripts/, src/, tests/, results/, logs/ and solutions-20260930/: only the ER-OBS-1 design "
    "entries (SLURM_JOBS.md:3720, its copy in the m7-crossdiag worktree, and er_coop_assessment_20261003.txt:76)",
    "scripted_carry.py reserves 0-9 (carry, lift_hold), 10-19 (lift_place), 20-39 (flush-pad scored), 100-119 "
    "(exploration and tuning), 120-124 (flush-pad probe); workstream W declares 125-129; the ER-OBS-1 smoke uses "
    "900-901 only and the CLI refuses 0-39, 100-129 always and 300-319 outside the scored run",
)
