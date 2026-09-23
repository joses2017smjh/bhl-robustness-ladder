# Mission 7 task checklist

Persistent, evidence-linked status for the Mission 7 campaign. Updated after
each meaningful batch. Status vocabulary: **DONE** (evidence linked),
**ACTIVE**, **BLOCKED** (names the blocker), **SUPERSEDED** (names what
replaced it). Route failures are never dropped from a denominator.

## Current status

- **Last updated:** 2026-09-23, before Campaign A submission.
- **Source checkpoint:** `1a526c7` pushed; chain-trace / `prev_actions_reset`
  patch in the working tree, under adversarial review.
- **Achieved:** Doors/1 stall mechanism identified on the instrumented local
  rerun — final 20 s: joint-target range 2.3 % of the episode's own walking
  amplitude, policy-output std 0.03 vs 1.04 walking, **both feet planted
  100 % of the window**, feedback state (`prev_actions`, norm 79) nearly
  static, 0.30 m/s forward command verifiably delivered to the policy input,
  actuator saturation identical to walking. A **policy fixed point**, not
  actuation or contact. A one-shot `prev_actions := 0` at the stall (45.80 s,
  delivered-zero verified) turned the 180 s timeout into route **success at
  82.7 s** (n = 1; `results/mission7-campaign-20260923/local-runs.json`).
- **World −x (Approach) — geometry, not control:** the goal posts are at
  `goal + (0.52, ±0.35)` for every layout, so only a −x approach passes
  *through* the post gap (0.61 m clear). The robot's lateral collision
  envelope at rest is **0.632 m** (elbow to elbow). All 8 failures are
  `excessive_collision` (>5 contact ticks) at t = 2.0–2.1 s during the
  first stride, 0.13 m past the spawn; the 8 successes are arm-swing phase.
  A lateral nudge cannot fix a robot wider than the gap; the untested
  `PulseApproachController` detours around the posts with geometry
  unchanged and was the sound candidate — but on layout 1 (baseline success,
  4.24 s reproduced) both `pulse` (11.84 s) and `guard` (3.24 s) failed with
  `excessive_collision`. Corridor geometry for the 16 −x layouts: outside-post
  margin 0.315 / 0.365 / 0.415 m at pitch 1.5 / 1.6 / 1.7 against a 0.316 m
  half-robot, so the outside route is impossible or marginal, and the
  between-post gap is 0.61 m everywhere. The 8 successes and 8 failures are
  distributed identically across pitches — arm-swing phase, not width. The
  16-layout three-arm comparison (`21400730/731/732`) measures this rather
  than asserting it.
- **Current limitation:** n = 1 on the recovery; no breadth yet.
- **World −x resolved as a geometric blocker** (six arms, 96 episodes; see task 10).
- **Next objective:** collect Campaign A, classify every stall, then Campaign B
  baseline vs `prev_actions_reset` on the exposed episodes.
- **Jobs:** completed `21400561`; pending none. Episodes used: 2 local + 3 smoke + 96 −x + 32 Campaign A = 133 / 512.

## Gates (unchanged unless evidence says otherwise)

| Gate | Status | Evidence |
|---|---|---|
| Pressure-plate exact replay, 10/10 upright | **PASS** | `21397732` |
| Doors ≥16 and Transport ≥16 (documented route protocol) | **FAIL** 1/16, 0/16 | `21398514` |
| Privileged Approach ≥60/64, zero falls, ≥14/16 per direction | **FAIL** 56/64, world −x 8/16 | `21386145` |
| Sensor-only Both readiness | **CLOSED** until both above pass | — |
| Four-sensor comparison | **CLOSED** | — |

## Campaign

| # | Task | Status | Evidence / blocker |
|---|---|---|---|
| 1 | Commit + push guarded probe, sbatch preflight, ignore shapes, CLAUDE.md | DONE | `1a526c7` = origin |
| 2 | Chain instrumentation: cmd→prev_actions→raw→targets→ctrl→joints→feet→base | DONE (pending review) | `--chain-trace`; local Doors/1 baseline classified `frozen_targets`; 99 s/episode, 31 MB trace |
| 3 | `prev_actions_reset` intervention with delivery verification | DONE (pending review) | local Doors/1: APPLIED, delivered-zero verified, route success 82.7 s |
| 4 | Campaign A: 16+16 early-handoff episodes, chain traced, mechanism per episode | ACTIVE | `21400752–755` running |
| 5 | Campaign B: baseline vs `prev_actions_reset` on ≤24 exposed episodes, fingerprint-matched | BLOCKED | needs Campaign A exposure list |
| 6 | Add command-only / combined arms if B leaves hypotheses undistinguished | BLOCKED | needs B |
| 7 | Freeze one recovery; predeclare eval set, metrics, n, stopping rule | BLOCKED | needs B |
| 8 | Confirmatory eval on fresh validation layouts 16–31 (never used), paired baseline | BLOCKED | needs 7 |
| 9 | Re-run 10/10 exact replay regression with the frozen candidate | BLOCKED | needs 7 |
| 10 | World −x / privileged Approach workstream | BLOCKED (geometric) | 6 arms / 96 episodes: 7–9/16 or 0/16; success set moves with crossing phase; robot 0.632 m > gap 0.61 m. Gate stays closed, unweakened. `approach-negx-summary.json` |
| 11 | Sensor-only and four-sensor studies | BLOCKED | gates closed |
| — | `forward_pulse` rejoin fix | SUPERSEDED | no-op by construction; `21399503` reinterpreted |
| — | "Post-stage route rejoin" framing | SUPERSEDED | stall is free-space, route-state valid (`21400561`) |

## Envelope (authorized 2026-09-23)

≤512 new episodes total · ≤8 concurrent Slurm jobs · ≤10 GB new retained
artifacts · keep ≥100 GB free on `/nfs/hpc/share` · `--constraint=haswell&el8`
by default, `--node cn-c22` only for bitwise replay. Episodes used: 0 / 512.

## Job ledger for this campaign

See `SLURM_JOBS.md` (append-only) for every receipt. Campaign-specific rows
are added here as they complete.

| Job | Purpose | Status |
|---|---|---|
| 21400730 | Approach −x, `recovery` baseline (16 layouts) | DONE 8/16, 0 falls — reproduces 21386145 |
| 21400731 | Approach −x, `guard` (16 layouts) | DONE 0/16 — rejected |
| 21400732 | Approach −x, `pulse` detour (16 layouts) | DONE 0/16 — rejected |
| 21400745 | Approach −x, `recovery030` 0.30 m/s onset | DONE 7/16 |
| 21400746 | Approach −x, `settle14` phase shift | DONE 8/16 |
| 21400747 | Approach −x, `settle16` phase shift | DONE 9/16 — success set moves with phase |
| 21400752–55 | Campaign A: 16 Doors + 16 Transport, chain traced | running |
