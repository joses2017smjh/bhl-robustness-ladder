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
- **Campaign A (32):** Doors 3/16, Transport 1/16. Failures: in-stage wall
  collision 15, post-exit fall 12, stall 5 (`frozen_targets`). Three separable
  in-stage fixes defined and smoke-validated on single layouts (F1–F3).
- **Current limitation:** factor arms and Campaign B not yet collected.
- **World −x resolved as a geometric blocker** (six arms, 96 episodes; see task 10).
- **Next objective:** collect Campaign A, classify every stall, then Campaign B
  baseline vs `prev_actions_reset` on the exposed episodes.
- **Jobs:** completed `21400561`; pending none. Episodes used: 217 + 48 no-lateral arms + 10 replay (wait-open) = 275 / 512.

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
| 4 | Campaign A: 16+16 early-handoff episodes, chain traced, mechanism per episode | DONE | 32 episodes; during-stage 15 (wall), after-stage 12 (falls), stall exposures 5 (`frozen_targets` 4); `campaign-a-summary.json` |
| 5 | Campaign B: baseline vs `prev_actions_reset` on the 5 exposed episodes, fingerprint-matched | DONE | 5/5 applied, fingerprints matched; Doors/1 → success, Doors/3 regressed downstream; un-sticks the fixed point 5/5 but not promoted alone |
| 6 | In-stage factors F1/F2/F3 | ACTIVE | F1 rejected (0/16 live, 8/10 replay); F2 load-bearing (falls 5→1); closed-door class remains (wait at pre-point presses nothing); no-lateral arms and press-and-hold next |
| 7 | Freeze one recovery; predeclare eval set, metrics, n, stopping rule | BLOCKED | needs B |
| 8 | Confirmatory eval on fresh validation layouts 16–31 (never used), paired baseline | BLOCKED | needs 7 |
| 9 | Exact replay regression for every PlateStage change | ACTIVE | F1: FAIL 8/10; F2 wait-open: `21400863` running |
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
| 21400752–54, 21400780 | Campaign A: 16 Doors + 16 Transport, chain traced | DONE Doors 3/16, Transport 1/16; `21400755` OOM, fixed, 13–15 rerun |
| 21400801–804 | In-stage arms F1, F1+F2, F1+F3, F1+F2+F3 on 16 Doors | DONE 0, 3, 0, 3 /16; falls 5→1 with F2 |
| 21400805 | Replay gate with F1 0.25 m | DONE **8/10 FAIL** — approach trips on the plate; F1 rejected |
| 21400806–807 | Campaign B: `prev_actions_reset` on 5 exposed | DONE 5/5 APPLIED; +1 −1; not promoted alone |
| 21400825–827 | In-stage arms F2, F3, F2+F3 (plate-centre lateral) | running |
| 21400863 | Replay gate with `--wait-open=2.0` (F2's PlateStage part) | running |

## Confirmatory protocol (predeclared 2026-09-23, before any in-stage or Campaign B result was read)

- **Candidate selection** (development only): validation layouts 0–15. The
  frozen candidate is the in-stage arm with the highest Doors route success on
  `21400801–804`, combined with `prev_actions_reset` only if Campaign B shows
  it APPLIED with ≥1 matched improvement and 0 matched regressions among the
  exposed episodes. Ties break toward fewer factors.
- **Regression gate:** the frozen candidate's PlateStage change must keep the
  exact ten-fall replay at 10/10 upright (`21400805` for F1); F2/F3 act in the
  live route controller and are outside the scripted replay by construction.
- **Evaluation set:** validation layouts **16–31**, never used by any run so
  far, both stages: 32 candidate episodes and 32 matched baseline episodes
  (unchanged controller, same early handoff), 64 total, run under the hardware
  constraint with the same source snapshot.
- **Primary metric:** route success (paired by stage/layout). Secondary: falls,
  in-stage collisions, post-exit falls, stall exposures. Denominator is every
  episode; nothing is excluded.
- **Precision:** with n = 32 per arm, a paired exact test on discordant pairs
  is reported with a 95 % Clopper–Pearson interval on each arm's rate; only a
  difference whose interval excludes zero is called an improvement. If the
  discordant-pair count is under 6 the result is reported as underpowered, not
  as null.
- **Stopping rule:** one evaluation. No tuning against layouts 16–31 and no
  second look; the test split (seeds 20000+) is untouched and reserved for its
  documented final use.
