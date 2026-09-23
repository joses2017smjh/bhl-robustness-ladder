# Mission 7 task checklist

Persistent, evidence-linked status for the Mission 7 campaign. Updated after
each meaningful batch. Status vocabulary: **DONE** (evidence linked),
**ACTIVE**, **BLOCKED** (names the blocker), **SUPERSEDED** (names what
replaced it). Route failures are never dropped from a denominator.

## Current status

- **Last updated:** 2026-09-23, campaign closeout.
- **Source checkpoint:** see `git log`; every job's source hashes are in its
  `submission.json`.
- **Achieved:** three mechanisms established with instrumented evidence —
  post-stage stall = gait feedback fixed point (`history_length 0`; the only
  persistent input is the 22-element `prev_actions`; 2 of 5 exposures were
  genuine stalls and `prev_actions := 0` un-stuck both, stall-anchored); in-stage failure = plate placement 0.29–0.39 m from
  the wall against a 0.316 m half-body, plus a wait that presses nothing;
  world −x = 0.632 m robot vs 0.61 m post gap. Infrastructure: preflighted,
  hash-frozen, pass-through submitters for route, Approach and replay gate;
  25 Hz chain trace with mechanism labels; exposure-aware delivery states;
  artifact size guard; 172 tests.
- **Baseline vs final route results (development set):** Doors 3/16 → best
  gate-passing arm 1/16 (falls 5 → 1); Transport 1/16, not re-run (no
  candidate). **No validated improvement.**
- **Current limitation:** every lateral arm loses layouts 4 and 7; the
  plate-press / wall-clearance / plate-trip constraint band admits no tested
  offset that raises route success; standstills feed the fixed point; the
  reset regresses downstream or is never exposed once in-stage factors are on.
- **Next objective (outside this campaign's evidence):** a stage that presses
  without standing still — one deliberate footfall onto the plate inside a
  continuous crossing — and a per-layout offset side chosen from the corridor
  geometry, since layouts 4 and 7 are lost by every lateral arm.
- **Jobs:** 31 completed this campaign; none pending.
- **Episodes used:** 354 / 512 (163 remained at reopening; 5 spent; ≤20 gates, ≤48 dev, 64 confirmatory planned). Retained artifacts 6.8 GB of 10 GB; disk
  ≥ 150 GB free throughout.

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
| 5 | Campaign B: baseline vs `prev_actions_reset` on the 5 exposed episodes, fingerprint-matched | DONE | 5/5 applied, fingerprints matched; 2/5 genuine stalls, both un-stuck (stall-anchored); 3/5 transient pauses; Doors/1 → success, Doors/3 regressed; not promoted. Review corrections applied |
| 6 | In-stage factors | REOPENED | the six lateral arms staged the robot 0.75 m past the plate (anchor bug, fixed) — withdrawn; no-lateral arms stand (F2/F3/F2F3/A1 all ≤ baseline). Root causes now traced: post-exit falls 8/8 and approach falls are raised-plate trips; crossing covers only 0.13 m |
| 7 | Freeze one recovery under the predeclared rule | ACTIVE | Doors/3 regression explained (mid-walk reset lurch → plate trip); `--stall-min-s 3.0` declared and under matched verification (`21401685–686`) |
| 8 | Confirmatory eval on validation 16–31 | BLOCKED | needs a candidate passing selection + replay gate; 64 episodes reserved |
| 9 | Exact replay regression for every PlateStage change | ACTIVE | offset gates withdrawn (misplaced target); wait-open PASS stands; new gates needed for pre-point 0.45 / cross-until-clear / corrected offset |
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
| 21400825–827 | In-stage arms F2, F3, F2+F3 (plate-centre lateral) | DONE 1, 1, 1 /16 — worse; wait-open standstill harms |
| 21400863 | Replay gate with `--wait-open=2.0` | DONE **10/10 PASS** |
| 21400895–896 | A1 activate-only; A2 A1 + lateral 0.35 | DONE 2/16, 0/16 — A2 clears walls, crossings find the door closed |
| 21400897 | Replay gate at lateral 0.35 | DONE **10/10 PASS** |
| 21400952 | F1+F2+F3 at 0.35 m (last predeclared arm) | DONE 1/16, falls 5→1 |
| 21400953 | Combined replay gate 0.35 m + wait-open 2.0 | DONE **10/10 PASS** |
| 21400961 | Interaction: reset + F1F2F3@0.35 on 4 exposed Doors | DONE 0/4 — withdrawn with the misplaced-target arms |
| 21401685–686 | Reset with `--stall-min-s 3.0`, unchanged stage, 5 exposures | running |

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
