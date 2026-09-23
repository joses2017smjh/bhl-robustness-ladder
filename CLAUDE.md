# CLAUDE.md — bhl-robustness-ladder

Orientation for anyone (human or agent) picking this repo up cold.
Written 2026-09-22 against `be2db75` on branch `mission7-approach-followup`.

The fragility section below is written as *findings plus what was done about
them*. Items marked **Fixed** were changed in the working tree on 2026-09-22
and are not yet committed; the rest still stand.

> Scope note: this file covers `Humanoid_Lite/bhl-robustness-ladder` only.
> The separate `Computer_Vision` tree (spur/DA2-DA3 depth work) was not reviewed here.

---

## 1. What this project is

Humanoid robotics research built around the upstream Berkeley Humanoid Lite.
Two things happen here, and keeping them apart is the whole point of the repo:

- **Training** — PPO in Isaac Lab / Isaac Sim, on GPU, via Slurm + Apptainer.
- **Evaluation** — exported gaits replayed in **MuJoCo 3.3.5**, on CPU, scored
  against explicit success predicates with matched negative controls.

A completed job is not a result. Nearly every doc in `docs/` exists to keep
"the job finished" separate from "the task succeeded".

**Current active workstream: Mission 7** — procedural sensor-driven navigation
missions (Doors / Transport) on generated 6×6 maze layouts. All Mission 7 jobs
are **CPU-only, zero GPU**, and bypass the Apptainer container entirely.

---

## 2. Layout

```
src/bhl_robust/          importable package (PYTHONPATH=src; not pip-installed)
  mission/               ** Mission 7 core ** layout, state, sensors, policy, env,
                         approach_debug (route controllers live here)
  eval/                  MuJoCo harnesses: harness, multi_robot, team_airlock,
                         inspection_maze, coop_replay, video, mjcf_assets
  tasks/                 Isaac Lab task configs, rewards, curriculum stages
  cloth/                 dual-arm folding / sorting study (separate from Mission 7)
  terrains/              maze, ice, stairs, bumpy generators
scripts/                 entry points, one file per experiment
  mission7*.py           ~18 Mission 7 runners and analyzers
  submit_mission7*.py    Slurm submitters (snapshot + sbatch + receipt)
  bench/                 gates and probes invoked by slurm/inner/*.sh
slurm/                   ~137 .sbatch launchers, numbered by era (00_ .. 99w_)
  _env.sh                shared paths + Apptainer wrapper — read this first
  inner/                 ~103 scripts that run *inside* the container
  mission7_*.sbatch      CPU Mission 7 jobs (no container, no GPU)
docs/                    ~35 .md — the real state of the project
results/                 dated campaign dirs + machine-readable evidence
tests/                   15 CPU test files, 171 tests
external/                Berkeley-Humanoid-Lite submodule (gitlink, pinned)
SLURM_JOBS.md            185 KB append-only job ledger
```

---

## 3. Key files and how they connect

### The Mission 7 job pipeline (the part being actively worked)

```
scripts/submit_mission7_route_handoff.py
    │  copies a frozen SOURCE SNAPSHOT into results/<campaign>/<name>/source/
    │  writes sha256.json over every copied file
    │  writes submission.json (receipt: job id, argv, git commit, hashes)
    │  appends one line to SLURM_JOBS.md, fsync'd
    ▼
slurm/mission7_route_handoff_probe.sbatch        (runs FROM the snapshot)
    │  re-verifies every sha256 before doing anything
    │  8 POSITIONAL args: repo snapshot out stage indices handoff diagnostic rejoin_fix
    ▼
scripts/mission7_route_handoff_probe.py          (the actual experiment)
    │  RouteHandoffController = PlateSafeRouteController + PlateStage
    ▼
results/<campaign>/<name>/result.json  +  <stage>-<index>.json
```

The snapshot is the good idea here: jobs run against immutable, hash-verified
source, so a result can always be traced to exact bytes. Use it — when a past
job's behaviour is in question, **diff the snapshots**, don't guess. That is how
the defect in §5.1 below was identified.

### Controller layering

- `src/bhl_robust/mission/approach_debug.py` — `DebugEnv`, `PlateSafeRouteController`,
  `wrap`, `yaw_of`. The route-following baseline.
- `scripts/mission7_plate_stage.py` — `PlateStage`, the guarded pressure-plate
  crossing controller (approach radius `.78` m, settle `.40` s, cross `1.20` s).
- `scripts/mission7_route_handoff_probe.py` — composes the two and controls
  *when* authority transfers (`--handoff switch|early`).

### Environment

- `slurm/_env.sh` — every path, both Isaac stacks, and `bhl_exec`. Its comments
  document real failures (Apptainer comma-splitting, MIG device masks,
  `--cleanenv` stripping variables). Read before touching any GPU job.
- Two Isaac stacks, opt-in per job via `BHL_STACK`:
  - `v51` (default) — Isaac Sim 5.1 + Lab 2.3.2, py3.11. **Every published
    number came from here.** RTX segfaults; depth comes from Warp ray-casting.
  - `v60` — Isaac Sim 6.0 + Lab 3.0.0b2, py3.12. RTX works, but the API moved
    and task overlays are unported — v60 numbers are not comparable to v51.

### Campaign tooling added 2026-09-23

- `docs/MISSION7_TASKS.md` — persistent checklist: gates, tasks with status
  vocabulary (DONE / ACTIVE / BLOCKED / SUPERSEDED), envelope usage, job table,
  and the **predeclared confirmatory protocol**. Read it before the ledger.
- `scripts/mission7_route_handoff_probe.py` flags: `--chain-trace` (25 Hz
  command-to-motion chain and mechanism labels), `--rejoin-fix
  prev_actions_reset`, `--stage-lateral`, `--stage-activate`, `--exit-ramp`,
  `--preflight`, `--allow-inactive-intervention`. Every run records the stall
  branch time and fingerprint; `intervention_delivery` is per episode.
- `scripts/submit_mission7_approach.py` + `--controller/--directions` in
  `mission7_approach_followup.py` — one-factor privileged Approach arms.
- `scripts/submit_mission7_plate_stage.py` — the 10/10 exact replay gate,
  pinned to `cn-c22` by protocol; re-run it for any PlateStage change.
- `scripts/mission7_campaign_analysis.py campaign-a|campaign-b` — compact
  aggregates, exposure lists, fingerprint-matched pairing; tolerates partial
  output dirs from jobs that died mid-run.
- All three sbatch files forward probe flags verbatim after their positional
  paths and run `--preflight` on the exact science argv first.

### Where the truth lives

Read in this order when returning to the project:
0. `docs/MISSION7_TASKS.md` — current status, gates, job table, protocol
1. `docs/MISSION7.md` — opens with a dated status paragraph
2. `docs/MISSION7_APPROACH_FOLLOWUP.md` — detailed current log
3. `SLURM_JOBS.md` (tail) — job ledger
4. newest `results/mission7-*/`
5. `git log --oneline --decorate -10`

---

## 4. Build and test

No build step. The package is used via `PYTHONPATH=src`, never installed.

**CPU tests (this is the only thing that runs without Isaac or GPU):**

```bash
cd /nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder
OMP_NUM_THREADS=2 PYTHONPATH=src \
  /nfs/hpc/share/sanchej7/Humanoid_Lite/venv/bin/python -m pytest -q tests
```

Verified 2026-09-22: **172 passed in ~35 s** (171 existing plus the new
artifact-size guard, §5.3). The README still says "146 CPU tests pass" — that
number is stale and should be updated when this work is committed.

The venv (`$WORKSPACE/venv`, Python 3.11.16, mujoco 3.3.5, numpy 1.26.0) lives
**outside the git tree on purpose**: it is ~30 GB and uv bakes absolute paths
into it. Do not try to recreate it inside the repo.

`requirements-test.txt` is for a *fresh* environment (plus torch 2.7.0 from the
CPU wheel index); it has not been independently validated. On this cluster,
just use the existing venv.

**GPU / Isaac work** needs `container/bhl.sif` + `slurm/_env.sh` and must go
through Slurm. Mission 7 CPU jobs deliberately do neither.

---

## 5. Odd and fragile

### 5.1 A job can succeed, report the intervention as applied, and have done nothing

**This is the most important thing in this file.**

Job `21399502` exited `0:0`, printed a clean completion line, and wrote a
`result.json` containing `"rejoin_fix": "forward_pulse"`. The pulse never fired.
`submission.json` also says `forward_pulse`. **The two receipt files agree with
each other and both are misleading** — the only distinguishing signal anywhere
in the output is `rejoin_fix_events: []` (0 events vs 2 in the corrected rerun
`21399503`).

`SLURM_JOBS.md` records the cause as *"the probe runner omitted the submitted
fix argument."* **That explanation is wrong.** The snapshot proves it: the
`21399502` snapshot's sbatch is byte-identical to the current one and does pass
`--rejoin-fix "$rejoin_fix"`, and the submitted argv ends in `forward_pulse`.

The real cause was a logic defect in the stall detector. In that snapshot,
`post_stage_last_progress_time` was refreshed whenever the base moved more than
2 cm (`observe_step`, lines 324–327). A humanoid stalled at a waypoint still
sways and steps in place, so "movement" kept resetting the timer and the 0.8 s
stall threshold was never crossed. The fix redefined progress as *distance to
the current waypoint decreasing by 2 cm*. **It measured motion where it meant
progress.**

There is a second, deeper instance. `21399503` — the "corrected" rerun — *did*
fire its pulse, and was still a no-op: the "0.30 m/s forward restart pulse"
imposed 0.300 m/s while the route controller was **already commanding
0.300 m/s forward**. Measured forward-command delta is `0.000000` m/s. So an
assertion on event count alone would have passed it. The intervention was
indistinguishable from doing nothing.

**Fixed.** The probe now reports a top-level `intervention_status` with four
states — `NOT_REQUESTED`, `REQUESTED_BUT_NEVER_FIRED`,
`FIRED_BUT_DID_NOT_CHANGE_FORWARD_COMMAND`, `APPLIED` — records the command
with and without the intervention at every step it fires, and exits non-zero on
the two failure states. Evidence is written first, so nothing is lost by
failing. `--allow-inactive-intervention` records a deliberate null result
instead. Replaying the `21399503` episode under the new build fails with
`FIRED_BUT_DID_NOT_CHANGE_FORWARD_COMMAND`, where the original exited `0:0`.

The `SLURM_JOBS.md` entries for both jobs have been corrected in an appended
block (the ledger is append-only, matching how `docs/FINDINGS.md` keeps
retracted claims rather than deleting them).

### 5.2 The intervention cannot be run without the instrumentation

The `forward_pulse` fix is gated on `self.rejoin_diagnostic` being true
(`mission7_route_handoff_probe.py:288`). Diagnostic mode also **monkey-patches
`self.route.action`** with a tracing closure (lines 66–83) and records two full
state snapshots per control step. So the intervention arm and any clean baseline
differ by more than the intervention. Decouple `--rejoin-fix` from
`--rejoin-diagnostic` before treating a pulse result as a controlled comparison.

Related: `observe_step` returned early when `rejoin_diagnostic` was false, and
that early return was what cleared `rejoin_fix_until`. If the fix were ever
enabled without the diagnostic, the pulse would have latched on forever.

**Fixed.** Stage-exit tracking and the pulse window are now maintained
unconditionally — they cost one timestamp and a counter — and only the heavy
tracing stays behind `--rejoin-diagnostic`. The intervention can now be run as
a clean one-factor change without the monkey-patch and the per-step snapshots.

### 5.3 1.44 GB of untracked, un-ignored files under `results/`

`.gitignore` blocks large traces by *filename* — line 91 is
`results/mission7-*/**/episodes.json`. But the newer probes write their traces
as `result.json`, `routes.json`, `doors.json`, `<stage>-<index>.json`, none of
which match. Currently un-ignored and untracked:

| Size | File |
|---:|---|
| 92.7 MB | `results/mission7-approach-followup-20260922/route-eval-cn-c22-v2/routes.json` |
| 84.1 MB | `...-rejoin/route-rejoin-diagnostic-cn-c22/result.json` |
| 71.4 MB | `...-early/route-early-handoff-cn-c22/result.json` |
| … | **1.44 GB total, 815 files** |

`.git` is already 702 MB. A single `git add -A` stages all of it, and git keeps
blobs forever — practically irreversible without a history rewrite. (Nothing
exceeds GitHub's 100 MB hard limit, but four files trip the 50 MB warning.)

**Fixed, three ways.**

1. *The producer.* `result.json` embedded every episode row whole, duplicating
   the per-episode files beside it. Measured on one row: `plate_contact_events`
   8.1 MB, `rejoin_diagnostic` 5.1 MB, `diagnostic_trace` 1.9 MB of 15.6 MB
   total. `result.json` now carries the verdict and a compact per-episode
   summary; traces stay in `<stage>-<index>.json`. A real Doors/1 run that
   previously produced a 27.8 MB `result.json` now produces **512 bytes**,
   with the 16 MB trace beside it and ignored.
2. *The patterns.* `.gitignore` now ignores by the shape the probes emit —
   `*episodes*`, `*routes*`, `*doors*`, `*transport*`, `*standstill*`,
   `*validation-*`, `replay/`, `*-trace` — with `*summary*`, `submission.json`
   and `sha256.json` re-included so compact evidence still commits. The closed
   2026-09-21 Approach campaign (nothing in it was ever tracked) is ignored
   wholesale, and the twelve legacy oversized `result.json` files are listed
   explicitly rather than pretended away.
3. *The invariant.* Patterns alone cannot settle this, because `result.json` is
   both the cited verdict and, historically, the raw trace.
   `tests/test_result_artifacts.py` asserts that nothing git *would* commit
   under `results/mission7-*` exceeds 5 MB — checked over tracked plus
   untracked-not-ignored, never a filesystem walk, and skipped outside a git
   checkout. Verified to fail on a planted 6.7 MB file.

Committable bytes under `results/mission7-*` went from ~1.4 GB to **20.7 MB**,
with nothing over 5 MB. The rule still holds: **stage Mission 7 paths
explicitly; never `git add -A` here.**

### 5.4 Six of the last ~14 submissions died before producing science

From `SLURM_JOBS.md`: `21396684`, `21397985`, `21398501`, `21399179`, `21399201`
all failed on infrastructure before executing an episode, and `21399502`
silently no-opped. Three recurring causes:

1. **Snapshot not import-closed** — the file list in
   `submit_mission7_route_handoff.py` is hand-maintained (`submit_mission7.py`
   has 5 more hardcoded paths; the overnight and debug submitters have none).
   Adding a `scripts/mission7*.py` glob fixed it twice.
   *Verified 2026-09-22: the current list IS closed — 17/18 snapshot entry
   points import cleanly with only the snapshot on `sys.path`. Nothing enforces
   this going forward.* (The 18th, `mission7_plate_safe_single.py`, reads
   `os.environ['MISSION7_REPO']` at module level — harmless, it is an array
   worker never imported by anything, but it will break any bulk-import check.)
2. **Wrong interpreter** — `cn-c22`'s bare `python3` has neither numpy nor
   mujoco. The sbatch now defaults to the shared venv; `PYTHON=` still overrides.
3. **Positional-argument drift** — the sbatch takes 8 positional args. Adding a
   parameter means editing the submitter's command list, the sbatch's `${N}`
   parsing, and the runner's argparse in lockstep. Nothing checks they agree.

**Five of the six were preventable by one preflight in the sbatch**, run before
the science: verify the interpreter imports mujoco + numpy, and dry-run the
runner's argparse against the exact forwarded argument list.

**Fixed.** `slurm/mission7_route_handoff_probe.sbatch` now verifies the source
hashes, then runs the probe with `--preflight` on the *exact* argument list the
science will use, before spending queue time. Importing the probe proves the
interpreter has numpy and mujoco and that the snapshot is import-closed;
parsing proves every forwarded flag is accepted. It cost ~4 s in production on
job `21400561`.

Cause 3 is gone rather than documented: the sbatch now takes only the three
paths positionally and forwards everything after them to the probe verbatim, so
a new probe flag needs no change in the sbatch or the submitter's argument
list. Adding `--allow-inactive-intervention` required no sbatch edit at all.

The sixth, `21399502`, was **not** — its argument was forwarded and accepted
correctly (see §5.1). It needed the other guard: fail the run when a requested
intervention produces zero events.

Ranking, if you only do one:
1. **Zero-events assertion (§5.1).** The preflight saves queue time; this one
   keeps a wrong result out of the record. `21399502` cost a day of
   interpretation and put an incorrect causal claim into `SLURM_JOBS.md`.
2. **Sbatch preflight.** Cheaper to write, recovers ~5 wasted submissions per
   14, and is what unblocks queueing jobs in bulk.

### 5.5 Smaller sharp edges

- **`.78` means two unrelated things.** It is the fall tilt threshold in
  `mission/env.py:227`, `mission/approach_debug.py:42`,
  `mission7_plate_stage.py:166` and `mission7_route_handoff_probe.py:382`, and
  it is the `PlateStage` capture radius **in metres** at
  `mission7_plate_stage.py:58`. Changing either will read as changing both.
  The tilt threshold is duplicated in four places rather than imported.
- **`SLURM_JOBS.md` is append-only and has duplicate/out-of-order entries.**
  `21399201` and `21399211` each appear twice in the tail, with later SUBMITTED
  lines written after their COMPLETED lines. Some job ids appear 5–6 times.
  Grep for the *last* mention of an id, not the first.
- **Node pinning — do not simply drop it.** The `share` partition is
  heterogeneous: sandybridge through skylake (AVX, AVX2, AVX-512) and **both
  el8 and el9**. Different ISA and different glibc change floating-point
  results, so an unpinned job is not comparable to the completed `cn-c22` runs.
  **Fixed** by replacing the default `--nodelist=cn-c22` with
  `--constraint=haswell&el8` — `cn-c22`'s exact hardware and OS class, 8 nodes
  instead of 1 — while `--node` is kept for the exact bitwise replay gate,
  where the single original physics node is the point. Verified the constraint
  actually schedules (a constraint matching nothing pends forever): a probe job
  allocated in ~8 s and landed on `cn-c22`.
- **`submit_*.py` refuses to reuse an output dir** (`destination exists`). This
  is why campaign dirs have grown `-v2`, `-v3`, `-early`, `-rejoin`,
  `-rejoin-fix` suffixes. Intentional (immutability), but the naming no longer
  says what distinguishes them — put that in `submission.json`, not the path.
- **`scripts/` has ~30 stale `__pycache__/*.pyc`** across py3.9/3.11/3.12/3.14,
  including for deleted modules. Harmless, but misleading when grepping.
- `python3` on the login node is **3.14.4**; the project needs **3.11**. Never
  rely on bare `python3`.

---

## 6. Conventions worth preserving

- One experiment per script in `scripts/`; no shared mutable driver.
- Every submission writes a `submission.json` receipt with git commit + hashes.
- Results claims in docs cite a compact JSON artifact, not a log.
- Negative controls are reported alongside positives, with the denominator kept.
- Retracted and corrected claims stay in `docs/FINDINGS.md` rather than deleted.
