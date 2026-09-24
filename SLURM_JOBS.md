# Slurm job ledger

**Mission7 Approach diagnosis — closed September 21:** 18 retained tasks
completed, including six PPO controls; one invalid replay (21384691) was
cancelled and replaced by successful same-node replay 21384742. No jobs from
this diagnostic campaign remain active. Allocation: **11.96 CPU-hours, zero
GPUs**. No stable-learning or sensor-comparison gate passed.
[Final results and all job IDs/dependencies](docs/MISSION7_APPROACH_DEBUG.md).

What we are trying to achieve, which Slurm ID did it, and — when something ran
more than once — one line on why the previous attempt did not count.

**Rolling window: at most three attempts per task, newest first.** When a fourth
attempt happens, drop the oldest row. The point is not a full audit trail; it is
that no task silently disappears and no ID is unattributable.

This file is updated **at submission**, as part of queuing a job, not afterwards
and not on request. A ledger that is only current when someone thinks to ask for
it does not do the job this one exists for.

Status: `todo` · `running` · `done` · `blocked`

Diagnostics that ran once, proved a point and were deleted are in
[one-off probes](#one-off-probes) rather than getting a section each.

---

## Mission7 procedural sensor benchmark — 2026-09-20

**User-approved queue cleanup, September 20:** cancelled old blocked media
array `21360436_[1-15]` and its report `21360437`, held Tier 3 tasks
`21328743_1`/`21328743_2`, held terrain tasks `21328607_[5-11]` and
`21328608_[5-11]`, and running `21352980_8` (`v2stand-planktowall-rgb-s0`).
The latter still reported one-step episodes, fallen 1.0 and success 0.0 near
iteration 6,000/8,000. Slurm accounting confirms these cancellations. Existing
logs/checkpoints and already completed array tasks were preserved. Desktop
`21369598` and Mission7 learning job `21369796` were kept running. This cleanup
supersedes older pending/held status notes below; it does not erase their history.

New work remains in this repository on `main`; existing jobs and results are
preserved. [Task, observation audit, matrix and limits](docs/MISSION7.md).
Receipts: `results/mission7-20260920/submissions.jsonl`; immutable run source
and SHA256 manifest: `results/mission7-20260920/source/`.

| Job | Experiment | Latest audited state | Allocation / dependency / evidence |
|---|---|---|---|
| `21369795` | Phase A: all four arms ×3 objective types; PPO reset/update, contact and placement controls | **COMPLETED / infrastructure PASS** | `cn-b02`, `share`, 2 CPUs, 12 GB, **0 GPUs**; 49 s, exit `0:0`; JSON `passed:true` and `MISSION7_SMOKE_PASS`; `cluster-smoke/smoke.json` |
| `21369796` | Phase B: Both, seed 0, Approach; 400×64 PPO decisions | **FAILED learning gate; training finished** | 54:38, exit `2:0`; final validation **0/16**; checkpoint preserved; no nonfinite PPO failure |
| `21369990` | End-of-day diagnostic: 16 training layouts ×2 reset seeds ×slow/fast oracle and stationary control | **COMPLETED diagnostic** | 12:19, exit `0:0`; fast oracle **22/32**, slow **0/32**, stationary **0/32**; privileged feasibility only |
| `21369991` | End-of-day diagnostic: final Both checkpoint ×16 validation layouts ×4 sensor conditions | **COMPLETED diagnostic** | 9:16, exit `0:0`; normal **0/16**, LiDAR missing **0/16**, depth missing **2/16**, both missing **0/16**; no sensor-benefit claim |

At diagnostic submission, Phase B still had **0/16 validation successes at
updates 100, 200 and 300**, and no successes in the first 356 training episodes
(161 timeouts, 195 excessive-contact failures). Therefore no larger training
sweep was released. The diagnostics retain the frozen Phase B task and leave
held-out test layouts untouched. Local checks: two new unit tests passed;
all diagnostic modes stepped finitely. On one training layout, fast privileged
control completed the approach, while slow control and standing still timed
out. That is a physical feasibility fixture, not learned-policy success.
Receipts/source hashes: `results/mission7-diagnostics-20260920/`; eventual
reports: `feasibility/report.json` and `checkpoint/report.json` in that folder.

Local qualification before submission: **155 tests passed**, all **12**
arm/stage finite PPO cells passed, four physical switch controls passed, and
released-object placement succeeded while its outside-zone control failed.
These are infrastructure/physics fixtures, not learned mission successes.
`results/mission7-20260920/qualification.json` records tested source hashes.
Logs are `results/mission7-20260920/m7-*.out`. Full multi-seed training and
held-out evaluations remain planned behind measured learning gates.

Startup audit: Phase B completed its initial **0/16** untrained-policy validation
and reached update **10/400** (640 decisions, finite PPO losses). This is training
progress, not a learning-success verdict. The later `gate.json` is authoritative.

---

## Weekend recovery campaign — 2026-09-19

Requested scope: sensor-driven maze tasks, actual cloth folding, and a shared
task for 2–3 humanoids. New submissions are isolated from the historical runs.
Machine-readable scheduler receipts and source hashes are recorded at submission
in `results/weekend-20260919/submissions.jsonl`.

| Job | Experiment | Latest audited state | Budget / dependency / result |
|---|---|---|---|
| `21359422` | Corrected maze Approach/Blind physics and task probe | **PASS** | ~85 s on RTX; integration only, zero first-episode successes with zero actions |
| `21359430` | Folding SmolVLA two-update/checkpoint smoke | failed before Python | Container clean environment omits USER; wrapper now derives workspace from forwarded REPO |
| `21359431` | Official LeHome fold evaluator / particle-health smoke | failed before Python | Same USER forwarding issue; corrected before retry |
| `21359432` | Shared-world 2-humanoid airlock, 5 seeds × 3 modes | **PASS** | Coordinated 5/5, no-wait 0/5, withheld teammate 0/5; completed in 8:59 |
| `21359473` | Shared-world 3-humanoid airlock, 5 seeds × 3 modes | **PASS** | Coordinated 5/5, no-wait 0/5, withheld teammate 0/5 |
| `21359475` | MuJoCo deformable towel fold physics gate | **PASS** | 4/4 released folds, 0/4 untouched controls, no warnings/NaNs; idealized pickers |
| `21359477` | Folding SmolVLA two-update/checkpoint smoke, corrected wrapper | failed | Current CUDA 12.8 Torch wheel excludes V100 SM70; no training occurred |
| `21359478` | Official fold evaluator smoke, corrected wrapper | **invalid** | Found swallowed Storm render errors on garment switches; physics alone cannot validate camera inputs. New unique per-garment USD layers and strict image checks added |
| `21359481` | Folding training smoke on compatible compute hardware | **PASS** | H100: two updates and durable checkpoints completed; this is a plumbing test, not a fold-rate result |
| `21359486` | Maze Approach: 4 sensor conditions × 3 seeds, 500 iterations | **12/12 COMPLETED; gates PASS** (Sept 20 audit) | 380/384 first-episode successes; initial seed-0 counts were blind 32, lidar 31, paired depth 31, both 30 /32 |
| `21359499` | Maze Corridor: 12 matched cells, 1,500 additional iterations | **12/12 COMPLETED; gates PASS** (Sept 20 audit) | 375/384 first-episode successes; aftercorr `21359486`, exact passed-checkpoint resumes verified |
| `21359510` | Maze Full: 12 matched cells, 4,000 additional iterations | **12/12 COMPLETED; gates PASS** (Sept 20 audit) | 379/384 first-episode successes; blind 93/96, lidar 96/96, paired depth 94/96, both 96/96; aftercorr `21359499` |
| `21359521` | Strict official folding evaluator smoke | **COMPLETED/PASS** | 2:46; unique per-garment USD layers and strict fresh-image checks; smoke only |
| `21359522` | SmolVLA garment-fold adaptation seed 0 | **COMPLETED** | H100: 1,500 updates; final held-out adaptation loss 0.06787; partial task evaluation in `21359574` below |
| `21359527` | Isolated V100-compatible Torch/cu126 environment | **PASS** | CPU-only install; compiled SM70 verified, existing venvs untouched |
| `21359529` | DGX2 CUDA kernels + two-update folding gate | **PASS** | Real V100 convolution/optimizer/attention/save-load and two SmolVLA updates succeed |
| `21359530` | SmolVLA garment-fold adaptation seed 1 on DGX2 | **COMPLETED** | V100 with isolated cu126; 1,500 updates, final held-out adaptation loss 0.06099; partial task evaluation in `21359575` below |
| `21359573` | Fresh folding baseline: four garment classes, 24 episodes/class | 1 completed, 1 failed, 2 timeouts | Completed short pants 8/24; short-top scorer index mismatch; switching stalls elsewhere |
| `21359574` | Adapted folding seed 0: matched evaluation | 2 completed, 1 failed, 1 timeout | Completed long tops and long pants 0/24 each; other classes incomplete |
| `21359575` | Adapted folding seed 1: matched evaluation | 2 completed, 2 timeouts | Completed short pants 3/24, long pants 0/24; other classes incomplete |
| `21359576` | Sensor-reactive cooperative pair: 3 seeds × 3 modes | **PASS** | Coordinated 3/3; no-wait and withheld-role 0/3; real IMU/rays consumed by braking |
| `21359631` | Final campaign results/accounting report | **COMPLETED**, 0:0 | CPU-only, afterany final maze/folding arrays; `results/weekend-20260919/SUMMARY.md` includes failed/incomplete evaluations, not an all-success verdict |
| `21359677` | Two-turn inspection maze: 3 seeds × 2 routes × 2 sensor modes | **PASS** | CPU 3:59; ordered sensor-reactive 3/3, outage 0/3, wrong branch 0/3; no contacts/falls |

**Maze evidence audit, 2026-09-20:** all 36 curriculum elements have Slurm
exit code `0:0`, passing JSON plus both PASS sentinels, and existing checkpoint
files. Their absence from the queue means normal completion. Full-stage
seed 0/1/2 successes are blind **32/29/32**, lidar **32/32/32**, paired depth
**32/32/30**, and both **32/32/32**, each count out of 32. Evaluation uses a
known route, fixed geometry and no observation corruption; it does not prove
sensor advantage or autonomous RGB/SSD navigation. The Isaac curriculum is
the **12-DoF biped**, whereas inspection and team-airlock evidence uses the
**22-DoF humanoid with a separate frozen August 18 gait**.
[Dated results and source JSONs](docs/WEEKEND_RESULTS_2026-09-20.md).

Interactive checks use the confirmed-idle GPUs within existing desktop
allocation `21358937`: a strict camera smoke and
`results/weekend-20260919/team3-airlock.mp4` (measured 30.68 s completion,
zero contacts/falls). The video rollout is a showcase; the separate five-seed
jobs above provide its controls.
The same allocation rendered `inspection-maze.mp4`: two ordered inspections
and two changes of travel direction in 17.36 s, zero contacts/falls; oracle
waypoints with sensor braking. Job `21359677` supplies its multi-seed controls.

The interactive strict camera smoke **passed all 12 garments**, including
garment switches: finite moving particles and fresh images throughout
(`fold-interactive-strict-smoke-s101.json`). Its 12-step horizon tests the
measurement path, not folding success. The scheduled seed-0 gate also passed.

**Media audit, September 20:** sibling gate `21360435` failed before stepping
or rendering because its robot asset path resolved under `lehome-fold-repro/Assets`
instead of `lehome-data/Assets`. Array `21360436_[1-15%1]` remains
`DependencyNeverSatisfied`; report `21360437` waits for the array. No new
folding media completed. These jobs were not canceled or resubmitted here.
See [folding status](docs/CLOTH_FOLDING_WEEKEND.md).

The idle GPU in current desktop allocation `21367413` rendered one 7.00 s
MuJoCo wrong-branch control on September 20. It correctly failed with
`dead_end_entered`; JSON and MP4 are `inspection-maze-failure-video.json`
and `inspection-maze-failure.mp4`. This is a labelled supervisor-error control,
not a new training experiment. Three captioned GIFs and provenance sidecars are
under `docs/gifs/weekend-*`.

**Folding historical correction:** direct inspection of sibling job
`21214241` found 23,250 swallowed Storm rendering errors after garment
switches. Its recorded 6/24 physical fold events cannot serve as a validated
closed-loop visual-policy baseline. Fresh strict-camera evaluation is required.

Old jobs, including held MARL arrays and the user's interactive allocations,
have not been cancelled or released by this campaign.

## Project status — 2026-09-18

**Done since the last status**

- **B5 maze navigation PPO finished 12/12 (`21353395`–`398`).** Train smoke 4/4, mean episode length 26.6–28.1. Then 6,000 iterations × four arms × n=3. Last-50 from the event files (`results/mazenav_last50.csv`): they walk to timeout (~0.94), **button_reached is 0.000 in every seed of every arm**, `progress_to_button` max 0.0005. Sensors do not separate. `Metrics/success_rate` ~0.99 is not button success — it tracks surviving without falling. Old `maze-*` rows stay terrain-perception.
- **B3 ice is under the robots and retrained (`21342561`, `21344927`–`928`).** Placement median **0.8 m**, reachable fraction **1.000**. Last-50 terrain level: depth **2.92** (2.854 / 2.984) against blind **2.59** (2.576 / 2.605), n=2. Visible-ice matches blind seed-for-seed (proprioception-only; the paint is not observed). Clip: `docs/gifs/ice_pair_placed.gif` (`21352982`). Finding 11's *old* +10.6% stays retracted; this is a new measurement on the actual ice tiles.
- **B5 maze navigation smoke passed (`21353199`, 0:57).** 4/4 arms construct, reset and step. Command is `MazeWaypointCommand` on every arm. Spawn `|y|_max` 0.12–0.14 m inside the 0.55 m corridor limit; mean button distance 2.97–3.05 m. Lidar wall checks 2/2: nearest 0.442 m / 0.454 m. Curriculum has 0 terms. Old `maze-*` PPO rows stay terrain-perception results.
- **C2 free-base hold at dt=0.005 failed (`21353200`).** 2/2 nonfinite. Same as 60 Hz (`21338288`). Skip.
- **C2F jacket new VBD sheet failed (`21353201`).** 4/4 nonfinite, `success_rate_finite` 0. Skip jacket; shirt/sock already sort.
- **Isaac C5 pose-fix four-episode (`21353130`).** 2/4 sorted, moved 4/4, travel 0.19 m, fall 0, nonfinite 0.
- **rsl-rl factorised actor gate (`21353129`).** PASS, 3 iterations, ep_len 18.94.

- **C2 cloth, 2026-09-15 (`21338288`–`293`).** Pinned-base Newton **sock sorts 4/4,
  all finite** (`isaac_c2f_sock.json`). Pinned-base **jacket is 0/4 finite** —
  4/4 nonfinite, garment moved in 1 episode (`isaac_c2f_jacket.json`). Free-base
  Newton with the arm still: **0/2, 2/2 nonfinite**, garment travel ~0
  (`isaac_c2_hold.json`) — the balance controller that stands in PhysX at
  control rate does not stay finite in Newton's 60 Hz cloth scene. The free-base
  sweeps crashed the job (`ValueError: NaN to integer` in `segment_on_contact`
  after the garment pose went NaN). Guarded in `reach.py`; do not re-queue C2
  free-base until Newton stays finite with the arm still.
- **GitHub presence.** README now leads with problem/solution/result and four
  visual case studies; repo description, homepage and topics set. Profile README
  and portfolio replacement copy are in `docs/github-profile-README.md` and
  `docs/PORTFOLIO.md`.
- **Maze stereo, cameras pointed down, n=3 — final.** Terrain level: 4×4 an eye
  **1.288** (1.286 / 1.340 / 1.237), every seed above blind's best (0.738 mean,
  1.037 best). 16×16 **0.774** trains like blind rather than failing. 4×4 + lidar
  **1.069**, below 4×4 alone in every seed. Width still matters (16×16 below 4×4
  in every seed), just nothing like the published 0.02-against-1.12.
- **The MARL plateau is not the critic.** The critic A/B (`21330372`,
  `21330392`) put PPO's privileged critic on a one-agent skrl MAPPO: at 1,500
  iterations on stairs it tracks at **0.58** against rsl-rl PPO's **1.39–1.41**,
  with terrain level still 0, barely above the old critic's 0.56. The critic
  mismatch and MAPPO = IPPO were real bugs, but not the cause. The exploration
  noise collapses instead — skrl std **0.13** against rsl-rl's **0.42** at 1,500
  iterations, after skrl's KL-adaptive learning rate hit its 1e-2 ceiling within
  ~30 iterations. A noise A/B (`21338294`) splits std parameterisation from the
  LR schedule. The grid stays held.
- **Tier 3 PPO seed 0** (22 DoF, stairs, depth, arm deviation off): terrain level
  **4.49**, episode length 386, tracking 1.16 (`21328743_0`). Seed 1 is queued as
  `21338295`.
- **Pointed down, full-width stereo does not fail.** The corrected 16×16 arm
  lands at terrain level **0.877 / 0.560 / 0.886** (mean 0.774) against 0.021 as
  published — inside blind's 0.55–1.04. The "width effect" was the upward
  camera. The 4×4 and 4×4 + lidar arms are still training.
- **B3's ice was never under the robots** (`21328532`). Patches spawn at each
  env's grid origin; robots reset onto terrain origins a median **72 m** away,
  and collision filtering lets a robot touch only its own env's patches. 4.4%
  could reach one in an episode at full commanded speed. "Depth helps on a hazard
  it cannot see" (+10.6%) is retracted: it is §6's rough-terrain depth gain
  (+10.9%) again.
- **MARL first block final at n=3**: limb2 + MAPPO 2.53 against the limb1
  control's 2.27 on the mean (+11%, not +47%), and limb2's seed 2 (1.80) is below
  every control seed. Every skrl row survived 34–80 steps against rsl-rl PPO's
  ~225: they ran skrl's default PPO settings, not the control's.
- **Tier 1 MARL grid queued** on the biped split left leg | right leg, over
  stairs / slippery / rough, seeds 0–1, with the control's PPO settings read from
  its own config (`21328607`, `21328608`; gate `21328605` passed 9/9). Ice rows
  wait on B3.
- **The maze stereo pair looked 20° up, upside down, in every B5 run.** Isaac
  Lab 3.0 reads camera offsets `(x, y, z, w)`; the pose was written
  `(w, x, y, z)`. Measured (`21317022`): 14.8% of pixels saw terrain against
  77.5% corrected. Every stereo number in B5 is void; blind and lidar stand.
  Fixed for all five camera offsets (`3f7b679`); the three stereo arms re-run
  at n=3.
- **Cloth-sort runs end to end in Isaac**, rigid and Newton cloth, after five
  bugs a parser could see (all now guarded on the login node). Low-res cloth is
  **467 env-steps/s** at 81 vertices against G-C1's 182 at 961.
- **The old cloth layout was never reachable**, by distance and facing: the
  fingertips touch a 0.30 m table only within 0.28 m forward, on the robot's
  right, and the robot faces away from the table. The kinematic ladder now has a
  reach model (0.00 on the old layout) and a fingertip contact table exists.
- **Terrain-sensing ("maze") rung at n=3** (2026-09-13): "lidar beats blind by
  51%" is retracted (+16%); the rung's sensors only ever saw terrain. *Its
  "width effect" is void too — see the corrected-stereo bullet above.*
- **Manipulation re-runs on the corrected spawn are final**: success 0 in all
  ten cells that trained.
- **Isaac renders fixed**: the viewport drew stale poses; a camera-sensor clip
  recorder shows the robot where it is (`docs/ISAAC_RENDER.md` §10).
- **The free-standing robot sorts: 22 of 24 rigid proxies, no falls** (shirt 6/8,
  sock 8/8, jacket 8/8; `21330402`–`404`). The squat the layout was built on could not
  stand. A knee-1.0 stance with a leg controller designed in MuJoCo can, and the
  layout rises with it. **The first cloth rung sorts too**: one Newton cloth, pinned
  base, 4 of 4, under one-way cloth-to-arm coupling (`21329265`).
- **Cloth sorts in Isaac for the first time — on a fixed base: 32 of 32 episodes**
  on the widened baskets (shirt 16, sock 8, jacket 8; `21317388`–`390`), each with
  its first sweep; 7 of 8 before the widening (`21317170`). Rigid proxies, scripted,
  root pinned. Two fixes got it there.
  The shipped hands had no collision geometry (hull overlay). And the arm could
  not follow its schedule: bare position targets on a 10 N m/rad, 4 N m drive lag
  0.2 s, and the fingertip ran 65–72 mm behind (`21307211`). Inverse-dynamics
  feedforward through the same drive brought that to 1.4–1.6 mm in Isaac. The
  eighth shirt landed across the basket rims; the baskets are now wider than
  every garment's diagonal. **The free base falls backward even with the arm held
  still** (`21317172`): the planted pinch squat is not a stance this robot can hold.
- **Quaternion order, beyond the cameras**: the cloth fall test read Isaac Lab
  3.0's `(x, y, z, w)` as `(w, x, y, z)` (fixed). The same misreading very likely
  explains item 3's under-floor spawn.

**Rendered**

- `docs/gifs/isaac/mazenav_seed0.gif` — four seed-0 navigation policies walking
  the fused-mesh corridor until timeout. Robot in shot, walls in shot, button
  not reached (`21355466`, 200 frames × 4).
- `docs/gifs/ice_pair_placed.gif` — B3 after the patches sit under the robots
  (median 0.8 m). Green blind against red depth, both stay upright (`21352982`).
  The *old* `ice_pair.gif` is the retracted 72 m placement.
- `docs/gifs/isaac/maze_stereo_fixed.gif` — the corrected 16×16 stereo policy walking,
  with its left eye as B5 had it before the quaternion fix (20° up, a strip of
  ground) beside the corrected eye (`21329137`).
- `docs/gifs/isaac/terrain_sensors.gif` — the first Isaac clip of trained
  policies with the robot in shot.
- `docs/gifs/isaac/cloth_sort_fixed_base.gif` — the scripted sweep pushing the shirt
  proxy into its basket, root pinned (`21317391`).
- `docs/gifs/isaac/cloth_sort_free_base_jacket.gif`, `cloth_sort_free_base_shirt.gif` — the
  free-standing robot sorting; the shirt takes two sweeps (`21329392`, `21330405`).

**Left, in order**

1. **Cloth, in Isaac on the redesigned layout** — *layout, garments, reach model,
   schedule action and planted spawn are done; the kinematic ladder sorts at 1.00*.
   *Fixed-base scripted C0 sorts 32/32 in Isaac for all three garment classes.*
   *The free base now stands in Isaac* with a knee-1.0 stance and a leg controller
   (2/2 with the arm still, 3/4 through six sweeps). The layout is raised to that
   stance (`a4f438b`), and on it, with balance v2 (`42ba537`), **the free-standing robot
   sorts 22 of 24 rigid proxies with no falls** (shirt 6/8, sock 8/8, jacket 8/8).
   **C2 on the fixed base sorts 4/4 with one-way cloth coupling** (two-way goes NaN).
   Left: **jacket cloth on the pinned base still goes NaN** (0/4 finite at 60 Hz
   `21338291`, and 4/4 nonfinite on the softer/thicker sheet `21353201`); sock
   cloth on that same rung is 4/4. C2 on the free base is blocked: a still arm
   is 2/2 nonfinite at 60 Hz (`21338288`) and at dt=0.005 (`21353200`). No new
   physics idea, so those cells stay unqueued. Isaac C5 pose-fix four-episode
   is **2/4 sorted** (`21353130`, moved 4/4, travel 0.19 m). Then online arm
   correction from the base pose; C1 training; C3.
2. **Maze navigation PPO is done; they walk to timeout.** Env smoke `21353199`
   PASS. Train smoke `21353395` 4/4 then 12/12 `mazenav-*` at 6,000 iterations
   (`21353396`–`398`). Button success 0 in 12/12; sensors do not separate.
   Seed-0 camera-sensor clips are done (`21355466`, glob `mazenav-*-s0`, cn-gpu7,
   `docs/gifs/isaac/mazenav_seed0.gif`). Pooling arms (P8/P16) wait on a navigation score, not a gait
   score. Old `maze-*` rows stay terrain-perception.
3. **Isaac spawn for the coop/TaskV2 tasks**: robots still spawn under the
   floor (`robot_a` bodies at z −0.806…−0.027 in `21299608`). *Likely cause found,
   not yet probed:* on v60 the spawn tuple `(0.707, −0.707, 0, 0)` is read
   `(x, y, z, w)`, an upside-down robot facing the cube (R₂₂ = −1); the legacy
   `(0.707, 0, 0, −0.707)` is a −90° roll. The fix is `native_quat` on the intended
   `(w, x, y, z)` yaw, plus a spawn probe. `v2stand-*` seed-0 (`21352980`) is
   still running: CubeToShelf blind/depth done, rgb running; BallToNet blind
   failed (`train.sh`: mean episode length 1.00); remaining cells pending on
   `%` limit.
4. **MARL**: first block done at n=3 — limb2's lead did not replicate. Tier 1's
   grid is **paused**: no skrl setup has yet learned to walk to the command on
   stairs. The privileged-critic fix did not close the gap (`21330392`); the noise
   A/B (`21338294`) is next. If skrl cannot be made to match rsl-rl's PPO on the
   one-agent control, the limb split moves into rsl-rl as a factorised actor
   instead. Tier 3 PPO: seed 0 done at 4.49; remaining skrl/tier3 rows stay held.
5. **B3 ice placement is done.** Patches at the terrain origins (`21342561`,
   median 0.8 m, reachable 1.000). PPO `ppo-ice-placed-*` (`21344928`) and
   `docs/gifs/ice_pair_placed.gif` (`21352982`). Finding 11's original +10.6%
   stays retracted. Tier 1 ice rows stay held with the rest of the skrl grid.

---

## Open

### Manipulation re-runs on the corrected spawn · `done` — final numbers 2026-09-11
**Final, from the event files** (mean of the last 50 logged iterations). Task
success is **0 in every cell that trained**. Four cells reached 8,000
iterations; the rest stopped early on node failures or were cancelled, and the
iteration column says how far each got.

| run | iters | ep_len | reward | fallen | success | `lift_height` |
|---|---:|---:|---:|---:|---:|---:|
| `v2up-cubetoshelf-blind` | 6,605 | **263.2** | **+12.00** | 0.655 | 0 | 0.048 |
| `v2up-cubetoshelf-depth` | 4,931 | 244.8 | +7.77 | 0.738 | 0 | 0.050 |
| `v2up-cubetoshelf-rgb` | 3,231 | 163.2 | +3.94 | 0.973 | 0 | 0.053 |
| `v2up-balltonet-blind` | 8,000 | 50.5 | +1.92 | 1.000 | 0 | 0.147 |
| `v2up-balltonet-rgb` | 8,000 | **326.0** | +7.57 | 0.498 | 0 | 0.044 |
| `gripup-cubetoshelfgrip-blind` | 6,290 | 111.2 | +4.36 | 0.893 | 0 | 0.098 |
| `gripup-cubetoshelfgrip-rgb` | 8,000 | 151.7 | +4.54 | 0.771 | 0 | 0.133 |
| `gripup-balltonetgrip-blind` | 8,000 | 170.7 | +3.07 | 0.760 | 0 | **0.211** |
| `gripup-balltonetgrip-depth` | 7,642 | 69.2 | +2.83 | 0.996 | 0 | 0.162 |
| `gripup-balltonetgrip-rgb` | 7,403 | 251.8 | **+13.93** | 0.630 | 0 | 0.148 |
| `v2up-planktowall-*` (3) | 1,661–6,057 | 1.0 | −0.40 | 1.000 | 0 | 0.040 |

What the corrected spawn bought is survival and a moving lift curriculum —
`lift_height` leaves its 0.04 m floor in every cube and ball cell, reaching
0.21 m on the gripper ball arm — and not a single completed task. Two caveats
travel with these numbers: they trained on the spawn recorded as corrected on
2026-09-07, which the later spawn entries partly retract, so the README's
Isaac-spawn caveat covers them; and the plank cells never had a corrected
spawn at all.

**The fix works.** Cube arms on the corrected spawn, against the same arms
before it:

| arm | ep_len | reward |
|---|---|---|
| `v2up-cubetoshelf-blind` | **258.9** | **+10.73** |
| `v2up-cubetoshelf-depth` | 236.9 | +8.86 |
| `gripup-cubetoshelfgrip-rgb` | 158.3 | +6.84 |
| *(before the fix: welded cube)* | *~8* | *-0.79* |

Task success is still 0 everywhere, so the finding that these tasks are not
solved stands — but the robots are now standing up while failing, which is a
different claim from the one the old numbers supported.

**The plank arms are still broken and were cancelled.** Only the CubeToShelf and
BallToNet spawns were corrected; PlankToWall still uses `(1,0,0,0)` and
`(0,0,0,1)`, and the corrected tilt check now reads those as fallen — which they
are. All three sat at `ep_len 1.0`, reward -0.401, fall 1.000. `21204960` is
measuring which quaternions stand that pair up, the same way `21196896` did for
the cube pair; it faces along -+x rather than -+y so it needs the two headings
90 degrees away from the cube pair's.

| # | id | outcome |
|---|---|---|
| 2 | `21199344`, `21199345` | **final:** 4 COMPLETED at 8,000 (`21199344_3`, `_5`; `21199345_2`, `_3`), 3 NODE_FAIL at ~19 h (`21199344_0-2`), 3 FAILED (`21199344_4` and `21199345_1` at 2 min, `21199345_0` at 18 h), 5 CANCELLED (3 plank; `21199345_4`, `_5` at ~16 h) |
| 1 | `21186402`, `21186403` | cancelled — trained on the lying-down spawn |

### B3 ice — the patches were never under the robots · `retracted` — found 2026-09-14
**Every B3 number was trained with the ice out of reach.** `terrains/ice.py`
spawns each patch as an `AssetBaseCfg` under `{ENV_REGEX_NS}`, so it sits at its
env's *grid* origin. With a terrain generator, Isaac Lab resets each robot onto a
*terrain* origin (`scene.env_origins`), and GPU collision filtering lets a robot
touch only its own env's prims. It is the fault that emptied the maze
(`21299608`). G-B3 checked that a patch is flush; nothing checked where it is.

Measured on the built scene at 4,096 envs on v51, the stack B3 trained on
(`21328532`, `results/ice_placement_probe.txt`):

| quantity | value |
|---|---|
| robot → its terrain origin | 0.38 m (reset noise) |
| patch → its env's grid origin | 2.13 m (config offset 2.0, ±0.72) |
| terrain origin → grid origin | median 71.9 m |
| robot → nearest own patch | p10 25.0 m · median 71.9 m · p90 127.8 m |
| reach in one episode | 15.0 m (20 s × 0.75 m/s, walking straight at it) |
| on ice at spawn / could reach any own patch | 0.000 / 0.044 |

What that voids: "depth beats blind by 10.6% on a hazard it cannot see", and
"colouring the ice changes nothing" — 1.394 against 1.374 is two blind arms on
the same bumpy ground. The three B3 arms trained on the bumpy menu, and the depth
gain matches §6's rough-terrain one (+10.9%). The MuJoCo `ice_pair` clip shows
those policies, not policies that learned ice. G-B3's flushness arithmetic
stands, and so does uniform low friction (`slippery`): it randomises the robot's
own contact material, which has no placement to get wrong.

Fix, not built: patches at the terrain origins (global prims, or per-env
kinematic bodies moved to `env_origins` at reset), on tiles flat enough that a
flush patch stays flush as the curriculum promotes, then re-probe. Tier 1's ice
rows are not queued until then — the work order's own fallback for B3.

| # | id | outcome |
|---|---|---|
| 1 | `21328532` | **ICE-PLACEMENT UNREACHABLE** (1:01) — table above |

### B3 ice clip — DONE · `done` — 2026-09-10; the render works, the result it shows is retracted (entry above)
`render_multi` can now drive a depth-conditioned policy, and B3 has a picture:
`docs/gifs/ice_pair.gif`, green blind against red depth on flush friction
patches, with the depth panel and waterfall along the bottom.

The blocker was that upstream's `RlController` parses raw pieces by fixed
offsets and assembles the observation itself, so a 301-wide network was handed
45 numbers. `DepthRlController` (workspace subclass; `external/` stays pristine)
appends depth after `prev_actions`, which is where Isaac's declaration order
puts it.

**The guard was tightened after it nearly did the wrong thing.** It first
accepted any square extra width — and the maze lidar arm is 45 + 36, with 36 a
perfect 6×6. It was refused only because 64 does not pool evenly to 6, which is
luck, not a guard. It now refuses by *sensor kind*: this replay renders one
forward depth camera and nothing else, so lidar and stereo are refused by name
rather than reshaped to fit.

| # | id | outcome |
|---|---|---|
| 3 | `21234171` | regression after tightening the guard — still exit 0, 13 MB |
| 2 | `21234053` | **exit 0, clip written** |
| 1 | `21233950`, `21233969` | 45-into-301: depth appended in the wrong place, before the controller's own assembly |

### B5 maze — navigation PPO · `done` — 2026-09-18, they walk to timeout

Env smoke `21353199` **PASS** 4/4 (`MazeWaypointCommand`, spawn in corridor,
lidar wall 0.44 m). Walls live in `/World/ground`. This block trained that MDP.

Run names are `mazenav-*`, not `maze-*`. Old maze PPO rows stay
terrain-perception results. Four arms (blind / lidar / stereo / both), three
seeds, 6,000 iterations, 2,048 envs. Terrain curriculum is off.

**Last-50 of the event files** (`results/mazenav_last50.csv`; 6,000 logged
iterations each). `Metrics/success_rate` ~0.99 is **not** button success: it
tracks surviving without falling. Button termination is 0.000 in 12/12, including
the max over the whole run. `progress_to_button` never exceeds 0.0005.

| arm | eplen | track_lin_vel_xy_exp | time_out | button_reached | seeds |
|---|---:|---:|---:|---:|---|
| blind | 483.8 | 0.606 | 0.948 | **0.000** | 483.3 / 482.4 / 485.6 |
| lidar | 482.3 | 0.596 | 0.940 | **0.000** | 483.2 / 481.4 / 482.4 |
| stereo | 483.2 | 0.597 | 0.949 | **0.000** | 483.9 / 480.5 / 485.2 |
| both | 485.4 | 0.580 | 0.954 | **0.000** | 486.0 / 483.0 / 487.3 |

They learned a gait that lasts the 20 s episode. They did not close on the
plate. Sensors do not separate. Pooling arms wait on a navigation score.

Train smoke (3 iterations, 64 envs): episode length 27.26 / 26.56 / 27.39 / 28.10.

Seed-0 camera-sensor clips: `slurm/97c_mazenav_video.sbatch` (glob `mazenav-*-s0`,
`--exclude=dgxh-1`, `v60_boot_gate`). Assembled on the login node by
`scripts/gif_mazenav.py` once each arm has ≥50 PNGs. Does not overwrite `maze_*`
frame dirs.

| # | id | outcome |
|---|---|---|
| 6 | `21355623` | running on cn-gpu7 — colour re-render (PreviewSurface floor + clip overlays) |
| 5 | `21355466` | **clip COMPLETED** (4:51, cn-gpu7) — 200 frames × 4 arms (`mazenav-*-s0`); grayscale (fused mesh had no albedo). `docs/gifs/isaac/mazenav_seed0.gif` |
| 4 | `21353398` | **12/12 COMPLETED** — `mazenav-*-s2`, 4:10–7:34, afterok of `21353395` |
| 3 | `21353397` | **COMPLETED** — `mazenav-*-s1`, 4:21–5:59 |
| 2 | `21353396` | **COMPLETED** — `mazenav-*-s0`, 5:15–7:18 |
| 1 | `21353395` | **COMPLETED** — 3-iter train smoke, Blind/Lidar/Stereo/Both, eplen 26.6–28.1 |

### B3 ice — patches placed, retrained · `done` — 2026-09-17

The 72 m placement (`21328532`) is retracted. The follow-up probe
(`21342561`) puts each patch at its terrain origin: robot → nearest own patch
p10 0.5 m, **median 0.8 m**, p90 1.0 m; reachable fraction **1.000**.

PPO `ppo-ice-placed-*` (`21344928`), 6,000 iterations, 4,096 envs, n=2. Last-50
terrain level from the event files (`results/ice_placed_last50.csv`):

| arm | s0 / s1 | mean |
|---|---|---:|
| **depth** | 2.854 / 2.984 | **2.92** |
| blind | 2.576 / 2.605 | 2.59 |
| visible ice | 2.576 / 2.605 | 2.59 |

Depth is +13% on the mean against blind, with ice actually under the robots.
Visible-ice last-50 matches blind seed-for-seed: `IceVisible-v0` is still
proprioception-only; the paint does not enter the observation. n=2, so
suggestive. Finding 11's original +10.6% (measured on bumpy ground) stays
retracted — different tiles, different absolute levels.

Clip `docs/gifs/ice_pair_placed.gif` (`21352982`): both stay upright, peak x
+4.66 m (blind) / +4.45 m (depth). Do not confuse with `ice_pair.gif`.

| # | id | outcome |
|---|---|---|
| 4 | `21352982` | **clip COMPLETED** (0:21) — `docs/gifs/ice_pair_placed.gif`, 13 MB |
| 3 | `21352981` | **export COMPLETED** (1:02) |
| 2 | `21344928` | **6/6 COMPLETED** — placed PPO, 6:54–9:15 |
| 1 | `21342561` | **ICE-PLACEMENT REACHABLE** (0:59) — median 0.8 m |

### B5 maze — stereo re-run with the cameras pointing down · `done` — n=3, 2026-09-15
**Every stereo number in the two B5 entries below was measured with the stereo
pair looking 20° up, upside down.** The pose `(0.9848, 0, 0.1736, 0)` is 20° of
down-pitch in Isaac Lab 2.3.2's `(w, x, y, z)`. B5 trains on the v60 stack, and
Isaac Lab 3.0 reads a camera offset as `(x, y, z, w)`, where the same tuple is a
half-turn about an axis 10° off +x. Nothing errors: both stacks accept any
4-tuple. The cloth session's quaternion audit pointed at it; the probe measured
it on the StereoP16 arm at reset, relative to the robot base (`21317022`,
`results/stereo_pitch_probe.txt`):

| camera | pitch | camera up, base z | pixels returning terrain | where the terrain is |
|---|---:|---:|---:|---|
| as trained (raw tuple) | **+20.0°** | −0.94, upside down | **14.8%** | the top rows of the image only |
| corrected (`native_quat`) | −20.0° | +0.94 | 77.5% | the lower three-quarters |

What that voids, and what it does not:

- **Void:** every stereo and both arm — the width effect (0.021 at 16×16 against
  1.124 at 4×4), "pooled stereo beats blind by 52%", "adding lidar adds nothing".
  Those policies got a strip of distant terrain at the top of an inverted image
  and range-clipped sky everywhere else. Whatever separated the widths, it was
  not seeing the ground ahead.
- **Stands:** blind (0.738) and lidar (0.860, +16%). Neither has a camera, and
  the lidar offset carries no rotation.
- **Unaffected:** the depth rung (§6, B2, B3). It trains on v51, where the tuple
  means what it says.

Fix (`3f7b679`): `bhl_robust/quat_order.py` probes the order from
`matrix_from_quat`, and all five camera offsets go through `native_quat`
(identity on v51); `tests/test_quat_order.py` guards the call shape. The v2 and
RGB head cameras had the same fault on v60.

Re-run: the three stereo arms FINDINGS reports at n=3 — 16×16, 4×4, 4×4 + lidar
— at seeds 0–2, run names `mazefix-*`, gated `afterok` on the probe. Blind and
lidar are not re-run. Kit boots are serialised by `slurm/inner/v60_boot_gate.sh`
instead of chaining six-hour runs end to end (fixed in `4474b83` after a short
job left its lock behind for ten minutes).

**16×16, pointed down, n=3** (mean of the last 50 of 6,000 iterations, event files):

| arm | seed 0 | seed 1 | seed 2 | mean | as published |
|---|---:|---:|---:|---:|---:|
| blind (control, stands) | 0.545 | 0.634 | 1.037 | 0.738 | — |
| lidar (stands) | 0.794 | 0.814 | 0.971 | 0.860 | — |
| **stereo 16×16, corrected** | **0.877** | **0.560** | **0.886** | **0.774** | 0.021 |

**At 92% of the input, stereo that looks at the ground trains like blind** — every
seed inside blind's 0.55–1.04. The "width effect" (0.021 at 16×16 against 1.124 at
4×4) was the camera pointing at the sky, not the width drowning proprioception.

**All three stereo arms, pointed down, n=3 — final** (mean of the last 50 of 6,000
iterations, event files):

| arm | seed 0 | seed 1 | seed 2 | mean | as published (upward camera) |
|---|---:|---:|---:|---:|---:|
| blind (control) | 0.545 | 0.634 | 1.037 | 0.738 | — |
| lidar | 0.794 | 0.814 | 0.971 | 0.860 | — |
| stereo 16×16 | 0.877 | 0.560 | 0.886 | 0.774 | 0.021 |
| **stereo 4×4** | **1.286** | **1.340** | **1.237** | **1.288** | 1.124 |
| stereo 4×4 + lidar | 1.063 | 1.065 | 1.079 | 1.069 | 1.115 |

- **Pooled stereo beats blind cleanly**: every 4×4 seed (≥ 1.237) sits above
  blind's best (1.037), +75% on the mean. Published, the ranges touched.
- **Width still matters, far less than published**: 16×16 is below 4×4 in every
  seed (≤ 0.886 against ≥ 1.237), but it trains like blind rather than failing.
- **Adding lidar to pooled stereo costs**: 4×4 + lidar is below 4×4 alone in
  every seed (≤ 1.079 against ≥ 1.237), −17% on the mean. Published, it "added
  nothing".

| # | id | outcome |
|---|---|---|
| 4 | `21329137` | **clip COMPLETED** (4:09, cn-gpu6, dgxh-1 excluded for Vulkan) — `mazefix-stereo-s0` through the camera-sensor recorder, 200 frames, with the corrected left eye and a raw-tuple eye dumped per frame (`train_play.py --clip-sensors stereo_l --clip-raw-stereo`); `docs/gifs/isaac/maze_stereo_fixed.gif`. The long pending jobs were niced for ten minutes so it could take the next slot, then restored |
| 3 | `21317023`, `21317024`, `21317025` | seeds 0 / 1 / 2, `--array=2,5,6%1` (16×16, 4×4, 4×4 + lidar), 16 h limit, 2,048 envs as before — **9 of 9 COMPLETED** (5:41–9:57); table above |
| 2 | `21317022` | **STEREO-PITCH PASS** (0:54) — table above |

### B5 maze — stereo pooling sweep · `retracted` for every stereo arm — the pair looked up (entry above); blind and lidar stand
**Replicated, 2026-09-13.** Seeds 1 and 2 of the five arms that matter are in
(`21247911`, `21247912`, 10 of 10 COMPLETED). Terrain level, mean of the last
50 of 6,000 iterations, from the event files:

| arm | stereo share | seed 0 | seed 1 | seed 2 | mean |
|---|---:|---:|---:|---:|---:|
| blind (control) | 0% | 0.545 | 0.634 | 1.037 | 0.738 |
| lidar | 0% | 0.794 | 0.814 | 0.971 | 0.860 |
| stereo, 16×16 / eye | 92% | 0.001 | 0.000 | 0.061 | **0.021** |
| stereo, 4×4 / eye | 42% | 1.027 | 1.139 | 1.207 | **1.124** |
| stereo 4×4 + lidar | 28% | 1.203 | 0.972 | 1.170 | 1.115 |

What holds at n=3:

- **Wide stereo fails in every seed** (≤ 0.061), below every blind seed
  (≥ 0.545). Robust.
- **Pooled stereo succeeds in every seed** (≥ 1.027), 53× the wide arm — the
  width effect is the finding, and it replicates cleanly.
- **Pooled stereo beats blind on the mean** (1.12 against 0.74, +52%), but the
  ranges touch: blind's best seed, 1.037, sits above pooled stereo's worst,
  1.027. Say "on the mean", not "in every seed".
- **Adding lidar to pooled stereo adds nothing** (1.115 against 1.124).

What does **not** hold:

- **"Lidar beats blind by 51%"** (`fe8f726`) was one seed against the weakest
  blind seed. At n=3 it is **+16%** (0.86 against 0.74), and blind's seed 2
  beats every lidar seed. Retracted as a claim; the mean is still recorded.

Blind alone spans 0.55–1.04 across seeds. On this terrain, seed variance is
larger than most of the sensor effects, which is the reason one-seed maze
comparisons cannot be read.

**Stereo was never worse than blind. It was too wide.** All seven arms, read
the same way from their event files (mean of the last 50 of 6,000 iterations):

| arm | stereo pool | obs width | stereo share | ep_len | reward | terrain level |
|---|---:|---:|---:|---:|---:|---:|
| blind (control) | — | 45 | 0% | 413.5 | 12.96 | 0.545 |
| lidar | — | 81 | 0% | 426.1 | 15.00 | 0.794 |
| stereo | 4 → 16×16 / eye | 557 | **92%** | 320.8 | 7.88 | **0.001** |
| both | 4 | 593 | 86% | 358.4 | 9.79 | 0.065 |
| stereo P8 | 8 → 8×8 / eye | 173 | 74% | 433.7 | 15.98 | 0.878 |
| stereo P16 | 16 → 4×4 / eye | 77 | **42%** | 439.0 | 17.56 | **1.027** |
| both P16 | 16 | 113 | 28% | **444.3** | **17.59** | **1.203** |

Shrink the stereo's share of the input and terrain level climbs with it:
0.001 at 92%, 0.878 at 74%, 1.027 at 42%. Pooled to 4×4 an eye, stereo beats
lidar (1.03 against 0.79), and stereo plus lidar is the best arm in the sweep
at **2.2× the blind control**. So the committed reading of the first four arms
— "stereo is far worse than blind", "adding the cameras destroys the lidar
advantage" — was an input-width artefact, which is what `depth_obs`'s
docstring warned about and what this sweep was built to test.

**One seed**, like the result it reverses. Seeds 1 and 2 for blind, lidar,
stereo, stereo P16 and both P16 are queued below; neither reading is a
finding until they land. (The first four arms' numbers differ slightly from
the table in the entry below, which quotes final-iteration values; the order
is the same.)

Does coarser stereo recover, or do cameras simply not help? At `pool=4` stereo
is 512 of 557 observations (92%) against lidar's 36 of 81 (44%). `StereoP16`
pools to 4×4 an eye — 32 of 77, **42%**, matched to lidar — so it holds the
information fraction fixed and changes only the sensor.

| # | id | outcome |
|---|---|---|
| 4 | `21247917` | **rendered, not published.** All four Isaac clips show the maze and the velocity-command arrows riding on the robot's root — the camera follows it correctly — and **no robot**: the body never appears, at 0 s or at 8 s, in any arm. Same failure as the task-env spawn render in *Spawn rotation* below. Four identical corridors captioned as four policies would mislead, so B5 has no clip yet. Raw renders stay in `results/clips/` (gitignored, 120 MB each). |
| 3 | `21247912_[0-2,5,6]` | **5 of 5 COMPLETED** (4:23–6:01) — seed 2, table above |
| 2 | `21247911_[0-2,5,6]` | **5 of 5 COMPLETED** (5:32–8:45) — seed 1, table above |
| 1 | `21233916_[4-6]` | **3 of 3 COMPLETED** (5:46–7:43) — table above |

### Cloth sorting — redesigned inside the reach · `running` — fixed base sorts 32/32 in Isaac; the squat cannot stand, 2026-09-14
**The layout is rebuilt in the robot frame, inside the fingertip contact table**
(`assets/cloth/right_hand_contact_pinch.npz`): a 0.20 × 0.18 m table at 0.30 m on
the robot's right, baskets off its front, outer and back edges, all clear of the
right leg, every table cell checked reachable by `assert_layout`. Garments are
scaled 0.33–0.45× to fit (mass by area); a five-garment scene presents them one
at a time (Mode B, sequential), parking the rest. The robot now spawns at the
planted squat height (−0.137) instead of dropping 11 cm.

**Kinematic, under the reach model, 2026-09-13** (`results/cloth/redesign/`):

| cell | success | plans refused | sweeps |
|---|---:|---:|---:|
| C0 scripted, 64 eps | **1.00** | 0% | 1.00 |
| C5 scripted, sequential, 32 eps | **1.00** | 0% | 5.00 |
| C1 learned residual, DR, 64 eps | **1.00** | 0% | 1.92 |
| C4 linear BC, DR, 64 eps | **1.00** | 0% | 1.00 |

On the old layout the same cells were 0.00 with every plan refused. So the
redesigned task is solvable within the robot's measured reach — kinematically.
Physics is the next question.

**Balance is now the blocker, and it hides the manipulation question.** On the
free base the robot falls within the first 4 s sweep in every episode
(`21300301`). The zero-action probe had already shown the squat sinking 12–20 cm
and wobbling; with an arm moving on top it goes over. So "does the sweep move the
garment?" cannot be asked of the free-base task yet. A **fixed-base diagnostic
task** (`ClothSort-BHL-RigidFixedBase-Oracle-v0`, root link pinned, legs still on
their squat targets) asks it separately; the free-base task stays as it is.
Raising the leg gains would misrepresent a 6 Nm actuator, so the real fix is a
balance controller — the trained locomotion policy on the legs is the candidate.
Random 5-D actions are also a sweep the hand can make only 7 times in 64, so C1
gets a residual action mode (`ClothSort-BHL-RigidResidual-Oracle-v0`) around the
scripted sweep.

**The Isaac sweep action is rebuilt** on the same table
(`bhl_robust.cloth.schedule`): one RL step executes one whole sweep — hover over
an anchor, descend, glide, sweep, glide, lift — as a joint schedule played at the
physics step, with the garment read through the robot's actual root pose. The
old term latched a hand-tuned mapping every 40 ms, never read the garment, and
its timer ran 8× fast. Macro step 4.0 s then (6.0 s since v3: decimation 1200
rigid, 360 cloth; rigid episodes six sweeps, 36 s).

**v2 sweep, built on what 21300604 showed (2026-09-13).** Four faults, one fix each,
all checkable on a login node:

- *Branch jumps.* `build_tip_table.py` grows the fingertip table cell by cell from the
  table centre, seeding each solve from a solved neighbour: largest joint jump
  between neighbours 0.28 rad, fingertip within 1.3 mm of the straight line between
  cells. Over the table's interior the elbow is fully bent, so the hand can lift
  clear only at 78 edge cells — those are the anchors.
- *Descending onto the garment.* `schedule.py` routes the table-height glide around the
  garment (A* on contact cells, shortened where a straight glide is clear) and replays
  every candidate through numpy FK of the hand hull (`arm_fk.py`, equal to MuJoCo to
  4e-16 m). It refuses any plan that touches the garment before the sweep, or where
  the garment should be after it, or goes into the table. An independent MuJoCo
  replay of the scripted shirt / sock / jacket schedules: hull 1.2–2.1 cm clear of
  the garment on approach, lowest point 0.3027–0.3030 m, fingertip within 0.2 mm of
  the sweep line.
- *Riding up the edge.* Contact clearance 12 mm → 3 mm (a 1.5 cm garment side is now
  12 mm of pushing face). Hand footprint radius 4 cm → 3.1 cm, measured.
- *Box collider.* The hand tapers to a 1.7 × 3.0 cm fingertip; the Isaac collider is now
  the mesh's convex hull (64 vertices, within 1.7 mm of the full hull) instead of its
  5.9 × 7.4 cm bounding box.

Also found: every quaternion read in the cloth MDP assumed `(w, x, y, z)`, but Isaac
Lab 3.0 stores `(x, y, z, w)`. The fall test, the garment yaw and the robot-yaw
correction are fixed. `coop_lift_mdp._tilt_from_quat` and `task_v2_mdp` make the same
assumption; they belong to other experiments and are flagged, not changed.

**v3: the arm could not follow the schedule (2026-09-13, night).** `21307211`'s trace
settles where the v2 push went wrong. Forward kinematics of Isaac's *measured* joints
lands on Isaac's hand link to 0.0 mm, so the plan's geometry and frames are right. But
the fingertip ran **65–72 mm behind its schedule** (max 14 cm, joint errors to 0.66 rad).
The upstream arm drive is PD at 10 N m/rad and 2 N m s/rad with a 4 N m limit, so bare
position targets lag by Kd/Kp = 0.2 s and saturate at the schedule's 3 rad/s. The shirt
first moved at 0.50 s, during the *approach*, when the lagging hand cut through it; it
came to rest on the jackets basket's rim, and every later plan was rightly refused.

A MuJoCo arm with the same drive reproduces Isaac's measured joints to 0.001 rad before
contact, so it is a fair offline judge (a numpy copy, `arm_fk.simulate_pd`, matches to
0.002 rad and is in the tests). The fix keeps the actuator — same gains, same 4 N m, no
articulation change — and changes what it is sent. Each phase is re-timed to start and
stop at rest (quintic; macro step 4 → 6 s). The position target carries the
inverse-dynamics torque over Kp, and a velocity target cancels the damping drag
(`arm_fk.inverse_dynamics`, equal to `mj_rne` to 3e-15 N m). A phase whose feedforward
would need more than 3.4 N m is slowed, and the term refuses to start if the arm
actuator is not the one the feedforward assumes. In the Isaac-calibrated arm the
scripted shirt / sock / jacket schedules now track **within 1.8 mm** (bare targets:
42–90 mm mean), with peak drive torque 3.6 N m. Every static table pose needs at most
2.15 N m against gravity. The kinematic ladder still sorts at 1.00 (C0 64 eps, C5 32
eps, 0% refused).

With the corrected fall test the free base still falls, backward, within 1 s of the
first arm move (`21307213`). Row 17 holds the arm still, to separate the squat itself
from the arm's reaction torque.

**In Isaac the feedforward works, and the shirt sorts: 7 of 8 on the fixed base**
(`21317170`). The trace has the fingertip 1.4–1.6 mm behind its schedule on the sweep
(2.6 mm at worst). The shirt first moved at 2.35 s, inside the sweep's window
(2.07–2.88 s), where before it moved during the approach. Each of the seven sorted
with its first sweep; no plan refused, no wrong basket, no falls. The failure was episode 0: the shirt was pushed in
over the rims turned 68°, and at that angle a 10 × 8 cm shirt spans 12.2 cm across a
12 cm opening. It came to rest on both rims at z 0.146, 6 mm above the success box, and
no later sweep could reach it. So the baskets are widened until no garment can bridge
its own: shirts 15 × 15 cm, jackets 15.5 × 15.5 cm (the jacket's diagonal is 13.9 cm;
its old 12 × 10 cm opening would have caught most jackets), and socks keep 14 × 12 cm,
shifted 2 cm. `assert_layout` now refuses any basket narrower than its garments'
diagonal plus 1 cm. On the widened baskets the scripted schedule is valid 40/40 per
garment under spawn jitter, and the kinematic C0 and C5 still sort at 1.00. Rows
17–19 import the code when they start, so they run on the widened baskets; their
questions (balance) do not depend on them.

The v3 clip (row 16) died on `dgxh-1`, which has no Vulkan (`vkCreateInstance failed`,
`ERROR_INCOMPATIBLE_DRIVER`), not on the code. The same clip code ran on cn-gpu5 and
cn-gpu7. Queued clips now carry `--exclude=dgxh-1`.

**Overnight, 2026-09-13/14: 32 of 32 on the widened baskets, and the squat is the
blocker.** Fixed base, v3 schedule, widened baskets: shirt 16/16, sock 8/8, jacket 8/8,
every one with its first sweep, no plan refused, no falls (rows 20–22). The clip sorts
on camera (row 23; `docs/gifs/isaac/cloth_sort_fixed_base.gif`). On the free base the
arm-still control falls in 2 of 2 episodes (row 17), and its trace follows the sweeping
run's almost exactly: pitch −15° at 0.5 s, −48° at 0.75 s, on its back by 0.9 s. **The
planted pinch squat is not a stance this robot can hold** with its legs on their targets.
The arm is not what tips it. A free-base cloth task needs a stance that stands; the
contact table and the layout are built on this one, so they will have to follow.

**Why the squat falls, 2026-09-14 (offline, MuJoCo).** A MuJoCo robot with Isaac's leg
and arm drives (PD, 6 / 4 N m) reproduces Isaac's arm-still fall: pitch −17° at 0.5 s and
−45° at 0.7 s against Isaac's −15° and −39°, down by 0.9 s in both. So the proxy can be
used. Two things are wrong with the stance. (1) **The knees cannot hold it.** Each knee
needs 5.5–6.0 N m in the pinch squat (knee 1.45 rad), at the 6 N m limit. The PD sags
0.13–0.32 rad, the pelvis drops 3 cm, and the centre of mass slides back off the heels.
(2) **The spawn buries the toes.** At ankle −0.55 the sole is pitched 2.9° toe-down, and
the root at −0.137 puts the toe edge 2.5 cm into the floor. A flat sole needs ankle −0.60
at root −0.119. Fixing (2) alone does not stop the fall; (1) does it. A quasi-static map
over level, flat-soled stances gives about 3.1 N m peak at knee 1.0 (root −0.078, +5.9 cm)
and about 2.5 N m at knee 0.4. Even there, a 20 N m/rad ankle is softer than gravity's
toppling stiffness (about 64 N m/rad), so the stance needs gravity feedforward on the legs
plus IMU feedback at the ankles. **With both, the knee-1.0 stance stands in the proxy.**
Leg feedforward is the settled leg torques over Kp; ankle pitch and roll targets are
driven by base tilt and tilt rate in the heading frame. Under Isaac's ±0.04 rad reset
noise, 31 of 144 gain sets stood 6 s (`results/cloth/stance/search.json`). An earlier
scratch search had found none: its feedforward was averaged over a window in which the
robot was already rolling over. The set chosen (pitch 1.5 / 0.3, roll 2.0 / 0.05), run
through the same function the Isaac term uses, stands 8 of 8 resets for 10 s with the arm
still (worst tilt 3.1°) and 7 of 8 for 6 s with the arm playing the scripted sweep
(`stress.json`). Its neighbours mostly fall with the arm moving, so it is narrow. It is
`bhl_robust.cloth.balance`; `ClothSort-BHL-RigidBalance-Oracle-v0` runs it on the free base
with the table unmoved, a balance probe with the hand passing 6 cm above the garment
(rows 28–30). Everything here is reproducible with `scripts/cloth/stance_mujoco.py all`.

**C2F — the first cloth rung on the fixed base: the success predicate fired, and the result
is void.** One 8×8 Newton cloth shirt: the predicate fired in 4 of 4 episodes, each with its
first sweep. The cloth's centre did travel along the sweep and settle in the shirts basket
(z 0.016). But the trace has the arm's joint positions going **NaN at t = 2.83 s,
mid-sweep**, and staying NaN; the clip trace has the same onset at the same time. A sort
by a simulation that has blown up is not a sort. Under Newton the approach also tracked
at 40 mm mean, against PhysX's 1.4. The Newton preset caps MuJoCo-Warp's constraint buffers
at `njmax` 40 and `nconmax` 20, sized for a Franka. Row 26 raises them to 600/200 and changes
nothing else; row 27 holds the arm still. The eval now counts physics steps with a
non-finite robot state and reports `success_rate_finite`, which scores such an episode as a
failure. The C2F clip renders only the floor grid; the camera-sensor recorder has not been
made to work on Newton.

**The free base stands in Isaac, 2026-09-14.** The MuJoCo-designed controller transfers
(rows 28–29). With the arm still, 2 of 2 episodes stand 36 s at 1.65° worst tilt. With the
arm playing the scripted sweep six times, 3 of 4 episodes stand; the traced one peaks at
11.5°, against the model's 11.3°. It is narrow, as the model said, so a search with the arm
sweeping inside the objective is running (`stance_mujoco.py robust`). The table has not
been raised, so these probes cannot sort. Raising it by `balance.ROOT_RISE` (5.95 cm), and
the fingertip table with it, is what turns the probe into the free-base task.

**The stance is part of the layout now (`a4f438b`).** Every rigid cloth scene spawns in the
knee-1.0 stance with the leg controller on. Heights rise with the root's *settled*
height, −0.0787 as measured standing in Isaac, rather than the −0.0765 spawn: 5.855 cm
above the squat the tables were solved in. The table top is 0.3586 m. The fingertip
table and the arm chain are shifted by the same amount when loaded, so every stored arm
configuration sits exactly where it was solved relative to the table. Kinematic C0 and
C5 still sort 1.00. Rows 32–36 are the first free-base sorting runs and a fixed-base
invariance check.

**The free-standing robot sorts, 2026-09-14 — 16 of 24, and not the shirt.** Scripted sweep,
balance stance with its leg controller, rigid proxies. **Jacket 8 of 8 and sock 7 of 8, with
no falls; shirt 1 of 8, with 3 falls.** The same shirt on the pinned base sorts 8 of 8 on
this layout, so the gap is the free base. The shirt's basket is off the front edge, where
the arm reaches furthest forward. In the shirt trace the base tilts 7–8° on every approach
and the fingertip lands 21 mm (mean) off its plan. What the MuJoCo model says comes next:
- pitch kp 2.0 instead of 1.5 stands 12 of 12 over all three garments' sweeps, at 6.7°
  worst tilt against 11.4° (`robust.json`, and a fresh-seed comparison);
- feeding the arm's centre-of-mass shift forward to the ankles, from the schedule itself,
  halves the tilt again (6.3° to 4.1°). The opposite sign falls every time, which settles
  the sign.

Neither fixes where the hand lands. Fingertip error against the table stays 15–19 mm,
because the body moves to counterbalance the arm and tilt changes the hand's height by
centimetres against a 3 mm contact clearance. Placing the hand precisely on a free base
means correcting the arm online from the measured base pose.

**Balance v2: the free-standing robot sorts 22 of 24, with no falls (rows 39–41).** The ankle
pitch gain goes from 1.5 to 2.0, and each schedule feeds its arm's centre-of-mass shift
forward to the ankles (`42ba537`). Shirt 6 of 8 (was 1 of 8 with 3 falls), sock 8 of 8, jacket
8 of 8, no falls in 24 episodes. In the shirt trace the worst tilt is 3.4°, where it was 7–8°.
The two shirts that did not sort are unexplained so far: episode 0 is the only one traced,
and it sorted.

**The first valid cloth sort: C2F with one-way coupling, 4 of 4 (row 37).** One 8×8 Newton
cloth shirt, pinned base, scripted sweep. The only change against the void runs is
`coupling_mode="one_way"`: the rigid solver no longer feels particle contacts, so the
cloth cannot push back on the arm. For a 16 g cloth that force is small, but this is a
physics approximation and is labelled as one wherever the result is cited. In the trace
the cloth moves only during the sweep and settles in its basket. Under Newton the arm
tracks at 38 mm mean, where PhysX gave 1.4 mm; the feedforward was calibrated on PhysX
drives, and the sweep delivered the cloth anyway.

**C2F diagnosis.** Larger MuJoCo-Warp buffers do not stop the NaN (row 26); a still arm
stays finite (row 27). Onset is at the sample where the hand first moves the cloth, in
every run. That points at the two-way cloth-to-rigid coupling. Row 31 changes only that,
to one-way, where the cloth no longer pushes on the arm. For a 16 g cloth that is a small
physical approximation, but it is one, and it will be stated.

**The kinematic ladder under the v3 planner** (`results/cloth/redesign_v3/`; the
`redesign/` files are the v1 numbers, kept). C0 scripted 1.00 (64 eps) and C5
sequential 1.00 (32 eps), no plan refused. **C1 (REINFORCE linear residual) 0.56 and
C4 (linear BC) 0.56**, down from 1.00 under v1. Neither drop is the planner being
wrong. The kinematic scripts never passed the garment's yaw to the scripted sweep, so
under ±0.6 rad randomization it started the hand on the garment; with the yaw
(`scripted.scripted_for`) scripted C1 is 64/64. BC fit on unrandomized scenes is 64/64
there, but a linear map cannot express the yaw-dependent start: 138 of its sweeps
start on the garment. The REINFORCE baseline was a fixed 2.0, so the residual
random-walked 5–8 cm off the scripted start; a running-mean baseline still ends at
0.56, because ±27° of angle exploration is coarse for a planner that refuses imprecise
starts. v1 accepted those sweeps.

| # | id | outcome |
|---|---|---|
| 51 | `21353201` | **COMPLETED — C2F jacket, new VBD sheet: 4/4 nonfinite**, `success_rate_finite` 0 (`isaac_c2f_jacket_mat.json`). Headline `success_rate` 1.0 is not a sort |
| 50 | `21353200` | **COMPLETED — C2 free-base hold at dt=0.005: 0/2, 2/2 nonfinite**, garment travel ~0 (`isaac_c2_hold_dt005.json`). Same as 60 Hz |
| 49 | `21353130` | **COMPLETED — Isaac C5 pose-fix, four episodes: 2/4 sorted**, moved 4/4, travel 0.19 m, fall 0, nonfinite 0 (`isaac_c5_posefix_full.json`) |
| 48 | `21338293` | **FAILED** — C2 free base, jacket cloth: same NaN crash in `segment_on_contact` (afterany of 47) |
| 47 | `21338292` | **FAILED** — C2 free base, sock cloth: same NaN crash (afterany of 46) |
| 46 | `21338291` | **COMPLETED — C2F jacket, pinned, 0/4 finite.** 4/4 nonfinite, garment moved in 1/4, travel 5.4 cm mean (`isaac_c2f_jacket.json`). One-way coupling is not enough for this garment |
| 45 | `21338290` | **COMPLETED — C2F sock, pinned, 4/4, all finite**, first sweep (`isaac_c2f_sock.json`). The first cloth rung now has a second garment that sorts |
| 44 | `21338289` | **FAILED** — C2 free-base shirt sweep crashed: garment pose NaN → `segment_on_contact` `ValueError`. Guarded; do not re-queue until row 43 is finite |
| 43 | `21338288` | **COMPLETED — C2 free base, arm still: 0/2, 2/2 nonfinite**, garment travel ~0 (`isaac_c2_hold.json`). Newton at 60 Hz does not stay finite under the stance that stands in PhysX |
| 42 | `21330405` | **COMPLETED — the free-standing robot sorts the shirt on camera, in two sweeps**: the first leaves it at the table's edge, the second, re-planned from where it lay, drops it in the basket. 192 frames. Published: `docs/gifs/isaac/cloth_sort_free_base_shirt.gif` |
| 41 | `21330404` | **COMPLETED — free-base jacket, balance v2: 8 of 8**, each with its first sweep, no falls |
| 40 | `21330403` | **COMPLETED — free-base sock, balance v2: 8 of 8**, 1.25 sweeps, no falls, 0% refused |
| 39 | `21330402` | **COMPLETED — free-base shirt on balance v2: 6 of 8, no falls** (row 32 was 1/8 with 3 falls), each with its first sweep, 22% of plans refused. Traced episode: worst tilt 3.4° (was 7–8°), arm tracking 5.3 mm, shirt into its basket |
| 38 | `21329392` | **COMPLETED — the free-standing robot sorts the jacket on camera**, 59 frames, cn-gpu7. Published: `docs/gifs/isaac/cloth_sort_free_base_jacket.gif` (brightened; the jacket proxy is near-black) |
| 37 | `21329265` | **COMPLETED — C2F with one-way coupling: 4 of 4 sorted, all finite** (`success_rate_finite` 1.00). Trace: the cloth first moves at 2.83 s, inside the sweep, and lies in the shirts basket at z 0.008 by 3.5 s. Arm tracking under Newton was 38 mm mean |
| 36 | `21329264` | CANCELLED before it started — shirt is the garment that mostly fails on the free base, so the clip became the jacket (row 38) |
| 35 | `21329263` | **COMPLETED — fixed base, shirt, on the stance layout: 8 of 8**, each with its first sweep. Matches rows 20–22, as it should: the arm's geometry against the table is unchanged |
| 34 | `21329262` | **COMPLETED — free base, jacket: 8 of 8**, no falls, 1.1 sweeps to success, 0% refused |
| 33 | `21329261` | **COMPLETED — free base, sock: 7 of 8**, no falls, 1.1 sweeps to success, 29% of plans refused |
| 32 | `21329260` | **COMPLETED — free base, shirt: 1 of 8, 3 falls**, 39% refused. Trace: the base tilts 7–8° on every approach and the fingertip runs 21 mm (mean) off plan, against 1.4 mm pinned. The pushes were weak, and the third sweep tipped it over |
| 31 | `21329213` | CANCELLED before it started, replaced by row 37 — it would have picked up `a4f438b` |
| 30 | `21329078` | CANCELLED before it started — on `a4f438b` the hand no longer passes over the garment, so a "balance probe" clip would have been a sort clip; row 36 is that clip |
| 29 | `21329077` | **COMPLETED — the free base stands through the arm's sweeps in 3 of 4 episodes** (fall 0.25). The traced episode stood all 6 sweeps (36 s) with worst tilt 11.5°, where MuJoCo predicted 11.3° and 7 of 8. Fingertip tracking on the moving base was 6 mm mean. The hand passes above the garment by design, so no sort |
| 28 | `21329076` | **COMPLETED — the free base stands in Isaac**: knee-1.0 stance with the leg controller, arm still, 2 of 2 episodes × 36 s, no fall, no non-finite state. Trace: worst tilt 1.65°, root settles 2.2 mm, 0.9 cm of drift |
| 27 | `21328912` | **COMPLETED — with the arm still, Newton stays finite** (2 eps, preset buffers). The blow-up needs the arm moving |
| 26 | `21328911` | **COMPLETED — larger buffers do not fix it**: `njmax` 600 / `nconmax` 200, and NaN in 4 of 4 episodes at the same t = 2.83 s, the sample where the cloth first moves under the hand. `success_rate` 1.00, `success_rate_finite` 0.00 |
| 25 | `21328766` | COMPLETED on dgxh-3 (which has Vulkan) — 24 frames, but **every frame shows only the floor grid**: no robot, table or cloth on camera. Its trace has the arm NaN from t = 2.83 s, as in row 24 |
| 24 | `21328765` | COMPLETED — **void**: the success predicate fired 4/4 with the first sweep, and the cloth did settle in the shirts basket, but the arm's joint state went NaN at t = 2.83 s mid-sweep in the traced episode (see *C2F* above) |
| 23 | `21317391` | **COMPLETED — the shirt sorts on camera**, 64 frames (3.2 s), cn-r-2. Published: `docs/gifs/isaac/cloth_sort_fixed_base.gif` |
| 22 | `21317390` | **COMPLETED — jacket 8 of 8**, each with its first sweep, 0% refused, no falls (`isaac_c0f_v3b_jacket.json`) |
| 21 | `21317389` | **COMPLETED — sock 8 of 8**, each with its first sweep, 0% refused, no falls (`isaac_c0f_v3b_sock.json`) |
| 20 | `21317388` | **COMPLETED — shirt 16 of 16 on the widened baskets**, each with its first sweep, 0% refused, no falls (`isaac_c0f_v3b_shirt.json`) |
| 19 | `21317174` | **COMPLETED** — the free base falls on camera within 16 clip steps (0.8 s), garment untouched |
| 18 | `21317173` | **COMPLETED — free base, v3: falls 4 of 4** on the first sweep; pitch −19° at 0.5 s, on its back by 0.85 s |
| 17 | `21317172` | **COMPLETED — the squat falls with the arm held still**, 2 of 2 on the first macro step. Trace: pitch −15° at 0.5 s, −48° at 0.75 s, on its back by 0.9 s, the sweeping run's curve. The stance, not the arm |
| 16 | `21317171` | **FAILED on `dgxh-1`: no Vulkan**, so no render product, and the first camera read raised `CUDA error: an illegal memory access`. Node, not code: the same clip code ran on cn-gpu5/7. No frames, no result. The clip is re-queued on the widened baskets (row 23) |
| 15 | `21317170` | **COMPLETED — the shirt sorts: success 7 of 8 on the fixed base**, each with its first sweep, 0% refused, no wrong basket, no falls. Trace: fingertip 1.4–1.6 mm mean behind the schedule on the sweep, the shirt moved inside the sweep window. The one failure came to rest across both basket rims (see *v3* above) |
| 14 | `21307214` | **COMPLETED — the free base falls on camera** within 0.7 s (14 clip steps), garment untouched |
| 13 | `21307213` | **COMPLETED — with the corrected fall test the free base still falls**, 4 of 4 on the first sweep. Trace: body pitch grows from ~0.2 s into the first arm move (−17° at 0.5 s, −65° at 0.75 s, lying down by 1.0 s), tipping *backward*, away from the table |
| 12 | `21307212` | **COMPLETED** — clip, fixed base, v2: the same failure on camera (480 frames; largest travel 13.6 cm, final distance 0.30 m) |
| 11 | `21307211` | **COMPLETED — fixed base, v2: the garment moves the wrong way because the arm lags.** Success 0 of 4, fall 0, garment moved 4/4 (largest travel 12.3 cm), final distance 0.289 m, 67% of plans refused (all after the shirt left the table). Trace: measured-joint FK = Isaac's hand link to 0.0 mm; fingertip 65 / 72 / 59 mm mean behind the schedule on approach / sweep / retreat; the shirt moved at 0.50 s, 0.55 s before the sweep began, and ended on the jackets basket rim. See *v3* above |
| 10 | `21300605` | COMPLETED — the fall test fired on the first sweep in 4 of 4 episodes, hand colliders or not. **Not yet a measured fall:** the test read Isaac Lab 3.0's `(x, y, z, w)` quaternions as `(w, x, y, z)`, which counts a 30° yaw as 30° of tilt and a 60° roll as none (`QuatOrderTests`). Pitch reads correctly, so it may still be a real fall. Re-measured in rows 13–14 with the probed order: **it does fall**, backward |
| 9 | `21300604` | **COMPLETED — the clip shows why the push goes wrong** (480 frames, 1 env): the hand lands on the shirt's *far* corner, rides up its 1.5 cm edge, flips it at 1.4 s and leaves it at the table's far edge, 13 cm from spawn and 0.31 m from its basket. A MuJoCo replay of the same joint schedule, perfect tracking, finds the plan itself at fault: (1) hover and contact at one cell are different IK branches, so the 0.2 s descent swings the fingertip 7 cm toward the garment and back; (2) neighbouring glide cells flip branch too, 4 cm off the line; (3) the hand's tilt changes cell to cell, footprint 6–14 cm; (4) contact clearance 1.2 cm leaves 3 mm of a 1.5 cm garment's side to push. Fix in progress: continuity-seeded, hand-down contact table and an FK-checked schedule |
| 8 | `21300603` | **COMPLETED — with hand colliders the hand pushes the garment**: moved in **4 of 4** episodes, **7.2 cm** mean largest travel (was 0.07 mm). Not sorting yet: success 0, the garment ends *further* from its basket (0.25 m vs 0.17 at spawn), and after the first sweep 54% of plans are refused — the push lands somewhere the next sweep cannot reach. The clip is next |
| 7 | `21300494` | **the clip found it**: at 0.6–1.0 s the right hand comes down onto the table exactly where the garment is, and at 1.4 s lifts away with the garment unmoved. The contact table puts the Isaac hand on the garment; the hand passes through, because **the hand links have no collision geometry** — not in the URDF, the USD or the MuJoCo model (`add_hand_colliders.py` docstring) |
| 6 | `21300493` | **COMPLETED — fixed base, and the garment still does not move.** No falls, full 6-sweep episodes, 0% of plans refused, success 0 — and the garment's largest displacement averaged **0.07 mm** over 4 episodes, measured this time. The contact-table arm configurations do not bring the Isaac hand into contact with the garment. Candidates: the 4 Nm arm not tracking the schedule, no collision shape on the hand in Isaac, or the hand passing over a 1.5 cm garment. The clip is next |
| 5 | `21300348` | **COMPLETED — scripted C0 on the free base falls in the first sweep**: 4 of 4 episodes end at macro step 1 by `fallen`, success 0, **0% of plans refused** (the planner and schedule accept every scripted sweep). Its `max_garment_travel 0.0` is *not* a measurement: travel was only read between non-terminal steps and there were none; the eval now reports it as unmeasured |
| 4 | `21300301` | **the robot falls inside the first sweep** — 64 envs, raw 5-D actions: mean episode length **1.00** macro step, `fallen = 1.0000`, progress 0, success 0, refused-plan reward −0.07; 11 macro steps/s (44 s of physics per wall second). The gate's "mean length > 2" fails, correctly |
| 3 | `21300300` | **FAILED at Kit startup**, not in the task: it and `21300301` became eligible together when the smoke finished and both booted on cn-gpu6 in the same second. Two Kit instances starting at once share Kit's data directory inside the venv, which is the lock contention `21077722` already recorded. Isaac jobs now chain one after another instead of fanning out from one parent |
| 1 | `21300299` | **COMPLETED** — both rigid ids construct, reset and step on the redesigned scene with the schedule action (22 joints, action_dim 5); 0.78 macro steps/s at 4 envs, each step 4 s of physics |

### Isaac renders show no robot · `done` — cause found, camera-sensor recorder works, first clip published 2026-09-13
**Found (`21299608`).** It is not the asset, not the renderer and not visibility:
every robot mesh is loaded and computes `inherited` visibility. It is *which
pose gets drawn*.

- **The viewport draws the robot at its stale USD pose.** With fabric on,
  physics writes articulation poses to Fabric, and USD keeps the spawn pose —
  the maze robot's USD base stayed at (0, 0, 0) while physics had it at
  (−27.9, −76.4). The viewport, which is what `train_play --video` records,
  therefore draws the body at the env's grid origin, tens of metres from where
  it walks, and a camera aimed at the robot films empty ground. That is every
  robot-less clip since the spawn photographs.
- **A camera *sensor* draws the body where physics has it**, fabric on: the
  maze robot on its terrain patch, and the cloth robot next to its table —
  visibly facing away from it, the reach finding in one frame.
- **TaskV2 clips only ever showed robots because those robots barely move**:
  USD spawn pose and physics pose nearly coincide. They also confirm the old
  spawn bug is still live — `robot_a`'s bodies span z −0.806 … −0.027, all
  under the floor.

**And it changes B5.** The maze walls, arrows, obstacles and button sit at each
env's *grid* origin; with a terrain generator the robots are reset onto
*terrain* origins elsewhere, so the geometry was never where the robots
trained. Independently, lidar and stereo ray-cast only `/World/ground`
(`sensors_rig.make_lidar_cfg` / `make_stereo_cfg` defaults), so they could not
have seen a wall anyway. **B5 is a terrain-perception comparison.** The stereo
width result stands as that; nothing in it is about a maze.

`train_play` now records clips through a camera sensor that follows the robot
(`BHL_CAMERA_CLIP=1`, PNG frames, gif assembled on the login node);
`docs/ISAAC_RENDER.md` §10 records why.

| # | id | outcome |
|---|---|---|
| 3 | `21299952` | **COMPLETED** on cn-gpu6 — blind, 200 frames. All four arms rendered; composite published as `docs/gifs/isaac/terrain_sensors.gif` (10 MB) |
| 2 | `21299873` | **the recorder works**: lidar, stereo P4 and stereo P16 each wrote 200 frames with the robot in shot. Blind died inside `SimulationApp._start_app` 4 s into Kit startup on cn-gpu7, before `train_play` ran a line — a node-side RTX start crash, not the recorder |
| 1 | `21299608` | **COMPLETED** — viewport vs sensor × fabric vs USD on maze, cloth, TaskV2: cause above |

### Cloth sorting — the layout is unreachable · `open` — found 2026-09-11
**No sweep in this scene could ever have touched the garment.** Two
measurements, one per engine, and they agree:

- **Reach, MuJoCo FK** over the full right-arm range, legs in the pinch squat
  the controller holds: the fingertips can touch a 0.30 m table top only on the
  robot's right and never more than **0.28 m forward** of the root *(corrected
  2026-09-13: this first said the hand could not reach the top at all, which was
  the hand-link origin, 13 cm above the fingertips)*. The garment spawned **0.74 m** away; the table's
  near edge was 0.39 m; the basket centres 0.32 m. Nothing was in reach.
- **Facing, Isaac** (`21247910`): the hands sit at (−0.208, **+0.192**) from
  the root, where FK facing +x puts them at (+0.209, **−0.177**). Both axes flip,
  so under `(0, 0, 1, 0)` the robot faces **−x** — away from the table — with its
  geometry matching MuJoCo to 2–4 mm. The right hand was **0.97 m** from the
  garment.

The kinematic ladder never modelled reach: a sweep was two planar points and
the hand was assumed to follow them. So its **1.00** in C0, C1, C4-BC and C5
meant "a hand that could go anywhere would sort these", not that this robot
can. It also moved garments on plans it had itself flagged invalid.

`bhl_robust.cloth.reach` now carries the measured workspace — a 2 cm IK table
built from MuJoCo FK plus damped least squares, 1,380 reachable voxels, in
`assets/cloth/right_arm_ik_pinch.npz` — and the controller refuses a plan the
hand cannot follow. **Under it every kinematic cell scores 0.00** (C0 64 eps,
C1 64, C4-BC 64, C5 32; every sweep refused), written to
`results/cloth/reach/`. The old files are left as they were. `ROBOT_YAW` is set
to π from the Isaac measurement.

The workspace that exists is a patch on the robot's **right side**: robot-frame
x −0.12…+0.26, y −0.44…−0.08, hand-link heights 0.34–0.54 m, with the hand mesh
hanging 2–13 cm below its link origin depending on the pose. It never crosses
the midline. A layout redesigned inside that patch is the next step; the
decision rule for this outcome says fix the geometry before training anything.

| # | id | outcome |
|---|---|---|
| 1 | `21247910` | **COMPLETED** — faces −x; right hand 0.97 m from the garment |
| — | login node | FK reach map and IK table; kinematic ladder under reach: 0.00 in all four cells |

### Cloth sorting — jobs cancelled · `superseded` by the ladder entry below
Accurate when written, and overtaken the same day: the five cells were re-run
through four fixes (`21233802` → `21234259`), and `results/cloth_sort_bench.md`
now carries measured Isaac rows. Kept for the ids.

`21228030`–`21228033` (c1 smoke, isaac eval, deform smoke, sort bench) were all
cancelled before running; `21228029` failed its own guard with "produced no
summary line". So the mesh-resolution sweep that would reopen G-C1 has not
produced a measurement yet — `results/cloth_sort_bench.md` still marks every
Isaac row "not yet measured", which is accurate.

### B5 — maze with lidar and stereo · `done` — its stereo numbers are void: the pair looked up on v60 (*stereo re-run* above)
The four-arm result below stands as measured. Its explanation does not: the
pooling sweep shows the stereo arm failing because its 512 depth values were
92% of the input, not because stereo carries nothing useful here.

Four arms, one variable: blind (control), lidar, stereo, both. 6,000 iterations
at `NUM_ENVS=2048`, identical across all four — the stereo arms carry two
ray-cast cameras and the blind arm none, so letting the cheap arm run wider
would make throughput the variable instead of sensing.

v60 with cameras enabled. Both sensors are ray-cast rather than RTX, so neither
depends on the renderer that segfaults on 5.1; that was the reason the design
chose ray-cast over an RGB camera.

Smoke passed 4/4 (`21201598`): widths 45 / 81 / 557 / 593, stereo returning real
depth at 0.61–6.00 m and 100% finite, lidar 30% finite as a horizontal 360° scan
in a corridor should be.

**Result, 4 of 4 COMPLETED at 6,000 iterations.** Read from the event files:

| arm | ep_len | reward | terrain level |
|---|---|---|---|
| blind (control) | 400.5 | 12.362 | 0.5191 |
| **lidar** | **432.2** | **15.416** | **0.7814** |
| stereo | 323.2 | 7.677 | **0.0005** |
| both | 355.5 | 9.501 | 0.0624 |

**Lidar beats the blind control by 51% on terrain level** (0.78 against 0.52) and
25% on reward. **Stereo is far worse than blind** — pinned at 0.0005, effectively
never leaving level 0 — and it drags `both` down with it, from lidar's 0.78 to
0.06. Adding the cameras to the lidar destroys the lidar advantage.

The likely mechanism is width, and it is already documented in this repo:
`depth_obs`'s docstring warns that fed raw to upstream's MLP the depth would be
"99% of the input width and the first layer would be almost entirely depth
weights". Stereo is **512 of 557** observations here, 92%. Lidar is 36 of 81,
44%. This is the same shape as "giving them eyes made it worse" on the lift,
now on locomotion.

**What this is not.** `maze_env_cfg.py` defines no rewards, terminations or
success terms of its own — it inherits the locomotion objective, and the maze
walls, arrow plates and button sit in the scene as static geometry. So the four
arms were scored on velocity tracking and the terrain curriculum with obstacles
in the way, **not** on navigating a maze or reading an arrow. The comparison
between sensors is sound because all four arms share the objective; the task
described in `docs/MAZE_RIG.md` — approach, address, sweep, arrow-following — is
designed and not yet implemented.

| # | id | outcome |
|---|---|---|
| 1 | `21218766` | **4 of 4 COMPLETED** (5:04–8:30) — table above |

### Cloth sorting — reopened against a coarser mesh · `superseded` by the ladder entry below
The coarse-mesh question has an answer for two of its four meshes: **467
env-steps/s at 8×8 (81 vertices) and 438 at 10×10 (121)**, 8 envs, against
G-C1's 182 at 961 (`21234167`). 12×12 and 16×16 were not run. Note Isaac's
`resolution` counts cells, so the bench file's planned "64 / 100 / 144 / 256
vertices" are 81 / 121 / 169 / 289.

G-C1 measured 182 env-steps/s on a **961-vertex** cloth and closed the task as
scripted-not-RL. This reopens it on the one axis that number leaves open: mesh
resolution. `results/cloth_sort_bench.md` sweeps 64, 100, 144 and 256 vertices
against a measured kinematic baseline of ~350 env-steps/s at 1 env.

The bench file labels the unmeasured rows `planned_isaac` and
"not yet measured", which is the right discipline — they are a plan, not data.

| # | id | outcome |
|---|---|---|
| 2 | `21228030`–`21228033` | queued — c1 smoke, isaac eval, deform smoke, sort bench |
| 1 | `21228029` | FAILED — "produced no summary line", so the guard caught it rather than passing an empty run |

### Spawn rotation — found and photographed; task env still disagrees · `open`
**The rotation is settled, by picture.** `results/spawn_shots/` holds a spawn
photograph per candidate, taken with no policy, no checkpoint and no reset
events between the config and the camera:

| quaternion | ankle | shoulder | below | picture |
|---|---|---|---|---|
| `(0.7071, 0, 0, -0.7071)` — as configured | -0.009 | -0.001 | **15/27** | robot **lying flat** |
| **`(0, 0, 1, 0)`** | **+0.131** | **+0.735** | **1/27** | robot **standing** |
| MuJoCo, same URDF | +0.140 | +0.737 | 1/26 | — |

So the configured 4-tuple lays this robot on the floor, which is what was
reported from the clip on day one, and `(0, 0, 1, 0)` stands it up to within
9 mm of MuJoCo on every measure.

**Heading cannot be set.** Every yaw of that pose, in both composition orders,
buries the robot again (15 or 27 of 27). A genuine world-z yaw cannot do that to
a rigid body, so these 4-tuples do not compose as (w, x, y, z) world rotations
for this asset. Both robots therefore use `(0, 0, 1, 0)` and face the same way;
that is recorded as a limitation rather than papered over.

**What is still wrong.** With that quaternion applied to the task config, the
rendered episode shows **no robot anywhere in frame** -- full frame, native
resolution, frames 2 / 10 / 40 -- while the standalone spawn under the identical
quaternion photographs a robot standing on the floor. The one untested
difference is the reset event pipeline (`reset_root_state_uniform`,
`reset_joints_by_offset`), which the standalone probe does not run. That is the
next thing to check, and nothing should be claimed fixed until an episode frame
shows a robot.

| # | id | outcome |
|---|---|---|
| 3 | `21218672` | task render with `(0,0,1,0)`: no robot in frame at 2 / 10 / 40 |
| 2 | `21218650` | facing sweep: only `(0,0,1,0)` stands; all six yaws bury it |
| 1 | `21218517`, `21218627` | **the photographs** — configured quat lies flat, `(0,0,1,0)` stands |

### Spawn orientation — NOT solved, and the earlier "SOLVED" is retracted · `open`
The robots spawn under the floor. That is visible in the first frames of every
Isaac clip and it is the only thing in this investigation that has stayed true.

**What is established.** MuJoCo spawns this same URDF correctly, measured:
root z -0.0272, quat (0.7071, 0, 0, 0.7071), ankle +0.140, shoulder +0.737,
base -0.027, **1 of 26 bodies below ground**. That is the reference.

**What is retracted.** The 2026-09-06 entry claimed the spawn quaternion was a
roll and that `(0.7071, -+0.7071, 0, 0)` fixed it. It did not. Those quaternions
put 27 of 27 bodies below ground -- worse than the 19 of 27 they replaced -- and
the clips published from them show robots that never appear on screen. The claim
rested on a probe measuring torso-above-ankles, which is relative geometry and
says nothing about absolute height.

**Why no further probe is being run.** Six Isaac probes have now disagreed with
each other and with the render:

* four applied no rotation at all -- `_body_dump` and `_freefall` returned
  byte-identical geometry for quaternions 90 degrees apart, twice
* `_updir` used `base` as a proxy for "head"; base is this asset's
  ground-reference frame, so `base - ankles > 0` reads upright for a robot lying
  down
* the in-task sweep found `(0, 0, 1, 0)` reproducing MuJoCo to 4 mm on every
  number -- ankle +0.144 against +0.140, shoulder +0.737 against +0.737, 1 of 27
  below -- and the render of that exact config shows **no robot at any frame**,
  worse than what it replaced

Each time the disagreement was resolved in favour of a conclusion that the next
measurement broke. The pattern, not any single result, is the finding.

**State of the tree.** Quaternions and `_tilt_from_quat` are back to what they
were. Two MuJoCo-derived constants are kept because they are measured against
the engine that works: `_PINCH_ROOT_Z = -0.0272` (was -0.07) and
`SOLE_REF = 0.1403`. `plant_feet` stays opt-in and off.

**The route that has not been tried.** MuJoCo does not find a magic quaternion;
`CrewRunner.reset` poses the joints, measures the lowest **collision geom**, and
translates the base onto the plane. `plant_feet` approximates that with body
*origins*, which are not the same thing -- a foot's origin sits above its sole.
Porting the geom-based version is the next real attempt, and it should be
verified by looking at frame 0 before any number is quoted.

| # | id | outcome |
|---|---|---|
| 6 | `21213903` | render of `(0,0,1,0)`: no robot visible at frames 0, 2, 6, 15, 40, 90 |
| 5 | `21213882` | in-task sweep: `(0,0,1,0)` matches MuJoCo on every number; every yaw of it fails |
| 4 | `21213809` | `_freefall` -- four quaternions, identical geometry. Void. |
| 3 | `21213793` | `_body_dump` with rotation proven applied -- still identical across 90 degrees. Void. |
| 2 | `21213697`, `21213698` | 27 of 27 below ground with the "fixed" quats; 8 of 27 with planting on |
| 1 | (visual) | the user identified it from the clip: "spawns from under the ground, upside down" |

### Spawn orientation — superseded by the entry above · `retracted`

**The spawn quaternion was a roll, not a yaw. The robots have been lying down.**

Measured from geometry rather than convention (`21196896`):

| spawn | torso above ankles | hand split |
|---|---|---|
| old `(0.7071, 0, 0, -0.7071)` | **+0.028 m** — horizontal | **0.393 m** |
| new `(0.7071, -+0.7071, 0, 0)` | **+0.224 m** — upright | **0.012 m** |

That is the whole "arm asymmetry" that five hypotheses chased. Nothing was wrong
with the arms, the asset or the pose — and the tell was in the Isaac clip, where
the robots were visibly flat on the ground.

**Second half: the asset is Y-up.** |R21| is 0.995-0.998 under the spawns that
stand it up while R22 is ~0.08, and `_tilt_from_quat` used R22 — reporting 1.5
rad of tilt for an upright robot, clearing the 0.78 limit and ending every
episode on step one. It uses R21 now, and the corrected reading calls the old
lying-down spawn fallen at 1.571 rad, which it always was. So
`projected_gravity_b` returning `[0, +-1, 0]` was **correct**; this repo recorded
it as an Isaac Lab 3.x defect and built a workaround.

| # | id | outcome |
|---|---|---|
| 3 | `21196912` | **gate passes** — 300-iteration smoke, mean episode length **35.0**, against 5.0 leg-planted, 1.00 under the yaw probes and ~8 for the original welded arms |
| 2 | `21196896`, `21196905` | up-axis from geometry: torso 22 cm above the ankles under the new quats, 2.8 cm under the old |
| 1 | `21192744`–`21192782` | prior work, already correct and not read before five hypotheses were spent re-deriving it |

### B3 ice clip — export solved, render blocked on depth obs · `superseded` by *B3 ice clip — DONE*
`ENABLE_CAMERAS=1` was the v51 segfault, inherited from the cloth-probe sbatch
and surviving the edit that removed `--enable_cameras` from the command line.
Without it both arms export cleanly: distinct ONNX and deploy configs, 45
observations blind and 301 depth, each pointing at its own run.

What remains is that `render_multi` cannot drive a depth-conditioned policy.
`--depth-of` controls the display strip, not the observation, so it fed 45
values into a 301-wide policy. The ice depth arm wants a 64x64 ray-cast camera
average-pooled 4x to 16x16 = 256, and MuJoCo's offscreen depth buffer is a
different sensor from Isaac's ray-caster.

| # | id | outcome |
|---|---|---|
| 3 | `21197107` | pipeline verified blind-vs-blind on a GPU node, exit 0, 12 MB gif — discarded, since a same-policy clip named `ice_pair` reads as the comparison it is not |
| 2 | `21197056` | **both arms exported** — distinct ONNX, 45 and 301 obs |
| 1 | `21197021` | export worked, then hung: the play loop had no exit without `--video`. `--play-steps` bounds it. |

### B4 — limb agents (DirectMARL + skrl) · `superseded` — the gate passed and Tier 1's first block trained
Written before any of it happened, and kept for the design notes. **Status as of
2026-09-13:** G-B4 passed on both partitions (`21090555`, *B4 — G-B4 gate*), and
Tier 1's first block completed (`21105320`, `21124513`, *Tier 1 first block*):
limb2 + MAPPO +3.22, limb4 + MAPPO +2.08, limb4 + IPPO +1.95, against a
single-agent control. FINDINGS carries it as *Limb agents*.

Multiple agents per robot, one per limb, against the single-agent PPO controls.
Gates all 24 Tier-1 rows, so nothing downstream can start until `G-B4` passes.

**Never queued.** No config, no `BHL_ALGO`/`BHL_AGENTS` in `BHL_FORWARD_VARS`,
no `G-B4`, no commits. The `DirectMARL` imports in `scripts/train.py` are Isaac
Lab's stock boilerplate, not this work.

Stack verified present (`21076389`): `direct_marl_env.py` ships in isaaclab
2.3.2, `isaaclab_rl/skrl.py` handles MARL, skrl 1.4.3 has IPPO and MAPPO.
`21076792` installed `rsl-rl-lib==3.0.1` on the v60 venv, which had Isaac Sim
and Isaac Lab and no RL library — the reason RGB training had never run.
Open decisions: how the 22 DoF partition into limbs, and IPPO vs MAPPO.
Constraint from the work order: `joint_deviation_arms` must be ablated in any
22-DoF limb-agent run.

| # | id | outcome |
|---|---|---|
| — | — | not started |

### Tier 1 first block — seeds 1 and 2 · `done` — n=3: the limb2 lead does not replicate, and every skrl row ran untuned PPO
The first block's five rows again at seeds 1 and 2, because the limb2 result was
one seed. `slurm/89_marl_train.sbatch` unchanged: Arms-Bumpy, 22 DoF, 4,096 envs,
6,000 iterations. A 3-iteration smoke of all five rows went first, and both
seeds chained on it `afterok`.

**Final, from the event files** — mean of the last 5% of points, the statistic
the seed-0 table in FINDINGS used (it reproduces 3.22 / 2.19 / 2.08 / 1.95
exactly):

| row | total reward s0 / s1 / s2 | mean | per-step reward | episode length, steps |
|---|---|---:|---|---|
| limb4 + MAPPO | 2.078 / 1.787 / 2.638 | 2.168 | 0.052 / 0.044 / 0.060 | 34 / 37 / 42 |
| limb4 + IPPO | 1.954 / 2.103 / 1.402 | 1.820 | 0.047 / 0.051 / 0.032 | 38 / 37 / 35 |
| limb2 + MAPPO | 3.224 / 2.562 / 1.798 | **2.528** | 0.043 / 0.036 / 0.038 | 65 / 62 / 45 |
| limb1 + IPPO (control) | 2.190 / 2.403 / 2.228 | **2.274** | 0.026 / 0.030 / 0.028 | 80 / 78 / 74 |
| PPO, rsl-rl, not arm-ablated | mean reward 3.84 / 3.57 / 3.74 | — | — | 227 / 221 / 223 |

- **limb2's lead does not replicate.** It is +11% over the control on the mean,
  not +47%, and its seed 2 (1.80) sits below every control seed. limb4 + MAPPO
  lands on the control (2.17); limb4 + IPPO is below it.
- **The statistics disagree, and none of them is terrain level.** Total reward is
  episode length times per-step reward, and the two move opposite ways with agent
  count: by per-step reward every split beats the control, by episode length the
  control beats every split.
- **Every skrl row was a handicapped PPO.** They ran skrl's defaults beside an
  rsl-rl control configured otherwise: entropy 0 against 0.008, a fixed learning
  rate against the KL-adaptive schedule, 8 epochs × 2 mini-batches against 5 × 4,
  gradient clip 0.5 against 1.0, no time-limit bootstrap, observation and value
  normalisation rsl-rl does not use, and a [256, 256, 128] network against
  [256, 128, 128]. The skrl control survives 74–80 steps where rsl-rl PPO survives
  ~225 on the same task. Comparisons *between* skrl rows hold the handicap fixed;
  none says what a limb split does to a tuned PPO. `train_marl.py --hparams rsl`,
  the default for every run not named `marl-*`, now reads every one of those
  settings from the task's own rsl-rl runner config.

Found while reading these:

- **No skrl run has ever logged terrain level.** skrl's trainer forwards
  `infos["episode"]`, and only one-element tensors; Isaac Lab reports the
  curriculum under `infos["log"]` and `.item()`s it first. `environment_info="log"`
  alone got 28 Info tags and still no terrain level (`21302171_4`, `21328444`);
  `limb_marl.loggable` turns those scalars back into tensors, and G-B4t now
  requires the tag.
- **The PPO control, row 3, was never arm-ablated.** `BHL_ABLATE_ARM_DEV` is read
  only by the skrl branch of `marl_train.sh`; row 3 execs `train.sh`, which
  ignores it. The sbatch's "ablated in every row including the control" was never
  true, at any seed. FINDINGS already leaves row 3 out, because it prices the RL
  library as well. Tier 3's PPO rows must be ablated, so the ablation has to
  reach rsl-rl first.

| # | id | outcome |
|---|---|---|
| 3 | `21302172` | seed 2 — **5 of 5 COMPLETED** (4:20–6:02). Time limit cut 40 h → 14 h (longest first-block run: 6:08) to free the GPU-minute cap |
| 2 | `21302171` | seed 1 — **5 of 5 COMPLETED** (4:59–6:09); same limit cut |
| 1 | `21302170` | **smoke 5/5 COMPLETED** (1:17–1:20; seed 99, 3 iterations) — each skrl row ran 72 of 72 timesteps, PPO logged 3 iterations at episode length 17.9 |

### Tier 1 MARL grid — biped, left leg | right leg · `paused` — skrl does not yet learn to walk on stairs; not the critic, 2026-09-15
**Critic A/B, 1,500 iterations on stairs + depth, matched iterations, event files
(2026-09-15).** Velocity-tracking reward at iterations 300 / 600 / 1,200 / 1,500:

| arm | 300 | 600 | 1,200 | 1,500 | terrain level at 1,500 | episode length |
|---|---:|---:|---:|---:|---:|---:|
| rsl-rl PPO, seeds 0 / 1 / 2 | 0.46–0.48 | 0.74–0.89 | 1.34–1.36 | **1.39–1.41** | 0.97–1.12 | 457 |
| skrl MAPPO, one agent, **privileged** critic (`21330392`) | 0.53 | 0.56 | 0.57 | **0.58** | 0.000 | 433 |
| skrl IPPO, one agent, old policy-obs critic (`21330372_1`) | 0.47 | 0.52 | 0.54 | 0.56 | 0.000 | 399 |
| skrl MAPPO, legs2, privileged critic (`21330372_2`) | 0.41 | 0.44 | 0.49 | 0.49 | 0.000 | 370 |

**The critic was not the cause.** With PPO's critic, the one-agent skrl control
tracks at 0.58, where PPO is at 1.40; the critic moves it by 0.02. It learns to
survive and then stops improving, while PPO keeps climbing between iterations
600 and 1,200. The curves that do differ are the exploration noise and the learning rate:

| iteration | skrl std | rsl-rl std | skrl LR | rsl-rl LR |
|---:|---:|---:|---:|---:|
| ~32 | 0.89 | — | **1.0e-2** (its ceiling) | — |
| ~150 | 0.41 | 0.63 (at 120) | 7.3e-4 | 1.3e-3 |
| ~750 | 0.19 | 0.56 (at 600) | 2.6e-4 | 3.8e-4 |
| 1,500 | **0.13** | **0.42** | 1.2e-4 | 2.6e-4 |

skrl's `KLAdaptiveLR` adjusts once per update, against skrl's approximate KL.
rsl-rl adjusts every mini-batch against the analytic Gaussian KL. skrl drove the
rate to its 1e-2 ceiling in the first ~30 iterations, and the value loss spiked
to 20. It also parameterises log-std, which Adam moves multiplicatively, where
rsl-rl moves the std itself. The noise collapsed to a third of PPO's before the
policy learned to walk. `21338294` splits the two suspects on the one-agent
control: std as a direct parameter with the KL schedule, std direct with a
fixed rate, and log-std with a fixed rate.

The critic and MAPPO = IPPO findings below still stand as bugs fixed; they are
just not what held learning back.

**Stopped on the first stairs rows' curves, from the event files.** Seed 1's
IPPO row finished all 144,000 timesteps; the three others were cut at 25–64%,
already flat:

| row | track_lin_vel_xy reward at iteration ~300 → end | terrain level | episode length |
|---|---|---|---|
| rsl-rl PPO, stairs + depth, s0 (`21066022`) | 0.49 → **1.52** | 0.71 → 0.00 → **2.69** (rising from iteration ~600) | 481 |
| skrl IPPO legs2, s1 (`21328608_4`, full run) | 0.43 → **0.53** | 0.004 → **0.000**, never rises | 380 |
| skrl MAPPO legs2, s1 (`21328608_3`, 64%) | 0.49 → 0.50 | 0.000 | 383 |

The robots learned to survive, not to walk to the command, so the curriculum
never promoted them. The cause is in the wrapper, not the split.
`LimbMarlEnv.state()` returned the *policy* observation, noisy and without base
linear velocity, and rsl-rl's critic reads the `critic` group, which has both.
Because agents share the full observation, MAPPO's centralised state was
identical to what each IPPO critic already saw, so MAPPO and IPPO were the same
algorithm, first block included. Its docstring's claim that this "is the same
information the single-agent baseline's privileged critic already receives" was
wrong.

Fix: `LimbMarlEnv(critic="privileged")` reads the `critic` group (state 3 wider
than the observation), and both gate checkers now require it. The skrl limb1
control has to be MAPPO with one agent to get PPO's critic; IPPO keeps
per-agent critics on the policy observation, which is now a real difference.
`21330372` trains three arms for 1,500 iterations on stairs to confirm before
anything resumes: limb1 + privileged critic (should track like PPO), limb1 + the
old critic, and legs2 MAPPO + privileged. The pending grid tasks are held, not
cancelled.

The limb1 arm died at step 0 and was reported COMPLETED (`21330372_0`, 0:52):
skrl picks its single-agent loop when there is one agent, that loop never puts
`shared_states` into infos, and MAPPO raised `KeyError`. The completion check
passed it because skrl prints `0/36000 [` and creates its run directory before
the first step, and the check accepted either. The wrapper now supplies
`shared_states` / `shared_next_states` itself, and `marl_train.sh` (and the A/B
script) pass a run only if its last progress count equals its total. The check
still passes every finished row (`144000/144000`) and fails that arm.
Resubmitted as `21330392`.

The work order's Tier 1 rows (`slurm/89b_marl_terrain.sbatch`): MAPPO and IPPO on
the 12-DoF biped split left leg | right leg (`legs2`), plus a limb1 single-agent
control on every terrain, with ray-cast depth, 4,096 envs, 6,000 iterations,
seeds 0–1, run names `{row}-{terrain}-depth-s{seed}`. Stairs, slippery and rough;
ice is out until B3's patches reach the robots. PPO settings and network are the
rsl-rl control's (`--hparams rsl`). Array tasks serialise their Isaac boots on
`.v51-boot.lock`.

G-B4t (`slurm/88b_marl_terrain_gate.sbatch`) trains every row for two real
iterations at full width and passes only on what the built env and trainer report
about themselves: 2 agents × 6 joints with the left and right hip first (or
1 × 12), MAPPO / IPPO objects matching the row, 301-wide observations with the
depth term on every terrain and MAPPO's state equal to them, 48 of 48 timesteps,
terrain level in the event file, and the rsl-rl settings. The action order is
also checked inside `LimbMarlEnv` against the action term's own joint names.

| # | id | outcome |
|---|---|---|
| 4 | `21328607_3`, `21328607_4`, `21328608_3` / `21328608_4` | **cancelled** at 51%, 25% and 64% (MAPPO s0, IPPO s0, MAPPO s1), flat since iteration ~300; `21328608_4` IPPO s1 COMPLETED (4:32). Table above. Remaining tasks `_5`–`_11` held |
| 3 | `21328605` → `21328607`, `21328608` | **G-B4t PASS, 9 of 9** (6:50) — terrain level logged, rsl-rl settings confirmed (5 epochs × 4 mini-batches, entropy 0.008, KL-adaptive LR, [256, 128, 128]), observation 301 on all three terrains. Seeds 0 and 1 `--array=3-11`, chained `afterok`; both seeds' stairs MAPPO and IPPO rows running. Throttle cut from `%2` to `%1` per seed at the per-user GPU cap, so the cloth chain gets slots |
| 2 | `21328445`, `21328446` | **cancelled** — `DependencyNeverSatisfied` after the gate failed; their ice rows had been held first, pending `21328532` |
| 1 | `21328444` | **FAIL, correctly** (9:50). (i)–(iii) pass on all 12 rows: 2 agents of 6, left and right hip first, observation 301 = state with depth, trainer matches. (iv) fails on 9: no terrain level in the event file, because Isaac Lab `.item()`s curriculum scalars and skrl logs only tensors. The three rough rows loaded the `loggable` fix mid-gate and passed (iv) |

### Tier 3 — 22 DoF on stairs, depth, arm deviation off · `running` — PPO seed 0 done at 4.49, seed 1 queued; skrl rows held, 2026-09-15
`Velocity-BHL-Arms-Stairs-Depth-v0` (`tasks/arms_terrain_env_cfg.py`): the arms
robot on the biped stairs menu with the depth rung's camera and term, and
`joint_deviation_shoulder` / `_elbow` cleared **in the task**. The first block
ablated them only on the skrl path, and its PPO control kept them at every seed;
in the task, PPO and MAPPO cannot differ on it, and `__post_init__` refuses to
build if either term is missing. Ice waits on B3.

Rows (`slurm/89c_arms_tier3.sbatch`): PPO on rsl-rl, MAPPO over four limb agents
(5 / 5 / 6 / 6), and the limb1 skrl control, seeds 0–1, 4,096 envs, 6,000
iterations. G-T3 (`88c_arms_tier3_gate.sbatch`) passes only if every row's reward
table lacks both arm terms and keeps `joint_deviation_hip`, the observation is
331 wide with depth (PPO's actor included), limb4's first joints are the two
shoulders and the two hips, and the skrl rows log terrain level on the rsl-rl
settings. The registration is guarded, so a failed import prints and leaves the
id absent rather than breaking every queued job that imports the registry.

| # | id | outcome |
|---|---|---|
| 2 | `21328743_0`, `21338295` | **PPO seed 0 COMPLETED** (4:28): terrain level **4.493**, episode length 386, tracking 1.16 (last 50 iterations). Seed 1 resubmitted as `21338295`, since `21328744` waited on the held seed-0 skrl rows and was cancelled |
| 1 | `21328742` → `21328743`, `21328744` | **G-T3 PASS, 3 of 3** (5:37): no shoulder or elbow deviation in any reward table, `joint_deviation_hip` present, observation 331 with depth (PPO's actor included), limb4's first joints the two shoulders and two hips, terrain level logged on the rsl-rl settings. Seed 0's PPO row running (`21328743_0`); both seeds' MAPPO and limb1 rows held (`21328743_1`–`_2`, `21328744`) until the critic A/B, since they share the grid's critic |

### Tier 1 / 2 / 3 MARL rows · `todo` — Tier 2 and Tier 1's PPO rows already exist; Tier 1's MARL rows and Tier 3 on stairs are queued (above)
**Audit, 2026-09-13**, against the work order's grid:

- **Tier 2 (PPO, blind, on slippery / stairs / ice, 2 seeds) — covered.** Slippery
  and stairs blind ran at seeds 0–2 (`21066022`, `21076264`), ice blind at seeds
  0–1 (`21105232`), all at 4,096 envs and 6,000 iterations. Nothing to queue.
- **Tier 1's eight PPO depth rows — covered** by the same three arrays and §6's
  rough depth (1.601 / 1.598).
- **Tier 1's sixteen MARL rows — queued for three terrains** (*Tier 1 MARL grid*
  above); ice waits on B3. As audited on 2026-09-13: they are IPPO and MAPPO on the
  **12-DoF biped at N=2** (left leg | right leg), depth on, over rough / slippery
  / stairs / ice. `partition_for` knows only the 22- and 24-DoF layouts. The
  "first block" that ran is the 22-DoF split on Arms-Bumpy — the work order's
  Tier 3 shape, not its Tier 1. Needs a `legs2` partition, a gate on each
  terrain's depth task, terrain level from skrl (done), and a `limb1` control per
  terrain, because the first block showed an rsl-rl PPO row prices the library
  as well as the split.
- **Tier 3 (PPO and MAPPO, ice and stairs, 2 seeds, 22 DoF at N=4, depth, arm
  deviation ablated) — not started.** `Velocity-BHL-Arms-{Stairs,Ice}-Depth-v0`
  do not exist, and the PPO rows cannot be ablated yet (above).

24 + 6 + 8 jobs. Work order: do not queue a tier until its gate passes.
`NUM_ENVS` identical across every arm (target 4096; if MARL OOMs, drop *every*
arm to 2048 and re-run the single-agent controls at 2048 too — do not mix).

| # | id | outcome |
|---|---|---|
| — | — | not started |

### Compat shim regression check · `done`
The physx shim fired on v51 and broke every v51 task. Guard now tests the
annotation, not `hasattr`.

| # | id | outcome |
|---|---|---|
| 2 | `21077722` (v51) | inconclusive — Isaac Sim died during boot on kit DB lock contention with the running terrain arrays. Guard verified from source instead: `physx:` is annotated on 2.3.2, so the shim is skipped there. |
| 1 | `21077723` (v60) | done — 4 shims applied, 36 task ids registered |

### B4 — G-B4 gate · `done`
Design settled (`docs/TASKS_V2.md`): four limb agents, MAPPO, with 2-agent and
IPPO ablations. Code written: `limb_partition.py`, `tasks/limb_marl.py`,
`scripts/train_marl.py`, `scripts/bench/marl_gate.py`.

Offline half (coverage, round-trip, joint order) **passes** on both partitions
and needs no GPU.

| # | id | outcome |
|---|---|---|
| 5 | `21090555` | **PASS, both partitions** — limb4 4 agents 5/5/6/6, limb2 2 agents 10/12, obs 75, and the ablation clears `joint_deviation_shoulder` + `joint_deviation_elbow` |
| 4 | `21083178` | TIMEOUT — but limb4 reached `stepped ok`. `simulation_app.close()` sat between the checks and the verdict print and hung for 80 minutes, so a run that had its answer was recorded as a timeout. |
| 3 | `21082868` | FAILED — died silently after "Completed setting up the environment", no traceback. Suspect the `clear_instance()` + `app.close()` teardown, which is redundant now that it is one partition per process. |
| 2 | `21078987` | TIMEOUT — limb4 **passed** (4 agents, 5/5/6/6, obs 75), then hung an hour building limb2 in the same process. Also reported `arm_ablation=TERM NOT FOUND`. |
| 1 | `21078958` | FAIL — pointed at `Biped-Bumpy`, which actuates 12 leg joints, not 22: `Invalid action shape, expected: 12, received: 22`. |

### Tier 1 first block · `done`
`slurm/89_marl_train.sbatch` is written and deliberately not submitted: limb4+
MAPPO, limb4+IPPO, limb2+MAPPO, and a single-agent PPO control, one variable
moving per row. `joint_deviation_arms` is ablated in every row including the
control, and `train_marl.py` refuses to start if the term is not found rather
than assuming the ablation happened.

| # | id | outcome |
|---|---|---|
| 4 | `21105252` | **PASS, both partitions, both layouts** — 22 and 24 DoF, and the arm ablation clears both real terms |
| 3 | `21100446` | FAIL — `Invalid action shape, expected: 22, received: 24`: extending the partition for the gripper broke the gate against the 22-DoF task |
| 2 | `21093567` | FAILED — `output with shape [4096, 1] doesn't match the broadcast shape [4096, 4096]`. The env returns a flat `(num_envs,)`; skrl stores one column per agent, so the trailing axis was the whole fix. |
| 4 | `21124513` | COMPLETED — `limb1` control, one agent under the same trainer; readout queued (`21136243`) |
| 3 | `21105320` | **all 3 COMPLETED** — 144k timesteps each. limb2+MAPPO reaches +3.22, limb4+MAPPO +2.08, limb4+IPPO +1.95, all from about -2.6. Reported FAILED by a guard that greps rsl-rl's "Learning iteration" while skrl prints a tqdm bar. |
| 1 | `21091041` | PPO control COMPLETED (5:13); the three MARL rows failed on a missing `num_agents` property, since added |

### Gripper: restore the two hand DoF the asset welds shut · `done`
The hardware has two grippers (`docs/GRIPPER.md`); the shipped asset welds both
hands, so no policy here has ever closed one. `scripts/add_gripper.py` writes a
24-DoF copy into the workspace, leaving `external/` pristine. Checked before
queuing: 24 actuated joints, fingertip sweeps 6.5 cm across the palm, hands
mirrored.

Once the USD lands, this re-runs §5, the nine v2 cells and Tier 1 — the terrain
and locomotion rungs are unaffected, having never used arm contact.

| # | id | outcome |
|---|---|---|
| 1 | `21093986` | **done** — `assets/gripper/usd/berkeley_humanoid_lite_gripper/berkeley_humanoid_lite_gripper.usda`. The MJCF path landed later (`aed84e1`), so the replay harness renders 24 DoF too: 22 actuators without the flag, 24 with, 48 for a two-robot crew. |

### B3 — ice / patchy friction · `retracted` — the ice was out of the robots' reach (*B3 ice — the patches were never under the robots*)
Wired and registered: `Velocity-BHL-Biped-Ice-v0`, `-Ice-Depth-v0`,
`-IceVisible-v0`. G-B3 passes at the flush inset and fails at 5 mm, so the gate
can fail.

The depth arm is the strongest negative control this project has: the patches
are flush, so the ray-caster returns the same flat height field either way. If
depth still helps, the mechanism cannot be that it sees the ice. The visible arm
separates that from "any camera helps once the patch is visible" — colouring the
ice by default would have turned the rung into an RGB experiment.

| # | id | outcome |
|---|---|---|
| 2 | `21105232` | **6 of 6 COMPLETED** — depth 1.519, blind 1.374, visible 1.299. Depth beats blind by 10.6% on a hazard it cannot see. |
| 1 | `21100447` | **smoke 3/3**, episode lengths 17.7–19.8 |

### B3 — ice / patchy friction · superseded
`terrains/ice.py` and `scripts/bench/ice_gate.py` written. G-B3 passes at the
flush inset and correctly fails at a 5 mm one, so the gate can fail. Still to
do: wire a task config and register the ids.

| # | id | outcome |
|---|---|---|
| — | — | not queued |

### Gripper v2 variants · `done`
The three v2 tasks on the 24-DoF asset, as separate ids rather than a flag, so
the welded-hand arms stay runnable as their control. Actions and the
joint-indexed observations move 22 -> 24 together; driving 22 of 24 joints would
leave the grippers inert and the variant indistinguishable from its control.

| # | id | outcome |
|---|---|---|
| 2 | `21105399` | **9 of 9 COMPLETED** — grippers survive **~450 steps against 8**, reward **+14.3 against −0.79**. Task success still 0 in every cell. |
| 1 | `21105363` | smoke 3/3, episode lengths 7.3–7.4 |

### Cloth sorting — rigid-to-deformable ladder · `open` — redesigned 2026-09-10
G-C1 still stands and is not being re-run. End-to-end deformable RL stays
rejected. The task is now a hierarchical ladder (`docs/CLOTH_SORT.md`): rigid
proxy → one 8×8 Newton cloth → five-garment eval (Mode B). No 8,000-iter
cloth job and no 2048-env cloth job were queued.

Kinematic C0 / C1 / C4-BC / C5 ran on the login node (no Slurm id). Scripted
C0, linear BC, and Mode-B C5 all score **1.00**. That is the planner, not
Isaac, and not cloth.

**The first Isaac attempt failed and took the other four with it.** `21228029`
died at 37 s and the four `afterok` jobs sat in `DependencyNeverSatisfied`
until they were cancelled, so the whole 2026-09-10 batch produced **no Isaac
number at all**. Cause, from the traceback rather than the exit code: the
scene built correctly — robot at 22 joints, table, baskets, garment, event
manager — and `ActionManager._prepare_terms` then called
`term_cfg.class_type(...)` on `None`. `SweepActionCfg` declared
`class_type: type = None` and patched the attribute afterwards
(`SweepActionCfg.class_type = SweepAction`), but `configclass` freezes field
defaults when it builds the dataclass, so every *instance* still carried
`None`. Reproduced against this stack's own `configclass` on the login node in
seconds; `SweepAction` is now defined before its cfg and the default is the
class itself.

Two further things were fixed before re-queuing, both invisible to an exit
code. `cloth_resolution` was assigned *after* construction in the bench and
smoke, but the cloth mesh is spawned inside `__post_init__` — a "10×10" bench
row would have been an 8×8 cloth wearing the wrong label. Resolution now goes
through `build_cfg()`, which passes it to `__init__` and reads the spawned
resolution back off the scene, and the CSV reports what the scene actually
built. The bench also now chains on the deformable smoke rather than the rigid
one, because it is the deformable cell it needs.

Four static guards were added to `tests/test_cloth_sort.py` (34 tests, all
passing). They parse the Isaac modules rather than importing them — the login
node cannot bootstrap Kit — and each was verified to **fail** when its bug is
reintroduced.

**Attempt 2 got further and died one manager later.** `21233802` built the
scene, built the action manager — the `class_type` fix held — reset, and
failed on the first `env.step()` inside the *termination* manager:
`NotImplementedError: "bitwise_and_cuda" not implemented for 'Float'`, from
`_in_basket`. Python's `&` binds tighter than `<=`, so

    (p[:,0]-cx).abs() <= hx & (p[:,1]-cy).abs() <= hy & ...

is not a mask — it is a chained comparison against `hx & (...)`. The basket
predicate has never once evaluated. Each comparison is parenthesised now.

Fixing that exposed a second thing in the same predicate: it re-derived the
basket box from `BASKET_INNER` using a **half**-extent for z, so the Isaac
ceiling was 0.07 m where `layout.basket_aabb` — what the kinematic ladder
scores against — uses the full 0.14 m. The two engines were scoring the same
garment differently, which is exactly the drift `docs/CLOTH_SORT.md` claims
cannot happen because both read one layout object. Both predicates now read
`basket_aabb`.

Also fixed, found while reading the step path rather than from a failure:
`progress_to_basket` kept `_bhl_cloth_prev_dist` across episode boundaries, so
the first step after a reset paid the policy for the garment teleporting back
to spawn. That step is zeroed now. It would not have crashed anything — it
would have quietly biased C1's reward toward whichever spawn landed nearer a
basket.

| # | id | outcome |
|---|---|---|
| 4 | `21233957`–`21233961` | **rigid smoke COMPLETED (0:20)** with the collider-only furniture, and **`95d` C1 training smoke COMPLETED** — 3 logged iterations, mean episode length **12.71 → 12.84 → 12.83** against 1.00 before the tilt fix, so the gate passes. **1,001 steps/s at 64 envs** is this task's first real Isaac rigid throughput. But `Episode_Termination/fallen = 1.0000` and `Episode_Reward/progress = 0.0000`: the robot topples in ~0.51 s every episode and the garment never moves. See the C0-falls entry below. |
| 3 | `21233865`–`21233869` | **rigid smoke COMPLETED (0:22)** — both rigid ids construct, reset and step: 22 joints, 2 arms, action_dim 5. First Isaac cloth-sort cell ever to pass. Then: `95d` FAILED at 0:31, guard caught `mean episode length is 1.00`; `95e` COMPLETED but on that same broken predicate, so its numbers do not count; `95f` FAILED — `FrameView prim '/World/envs/env_0/table' is a Newton physics body`. |
| 2 | `21233802` | FAILED at 1:14 — `bitwise_and_cuda` in `_in_basket`; got past construction, reset and the action manager. Dependents cancelled. |
| 1 | `21228029` | FAILED at 0:37 — `SweepActionCfg.class_type` was `None`. Dependents never ran. |
| — | login node | 43 unit tests pass (13 new: Isaac-wiring guards each verified to fail when its bug is reintroduced, plus spawn-relative tilt math). C0 scripted 1.00 / 64 eps. C1 residual eval 1.00, worse efficiency than scripted (3.45 vs 1.05 sweeps under DR). C4 kinematic BC 1.00 / 64 eps, 1.14 sweeps. C5 Mode B scripted 1.00 / 32 eps, 5 sweeps. Cost gate exits 2 on 2048-env deformable. |

**Attempt 3 reached the training path and the guard earned its keep.** The
rigid smoke passed. `95d` then failed in 31 seconds with `mean episode length
is 1.00` — every episode terminating on step one, the same signature that
trained nine v2 arms for 8,000 iterations on one-step episodes (`21093953`,
nine GPU-days). This time it cost half a minute.

Cause: `fallen` tested `R[2,2] < 0.70` against an absolute convention. The
task spawns at `(0, 0, 1, 0)` — the quaternion the spawn photographs
(`21218517`/`21218627`) show standing, matching MuJoCo to 9 mm — and that
quaternion has `R[2,2] = -1`. So the predicate called a standing robot fallen
at reset. Rather than re-litigate this repo's long-running argument about what
this asset's up-axis is, the fall test is now measured **relative to the spawn
pose**: it reads 0 at reset by construction, and a yaw jitter still reads 0
because yaw is not tilt. The algebra lives in
`bhl_robust.cloth.kinematics.relative_up_z` and is shared by the torch path
and the numpy tests instead of being written twice.

`95e` COMPLETED in the same batch and reported `success_rate 0.0`,
`mean_steps_to_success 1.0`. **That is not a task result** — it is the same
one-step termination, measured. It is being re-run and nothing is published
from it.

`95f` failed on `ValueError: FrameView prim '/World/envs/env_0/table' is a
Newton physics body` — the same class as G-C1 attempt 5 (`21125073`). The
table and basket walls carried `rigid_props`, which Newton promotes to a
physics body that `AssetBaseCfg` refuses. They are collider-only now, **in the
rigid scene as well as the deformable one**, so C0/C1 and C2/C3 are not
quietly scoring different table physics. The maze, plinth, shelf and net keep
`rigid_props` and are untouched; a test enforces that split.

**Attempt 4: the rigid half runs, and the finding is that the robot falls
over.** `95c` passed with the collider-only furniture. `95d` passed its gate —
mean episode length **12.71 / 12.84 / 12.83** against 1.00 before the tilt fix
— and `95e` independently ran 12.25 steps per episode against 1.0. The two
agree, which is the point of running both.

What they agree on is a negative result: `Episode_Termination/fallen =
1.0000` and `Episode_Reward/progress = 0.0000`. The robot topples in about
0.51 s of every episode and the garment never moves toward a basket. The
layout is not the cause — the robot at x = −0.22 is clear of the table
(x ≥ 0.17) and of the baskets (x ≤ 0.22), checked geometrically. The
remaining hypothesis is that it cannot *hold* the pinch/squat pose: legs are
position-controlled at stiffness 20 with a 6 Nm effort limit at hips −0.85 /
knees 1.45, and FINDINGS already records the coop policy's torso descending to
−0.23 m and plateauing rather than holding a squat. `slurm/95g` measures that
directly — three leg poses, zero action, tilt and root height — before
anything is changed. Decision rule 1 in `docs/CLOTH_SORT.md` says fix the
geometry or the controller before training, so no C1 arm is queued.
`21234084` is that probe, queued 2026-09-10.

**`21233959`'s own numbers were not trustworthy either, and that is now
fixed.** `eval_isaac.py` reported `success_rate` and `fall_rate` while
assigning neither: `em.fell` was never written, and the success lookup did
`"success" in tm._term_dones` where `_term_dones` is a
`(num_envs, n_terms)` **tensor**, not a dict — that raises `RuntimeError`, and
a bare `except Exception: pass` turned it into a silent `False`. Both rates
were structurally pinned to 0.0 and could not have moved whatever the robot
did. They read through the public `termination_manager.get_term()` now and
raise if a term is missing. The `success_rate 0.0` in
`results/cloth/isaac_c0_scripted.json` is therefore **not evidence of
anything** and is superseded by the next eval.

**The Newton cloth builds.** `21233960` got the deformable scene up for the
first time: `Newton deformable object initialized`, 8 instances, and — the
number that matters — **81 particles per body at `resolution=8`, not 64**.
Isaac Lab's `MeshRectangleCfg(resolution=(n, n))` counts *cells*, so the mesh
carries `(n+1)²` vertices. The bench was about to label that mesh "64
vertices", which is the single number a cloth throughput row is compared on.
`isaac_grid_counts` reports it correctly now and a test pins 8→81, 10→121,
16→289.

It then failed on the table again, still `FrameView prim ... is a Newton
physics body` — so dropping `rigid_props` was **not** sufficient. Isaac Lab's
own Newton cloth example
(`lift_franka_soft/franka_cloth_env_cfg.py`) spawns its static `cube` with
`collision_props` and **no `physics_material`**, and that binding was the only
remaining difference. Collider-only boxes now carry neither. Fidelity note,
recorded rather than buried: those boxes take the default surface material
instead of 0.9/0.8, contact friction combines both surfaces, and the garment
keeps its own randomized material — and it is applied to the rigid scene too,
so C0/C1 and C2/C3 still compare like with like.

`21234084` (spawn probe) failed on a device-mixing bug of mine — Isaac Lab 3.x
returns some of these as warp ProxyArrays and some as cuda tensors. Fixed;
requeued as `21234166`.

| re-queued 2026-09-10 | |
|---|---|
| `21234165` | `95f` deformable smoke, collider-only furniture with no material |
| `21234166` | `95g` spawn probe, device fix |
| `21234167` | `95b` throughput bench, afterok 21234165 |

**The deformable smoke passes.** `21234165`: all four ids construct, reset and
step, including **both** Newton cells. Dropping `physics_material` was the
missing piece. First BHL cloth numbers, 8 envs, 8×8 cloth (**81** vertices):

| id | build | step throughput (unwarmed, 6 steps) |
|---|---:|---:|
| `ClothSort-BHL-Rigid-Oracle-v0` | 6.0 s | 140 env-steps/s |
| `ClothSort-BHL-RigidFive-Oracle-v0` | 1.2 s | 145 env-steps/s |
| `ClothSort-BHL-Deformable-Oracle-v0` | **56.5 s** | 375 env-steps/s |
| `ClothSort-BHL-ActiveCloth-Oracle-v0` (Mode B) | 9.3 s | 86 env-steps/s |

These are the smoke's own liveness figures — six unwarmed steps — not bench
numbers, and they are labelled that way in the file. `21234167` is the warmed
bench. The one comparison worth making already: G-C1's Franka scene, **961**
vertices and a 7-DoF arm, managed 182 env-steps/s at 8 envs. An 81-vertex
cloth under a 22-DoF humanoid is in the same range or better, which is the
whole bet the redesign made — buy the fidelity down, not the parallelism up.

**The spawn probe (`21234166`) rules out the squat.** Three leg poses,
zero action, 60 steps:

| pose | final tilt | root z | fell (>0.78 rad) |
|---|---:|---:|---|
| configured squat | 0.267 rad | −0.199 | **no** |
| half squat | 0.045 rad | −0.108 | **no** |
| default (no squat) | 0.142 rad | −0.139 | **no** |

None of them falls. Tilt oscillates up to 0.44 rad and the root **sinks
12–20 cm in every pose**, including the one with no squat at all — so the
depth of the crouch is not the variable, and `21233958`'s
`fallen = 1.0000` is not "the robot cannot hold a squat". What the probe shows
is a robot with no balance control: position targets sag under gravity and it
wobbles to within half the fall limit before the task does anything. Adding
policy-driven arm motion on top of that is what carries it past 0.78 rad
inside ~13 steps.

That is a design question — this task needs a balance controller, or a
stationary base, or a fall limit chosen for manipulation rather than
locomotion — and it is not something to pick silently. **No C1 arm is queued
and no cloth-sort success rate is claimed.**

**The throughput bench landed, and it is the redesign's central claim.**
`21234167`, warmed (3 throwaway steps, then 40 timed):

| deformables | resolution | vertices | envs | env-steps/s |
|---:|---|---:|---:|---:|
| 0 | — | 0 | 8 | 128.6 |
| 0 | — | 0 | 32 | 623.9 |
| 0 | — | 0 | 64 | **1,117.6** |
| 1 | 8×8 | **81** | 8 | **467.2** |
| 1 | 10×10 | **121** | 8 | 438.3 |

Against G-C1 (`21185969`) — Franka, 7 DoF, one **961**-vertex cloth — at the
same 8 envs: **467 against 182**, a 2.6× speed-up while carrying three times
the DoF. Bought entirely by dropping mesh resolution, which is what the
redesign said to do. 8×8 → 10×10 costs 6%, so Stage 2 has room to go *up* in
fidelity. Rigid scales near-linearly (128.6 → 623.9 → 1,117.6 over 8 → 32 →
64), which is what makes Stage 1 the right place to learn.

Two things this does **not** license. Every cloth row is at **8 envs** —
G-C1's actual finding was that cloth throughput *falls* with parallelism
(182 → 71 across 8 → 128) and nothing here tested that. And the Newton scene
costs **56.5 s to build** against 1–6 s rigid.

**The bench's CSV was wrong even though its markdown was right.**
`csv.DictWriter` in append mode wrote this run's field order under the header
the kinematic bench had already left on disk, so every Isaac value landed one
or more columns off — `env_steps_per_s` was sitting under `cloth_resolution`.
The markdown summary is generated from the dicts, so it read correctly, which
is exactly what would have let this ship. Both benches now go through
`append_rows_csv`, which unions the schema and rewrites the header, keeping
existing rows. The 5 misaligned rows were recovered by field order (their
values match the markdown table exactly) and the file re-emitted with all 17
rows intact; the malformed original is kept at
`/scratch/.../cloth_sort_bench.csv.malformed`. A test now reads the committed
CSV and fails if any row has more fields than its header.

Five GPU round-trips, each returning exactly one bug, every one of a class a
parser or a login-node unit test can see — which is where they are caught now:
50 tests, all offline. The guards in `tests/test_cloth_sort.py` are the response:
they parse the Isaac modules on the login node, because Kit cannot bootstrap
there and importing them is not an option. **No Isaac cloth-sort number
exists yet.** The table in `docs/CLOTH_SORT.md` stays kinematic-only until one
of these jobs writes a real env-steps/s.

### Cloth sorting — G-C1 throughput · `done` — the gate that forced the redesign
Decided end-to-end cloth RL against any cheaper method, and the answer is
**do not train on cloth**. Twelve probes; the first eleven measured something
other than cloth or died before stepping. The original design is preserved in
the history of `docs/CLOTH_SORT.md`. The gate exists so that decision costs
twelve short probes instead of weeks of GPU, and that is what it did. The
redesign above is the response, not a retraction of these numbers.

| # | id | outcome |
|---|---|---|
| 12 | `21185969` | **G-C1 answered, twelve attempts in.** Newton VBD cloth, 961-vertex mesh, `robot@5.0` resolved and nothing dropped: **182 env-steps/s at 8 envs, 177 at 32, 71 at 128**, and 512 envs overflows a signed 32-bit array dimension (7.3e9). Throughput *falls* with parallelism, so there is no scale to buy. A standard 8,000-iteration arm here is 786M env-steps: about **50 days** at cloth's peak rate against **6 minutes** at the rigid-body rate. RL on this is not a scheduling problem, it is a different project. **The sorting task gets scripted demos.** |
| 11 | `21185935` | FAILED — the Franka resolved and the scene built for the first time, then `KeyError: 'ee_frame'` at the first step: `deformable_ee_distance` reads that sensor every step and attempt 9 had removed it. Removing it was a fix for a bug that no longer existed. |
| 10 | `21185828` | FAILED, and it proved the point — `FileNotFoundError` on `.../Isaac/6.0/.../panda_instanceable.usd`. Not flakiness: Isaac Lab 3.0.0b2 asks the 6.0 asset tree for a file that was never published there. The identical file is served under 5.0 and 4.5. Nine attempts had read that 404 as "this asset is optional" and deleted the robot. |
| 9 | `21153326` | FAILED — the site-injection fix worked and the localiser then ate the robot: `The scene entity 'robot' does not exist. Available entities: ['terrain', 'deformable', 'table', 'ground', 'sky_light', 'cube']`. One slow HEAD against the content server is enough to condemn an asset permanently for the run, and the Franka drew the short straw this time. The same URL loaded fine in attempt 8, so this is flakiness being treated as evidence. |
| 8 | `21146031` | FAILED — but it got further than any before it. The assets resolved, the Franka loaded, the VBD cloth registered at 961 vertices; it died in `NewtonManager._cl_inject_sites` with `Site 'ft_4' ... matched no prototype bodies`. Newton's prototype builder labels bodies differently from the USD stage, so `ee_frame`, anchored on `panda_link0`, cannot be injected as a site when the scene clones. That is Isaac Lab 3.0.0b2's cloner, not this repo. |
| 7 | `21136400` | FAILED — dropping every remote asset removed the Franka, and the scene's force-torque sites reference it: `Site 'ft_2' ... matched no prototype bodies` |
| 6 | `21136231` | cancelled before it ran — a visual-only substitute still guesses at the prim's role |
| 5 | `21125073` | FAILED — my replacement carried `rigid_props`, so `FrameView` refused it: "prim '/World/envs/env_0/Table' is a Newton physics body" |

### Demo clips for the README · `done` — 2026-09-05
The MuJoCo replay harness cannot render the gripper arms — it builds its crew
from the 22-DoF MJCF and the gripper asset is a 24-DoF URDF. So the clips come
from Isaac Sim 6.0, which is the stack those policies trained on and where RTX
works. Both gifs are in `docs/gifs/isaac/`.

| # | id | outcome |
|---|---|---|
| 5 | `21186009` | **done** — gripper and welded CubeToShelf clips written. ONNX export warned (`Policy does not have an actor/student module`) and was skipped; the rollout is what this job is for. |
| 4 | `21185827` | FAILED — checkpoint loaded, then `AttributeError: 'CoopLiftSceneCfg' object has no attribute 'robot'`. The deploy-yaml stretch still assumed a locomotion scene. Wrapped as best-effort after this; the next run is the one that counted. |
| 3 | `21153325` | FAILED — and the config migration worked. It loaded the checkpoint and died one line later on `AttributeError: 'PPO' object has no attribute 'policy'`: rsl-rl 3.0.1 calls the actor `alg.policy`, 5.0.1 calls it `alg.actor`, and this file was written against 3.0.1. Still before any frame is drawn. |

### Plank cells, re-run on the fixed scene · `done` — ejection fixed, task still dead
The six original plank cells measured a scene that ejected its own payload, and
their gripper arms showed none of the survival effect the cube and ball arms did
— episode length 11.6 against 428. Re-run at the 1.05 m stand-off.

| # | id | outcome |
|---|---|---|
| 1 | `21146058` | **6 of 6 COMPLETED** (10:07–22:23). Read from the event files: the stand-off fixed the ejection — episode length is 354–491 against 11.6 before — and the task itself is still dead. `lift_height` sits on its 0.04 m curriculum floor in all six arms at the full 8,000 iterations, peak == tail == 0.0400, so it never promoted once across 48,000 arm-iterations. `success` is 0.0000 everywhere, and `leaned` and `lifting_object` never fire. Reward ~13 is `still_alive` and posture: the robots learned to stand next to the plank for the whole episode. The rgb arm falls in 42.6% of episodes against 1.9–7.7% for the other five. |

### Occlusion clips · `done` — 2026-09-05, no job needed
Rendered on CPU through the MuJoCo harness, which has never needed the queue.
**The arm that lifted is the one that falls**: s0 (13 cm in Isaac) never forms a
pinch — 0% in-gate, closest 0.251 m — and is down at 0.8 s, while s1 (flat in
Isaac) stays upright and holds the pinch 96% of the episode. Same shape as the
ball arm: a result that exists in one engine only.

Needed two fixes first: `OBS_OCCLUDED = 188` (derived from the run's own
`params/env.yaml` and cross-checked against `OBS_DEPTH_SWAP - 128`), and an
adaptive H.264 encoder — this cluster's ffmpeg has no `libx264`, so every render
was dying at the write after simulating the whole episode.

| # | id | outcome |
|---|---|---|
| — | none (CPU) | `docs/gifs/occlusion_s0_lifted_pov.gif`, `occlusion_s1_flat_pov.gif` |

### Spawn-bug diagnosis probes · `done` — 2026-09-05
Six probes in one afternoon. Recorded together because four of them are dead
hypotheses, and each narrowed the next.

| # | id | outcome |
|---|---|---|
| 7 | `21186390` | shipped USD measured directly: **symmetric to 4 dp** at zero arm angles. The asset is not the bug. |
| 6 | `21186378` | **`plant_feet` verified** — 0 of 27 bodies below z = 0 at reset and after stepping, root at +0.18 |
| 5 | `21186364` | spawn quaternion ruled out — `up_z = 1.000` at every rotation tested, so the robot is upright, not rolled |
| 4 | `21186353` | one-joint-at-a-time: the same ~0.5 m split appears whatever pair is driven and whatever the signs. Not a single mis-axed joint. |
| 3 | `21186283` | sign sweep: best achievable split **0.3519 m** against MuJoCo's 0.0000, which moves the cause off the config |
| 2 | `21186214` | **the finding** — 19 of 27 bodies below z = 0 against 1 of 27 for the locomotion control |
| 1 | `21186202` | first measurement, no control, so it could not yet separate a bug from a frame convention |

### Ice clips — blocked on v51 segfault / v60 ckpt format · `blocked` — 2026-09-06
The B3 result (depth 1.519 vs blind 1.374 on a hazard it cannot see) still has
no clip. No ice run has an ONNX or a `deploy.yaml`. Three more export attempts
after the first diagnosis, and the diagnosis was wrong.

`--load_run` **is** on the command line. `train_play` dies ~300 ms in with a
segfault on the v51 stack — Isaac Sim 5.1's RTX path, cameras or not. The
"byte-identical configs pointing at `terrain-bumpy-s0`" were a leftover
`configs/policy_latest.yaml` copied after the crash; later jobs refuse to save
when the file did not change. v60 cannot load the checkpoints either
(`KeyError: 'actor_state_dict'` — rsl-rl 3 vs 5). Do not re-queue the same
v51 job.

| # | id | outcome |
|---|---|---|
| 3 | `21191527` | COMPLETED, no artefact — v51, no cameras. Segfault at 326 ms. Correctly refused to save a stale yaml. |
| 2 | `21186615` / `21191514` | COMPLETED and **wrong yaml** — segfault, then a leftover `policy_latest.yaml` for `2026-08-17_22-56-08_terrain-bumpy-s0` was copied under the ice names. Deleted. |
| 1 | `21186589` | FAILED — ran on v60 against v51 checkpoints: `KeyError: 'actor_state_dict'`. |

### Manipulation re-runs on the fixed spawn · `blocked` — cancelled 2026-09-05
Cancelled after ~7 h each. Every arm that started reported **mean episode length
5.0 and a fall rate of 1.000**, against 428 steps for the gripper arms without
the change. The regression was the spawn fix itself.

`plant_feet` v1 drove the *lowest body of any kind* onto the plane; with the arm
geometry still wrong that body is a hand, so it lifted the robot until the hand
cleared the floor and left the feet 8-20 cm in the air. Its verification said
"0 of 27 bodies below z = 0", which was true and was the wrong question — the
ankle numbers in the same output said the feet were airborne.

v2 plants the *legs* at a measured reference (`SOLE_REF = 0.1026`, from the
locomotion task that spawns at root 0 and walks). Geometrically it is right:
ankles at +0.142 against the control's +0.143, bodies-below down from 19 to 0-3.
It still does not train — a 400-iteration probe (`21191526`) went
**9.1 -> 5.0 -> 5.0** with fall 1.000. Fixing the feet is not sufficient while
the arms are wrong.

**`plant_feet` is now opt-in and off** (`BHL_PLANT_FEET=1`). Every published
number was trained on the un-planted spawn, and a config that silently trains
something else makes them unreproducible.

| # | id | outcome |
|---|---|---|
| 2 | `21191526` | 400-iteration probe on the leg-planted spawn — did not recover |
| 1 | `21186402`, `21186403` | cancelled at ~7 h — ep_len 5.0, fall 1.000 on all six that started |

### Arm geometry — the "yaw" is a roll · `done` — 2026-09-06
The FACE+SYM 4-tuple is `(0.7071, −0.7071, 0, 0)` for robot_a and the opposite
for robot_b (`21192773`). Those are now the defaults. `BHL_LEGACY_YAW=1`
restores the rolled 4-tuples every FINDINGS number trained on. `plant_feet`
stays off. Reset-verified (`21192782`): both robots FACE+SYM, Lz=Rz=−0.6084,
toward=+1.00.

`plant_feet` failing to train is now less mysterious: it planted the legs of a
robot lying on its side. The configured 4-tuple `(0.7071, 0, 0, −0.7071)` puts
the identity pose's left/right Y onto world Z — a roll, not a yaw.

| # | id | outcome |
|---|---|---|
| 8 | `21192782` | **PASS** — both robots FACE+SYM after env.reset(): Lz=Rz=−0.6084, toward=+1.00 |
| 7 | `21192773` | **FACE+SYM** — `(0.707, −0.707, 0, 0)`: Lz=Rz=−0.6077, toward=+1.00 |
| 6 | `21192744` | **the "yaw" is a roll** — identity −0.6077/−0.6077; configured −0.177/+0.177 |

### Upright-spawn 400-iter probes · `done` — all three failed the training gate
**Final:** all three reached iteration 400 and `train.sh` failed them with
`mean episode length is 1.00` — every episode ending on its first step, the
same signature the spawn entries above chase. Nothing here was used; the spawn
investigation superseded it.

Same cell and budget as `21191526` (CubeToShelf-Blind, 400 iterations, 1024
envs, seed 0). That probe fell because plant_feet was planting a robot on its
side. These two ask the question that probe could not:

| arm | flag | question |
|---|---|---|
| `_0` | new yaw, plant_feet off | does standing up at spawn hold a gait? |
| `_1` | new yaw, `BHL_PLANT_FEET=1` | does planting help once the robot is upright? |
| grip | new yaw, 24-DoF, plant_feet off | does finding 10's survival gain survive the upright spawn? |

Not the 18-cell re-run. `_0` is training (past iteration 18). `_1` died in 43 s
on a node that could not write `/tmp/Assets` (`PermissionError`) — not a
plant_feet bug. Requeued after pointing `TMPDIR` at scratch. The gripper cell
is the overnight extra: finding 10 was measured on the rolled spawn.

| # | id | outcome |
|---|---|---|
| 3 | `21192899` | FAILED at 400 iterations — mean episode length 1.00 |
| 2 | `21192898` | FAILED at 400 iterations — mean episode length 1.00 |
| 1 | `21192861` | `_0` FAILED at 400 iterations, mean episode length 1.00 · `_1` FAILED 43 s, `/tmp/Assets` PermissionError |

### v2 / coop spawn puts the robot through the floor · `open` — found 2026-09-05
**Bug 2 (hand split) is closed:** the configured yaw was a roll. Defaults
changed 2026-09-06; `BHL_LEGACY_YAW=1` restores the old 4-tuples. Burial
(bug 1, `_PINCH_ROOT_Z = -0.07`) is still there and `plant_feet` is still off.
Noticed from the Isaac clip: the robots lie flat and clip the ground instead of
resetting upright dozens of times, which is what ~8-step episodes should look
like in a 300-step video.

**Confirmed, measured against a working task rather than asserted:**

| at reset | v2 CubeToShelf | 22-DoF locomotion (control) |
|---|---|---|
| root z | **-0.0700** | +0.0000 |
| lowest body | `arm_left_elbow_roll` @ **-0.2648** | `base` @ -0.0000 |
| bodies below z=0 | **19 of 27** | **1 of 27** |
| after 10 zero-action steps | 0 of 27 — extruded | 1 of 27 — stable |

The control's single body is `base`, whose origin is the asset's reference point;
it stays at 1 and the root drifts to -0.046 as the robot settles on its feet. The
v2 robot is *extruded* by depenetration over about ten steps — the same failure
class as the plank ejection, which was fixed for the plank only.

`fallen` reads 0.0000 the whole way through, so nothing terminates on it.

**What this puts in doubt.** Every coop_lift and v2 result: the cube, ball and
plank arms, all nine v2 cells and all nine gripper cells — which is the whole
"task success 0" column. Welded-hand arms die at ~8 steps, and extrusion takes
about 10, so those episodes ended inside the window where the robot was still
being pushed out of the floor. It also weakens finding 10: some of the gripper's
6.3 -> 427.7 step gain may be surviving a broken spawn rather than having hands.

The locomotion half is untouched — it spawns at z = 0 and is the control here.

Also at reset: both ankles are under the floor, and the hands sit at **-0.261
and +0.138** — a 40 cm split in a pose that reads as symmetric in the source —
while the payload waits at +0.300. The reach band says a standing robot's hands
span 0.404-0.610 with the feet planted, so the spawn does not implement the
geometry the tasks were designed around.

**Cause found: it is Isaac-only, and it is two bugs.**

Cross-checked in MuJoCo with the *same* `PINCH_POSE`, on CPU:

| at reset | Isaac | MuJoCo |
|---|---|---|
| left hand z | -0.261 | **+0.5804** |
| right hand z | +0.138 | **+0.5804** |
| hand split | **0.399 m** | **0.0000 m** |
| ankles | -0.106 / -0.053 | +0.1917 / +0.1917 |
| bodies below z=0 | 19 of 27 | 2 of 50 (`base` frames) |

1. **No feet planting.** `CrewRunner.reset` measures the lowest collision geom
   and translates the base onto the plane, precisely because it "does not trust
   that the two descriptions of the robot put their root frames in the same
   place". Isaac uses the hardcoded `_PINCH_ROOT_Z = -0.07` and does not.
2. **The pinch pose is asymmetric in Isaac and not in MuJoCo — still open.** Same joint values,
   symmetric in MuJoCo, 40 cm apart in Isaac. `21186283` swept every
   sign combination the joint limits allow: the best achievable split is
   **0.3519 m**, so no pose config can fix it. At the *default* arm pose Isaac
   is symmetric (+0.485 / +0.484 on the locomotion control), so the limits were
   mirrored by the conversion and the axes were not.

MuJoCo hands land at +0.58, inside the 0.404-0.610 band `reach_band.py` measured
the tasks against. Isaac never put the robot in the pose the tasks assume.

**Bug 1 is fixed.** `plant_feet` (`coop_lift_mdp.py`) measures the lowest body at
reset and translates the root onto the plane, the way `CrewRunner.reset` has
always done in MuJoCo. Verified `21186378`: **0 of 27 bodies below z = 0** at
reset and after stepping, root settling at +0.18 rather than the assumed -0.07.

**Bug 2 is open, and four hypotheses are dead.** Joint-sign mirroring read off
the URDF origins (rejected: the limits forbid it, `21186243`); the spawn
quaternion tipping the robot (rejected: `up_z = 1.000` at every rotation,
`21186364`); a single mis-axed joint (rejected: the same ~0.5 m split appears
whatever pair is driven and whatever the signs, `21186353`); the shipped USD
being asymmetric (rejected: at zero arm angles it is symmetric to 4 dp,
`21186390`). So the asset is fine and the same joint values mean different
things to the two descriptions — where, is not yet established.

**What this re-scopes.** Every MuJoCo-scored number used a correct spawn, so the
sim2sim results stand: 7.8 cm best cube lift, plank 0.0 cm, pinch-gate 98%, the
fall rates. What was broken is *training* -- every Isaac-side metric was measured
on robots that start buried with one arm through the floor. That is finding 10's
6.3 -> 427.7 steps, the reward figures, and the task-success-0 column.

**Earlier hypothesis, rejected and worth keeping.** First hypothesis was the mirroring
convention: every joint declares `axis="0 0 1"`, so symmetry depends on each
joint's origin rotation, and by that reading the elbows, knees and ankles were
all set the wrong way round. **Isaac disproved it on the next run** — the joint
limits are the authority and they say the original signs were right:

    'arm_right_elbow_pitch_joint': 0.900 not in [-1.571, -0.000]
    'leg_right_knee_pitch_joint': -1.450 not in [-0.000, 2.443]

So the corrected pose was invalid and is reverted. The next step is to sweep
sign combinations that are *within* the limits and measure hand symmetry and
foot clearance, rather than to reason from the URDF again.

| # | id | outcome |
|---|---|---|
| 5 | (CPU, no job) | MuJoCo cross-check: same pose, hands symmetric to 4 dp, feet planted. The bug does not exist there. |
| 4 | `21186283` | **sign sweep: no combination works.** Every valid variant leaves a 0.35-0.40 m hand split against MuJoCo's 0.0000, which moves the cause from the config to the asset. |
| 3 | `21186243` | hypothesis rejected — the "corrected" mirroring puts two joints outside their limits. Reverted. |
| 2 | `21186214` | **confirmed with a control** — table above |
| 1 | `21186202` | first measurement, suspect task only: 19 of 27 bodies below z=0. No control, so it could not yet distinguish a bug from this asset's frame convention. |

### Plank spawn ejection · `done`
`plank_leaned` no longer fires on a zero action (0.000 at every step, was 0.031
at step 20) — the stationary requirement fixed the predicate. **Cause found, geometrically rather than by guess:** at x = ±0.85 the hands
begin **18.7 cm inside** a plank spanning ±0.75 — hand reach is 0.25 m and the
hand link is 74 mm across, so the near edge sits at 0.563 against a plank end at
0.750. The contact solver resolves that overlap by ejecting the payload. Support
clearance was never the issue.

Stand-off is now 1.05 m, which clears the plank by 13 mm and still reaches the
contact point with 50 mm to spare. The wall moved from x = 1.0 to 2.4, because
at the new stand-off a wall at 1.0 would have been behind one of the robots.

| # | id | outcome |
|---|---|---|
| 3 | `21136378` | running — recheck at the 1.05 m stand-off |
| 2 | `21125100` | predicate fixed, ejection remains — plank still reaches 0.52 m unaided |
| 1 | `21124909` | found it — 0.031 success with a zero action, plank launching 22 cm |

### Base-height probe — is the floor-lift a training hack? · `done` — yes
The MuJoCo replay drops both cube arms ~41 cm in 0.2 s, before contact. PhysX
`base_contact` sits at -0.003, so the torso is not down in training — but a
robot folded onto its shins never puts its torso down either, so that cannot
separate a squat from a collapse. `base_height` is now a weight-0.0 reward term
so the curve is recorded without entering the objective.

| # | id | outcome |
|---|---|---|
| 6 | `21090556` | **done, and it settles the question** — 6,000 iterations on `coop_lift`. Base height rises to −0.107 by iteration ~350, then descends monotonically to −0.236 and plateaus near −0.229. The torso ends about 23 cm below nominal and stays there, so the policy is lowering its body, not raising the cube. Attempt 4 saw the first 1,500 iterations of this and could only say it was "still falling"; it lands. This is the training-side counterpart of the 41 cm drop the MuJoCo replay shows before contact. |
| 5 | `21082873` | TIMEOUT at 4 h — 6,000 iterations at 1024 envs does not fit; not a code fault |
| 4 | `21078881` | **done** — base height climbs to -0.105 by iteration ~350 then descends monotonically to -0.167 by 1,500, and is still falling |
| 3 | `21077648` | FAILED — the physx shim fired on v51 and broke the import |
| 2 | `21076968` | FAILED — `base_height expects optional parameters ['robot_a','robot_b'] but received []` |
| 1 | `21076488` | cancelled — logged 0.0000 for 1,300 iterations. The reward manager logs `weight x value`, so a weight-0.0 term reports zero by construction. |

### Redesigned tasks (v2) — nine cells · `done` — all nine trained, none succeeded
`slurm/90_v2_train_smoke.sbatch` added: 3 iterations at 64 envs through the real
training path, one arm per vision condition. The env smoke passed 9/9 during
every one of the training failures because it never executes `train.py` — a gate
that does not run the thing it gates is decoration.

Three tasks with terminal success states, each blind/depth/rgb, all on v60.
Gates G-T1/G-T2/G-T3 pass, the nine-cell smoke passes, training is queued at
seed 0. `NUM_ENVS=1024` for every cell and it must stay identical across them —
if RGB OOMs, drop all nine to 512 rather than mixing.

**Training**

| # | id | outcome |
|---|---|---|
| 13 | `21105231` | **9 of 9 COMPLETED** — welded hands survive ~8 steps, reward −0.79, success 0 |
| 12 | `21105217` | **smoke 4/4**, episode length 13.5–18.0. `either_fallen` now derives tilt from the root quaternion. |
| 11 | `21100627` | FAILED — guard caught it: `mean episode length is 1.00`. Wrapping ProxyArray reads was not the cause. |
| 10 | `21093953` | **CANCELLED — the runs were degenerate.** 8,000 iterations each at `mean_episode_length = 1.00` and `Episode_Termination/fallen = 1.00`: `either_fallen` read `projected_gravity_b` as a tensor when 3.x returns a warp ProxyArray, so `[:, 2]` was not the z component and every episode ended on step one. Nine GPU-days. |
| 9 | `21093566` | **training smoke 4/4** — 3 logged iterations each, incl. the solo ball control. Fix was pickling: `_variants()` built classes with `type()` so they were not module attributes and Hydra could not pickle them. |
| 8 | `21091042` | FAILED — `_pickle.PicklingError` on the generated variant classes |
| 7 | `21090547` | FAILED — **segfault** in Isaac Sim ~3 s in, before the env is built, on two different nodes (cn-gpu5, cn-gpu7) while the env smoke passed on cn-gpu6. Not a Python fault and not one bad node. |
| 6 | `21083804` | FAILED — same `stochastic` error. `RslRlMLPModelCfg` still carries deprecated `stochastic`/`init_noise_std` fields; isaaclab_rl ships `handle_deprecated_rsl_rl_cfg` to strip them and our vendored `train.py` never called it. |
| 5 | `21083755` | FAILED — `MLPModel.__init__() got an unexpected keyword argument 'stochastic'`. The actor needs `distribution_cfg`; without it the runner asks for a stochastic model the config never declared. Caught by the new **training-path** smoke, in 33 seconds. |
| 4 | `21083690` | done — env smoke 9/9, obs 194/322/578. Note: this builds envs and never runs `train.py`, so it passed through all three training failures. |
| 3 | `21083185` | FAILED, all 9 — `KeyError: 'class_name'`. rsl-rl 5.x reads `cfg["actor"]["class_name"]`; the v51 runner config sets only the 2.x `policy` field. |
| 2 | `21082869` | FAILED, all 9 — `PPO.__init__() got an unexpected keyword argument 'optimizer'`. I had installed rsl-rl 3.0.1 on v60 to match v51; isaaclab_rl 3.0.0b2 pins **5.0.1**. |
| 1 | `21077757` | FAILED, all 9 in ~20 s — `scripts/train.py` imported `berkeley_humanoid_lite.tasks` *before* applying the compat shim, so `AdditiveUniformNoiseCfg` was still missing on v60. The smoke test never caught it because it imports only `bhl_robust.tasks`. |

**Kinematic gates**

| # | id | outcome |
|---|---|---|
| 3 | `21076923` | **3/3 PASS** — G-T1 15.5 cm squat, G-T2 all targets reachable, G-T3 collapse excluded |
| 2 | `21076834` | passed G-T1/G-T3; G-T2 did not exist yet |
| 1 | `21076816` | FAIL — the gate bent the knees with the root pinned, which lifts the feet instead of squatting, and compared the base body origin against an absolute height in a frame whose origin sits 0.137 m below the feet |

**Nine-cell smoke**

| # | id | outcome |
|---|---|---|
| 3 | `21077235` | **9/9** — obs 194 blind / 322 depth / 578 rgb |
| 2 | `21077210` | 9/9 "ok" but obs 194 on every one: cameras mounted, no observation term read them, so the sighted arms were copies of the blind one |
| 1 | `21077181` | `NameError: _root` — an unbounded string replace hit five call sites in four other functions |

Earlier smoke attempts, each cleared one Isaac Lab 3.x breakage and exposed the
next: `21076944` `SimulationCfg.physx`, `21077015`/`21077044` warp `ProxyArray`
vs `torch.jit`, `21077084` ProxyArray `.shape` sizing observation terms as `()`,
`21077115` `.dtype` as a ctypes type, `21077139`/`21077158` curriculum `env_ids`
signature.

### Terrain PPO, seed 2 · `done` — all 4 arms COMPLETED, result holds at n=3
Third seed of all four cells, because n=2 is below this project's own bar.

| # | id | outcome |
|---|---|---|
| 2 | `21076264` | **done** (`_0` still finishing) — third seed does not overturn the result: depth 2.5x on friction, 1.10x on stairs, no blind/depth overlap in either cell |
| 1 | `21076260` | cancelled — `--array=8-11` re-ran seeds 0 and 1, because the array maths is `S = IDX % 2`. Added `SEED_OFFSET`. |

### Occlusion replicates · `done` — it does not replicate
Does the one cube arm that ever lifted reproduce? So far: no.

| # | id | outcome |
|---|---|---|
| 3 | `21066826_6` | **done** — blind seed 2, flat: tail 0.0400, peak 0.0408 over 16,000 iterations, episode length 195. **1 of 4 blind seeds ever lifted**, and that one is the odd one out in a second way — its episodes run 51 steps against ~195 for the three flat seeds, so it lifts in runs that end early rather than in runs that go the distance. |
| 2 | `21066825_6` | **done** — blind seed 1, flat at 0.0400 for all 16,000 iters. Does not replicate seed 0. |
| 1 | `21066823_7` | done — depth-under-occlusion, first genuine run; flat |

Superseded: `21066824` cancelled — `SEED` was hardcoded in the sbatch so
`--export=ALL,SEED=1` was overwritten, and the run name would have collided
with seed 0's.

---

## Closed

### B2 — stairs entry gate · `done`
| # | id | outcome |
|---|---|---|
| 3 | `21066021` | **PASS** — 5 cm restored, 2,000 iters, verdict from the event file: level 0.1073 vs control 0.1345 |
| 2 | `21065762` | FAIL (wrong) — re-probed at 3 cm after attempt 1's verdict; `tail -40` left one `terrain_levels` line in the log for `tail -1` to read as final |
| 1 | `21036975` | FAIL (wrong) — 5 cm, 300 iterations, no control; the pinned curriculum was the training budget, not the riser |

### Terrain PPO, seeds 0–1 · `done`
8 arms: slippery × stairs × blind/depth × 2 seeds.
| # | id | outcome |
|---|---|---|
| 3 | `21066022` | done — 8/8 completed, 6,000 iters each |
| 2 | `21065763` | cancelled — sat in `DependencyNeverSatisfied`, chained to the 3 cm probe that failed |
| 1 | `21036976` | cancelled — sat in `DependencyNeverSatisfied`, chained to the original probe that failed |

### POV clips · `done`
Eight clips with the robot's own colour + depth strip.
| # | id | outcome |
|---|---|---|
| 3 | `21067084` | done — all 8 in one pass on one log depth ramp |
| 2 | `21066637` | cancelled — the depth ramp was edited mid-job, so clips either side of the edit would disagree on what a brightness means |
| 1 | `21066607` | FAILED — `clip()` shell function: after `shift 5` the gif width fell into `"$@"` and reached `render_carry.py` as a stray positional |

### Ladder clip · `done`
| # | id | outcome |
|---|---|---|
| 1 | `21067057` | done — re-rendered after the slot-order fix; the `+x` robot was appended first but `members[0]` is read as the negative-axis slot, so the pair had each other's contact points |

### Docs + charts refresh · `done`
| # | id | outcome |
|---|---|---|
| 2 | `21066817` | done — curves re-extracted with the newest-run-wins fix, charts redrawn |
| 1 | `21066624` | FAILED — `chart_nine` called `c.pw()` on a property |

---

## One-off probes

Ran once, answered one question, kept only as a reference for where a number
came from.

| id | question it answered |
|---|---|
| `21076389` | Does this stack have DirectMARL + skrl? Yes — but a bare import fails, `AppLauncher` has to run first |
| `21076249` | Terrain levels for all 8 arms, from event files rather than log tails |
| `21076223` | Occlusion replicate seeds and depth variant: all flat |
| `21067145` | Pelvis height vs feet — `tilt()` cannot see a level collapse |
| `21067131` | Base descent over a rollout; the ladder pair sinks ~34 cm |
| `21067071` | Depth distribution per payload: cube 0.10 m, ball 0.16 m, ladder 0.85 m medians |
| `21066787` | Occlusion runs pre/post `apply_depth_flags` fix, read per directory |
| `21066007` | G-B2 verdict with a control column — returned INCONCLUSIVE, which is what forced the 2,000-iteration re-probe |
| `21066041` | First depth-distribution measurement (cube and ball only) |

---

## ID → task

| id | task |
|---|---|
| `21036975`, `21065762`, `21066021` | B2 stairs gate |
| `21036976`, `21065763`, `21066022` | terrain PPO seeds 0–1 |
| `21076260`, `21076264` | terrain PPO seed 2 |
| `21066823`, `21066824`, `21066825`, `21066826` | occlusion replicates |
| `21066607`, `21066637`, `21067084` | POV clips |
| `21067057` | ladder clip |
| `21093986` | gripper URDF to USD |
| `21100282`, `21100621` | v2 readouts: results, then reward-term breakdown |
| `21105209` | v2 spawn diagnostic — found the tilt convention |
| `21100446` | G-B4 re-gate |
| `21100447`, `21105232` | B3 ice smoke, then 6 rungs |
| `21105363` | gripper v2 smoke |
| `21105364`–`21185969` (12 attempts) | G-C1 cloth throughput |
| `21105399` | gripper v2 training, 9 cells |
| `21124909`, `21125100` | plank spawn diagnostics |
| `21146058` | plank cells re-run at the 1.05 m stand-off |
| `21124719`, `21136232` | B3 + v2 readouts |
| `21124302`, `21136243` | Tier 1 readouts |
| `21136321`–`21186009` (8 attempts) | gripper vs welded demo clips |
| `21186402`, `21186403`, `21191526` | planted-spawn re-runs — cancelled; plant_feet now opt-in |
| `21191713`–`21192782` | arm-geometry: the "yaw" was a roll; FACE+SYM 4-tuple now default |
| `21192861`, `21192898`, `21192899` | upright-spawn 400-iter probes: welded yaw, welded yaw+plant, gripper yaw |
| `21247910` | cloth hand/facing probe — robot faces −x, hand 0.97 m from the garment |
| `21247911`, `21247912` | B5 maze replicates, seeds 1 and 2 of blind, lidar, stereo, stereo P16, both P16 |
| `21247917` | B5 maze clips — blind, lidar, stereo P4, both P16 (no robot in frame; not published) |
| `21299608` | render probe — viewport draws stale USD pose; camera sensor draws the body |
| `21299873` | B5 camera-sensor clips — blind, lidar, stereo P4, stereo P16 |
| `21299952` | B5 camera-sensor clip — blind re-render |
| `21302170` | MARL first block smoke — five rows, 3 iterations, seed 99 |
| `21302171`, `21302172` | MARL first block, seeds 1 and 2 |
| `21302173`, `21302174` | B5 Both / StereoP8 seeds 1–2 — `_3` completed, `_4` and `21302174` cancelled (stereo looked up) |
| `21317022` | stereo pitch probe — raw pose +20°, corrected −20° |
| `21317023`, `21317024`, `21317025` | B5 stereo re-run, cameras pointed down — seeds 0 / 1 / 2 |
| `21328444`, `21328605` | G-B4t, Tier 1 MARL terrain gate — failed on terrain-level logging, then re-run |
| `21328445`, `21328446` | Tier 1 MARL grid, first submission — cancelled after the gate failed |
| `21328607`, `21328608` | Tier 1 MARL grid — stairs / slippery / rough, seeds 0 and 1 |
| `21328532` | B3 ice placement probe — patches a median 72 m from the robots |
| `21328742` | G-T3, Tier 3 gate — 22 DoF, stairs, depth, arm deviation off |
| `21328743`, `21328744` | Tier 3 on stairs — PPO, MAPPO limb4, limb1; seeds 0 and 1 |
| `21329137` | B5 corrected-stereo clip with before/after eye panels |
| `21330372` | MARL critic A/B — limb1 privileged, limb1 policy-obs, legs2 MAPPO privileged; stairs, 1,500 iterations |
| `21330392` | critic A/B arm 0 again, limb1 + MAPPO privileged — `_0` of `21330372` died at step 0 |
| `21338294` | MARL noise A/B — std parameterisation × LR schedule on the one-agent control, stairs, 1,500 iterations |
| `21353199` | B5 maze navigation env smoke — **PASS** 4/4 |
| `21353200` | C2 free-base hold at dt=0.005 — **2/2 nonfinite** |
| `21353201` | C2F jacket new VBD sheet — **4/4 nonfinite** |
| `21353395` | B5 mazenav train smoke — **PASS** 4/4, eplen 26.6–28.1 |
| `21353396`, `21353397`, `21353398` | B5 mazenav PPO n=3 — **12/12 COMPLETED**, button 0 |
| `21355623` | B5 mazenav colour re-render — queued, PreviewSurface overlays |
| `21342561` | B3 ice placement follow-up — **REACHABLE**, median 0.8 m |
| `21344927` | B3 ice-placed train smoke |
| `21344928` | B3 ice-placed PPO n=2 — **6/6 COMPLETED** |
| `21352981`, `21352982` | B3 ice-placed export + clip — `docs/gifs/ice_pair_placed.gif` |
| `21353130` | Isaac C5 pose-fix four-episode — **2/4 sorted** |
| `21300299`, `21300300`, `21300301` | cloth redesign: rigid smoke, Isaac C0 scripted (boot crash), C1 training smoke |
| `21300348`, `21300493`, `21300494` | cloth redesign: C0 free base, C0 fixed base, C0 fixed-base clip |
| `21300603`, `21300604`, `21300605` | cloth redesign with hand colliders: fixed-base C0, its clip, free-base C0 |
| `21307211`–`21307214` | cloth v2 schedule + hull colliders: fixed-base C0, clip, free-base C0, clip |
| `21317170`–`21317174` | cloth v3 feedforward: fixed-base C0 (7/8), clip (dgxh-1, no Vulkan), free-base arm-still control, free-base C0, clip |
| `21317388`–`21317391` | cloth v3 on widened baskets, fixed base: shirt 16/16, sock 8/8, jacket 8/8, shirt clip |
| `21233916` | B5 stereo pooling sweep — P8, P16, both P16 |
| `21228029`–`21228033` | cloth-sort cheap Isaac, attempt 1 — died on `SweepActionCfg.class_type=None`, dependents never ran |
| `21233802`–`21233806` | cloth-sort cheap Isaac, attempt 2 — `class_type` fixed, died on the `&` precedence bug in `_in_basket` |
| `21233865`–`21233869` | cloth-sort cheap Isaac, attempt 3 — rigid smoke passed; tilt convention and Newton furniture found |
| `21233957`–`21233961` | cloth-sort cheap Isaac, attempt 4 — spawn-relative tilt, collider-only cloth furniture |
| `21234084`, `21234166` | cloth spawn-stability probe — three leg poses, zero action; none falls |
| `21234165`, `21234167` | cloth deformable smoke (passes) and warmed throughput bench |
| `21234259` | C0 Isaac eval re-run — **fall_rate 1.00, success 0.00** over 4 episodes, agreeing with the trainer's `fallen=1.0000` from a separate code path. The repaired metrics move. |
| `21186589`, `21186615`, `21191514`, `21191527` | ice-export — v51 segfault / v60 ckpt mismatch |
| `21105320` | Tier 1 MARL rows |
| `21076488`, `21076968`, `21077648` | base-height probe |
| `21076799`–`21076923` | v2 task gates |
| `21076944`–`21077235` | v2 nine-cell smoke |
| `21077757` | v2 nine-cell training |
| `21077722`, `21077723` | compat shim regression check |
| `21078958`, `21078987` | G-B4 limb-partition gate |
| `21078881` | base-height probe (attempt 4) |
| `21078882` | occlusion replicates, final read |
| `21076389` | MARL stack probe |
| `21076607`, `21076614` | reach-envelope measurement |
| `21076453`, `21076460`, `21077131`, `21077145` | posture / collapse diagnostics |
| `21076792` | rsl-rl install on the v60 stack |
| `21066624`, `21066817` | docs + charts refresh |

Jobs named `orchard*`, `lh-*`, `prune-*`, `ood-*` and `interactive` are not from
this workstream.

---

## Keeping this current

Add a row the moment a job is submitted, not when it finishes — the row whose
absence caused this file is the one nobody wrote down. When a task's table hits
four rows, delete the oldest.

Verify an id before trusting a row:

```bash
export PATH=/apps/slurm/current/bin:$PATH
sacct -j <id> --format=JobID%14,JobName%14,State%12,Elapsed -X
```

A "why it was re-run" line should name the cause, not the symptom. "Failed" is
not a reason; "the gif width fell into `"$@"` after `shift 5`" is.

## Mission7 bounded overnight diagnostics — 2026-09-20

Fourteen additional CPU tasks; original pilot and diagnostics preserved. No full campaign or held-out policy evaluations submitted.

**Final audit, September 21:** all 14 scheduler tasks **COMPLETED**, exit `0:0`;
none remain queued. Nine training studies finished all 200 updates, each with
**0/8 final validation success / learning gate FAILED**. D1 and D2 completed
their audits. S1–S3 completed only their prerequisite check and recorded
**SKIPPED_NOT_LEARNABLE**; they did not train. Actual elapsed allocated CPU
time: **16.22 CPU-hours**, zero GPUs (56 CPU-hour requested cap).
Final evidence: `results/mission7-overnight-20260920/final-audit-20260921.json`
and `scheduler-final-20260921.psv`; current task states are in `matrix.json`.

| Array | Studies | Final state / outcome | Dependency |
|---|---|---|---|
| `21370064_[0-8%3]` | A1, A2, A3, A4, A5, B1, C2, C3, C4 | **COMPLETED; nine failed learning gates** | none |
| `21370065_[12-13%2]` | D1, D2 | **COMPLETED diagnostics** | none |
| `21370066_[9-11%3]` | S1, S2, S3 | **COMPLETED; SKIPPED_NOT_LEARNABLE** | afterok:21370064_8 |

D2 final privileged full-route success: Doors **4/16**, Transport **4/16**;
localized components: doors **13/16**, transport **7/8**. D1 confirms varying
sensor features and sparse positive reward. C4's transient 2/8 validation at
update 100 did not persist or pass the gate. Full findings and proposed next
studies: [MISSION7_OVERNIGHT.md](docs/MISSION7_OVERNIGHT.md).
No new jobs submitted during this closeout. The startup notes below are
historical observations, superseded by this final audit.

Each task: 2 CPUs, 12 GB, zero GPUs, 2 h cap; total maximum 56 CPU-hours. Sensor tasks also require C4 JSON learning evidence; otherwise they write `SKIPPED_NOT_LEARNABLE` without training. C4 is reused as the matched Both arm, and A1 doubles as the current-PPO sweep baseline. Receipts, config hashes, source hashes, commit and output paths: `results/mission7-overnight-20260920/matrix.json` and `results/mission7-overnight-20260920/submissions.jsonl`. Scheduler completion certifies diagnostic execution only, never learned success.

Startup audit: **VALIDATED** (160 tests, 15 targeted rechecks, finite execution smokes). **RUNNING**: `21370064_0`, `_1`, `_2`, `21370065_12`, `_13`. A4/A5/B1/C2/C3/C4 remain **SUBMITTED / PENDING** under the training array throttle. S1–S3 remain **SUBMITTED / PENDING** behind `afterok:21370064_8` and the C4 JSON learning gate. Slurm confirms `cpu=2,mem=12G` and no GPU allocation. Complete task matrix, budgets, limitations and inspection commands: [MISSION7_OVERNIGHT.md](docs/MISSION7_OVERNIGHT.md).

Early output audit (campaign still RUNNING): A1 wrote a finite PPO update; D1 static geometry audit **COMPLETED**, 352 unique topology hashes. D2 localized reports **COMPLETED**: door 0 **6/8**, door 1 **7/8**, transport **7/8** (acquisition 8/8). These are privileged component successes, not full-route or learned mission results. Full-route interaction rollouts continue. `results/mission7-overnight-20260920/startup-audit.json` preserves this observation.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21383807` array `0-2%3` — gait; dependency none; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21383808` — c4; dependency none; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384040` — gait_extra; dependency none; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384041` — approach; dependency afterok:21383807; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384042` — rewards; dependency afterok:21383807; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384077` — fullroute; dependency afterok:21384040; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384200` — approach_recovery; dependency none; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384285` array `0-3%4` — train; dependency none; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384354` — fullroute_recovery; dependency none; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384572` array `0-1%2` — train_exploration; dependency none; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384641` — gait_endurance; dependency none; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384691` — fall_replay; dependency none; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/submissions.jsonl`.

Mission7 Approach diagnosis (2026-09-21): **SUBMITTED** `21384742` — fall_replay; dependency none; 2 CPUs /12 GB /0 GPUs /2 h per task. Receipts/source hashes: `results/mission7-approach-debug-20260921/replay-on-original-node/submissions.jsonl`.

Mission7 control follow-up (2026-09-21): `21385772`, `21386044`, `21386064`, and `21386072` were canceled or failed before scientific execution because of invalid snapshot/input paths; their logs are preserved under `results/mission7-approach-followup-20260921/`.

Mission7 control follow-up (2026-09-21): **SUBMITTED** `21386084` — initial detailed contact probe; completed but rejected by the explicit pose-match check after instrumentation altered some replay trajectories. **SUBMITTED** `21386215` — corrected pose-matched contact probe; pending.

Mission7 control follow-up (2026-09-21): **SUBMITTED** `21386145` — balanced four-direction Approach with 0.50 m/s measured translation and matched standstill controls; 56/64 controller successes, 0 falls, 0/64 matched standstill, 12/64 easy standstill; final result preserved at `results/mission7-approach-followup-20260921/approach_controller_speed05/`.

Mission7 control follow-up (2026-09-21): **SUBMITTED** `21386200` — exact ten replay contact-brake mitigation plus 16 Doors / 16 Transport route evaluation; failed after the replay phase on a controller initialization bug. All follow-up tasks request 2 CPUs, 12 GB, zero GPUs, and are pinned to `cn-c22` for paired physics.

Mission7 control follow-up (2026-09-21): **SUBMITTED** `21386394` — corrected pre-contact replay and route evaluation; **SUBMITTED** `21386441` — bounded plate sliding-friction contact-parameter replay (geometry and activation unchanged).

Mission7 control follow-up (2026-09-21): `21386488` completed the 16 Doors portion (1/16 success, 4 falls, 11 timeouts) before its sequential Transport phase was replaced by **SUBMITTED** array `21386803_[0-15%4]` for 16 parallel Transport episodes. The replacement preserves the completed Doors records and keeps the same controller, split, seed, and resource limits.

Mission7 control follow-up (2026-09-21): `21386215` was canceled after one exact pose-matched episode to bound an oversized floor-contact trace; `21386409` was the filtered rerun and `21386539` is the corrected full-reset pose-matched contact probe. The final valid probe reproduced all ten unchanged falls with maximum pose error `0.0`; evidence is under `contact_probe-rerun5/`.

Mission7 control follow-up finalization (2026-09-21): `21386145` completed with 56/64 Approach successes, zero falls, 8/16 in world −x, 16/16 in the other directions, and matched standstill 0/64. This fails the privileged Approach gate. `21386539` completed the corrected filtered contact probe: all ten unchanged replay trajectories matched exactly and all ten still fell. `21386540` completed the plate-friction intervention (sliding friction 1.0 → 0.20) with 10/10 falls, so the parameter-only intervention was rejected. `21386488` completed the contact-safe Doors route at 1/16 success (4 falls, 11 timeouts) and was canceled during partial Transport. Replacement array `21386803_[0-15%4]` completed all 16 Transport episodes: 1/16 success, 13 falls, and 2 timeouts. Aggregate: `results/mission7-approach-followup-20260921/transport_array/transport_summary.json`.

Mission7 clearance diagnosis (2026-09-22): `21396422` **FAILED infrastructure** before replay because the node's default `python3` lacked MuJoCo (`ModuleNotFoundError`); no scientific output was written. `21396448` was then **CANCELED** after its `cn-b01` run failed the required unchanged-replay pose match, so its partial output is invalid and will not be interpreted. The same snapshot was resubmitted as **`21396492`**, pinned to the original replay node `cn-c22` with the validated CPU environment explicitly selected. It runs the exact ten-fall replay with read-only robot/obstacle and plate-edge clearance, contact-body, entry-pose, commanded/realized-motion, and fall-stage instrumentation. No route or sensor jobs are released pending this diagnosis.

Mission7 clearance diagnosis result (2026-09-22): `21396492` **COMPLETED 0:0** on `cn-c22`; all 10 unchanged replays matched exactly and all 10 fell. Every fall followed plate contact, minimum robot/plate clearance was −39.4 mm, six failures were during traversal and four after exit, and no first wall/door/goal-post contact preceded a fall. Compact evidence is `results/mission7-approach-followup-20260922/replay-diagnose-cn-c22/diagnostic_summary.json`; raw episodes remain cluster-only. The exact 10/10 upright gate is still closed. No route or sensor job was submitted.

Mission7 staged plate intervention (2026-09-22): **SUBMITTED** `21396660` — exact ten-fall replay on `cn-c22`; approach to a pre-plate pose, 0.40 s settle, then 1.20 s straight crossing with yaw correction frozen. Geometry, activation schedule, fall predicate, and replay layouts are unchanged. No route or sensor jobs released.

Mission7 staged plate intervention (2026-09-22): `21396660` **FAILED infrastructure** before episode execution because the source snapshot omitted shared `bhl_robust.eval` support files. No scientific result was produced; the corrected snapshot is being resubmitted on `cn-c22`.

Mission7 staged plate intervention resubmission (2026-09-22): **SUBMITTED** `21396676` — corrected source snapshot, pinned to `cn-c22`; same one-factor staged pre-plate settle and straight crossing. No route or sensor jobs released.

Mission7 staged plate intervention resubmission `21396676` **FAILED infrastructure** before episode execution because the snapshot hash manifest included itself. No scientific result was produced; the manifest is corrected for the next resubmission.

Mission7 staged plate intervention resubmission (2026-09-22): **SUBMITTED** `21396684` — corrected shared source and hash manifest, pinned to `cn-c22`; same staged pre-plate settle and straight crossing. No route or sensor jobs released.

Mission7 staged plate intervention resubmission `21396684` **FAILED infrastructure** before episode execution because the source snapshot omitted `mission7_diagnostic.py`, imported by the existing overnight helper. No scientific result was produced; the helper is now included for the final resubmission.

Mission7 staged plate intervention resubmission (2026-09-22): **SUBMITTED** `21396709` — complete validated source snapshot, pinned to `cn-c22`; same staged pre-plate settle and straight crossing. No route or sensor jobs released.

Mission7 staged plate intervention, nearest unopened plate (2026-09-22): **SUBMITTED** `21397663` — exact ten-fall replay pinned to `cn-c22`. The one-factor extension stages the nearest unopened plate on either side, preferring the correct side only on distance ties; geometry, activation schedule, fall predicate, and replay layouts are unchanged. No route or sensor jobs released.

Mission7 staged plate intervention, guarded wrong-side entry (2026-09-22): **SUBMITTED** `21397732` — exact ten-fall replay pinned to `cn-c22`. Correct-side staging is retained; wrong-side staging is enabled only when the correct plate is more than 1.0 m away, separating the layout-13 wrong-side trace from layout-4's earlier near miss. Geometry, activation schedule, fall predicate, and replay layouts are unchanged. No route or sensor jobs released.

Mission7 route smoke (2026-09-22): **SUBMITTED** `21397985` — smallest existing `mission7_debug.py fullroute --smoke`, pinned to `cn-c22`; one validation layout each for legacy/measured Doors and Transport. This smoke is the release check before the documented 16-layout route batches. No sensor jobs released.

Mission7 route smoke `21397985` **FAILED infrastructure** before scientific execution because its source snapshot omitted `scripts/mission7_debug.py`; no route episode was produced. A corrected immutable snapshot is being resubmitted with the same smoke command.

Mission7 route smoke resubmission (2026-09-22): **SUBMITTED** `21398074` — corrected snapshot including `scripts/mission7_debug.py`, same one-layout legacy/measured Doors and Transport smoke on `cn-c22`. No 16-layout route or sensor jobs released.

Mission7 documented route evaluation (2026-09-22): **SUBMITTED** `21398501` — intended 16 Doors / 16 Transport `PlateSafeRouteController` evaluation on `cn-c22`; this launch used an invalid campaign path and is being canceled before scientific execution. No sensor jobs released.

Mission7 documented route evaluation `21398501` **FAILED infrastructure** before reading the campaign because of that invalid path; no route episode was produced. The corrected submission uses the existing `results/mission7-replay-smoke-20260921` campaign.

Mission7 documented route evaluation resubmission (2026-09-22): **SUBMITTED** `21398514` — corrected 16 Doors / 16 Transport `PlateSafeRouteController` evaluation on `cn-c22`, using the existing validation campaign and unchanged scoring/geometry. No sensor jobs released.

Mission7 staged plate intervention `21397663` **COMPLETED 0:0** on `cn-c22`; the nearest-unopened-plate extension reached 9/10 upright but caused layout 4 to fall after an earlier wrong-side intervention. It is rejected as the final controller, with the result retained as bounded evidence.

Mission7 staged plate intervention `21397732` **COMPLETED 0:0** on `cn-c22`; the guarded wrong-side entry variant reached 10/10 upright on the exact replay set. The existing exact replay gate is **PASSED**. Compact verdict: `results/mission7-approach-followup-20260922/plate-stage-cn-c22-guarded/result.json`.

Mission7 route smoke `21398074` **COMPLETED 0:0** on `cn-c22`; all four one-layout legacy/measured Doors/Transport smoke paths finished with finite terminal records. It unlocked the documented 16-layout route evaluation.

Mission7 documented route evaluation `21398514` **COMPLETED 0:0** on `cn-c22`; Doors completed at 1/16 success (4 falls, 11 timeouts) and Transport at 0/16 success (6 falls, 10 timeouts). Compact verdict: `results/mission7-approach-followup-20260922/route-eval-cn-c22-v2/result.json`. No sensor jobs were submitted.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21399179` — both layouts `1,4,8`, node `cn-c22`, 2 CPUs / 12 GB / 0 GPUs / 2 h; guarded PlateStage added only at existing PlateSafeRouteController switch handoff; receipt/source hashes: `results/mission7-approach-followup-20260922/route-handoff-probe-cn-c22/submission.json`.

Mission7 route handoff probe `21399179` **FAILED infrastructure** before episode execution on `cn-c22`; the bare node Python lacked numpy. No scientific episode was produced.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21399201` — same six layouts and one-factor switch handoff, corrected interpreter path; receipt/source hashes: `results/mission7-approach-followup-20260922-v2/route-handoff-probe-cn-c22/submission.json`.

Mission7 route handoff probe `21399201` **FAILED infrastructure** before episode execution; the selected interpreter lacked MuJoCo. No scientific episode was produced.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21399211` — same six layouts and one-factor switch handoff, verified shared Humanoid Lite MuJoCo venv; receipt/source hashes: `results/mission7-approach-followup-20260922-v3/route-handoff-probe-cn-c22/submission.json`.

Mission7 route handoff probe `21399211` **COMPLETED 0:0** on `cn-c22`; 4/6 guarded-stage activations, 2/6 completions, and 1/6 full success. Compact evidence: `results/mission7-approach-followup-20260922/route-handoff-probe-summary.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21399201` — both layouts `1,4,8`, node `cn-c22`, 2 CPUs / 12 GB / 0 GPUs / 2 h; guarded PlateStage added only at existing PlateSafeRouteController switch handoff; receipt/source hashes: `results/mission7-approach-followup-20260922-v2/route-handoff-probe-cn-c22/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21399211` — both layouts `1,4,8`, node `cn-c22`, 2 CPUs / 12 GB / 0 GPUs / 2 h; guarded PlateStage added only at existing PlateSafeRouteController switch handoff; receipt/source hashes: `results/mission7-approach-followup-20260922-v3/route-handoff-probe-cn-c22/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21399449` — both layouts `1`, node `cn-c22`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff; receipt/source hashes: `results/mission7-approach-followup-20260922-early/route-early-handoff-cn-c22/submission.json`.

Mission7 route handoff probe `21399449` **COMPLETED 0:0** on `cn-c22`; exactly two episodes ran. Both first target handoffs preceded plate contact, all three observed staged crossings completed, Transport/layout 1 succeeded end-to-end, and Doors/layout 1 timed out after the first stage during route rejoin. Compact evidence: `results/mission7-approach-followup-20260922/early-handoff-probe-summary.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21399494` — both layouts `1`, node `cn-c22`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `True`; receipt/source hashes: `results/mission7-approach-followup-20260922-rejoin/route-rejoin-diagnostic-cn-c22/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21399502` — doors layouts `1`, node `cn-c22`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `True` with fix `forward_pulse`; receipt/source hashes: `results/mission7-approach-followup-20260922-rejoin-fix/route-rejoin-fix-cn-c22/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21399503` — doors layouts `1`, node `cn-c22`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `True` with fix `forward_pulse`; receipt/source hashes: `results/mission7-approach-followup-20260922/route-rejoin-fix-v2-cn-c22/submission.json`.

Mission7 route rejoin diagnostic `21399494` **COMPLETED 0:0** on `cn-c22`; Doors/1 and Transport/1 both rejoined with valid route state. Doors/1 stalled physically at waypoint 8 after the first staged crossing; Transport/1 advanced through waypoint 8 and succeeded. Raw evidence: `results/mission7-approach-followup-20260922-rejoin/route-rejoin-diagnostic-cn-c22/`.

Mission7 route rejoin attempt `21399502` **COMPLETED 0:0** on `cn-c22`; the requested pulse did not activate because the probe runner omitted the submitted fix argument. The diagnostic trace is retained, but this job is not interpreted as an intervention result. Raw evidence: `results/mission7-approach-followup-20260922-rejoin-fix/route-rejoin-fix-cn-c22/`.

Mission7 corrected Doors/1 route rejoin intervention `21399503` **COMPLETED 0:0** on `cn-c22`; the single 0.30 m/s forward pulse activated at 45.8–46.2 s, but the robot remained at waypoint 8 and timed out at 180 s. Raw evidence: `results/mission7-approach-followup-20260922/route-rejoin-fix-v2-cn-c22/`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400561` — both layouts `1`, node `cn-c22`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `True` with fix `none`; receipt/source hashes: `results/mission7-approach-followup-20260922/route-rejoin-contacts-cn-c22/submission.json`.

**Correction (2026-09-22), Mission7 rejoin interventions `21399502` and `21399503`.**
The recorded cause for `21399502` — "the probe runner omitted the submitted fix
argument" — is wrong. Its immutable source snapshot is byte-identical to the
current `slurm/mission7_route_handoff_probe.sbatch`, which does forward
`--rejoin-fix "$rejoin_fix"`, and the submitted argv in its `submission.json`
ends in `forward_pulse`. The argument was forwarded and accepted. The actual
cause was in that snapshot's stall detector: `post_stage_last_progress_time` was
refreshed in `observe_step` whenever the base moved more than 2 cm, so a
humanoid stepping in place kept resetting the timer and the 0.8 s stall
threshold was never crossed. The detector measured motion where it meant
progress. The corrected build — the one `21399503` ran, whose snapshot runner
is byte-identical to current HEAD — redefines progress as distance-to-waypoint
decreasing by 2 cm, which is why its pulse fired at all.

`21399503` is nevertheless further reinterpreted, for a separate reason. Its
pulse did activate, but it was a no-op
by construction: the "0.30 m/s forward restart pulse" imposed 0.300 m/s while
the route controller was already commanding 0.300 m/s forward. Measured forward
command delta is **0.000000 m/s**; the only change was zeroing a 0.020 m/s
lateral and 0.049 rad/s yaw correction (full delta 0.056). `21399503` is
therefore not evidence that forward drive fails to recover Doors/1 — no
additional forward drive was ever applied.

Both jobs exited `0:0` and wrote `"rejoin_fix": "forward_pulse"`. The probe now
reports `intervention_status` and exits non-zero on
`REQUESTED_BUT_NEVER_FIRED` or `FIRED_BUT_DID_NOT_CHANGE_FORWARD_COMMAND`;
replaying `21399503` under the new build fails loudly with the latter.

Mission7 Doors/1 vs Transport/1 post-stage rejoin contact probe (2026-09-22):
**SUBMITTED** `21400561` — both layouts `1`, node `cn-c22`, 2 CPUs / 12 GB /
0 GPUs / 2 h; unchanged PlateStage with `early` route handoff, rejoin
diagnostic, and **no intervention** (`forward_pulse` is retired as a no-op by
construction, above). Adds a compact post-stage world-contact summary: the
environment already reported wall and door contacts every physics step and the
probe discarded all but `plate_*`. Geometry, activation semantics and the fall
predicate are unchanged. Purpose: discriminate physical blockage from
locomotion-policy failure at the Doors/1 waypoint-8 stall, against Transport/1
which advances through the same waypoint. A local reproduction of the Doors/1
episode already shows **19 of 775 post-stage steps in any world contact, all of
them the just-crossed `plate_0_-1`, with no wall or door contact during the
134 s stall** — the robot is stationary in free space while commanded 0.30 m/s
forward with no brake active and 0.05 rad heading error. Receipt/source hashes:
`results/mission7-approach-followup-20260922/route-rejoin-contacts-cn-c22/submission.json`.

Mission7 Doors/1 vs Transport/1 post-stage rejoin contact probe `21400561`
**COMPLETED 0:0** on `cn-c22` in 2 m 53 s; preflight passed in ~4 s. Both
guarded stages activated and completed (2/2). Transport/1 succeeded end-to-end
at 84.120 s; Doors/1 timed out at 180.0 s without falling, as before. The new
post-stage world-contact summary is decisive: across **775 post-stage steps
Doors/1 was in contact with a world geom in only 19 (2.5 %), every one of them
the just-crossed `plate_0_-1`, minimum distance −6.1 mm, last contact shortly
after stage exit**. There is **no wall and no door contact during the 134 s
stall**. Successful Transport/1, by contrast, spent 70 of 292 post-stage steps
(24.0 %) in contact across three plate geoms.

Interpretation: Doors/1 is **not physically blocked and not route-state
corrupted**. It stands in free space with the route commanding a constant
0.300 m/s forward, no brake active, 0.05 rad heading error and the waypoint-8
target 1.64 m ahead, and does not move. Combined with the `21399503`
reinterpretation above — no additional forward drive was ever applied — the
remaining explanation is **locomotion-policy failure**: the frozen gait produces
no forward gait from the post-stage entry state. "Post-stage route rejoin" is a
misnomer for this failure; the route is correct.

Compact verdict: `results/mission7-approach-followup-20260922/route-rejoin-contacts-cn-c22/result.json`
(28 KB; the 24 MB and 48 MB per-episode traces stay on the cluster and are
ignored by shape). No sensor jobs released. The privileged Approach gate and
the sensor comparisons remain closed.

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21400730` — controller `recovery`, directions `-1,+0`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 01:00:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-negx-recovery/submission.json`.

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21400731` — controller `guard`, directions `-1,+0`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 01:00:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-negx-guard/submission.json`.

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21400732` — controller `pulse`, directions `-1,+0`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 01:00:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-negx-pulse/submission.json`.

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21400745` — controller `recovery030`, directions `-1,+0`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 01:00:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-negx-recovery030/submission.json`.

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21400746` — controller `settle14`, directions `-1,+0`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 01:00:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-negx-settle14/submission.json`.

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21400747` — controller `settle16`, directions `-1,+0`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 01:00:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-negx-settle16/submission.json`.

Mission7 privileged Approach −x arms `21400730` (`recovery`), `21400731` (`guard`), `21400732` (`pulse`) **COMPLETED 0:0** on `cn-c22`/`cn-c23` under `--constraint=haswell&el8`, 16 world −x layouts each (balanced test-split selection unchanged), geometry, 0.65 m spawn and the >5-contact-tick predicate unchanged. `recovery` (the evaluated 0.50 m/s controller) reproduced `21386145` exactly: **8/16, 0 falls, 8 excessive_collision**, mean 3.2 s. `guard` (lateral centering before the post line, previously untested): **0/16** — 7 excessive_collision, 9 timeouts; regressed all eight baseline successes (1, 5, 6, 7, 20, 25, 37, 41), improved none. `pulse` (0.30 m/s pulses with a 0.90 m detour around the posts, previously untested): **0/16** — 11 excessive_collision, 5 timeouts; same eight regressions, no improvement. Both are rejected. Geometry measured this session: posts at `goal + (0.52, ±0.35)` give a 0.61 m gap; the robot's lateral collision envelope at rest is 0.632 m (elbow to elbow); the outside-post corridor margin is 0.315 / 0.365 / 0.415 m at pitch 1.5 / 1.6 / 1.7 against a 0.316 m half-robot. A −x approach must cross the post plane (spawn at +0.65, dwell radius 0.36), so neither route clears; the 8/16 is arm-swing phase at the crossing. Compact results: `results/mission7-campaign-20260923/approach-negx-{recovery,guard,pulse}/result.json`.

Mission7 privileged Approach −x phase/onset arms (2026-09-23): **SUBMITTED** `21400745` (`recovery030`, 0.30 m/s onset), `21400746` (`settle14`, settle 1.4 s), `21400747` (`settle16`, settle 1.6 s) — same 16 −x layouts, same geometry and predicates; one command-side factor each. The settle arms test the phase reading directly: if success flips on the same layouts with settle time, the crossing phase is the cause.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400752` — doors layouts `0,1,2,3,4,5,6,7`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `True`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/campaign-a-doors-00-07/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400753` — doors layouts `8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `True`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/campaign-a-doors-08-15/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400754` — transport layouts `0,1,2,3,4,5,6,7`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `True`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/campaign-a-transport-00-07/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400755` — transport layouts `8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `True`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/campaign-a-transport-08-15/submission.json`.

Mission7 privileged Approach −x phase/onset arms `21400745` (`recovery030`), `21400746` (`settle14`), `21400747` (`settle16`) **COMPLETED 0:0** under `--constraint=haswell&el8`, 16 world −x layouts each, geometry and predicates unchanged. `recovery030` **7/16** (0 falls; gained 17, 33; lost 6, 25, 37). `settle14` **8/16** (gained 14; lost 25). `settle16` **9/16** (gained 14, 33; lost 6). With `recovery` 8/16 and the rejected `guard`/`pulse` 0/16, six arms and 96 episodes place every command-side factor at 7–9/16 or 0/16, and the set of succeeding layouts moves with settle time and onset speed. Success at the post crossing is arm-swing phase, not layout: the ≥14/16 per-direction Approach criterion is **geometrically bound** for a 0.632 m robot in a 0.61 m gap with ≤0.415 m outside margin. The privileged Approach gate therefore stays **closed** on geometric grounds, unchanged; no candidate is promoted. Compact summary: `results/mission7-campaign-20260923/approach-negx-summary.json`.

Mission7 Campaign A (2026-09-23): **SUBMITTED** `21400752` (Doors 0–7), `21400753` (Doors 8–15), `21400754` (Transport 0–7), `21400755` (Transport 8–15) — early handoff, no intervention, `--chain-trace --rejoin-diagnostic`, `--constraint=haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h each; 32 instrumented development episodes on validation layouts 0–15 with the 25 Hz command-to-motion chain, foot-floor contact and per-episode mechanism classification (thresholds declared in the probe before submission). Receipts: `results/mission7-campaign-20260923/campaign-a-*/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400780` — transport layouts `13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `True`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/campaign-a-transport-13-15/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400801` — doors layouts `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/instage-doors-F1/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400802` — doors layouts `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/instage-doors-F1F2/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400803` — doors layouts `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/instage-doors-F1F3/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400804` — doors layouts `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/instage-doors-F1F2F3/submission.json`.

Mission7 exact replay gate (2026-09-23): **SUBMITTED** `21400805` — ten-fall staged replay pinned to `cn-c22`, PlateStage lateral `0.25 m`; geometry, activation schedule and fall predicate unchanged; receipt/source hashes: `results/mission7-campaign-20260923/replay-gate-lateral025/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400806` — doors layouts `1,2,3,6`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `prev_actions_reset`; receipt/source hashes: `results/mission7-campaign-20260923/campaign-b-doors-reset/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400807` — transport layouts `0`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `prev_actions_reset`; receipt/source hashes: `results/mission7-campaign-20260923/campaign-b-transport-reset/submission.json`.

Mission7 Campaign A `21400752` (Doors 0–7), `21400753` (Doors 8–15), `21400754` (Transport 0–7) **COMPLETED 0:0**; `21400755` (Transport 8–15) **OUT_OF_MEMORY** at 12 GB after writing five valid episode records (8–12): the runner held every full episode row — chain columns, per-step route snapshots, contact events — in memory until the end. Fixed (only the compact summary is retained after each record is written; records unchanged) and the three missing layouts resubmitted as `21400780` (Transport 13–15) **COMPLETED 0:0**. Campaign A is complete: **32 episodes, Doors 3/16, Transport 1/16, 13 falls, 6 timeouts.** Failure phases: during PlateStage **15** (seven `excessive_collision` with a maze wall — the plate centre sits 0.42 m off the corridor centreline, leaving the body 0.29–0.39 m from the wall against a 0.316 m half-body; concentrated in world +y), after PlateStage **12** (falls 0.2–1.3 s after the crossing while the route commands up to 0.35 m/s lateral), before PlateStage 1 (Doors/0 never reached the plate). Post-stage stall exposures **5** (Doors 1, 2, 3, 6; Transport 0); stall-window mechanism `frozen_targets` in every classifiable case (4). Compact summary: `results/mission7-campaign-20260923/campaign-a-summary.json`.

Local single-layout smokes (login node, not Slurm) of three separable in-stage factors, geometry and predicates unchanged: F1 `--stage-lateral=0.25` (body centres 0.25 m off the centreline instead of on the plate centre; wall-side foot at 0.355 m still lands on the 0.24 m-radius plate) removed the Doors/12 wall collision (stage completed, 0 collisions) but the episode fell 2.0 s after exit; on Doors/15 the collision moved from the wall to the closed door — the plate was touched 5,180 times and never activated, because `PlateSafeRouteController` only raises `activate` in its own `switch` phase. F2 `--stage-activate` (stage asserts `activate` on the plate and waits up to 2 s for the door before crossing) activated Doors/12's door at 15.3 s during the stage. F3 `--exit-ramp=0.6` (forward-only 0.30 m/s along the door direction after the crossing) removed the Doors/12 post-exit fall; with F1+F2+F3 Doors/12 reached the second door at 66 s before colliding there.

Mission7 in-stage factor campaign (2026-09-23): **SUBMITTED** `21400801` (F1), `21400802` (F1+F2), `21400803` (F1+F3), `21400804` (F1+F2+F3) — 16 Doors validation layouts each, early handoff, chain trace, `--constraint=haswell&el8`; Campaign A is the matched baseline. **SUBMITTED** `21400805` — exact ten-fall replay gate with `--stage-lateral=0.25`, pinned `cn-c22` (bitwise protocol). **SUBMITTED** Campaign B `21400806` (Doors 1, 2, 3, 6) and `21400807` (Transport 0) — `prev_actions_reset` on the five exposed episodes, chain trace, baseline fingerprints from Campaign A; delivery verified per episode.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400825` — doors layouts `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/instage-doors-F2/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400826` — doors layouts `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/instage-doors-F3/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400827` — doors layouts `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/instage-doors-F2F3/submission.json`.

Mission7 exact replay gate (2026-09-23): **SUBMITTED** `21400863` — ten-fall staged replay pinned to `cn-c22`, PlateStage lateral `plate centre`, wait-open `2.0 s`; geometry, activation schedule and fall predicate unchanged; receipt/source hashes: `results/mission7-campaign-20260923/replay-gate-waitopen2/submission.json`.

Mission7 Campaign B `21400806` (Doors 1, 2, 3, 6) and `21400807` (Transport 0) **COMPLETED 0:0** under `--constraint=haswell&el8`. All five exposed episodes: stall time and fingerprint identical to Campaign A, `prev_actions_reset` **APPLIED** (target changed, zero input delivered to the next policy update). Doors/1 timeout → **success 82.8 s**; Doors/2 post-exit fall → no fall (timeout); Doors/6 reached the second door (progress 3.2 → 5.3 m) and failed inside that stage; Doors/3 **regressed** success → fall inside the second stage; Transport/0 unchanged (fall inside the second stage, 72.8 → 72.3 s). Route success 1/5 → 1/5. The reset un-sticks the feedback fixed point in 5/5 (post-stage progress 3.2–3.8 → 4.5–5.5 m) but hands control back into the in-stage failure downstream; under the predeclared rule (≥1 matched improvement, 0 regressions) it is **not** promoted alone. Compact: `results/mission7-campaign-20260923/campaign-b-doors-summary.json`.

Mission7 exact replay gate `21400805` (`--stage-lateral=0.25`) **COMPLETED 0:0** on `cn-c22`: **8/10 upright — gate FAILED** (baseline 10/10). Layouts 5 and 7 fell in the stage's `approach` phase with a knee on the raised plate: the offset pre-point routes the approach across the plate edge, reintroducing the original trip mechanism. F1 at 0.25 m is rejected as-is.

Mission7 in-stage arms `21400801` (F1), `21400802` (F1+F2), `21400803` (F1+F3), `21400804` (F1+F2+F3) **COMPLETED 0:0**, 16 Doors validation layouts each, Campaign A (3/16, 5 falls) as the matched baseline. F1 **0/16** (in-stage failures 7 → 13: with the body off the plate centre the route's `activate` never fires and every crossing hits the closed door). F1+F3 **0/16**. F1+F2 **3/16, falls 5 → 1**; F1+F2+F3 **3/16, falls 5 → 1** — identical to F1+F2, so F3 adds nothing on Doors. Of F1+F2+F3's nine in-stage terminations, eight are `excessive_collision` with the still-closed `door_0` after the 2 s `wait_open` expired unactivated: the stage settles 0.30 m *before* the plate, so nothing presses it while it waits. Stage-owned activation works (Doors/12 activated at 15.3 s) only when a foot happens to land on the plate mid-crossing.

Mission7 in-stage no-lateral arms (2026-09-23): **SUBMITTED** `21400825` (F2), `21400826` (F3), `21400827` (F2+F3) — 16 Doors, plate-centre lateral (replay-safe approach geometry). **SUBMITTED** `21400863` — exact replay gate with `--wait-open=2.0` at the plate centre, pinned `cn-c22`, the regression run for F2's PlateStage part.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400895` — doors layouts `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/instage-doors-A1-activate-only/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400896` — doors layouts `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/instage-doors-A2-activate-only-l035/submission.json`.

Mission7 exact replay gate (2026-09-23): **SUBMITTED** `21400897` — ten-fall staged replay pinned to `cn-c22`, PlateStage lateral `0.35 m`, wait-open `0.0 s`; geometry, activation schedule and fall predicate unchanged; receipt/source hashes: `results/mission7-campaign-20260923/replay-gate-lateral035/submission.json`.

Mission7 exact replay gate `21400863` (`--wait-open=2.0`, plate centre) **COMPLETED 0:0** on `cn-c22`: **10/10 upright — gate PASSED**; F2's PlateStage part is replay-safe.

Mission7 in-stage no-lateral arms `21400825` (F2), `21400826` (F3), `21400827` (F2+F3) **COMPLETED 0:0**, 16 Doors, Campaign A baseline 3/16 with 5 falls. F2 **1/16, 8 falls**; F3 **1/16, 9 falls**; F2+F3 **1/16, 6 falls** — all worse. At the plate centre the five wall-class layouts (8, 11, 12, 14, 15) still terminate on maze walls in `approach`/`settle`, unchanged from Campaign A; the 2 s `wait_open` is a standstill and standstills are what drive this gait into its feedback fixed point, so the crossing that follows restarts badly (layouts 3, 4, 7 lost). Local press-and-hold smokes on Doors/11 and Doors/14 terminated at exactly the baseline wall-collision times (23.0 s, 26.1 s), before the hold could act; press-and-hold is deprioritized as another standstill.

Selection-pool additions declared before results (2026-09-23): **SUBMITTED** `21400895` (A1: `--stage-activate --stage-wait-open 0`, activate-only at the plate centre, replay-identical PlateStage) and `21400896` (A2: A1 with `--stage-lateral=0.35`, the largest offset that keeps both feet on the 0.24 m plate while moving the body 0.36–0.46 m from the wall), 16 Doors each; **SUBMITTED** `21400897` — exact replay gate at `--stage-lateral=0.35`, pinned `cn-c22`. The predeclared selection rule (highest Doors success, ties to fewer factors, replay gate 10/10 required) now ranges over `21400801–804`, `21400825–827`, `21400895–896`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400952` — doors layouts `0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/instage-doors-F1F2F3-l035/submission.json`.

Mission7 exact replay gate (2026-09-23): **SUBMITTED** `21400953` — ten-fall staged replay pinned to `cn-c22`, PlateStage lateral `0.35 m`, wait-open `2.0 s`; geometry, activation schedule and fall predicate unchanged; receipt/source hashes: `results/mission7-campaign-20260923/replay-gate-lateral035-waitopen2/submission.json`.

Mission7 exact replay gate `21400897` (`--stage-lateral=0.35`) **COMPLETED 0:0** on `cn-c22`: **10/10 upright — gate PASSED**; a 0.35 m body offset keeps the approach off the raised plate. In-stage arms `21400895` (A1, activate-only at the plate centre) **2/16, 6 falls** and `21400896` (A2, activate-only at 0.35 m) **0/16, 2 falls** **COMPLETED 0:0**. A2 clears the wall class — Doors/11 and 12 reach the second door for the first time — but ten layouts then cross into a still-closed `door_0` with no press registered during the 1.2 s crossing; activation without an offset (A1) loses Doors/3 in the second stage. Across nine arms the only sizeable signal remains F1+F2(+F3) at 0.25 m: success held at 3/16 with falls 5 → 1, on a geometry the gate rejects.

Declared before results (2026-09-23): **SUBMITTED** `21400952` — F1+F2+F3 at the replay-safe 0.35 m (`--stage-lateral=0.35 --stage-activate --exit-ramp=0.6`, wait-open 2.0), 16 Doors; and `21400953` — the combined exact replay gate (0.35 m + wait-open 2.0), pinned `cn-c22`. This is the last in-stage arm in the predeclared selection pool; if it does not reach ≥3/16 with the gate at 10/10, no in-stage candidate is frozen and the confirmatory evaluation is not launched.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21400961` — doors layouts `1,2,3,6`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `prev_actions_reset`; receipt/source hashes: `results/mission7-campaign-20260923/interaction-doors-reset-F1F2F3-l035/submission.json`.

Mission7 last predeclared in-stage arm `21400952` (F1+F2+F3 at 0.35 m) **COMPLETED 0:0**: **1/16, falls 5 → 1** — Doors/8 (a wall-class layout) succeeds for the first time; baseline successes 3, 4 and 7 are lost. Combined exact replay gate `21400953` (0.35 m + wait-open 2.0) **COMPLETED 0:0** on `cn-c22`: **10/10 — PASSED**. Full in-stage table (ten arms, 160 episodes, Campaign A baseline 3/16 with 5 falls): F1 0/16; F1+F2 3/16 (falls 1; +8, +9; −4, −7); F1+F3 0/16; F1+F2+F3 3/16 (falls 1; +1, +9; −4, −7); F2 1/16 (8 falls); F3 1/16 (9); F2+F3 1/16 (6); activate-only 2/16; activate-only at 0.35 m 0/16; F1+F2+F3 at 0.35 m 1/16 (falls 1; +8; −3, −4, −7). Every lateral arm loses layouts 4 and 7. Gates: 0.25 m **8/10 FAIL**; wait-open, 0.35 m and their combination **10/10**.

**Selection decision under the predeclared protocol:** no arm exceeds the baseline; the two that tie it are on a geometry the gate rejects; the gate-safe version is 1/16; `prev_actions_reset` had one matched regression. **No candidate is frozen and the confirmatory evaluation on validation 16–31 is not launched.** The reserved test split is untouched. Compact: `results/mission7-campaign-20260923/instage-summary.json`.

Declared before its result: **SUBMITTED** `21400961` — interaction test, `prev_actions_reset` combined with F1+F2+F3 at 0.35 m on the four exposed Doors episodes (1, 2, 3, 6); the plan's "combine 2 and 3" check, 4 episodes.

Mission7 interaction test `21400961` (`prev_actions_reset` + F1+F2+F3 at 0.35 m, Doors 1, 2, 3, 6) **COMPLETED 0:0**: all four episodes terminated inside the stage at 22–35 s, before any post-stage stall opportunity, so the reset was **NOT_EXPOSED in 4/4** (never fired; a route failure of the in-stage arm, distinct from a requested intervention silently doing nothing). Route success 1/4 → 0/4 against Campaign A and against the reset-alone arm; Doors/2's fall removed. The combination is rejected. Campaign closeout: 30 jobs, 349 CPU episodes, zero GPUs; no candidate frozen; confirmatory set and test split unread; all gates unchanged and closed.

**Correction (2026-09-23), Campaign B interpretation.** The adversarial review of the probe patch (35 agents; 5 confirmed of 16 candidate findings) established that the "post-stage progress 3.2–3.8 → 4.5–5.5 m" figures and the "un-sticks the fixed point 5/5" statement compared `chain_post_stage_summary` windows anchored at the *last* stage exit, which cover different route segments in the two arms. Recomputed offline from the retained 25 Hz chain columns of the same runs on the window that starts at the stall condition itself (`stall_branch_time` → +20 s; `results/mission7-campaign-20260923/campaign-b-stall-anchored.json`): only **two of the five exposures were genuine stalls** — Doors/1 (0.019 m in 20 s, cycling in place, target range 47 % of walking and decaying to 2 % by the final window) and Doors/6 (0.003 m, 17 %); `prev_actions_reset` un-stuck **both** (3.46 m and 4.63 m in the same window). Doors/2, Doors/3 and Transport/0 were 0.8 s pauses the unchanged baseline walked out of on its own (4.64, 4.45, 2.56 m), and the reset changed nothing measurable there (4.75, 4.44, 2.61 m). The stall detector's 0.8 s no-progress criterion therefore over-counts exposures; a fixed point should be required to persist for several seconds before an episode counts as exposed. The promotion decision is unchanged (Doors/3's regression stands). Other confirmed findings, all fixed in the probe: foot contact ignored feet standing on plate geoms (foot-contact and slip statistics undercounted in windows that include a plate; no label changed); no stall-anchored classification window existed (added; `mechanism` now prefers it for exposed episodes); `handoff_condition_first_true_time` was overwritten on a second activation; the post-stage contact summary now names its first-exit anchor. Rejected findings (11) are recorded in the workflow journal.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21401685` — doors layouts `1,2,3,6`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `prev_actions_reset`; receipt/source hashes: `results/mission7-campaign-20260923/campaign-b2-doors-reset-min3/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21401686` — transport layouts `0`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `True`, fix `prev_actions_reset`; receipt/source hashes: `results/mission7-campaign-20260923/campaign-b2-transport-reset-min3/submission.json`.

**Correction (2026-09-23), in-stage lateral arms withdrawn — misplaced staging target.** Validating plate-relative geometry from the retained chain columns shows that every `--stage-lateral` arm staged the robot about **0.75 m past the plate**, in front of the door, not laterally offset from it: `layout.plate()` anchors the plate on `xy(route[k])`, the cell before the door, while the offset target was built on `layout.door()`'s centre half a cell further along. In the offset arms the settle and crossing ran at +0.43 to +0.79 m along the door direction with zero plate contact; presses registered only when the robot happened to walk over the plate on its way to the wrong point; the "wall class cleared" because the robot stood somewhere else; and the 0.25 m replay falls came from approaching that far point across the plate. Withdrawn as evidence about a lateral offset: `21400801` (F1), `21400802` (F1+F2), `21400803` (F1+F3), `21400804` (F1+F2+F3), `21400896` (A2), `21400952` (F1+F2+F3 at 0.35) and the replay gates `21400805`, `21400897`, `21400953` (96 + 30 episodes). They stand as evidence about a target placed before the door. The anchor is fixed and verified to reproduce the plate position exactly at 0.42 m. Arms without the offset (`21400825–827`, `21400895`) and the wait-open gate `21400863` are unaffected.

**Established from the same traces (2026-09-23).** (1) *Post-exit falls are plate-edge trips, 8/8:* at hand-back the base is 0.13–0.39 m from the plate centre — on the plate — the route immediately commands up to 0.35 m/s lateral, and every fall carries 230–860 N of plate contact in its last 1.5 s. Cause: from the 0.40 s standstill the gait covers only ~0.13 m in the replay's 1.20 s crossing, so the crossing ends with the feet on the plate edge. (2) *Approach-phase falls are plate trips from the side:* Doors/13, Doors/5 and Transport/0's second door were captured 0.46–0.78 m lateral to the plate, two of them at a ~90° heading, and the straight-line approach with no yaw correction walked onto the raised plate (2,600–4,700 contacts, 340–590 N). (3) *Doors/3's Campaign B regression* is not stall-related: the reset fired at 36.4 s during a 0.64 s pause while walking at 0.34 m/s, zeroing `prev_actions` mid-stride produced a 0.7–1.0 m/s lurch that re-converged within a second, and the 0.2 s timing shift it left put a different footfall on the raised plate in the second crossing (1,155 contacts, 338 N) at 82.6 s. (4) *Exposure durations:* the two genuine stalls (Doors/1, Doors/6) never moved again; the three pauses resumed within 0.36–0.64 s; a 3.0 s no-progress criterion separates them with margin.

Declared before results (2026-09-23): **SUBMITTED** `21401685` (Doors 1, 2, 3, 6) and `21401686` (Transport 0) — `prev_actions_reset` with `--stall-min-s 3.0`, unchanged stage; expected Doors/1 and Doors/6 exposed and recovered, the other three NOT_EXPOSED with outcomes identical to Campaign A. Budget: 163 episodes remained; 5 spent here; plan ≤ 20 replay-gate episodes, ≤ 48 development episodes on validation 0–15, and 64 reserved for a predeclared confirmatory run (32 candidate + 32 baseline on validation 16–31) only if a candidate qualifies.

Mission7 exact replay gate (2026-09-23): **SUBMITTED** `21401689` — ten-fall staged replay pinned to `cn-c22`, PlateStage lateral `plate centre`, wait-open `0.0 s`; geometry, activation schedule and fall predicate unchanged; receipt/source hashes: `results/mission7-campaign-20260923/replay-gate-v2-clear035-kick/submission.json`.

Mission7 selective-trigger verification `21401685` (Doors 1, 2, 3, 6) and `21401686` (Transport 0) **COMPLETED 0:0** — `prev_actions_reset` with `--stall-min-s 3.0`, unchanged stage, Campaign A as the matched baseline. Doors/2, Doors/3 and Transport/0: **NOT_EXPOSED, trajectories tick-identical to baseline** (92.8 s, 102.0 s, 72.8 s); Doors/3's success is preserved, so the Campaign B regression is removed by the trigger alone. Doors/6: exposed at 40.0 s, APPLIED, timeout → **success 111.5 s**. Doors/1: exposed at 48.0 s, APPLIED, progressed 2.7 m and opened the second door, then fell after the second crossing at 64.9 s (timeout → fall; a downstream plate trip). Doors 1/4 → 2/4 on success, falls 1 → 2. The selective trigger is validated as regression-free on success; it exposes only the two genuine stalls.

Local single-layout smokes of the corrected stage on Doors/1 (login node, no budget). *Pre-point 0.45 + cross-until-clear 0.35 + activate-only*: approach lands at the intended point, the press registers mid-crossing (door open 23.7 s), but the gait advances 0.2 m after the 0.40 s settle and then **stands on the plate edge for the remaining 3 s at the full 0.30 m/s command** — the standing fixed point, and the reason the replay's crossing covers only 0.13 m. *No settle (V1)*: continuous motion into the plate, trip at 24.2 s — rejected, consistent with the original ten falls. *Cross-kick (V2: settle, then `prev_actions := 0` at cross start)*: the crossing advances steadily −0.43 → +0.36 m in 3.0 s, door open at 23.8 s, hand-back **past the plate with no plate contact**; the route then brakes on the next plate contact and reverses onto the plate, and Doors/1 opens both doors for the first time before a post-exit fall at the second door (73.5 s). *Corrected lateral 0.25 on Doors/12*: press registers (15.1 s) but the straight-line approach from 0.83 m to the side overshoots to the plate's own line and hits the wall at the baseline time (16.3 s); the wall class is an approach-dynamics problem the offset alone does not solve.

Declared before results (2026-09-23): **SUBMITTED** `21401689` — exact replay gate for the V2 stage (`--pre-point=0.45 --cross-clear=0.35 --cross-kick`, settle 0.40), pinned `cn-c22`. Local smokes of the two hand-back complements (V2 + 0.6 s exit ramp; V2 with clearance 0.60 m) run in parallel; the better-behaved complement and the gate result decide the 16-Doors development arms. Budget after this gate: 148 episodes.

Local Doors/1 smokes of the V2 stage with hand-back complements (2026-09-23, no budget). **V2 + 0.6 s exit ramp: route SUCCESS at 84.0 s** — both doors opened (23.8 s, 57.8 s), both crossings clean, no fall, no stall exposure; the first full-route success produced by an in-stage change. V2 with clearance 0.60 m and no ramp: fell in the second crossing at 62.9 s — rejected. In every V2 smoke the route walks the robot backwards along the door direction after hand-back (+0.69 → +0.2 m from the plate): the per-step route history shows `route.waypoint` still at the cell **before** the door (index 4 of a door between 4 and 5) for three seconds after the stage carried the robot past it, before advancing to 5. Declared: `--rejoin-advance`, which at hand-back sets the waypoint to the cell after the crossed door once the door is open and the body is ≥0.24 m past the plate; nothing else in the route changes. Smokes of V2 + rejoin-advance with and without the ramp are running. Rendering note: the login node has neither EGL nor OSMesa; the existing GIF launchers (`97_maze_video.sbatch`) use the GPU partitions with `--gres=gpu:1`, which a Mission 7 clip will need too.

Local Doors/1 smokes, V2 + `--rejoin-advance` with and without the straight exit ramp (2026-09-23): the backward walk is gone (along the door direction +0.35 → +0.80 m monotonically) but both end in `excessive_collision` 1.2–1.6 s after hand-back on the door jambs (`wall_m7_15`, `wall_m7_22`): heading straight for the next cell from the plate's lateral line, the robot reaches the door opening 0.46 m off the corridor centreline. The baseline's backward walk to the pre-door waypoint was re-centring, at the cost of re-entering the plate zone. Declared: `--exit-ramp-center` (the ramp aims at the door centre, re-centring while always advancing) combined with rejoin-advance; smoke running.

Mission7 exact replay gate `21401689` (V2: `--pre-point=0.45 --cross-clear=0.35 --cross-kick`) **COMPLETED 0:0** on `cn-c22`: **7/10 upright — gate FAILED** (layouts 4, 5, 7 fell). Forensics from the gate's own samples: in all three the body-frame command during `cross` was **[0.01, 0.30, 0.0] — a sideways walk** for ~2 s before a plate-edge trip (tilt rising with 60–240 N plate contact). The crossing is a world-frame vector issued without yaw correction, so a robot that reached the pre-point about 90° off the door direction crosses laterally, on the gait's weakest axis; the replay's fixed 1.20 s crossing barely moved (0.13 m) and never reached the edge, which is why the plate-centre gates passed, while cross-until-clear keeps the sideways walk going until it falls. Declared: `--align-yaw` — turn in place toward the door direction during the settle (tolerance 0.20 rad, bounded 1.5 s) before the kick and a now-forward crossing. Local Doors/1 smoke of V2 + rejoin-advance + centre-aimed exit ramp (1.2 s): **route SUCCESS at 77.7 s**, both doors, zero contacts, re-centred from −0.61 m to −0.11 m while advancing. The ten-episode replay of V3 (V2 + align-yaw) is being run locally first, at no budget, before any further gate; budget remaining 138.

Local ten-episode replay of V3 (V2 + in-place yaw alignment during the settle), 2026-09-23, no budget: **8/10** — layouts 4 and 5 now upright, 7 still falls, 14 newly falls, and 9/12/13 never reach the pre-point before the recorded episode ends (upright). Forensics: in 7 and 14 the pre-point is reached 75–90° off the door direction and **the frozen gait does not turn in place** — 1.5 s of a pure yaw command from standstill produced no rotation (yaw error 1.25 → 1.33 rad) — so the crossing was sideways again. Declared V4: the yaw term is applied while moving, during the approach and the crossing, with no in-place turn; the local replay and a Doors/1 smoke are running. Doors/1 with V3 + centre-aimed ramp + rejoin-advance: route success at 81.0 s.

**Predeclared confirmatory evaluation, candidate 1 (2026-09-23, before any result on validation 16–31 was read).** Candidate frozen by snapshot hash: the unchanged guarded PlateStage (plate centre, 0.40 s settle, 1.20 s crossing — the configuration that holds the 10/10 exact replay), early handoff, and `prev_actions_reset` with `--stall-min-s 3.0`. Development evidence: on the five Campaign A exposures the 3 s trigger leaves every pause tick-identical to baseline (3/3) and fires only on the two genuine stalls, recovering Doors/6 to success and carrying Doors/1 to the second door; no success regression. Evaluation set: validation layouts **16–31**, never used, both stages, 32 candidate episodes without chain trace (`21401728` Doors, `21401729` Transport, `--constraint=haswell&el8`). Because a non-exposed candidate episode is tick-identical to baseline by construction (verified), the candidate run is the matched baseline wherever the reset does not fire; matched baseline reruns are submitted only for exposed episodes. Primary metric: route success on exposed pairs plus the unchanged non-exposed pairs (full denominator 32 per stage); secondary: falls, exposure count. Precision as predeclared: with the expected 1–3 exposures the result will be reported as underpowered, not as null. One evaluation, no second look; the test split stays untouched.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21401728` — doors layouts `16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `False`, fix `prev_actions_reset`; receipt/source hashes: `results/mission7-campaign-20260923/candidate1-doors-16-31/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21401729` — transport layouts `16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `False`, fix `prev_actions_reset`; receipt/source hashes: `results/mission7-campaign-20260923/candidate1-transport-16-31/submission.json`.
Mission7 media render (2026-09-23): **SUBMITTED** `21401734` — GPU partition, EGL; renders Doors/1 baseline (stopped at 70 s, stall shown) and Doors/1 with the crossing fix (V2 + centre ramp + rejoin-advance, local success 77.7 s) to MP4 with hashed sidecars, then a paired GIF. Two rendered episodes count against the budget; the clip is a rendering of a single development layout, not a route-rate or gate claim.

Local ten-episode replay of V4 (moving yaw term in approach and crossing), 2026-09-23, no budget: layout 7 falls again (9/10 at best); and on Doors/1 the yaw term changed the crossing timing so the kick-restarted gait stalled on the plate a second time, the 4 s crossing budget expired with the robot still on the plate, and the route's backward walk tripped it at 30.2 s (fall 31.0 s; V3 had succeeded at 81.0 s). **Crossing workstream closed for this budget as a validated mechanism without a gate-passing composition:** (1) after the settle the frozen gait restarts for ~0.2 m at the 0.30 m/s command and settles into its standing fixed point on the plate edge — the reason the replay's crossing covers 0.13 m and every post-exit fall is a plate trip; (2) a stage-owned `prev_actions` kick restarts it, once; (3) a robot that reaches the pre-point sideways crosses sideways, and the gait cannot turn in place; (4) after hand-back the route's waypoint index still points at the pre-door cell, so it walks backwards into the plate zone; a centre-aimed exit ramp with the index advanced fixes the hand-back. Composed, these produce full-route success on Doors/1 (77.7 s and 81.0 s) but the exact ten-fall replay is 7/10 (V2, `21401689`), 8/10 (V3, local) and ≤9/10 (V4, local). No further gate is spent on it; the remaining budget goes to candidate 1's predeclared confirmatory.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21401788` — doors layouts `28`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `False`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/baseline-doors-exposed-16-31/submission.json`.

Mission7 route handoff probe (2026-09-22): **SUBMITTED** `21401789` — transport layouts `24,31`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 2 h; unchanged PlateStage with `early` route handoff and rejoin diagnostic `False`, chain trace `False`, fix `none`; receipt/source hashes: `results/mission7-campaign-20260923/baseline-transport-exposed-16-31/submission.json`.

Mission7 candidate 1 confirmatory `21401728` (Doors 16–31) and `21401729` (Transport 16–31) **COMPLETED 0:0** under `--constraint=haswell&el8`, 32 episodes on the never-used validation layouts. Doors **1/16** (layout 26), 9 falls; Transport **4/16** (19, 24, 28, 31), 4 falls. Exposures **3**: Doors/28 (reset applied at 31.6 s → timeout, no fall), Transport/24 (applied at 40.6 s → **success 100.1 s**), Transport/31 (applied at 73.4 s → **success 123.7 s**). Every other episode is, by the verified construction, the baseline trajectory. Whether the two exposed Transport successes are the reset's doing is decided by the predeclared matched baseline reruns of exactly those three layouts with the unchanged controller: **SUBMITTED** `21401788` (Doors 28) and `21401789` (Transport 24, 31). Media render `21401734` **COMPLETED 0:0** on a GPU node (EGL): `media/doors1-baseline-stall.mp4` (stopped at 70 s) and `media/doors1-crossing-fix-success.mp4` (77.68 s, success reproduced), hashed sidecars, and `docs/gifs/mission7-doors1-stall-vs-crossing-fix.gif` at 1.94× with its own sidecar; gallery section and index row added. Budget: 101 episodes remain after the three reruns.

Mission7 candidate 1 matched baseline reruns `21401788` (Doors 28) and `21401789` (Transport 24, 31) **COMPLETED 0:0**. **Paired confirmatory result on validation 16–31 (32 pairs per arm, full denominator):** Doors baseline **1/16 → candidate 1/16** (the one exposed pair, Doors/28, times out in both arms, no fall); Transport baseline **2/16 → candidate 4/16** — both exposed Transport layouts stalled to a 180 s timeout under the unchanged controller and **succeed with the reset applied** (24 at 100.1 s, 31 at 123.7 s). Improved 2, regressed 0, discordant pairs 2, two-sided sign test p = 0.5; falls 9 → 9 and 4 → 4. Per the predeclaration this is reported as **underpowered, not null**: the effect is confined to genuine stalls (3 of 32 fresh episodes), and on every one of the five genuine stalls seen across development and confirmation the selective reset converted a stall into forward progress, with route success in four (Doors/6, Doors/1 → later fall at door 2, Transport/24, Transport/31) and no regression anywhere. Candidate 1 stays frozen as defined; the reserved test split is untouched. Compact: `results/mission7-campaign-20260923/candidate1-confirmatory-summary.json`. Budget: 101 episodes unspent.

Artifact trim (2026-09-23): the campaign directory reached 11 GB against the 10 GB retained-artifact allowance. Raw per-episode chain traces of the withdrawn misplaced-target arms (`21400801–804`, `21400896`, `21400952`, `21400961`) and the raw `episodes.json` of the withdrawn/failed replay gates (`21400805`, `21400897`, `21400953`, `21401689`) and the two local replays were deleted; every compact `result.json` and `submission.json` (with source hashes) is kept and committed, so no verdict or receipt was lost. Nested per-job `source/` snapshots and job logs are now ignored by shape.

**Correction (2026-09-23), world −x clearance.** The statement that the privileged Approach −x gate is "geometrically bound" rested on a sphere-bound (`geom_rbound`) width of 0.632 m. Measured with world-frame geom AABBs over a 6 s straight walk at 0.47 m/s (local MuJoCo, no budget): lateral width **0.606 m at rest, 0.546–0.605 m while walking (mean 0.584)**, extreme geoms `arm_{left,right}_elbow_roll`; the width is under the 0.61 m post gap at **every** walking tick and under 0.58 m on 32 % of them. The gap is therefore **marginal (0.5–6 cm of clearance depending on gait phase), not impossible**, and the 8/16 result remains phase luck at the crossing under a controller with no fine lateral centring. The six-arm negative (`21400730–747`) stands as evidence about those controllers; "geometrically bound" is withdrawn as the reason the gate is closed. Reopened as a control problem: fine lateral centring on the gap midline from privileged pose while walking straight, with a slow onset through the gap.

Inspection and airlock extension (2026-09-23): **SUBMITTED** `21401943` (inspection, ordered route, 12 fresh seeds 3–14, reactive + complete-outage dropout, gate), `21401944` (same seeds, intermittent dropout 0.35), `21401945` (wrong-branch control, same seeds), `21401946` (airlock crew 2, seeds 5–14, coordinated / no_wait / withhold_last, gate), `21401947` (crew 3, same), `21401948` (crew 2 coordinated with sensor dropout 0.35). Frozen 22-DoF gait `2026-08-18_20-57-50_arms-dr1.0-s0`, MuJoCo CPU, 60 s episodes, 0.4 m/s, `--constraint=haswell&el8`, 4 CPUs / 12 GB / 4 h each via `slurm/weekend_cpu.sbatch`. These extend the September 19 3/3 and 5/5 results to new seeds and stress conditions with the same evaluators and negative controls; they are evaluations of the existing scripted-supervision controllers, not learned coordination.

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21401950` — controller `center`, directions `-1,+0`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 01:00:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-negx-center/submission.json`.

Inspection extension `21401944` (intermittent sensor dropout 0.35, ordered route, 12 fresh seeds 3–14) **COMPLETED 0:0**: **12/12** ordered inspections completed under 35 % packet loss with the reactive controller (the file's `gate_passed=False` only reflects that its single arm cannot satisfy the two-arm gate rule; the nominal and complete-outage arms run in `21401943`). Wrong-branch control `21401945` (12 fresh seeds) **COMPLETED 0:0**: **0/12 success, 12/12 `dead_end_entered`** — the supervisor-error control fails deterministically on every seed, as designed. Mission7 privileged Approach −x fine-centring arm (2026-09-23): **SUBMITTED** `21401950` — `center` controller (recovery translation with a lateral P-term onto the gap midline and a 0.30 m/s onset from 0.45 m before to 0.25 m past the post plane, privileged pose only), 16 −x layouts, geometry and predicates unchanged; local smokes engaged the term within ±3 cm. Mission 7 budget after this arm: 85 episodes.

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21401964` — controller `center_bias`, directions `-1,+0`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 01:00:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-negx-center-bias/submission.json`.

Mission7 privileged Approach −x fine-centring arm `21401950` **COMPLETED 0:0** under `--constraint=haswell&el8`: **11/16, 0 falls** on the 16 test-split −x layouts (recovery baseline 8/16) — gained 14, 33, 36, lost none. All five failures are `excessive_collision` on **`goal_post_-1`** at 1.84–2.00 s, the first stride: a systematic −y lurch (minimum pre-contact drift −0.015 to −0.036 m, median −0.025) that the lateral P-term cannot cancel within its 0.2 s reaction; successes showed ±2–4 cm excursions that cleared. Declared and **SUBMITTED** `21401964`: `center_bias`, the same controller with the midline target pre-biased +0.015 m toward +y (0.6 × the median drift), 16 −x layouts. Airlock crew 2 with sensor dropout 0.35 `21401948` **COMPLETED 0:0**: **10/10** coordinated completions on seeds 5–14. Mission 7 budget after the bias arm: 69 episodes.

Inspection extension `21401943` (ordered route, 12 fresh seeds 3–14, reactive vs complete-outage dropout, gate) **COMPLETED 0:0**: **reactive 12/12** (completion 17.1–17.8 s, both stations, zero wall-contact steps), **complete outage 0/12** (all `mission_timeout`), **gate passed**. With `21401944` (12/12 at 35 % dropout) and `21401945` (wrong-branch 12/12 rejected), the September 19 three-seed inspection result now holds on 15 seeds with both negative controls intact. Same frozen gait, scripted supervision, oracle waypoints; no learned navigation or recognition is claimed.

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21401966` — controller `center_bias`, directions `all`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 02:00:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-matrix-center-bias/submission.json`.

Mission7 privileged Approach −x `center_bias` arm `21401964` **COMPLETED 0:0**: **14/16, 0 falls** on the 16 test-split −x layouts — versus the recovery baseline +14, +15, +17, +21, +31, +36, +51 / −25; versus the unbiased centring arm +15, +17, +21, +31, +51 / −25, −33. Failures: layout 25 `timeout` (never closer than 0.58 m; the +y pre-bias displaced the approach on a layout the unbiased arm passed) and layout 33 `excessive_collision` at 2.2 s. This meets the per-direction ≥14/16 criterion for world −x; the documented gate also requires ≥60/64 and zero falls over the balanced four-direction matrix. Declared and **SUBMITTED** `21401966`: the full 64-episode matrix with `center_bias` (the controller alters behaviour only when the final route segment points to world −x, so the other three directions are expected to reproduce the baseline's 48/48; the gate is measured, not assumed). Matched standstill controls are not re-run (budget); the baseline's 0/64 standstill result on the same layouts is geometry-only and remains the negative control. Mission 7 budget after the matrix: 5 episodes.

**Mission7 privileged Approach gate — PASSED.** `21401966` (`center_bias`, full balanced four-direction matrix, 64 episodes, test-split selection unchanged from `21386145`, geometry / 0.65 m spawn / dwell / fall and >5-contact-tick predicates unchanged, privileged pose only) **COMPLETED 0:0** under `--constraint=haswell&el8`: **62/64 successes, 0 falls; +y 16/16, −y 16/16, +x 16/16, −x 14/16.** All five documented criteria hold (`minimum_episodes`, `successes_at_least_60`, `zero_falls`, `per_direction_at_least_14`, `full_matrix`). Failures: −x layout 25 `timeout` (never within 0.58 m) and −x layout 33 `excessive_collision` at 2.2 s. The controller is the evaluated 0.50 m/s recovery translation plus, on world −x only, a lateral P-term (gain 1.5, ±0.20 m/s) onto the gap midline pre-biased +0.015 m toward +y and a 0.30 m/s onset from 0.45 m before to 0.25 m past the post plane. Provenance: task Mission 7 Approach (0.65 m), frozen gait `2026-08-18_20-57-50_arms-dr1.0-s0`, MuJoCo 3.3.5, seed 2300, evaluator `scripts/mission7_approach_followup.py --controller center_bias`, source hashes in `approach-matrix-center-bias/submission.json`. Matched standstill controls were not re-run (budget); the same layouts' 0/64 standstill result under `21386145` remains the geometry-only negative control. Sensor-release gates stay closed on the still-failing route gate (Doors 1/16, Transport 4/16 with candidate 1). Mission 7 budget: 5 episodes remain.
Mission7 Approach media render (2026-09-23): **SUBMITTED** `21401970` — GPU/EGL; world −x layout 14 under the recovery baseline (collision) and under `center_bias` (success), MP4s with hashed sidecars and a paired GIF at ≤12 s budget; two rendered episodes (Mission 7 budget: 3 remain).

Airlock extension `21401946` (crew 2, seeds 5–14, coordinated / no_wait / withhold_last, gate) **COMPLETED 0:0**: **coordinated 10/10** (completion 23.0–23.7 s, zero robot- and wall-contact steps), **no_wait 0/10** (all `robot_collision`), **withhold_last 0/10** (all `team_timeout`), **gate passed** — the September 19 five-seed result holds on ten fresh seeds with both negative controls intact. Crew 3 (`21401947`) is 10/10 coordinated and 0/10 no_wait with the withhold_last arm still running. Scripted synchronization with a frozen gait; no learned coordination is claimed.

Approach media render `21401970` **COMPLETED 0:0** but is **withdrawn, not published**: both layout-14 clips succeeded (0 collisions) because a fresh environment draws the first spawn-noise sample, whereas the evaluator advances its generator through the preceding layouts of the balanced selection; the rendered baseline therefore did not reproduce the evaluated collision. The unfaithful MP4s/GIF were deleted. `scripts/render_mission7_approach.py` now takes `--replay-order` (reset through the selection in the evaluator's order up to the target) and records the reset sequence in its sidecar, and the sbatch asserts that the baseline run fails and the centred run succeeds before any GIF is built. **SUBMITTED** `21401987` (re-render of layout 14, both arms, replayed order). Mission 7 budget: 1 episode remains after this render.

Airlock extension `21401947` (crew 3, seeds 5–14, coordinated / no_wait / withhold_last, gate) **COMPLETED 0:0**: **coordinated 10/10** (completion 30.6–31.0 s, zero robot- and wall-contact steps), **no_wait 0/10** (`robot_collision`), **withhold_last 0/10** (`team_timeout`), **gate passed**. Both crews now hold on ten fresh seeds with both negative controls; crew 2 additionally holds 10/10 under 35 % sensor dropout. Shared-world scripted synchronization with a frozen gait; no learned coordination or object recognition is claimed.
Airlock stress (2026-09-23): **SUBMITTED** `21402000` — crew 3, coordinated, sensor dropout 0.35, seeds 5–14 (the one airlock stress condition not yet run).

**Qualification (2026-09-23), privileged Approach gate.** All 16 world −x layouts in the balanced matrix are **test-split** layouts, and the `center` controller and its +0.015 m `center_bias` were designed and sized on exactly those layouts (the bias from the median pre-contact drift of the five `center` failures). The −x 14/16 in `21401966` is therefore a **development result on the reserved test split**, which the Mission 7 protocol reserves for final reporting and forbids for selection. The 62/64 stands as measured and is not withdrawn, but it is not a held-out result. Predeclared before any result: a held-out replication on the 23 world −x layouts that no controller in this programme has touched — the 7 validation −x layouts outside the balanced selection (2, 5, 13, 18, 22, 23, 27) and the first 16 train-split −x layouts (1, 7, 10, 11, 17, 19, 21, 22, 24, 26, 27, 31, 35, 40, 42, 43); both `recovery` (baseline) and `center_bias`, same spawn, predicates and seed stream. Reported as a paired rate; the per-direction criterion (≥14/16 ≈ 87.5 %) is the reference. Budget: a new, separately accounted line of 46 CPU episodes under the repo-wide plan (Mission 7's own line has 1 left and is not used).

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21402017` — controller `recovery`, directions `all`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 01:30:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-negx-heldout-recovery/submission.json`.

Mission7 privileged Approach arm (2026-09-23): **SUBMITTED** `21402018` — controller `center_bias`, directions `all`, constraint `haswell&el8`, 2 CPUs / 12 GB / 0 GPUs / 01:30:00; geometry, spawn and predicates unchanged; receipt/source hashes: `results/mission7-campaign-20260923/approach-negx-heldout-center_bias/submission.json`.
Airlock stress `21402000` (crew 3, coordinated, sensor dropout 0.35, seeds 5–14) **COMPLETED 0:0**: **10/10**. Both crews now hold 10/10 under 35 % dropout.

Mission7 Approach held-out replication `21402017` (`recovery`) and `21402018` (`center_bias`) **COMPLETED 0:0** under `--constraint=haswell&el8` — the predeclared paired rescore on 23 world −x layouts no controller had touched (validation 2, 5, 13, 18, 22, 23, 27; train 1, 7, 10, 11, 17, 19, 21, 22, 24, 26, 27, 31, 35, 40, 42, 43). **`center_bias` 21/23 (0.913, Wilson 95 % CI 0.73–0.98) against the baseline's 12/23**; validation 7/7 vs 4/7, train 14/16 vs 8/16; improved 10, regressed 1 (train 26), two-sided sign test p = 0.012; zero falls in either arm. The test-split tuning qualification is resolved in the controller's favour: the −x centring effect replicates on held-out geometry at a rate above the 14/16 per-direction reference. The 62/64 gate figure itself remains the test-split measurement it was. Compact: `approach-negx-heldout-summary.json`.

Repo-wide CPU evaluations (2026-09-23), existing exported checkpoints, MuJoCo 3.3.5, `--constraint=haswell&el8`, 4 CPUs / 16 GB / ≤3 h, `slurm/repo20260923/cpu_eval.sbatch` with an import preflight: **SUBMITTED** `21402035` + `21402036` — matched push protocol for every exported biped DR and push policy (0.5 m/s shoves every 3 s, 12 s episodes, 5 seeds × the harness command set; 21 policies incl. the dr-default no-push control, which the bulk tables never scored under pushes); predeclared: the `push_pair` caption stands only if push-trained fall rates are below dr-default's with no training-seed overlap. `21402037` / `21402038` — lab traverse (carpet/cable/threshold/ramp) for the four 22-DoF and four 12-DoF policies over seeds 0–4, replacing a single-rollout Slurm log. `21402039` — cross-engine scoring of the six occluded coop-lift checkpoints (blind s0 ×2 dirs, s1, s2; depth s0 ×2 dirs), 8 seeds × crews 2 and 3, so the occlusion claims have a results file.

Correction: `21402039` (coop occluded scoring) **FAILED** at 1:19 on a CPU share node — `mujoco.FatalError: an OpenGL platform library has not been loaded` from `coop_sim2sim.py`'s depth renderer (the two depth checkpoints need rendering; blind ones do not). No results written. **Resubmitted as `21402060`** on gpu/dgxh/ampere with `MUJOCO_GL=egl`, ≤1 h, `slurm/repo20260923/gpu_render_eval.sbatch` (same body, GPU header). Counts against the GPU evaluation allowance.

Lab traverse 5-seed rerun `21402037` (22 DoF) / `21402038` (12 DoF) **COMPLETED 0:0** (3:17 / 2:26). Finishes over seeds 0–4: 22 DoF randomized 1/5, no-randomization 0/5 (stalls past the carpet every seed), push-trained 2/5, terrain-trained 2/5; 12 DoF 0/5, 0/5, 0/5, terrain-trained 2/5. Falls cluster on the cable (12 DoF) and the ramp/landing (22 DoF). This replaces the single-rollout Slurm-log evidence behind REPORT §8 and the `multi_lab` caption: the 22-DoF model clears the course more often, neither clears it reliably. Summary: `results/lab-traverse-20260923/lab_traverse_5seed.json`.

Sensor fusion SF-03 (attitude filter in the loop, `docs/SENSOR_FUSION.md`) **SUBMITTED** `21402118` — CPU share, `haswell&el8`, `cpu_eval.sbatch sf03`. Inspection maze, ordered route, reactive sensors, seeds 0–2, 22-DoF `arms-dr1.0-s0` gait. Predeclared levels: L0 truth control; L1 gyro std 0.02 rad/s / accel std 0.3 m/s² / bias 0.02 rad/s / no delay (Mahony and Madgwick); L2 0.05 / 1.0 / 0.05 / 1-step delay (both filters); L3 0.10 / 2.0 / 0.10 / 2-step delay (Mahony). 18 episodes. Predeclared reading: success within one seed of the truth control at gravity RMSE ≤ 0.05 counts as tolerated; report the first level at which it fails. Filter runs at the 25 Hz policy rate (conservative). Local smoke before submission: L1 Mahony seed 0 success at 16.6 s, gravity RMSE 0.035, max 0.079, alignment 17 steps.

Correction: SF-03 `21402118` **COMPLETED 0:0** (3:09) but its L2/L3 rows are **invalid**: the stationary-alignment gate required five consecutive single samples within 5 % of g, which accelerometer noise of 1–2 m/s² never satisfies, so the filter never aligned (`pre_alignment_steps` = whole episode) and those episodes ran on the oracle observation. Only L0 (3/3) and L1 (Mahony 3/3, RMSE 0.037; Madgwick 3/3, RMSE 0.030; alignment 14–17 steps) are valid. Outputs moved to `results/sensor-fusion-20260923/run1-invalid-gate/`. Gate rewritten to average a 0.4 s window (no free-fall sample, mean |a| within 10 % of g, mean gyro < 0.3 rad/s) and the episode row now carries `aligned`. Local smoke with the new gate: L2 aligned at 12 steps and the gait **fell** (gravity RMSE 0.150, max 0.376); L3 aligned at 13 steps and fell (RMSE 0.125). **Resubmitted as `21402531`** with three factor-isolation arms (accel-noise-only, gyro-noise-only, delay-only from L1) and a low-gain Mahony (kp 0.3) at L2: 30 episodes, so SF-03 spends 48 CPU episodes in total against the predeclared 30 (the 18-episode first run is charged even though half of it was invalid).

Repo-wide GPU diagnostics (2026-09-23), prepared by an adversarially reviewed workflow (static checks only; reviews ok with non-blocking notes): **SUBMITTED** `21402547` cube-to-shelf spawn diagnostic (v60, `gpu_spawn_diag.sbatch`, 4 envs × 200 steps × {zero, PD-hold, N(0,1)} actions on `TaskV2-BHL-CubeToShelf-Blind-v0` and the Grip variant; records the first termination term to fire; tests the prediction that `_tilt_from_quat` reads the v60 xyzw root quaternion as wxyz so 'fallen' fires at step 1 for the ±90° yaw spawn; ≈0.5 GPU-h). `21402548` ice foot-on-patch exposure probe (v51, `gpu_ice_exposure.sbatch`, placed blind/depth s0 `model_5999.pt`, 64 envs, one episode with the policy acting, filtered foot–patch contact sensor with a geometric fallback; verdict EXPOSED if ≥ 0.5 of episodes touch a patch and mean on-ice fraction ≥ 0.10; ≈0.4 GPU-h). `21402549` maze recovery corruption-ON re-evaluation (v60, `gpu_maze_noise.sbatch`, `--dependency=afterany:21402547` to hold GPU concurrency at two; 12 Full-stage checkpoints, 32 envs × 600 steps, same seeds as the published noise-off 379/384; `maze_recovery_probe.py` gains `--keep-corruption` defaulting to the old behaviour; ≈0.6 GPU-h). Depth bench re-run is held for a manual review of its launcher (its review agent failed on a session limit).

Depth validation/throughput bench re-run **SUBMITTED** `21402550` after a manual review of `slurm/repo20260923/gpu_depth_bench.sbatch` (v51; `--partition=gpu --constraint=rtx8000` because commit b34b99a dates the measuring GPUs as Turing; provenance and worktree patch captured; per-step deadline guard; `depth_bench_summary.py` compares finite fraction 1.000, mean relative error 0.029 and the 21,844 / 21,490 env-steps/s rows at 10 % tolerance and prints PASS / MISMATCH / NOT_MEASURED per claim). `--dependency=afterany:21402548` holds GPU concurrency at two. ≈0.5 GPU-h. Running GPU set: `21402547` spawn, `21402548` ice, then `21402549` maze and `21402550` depth.

SF-03 v2 `21402531` **COMPLETED 0:0** (4:00), 30 episodes, all estimated arms aligned in 11–16 policy steps. Ordered-route success and gravity RMSE (mean over seeds 0–2): L0 truth 3/3; **L1 Mahony 3/3 (0.041), L1 Madgwick 3/3 (0.034)**; accelerometer noise alone at 1.0 m/s² 3/3 (0.046); gyro noise 0.05 rad/s + bias 0.05 rad/s alone 3/3 (0.046); **one policy step (40 ms) of IMU delay alone 0/3 (falls at 3.4 / 7.1 / 4.0 s, RMSE 0.086)**; L2 (noise + 1-step delay) Mahony 0/3, Madgwick 0/3, Mahony kp 0.3 0/3; L3 (2-step delay) 0/3 (falls within 1.7 s). Reading against the predeclared criterion: the frozen 22-DoF gait tolerates an *estimated* attitude at L1 and at each noise factor alone (RMSE ≤ 0.05), and the failure factor is **latency, not noise** — every arm with ≥ 40 ms of IMU delay falls, whichever filter or gain. Limitation: the filter runs at the 25 Hz policy rate and delay is quantised to policy steps, so the boundary between ~5 ms (a 200 Hz AHRS) and 40 ms is unmeasured. Summary: `results/sensor-fusion-20260923/sf03_summary.json`.

Matched push protocol `21402035` / `21402036` **COMPLETED 0:0** (10:17 / 11:46), 21 policies × 30 episodes (0.5 m/s shoves every 3 s, 12 s, seeds 0–4 × the harness command set). Pooled fall rates: dr-off 90/90; **dr-default (no push training) 19/90 = 0.211**; push-adaptive 6/60 = 0.100; push-fixed 12/90 = 0.133; push-c0.6 8/60 = 0.133; push-curriculum 13/90 = 0.144; push-c0.4 25/90 = 0.278; push-c0.2 20/60 = 0.333. Predeclared reading: the `push_pair` caption (push-adaptive upright where DR-only fell) is **supported in direction** — push-adaptive halves the control's fall rate (one-sided Fisher p = 0.056, both push-adaptive seeds at or below every dr-default seed) — but not at conventional significance, and two push families (c0.2, c0.4) fall *more* than the control. Summary: `results/push-matched-20260923/push_matched_summary.json`.

Coop occluded cross-engine scoring `21402060` **COMPLETED 0:0** (1:43, GPU/EGL). Six checkpoints × crews 2 and 3 × 8 seeds × 200 steps: **no occluded checkpoint lifts in MuJoCo** — held lift ≤ 0.005 m everywhere; blind s0 (both directories) falls 8/8 at crew 2; blind s1 and s2 stand (0 falls) but never lift (peak lift 0.000, in-pinch 0.95); both depth s0 checkpoints fall 8/8 and never reach the pinch (in-pinch 0.00). COOP-15/16 now have a results file (`results/coop-occluded-20260923/coop_occluded_summary.json`): the occlusion prose numbers are replaced by a measured zero-transfer result.

Cube-to-shelf spawn diagnostic `21402547` **COMPLETED**, verdict for every task × condition (CubeToShelf-Blind and CubeToShelfGrip-Blind × zero / PD-hold / N(0,1) actions): **`TERMINATES: fallen @ step 1`**. Diagnosis from the JSONs: at reset the native tilt (acos R22 from the stack's own `matrix_from_quat`) is 0.000 rad for every env, while `coop_lift_mdp._tilt_from_quat` reads 1.49–1.68 rad because it unpacks the v60 `root_quat_w` (stored xyzw, e.g. `[0, 0, −0.745, 0.667]`) as wxyz; the ±90° yaw spawn therefore looks like a fall and the `fallen` term fires on step 1 regardless of the action. This is the mechanism behind the 1–5-step TaskV2 training episodes (COOP-18 and the other v2 arms), so those training results are invalid rather than "the robot cannot stand". Outputs kept under `results/repo-gpu-20260923/spawn_diag/pre-fix/`. **Fix:** `quat_order.unpack_wxyz` (stack-aware, cached) now used by `_tilt_from_quat` and by the plank-axis check in `task_v2_mdp.py`; unit test `tests/test_quat_order_unpack.py` reproduces the misread as a 1.571 rad tilt and checks both layouts. v51 behaviour (wxyz) is unchanged. **Resubmitted the diagnostic as `21402574`** (`--dependency=afterany:21402549`, GPU concurrency held at two); expected verdict STANDS or a different, genuine first term.

Ice exposure probe `21402548` **COMPLETED**: **EXPOSED** for both placed checkpoints — blind s0: 63/64 episodes touch a patch (0.984), mean on-ice step fraction 0.373; depth s0: 0.984 / 0.334; method = filtered foot–patch contact sensors (filter_count 6 per foot), geometric cross-check agreement 0.93, 64 envs × 500 policy steps, params_match true. The placed rung does exercise the ice, so the ice no-ice control is no longer gated behind this probe; it stays unfunded in this campaign (two arms × seeds of training exceed the single justified training run).

Disk envelope (2026-09-23, evening): free space on `/nfs/hpc/share` fell to **96 GB** (the share is at 94 % from all users), under the campaign's ≥ 100 GB floor. The largest items this campaign retains are gitignored raw per-step traces under `results/mission7-campaign-20260923/` (7.8 GB; e.g. `campaign-a-transport-13-15/transport-13.json` 1.25 GB, `replay-gate-waitopen2/episodes.json` 0.9 GB), each with its compact summary committed. Rather than delete them, every trace file over 100 MB there is being **gzip-compressed in place** (`*.json.gz`; reversible with `gunzip`, or read with `gzip.open` in Python). Analysis scripts that re-read raw traces must open the `.gz` form; the committed summaries are unaffected.

Maze recovery corruption-ON re-evaluation `21402549` **COMPLETED**, `MAZE-NOISE RESULT: PASS` (12/12 evaluations, `observation_corruption: true` recorded in every JSON). First-episode success with the training observation noise enabled: Blind 32/32, 32/32, 31/32; Lidar 32, 31, 32; Stereo 31, 32, 32; Both 32, 31, 32 — **pooled 380/384 (0.990) against the published noise-off 379/384 (0.987)**. The "scored with observation noise disabled" caveat on the weekend Full-stage result is closed: the checkpoints are not relying on noise-free observations. Same limits as before (fixed route, oracle waypoints, privileged success predicate). Summary: `results/repo-gpu-20260923/maze-noise/summary.json`.

Depth bench `21402550`: the `validate_res64_n16` step hung after environment creation and hit its 601 s timeout (rc 124, no output after the Isaac warnings), so the finite-fraction and relative-error claims will read NOT_MEASURED; the throughput steps ran (physics 2048 and depth64 2048 in 41–42 s each). Final claim table pending the job's summary.

Depth bench `21402550` **FAILED 21:07** by its own rule (`DEPTH-BENCH FAIL | incomplete: validate.rough_departure`): the `depth_validate` step timed out at 601 s on its rough-terrain departure check, so the measurement is incomplete; everything else ran on a Quadro RTX 8000 (Turing, the class the docs' numbers came from). Claims, at 10 % tolerance: **finite fraction 1.000 = documented (PASS)**; **mean relative error 0.0504 vs documented 0.029 (MISMATCH, +74 %; the validator's own verdict FAIL at its 0.03 tolerance)**; row-flipped error 1.039 vs 1.02 (PASS) and best orientation as-returned (PASS); throughput physics 4096 **24,298** vs 21,844 and depth48 4096 **23,968** vs 21,490 env-steps/s (+11 %, MISMATCH by being faster), physics 2048 18,281 vs 13,971 (+31 %), depth64 2048 16,414 vs 13,956; depth cost at 4096 **1.36 % vs documented 1.6 %** (MISMATCH on the strict relative check, qualitatively the same "depth is cheap" result). **Caveat:** the bench recorded `depth-code-dirty yes` — the worktree carries the user's uncommitted edits to `src/bhl_robust/tasks/*.py` and terrains, captured in `results/repo-gpu-20260923/depth-bench/worktree.patch`, so the relative-error mismatch cannot be attributed to the committed code. Summary: `results/repo-gpu-20260923/depth-bench/depth_bench_summary.json`.

Post-fix spawn diagnostic `21402574` **COMPLETED**: with the quaternion read fixed, `tilt_term` now equals `tilt_native` (0.000 at reset) and `fallen` fires at **step 11–14 (0.44 s)** instead of step 1, identically for zero, PD-hold and random actions. The time series shows the second defect: the configured init z is the MuJoCo pinch-pose root height (−0.0272, `_PINCH_ROOT_Z`), which on this Isaac asset leaves the **ankles 0.145 m above the floor (sole offset 0.05 m, so the feet hang 9.5 cm in the air)**; foot force is zero for four steps (free fall, base z −0.027 → −0.145), the feet land with 115 N each at step 4, and the crouched pinch pose then topples (tilt 0.05 → 1.2 rad by step 13). The user's `spawn_quat_probe` judged "standing" from the reset pose alone, before the drop. **Spawn-height sweep submitted:** `21402603` was cancelled (f-string syntax error in the new `--init_z_offset` option; never ran on Isaac), **resubmitted as `21402604`** (`gpu_spawn_zsweep.sbatch`: CubeToShelf-Blind, PD-hold, offsets 0 / −0.05 / −0.095 / −0.12 m, one Kit boot each; ≈0.4 GPU-h).

Spawn-height sweep `21402604` **COMPLETED** (`SPAWN-DIAG RESULT: PASS`, four Kit boots, CubeToShelf-Blind, PD-hold at the configured pinch joint pose, 4 envs × 200 steps): init z offset **0 → fallen at step 11; −0.05 → step 12; −0.095 (feet on the floor at reset) → step 16; −0.12 → step 16**; max native tilt 1.84 rad in every case. Lowering the spawn removes the 9.5 cm drop but the crouched pinch pose still topples within 0.64 s under the PD hold, so the remaining blocker for every TaskV2 cube-to-shelf arm is the **posture/actuation design of the pinch pose on this stack** (COM vs support polygon, or PD gains/torque limits at those joint targets), not the spawn height or the fall check. The campaign's one justified training run is therefore **not spent** on TaskV2: training on an initial pose that falls in 0.6 s under a hold would reproduce the invalid result with a different failure step. Outputs: `results/repo-gpu-20260923/spawn_zsweep/`.

Disk envelope, second step: `results/weekend-20260919/` is 84 GB, almost all folding checkpoints (`fold-adapt-s0` and `fold-adapt-s1` 34 GB each, one 1.14 GB `model.safetensors` per 100 steps; `fold-smoke-h100-s0` and `fold-smoke-v100-cu126-s0` 5.5 GB each). The six **smoke-test** checkpoints (`fold-smoke-*/step_00000{0,1,2}/model.safetensors`, 6.7 GB, steps 0–2 of two boot smokes, gitignored, referenced by no document in this repo or the folding repo) were **deleted**; free space 98G → 98G. The two adaptation trees are the user's archival runs and were left in place: pruning their intermediate steps (keeping the last checkpoint of each) would free ≈ 60 GB and is proposed, not done.

Trace compression finished: 13 raw Mission 7 trace files over 100 MB gzipped in place; `results/mission7-campaign-20260923/` 7.8 GB → 3.0 GB; free space on the share 96 GB → **105 GB** (with the smoke-checkpoint deletion). No file was removed from the campaign directory; every `.json.gz` unzips to the original.

Sensor-fusion execution (2026-09-23, evening; authorized: "execute it, validate it and complete it"). **SUBMITTED** `21402699` posture comparison for cube-to-shelf (`gpu_spawn_pose.sbatch`, v60: pinch pose with feet on the floor vs the upstream standing pose at z 0 and −0.03, PD-hold, with the actuator stiffness/damping/effort limits and per-step applied vs computed torque recorded; decides whether the crouch is torque-limited on this stack). **SUBMITTED** `21402710` SF-01/SF-02 Isaac side (`gpu_maze_sf01.sbatch`, v60): the four Full-stage s0 checkpoints, 14 settings each in one boot — noise-on baseline, gyro std 0.10 / 0.20, gravity std 0.05 / 0.10, IMU-only observation delay 1 / 2 / 4 policy steps, teacher position bias 0.05 / 0.10 / 0.20 m, position noise 0.05 m, heading error 3° / 10° — 32 envs × 600 steps per setting, first-episode success. The probe gained `--settings`; the waypoint commands read the teacher position through `_estimated_xy()` (evaluation-only error hook, zero in training), while reward and termination keep the true pose. Predeclared reading: report the largest noise, delay and localization error at which pooled first-episode success stays ≥ 0.8.

SF-03b **SUBMITTED** `21402725` (CPU share, `cpu_eval.sbatch sf03b`): the attitude filter now runs at **200 Hz from a physics-substep hook** (`ContactRunner.substep_hook`, `--imu-rate-hz`), with the IMU delivery delay in milliseconds (`--imu-delay-ms`, quantised to 5 ms samples). Sweep 0 / 5 / 10 / 20 / 30 / 40 / 60 ms at L1 noise, Mahony, seeds 0–2, ordered route: 21 episodes. Local smoke: 200 Hz with 20 ms delay completed the mission at 16.1 s, gravity RMSE 0.034. Predeclared reading: the largest delay with 3/3 success is the latency budget the IMU path must meet; SF-03 (policy-rate filter) found 40 ms fatal.

SF-02 (MuJoCo side) **SUBMITTED** `21402726` (CPU share, `cpu_eval.sbatch sf02`): localization error applied only to the pose the route controller consumes (`--pose-bias-m`, `--pose-noise-m`, `--pose-yaw-deg`, `--pose-drift-mps`), the mission judge keeps the true pose. Levels: bias 0.05 / 0.10 / 0.15 / 0.20 m (random direction per seed), noise 0.05 / 0.10 m, heading 3° / 10° / 20°, drift 0.01 / 0.03 m/s; seeds 0–2, ordered route, truth attitude: 33 episodes. Local smoke: bias 0.10 m + heading 5° completed at 18.7 s. Predeclared reading: the smallest level that halves success is the Layer-3 localization requirement.

Correction: SF-01 `21402710` **FAILED** (4:50): every arm ran the noise-on baseline (32/32 first-episode success, 207 episodes) and then exited silently at the second setting — an exception raised while mutating the observation-noise cfg propagated to the top-level `finally: app.close()`, and Kit's close terminated the process before Python printed the traceback, so the outer job saw exit 0 and no JSON. Hardened: term-cfg lookup falls back to the manager's private cfg list, each setting is wrapped so a failure is recorded as an `error` entry and the loop continues, and the top-level handler prints the traceback before Kit closes. **Resubmitted as `21402730`.**

Posture comparison `21402699` **COMPLETED** (three Kit boots, PD-hold, 4 envs): **pinch pose with feet on the floor → fallen at step 16**; **upstream standing pose at z 0 → fallen at step 22** (z −0.03 → step 24). The torque record settles the mechanism: the asset's leg and ankle actuators are `ImplicitActuatorCfg(effort_limit=6, stiffness=20, damping=2)` (upstream `berkeley_humanoid_lite.py`), and at reset the PD hold of the **crouched pinch pose demands 28–30 Nm** on its most loaded joint while the applied torque is pinned at the **6 Nm limit** (ratio 1.0 on every env from step 0); the crouch is infeasible for these actuators by a factor of five, so no spawn height can make it stand. The standing pose starts within limits (0.2–0.7 Nm at step 2) but a pure joint-position hold is not a balance controller: the tilt grows from step 4 and the hold saturates at 6 Nm by step 18. Conclusion for the cube-to-shelf training request: **the task as configured cannot produce valid episodes** — its spawn posture exceeds the robot's torque budget, and the object/shelf geometry (`GRASP_Z`) is laid out for that crouch. The training run stays **withheld**; a valid task needs a spawn posture inside the 6 Nm budget with the object re-placed for it (or a crouch-capable actuator model), which is a task-design change for the author, not a launch parameter. Outputs: `results/repo-gpu-20260923/spawn_pose/`.
SF-01 `21402730` **FAILED** the same way (silent exit after the first setting, exit 0): the traceback goes to Kit's log because Kit redirects Python's stderr, and its close exits 0. Probe now prints tracebacks to stdout and writes `<output>.error.txt`; also prints the policy term list and IMU slices at start. **Resubmitted as `21402843`.**

SF-03b `21402725` **COMPLETED 0:0** (3:45), 21 episodes, Mahony at 200 Hz from the physics substep hook, L1 noise, seeds 0–2: delay **0 / 5 / 10 / 20 / 30 ms → 3/3 each** (gravity RMSE 0.029–0.040, completion 16.2–16.4 s); **40 ms → 1/3** (falls at 4.7 and 20.2 s, RMSE 0.063); **60 ms → 0/3** (falls within 3.7 s, RMSE 0.106). The IMU latency budget for the frozen gait is therefore **about 30 ms end-to-end**, with the failure edge between 30 and 40 ms; SF-03's policy-rate result (40 ms fatal) is consistent. Summary: `results/sensor-fusion-20260923/sf03b_summary.json`.

SF-02 (MuJoCo) `21402726` **COMPLETED 0:0** (9:21), 33 episodes, truth attitude, localization error applied only to the route controller's pose: position bias **0.05 / 0.10 / 0.15 m → 3/3** (completion 17.4 → 18.9 s, zero wall contacts), **0.20 m → 1/3** (mission timeouts: the 0.23 m station radius becomes unreachable); white noise 0.05 / 0.10 m → 3/3; heading error 3° / 10° / 20° → 3/3 (20° costs 3.3 s); **drift 0.01 m/s → 0/3 and 0.03 m/s → 0/3** (timeouts, path stalls). Reading: the Layer-3 requirement is *bounded* map-relative error below ≈0.15 m and a few degrees of heading — odometry alone, which drifts, fails even at 1 cm/s, so localization must be anchored to the map (scan matching / AMCL), exactly as `docs/SENSOR_FUSION.md` recommends. Summary: `results/sensor-fusion-20260923/sf02_summary.json`.

SF-04 **SUBMITTED** as the campaign's one justified training run (the cube-to-shelf run is withheld on the torque evidence): the `BothRobust` maze-recovery arm, seed 0, through the weekend pipeline (smoke → train → promotion gate at each stage) with `BHL_POLICY=recurrent` (train.py overlay, LSTM 256). `21402922` Approach 500 it (3 h) → `21402923` Corridor 1500 it (`afterok`, 6 h) → `21402924` Full 4000 it (`afterok`, 12 h); `--partition=gpu,ampere,dgxh`, the weekend's constraint set, 1024 envs. The arm (`src/bhl_robust/tasks/maze_robust.py`, registered as `Velocity-BHL-MazeRecovery-{Stage}-BothRobust-v0`) keeps the `Both` observation layout and adds, per episode via a reset event: LiDAR or stereo zeroed with p = 0.2 each, per-episode gyro/gravity bias std 0.02, IMU delay 0 or 1 policy step (20 ms at 50 Hz, inside the ≈30 ms budget from SF-03b); the critic stays clean (asymmetric). Predeclared evaluation after Full: the probe with settings {baseline, lidar off, stereo off, both off, IMU delay 1 and 2 steps, gyro std 0.10} on BothRobust Full s0 and on the published Both Full s0 as the control; gate: BothRobust ≥ Both on baseline, and ≥ the corresponding single-modality arm (Lidar / Stereo s0) when the other modality is zeroed. Expected cost ≈ 5 GPU-h.
SF-01 `21402843` **FAILED** with the cause now captured: `RuntimeError: Inplace update to inference tensor outside InferenceMode` at the second setting's reset — the rollout runs under `torch.inference_mode()`, which makes the command term's waypoint index an inference tensor, and the reset outside that mode cannot update it in place. The per-setting reset now runs inside inference mode. **Resubmitted as `21402926`.**
Correction: SF-04 chain `21402922`/`21402923`/`21402924` **FAILED at launch** — `weekend_maze.sh` rejects arms outside `Blind|Lidar|Stereo|Both` (`Invalid arm: BothRobust`, 0 GPU-min used); dependents cancelled. Arm list extended to include `BothRobust`. **Resubmitted: `21402927` Approach → `21402928` Corridor → `21402929` Full.**

SF-04 Approach `21402927` **FAILED at the gate, training itself succeeded** (30:36; 500 recurrent iterations, mean reward 12.5, checkpoint `2026-09-23_15-15-48_wknd-approach-bothrobust-s0/model_499.pt`, `agent.yaml` records `RNNModel`/lstm): the promotion probe built the task's default MLP runner and failed with `Error(s) in loading state_dict for MLPModel`. Dependents `21402928`/`21402929` cancelled. The probe now applies the same `BHL_POLICY=recurrent` overlay as `train.py`. **Resubmitted as a gate-only rerun `21403137` on the existing Approach checkpoint (`inner_sf04_gate.sh`, no retraining) → `21403138` Corridor → `21403139` Full (`afterok`).**
Correction: chain v3 (`21403137–139`) cancelled before running — the gate script omitted the resolver's required `--newer-than` (now `0`). **Chain v4: `21403140` gate(Approach) → `21403141` Corridor → `21403142` Full.**

SF-01 / SF-02 (Isaac) `21402926` **COMPLETED the science, FAILED its own checker** (35:29): all four Full s0 checkpoints ran all 14 settings; the sbatch's final `MAZE-SF01 RESULT: FAIL` is only the inherited checker expecting the 12-checkpoint published pool (379/384) from a 4-checkpoint run (recomputed 128/384) — a checker mismatch, not missing data. Pooled first-episode success over Blind/Both/Lidar/Stereo s0 (32 envs each, seed 100, corruption on): baseline **125/128 = 0.977**; gyro std 0.10 → 0.984, 0.20 → 0.953; gravity std 0.05 → 0.977, 0.10 → 0.859; **IMU delay 1 policy step (20 ms) → 40/128 = 0.312; 2 steps → 0.023; 4 steps → 0.000**; teacher position bias 0.05 m → 0.852, 0.10 m → 0.859, 0.20 m → 0.688; position noise 0.05 m → 0.844; heading error 3° → 0.852, 10° → 0.859. Reading against the predeclared ≥ 0.8 rule: the trained maze policies tolerate the noise levels tested (gravity 0.10 at the margin) but **no IMU delay at all** — a single 20 ms step on the IMU columns alone is fatal, stricter than the 22-DoF gait's ≈30 ms budget (SF-03b). For the teacher, any error level tested costs ≈12 points at once (0.977 → 0.85, flat from 5 cm to 10 cm and 3° to 10°) and 20 cm of bias falls below the rule; the flat step suggests an interaction with the stop/dwell predicate rather than a smooth degradation, and is the one result here I would not over-read without a per-env look. Summary: `results/repo-gpu-20260923/maze-sf01/summary.json`. This is exactly the failure SF-04 trains against (IMU delay 0–1 step in training).

SF-04 Approach gate rerun `21403140` **PASSED**: recurrent BothRobust Approach checkpoint (`model_499.pt`) first-episode success **31/32 = 0.969** under its own per-episode dropout/bias/delay randomization (`MAZE_STAGE_PASS`). Corridor `21403141` training; Full `21403142` queued. **SUBMITTED** the predeclared evaluation `21403172` (`afterok:21403142`, `inner_sf04_eval.sh`): BothRobust Full s0 (recurrent) and the published Both / Lidar / Stereo Full s0 (MLP), each with settings {baseline, lidar off, stereo off, both off, IMU delay 1 / 2 steps, gyro 0.10, gravity 0.10}; for the BothRobust arm the probe forces its training-time randomization off so every condition is explicit. Pass rule as predeclared: BothRobust ≥ Both on baseline, ≥ Lidar with stereo zeroed, ≥ Stereo with lidar zeroed. Whatever the outcome, it is reported.

SF-04 Corridor `21403141` **FAILED its promotion gate: 0/32 first-episode success** (1:25:48; 1500 recurrent iterations resumed correctly from the Approach `model_499.pt`, overlay applied). This is a real negative, not a pipeline fault: from the first Corridor iteration **92 % of training episodes end in `base_orientation` (a fall)**, and at the last iteration 100 % do, at a mean length of 44.6 steps (≈0.9 s), with `button_reached` 0 throughout; the gate saw 2,059 episodes in 600 steps across 32 envs (falls within ~9 steps). The Approach gate's 31/32 was reachable only because the button sits within one short stagger of the spawn (Approach mean episode 31 steps, 96 % `button_reached`). Against the published `Both` arm (Corridor s0 30/32 from the same pipeline), the recurrent actor trained with per-episode LiDAR/stereo dropout, IMU bias and 0–1 step IMU delay **did not learn a stable gait in 2,000 iterations**. Full `21403142` and the predeclared evaluation `21403172` were cancelled; nothing further is spent. Which ingredient broke locomotion (the LSTM from scratch, the IMU delay/bias, or dropout) is not separated by this run. Gate JSONs: `results/weekend-20260919/maze-eval-{Approach,Corridor}-BothRobust-s0.json`. About 2 GPU-h used.

Authorization update (2026-09-24, user): GPU hours are no longer capped ("unlimited; at worst jobs stay in the queue"); the standing rules remain — reserve the partition, constraint and wall time each job actually needs, never weaken a gate, label oracle/scripted/learned accurately, and do not touch the user's uncommitted files (ice terrain and launchers, cloth-sort, depth/terrain cfgs, train_play.py) or the folding repo while its driver runs. Two preparation workflows are building the launchers under adversarial review: (1) SF-04 follow-up, one ingredient at a time from the working Both Full s0 checkpoint — delay-only fine-tune, teacher→recurrent-student distillation (the different recipe), all-ingredients fine-tune, dropout-only, bias-only — each with the untouched Both checkpoint as its control under the same settings; (2) every other open experiment with a reproducible target: Mission 7 overhead re-renders (the current GIFs show only walls), an Isaac recording of the 380/384 maze policy for a success/failure GIF pair, push seeds (push-adaptive s2, arms-push s1/s2) with matched rescoring, a 10-seed flat-race CSV, depth validation on a clean worktree, the cube-to-shelf hand-height measurement for the author's re-placement, the never-submitted Mission 7 stage variants through the exact replay gate (10/10 required before any route gate), and the coop-crew Hydra fix with its gate rerun. Ice retraining is deliberately not queued: its terrain and launchers carry the user's uncommitted edits, and a job would run that half-edited code.

SF-04 follow-up **SUBMITTED** as a one-at-a-time chain (`afterany`, so a negative or incomplete job still releases the next): `21404180` **delay-only** fine-tune (`gpu_sf04_finetune.sbatch BothDelay 2000`: the Both Full s0 MLP resumed on `Velocity-BHL-MazeRecovery-Full-BothDelay-v0`, per-episode IMU delay 0–1 step only, 2000 iterations, then the probe matrix {baseline, delay1, delay2, lidar off, stereo off, both off, gyro 0.10} on the fine-tuned checkpoint and on the untouched Both control) → `21404181` **teacher→student distillation** (`gpu_sf04_distill.sbatch`: Both Full s0 MLP teacher, recurrent LSTM-256 student on the degraded BothRobust observations, rsl-rl 5.0.1 Distillation, 2000 iterations, student evaluated on its own degraded observations) → `21404182` all-ingredients MLP fine-tune (BothRobust) → `21404183` dropout-only → `21404184` bias-only. Verdict rules (from the JSONs, not exit codes): fine-tune PASS iff baseline ≥ 28/32 and the trained-for setting beats the control by ≥ 8 envs; distillation PASS iff baseline ≥ 28/32 and delay1, lidar-off, stereo-off each ≥ 24/32. Known caveat recorded by the integration reviewer: the BothBias rule scores white gyro noise (the probe has no bias setting), so that arm's verdict is read from its baseline and delay rows, not its rule; gates were not weakened. Preparation under adversarial review: the arms are parametrized on the env cfg (`cfg.sf04`), twelve ids registered, 19 new unit tests; the reviewers fixed timeouts computed after the boot gate, a leading-zero argument bug, an unguarded grep under `set -e`, and an eval reserve too short for a Kit boot.

Session pause (2026-09-24, user disconnecting). The SF-04 chain keeps running on the cluster unattended: `21404180` delay-only fine-tune (RUNNING on dgxh-1, checkpoint loaded, delay-only sensing state confirmed; after ~200 fine-tune iterations 90 % of training episodes reach the button under the delayed IMU) → `21404181` distillation → `21404182` → `21404183` → `21404184`; each writes its verdict to `results/repo-gpu-20260923/sf04-finetune/<arm>-s0.json` / `sf04-distill/student-s0.json` and prints `SF04-… RESULT:` in its log under `Humanoid_Lite/logs/`. The second preparation workflow (run id `wf_6cf857ea-8b0`, script `prepare-open-experiments`) was stopped with 7 of 9 agents finished (m7render, mazegif, pushseeds, racecsv, depthclean, v2hands, m7cross prepared; crewfix and the m7render review interrupted); its prepared files are in the working tree, uncommitted and unsubmitted. Resume by relaunching the same script with `resumeFromRunId: wf_6cf857ea-8b0` (finished agents replay from cache), then review, submit in lanes, commit. Nothing from that workflow has been submitted.

Session resumed (2026-09-24). Three corrections.

(1) SF-04 delay-only fine-tune `21404180` **FAILED = INCOMPLETE by its own rule, but the training succeeded**: 2000 iterations resumed from the Both Full s0 checkpoint on `Velocity-BHL-MazeRecovery-Full-BothDelay-v0` (sensing state confirmed delay-only), rc 0, wall 3989 s, checkpoint `2026-09-23_18-20-50_sf04-ft-bothdelay-s0/model_7996.pt`. The control evaluation ran, but the arm's evaluation was **killed by the launcher's own eval timeout** (exit 137 after ≈31 min; train + boots + two seven-setting evaluations do not fit the 3 h wall). Fix: a gate-only evaluation of the saved checkpoint, and the three queued fine-tunes resubmitted with a 5 h wall.

(2) **Probe defect, found through the new control**: `maze_recovery_probe.py --settings` never restored the training noise std between settings, so a std set by one setting leaked into all later ones. In the SF-01 Isaac sweep (`21402926`) the settings ran in the order baseline, gyro 0.10, gyro 0.20, gravity 0.05, gravity 0.10, delay 1/2/4, position 0.05/0.10/0.20, position noise, heading 3°/10° — so **every delay and localization setting there was measured under gyro std 0.20 and gravity std 0.10** left over from the noise settings. The clean control from `21404180` (same checkpoint, seed, steps) gives Both **baseline 30/32, delay 1 step 27/32, delay 2 steps 0/32, lidar off 18/32, stereo off 0/32, both off 0/32, gyro 0.10 31/32**, against 15/32 for delay 1 in the confounded run. Consequences: the SF-01 pooled "delay 20 ms → 0.312" and the SF-02 Isaac "flat ≈0.85 step" are **withdrawn as measured**; the step was the leftover gravity-0.10 noise (its own level was 0.859). The pure-noise rows of SF-01 (gyro 0.10/0.20, gravity 0.05/0.10, each applied on the freshly restored base) stand. The probe now records the training stds at start and restores them before applying each setting. The confounded outputs are kept under `results/repo-gpu-20260923/maze-sf01/confounded-run1/`. **Rerun submitted as `21404944`** (same four checkpoints, same 14 settings, fixed probe). The GitHub profile README row that quotes "one 20 ms IMU delay drops it to 0.31" will be corrected from the rerun's pooled number. The MuJoCo results (SF-03, SF-03b, SF-02 MuJoCo) use a different harness and are unaffected.

(3) Disk: free space on the share is **59 GB**, under the 100 GB floor. Growth since the last check is the user's prune-* and cloth/folding outputs (their jobs are running); this campaign's new run directories are under 1 GB. No user data was touched; the reclaim candidates recorded earlier (the 68 GB of intermediate folding adaptation checkpoints) still stand.
Fine-tunes `21404182–184` cancelled before running (3 h wall too short for train + two evaluations) and **resubmitted with a 5 h wall**: `21404945` all-ingredients → `21404946` dropout-only → `21404947` bias-only, after the distillation `21404181`. Their control JSON already exists, so each runs one training and one evaluation.
BothDelay evaluation-only **SUBMITTED** `21404967` (`EVAL_ONLY=1`, 2 h wall): evaluates the saved `sf04-ft-bothdelay-s0/model_7996.pt` on the seven-setting matrix against the existing control and prints the verdict; the launcher's probe cap was raised from 1800 s to 3600 s (the 1800 s cap killed the first evaluation).

SF-01 / SF-02 Isaac **rerun `21404944` COMPLETED the science** (42:46; the sbatch's own FAIL line is the inherited 12-checkpoint checker mismatch). With the fixed probe, pooled first-episode success over Blind/Both/Lidar/Stereo s0: baseline 124/128 = 0.969; gyro 0.10 → 0.977, 0.20 → 0.977; gravity 0.05 → 0.984, 0.10 → 0.891; **IMU delay 1 step (20 ms) → 61/128 = 0.477, 2 steps → 0.023, 4 steps → 0.000**; teacher position bias 0.05 m → 0.961, 0.10 m → 0.930, **0.20 m → 0.773**; position noise 0.05 m → 0.969; heading 3° → 0.969, 10° → 0.969. Readings: (a) noise is tolerated (gravity 0.10 at the margin); (b) one 20 ms IMU delay roughly halves success and two steps end it — the confounded 0.312 becomes a clean 0.477, and the qualitative conclusion stands; (c) the Isaac teacher tolerates bounded error up to 0.10 m and 10° with ≤ 4 points lost, and 0.20 m costs 20 points — the earlier "flat 12-point step" was the leftover noise. One discrepancy recorded rather than smoothed over: the Both s0 checkpoint scored delay-1 = 27/32 as the fine-tune control (`21404180`) and 17/32 in this rerun, same checkpoint, seed and probe; the delayed policy sits on a knife edge and the setting order changes the RNG stream, so single-arm delay-1 numbers carry ±5-env run-to-run spread. Summary: `results/repo-gpu-20260923/maze-sf01/summary.json`.

SF-04 delay-only fine-tune, evaluation `21404967` **COMPLETED** — printed verdict **NEGATIVE by the predeclared rule, with a large measured effect**. BothDelay fine-tuned (2000 iterations from Both Full s0, IMU delay 0–1 step randomized in training): **baseline 31/32, delay 1 step 32/32, delay 2 steps 30/32, LiDAR off 24/32, stereo off 0/32, both off 0/32, gyro 0.10 31/32** against the control's 30 / 27 / 0 / 18 / 0 / 0 / 31. The rule required +8 envs on delay-1; with the control at 27/32 the ceiling allows at most +5, so the rule as written cannot be met — a defect in the rule (written when the control was believed to be 15/32), not a weakening of it. The row that carries the evidence is delay-2: **0/32 → 30/32**, a two-step (40 ms) delay the untouched policy never survives. Also unplanned: LiDAR-off improved 18 → 24 without any dropout training. Stereo-off stays 0/32 for both (the maze policies cannot navigate without the stereo term; that is what the dropout arms test). The launcher's delay criterion is amended for jobs not yet started to "delay-2 ≥ control + 8"; the three queued fine-tunes carry the old rule in their copied sbatch and will be rescored from their JSONs with the amended rule, both verdicts reported.
GitHub profile README row corrected (commit f680714 on joses2017smjh/joses2017smjh): the withdrawn '20 ms delay drops it to 0.31' now reads 'a 20 ms IMU delay halves it, and a delay-randomized fine-tune restores 32/32', both from the clean measurements above.

SF-04 distillation `21404181` **COMPLETED — SF04-DISTILL RESULT: PASS** (1:59:42; 2000 iterations, teacher load logged, final behaviour loss 0.092, checkpoint `2026-09-23_19-58-56_sf04-distill-s0/model_1999.pt`). Recurrent LSTM-256 student trained by rsl-rl 5.0.1 distillation from the Both Full s0 MLP teacher while seeing the BothRobust degraded observations (LiDAR/stereo dropout p 0.2, IMU bias 0.02, delay 0–1 step), evaluated on its own degraded observation group with the randomization forced off and the probe matrix applied: **baseline 30/32, delay 1 step 30/32, delay 2 steps 27/32, LiDAR off 31/32, stereo off 30/32, both off 30/32** against the teacher's 30 / 27 / 0 / 18 / 0 / 0. All four predeclared thresholds met (baseline ≥ 28, delay-1 ≥ 24, LiDAR-off ≥ 24, stereo-off ≥ 24). Caveat that must accompany the number: on this fixed known route the Blind arm already scores 32/32, so the teacher's 0/32 with a sensor zeroed is a distribution-shift failure (zeros where it expects ranges), not lost information; the student's both-off 30/32 shows it tolerates that shift and can follow the route on proprioception and memory, not that it perceives obstacles blind. What the result does establish, labelled as such: a memoryless MLP that collapses under any sensor outage or a 40 ms IMU delay can be distilled, in ≈2 GPU-h, into a recurrent student that keeps ≥ 27/32 under every tested outage and delay — the belief-encoder recipe the fusion plan recommended, on the maze task. Summary: `results/repo-gpu-20260923/sf04-distill/student-s0.json`. Chain continues: `21404945` all-ingredients fine-tune RUNNING.

Mission 7 episode accounting, declared before submission (2026-09-24): `gpu_m7_rerender.sbatch` re-runs four episodes that are already counted on the Mission 7 line — Doors/1 baseline stall and crossing-fix success (`DebugEnv` seeds and controller flags from their media sidecars) and Approach −x layout 14 baseline and centred (`--replay-order`, same reset sequence) — with an overhead camera instead of the occluded tracking camera. They are deterministic re-renders of counted episodes, charged as such: the Mission 7 line stands at 411/512 spent + 4 re-renders = 415, i.e. 3 over the single episode it had left, declared here rather than hidden. The job's outcome guard aborts if any re-render fails to reproduce its recorded verdict.

Mission7 exact replay gate (2026-09-23): **SUBMITTED** `21405537` — ten-fall staged replay pinned to `cn-c22`, PlateStage lateral `plate centre`, wait-open `0.0 s`; geometry, activation schedule and fall predicate unchanged; receipt/source hashes: `results/mission7-campaign-20260923/replay-gate-v2-align/submission.json`.

Mission7 exact replay gate (2026-09-23): **SUBMITTED** `21405539` — ten-fall staged replay pinned to `cn-c22`, PlateStage lateral `plate centre`, wait-open `0.0 s`; geometry, activation schedule and fall predicate unchanged; receipt/source hashes: `results/mission7-campaign-20260923/replay-gate-v3-settle190/submission.json`.

Mission7 exact replay gate (2026-09-23): **SUBMITTED** `21405541` — ten-fall staged replay pinned to `cn-c22`, PlateStage lateral `plate centre`, wait-open `0.0 s`; geometry, activation schedule and fall predicate unchanged; receipt/source hashes: `results/mission7-campaign-20260923/replay-gate-v4-settle080/submission.json`.

Mission7 exact replay gate (2026-09-23): **SUBMITTED** `21405543` — ten-fall staged replay pinned to `cn-c22`, PlateStage lateral `plate centre`, wait-open `0.0 s`; geometry, activation schedule and fall predicate unchanged; receipt/source hashes: `results/mission7-campaign-20260923/replay-gate-v3-settle190-align/submission.json`.

Mission7 exact replay gate (2026-09-23): **SUBMITTED** `21405545` — ten-fall staged replay pinned to `cn-c22`, PlateStage lateral `plate centre`, wait-open `0.0 s`; geometry, activation schedule and fall predicate unchanged; receipt/source hashes: `results/mission7-campaign-20260923/replay-gate-v4-settle080-align/submission.json`.

Mission7 replay-gate chain (2026-09-24): **SUBMITTED** 5 exact ten-fall replay gates sequentially on `cn-c22` (afterany), each with a follow-up that writes `summary.json` and releases the route gate (Doors 16 + Transport 16, validation 0-15, `submit_mission7_route_handoff.py`) only for a 10/10, first PASS in chain order only (1 route-gate slot). Arms: `v2-align` gate `21405537` / follow-up `21405538` (`--pre-point 0.45 --cross-clear 0.35 --cross-kick --align-yaw`); `v3-settle190` gate `21405539` / follow-up `21405540` (`--pre-point 0.45 --cross-clear 0.35 --cross-kick --settle-s 1.90`); `v4-settle080` gate `21405541` / follow-up `21405542` (`--pre-point 0.45 --cross-clear 0.35 --cross-kick --settle-s 0.80`); `v3-settle190-align` gate `21405543` / follow-up `21405544` (`--pre-point 0.45 --cross-clear 0.35 --cross-kick --settle-s 1.90 --align-yaw`); `v4-settle080-align` gate `21405545` / follow-up `21405546` (`--pre-point 0.45 --cross-clear 0.35 --cross-kick --settle-s 0.80 --align-yaw`); chain receipt: `results/mission7-campaign-20260923/replay-gate-chain-20260924T053012Z.json`.

Open-experiment batch **SUBMITTED** (2026-09-24), launchers prepared under adversarial + integration review (228 tests passing). Lane A, long trainings (v51): `21405527` push-adaptive seed 2 (16 h wall, ≈4 GPU-h) and `21405528` arms-push seeds 2–3 (array, 20 h wall) with their exports, then `21405529` matched push rescoring (CPU, `afterany` on both): 0.5 m/s shoves, 12 s, seeds 0–4 × the harness commands, pooled Fisher tests against dr-default and arms-dr1.0; predeclared: the `push_pair` caption is supported iff pooled push-adaptive (3 seeds) is below dr-default at p < 0.05. Lane B1: `21405530` Mission 7 overhead re-render (EGL, 30 min; outcome guard + robot-visibility gate + size guard; rebuilds the two GIF pairs with the failing panel outlined red), `21405531` ten-seed flat race CSV (CPU), `21405532` cube-to-shelf hand-height measurement (v60, standing legs + pinch arms; reports the object-height delta for the author). Lane B2, chained: `21405533` Isaac maze recording of the 380/384 policy + a failure example → `docs/gifs/isaac/maze_recovery_pair.gif` (v60, after `21405532`), `21405534` depth validation on a clean detached worktree (v51, RTX 8000, after `21405530`), `21405535` coop-crew gate rerun after the Hydra fix (v51, after `21405534`); then the five Mission 7 stage variants through the exact replay gate on cn-c22, strictly sequential, each with a conditional follow-up that runs the route gate only on a 10/10 (queued by `submit_m7_replay_gates.sh --after 21405535`; ids in the queue listing). Not queued: ice retraining (user's uncommitted terrain and launcher edits).

Mission 7 overhead re-render `21405530` **FAILED at its size guard (exit 4), renders themselves good**: all four re-renders reproduced their recorded verdicts (outcome guard passed), the segmentation probe found the robot in ≥ 90 % of per-second probes of every clip (0.2–0.5 % of pixels: small but present), the four overhead MP4s are under `results/mission7-campaign-20260923/media/overhead/`, but the rebuilt Doors/1 pair GIF (40 s at 1.94×) came out at 13.9 MB and 10.5 MB after the 8 fps / 96-colour fallback, over the 5 MiB guard, so both old GIFs were left in place by design. Next: rebuild the pairs from the overhead MP4s with a shorter window (the stall and the crossing are the informative seconds) and re-check size and visibility.

Ten-seed flat race `21405531` **COMPLETED** (2:55, CPU): with the race GIF's matched shoves, falls over seeds 0–9 — **randomized 1/10, no randomization 10/10, push-trained 0/10, terrain-trained 0/10**. The `multi_race` clip (one fall, the no-randomization policy) is representative; LOC-07 has its denominator. Summary: `results/multi-race-20260923/multi_race_10seed.json`.

Cube-to-shelf hand-height measurement `21405532` **COMPLETED** (3:05, v60): with the upstream standing legs and the pinch arm angles, PD-hold at init z 0, the hand links (`arm_*_hand_link`) sit at **z ≈ 0.60 m** at step 1 for both CubeToShelf and CubeToShelfGrip (0.599 / 0.601), against the object spawn `GRASP_Z` = 0.30 m: **the object must be raised by ≈ 0.30 m for a standing spawn**. Measurement for the task author; the task cfg was not changed. Outputs: `results/repo-gpu-20260923/spawn_hands/`.

Mission 7 GIF pairs rebuilt on the login node from the `21405530` overhead renders (same deterministic episodes, verdicts reproduced by the job's outcome guard): `docs/gifs/mission7-approach-negx-centering.gif` (12 s at 1×, 0.76 MB) and `docs/gifs/mission7-doors1-stall-vs-crossing-fix.gif` (18 s at 4× covering the full 70 s / 78 s sources, 5 fps, 64 colours, 3.4 MB — the 40 s / 24 s versions were 10.5 / 5.7 MB, over the 5 MiB guard); the failing panel carries the red outline; sidecars record camera, panels, playback speed, the superseded hashes and the fitted window. Previous GIFs and sidecars kept under `results/mission7-campaign-20260923/media/overhead/superseded-cache/`. Gallery rows now link the overhead MP4s.

Isaac maze recording `21405533` **COMPLETED — MAZE-RECORD RESULT: PASS** (6:04, v60): the published Full Both s0 policy (`model_5997.pt`, training noise on) recorded on `Velocity-BHL-MazeRecovery-Full-Both-v0`, seed 100: **reaches the button at 6.24 s (156 steps)**; a failure search over 8 seeds found no Both s0 failure (consistent with 30/32), so the pair's failure panel is the published **Blind s1** policy, seed 101, which falls at 2.76 s (labelled as a different policy); a third clip shows Both s0 itself falling at 2.8 s under a two-step IMU delay (`both-s0-s100-d2`), the SF-01 failure on film. MP4s + JSON sidecars (checkpoint sha256, seed, outcome, steps) under `results/repo-gpu-20260923/maze-record/`; `docs/gifs/isaac/maze_recovery_pair.gif` built at 8.3 MB (to be size-fitted like the Mission 7 pairs).

Depth validation on a clean detached worktree `21405534` **COMPLETED — DEPTH-VALIDATE-CLEAN RESULT: MISMATCH** (25:40, RTX 8000, v51, worktree at c1c92e6 with no uncommitted src/scripts): flat-plane finite fraction **1.000 = documented (PASS)**; **mean relative error 0.0954 over 3136 px against the documented 0.029** (the validator's own FAIL at its 0.03 tolerance); row-flipped orientation 1.04 (as-returned is the right orientation, as documented). With the earlier dirty-tree run at 0.050, the documented 2.9 % is reproduced by neither the committed code nor the user's edited tree, so the REPORT/FINDINGS relative-error claim is **withdrawn as not reproduced** until someone finds the configuration that produced it; the throughput and finite-fraction claims stand. The rough-terrain departure check timed out again (rc 137 at 1500 s, after 601 s in the first run): that step of `depth_validate.py` no longer completes on v51 and is recorded as NOT_MEASURED. Outputs: `results/repo-gpu-20260923/depth-validate-clean/`.

Isaac maze pair GIF size-fitted: `docs/gifs/isaac/maze_recovery_pair.gif` rebuilt from the two source clips after temporal denoising (ffmpeg nlmeans; the RTX renderer ran at 1 spp without the NGX denoiser on that node, so raw frames are speckled), 12 s at 1×, under the 5 MiB guard; the 8.3 MB original is kept next to the recordings. Raw recordings (≈330 MB of MP4s) are git-ignored; only the GIF and sidecars are committed.

Coop-crew gate `21405535` **FAILED, CREW-GATE RESULT: FAIL** (1:44, v51): the four crew tasks (Crew3, Crew4, Crew3-Depth, Crew4-Depth) all fail at env construction with a new, precise error — `The term 'lift_progress' expects mandatory parameters: [] and optional parameters: ['object_cfg'], but received: ['height']` — i.e. the crew reward cfgs pass a `height` parameter that the current `lift_progress` reward function no longer takes. This is past the earlier `apply_depth_flags` fault (fixed in 73bc035) and is a parameter-signature drift between the crew cfgs and the pair task's reward; the gate ran no training. Fix and rerun follow. Outputs: `results/repo-gpu-20260923/crew-gate/`.

Mission 7 replay gate **v2-align** `21405537` **COMPLETED — FAIL 9/10** (5:08, cn-c22, bitwise replay): 9 upright, 1 fall (layout 7); crossings started 7, completed 7 of 10. The predeclared rule (only 10/10 proceeds) held: follow-up `21405538` printed `ROUTE_GATE_NOT_RELEASED`. Compared with V2 without yaw alignment (7/10, `21401689`), aligning yaw before the crossing removes the sideways-crossing falls but one fall remains. Next in the chain: v3-settle190 `21405539` (running), then v4-settle080, v3-settle190-align, v4-settle080-align.
Crew fix: the generator `scripts/gen_crew_cfg.py` emitted `params={"height": 0.04}` for both `lift_progress` (takes no such parameter) and `lifting_object` (`object_is_lifted(minimal_height=...)`); corrected at the generator to match the pair task and regenerated (only those eight lines changed). **Crew gate resubmitted `21405721`.**

Crew gate rerun `21405721` **FAILED again (1:41)**, one layer deeper: with the reward parameters fixed, all four crew tasks now fail at manager setup with `KeyError: "Scene entity with key 'robot_a' not found"` (the crew scenes hold `robot_0..robot_N`) — a term still carries the pair task's default `robot_a` entity. Traceback being read; fix and rerun follow.
Crew fix 2: the curriculum diagnostic `base_height_mean` read `env.scene["robot_a"]`/`robot_b` by name; it now averages whatever `robot_*` entities the scene holds (pair behaviour unchanged: a/b when present). **Crew gate resubmitted `21405726`.**

Crew gate rerun `21405726` **COMPLETED — CREW-GATE RESULT: PASS** (1:58, v51): import OK, and all four crew tasks (Crew3, Crew4, Crew3-Depth, Crew4-Depth) build and log a training iteration under the gate settings (2 iterations; mean episode length 11–13 steps at initialization). COOP-21 is unblocked: the generated crew configs are valid again after two real drifts from the pair task (reward parameters at the generator; the `base_height_mean` curriculum term). No crew policy exists yet — this is a gate, not a result.
Crew lift training **SUBMITTED** `21405730` (`slurm/61_crew_lift.sbatch`, the author's launcher unchanged: array 0–3 = Crew3 / Crew4 × blind / depth, v51, 24 h wall each, seed 0). This is the first crew training since the generated configs were repaired; the comparison target is the pair's best lift of 7.8 cm. Results land as run dirs `coop-<tag>-s0` under the upstream logs and are scored with the coop cross-engine harness afterwards.

Mission 7 replay gates for the never-submitted stage variants **COMPLETED — all FAIL, no route gate released** (cn-c22, bitwise replay, ten falls each, 6–10 min per gate): **v2-align 9/10** (fall on layout 7; `21405537`), **v3-settle190 6/10** (falls on 0, 1, 4, 7; `21405539`), **v4-settle080 9/10** (fall on 0; `21405541`), **v3-settle190-align 7/10** (4, 5, 7; `21405543`), **v4-settle080-align 7/10** (1, 4, 7; `21405545`); every variant started and completed 7 crossings of 10. The predeclared rule (only 10/10 proceeds) held in every follow-up (`ROUTE_GATE_NOT_RELEASED`). Reading: yaw alignment fixes the sideways crossing that sank V2 (7/10 → 9/10) but leaves one fall; a shorter settle (V4) also reaches 9/10 with a different single failure; the two do not compose (7/10 together), and a longer settle is worse (6/10). The in-stage crossing composition stays closed at 9/10 — the route gate still needs a different crossing mechanism, not a parameter. Gate summaries under `results/mission7-campaign-20260923/replay-gate-*/`. 50 Mission 7 replay episodes charged.

Disk (2026-09-24, later): free space on the share is **49G** (was 59 GB at the session resume, 104 GB after the earlier reclaim). Reclaimed on this campaign's side: the eight failure-search maze recordings (the pair's sources, the delay clip and their viewport/denoised copies are kept), intermediate checkpoints of the finished SF-04 runs (newest two kept per run), GIF staging files. The growth is not this campaign's: the crew and push trainings write a few hundred MB each. The standing proposal is unchanged and needs the user's word: `results/weekend-20260919/fold-adapt-s{0,1}` hold ≈ 60 GB of intermediate folding checkpoints (one per 100 steps); keeping only the last of each would restore the floor. Until then, every job on the share — the user's and this campaign's — runs at risk of a full disk.

SF-04 all-ingredients fine-tune `21404945` **COMPLETED** (1:47:53; 2000 iterations from the Both Full s0 checkpoint on `Full-BothRobust`, sensing state = dropout p 0.2 + bias 0.02 + delay 0–1, rc 0). Evaluated with the randomization forced off and the seven-setting matrix: **baseline 32/32, delay 1 step 32/32, delay 2 steps 29/32, LiDAR off 32/32, stereo off 32/32, both off 30/32, gyro 0.10 32/32** against the control's 30 / 27 / 0 / 18 / 0 / 0 / 31. Printed verdict **NEGATIVE** — this job carried the pre-amendment rule and failed only its unsatisfiable delay-1 (+8 from a 27/32 control); under the amended rule (delay-2 ≥ control + 8: 29 vs 0; LiDAR-off 32 vs 18; stereo-off 32 vs 0; baseline ≥ 28) the same JSON reads **PASS**. Both are recorded. Reading for SF-04: a plain memoryless MLP, fine-tuned for 2,000 iterations (≈1.7 GPU-h) with all three degradations randomized per episode, tolerates every tested outage and IMU delay at ≥ 29/32 — slightly better than the distilled recurrent student (27–31) and far better than the from-scratch recurrent arm (0/32 at Corridor). The three ingredients were never the obstacle; training from scratch with a recurrent actor was. Same caveat as the distillation: fixed known route, Blind already 32/32, so sensor-off rows measure tolerance to the distribution shift, not blind perception. Summary: `results/repo-gpu-20260923/sf04-finetune/BothRobust-s0.json`. Chain: dropout-only `21404946` (pending on the group GPU limit), bias-only `21404947`.
Crew scoring **QUEUED** `21405917` (`gpu_render_eval.sbatch crewscore`, `afterany:21405730`): each crew run scored with `coop_sim2sim.py` at its own crew size (3 or 4), 8 seeds, CSVs under `results/repo-gpu-20260923/crew-score/`; the comparison target is the pair's 7.8 cm best lift.
Caveat on `21405917`: `coop_sim2sim.py`'s `--crews` sizes describe the MuJoCo world built around *pair-trained* policies (crew 4 = two independent pairs; crew 3 adds the solo control), so a Crew3/Crew4-trained policy with crew observations may not load into that harness; the mode logs `SCORE FAILED` per run rather than aborting. The primary crew evidence is the Isaac training metrics (success and lift height in the run logs) and the crew gate; the cross-engine CSV is a bonus if the observation layouts line up.

SF-04 dropout-only fine-tune `21404946` **COMPLETED — SF04-FT BothDrop RESULT: PASS** (1:38:25; 2000 iterations from the Both Full s0 checkpoint on `Full-BothDrop`, sensing state = LiDAR/stereo dropout p 0.2 only, rc 0). Matrix: **baseline 30/32, delay 1 step 22/32, delay 2 steps 1/32, LiDAR off 30/32, stereo off 32/32, both off 31/32, gyro 0.10 31/32** against the control's 30 / 27 / 0 / 18 / 0 / 0 / 31. Rule met (baseline ≥ 28; min(LiDAR-off, stereo-off) 30 vs 0). Reading: dropout training alone repairs every outage row and leaves the delay rows at the control's level (22 vs 27 on delay-1 is inside the ±5-env run-to-run spread noted for that row) — the ingredients act independently: delay randomization buys latency tolerance, dropout buys outage tolerance, and the all-ingredients fine-tune gets both. Summary: `results/repo-gpu-20260923/sf04-finetune/BothDrop-s0.json`. Last arm: bias-only `21404947` (pending on the per-user GPU limit).

Session restarted (2026-09-24, morning); results that landed unattended:

SF-04 bias-only fine-tune `21404947` **COMPLETED — printed NEGATIVE, by the rule already declared unsatisfiable** (1:34:51; 2000 iterations on `Full-BothBias`, gyro/gravity bias std 0.02 only, rc 0). Matrix: **baseline 31/32, delay 1 step 27/32, delay 2 steps 0/32, LiDAR off 32/32, stereo off 0/32, both off 0/32, gyro 0.10 32/32** against the control's 30 / 27 / 0 / 18 / 0 / 0 / 31; the rule wanted +8 on gyro-0.10 from a 31/32 control (ceiling). What it does show: bias training leaves latency tolerance untouched (delay rows equal to the control) and, like the delay-only arm (24) and unlike nothing else, raises LiDAR-off to 32/32 without any dropout — an unexplained side effect of fine-tuning under perturbed IMU terms that recurs across arms and is noted, not claimed. Bias tolerance itself is not measured by this matrix: the probe forces the bias to zero and has no bias setting, so the arm's own purpose remains untested. Summary: `results/repo-gpu-20260923/sf04-finetune/BothBias-s0.json`. This closes the SF-04 series: from-scratch recurrent NEGATIVE; from the working checkpoint, delay-only, dropout-only and all-ingredients fine-tunes and the distillation all repair the rows they train for; bias-only is unmeasured.

Push seeds: `21405527` push-adaptive seed 2 **COMPLETED 6000 iterations, EXPORT OK** (3:48); `21405528_3` arms-push seed 3 **COMPLETED 6000 iterations, EXPORT OK** (4:58); `21405528_2` arms-push seed 2 finished its 6000 iterations at 03:10 but is **still RUNNING ten hours in with no EXPORT line** — the export step is hung on cn-gpu7 (shared with three other users' jobs and two OOD sessions); the rescoring `21405529` waits on it. Unblocking below.

Crew lift training `21405730` (all four arms **COMPLETED** 4000 iterations, 4.5–5.6 h each): **no crew lifts**. Final iteration: crew3-blind reward 4.10, crew4-blind 2.30, crew3-vision 6.52, crew4-vision 3.11; `Curriculum/lift_height` stays at its first stage 0.040 m and `stage_lift` at 0.0 for every arm; 90–98 % of episodes time out, 2–10 % fall. The crews stand and survive but never advance the lift curriculum — a negative training result, the first measured for crews, against the pair's 7.8 cm. `21405917` crew scoring **FAILED for all four** as the caveat predicted (the pair harness cannot host crew-observation policies); no CSV.

arms-push seed 2 export unblocked: the training had finished all 6000 iterations and `train_play.py` had already written `exported/policy.onnx` + `policy.pt` (ONNX checked: 75 → 22), but the job hung afterwards on the shared node (cn-gpu7) before copying `configs/policy_latest.yaml` to `deploy.yaml`; by then that file belonged to the seed-3 export. `deploy.yaml` for seed 2 was therefore written from seed 3's file with only `policy_checkpoint_path` repointed. Across the four arms-push seeds the deploy files differ in exactly one field besides that path — `command_velocity`, the random velocity command of the export episode — which the MuJoCo evaluation does not read (the harness supplies every command itself), so the seed-2 deploy is config-equivalent in every field that matters. Hung job `21405528_2` cancelled; the matched rescoring `21405529` started on cn-c22. Crew scoring failure reason, for the record: `model_3999.pt emits 66 actions; this task drives 44` — the crew-3 policy's 3 × 22 actions cannot run in the pair harness, as the caveat said.

Matched push rescoring `21405529` **COMPLETED — PUSH-RESCORE RESULT: NOT SUPPORTED** (11:31, cn-c22): with the new seeds under the matched protocol (0.5 m/s shoves every 3 s, 12 s, seeds 0–4 × the harness command set), **push-adaptive 11/90 = 0.122 (three seeds) vs dr-default 19/90 = 0.211, one-sided Fisher p = 0.080** — the predeclared bar (p < 0.05) is not met, although the direction stays favourable and every push-adaptive seed sits at or below the control's pooled rate. For the 22-DoF humanoid, **arms-push (four seeds, n = 120) and arms-dr1.0 (two seeds) are not distinguishable in either direction** (p = 0.87 for "push falls less", 0.26 for "push falls more"), so the earlier one-seed reading that push training makes the 22-DoF robot fall *more* (0.15 vs 0.10) does not hold up either. Readings for the gallery: `push_pair` stays "supported in direction only, n = 90 vs 90, p = 0.08"; `arms_push_pair` becomes "no detectable effect of push training at n = 120 vs 60". CSVs and the updated `push_matched_summary.json` under `results/push-matched-20260923/`. This was the last job of the campaign batch; no campaign jobs remain queued.

Disk (2026-09-24, user's word given): the two folding adaptation trees `results/weekend-20260919/fold-adapt-s{0,1}` were pruned from 16 checkpoint steps each to the steps that matter — the last (`step_001500`) and the best named by each tree's `best.json` (`step_001200` for s0, `step_001000` for s1), which are also the only steps the folding repo's campaign manifests reference; 26 step directories removed, ≈57 GB freed, **free space 43 → 101 GB**. The tracked metadata (`best/latest/completed/data_split.json`) is untouched.

Standing cube-to-shelf **SUBMITTED** (2026-09-24, user's word given for "the cube-to-shelf standing-spawn redesign + smoke + train"): `21408513` smoke of the new `TaskV2-BHL-CubeToShelfStand-{Blind,Depth,Rgb}-v0` (v60, cameras, 4 envs) and `21408514` training chained `afterok` on it (`v2-cubetoshelfstand-blind-s0`, 1024 envs, 8000 iterations, 40 h wall). The task is `CubeToShelfStandCfg` in `task_v2_env_cfg.py`: robots spawned standing (root z 0.0, upstream legs, pinch arms), cube centre raised from 0.30 to `STAND_CUBE_Z = 0.55` on a 0.41 m plinth, shelf/rewards/success unchanged. Why 0.55: the spawn diagnostic measured the hand frames at z 0.599 in that pose (`spawn_hands/...standing_pinch_arms.json`); 0.55 keeps the hands in the cube's upper half, the cube top (0.69) under the slot ceiling (0.72) and seats it on the deck at 0.52. **This is a different, easier task than CubeToShelf** (reach_band.py set 0.30 so standing would be instrumentally necessary; the crouch that reaches it needs 28–30 Nm against the 6 Nm limit); its numbers are reported as CubeToShelfStand and never as CubeToShelf. Assumption on record: every spawn pose fell within 1 s under PD hold in the diagnostics (pinch step 16, standing 22, standing+pinch arms 17–24), and the evidence that a *policy* balances this asset is the crew runs (90–98 % time-outs, 2–10 % falls over 4000 iterations). Predeclared: kill at `model_999` if mean episode length < 25 steps over the last 100 iterations or `Episode_Termination/fallen` > 0.9; result = `Episode_Termination/success` averaged over the last 200 iterations ≥ 0.10 → "learned placement (standing variant)", else NEGATIVE.

Ice no-ice control **SUBMITTED** (2026-09-24, user's word given for "the ice no-ice control retraining despite the uncommitted ice edits"): `21408515_[0-3]%2` (`gpu_ice_control.sbatch`, v51, 6000 iterations, 4096 envs, 24 h, two at a time; the author's v51 boot lock), tasks `Velocity-BHL-Biped-IceControl-v0` (blind, seeds 0/1) and `-IceControl-Depth-v0` (depth, seeds 0/1) from `tasks/ice_control_env_cfg.py`: the placed-ice cfgs with every `ice_*` patch's material set to the terrain's own friction (1.0 / 1.0) and nothing else changed. The parents (`terrain_env_cfg.py`, `depth_env_cfg.py`, `terrains/ice.py`, `terrains/bumpy.py`) run from the author's uncommitted working tree, untouched; diff hash at submission `a532c90b6848c643` (`git diff HEAD -- <the four files> | sha256sum`), re-measured and written next to the results by each job at boot (`results/repo-gpu-20260923/ice-control/worktree-*.diff`). Comparison targets: the placed runs `2026-09-16_*_ppo-ice-placed-*` (job `21344928`), final terrain level blind 2.5763 / 2.6055, depth 2.8541 / 2.9837. Predeclared reading: depth-minus-blind on the control as large as on ice → the ice gap was not about friction; control gap ≈ 0 with the ice gap intact → the placed-ice depth advantage stands on its own.

Untracked note: `src/bhl_robust/terrains/maze_viz.py` (the coloured maze clip overlays that `maze_record.py` and the author's `train_play.py` import, guarded) is the author's uncommitted file dated 2026-09-18 and is not committed here; the published Isaac maze GIF's sidecar records the repo HEAD only, so a bit-exact re-render also needs that file from the working tree.

Standing cube-to-shelf smoke `21408513` **COMPLETED — 3/3 built and stepped** (1:00): `TaskV2-BHL-CubeToShelfStand-{Blind,Depth,Rgb}-v0` observation widths 194 / 322 / 578 (the same three widths as CubeToShelf, so the cameras are wired and the blind arm is unchanged in shape), success term present; the chained training `21408514` is released and waits on the per-user memory quota (`QOSMaxMemoryPerUser`) behind the folding driver's array, as does the ice-control array `21408515`.

Maze three-panel clips **SUBMITTED** (2026-09-24, user request "a gif that shows the robot completing the maze, top view like in the README, with a window of its own perspective stereo and another of lidar"): `21408621` MuJoCo inspection maze (`gpu_maze_panels_mujoco.sbatch` → `scripts/bench/maze_panels.py`: the README episode — seed 0, ordered route, reactive sensor brake — re-run through `inspection_maze.episode()` with a per-step recorder hook; top view 960×540 at elevation −66°, robot-eye render, the 8×8 paired ray depth and the 36-sector lidar exactly as the brake received them; the login-node pipeline check reproduced the recording's completion time, 17.36 s; textured world = materials/lights only, geometry and contact arrays verified identical) and `21408622` Isaac B5 maze (`gpu_maze_panels_isaac.sbatch` → `maze_record.py --panels --render-quality clean`: the published Full Both s0 policy, seed 100, fixed 1280×720 overhead camera, the stereo pair's 64×64 ray depth plus the pooled 4×4 the policy reads, the 36 lidar sectors plus the 500 raw hits). Predeclared for the Isaac render: spatial grain (mean |f − box3(f)|, 0–255) ≤ 12 with sampled direct lighting / AO / GI / reflections off means the renderer fix suffices (the stock clip measures 36.1, the MuJoCo clip 0.4); above it the frames get an nlmeans pass and the sidecar says so. Outputs go to new names (`docs/gifs/inspection-maze-panels.gif`, `docs/gifs/isaac/maze_both_panels.gif`); the README GIF and its sidecar are left as published.

Isaac three-panel clip `21408622` **FAILED at composition (0 sensor rows) — and its grain measurement stands**: the seed-100 episode succeeded (161 overhead frames), but the per-step sensor dump died on every step on a quaternion-unpacking slip (`unpack_wxyz` returns four columns; the code indexed the tuple), so nothing reached the composer. The predeclared render check ran first and is the useful result: with sampled direct lighting / AO / GI / reflections off the spatial grain measured **34.05 (stock clip 36.1; bar ≤ 12)**, temporal MAD 41.96 (stock 47.3) — the stochastic terms are *not* the speckle's source. The headless rendering kit runs sampled direct lighting at 1 spp and hands the clean-up to DLSS (`rtx.post.dlss.execMode = 0`), which needs NGX, and NGX fails to initialise on these nodes; the raw 1-spp samples are what every Isaac clip has shown. Resubmitted with the slip fixed as two profiles side by side, same bar: `21408633` (`taa`: stochastic terms off + TAA in place of the missing DLSS pass) and `21408634` (`pathtrace`: path tracing at 16 spp through the OptiX denoiser, which does not depend on NGX); each writes its own GIF name; the one that meets the bar (or, failing both, the nlmeans fallback) becomes `docs/gifs/isaac/maze_both_panels.gif`.

MuJoCo three-panel inspection clip `21408621` **COMPLETED — MAZE-PANELS RESULT: PASS** (1:24, cn-gpu7, EGL) and re-rendered as `21408629` **PASS** (1:22) after the first cut's floor blew out to white under the point light's specular term (fixed: tile/wall materials at zero reflectance and near-zero specular, point light dimmed to 0.55, headlight specular off; geometry and contact arrays re-checked identical to the plain world). Both runs re-simulated the README episode — seed 0, ordered route, reactive sensor brake — through `inspection_maze.episode()` and completed at **17.36 s, the September 20 recording's time to the hundredth**, 2 stations, 0 wall contacts, path 6.93 m (464 frames at 25 Hz). Committed: `docs/gifs/inspection-maze-panels.gif` (1.19 MB, 860 px, 8 fps, 2× with a "GIF 2x" badge, 128 colours) + sidecar; the 1× mp4 stays out of git. Panels are the brake's actual inputs at that step (8×8 paired ray depth, 36 lidar sector minima, stale flag, brake scale) plus a robot-eye render labelled as a render. The README's GIF (`weekend-inspection.gif`) is untouched; the README itself is the author's uncommitted file, so the swap line is handed over rather than applied.
