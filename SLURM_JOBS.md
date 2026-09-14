# Slurm job ledger

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

## Project status — 2026-09-14

**Done since the last status**

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

- `docs/gifs/isaac/maze_stereo_fixed.gif` — the corrected 16×16 stereo policy walking,
  with its left eye as B5 had it before the quaternion fix (20° up, a strip of
  ground) beside the corrected eye (`21329137`).
- `docs/gifs/isaac/terrain_sensors.gif` — the first Isaac clip of trained
  policies with the robot in shot.
- `docs/gifs/isaac/cloth_sort_fixed_base.gif` — the scripted sweep pushing the shirt
  proxy into its basket, root pinned (`21317391`).

**Left, in order**

1. **Cloth, in Isaac on the redesigned layout** — *layout, garments, reach model,
   schedule action and planted spawn are done; the kinematic ladder sorts at 1.00*.
   *Fixed-base scripted C0 sorts 32/32 in Isaac for all three garment classes.*
   *The free base now stands in Isaac* with a knee-1.0 stance and a leg controller
   (2/2 with the arm still, 3/4 through six sweeps). The layout is raised to that
   stance (`a4f438b`), and on it **the free-standing robot sorts jacket 8/8 and sock
   7/8, shirt 1/8.** Left: the shirt, via stiffer ankle gains and the arm's
   centre-of-mass feedforward, then online arm correction from the base pose; C2 on
   the fixed base, void so far because Newton goes NaN when the hand pushes the
   cloth (one-way coupling test `21329265`); then the sequential five-garment scene
   in Isaac, C1 training, and C3–C5.
2. **Make the maze a maze**: walls into the terrain mesh at the terrain origins,
   sensors pointed at them, and the navigation objective `docs/MAZE_RIG.md`
   designs. *Before that, the terrain rung's stereo arms finish re-running with
   the cameras pointed down (`21317023`–`21317025`; 16×16 done, 0.774 at n=3).*
3. **Isaac spawn for the coop/TaskV2 tasks**: robots still spawn under the
   floor (`robot_a` bodies at z −0.806…−0.027 in `21299608`). *Likely cause found,
   not yet probed:* on v60 the spawn tuple `(0.707, −0.707, 0, 0)` is read
   `(x, y, z, w)`, an upside-down robot facing the cube (R₂₂ = −1); the legacy
   `(0.707, 0, 0, −0.707)` is a −90° roll. The fix is `native_quat` on the intended
   `(w, x, y, z)` yaw, plus a spawn probe.
4. **MARL**: first block done at n=3 — limb2's lead did not replicate. Tier 1's
   MARL rows are running for stairs / slippery / rough (`21328607`, `21328608`;
   gate `21328605` passed 9/9); ice waits on item 5. Tier 3 on stairs is queued
   behind its gate (`21328742` → `21328743`, `21328744`).
5. **Put B3's ice where the robots are**: patches at the terrain origins, or moved
   to `env_origins` at reset, on tiles flat enough that a flush patch stays flush;
   re-probe with `scripts/bench/ice_placement_probe.py`; then the B3 PPO arms and
   Tier 1's ice rows. Until then B3 and finding 11 stay retracted.

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

### B5 maze — stereo re-run with the cameras pointing down · `running` — queued 2026-09-13
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
The 4×4 and 4×4 + lidar arms are mid-run and not results yet: 4×4 seed 1 reads
1.206 at 5,137 iterations, 4×4 + lidar seed 2 0.839 at 5,078, 4×4 seed 0 0.428
at 3,419.

| # | id | outcome |
|---|---|---|
| 4 | `21329137` | **clip COMPLETED** (4:09, cn-gpu6, dgxh-1 excluded for Vulkan) — `mazefix-stereo-s0` through the camera-sensor recorder, 200 frames, with the corrected left eye and a raw-tuple eye dumped per frame (`train_play.py --clip-sensors stereo_l --clip-raw-stereo`); `docs/gifs/isaac/maze_stereo_fixed.gif`. The long pending jobs were niced for ten minutes so it could take the next slot, then restored |
| 3 | `21317023`, `21317024`, `21317025` | seeds 0 / 1 / 2, `--array=2,5,6%1` (16×16, 4×4, 4×4 + lidar), 16 h limit, 2,048 envs as before — **16×16 COMPLETED in all three** (8:42, 5:41, 5:42; table above); 4×4 seeds 0–1 and 4×4 + lidar seed 2 running, three tasks pending |
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
| 38 | `21329392` | queued, afterany `21329265` — clip, **free base, jacket**, stance layout (`results/clips/frames/cloth_c0_stance_jacket`), excludes dgxh-1 |
| 37 | `21329265` | queued, afterany `21329264` — C2F with `BHL_NEWTON_COUPLING=one_way`, nothing else changed against rows 24/26 (the stance change leaves a pinned arm's geometry as it was), 4 eps, trace (`isaac_c2f_v3_oneway.json`) |
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

### Tier 1 MARL grid — biped, left leg | right leg · `running` — G-B4t passed 9/9, 2026-09-14
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
| 3 | `21328605` → `21328607`, `21328608` | **G-B4t PASS, 9 of 9** (6:50) — terrain level logged, rsl-rl settings confirmed (5 epochs × 4 mini-batches, entropy 0.008, KL-adaptive LR, [256, 128, 128]), observation 301 on all three terrains. Seeds 0 and 1 `--array=3-11`, chained `afterok`; both seeds' stairs MAPPO and IPPO rows running. Throttle cut from `%2` to `%1` per seed at the per-user GPU cap, so the cloth chain gets slots |
| 2 | `21328445`, `21328446` | **cancelled** — `DependencyNeverSatisfied` after the gate failed; their ice rows had been held first, pending `21328532` |
| 1 | `21328444` | **FAIL, correctly** (9:50). (i)–(iii) pass on all 12 rows: 2 agents of 6, left and right hip first, observation 301 = state with depth, trainer matches. (iv) fails on 9: no terrain level in the event file, because Isaac Lab `.item()`s curriculum scalars and skrl logs only tensors. The three rough rows loaded the `loggable` fix mid-gate and passed (iv) |

### Tier 3 — 22 DoF on stairs, depth, arm deviation off · `queued` — G-T3 passed 3/3, 2026-09-14
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
| 1 | `21328742` → `21328743`, `21328744` | **G-T3 PASS, 3 of 3** (5:37): no shoulder or elbow deviation in any reward table, `joint_deviation_hip` present, observation 331 with depth (PPO's actor included), limb4's first joints the two shoulders and two hips, terrain level logged on the rsl-rl settings. Seed 0 `--array=0-2%1` pending at the per-user GPU cap, seed 1 after seed 0 |

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
