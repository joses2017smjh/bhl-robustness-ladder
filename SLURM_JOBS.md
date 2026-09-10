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

## Open

### Manipulation re-runs on the corrected spawn · `running` — 2026-09-07
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
| 2 | `21199344`, `21199345` | 1 COMPLETED, 5 running, 3 NODE_FAIL at ~19 h (infrastructure), 1 CUDA illegal access, 1 abort, 6 plank cells cancelled |
| 1 | `21186402`, `21186403` | cancelled — trained on the lying-down spawn |

### B3 ice clip — DONE · `done` — 2026-09-10
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

### B5 maze — stereo pooling sweep · `running` — 2026-09-10
Does coarser stereo recover, or do cameras simply not help? At `pool=4` stereo
is 512 of 557 observations (92%) against lidar's 36 of 81 (44%). `StereoP16`
pools to 4×4 an eye — 32 of 77, **42%**, matched to lidar — so it holds the
information fraction fixed and changes only the sensor.

| # | id | outcome |
|---|---|---|
| 1 | `21233916_[4-6]` | running — StereoP8, StereoP16, BothP16 |

### Cloth sorting — jobs cancelled · `open`
`21228030`–`21228033` (c1 smoke, isaac eval, deform smoke, sort bench) were all
cancelled before running; `21228029` failed its own guard with "produced no
summary line". So the mesh-resolution sweep that would reopen G-C1 has not
produced a measurement yet — `results/cloth_sort_bench.md` still marks every
Isaac row "not yet measured", which is accurate.

### B5 — maze with lidar and stereo · `running` — queued 2026-09-08
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

### Cloth sorting — reopened against a coarser mesh · `running` — 2026-09-09
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

### B3 ice clip — export solved, render blocked on depth obs · `open`
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

### B4 — limb agents (DirectMARL + skrl) · `todo`
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

### Tier 1 / 2 / 3 MARL rows · `blocked` on B4
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

### B3 — ice / patchy friction · `done`
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

### Upright-spawn 400-iter probes · `running` — queued 2026-09-06
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
| 3 | `21192899` | running — gripper CubeToShelf-Blind, 400 iter, new yaw (`yaw-cubetoshelfgrip-blind-s0`) |
| 2 | `21192898` | running — `_1` requeue after TMPDIR-on-scratch |
| 1 | `21192861` | `_0` running on cn-gpu7 (past iter 18) · `_1` FAILED 43 s, `/tmp/Assets` PermissionError |

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
