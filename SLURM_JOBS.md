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

### Tier 1 first block · `running`
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
| 1 | `21091041` | PPO control **COMPLETED** (5:13); the three MARL rows failed on a missing `num_agents` property, since added |

### Gripper: restore the two hand DoF the asset welds shut · `running`
The hardware has two grippers (`docs/GRIPPER.md`); the shipped asset welds both
hands, so no policy here has ever closed one. `scripts/add_gripper.py` writes a
24-DoF copy into the workspace, leaving `external/` pristine. Checked before
queuing: 24 actuated joints, fingertip sweeps 6.5 cm across the palm, hands
mirrored.

Once the USD lands, this re-runs §5, the nine v2 cells and Tier 1 — the terrain
and locomotion rungs are unaffected, having never used arm contact.

| # | id | outcome |
|---|---|---|
| 1 | `21093986` | running — URDF to USD conversion; `get_gripper_cfg()` raises until it exists |

### B3 — ice / patchy friction · `running`
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
| 2 | `21105232` | running — 6 rungs: blind/depth/visible x 2 seeds, 6,000 iters |
| 1 | `21100447` | **smoke 3/3**, episode lengths 17.7–19.8 |

### B3 — ice / patchy friction · superseded
`terrains/ice.py` and `scripts/bench/ice_gate.py` written. G-B3 passes at the
flush inset and correctly fails at a 5 mm one, so the gate can fail. Still to
do: wire a task config and register the ids.

| # | id | outcome |
|---|---|---|
| — | — | not queued |

### Gripper v2 variants · `running`
The three v2 tasks on the 24-DoF asset, as separate ids rather than a flag, so
the welded-hand arms stay runnable as their control. Actions and the
joint-indexed observations move 22 -> 24 together; driving 22 of 24 joints would
leave the grippers inert and the variant indistinguishable from its control.

| # | id | outcome |
|---|---|---|
| 1 | `21105363` | running — smoke through the real training path, 3 iters, blind/depth/rgb |

### Cloth sorting — G-C1 throughput · `running`
Decides RL against scripted demo. The folding repo's stack is CPU-only at one
env per process; 6.0's Newton runs cloth on GPU, so the question is how many
environments at once. At 1,024 this is an RL task, at 16 it is a demonstration.

| # | id | outcome |
|---|---|---|
| 3 | `21105721` | running — self-contained probe, local cloth USD, no external assets |
| 2 | `21105405` | FAILED — Isaac Lab's cloth task pulls a table and sky from the Omniverse content server and one path 404s. The sorting task brings its own garments, so that dependency was never needed. |
| 1 | `21105364` | FAILED — `'str' object is not callable`: Isaac Lab's tasks use a module-path string for `env_cfg_entry_point`, this repo's use a class |

### Base-height probe — is the floor-lift a training hack? · `running`
The MuJoCo replay drops both cube arms ~41 cm in 0.2 s, before contact. PhysX
`base_contact` sits at -0.003, so the torso is not down in training — but a
robot folded onto its shins never puts its torso down either, so that cannot
separate a squat from a collapse. `base_height` is now a weight-0.0 reward term
so the curve is recorded without entering the objective.

| # | id | outcome |
|---|---|---|
| 6 | `21090556` | running — wall time raised to 12 h |
| 5 | `21082873` | TIMEOUT at 4 h — 6,000 iterations at 1024 envs does not fit; not a code fault |
| 4 | `21078881` | **done** — base height climbs to -0.105 by iteration ~350 then descends monotonically to -0.167 by 1,500, and is still falling |
| 3 | `21077648` | FAILED — the physx shim fired on v51 and broke the import |
| 2 | `21076968` | FAILED — `base_height expects optional parameters ['robot_a','robot_b'] but received []` |
| 1 | `21076488` | cancelled — logged 0.0000 for 1,300 iterations. The reward manager logs `weight x value`, so a weight-0.0 term reports zero by construction. |

### Redesigned tasks (v2) — nine cells · `running`
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
| 13 | `21105231` | running — the nine cells, after the tilt fix |
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

### Occlusion replicates · `running`
Does the one cube arm that ever lifted reproduce? So far: no.

| # | id | outcome |
|---|---|---|
| 3 | `21066826_6` | running — blind seed 2, flat at 0.0400 |
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
| `21105364`, `21105405`, `21105721` | G-C1 cloth throughput |
| `21105399` | gripper v2 training, 9 cells |
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

Jobs named `orchard*`, `lh-*` and `interactive` are not from this workstream.

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
