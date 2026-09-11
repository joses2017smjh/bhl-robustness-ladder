# Sorting garments into baskets

> **Status, 2026-09-11: the ladder runs end to end in Isaac, the low-resolution
> cloth is fast — and the scene it runs was never reachable.** The hand cannot
> touch the table, the garment sat 0.74 m from a 0.305 m reach, and in Isaac the
> robot faces away from it. Every earlier success rate in this file came from a
> model with no reach; under one they are all 0.00. Layout redesign next.
> End-to-end deformable RL stays rejected. The task is not abandoned.

This is an evidence-driven redesign. G-C1 answered that Newton cloth cannot
carry the training workload. The research question moved with that measurement:
can a sweep strategy learned or planned on cheap object representations
transfer onto low-resolution cloth, and then onto a five-garment scene?


## The layout was never reachable (2026-09-11)

Found while asking why no Isaac cell ever moved the garment, and it outranks
every other result in this file.

**Reach, by forward kinematics** (MuJoCo, the same crew builder that measured
the vertical reach band). Right arm over its full joint range, legs held in the
pinch squat the controller commands:

| hand-link height | furthest forward reach | lateral span there |
|---|---:|---|
| 0.30–0.35 m | +0.064 m | right side only |
| 0.35–0.40 m | +0.170 m | right side only |
| 0.45–0.50 m | +0.258 m | right side only |
| 0.60–0.65 m | +0.289 m | right side only |

The hand's lowest point is **0.339 m** — it cannot reach the 0.30 m table top
at all — and at table height its horizontal reach tops out at **0.305 m**.

| feature of the old layout | distance from the robot root |
|---|---:|
| basket centre | 0.320 m |
| table near edge | 0.390 m |
| **garment spawn** | **0.740 m** |

**Facing, in Isaac** (`21247910`, `scripts/cloth/hand_probe.py`). The hands sit
at (−0.208, +0.192) from the root; FK facing +x puts them at (+0.209, −0.177).
Both axes flip — a half-turn, not a mirror — so under `(0, 0, 1, 0)` the robot
faces **−x**, away from the table, with its geometry matching MuJoCo to 2–4 mm.
The right hand was **0.97 m** from the garment. `ROBOT_YAW` is now π.

**What that does to the numbers below.** The kinematic env had no reach model:
a sweep was two planar points the hand was assumed to follow, so its 1.00s meant
"a hand that could go anywhere would sort these". It also let a plan it had
flagged invalid push the garment. `bhl_robust.cloth.reach` now carries the
measured workspace — a 2 cm IK table from FK plus damped least squares, 1,380
reachable voxels, `assets/cloth/right_arm_ik_pinch.npz` — and the controller
refuses a plan the hand cannot follow:

| Experiment | Physics | Policy | Episodes | Success | Sweeps refused |
|---|---|---|---:|---:|---:|
| C0 | kinematic + reach | scripted | 64 | **0.00** | 100% |
| C1 | kinematic + reach, DR | learned residual | 64 | **0.00** | 100% |
| C4 | kinematic + reach, DR | linear BC | 64 | **0.00** | 100% |
| C5 | kinematic + reach, Mode B | scripted | 32 | **0.00** | 100% |

`results/cloth/reach/`. The no-reach files are left where they were.

**What is reachable** is a patch on the robot's right: robot-frame x
−0.12…+0.26 m, y −0.44…−0.08 m, hand-link heights 0.34–0.54 m, never crossing
the midline. The hand mesh hangs 2–13 cm below its link origin depending on
the pose, so "contact height" has to be defined on the hand's underside, not
its origin. The robot's own right hip and foot reach y = −0.17. A layout inside
that patch — smaller garments, a side table, baskets off its reachable edges —
is the next step, and it is what decision rule 1 asks for.

The robot falling in ~13 steps (below) is now the smaller problem, and it may
share a cause with this one: the cloth robot spawns at the *standing* planted
height, −0.027 m, while the squat plants at **−0.137 m**, so it drops 11 cm at
every reset. The zero-action probe's root bounced between −0.05 and −0.20.
Not yet tested.


## Original gate result

Deformable simulation **failed the throughput gate**.

G-C1 (`21185969`, 2026-09-05) ran Isaac Lab's Franka cloth scene — **one**
961-vertex deformable, a **7-DoF** arm, assets localised — and measured:

| envs | env-steps/s |
|---:|---:|
| 8 | **182** |
| 32 | 177 |
| 128 | **71** |
| 512 | overflow (signed 32-bit array dim, 7.3e9) |

Parallel throughput **falls** as environment count increases. There is no
scale to buy.

### Why that kills end-to-end RL

A normal training arm here is:

- 8,000 iterations
- 2,048 environments
- ~48 environment steps per iteration
- **786 million** environment steps

At 182 env-steps/s that is about **50 days** of wall clock against **~6
minutes** at this project's rigid-body rate. At the 128-env point the same
job is ~128 days (`scripts/cloth/estimate_cost.py`).

And that measurement is generous. This design needs **five** garments and a
**22-DoF** humanoid: more collision geometry, more robot-cloth contacts,
cloth-cloth and cloth-environment interactions. The scene that already fails
the gate is lighter than the one the original proposal asked for.

### Consequence

**Full five-garment, 22-DoF end-to-end deformable RL is rejected as the
primary training method.**

The previous write-up concluded "scripted demonstrations, not RL" and stopped.
That was the right call *for cloth physics as the training engine*. It was
the wrong call for the experiment. The experiment is still worth running if
learning happens where it is cheap.


## Stage 2: one low-resolution Newton cloth

`ClothSort-BHL-Deformable-Oracle-v0` replaces the rigid `garment_0` with an
Isaac Lab `MeshRectangleCfg` surface cloth. Default resolution is **8×8 (64
vertices)**, not the 961-vertex Franka mesh from G-C1. `sim.dt` is `1/60` to
match the Franka cloth scene; that change is documented. Construction raises
if Newton is missing — it does not silently train a cube.

`ClothSort-BHL-ActiveCloth-Oracle-v0` is Mode B: **one** live cloth (`sock_a`
as `garment_0`) plus four rigid proxies. That is not five simultaneous cloths.

Success on cloth is the vertex-fraction predicate (`>= 0.60` in the correct
basket AABB), not the rigid CoM test.


## New approach

```
rigid proxy training
    → single-cloth validation
        → limited deformable adaptation if necessary
            → five-garment evaluation
```

| Stage | Physics | Parallelism | Role |
|---|---|---|---|
| 1 | thin rigid rectangles | 1024–2048 *after a measured rigid throughput* | learn / plan where to sweep |
| 2 | 1 low-res deformable (8×8 … 16×16, not 961) | 8–32 | does the primitive transfer? |
| 3 | 5 garments, 3 baskets | 1–8, eval only | demonstration. Mode B if five live cloths do not fit |

Cheap simulation for learning. Expensive simulation for validation.


## Research questions

Primary:

> Can manipulation behavior learned or planned on inexpensive object
> representations transfer to low-resolution deformable garments and then to
> a realistic multi-garment cloth-sorting scene?

Secondary:

- How large is the rigid-to-deformable gap? (`transfer_gap = rigid_success − deformable_success`)
- Which failures appear only when the garment bends?
- How much deformable fine-tuning is required — if any?
- Does a learned sweep planner beat a fixed scripted sweep?
- How sensitive is success to mass, friction, size, spawn, basket pose?


## Morphology, unchanged

No fingers. 4 Nm arms. Hands that do not adduct past 0.355 m. The robot is
the stock **22-DoF** Berkeley Humanoid Lite with its normal left and right
arms. "Experimental arm" means an experimental *condition*. It must never
mean a third physical limb.

A humanoid with no fingers cannot pick up a sock. The task is **sweeping**,
not pick-and-place. The table top sits at `GRASP_Z = 0.30 m` (the measured
squat band). Baskets sit under the robot-facing edge. The robot spawns
already addressing the table — walking there is a different problem and is
not mixed into this reward.


## Architecture

```
Observation / Perception  (oracle state first; sensors later)
        → Garment localization
        → Target selection
        → Sweep planner     (scripted geometry  or  learned 5-D action)
        → Sweep controller  (pinch-pose + bounded arm offsets)
        → Humanoid          (22-DoF, unchanged articulation)
        → Object physics    (rigid proxy  or  low-res cloth)
```

The policy outputs

`[dx_start, dy_start, sweep_angle, sweep_distance, sweep_speed]`

in `[-1, 1]`. It does **not** rediscover 22-DoF motor coordination. The
controller reuses the measured pinch/squat hold from the coop tasks and
refuses trajectories that leave the joint walls.

Oracle observations are privileged and labeled as such. A later sensor rung
can replace `garment_xy` without rewriting the planner or the controller.
Blind / LiDAR / stereo variants are compatible and are not a blocker for
Stage 1.


## Garment set and baskets

Five garments, three baskets, one class per basket — at least one basket
takes two items, so a policy cannot succeed by memorising a 1-1 layout.

| garment | class | basket | rigid proxy (m) | mass (kg) |
|---|---|---|---|---:|
| sock_a, sock_b | sock | socks | 0.18 × 0.10 × 0.02 | 0.04 |
| shirt_a, shirt_b | shirt | shirts | 0.28 × 0.22 × 0.02 | 0.12 |
| jacket | jacket | jackets | 0.34 × 0.26 × 0.025 | 0.22 |

Success is geometric:

- rigid: centre of mass inside the correct basket AABB
- deformable: `fraction_of_vertices_in_correct_basket >= threshold` (default 0.60)

Wrong-basket is a separate bit from "not yet sorted".


## Experiment ladder

| id | physics | policy | purpose |
|---|---|---|---|
| **C0** | rigid proxy | scripted | geometry + controller |
| **C1** | rigid proxy | learned 5-D | high-throughput strategy |
| **C2** | 1 low-res cloth | scripted | does the primitive transfer? |
| **C3** | 1 low-res cloth | C1 zero-shot | measure `transfer_gap` |
| **C4** | 1 low-res cloth | BC / residual / small RL | only if C3 is poor; cost-gated |
| **C5** | 5-garment eval | best of the above | Mode A: five live cloths. Mode B: one active, others frozen/rigid |

Decision rules:

- C0 fails → fix controller/layout, do not train
- C1 fails → fix reward/action, do not introduce cloth
- C2 fails → physics/controller, not RL
- C2 works and C3 fails → transfer gap, not "cloth is unsolvable"
- C3 is already good → do not fine-tune for the sake of fine-tuning
- five live cloths do not fit → Mode B, and say so

C4 is refused unless `scripts/cloth/estimate_cost.py` accepts the job.


## What has been measured

**Superseded 2026-09-11 — this table ran with no reach model.** Under one,
every cell is 0.00; see *The layout was never reachable*. Kept because the
numbers were published.

Kinematic rigid proxy, login node, 2026-09-10. **Not Isaac rigid-body
physics. Not cloth.** Fall rate is 0 because this integrator does not
simulate balance; that number is not a locomotion result.

Success and sweep counts here are deterministic under their seeds and
reproduce exactly on re-run. The **throughput column does not**: it was
re-measured at 600 and 572 env-steps/s against the 350 and 357 recorded below,
on a shared login node under different load. Treat those two cells as an order
of magnitude, not a measurement — a real env-steps/s for this task has to come
off a compute node.

| Experiment | Physics | Policy | NUM_ENVS | Throughput | Success |
|---|---|---|---:|---:|---:|
| C0 | kinematic rigid | scripted | 1 | 350 env-steps/s | **1.00** (64 eps, 1.0 sweep) |
| C1 | kinematic rigid + DR | learned residual | 1 | 554 (train) | **1.00** eval (64 eps, 3.45 sweeps) |
| C1 control | kinematic rigid + DR | scripted | 1 | — | **1.00** (64 eps, 1.05 sweeps) |
| C2 | deformable | scripted | — | not yet measured | N/A |
| C3 | deformable | C1 zero-shot | — | not yet measured | N/A (`transfer_gap` refused) |
| C4 | kinematic rigid + DR | linear BC | 1 | — | **1.00** (64 eps, 1.14 sweeps). **Not cloth.** |
| C4 deformable | 1 low-res cloth | adapted | — | not yet measured | N/A |
| C5 | kinematic rigid, Mode B | scripted | 1 | 357 env-steps/s | **1.00** (32 eps, 5.0 sweeps) |

C1 matches C0's success and loses on efficiency. On this model the scripted
planner is the better high-level policy. Kinematic C4 BC recovers that
efficiency (1.14 sweeps vs C1's 3.45) because it clones the scripted
action. That is **not** a deformable fine-tune and is not a reason to skip
C3.

### Isaac, 2026-09-10 — the scene runs and the robot falls over

The rigid ids now construct, reset and step (`21233957`), and a 3-iteration
pass through the real `scripts/train.py` clears the training gate
(`21233958`). Both report the stock articulation: **22 joints, 2 arms**,
5-D action.

| cell | job | what it measured |
|---|---|---|
| rigid construct/step | `21233957` | 22 joints, 2 arms, action_dim 5, 4 envs |
| C1 training smoke | `21233958` | **1,001 steps/s at 64 envs**, 3 iterations, mean episode length **12.83** |
| C0 scripted | `21233959` | 12.25 steps/episode |
| deformable build | `21233960` | Newton cloth initialises: 8 instances, **81 vertices at `resolution=8`** |

**The result is negative and it is not about cloth.** In both the trained and
the scripted cell, `Episode_Termination/fallen = 1.0000` and
`Episode_Reward/progress = 0.0000`: the humanoid topples about 0.51 s into
every episode and the garment never moves toward a basket. The layout is not
the cause — the robot at x = −0.22 clears the table (x ≥ 0.17) and the baskets
(x ≤ 0.22), checked geometrically. The open hypothesis is that it cannot hold
the pinch/squat pose: legs are position-controlled at stiffness 20 against a
6 Nm effort limit at hips −0.85 / knees 1.45, and FINDINGS already records the
coop policy's torso sinking to −0.23 m rather than holding a squat.
`slurm/95g_cloth_spawn_probe.sbatch` measured that directly, and **it is not
the squat**. Three leg poses, zero action, 60 steps (`21234166`):

| pose | final tilt | root z | fell (>0.78 rad) |
|---|---:|---:|---|
| configured squat | 0.267 rad | −0.199 | **no** |
| half squat | 0.045 rad | −0.108 | **no** |
| default, no squat | 0.142 rad | −0.139 | **no** |

None of the three falls, and the root sinks **12–20 cm in every one of them**,
including the pose with no crouch at all. So crouch depth is not the variable.
What the probe shows is a robot with **no balance control**: the leg position
targets sag under gravity and it oscillates to within half the fall limit
before the task has done anything. Policy-driven arm motion on top of that is
what carries it past 0.78 rad inside ~13 steps.

That is a design decision, not a bug to patch quietly. The task needs one of:
a balance controller, a fixed or supported base for the manipulation stage, or
a fall criterion chosen for a stationary manipulator rather than the
locomotion limit of 0.78 rad. Whichever is chosen changes what a cloth-sort
success rate *means*, so it is left open rather than picked here.

**Decision rule 1 applies: C0 fails, so fix the controller or the layout, and
do not train.** No C1 arm is queued and no success number is claimed.

**`21233959`'s success and fall rates were not measurements.** `eval_isaac.py`
reported both while assigning neither — `em.fell` was never written, and the
success lookup indexed a `(num_envs, n_terms)` **tensor** with a string, which
raises into a bare `except: pass`. Both were pinned to 0.0 and could not have
moved whatever the robot did. They read through
`termination_manager.get_term()` now and raise on a missing term.

Re-run as `21234259`, the same cell reports **`fall_rate` 1.00** and
**`success_rate` 0.00** over 4 episodes — matching the trainer's
`Episode_Termination/fallen = 1.0000` from an entirely separate code path.
That agreement is what makes the 0.00 success a measurement rather than a
default.

| Experiment | Physics | Policy | NUM_ENVS | Throughput | Success |
|---|---|---|---:|---:|---:|
| C0 Isaac | Isaac rigid | scripted | 4 | — | **0.00** success, **1.00 fall rate**, 12.75 steps/episode (`21234259`) |
| C1 Isaac | Isaac rigid | learned 5-D | 64 | **1,001 steps/s** (3-iter smoke) | **N/A** — `fallen = 1.0000`, `progress = 0.0000` |
| C2/C3/C4 Isaac | 1 low-res cloth (81 verts) | — | 8 | **467 env-steps/s** | N/A — blocked behind the same fall |

### Isaac throughput, measured (`21234167`)

Warmed, three throwaway steps then 40 timed, one GPU node.
`results/cloth_sort_bench.csv`, summarised in `results/cloth_sort_bench.md`.

| robot | deformables | resolution | vertices | envs | env-steps/s |
|---|---:|---|---:|---:|---:|
| BHL 22-DoF | 0 | — | 0 | 8 | 128.6 |
| BHL 22-DoF | 0 | — | 0 | 32 | 623.9 |
| BHL 22-DoF | 0 | — | 0 | 64 | **1,117.6** |
| BHL 22-DoF | 1 | 8×8 | **81** | 8 | **467.2** |
| BHL 22-DoF | 1 | 10×10 | **121** | 8 | 438.3 |

**This is the number the redesign was betting on, and it landed.** Against
G-C1 — Franka, 7 DoF, one **961**-vertex cloth — at the same 8 environments:

| scene | robot | vertices | envs | env-steps/s |
|---|---|---:|---:|---:|
| G-C1 (`21185969`) | Franka, 7 DoF | 961 | 8 | 182 |
| this task (`21234167`) | **BHL, 22 DoF** | **81** | 8 | **467** |

A **2.6×** speed-up while carrying three times the DoF, bought entirely by
dropping mesh resolution. Going 8×8 → 10×10 costs only 6% (467 → 438), so
there is headroom to raise fidelity in Stage 2 rather than lower it.

Two cautions, because neither of these is a licence to scale up:

- **Deformable parallelism is still unmeasured.** Every cloth row above is at
  **8 envs**. G-C1's finding was that cloth throughput *falls* with
  environment count (182 → 71 from 8 → 128). Nothing here contradicts that; it
  was not tested. Do not read 467 env-steps/s as an 8,000-iteration budget.
- **The rigid path scales and the cloth path is untested at scale.** Rigid
  goes 128.6 → 623.9 → 1,117.6 across 8 → 32 → 64 envs, close to linear, which
  is what makes Stage 1 the place to learn. The 1024–2048 claim is still an
  extrapolation from 64.

Cloth **build** cost is real and separate from stepping: the Newton scene took
**56.5 s** to construct against 1–6 s for rigid (`21234165`). That is a fixed
cost per job, not per step, but it is why a deformable cell is not something
to launch casually.

G-C1 (Franka, 961 vertices) remains this repo's reference cloth measurement:
182 / 177 / 71 env-steps/s at 8 / 32 / 128 envs.


## Isaac bring-up: what the port actually cost

The kinematic ladder ran on a login node from the first day. Getting the same
task to construct in Isaac took four GPU round-trips, and each returned exactly
one bug. They are recorded because every one of them was a class of mistake
that a login-node check can catch, and three of the four are now caught there.

| # | job | died on | why it was invisible offline |
|---|---|---|---|
| 1 | `21228029` | `TypeError: 'NoneType' object is not callable` | `SweepActionCfg` declared `class_type: type = None` and patched the class attribute afterwards. `configclass` freezes field defaults when it builds the dataclass, so every *instance* still held `None`. |
| 2 | `21233802` | `"bitwise_and_cuda" not implemented for 'Float'` | `&` binds tighter than `<=` in Python, so the basket test was a chained comparison, not a mask. The predicate had never evaluated. |
| 3 | `21233866` | `mean episode length is 1.00` | the fall test used an absolute `R[2,2] < 0.70`, and this asset's stand-up quaternion `(0,0,1,0)` has `R[2,2] = -1`. A standing robot read as fallen at reset. |
| 4 | `21233868` | `FrameView prim '.../table' is a Newton physics body` | the table and basket walls carried `rigid_props`; Newton promotes that to a physics body, which `AssetBaseCfg` refuses. |

Three consequences worth stating outright, because two of them are fidelity
decisions rather than repairs:

**The fall test is measured relative to the spawn pose.** Not against an
absolute body-z convention. This asset's identity orientation is not upright —
the quaternion the spawn photographs (`21218517`/`21218627`) show standing, and
which matches MuJoCo to 9 mm, has `R[2,2] = -1`. Rather than re-open this
repo's long-running argument about which matrix element is "up" for this
asset, `fallen` measures the angle between the robot's current orientation and
the one it spawned with. That reads 0 at reset by construction, and a yaw
jitter still reads 0 because yaw is not tilt. The algebra is
`bhl_robust.cloth.kinematics.relative_up_z`, shared by the torch path and the
numpy tests rather than written twice. **Caveat:** this measures orientation
only, and it cannot tell a squat from a collapse — the same limitation
FINDINGS records for the coop tilt term.

**The cloth table and baskets are collider-only, in the rigid scene too.**
They were kinematic rigid bodies. Newton rejects that, and the obvious fix —
collider-only under Newton, rigid body under PhysX — would have put a physics
difference between C0/C1 and C2/C3 and then charged the difference to the
cloth, which is precisely the measurement this ladder exists to make. Both
scenes now spawn the same static geometry. The maze, plinth, shelf and net are
untouched and keep `rigid_props`; a test enforces that split so no future edit
silently re-times a published number.

**One basket volume, not two.** The Isaac predicate re-derived the basket box
from `BASKET_INNER` with a *half*-extent for z, giving a 0.07 m ceiling where
`layout.basket_aabb` — what the kinematic ladder scores against — uses the full
0.14 m. The two engines were scoring the same garment differently, in a
document that claims they cannot drift because they read one layout object.
Both now call `basket_aabb`.

Also fixed while reading the step path, without a job failing on it:
`progress_to_basket` carried its previous-distance state across episode
boundaries, so the first step after a reset paid the policy for the garment
teleporting back to spawn. That would not have crashed anything — it would
have biased C1's reward toward whichever spawn happened to land nearer a
basket.

### Guards, on the login node

`tests/test_cloth_sort.py` grew from 28 to 43 tests. The new ones parse the
Isaac modules rather than importing them, because a login node cannot
bootstrap Kit — the reason these four bugs reached a GPU queue in the first
place. Each guard was verified to **fail** when its bug is reintroduced:

- no `@configclass` may default `class_type` to a constant, and no module-level
  `SomeCfg.class_type = ...` patching
- the action term class must be defined before the cfg that defaults to it
- no bitwise `&`/`|` inside an unparenthesised comparison
- `cloth_resolution` may not be assigned after construction, and the Isaac
  scripts must build through `build_cfg()`
- `cloth_sort_mdp` may not re-derive the basket box from `BASKET_INNER`
- cloth furniture collider-only, maze/coop furniture not
- the spawn-relative tilt algebra, including the exact `(0,0,1,0)` case that
  broke `21233866`

### A benchmark that cannot mislabel itself

`cloth_resolution` was being assigned *after* the env cfg was constructed, but
the cloth mesh is spawned inside `__post_init__`. A "10×10" bench row would
have been an 8×8 cloth wearing the wrong label. Resolution now goes through
`build_cfg()`, which passes it to `__init__` and then reads the resolution back
off the spawned scene; a mismatch raises instead of writing the row. The CSV
reports what the scene built, not what was requested.

The smoke's throughput field was also folding scene construction into its
timing — tens of seconds of USD loading divided by six steps. It now reports
`build_s` and `step_env_steps_per_s` separately, and neither is the bench
number: `scripts/bench/cloth_sort_isaac_bench.py` warms up first and is the
only thing here that measures throughput.


## Cost gate

```bash
python3 scripts/cloth/estimate_cost.py --physics deformable --num-envs 2048
# exits 2: ~128 days at the G-C1 128-env rate. Refused.

python3 scripts/cloth/estimate_cost.py --physics deformable --num-envs 8 \
    --iterations 20 --steps-per-iter 24
# accepted: ~21 s at 182 env-steps/s
```

A deformable training job that cannot print this report is not submitted.


## Commands

Kinematic ladder (no GPU, no Isaac). **Use the stack's python, not the login
node's** — `/usr/local/apps/python/3.14` has no numpy, so a bare `python3`
fails at the first import:

```bash
export PYTHONPATH=src
PY=/nfs/hpc/share/$USER/Humanoid_Lite/venv/bin/python3   # v51 stack; v60 works too

$PY tests/test_cloth_sort.py -v      # 50 tests, ~0.2 s, no GPU
$PY scripts/cloth/eval_scripted.py --rung C0 --episodes 64 \
    --out results/cloth/c0_scripted.json
$PY scripts/cloth/train_rigid.py --episodes 200 --eval-episodes 64 \
    --out results/cloth/c1_learned.json
$PY scripts/cloth/eval_five.py --episodes 32 \
    --out results/cloth/c5_scripted.json
$PY scripts/cloth/eval_transfer.py --rigid results/cloth/c0_scripted.json \
    --out results/cloth/c3_transfer.json
$PY scripts/cloth/adapt.py --collect 64 --eval-episodes 64 \
    --out results/cloth/c4_bc.json
$PY scripts/bench/cloth_sort_bench.py
$PY scripts/cloth/estimate_cost.py --physics deformable --num-envs 2048
```

Isaac (v60, GPU, after SimulationApp). **Smoke first. No 8,000-iter cloth
job. No 2048-env cloth job.**

```bash
sbatch slurm/95c_cloth_smoke.sbatch          # rigid construct/reset/step
sbatch slurm/95d_cloth_c1_smoke.sbatch     # 3 iters, 64 envs, rigid only
sbatch slurm/95e_cloth_isaac_eval.sbatch   # 4 scripted C0 episodes
sbatch slurm/95f_cloth_deform_smoke.sbatch  # 8×8 cloth, 8 envs, cost-gated
sbatch slurm/95b_cloth_sort_bench.sbatch  # throughput; stop on fall/OOM
sbatch slurm/95g_cloth_spawn_probe.sbatch  # why the robot topples: 3 leg poses, zero action

# after 95c, a human can also:
# python scripts/cloth/eval_isaac.py --rung C0 --headless
```

Replay / render of an Isaac run uses the existing play path:

```bash
python3 scripts/cloth/replay.py --task ClothSort-BHL-Rigid-Oracle-v0
# → BHL_STACK=v60 ENABLE_CAMERAS=1 python scripts/train_play.py --task ... --headless --enable_cameras
```

Registered gym ids (imported after SimulationApp, like every other overlay):

| id | stage |
|---|---|
| `ClothSort-BHL-Rigid-Oracle-v0` | C0 / C1, one rigid proxy |
| `ClothSort-BHL-RigidFive-Oracle-v0` | C5 rigid stand-in, Mode B |
| `ClothSort-BHL-Deformable-Oracle-v0` | Stage 2: one 8×8 Newton `MeshRectangle` |
| `ClothSort-BHL-ActiveCloth-Oracle-v0` | Mode B: 1 cloth + 4 rigid proxies |


## What this is not

- Not five simultaneously active high-res cloths unless a bench says that
  fits, and then only for evaluation.
- Not a report of Mode B as "five live garments".
- Not a silent fidelity drop: kinematic cells are labeled kinematic.
- Not a change to the robot articulation, the maze, or any trained checkpoint.
- Not a deletion of `results/cloth_probe.txt` or the G-C1 ledger rows.


## Layout numbers

Table top at `GRASP_Z`. Robot at `(-0.22, 0)` — facing **−x** as measured (`21247910`), not +x as first written. Baskets at
`x = 0.10`, `y ∈ {+0.34, 0.00, −0.34}`. Constants live in
`src/bhl_robust/cloth/layout.py` and are the same object the Isaac scene
spawns from, so the two cannot drift by editing one file.
