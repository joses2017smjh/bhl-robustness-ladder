# bhl-robustness-ladder

**An 11.3 kg 3D-printed humanoid with 6 Nm joints. How much can it take before
it stops learning?**

Those two numbers drive everything here. This machine has very little authority
to arrest a disturbance, so the question is not rhetorical — and answering it
needed an instrument, because upstream ships flat-ground locomotion with no
curriculum and **no way to score a policy at all**.

So: train in Isaac Lab (PhysX), score in MuJoCo. Different simulator on purpose.
A policy that only works where it trained has learned PhysX, not locomotion.

<p align="center">
  <img src="docs/gifs/multi_race.gif" width="880" alt="Four policies in one MuJoCo world taking identical shoves. Three stay up; the un-randomized robot is on the ground."><br>
  <sub>Four policies, one world, identical 0.45 m/s shoves. Not a composite —
  they share a solver and a clock. The un-randomized robot is the one on the
  ground.</sub>
</p>

<p align="center">
<b>156 policies trained</b> · <b>6,348 scored sim2sim episodes</b> · <b>288 rendered rollouts</b> · <b>13 findings, 4 of them retractions</b><br>
<a href="https://claude.ai/code/artifact/de955af8-2236-4912-84fb-577e0a43ccbe"><b>Explore every run interactively</b></a> — isolate a run, switch metrics, watch the axis rescale.
</p>

---

## Reading this honestly

Four of the thirteen findings below are **retractions of earlier findings in this
same README**, and they are left in rather than edited away. A repo whose whole
argument is "the measurement was wrong" cannot quietly fix its own measurements.

| | |
|---|---|
| **settled** | 3 seeds, no overlap, read from event files — findings 1–6, 11 |
| **suggestive** | 1–2 seeds, or a control still running — findings 10, 12 |
| **retracted** | claimed here earlier, withdrawn on evidence — 7, 9, 13, and the wedge argument in §5 |

Every number is read from the trainer's TensorBoard event file, never from a
scraped log. That rule exists because four published claims were wrong when it
did not.

---

## What came out of it

| | Finding |
|---|---|
| **1** | **Transfer inverts the training-reward ranking.** Highest training reward falls **23%** of the time in MuJoCo; the repo default falls **0%**. |
| **2** | **0.2 m/s of shove-rejection is free** — and the "0.87 m/s ceiling" was an artifact of a safety cap. Uncapped, the curriculum oscillates 0 → 1.8 m/s and never converges. |
| **3** | **Randomization alone buys most of terrain robustness.** A blind policy that never saw rough ground holds to d≈0.4. Arms push that to d≈0.6. |
| **4** | **The terrain plateau is torque, not sensing.** An exact height map of the ground underfoot moves the curriculum **not at all**. Looking *ahead* does. |
| **5** | **Depth never needed the RTX renderer, and what it buys is retention.** It costs **1.6%** of throughput at 4,096 envs. On low friction all six runs peak at ~0.68 and only the sighted three stay there — **0.65 against 0.26**, three seeds, no overlap. |
| **6** | **The sim2sim gap is physics, not bookkeeping.** URDF, USD and MJCF agree; swapping collision primitives for convex meshes moves neither reward nor transfer. |
| **7** | **The only cube arm that ever lifted is a single seed, and it has not replicated.** Occlusion reached 13 cm where nine other interventions sat on the 4 cm floor — but two further seeds and the depth variant are all flat. |
| **8** | **Arms buy recoverable perturbation, not a higher step.** No 12-DoF policy crosses the lab floor; two of four 22-DoF policies do. |
| **9** | **The fall detector cannot see a level collapse.** `bad_orientation` tests torso *orientation*, so a robot that sinks 34 cm with its torso level scores as upright — which is what the ladder pair does for twelve seeds out of twelve. |
| **10** | **The hands were welded shut.** Every manipulation result came from an asset whose grippers are `type="fixed"`. Restoring the two DoF the hardware has takes mean episode length from **6.3 to 427.7 steps**, six cells a side. |
| **11** | **Depth helps on a hazard it cannot see.** On friction patches flush with the floor — ray-cast-verified — depth beats blind by **10.6%**, while colouring them so a camera *could* see them changes nothing (1.394 vs 1.374). |
| **12** | **Splitting the policy at the arms/legs seam helps; splitting further does not.** limb2 finishes **47% above** a no-split control under the same trainer; limb4 lands on it. One seed. |
| **13** | **A gate with no control measures its own budget.** G-B2 rejected a 5 cm stair riser twice on a 300-iteration probe. The walkable-terrain control is *also* pinned at 0.0000 there. Re-run to 2,000 with 5 cm restored, the same probe **passes** at level 0.107. |

Every claim below links into the [full technical report](docs/REPORT.md), which
carries the protocols, the caveats, and the corrections. What is currently
running, what is still outstanding, and which Slurm id produced which number is
tracked in [the job ledger](SLURM_JOBS.md).

---

## The ladder: what randomization costs, and what it buys

Every range in upstream's `EventsCfg` is rewritten as one scale `s`, so
randomization becomes a continuous axis instead of three named presets.

<p align="center">
  <img src="docs/gifs/dr_pair.gif" width="880" alt="Left: randomized policy strafing. Right: un-randomized policy falling on identical ground."><br>
  <sub>Identical strafe command in MuJoCo. Left <code>s=1.0</code>, right
  <code>s=0</code>. Neither policy ever saw MuJoCo during training.</sub>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/dr_ladder_summary-dark.svg">
  <img alt="Reward declines smoothly with randomization scale while fall rate knees upward after s=1.0." src="results/charts/dr_ladder_summary-light.svg">
</picture>

The training column and the transfer column disagree, and that disagreement is
the point. `s = 0` wins training by 49% and loses transfer outright. Reward
declines smoothly; **fall rate** is the quantity with a knee — and upstream's
shipped default sits on the last rung before it.

[Read §1](docs/REPORT.md#1--domain-randomization-the-fidelity-ladder)

---

## How hard a shove is learnable

<p align="center">
  <img src="docs/gifs/push_pair.gif" width="880" alt="Left: push-trained policy staggering and recovering. Right: baseline knocked flat by the same shove."><br>
  <sub>Identical 0.5 m/s shoves. Left trained with a push curriculum, right
  without. <b>0/6 falls against 3/6.</b></sub>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/push_sweep-dark.svg">
  <img alt="Final reward against push ceiling: 33 at 0.2 m/s, 29 at 0.4, 22 at 0.6, 3 at 1.5." src="results/charts/push_sweep-light.svg">
</picture>

0.2 m/s is free. 1.5 m/s destroys the gait — reward climbs to 24.7 and then
collapses to 3.0 as the ramp passes 0.7 m/s. A competence-gated rule does far
better, but lifting its safety cap showed it never actually converges: it
climbs until the gait breaks, drops to zero, and repeats.

[Read §2](docs/REPORT.md#2--push-recovery-how-hard-a-shove-is-learnable)

---

## Rough ground, and what actually pays for it

<p align="center">
  <img src="docs/gifs/terrain_pair.gif" width="880" alt="Left: terrain-trained policy crossing rough ground. Right: flat-trained policy falling on the same ground."><br>
  <sub>Identical rough ground at <code>d = 0.80</code>. Left trained on terrain,
  right flat-trained with the same randomization. <b>0/6 falls against 3/6.</b></sub>
</p>

<p align="center">
  <img src="docs/gifs/arms_terrain_pair.gif" width="292" alt="22-DoF pair on rough ground.">
  <img src="docs/gifs/arms_push_pair.gif" width="292" alt="22-DoF pair taking a shove.">
  <img src="docs/gifs/arms_dr_pair.gif" width="292" alt="22-DoF pair strafing, randomized against un-randomized."><br>
  <sub>The same three comparisons on the 22-DoF body: terrain, shove, strafe.
  Arms are not decoration — at <code>d = 1.0</code> the humanoid falls
  <b>11.7%</b> where the biped falls <b>37.8%</b>.</sub>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/terrain_retention-dark.svg">
  <img alt="Fall rate against terrain difficulty: no randomization fails immediately, randomization-only degrades past d=0.4, terrain-trained stays near zero." src="results/charts/terrain_retention-light.svg">
</picture>

Plain domain randomization, with **zero terrain exposure**, carries the robot to
roughly d = 0.4. Terrain training is what holds past that — 11× fewer falls at
d = 1.0. Evaluation terrain is generated by completely different code from
training terrain, so a policy that memorised its height field scores nothing.

[Read §3](docs/REPORT.md#3--rough-terrain-what-actually-buys-terrain-robustness)

---

## The plateau is torque, not sensing

The curriculum stalls at level 1.4 of 9. That was attributed to "6 Nm joints
**and** no height sensing" — two hypotheses in one sentence. Giving the policy a
perfect height map of the ground underfoot separates them.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/terrain_plateau-dark.svg">
  <img alt="Terrain curriculum level against iteration: the privileged height-scan curve lands on the blind curve; only forward depth rises above it." src="results/charts/terrain_plateau-light.svg">
</picture>

The privileged arm lands **on** the blind curve. Perfect terrain knowledge buys
nothing, which leaves joint authority as the thing that runs out — a hardware
claim about this robot, not an observation-design one. What does move it is
looking *ahead*.

[Read §3b](docs/REPORT.md#3--rough-terrain-what-actually-buys-terrain-robustness)

---

## Depth, without the renderer that will not start

Isaac Sim 5.1's RTX renderer segfaults on this cluster inside
`omni.usd.create_hydra_engine`. That was recorded as blocking every vision
experiment. It was not: `RayCasterCamera` intersects a pinhole ray bundle with
the scene mesh in warp and never asks for a Hydra engine at all.

<p align="center">
  <img src="docs/gifs/depth_pair.gif" width="880" alt="Left: the terrain policy crossing rough ground. Right: its own 64x64 egocentric depth image, near in blue, far in red."><br>
  <sub>Left, the scored episode. Right, the robot's own 64×64 depth image. This
  one is MuJoCo's offscreen depth buffer, not Isaac Lab's ray-caster — a
  different renderer, a different projection, a different simulator. Two
  independent paths agreeing is the argument this repo makes everywhere.</sub>
</p>

| | envs | env-steps/s |
|---|---|---|
| physics only | 4,096 | 21,844 |
| **+ depth 48×48** | 4,096 | **21,490** |

Validated against the closed-form depth image of a flat plane before anything
trained on it: **100% finite pixels, 2.9% mean relative error**. Warp returns
all-NaN rather than an error when the mesh paths are wrong, so the check is not
optional.

### What depth actually buys: it holds the level, it does not raise it

Two terrains, blind against depth, two seeds each, 6,000 iterations. Levels are
the mean over the last 5% of training, not the last logged line — on one of
these arms those two numbers differ by a factor of four, and the difference is
the entire result.

| terrain | | peak level | **final level** | mean |
|---|---|---|---|---|
| low friction | blind | 0.68 / 0.67 / 0.68 | **0.29 / 0.17 / 0.32** | 0.261 |
| low friction | **depth** | 0.67 / 0.72 / 0.67 | **0.61 / 0.70 / 0.64** | **0.652** |
| 5 cm stairs | blind | 2.25 / 2.39 / 2.36 | 2.20 / 2.31 / 2.32 | 2.276 |
| 5 cm stairs | **depth** | 2.70 / 2.65 / 2.59 | 2.68 / 2.52 / 2.33 | **2.511** |

**Finding — on low friction, blind and sighted policies reach the same ceiling
and only one of them stays there.** All six runs peak between 0.67 and 0.68. The
blind three then regress to a mean of 0.26; the depth three hold 0.65. A
final-value comparison reads as "depth is 2.5× better" and a peak comparison
reads as "depth changes nothing"; both are wrong, and the curve is the finding.
Depth does not raise the ceiling here, it stops the fall-back from it.

**And it inverts the prediction this repo wrote down.** `BipedSlipperyDepthEnvCfg`
exists to test "depth pays on geometry, not material", on the reasoning that a
camera cannot see friction. Depth's large effect is on friction, which it cannot
see. Its effect on pure geometry — the stairs — is **+10%**, real and consistent
across seeds but far smaller. The likely mechanism is that seeing the ground
ahead buys foot placement that does not need friction margin, which is a
geometric answer to a material problem; that is a hypothesis, not a result.

**Three seeds per cell, and the third did not overturn it** — which is the test
§1 of this project failed. Blind and depth do not overlap on final level in any
cell: slippery 0.17–0.32 against 0.61–0.70, stairs 2.20–2.32 against 2.33–2.68.
The stairs effect shrank from 1.16× at two seeds to **1.10×** at three, so the
geometry win is real but smaller than the first pass suggested; the friction
win held at **2.5×**.

Worth stating plainly, because it nearly did not get measured: the stairs rung
runs at a **5 cm riser** only because the gate that rejected 5 cm was re-checked
against a control. Both blind and sighted arms clear level 2.2 on it — this is
easily the most learnable terrain in the project — and at the 3 cm the bad gate
argued for, the rung would have measured a ramp.

**Why there is no RGB anywhere in this repo.** Not a preference. Colour needs
the RTX renderer, and a Warp mesh query has no colour to return — depth was the
only modality this cluster could produce. That constraint is being lifted rather
than accepted: `isaaclab 3.0.0b2` pins `isaacsim 6.0.0.1` on Python 3.12, and a
probe already showed 6.0 surviving the call that kills 5.1. It builds as a
**second venv beside the locked 5.1 stack** (`BHL_STACK=v60`), because every
published number here came from 5.1 and none of them should move to find out
whether a renderer works.

[Read §6](docs/REPORT.md#6--depth-without-a-renderer)

---

## The lab floor

Tile, carpet strip, cable, door threshold, ramp — the geometry a depth sensor
would actually be aimed at, and a course to cross rather than a texture to
sample.

<p align="center">
  <img src="docs/gifs/multi_lab.gif" width="880" alt="Four 22-DoF policies crossing a composed lab floor, with the hero policy's egocentric depth image running along the bottom."><br>
  <sub>Four 22-DoF policies, no shove. The hero run wears the orange shell; a
  fallen robot darkens and the frame takes a red outline. Along the bottom is
  that robot's own egocentric depth, with a scrolling waterfall of the centre
  column — so an obstacle is visible in the sensor <i>before</i> it matters to
  the feet.</sub>
</p>

No 12-DoF policy finishes upright. Two of four 22-DoF policies cross the whole
course. Three of the four biped runs end at a **2.5 cm cable** — 9% of leg
length. The sharpest comparison is the same recipe on both bodies: `randomized`
falls at the cable on 12 DoF and finishes on 22.

Arms are not free stability, and the table says so: `no-randomization` is
*worse* with arms. More mass higher up, with no policy trained to use it, is a
liability.

[Read §8](docs/REPORT.md#8--do-the-arms-buy-stability-walk-the-lab-floor-and-see)

---

## Two robots, one object

16 kg each, 6 Nm joints, no fingers, shoulders that cannot adduct past ~36 cm.
Can a pair *learn* a non-prehensile side-lift? One PPO, 44 actions, privileged
critic — the recipe contact-rich dual-arm papers actually train with.

<p align="center">
  <img src="docs/gifs/squat_pick.gif" width="430" alt="Scripted interpolated-joint squat and pick, showing the kinematics are reachable.">
  <img src="docs/gifs/carry_3.gif" width="430" alt="Three robots running the learned lift policy."><br>
  <sub>Left is <b>not</b> a policy — it is a scripted joint interpolation, the
  reachability check that says the pose exists. Right is the learned thing.
  Everything below is the distance between those two clips.</sub>
</p>

<p align="center">
  <img src="docs/gifs/carry_2.gif" width="430" alt="Two 22-DoF robots closing on a cube from opposite sides.">
  <img src="docs/gifs/carry_4.gif" width="430" alt="Two pairs of robots running the same learned lift policy side by side."><br>
  <sub>Left, one pair. Right, two pairs running the same learned policy. They
  close on the object and hold — the pinch forms, the lift does not. Each clip
  runs the seed that stays upright longest out of twelve, picked by
  <code>scripts/bench/pick_seed.py</code> rather than left at seed 0, so the
  clip and the numbers below it are the same rollout. For one pair that is
  <b>7.8 cm of lift, hands inside the pinch gate 98% of the time, both robots
  still standing at 12 s</b> — four seconds past the trained horizon. The median
  across all twelve seeds is 5.5 cm.</sub>
</p>

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/coop_ablation-dark.svg">
  <img alt="Cube recipe ablation: the arm that never pays for height is the one that forms a pinch." src="results/charts/coop_ablation-light.svg">
</picture>

Fourteen knobs, then three seeds on the four that mattered — and the seeds
overturned the headline. The pinch gap between arms is inside seed noise. What
survives three seeds out of three is the **fall rate**: never paying for height
produces the only arm that reliably stays upright. Ordering the reward — pinch
first, height second — then gets a pinch, a clamp, a lift bonus and the highest
reward of any arm, with height still pinned at 4 cm.

### The lift is performed from the floor

Watching the clips rather than the scalars raises a question the tables never
could: the robots appear to go down *first* and grab the cube afterwards. They
do.

| cube arm, MuJoCo | half of its descent | first pinch | first 1 cm of lift |
|---|---|---|---|
| staged | **t = 0.20 s** | t = 0.28 s | t = 0.52 s |
| pinch-only | **t = 0.24 s** | t = 3.76 s | t = 0.60 s |

Both drop ~41 cm within a fifth of a second, before any contact with the
payload. The staged arm then holds one pose for the remaining 11 seconds — sink
40.5 cm, pinch 0.076 m, lift 4.4 cm, all flat to three digits. The "best rollout
in the project" is a controlled collapse into a static brace that happens to
hold the cube 4 cm off the ground.

**Nothing in the task charges for it.** Both fall tests read *orientation* only —
`either_fallen` on a 0.78 rad limit and `flat_orientation_l2` on projected
gravity — and there is **no base-height term anywhere in the rewards or the
terminations**. Meanwhile getting low is worth a great deal: it puts the hands at
the height of a 28 cm cube, which is `reaching_coarse` (1.0), `reaching_fine`
(1.0) and `opposing_clamp` (1.5), and it unlocks `lift_progress` (2.0) and
`lifting_object` (**15.0**, the largest weight in the task). The gradient points
down and there is no counter-gradient. That is a reward specification that
*permits* the behaviour by construction.

**Whether training exploited it is not yet established, and the PhysX side
argues against the strong version.** `base_contact_a/b` — which fires when the
torso geom touches the ground — sits at **-0.003** across training, and
`still_alive` at 0.98. In Isaac the torso is not on the floor. So the 41 cm
collapse is either a sim2sim failure that MuJoCo exposes and PhysX does not, or
a deep squat that MuJoCo exaggerates. Base contact cannot separate those,
because a robot folded onto its shins never touches its torso down either.

The test that would settle it is base height logged during training, which this
project does not currently record — the term does not exist, which is the same
reason the behaviour is unpenalised. It is being added.

### Giving them eyes made it worse

The obvious next move is to put a camera on the lift. A ray-cast depth camera
per robot tracks the payload's transform, so the cube stays visible as it moves
— no RTX renderer needed. Two arms: depth *replacing* the privileged object
pose, and depth *added alongside* it.

<p align="center">
  <img src="docs/gifs/carry_2.gif" width="430" alt="Blind policy: two robots closed on the cube, both upright.">
  <img src="docs/gifs/carry_vision_swap_2.gif" width="430" alt="Depth-conditioned policy: one robot face-down on the floor, fall outline on the frame."><br>
  <sub>Same task, same 4,000 iterations. Left blind, right with depth replacing
  the object pose. The red outline is a fall, held for the rest of the clip.
  The strip down the right of the sighted clip is <b>what that policy is
  reading</b> — colour under the white rule for context, the raw 64×64 depth
  frame under the orange, and under the blue the <b>8×8 vector it is actually
  handed</b>. That bottom pane is the entire visual input: 64 numbers, pooled,
  in place of an exact object pose.</sub>
</p>

<p align="center">
  <img src="docs/gifs/carry_vision_swap_3.gif" width="292" alt="Depth-swap policy, three robots.">
  <img src="docs/gifs/carry_vision_both_2.gif" width="292" alt="Depth-alongside policy, two robots, both down.">
  <img src="docs/gifs/carry_vision_both_4.gif" width="292" alt="Depth-alongside policy, four robots, all down."><br>
  <sub>Not one unlucky rollout. Left, depth replacing the pose; centre and
  right, depth added alongside it — the arm that never forms a pinch at all.
  Every one carries its own POV strip, and in every one the cube is plainly
  visible in the depth panes for the whole clip. These policies are not
  failing to see the payload. They are failing to act on 64 pooled numbers
  where the blind arm was handed the pose exactly.</sub>
</p>

| | pinch | held lift | fell |
|---|---|---|---|
| **blind** | **0.42 – 0.91** | **4.0 – 4.8 cm** | **0.12 – 0.25** |
| depth replaces pose | 0.03 – 0.18 | 0.2 – 0.7 cm | 0.62 – 1.00 |
| depth added alongside | 0.00 | −0.1 cm | 0.88 – 1.00 |

**Finding.** Not a small regression — the sighted policies stop forming a pinch
at all and fall in almost every episode. The reason is visible in the setup
rather than the training: the actor is *already handed the exact object pose*.
Depth here does not add information, it substitutes a 64-dim pooled image for a
quantity the policy was being given exactly, and pools away the spatial
precision a pinch depends on. Vision earns its place when the payload is unknown
or occluded, which is a different experiment.

### And the other two objects

The recipe that lifts the cube was only ever run on the cube. Rerun on both:

| object | pinch | lift bonus |
|---|---|---|
| cube | 0.298 | 0.93 |
| yoga ball | 0.063 | 0.51 |
| ladder | **0.000** | 0.00 |

**The ball is the only arm whose height curriculum ever moved, and it did not
survive the cross-check.** In PhysX it climbs 4 cm → 6 cm → 13 cm → **20.9 cm**,
stopping just under the 22 cm cap; that curriculum promotes only on repeated
success, so inside the trainer this is a real lift. Replayed in MuJoCo, the same
weights fall over in **0.72 s**, in **0 of 6 seeds**, and — put in the cube scene
instead — in 0 of 6 there too, at 0.77 s. The collapse travels with the weights,
not the payload. The 21 cm is a single-engine number and this repo does not
count those.

**A geometry advantage looked real at six seeds and did not survive twelve.**
The tempting story is mechanical: squeezing a sphere from two sides gives
contact normals angled inward *and upward*, so the squeeze carries its own
vertical component, where a cube's vertical faces give purely horizontal normals
and every newton of lift has to come from friction. Drop the *cube*-trained
policy into a ball scene it has never seen and the first six seeds agree — 9.1 cm
against 6.1 cm on its own cube.

Twelve seeds reverse it.

| cube policy, 12 seeds | median lift | mean lift | max lift | upright at 12 s |
|---|---|---|---|---|
| in the **cube** scene | **5.1 cm** | **5.2 cm** | 7.8 cm | **6/12** |
| in the **ball** scene | 3.1 cm | 3.7 cm | 9.1 cm | 1/12 |

Both 6-seed figures were maxima, and the ball's distribution is the one with the
long tail: a higher best, a lower middle, and a fall in eleven runs out of
twelve at a mean of 1.5 s. A peak height reached during a topple is not a lift,
and *peak* is the wrong statistic for a claim about lifting — the median says
the ball is worse, and the upright count says it is much worse. **The wedge
argument is not supported by this data.** It is left here as the hypothesis it
is, and what the run actually shows is that a 65 cm sphere is harder to stand
next to than a 28 cm box.

| policy | scene | upright at 2 s | mean fall |
|---|---|---|---|
| ball | ball | 0/6 | 0.72 s |
| ball | cube | 0/6 | 0.77 s |
| cube | ball | 0/6 | 1.73 s |
| cube | cube | **5/6** | **6.65 s** |

Nothing here is a solved lift. The cube arm is the only one that reliably stays
upright, and it lifts 5 cm. Cooperative lifting on this robot is not done.

<p align="center">
  <img src="docs/gifs/carry_ball_transfer_pov.gif" width="440" alt="A cube-trained policy lifting a yoga ball it never saw in training, with the robot's colour and depth views alongside.">
  <img src="docs/gifs/carry_cube_pov.gif" width="440" alt="The same policy on the cube it was trained on, with the robot's colour and depth views alongside."><br>
  <sub>Right is the best rollout in the project: <b>7.8 cm of lift, hands inside
  the pinch gate 98% of the time, and the fall criterion never triggered across
  the full 12 s</b> — four seconds longer than the horizon it was trained on.
  "Never fell" is doing exact work there: the test is training's own
  torso-orientation termination, and the pair drops <b>41 cm in the first
  0.2 s</b> — before touching the cube — then holds one static pose for the
  remaining 11 s. See <a href="#the-lift-is-performed-from-the-floor">below</a>. Left is the same
  weights on a ball they have never seen, which reaches higher on its best seed
  and falls on eleven of twelve, and is why the table above reads medians. The
  strip on the right of each frame is that robot's own head camera, three views
  from one pose at one instant: <b>colour</b> under the white rule, the
  <b>raw 64x64 depth frame</b> under the orange, and under the blue the
  <b>8x8 the network is actually handed</b>. The two depth panes are masked to
  floor and payload because that is the target list the ray-caster casts
  against — they are the tensor the policy receives, not a picture of the room,
  and the gap between them is the resolution it does not get.</sub>
</p>

<p align="center">
  <img src="docs/gifs/carry_ball_native_pov.gif" width="440" alt="The ball-trained policy toppling in MuJoCo without reaching the ball."><br>
  <sub>The 21 cm arm, cross-checked. It falls at 0.72 s having never touched the
  payload — the ball does not move a millimetre. This is what a single-engine
  result looks like from the other engine.</sub>
</p>

The ladder is at pinch identically zero for the third recipe in a row, and its
staging latch never fires at all. That stops being "we never retried it" and
becomes a property of the object: long, thin and light puts the two contact
points further apart than the shoulders can span, with nothing to squeeze
against.

<p align="center">
  <img src="docs/gifs/carry_ladder_pov.gif" width="620" alt="Two robots either side of a 1.5 m plank, never closing on it; the depth panes show the plank sitting untouched at the bottom of the frame."><br>
  <sub>The third object, rendered for the first time. <b>Twelve seeds out of
  twelve: 0.0 cm of lift, closest approach 39 cm, and the fall criterion never
  triggered.</b> That last part is misleading on its own, and the clip is why
  it is here: the pair sinks about 34 cm in the first two seconds and stays
  down, torso level the whole way, which training's orientation test does not
  score as a fall. They do not lift and they do not stay up — they subside. The
  pair starts 1.7 m apart across a 1.5 m plank whose contact points are 75 cm
  from its centre,
  further apart than shoulders that cannot adduct past 36 cm can span. Note how
  much darker the depth panes are here than in the cube clips: the same camera,
  the same fixed ramp, a payload 8 cm tall seen from 85 cm away. Unlike every
  other null in this section, this one is not a stability failure.</sub>
</p>

### Nine interventions, one seed that moved, and no replication yet

Height sat at exactly 4 cm — the curriculum's floor — in every cube arm that had
been run. So it got attacked from nine directions at once: twice the training,
the tilt penalty halved and removed, an exploration bonus at two weights, a
randomised payload, occlusion, and a recurrent policy.

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="results/charts/coop_nine-dark.svg">
  <img alt="Lift-height curriculum curves. Seven arms lie on top of each other at the 4 cm floor for up to 16,000 iterations; one occluded seed climbs to 13 cm and holds it, while two further occluded seeds stay on the floor." src="results/charts/coop_nine-light.svg">
</picture>

| arm | pinch | lift bonus | height |
|---|---|---|---|
| 16k iterations, seed 1 / 2 | 0.290 / 0.320 | 0.00 | 0.0400 |
| tilt penalty × 0.5 | 0.223 | 0.00 | 0.0400 |
| tilt penalty removed | 0.273 | 0.00 | 0.0400 |
| exploration bonus 0.003 | 0.207 | 0.00 | 0.0400 |
| exploration bonus 0.010 | 0.048 | 0.00 | 0.0400 |
| randomised payload | 0.254 | 0.00 | 0.0400 |
| **occluded, blind** | 0.397 peak | **0.53** | **0.1250** (sustained peak 0.1510) |
| occluded, blind + LSTM | 0.408 peak | 3.19 peak, 0.00 final | 0.0947 peak, 0.0400 final |

**Finding — one seed lifted, and so far only that seed.** Seven of the nine sit
on the floor with `stage_lift` at 0.0000: the latch never fires, so the lift
reward stays at zero, so the curriculum never sees a success. They all landed
upstream of a switch that stayed off.

One occlusion arm is the exception, and it is the only cube arm in the project
that has ever lifted: the latch fires, holds, and the height curriculum climbs
to a **15.1 cm sustained peak, ending near 13 cm after 16,000 iterations**. Not
a spike — a level it holds for thousands of iterations. (Heights are read off a
133-iteration mean; the raw series touches 21.1 cm for a single iteration, which
is not a number worth leading with.)

**It has not replicated.** Two further seeds of the same arm and the depth
variant were queued precisely to test it, and all three are flat:

| occluded arm, 16k iters | stage_lift | lift bonus | height |
|---|---|---|---|
| blind, seed 0 | **1.0000** | **0.67** | **0.1313** |
| blind, seed 1 | 0.0000 | 0.00 | 0.0400 |
| blind, seed 2 | 0.0000 | 0.00 | 0.0400 |
| depth, seed 0 | 0.0000 | 0.00 | 0.0400 |

All four have now run their full 16,000 iterations, so this is a finished
comparison rather than a partial one. Seeds 1 and 2 never leave the floor, and
never fire the latch, over the whole run — including well past the ~11,500 mark
where seed 0 took off. This project has been here before and wrote the rule down
in §1: the single-seed "pinch 0.40 against 0.08" did not survive three seeds
either. **One of three is a lead, not a result**, and it is reported here as
one.

The control for that is unusually clean, because the arm was accidentally run
twice. The first run predates the `apply_depth_flags` fix, which was injecting
`object_pos_a`/`object_pos_b` into every task's policy group — so the "occluded"
arm was quietly handed the exact object pose after all. Same config, same 16,000
iterations, same seed:

| occluded-blind, 16k iters | pose | stage_lift | lift bonus | final height |
|---|---|---|---|---|
| before the fix | **restored by the bug** | 0.0000 | 0.00 | 0.0400 |
| after the fix | **actually hidden** | 1.0000 | 0.53 | **0.1250** |

The tempting reading — that the blind policy was never short of information, was
handed the pose exactly, and only learns to close on it once the pose is taken
away — lines up neatly with why depth made things worse two sections above. It
is also exactly the kind of story that a single seed will happily tell you.
Until seed 1 or seed 2 moves, the honest version is narrower: **occlusion is the
only condition under which this task has ever been solved once.**

The recurrent variant finds the same thing and cannot hold it: the highest lift
bonus in the project, 3.19, and then a collapse back to the floor by iteration
8,000 with the bonus at zero.

Depth-under-occlusion has now run for real. Both earlier attempts predated the
`apply_depth_flags` fix and so were never actually occluded; the repeat is in
the table above, pinned at 0.0400 with a pinch reward of 0.019 — the worst in
the section. A camera does not rescue this task even under the one condition
where §"giving them eyes" predicts it might.

It also retires an earlier explanation. `notilt` peaked at 15.9 cm and the tilt
penalty looked like the cap; removing it entirely changes nothing. And the seven
pinned arms reporting the *identical* 0.0400 rather than a spread was always the
tell — a genuine plateau scatters, a switch that never flips does not.

> **Corrections.** Two earlier readings of this section were wrong, and both are
> worth stating because the same mistake produced them.
>
> The first blamed an unreachable staging threshold, comparing a best pinch of
> 0.32 against a threshold of 0.40. Those are different quantities — 0.32 is an
> episode-mean *reward* for a term carrying weight 2.0, and the threshold is
> read against an instantaneous kernel in [0, 1].
>
> The second said the number never moved in any of the nine. It moved in two of
> them, to 12.5 cm. That claim came from `tail -1` on a training log rather than
> from the trainer's event file, and the log's last line was not the last
> iteration. Both readings scraped a summary instead of reading the series.

[Read §5](docs/REPORT.md#5--cooperative-lift-can-two-of-them-learn-to-pick-something-up)

---

## Everything that was tested, and how it went

Every clip below is a real rollout from a trained policy in MuJoCo, scored by
the same harness that produced the numbers. Nothing here is a scripted
animation except where it says so.

### What works

<p align="center">
  <img src="docs/gifs/multi_race.gif" width="860" alt="Four policies in one MuJoCo world; three stay up, the un-randomized one falls."><br>
  <sub><b>Randomization buys most of terrain robustness.</b> Four policies, one
  world, one command. The un-randomized robot is the one on the ground.</sub>
</p>

<p align="center">
  <img src="docs/gifs/dr_pair.gif" width="425" alt="Randomized policy walking; un-randomized policy falling.">
  <img src="docs/gifs/depth_pair.gif" width="425" alt="Terrain policy crossing rough ground beside its own 64x64 depth image."><br>
  <sub>Left: the same command on randomized and un-randomized policies. Right:
  ray-cast depth at <b>1.6% of throughput</b> — the sensor that made §6 and the
  ice rung possible without an RTX renderer.</sub>
</p>

<p align="center">
  <img src="docs/gifs/push_pair.gif" width="425" alt="Push recovery, 12-DoF biped.">
  <img src="docs/gifs/arms_push_pair.gif" width="425" alt="Push recovery with arms, 22 DoF."><br>
  <sub><b>Arms buy recoverable perturbation.</b> Left 12 DoF, right 22. The arms
  move angular momentum away from the legs — worth 0.2 m/s of shove rejection,
  and it is free.</sub>
</p>

### What does not

<p align="center">
  <img src="docs/gifs/carry_cube_pov.gif" width="500" alt="Two robots holding a cube 7.8 cm off the floor with the robot's own camera views alongside."><br>
  <sub><b>The best cooperative rollout in the project</b> — 7.8 cm of lift, hands
  inside the pinch gate 98% of the time, twelve seconds without a fall. It is
  also a controlled collapse: the pair drops 41 cm in the first 0.2 s, before
  touching the cube. The strip is that robot's head camera — colour, raw depth,
  and the 8×8 the network actually receives.</sub>
</p>

<p align="center">
  <img src="docs/gifs/carry_ladder_pov.gif" width="425" alt="Two robots either side of a plank, never closing on it.">
  <img src="docs/gifs/carry_ball_native_pov.gif" width="425" alt="The ball policy toppling without touching the payload."><br>
  <sub>Left: the plank, <b>12 seeds out of 12 at 0.0 cm</b> — they do not fall and
  they do not try; the contact points are further apart than shoulders that
  cannot adduct past 36 cm can span. Right: the arm that reported 21 cm in
  PhysX, cross-checked in MuJoCo — it falls at 0.72 s without moving the ball a
  millimetre.</sub>
</p>

<p align="center">
  <img src="docs/gifs/carry_vision_swap_2.gif" width="425" alt="Depth-conditioned policy with one robot face-down.">
  <img src="docs/gifs/carry_vision_both_4.gif" width="425" alt="Depth-alongside policy, four robots, all down."><br>
  <sub><b>Giving them eyes made it worse.</b> The cube is plainly visible in the
  depth panes throughout. These policies are not failing to see it — they are
  failing to act on 64 pooled numbers where the blind arm was handed the pose
  exactly.</sub>
</p>

### The reachability check, which is not a policy

<p align="center">
  <img src="docs/gifs/squat_pick.gif" width="425" alt="Scripted joint interpolation through a squat and pick."><br>
  <sub>A scripted joint interpolation, included because it is the control for
  every failure above: <b>the pose exists and is reachable.</b> Everything the
  learned policies could not do, they could not do for reasons other than
  kinematics.</sub>
</p>

---

## The hands were welded shut the whole time

Every manipulation result above was produced by a robot that cannot close a
hand. Upstream's own driver commands two grippers over serial — `bimanual.py`
maps a `[0, 1]` target onto a raw `[0.2, 0.8]`, documents 0.2 open and 0.85
closed, and `run_teleop.py` places them at `robot_actions[10]` and `[11]`. The
simulation asset has neither: both hand joints are `type="fixed"`, and
`arm_*_hand_link` is a rigid 74 × 69 × 136 mm block bolted to the elbow.

So §5's whole arc — a pinch that forms and never lifts, nine interventions that
move nothing, a policy that discovers a braced collapse because squeezing
destabilises it — is the behaviour of a machine trying to hold things between
two fixed blocks. Not a property of the robot. A property of the model.

`scripts/add_gripper.py` writes a 24-DoF copy into the workspace, leaving
`external/` pristine. One DoF per hand, the finger closing against the palm,
which is the grasp the hardware actually performs: lay the open hand over the
object, close, and let finger and palm retain it geometrically rather than by
friction. Checked before anything trained on it — 24 actuated joints, fingertip
sweeping 6.5 cm across the palm, hands mirrored.

Six cells a side, cube and ball, 8,000 iterations each, everything identical
except the two hand DoF:

| | welded hands (n=8) | **grippers (n=6)** |
|---|---|---|
| mean episode length | 6.3 | **427.7** |
| mean reward | −0.71 | **+11.49** |
| task success | 0 | 0 |

**Sixty-eight times the episode length for two joints.** The welded-hand arms
fall over in six steps; the gripper arms stay up for four hundred and
twenty-seven. Task success is still zero in both, so this is survival rather
than completion — but the welded-hand arms were never going to complete
anything, and that is now measured instead of argued.

The plank cells are excluded from both columns and are not reported: they
trained on a scene that ejected its own payload, and their gripper arms show
none of this effect (episode length 11.6, reward −1.0). That is what a task
looks like when the payload leaves before the policy can act, and it is being
re-run now that the stand-off is fixed.

The honest reading is that the manipulation ceiling this project spent six
sections measuring was a property of the asset, and the number that moved when
it was fixed is the largest single effect here.

---

## Ice: depth helps on a hazard it cannot see

§6 found depth beating blind **2.5×** on uniformly low friction — a result that
inverted this repo's own written prediction, since a ray-cast depth camera
returns geometry and friction has none. The obvious explanation is that depth
was seeing something incidental. This rung removes that possibility.

Low-friction patches on ground that is geometrically **flat**. The patches sit
exactly coplanar with the floor, and `scripts/bench/ice_gate.py` ray-casts the
boundary to prove it rather than trusting the config — it passes at the flush
inset and fails at a 5 mm proud patch. So the ray-caster returns the same height
field with or without the ice.

| arm | terrain level (2 seeds) |
|---|---|
| **depth** | **1.519** |
| visible ice | 1.394 |
| blind | 1.374 |

**Depth still wins, by 10.6%, on a hazard it provably cannot perceive.** That
rules out "it sees the ice" as the mechanism and leaves the one this repo
guessed at in §6: looking ahead buys foot placement that needs less friction
margin everywhere, which is a geometric answer to a material problem.

Colouring the patches so a camera *could* see them changes essentially nothing —
1.394 against blind's 1.374, well inside the spread between the two blind seeds.
That is the control on the control, and it is the sharper half of the result:
making the hazard **visible** does not help, while depth that **cannot see it**
helps by 10.6%. Whatever depth is buying, it is not hazard detection.

---

## Limb agents: the split helps, and the two-way split helps more

One agent per limb, under MAPPO, against the same task the single-agent PPO
rungs use. `limb_partition.py` slices the 22 DoF as 5/5/6/6 by index, and G-B4
checks that the four action slices reassemble into exactly the vector a
single-agent policy would have emitted — including **order**, since concatenating
in dict order passes a naive round-trip and still permutes the robot's joints.

| arm | mean reward, first → final | gain |
|---|---|---|
| **limb2 (arms / legs) + MAPPO** | −1.31 → **+3.22** | +4.53 |
| **limb1 (no split, control)** | −0.65 → **+2.19** | +2.84 |
| limb4 (one per limb) + MAPPO | −2.59 → +2.08 | +4.67 |
| limb4 + IPPO | −2.56 → +1.95 | +4.51 |

**The two-way split beats not splitting; the four-way split does not.** limb2
finishes 47% above the control, while limb4 lands on it — 2.08 against 2.19.
That is the opposite of what the design note argued when it called limb4 the
headline and limb2 the ablation, and it lands on exactly the credit-assignment
boundary §5 characterised, where the lift lives in the arms and the legs do the
standing. Splitting along that seam helps; splitting further does not.

> **Two caveats, and both are load-bearing.**
>
> **The starting points are not equal, and they scale with agent count** — limb1
> −0.65, limb2 −1.31, limb4 −2.59. More independent random policies coordinate
> worse at initialisation, which is expected but means the *final* column and the
> *gain* column disagree: limb4 gains the most (+4.67) and finishes lowest. The
> claim above rests on final value, which is the quantity the task cares about,
> but the disagreement is real and is why the gain column is printed.
>
> **This is one seed.** §1 of this project is a single-seed result that died on
> its third seed, and the same standard applies here. Suggestive, not settled.
>
> The original PPO control is not in the table: it routes through rsl-rl while
> every MARL row routes through skrl, so it prices the RL library as well as the
> factorisation. `limb1` replaced it — one agent owning every joint under the
> same trainer, models and hyperparameters.

---

## How any of this is measured

Upstream's sim2sim script constructs a gamepad, blocks on joystick input,
requires a GUI, sleeps to hold realtime, and emits **no metric of any kind**. It
is a thing to watch, not a thing to measure.

<p align="center">
  <img src="docs/img/pipeline.svg" width="100%"
       alt="Train in Isaac Lab at 4096 envs with PPO for 6000 iterations; export the checkpoint through the vendored play.py to policy.onnx and deploy.yaml; drive it in a headless scripted MuJoCo harness, which emits per-episode CSV that becomes curves and ladder charts, and MP4 that becomes paired GIFs.">
</p>

The harness drives the policy through upstream's **own** `RlController`, so
observation construction is bit-identical to the sim2real deployment path. An
evaluator that rebuilt the 45-dim observation independently would be measuring
its own reimplementation. A fall reuses training's own `bad_orientation`
threshold; velocity error is computed in the yaw frame, because that is the
frame upstream rewards.

Six commanded velocities × five seeds × 10 s episodes, identical for every
policy, on one fixed terrain seed.

[Protocols, metric definitions, and the cluster notes](docs/REPORT.md#how-any-of-this-is-measured)

---

## Reproducing

```bash
git clone --recurse-submodules https://github.com/joses2017smjh/bhl-robustness-ladder.git
cd bhl-robustness-ladder
sbatch slurm/00_build_container.sbatch   # ~4 min
sbatch slurm/01_uv_sync.sbatch           # ~30-60 min, ~30 GB
sbatch slurm/02_smoke_train.sbatch       # gate check
```

Isaac Sim needs glibc ≥ 2.34 and the login node is Rocky 8, so everything runs
inside an Apptainer image with the venv on shared storage outside the git tree.
Jobs declare several partitions and an explicit GPU-architecture constraint —
one partition's per-user GPU cap is what the queue actually binds on, and the
OS tag alone will happily hand you a card the pinned torch has no kernels for.

The full job list, the partition map, and what each one produces are in the
[report](docs/REPORT.md#reproducing).

---

## Repo layout

```
external/Berkeley-Humanoid-Lite   upstream, pinned (submodule, unmodified)
src/bhl_robust/
  tasks/        overlay env configs registering new gym task ids
  curricula/    push-magnitude ramp + adaptive rule
  terrains/     rough / slope / obstacle / stairs generators
  eval/         headless MuJoCo harness, MJCF repair, height fields, depth, video
  audit/        three-way URDF / USD / MJCF consistency
  usd/          scripted OpenUSD stages (terrain variants, lab scene)
scripts/        vendored train/play entrypoints, curves, charts, GIFs, benches
slurm/          sbatch scripts (OSU COE HPC)
docs/           REPORT.md, interactive run explorer, README clips
results/        curves, charts, per-episode CSVs, aggregated tables, audit JSON
```

## What was found in upstream

Six genuine breakages, all worked around without modifying `external/` —
including a curriculum config bound to the wrong attribute name (every
curriculum term dead code), and an MJCF whose mesh paths mean **the MuJoCo model
does not load at all**. [The list](docs/REPORT.md#what-was-found-in-upstream).

## License

Experiment code MIT. Upstream BHL retains its own license under `external/`.
