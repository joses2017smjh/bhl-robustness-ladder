# Every clip, with what made it

**23 clips from MuJoCo, 7 from Isaac Sim.** Renderer is in the folder name:
`docs/gifs/` is MuJoCo, `docs/gifs/isaac/` is Isaac Sim. Every clip is a real
scored episode from the harness that produced the numbers, except `squat_pick`,
which says on its face that it is scripted.

Tell me which names you want on the front page.

## Index

| clip | renderer | task / rung | verdict |
|---|---|---|---|
| [`dr_pair`](#locomotion) | MuJoCo | domain randomization | **works** |
| [`push_pair`](#locomotion) | MuJoCo | push curriculum | **works** |
| [`terrain_pair`](#locomotion) | MuJoCo | terrain curriculum | **works** |
| [`arms_dr_pair`](#locomotion) | MuJoCo | 22-DoF vs 12-DoF, randomization | **works** |
| [`arms_push_pair`](#locomotion) | MuJoCo | 22-DoF vs 12-DoF, shove | **works** |
| [`arms_terrain_pair`](#locomotion) | MuJoCo | 22-DoF vs 12-DoF, terrain | **works** |
| [`multi_race`](#four-policies-at-once) | MuJoCo | 4 policies, one shove | **works** |
| [`multi_lab`](#four-policies-at-once) | MuJoCo | 4 policies, obstacle course + depth | **works** |
| [`depth_pair`](#depth) | MuJoCo | ray-cast depth | **works** |
| [`ice_pair`](#b3--ice) | MuJoCo | B3 — blind vs depth on flush friction patches | render works — **result retracted**: in training the ice was never under the robots |
| [`isaac/maze_stereo_fixed`](#terrain-sensing-the-b5-maze-rung) | Isaac Sim | B5 — the corrected 16×16 stereo policy, and what its camera saw before and after the fix | **works** — terrain level 0.88, was 0.02 |
| [`isaac/terrain_sensors`](#terrain-sensing-the-b5-maze-rung) | Isaac Sim | B5 — blind, lidar, stereo at two widths | render only — the result is the table, not the clip |
| [`isaac/cloth_sort_free_base_jacket`](#cloth-sorting-free-base) | Isaac Sim | cloth-sort C0, scripted sweep, rigid jacket proxy, **free-standing robot** | **works** — jacket 8/8 free-base; shirts only 1/8 |
| [`isaac/cloth_sort_fixed_base`](#cloth-sorting-fixed-base-diagnostic) | Isaac Sim | cloth-sort C0, scripted sweep, rigid shirt proxy, **root pinned** | **works** — 32/32 on this layout; not cloth, not a standing robot |
| [`squat_pick`](#the-cooperative-lift) | MuJoCo | scripted reachability control | n/a — scripted |
| [`carry_2`](#the-cooperative-lift) | MuJoCo | cooperative cube lift | **fails** |
| [`carry_3`](#the-cooperative-lift) | MuJoCo | learned vs scripted | **fails** |
| [`carry_4`](#the-cooperative-lift) | MuJoCo | one pair vs two | **fails** |
| [`carry_cube_pov`](#robot-pov) | MuJoCo | cube lift, POV | **fails** — best is 7.8 cm |
| [`carry_ball_native_pov`](#robot-pov) | MuJoCo | ball lift, POV | **fails** — 0/6 seeds |
| [`carry_ball_transfer_pov`](#robot-pov) | MuJoCo | ball, transfer condition | **fails** |
| [`carry_ladder_pov`](#robot-pov) | MuJoCo | plank lift, POV | **fails** — 0.0 cm, 18 seeds |
| [`carry_vision_swap_2`](#vision-made-it-worse) | MuJoCo | depth replacing object pose | **fails** |
| [`carry_vision_swap_3`](#vision-made-it-worse) | MuJoCo | same, three pairs | **fails** |
| [`carry_vision_swap_4`](#vision-made-it-worse) | MuJoCo | same, four pairs — unused | **fails** |
| [`carry_vision_both_2`](#vision-made-it-worse) | MuJoCo | depth alongside object pose | **fails** |
| [`carry_vision_both_3`](#vision-made-it-worse) | MuJoCo | same, three pairs — unused | **fails** |
| [`carry_vision_both_4`](#vision-made-it-worse) | MuJoCo | same, four pairs | **fails** |
| [`occlusion_s0_lifted_pov`](#the-occlusion-seed-cross-checked) | MuJoCo | occlusion, the seed that lifted | **fails in MuJoCo** |
| [`occlusion_s1_flat_pov`](#the-occlusion-seed-cross-checked) | MuJoCo | occlusion, a flat sibling | holds the pinch, never lifts |
| [`isaac/cubetoshelf_upright_welded`](#isaac-sim) | **Isaac Sim** | v2 CubeToShelf, corrected spawn | spawns underground, extruded upward — see retraction |
| [`isaac/cubetoshelf_upright_gripper`](#isaac-sim) | **Isaac Sim** | same, 24-DoF gripper | spawns underground |
| [`isaac/cubetoshelf_gripper`](#isaac-sim) | **Isaac Sim** | v2 CubeToShelf, 24-DoF gripper | **fails** — survives, never lifts |
| [`isaac/cubetoshelf_welded`](#isaac-sim) | **Isaac Sim** | v2 CubeToShelf, welded hands | **fails** — ~8 steps |

---

## Locomotion

Two policies, one command, one world. Left is the intervention, right the control.

| | |
|---|---|
| <img src="gifs/dr_pair.gif" width="420"> | **`dr_pair`** — identical strafe. Left `s=1.0`, right `s=0`. Neither policy ever saw MuJoCo in training. |
| <img src="gifs/push_pair.gif" width="420"> | **`push_pair`** — identical 0.5 m/s shoves. Left has a push curriculum. **0/6 falls against 3/6.** |
| <img src="gifs/terrain_pair.gif" width="420"> | **`terrain_pair`** — rough ground at `d = 0.80`. Left terrain-trained, right flat-trained. **0/6 against 3/6.** |
| <img src="gifs/arms_dr_pair.gif" width="420"> | **`arms_dr_pair`** — the same three comparisons on the 22-DoF body. At `d = 1.0` the humanoid falls 11.7% where the biped falls 37.8%. |
| <img src="gifs/arms_push_pair.gif" width="420"> | **`arms_push_pair`** — 12 DoF against 22. Arms move angular momentum away from the legs: 0.2 m/s of shove rejection, free. |
| <img src="gifs/arms_terrain_pair.gif" width="420"> | **`arms_terrain_pair`** — arms on rough ground. |

## Four policies at once

| | |
|---|---|
| <img src="gifs/multi_race.gif" width="420"> | **`multi_race`** — identical 0.45 m/s shoves. Same solver, same clock, not a composite. The un-randomized robot is the one on the ground. |
| <img src="gifs/multi_lab.gif" width="420"> | **`multi_lab`** — current README hero. Four colour-coded policies crossing plank, beam and ramp, with the orange robot's egocentric depth along the bottom. **14 MB.** |

## Depth

| | |
|---|---|
| <img src="gifs/depth_pair.gif" width="420"> | **`depth_pair`** — left the scored episode, right the robot's own 64×64 depth. MuJoCo's offscreen depth buffer, not Isaac's ray-caster. |

## B3 — ice

> **Retracted, 2026-09-14.** In Isaac training the patches spawned a median 72 m
> from the robots, and 4.4% could have reached one in an episode
> (`scripts/bench/ice_placement_probe.py`, FINDINGS *Ice*). These are policies
> trained on bumpy ground, dropped onto ice in MuJoCo. The clip stays as a render
> of `DepthRlController`; it is not evidence about ice.

The strongest result in the repo, and until now the only one with no picture.
Depth beats blind by **10.6%** (final curriculum level 1.519 against 1.374) on
friction patches that are **flush with the floor** — ray-cast-verified, so the
sensor cannot see them. Colouring them so a camera *could* see them changes
nothing (1.394).

It had no clip because `render_multi` could not drive a depth-conditioned
policy: upstream's controller assembles the observation from raw pieces and
knows nothing about depth, so a 301-wide network was handed 45 numbers and the
run raised before drawing a frame. `DepthRlController` appends the term where
Isaac puts it — after `prev_actions` — and the clip is the same harness that
produced the numbers.

| | |
|---|---|
| <img src="gifs/ice_pair.gif" width="640"> | **`ice_pair`** — green blind, red depth, identical command. Along the bottom is the depth robot's own egocentric view and a scrolling waterfall of the centre column. **14 MB.** |


## The cooperative lift

| | |
|---|---|
| <img src="gifs/squat_pick.gif" width="420"> | **`squat_pick`** — scripted joint interpolation, not a policy. The control for every failure below: the pose exists and is reachable. |
| <img src="gifs/carry_3.gif" width="420"> | **`carry_3`** — left the scripted check, right the learned thing. The gap is the finding. |
| <img src="gifs/carry_2.gif" width="420"> | **`carry_2`** — one pair closing on the cube. |
| <img src="gifs/carry_4.gif" width="420"> | **`carry_4`** — one pair against two. The pinch forms, the lift does not. Each is the seed that stays upright longest out of twelve. |

## Robot POV

Colour, raw 64×64 depth, and the 8×8 the network actually receives.

| | |
|---|---|
| <img src="gifs/carry_cube_pov.gif" width="420"> | **`carry_cube_pov`** — the best rollout in the project. 7.8 cm of lift, hands in the pinch gate 98% of the time, 12 s without a fall. Also a controlled collapse: the pair drops 41 cm before contact. |
| <img src="gifs/carry_ball_native_pov.gif" width="420"> | **`carry_ball_native_pov`** — the 21 cm arm cross-checked in the other engine. Falls at 0.72 s having never touched the ball. |
| <img src="gifs/carry_ball_transfer_pov.gif" width="420"> | **`carry_ball_transfer_pov`** — the same arm, transfer condition. |
| <img src="gifs/carry_ladder_pov.gif" width="420"> | **`carry_ladder_pov`** — the plank. 18 seeds at 0.0 cm, closest approach 39 cm. Contact points are further apart than shoulders that cannot adduct past 36 cm can span. |

## Vision made it worse

| | |
|---|---|
| <img src="gifs/carry_vision_swap_2.gif" width="420"> | **`carry_vision_swap_2`** — left blind, right depth *replacing* the object pose. The red outline is a fall, held for the rest of the clip. |
| <img src="gifs/carry_vision_swap_3.gif" width="420"> | **`carry_vision_swap_3`** — three pairs. |
| <img src="gifs/carry_vision_both_4.gif" width="420"> | **`carry_vision_both_4`** — depth *added alongside* the pose. The cube is plainly visible in every depth pane. They are not failing to see it. |
| <img src="gifs/carry_vision_both_2.gif" width="420"> | **`carry_vision_both_2`** — two robots. |
| <img src="gifs/carry_vision_swap_4.gif" width="420"> | **`carry_vision_swap_4`** — unused anywhere. |
| <img src="gifs/carry_vision_both_3.gif" width="420"> | **`carry_vision_both_3`** — unused anywhere. |

## The occlusion seed, cross-checked

Finding 7 is that the only cube arm which ever lifted is a single seed that did
not replicate. Replayed in the other engine it does worse than that: **the arm
that lifted is the one that falls.**

| | closest pinch | peak lift | in-pinch | fell |
|---|---|---|---|---|
| **s0** — reached 13 cm in Isaac | 0.251 m | 0.0 cm | **0%** | **yes, at 0.8 s** |
| **s1** — flat at the 4 cm floor in Isaac | 0.167 m | 0.0 cm | **96%** | no |

The seed that looked best in training never forms a pinch and goes down inside a
second; the seed that looked like a failure stays up and holds the pinch for
almost the whole episode. Same pattern as the ball arm in §1 — a result that
exists in one engine and not the other.

| | |
|---|---|
| <img src="gifs/occlusion_s0_lifted_pov.gif" width="420"> | **`occlusion_s0_lifted_pov`** — the 13 cm seed. Every pair down at 0.8 s. |
| <img src="gifs/occlusion_s1_flat_pov.gif" width="420"> | **`occlusion_s1_flat_pov`** — the flat seed. Upright, pinch held 96% of the episode, 0.0 cm of lift. |


## Isaac Sim

The only two clips in this repo that Isaac rendered, and the first frames it has
ever produced here. PhysX/RTX, 1280×720, cropped to the subject and denoised —
the path-traced floor grain makes an uncropped GIF 25 MB.

They are unflattering and that is the result. The grey box is the shelf, the
blue box the cube, and the small orange shapes are the robots — **lying down**.
Success is 0 on this task either way; the gripper arm's contribution is that it
stays alive for 427 steps against 8 while doing it.

| | |
|---|---|
| <img src="gifs/isaac/cubetoshelf_gripper.gif" width="420"> | **`isaac/cubetoshelf_gripper`** — `TaskV2-BHL-CubeToShelfGrip-Blind-v0`, the 24-DoF gripper asset. |
| <img src="gifs/isaac/cubetoshelf_welded.gif" width="420"> | **`isaac/cubetoshelf_welded`** — `TaskV2-BHL-CubeToShelf-Blind-v0`, the shipped welded-hand asset. |

### Terrain sensing (the B5 "maze" rung)

| | |
|---|---|
| <img src="gifs/isaac/maze_stereo_fixed.gif" width="560"> | **`isaac/maze_stereo_fixed`** — the stereo policy fed 16×16 an eye, retrained with its cameras pointed down (`mazefix-stereo-s0`, terrain level 0.88; the same arm scored 0.02 with the old pose). Right, the left eye's 64×64 depth each frame, bright near and black at the 6 m limit. **Top: the pose every B5 run on Isaac Lab 3.0 had before `3f7b679`** — the quaternion read in the wrong order, 20° up and upside down, with ground only in a strip along the top. **Bottom: the corrected pose** this policy trained on. Both eyes ride the same robot in the same run (`21329137`). Cropped and lightly blurred, because the bumpy terrain is per-pixel noise GIF cannot compress. **9.5 MB.** |
| <img src="gifs/isaac/terrain_sensors.gif" width="560"> | **`isaac/terrain_sensors`** — the four seed-0 policies, each followed by its own camera: blind, lidar, stereo fed at 16×16 an eye, and stereo pooled to 4×4. Labels are each clip's own terrain level (3-seed means are in FINDINGS). **Watch it for what the policies look like, not for the result**: over twelve seconds all four walk, because the difference is how far the terrain curriculum promoted them, which one clip on one patch cannot show. There are no walls in shot because there never were any near the robots — see FINDINGS. **Both stereo policies here trained with their cameras looking 20° up, upside down** — `isaac/maze_stereo_fixed` above shows the difference (Isaac Lab 3.0 reads the pose quaternion in a different order; FINDINGS, *Terrain sensing*), so those two panels show what the policies did, not what stereo sees. Denoised and cropped from 640×360 path-traced frames. **10 MB.** |

**How the render got fixed.** The first attempt (`21247917`) filmed corridors
and no robot. The viewport, which `RecordVideo` records, draws an articulation
at its stale USD pose when fabric is on — here the env's grid origin, tens of
metres from the terrain patch the robot walks on. A camera *sensor* draws the
body where physics has it (`21299608`), so `train_play` now records through
one; `docs/ISAAC_RENDER.md` §10 has the recipe.

### Cloth sorting, free base

| | |
|---|---|
| <img src="gifs/isaac/cloth_sort_free_base_jacket.gif" width="440"> | **`isaac/cloth_sort_free_base_jacket`** — `ClothSort-BHL-Rigid-Oracle-v0` with `--garment jacket`, job `21329392`. **The robot is free-standing**: its legs hold a knee-bent stance with gravity feedforward and IMU ankle feedback (`bhl_robust.cloth.balance`), and the table stands 5.9 cm higher to match. The hand sweeps the dark jacket proxy off the table's back edge into the grey jackets basket. On this layout the free base sorts jackets 8/8 and socks 7/8, but shirts only 1/8 (`21329260`–`262`). A rigid box, not cloth. Frames brightened because the jacket is near-black. **3.8 MB.** |

### Cloth sorting, fixed-base diagnostic

| | |
|---|---|
| <img src="gifs/isaac/cloth_sort_fixed_base.gif" width="440"> | **`isaac/cloth_sort_fixed_base`** — `ClothSort-BHL-RigidFixedBase-Oracle-v0`, job `21317391`. The scripted sweep: the right hand descends at an anchor beside the table, glides around the red shirt proxy, and sweeps it off the front edge into the red shirts basket. **The robot's root is pinned and the shirt is a rigid 10 × 8 cm box.** With the root free this robot falls backward within a second even with its arm still (`21317172`), so this clip shows the manipulation half alone. On this layout shirt, sock and jacket sort 32 of 32 (`21317388`–`390`). The episode ends when the shirt's centre enters the basket, so the clip stops as it drops. Camera sensor, 640×360 path-traced, cropped and denoised. **3.9 MB.** |

### On the corrected *rotation* — and a spawn that is still wrong

**Retraction.** These were published as "the first clips of this robot spawning
upright". They are not. Watch the first second: the robots are not visible at
all, because they spawn **entirely underneath the floor** and are extruded
upward by the physics solver over the following twenty frames.

Measured at reset with the corrected rotation: **27 of 27 bodies below z = 0**,
the lowest at -0.85 m. The rotation fix made the burial *worse* -- it was 19 of
27 when the robot was lying on its side, because standing it upright put its
full height below a root that sits at -0.07.

The numbers below are real and were produced in that state, which is what makes
them hard to interpret rather than simply good.

| | start | tail |
|---|---|---|
| `Curriculum/base_height` | +0.684 | **+0.773** — rising, not collapsing |
| `Curriculum/lift_height` | 0.0400 | **0.0478** — off the floor for the first time |
| `Episode_Reward/lifting_object` | 0.0003 | **+1.212** |
| `Episode_Termination/time_out` | 0.020 | **0.331** |
| mean episode length | ~8 (before the fix) | **258.9** |

Every earlier arm in this project sat on `lift_height = 0.0400` exactly and
never promoted. This one does — but it does so while being pushed out of the
ground for the first fifth of every episode, so what the promotion is measuring
is not yet established. **Task success is still 0.**

Two thirds of episodes still end in a fall, so several seconds of both clips are
robots tangled around a tipped cube.

| | |
|---|---|
| <img src="gifs/isaac/cubetoshelf_upright_welded.gif" width="420"> | **`isaac/cubetoshelf_upright_welded`** — welded hands, corrected spawn. Mean episode length 258.9, reward +10.73. |
| <img src="gifs/isaac/cubetoshelf_upright_gripper.gif" width="420"> | **`isaac/cubetoshelf_upright_gripper`** — the 24-DoF gripper on the same spawn. |


Framing is still wide — the viewer camera is set from `BHL_VIEW_EYE` /
`BHL_VIEW_LOOKAT` (colon-separated; both `sbatch --export` and Apptainer's
`--env` split on commas) and has not been tuned. The source mp4s are 77–88 MB
and stay out of the repo.
