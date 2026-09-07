# B5 — maze navigation with a lidar and a stereo pair

## The parts

| | modelled as | why |
|---|---|---|
| **RPLIDAR C1** | `LidarPatternCfg`, 500 rays, 0.72°, 12 m, 10 Hz | ray-cast, so it costs what the depth rung measured — 1.6% of throughput — and works on both stacks |
| **MMlove global-shutter stereo** | two `RayCasterCameraCfg`, 60 mm baseline, 64×64 → pooled 16×16 | the baseline is the part that makes stereo depth possible |
| **7″ screen** | **not modelled** | it is an output device. It does not enter the observation, and its absence is deliberate rather than forgotten |

A note on "global shutter": simulation has no rolling shutter, so it cannot show
the advantage. What it can show is the baseline geometry. The honest statement
is that a rolling-shutter part would be *worse on hardware* in a way this sim
cannot reproduce — this robot's own clips show the base oscillating several
centimetres a step, which is exactly the motion that skews a rolling frame.

## Built on the half that works

On `BipedBumpyEnvCfg`, the locomotion rung — not the v2 manipulation tasks.
Locomotion is where this project has real effects (push, terrain, arms). Every
manipulation task here has scored zero. A new sensor capability belongs on the
half that walks.

## Four arms, and why each hazard is chosen

| arm | obs width |
|---|---|
| blind | 45 |
| lidar | 81 (45 + 36 sector minima) |
| stereo | 557 (45 + 2 × 256) |
| both | 593 |

Each hazard is visible to exactly one sensor, which is what makes the
comparison mean anything:

* **floor obstacles at 0.10 m** — under the 0.34 m lidar plane, inside the
  cameras' down-pitched cone. Stereo's hazard.
* **corners and walls** — at lidar height, outside a 64-pixel forward cone
  until the turn is already made. Lidar's.
* **arrow plates** — geometric, not coloured, so they read as depth structure
  to stereo and as one more wall to lidar. Stereo's, and the reason the maze
  cannot be solved by wall-following alone.

Deliberately geometric rather than textured: 5.1's RTX renderer segfaults on
this cluster, and a marker only a colour camera could read would make the task
unrunnable there.

The blind arm is the control. Without it a lidar number is a fact about the
maze, not about lidar — the mistake G-B2 made when it measured its own
iteration budget and called it a terrain verdict.

## Button pressing: in

A wall plate, pressed with a hand. Non-prehensile, one contact point, no fingers
required — the same reasoning that made the cloth task a sweeping task rather
than a picking one.

## Rubik's cube: out, and this is not a scheduling problem

This robot has **one revolute DoF per hand**, a single finger closing against
the palm. Solving a Rubik's cube requires holding the cube in one hand while
rotating a face with the other, independently actuated fingers, and force
control fine enough not to crush or drop it.

This project already established the weaker claim and it settles the stronger
one: *"A humanoid with no fingers cannot pick up a sock. Not with a better
policy, not with more iterations."* A cube is rigid, so it is easier to grip
than fabric — but face rotation needs in-hand manipulation, which is
categorically further from one DoF than picking is.

Adding it would mean either a different robot or a scripted animation labelled
as a result. Both are worse than saying so.

## Status

`Velocity-BHL-Maze-{Blind,Lidar,Stereo,Both}-v0` are registered and the smoke
passes 4/4: all four construct, reset and step, with the widths above and stereo
returning real depth (0.61–6.00 m, 100% finite). Lidar reads 30% finite, which
is what a horizontal 360° scan in a corridor should look like — most rays hit
nothing and clamp to "clear".

Not yet trained. The GPU cap here is 9 concurrent, and the 18 manipulation
re-runs on the corrected spawn have it. Training these four is the next
allocation, not a parallel one.
