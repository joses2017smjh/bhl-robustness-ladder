# Three tasks that can actually be completed

The original lift tasks had no completion criterion. A policy could earn every
reward term and the episode would still end on a timeout, so "success" was never
a state the environment recognised — only a number that went up. Worse, the
payload sat on the floor, inside the reach of a robot that had already fallen
over, and nothing in the reward or the terminations referred to base height.
Both cube arms learned to drop ~41 cm in the first 0.2 s and hold a brace.

These three replace it. Each has a terminal success state, and each puts its
payload where reaching it requires standing up.

## The one number the redesign turns on

Measured by forward kinematics over the shoulder and elbow ranges, feet planted
(`slurm/inner/_reach.sh`, constants in `tasks/reach_band.py`):

| condition | hands can reach |
|---|---|
| standing, arms only | 0.404 – 0.610 m |
| collapsed 41 cm, arms only | −0.006 – **0.200 m** |

`GRASP_Z = 0.30 m` for all three payloads:

* **10 cm below** the standing arms-only floor, so the knees have to bend — the
  arms cannot get there on their own.
* **10 cm above** the collapsed ceiling, so falling over *loses* the payload
  rather than winning it.

That is a structural fix rather than a tuned one. It does not require a height
penalty large enough to outweigh a 15.0-weight lift bonus; it makes the collapse
stop paying at all.

---

## 1 · Cube to a shelf that barely fits it

| | |
|---|---|
| payload | 0.28 m cube, on a 0.16 m plinth, centre at 0.30 m |
| target | shelf slot **0.34 × 0.34 m** — 6 cm of total clearance on a 28 cm cube |
| deck height | 0.38 m, so the carried cube centre ends at 0.52 m (inside the 0.61 m standing ceiling) |
| carry | shelf 1.2 m from the spawn, so it has to be moved, not just raised |
| success | cube centre inside the slot volume, speed < 0.05 m/s, held 0.5 s |

The tight slot is the point: a 6 cm margin cannot be hit by throwing the cube at
the shelf, so the terminal condition rewards placement rather than displacement.

## 2 · Ball carried to a spot and thrown into a net

| | |
|---|---|
| payload | r = 0.18 m ball, on a 0.12 m tee, centre at 0.30 m |
| release zone | 0.6 m radius marker, 1.5 m from the spawn |
| net | 0.7 × 0.7 m mouth, rim at 0.60 m, 2.0 m beyond the release zone |
| success | ball centre inside the net volume with downward velocity |

Staged, because the throw is worthless before the carry works: reach → lift →
both robots inside the release zone → release with horizontal velocity toward
the net → net entry. This is the only one of the three with a ballistic phase,
and the only one where the pair must agree on a moment as well as a position.

## 3 · Plank leaned against a wall

| | |
|---|---|
| payload | 1.5 × 0.4 × 0.08 m plank, on two supports at 0.26 m, centre 0.30 m |
| target | wall 1.0 m away |
| success | lower end within 0.30 m of the wall base, upper end touching the wall above 0.5 m, plank 50–80° from horizontal, stable 0.5 s |

The plank is 1.5 m and one robot's hands span 0.355 m, so neither robot can
bracket it alone — this is the one task where the two-robot requirement is
geometric rather than a matter of load.

---

## Vision: three conditions per task

Every task runs blind, with depth, and with RGB. Nine cells.

| condition | sensor | stack |
|---|---|---|
| blind | privileged object pose | v51 |
| depth | `RayCasterCamera`, 64×64 → 8×8 pooled | v51 |
| **RGB** | `TiledCameraCfg`, RTX | **v60** |

RGB has never been run in this project — 5.1's RTX renderer segfaults on this
cluster, which is why depth came from Warp ray-casting. 6.0 renders (measured:
rgb std 24.4 across 1,902 unique colours), so the RGB arm runs on `BHL_STACK=v60`
and its numbers do **not** go in a table with the v51 numbers. Isaac Sim changed
major version underneath; that is a separate stack producing separate results.

## Gates, before any of it trains

The lesson from G-B2 is that a gate without a control measures its own budget.
Each task gets a *kinematic* gate first, which needs no training at all:

* **G-T1** — can a scripted standing robot place its hands on the payload at
  `GRASP_Z` without its base dropping below 0.35 m? If not, the height is wrong
  and every arm trained on it would measure the height.
* **G-T2** — is the success volume actually enterable? Drive the payload there
  kinematically and confirm the terminal condition fires.
* **G-T3** — is the collapsed posture genuinely excluded? Replay a current
  cube policy into the new scene; it should reach nothing.

No tier trains until its gate passes.
