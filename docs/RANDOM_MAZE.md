# Randomized mazes, mapped by the robot's own lidar

**Ask (2026-09-24):** mazes that change every run so a clip cannot be a
memorised route, the sensor panels of the three-panel clips, a view of how the
robot mapped the maze along its trajectory, and a robot that turns and then
walks forward instead of sliding sideways.

## What runs

| Piece | What it is | Learned / scripted / oracle |
|---|---|---|
| Maze | recursive-backtracker maze on an n×m grid of 1.4 m cells (1.32 m clear corridors, 1.1 m walls) plus 1–2 random extra openings so routes have loops; **a new maze per seed** (`bhl_robust/eval/random_maze.py`) | generated; unknown to the planner |
| Gait | the biped `dr-default-s0` PPO checkpoint, frozen, driven by body-frame (vx, 0, wz) commands | **learned** |
| Sensing | the `team_sensors` rig on the biped's mounts (lidar +0.34 m, stereo pair +0.30 m, 20° down): 108-ray 12 m lidar and an 8×8 paired ray depth at 10 Hz, plus the existing speed brake | actual simulated sensors |
| Map | log-odds occupancy grid, 0.10 m cells, integrated from the raw lidar returns | built online |
| Localization | the simulator's true pose | **oracle** (says so on every frame) |
| Planner | A* on the inflated map (0.30 m), **unknown = free**, replanned every 0.4 s; greedy line-of-sight simplification to corner waypoints; the last waypoint is the exact goal coordinate | scripted; goal coordinate is oracle |
| Motion | turn in place toward the next waypoint (|heading error| > 0.40 rad, 0.6 rad/s), walk forward once within 0.15 rad (0.30 m/s, yaw correction ≤ 0.4 rad/s); **vy is always 0** | scripted |

Success: goal disc reached within the time limit with no fall. *Clean*: also
no wall contact at any physics substep. The table over seeds is the evidence;
the clip is one seed at the median clean completion time, never the fastest.

## Why the biped, and why earlier clips walk sideways

Measured on the login node (flat world, 6 s of a constant command after 1 s of
standing; yaw change and displacement of the base):

| Gait | (0, 0, 0.6 rad/s) | (0, 0, 1.0) | (0.2, 0, 0.8) | (0.35, 0, 0) |
|---|---|---|---|---|
| humanoid `arms-dr1.0-s0` (22 DoF, the Mission 7 / inspection gait) | **0.2°**, 0.06 m | 4.2° | 1.1° | −18° drift over 2.0 m |
| biped `dr-default-s0` (12 DoF) | **239°**, 0.07 m | 393° | 318°, 0.14 m | 7° over 2.1 m |
| biped `push-adaptive-s0` | 6° | 7° | 199°, 0.57 m | −54° over 1.9 m |
| biped `terrain-bumpy-s0` | 7° | 9° | 311°, 0.19 m | 13° over 2.1 m |

The humanoid checkpoint ignores the yaw-rate command. That is why every
Mission 7 and inspection clip translates holonomically with a fixed facing:
the route controllers never had a turn available to them. `dr-default-s0`
tracks a yaw-rate command at about 1.2× the commanded rate and walks
straight, so this mission uses it. The biped has no arms and no vision arms
trained on it.

## Results

Filled from `results/maze-explore-20260924/*/summary.json` when the table job
lands (see `SLURM_JOBS.md`).

## Reproduce

```bash
# multi-seed table, CPU MuJoCo, no render
python scripts/bench/maze_explore.py --upstream external/Berkeley-Humanoid-Lite --cache-dir /tmp/mz \
    --seeds 12 --n 6 --m 6 --extra-openings 1 --time-limit 180 --out-dir results/maze-explore-20260924/hard-6x6
# three-panel clip of one seed (needs EGL: MUJOCO_GL=egl on a GPU node)
python scripts/bench/maze_explore.py ... --seeds 1 --seed-start K --render --gif docs/gifs/random-maze-explore.gif
```
