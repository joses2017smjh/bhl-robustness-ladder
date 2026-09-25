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

Job `21412035` (CPU MuJoCo, 9:52 for all 24 episodes), one maze per seed, seeds 0–11, initial heading +x:

| configuration | seeds | reached goal | clean (no wall contact) | falls | completion, median (range) | route cells, median (range) | path walked, median | turns, median (range) | median-time seed |
|---|---|---|---|---|---|---|---|---|---|
| base-5x5 | 12 | **12/12** | **12/12** | 0 | 37.2 s (28.9–70.2) | 9 (9–19) | 12.4 m | 5 (1–22) | seed 5 |
| hard-6x6 | 12 | **12/12** | **12/12** | 0 | 73.1 s (56.8–116.4) | 17 (11–31) | 22.6 m | 15 (9–38) | seed 1 |

Every episode reached its goal with no fall and no wall contact at any physics substep. The mapped fraction of the
grid at the goal was 0.43–0.78: the planner never needed the whole maze, only what the 12 m lidar had returned along
the way. "Turns" counts walk→turn transitions, i.e. every time the next waypoint needed a turn in place.

Predeclared rule applied: the harder configuration scored ≥ 10/12 clean, so the clip is its median-time seed
(seed 1: 73.1 s, 24.8 m walked for a 21-cell route, 23 turns, 172 replans), rendered by job `21412068` →
`docs/gifs/random-maze-explore.gif`. Evidence: `results/maze-explore-20260924/{base-5x5,hard-6x6}/seed*.json`
(per-step traces) and `summary.json`.

## Reproduce

```bash
# multi-seed table, CPU MuJoCo, no render
python scripts/bench/maze_explore.py --upstream external/Berkeley-Humanoid-Lite --cache-dir /tmp/mz \
    --seeds 12 --n 6 --m 6 --extra-openings 1 --time-limit 180 --out-dir results/maze-explore-20260924/hard-6x6
# three-panel clip of one seed (needs EGL: MUJOCO_GL=egl on a GPU node)
python scripts/bench/maze_explore.py ... --seeds 1 --seed-start K --render --gif docs/gifs/random-maze-explore.gif
```

## A learned policy on the same interface (NavGym)

`bhl_robust/navgym/env.py` is a Gymnasium environment whose action is exactly
the gait's command (vx, wz; no vy) and whose observation is what the physics
runner can reproduce: the 36 lidar sectors, a 3×24×24 egocentric crop of a
0.2 m log-odds map built from the same 108 rays, and goal distance/bearing.
The proxy dynamics are a unicycle with the gait's measured yaw-rate gain, a
first-order lag, walking drift and command latency, randomized per episode.
`scripts/bench/navgym_train.py` trains it with Stable-Baselines3 PPO on CPU
under a maze-size curriculum, evaluates on mazes with seeds ≥ 10 000 (never
trained on) and exports the deterministic actor to ONNX;
`scripts/bench/maze_explore.py --policy actor.onnx` then drives the physics
robot with it. Jobs `21412154` / `21412155`; the predeclared transfer test is in
`SLURM_JOBS.md`. Results land here when they exist.

