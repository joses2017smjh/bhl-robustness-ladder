"""A Gymnasium environment for learning maze navigation on the gait's command interface.

The point of this gym is transfer: everything the policy sees and does here
can be reproduced on the physics robot by `scripts/bench/maze_explore.py`.

* action    (vx, wz) in [-1, 1]^2, scaled to vx in [0, 0.35] m/s and wz in
            [-1, 1] rad/s -- the biped gait's command interface; there is no vy.
* dynamics  a unicycle with a first-order lag on both channels, a yaw-rate
            gain, a heading drift while walking and a command latency, all
            randomized per episode around the values measured on the gait
            (gain 1.17, ~7 deg of drift over 2 m).
* obs       "lidar": 36 sector minima of a 108-ray 12 m scan over the maze
            walls, scaled by the range (what `team_sensors` returns);
            "map": an egocentric 24x24 crop (0.2 m cells, 4.8 m window) of a
            log-odds occupancy map built from the same rays, three channels:
            occupied, free, unknown -- rotated so forward is up;
            "goal": goal distance / 10 (clipped), sin and cos of the goal
            bearing in the body frame, and the previous action.
* reward    potential-based progress along the true shortest path distance
            (dense, cheap, and not gameable by circling), a goal bonus, a step
            cost, and a collision penalty; collision ends the episode.
* mazes     `random_maze.generate` on a curriculum of sizes; training seeds are
            < 10_000, evaluation seeds >= 10_000, so no evaluation maze was seen.

Oracle inputs, stated once: the pose that anchors the map and the goal
coordinate. The maze walls are only ever seen through the rays.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from bhl_robust.eval.random_maze import CELL, WALL_T, Maze, generate

LIDAR_RAYS = 108
LIDAR_SECTORS = 36
LIDAR_RANGE = 12.0
MAP_RES = 0.2
MAP_CROP = 24              # cells per side of the egocentric crop
DT = 0.04                  # the gait's policy period
V_MAX, W_MAX = 0.35, 1.0
ROBOT_RADIUS = 0.22        # measured collision half-width of the biped


@dataclass
class Dynamics:
    """Per-episode proxy of the gait: gain, lag, drift, latency."""
    w_gain: float = 1.17
    v_gain: float = 1.0
    tau: float = 0.25          # s, first-order lag on both channels
    drift: float = -0.02       # rad/s of heading drift while walking at full speed
    latency: int = 1           # command steps of delay
    v_noise: float = 0.02
    w_noise: float = 0.05


def sample_dynamics(rng: np.random.Generator, randomize: bool = True) -> Dynamics:
    if not randomize:
        return Dynamics()
    return Dynamics(w_gain=float(rng.uniform(0.9, 1.4)), v_gain=float(rng.uniform(0.85, 1.05)),
                    tau=float(rng.uniform(0.15, 0.45)), drift=float(rng.uniform(-0.06, 0.06)),
                    latency=int(rng.integers(0, 3)), v_noise=float(rng.uniform(0.0, 0.04)),
                    w_noise=float(rng.uniform(0.0, 0.10)))


# --------------------------------------------------------------- ray casting
def wall_boxes(maze: Maze) -> np.ndarray:
    """(N, 4) array of axis-aligned boxes: xmin, xmax, ymin, ymax."""
    out = []
    for (cx, cy), (hx, hy) in maze.wall_segments():
        out.append((cx - hx, cx + hx, cy - hy, cy + hy))
    return np.asarray(out, dtype=np.float64)


def cast_rays(boxes: np.ndarray, origin, angles: np.ndarray, max_range: float) -> np.ndarray:
    """Slab-method ray/AABB intersection, vectorised over rays and boxes."""
    ox, oy = origin
    dx, dy = np.cos(angles), np.sin(angles)
    with np.errstate(divide="ignore", invalid="ignore"):
        inv_dx = np.where(np.abs(dx) < 1e-12, np.inf, 1.0 / dx)[:, None]
        inv_dy = np.where(np.abs(dy) < 1e-12, np.inf, 1.0 / dy)[:, None]
        tx1 = (boxes[None, :, 0] - ox) * inv_dx
        tx2 = (boxes[None, :, 1] - ox) * inv_dx
        ty1 = (boxes[None, :, 2] - oy) * inv_dy
        ty2 = (boxes[None, :, 3] - oy) * inv_dy
        tmin = np.maximum(np.minimum(tx1, tx2), np.minimum(ty1, ty2))
        tmax = np.minimum(np.maximum(tx1, tx2), np.maximum(ty1, ty2))
    hit = (tmax >= np.maximum(tmin, 0.0))
    t = np.where(hit, np.where(tmin > 0, tmin, tmax), np.inf)
    d = t.min(axis=1)
    return np.minimum(np.where(np.isfinite(d), d, max_range), max_range)


def point_in_wall(boxes: np.ndarray, x: float, y: float, radius: float) -> bool:
    return bool(np.any((x + radius > boxes[:, 0]) & (x - radius < boxes[:, 1]) & (y + radius > boxes[:, 2]) & (y - radius < boxes[:, 3])))


# ------------------------------------------------------------ shortest path
def geodesic_field(maze: Maze) -> dict:
    """Cell -> number of hops to the goal (BFS over open edges)."""
    dist = {maze.goal: 0}
    queue = [maze.goal]
    while queue:
        c = queue.pop(0)
        for d in maze.neighbours(c):
            if d not in dist:
                dist[d] = dist[c] + 1
                queue.append(d)
    return dist


def geodesic_distance(maze: Maze, field: dict, x: float, y: float) -> float:
    """Metres to the goal along the maze: hops from the current cell plus the
    straight-line leg from the position to that cell's exit, in cell units."""
    i = int(round(x / CELL)); j = int(round(y / CELL))
    i = min(max(i, 0), maze.n - 1); j = min(max(j, 0), maze.m - 1)
    c = (i, j)
    if c == maze.goal:
        gx, gy = maze.centre(maze.goal)
        return math.hypot(x - gx, y - gy)
    hops = field.get(c)
    if hops is None:
        return 1e3
    # the next cell toward the goal, and the distance to its centre
    nxt = min(maze.neighbours(c), key=lambda d: field.get(d, 1e9))
    nx, ny = maze.centre(nxt)
    return math.hypot(x - nx, y - ny) + (field[nxt]) * CELL


# ---------------------------------------------------------------- ego map
class EgoMap:
    """Log-odds occupancy at MAP_RES, integrated from raw rays, cropped
    egocentrically (forward = up). Used by the gym and, unchanged, by the
    physics runner so the policy sees the same thing in both."""

    L_FREE, L_OCC, L_MIN, L_MAX = -0.2, 0.85, -4.0, 4.0

    def __init__(self, bounds, margin: float = 1.0, res: float = MAP_RES, max_range: float = LIDAR_RANGE):
        xmin, xmax, ymin, ymax = bounds
        self.res, self.max_range = float(res), float(max_range)
        self.x0, self.y0 = xmin - margin, ymin - margin
        self.nx = int(math.ceil((xmax - xmin + 2 * margin) / res)); self.ny = int(math.ceil((ymax - ymin + 2 * margin) / res))
        self.l = np.zeros((self.nx, self.ny), dtype=np.float32)
        self._ks = np.arange(0.0, self.max_range, res * 0.5)

    def update(self, x: float, y: float, yaw: float, angles: np.ndarray, ranges: np.ndarray) -> None:
        ang = yaw + angles
        ca, sa = np.cos(ang)[:, None], np.sin(ang)[:, None]
        ks = self._ks[None, :]
        free_mask = ks < (ranges[:, None] - 0.1)
        px = x + ks * ca; py = y + ks * sa
        ii = np.floor((px - self.x0) / self.res).astype(int); jj = np.floor((py - self.y0) / self.res).astype(int)
        ok = free_mask & (ii >= 0) & (ii < self.nx) & (jj >= 0) & (jj < self.ny)
        fi, fj = ii[ok], jj[ok]
        # each cell at most once per scan: unique pairs, then one free decrement
        if fi.size:
            flat = np.unique(fi * self.ny + fj)
            self.l.ravel()[flat] = np.maximum(self.L_MIN, self.l.ravel()[flat] + self.L_FREE)
        hit = ranges < self.max_range - 1e-3
        if hit.any():
            hx = x + ranges[hit] * ca[hit, 0]; hy = y + ranges[hit] * sa[hit, 0]
            hi = np.floor((hx - self.x0) / self.res).astype(int); hj = np.floor((hy - self.y0) / self.res).astype(int)
            okh = (hi >= 0) & (hi < self.nx) & (hj >= 0) & (hj < self.ny)
            flat = np.unique(hi[okh] * self.ny + hj[okh])
            self.l.ravel()[flat] = np.minimum(self.L_MAX, self.l.ravel()[flat] + self.L_OCC)

    def crop(self, x: float, y: float, yaw: float, size: int = MAP_CROP) -> np.ndarray:
        """(3, size, size) egocentric crop: occupied, free, unknown; forward = up, left = left."""
        half = size // 2
        # Offsets at whole cells (2.4, 2.2, ..., -2.2 m for 24 cells): a robot standing on a
        # cell centre then samples cell centres, never the edges where float rounding
        # would drop the sample into a neighbour. One more row ahead than behind.
        u = (half - np.arange(size)) * self.res
        v = (half - np.arange(size)) * self.res
        U, V = np.meshgrid(u, v, indexing="ij")
        c, s = math.cos(yaw), math.sin(yaw)
        wx = x + U * c - V * s; wy = y + U * s + V * c
        ii = np.floor((wx - self.x0) / self.res).astype(int); jj = np.floor((wy - self.y0) / self.res).astype(int)
        ok = (ii >= 0) & (ii < self.nx) & (jj >= 0) & (jj < self.ny)
        l = np.zeros((size, size), dtype=np.float32)
        l[ok] = self.l[ii[ok], jj[ok]]
        p = 1.0 / (1.0 + np.exp(-l))
        occ = (p > 0.65).astype(np.float32); free = (p < 0.35).astype(np.float32)
        unk = 1.0 - occ - free
        unk[~ok] = 1.0; occ[~ok] = 0.0; free[~ok] = 0.0
        return np.stack([occ, free, unk])

    def known_fraction(self) -> float:
        p = 1.0 / (1.0 + np.exp(-self.l))
        return float(np.mean((p > 0.65) | (p < 0.35)))


def sector_minima(ranges: np.ndarray, sectors: int = LIDAR_SECTORS, clip: float = LIDAR_RANGE) -> np.ndarray:
    n = ranges.shape[0] - (ranges.shape[0] % sectors)
    return np.minimum(ranges[:n].reshape(sectors, -1).min(axis=1), clip) / clip


def goal_features(x: float, y: float, yaw: float, goal_xy, prev_action) -> np.ndarray:
    dx, dy = goal_xy[0] - x, goal_xy[1] - y
    dist = math.hypot(dx, dy)
    bearing = math.atan2(dy, dx) - yaw
    return np.array([min(dist / 10.0, 1.0), math.sin(bearing), math.cos(bearing), float(prev_action[0]), float(prev_action[1])], dtype=np.float32)


# ------------------------------------------------------------------ the env
class MazeNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sizes=((3, 3), (4, 4), (5, 5), (6, 6)), extra_openings: int = 1, max_steps: int = 1500,
                 seed_base: int = 0, seed_span: int = 10_000, randomize_dynamics: bool = True, size_weights=None):
        super().__init__()
        self.sizes = tuple(tuple(s) for s in sizes)
        self.size_weights = None if size_weights is None else np.asarray(size_weights, dtype=float) / np.sum(size_weights)
        self.extra_openings = extra_openings
        self.max_steps = max_steps
        self.seed_base, self.seed_span = seed_base, seed_span
        self.randomize_dynamics = randomize_dynamics
        self.angles = np.linspace(-np.pi, np.pi, LIDAR_RAYS, endpoint=False)
        self.observation_space = gym.spaces.Dict({
            "lidar": gym.spaces.Box(0.0, 1.0, (LIDAR_SECTORS,), np.float32),
            "map": gym.spaces.Box(0.0, 1.0, (3, MAP_CROP, MAP_CROP), np.float32),
            "goal": gym.spaces.Box(-1.0, 1.0, (5,), np.float32),
        })
        self.action_space = gym.spaces.Box(-1.0, 1.0, (2,), np.float32)
        self._rng = np.random.default_rng(0)
        self.episode_seed = None

    # ----------------------------------------------------------- helpers
    def _sample_size(self):
        k = int(self._rng.choice(len(self.sizes), p=self.size_weights)) if self.size_weights is not None else int(self._rng.integers(len(self.sizes)))
        return self.sizes[k]

    def set_sizes(self, sizes, size_weights=None):
        """Curriculum hook: change the maze sizes drawn at the next reset."""
        self.sizes = tuple(tuple(s) for s in sizes)
        self.size_weights = None if size_weights is None else np.asarray(size_weights, dtype=float) / np.sum(size_weights)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        n, m = self._sample_size()
        self.episode_seed = int(self.seed_base + self._rng.integers(self.seed_span))
        self.maze = generate(n, m, self.episode_seed, extra_openings=self.extra_openings)
        self.boxes = wall_boxes(self.maze)
        self.field = geodesic_field(self.maze)
        self.dyn = sample_dynamics(self._rng, self.randomize_dynamics)
        sx, sy = self.maze.centre(self.maze.start)
        self.x, self.y = sx + self._rng.normal(0, 0.03), sy + self._rng.normal(0, 0.03)
        self.yaw = float(self._rng.uniform(-np.pi, np.pi))
        self.v = self.w = 0.0
        self.queue = [np.zeros(2) for _ in range(self.dyn.latency)]
        self.prev_action = np.zeros(2, dtype=np.float32)
        self.t = 0
        self.emap = EgoMap(self.maze.bounds())
        self.goal_xy = self.maze.centre(self.maze.goal)
        self.potential = geodesic_distance(self.maze, self.field, self.x, self.y)
        self._scan()
        return self._obs(), {"maze_seed": self.episode_seed, "size": (n, m)}

    def _scan(self):
        self.ranges = cast_rays(self.boxes, (self.x, self.y), self.yaw + self.angles, LIDAR_RANGE)
        self.emap.update(self.x, self.y, self.yaw, self.angles, self.ranges)

    def _obs(self):
        return {"lidar": sector_minima(self.ranges).astype(np.float32), "map": self.emap.crop(self.x, self.y, self.yaw),
                "goal": goal_features(self.x, self.y, self.yaw, self.goal_xy, self.prev_action)}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        self.prev_action = a.copy()
        v_cmd = (a[0] + 1.0) * 0.5 * V_MAX
        w_cmd = a[1] * W_MAX
        self.queue.append(np.array([v_cmd, w_cmd]))
        v_cmd, w_cmd = self.queue.pop(0)
        d = self.dyn
        alpha = DT / max(d.tau, DT)
        self.v += alpha * (d.v_gain * v_cmd - self.v)
        self.w += alpha * (d.w_gain * w_cmd - self.w)
        v = self.v + self._rng.normal(0, d.v_noise)
        w = self.w + self._rng.normal(0, d.w_noise) + d.drift * (self.v / V_MAX)
        self.yaw = (self.yaw + w * DT + math.pi) % (2 * math.pi) - math.pi
        nx, ny = self.x + v * DT * math.cos(self.yaw), self.y + v * DT * math.sin(self.yaw)
        self.t += 1
        terminated, truncated = False, False
        reward = -0.01
        if point_in_wall(self.boxes, nx, ny, ROBOT_RADIUS):
            reward -= 2.0
            terminated = True
            info = {"outcome": "collision"}
        else:
            self.x, self.y = nx, ny
            new_pot = geodesic_distance(self.maze, self.field, self.x, self.y)
            reward += 1.0 * (self.potential - new_pot)
            self.potential = new_pot
            info = {"outcome": None}
            if math.hypot(self.goal_xy[0] - self.x, self.goal_xy[1] - self.y) < 0.30:
                reward += 5.0
                terminated = True
                info = {"outcome": "goal"}
            elif self.t >= self.max_steps:
                truncated = True
                info = {"outcome": "time_out"}
        self._scan()
        info["maze_seed"] = self.episode_seed
        return self._obs(), float(reward), terminated, truncated, info
