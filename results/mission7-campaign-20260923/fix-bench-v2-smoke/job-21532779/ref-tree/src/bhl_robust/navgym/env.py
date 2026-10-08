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

Versions. ``MazeNavEnv(version=1)`` (the constructor default, so every existing
caller is unchanged) is the environment of results/navgym-20260924 and is kept
bit-for-bit. ``version=2`` (what ``navgym_train.py --env-version 2`` builds)
fixes the five faults found in the v1 post-mortem (2026-09-25):

* footprint   ROUND: a collision is a wall box closer than ROBOT_RADIUS to the
              robot centre (v1 tested an axis-aligned square of half-width 0.22,
              which clips corners up to 0.311 m away);
* time limit  route-scaled: ceil(3.0 * route / (V_MAX * DT)) + 400 steps, capped
              at 4500 (180 s, the physics 6x6 limit); route = the continuous
              geodesic from the start cell centre (v1: a flat 1500, shorter than
              many 6x6 routes need). Budget (2026-09-26, the scripted A* +
              turn-then-walk driver on the 48 held-out gate mazes under a flat
              4500): its (steps - 400) / full-speed route time has median 1.26
              (5x5) / 1.19 (6x6) and max 2.84 / 3.04; factor 3.0 keeps 24/24 of
              its 5x5 and 22/23 of its 6x6 successes inside the limit (2.5 would
              cut one more of each), while the median limit (2629 / 3636 steps,
              105 / 145 s) stays under the physics limits (150 / 180 s);
* reward      step cost -0.002, collision -5 (ends), goal +5 (ends), and
              potential-based shaping F = gamma * phi(s') - phi(s) with
              phi = -geodesic (Ng et al. 1999); gamma is a constructor argument.
              The v2 trainer passes gamma = 1.0 (reward = metres of progress, the v1
              form) because the learner's-gamma form pays for standing still (below). On the collision step the
              robot does not move and no shaping is paid (the crash forfeits the
              shaping stream as well as paying -5). Because phi <= 0,
              (1 - gamma) * geodesic > 0 is paid every step, so a robot standing
              still far from the goal collects a small positive reward: the
              undiscounted episode return is therefore NOT a progress measure in
              v2; use the held-out success/collision/time-out rates;
* potential   continuous: a 5 cm, 16-connected Dijkstra from the goal over free
              space dilated by ROBOT_RADIUS, computed once per maze (LRU cache),
              read by bilinear interpolation (v1's cell-hop potential jumped by
              0.3-2.4 m at doorways);
* obs         one extra key, "near": the 36 sector minima clipped at 2 m and
              scaled by 2 m (near-field resolution the 12 m "lidar" key lacks).
              ``build_obs`` is the single builder used by the gym and by the
              physics runner (maze_explore.py --policy), which picks the keys
              from the ONNX actor's input names so v1 actors still run.

Stall price (NavGym v4, 2026-09-28; opt-in, version 2 only). ``MazeNavEnv(version=2,
idle_cost=V4_IDLE_COST)`` (``navgym_train.py --idle-cost 0.008``) subtracts ``idle_cost``
on every step whose commanded forward action is ~0 (a[0] <= V4_IDLE_A0_MAX = -0.9, i.e.
v_cmd <= 5 % of V_MAX) while the straight-line goal distance at the start of the step is
> V4_IDLE_FAR_M = 0.5 m; the step's info then carries ``idle_charged``. With the -0.002
step cost an endless idle stall is worth (-0.002 - 0.008) / (1 - 0.998) = -5.0 at the
learner's gamma (SB3 bootstraps truncations), the crash penalty, instead of -1.0. The
default idle_cost 0.0 leaves every existing path unchanged.

Version 3 (NavGym v5, 2026-10-01; ``MazeNavEnv(version=3)``, ``navgym_train.py --env-version 3``).
Version 2's dynamics, rewards, time limit and (opt-in) stall price, PLUS exactly these four
changes, frozen before any v5 training (docs/SOLUTIONS_2026-10-01.md, N2); versions 1 and 2
never read any of it:

* visitation  a world-fixed recency grid (``VisitMap``) on the ego map's own 0.2 m lattice, r in
              [0, 1]: every step all cells decay r *= exp(-DT / 60 s), then the cells whose centres
              lie within ROBOT_RADIUS of the robot are set to 1 (decay first, so the footprint reads
              exactly 1; the start pose is marked at reset). Cropped and rotated exactly like the
              ego map and stacked onto it as a 4th channel: obs key "map_visit" (4, 24, 24) =
              occupied, free, unknown, visitation (replaces v2's "map");
* coarse map  obs key "map_coarse" (4, 24, 24): the same four channels, block-averaged over 3x3
              fine cells (0.6 m cells anchored at the fine grid's origin; cells past the grid edge
              count as unknown, unvisited), cropped 24 x 24 (14.4 m) in the same egocentric frame;
* yaw change  reward -= 0.005 * |a_yaw(t) - a_yaw(t-1)| on every step (actions in [-1, 1];
              a_yaw(-1) = 0, the reset's previous action); chosen, not tuned;
* brake       the deployment speed brake's lidar RANGE TERM only (team_sensors.brake_command's formula,
              as the frozen spec and the sysid_env.py port give it; not the whole physics brake, see
              below): v_cmd *= clip((clearance - 0.42) / 0.48, 0, 1), clearance = the minimum of
              the 10 Hz packet's sector minima whose centres lie within +-25 deg of travel (straight
              ahead: the gait has no vy). The packet is refreshed on team_sensors' schedule (captured at
              the top of a step when t * DT >= the next capture time, every 0.1 s: steps 0, 3, 5, 8,
              10, ...); captured at the step's start pose, it equals the last scan. The brake scales the
              forward command before the latency queue; the stall price still reads the actor's a[0].
              Not modelled (as in solutions-20260930/sysid/sysid_env.py): brake_command's stereo-depth
              tightening and IMU limit -- the gym has no depth camera or IMU. The depth term is not
              minor for a wall straight ahead: the biped's stereo centre sits 0.12 m ahead of the lidar,
              pitched 20 deg, so its horizontal pixel row reads ~0.94 x (d - 0.12) for a wall at lidar
              distance d, and the physics brake starts at d ~ 1.08 m and stops the robot at d ~ 0.57 m
              (here: 0.90 m and 0.42 m). Disclosed at review, 2026-10-02; the frozen spec is unchanged.

Observation keys "lidar", "near" and "goal" are v2's. ``build_obs(..., visit=VisitMap)`` builds the
v3 keys, so the physics runner builds them with the same code from the ONNX actor's input names.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import gymnasium as gym
import numpy as np

from collections import OrderedDict

from bhl_robust.eval.random_maze import CELL, WALL_T, Maze, generate

LIDAR_RAYS = 108
LIDAR_SECTORS = 36
LIDAR_RANGE = 12.0
MAP_RES = 0.2
MAP_CROP = 24              # cells per side of the egocentric crop
DT = 0.04                  # the gait's policy period
V_MAX, W_MAX = 0.35, 1.0
ROBOT_RADIUS = 0.22        # measured collision half-width of the biped

# ---- version 2 only (version 1 never reads these)
NEAR_RANGE = 2.0           # m, clip/scale of the near-field sector channel
V2_STEP_COST = -0.002
V2_COLLISION = -5.0
V2_GOAL = 5.0
V2_GAMMA = 0.998           # default shaping gamma (the trainer passes PPO's gamma)
V2_TIME_FACTOR = 3.0       # x the full-speed time along the route (budget in the module docstring)
V2_TIME_SLACK = 400        # steps (16 s) for the initial turn, lag and waiting
V2_MAX_STEPS_CAP = 4500    # 180 s, the physics 6x6 time limit
FINE_RES = 0.05            # m, resolution of the continuous geodesic field

# ---- v4 stall price (opt-in via MazeNavEnv(idle_cost=...); the default 0.0 never reads these)
V4_IDLE_A0_MAX = -0.9      # commanded forward action at or below this is "forward command ~0" (v_cmd <= 0.05 * V_MAX)
V4_IDLE_FAR_M = 0.5        # m, straight-line goal distance (start of the step) beyond which idling is charged
V4_IDLE_COST = 0.008       # the v4 value: (V2_STEP_COST - 0.008) / (1 - 0.998) = -5.0 = V2_COLLISION

# ---- version 3 only (NavGym v5, 2026-10-01; versions 1 and 2 never read these). FROZEN: chosen, not tuned.
V3_VISIT_TAU_S = 60.0      # s, visitation recency decay: r *= exp(-dt / 60 s) every step
V3_VISIT_RADIUS = ROBOT_RADIUS  # m, cells whose centre lies within this of the robot are set to 1
V3_COARSE_BLOCK = 3        # fine cells per coarse cell side: 3 x 0.2 m = 0.6 m
V3_COARSE_RES = 0.6        # m (= V3_COARSE_BLOCK * MAP_RES; recorded, the crop uses the product)
V3_COARSE_CROP = 24        # coarse cells per side of the coarse crop (14.4 m)
V3_YAW_CHANGE_COST = 0.005  # reward -= 0.005 * |a_yaw(t) - a_yaw(t-1)|
V3_BRAKE_LO = 0.42         # m, team_sensors.brake_command: scale = clip((clearance - 0.42) / 0.48, 0, 1)
V3_BRAKE_SPAN = 0.48       # m
V3_BRAKE_HALF_ANGLE_DEG = 25.0  # sectors whose centres lie within +-25 deg of travel
V3_BRAKE_PERIOD_S = 0.1    # the 10 Hz exteroceptive packet the brake reads (team_sensors.EXTERO_PERIOD)
V3_YAW_SAT = 0.99          # reported only: |a_yaw| >= 0.99 counts as a saturated (full-rate) yaw command

# ---- held-out evaluation sets (training draws maze seeds < 10 000)
HELDOUT_BASE = 10_000      # the held-out set every run so far is evaluated on (maze seeds 10 000 + k)
FRESH_BASE = 20_000        # NavGym v3's scored set (maze seeds 20 000 + k), never used before 2026-09-27


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
    """v1 collision test: an axis-aligned SQUARE of half-width `radius` (kept for v1)."""
    return bool(np.any((x + radius > boxes[:, 0]) & (x - radius < boxes[:, 1]) & (y + radius > boxes[:, 2]) & (y - radius < boxes[:, 3])))


def box_clearance(boxes: np.ndarray, x, y) -> np.ndarray:
    """Euclidean distance from point(s) (x, y) to the nearest wall box (0 inside a box).
    Scalars or arrays of any matching shape."""
    x = np.asarray(x, dtype=np.float64)[..., None]
    y = np.asarray(y, dtype=np.float64)[..., None]
    dx = np.maximum(np.maximum(boxes[:, 0] - x, 0.0), x - boxes[:, 1])
    dy = np.maximum(np.maximum(boxes[:, 2] - y, 0.0), y - boxes[:, 3])
    return np.sqrt(dx * dx + dy * dy).min(axis=-1)


def circle_in_wall(boxes: np.ndarray, x: float, y: float, radius: float) -> bool:
    """v2 collision test: a ROUND footprint -- some wall box is closer than `radius` to (x, y)."""
    dx = np.maximum(np.maximum(boxes[:, 0] - x, 0.0), x - boxes[:, 1])
    dy = np.maximum(np.maximum(boxes[:, 2] - y, 0.0), y - boxes[:, 3])
    return bool(np.any(dx * dx + dy * dy < radius * radius))


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


class FineGeodesic:
    """Continuous geodesic distance to the goal (v2 potential).

    Nodes on a `res` lattice over the maze bounds; a node is free when a disc of
    `radius` centred on it touches no wall box (free space dilated by the robot
    radius, i.e. the robot centre's configuration space). Dijkstra from the goal
    over free nodes with 16-connectivity (orthogonal, diagonal and knight moves,
    no corner cutting) keeps the metric within ~3 % of Euclidean. Nodes that are
    not free, or not reachable, take the value of their nearest reachable free
    node plus the distance to it, so a bilinear stencil that straddles the
    dilated boundary never reads an infinity. Lookup is bilinear."""

    # (move, lattice nodes the straight segment passes next to, all of which must be free)
    _MOVES = (((1, 0), ()), ((0, 1), ()),
              ((1, 1), ((1, 0), (0, 1))), ((1, -1), ((1, 0), (0, -1))),
              ((1, 2), ((0, 1), (1, 1))), ((1, -2), ((0, -1), (1, -1))),
              ((2, 1), ((1, 0), (1, 1))), ((2, -1), ((1, 0), (1, -1))))

    def __init__(self, maze: Maze, boxes: np.ndarray, goal_xy, res: float = FINE_RES, radius: float = ROBOT_RADIUS):
        from scipy.ndimage import distance_transform_edt
        from scipy.sparse import coo_matrix
        from scipy.sparse.csgraph import dijkstra
        xmin, xmax, ymin, ymax = maze.bounds()
        self.res = float(res)
        self.x0, self.y0 = xmin, ymin
        nx = int(round((xmax - xmin) / res)) + 1
        ny = int(round((ymax - ymin) / res)) + 1
        self.nx, self.ny = nx, ny
        xs = xmin + np.arange(nx) * res
        ys = ymin + np.arange(ny) * res
        free = np.ones((nx, ny), dtype=bool)
        for bx0, bx1, by0, by1 in boxes:
            # only the nodes within `radius` of this box's bounding rectangle can be blocked by it
            ia, ib = max(0, int(math.floor((bx0 - radius - xmin) / res))), min(nx, int(math.ceil((bx1 + radius - xmin) / res)) + 1)
            ja, jb = max(0, int(math.floor((by0 - radius - ymin) / res))), min(ny, int(math.ceil((by1 + radius - ymin) / res)) + 1)
            if ia >= ib or ja >= jb:
                continue
            dx = np.maximum(np.maximum(bx0 - xs[ia:ib], 0.0), xs[ia:ib] - bx1)[:, None]
            dy = np.maximum(np.maximum(by0 - ys[ja:jb], 0.0), ys[ja:jb] - by1)[None, :]
            free[ia:ib, ja:jb] &= (dx * dx + dy * dy) >= radius * radius
        idx = np.arange(nx * ny).reshape(nx, ny)
        rows, cols, wts = [], [], []
        for (di, dj), via in self._MOVES:
            # node (i, j) -> (i + di, j + dj): both ends free and every node in `via` free
            i0, i1 = 0, nx - di
            j0, j1 = max(0, -dj), ny - max(0, dj)
            ok = free[i0:i1, j0:j1] & free[i0 + di:i1 + di, j0 + dj:j1 + dj]
            for vi, vj in via:
                ok &= free[i0 + vi:i1 + vi, j0 + vj:j1 + vj]
            a = idx[i0:i1, j0:j1][ok]
            b = idx[i0 + di:i1 + di, j0 + dj:j1 + dj][ok]
            rows.append(a); cols.append(b); wts.append(np.full(a.size, res * math.hypot(di, dj)))
        # a virtual source (index nx*ny) tied to the free corners of the goal's lattice square
        gx, gy = float(goal_xy[0]), float(goal_xy[1])
        gi, gj = int(math.floor((gx - xmin) / res)), int(math.floor((gy - ymin) / res))
        src_r, src_c, src_w = [], [], []
        for a in (gi, gi + 1):
            for b in (gj, gj + 1):
                if 0 <= a < nx and 0 <= b < ny and free[a, b]:
                    src_r.append(nx * ny); src_c.append(idx[a, b]); src_w.append(max(1e-9, math.hypot(xs[a] - gx, ys[b] - gy)))
        if not src_r:
            raise ValueError("goal is not in free space")
        rows.append(np.asarray(src_r)); cols.append(np.asarray(src_c)); wts.append(np.asarray(src_w))
        n = nx * ny + 1
        g = coo_matrix((np.concatenate(wts), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n)).tocsr()
        dist = dijkstra(g, directed=False, indices=n - 1)[:-1].reshape(nx, ny)
        reach = np.isfinite(dist)
        if not reach.all():
            _, (ii, jj) = distance_transform_edt(~reach, return_indices=True)
            fill = dist[ii, jj] + res * np.hypot(np.arange(nx)[:, None] - ii, np.arange(ny)[None, :] - jj)
            dist = np.where(reach, dist, fill)
        self.free = free
        self.dist = dist.astype(np.float64)

    def __call__(self, x: float, y: float) -> float:
        fx = (x - self.x0) / self.res
        fy = (y - self.y0) / self.res
        i = min(max(int(math.floor(fx)), 0), self.nx - 2)
        j = min(max(int(math.floor(fy)), 0), self.ny - 2)
        u = min(max(fx - i, 0.0), 1.0)
        v = min(max(fy - j, 0.0), 1.0)
        d = self.dist
        return float((1 - u) * (1 - v) * d[i, j] + u * (1 - v) * d[i + 1, j] + (1 - u) * v * d[i, j + 1] + u * v * d[i + 1, j + 1])


_FINE_CACHE: "OrderedDict[tuple, FineGeodesic]" = OrderedDict()
_FINE_CACHE_MAX = 256


def fine_geodesic(maze: Maze, boxes: np.ndarray | None = None, res: float = FINE_RES, radius: float = ROBOT_RADIUS) -> FineGeodesic:
    """FineGeodesic for this maze, cached by (size, seed, openings, res, radius) with an LRU bound."""
    key = (maze.n, maze.m, maze.seed, maze.extra_openings, float(res), float(radius))
    hit = _FINE_CACHE.get(key)
    if hit is not None:
        _FINE_CACHE.move_to_end(key)
        return hit
    if boxes is None:
        boxes = wall_boxes(maze)
    f = FineGeodesic(maze, boxes, maze.centre(maze.goal), res=res, radius=radius)
    _FINE_CACHE[key] = f
    while len(_FINE_CACHE) > _FINE_CACHE_MAX:
        _FINE_CACHE.popitem(last=False)
    return f


def route_time_limit(route_m: float) -> int:
    """v2 episode limit in steps: 3x the full-speed time along the route, plus 16 s, capped at 180 s."""
    return int(min(V2_MAX_STEPS_CAP, math.ceil(V2_TIME_FACTOR * route_m / (V_MAX * DT)) + V2_TIME_SLACK))


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


def near_field(ranges: np.ndarray) -> np.ndarray:
    """v2 "near" key: the 36 sector minima clipped at NEAR_RANGE (2 m) and scaled by it."""
    return sector_minima(ranges, clip=NEAR_RANGE)


# ------------------------------------------------- version 3 (NavGym v5): visitation, coarse map, brake
def ego_indices(x0: float, y0: float, nx: int, ny: int, res: float, x: float, y: float, yaw: float, size: int):
    """Grid indices (ii, jj) and in-grid mask of a (size, size) egocentric crop of a world-fixed grid with origin
    (x0, y0), shape (nx, ny) and cell `res`: EgoMap.crop's sampling, expression for expression (offsets at whole
    cells, one more row ahead than behind; forward = up, left = left)."""
    half = size // 2
    u = (half - np.arange(size)) * res
    v = (half - np.arange(size)) * res
    U, V = np.meshgrid(u, v, indexing="ij")
    c, s = math.cos(yaw), math.sin(yaw)
    wx = x + U * c - V * s; wy = y + U * s + V * c
    ii = np.floor((wx - x0) / res).astype(int); jj = np.floor((wy - y0) / res).astype(int)
    ok = (ii >= 0) & (ii < nx) & (jj >= 0) & (jj < ny)
    return ii, jj, ok


class VisitMap:
    """v3 visitation memory: a world-fixed recency grid on the EgoMap lattice for the same bounds (same origin,
    resolution and shape), r in [0, 1]. ``step(x, y)``: every cell decays r *= exp(-dt / tau), then the cells whose
    centres lie within `radius` of (x, y) are set to 1. Used, unchanged, by the gym (once per step at the post-step
    pose, and at reset) and by the physics runner (once per control step at the pose the observation is built at)."""

    def __init__(self, bounds, margin: float = 1.0, res: float = MAP_RES, dt: float = DT, tau: float = V3_VISIT_TAU_S,
                 radius: float = V3_VISIT_RADIUS):
        xmin, xmax, ymin, ymax = bounds
        self.res = float(res)
        # EgoMap's geometry, the same expressions
        self.x0, self.y0 = xmin - margin, ymin - margin
        self.nx = int(math.ceil((xmax - xmin + 2 * margin) / res)); self.ny = int(math.ceil((ymax - ymin + 2 * margin) / res))
        self.r = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.dt, self.tau, self.radius = float(dt), float(tau), float(radius)
        self.decay = math.exp(-self.dt / self.tau)
        self.updates = 0

    def footprint(self, x: float, y: float):
        """(ii, jj) of the cells whose centres lie within `radius` of (x, y) (in-grid only)."""
        i0 = max(0, int(math.floor((x - self.radius - self.x0) / self.res)))
        i1 = min(self.nx - 1, int(math.floor((x + self.radius - self.x0) / self.res)))
        j0 = max(0, int(math.floor((y - self.radius - self.y0) / self.res)))
        j1 = min(self.ny - 1, int(math.floor((y + self.radius - self.y0) / self.res)))
        if i0 > i1 or j0 > j1:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=int)
        ii, jj = np.meshgrid(np.arange(i0, i1 + 1), np.arange(j0, j1 + 1), indexing="ij")
        cx = self.x0 + (ii + 0.5) * self.res; cy = self.y0 + (jj + 0.5) * self.res
        inside = (cx - x) ** 2 + (cy - y) ** 2 <= self.radius ** 2
        return ii[inside], jj[inside]

    def step(self, x: float, y: float) -> None:
        self.r *= np.float32(self.decay)
        ii, jj = self.footprint(x, y)
        self.r[ii, jj] = 1.0
        self.updates += 1

    def crop(self, x: float, y: float, yaw: float, size: int = MAP_CROP) -> np.ndarray:
        """(size, size) egocentric crop, sampled exactly as EgoMap.crop; 0 outside the grid."""
        ii, jj, ok = ego_indices(self.x0, self.y0, self.nx, self.ny, self.res, x, y, yaw, size)
        out = np.zeros((size, size), dtype=np.float32)
        out[ok] = self.r[ii[ok], jj[ok]]
        return out


def fine_channels(emap: "EgoMap", visit: VisitMap) -> np.ndarray:
    """(4, nx, ny) world-fixed fine channels: EgoMap.crop's occupied / free / unknown classes of every cell (the same
    float32 sigmoid and thresholds) and the visitation value."""
    if (visit.nx, visit.ny, visit.x0, visit.y0, visit.res) != (emap.nx, emap.ny, emap.x0, emap.y0, emap.res):
        raise ValueError("the visitation grid and the ego map must share one lattice")
    p = 1.0 / (1.0 + np.exp(-emap.l))
    occ = (p > 0.65).astype(np.float32); free = (p < 0.35).astype(np.float32)
    unk = 1.0 - occ - free
    return np.stack([occ, free, unk, visit.r.astype(np.float32)])


def coarse_grid(emap: "EgoMap", visit: VisitMap, block: int = V3_COARSE_BLOCK) -> np.ndarray:
    """(4, NX, NY) coarse grid: fine_channels averaged over block x block fine cells, blocks anchored at the fine
    grid's origin; fine cells past the grid's edge (padding up to a multiple of `block`) count as unknown, unvisited."""
    ch = fine_channels(emap, visit)
    nx, ny = ch.shape[1:]
    NX, NY = -(-nx // block), -(-ny // block)
    pad = ((0, NX * block - nx), (0, NY * block - ny))
    padded = np.stack([np.pad(ch[0], pad), np.pad(ch[1], pad), np.pad(ch[2], pad, constant_values=1.0), np.pad(ch[3], pad)])
    return padded.reshape(4, NX, block, NY, block).mean(axis=(2, 4)).astype(np.float32)


def coarse_crop(emap: "EgoMap", visit: VisitMap, x: float, y: float, yaw: float, size: int = V3_COARSE_CROP,
                block: int = V3_COARSE_BLOCK) -> np.ndarray:
    """v3 "map_coarse": (4, size, size) egocentric crop of coarse_grid at block * MAP_RES (0.6 m) cells, sampled as
    EgoMap.crop samples the fine grid (same frame: forward = up); outside the grid: unknown (0, 0, 1, 0)."""
    grid = coarse_grid(emap, visit, block)
    ii, jj, ok = ego_indices(emap.x0, emap.y0, grid.shape[1], grid.shape[2], emap.res * block, x, y, yaw, size)
    out = np.zeros((4, size, size), dtype=np.float32)
    out[2] = 1.0
    out[:, ok] = grid[:, ii[ok], jj[ok]]
    return out


_V3_SECTOR_CENTRES = np.linspace(-np.pi, np.pi, LIDAR_RAYS, endpoint=False).reshape(LIDAR_SECTORS, -1).mean(axis=1)
# team_sensors.brake_command's selection for a forward command (heading = atan2(vy = 0, vx > 0) = 0)
V3_BRAKE_SECTORS = np.abs(np.arctan2(np.sin(_V3_SECTOR_CENTRES - 0.0), np.cos(_V3_SECTOR_CENTRES - 0.0))) <= np.deg2rad(V3_BRAKE_HALF_ANGLE_DEG)


def brake_scale(ranges: np.ndarray) -> float:
    """team_sensors.brake_command's range term for a forward command, from one raw 108-ray scan: the clearance is the
    minimum of the sector minima whose centres lie within +-25 deg of travel; scale = clip((clearance - 0.42) / 0.48, 0, 1)."""
    sec = np.asarray(ranges, dtype=np.float64).reshape(LIDAR_SECTORS, -1).min(axis=1)
    clearance = float(np.min(sec[V3_BRAKE_SECTORS]))
    return float(np.clip((clearance - V3_BRAKE_LO) / V3_BRAKE_SPAN, 0.0, 1.0))


OBS_KEYS_V1 = ("lidar", "map", "goal")
OBS_KEYS_V2 = ("lidar", "near", "map", "goal")
OBS_KEYS_V3 = ("lidar", "near", "map_visit", "map_coarse", "goal")
V3_VISIT_KEYS = ("map_visit", "map_coarse")   # the keys that need a VisitMap (build_obs(..., visit=...))


def build_obs(keys, ranges: np.ndarray, emap: "EgoMap", x: float, y: float, yaw: float, goal_xy, prev_action, visit=None) -> dict:
    """The policy observation for the requested keys, from one raw 108-ray scan (body-frame
    angles linspace(-pi, pi, 108, endpoint=False)), the ego map, the pose, the goal and the
    previous action. Shared by the gym (v2) and the physics runner, so both build it the same way.
    The v3 keys "map_visit" and "map_coarse" also need the visitation memory `visit` (a VisitMap)."""
    out = {}
    for k in keys:
        if k == "lidar":
            out[k] = sector_minima(ranges).astype(np.float32)
        elif k == "near":
            out[k] = near_field(ranges).astype(np.float32)
        elif k == "map":
            out[k] = emap.crop(x, y, yaw).astype(np.float32)
        elif k == "goal":
            out[k] = goal_features(x, y, yaw, goal_xy, prev_action)
        elif k in V3_VISIT_KEYS:
            if visit is None:
                raise ValueError(f"observation key {k!r} needs the visitation memory (build_obs(..., visit=VisitMap))")
            if k == "map_visit":
                out[k] = np.concatenate([emap.crop(x, y, yaw), visit.crop(x, y, yaw)[None]]).astype(np.float32)
            else:
                out[k] = coarse_crop(emap, visit, x, y, yaw)
        else:
            raise KeyError(f"unknown observation key {k!r}")
    return out


# ------------------------------------------------------------------ the env
class MazeNavEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, sizes=((3, 3), (4, 4), (5, 5), (6, 6)), extra_openings: int = 1, max_steps: int | None = None,
                 seed_base: int = 0, seed_span: int = 10_000, randomize_dynamics: bool = True, size_weights=None,
                 version: int = 1, gamma: float = V2_GAMMA, idle_cost: float = 0.0):
        """`version` 1 (default) is the v1 environment, unchanged; 2 is NavGym v2 (module
        docstring). `max_steps` None means the version's default -- v1: a flat 1500; v2: the
        route-scaled limit of `route_time_limit`, set at every reset -- and an int is a flat
        limit in either version. `gamma` is the shaping discount (v2 only; pass the learner's).
        `idle_cost` (v2 only, >= 0; default 0.0 = off) is the v4 stall price (module docstring).
        `version` 3 is NavGym v5 (module docstring): version 2 plus the visitation and coarse map channels, the
        yaw-change penalty and the speed brake; it takes v2's max_steps, gamma and idle_cost semantics."""
        super().__init__()
        if version not in (1, 2, 3):
            raise ValueError(f"MazeNavEnv version must be 1, 2 or 3, got {version!r}")
        if idle_cost < 0 or (idle_cost and version not in (2, 3)):
            raise ValueError(f"idle_cost must be >= 0 and needs version 2 or 3, got {idle_cost!r} with version {version!r}")
        self.idle_cost = float(idle_cost)
        self.version = int(version)
        self.sizes = tuple(tuple(s) for s in sizes)
        self.size_weights = None if size_weights is None else np.asarray(size_weights, dtype=float) / np.sum(size_weights)
        self.extra_openings = extra_openings
        self.max_steps_arg = max_steps
        self.max_steps = (1500 if max_steps is None else int(max_steps)) if self.version == 1 else (V2_MAX_STEPS_CAP if max_steps is None else int(max_steps))
        self.gamma = float(gamma)
        self.seed_base, self.seed_span = seed_base, seed_span
        self.randomize_dynamics = randomize_dynamics
        self.angles = np.linspace(-np.pi, np.pi, LIDAR_RAYS, endpoint=False)
        spaces = {
            "lidar": gym.spaces.Box(0.0, 1.0, (LIDAR_SECTORS,), np.float32),
            "map": gym.spaces.Box(0.0, 1.0, (3, MAP_CROP, MAP_CROP), np.float32),
            "goal": gym.spaces.Box(-1.0, 1.0, (5,), np.float32),
        }
        if self.version == 2:
            spaces["near"] = gym.spaces.Box(0.0, 1.0, (LIDAR_SECTORS,), np.float32)
        if self.version == 3:
            # v2's keys with "map" replaced by the 4-channel "map_visit" and the second map input "map_coarse"
            del spaces["map"]
            spaces["near"] = gym.spaces.Box(0.0, 1.0, (LIDAR_SECTORS,), np.float32)
            spaces["map_visit"] = gym.spaces.Box(0.0, 1.0, (4, MAP_CROP, MAP_CROP), np.float32)
            spaces["map_coarse"] = gym.spaces.Box(0.0, 1.0, (4, V3_COARSE_CROP, V3_COARSE_CROP), np.float32)
        self.obs_keys = OBS_KEYS_V1 if self.version == 1 else (OBS_KEYS_V2 if self.version == 2 else OBS_KEYS_V3)
        self.observation_space = gym.spaces.Dict(spaces)
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
        if self.version == 1:
            self.potential = geodesic_distance(self.maze, self.field, self.x, self.y)
            self._scan()
            return self._obs(), {"maze_seed": self.episode_seed, "size": (n, m)}
        # v2: continuous geodesic (cached per maze), route-scaled time limit
        self.geo = fine_geodesic(self.maze, self.boxes)
        self.route_m = self.geo(*self.maze.centre(self.maze.start))
        self.max_steps = route_time_limit(self.route_m) if self.max_steps_arg is None else int(self.max_steps_arg)
        self.potential = self.geo(self.x, self.y)          # metres to the goal; phi = -potential
        if self.version == 3:
            # v3 state (draws nothing from the RNG, so v3 episodes share v2's mazes, dynamics and starts)
            self.visit = VisitMap(self.maze.bounds())       # marked at the start pose by _scan below
            self.next_capture = 0.0                         # the brake's 10 Hz packet schedule (team_sensors)
            self.brake_k = 1.0                              # brake scale of the current packet (set at the first capture)
            self.yaw_flips = self.yaw_sat_steps = self.braked_steps = 0
        self._scan()
        return self._obs(), {"maze_seed": self.episode_seed, "size": (n, m), "version": self.version, "max_steps": self.max_steps,
                             "route_m": float(self.route_m)}

    def _scan(self):
        self.ranges = cast_rays(self.boxes, (self.x, self.y), self.yaw + self.angles, LIDAR_RANGE)
        self.emap.update(self.x, self.y, self.yaw, self.angles, self.ranges)
        if self.version == 3:
            self.visit.step(self.x, self.y)                 # once per step (and at reset), at the pose the obs is built at

    def _obs(self):
        if self.version == 3:
            return build_obs(OBS_KEYS_V3, self.ranges, self.emap, self.x, self.y, self.yaw, self.goal_xy, self.prev_action,
                             visit=self.visit)
        if self.version == 2:
            return build_obs(OBS_KEYS_V2, self.ranges, self.emap, self.x, self.y, self.yaw, self.goal_xy, self.prev_action)
        return {"lidar": sector_minima(self.ranges).astype(np.float32), "map": self.emap.crop(self.x, self.y, self.yaw),
                "goal": goal_features(self.x, self.y, self.yaw, self.goal_xy, self.prev_action)}

    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        # v4 stall price (off by default): forward command ~0 while > V4_IDLE_FAR_M from the goal, judged at the step start
        idle = bool(self.idle_cost) and float(a[0]) <= V4_IDLE_A0_MAX and \
            math.hypot(self.goal_xy[0] - self.x, self.goal_xy[1] - self.y) > V4_IDLE_FAR_M
        v3 = self._v3_pre_step(a) if self.version == 3 else None     # reads prev_action: before it is overwritten
        self.prev_action = a.copy()
        v_cmd = (a[0] + 1.0) * 0.5 * V_MAX
        w_cmd = a[1] * W_MAX
        if v3 is not None:
            v_cmd = v_cmd * v3["brake_scale"]                         # v3 speed brake, before the latency queue
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
        if self.version == 2:
            return self._finish_v2(nx, ny, idle)
        if self.version == 3:
            return self._finish_v3(nx, ny, idle, v3)
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

    def _finish_v2(self, nx: float, ny: float, idle: bool = False):
        """v2 transition: round footprint, continuous potential-based shaping, -5 crash
        (and, only when idle_cost > 0, the v4 idle cost on an idle step of any outcome)."""
        terminated, truncated = False, False
        reward = V2_STEP_COST
        if circle_in_wall(self.boxes, nx, ny, ROBOT_RADIUS):
            # the robot does not move and no shaping is paid on the crash step
            reward += V2_COLLISION
            terminated = True
            info = {"outcome": "collision"}
        else:
            self.x, self.y = nx, ny
            new_pot = self.geo(self.x, self.y)
            # F = gamma * phi(s') - phi(s), phi = -geodesic
            reward += self.gamma * (-new_pot) - (-self.potential)
            self.potential = new_pot
            info = {"outcome": None}
            if math.hypot(self.goal_xy[0] - self.x, self.goal_xy[1] - self.y) < 0.30:
                reward += V2_GOAL
                terminated = True
                info = {"outcome": "goal"}
            elif self.t >= self.max_steps:
                truncated = True
                info = {"outcome": "time_out"}
        if self.idle_cost:
            if idle:
                reward -= self.idle_cost
            info["idle_charged"] = bool(idle)
        self._scan()
        info["maze_seed"] = self.episode_seed
        info["t"] = self.t
        return self._obs(), float(reward), terminated, truncated, info

    def _v3_pre_step(self, a) -> dict:
        """v3, at the top of a step, before prev_action is overwritten: refresh the brake's packet when a capture is
        due (team_sensors.filter_commands' schedule), and the yaw change and yaw statistics against the previous action."""
        now = self.t * DT
        if now + 1e-9 >= self.next_capture:
            while self.next_capture <= now + 1e-9:
                self.next_capture += V3_BRAKE_PERIOD_S
            # a packet captured at this step's start pose: that is the pose of the last scan, so the packet is that scan
            self.brake_k = brake_scale(self.ranges)
        prev, cur = float(self.prev_action[1]), float(a[1])
        self.yaw_flips += int(np.sign(cur) * np.sign(prev) < 0)
        self.yaw_sat_steps += int(abs(cur) >= V3_YAW_SAT)
        self.braked_steps += int(self.brake_k < 1.0 and float(a[0]) > -1.0)
        return {"yaw_change": abs(cur - prev), "brake_scale": self.brake_k}

    def _finish_v3(self, nx: float, ny: float, idle: bool, v3: dict):
        """v3 transition: the v2 transition exactly (its scan also steps the visitation memory, and its observation is
        the v3 one), minus the yaw-change penalty on every step of any outcome. Episode-cumulative yaw and brake counts
        ride along in info (reported, never rewarded)."""
        obs, reward, terminated, truncated, info = self._finish_v2(nx, ny, idle)
        cost = V3_YAW_CHANGE_COST * v3["yaw_change"]
        info.update({"yaw_change_cost": cost, "brake_scale": v3["brake_scale"], "yaw_flips": self.yaw_flips,
                     "yaw_sat_steps": self.yaw_sat_steps, "braked_steps": self.braked_steps})
        return obs, float(reward - cost), terminated, truncated, info

    def heldout_reset(self, size, seed: int, maze_base: int = HELDOUT_BASE, dyn_base: int | None = None) -> dict:
        """Make this env held-out episode `seed` on a `size` maze -- the same episode as
        `heldout_env(size, seed, version, gamma, maze_base=..., dyn_base=...)` -- and reset it;
        returns the observation. (For a pool of evaluation workers; the env keeps its version,
        gamma and max_steps.)"""
        mb, db = heldout_bases(maze_base, dyn_base)
        self.set_sizes((tuple(size),))
        self.randomize_dynamics = True
        self.seed_base, self.seed_span = mb + int(seed), 1
        obs, _ = self.reset(seed=db + int(seed))
        return obs


def heldout_bases(maze_base: int = HELDOUT_BASE, dyn_base: int | None = None) -> tuple:
    """(maze seed base, dynamics reset-seed base) of a held-out set. The original set (maze_base
    10 000) draws its dynamics from reset(seed=k) -- dyn_base 0 -- as every published evaluation
    did; any other set defaults to dyn_base = maze_base (reset(seed=maze_base + k))."""
    mb = int(maze_base)
    db = (0 if mb == HELDOUT_BASE else mb) if dyn_base is None else int(dyn_base)
    return mb, db


def heldout_env(size, seed: int, version: int = 1, gamma: float = V2_GAMMA, max_steps: int | None = None,
                maze_base: int = HELDOUT_BASE, dyn_base: int | None = None) -> tuple:
    """The trainer's held-out evaluation episode `seed` on a `size` maze: returns (env, obs, info).
    Default: maze seed 10 000 + seed, randomized dynamics drawn from reset(seed=seed) -- the
    published held-out set. `maze_base=FRESH_BASE` gives the v3 scored set: maze seed 20 000 + seed,
    dynamics from reset(seed=20 000 + seed) (`heldout_bases`)."""
    mb, db = heldout_bases(maze_base, dyn_base)
    env = MazeNavEnv(sizes=(tuple(size),), randomize_dynamics=True, seed_base=mb + int(seed), seed_span=1,
                     version=version, gamma=gamma, max_steps=max_steps)
    obs, info = env.reset(seed=db + int(seed))
    return env, obs, info
