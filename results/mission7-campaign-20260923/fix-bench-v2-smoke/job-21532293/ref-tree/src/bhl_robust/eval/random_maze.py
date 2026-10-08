"""Randomized mazes for a sensor-driven navigation mission.

A new maze per seed (recursive backtracker on a grid, plus a few extra
openings so there is more than one route), an MJCF world for it, a log-odds
occupancy grid built from the robot's own lidar returns, an A* planner over
that grid that treats unknown space as free and replans as walls appear, and a
turn-then-walk command generator for a gait that turns in place.

Nothing here is a learned policy, and nothing here knows the maze: the planner
sees only what the lidar has mapped. Localization is the simulator's true pose
(an oracle), and every frame that shows this says so. Pure numpy so the tests
run on a login node; MuJoCo enters only through the XML this module writes.
"""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass, field

import numpy as np

CELL = 1.4          # m, cell pitch (1.32 m clear corridor)
WALL_T = 0.08       # m, wall thickness
WALL_H = 1.1        # m, wall height (the inspection maze's)
DIRS = ((1, 0), (0, 1), (-1, 0), (0, -1))


# --------------------------------------------------------------------- maze
@dataclass
class Maze:
    n: int
    m: int
    seed: int
    open_edges: set = field(default_factory=set)      # frozenset({cell, cell})
    extra_openings: int = 0

    @property
    def start(self):
        return (0, 0)

    @property
    def goal(self):
        return (self.n - 1, self.m - 1)

    def cells(self):
        return [(i, j) for i in range(self.n) for j in range(self.m)]

    def is_open(self, a, b) -> bool:
        return frozenset((a, b)) in self.open_edges

    def neighbours(self, c):
        for dx, dy in DIRS:
            d = (c[0] + dx, c[1] + dy)
            if 0 <= d[0] < self.n and 0 <= d[1] < self.m and self.is_open(c, d):
                yield d

    @staticmethod
    def centre(c):
        return (c[0] * CELL, c[1] * CELL)

    def bounds(self):
        """(xmin, xmax, ymin, ymax) of the walled area."""
        return (-CELL / 2, (self.n - 1) * CELL + CELL / 2, -CELL / 2, (self.m - 1) * CELL + CELL / 2)

    def wall_segments(self):
        """Axis-aligned wall boxes as ((cx, cy), (hx, hy)), deduplicated."""
        seen = {}
        for c in self.cells():
            cx, cy = self.centre(c)
            for dx, dy in DIRS:
                d = (c[0] + dx, c[1] + dy)
                inside = 0 <= d[0] < self.n and 0 <= d[1] < self.m
                if inside and self.is_open(c, d):
                    continue
                centre = (round(cx + dx * CELL / 2, 4), round(cy + dy * CELL / 2, 4))
                half = (WALL_T / 2, CELL / 2 + WALL_T / 2) if dx else (CELL / 2 + WALL_T / 2, WALL_T / 2)
                seen[(centre, half)] = None
        return [(k[0], k[1]) for k in seen]

    def solution(self):
        """Breadth-first shortest cell path from start to goal (the oracle's answer)."""
        prev = {self.start: None}
        queue = [self.start]
        while queue:
            c = queue.pop(0)
            if c == self.goal:
                break
            for d in self.neighbours(c):
                if d not in prev:
                    prev[d] = c
                    queue.append(d)
        if self.goal not in prev:
            return None
        path, c = [], self.goal
        while c is not None:
            path.append(c)
            c = prev[c]
        return path[::-1]


def generate(n: int = 5, m: int = 5, seed: int = 0, extra_openings: int = 2) -> Maze:
    """Recursive-backtracker perfect maze, then `extra_openings` closed internal
    walls removed at random so the maze has loops (and a wrong branch can
    rejoin instead of only dead-ending)."""
    if n < 2 or m < 2:
        raise ValueError("a maze needs at least 2x2 cells")
    rng = np.random.default_rng(seed)
    maze = Maze(n, m, seed, extra_openings=extra_openings)
    visited = {maze.start}
    stack = [maze.start]
    while stack:
        c = stack[-1]
        options = [(c[0] + dx, c[1] + dy) for dx, dy in DIRS]
        options = [d for d in options if 0 <= d[0] < n and 0 <= d[1] < m and d not in visited]
        if not options:
            stack.pop()
            continue
        d = options[int(rng.integers(len(options)))]
        maze.open_edges.add(frozenset((c, d)))
        visited.add(d)
        stack.append(d)
    closed = []
    for c in maze.cells():
        for dx, dy in ((1, 0), (0, 1)):
            d = (c[0] + dx, c[1] + dy)
            if d[0] < n and d[1] < m and not maze.is_open(c, d):
                closed.append((c, d))
    for idx in rng.permutation(len(closed))[:extra_openings]:
        maze.open_edges.add(frozenset(closed[idx]))
    return maze


def world_xml(maze: Maze, textured: bool = True) -> str:
    """The maze as MJCF: floor, walls named `wall_maze_*` (the contact judge
    keys on the prefix), an orange start disc and a green goal disc."""
    geoms = []
    wall_look = 'material="wall"' if textured else 'rgba=".46 .51 .57 1"'
    for i, ((x, y), (hx, hy)) in enumerate(maze.wall_segments()):
        geoms.append(f'<geom name="wall_maze_{i}" type="box" pos="{x} {y} {WALL_H / 2}" '
                     f'size="{hx} {hy} {WALL_H / 2}" {wall_look}/>')
    sx, sy = maze.centre(maze.start)
    gx, gy = maze.centre(maze.goal)
    geoms.append(f'<geom name="start_marker" type="cylinder" pos="{sx} {sy} .002" size=".23 .001" '
                 'rgba=".96 .63 .12 1" contype="0" conaffinity="0"/>')
    geoms.append(f'<geom name="goal_marker" type="cylinder" pos="{gx} {gy} .002" size=".26 .001" '
                 'rgba=".15 .8 .35 1" contype="0" conaffinity="0"/>')
    xmin, xmax, ymin, ymax = maze.bounds()
    cx, cy = (xmin + xmax) / 2, (ymin + ymax) / 2
    if not textured:
        return f'''<mujoco model="bhl-random-maze-{maze.seed}">
      <compiler angle="radian"/><option timestep=".0005"/>
      <visual><global offwidth="1280" offheight="720"/></visual>
      <worldbody><light pos="{cx} {cy} 6" dir="0 0 -1"/>
      <geom name="floor" type="plane" size="12 12 .05" rgba=".2 .24 .28 1"/>
    ''' + "\n".join(geoms) + "</worldbody></mujoco>"
    return f'''<mujoco model="bhl-random-maze-{maze.seed}">
      <compiler angle="radian"/><option timestep=".0005"/>
      <visual>
        <global offwidth="1280" offheight="720"/>
        <headlight diffuse="0.40 0.40 0.40" ambient="0.30 0.30 0.32" specular="0 0 0"/>
        <quality shadowsize="4096"/>
        <rgba haze="0.16 0.18 0.22 1"/>
      </visual>
      <asset>
        <texture type="skybox" builtin="gradient" rgb1="0.50 0.56 0.64" rgb2="0.10 0.12 0.16" width="512" height="3072"/>
        <texture type="2d" name="tile" builtin="checker" mark="edge" rgb1="0.66 0.65 0.62" rgb2="0.53 0.53 0.51"
                 markrgb="0.40 0.40 0.40" width="300" height="300"/>
        <material name="tile" texture="tile" texuniform="true" texrepeat="6 6" reflectance="0.0" specular="0.05" shininess="0.05"/>
        <texture type="2d" name="brick" builtin="checker" mark="edge" rgb1="0.52 0.57 0.64" rgb2="0.46 0.51 0.58"
                 markrgb="0.36 0.40 0.46" width="128" height="128"/>
        <material name="wall" texture="brick" texuniform="true" texrepeat="3 3" reflectance="0.0" specular="0.1" shininess="0.1"/>
      </asset>
      <worldbody>
      <light pos="{cx} {cy} 7" dir="0 0 -1" castshadow="true" diffuse="0.55 0.55 0.55" specular="0.05 0.05 0.05"/>
      <light pos="{cx - 3} {cy + 3} 5" dir="0.45 -0.55 -0.7" directional="true" castshadow="false" diffuse="0.30 0.30 0.32"/>
      <geom name="floor" type="plane" size="12 12 .05" material="tile"/>
    ''' + "\n".join(geoms) + "</worldbody></mujoco>"


# ------------------------------------------------------------- occupancy grid
class OccupancyGrid:
    """Log-odds occupancy on a fixed square lattice covering the maze."""

    L_FREE = -0.40
    L_OCC = 0.85
    L_MIN, L_MAX = -4.0, 4.0

    def __init__(self, bounds, res: float = 0.10, margin: float = 0.6):
        xmin, xmax, ymin, ymax = bounds
        self.res = float(res)
        self.x0, self.y0 = xmin - margin, ymin - margin
        self.nx = int(math.ceil((xmax - xmin + 2 * margin) / res))
        self.ny = int(math.ceil((ymax - ymin + 2 * margin) / res))
        self.l = np.zeros((self.nx, self.ny), dtype=np.float32)
        self.updates = 0

    def to_cell(self, x, y):
        return int((x - self.x0) / self.res), int((y - self.y0) / self.res)

    def to_world(self, i, j):
        return self.x0 + (i + 0.5) * self.res, self.y0 + (j + 0.5) * self.res

    def inside(self, i, j):
        return 0 <= i < self.nx and 0 <= j < self.ny

    def update(self, x, y, yaw, angles, ranges, max_range):
        """Integrate one scan taken at world pose (x, y, yaw). `angles` are
        body-frame ray angles, `ranges` metres; a return at max_range is a
        miss: free along the ray, nothing marked occupied."""
        for a, r in zip(angles, ranges):
            hit = r < max_range - 1e-3
            reach = min(float(r), max_range)
            ca, sa = math.cos(yaw + a), math.sin(yaw + a)
            n = int(reach / (self.res * 0.5))
            for k in range(n):
                d = k * self.res * 0.5
                if d >= reach - self.res * 0.5:
                    break
                i, j = self.to_cell(x + d * ca, y + d * sa)
                if self.inside(i, j):
                    self.l[i, j] = max(self.L_MIN, self.l[i, j] + self.L_FREE * 0.5)
            if hit:
                i, j = self.to_cell(x + reach * ca, y + reach * sa)
                if self.inside(i, j):
                    self.l[i, j] = min(self.L_MAX, self.l[i, j] + self.L_OCC)
        self.updates += 1

    def prob(self):
        return 1.0 / (1.0 + np.exp(-self.l))

    def occupied(self, thresh: float = 0.65):
        return self.prob() > thresh

    def known_fraction(self):
        p = self.prob()
        return float(np.mean((p > 0.65) | (p < 0.35)))


# ------------------------------------------------------------------ planning
def inflate(occ: np.ndarray, radius_cells: int) -> np.ndarray:
    out = occ.copy()
    r = int(radius_cells)
    if r <= 0:
        return out
    ii, jj = np.nonzero(occ)
    nx, ny = occ.shape
    for di in range(-r, r + 1):
        for dj in range(-r, r + 1):
            if di * di + dj * dj > r * r:
                continue
            a = np.clip(ii + di, 0, nx - 1)
            b = np.clip(jj + dj, 0, ny - 1)
            out[a, b] = True
    return out


def astar(blocked: np.ndarray, start, goal):
    """8-connected A* over a boolean blocked grid; returns cell path or None."""
    nx, ny = blocked.shape
    if blocked[goal]:
        # aim at the nearest unblocked cell around the goal
        best, bd = None, 1e9
        for di in range(-3, 4):
            for dj in range(-3, 4):
                g = (goal[0] + di, goal[1] + dj)
                if 0 <= g[0] < nx and 0 <= g[1] < ny and not blocked[g] and di * di + dj * dj < bd:
                    best, bd = g, di * di + dj * dj
        if best is None:
            return None
        goal = best
    h = lambda c: math.hypot(c[0] - goal[0], c[1] - goal[1])
    frontier = [(h(start), 0.0, start)]
    came, cost = {start: None}, {start: 0.0}
    while frontier:
        _, g, c = heapq.heappop(frontier)
        if c == goal:
            path, x = [], c
            while x is not None:
                path.append(x)
                x = came[x]
            return path[::-1]
        if g > cost.get(c, 1e18):
            continue
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di == 0 and dj == 0:
                    continue
                d = (c[0] + di, c[1] + dj)
                if not (0 <= d[0] < nx and 0 <= d[1] < ny) or blocked[d]:
                    continue
                if di and dj and (blocked[c[0] + di, c[1]] or blocked[c[0], c[1] + dj]):
                    continue                      # no corner cutting
                ng = g + math.hypot(di, dj)
                if ng < cost.get(d, 1e18):
                    cost[d] = ng
                    came[d] = c
                    heapq.heappush(frontier, (ng + h(d), ng, d))
    return None


def line_free(blocked: np.ndarray, a, b) -> bool:
    """Bresenham line-of-sight through unblocked cells."""
    (x0, y0), (x1, y1) = a, b
    dx, dy = abs(x1 - x0), abs(y1 - y0)
    sx, sy = (1 if x1 > x0 else -1), (1 if y1 > y0 else -1)
    err = dx - dy
    x, y = x0, y0
    while True:
        if blocked[x, y]:
            return False
        if (x, y) == (x1, y1):
            return True
        e2 = 2 * err
        if e2 > -dy:
            err -= dy
            x += sx
        if e2 < dx:
            err += dx
            y += sy


def simplify(blocked: np.ndarray, path):
    """Greedy line-of-sight pruning: keep the farthest visible cell each hop."""
    if not path or len(path) < 3:
        return list(path or [])
    out = [path[0]]
    i = 0
    while i < len(path) - 1:
        j = len(path) - 1
        while j > i + 1 and not line_free(blocked, path[i], path[j]):
            j -= 1
        out.append(path[j])
        i = j
    return out


class Planner:
    """A* on the inflated occupancy map; unknown cells count as free."""

    def __init__(self, grid: OccupancyGrid, inflate_m: float = 0.30):
        self.grid = grid
        self.r_cells = int(round(inflate_m / grid.res))
        self.last_blocked = None
        self.replans = 0

    def plan(self, xy, goal_xy):
        blocked = inflate(self.grid.occupied(), self.r_cells)
        s = self.grid.to_cell(*xy)
        g = self.grid.to_cell(*goal_xy)
        s = (min(max(s[0], 0), self.grid.nx - 1), min(max(s[1], 0), self.grid.ny - 1))
        g = (min(max(g[0], 0), self.grid.nx - 1), min(max(g[1], 0), self.grid.ny - 1))
        if blocked[s]:
            # the robot is inside an inflated wall band: free its own cell and neighbours
            i0, j0 = s
            blocked[max(0, i0 - 2):i0 + 3, max(0, j0 - 2):j0 + 3] = False
        self.last_blocked = blocked
        self.replans += 1
        cells = astar(blocked, s, g)
        if cells is None:
            return None
        cells = simplify(blocked, cells)
        pts = [self.grid.to_world(*c) for c in cells]
        # The last waypoint is the exact goal, never the goal *cell's* centre:
        # once the robot stands in that cell a one-cell path would otherwise
        # collapse onto its own position and read as "arrived" 0.3 m short.
        pts[-1] = (float(goal_xy[0]), float(goal_xy[1]))
        if len(pts) >= 2 and math.hypot(pts[-1][0] - pts[-2][0], pts[-1][1] - pts[-2][1]) < 0.05:
            pts.pop(-2)
        return pts


# --------------------------------------------------------------- controller
def wrap(a: float) -> float:
    return (a + math.pi) % (2 * math.pi) - math.pi


class TurnWalkController:
    """Turn in place toward the next waypoint, then walk forward.

    The gait's yaw-rate tracking was measured (dr-default-s0: 239 deg in 6 s at
    a 0.6 rad/s command, no fall), so turning is closed on the measured heading
    rather than the commanded rate. Commands are body-frame (vx, vy, wz); vy is
    always 0 -- this controller never walks sideways.
    """

    def __init__(self, turn_enter: float = 0.40, turn_exit: float = 0.15, turn_rate: float = 0.6,
                 cruise: float = 0.30, k_yaw: float = 1.2, wz_walk: float = 0.4, waypoint_radius: float = 0.30,
                 goal_radius: float = 0.30):
        self.turn_enter, self.turn_exit, self.turn_rate = turn_enter, turn_exit, turn_rate
        self.cruise, self.k_yaw, self.wz_walk = cruise, k_yaw, wz_walk
        self.waypoint_radius, self.goal_radius = waypoint_radius, goal_radius
        self.state = "turn"
        self.turns = 0

    def command(self, xy, yaw, waypoint, is_goal: bool):
        dx, dy = waypoint[0] - xy[0], waypoint[1] - xy[1]
        dist = math.hypot(dx, dy)
        err = wrap(math.atan2(dy, dx) - yaw)
        if dist < (self.goal_radius if is_goal else self.waypoint_radius):
            return np.zeros(3), "arrived", err
        if self.state == "walk" and abs(err) > self.turn_enter:
            self.state = "turn"
            self.turns += 1
        elif self.state == "turn" and abs(err) < self.turn_exit:
            self.state = "walk"
        if self.state == "turn":
            return np.array([0.0, 0.0, math.copysign(self.turn_rate, err)]), "turn", err
        wz = float(np.clip(self.k_yaw * err, -self.wz_walk, self.wz_walk))
        vx = self.cruise * max(0.0, math.cos(err))
        return np.array([vx, 0.0, wz]), "walk", err
