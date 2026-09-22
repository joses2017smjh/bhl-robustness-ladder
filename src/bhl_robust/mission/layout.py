"""Deterministic seven-decision mazes; all coordinates are scoring/world data."""
from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import numpy as np

SPLITS = {"train": (0, 256), "validation": (10000, 32), "test": (20000, 64)}
STAGES = ("approach", "branches", "navigation", "doors", "transport")


def edge(a, b):
    return tuple(sorted((tuple(a), tuple(b))))


@dataclass(frozen=True)
class Layout:
    split: str
    seed: int
    cell_m: float
    edges: tuple
    route: tuple
    door_indices: tuple
    correct_sides: tuple
    object_index: int

    @property
    def adjacency(self):
        out = {}
        for a, b in self.edges:
            out.setdefault(a, []).append(b)
            out.setdefault(b, []).append(a)
        return out

    def xy(self, cell):
        return np.asarray(cell, dtype=float) * self.cell_m

    @property
    def decisions(self):
        return sum(len(self.adjacency[c]) >= 3 for c in self.route[1:-1])

    @property
    def turns(self):
        d = np.diff(np.asarray(self.route), axis=0)
        return int(np.any(d[1:] != d[:-1], axis=1).sum())

    @property
    def fingerprint(self):
        # Exclude split/seed: duplicate geometry must not cross split boundaries.
        return hashlib.sha256(json.dumps(self.edges).encode()).hexdigest()

    def metadata(self):
        return {**asdict(self), "geometry_sha256": self.fingerprint,
                "route_decisions": self.decisions, "route_turns": self.turns,
                "centerline_m": (len(self.route)-1)*self.cell_m,
                "decision_ratio_to_one_decision_reference": self.decisions,
                "legacy_reference_route_m": 3.0}

    def door(self, i):
        k = self.door_indices[i]
        a, b = self.xy(self.route[k]), self.xy(self.route[k+1])
        return (a+b)/2, (b-a)/self.cell_m

    def plate(self, i, side):
        k = self.door_indices[i]
        _, direction = self.door(i)
        lateral = np.array([-direction[1], direction[0]])
        return self.xy(self.route[k]) + lateral*side*.42 - direction*.12


def generate(split, index):
    if split not in SPLITS or not 0 <= index < SPLITS[split][1]:
        raise ValueError("layout index must belong to an explicit disjoint split")
    seed = SPLITS[split][0] + index
    rng = np.random.default_rng(seed)
    cells = [(x, y) for x in range(6) for y in range(6)]
    candidates = [edge(c, (c[0]+dx, c[1]+dy)) for c in cells
                  for dx, dy in ((1, 0), (0, 1)) if (c[0]+dx, c[1]+dy) in cells]
    for _ in range(10000):
        parent = {c: c for c in cells}

        def root(c):
            while parent[c] != c:
                c = parent[c]
            return c

        edges = []
        for idx in rng.permutation(len(candidates)):
            a, b = candidates[idx]
            if root(a) != root(b):
                parent[root(a)] = root(b)
                edges.append((a, b))
        adjacency = {c: [] for c in cells}
        for a, b in edges:
            adjacency[a].append(b)
            adjacency[b].append(a)
        paths = {(0, 0): ((0, 0),)}
        queue = [(0, 0)]
        for c in queue:
            for nxt in adjacency[c]:
                if nxt not in paths:
                    paths[nxt] = paths[c] + (nxt,)
                    queue.append(nxt)
        valid = [p for p in paths.values() if 14 <= len(p) <= 21
                 and len(adjacency[p[-1]]) == 1
                 and sum(len(adjacency[c]) >= 3 for c in p[1:-1]) == 7
                 and np.any(np.diff(np.asarray(p), axis=0)[1:] != np.diff(np.asarray(p), axis=0)[:-1], axis=1).sum() >= 4]
        if valid:
            route = valid[int(rng.integers(len(valid)))]
            # Vary pitch, branch geometry, goal, plate side, gates and pickup.
            gates = (int(rng.integers(3, 6)), int(rng.integers(8, len(route)-2)))
            return Layout(split, seed, float(rng.choice([1.5, 1.6, 1.7])),
                          tuple(sorted(edges)), route, gates,
                          tuple(int(x) for x in rng.choice([-1, 1], 2)),
                          int(rng.integers(1, gates[0])))
    raise RuntimeError(f"could not generate seven-decision maze for {seed}")


def wall_segments(layout):
    links = set(layout.edges)
    seen = set()
    for c in sorted(layout.adjacency):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            other = (c[0]+dx, c[1]+dy)
            key = edge(c, other)
            if key in links or key in seen:
                continue
            seen.add(key)
            center = layout.xy(c) + np.array([dx, dy])*layout.cell_m/2
            size = (.04, layout.cell_m/2+.04) if dx else (layout.cell_m/2+.04, .04)
            yield center, size


def world_xml(layout):
    parts = ['<mujoco model="mission7"><compiler angle="radian"/>'
             '<option timestep=".0005"/><visual><global offwidth="960" offheight="540"/></visual>'
             '<worldbody><light pos="4 4 9" dir="0 0 -1"/>'
             '<geom name="floor" type="plane" size="12 12 .05" rgba=".22 .25 .28 1"/>']
    for i, ((x, y), (sx, sy)) in enumerate(wall_segments(layout)):
        parts.append(f'<geom name="wall_m7_{i}" type="box" pos="{x} {y} .55" '
                     f'size="{sx} {sy} .55" rgba=".45 .51 .57 1"/>')
    for i in range(2):
        (x, y), direction = layout.door(i)
        sx, sy = (.045, layout.cell_m/2) if direction[0] else (layout.cell_m/2, .045)
        parts.append(f'<body name="gate_{i}" mocap="true" pos="{x} {y} .55">'
                     f'<geom name="door_{i}" type="box" size="{sx} {sy} .55" '
                     'rgba=".8 .4 .12 1"/></body>')
        for side in (-1, 1):
            x, y = layout.plate(i, side)
            correct = side == layout.correct_sides[i]
            shape = 'type="cylinder" size=".24 .015"' if correct else 'type="box" size=".24 .24 .015"'
            parts.append(f'<geom name="plate_{i}_{side}" {shape} pos="{x} {y} .015" '
                         'rgba=".75 .65 .2 1"/>')
    x, y = layout.xy(layout.route[layout.object_index])
    parts.append(f'<body name="parcel" pos="{x} {y} .075"><freejoint name="parcel_free"/>'
                 '<geom name="parcel_geom" type="box" size=".07 .07 .07" mass=".12" '
                 'rgba=".85 .3 .3 1"/></body>')
    x, y = layout.xy(layout.route[-1])
    # A visible geometric marker, without oracle class IDs in policy observations.
    parts.append(f'<geom name="drop_zone" type="cylinder" pos="{x} {y} .002" size=".36 .002" '
                 'contype="0" conaffinity="0" rgba=".1 .7 .4 1"/>')
    for side in (-1, 1):
        parts.append(f'<geom name="goal_post_{side}" type="capsule" '
                     f'fromto="{x+.52} {y+side*.35} .05 {x+.52} {y+side*.35} .85" '
                     'size=".045" rgba=".1 .7 .4 1"/>')
    return ''.join(parts) + '</worldbody></mujoco>'
