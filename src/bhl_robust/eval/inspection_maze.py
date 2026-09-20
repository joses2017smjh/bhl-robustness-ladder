"""A wall-bounded dogleg maze with ordered inspection and a dead-end branch.

The route is an oracle-map baseline. Inspection is position-and-dwell, not
classification or physical button manipulation. Each 1.6 m cell gives BHL
clearance to change travel direction using its trained holonomic gait.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


CELL_M = 1.6
OPEN_CELLS = ((0, 0), (1, 0), (1, 1), (2, 1), (3, 1), (1, -1))
STATIONS = ((1.0, 0.0), (3.2, 1.6))
EXIT = (4.8, 1.6)
DEAD_END = (1.6, -1.6)


@dataclass
class InspectionMaze:
    route: str = "ordered"
    radius_m: float = .23
    dwell_s: float = .6
    stage: int = 0
    held_s: float = 0.0
    completed_at: float | None = None
    failure: str | None = None
    station_times: list = field(default_factory=list)
    last_update_s: float | None = None

    def __post_init__(self):
        if self.route not in ("ordered", "wrong_branch"):
            raise ValueError("route must be ordered or wrong_branch")
        self.waypoints = [(1.0, 0.0, "inspect_A", True),
                          (1.6, 0.0, "turn_north", False),
                          (1.6, 1.6, "turn_east", False),
                          (3.2, 1.6, "inspect_B", True),
                          (4.8, 1.6, "exit", True)]

    def target(self):
        if self.route == "wrong_branch" and self.stage >= 1:
            return np.array(DEAD_END)
        return np.array(self.waypoints[min(self.stage, len(self.waypoints)-1)][:2])

    @property
    def active_label(self):
        return self.waypoints[min(self.stage, len(self.waypoints)-1)][2]

    def update(self, position, upright, dt, now):
        if self.completed_at is not None or self.failure is not None or self.last_update_s == now:
            return
        if dt <= 0 or not np.isfinite(position).all() or not np.isfinite(now):
            raise ValueError("finite pose/time and positive sample interval required")
        if self.last_update_s is not None and now < self.last_update_s:
            raise ValueError("mission timestamps must be monotonic")
        self.last_update_s = now
        if not upright:
            self.failure = "fall"
            return
        position = np.asarray(position)
        # Evaluation guards; neither changes the low-level action or hides a
        # physical wall. The body center must stay within the open-cell union.
        if not any(np.all(np.abs(position - np.array(cell)*CELL_M) <= CELL_M/2)
                   for cell in OPEN_CELLS):
            self.failure = "left_maze"
            return
        if .8 < position[0] < 2.4 and position[1] < -.95:
            self.failure = "dead_end_entered"
            return
        x, y, label, needs_dwell = self.waypoints[self.stage]
        inside = np.linalg.norm(position - [x, y]) < self.radius_m
        self.held_s = self.held_s + dt if inside else 0.0
        if inside and (not needs_dwell or self.held_s + 1e-9 >= self.dwell_s):
            if label.startswith("inspect_"):
                self.station_times.append({"station": label, "time_s": now})
            self.stage += 1
            self.held_s = 0.0
            if label == "exit":
                self.completed_at = now


def wall_segments():
    """Boundary segments of the open-cell union; shared edges remain open."""
    cells = set(OPEN_CELLS)
    segments = []
    for ix, iy in sorted(cells):
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            if (ix+dx, iy+dy) in cells:
                continue
            center = ((ix+dx*.5)*CELL_M, (iy+dy*.5)*CELL_M)
            half_size = (.04, CELL_M/2+.04) if dx else (CELL_M/2+.04, .04)
            segments.append((center, half_size))
    return segments


def world_xml():
    geoms = []
    for i, ((x, y), (sx, sy)) in enumerate(wall_segments()):
        geoms.append(f'<geom name="wall_maze_{i}" type="box" pos="{x} {y} .55" '
                     f'size="{sx} {sy} .55" rgba=".46 .51 .57 1"/>')
    for i, (x, y) in enumerate(STATIONS):
        geoms.append(f'<geom name="inspection_{i}" type="cylinder" pos="{x} {y} .002" '
                     'size=".23 .001" rgba=".96 .63 .12 1" contype="0" conaffinity="0"/>')
    geoms.append(f'<geom name="exit_marker" type="cylinder" pos="{EXIT[0]} {EXIT[1]} .002" '
                 'size=".23 .001" rgba=".15 .8 .35 1" contype="0" conaffinity="0"/>')
    geoms.append(f'<geom name="dead_end_marker" type="cylinder" pos="{DEAD_END[0]} {DEAD_END[1]} .002" '
                 'size=".23 .001" rgba=".85 .15 .12 1" contype="0" conaffinity="0"/>')
    return '''<mujoco model="bhl-inspection-maze">
      <compiler angle="radian"/><option timestep=".0005"/>
      <visual><global offwidth="1280" offheight="720"/></visual>
      <worldbody><light pos="2 0 6" dir="0 0 -1"/>
      <geom name="floor" type="plane" size="9 9 .05" rgba=".2 .24 .28 1"/>
    ''' + "\n".join(geoms) + "</worldbody></mujoco>"
