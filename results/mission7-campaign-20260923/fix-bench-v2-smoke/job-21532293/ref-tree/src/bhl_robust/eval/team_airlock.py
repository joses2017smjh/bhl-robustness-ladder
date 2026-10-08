"""A shared-state 2/3-humanoid inspection and airlock coordination task.

The high-level controller is an oracle-map baseline, not a learned perception
policy. Physical walking is provided by frozen Isaac-trained BHL policies.
No object grasping, lifting, or cloth manipulation is implied by this task.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class TeamAirlock:
    """All members inspect their station together, then share one doorway.

    A single robot cannot finish the task. The door latches only after every
    assigned station is occupied continuously for ``dwell_s``. Crossing is
    serialized; all members must subsequently hold their own exit station.
    """

    crew: int
    mode: str = "coordinated"
    station_radius: float = 0.23
    dwell_s: float = 0.6
    door_open: bool = False
    joint_dwell: float = 0.0
    exit_dwell: float = 0.0
    released_at: float | None = None
    completed_at: float | None = None
    active: int = 0
    visited: np.ndarray = field(init=False)
    stages: np.ndarray = field(init=False)

    def __post_init__(self):
        if self.crew not in (2, 3):
            raise ValueError("team task supports exactly two or three robots")
        if self.mode not in ("coordinated", "no_wait", "withhold_last"):
            raise ValueError(f"unknown team control mode {self.mode}")
        ys = np.linspace(-0.9, 0.9, self.crew)
        self.starts = np.column_stack([np.zeros(self.crew), ys])
        self.stations = np.column_stack([np.full(self.crew, 0.7), ys])
        self.exits = np.column_stack([np.full(self.crew, 3.6), ys])
        self.visited = np.zeros(self.crew, dtype=bool)
        self.stages = np.zeros(self.crew, dtype=int)
        # Merge before the wall, traverse the opening, and spread only after
        # the rear of the body has cleared the wall. No diagonal corner cuts.
        self.crossing = np.array([[1.15, 0.0], [2.7, 0.0]])

    def update(self, positions, upright, dt: float, time_s: float):
        positions = np.asarray(positions)
        upright = np.asarray(upright, dtype=bool)
        if positions.shape != (self.crew, 2) or upright.shape != (self.crew,):
            raise ValueError("mission needs one planar position and health flag per member")
        on_station = np.linalg.norm(positions - self.stations, axis=1) < self.station_radius
        self.visited |= on_station & upright
        if not self.door_open:
            self.joint_dwell = self.joint_dwell + dt if np.all(on_station & upright) else 0.0
            if self.joint_dwell + 1e-9 >= self.dwell_s:
                self.door_open = True
                self.released_at = time_s

        for i in range(self.crew):
            authorized = self.door_open and i <= self.active
            if self.mode == "no_wait":
                authorized = bool(self.visited[i])
            if authorized and self.stages[i] < len(self.crossing):
                if np.linalg.norm(positions[i] - self.crossing[self.stages[i]]) < 0.18:
                    self.stages[i] += 1
            # Release the next member only once this member has moved sideways
            # out of the shared exit lane, not just after touching the door.
            if i == self.active and self.stages[i] == len(self.crossing):
                if np.linalg.norm(positions[i] - self.exits[i]) < self.station_radius:
                    self.active += 1

        at_exit = np.linalg.norm(positions - self.exits, axis=1) < self.station_radius
        finished = self.door_open and np.all(self.stages == len(self.crossing))
        self.exit_dwell = self.exit_dwell + dt if finished and np.all(at_exit & upright) else 0.0
        if self.completed_at is None and self.exit_dwell + 1e-9 >= self.dwell_s:
            self.completed_at = time_s

    def targets(self):
        targets = self.stations.copy()
        for i in range(self.crew):
            authorized = self.door_open and i <= self.active
            if self.mode == "no_wait":
                authorized = bool(self.visited[i])
            if authorized:
                targets[i] = (self.crossing[self.stages[i]] if self.stages[i] < len(self.crossing)
                              else self.exits[i])
        if self.mode == "withhold_last":
            targets[-1] = self.starts[-1]
        return targets


def velocity_command(position, yaw, target, max_speed=0.28):
    """Bounded body-frame holonomic command; hold the original facing."""
    error = np.asarray(target) - np.asarray(position)
    distance = float(np.linalg.norm(error))
    world = 1.2 * error
    if np.linalg.norm(world) > max_speed:
        world *= max_speed / np.linalg.norm(world)
    if distance < 0.045:
        world[:] = 0.0
    c, s = np.cos(yaw), np.sin(yaw)
    body = np.array([c * world[0] + s * world[1], -s * world[0] + c * world[1]])
    return np.array([*body, np.clip(-1.2 * yaw, -0.35, 0.35)])


def world_xml(crew):
    """Physical dividing wall/door; coloured floor marks are visual-only."""
    mission = TeamAirlock(crew)
    markers = []
    for name, points, color in (("station", mission.stations, "0.9 0.6 0.1 0.8"),
                                ("exit", mission.exits, "0.1 0.8 0.4 0.8")):
        for i, (x, y) in enumerate(points):
            markers.append(f'<geom name="{name}_{i}" type="cylinder" size="0.23 0.001" '
                           f'pos="{x} {y} 0.002" rgba="{color}" contype="0" conaffinity="0"/>')
    return '''<mujoco model="bhl-team-airlock">
      <compiler angle="radian"/>
      <option timestep="0.0005"/>
      <visual><global offwidth="1280" offheight="720"/></visual>
      <worldbody>
        <light pos="1 0 5" dir="0 0 -1"/>
        <geom name="floor" type="plane" size="8 8 0.05" rgba="0.23 0.27 0.3 1"/>
        <geom name="wall_left" type="box" pos="2.0 -1.65 0.55" size="0.08 0.9 0.55"/>
        <geom name="wall_right" type="box" pos="2.0 1.65 0.55" size="0.08 0.9 0.55"/>
        <geom name="side_left" type="box" pos="1.8 -2.6 0.55" size="3.0 0.05 0.55"/>
        <geom name="side_right" type="box" pos="1.8 2.6 0.55" size="3.0 0.05 0.55"/>
        <body name="airlock" mocap="true" pos="2 0 0.55">
          <geom name="door" type="box" size="0.08 0.75 0.55" rgba="0.8 0.25 0.15 1"/>
        </body>
    ''' + "\n".join(markers) + "</worldbody></mujoco>"
