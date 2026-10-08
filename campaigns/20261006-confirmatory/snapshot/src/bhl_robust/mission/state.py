"""Scoring-only state machine, sampled once per physical control interval."""
from dataclasses import dataclass, field
import numpy as np


@dataclass
class MissionState:
    stage: str
    open: list = field(default_factory=lambda: [False, False])
    crossed: list = field(default_factory=lambda: [False, False])
    activation_s: list = field(default_factory=lambda: [None, None])
    wrong_buttons: int = 0
    retries: int = 0
    acquired: bool = False
    carrying: bool = False
    drops: int = 0
    distance_m: float = 0.
    carried_m: float = 0.
    collisions: int = 0
    dead_end_entries: int = 0
    hold_s: float = 0.
    completed_s: float | None = None
    failure: str | None = None
    last_s: float = 0.
    previous_buttons: set = field(default_factory=set)

    def update(self, *, now, dt, upright, collision, button_contacts, activate,
               crossing, xy_delta, goal_inside, slow, object_inside, object_slow):
        if self.completed_s is not None or self.failure is not None or now == self.last_s:
            return 0.
        if not np.isfinite([now, dt, xy_delta]).all() or dt <= 0 or now < self.last_s:
            raise ValueError("finite increasing time required")
        if abs(now-self.last_s-dt) > 1e-6:
            raise ValueError("dwell interval must match elapsed simulation time")
        self.last_s = now
        self.distance_m += xy_delta
        self.carried_m += xy_delta if self.carrying else 0.
        self.collisions += int(collision)
        reward = -.01*dt - .2*int(collision)
        if not upright:
            self.failure = "fall"
            self.hold_s = 0.
            return reward - 5.
        if self.collisions > 5:
            self.failure = "excessive_collision"
            self.hold_s = 0.
            return reward - 5.
        pressed = set(button_contacts) if activate else set()
        for door, correct in pressed-self.previous_buttons:
            if self.stage not in ("doors", "transport"):
                continue
            if not correct:
                self.wrong_buttons += 1
                reward -= 1.
            elif self.open[door]:
                self.retries += 1
            else:
                self.open[door] = True
                self.activation_s[door] = now
                reward += 2.
        self.previous_buttons = pressed
        for door in crossing:
            if self.open[door] and not self.crossed[door]:
                self.crossed[door] = True
                reward += 2.
        ready = self.stage not in ("doors", "transport") or all(self.crossed)
        valid = ready and slow and goal_inside
        if self.stage == "transport":
            # Arrival alone never passes transport. The released object must settle.
            valid = ready and self.acquired and not self.carrying and object_inside and object_slow
        self.hold_s = self.hold_s + dt if valid else 0.
        if self.hold_s + 1e-9 >= 1.0:
            self.completed_s = now
            reward += 10.
        return reward

    def acquire(self):
        if self.carrying:
            return 0.
        first = not self.acquired
        self.acquired = self.carrying = True
        return 2. if first else 0.

    def release(self, accidental=False):
        if self.carrying:
            self.carrying = False
            self.drops += int(accidental)
