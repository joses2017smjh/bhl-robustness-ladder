"""STUB Waiter environment implementing the frozen WaiterEnv contract (TOY task, not the waiter task).

Contract (docs/WAITER_INTERFACES.md): ``reset(seed, instruction_id=None) -> obs``,
``step(a) -> (obs, reward, done, info)`` with a 15-D action at 12.5 Hz (0.08 s per step), a 30-D state,
a 224x224 RGB ``image``, an ``instruction`` string, ``reward`` 1.0 on the step success is reached,
``done`` on success / fall / time limit, and ``info`` keys
``success, fell, stage, tumbler_on_tray, tray_on_target, t``; ``oracle()`` returns poses.

TOY task (labelled as such everywhere it is used): two markers (red, green) are placed at
seed-dependent positions on a 2-D "table plane"; the instruction names one of them. The action's
left-arm entries ``a[3], a[4]`` are read as a planar hand TARGET (toy kinematics, metres), which the
hand follows with a first-order lag. Success = hand within ``tol`` of the instructed marker with both
grippers open (``a[13:15] < 0.5``), held for ``hold_s`` seconds. "Fell" = a hand target farther than
``fall_radius`` from the origin (exercises the fall/terminal path). Pure numpy, no MuJoCo.

This file never imports torch, so it runs in any interpreter with numpy (env-worker processes).
"""
from __future__ import annotations

import numpy as np

STEP_DT = 0.08          # contract: high-level step 0.08 s (12.5 Hz)
ACTION_DIM = 15
STATE_DIM = 30
IMAGE_HW = 224
COLORS = ("red", "green")
_RGB = {"red": (220, 30, 30), "green": (30, 180, 30), "hand": (30, 60, 230)}
# Contract action clipping (base velocity) is applied; arm entries are clipped to [-1, 1] (toy).
_V_LIMITS = np.array([0.4, 0.3, 0.6], np.float32)
_YY, _XX = np.mgrid[0:IMAGE_HW, 0:IMAGE_HW]


def instruction_for(color: str) -> str:
    return f"move the left hand to the {color} marker"


class StubWaiterEnv:
    """TOY env with the WaiterEnv API. Extra keyword arguments are TOY parameters."""

    toy = True

    def __init__(self, deploy=None, upstream=None, cache_dir=None, render=True, fixed_base=False, *,
                 time_limit_s: float = 16.0, tol: float = 0.06, hold_s: float = 0.4, lag_s: float = 0.1,
                 marker_box: float = 0.3, min_sep: float = 0.25, fall_radius: float = 0.9):
        self.render_enabled = bool(render)
        self.fixed_base = bool(fixed_base)
        self.time_limit_s = float(time_limit_s)
        self.tol = float(tol)
        self.hold_steps = max(1, int(round(hold_s / STEP_DT)))
        self.alpha = float(1.0 - np.exp(-STEP_DT / lag_s))
        self.marker_box = float(marker_box)
        self.min_sep = float(min_sep)
        self.fall_radius = float(fall_radius)
        self._max_steps = int(round(self.time_limit_s / STEP_DT))
        self._seed = None

    # ---------------------------------------------------------------- contract API
    def reset(self, seed, instruction_id=None):
        rng = np.random.default_rng(int(seed))
        while True:
            markers = rng.uniform(-self.marker_box, self.marker_box, size=(2, 2))
            if np.linalg.norm(markers[0] - markers[1]) >= self.min_sep:
                break
        iid = int(rng.integers(2)) if instruction_id is None else int(instruction_id) % 2
        self._seed = int(seed)
        self.markers = markers.astype(np.float64)
        self.instruction_id = iid
        self.color = COLORS[iid]
        self.goal = self.markers[iid].copy()
        self.hand = np.zeros(2)
        self.target = np.zeros(2)
        self.grip = np.zeros(2)
        self.base_cmd = np.zeros(3)
        self.steps = 0
        self.hold = 0
        self.stage_max = 0
        return self._obs()

    def step(self, a):
        a = np.asarray(a, np.float64).reshape(ACTION_DIM)
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite action")
        self.base_cmd = np.clip(a[0:3], -_V_LIMITS, _V_LIMITS)
        self.target = np.clip(a[3:5], -1.0, 1.0)
        self.grip = np.clip(a[13:15], 0.0, 1.0)
        self.hand = self.hand + self.alpha * (self.target - self.hand)
        self.steps += 1
        t = self.steps * STEP_DT
        dist = float(np.linalg.norm(self.hand - self.goal))
        fell = bool(np.linalg.norm(self.target) > self.fall_radius)
        near = dist < self.tol and bool(np.all(self.grip < 0.5))
        self.hold = self.hold + 1 if near else 0
        success = (self.hold >= self.hold_steps) and not fell
        stage = 3 if self.hold > 0 else (2 if dist < self.tol else (1 if dist < 3 * self.tol else 0))
        self.stage_max = max(self.stage_max, stage)
        done = bool(success or fell or self.steps >= self._max_steps)
        info = {"success": bool(success), "fell": fell, "stage": int(stage),
                "tumbler_on_tray": bool(dist < self.tol), "tray_on_target": bool(success),
                "t": float(t), "toy": True}
        return self._obs(), (1.0 if success else 0.0), done, info

    def oracle(self):
        return {"markers": self.markers.copy(), "goal": self.goal.copy(), "hand": self.hand.copy(),
                "target": self.target.copy(), "instruction_id": self.instruction_id, "toy": True}

    # ---------------------------------------------------------------- rendering / state
    def _state(self):
        s = np.zeros(STATE_DIM, np.float32)
        s[3:5] = self.hand            # toy: the two left-arm "joint positions" carry the hand position
        s[22:24] = self.grip * 1.20   # gripper joint angles (rad), contract scale
        s[24:27] = (0.0, 0.0, -1.0)   # projected gravity (upright)
        return s

    def _px(self, xy):
        # table plane [-1, 1]^2 -> pixel coordinates (row, col); +y is up in the image
        col = (xy[0] + 1.0) * 0.5 * (IMAGE_HW - 1)
        row = (1.0 - (xy[1] + 1.0) * 0.5) * (IMAGE_HW - 1)
        return row, col

    def _image(self):
        img = np.full((IMAGE_HW, IMAGE_HW, 3), 245, np.uint8)
        if not self.render_enabled:
            return img
        for xy, rgb, r in ((self.markers[0], _RGB["red"], 9), (self.markers[1], _RGB["green"], 9),
                           (self.hand, _RGB["hand"], 6)):
            row, col = self._px(xy)
            r0, r1 = max(0, int(row - r - 1)), min(IMAGE_HW, int(row + r + 2))
            c0, c1 = max(0, int(col - r - 1)), min(IMAGE_HW, int(col + r + 2))
            m = (_YY[r0:r1, c0:c1] - row) ** 2 + (_XX[r0:r1, c0:c1] - col) ** 2 <= r * r
            img[r0:r1, c0:c1][m] = rgb
        return img

    def _obs(self):
        return {"image": self._image(), "state": self._state(), "instruction": instruction_for(self.color)}
