"""SYNTHETIC stand-in for ``bhl_robust.eval.waiter_env.WaiterEnv`` (contract API only; no physics, no robot).

It exists so the phase 3 pipeline (dataset -> SmolVLA fine-tune -> closed-loop rollout) can be exercised end to
end before phase 2 delivers the MuJoCo scene. Nothing measured on it says anything about the waiter task.

Contract surface reproduced (``docs/WAITER_INTERFACES.md``):
- ``StubWaiterEnv(deploy=None, upstream=None, cache_dir=None, render=True, fixed_base=False)``
- ``reset(seed, instruction_id=None) -> obs``; ``obs = {"image": (224,224,3) uint8, "state": (30,) f32,
  "instruction": str}``; the layout is a pure function of ``seed``.
- ``step(a) -> (obs, reward, done, info)``: 0.08 s per step; reward 1.0 on the success step; done on success, fall
  or the 60 s limit; ``info`` keys success, fell, stage, tumbler_on_tray, tray_on_target, t.
- ``oracle()`` -> poses (for the synthetic expert only).

Synthetic task: four stages. 0 reach (arms to the goal pose of the instructed tumbler colour), 1 grip (both
grippers closed), 2 carry (base translated onto the target zone of the instructed side; its distance varies with
the seed and is drawn in the image), 3 release (grippers open, base still) and hold 1.0 s -> success. A jump in
commanded base velocity > 0.5 m/s in one step "trips" the robot (fell). Everything needed to succeed is observable
(instruction + image), so a working fine-tune pipeline can learn it; a broken one (e.g. mismatched normalisation)
cannot.
"""
from __future__ import annotations

import numpy as np

SYNTHETIC_TEMPLATES = {
    "T0": "pick up the {colour} tumbler, put it on the tray and bring the tray to the {side} table",
    "T1": "put the {colour} tumbler on the tray, then carry the tray to the {side} table",
    "T2": "take the {colour} tumbler by its handle, set it on the tray and deliver the tray to the {side} table",
    "T3": "bring the tray with the {colour} tumbler on it to the {side} table",
}
SYNTHETIC_SLOTS = {"colour": ["red", "blue", "green", "yellow"], "side": ["left", "right"]}
SYNTHETIC_HELD_OUT = ["T3"]
COLOURS_RGB = {"red": (220, 40, 40), "blue": (40, 70, 220), "green": (40, 180, 70), "yellow": (230, 210, 40)}

DT = 0.08
TIME_LIMIT_S = 60.0
BASE_LIMITS = np.array([0.4, 0.3, 0.6], dtype=np.float32)
HOLD_S = 1.0


def templates_json() -> dict:
    return {"templates": dict(SYNTHETIC_TEMPLATES), "slots": {k: list(v) for k, v in SYNTHETIC_SLOTS.items()},
            "held_out": list(SYNTHETIC_HELD_OUT), "synthetic": True}


class StubWaiterEnv:
    synthetic = True

    def __init__(self, deploy=None, upstream=None, cache_dir=None, render=True, fixed_base=False):
        self.render_enabled = bool(render)
        self.fixed_base = bool(fixed_base)
        self._seed = None

    # -- layout -------------------------------------------------------------------------------------------------
    def _layout(self, seed: int):
        rng = np.random.default_rng([int(seed), 0xC0FFEE])
        colour_i = int(rng.integers(4))
        colour = SYNTHETIC_SLOTS["colour"][colour_i]
        side = SYNTHETIC_SLOTS["side"][int(rng.integers(2))]
        # arm goal is a fixed function of the instructed colour (language-grounded, not seed noise)
        arm_goal = np.random.default_rng([colour_i, 0xA4]).uniform(-0.6, 0.6, size=10).astype(np.float32)
        target_y = (1.0 if side == "left" else -1.0) * float(rng.uniform(0.6, 1.0))
        background = rng.integers(60, 120, size=3)
        return {"colour": colour, "side": side, "arm_goal": arm_goal, "target_y": target_y,
                "target_x": 0.0 if self.fixed_base else float(rng.uniform(-0.3, 0.3)),
                "background": background, "default_template": sorted(SYNTHETIC_TEMPLATES)[int(rng.integers(4))]}

    def reset(self, seed: int, instruction_id=None):
        self._seed = int(seed)
        self.layout = self._layout(seed)
        if instruction_id is None:
            template = self.layout["default_template"]
        else:
            keys = sorted(SYNTHETIC_TEMPLATES)
            template = instruction_id if isinstance(instruction_id, str) else keys[int(instruction_id) % len(keys)]
        self.template_id = template
        self.instruction = SYNTHETIC_TEMPLATES[template].format(colour=self.layout["colour"], side=self.layout["side"])
        self.q = np.zeros(24, dtype=np.float32)
        self.base = np.zeros(3, dtype=np.float64)  # x, y, yaw
        self.prev_cmd = np.zeros(3, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.stage = 0
        self.hold = 0.0
        self.steps = 0
        self.done = False
        self.fell = False
        self.success = False
        self._noise = np.random.default_rng([int(seed), 7])
        return self._obs()

    # -- dynamics -----------------------------------------------------------------------------------------------
    def step(self, action):
        if self._seed is None or self.done:
            raise RuntimeError("step() before reset() or after done")
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        if a.shape != (15,) or not np.isfinite(a).all():
            raise ValueError(f"action must be 15 finite floats, got shape {a.shape}")
        cmd = np.clip(a[:3], -BASE_LIMITS, BASE_LIMITS)
        if self.fixed_base:
            cmd = np.zeros(3, dtype=np.float32)
        arms = np.clip(a[3:13], -2.5, 2.5)
        grip = np.clip(a[13:15], 0.0, 1.0)
        if np.abs(cmd - self.prev_cmd).max() > 0.5:
            self.fell = True
        self.prev_cmd = cmd
        self.q[0:10] += 0.35 * (arms - self.q[0:10])
        self.q[22:24] += 0.5 * (1.2 * grip - self.q[22:24])
        self.q[10:22] = 0.01 * self._noise.standard_normal(12)
        c, s = np.cos(self.base[2]), np.sin(self.base[2])
        self.base[0] += DT * (c * cmd[0] - s * cmd[1])
        self.base[1] += DT * (s * cmd[0] + c * cmd[1])
        self.base[2] += DT * cmd[2]
        self.ang_vel = np.array([0.0, 0.0, cmd[2]], dtype=np.float32)
        self.steps += 1
        self._advance_stage(grip)
        t = self.steps * DT
        reward = 0.0
        if self.success:
            reward = 1.0
        self.done = self.success or self.fell or t >= TIME_LIMIT_S - 1e-9
        info = {"success": self.success, "fell": self.fell, "stage": int(self.stage),
                "tumbler_on_tray": self.stage >= 2, "tray_on_target": self._at_target(), "t": float(t),
                "synthetic": True}
        return self._obs(), reward, self.done, info

    def _at_target(self) -> bool:
        return bool(np.hypot(self.base[0] - self.layout["target_x"], self.base[1] - self.layout["target_y"]) < 0.15)

    def _advance_stage(self, grip):
        arm_err = np.abs(self.q[0:10] - self.layout["arm_goal"]).max()
        closed = self.q[22:24].min() > 0.9
        opened = self.q[22:24].max() < 0.15
        if self.stage == 0 and arm_err < 0.1:
            self.stage = 1
        elif self.stage == 1 and closed:
            self.stage = 2
        elif self.stage == 2 and self._at_target() and np.abs(self.prev_cmd).max() < 0.05:
            self.stage = 3
        elif self.stage == 3:
            if opened and np.abs(self.prev_cmd).max() < 0.05:
                self.hold += DT
            else:
                self.hold = 0.0
            if self.hold >= HOLD_S - 1e-9:
                self.success = True

    # -- observations -------------------------------------------------------------------------------------------
    def _obs(self):
        tilt = np.array([0.05 * self.prev_cmd[0], 0.05 * self.prev_cmd[1], 0.0])
        gravity = np.array([tilt[0], tilt[1], -1.0]) / np.linalg.norm([tilt[0], tilt[1], -1.0])
        state = np.concatenate([self.q, gravity, self.ang_vel]).astype(np.float32)
        return {"image": self._render(), "state": state, "instruction": self.instruction}

    def _render(self):
        img = np.empty((224, 224, 3), dtype=np.uint8)
        if not self.render_enabled:
            img[:] = 0
            return img
        img[:] = self.layout["background"].astype(np.uint8)
        # target table band on the left or right edge
        x0 = 0 if self.layout["side"] == "left" else 196
        img[60:160, x0:x0 + 28] = (150, 110, 70)
        # target zone outline: where the tray must end up (seed-dependent distance)
        ty0 = int(np.clip(112 - 60 * self.layout["target_x"], 20, 204))
        tx0 = int(np.clip(112 - 80 * self.layout["target_y"], 20, 204))
        img[ty0 - 21:ty0 + 21, tx0 - 33:tx0 + 33] = (250, 250, 250)
        img[ty0 - 18:ty0 + 18, tx0 - 30:tx0 + 30] = self.layout["background"].astype(np.uint8)
        # tray moves with the base (top-down synthetic view)
        cy = int(np.clip(112 - 60 * self.base[0], 20, 204))
        cx = int(np.clip(112 - 80 * self.base[1], 20, 204))
        img[cy - 18:cy + 18, cx - 30:cx + 30] = (190, 190, 190)
        # tumbler: offset by the left-arm pose until it is on the tray
        off = 0 if self.stage >= 2 else int(np.clip(40 * self.q[0], -60, 60))
        ty, tx = int(np.clip(cy - 30 + off, 8, 216)), int(np.clip(cx + off, 8, 216))
        img[ty - 8:ty + 8, tx - 8:tx + 8] = COLOURS_RGB[self.layout["colour"]]
        # gripper indicators
        img[200:216, 8:24] = int(200 * self.q[22] / 1.2)
        img[200:216, 200:216] = int(200 * self.q[23] / 1.2)
        noise = self._noise.integers(0, 3, size=(224, 224, 1), dtype=np.uint8)
        return np.minimum(img.astype(np.uint16) + noise, 255).astype(np.uint8)

    def oracle(self):
        return {"synthetic": True, "arm_goal": self.layout["arm_goal"].copy(), "target_xy":
                (self.layout["target_x"], self.layout["target_y"]), "base": self.base.copy(), "q": self.q.copy(),
                "stage": int(self.stage)}


class StubExpert:
    """SYNTHETIC scripted expert for the stub (oracle access, like phase 2's expert)."""

    def __init__(self, env: StubWaiterEnv, noise: float = 0.0, seed: int = 0):
        self.env = env
        self.noise = noise
        self.rng = np.random.default_rng(seed)
        self.prev = np.zeros(3, dtype=np.float32)

    def act(self):
        o = self.env.oracle()
        a = np.zeros(15, dtype=np.float32)
        a[3:13] = o["arm_goal"]
        stage = o["stage"]
        if stage >= 1:
            a[13:15] = 1.0
        if stage == 2:
            tx, ty = o["target_xy"]
            dx, dy = tx - o["base"][0], ty - o["base"][1]
            yaw = o["base"][2]
            c, s = np.cos(yaw), np.sin(yaw)
            v = np.array([c * dx + s * dy, -s * dx + c * dy, -yaw], dtype=np.float32) * 1.5
            if np.hypot(dx, dy) < 0.08:
                v[:] = 0.0
            v = np.clip(v, -BASE_LIMITS * 0.9, BASE_LIMITS * 0.9)
            a[:3] = np.clip(v, self.prev - 0.1, self.prev + 0.1)  # ramp: never trips
        if stage == 3:
            a[13:15] = 0.0
        if self.noise > 0:
            a[3:13] += self.noise * self.rng.standard_normal(10).astype(np.float32)
        a[:3] = np.clip(a[:3], -BASE_LIMITS, BASE_LIMITS)
        self.prev = a[:3].copy()
        return a
