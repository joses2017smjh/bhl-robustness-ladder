"""Waiter phase 2 environment (docs/WAITER_INTERFACES.md): `WaiterEnv`, the contract every later phase uses.

The scene is `waiter_scene` (robot, pick table A with two handled tumblers and a handled tray, two delivery tables,
SIMULATED head camera). The low level is the selected phase 1 WBC (`gait_clock.make_controller` on a stamped
deploy.yaml: the learned leg policy at 25 Hz, arms and grippers PD-driven to the commanded targets). One `step` is
0.08 s = 2 WBC steps.

Action (15, float32): [vx, vy, wz] body-frame base velocity command (clipped |vx| <= 0.4, |vy| <= 0.3,
|wz| <= 0.6), the 10 arm targets (absolute rad, clipped to the joint limits), the 2 gripper closures in [0, 1]
(x 1.20 rad). State (30): the 24 joint positions (deploy order), base projected gravity, base angular velocity.

Success (frozen, the gate of every phase): the instructed tumbler upright (tilt < 20 deg) resting on the tray, the
tray resting on the instructed delivery table with its centre inside the target zone, both grippers open, the robot
upright; held 1.0 s. `reward` is 1.0 on the step success is reached, else 0.0; `done` on success, a fall (base tilt
>= 0.78 rad) or the 60 s limit.

Stance hold (low-level, declared, `stance_hold=True` by default): while the commanded base velocity is ~0 the WBC's
gait-clock INPUT is held at the next double-support phase (both feet stance by the training reward), so the robot
stands instead of stepping in place. The clock always advanced in training: this is out of the WBC's training
distribution and is measured by the phase 2 dev tests and gate, never assumed. Any non-zero command releases it.

`fixed_base=True` welds the robot base where it stands (development only; never a scored run).
"""

from __future__ import annotations

import math
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.eval import waiter_scene as W

STEP_DT = 0.08
WBC_STEPS_PER_STEP = 2
TIME_LIMIT_S = 60.0
TILT_FALL = 0.78
V_LIMITS = np.array([0.4, 0.3, 0.6], np.float32)
GRIPPER_CLOSED = 1.20
GRIPPER_OPEN_MAX_RAD = 0.25            # "open" = measured gripper angle below this
CUP_UPRIGHT_DEG = 20.0
SUCCESS_HOLD_STEPS = int(math.ceil(1.0 / STEP_DT))   # 13 steps = 1.04 s
REST_TOL_M = 0.012                     # "resting": bottom within this of the supporting surface
SETTLE_S = 2.0                         # reset: WBC runs this long (command 0, default arms) before t = 0
# The free robot, dropped from its model pose (qpos0, 4 cm above standing; the gates' start), stumbles back 12-14 cm in
# its first 0.4 s, then (stance hold at phase 0.5) stays within ~1 cm: dev2/settle2 (seeds 1000-1004: x -0.122 to
# -0.142, |y| <= 0.016, |yaw| <= 2 deg after 1 s). So it is spawned SPAWN_X ahead and settles near the origin.
SPAWN_X = 0.13
PARK_DX = 5.0                          # table A and its objects are parked this far away during that settle
STANCE_EPS = (0.02, 0.02, 0.05)        # |command| below these on every axis = "stand"
STANCE_PHASE_STEPS = (10,)             # clock step (period 20 x 0.04 s) at phase 0.5: double support (phase 0.0 is
                                       # double support too, but held there the base yawed 24-40 deg in 8 s: dev2/settle)
ACTION_DIM, STATE_DIM, IMAGE_HW = 15, 30, W.IMAGE_SIZE

# Instruction templates (declared by phase 2; phase 3 holds out some of them). Slots: colour, side.
TEMPLATES = {
    "T0": "pick up the {colour} tumbler, put it on the tray and bring the tray to the {side} table",
    "T1": "put the {colour} tumbler on the tray and carry the tray to the {side} table",
    "T2": "serve the {colour} tumbler to the {side} table on the tray",
    "T3": "load the {colour} tumbler onto the tray, then deliver the tray to the {side} table",
    "T4": "the {side} table wants the {colour} tumbler; bring it over on the tray",
    "T5": "take the tray with the {colour} tumbler to the {side} table",
}
SLOTS = {"colour": list(W.COLOURS), "side": list(W.SIDES)}

# Progress stages (info["stage"], the furthest reached so far; npz `stage`).
STAGES = ("start", "tumbler_lifted", "tumbler_on_tray", "tray_lifted", "tray_at_table", "tray_on_target", "success")


def instruction_text(layout: W.Layout, template_id: str | None = None) -> tuple[str, str]:
    tid = template_id or f"T{layout.instruction_template % len(TEMPLATES)}"
    return TEMPLATES[tid].format(colour=layout.target_colour, side=layout.target_side), tid


def quat_rotate_inverse(q, v):
    q_w, q_vec = q[0], np.asarray(q[1:4])
    return v * (2.0 * q_w ** 2 - 1.0) - np.cross(q_vec, v) * q_w * 2.0 + q_vec * np.dot(q_vec, v) * 2.0


class _OnnxPolicy:
    def __init__(self, path):
        import onnxruntime as ort
        o = ort.SessionOptions()
        o.intra_op_num_threads = 1
        o.inter_op_num_threads = 1
        self.s = ort.InferenceSession(str(path), o, providers=["CPUExecutionProvider"])
        self.k = self.s.get_inputs()[0].name

    def forward(self, x):
        return self.s.run(None, {self.k: x})[0]


class WaiterEnv:
    toy = False

    def __init__(self, deploy, upstream, cache_dir, render=True, fixed_base=False, stance_hold=True,
                 time_limit_s: float = TIME_LIMIT_S, stance_mode: str = "clock"):
        from omegaconf import OmegaConf
        from bhl_robust.eval.gait_clock import make_controller
        from bhl_robust.eval.multi_robot import MultiRunner
        self.deploy = Path(deploy)
        self.cfg = OmegaConf.load(self.deploy)
        if "waiter_wbc" not in self.cfg:
            raise ValueError(f"{deploy} is not a stamped Waiter WBC deploy.yaml")
        self.model, self.ids = W.build_scene(Path(upstream), Path(cache_dir))
        self.ctrl = make_controller(self.cfg)
        self.ctrl.policy = _OnnxPolicy(self.cfg.policy_checkpoint_path)
        self.runner = MultiRunner(self.model, [self.ids.slot], [self.cfg], [self.ctrl])
        self.d = self.runner.d
        if abs(float(self.cfg.policy_dt) * WBC_STEPS_PER_STEP - STEP_DT) > 1e-9:
            raise ValueError(f"policy_dt {self.cfg.policy_dt} x {WBC_STEPS_PER_STEP} != {STEP_DT}")
        self.render_enabled = bool(render)
        self.fixed_base = bool(fixed_base)
        self.stance_hold = bool(stance_hold)
        if stance_mode not in ("clock", "pd"):
            raise ValueError(stance_mode)
        self.stance_mode = stance_mode             # dev: "pd" = while held, the legs keep their last targets
        self._last_targets = None
        self.stance_phases = STANCE_PHASE_STEPS
        self.spawn_z = None                 # dev: base height at reset (None = the model's qpos0, as the gates)
        self.stance_min_t = 0.0             # the hold may engage only after this much WBC time since reset
        self.t_wbc = 0.0
        self.time_limit_s = float(time_limit_s)
        self._renderer = None
        sl = self.ids.slot
        self._lo = np.r_[self.ids.arm_range["left"][:, 0], self.ids.arm_range["right"][:, 0]]
        self._hi = np.r_[self.ids.arm_range["left"][:, 1], self.ids.arm_range["right"][:, 1]]
        self._sl = sl
        self.layout = None

    # ------------------------------------------------------------------------------------------------ contract API
    def reset(self, seed, instruction_id=None):
        seed = int(seed)
        rng = np.random.default_rng(seed)
        self.runner.reset(rng)                                  # joint noise N(0, 0.02) around the deploy default
        self.layout = W.sample_layout(seed)
        W.apply_layout(self.model, self.d, self.ids, self.layout)
        park = not self.fixed_base                  # table A and its objects wait PARK_DX away while the robot settles
        if park:
            self._shift_pick_area(PARK_DX)
        a = self._sl.qpos_adr
        self.d.qpos[a:a + 2] = (0.0 if self.fixed_base else SPAWN_X, 0.0)   # facing +x (qpos0 keeps z: the
        self.d.qpos[a + 3:a + 7] = [1.0, 0.0, 0.0, 0.0]             # gates' start; spawning at the standing height fell)
        if self.spawn_z is not None:
            self.d.qpos[a + 2] = float(self.spawn_z)
        self.d.eq_active[self.ids.weld] = 0
        mujoco.mj_forward(self.model, self.d)
        if self.fixed_base:                                     # welded at the free robot's standing height
            self.d.qpos[a + 2] = W.STAND_BASE_Z
            mujoco.mj_forward(self.model, self.d)
            self.d.mocap_pos[self.ids.anchor_mocap] = self.d.qpos[a:a + 3]
            self.d.mocap_quat[self.ids.anchor_mocap] = [1.0, 0.0, 0.0, 0.0]
            self.d.eq_active[self.ids.weld] = 1
        tid = None if instruction_id is None else f"T{int(instruction_id) % len(TEMPLATES)}"
        self.instruction, self.template_id = instruction_text(self.layout, tid)
        self.ctrl.reset_phase()
        self._held_clock = None
        self.t_wbc = 0.0
        self._ub = np.r_[np.zeros(10), 0.0, 0.0].astype(np.float32)
        self._cmd = np.zeros(3, np.float32)
        self.ctrl.set_upper_body(self._ub)
        for _ in range(int(round(SETTLE_S / float(self.cfg.policy_dt)))):
            self._wbc_step()
            if self.runner.tilt(0) >= TILT_FALL:
                break
        if park:                                    # then everything goes to its layout pose (pure function of the seed)
            W.apply_layout(self.model, self.d, self.ids, self.layout)
            mujoco.mj_forward(self.model, self.d)
        self.steps, self.t = 0, 0.0
        self._hold, self._stage, self._done = 0, 0, False
        self._fell = bool(self.runner.tilt(0) >= TILT_FALL)     # a fall while settling ends the first step
        return self._obs()

    def step(self, a):
        if self._done:
            raise RuntimeError("step() after done; call reset()")
        a = np.asarray(a, np.float64).reshape(ACTION_DIM)
        if not np.all(np.isfinite(a)):
            raise ValueError("non-finite action")
        self._cmd = np.clip(a[0:3], -V_LIMITS, V_LIMITS).astype(np.float32)
        arms = np.clip(a[3:13], self._lo, self._hi)
        grip = np.clip(a[13:15], 0.0, 1.0) * GRIPPER_CLOSED
        self._ub = np.r_[arms, grip].astype(np.float32)
        self.ctrl.set_upper_body(self._ub)
        for _ in range(WBC_STEPS_PER_STEP):
            self._wbc_step()
            if self.runner.tilt(0) >= TILT_FALL:
                self._fell = True
                break
        self.steps += 1
        self.t = self.steps * STEP_DT
        ev = self.evaluate()
        self._stage = max(self._stage, ev["stage_now"])
        self._hold = self._hold + 1 if ev["success_now"] else 0
        success = self._hold >= SUCCESS_HOLD_STEPS and not self._fell
        if success:
            self._stage = len(STAGES) - 1
        done = bool(success or self._fell or self.t >= self.time_limit_s - 1e-9)
        self._done = done
        info = {"success": bool(success), "fell": bool(self._fell), "stage": int(self._stage),
                "tumbler_on_tray": bool(ev["tumbler_on_tray"]), "tray_on_target": bool(ev["tray_on_target"]),
                "t": float(self.t), "applied_action": self.applied_action()}
        return self._obs(), (1.0 if success else 0.0), done, info

    def applied_action(self) -> np.ndarray:
        """The action as the env applied it (after clipping), contract units."""
        return np.r_[self._cmd, self._ub[:10], self._ub[10:] / GRIPPER_CLOSED].astype(np.float32)

    def oracle(self) -> dict:
        """Object and robot poses for the SCRIPTED expert only (never a VLA input)."""
        d, ids = self.d, self.ids
        a = self._sl.qpos_adr
        out = {"base_pos": d.qpos[a:a + 3].copy(), "base_quat": d.qpos[a + 3:a + 7].copy(),
               "base_yaw": W.yaw_of_quat(d.qpos[a + 3:a + 7]), "tilt": self.runner.tilt(0),
               "q": d.sensordata[self._sl.jpos_adr].copy(), "qpos": d.qpos.copy(), "t": self.t,
               "target_colour": self.layout.target_colour, "target_side": self.layout.target_side,
               "layout": self.layout.as_dict(), "clock_held": self._held_clock is not None}
        for c in W.COLOURS:
            out[f"tumbler_{c}"] = W.body_pose(d, ids.tumbler_body[c])
        out["tray"] = W.body_pose(d, ids.tray_body)
        for s in W.SIDES:
            out[f"table_{s}"] = W.body_pose(d, ids.table_body[s])
            out[f"zone_{s}"] = W.zone_world(d, ids, s)
            out[f"pivot_{s}"] = d.xpos[ids.finger_body[s]].copy()
        return out

    # ------------------------------------------------------------------------------------------------ evaluation
    def evaluate(self) -> dict:
        d, ids, lay = self.d, self.ids, self.layout
        cp, cR = W.body_pose(d, ids.tumbler_body[lay.target_colour])
        tp, tR = W.body_pose(d, ids.tray_body)
        cup_tilt = math.degrees(W.tilt_of(cR))
        loc = tR.T @ (cp - tp)
        cup_in_tray = (abs(loc[0]) < W.TRAY_HALF[0] - 0.5 * W.CUP_R and abs(loc[1]) < W.TRAY_HALF[1] - 0.5 * W.CUP_R)
        cup_on_floor = abs(loc[2] - W.TRAY_FLOOR_T) < REST_TOL_M
        tumbler_on_tray = bool(cup_in_tray and cup_on_floor and cup_tilt < CUP_UPRIGHT_DEG)
        tumbler_lifted = cp[2] > W.TABLE_TOP + 0.015
        tray_lifted = tp[2] > W.TABLE_TOP + 0.02
        side = lay.target_side
        bp, bR = W.body_pose(d, ids.table_body[side])
        tloc = bR.T @ (tp - bp)
        over_table = abs(tloc[0]) < W.DELIVERY_HALF[0] and abs(tloc[1]) < W.DELIVERY_HALF[1]
        in_zone = (abs(tloc[0] - W.ZONE_CENTER[0]) < W.ZONE_HALF[0] and abs(tloc[1] - W.ZONE_CENTER[1]) < W.ZONE_HALF[1])
        tray_resting = abs(tp[2] - W.TABLE_TOP) < REST_TOL_M and math.degrees(W.tilt_of(tR)) < 10.0
        tray_on_target = bool(over_table and in_zone and tray_resting)
        q = d.sensordata[self._sl.jpos_adr]
        grippers_open = bool(np.all(np.abs(q[22:24]) < GRIPPER_OPEN_MAX_RAD))
        upright = self.runner.tilt(0) < TILT_FALL
        stage_now = 0
        if tumbler_lifted or tumbler_on_tray:
            stage_now = 1
        if tumbler_on_tray:
            stage_now = 2
            if tray_lifted:
                stage_now = 3
            if over_table:
                stage_now = 4
            if tray_on_target:
                stage_now = 5
        return {"tumbler_on_tray": tumbler_on_tray, "tray_on_target": tray_on_target, "cup_tilt_deg": cup_tilt,
                "grippers_open": grippers_open, "upright": upright, "stage_now": stage_now,
                "success_now": bool(tumbler_on_tray and tray_on_target and grippers_open and upright)}

    # ------------------------------------------------------------------------------------------------ internals
    def _shift_pick_area(self, dx: float):
        """Move table A, both tumblers and the tray by dx along world x (reset only)."""
        ids, m, d = self.ids, self.model, self.d
        m.body_pos[ids.table_body["A"], 0] += dx
        for q in list(ids.tumbler_qadr.values()) + [ids.tray_qadr]:
            d.qpos[q] += dx
        mujoco.mj_forward(m, d)

    def _stand_requested(self) -> bool:
        return bool(np.all(np.abs(self._cmd) < np.asarray(STANCE_EPS)))

    def _wbc_step(self):
        c = self.ctrl
        if self.stance_hold and not self.fixed_base and self._stand_requested():
            if (self._held_clock is None and (c.clock_step % 20) in self.stance_phases
                    and self.t_wbc >= self.stance_min_t):
                self._held_clock = int(c.clock_step)
            if self._held_clock is not None:
                c.clock_step = self._held_clock
                if self.stance_mode == "pd" and self._last_targets is not None:
                    t = self._last_targets.copy()
                    t[c.ub_idx] = c.upper_body_target
                    self.runner.step([t])
                    return
        elif self._held_clock is not None:
            c.clock_step = self._held_clock                      # resume from the held phase
            self._held_clock = None
        self._last_targets = c.update(self.runner.observe(0, self._cmd))
        self.runner.step([self._last_targets])
        self.t_wbc += float(self.cfg.policy_dt)

    def _state(self) -> np.ndarray:
        sd = self.d.sensordata
        q = sd[self._sl.jpos_adr]
        quat = sd[self._sl.quat_adr:self._sl.quat_adr + 4]
        g = quat_rotate_inverse(quat, np.array([0.0, 0.0, -1.0]))
        w = sd[self._sl.gyro_adr:self._sl.gyro_adr + 3]
        return np.r_[q, g, w].astype(np.float32)

    def _image(self) -> np.ndarray:
        if not self.render_enabled:
            return np.zeros((IMAGE_HW, IMAGE_HW, 3), np.uint8)
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=IMAGE_HW, width=IMAGE_HW)
        self._renderer.update_scene(self.d, camera=self.ids.cam)
        return self._renderer.render().copy()

    def _obs(self) -> dict:
        return {"image": self._image(), "state": self._state(), "instruction": self.instruction}

    def close(self):
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
