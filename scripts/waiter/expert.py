"""Waiter phase 2 SCRIPTED expert with ORACLE poses (docs/WAITER_INTERFACES.md).

`WaiterExpert(env).act()` returns the next 15-D contract action for a `bhl_robust.eval.waiter_env.WaiterEnv`; it
reads the simulator state (`env.oracle()`, `env.d`) and is never an input to a learned policy. The plan is a Python
generator, one yield per 0.08 s env step:

  A. tumbler: the instructed tumbler's own-side hand rises sideways, slides its OPEN finger under the tumbler's lid
     handle (bar along the tumbler's y, opening facing the robot), lifts it 5 cm, carries it sideways over the tray,
     re-aims on its measured position, lowers it until it stands on the tray floor and the finger leaves the bar,
     centres the finger in the handle's gap, withdraws it level until it clears the bar, and rises (retrying if the
     tumbler comes up with it).
  B. tray: each hand moves above its handle loop (one at a time), both slide their open fingers under the loops' flat
     top bars and lift together; the low side is raised to level the tray.
  C. delivery: back off table A, turn about +-90 deg toward the instructed table, walk to the stand point (closed
     loop on the base pose; the goal follows where the carried tray actually hangs), stop.
  D. set down: lower the tray until it rests on the table, drop the fingers clear, withdraw, hold still.

Arm targets come from world-frame IK of the finger pivot (`ArmIK`, synced to the current base pose every step,
regularised toward the previous target) plus gravity feed-forward (target += tau_g / kp) and a small clamped
integral; joint-space moves are checked for clearance over table A and the tumblers. While walking the arm joint
targets are held fixed (the tray moves with the body). Base velocity commands are 0 except in C (and the free-base
station keeping), so the env's stance hold, and by default its stance lock (a declared simulation fixture), keep
the robot standing still while the hands work. With `stance_lock=False` the expert also re-centres and steps
forward for the tray (the free-base baseline).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import mujoco
import numpy as np

SIDES = ("left", "right")
ARM = {"left": slice(0, 5), "right": slice(5, 10)}
KP_ARM = 20.0


class ArmIK:
    """Damped-least-squares IK of one arm's finger pivot (and the finger hinge axis) by MuJoCo Jacobians
    (`mj_jacBody` on the finger body), on a private MjData so the simulation is never touched. Joint limits are
    enforced by clamping every iterate. Targets are WORLD frame; the base pose is copied from the simulation
    (`sync`), so the same code serves fixed and free base."""

    def __init__(self, model, ids):
        self.m = model
        self.ids = ids
        self.d = mujoco.MjData(model)
        self.jp = np.zeros((3, model.nv))
        self.jr = np.zeros((3, model.nv))
        self._bias = np.zeros(model.nv)

    def sync(self, qpos) -> None:
        self.d.qpos[:] = qpos
        self.d.qvel[:] = 0.0

    def _set(self, side, q) -> None:
        self.d.qpos[self.ids.arm_qadr[side]] = q
        mujoco.mj_kinematics(self.m, self.d)

    def tool(self, side):
        """(pivot position, hinge axis, finger direction when open, palm direction) of that side, world frame.
        The finger closes from `fwd` toward `palm` (measured by FK: lateral for both hands)."""
        fb = self.ids.finger_body[side]
        R = self.d.xmat[fb].reshape(3, 3)
        sg = 1.0 if side == "left" else -1.0
        return (self.d.xpos[fb].copy(), R @ np.array([0.0, sg, 0.0]), -R[:, 2].copy(),
                R @ np.array([-sg, 0.0, 0.0]))

    def fk(self, side, q):
        self._set(side, np.asarray(q, dtype=float))
        return self.tool(side)

    def solve(self, side, q0, p_des, hinge_des=None, w_hinge=0.05, iters=30, w_reg=0.0):
        """Bounded least squares (scipy 'trf', analytic Jacobian from mj_jacBody) from the warm start q0:
        minimise |p - p_des|^2 + w_hinge^2 |h - hinge_des|^2 + w_reg^2 |q - q0|^2 inside the joint limits. w_hinge 0.05
        trades 1 cm of position for ~11 deg of finger tilt (0.2, used before 2026-10-08, only ~3 deg: it cost 1-3 cm of
        reach where the grasps tolerate 15-20 deg of tilt; dev2/reach3). w_reg > 0 (per-step tracking) keeps the
        solution next to the warm start: without it the null space wandered near the hanging (singular) arm, ~9 rad of
        joint path for a 2 s, 25 cm rise, and those swings turned the standing robot (dev2 trace1).
        Returns (q, position error m, hinge error rad)."""
        from scipy.optimize import least_squares
        rng = self.ids.arm_range[side]
        lo, hi = rng[:, 0], rng[:, 1]
        dadr = self.ids.arm_dadr[side]
        fb = self.ids.finger_body[side]
        p_des = np.asarray(p_des, dtype=float)
        hd = None if hinge_des is None else np.asarray(hinge_des, dtype=float)

        q_ref = np.asarray(q0, dtype=float).copy()

        def fun(q):
            self._set(side, q)
            p, h, _, _ = self.tool(side)
            r = [p - p_des]
            if hd is not None:
                r.append(w_hinge * (h - hd))
            if w_reg > 0:
                r.append(w_reg * (q - q_ref))
            return np.concatenate(r)

        def jac(q):
            self._set(side, q)
            mujoco.mj_comPos(self.m, self.d)
            mujoco.mj_jacBody(self.m, self.d, self.jp, self.jr, fb)
            Jp, Jr = self.jp[:, dadr], self.jr[:, dadr]
            J = [Jp]
            if hd is not None:
                _, h, _, _ = self.tool(side)
                J.append(w_hinge * np.cross(Jr.T, h).T)
            if w_reg > 0:
                J.append(w_reg * np.eye(len(dadr)))
            return np.vstack(J)

        x0 = np.clip(np.asarray(q0, dtype=float), lo + 1e-6, hi - 1e-6)
        r = least_squares(fun, x0, jac=jac, bounds=(lo, hi), method="trf", max_nfev=iters, xtol=1e-8,
                          ftol=1e-10, gtol=1e-10)
        q = r.x
        p, h, _, _ = self.fk(side, q)
        herr = 0.0 if hd is None else float(np.arccos(np.clip(h @ hd, -1.0, 1.0)))
        return q, float(np.linalg.norm(p - p_des)), herr

    def gravity(self, side, q) -> np.ndarray:
        """Arm joint torques that hold the arm still against gravity at q (zero velocity), for feedforward through
        the PD position target (target += tau / kp)."""
        self._set(side, np.asarray(q, dtype=float))
        mujoco.mj_comPos(self.m, self.d)
        mujoco.mj_comVel(self.m, self.d)
        mujoco.mj_rne(self.m, self.d, 0, self._bias)
        return self._bias[self.ids.arm_dadr[side]].copy()


@dataclass
class ExpertParams:
    # A. tumbler (fixed-base dev sweeps, phase2/dev: cup x 0.30, tray x 0.38)
    pre_back: float = 0.10          # pre point behind the handle centre (pivot), along the handle's opening axis
    above: float = 0.04             # the pre point is approached from this much above it,
    approach_high: float = 0.06     # after a joint-space move to this much higher still (clear of the handles)
    ins_back: float = 0.05          # inserted pivot this far behind the tumbler axis (the palm cleared the rim by
                                    # only 0.7 cm at 0.045 and pushed the tumbler: dev2 seeds 1005/1006)
    lift: float = 0.05
    carry_s: float = 3.0
    settle_s: float = 1.0
    place_dx: float = -0.05         # placement: the tumbler goes to tray centre + (place_dx, +-place_dy) (tray frame), on
    place_dy: float = 0.05          # its own side (2026-10-08: the sign was flipped; tray half-width 0.13). -0.06 left
                                    # 2.4 cm to the back rim and the carries fell 2-3 cm short (dev2 F runs, 2026-10-09)
    place_fix_tol: float = 0.012    # after the carry: re-aim the hanging tumbler while it is off by more than this
    place_fix_iters: int = 2
    w_hinge_release: float = 0.2    # IK hinge weight while lowering, freeing and withdrawing the finger (keeps it level)
    withdraw: float = 0.07
    ins_tol: float = 0.02           # after insertion: pivot error, tumbler shift and tilt limits; else back out, retry
    cup_shift_tol: float = 0.012
    cup_tilt_tol_deg: float = 6.0
    retries: int = 2
    # B. tray
    ear_z: float = 0.165            # finger pivot above the tray bottom (finger top 1 cm under the peg cap)
    ear_back: float = 0.035         # pivot behind the handle loop's bar: 3.5 cm of the 7 cm finger passes under it
    ear_in: float = 0.0             # pivot lateral offset toward the tray centre (centred between the loop's posts)
    ear_pre: float = 0.08
    tray_lift: float = 0.045        # 1 cm until the finger meets the cap, then ~3.5 cm of tray lift
    # C. delivery
    backoff_x: float = -0.35        # base x before turning (table A's front edge is at 0.18)
    walk_v: float = 0.35
    cmd_slew: float = 0.04          # max change of vx, vy per 0.08 s step (0.5 m/s^2); wz twice that
    turn_w: float = 0.6
    pos_tol: float = 0.05
    yaw_tol_deg: float = 4.0
    # station keeping before manipulation (the base drifts while the WBC settles and while the arms move)
    station_tol_x: float = 0.08
    station_tol_y: float = 0.08
    station_yaw_tol_deg: float = 12.0
    station_vmin: float = 0.04
    station_v: float = 0.15
    station_tries: int = 4
    tray_station_x: float = 0.10   # base x (world) for hooking the tray: the robot steps forward with its hands out
                                    # to the side, and moves ~5 cm further forward as it leans back with them at the loops
    recenter_k: float = 0.5         # _recenter: P gain (1/s), speed cap (m/s), lateral command bias while it steps
    recenter_vmax: float = 0.06     # (the s7 WBC drifts to its left when it starts stepping: dev2 cmd1), time limit
    recenter_vy_bias: float = 0.0
    recenter_timeout: float = 3.0
    recenter_settle: float = 0.8    # stand still this long after re-centring before the arms move
    station_ax: float = 0.0         # station A goal x (world)
    station_settle: float = 0.0     # stand still this long before station A (the arm-pose yaw creep settles first)
    station_yaw_deg: float = 0.0    # station A goal yaw: raising the picking arm then turns the robot ~5-9 deg clockwise
    tray_recenter_tol: float = 0.03 # re-centre for the tray step when the base is off by more than this (x or y)
    tray_side_settle: float = 1.5   # hands out to the side: stand this long before stepping (posture creep)
    tray_above_settle: float = 1.0  # hands above the loops: stand this long before re-aiming (the lean develops)
    tray_yaw_tol_deg: float = 3.0   # then turn back to heading 0 when off by more than this
    step_k: float = 1.5             # _step_to: P gain, minimum and maximum speed (m/s), time limit (s)
    step_vmin: float = 0.08
    step_vmax: float = 0.15
    step_timeout: float = 4.0
    ready_idle: bool = False        # raise the idle hand to its loop early (v5); off: it hangs during the tumbler step
    # D. set down
    set_extra: float = 0.025        # lower this much once the tray rests: the fingers drop clear of the tunnel roofs
    zone_aim_dx: float = 0.03       # aim the tray this far past the zone centre (zone frame x, away from the robot)
    tray_off_alpha: float = 0.1     # low-pass gain (per step) of the carried tray's offset in the base frame
    setdown_settle: float = 1.0     # stand this long at the stand point before lowering (the hanging tray swings)
    debug: bool = False
    integral_ki: float = 0.15
    integral_max: float = 0.06
    # arm motion shaping (2026-10-09, dev2 trace1: with the base free, nearly all drift came from arm motion, ~0.4 deg
    # of yaw per rad of joint path, and 0.06 deg/s when the arms were still)
    ik_reg: float = 0.01            # per-step IK regularisation toward the previous target (m per rad; 0 = off)
    qd_max: float = 0.0             # joint speed cap (rad/s): move durations are stretched to it and every per-step
                                    # target is rate limited (0 = off)


def smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def yaw_rot(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


class Tracker:
    """One arm: IK target -> joint target with gravity feed-forward and a small clamped integral."""

    def __init__(self, ex: "WaiterExpert", side: str):
        self.ex, self.side = ex, side
        self.sg = 1.0 if side == "left" else -1.0
        self.hinge = np.array([0.0, 0.0, self.sg])        # measured: left hinge +z, right -z with the forearm level
        self.qdes = ex.q_arm(side).copy()
        self.I = np.zeros(5)
        self.pe = 0.0

    def reset_integral(self):
        self.I[:] = 0.0

    def target(self, p_des, hinge=True, w_hinge=0.05) -> np.ndarray:
        ex, p = self.ex, self.ex.p
        ex.ik.sync(ex.env.d.qpos)
        # warm-started only: branch choices happen once per approach, in joint space (approach_q + jmove2); a per-step
        # multi-start fallback flipped branches between steps and swung the arms (the standing robot slid)
        q, self.pe, _ = ex.ik.solve(self.side, self.qdes, p_des, hinge_des=self.hinge if hinge else None,
                                    w_hinge=w_hinge, w_reg=p.ik_reg)
        q = self.limit(q)
        self.qdes = q
        g = ex.ik.gravity(self.side, q)
        self.I = np.clip(self.I + p.integral_ki * (q - ex.q_arm(self.side)), -p.integral_max, p.integral_max)
        rng = ex.env.ids.arm_range[self.side]
        return np.clip(q + g / KP_ARM + self.I, rng[:, 0], rng[:, 1])

    def limit(self, q) -> np.ndarray:
        """Rate-limit a new joint target against the previous one (ExpertParams.qd_max per 0.08 s step)."""
        v = self.ex.p.qd_max
        if v <= 0:
            return np.asarray(q, float)
        return self.qdes + np.clip(np.asarray(q, float) - self.qdes, -v * 0.08, v * 0.08)

    def joint_target(self, q) -> np.ndarray:
        ex = self.ex
        ex.ik.sync(ex.env.d.qpos)
        q = self.limit(q)
        g = ex.ik.gravity(self.side, q)
        rng = ex.env.ids.arm_range[self.side]
        self.qdes = np.asarray(q, float).copy()
        return np.clip(q + g / KP_ARM, rng[:, 0], rng[:, 1])

    def pivot(self) -> np.ndarray:
        return self.ex.env.d.xpos[self.ex.env.ids.finger_body[self.side]].copy()

    def fwd(self) -> np.ndarray:
        self.ex.ik.sync(self.ex.env.d.qpos)
        _, _, f, _ = self.ex.ik.tool(self.side)
        f[2] = 0.0
        return f / max(1e-9, np.linalg.norm(f))


class WaiterExpert:
    def __init__(self, env, params: ExpertParams | None = None, log=None):
        from bhl_robust.eval import waiter_scene as W
        self.W = W
        self.env = env
        self.p = params or ExpertParams()
        self.ik = ArmIK(env.model, env.ids)
        self.log = log or (lambda *a: None)
        self.gen = None
        # the base cannot drift (stance lock or fixed base): no stepping, no heading corrections at the tray
        self.base_held = bool(getattr(env, "stance_lock", False) or getattr(env, "fixed_base", False))

    # ------------------------------------------------------------------------------------------------ public
    def reset(self):
        self.arms = self.env.applied_action()[3:13].astype(float).copy()
        self.grip = {s: 0.0 for s in SIDES}
        self.cmd = np.zeros(3)
        self.trk = {s: Tracker(self, s) for s in SIDES}
        self.phase = "A"
        self.notes = {}
        self.gen = self._plan()

    def act(self) -> np.ndarray:
        try:
            return next(self.gen)
        except StopIteration:
            self.cmd[:] = 0.0
            return self._action()

    # ------------------------------------------------------------------------------------------------ helpers
    def q_arm(self, side):
        return self.env.d.sensordata[self.env.ids.slot.jpos_adr][ARM[side]].copy()

    def _action(self):
        return np.r_[self.cmd, self.arms, self.grip["left"], self.grip["right"]].astype(np.float32)

    def base(self):
        o = self.env.d.qpos[self.env.ids.slot.qpos_adr:self.env.ids.slot.qpos_adr + 7]
        return o[:3].copy(), self.W.yaw_of_quat(o[3:7])

    def att(self):
        """Base (roll, pitch) in degrees."""
        R = self.env.d.xmat[self.env.ids.base_body].reshape(3, 3)
        return math.degrees(math.atan2(R[2, 1], R[2, 2])), math.degrees(math.asin(-max(-1.0, min(1.0, R[2, 0]))))

    def shoulder(self, side):
        """Shoulder (pitch joint) position in the base-yaw frame, relative to the base origin."""
        m = self.env.model
        j = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, f"r0_arm_{side}_shoulder_pitch_joint")
        bp, yaw = self.base()
        return yaw_rot(yaw).T @ (self.env.d.xpos[m.jnt_bodyid[j]] - bp)

    def cup(self):
        return self.W.body_pose(self.env.d, self.env.ids.tumbler_body[self.env.layout.target_colour])

    def tray(self):
        return self.W.body_pose(self.env.d, self.env.ids.tray_body)

    def cup_tilt(self):
        return math.degrees(self.W.tilt_of(self.cup()[1]))

    def arm_contacts(self, side):
        """Names of (arm geom, other geom) pairs in contact for that arm (debug)."""
        import mujoco as mj
        m, d = self.env.model, self.env.d
        out = set()
        for i in range(d.ncon):
            c = d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                bn = mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])) or ""
                if f"arm_{side}" in bn:
                    on = mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, int(o)) or mj.mj_id2name(
                        m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[o])) or str(o)
                    out.add(f"{bn.replace('r0_arm_', '')}~{on.replace('r0_', '')}")
        return sorted(out)

    def dbg(self, tag, side, goal):
        if not self.p.debug:
            return
        bp, yaw = self.base()
        Rb = yaw_rot(yaw)
        t = self.trk[side]
        gb = Rb.T @ (np.asarray(goal) - bp)
        pb = Rb.T @ (t.pivot() - bp)
        q, qd = self.q_arm(side), t.qdes
        self.log(f"dbg {tag:8s} contacts {self.arm_contacts(side)}")
        self.log(f"dbg {tag:8s} {side:5s} base {np.round(bp, 3)} yaw {math.degrees(yaw):.1f} | goal(base) "
                 f"{np.round(gb, 3)} pivot(base) {np.round(pb, 3)} err {np.linalg.norm(t.pivot() - goal):.3f} "
                 f"ik_res {t.pe:.3f} | q {np.round(q, 2)} qdes {np.round(qd, 2)} I {np.round(t.I, 2)}")

    def dbg_cup(self, tag, side):
        """Debug: the instructed tumbler on the tray (tray frame), its tilt, the finger pivot in the tumbler frame
        and what the tumbler touches."""
        if not self.p.debug:
            return
        import mujoco as mj
        m, d = self.env.model, self.env.d
        cb = self.env.ids.tumbler_body[self.env.layout.target_colour]
        cp, cR = self.cup()
        tp, tR = self.tray()
        touch = set()
        for i in range(d.ncon):
            c = d.contact[i]
            for g, o in ((c.geom1, c.geom2), (c.geom2, c.geom1)):
                if int(m.geom_bodyid[g]) == cb:
                    touch.add(f"{mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, int(g))}~"
                              f"{mj.mj_id2name(m, mj.mjtObj.mjOBJ_GEOM, int(o)) or mj.mj_id2name(m, mj.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[o]))}")
        self.log(f"dbg {tag:8s} cup on tray {np.round(tR.T @ (cp - tp), 3)} tilt {self.cup_tilt():.1f} pivot in cup "
                 f"{np.round(cR.T @ (self.trk[side].pivot() - cp), 3)} fwd in cup {np.round(cR.T @ self.trk[side].fwd(), 2)} "
                 f"touch {sorted(t.replace('tumbler_', '') for t in touch)}")

    def dbg_setdown(self, tag):
        """Debug: the tray on the target table (table frame, zone), its tilt and height, the tumbler on the tray."""
        if not self.p.debug:
            return
        W, d, ids = self.W, self.env.d, self.env.ids
        side = self.env.layout.target_side
        tp, tR = self.tray()
        bp, bR = W.body_pose(d, ids.table_body[side])
        tloc = bR.T @ (tp - bp)
        cp, cR = self.cup()
        q = d.sensordata[ids.slot.jpos_adr]
        self.log(f"dbg {tag:8s} tray in table frame {np.round(tloc, 3)} (zone centre {W.ZONE_CENTER}, half {W.ZONE_HALF}; "
                 f"table half {W.DELIVERY_HALF}) tray z-top {tp[2] - W.TABLE_TOP:+.3f} tilt "
                 f"{math.degrees(W.tilt_of(tR)):.1f} | cup on tray {np.round(tR.T @ (cp - tp), 3)} tilt {self.cup_tilt():.1f} "
                 f"| grippers {np.round(q[22:24], 2)} | arm contacts L {self.arm_contacts('left')} R {self.arm_contacts('right')}")

    def finger_touch(self, side, part) -> bool:
        """Does that finger touch the instructed tumbler's lid-handle top bar (`part` "bar") or its body ("body")?"""
        m, d = self.env.model, self.env.d
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM,
                              f"tumbler_{self.env.layout.target_colour}_{'bar_top' if part == 'bar' else 'body'}")
        fg = self.env.ids.finger_geom[side]
        return any({int(d.contact[i].geom1), int(d.contact[i].geom2)} == {g, fg} for i in range(d.ncon))

    def cup_on_floor(self) -> bool:
        """Does the instructed tumbler's body touch the tray floor?"""
        m, d = self.env.model, self.env.d
        g = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, f"tumbler_{self.env.layout.target_colour}_body")
        f = self.env.ids.tray_floor_geom
        return any({int(d.contact[i].geom1), int(d.contact[i].geom2)} == {g, f} for i in range(d.ncon))

    def finger_clear(self, side, margin=0.006) -> bool:
        """Every corner of the finger box is behind the lid handle's top bar, or beyond its ends (tumbler frame)."""
        d, fg = self.env.d, self.env.ids.finger_geom[side]
        sz = self.env.model.geom_size[fg]
        R = d.geom_xmat[fg].reshape(3, 3)
        cp, cR = self.cup()
        sg = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)]) * sz
        loc = (d.geom_xpos[fg] + sg @ R.T - cp) @ cR
        W = self.W
        return bool(np.all(loc[:, 0] < -W.HANDLE_T / 2 - margin)
                    or np.all(np.abs(loc[:, 1]) > W.BAIL_HALF_SPAN + W.HANDLE_T / 2 + margin))

    def hold(self, dur):
        for _ in range(max(1, int(round(dur / 0.08)))):
            yield self._action()

    def stretch(self, dur, dq: dict) -> float:
        """Duration of a raised-cosine move whose peak joint speed, (pi/2) max|dq| / T, stays under qd_max."""
        v = self.p.qd_max
        if v <= 0 or not dq:
            return dur
        return max(dur, 0.5 * math.pi * max(float(np.max(np.abs(d))) for d in dq.values()) / v)

    def move(self, goals: dict, dur: float, hinge=True, w_hinge=0.05):
        """Raised-cosine Cartesian move of the listed pivots from where they are now to `goals` (world)."""
        starts = {s: self.trk[s].pivot() for s in goals}
        if self.p.qd_max > 0:                     # the joint motion this move needs, from IK at its goal
            self.ik.sync(self.env.d.qpos)
            dq = {}
            for s, g in goals.items():
                h = hinge if not isinstance(hinge, dict) else hinge[s]
                q1, _, _ = self.ik.solve(s, self.trk[s].qdes, g, hinge_des=self.trk[s].hinge if h else None,
                                         w_hinge=w_hinge, w_reg=self.p.ik_reg)
                dq[s] = q1 - self.trk[s].qdes
            dur = self.stretch(dur, dq)
        n = max(1, int(round(dur / 0.08)))
        for k in range(n):
            u = smooth((k + 1) / n)
            for s, g in goals.items():
                self.arms[ARM[s]] = self.trk[s].target(starts[s] + (np.asarray(g) - starts[s]) * u,
                                                       hinge=hinge if not isinstance(hinge, dict) else hinge[s],
                                                       w_hinge=w_hinge)
            yield self._action()

    def jmove(self, side, q1, dur):
        q0 = self.q_arm(side)
        n = max(1, int(round(self.stretch(dur, {side: np.asarray(q1) - q0}) / 0.08)))
        for k in range(n):
            u = smooth((k + 1) / n)
            self.arms[ARM[side]] = self.trk[side].joint_target(q0 + (np.asarray(q1) - q0) * u)
            yield self._action()

    IK_STARTS = (np.array([-0.3, 0.2, 0.0, 1.3, -0.4]), np.zeros(5), np.array([-0.6, 0.4, 0.3, 1.0, -0.6]),
                 np.array([-0.15, 0.1, 0.0, 0.65, -0.2]))

    def best_ik(self, side, goal, hinge=True):
        """Multi-start IK (the current configuration and canonical arm poses): the joint solution with the smallest
        residual, so a Cartesian approach starts in a configuration that can reach (warm starts get stuck)."""
        sg = 1.0 if side == "left" else -1.0
        self.ik.sync(self.env.d.qpos)
        h = self.trk[side].hinge if hinge else None
        best = None
        for q0 in (self.trk[side].qdes,) + tuple(sg * q for q in self.IK_STARTS):
            q, pe, _ = self.ik.solve(side, q0, goal, hinge_des=h, iters=60)
            if best is None or pe < best[1] - 1e-4:
                best = (q, pe)
        return best

    def approach_q(self, side, final, approach, hinge=True):
        """Joint configuration at `approach` in the IK branch that reaches `final`: multi-start IK on the FINAL point
        (the hardest), then the approach point warm-started from it. Solving the approach point alone picked a
        straight-arm branch that cannot reach a hook point 8 cm lower (dev2 tray4: shoulder roll on its limit)."""
        qf, pef = self.best_ik(side, final)
        self.ik.sync(self.env.d.qpos)
        qa, pea, _ = self.ik.solve(side, qf, approach, hinge_des=self.trk[side].hinge if hinge else None, iters=60)
        return qa, pea, pef

    def path_clearance(self, side, q0, q1, n=16) -> float:
        """Lowest height of that arm's finger (pivot and tip) and hand along the joint-space line q0 -> q1 above the
        obstacle under it: table A's top (points above or within 4 cm of the table) and each tumbler's handle top
        (points within 4 cm of its footprint, handle included); the base pose as it is now. +inf if none."""
        W = self.W
        tc = self.env.model.body_pos[self.env.ids.table_body["A"]]
        hb = self.env.ids.hand_body[side]
        cups = [self.W.body_pose(self.env.d, b)[0] for b in self.env.ids.tumbler_body.values()]
        cup_top = W.CUP_H + W.BAIL_GAP + W.HANDLE_T
        cup_r = W.BAIL_HALF_SPAN + W.HANDLE_T / 2 + 0.04
        self.ik.sync(self.env.d.qpos)
        low = math.inf
        q0, q1 = np.asarray(q0, float), np.asarray(q1, float)
        for u in np.linspace(0.0, 1.0, n):
            pv, _, f, _ = self.ik.fk(side, q0 + (q1 - q0) * u)
            for pt in (pv, pv + 0.07 * f, self.ik.d.xpos[hb]):
                if pt[0] > tc[0] - W.TABLE_A_HALF[0] - 0.04 and abs(pt[1] - tc[1]) < W.TABLE_A_HALF[1] + 0.04:
                    low = min(low, float(pt[2]) - W.TABLE_TOP)
                for c in cups:
                    if float(np.hypot(pt[0] - c[0], pt[1] - c[1])) < cup_r:
                        low = min(low, float(pt[2]) - (c[2] + cup_top))
        return low

    def via_q(self, side, q0, q1):
        """A joint configuration to pass through so both halves of q0 -> q1 keep the hand >= 3 cm above table A (a
        straight joint-space line from a low hand dipped the finger onto the table's edge: dev2 F runs, seed 1000)."""
        W = self.W
        self.ik.sync(self.env.d.qpos)
        p0 = self.ik.fk(side, q0)[0]
        p1 = self.ik.fk(side, q1)[0]
        for dz, dx in ((0.12, 0.0), (0.18, 0.0), (0.12, -0.08), (0.18, -0.08)):
            g = 0.5 * (p0 + p1) + np.array([dx, 0.0, 0.0])
            g[2] = max(p0[2], p1[2], W.TABLE_TOP + dz)
            self.ik.sync(self.env.d.qpos)
            qv, _, _ = self.ik.solve(side, 0.5 * (np.asarray(q0) + np.asarray(q1)), g, hinge_des=None, iters=60)
            if min(self.path_clearance(side, q0, qv), self.path_clearance(side, qv, q1)) >= 0.03:
                return qv
        return None

    def jmove2(self, qs: dict, dur, clear=True):
        """Raised-cosine joint-space move of the listed arms to `qs` (with gravity feed-forward). With `clear`, an arm
        whose straight joint-space path would bring the hand within 3 cm of table A's top first goes through a via
        configuration (via_q)."""
        if clear:
            vias = {}
            for s_, q1 in qs.items():
                if self.path_clearance(s_, self.q_arm(s_), q1) < 0.03:
                    qv = self.via_q(s_, self.q_arm(s_), np.asarray(q1, float))
                    self.log(f"expert: {s_} joint path near table A; via {'found' if qv is not None else 'NOT found'}")
                    if qv is not None:
                        vias[s_] = qv
            if vias:
                yield from self.jmove2(vias, 0.5 * dur, clear=False)
        q0 = {s: self.q_arm(s) for s in qs}
        n = max(1, int(round(self.stretch(dur, {s: np.asarray(q1) - q0[s] for s, q1 in qs.items()}) / 0.08)))
        for k in range(n):
            u = smooth((k + 1) / n)
            for s, q1 in qs.items():
                self.arms[ARM[s]] = self.trk[s].joint_target(q0[s] + (np.asarray(q1) - q0[s]) * u)
            yield self._action()

    def grip_to(self, sides, value, dur):
        g0 = {s: self.grip[s] for s in sides}
        n = max(1, int(round(dur / 0.08)))
        for k in range(n):
            u = (k + 1) / n
            for s in sides:
                self.grip[s] = g0[s] + (value - g0[s]) * u
            yield self._action()

    # ------------------------------------------------------------------------------------------------ the plan
    def _ready_idle(self, idle):
        """Early in the episode: the idle hand rises (Cartesian, behind the table edge) and goes, slowly, to its hook-ready
        configuration above its tray peg, so the tray step needs only a small motion (big arm swings make the standing
        robot slide and turn: dev2 env21-tray5)."""
        yield from self.raise_hand(idle, dur=2.0)
        W = self.W
        tp, tR = self.tray()
        sg = 1.0 if idle == "left" else -1.0
        ins = tp + tR @ (np.array([W.EAR_DX, sg * W.ear_center_y(), 0.0])
                         + np.array([-self.p.ear_back, -sg * self.p.ear_in, self.p.ear_z]))
        qa, pea, pef = self.approach_q(idle, ins, ins - self.p.ear_pre * tR[:, 0] + np.array([0, 0, 0.06]))
        yield from self.jmove2({idle: qa}, 2.5)
        self.log(f"expert: idle {idle} ready (res above {pea:.3f} ins {pef:.3f})")

    def plan_from_tray(self, side):
        """Development: start at the tray step (the tumbler already on the tray), `side` = the hand that 'placed' it."""
        self.side = side
        self.notes["tumbler_ok"] = True
        yield from self.hold(0.4)
        if self.p.ready_idle:
            yield from self._ready_idle("left" if side == "right" else "right")
        ok = yield from self._tray()
        self.notes["tray_ok"] = bool(ok)
        if not ok:
            self.log("expert: tray step failed; holding still")
            yield from self.hold(1e9)
        yield from self._deliver()
        yield from self._set_down()
        yield from self.hold(1e9)

    def _plan(self):
        yield from self.hold(0.4)                          # stance hold engages at the next double support
        cy = self.env.layout.tumblers[self.env.layout.target_colour][1]
        if self.p.ready_idle:
            yield from self._ready_idle("right" if cy > 0 else "left")
        ok = yield from self._tumbler()
        self.notes["tumbler_ok"] = ok
        if not ok:
            self.log("expert: tumbler step failed; holding still")
            yield from self.hold(1e9)
        ok = yield from self._tray()
        self.notes["tray_ok"] = bool(ok)
        if not ok:
            self.log("expert: tray step failed; holding still")
            yield from self.hold(1e9)
        yield from self._deliver()
        yield from self._set_down()
        yield from self.hold(1e9)

    # A ---------------------------------------------------------------------------------------------------------
    def _tumbler(self):
        p, W = self.p, self.W
        if p.station_settle > 0:
            yield from self.hold(p.station_settle)
        yield from self._station(goal_xy=(p.station_ax, 0.0), goal_yaw=math.radians(p.station_yaw_deg), tag="A")
        cp, cR = self.cup()
        side = "left" if cp[1] - self.base()[0][1] > 0 else "right"
        self.side = side
        sg = 1.0 if side == "left" else -1.0
        trk = self.trk[side]
        # prep: finger open, the hanging hand rises SIDEWAYS (raise_hand), clear of table A's near edge
        if trk.pivot()[2] < 0.55:
            yield from self.raise_hand(side)
        self.dbg("prep", side, trk.pivot())
        for attempt in range(p.retries + 1):
            cp, cR = self.cup()
            a = cR[:, 0]
            c = cp + cR @ W.handle_center_local()
            pz = c[2] - 0.002
            pre = c - p.pre_back * a
            pre[2] = pz
            above = pre + np.array([0, 0, p.above])
            ins = c - p.ins_back * a
            ins[2] = pz
            trk.reset_integral()
            high = above + np.array([0, 0, p.approach_high])
            qa, pea, pef = self.approach_q(side, ins, high)           # the branch that reaches the insertion
            yield from self.jmove2({side: qa}, 1.2)                   # (joint space, high: clears the handle)
            yield from self.move({side: above}, 0.5)
            self.dbg("above", side, above)
            yield from self.move({side: pre}, 0.8)
            self.dbg("pre", side, pre)
            trk.reset_integral()
            yield from self.move({side: ins}, 0.8)
            yield from self.move({side: ins}, 0.3)
            self.dbg("ins", side, ins)
            cp2, cR2 = self.cup()
            err = float(np.linalg.norm(trk.pivot() - ins))
            shift = float(np.linalg.norm(cp2[:2] - cp[:2]))
            tilt = self.cup_tilt()
            loc = cR2.T @ (trk.pivot() - cp2)                 # pivot in the tumbler frame: is the finger under the bar?
            under = bool(abs(loc[1]) < W.BAIL_HALF_SPAN - W.HANDLE_T / 2 - 0.016 and -0.065 < loc[0] < -0.035
                         and W.CUP_H + 0.02 < loc[2] < W.CUP_H + W.BAIL_GAP - 0.02)
            self.log(f"expert: insert attempt {attempt} pivot err {err:.3f} cup shift {shift:.3f} tilt {tilt:.1f} "
                     f"pivot in cup {np.round(loc, 3)} under bar {under}")
            if under and tilt < p.cup_tilt_tol_deg and cp2[2] < W.TABLE_TOP + 0.01:
                break
            # back out along the opening axis, then up, and retry from the new tumbler pose
            out = trk.pivot() - 0.06 * a
            yield from self.move({side: out}, 1.0)
            yield from self.move({side: out + np.array([0, 0, 0.05])}, 0.8)
            if tilt > 30.0:
                return False
        else:
            return False
        # lift
        cur = trk.pivot()
        z0 = self.cup()[0][2]
        yield from self.move({side: cur + np.array([0, 0, p.lift])}, 1.2)
        yield from self.hold(0.4)
        if self.cup()[0][2] - z0 < 0.02:
            self.log("expert: lift failed")
            return False
        # carry over the tray (one raised-cosine move), settle
        tp, tR = self.tray()
        cur = trk.pivot()
        off = cur - self.cup()[0]
        off[2] = 0.0
        dest = tp + tR @ np.array([p.place_dx, sg * p.place_dy, 0.0]) + off
        dest[2] = cur[2]
        yield from self.move({side: dest}, p.carry_s)
        yield from self.move({side: dest}, p.settle_s)
        # the carry falls short (the arm reaches out to the side and the robot leans): re-aim on the tumbler's measured
        # position on the tray, a small horizontal move per try
        goal_loc = np.array([p.place_dx, sg * p.place_dy])
        for _ in range(p.place_fix_iters):
            tp, tR = self.tray()
            err = goal_loc - (tR.T @ (self.cup()[0] - tp))[:2]
            if float(np.linalg.norm(err)) < p.place_fix_tol:
                break
            dest = dest + tR @ np.array([err[0], err[1], 0.0])
            yield from self.move({side: dest}, 0.8)
            yield from self.move({side: dest}, 0.6)
        self.dbg_cup("settled", side)
        # lower 3 mm per 0.16 s until the tumbler stands on the tray floor and the finger has come off the lid handle's
        # bar (the old test, the tumbler's height alone, fired a few steps late and put the finger on the lid)
        rest_z = self.tray()[0][2] + W.TRAY_FLOOR_T
        cur = dest.copy()
        free = 0
        for _ in range(50):
            cur = cur - np.array([0, 0, 0.003])
            yield from self.move({side: cur}, 0.16)
            free = 0 if self.finger_touch(side, "bar") else free + 1
            if free and (self.cup_on_floor() or self.cup()[0][2] < rest_z + 0.002) or free >= 2:
                break
        self.dbg_cup("rests", side)
        yield from self.hold(0.4)
        # centre the finger in the handle's gap (a 5 cm tall plate in 6.5 cm: ~1.5 cm of play), never on the lid
        cur = cur - np.array([0, 0, 0.003])
        yield from self.move({side: cur}, 0.24, w_hinge=p.w_hinge_release)
        for _ in range(3):
            if not self.finger_touch(side, "body"):
                break
            cur = cur + np.array([0, 0, 0.003])
            yield from self.move({side: cur}, 0.24, w_hinge=p.w_hinge_release)
        self.dbg_cup("freed", side)
        # withdraw along the finger's own direction, level, until the whole finger is behind the bar (tumbler frame)
        zf = cur[2]
        for k in range(3):
            cur = trk.pivot() - (p.withdraw if k == 0 else 0.03) * trk.fwd()
            cur[2] = zf
            yield from self.move({side: cur}, 1.0 if k == 0 else 0.5, w_hinge=p.w_hinge_release)
            yield from self.hold(0.24)
            if self.finger_clear(side):
                break
        self.dbg_cup("withdrawn", side)
        # rise; if the tumbler comes up with the finger (it caught the bar), go back down, withdraw 3 cm more, retry
        z0 = self.cup()[0][2]
        for _ in range(3):
            yield from self.move({side: cur + np.array([0, 0, 0.015])}, 0.3, w_hinge=p.w_hinge_release)
            if self.cup()[0][2] < z0 + 0.004:
                yield from self.move({side: cur + np.array([0, 0, 0.06])}, 0.5)
                break
            yield from self.move({side: cur}, 0.3, w_hinge=p.w_hinge_release)
            cur = trk.pivot() - 0.03 * trk.fwd()
            cur[2] = zf
            yield from self.move({side: cur}, 0.5, w_hinge=p.w_hinge_release)
        self.dbg_cup("risen", side)
        cp, cR = self.cup()
        tp, tR = self.tray()
        loc = tR.T @ (cp - tp)
        ok = bool(abs(loc[0]) < W.TRAY_HALF[0] - W.CUP_R and abs(loc[1]) < W.TRAY_HALF[1] - W.CUP_R
                  and abs(loc[2] - W.TRAY_FLOOR_T) < 0.01 and self.cup_tilt() < 20.0)
        self.log(f"expert: tumbler placed ok={ok} tilt {self.cup_tilt():.1f} loc {np.round(loc, 3)}")
        return ok

    # B ---------------------------------------------------------------------------------------------------------
    Q_SIDE = np.array([0.0, 1.0, 0.0, 0.2, 0.0])           # left arm abducted ~57 deg (the right arm: -Q_SIDE)

    def raise_hand(self, side, dur=1.5):
        """A hanging hand rises SIDEWAYS in joint space (shoulder roll out, elbow slightly bent: pivot ~(0.0, +-0.55,
        0.60) m in the base frame): it stays behind table A's near edge (x 0.13) until it is above the table top, and
        the next joint-space move (clearance-checked, jmove2) brings it over the table. The v5 Cartesian rise behind
        the edge asked for points closer to the shoulder than the arm's ~0.30 m minimum reach (IK residual 0.12-0.15 m,
        dev2 t_fk), and the joint-space moves from where it stopped dipped onto the table's edge (dev2 F runs)."""
        q = self.Q_SIDE if side == "left" else -self.Q_SIDE
        yield from self.jmove2({side: q}, dur, clear=False)
        self.dbg("raised", side, self.trk[side].pivot())

    def raise_hand_cartesian(self, side, dur=1.5):
        """A hanging hand rises behind table A's front edge (x 0.18) to ~0.62 m along a CARTESIAN path: first forward-up
        to just above the table top (base x 0.04, z 0.50), then up (x 0.08, z 0.62). A joint-space path from the hanging
        pose swings the forearm forward below table height into the table's edge (dev2 dbg23/dbg24)."""
        bp, yaw = self.base()
        Rb = yaw_rot(yaw)
        y = (Rb.T @ (self.trk[side].pivot() - bp))[1]
        for xb, zw, d in ((0.04, 0.50, 0.45 * dur), (0.08, 0.62, 0.55 * dur)):
            g = bp + Rb @ np.array([xb, y, 0.0])
            g[2] = zw
            yield from self.move({side: g}, d, hinge=False)
        self.dbg("raised", side, g)

    def _tray(self):
        p, W = self.p, self.W
        self.phase = "B"
        side = self.side
        idle = "left" if side == "right" else "right"
        sgn = {"left": 1.0, "right": -1.0}
        peg_loc = {s_: np.array([W.EAR_DX, sgn[s_] * W.ear_center_y(), 0.0]) for s_ in SIDES}
        ins_loc = {s_: peg_loc[s_] + np.array([-p.ear_back, -sgn[s_] * p.ear_in, p.ear_z]) for s_ in SIDES}

        def targets():
            tp_, tR_ = self.tray()
            ins_ = {s_: tp_ + tR_ @ ins_loc[s_] for s_ in SIDES}
            tR2 = self.tray()[1]
            return ins_, {s_: ins_[s_] - self.p.ear_pre * tR2[:, 0] + np.array([0, 0, 0.06]) for s_ in SIDES}

        bp0, yaw0 = self.base()
        tp0, tR0 = self.tray()
        lx, ly, lyaw = self.env.layout.tray
        self.log(f"expert: tray step start base {np.round(bp0, 3)} yaw {math.degrees(yaw0):.1f} | tray moved "
                 f"{np.round(tp0[:2] - (lx, ly), 3)} yaw {math.degrees(self.W.yaw_of_quat(self.env.d.xquat[self.env.ids.tray_body]) - lyaw):.1f}")
        if self.base_held:
            # The base is held (stance lock / fixed base): one arm at a time to its approach configuration above its
            # loop (v5 sequence: 8/8 hooks welded, dev2 W1). The picking hand is above its placed tumbler; the idle
            # hand still hangs and rises in front, behind the table edge, then moves above its loop.
            cur = self.trk[side].pivot()
            if cur[2] < 0.55:
                yield from self.raise_hand_cartesian(side)
            else:
                yield from self.move({side: np.array([cur[0], cur[1], 0.70])}, 0.6, hinge=False)
            ins, above = targets()
            qa, pea, pef = self.approach_q(side, ins[side], above[side])
            self.log(f"dbg approach {side} res above {pea:.3f} ins {pef:.3f}")
            yield from self.jmove2({side: qa}, 2.0)
            if self.trk[idle].pivot()[2] < 0.55:
                yield from self.raise_hand_cartesian(idle)
            ins, above = targets()
            qa, pea, pef = self.approach_q(idle, ins[idle], above[idle])
            self.log(f"dbg approach {idle} res above {pea:.3f} ins {pef:.3f}")
            yield from self.jmove2({idle: qa}, 1.2)
        else:
            # Two stances (dev2 2026-10-09): standing still the s7 robot leans back as its arms reach forward (base pitch
            # +3 deg arms down, -8..-12 deg both hands at the loops; the shoulders move ~15 cm back), so the loops are in
            # reach only from x >= ~0.10, where the robot's hips would touch table A's edge with its arms down. Hence:
            # 1) both hands out to the side (Q_SIDE), the picking hand first up off its tumbler, and the posture settles;
            cur = self.trk[side].pivot()
            if cur[2] >= 0.55:
                yield from self.move({side: np.array([cur[0], cur[1], 0.70])}, 0.6, hinge=False)
            yield from self.jmove2({s_: (self.Q_SIDE if s_ == "left" else -self.Q_SIDE) for s_ in SIDES}, 2.0)
            yield from self.hold(p.tray_side_settle)
            # 2) step forward to the hook stance (straight ahead: small P commands sat in the WBC's dead band, dev2 K runs);
            yield from self._step_to(p.tray_station_x)
            # 3) both hands above their loops (clearance-checked joint moves); as the lean develops the robot turns
            #    clockwise (posture creep, -5..-15 deg): wait, turn back in place (the hands are above the bars), re-aim.
            ins, above = targets()
            qs = {}
            for s_ in SIDES:
                qs[s_], pea, pef = self.approach_q(s_, ins[s_], above[s_])
                self.log(f"dbg approach {s_} res above {pea:.3f} ins {pef:.3f}")
            yield from self.jmove2(qs, 2.5)
            yield from self.hold(p.tray_above_settle)
            if abs(math.degrees(self.base()[1])) > p.tray_yaw_tol_deg:
                yield from self._turn_to(0.0, tol_deg=2.0, timeout=3.0)
                yield from self.hold(0.6)
            ins, above = targets()
            yield from self.jmove2({s_: self.approach_q(s_, ins[s_], above[s_])[0] for s_ in SIDES}, 1.0)
        self.dbg("idle-ab", idle, above[idle])
        self.dbg("pick-ab", side, above[side])
        bp, yaw = self.base()
        rl, pt = self.att()
        self.log(f"expert: hook start base {np.round(bp, 3)} yaw {math.degrees(yaw):.1f} roll {rl:.1f} pitch {pt:.1f} "
                 f"shoulders {np.round(self.shoulder('left'), 3)} {np.round(self.shoulder('right'), 3)}")

        def in_place(s_):
            """The pivot, relative to its handle loop (tray frame): behind the bar, between the posts, under the bar."""
            tp_, tR_ = self.tray()
            rel = tR_.T @ (self.trk[s_].pivot() - tp_) - peg_loc[s_]
            return bool(-0.06 < rel[0] < -0.005 and abs(rel[1]) < 0.025
                        and W.EAR_TOP - 0.06 < rel[2] < W.EAR_TOP - 0.022), rel

        ok = False
        for attempt in range(p.retries + 1):
            tp, tR = self.tray()
            ax = tR[:, 0]
            ins = {s_: tp + tR @ ins_loc[s_] for s_ in SIDES}
            pre = {s_: ins[s_] - p.ear_pre * ax for s_ in SIDES}              # behind the loop, at entry height
            above = {s_: pre[s_] + np.array([0, 0, 0.06]) for s_ in SIDES}
            for s_ in SIDES:
                self.trk[s_].reset_integral()
            yield from self.move(above, 0.5)
            yield from self.move(pre, 0.8)
            yield from self.move(ins, 1.0)                                 # slide in under the bar
            yield from self.move(ins, 0.3)
            for s_ in SIDES:
                self.dbg("t-ins", s_, ins[s_])
            placed = {s_: in_place(s_) for s_ in SIDES}
            self.notes["ear_err"] = max(float(np.linalg.norm(self.trk[s_].pivot() - ins[s_])) for s_ in SIDES)
            ok = all(v[0] for v in placed.values())
            self.log(f"expert: loop attempt {attempt} placed "
                     f"{ {k: (v[0], np.round(v[1], 3).tolist()) for k, v in placed.items()} } ok {ok}")
            if ok:
                break
            yield from self.move(pre, 0.8)                                 # back out and re-plan
        if not ok:
            return False
        tz0 = self.tray()[0][2]
        yield from self.move({s: ins[s] + np.array([0, 0, p.tray_lift]) for s in SIDES}, 2.0)
        yield from self.hold(0.5)
        for _ in range(2):                                   # level the tray: raise the low side's hand
            tp, tR = self.tray()
            dz = (tR @ np.array([0.0, 1.0, 0.0]))[2] * W.ear_center_y()     # left ridge height minus the centre's
            if abs(dz) < 0.004:
                break
            goals = {s_: self.trk[s_].pivot() + np.array([0, 0, (-dz if s_ == "left" else dz) * 0.9]) for s_ in SIDES}
            yield from self.move(goals, 0.6)
            yield from self.hold(0.3)
        tp, tR = self.tray()
        self.notes["tray_lift"] = float(tp[2] - tz0)
        self.notes["tray_tilt_lifted"] = round(math.degrees(W.tilt_of(tR)), 1)
        self.log(f"expert: tray lifted {self.notes['tray_lift']:.3f} tilt {self.notes['tray_tilt_lifted']:.1f} "
                 f"ear err {self.notes['ear_err']:.3f}")
        if self.notes["tray_lift"] < 0.015 or self.notes["tray_tilt_lifted"] > 12.0:
            return False
        # freeze the arm joint targets for the walk (the tray moves with the body)
        self.carry_q = self.arms.copy()
        return True

    # C ---------------------------------------------------------------------------------------------------------
    def _goto(self, goal_xy, goal_yaw, max_v, timeout, vmin=0.08, track=None):
        """Closed loop on the base pose: P control in the body frame with speed caps and a minimum speed (the WBC
        barely moves for very small commands). `track(...)` is called every step (logging)."""
        p = self.p
        n = int(round(timeout / 0.08))
        for _ in range(n):
            bp, yaw = self.base()
            e = np.asarray(goal_xy() if callable(goal_xy) else goal_xy) - bp[:2]
            c, s = math.cos(yaw), math.sin(yaw)
            ex, ey = c * e[0] + s * e[1], -s * e[0] + c * e[1]
            eyaw = math.remainder(goal_yaw - yaw, 2 * math.pi)
            dist = math.hypot(ex, ey)
            if dist < p.pos_tol and abs(math.degrees(eyaw)) < p.yaw_tol_deg:
                break
            if dist >= p.pos_tol and max_v > 0:
                sp = min(max(1.5 * dist, vmin), max_v)
                vx, vy = sp * ex / dist, float(np.clip(sp * ey / dist, -0.3, 0.3))
            else:
                vx = vy = 0.0
            wz = 0.0 if abs(math.degrees(eyaw)) < p.yaw_tol_deg and dist < p.pos_tol else \
                float(np.clip(1.8 * eyaw, -p.turn_w, p.turn_w))
            if 0.0 < abs(wz) < 0.12:
                wz = math.copysign(0.12, wz)
            prev = self.cmd.copy()
            lim = np.array([p.cmd_slew, p.cmd_slew, 2.0 * p.cmd_slew])
            self.cmd[:] = prev + np.clip(np.array([vx, vy, wz]) - prev, -lim, lim)
            if track is not None:
                track()
            yield self._action()
        self.cmd[:] = 0.0

    def _recenter(self, goal_xy, goal_yaw=0.0, k=0.5, vmax=0.06, timeout=3.0):
        """Gentle re-centring of the standing base: proportional commands capped at vmax, NO minimum speed, so once the
        error is small the command falls under the stance threshold and the clock hold re-engages (the station loop's
        minimum speed and gain overshot by 5-15 cm with this WBC: dev2 env22/dbg25)."""
        for _ in range(int(round(timeout / 0.08))):
            bp, yaw = self.base()
            e = np.asarray(goal_xy, float) - bp[:2]
            c, s_ = math.cos(yaw), math.sin(yaw)
            ex, ey = c * e[0] + s_ * e[1], -s_ * e[0] + c * e[1]
            eyaw = math.remainder(goal_yaw - yaw, 2 * math.pi)
            v = np.array([np.clip(k * ex, -vmax, vmax), np.clip(k * ey, -vmax, vmax), np.clip(1.0 * eyaw, -0.15, 0.15)])
            if np.all(np.abs(v) < np.array([0.02, 0.02, 0.05])):
                break
            v[1] += self.p.recenter_vy_bias
            self.cmd[:] = v
            yield self._action()
        self.cmd[:] = 0.0
        yield from self.hold(self.p.recenter_settle)
        bp, yaw = self.base()
        self.log(f"expert: recentered base {np.round(bp[:2], 3)} yaw {math.degrees(yaw):.1f}")

    def _step_to(self, goal_x, goal_yaw=0.0):
        """Walk straight to base x = goal_x (world): first turn in place to goal_yaw if off by more than 3 deg, then vx
        only, at >= step_vmin (the WBC barely moves for smaller commands), stop within 2 cm; then stand still."""
        p = self.p
        if abs(math.degrees(math.remainder(goal_yaw - self.base()[1], 2 * math.pi))) > 3.0:
            yield from self._turn_to(goal_yaw, tol_deg=2.0, timeout=3.0)
        bp0, yaw0 = self.base()
        for _ in range(int(round(p.step_timeout / 0.08))):
            bp, yaw = self.base()
            ex = math.cos(yaw) * (goal_x - bp[0]) + math.sin(yaw) * (0.0 - bp[1])
            if abs(ex) < 0.02:
                break
            self.cmd[:] = (math.copysign(min(max(p.step_k * abs(ex), p.step_vmin), p.step_vmax), ex), 0.0, 0.0)
            yield self._action()
        self.cmd[:] = 0.0
        for _ in range(int(round(p.step_timeout / 0.08))):     # then sideways, if the walk drifted (it does, to the left)
            bp, yaw = self.base()
            ey = -math.sin(yaw) * (goal_x - bp[0]) + math.cos(yaw) * (0.0 - bp[1])
            if abs(ey) < 0.02:
                break
            self.cmd[:] = (0.0, math.copysign(min(max(p.step_k * abs(ey), p.step_vmin), p.step_vmax), ey), 0.0)
            yield self._action()
        self.cmd[:] = 0.0
        yield from self.hold(p.recenter_settle)
        bp, yaw = self.base()
        self.log(f"expert: stepped base {np.round(bp0[:2], 3)} -> {np.round(bp[:2], 3)} yaw {math.degrees(yaw0):.1f} -> "
                 f"{math.degrees(yaw):.1f} (goal x {goal_x:.2f})")

    def _turn_to(self, goal_yaw, tol_deg=3.0, timeout=3.0):
        """Turn in place (command wz only, P with a minimum rate), then stop and let the stance hold engage."""
        for _ in range(int(round(timeout / 0.08))):
            e = math.remainder(goal_yaw - self.base()[1], 2 * math.pi)
            if abs(math.degrees(e)) < tol_deg:
                break
            self.cmd[:] = (0.0, 0.0, math.copysign(min(max(1.5 * abs(e), 0.15), 0.4), e))
            yield self._action()
        self.cmd[:] = 0.0
        yield from self.hold(0.6)

    def _station(self, goal_xy=(0.0, 0.0), goal_yaw=0.0, tag="", tol=None):
        """Walk back onto the manipulation station (closed loop), stop, and let the stance hold engage."""
        p = self.p

        def errs():
            bp, yaw = self.base()
            e = np.asarray(goal_xy, float) - bp[:2]
            c, s_ = math.cos(yaw), math.sin(yaw)
            return c * e[0] + s_ * e[1], -s_ * e[0] + c * e[1], math.remainder(goal_yaw - yaw, 2 * math.pi)

        tx_, ty_, tyaw_ = tol if tol is not None else (p.station_tol_x, p.station_tol_y, p.station_yaw_tol_deg)

        def inside(ex, ey, eyaw, f=1.0):
            return abs(ex) < f * tx_ and abs(ey) < f * ty_ and abs(math.degrees(eyaw)) < f * tyaw_

        for k in range(p.station_tries):
            if inside(*errs()):
                break
            for _ in range(int(round(4.0 / 0.08))):          # P control with a minimum speed (the WBC ignores
                ex, ey, eyaw = errs()                        # very small commands), until well inside
                if inside(ex, ey, eyaw, 0.5):
                    break
                v = [ex, ey, eyaw]
                lim = (p.station_v, p.station_v, 0.3)
                vmin = (p.station_vmin, p.station_vmin, 0.08)
                tol = (0.5 * tx_, 0.5 * ty_, math.radians(0.5 * tyaw_))
                for i in range(3):
                    v[i] = 0.0 if abs(v[i]) < tol[i] else math.copysign(min(max(abs(1.0 * v[i]), vmin[i]), lim[i]), v[i])
                self.cmd[:] = v
                yield self._action()
            self.cmd[:] = 0.0
            yield from self.hold(0.6)
        bp, yaw = self.base()
        self.log(f"expert: station {tag} base {np.round(bp[:2], 3)} yaw {math.degrees(yaw):.1f}")

    def _deliver(self):
        p, W = self.p, self.W
        self.phase = "C"
        side_t = self.env.layout.target_side
        bp, yaw = self.base()
        Rb = yaw_rot(yaw)
        tp, _ = self.tray()
        self.tray_off = (Rb.T @ (tp - bp))[:2]           # tray centre in the base frame while carried
        mx = {"tray": 0.0, "cup": 0.0}

        def track():
            mx["tray"] = max(mx["tray"], math.degrees(W.tilt_of(self.tray()[1])))
            mx["cup"] = max(mx["cup"], self.cup_tilt())
        # back off table A keeping the heading, turn in place toward the instructed table, walk to the stand point
        yield from self._goto(np.array([p.backoff_x, bp[1]]), yaw, 0.25, 5.0, track=track)
        goal_yaw = W.DELIVERY_YAW[side_t]
        yield from self._goto(self.base()[0][:2], goal_yaw, 0.0, 5.0, track=track)
        zc, zyaw = W.zone_world(self.env.d, self.env.ids, side_t)
        aim = zc + yaw_rot(zyaw)[:2, :2] @ np.array([p.zone_aim_dx, 0.0])
        # the stand point follows where the tray actually hangs in the base frame (low-passed: it swings): walking with
        # the tray the free robot leans back and the tray hung ~15 cm nearer than at table A (dev2 P3)
        off = {"v": self.tray_off.copy()}

        def stand():
            bp_, yaw_ = self.base()
            now = (yaw_rot(yaw_).T @ (self.tray()[0] - bp_))[:2]
            off["v"] = off["v"] + p.tray_off_alpha * (now - off["v"])
            return aim - yaw_rot(goal_yaw)[:2, :2] @ off["v"]
        yield from self._goto(stand, goal_yaw, p.walk_v, 12.0, track=track)
        self.cmd[:] = 0.0
        yield from self.hold(p.setdown_settle)
        stand = stand()
        bp, yaw = self.base()
        self.notes.update(walk_tray_tilt_max=round(mx["tray"], 1), walk_cup_tilt_max=round(mx["cup"], 1))
        self.log(f"expert: at stand point err {np.round(stand - bp[:2], 3)} yaw err "
                 f"{math.degrees(math.remainder(goal_yaw - yaw, 2 * math.pi)):.1f} | walk max tilt tray {mx['tray']:.1f} "
                 f"cup {mx['cup']:.1f}")

    # D ---------------------------------------------------------------------------------------------------------
    def _set_down(self):
        p, W = self.p, self.W
        self.phase = "D"
        for s in SIDES:
            self.trk[s].qdes = self.q_arm(s)
            self.trk[s].reset_integral()
        start = {s: self.trk[s].pivot() for s in SIDES}
        drop = 0.0
        for _ in range(40):                                  # 4 mm per 0.16 s until the tray rests
            drop += 0.004
            yield from self.move({s: start[s] - np.array([0, 0, drop]) for s in SIDES}, 0.16)
            if self.tray()[0][2] < W.TABLE_TOP + 0.003:
                break
        self.dbg_setdown("rests")
        yield from self.move({s: start[s] - np.array([0, 0, drop + p.set_extra]) for s in SIDES}, 0.6)
        yield from self.grip_to(SIDES, 0.0, 0.3)
        f = {s: self.trk[s].fwd() for s in SIDES}
        cur = {s: self.trk[s].pivot() - 0.075 * f[s] for s in SIDES}
        yield from self.move(cur, 1.2)
        yield from self.move({s: cur[s] + np.array([0, 0, 0.05]) for s in SIDES}, 0.8)
        self.dbg_setdown("released")
