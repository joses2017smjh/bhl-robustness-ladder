"""A scratch subclass of NavGym's MazeNavEnv (version 2) that swaps in the physics-identified pieces one at a time,
for the predictive check (does the identified model reproduce the physics transfer failures?) and for trying
deployment-time command filters. Nothing in the repository is modified; this only subclasses it.

Options (all off = MazeNavEnv(version=2) behaviour, step for step, except that the dynamics object is pluggable):
  dyn_model      None (the gym's own per-episode Dynamics) or a callable rng -> model with .reset() and
                 .step(v_cmd, w_cmd) -> (v, w) in m/s, rad/s (see IdentifiedDynamics below)
  brake          team_sensors.brake_command's range term on the 10 Hz packet: v_cmd *= clip((clear - 0.42)/0.48, 0, 1),
                 clear = min of the packet's sector minima whose centres are within 25 deg of straight ahead
  obs10hz        lidar packets at the physics schedule (EXTERO_PERIOD 0.1 s -> captures at steps 0,3,5,8,10,...), the
                 packet captured at the PRE-step pose and integrated into the ego map after the step at the NEW pose
                 (maze_explore's biped path); the lidar/near obs keys hold the packet's body-frame sectors until the next
  capture_fix    with obs10hz: integrate each packet at its capture pose instead (the proposed deployment fix)
  heading0       initial heading 0 (maze_explore without --random-heading) instead of uniform(-pi, pi)
  settle_steps   first N steps: the command is zeroed (maze_explore's 1.0 s settle = 25 steps); prev_action keeps the actor's
  cmd_filter     None or an object with .reset() and .apply(v_cmd, w_cmd) -> (v_cmd, w_cmd): a deployment filter placed
                 between the actor and the brake/gait (the actor's prev_action observation stays its own raw action)
"""
from __future__ import annotations

import math
import numpy as np

from bhl_robust.navgym.env import (DT, V_MAX, W_MAX, LIDAR_RANGE, LIDAR_RAYS, LIDAR_SECTORS, EgoMap, MazeNavEnv, V4_IDLE_A0_MAX,
                                   V4_IDLE_FAR_M, cast_rays, sector_minima)

EXTERO_PERIOD = 0.1
_ANG = np.linspace(-np.pi, np.pi, LIDAR_RAYS, endpoint=False)
_CENTRES = _ANG.reshape(LIDAR_SECTORS, -1).mean(axis=1)
_FWD = np.abs(np.arctan2(np.sin(_CENTRES), np.cos(_CENTRES))) <= np.deg2rad(25)


class GymDefaultDynamics:
    """The gym's own Dynamics (d) with its step equations (latency queue, Euler lag, drift, noise)."""

    def __init__(self, d, rng):
        self.d, self.rng = d, rng

    def reset(self):
        self.q = [np.zeros(2) for _ in range(self.d.latency)]
        self.v = self.w = 0.0

    def step(self, v_cmd, w_cmd):
        d = self.d
        self.q.append(np.array([v_cmd, w_cmd]))
        vc, wc = self.q.pop(0)
        a = DT / max(d.tau, DT)
        self.v += a * (d.v_gain * vc - self.v)
        self.w += a * (d.w_gain * wc - self.w)
        v = self.v + self.rng.normal(0, d.v_noise)
        w = self.w + self.rng.normal(0, d.w_noise) + d.drift * (self.v / V_MAX)
        return v, w


class IdentifiedDynamics:
    """Separate first-order-plus-dead-time channels (Euler lag, integer dead time in steps, as the gym discretises),
    an optional input nonlinearity on wz (callable), an optional wz-dependent forward-speed factor, drift, and
    an optional periodic yaw wobble (amplitude A rad/s at f Hz, random phase) plus white noise."""

    def __init__(self, rng, Kw=1.17, tau_w=0.25, Lw=1, Kv=1.0, tau_v=0.25, Lv=1, drift=0.0, w_noise=0.0, v_noise=0.0,
                 w_in=None, v_turn_factor=None, wobble_amp=0.0, wobble_hz=1.8, w_state_fn=None, v_in=None, chatter_v_factor=1.0,
                 drift_range=None):
        self.rng = rng
        self.Kw, self.tau_w, self.Lw, self.Kv, self.tau_v, self.Lv = Kw, tau_w, int(Lw), Kv, tau_v, int(Lv)
        self.drift, self.w_noise, self.v_noise = drift, w_noise, v_noise
        if drift_range is not None:
            self.drift = float(rng.uniform(*drift_range))
        self.w_in, self.v_turn_factor = w_in, v_turn_factor
        self.wobble_amp, self.wobble_hz = wobble_amp, wobble_hz
        self.w_state_fn = w_state_fn
        self.v_in, self.chatter_v_factor = v_in, chatter_v_factor

    def reset(self):
        self.qw = [0.0] * self.Lw
        self.qv = [0.0] * self.Lv
        self.v = self.w = 0.0
        self.k = 0
        self.phase = float(self.rng.uniform(0, 2 * np.pi))
        self.hist = []

    def step(self, v_cmd, w_cmd):
        self.hist.append(w_cmd)
        wi = self.w_in(w_cmd, self.hist) if self.w_in is not None else w_cmd
        self.qw.append(wi); wc = self.qw.pop(0)
        self.qv.append(self.v_in(v_cmd) if self.v_in is not None else v_cmd); vc = self.qv.pop(0)
        aw = DT / max(self.tau_w, DT); av = DT / max(self.tau_v, DT)
        self.w += aw * (self.Kw * wc - self.w)
        vf = self.v_turn_factor(self.w) if self.v_turn_factor is not None else 1.0
        if self.chatter_v_factor != 1.0 and len(self.hist) >= 2 and self.hist[-1] * self.hist[-2] < 0:
            vf *= self.chatter_v_factor
        self.v += av * (self.Kv * vf * vc - self.v)
        self.k += 1
        wob = self.wobble_amp * math.sin(2 * math.pi * self.wobble_hz * self.k * DT + self.phase) if self.wobble_amp else 0.0
        v = self.v + (self.rng.normal(0, self.v_noise) if self.v_noise else 0.0)
        w = self.w + wob + (self.rng.normal(0, self.w_noise) if self.w_noise else 0.0) + self.drift * (self.v / V_MAX)
        return v, w


class SysidNavEnv(MazeNavEnv):
    def __init__(self, *args, dyn_model=None, brake=False, obs10hz=False, capture_fix=False, heading0=False, settle_steps=0,
                 cmd_filter=None, brake_lo=0.42, brake_span=0.48, **kw):
        kw.setdefault("version", 2)
        self.brake_lo, self.brake_span = float(brake_lo), float(brake_span)
        super().__init__(*args, **kw)
        self.dyn_model_factory, self.brake, self.obs10hz, self.capture_fix = dyn_model, brake, obs10hz, capture_fix
        self.heading0, self.settle_steps, self.cmd_filter = heading0, int(settle_steps), cmd_filter
        self.stats = {}

    # ------------------------------------------------------------------ reset
    def reset(self, seed=None, options=None):
        self._pending = None
        obs, info = super().reset(seed=seed, options=options)     # draws dynamics, random yaw, scans (at the random yaw)
        if self.heading0:
            self.yaw = 0.0
            self.emap = EgoMap(self.maze.bounds())
            self._pending = None
            self.ranges = cast_rays(self.boxes, (self.x, self.y), self.yaw + self.angles, LIDAR_RANGE)
            self.emap.update(self.x, self.y, self.yaw, self.angles, self.ranges)
            obs = self._obs()
        self.model = (self.dyn_model_factory(self._rng) if self.dyn_model_factory is not None else GymDefaultDynamics(self.dyn, self._rng))
        self.model.reset()
        if self.cmd_filter is not None:
            self.cmd_filter.reset()
        self.next_capture = 0.0
        self.packet = {"ranges": self.ranges.copy(), "pose": (self.x, self.y, self.yaw)}   # the reset scan stands in for the first packet
        self.stats = {"steps": 0, "braked": 0, "brake_scale_sum": 0.0, "captures": 0}
        return obs, info

    # ------------------------------------------------------------------ physics-like sensing
    def _capture_if_due(self):
        """maze_explore: TeamSensors.filter_commands captures at the top of the step when now >= next_capture."""
        now = self.t * DT
        if now + 1e-9 >= self.next_capture:
            while self.next_capture <= now + 1e-9:
                self.next_capture += EXTERO_PERIOD
            rays = cast_rays(self.boxes, (self.x, self.y), self.yaw + self.angles, LIDAR_RANGE)
            self.packet = {"ranges": rays, "pose": (self.x, self.y, self.yaw)}
            self._pending = self.packet
            self.stats["captures"] += 1

    def _scan(self):
        if not getattr(self, "obs10hz", False) or not hasattr(self, "packet"):
            return super()._scan()
        # obs10hz: integrate a packet captured on this step, at the NEW pose (as maze_explore's biped path) or at its capture pose
        if self._pending is not None:
            px, py, pyaw = self._pending["pose"] if self.capture_fix else (self.x, self.y, self.yaw)
            self.emap.update(px, py, pyaw, self.angles, self._pending["ranges"])
            self.ranges = self._pending["ranges"]
            self._pending = None

    # ------------------------------------------------------------------ step
    def step(self, action):
        a = np.clip(np.asarray(action, dtype=np.float32), -1.0, 1.0)
        idle = bool(self.idle_cost) and float(a[0]) <= V4_IDLE_A0_MAX and \
            math.hypot(self.goal_xy[0] - self.x, self.goal_xy[1] - self.y) > V4_IDLE_FAR_M
        self.prev_action = a.copy()
        v_cmd = (a[0] + 1.0) * 0.5 * V_MAX
        w_cmd = a[1] * W_MAX
        if self.t < self.settle_steps:
            v_cmd = w_cmd = 0.0
        if self.cmd_filter is not None:
            v_cmd, w_cmd = self.cmd_filter.apply(v_cmd, w_cmd)
        if self.obs10hz or self.brake:
            self._capture_if_due()
        if not self.obs10hz:
            self._pending = None
        if self.brake and v_cmd > 1e-8:
            sec = self.packet["ranges"].reshape(LIDAR_SECTORS, -1).min(axis=1)
            clear = float(np.min(sec[_FWD]))
            scale = float(np.clip((clear - self.brake_lo) / self.brake_span, 0.0, 1.0))
            if scale < 1.0:
                self.stats["braked"] += 1
            self.stats["brake_scale_sum"] += scale
            v_cmd *= scale
        self.stats["steps"] += 1
        v, w = self.model.step(v_cmd, w_cmd)
        self.yaw = (self.yaw + w * DT + math.pi) % (2 * math.pi) - math.pi
        nx, ny = self.x + v * DT * math.cos(self.yaw), self.y + v * DT * math.sin(self.yaw)
        self.t += 1
        return self._finish_v2(nx, ny, idle)


# ----------------------------------------------------------------------- deployment filters
class LowPassW:
    """First-order low-pass on wz only (Euler, time constant tau_f)."""

    def __init__(self, tau_f=0.2, on_v=False):
        self.tau_f, self.on_v = tau_f, on_v

    def reset(self):
        self.w = 0.0; self.v = 0.0

    def apply(self, v, w):
        a = DT / max(self.tau_f, DT)
        self.w += a * (w - self.w)
        if self.on_v:
            self.v += a * (v - self.v)
            return self.v, self.w
        return v, self.w


class MovingAvgW:
    def __init__(self, n=5):
        self.n = n

    def reset(self):
        self.buf = []

    def apply(self, v, w):
        self.buf.append(w)
        self.buf = self.buf[-self.n:]
        return v, float(np.mean(self.buf))


class RateLimitW:
    def __init__(self, rate=5.0):
        self.rate = rate

    def reset(self):
        self.w = 0.0

    def apply(self, v, w):
        d = self.rate * DT
        self.w = float(np.clip(w, self.w - d, self.w + d))
        return v, self.w


class PhysDynamics:
    """The physics-identified gait model (dr-default-s0, MuJoCo, flat floor; data/*_summary.json):
    forward  v_ss = max(0, 1.19 v_cmd - 0.063) (gain 1.01 @0.35, 0.95 @0.25, 0.77 @0.15; dead below 0.053 m/s),
             Euler lag tau 0.13 s, no dead time; from standstill with v_cmd < 0.2 a slow start (tau 1.0 s);
             x0.93 on steps where the wz command flips sign (chatter costs 5-10 % speed, PRBS p=0.5-0.7)
    yaw      walking (v >= 0.06): lag alpha 0.62/step (tau 0.065 s, FIR 0.62 then 1.0), dead time 0, input map
             g(u) = u (1 - 0.296|u|)/0.704 (concave: 1.33x @0.3, 1.2x @0.6, 1x @1), gain Kw = 1.05 at |u| = 1;
             standing (v < 0.06): in-place turns need a sustained command -- u_eff = sign(m) max(0, |m| - 0.3)/0.7 of the
             0.48 s mean m of the command (PWM/chatter at |m| <= 0.5 barely turns, const 0.33 does not turn), gain 1.15
    wobble   stride-locked yaw-rate oscillation 0.40 rad/s amplitude at 1.75 Hz, scaled by v/0.35 (heading +-2 deg)
    drift    uniform(-0.02, 0.02) rad/s per episode while walking (measured -0.009..+0.017)"""

    def __init__(self, rng, Kw=1.05, wobble=True, drift=True, standing_deadzone=True, v_map=True, chatter_v=0.93):
        self.rng, self.Kw, self.wobble_on, self.standing_deadzone, self.v_map, self.chatter_v = rng, Kw, wobble, standing_deadzone, v_map, chatter_v
        self.drift = float(rng.uniform(-0.02, 0.02)) if drift else 0.0

    def reset(self):
        self.v = self.w = 0.0
        self.hist = []
        self.k = 0
        self.phase = float(self.rng.uniform(0, 2 * np.pi))

    def step(self, v_cmd, w_cmd):
        self.hist.append(float(w_cmd)); self.hist = self.hist[-12:]
        vt = max(0.0, 1.19 * v_cmd - 0.063) if self.v_map else v_cmd
        if len(self.hist) >= 2 and self.hist[-1] * self.hist[-2] < 0:
            vt *= self.chatter_v
        tau_v = 1.0 if (self.v < 0.05 and v_cmd < 0.2 and self.v_map) else 0.13
        self.v += (DT / tau_v) * (vt - self.v)
        if self.standing_deadzone and self.v < 0.06:
            m = float(np.mean(self.hist))
            u = math.copysign(max(0.0, abs(m) - 0.3) / 0.7, m)
            self.w += (DT / 0.10) * (1.15 * u - self.w)
        else:
            u = w_cmd * (1.0 - 0.296 * abs(w_cmd)) / 0.704
            self.w += 0.62 * (self.Kw * u - self.w)
        self.k += 1
        wob = 0.40 * (self.v / 0.35) * math.sin(2 * math.pi * 1.75 * self.k * DT + self.phase) if self.wobble_on else 0.0
        return self.v, self.w + wob + self.drift * (self.v / V_MAX)
