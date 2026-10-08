"""Deployment-side smoothing of the learned NavGym navigator's command (2026-10-07; SCRIPTED, opt-in).

Why. The NavGym v5 actors turn bang-bang: |wz| is saturated on most physics steps and flips sign many times a second
(solutions-20260930/REPORTS_completed.md, section D). The gym models the gait as a first-order lag on both channels
(``navgym/env.py`` ``Dynamics``: tau uniform 0.15-0.45 s, nominal 0.25 s), which averages the chatter away, so the
chatter costs the actor almost nothing in training. The physical gait answers much faster: the PRBS identification in
``solutions-20260930/sysid`` fits w_gain 1.03, tau 0.06 s, latency 0. In MuJoCo the chatter therefore reaches the gait.
In the 2026-10-06 confirmation, every one of the 6 falls (288 learned episodes; A* 0 in 96) happened while the
commanded wz flipped between -1 and +1 rad/s at walking speed.

``CommandLowPass`` applies the gym's own lag, ``x += alpha * (u - x)`` with ``alpha = dt / max(tau, dt)`` (the update in
``MazeNavEnv.step``), to chosen command channels. ``DEFAULT_TAU_S`` = 0.2 s is the gym's nominal 0.25 s minus the
gait's measured 0.06 s, so the filtered physics robot answers like the gym's nominal one. It was chosen from those two
numbers, not tuned on maze outcomes.

Modes (``maze_explore.py --policy-cmd-filter``):
  none            no filter, but the command statistics are recorded (the explicit control arm);
  wz-lpf          the actor's wz is filtered BEFORE the speed brake; vx is untouched;
  lpf-post-brake  vx and wz are filtered AFTER the brake (the gym's order: brake, then lag).
The actor's ``prev_action`` observation stays its own raw output in every mode, as in the gym.
"""
from __future__ import annotations

import numpy as np

CMD_FILTER_MODES = ("none", "wz-lpf", "lpf-post-brake")
DEFAULT_TAU_S = 0.2
GYM_NOMINAL_TAU_S = 0.25        # navgym/env.py Dynamics().tau
GAIT_MEASURED_TAU_S = 0.06      # solutions-20260930/sysid PRBS best fit (w_gain 1.03, latency 0)
SATURATED = 0.99                # |wz| >= 0.99 * W_MAX counts as saturated (navgym/env.py V3_YAW_SAT)

MODE_CHANNELS = {"wz-lpf": (2,), "lpf-post-brake": (0, 2)}


class CommandLowPass:
    """First-order low-pass on selected channels of a (vx, vy, wz) command, discretized as the NavGym lag."""

    def __init__(self, dt: float, tau: float, channels=(2,)):
        if not (dt > 0.0 and tau > 0.0):
            raise ValueError(f"CommandLowPass needs dt > 0 and tau > 0, got dt={dt}, tau={tau}")
        channels = tuple(int(c) for c in channels)
        if not channels or any(c not in (0, 1, 2) for c in channels):
            raise ValueError(f"channels must be a non-empty subset of (0, 1, 2), got {channels}")
        self.dt, self.tau = float(dt), float(tau)
        self.alpha = self.dt / max(self.tau, self.dt)
        self.channels = channels
        self.state = np.zeros(3)

    def __call__(self, command) -> np.ndarray:
        u = np.asarray(command, dtype=float)
        out = u.copy()
        for c in self.channels:
            self.state[c] += self.alpha * (u[c] - self.state[c])
            out[c] = self.state[c]
        return out

    def describe(self) -> dict:
        return {"tau_s": self.tau, "alpha": round(self.alpha, 6), "channels": list(self.channels),
                "form": "x += alpha * (u - x), alpha = dt / max(tau, dt) (the NavGym lag)"}


def make_filter(mode: str, dt: float, tau: float = DEFAULT_TAU_S):
    """The filter for a --policy-cmd-filter mode, or None for 'none' (statistics only)."""
    if mode not in CMD_FILTER_MODES:
        raise ValueError(f"unknown command filter mode {mode!r}; expected one of {CMD_FILTER_MODES}")
    if mode == "none":
        return None
    return CommandLowPass(dt, tau, MODE_CHANNELS[mode])


class CommandStats:
    """Per-episode statistics of a wz command stream at the control rate: sign flips per second, saturated share."""

    def __init__(self, dt: float, w_max: float = 1.0):
        self.dt, self.w_max = float(dt), float(w_max)
        self.n = self.flips = self.saturated = 0
        self.abs_sum = 0.0
        self.prev = 0.0

    def update(self, wz: float) -> None:
        wz = float(wz)
        self.n += 1
        self.flips += int(np.sign(wz) * np.sign(self.prev) < 0)
        self.saturated += int(abs(wz) >= SATURATED * self.w_max)
        self.abs_sum += abs(wz)
        self.prev = wz

    def summary(self) -> dict:
        t = self.n * self.dt
        return {"steps": self.n, "flips_per_s": round(self.flips / t, 3) if t > 0 else None,
                "saturated_share": round(self.saturated / self.n, 4) if self.n else None,
                "mean_abs_wz": round(self.abs_sum / self.n, 4) if self.n else None}
