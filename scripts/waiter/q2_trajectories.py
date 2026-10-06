"""Waiter phase 1 gate Q2: the four upper-body trajectories (docs/WAITER_PROGRAM.md; frozen 2026-10-05).

Committed before any Waiter WBC checkpoint was evaluated. Each trajectory maps t in [0, 8] s to the 12 upper-body
targets (absolute rad, deploy order: left arm 5, right arm 5, left gripper, right gripper). Kinematics (MuJoCo FK
of the Waiter model, arms hanging at 0): left shoulder pitch < 0 swings the arm forward, left elbow pitch > 0
brings the forearm forward; the right arm mirrors the left by negation (every limit pair is negated).

T1 carry: both arms to a tray-carry pose (upper arm 0.25 rad forward, forearm about level and forward) over
   1.5 s, then held; grippers close at 1.5-2.0 s and stay closed.
T2 reach: the right arm reaches forward-down (shoulder 0.9 rad forward, elbow 0.6), closes its gripper, returns,
   opens; a 4 s cycle, twice.
T3 raise: both arms raise forward to 1.2 rad and lower, three times in 8 s (raised-cosine profile), grippers open.
T4 random: smooth random goals from the training distribution (waiter_wbc_mdp.sample_goals' rule, re-implemented
   in numpy), a new goal every 2.0 s reached by linear interpolation over 0.8 s, numpy seed 20261005.
"""
from __future__ import annotations

import math

import numpy as np

GRIPPER_CLOSED = 1.20
EPISODE_S = 8.0
LEFT_LIMITS = np.array([(-1.5708, 0.785398), (-0.261799, 1.309), (-0.785398, 0.785398), (0.0, 1.5708),
                        (-0.785398, 0.785398)])
RIGHT_LIMITS = -LEFT_LIMITS[:, ::-1]
CARRY_LEFT = np.array([-0.25, 0.0, 0.0, 1.35, 0.0])
REACH_RIGHT = -np.array([-0.9, 0.0, 0.0, 0.6, 0.0])       # the right arm's own sign convention
RAISE_LEFT = np.array([-1.2, 0.0, 0.0, 0.0, 0.0])
T4_SEED = 20261005


def _smooth(u: float) -> float:
    u = min(max(u, 0.0), 1.0)
    return 0.5 - 0.5 * math.cos(math.pi * u)


def _pack(left, right, gl, gr) -> np.ndarray:
    return np.concatenate([np.asarray(left, float), np.asarray(right, float), [gl, gr]]).astype(np.float32)


def t1_carry(t: float) -> np.ndarray:
    a = _smooth(t / 1.5)
    g = GRIPPER_CLOSED * _smooth((t - 1.5) / 0.5)
    return _pack(a * CARRY_LEFT, -a * CARRY_LEFT, g, g)


def t2_reach(t: float) -> np.ndarray:
    c = t % 4.0                                     # 0-1.5 reach, 1.5-2.0 close, 2.0-3.5 return, 3.5-4.0 open
    if c < 1.5:
        a, g = _smooth(c / 1.5), 0.0
    elif c < 2.0:
        a, g = 1.0, GRIPPER_CLOSED * _smooth((c - 1.5) / 0.5)
    elif c < 3.5:
        a, g = 1.0 - _smooth((c - 2.0) / 1.5), GRIPPER_CLOSED
    else:
        a, g = 0.0, GRIPPER_CLOSED * (1.0 - _smooth((c - 3.5) / 0.5))
    return _pack(np.zeros(5), a * REACH_RIGHT, 0.0, g)


def t3_raise(t: float) -> np.ndarray:
    a = 0.5 - 0.5 * math.cos(2.0 * math.pi * t / (EPISODE_S / 3.0))
    return _pack(a * RAISE_LEFT, -a * RAISE_LEFT, 0.0, 0.0)


def _sample_goal(rng: np.random.Generator) -> np.ndarray:
    """One goal by waiter_wbc_mdp.sample_goals' rule (p_default 0.25, p_uniform 0.5, rest mirrored; middle 70%;
    each gripper closed with p 0.5)."""
    lo = np.r_[LEFT_LIMITS[:, 0], RIGHT_LIMITS[:, 0]]
    hi = np.r_[LEFT_LIMITS[:, 1], RIGHT_LIMITS[:, 1]]
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * 0.7
    u = rng.random()
    arms = mid + (2.0 * rng.random(10) - 1.0) * half
    if u < 0.25:
        arms = np.zeros(10)
    elif u >= 0.75:
        arms[5:] = np.clip(-arms[:5], lo[5:], hi[5:])
    g = (rng.random(2) < 0.5).astype(float) * GRIPPER_CLOSED
    return np.r_[arms, g].astype(np.float32)


def _t4_table() -> list[np.ndarray]:
    rng = np.random.default_rng(T4_SEED)
    return [np.zeros(12, np.float32)] + [_sample_goal(rng) for _ in range(int(EPISODE_S / 2.0) + 1)]


_T4 = _t4_table()


def t4_random(t: float) -> np.ndarray:
    k = int(t // 2.0)
    a = min(max((t - 2.0 * k) / 0.8, 0.0), 1.0)
    return (_T4[k] + a * (_T4[k + 1] - _T4[k])).astype(np.float32)


TRAJECTORIES = {"T1_carry": t1_carry, "T2_reach": t2_reach, "T3_raise": t3_raise, "T4_random": t4_random}


def within_limits() -> bool:
    lo = np.r_[LEFT_LIMITS[:, 0], RIGHT_LIMITS[:, 0], 0.0, 0.0] - 1e-6
    hi = np.r_[LEFT_LIMITS[:, 1], RIGHT_LIMITS[:, 1], GRIPPER_CLOSED, GRIPPER_CLOSED] + 1e-6
    return all(bool(np.all((f(t) >= lo) & (f(t) <= hi))) for f in TRAJECTORIES.values()
               for t in np.arange(0.0, EPISODE_S, 0.04))
