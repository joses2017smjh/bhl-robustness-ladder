"""Sweep primitive: garment pose → planar trajectory.

The policy (scripted or learned) outputs where and how to sweep. This module
turns that into a start point, a direction, and a timed approach/contact/
sweep/retract plan. It does not command the robot; ``controller.py`` does.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

ACTION_DIM = 5
ACTION_NAMES = (
    "dx_start",
    "dy_start",
    "sweep_angle",
    "sweep_distance",
    "sweep_speed",
)

#: Physical bounds the 5-D action is scaled into. Kept as config so a later
#: domain-randomisation pass can widen them without rewriting the trainer.
ACTION_SCALE = {
    "dx_start": 0.15,
    "dy_start": 0.15,
    "sweep_angle": float(np.pi),
    "sweep_distance": (0.10, 0.85),
    "sweep_speed": (0.08, 0.80),
}


@dataclass
class SweepConfig:
    """Timing and geometry of one sweep execution."""

    start_offset: float = 0.12
    hand_height: float = 0.36
    contact_height: float = 0.315
    approach_duration: float = 0.40
    sweep_duration: float = 0.80
    retract_duration: float = 0.30
    hand_radius: float = 0.04
    max_distance: float = 0.90
    min_speed: float = 0.05
    max_speed: float = 1.20


@dataclass(frozen=True)
class SweepPlan:
    """One planned sweep, in world XY plus a height schedule."""

    start_xy: np.ndarray
    end_xy: np.ndarray
    direction: np.ndarray
    angle: float
    distance: float
    speed: float
    contact_height: float
    hand_height: float
    approach_duration: float
    sweep_duration: float
    retract_duration: float

    @property
    def duration(self) -> float:
        return self.approach_duration + self.sweep_duration + self.retract_duration


def decode_action(action: np.ndarray) -> dict[str, float]:
    """Map a normalised ``[-1, 1]^5`` action into physical sweep parameters."""
    a = np.asarray(action, dtype=float).reshape(-1)
    if a.size != ACTION_DIM:
        raise ValueError(f"sweep action must have {ACTION_DIM} entries, got {a.size}")
    a = np.clip(a, -1.0, 1.0)
    lo_d, hi_d = ACTION_SCALE["sweep_distance"]
    lo_s, hi_s = ACTION_SCALE["sweep_speed"]
    return {
        "dx_start": float(a[0]) * ACTION_SCALE["dx_start"],
        "dy_start": float(a[1]) * ACTION_SCALE["dy_start"],
        "sweep_angle": float(a[2]) * ACTION_SCALE["sweep_angle"],
        "sweep_distance": lo_d + 0.5 * (float(a[3]) + 1.0) * (hi_d - lo_d),
        "sweep_speed": lo_s + 0.5 * (float(a[4]) + 1.0) * (hi_s - lo_s),
    }


def encode_physical(
    dx_start: float,
    dy_start: float,
    sweep_angle: float,
    sweep_distance: float,
    sweep_speed: float,
) -> np.ndarray:
    """Inverse of ``decode_action``. Used by the scripted baseline."""
    lo_d, hi_d = ACTION_SCALE["sweep_distance"]
    lo_s, hi_s = ACTION_SCALE["sweep_speed"]
    return np.array([
        np.clip(dx_start / ACTION_SCALE["dx_start"], -1.0, 1.0),
        np.clip(dy_start / ACTION_SCALE["dy_start"], -1.0, 1.0),
        np.clip(sweep_angle / ACTION_SCALE["sweep_angle"], -1.0, 1.0),
        np.clip(2.0 * (sweep_distance - lo_d) / (hi_d - lo_d) - 1.0, -1.0, 1.0),
        np.clip(2.0 * (sweep_speed - lo_s) / (hi_s - lo_s) - 1.0, -1.0, 1.0),
    ], dtype=float)


def plan_sweep(
    garment_xy: np.ndarray,
    params: dict[str, float],
    cfg: SweepConfig | None = None,
) -> SweepPlan:
    """Build a planar sweep from a garment position and decoded parameters."""
    cfg = cfg or SweepConfig()
    direction = np.array([np.cos(params["sweep_angle"]), np.sin(params["sweep_angle"])])
    start = np.asarray(garment_xy, dtype=float)[:2] + np.array(
        [params["dx_start"], params["dy_start"]]
    )
    distance = float(np.clip(params["sweep_distance"], 0.0, cfg.max_distance))
    speed = float(np.clip(params["sweep_speed"], cfg.min_speed, cfg.max_speed))
    end = start + direction * distance
    sweep_duration = max(distance / max(speed, 1e-6), 0.15)
    return SweepPlan(
        start_xy=start,
        end_xy=end,
        direction=direction,
        angle=float(params["sweep_angle"]),
        distance=distance,
        speed=speed,
        contact_height=cfg.contact_height,
        hand_height=cfg.hand_height,
        approach_duration=cfg.approach_duration,
        sweep_duration=sweep_duration,
        retract_duration=cfg.retract_duration,
    )


def hand_pose_at(plan: SweepPlan, t: float) -> tuple[np.ndarray, float]:
    """World (x, y) and z of the sweeping hand at time ``t`` into the plan.

    Phases: approach (lower onto the table behind the garment), sweep
    (translate along ``direction``), retract (lift).
    """
    t0 = plan.approach_duration
    t1 = t0 + plan.sweep_duration
    t2 = t1 + plan.retract_duration
    t = float(np.clip(t, 0.0, t2))
    if t <= t0:
        alpha = t / max(t0, 1e-6)
        xy = plan.start_xy
        z = plan.hand_height + alpha * (plan.contact_height - plan.hand_height)
    elif t <= t1:
        alpha = (t - t0) / max(plan.sweep_duration, 1e-6)
        xy = plan.start_xy + alpha * (plan.end_xy - plan.start_xy)
        z = plan.contact_height
    else:
        alpha = (t - t1) / max(plan.retract_duration, 1e-6)
        xy = plan.end_xy
        z = plan.contact_height + alpha * (plan.hand_height - plan.contact_height)
    return xy, float(z)


def segment_hits_disk(
    start: np.ndarray,
    end: np.ndarray,
    center: np.ndarray,
    radius: float,
) -> bool:
    """True if the segment comes within ``radius`` of ``center``."""
    s = np.asarray(start, dtype=float)[:2]
    e = np.asarray(end, dtype=float)[:2]
    c = np.asarray(center, dtype=float)[:2]
    d = e - s
    length2 = float(d @ d)
    if length2 < 1e-12:
        return bool(np.linalg.norm(c - s) <= radius)
    alpha = float(np.clip(((c - s) @ d) / length2, 0.0, 1.0))
    closest = s + alpha * d
    return bool(np.linalg.norm(c - closest) <= radius)
