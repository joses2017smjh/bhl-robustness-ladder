"""Deterministic geometric sweep. The C0 / C2 baseline.

Establishes whether the task is physically (or kinematically) solvable before
RL is blamed for failures. One sweep:

1. garment centre
2. target basket centre
3. planar direction garment → basket
4. hand placed behind the garment relative to that direction
5. sweep toward the basket
6. retract

Repeating is the caller's job: the env asks for an action every macro-step.
"""

from __future__ import annotations

import numpy as np

from bhl_robust.cloth.garments import GarmentSpec
from bhl_robust.cloth.layout import basket_center
from bhl_robust.cloth.sweep import SweepConfig, encode_physical


def scripted_physical(
    garment_xy: np.ndarray,
    spec: GarmentSpec,
    cfg: SweepConfig | None = None,
) -> dict[str, float]:
    """Physical sweep parameters that push ``spec`` toward its basket."""
    cfg = cfg or SweepConfig()
    g = np.asarray(garment_xy, dtype=float)[:2]
    target = basket_center(spec.target_basket)[:2]
    delta = target - g
    dist = float(np.linalg.norm(delta))
    if dist < 1e-6:
        direction = np.array([-1.0, 0.0])
        dist = 0.15
    else:
        direction = delta / dist
    angle = float(np.arctan2(direction[1], direction[0]))
    # Start behind the garment so contact happens on the far side of the push.
    start_offset = -direction * cfg.start_offset
    # Overshoot a little so the garment is carried through the basket mouth
    # rather than stopping on the rim.
    # Aim at the basket centre. Adding a large overshoot is how garments
    # skipped the box and landed on the floor in the first C5 run.
    distance = float(np.clip(dist, 0.15, cfg.max_distance))
    return {
        "dx_start": float(start_offset[0]),
        "dy_start": float(start_offset[1]),
        "sweep_angle": angle,
        "sweep_distance": distance,
        "sweep_speed": 0.45,
    }


def scripted_action(
    garment_xy: np.ndarray,
    spec: GarmentSpec,
    cfg: SweepConfig | None = None,
) -> np.ndarray:
    """Normalised ``[-1, 1]^5`` action for the scripted policy."""
    p = scripted_physical(garment_xy, spec, cfg)
    return encode_physical(
        p["dx_start"], p["dy_start"], p["sweep_angle"],
        p["sweep_distance"], p["sweep_speed"],
    )


def pick_unsorted(sorted_mask: np.ndarray, positions: np.ndarray) -> int:
    """Closest unsorted garment. Ties break by index, so the order is stable."""
    mask = np.asarray(sorted_mask, dtype=bool)
    if mask.all():
        return 0
    pos = np.asarray(positions, dtype=float).reshape(-1, 2)
    robot = np.array([-0.22, 0.0])
    d = np.linalg.norm(pos - robot, axis=1)
    d[mask] = np.inf
    return int(np.argmin(d))
