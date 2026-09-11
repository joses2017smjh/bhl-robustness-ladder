"""Table, baskets, and spawn ranges for the cloth-sort scene.

Heights come from ``bhl_robust.reach_band``: the table top sits at ``GRASP_Z``
so a standing squat can reach a garment and a collapsed robot cannot. Baskets
sit under the table's robot-facing edge. Success is a geometric AABB test,
never a visual judgement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from bhl_robust.cloth.garments import BASKET_IDS, BASKET_JACKETS, BASKET_SHIRTS, BASKET_SOCKS
from bhl_robust.reach_band import GRASP_Z, HAND_SPAN

#: Table top. Same height every redesigned payload is grasped at.
TABLE_TOP_Z = GRASP_Z
TABLE_THICKNESS = 0.04
TABLE_SIZE_XY = (0.70, 0.96)
TABLE_CENTER_XY = (0.52, 0.0)

#: Open-topped sorting boxes under the front lip. Inner clear volume.
BASKET_INNER = (0.24, 0.28, 0.14)
BASKET_WALL = 0.02
#: Basket centres in the robot-table plane. x is between the robot and the
#: table so a sweep off the front edge drops into a box rather than onto the
#: floor behind the table.
BASKET_X = 0.10
BASKET_Y = {
    BASKET_SOCKS: 0.34,
    BASKET_SHIRTS: 0.00,
    BASKET_JACKETS: -0.34,
}

#: Robot spawn, facing +x, already at the table. Walking-to-the-table is a
#: different problem and is not part of C0–C3.
ROBOT_XY = (-0.22, 0.0)
#: Measured, not assumed. The Isaac scene spawns under ``(0, 0, 1, 0)`` and a
#: heading cannot be set on that pose; 21247910 read the hands at (-0.208,
#: +0.192) from the root where MuJoCo FK facing +x puts them at (+0.209,
#: -0.177). Both axes flip, so this is a half-turn, not a mirror: the robot faces
#: **-x**, away from where this file put the table.
ROBOT_YAW = float(__import__("math").pi)

#: Hand contact just above the table so a sweep grazes rather than stubs.
CONTACT_HEIGHT = TABLE_TOP_Z + 0.015
HAND_HEIGHT = TABLE_TOP_Z + 0.06
HAND_RADIUS = 0.04

#: Fraction of a deformable's vertices that must lie in the correct basket.
DEFORMABLE_SUCCESS_FRACTION = 0.60

#: Rigid-proxy success uses the centre of mass. Configurable so a later
#: evaluation can tighten it without rewriting the predicate.
RIGID_SUCCESS_FRACTION = 1.0


@dataclass(frozen=True)
class AABB:
    """Axis-aligned box. ``low`` and ``high`` are (3,) world-frame corners."""

    low: np.ndarray
    high: np.ndarray

    def contains(self, points: np.ndarray, fraction: float = 1.0) -> bool:
        """True if at least ``fraction`` of the points lie inside.

        ``points`` is (3,) or (N, 3). A single CoM is the rigid-proxy case;
        a vertex cloud is the deformable case.
        """
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        inside = np.all((pts >= self.low) & (pts <= self.high), axis=1)
        return float(inside.mean()) >= fraction

    def center(self) -> np.ndarray:
        return 0.5 * (self.low + self.high)

    def contains_xy(self, xy: np.ndarray) -> bool:
        p = np.asarray(xy, dtype=float)
        return bool(self.low[0] <= p[0] <= self.high[0] and self.low[1] <= p[1] <= self.high[1])


def table_aabb() -> AABB:
    hx, hy = TABLE_SIZE_XY[0] / 2.0, TABLE_SIZE_XY[1] / 2.0
    z0 = TABLE_TOP_Z - TABLE_THICKNESS
    c = TABLE_CENTER_XY
    return AABB(
        low=np.array([c[0] - hx, c[1] - hy, z0]),
        high=np.array([c[0] + hx, c[1] + hy, TABLE_TOP_Z]),
    )


def table_top_rect() -> tuple[np.ndarray, np.ndarray]:
    """((xmin, xmax), (ymin, ymax)) of the table top."""
    box = table_aabb()
    return box.low[:2], box.high[:2]


def basket_aabb(basket_id: str) -> AABB:
    """Inner clear volume of one sorting basket."""
    hx, hy, hz = (s / 2.0 for s in BASKET_INNER)
    y = BASKET_Y[basket_id]
    return AABB(
        low=np.array([BASKET_X - hx, y - hy, 0.0]),
        high=np.array([BASKET_X + hx, y + hy, BASKET_INNER[2]]),
    )


def basket_center(basket_id: str) -> np.ndarray:
    return basket_aabb(basket_id).center()


def all_basket_aabbs() -> dict[str, AABB]:
    return {bid: basket_aabb(bid) for bid in BASKET_IDS}


def garment_spawn_range() -> tuple[np.ndarray, np.ndarray]:
    """Inclusive xy range garments spawn in, inset from the table rim.

    Inset keeps a reset from placing a garment already hanging off an edge,
    which would make the first sweep's success a spawn artifact.
    """
    low, high = table_top_rect()
    inset = 0.10
    return low + inset, high - inset


def default_spawn_xy(spec, sibling_index: int = 0, n_siblings: int = 1) -> tuple[float, float]:
    """Table-centre x, target-basket y-lane, with a split if two items share a basket."""
    spawn_lo, spawn_hi = garment_spawn_range()
    x = 0.5 * (float(spawn_lo[0]) + float(spawn_hi[0]))
    y = float(basket_center(spec.target_basket)[1])
    if n_siblings > 1:
        y += (sibling_index - 0.5 * (n_siblings - 1)) * 0.12
    y = float(np.clip(y, float(spawn_lo[1]), float(spawn_hi[1])))
    return x, y


def assert_layout() -> None:
    """Sanity the geometry cannot silently violate the morphology.

    Called by tests and by env construction. A basket wider than the hand
    span is fine (the robot sweeps, it does not pinch the basket); a garment
    spawn outside the table is not.
    """
    if HAND_SPAN <= 0:
        raise AssertionError("hand span must be positive")
    top = table_aabb()
    if top.high[2] != TABLE_TOP_Z:
        raise AssertionError("table top is not GRASP_Z")
    spawn_lo, spawn_hi = garment_spawn_range()
    if np.any(spawn_lo[:2] <= top.low[:2]) or np.any(spawn_hi[:2] >= top.high[:2]):
        raise AssertionError("spawn range is not inset from the table")
    for bid in BASKET_IDS:
        b = basket_aabb(bid)
        if b.high[2] >= TABLE_TOP_Z:
            raise AssertionError(f"{bid} rim is not below the table top")
        # Baskets must sit on the robot-facing side, not behind the table.
        if b.center()[0] > top.low[0]:
            raise AssertionError(f"{bid} is not under the front edge")
