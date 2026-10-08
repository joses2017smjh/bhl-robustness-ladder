"""Table, baskets, and spawn ranges for the cloth-sort scene.

**Redesigned 2026-09-13, inside the measured reach.** The first layout put the
table in front of the robot with its near edge 0.39 m away and the garment at
0.74 m. The fingertips reach 0.28 m forward, and in Isaac the robot faces the
other way, so nothing in it could be touched (``docs/CLOTH_SORT.md``).

This one is placed in the **robot frame**, inside the fingertip contact region
of ``assets/cloth/right_hand_contact_pinch.npz`` -- a hand-fixed contact point
IK'd onto a 2 cm grid against a 0.30 m table top, the height with the most
usable cells -- and converted to world coordinates through the measured facing.
``assert_layout`` re-checks every table cell against that table, so an edit that
walks the table out of reach fails loudly instead of building another scene
nothing can touch.

Robot frame: root at the origin, +x forward, -y the robot's right.
World: ``ROBOT_XY + R(ROBOT_YAW) @ robot``.

The right hand sweeps. The table sits on the robot's right; the baskets sit off
its front, outer and back edges, clear of the robot's own right leg. Success is
a geometric AABB test, never a visual judgement.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from bhl_robust.cloth import balance as _balance
from bhl_robust.cloth.garments import BASKET_IDS, BASKET_JACKETS, BASKET_SHIRTS, BASKET_SOCKS, GARMENTS
from bhl_robust.reach_band import HAND_SPAN

#: Robot spawn, already at the table. Walking to it is a different problem.
ROBOT_XY = (-0.22, 0.0)
#: Measured, not assumed. The Isaac scene spawns under ``(0, 0, 1, 0)`` and a
#: heading cannot be set on that pose; 21247910 read the hands at (-0.208,
#: +0.192) from the root where MuJoCo FK facing +x puts them at (+0.209,
#: -0.177). Both axes flip, so this is a half-turn, not a mirror: the robot
#: faces **-x**.
ROBOT_YAW = math.pi
#: Root height of the pinch squat the reach, contact and fingertip tables were
#: solved in (MuJoCo ``planted_root_z``). Every height those tables store is in
#: that frame; ``reach.load_contact`` and ``arm_fk.load_chain`` shift them to
#: the stance below.
SOLVE_ROOT_Z = -0.1372496766232026
#: The robot stands in ``balance.STANCE``, not the pinch squat: the squat's knees
#: saturate and it falls backward in under a second (21317172), while this
#: stance with its leg controller stands in Isaac (21329076-077). Was -0.137.
ROBOT_ROOT_Z = _balance.ROOT_Z
#: Where the root stands once settled, which is what the arm's reach hangs off.
STANDING_ROOT_Z = _balance.SETTLED_ROOT_Z
#: How far every height rises with the stance. The arm hangs off the root, so
#: raising the table by the same amount keeps each fingertip cell's arm
#: configuration exactly as solved.
STANCE_RISE = STANDING_ROOT_Z - SOLVE_ROOT_Z

#: Table top: 0.30 m above the floor in the squat frame, the contact height with
#: the most usable fingertip cells (252 of the heights 0.28-0.42 tried); risen
#: with the stance.
TABLE_TOP_Z = 0.30 + STANCE_RISE
TABLE_THICKNESS = 0.04

#: Robot-frame rectangles, ((x_min, x_max), (y_min, y_max)).
TABLE_ROBOT = ((-0.04, 0.16), (-0.38, -0.20))
#: Openings wider than every garment bound for them is across its diagonal
#: (``assert_layout``). At 12 x 12 cm the shirts basket let a pushed shirt, turned
#: 68 deg, land across both rims and rest there -- 21317170, the one failure in 8
#: -- and a 12 x 10 cm jackets opening would have caught most jackets.
BASKET_ROBOT = {
    BASKET_SHIRTS: ((0.18, 0.33), (-0.355, -0.205)),    # off the front edge, 15 x 15 cm
    BASKET_SOCKS: ((0.00, 0.14), (-0.52, -0.40)),       # off the outer edge, 14 x 12 cm
    BASKET_JACKETS: ((-0.21, -0.055), (-0.41, -0.255)),  # off the back edge, 15.5 x 15.5 cm
}
#: A rigid garment cannot bridge an opening at least this much wider than its diagonal.
BRIDGE_MARGIN = 0.01
BASKET_DEPTH = 0.14
BASKET_WALL = 0.02
#: Garments spawn in the middle of the table, inset so a reset never leaves one
#: overhanging an edge.
SPAWN_ROBOT = ((0.03, 0.09), (-0.32, -0.26))
#: The right leg in the pinch squat, robot frame (union of the hip, knee and
#: ankle body extents). Nothing on the table or floor may enter it.
RIGHT_LEG_ROBOT = ((-0.08, 0.17), (-0.18, 0.07))

#: Heights of the hand's lowest point the sweep controller commands. Must match
#: the fingertip table they were solved against (``build_tip_table.py``).
#: Contact was 12 mm, which left 3 mm of a 1.5 cm garment's side for the hand
#: to push; the Isaac clip (21300604) shows the hand riding up onto the shirt.
CONTACT_CLEARANCE = 0.003
HOVER_CLEARANCE = 0.070
CONTACT_HEIGHT = TABLE_TOP_Z + CONTACT_CLEARANCE
HAND_HEIGHT = TABLE_TOP_Z + HOVER_CLEARANCE
#: Footprint of the hand within 3 cm of the table, about the fingertip: the
#: largest horizontal reach of any hull vertex that low, over every contact cell
#: (2.96 cm median, 3.05 cm max). Was 4 cm, a guess.
HAND_RADIUS = 0.031

#: Fraction of a deformable's vertices that must lie in the correct basket.
DEFORMABLE_SUCCESS_FRACTION = 0.60
#: Rigid-proxy success uses the centre of mass.
RIGID_SUCCESS_FRACTION = 1.0


def robot_rect_to_world(rect) -> tuple[np.ndarray, np.ndarray]:
    """World (low_xy, high_xy) of a robot-frame axis-aligned rectangle.

    Exact for yaw at multiples of 90 degrees, which is all this layout uses.
    """
    (x0, x1), (y0, y1) = rect
    c, s = math.cos(ROBOT_YAW), math.sin(ROBOT_YAW)
    corners = np.array([[x, y] for x in (x0, x1) for y in (y0, y1)])
    w = np.stack([ROBOT_XY[0] + c * corners[:, 0] - s * corners[:, 1],
                  ROBOT_XY[1] + s * corners[:, 0] + c * corners[:, 1]], axis=1)
    return w.min(axis=0).round(6), w.max(axis=0).round(6)


_T_LO, _T_HI = robot_rect_to_world(TABLE_ROBOT)
TABLE_CENTER_XY = (float(0.5 * (_T_LO[0] + _T_HI[0])), float(0.5 * (_T_LO[1] + _T_HI[1])))
TABLE_SIZE_XY = (float(_T_HI[0] - _T_LO[0]), float(_T_HI[1] - _T_LO[1]))


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
    return AABB(
        low=np.array([_T_LO[0], _T_LO[1], TABLE_TOP_Z - TABLE_THICKNESS]),
        high=np.array([_T_HI[0], _T_HI[1], TABLE_TOP_Z]),
    )


def table_top_rect() -> tuple[np.ndarray, np.ndarray]:
    """((xmin, ymin), (xmax, ymax)) of the table top."""
    box = table_aabb()
    return box.low[:2], box.high[:2]


def basket_aabb(basket_id: str) -> AABB:
    """Inner clear volume of one sorting basket."""
    lo, hi = robot_rect_to_world(BASKET_ROBOT[basket_id])
    return AABB(low=np.array([lo[0], lo[1], 0.0]), high=np.array([hi[0], hi[1], BASKET_DEPTH]))


def basket_center(basket_id: str) -> np.ndarray:
    return basket_aabb(basket_id).center()


def all_basket_aabbs() -> dict[str, AABB]:
    return {bid: basket_aabb(bid) for bid in BASKET_IDS}


def garment_spawn_range() -> tuple[np.ndarray, np.ndarray]:
    """Inclusive world xy range garments spawn in."""
    return robot_rect_to_world(SPAWN_ROBOT)


def default_spawn_xy(spec=None, sibling_index: int = 0, n_siblings: int = 1) -> tuple[float, float]:
    """Centre of the spawn range.

    Every garment starts here. The table holds one garment at a time -- a
    five-garment scene presents them in turn and parks the rest (``parking_xy``)
    -- so there are no per-basket lanes to split siblings into any more. The
    arguments are kept so existing callers do not break.
    """
    lo, hi = garment_spawn_range()
    return float(0.5 * (lo[0] + hi[0])), float(0.5 * (lo[1] + hi[1]))


def parking_xy(index: int) -> tuple[float, float]:
    """Where a garment waits before its turn: well clear of table, baskets and robot."""
    return ROBOT_XY[0] + 1.20 + 0.25 * index, ROBOT_XY[1] - 1.20


def _grow(rect, d):
    (x0, x1), (y0, y1) = rect
    return (x0 - d, x1 + d), (y0 - d, y1 + d)


def _overlap(a, b) -> bool:
    """Strict overlap of two ((x0, x1), (y0, y1)) rectangles; touching is not overlap."""
    (ax0, ax1), (ay0, ay1) = a
    (bx0, bx1), (by0, by1) = b
    return ax0 < bx1 - 1e-9 and bx0 < ax1 - 1e-9 and ay0 < by1 - 1e-9 and by0 < ay1 - 1e-9


def assert_layout() -> None:
    """The geometry cannot silently violate the morphology or the measured reach."""
    if HAND_SPAN <= 0:
        raise AssertionError("hand span must be positive")
    top = table_aabb()
    if abs(top.high[2] - TABLE_TOP_Z) > 1e-9:
        raise AssertionError("table top is not TABLE_TOP_Z")
    (sx0, sx1), (sy0, sy1) = SPAWN_ROBOT
    (tx0, tx1), (ty0, ty1) = TABLE_ROBOT
    if not (tx0 < sx0 and sx1 < tx1 and ty0 < sy0 and sy1 < ty1):
        raise AssertionError("spawn range is not inset from the table")
    if _overlap(TABLE_ROBOT, RIGHT_LEG_ROBOT):
        raise AssertionError("table intrudes on the robot's right leg")
    ids = list(BASKET_IDS)
    for i, bid in enumerate(ids):
        if basket_aabb(bid).high[2] >= TABLE_TOP_Z - TABLE_THICKNESS:
            raise AssertionError(f"{bid} rim is not below the underside of the table")
        inner, walls = BASKET_ROBOT[bid], _grow(BASKET_ROBOT[bid], BASKET_WALL)
        if _overlap(inner, TABLE_ROBOT):
            raise AssertionError(f"{bid} opening sits under the table instead of off an edge")
        if _overlap(walls, RIGHT_LEG_ROBOT):
            raise AssertionError(f"{bid} walls intrude on the robot's right leg")
        for other in ids[i + 1:]:
            if _overlap(walls, _grow(BASKET_ROBOT[other], BASKET_WALL)):
                raise AssertionError(f"{bid} and {other} walls overlap")
        (bx0, bx1), (by0, by1) = inner
        for g in GARMENTS:
            if g.target_basket == bid and min(bx1 - bx0, by1 - by0) < math.hypot(*g.proxy_size[:2]) + BRIDGE_MARGIN:
                raise AssertionError(f"{g.name} can rest across the rims of the {bid} basket")
    from bhl_robust.cloth.reach import load_contact

    t = load_contact()
    if abs(t.table_top - TABLE_TOP_Z) > 1e-9 or abs(t.contact_clearance - CONTACT_CLEARANCE) > 1e-9 \
            or abs(t.hover_clearance - HOVER_CLEARANCE) > 1e-9:
        raise AssertionError("layout heights do not match the contact table they were placed against")
    for x in np.arange(tx0, tx1 + 1e-9, t.step):
        for y in np.arange(ty0, ty1 + 1e-9, t.step):
            k = t.index((x, y))
            if k is None or not t.contact_mask[k]:
                raise AssertionError(
                    f"table cell at robot ({x:+.2f}, {y:+.2f}) is outside fingertip reach")
