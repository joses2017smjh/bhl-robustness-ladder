"""Where the sweeping hand can actually go, and joint angles that put it there.

Until this module existed the kinematic ladder had **no reach model**. A sweep
was two planar points and the hand was assumed to travel between them, so
kinematic C0 scoring 1.00 meant "if a hand could go anywhere, this plan sorts
the garment" and nothing more. Measured by forward kinematics, the right hand's
fingertips can touch a 0.30 m table top only on the robot's right and never more
than 0.28 m forward of the root (``load_contact``); the hand-link origin itself
bottoms out at 0.339 m. The original layout put the table edge 0.39 m away and
the garment 0.74 m. Nothing in it was reachable.

The table in ``assets/cloth/right_arm_ik_pinch.npz`` is built from MuJoCo FK
and refined with damped least squares (``scripts/cloth/build_ik_table.py``):
2 cm voxels in the robot frame (root at the origin, +x forward, legs in the
pinch squat the controller holds, left arm at its pinch pose). A voxel is
reachable when the right hand-link origin was driven to its centre within
5 mm, inside the joint limits. Each reachable voxel stores the five right-arm
joint angles that got it there, so the same artefact is the reach model *and*
the inverse kinematics -- the kinematic ladder and the Isaac action term cannot
disagree about where the hand is.

Heights here are hand-link-origin heights above the floor, which is also the
world frame of a planted robot, so a layout constant compares directly.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_DEFAULT = Path(__file__).resolve().parents[3] / "assets" / "cloth" / "right_arm_ik_pinch.npz"


@dataclass(frozen=True)
class ReachTable:
    x: np.ndarray
    y: np.ndarray
    z: np.ndarray
    mask: np.ndarray          # (nx, ny, nz) bool
    q: np.ndarray             # (nx, ny, nz, 5) right-arm joint angles
    joints: tuple[str, ...]
    q_pinch: np.ndarray
    planted_root_z: float
    step: float

    def index(self, p_robot: np.ndarray) -> tuple[int, int, int] | None:
        """Nearest voxel to a robot-frame point, or None outside the grid."""
        p = np.asarray(p_robot, dtype=float).reshape(3)
        idx = []
        for axis, v in zip((self.x, self.y, self.z), p):
            i = int(np.rint((v - axis[0]) / self.step))
            if i < 0 or i >= len(axis):
                return None
            idx.append(i)
        return tuple(idx)  # type: ignore[return-value]


@lru_cache(maxsize=4)
def load_reach(path: str | None = None) -> ReachTable:
    f = np.load(Path(path) if path else _DEFAULT, allow_pickle=False)
    x = f["x"]
    return ReachTable(
        x=x, y=f["y"], z=f["z"], mask=f["mask"].astype(bool), q=f["q"],
        joints=tuple(str(j) for j in f["joints"]), q_pinch=f["q_pinch"],
        planted_root_z=float(f["planted_root_z"]),
        step=float(round(float(x[1] - x[0]), 6)),
    )


def to_robot_frame(p_world: np.ndarray, robot_xy=None, robot_yaw: float | None = None) -> np.ndarray:
    """World point -> robot frame (root at origin, +x forward). z is unchanged."""
    from bhl_robust.cloth.layout import ROBOT_XY, ROBOT_YAW

    rx, ry = robot_xy if robot_xy is not None else ROBOT_XY
    yaw = ROBOT_YAW if robot_yaw is None else robot_yaw
    p = np.asarray(p_world, dtype=float).reshape(-1)
    dx, dy = p[0] - rx, p[1] - ry
    c, s = np.cos(-yaw), np.sin(-yaw)
    z = p[2] if p.size > 2 else 0.0
    return np.array([c * dx - s * dy, s * dx + c * dy, z])


def robot_to_world(p_robot, robot_xy=None, robot_yaw: float | None = None) -> np.ndarray:
    """Inverse of ``to_robot_frame``: robot-frame point -> world. z unchanged."""
    from bhl_robust.cloth.layout import ROBOT_XY, ROBOT_YAW

    rx, ry = robot_xy if robot_xy is not None else ROBOT_XY
    yaw = ROBOT_YAW if robot_yaw is None else robot_yaw
    p = np.asarray(p_robot, dtype=float).reshape(-1)
    c, s = np.cos(yaw), np.sin(yaw)
    z = p[2] if p.size > 2 else 0.0
    return np.array([rx + c * p[0] - s * p[1], ry + s * p[0] + c * p[1], z])


def is_reachable(p_world: np.ndarray, table: ReachTable | None = None) -> bool:
    """True if the right hand-link origin can be put at this world point."""
    t = table or load_reach()
    i = t.index(to_robot_frame(p_world))
    return bool(i is not None and t.mask[i])


def ik(p_world: np.ndarray, table: ReachTable | None = None) -> dict[str, float] | None:
    """Right-arm joint angles for a world hand position, or None if unreachable."""
    t = table or load_reach()
    i = t.index(to_robot_frame(p_world))
    if i is None or not t.mask[i]:
        return None
    return {j: float(v) for j, v in zip(t.joints, t.q[i])}


def reachable_xy(z: float, table: ReachTable | None = None) -> np.ndarray:
    """(N, 2) robot-frame voxel centres reachable at hand-origin height ``z``."""
    t = table or load_reach()
    iz = int(np.rint((z - t.z[0]) / t.step))
    if iz < 0 or iz >= len(t.z):
        return np.zeros((0, 2))
    ix, iy = np.nonzero(t.mask[:, :, iz])
    return np.stack([t.x[ix], t.y[iy]], axis=1)


#: Furthest the hand mesh hangs below the hand-link origin, over every sweep IK
#: config measured (0.134-0.135 m, ``scripts/cloth/build_ik_table.py``). A
#: waypoint at table height is only touchable if the link origin can sit
#: somewhere within this far above it.
HAND_DROP_MAX = 0.14


def can_reach_xy(xy_world, z_lo: float, z_hi: float, table: ReachTable | None = None) -> bool:
    """Can the right hand-link origin get over this (x, y) anywhere in [z_lo, z_hi]?

    Strict in the plane, permissive in height: it answers "can the hand get
    over this point at all", which is the question the original layout fails.
    A contact-point model that also pins where the hand's underside is comes
    with the layout redesign.
    """
    t = table or load_reach()
    p = to_robot_frame((float(xy_world[0]), float(xy_world[1]), 0.0))
    ix = int(np.rint((p[0] - t.x[0]) / t.step))
    iy = int(np.rint((p[1] - t.y[0]) / t.step))
    if not (0 <= ix < len(t.x) and 0 <= iy < len(t.y)):
        return False
    lo = max(0, int(np.ceil((z_lo - t.z[0]) / t.step - 1e-9)))
    hi = min(len(t.z) - 1, int(np.floor((z_hi - t.z[0]) / t.step + 1e-9)))
    if hi < lo:
        return False
    return bool(t.mask[ix, iy, lo:hi + 1].any())


# ------------------------------------------------------------------ contact

_CONTACT_DEFAULT = Path(__file__).resolve().parents[3] / "assets" / "cloth" / "right_hand_tip_table.npz"
#: The first contact table, kept for the record: cells solved independently, so
#: neighbours -- and hover vs contact at one cell -- could sit on different IK
#: branches, and a joint-space move between them left the planned line by up
#: to 7 cm (21300604). Nothing loads it by default.
CONTACT_V1 = Path(__file__).resolve().parents[3] / "assets" / "cloth" / "right_hand_contact_pinch.npz"


@dataclass(frozen=True)
class ContactTable:
    """Joint angles that put the hand's fingertip over a cell at a table-relative height.

    Built by ``scripts/cloth/build_tip_table.py``. A 2 cm robot-frame grid in
    (x, y); per cell, one right-arm configuration with the fingertip (``p_hand``,
    the centre of the hand hull's bottom face) over the cell and the hand's
    lowest point at ``table_top + contact_clearance``; the same branch lifted to
    ``hover_clearance`` and ``high_clearance`` where the arm allows it. Cells
    were grown from the table centre with neighbour seeding, so adjacent cells
    are joint-space neighbours and ``q_at`` can blend them. Cells that would
    crowd the robot's own right thigh are excluded.
    """

    x: np.ndarray
    y: np.ndarray
    table_top: float
    contact_clearance: float
    hover_clearance: float
    contact_q: np.ndarray      # (nx, ny, 5)
    contact_mask: np.ndarray   # (nx, ny)
    hover_q: np.ndarray
    hover_mask: np.ndarray
    joints: tuple[str, ...]
    p_hand: np.ndarray
    step: float
    high_clearance: float | None = None
    high_q: np.ndarray | None = None
    high_mask: np.ndarray | None = None

    @property
    def contact_z(self) -> float:
        return self.table_top + self.contact_clearance

    @property
    def hover_z(self) -> float:
        return self.table_top + self.hover_clearance

    def index(self, p_robot_xy) -> tuple[int, int] | None:
        px, py = float(p_robot_xy[0]), float(p_robot_xy[1])
        i = int(np.rint((px - self.x[0]) / self.step))
        j = int(np.rint((py - self.y[0]) / self.step))
        if 0 <= i < len(self.x) and 0 <= j < len(self.y):
            return i, j
        return None

    def mask(self, phase: str) -> np.ndarray:
        if phase == "high":
            return self.high_mask if self.high_mask is not None else np.zeros_like(self.contact_mask)
        return self.contact_mask if phase == "contact" else self.hover_mask

    def q(self, phase: str) -> np.ndarray:
        if phase == "high":
            return self.high_q if self.high_q is not None else np.zeros_like(self.contact_q)
        return self.contact_q if phase == "contact" else self.hover_q

    def q_at(self, p_robot_xy, phase: str = "contact", strict: bool = False) -> np.ndarray | None:
        """Joint angles at a robot-frame (x, y), blended over the surrounding cells.

        None unless the nearest cell is valid in this phase -- the same test as
        ``sweep_cell_ok`` -- so interpolation never extends reach. Invalid corners
        drop out of the bilinear weights, which near the edge of the region can
        put the fingertip up to 1.3 cm from ``p``; ``strict`` instead returns None
        unless every corner with weight is valid (fingertip within 1.3 mm).
        A nearest-cell lookup steps the arm between 2 cm cells, which a
        10 N m/rad arm plays back as a staircase.
        """
        k = self.index(p_robot_xy)
        m = self.mask(phase)
        if k is None or not m[k]:
            return None
        Q = self.q(phase)
        fx = (float(p_robot_xy[0]) - float(self.x[0])) / self.step
        fy = (float(p_robot_xy[1]) - float(self.y[0])) / self.step
        i0, j0 = int(np.floor(fx)), int(np.floor(fy))
        ax, ay = fx - i0, fy - j0
        acc, wsum = np.zeros(Q.shape[-1]), 0.0
        for di, wx in ((0, 1.0 - ax), (1, ax)):
            for dj, wy in ((0, 1.0 - ay), (1, ay)):
                i, j = i0 + di, j0 + dj
                w = wx * wy
                if w <= 1e-9:
                    continue
                if 0 <= i < len(self.x) and 0 <= j < len(self.y) and m[i, j]:
                    acc += w * Q[i, j]
                    wsum += w
                elif strict:
                    return None
        return acc / wsum if wsum > 0 else np.asarray(Q[k], dtype=float)


@lru_cache(maxsize=4)
def load_contact(path: str | None = None) -> ContactTable:
    f = np.load(Path(path) if path else _CONTACT_DEFAULT, allow_pickle=False)
    x = f["x"]
    high = "high_q" in f.files
    return ContactTable(
        x=x, y=f["y"], table_top=float(f["table_top"]),
        contact_clearance=float(f["contact_clearance"]), hover_clearance=float(f["hover_clearance"]),
        contact_q=f["contact_q"].astype(float), contact_mask=f["contact_mask"].astype(bool),
        hover_q=f["hover_q"].astype(float), hover_mask=f["hover_mask"].astype(bool),
        joints=tuple(str(j) for j in f["joints"]), p_hand=f["p_hand"],
        step=float(round(float(x[1] - x[0]), 6)),
        high_clearance=float(f["high_clearance"]) if high else None,
        high_q=f["high_q"].astype(float) if high else None,
        high_mask=f["high_mask"].astype(bool) if high else None,
    )


def sweep_cell_ok(xy_world, phase: str = "contact", table: ContactTable | None = None) -> bool:
    """Can the hand's underside be put over this world (x, y) in this phase?"""
    t = table or load_contact()
    i = t.index(to_robot_frame((float(xy_world[0]), float(xy_world[1]), 0.0)))
    return bool(i is not None and t.mask(phase)[i])


def sweep_joints(xy_world, phase: str = "contact", table: ContactTable | None = None) -> dict[str, float] | None:
    """Right-arm joint angles for this phase over this world (x, y), or None."""
    t = table or load_contact()
    q = t.q_at(to_robot_frame((float(xy_world[0]), float(xy_world[1]), 0.0)), phase)
    if q is None:
        return None
    return {j: float(v) for j, v in zip(t.joints, q)}


def segment_on_contact(a, b, table: ContactTable | None = None, step: float = 0.01) -> bool:
    """True if every point from ``a`` to ``b`` (world xy) is a contact cell."""
    t = table or load_contact()
    a, b = np.asarray(a, dtype=float)[:2], np.asarray(b, dtype=float)[:2]
    n = max(2, int(np.ceil(np.linalg.norm(b - a) / step)) + 1)
    return all(sweep_cell_ok(a + k * (b - a), "contact", t) for k in np.linspace(0.0, 1.0, n))
