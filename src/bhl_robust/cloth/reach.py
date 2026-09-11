"""Where the sweeping hand can actually go, and joint angles that put it there.

Until this module existed the kinematic ladder had **no reach model**. A sweep
was two planar points and the hand was assumed to travel between them, so
kinematic C0 scoring 1.00 meant "if a hand could go anywhere, this plan sorts
the garment" and nothing more. Measured by forward kinematics, the right hand
of the pinch squat cannot get below z = 0.339 m -- above a 0.30 m table top --
and never reaches more than 0.29 m forward of the root. The original layout put
the garment 0.74 m away. Nothing in it was reachable.

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
