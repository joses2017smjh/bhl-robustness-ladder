"""Numpy forward kinematics of the right arm, and where its hand hull is.

The contact table gives joint angles at 2 cm cells; a schedule interpolates
between them in joint space. Whether the *hand* stays on the planned line in
between, stays off the garment on its way in, and stays out of the table is a
property of the interpolated joints, not of the cells -- and 21300604 showed a
schedule that passed every cell check swing the hand 7 cm into the garment.
This answers those questions without MuJoCo or Isaac, so the schedule builder
can refuse such plans and the login-node tests can pin them down.

Robot frame throughout: root at ``(0, 0, root_z)``, +x forward, +y left, legs
in the pinch squat (the arm hangs off the base, so the legs do not enter).
``scripts/cloth/export_arm_chain.py`` writes the chain and checks this FK
against MuJoCo to 1e-6 m.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

_CHAIN_DEFAULT = Path(__file__).resolve().parents[3] / "assets" / "cloth" / "right_arm_chain.npz"


@dataclass(frozen=True)
class ArmChain:
    joints: tuple[str, ...]
    body_pos: np.ndarray   # (6, 3) parent-relative: five joint bodies, then the hand link
    body_rot: np.ndarray   # (6, 3, 3)
    hull: np.ndarray       # (P, 3) hand-link-frame convex hull of the hand mesh
    root_z: float
    lower: np.ndarray      # (5,)
    upper: np.ndarray
    #: Inertials of the six bodies, for ``inverse_dynamics``: mass, centre of
    #: mass in the body frame, and inertia about the centre of mass in the body
    #: frame. None in a chain exported before they were.
    mass: np.ndarray | None = None      # (6,)
    com: np.ndarray | None = None       # (6, 3)
    inertia: np.ndarray | None = None   # (6, 3, 3)


@lru_cache(maxsize=2)
def load_chain(path: str | None = None) -> ArmChain:
    from bhl_robust.cloth.layout import STANDING_ROOT_Z

    f = np.load(Path(path) if path else _CHAIN_DEFAULT, allow_pickle=False)
    has = "mass" in f.files
    # The chain is exported root-relative; its stored root_z is the squat it was
    # checked in. The root stands where the layout's stance settles it.
    return ArmChain(
        joints=tuple(str(j) for j in f["joints"]), body_pos=f["body_pos"], body_rot=f["body_rot"],
        hull=f["hull"], root_z=STANDING_ROOT_Z, lower=f["lower"], upper=f["upper"],
        mass=f["mass"] if has else None, com=f["com"] if has else None,
        inertia=f["inertia"] if has else None,
    )


def _rz(q: np.ndarray) -> np.ndarray:
    c, s = np.cos(q), np.sin(q)
    out = np.zeros(q.shape + (3, 3))
    out[..., 0, 0], out[..., 0, 1] = c, -s
    out[..., 1, 0], out[..., 1, 1] = s, c
    out[..., 2, 2] = 1.0
    return out


def fk(chain: ArmChain, q) -> tuple[np.ndarray, np.ndarray]:
    """Hand-link position ``(N, 3)`` and rotation ``(N, 3, 3)`` for arm angles ``(N, 5)``."""
    q = np.atleast_2d(np.asarray(q, dtype=float))
    n = q.shape[0]
    p = np.zeros((n, 3))
    p[:, 2] = chain.root_z
    R = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    for k in range(5):
        p = p + R @ chain.body_pos[k]
        R = R @ chain.body_rot[k] @ _rz(q[:, k])
    p = p + R @ chain.body_pos[5]
    R = R @ chain.body_rot[5]
    return p, R


def hull_points(chain: ArmChain, q) -> np.ndarray:
    """Hand hull vertices in the robot frame, ``(N, P, 3)``."""
    p, R = fk(chain, q)
    return p[:, None, :] + chain.hull @ R.transpose(0, 2, 1)


def hand_point(chain: ArmChain, q, p_hand) -> np.ndarray:
    """A hand-fixed point in the robot frame, ``(N, 3)``."""
    p, R = fk(chain, q)
    return p + R @ np.asarray(p_hand, dtype=float)


def inverse_dynamics(chain: ArmChain, q, qd, qdd, gravity=(0.0, 0.0, -9.81), armature=0.0) -> np.ndarray:
    """Joint torques ``(N, 5)`` that produce ``qdd`` at ``(q, qd)`` against gravity.

    Recursive Newton-Euler over the five hinges, base fixed and not moving --
    the fixed-base task exactly, the planted free base approximately. Gravity
    enters as an upward acceleration of the base; ``gravity`` is expressed in
    the robot frame, so a tilted base can pass its own. ``armature`` (rotor
    inertia per joint, scalar or ``(5,)``) is added as ``armature * qdd``, as
    PhysX adds it. Checked against MuJoCo's ``mj_rne`` to 1e-9 N m by
    ``scripts/cloth/export_arm_chain.py``.
    """
    if chain.mass is None:
        raise ValueError("this arm chain has no inertials; re-run scripts/cloth/export_arm_chain.py")
    q, qd, qdd = (np.atleast_2d(np.asarray(x, dtype=float)) for x in (q, qd, qdd))
    n = q.shape[0]
    p_prev = np.zeros((n, 3))
    p_prev[:, 2] = chain.root_z
    R_prev = np.broadcast_to(np.eye(3), (n, 3, 3)).copy()
    w = np.zeros((n, 3))
    al = np.zeros((n, 3))
    a = np.broadcast_to(-np.asarray(gravity, dtype=float), (n, 3)).copy()
    P, Z, F, M, C = [], [], [], [], []
    for k in range(6):
        p = p_prev + R_prev @ chain.body_pos[k]
        r = p - p_prev
        a = a + np.cross(al, r) + np.cross(w, np.cross(w, r))     # origin rides on the parent
        if k < 5:
            R_fix = R_prev @ chain.body_rot[k]
            z = R_fix[:, :, 2]
            R = R_fix @ _rz(q[:, k])
            spin = z * qd[:, k:k + 1]
            al = al + z * qdd[:, k:k + 1] + np.cross(w, spin)
            w = w + spin
        else:                                                      # hand link, welded to the elbow roll
            R, z = R_prev @ chain.body_rot[5], None
        c = p + R @ chain.com[k]
        rc = c - p
        ac = a + np.cross(al, rc) + np.cross(w, np.cross(w, rc))
        Iw = R @ chain.inertia[k] @ R.transpose(0, 2, 1)
        P.append(p); Z.append(z); C.append(c)
        F.append(chain.mass[k] * ac)
        M.append((Iw @ al[:, :, None])[:, :, 0] + np.cross(w, (Iw @ w[:, :, None])[:, :, 0]))
        p_prev, R_prev = p, R
    f = np.zeros((n, 3))
    mom = np.zeros((n, 3))
    tau = np.zeros((n, 5))
    for k in range(5, -1, -1):
        m_k = M[k] + np.cross(C[k] - P[k], F[k])
        if k < 5:
            m_k = m_k + mom + np.cross(P[k + 1] - P[k], f)
        f = F[k] + f
        mom = m_k
        if k < 5:
            tau[:, k] = np.einsum("ij,ij->i", mom, Z[k])
    return tau + np.asarray(armature, dtype=float) * qdd


def simulate_pd(chain: ArmChain, q0, q_cmd, qd_cmd, dt: float, kp: float, kd: float, effort: float,
                armature: float, substeps: int = 10, qd0=None, gravity=(0.0, 0.0, -9.81)) -> np.ndarray:
    """The arm under a joint PD drive, targets held for each ``dt``; returns ``q`` at each step's start.

    Forward dynamics from ``inverse_dynamics`` (mass matrix by unit
    accelerations, bias at rest acceleration), semi-implicit Euler at
    ``dt / substeps``, torque clipped at ``effort``. Base fixed, no contacts,
    no joint limits -- the free-space arm. With Isaac's arm gains it reproduces
    the joint trajectory Isaac measured in 21307211 before the hand touched
    anything (``tests/test_cloth_sort.py``), which is what makes it a fair
    login-node judge of a schedule's tracking.
    """
    q = np.array(q0, dtype=float)
    v = np.zeros(5) if qd0 is None else np.array(qd0, dtype=float)
    q_cmd, qd_cmd = np.asarray(q_cmd, dtype=float), np.asarray(qd_cmd, dtype=float)
    h = dt / substeps
    acc = np.vstack([np.zeros((2, 5)), np.eye(5)])
    out = np.zeros((len(q_cmd), 5))
    for k in range(len(q_cmd)):
        out[k] = q
        for _ in range(substeps):
            tau = np.clip(kp * (q_cmd[k] - q) + kd * (qd_cmd[k] - v), -effort, effort)
            vel = np.vstack([v[None], np.zeros((6, 5))])
            r = inverse_dynamics(chain, np.repeat(q[None], 7, axis=0), vel, acc, gravity=gravity, armature=armature)
            M = (r[2:] - r[1]).T
            v = v + h * np.linalg.solve(M, tau - r[0])
            q = q + h * v
    return out


def reduce_hull(points, max_vertices: int = 64, keep_below: float = 1e-3) -> np.ndarray:
    """At most ``max_vertices`` hull vertices, spread by farthest-point sampling.

    PhysX cooks a convex collider with a vertex limit (64 by default) and
    enlarges or trims the hull to meet it; handing it a hull already under the
    limit keeps the Isaac collider the shape this module checks against. Every
    vertex within ``keep_below`` of the lowest one is kept, so the fingertip face
    the tip table stands on is exact.
    """
    from scipy.spatial import ConvexHull

    p = np.asarray(points, dtype=float)
    v = p[ConvexHull(p).vertices]
    if len(v) <= max_vertices:
        return v
    chosen = list(np.nonzero(v[:, 2] <= v[:, 2].min() + keep_below)[0])
    d = np.min(np.linalg.norm(v[:, None, :] - v[None, chosen, :], axis=2), axis=1)
    while len(chosen) < max_vertices:
        k = int(np.argmax(d))
        chosen.append(k)
        d = np.minimum(d, np.linalg.norm(v - v[k], axis=1))
    r = v[chosen]
    return r[ConvexHull(r).vertices]


def hull_excess(points, hull_vertices) -> float:
    """How far any of ``points`` lies outside the convex hull of ``hull_vertices`` (m)."""
    from scipy.spatial import ConvexHull

    eq = ConvexHull(np.asarray(hull_vertices, dtype=float)).equations
    p = np.asarray(points, dtype=float)
    return float(max(0.0, (p @ eq[:, :3].T + eq[:, 3]).max()))
