#!/usr/bin/env python3
"""Fingertip table: one continuous arm configuration per 2 cm cell, at three heights.

Replaces ``right_hand_contact_pinch.npz`` for the sweep. That table solved each
cell independently from the three nearest reach-table seeds, so neighbouring
cells, and the contact and hover solutions *of the same cell*, could sit on
different IK branches. Interpolating between them in joint space -- which is
what an arm does between two targets -- swung the hand far off the planned line:
a MuJoCo replay of 21300603's schedule puts the fingertip 7 cm into the garment
during a 0.2 s "vertical" descent (docs/CLOTH_SORT.md). Its contact point was also
1.2 cm above the table, which left 3 mm of a 1.5 cm garment's side to push.

This table:

1. puts the *fingertip* -- the centre of the hand hull's flat bottom face, not
   a median lowest vertex -- over the cell, with the hull's lowest point at
   ``top + CONTACT_CLEARANCE`` (3 mm). The hand is tilted, so the face's lower
   edge, not its centre, is what reaches the table;
2. grows outward cell by cell from the table centre, seeding each solve with an
   already-solved neighbour and penalising distance from it, so adjacent cells
   share a branch; a cell whose best solution still jumps is dropped;
3. prefers the least-tilted hand among solutions (the arm cannot hold the hand
   vertical at this height: shoulder-yaw and elbow-roll limits leave 15-45 deg);
4. solves hover (``top + HOVER_CLEARANCE``) and high (``top + HIGH_CLEARANCE``)
   from that cell's own contact solution, and keeps them only if the
   joint-space move between them stays within 1 cm of vertical;
5. keeps a contact cell only if no vertex of the hand hull is below the table.

Everything uses ``bhl_robust.cloth.arm_fk`` (checked against MuJoCo to 1e-6 m by
``export_arm_chain.py``), so the tests and the schedule builder check plans
against the same kinematics this table was solved with.
Writes ``assets/cloth/right_hand_tip_table.npz``.
"""

from __future__ import annotations

import sys
import warnings
from collections import deque
from pathlib import Path

import numpy as np
from scipy.optimize import minimize

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bhl_robust.cloth.arm_fk import fk, hull_points, load_chain  # noqa: E402

from bhl_robust.cloth.layout import TABLE_TOP_Z as TABLE_TOP  # noqa: E402  risen with the stance
CONTACT_CLEARANCE = 0.003
HOVER_CLEARANCE = 0.070
HIGH_CLEARANCE = 0.140
BODY_CLEAR_Y = -0.18            # the right thigh reaches y = -0.165 in the pinch squat
STEP = 0.02
XS = np.round(np.arange(-0.16, 0.3001, STEP), 3)
YS = np.round(np.arange(-0.46, -0.1799, STEP), 3)
SEED_CELL = (0.06, -0.30)       # the table centre, robot frame
MAX_JUMP = 0.30                 # rad, any joint, between 2 cm neighbours
TIP_TOL = 0.002                 # m
DESCENT_TOL = 0.010             # m of horizontal wander between heights at one cell
OUT = REPO / "assets" / "cloth" / "right_hand_tip_table.npz"

ARM = load_chain()


def tip_point() -> np.ndarray:
    """Centre of the hull's bottom face (vertices within 1 mm of its lowest z), hand frame."""
    h = ARM.hull
    bottom = h[h[:, 2] <= h[:, 2].min() + 1e-3]
    lo, hi = bottom.min(axis=0), bottom.max(axis=0)
    return np.array([0.5 * (lo[0] + hi[0]), 0.5 * (lo[1] + hi[1]), float(h[:, 2].min())])


P_TIP = tip_point()


def tip(q: np.ndarray) -> np.ndarray:
    p, R = fk(ARM, q)
    return p + R @ P_TIP


def tilt_deg(q: np.ndarray) -> np.ndarray:
    _, R = fk(ARM, q)
    return np.degrees(np.arccos(np.clip(R[:, 2, 2], -1.0, 1.0)))


def low_z(q: np.ndarray, tau: float = 5e-4) -> np.ndarray:
    """Smooth minimum of the hull's z (log-sum-exp, 0.5 mm temperature), ``(N,)``."""
    z = hull_points(ARM, q)[:, :, 2]
    m = z.min(axis=1, keepdims=True)
    return m[:, 0] - tau * np.log(np.exp(-(z - m) / tau).sum(axis=1))


def placement_error(q: np.ndarray, target: np.ndarray) -> float:
    """Fingertip xy off the cell, and the hull's lowest point off the target height."""
    t = tip(q[None])[0]
    z = float(hull_points(ARM, q[None])[0, :, 2].min())
    return float(np.hypot(np.linalg.norm(t[:2] - target[:2]), z - target[2]))


def solve(target: np.ndarray, q_ref: np.ndarray, w_ref: float) -> tuple[np.ndarray, float]:
    """Least-tilted configuration with the fingertip over ``target[:2]`` and the
    hand's lowest point at ``target[2]``, near ``q_ref``."""
    def cost(q):
        _, R = fk(ARM, q[None])
        return float(1.0 - R[0, 2, 2]) + w_ref * float(np.sum((q - q_ref) ** 2))

    def eq(q):
        t = tip(q[None])[0]
        return np.array([t[0] - target[0], t[1] - target[1], low_z(q[None])[0] - target[2]])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        res = minimize(
            cost, np.clip(q_ref, ARM.lower, ARM.upper), method="SLSQP",
            bounds=list(zip(ARM.lower, ARM.upper)),
            constraints=[{"type": "eq", "fun": eq}],
            options={"maxiter": 300, "ftol": 1e-10},
        )
    q = np.clip(res.x, ARM.lower, ARM.upper)
    return q, placement_error(q, target)


def vertical_wander(q_a: np.ndarray, q_b: np.ndarray, xy: np.ndarray) -> float:
    """Largest horizontal distance of the fingertip from ``xy`` on the joint-space move."""
    s = np.linspace(0.0, 1.0, 11)[:, None]
    path = tip(q_a[None] * (1 - s) + q_b[None] * s)
    return float(np.linalg.norm(path[:, :2] - xy, axis=1).max())


def neighbours(i: int, j: int):
    for di in (-1, 0, 1):
        for dj in (-1, 0, 1):
            if (di or dj) and 0 <= i + di < len(XS) and 0 <= j + dj < len(YS):
                yield i + di, j + dj


def main() -> None:
    nx, ny = len(XS), len(YS)
    Qc = np.zeros((nx, ny, 5)); Mc = np.zeros((nx, ny), bool)
    Qh = np.zeros((nx, ny, 5)); Mh = np.zeros((nx, ny), bool)
    Qg = np.zeros((nx, ny, 5)); Mg = np.zeros((nx, ny), bool)
    tilt = np.full((nx, ny), np.nan)
    z_contact = TABLE_TOP + CONTACT_CLEARANCE
    print(f"fingertip (hand frame): {np.round(P_TIP, 4)}; hull {len(ARM.hull)} vertices")

    # ---- seed cell: multi-start, least tilt -----------------------------------
    i0, j0 = int(np.argmin(np.abs(XS - SEED_CELL[0]))), int(np.argmin(np.abs(YS - SEED_CELL[1])))
    target0 = np.array([XS[i0], YS[j0], z_contact])
    rng = np.random.default_rng(0)
    old = np.load(REPO / "assets" / "cloth" / "right_hand_contact_pinch.npz")
    starts = [old["contact_q"][i0, j0]] + list(rng.uniform(ARM.lower, ARM.upper, size=(40, 5)))
    best = None
    for s0 in starts:
        q, e = solve(target0, s0, 0.0)
        if e <= TIP_TOL and hull_points(ARM, q[None])[0, :, 2].min() >= TABLE_TOP:
            t = float(tilt_deg(q[None])[0])
            if best is None or t < best[1]:
                best = (q, t)
    assert best is not None, "no valid solution at the seed cell"
    Qc[i0, j0], Mc[i0, j0] = best[0], True
    print(f"seed cell ({XS[i0]:+.2f}, {YS[j0]:+.2f}): tilt {best[1]:.1f} deg")

    # ---- grow outward, each cell seeded by solved neighbours ---------------------
    queue = deque([(i0, j0)])
    tried = {(i0, j0)}
    dropped_jump = dropped_other = 0
    while queue:
        i, j = queue.popleft()
        for a, b in neighbours(i, j):
            if (a, b) in tried or YS[b] > BODY_CLEAR_Y:
                continue
            tried.add((a, b))
            refs = [Qc[c] for c in neighbours(a, b) if Mc[c]]
            target = np.array([XS[a], YS[b], z_contact])
            cand = []
            for ref in refs:
                q, e = solve(target, ref, 0.05)
                jump = float(np.abs(q - ref).max())
                if e > TIP_TOL or hull_points(ARM, q[None])[0, :, 2].min() < TABLE_TOP:
                    continue
                cand.append((float(tilt_deg(q[None])[0]), jump, q))
            ok = [c for c in cand if all(np.abs(c[2] - r).max() <= MAX_JUMP for r in refs)]
            if not ok:
                dropped_jump += bool(cand)
                dropped_other += not cand
                continue
            t, _, q = min(ok, key=lambda c: c[0])
            Qc[a, b], Mc[a, b], tilt[a, b] = q, True, t
            queue.append((a, b))
    tilt[i0, j0] = best[1]
    print(f"contact cells: {int(Mc.sum())} (dropped: {dropped_jump} would jump branch, "
          f"{dropped_other} unreachable or into the table)")

    # ---- hover and high at each contact cell, from its own solution ----------------
    for i, j in zip(*np.nonzero(Mc)):
        xy = np.array([XS[i], YS[j]])
        q, e = solve(np.array([XS[i], YS[j], TABLE_TOP + HOVER_CLEARANCE]), Qc[i, j], 0.5)
        if e <= TIP_TOL and vertical_wander(Qc[i, j], q, xy) <= DESCENT_TOL:
            Qh[i, j], Mh[i, j] = q, True
            q2, e2 = solve(np.array([XS[i], YS[j], TABLE_TOP + HIGH_CLEARANCE]), q, 0.5)
            if e2 <= TIP_TOL and vertical_wander(q, q2, xy) <= DESCENT_TOL:
                Qg[i, j], Mg[i, j] = q2, True
    print(f"hover cells: {int(Mh.sum())}; high cells: {int(Mg.sum())}")

    # ---- continuity report over contact edges -------------------------------------
    worst_jump, worst_dev, low_band = 0.0, 0.0, [np.inf, -np.inf]
    for i, j in zip(*np.nonzero(Mc)):
        for a, b in ((i + 1, j), (i, j + 1)):
            if a < nx and b < ny and Mc[a, b]:
                worst_jump = max(worst_jump, float(np.abs(Qc[i, j] - Qc[a, b]).max()))
                s = np.linspace(0, 1, 9)[:, None]
                qs = Qc[i, j][None] * (1 - s) + Qc[a, b][None] * s
                path = tip(qs)[:, :2]
                line = np.array([XS[i], YS[j]])[None] * (1 - s) + np.array([XS[a], YS[b]])[None] * s
                worst_dev = max(worst_dev, float(np.linalg.norm(path - line, axis=1).max()))
                zl = hull_points(ARM, qs)[:, :, 2].min(axis=1)
                low_band = [min(low_band[0], float(zl.min())), max(low_band[1], float(zl.max()))]
    valid_tilt = tilt[Mc]
    print(f"neighbour edges: max joint jump {worst_jump:.3f} rad, max fingertip deviation "
          f"{worst_dev * 1000:.1f} mm from the straight line, hand's lowest point "
          f"{low_band[0]:.4f}..{low_band[1]:.4f} m (target {z_contact:.3f})")
    print(f"tilt over contact cells: min {valid_tilt.min():.1f}, median {np.median(valid_tilt):.1f}, "
          f"max {valid_tilt.max():.1f} deg")
    print("\ncells ('#' contact+hover+high, 'h' contact+hover, '+' contact only); x up = forward")
    print("        y: " + " ".join(f"{y:+.2f}"[1:4] if k % 3 == 0 else "   " for k, y in enumerate(YS)))
    for i in range(nx - 1, -1, -1):
        row = "".join("  #" if Mg[i, j] else ("  h" if Mh[i, j] else ("  +" if Mc[i, j] else "  ."))
                      for j in range(ny))
        print(f"  x={XS[i]:+.2f} {row}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, x=XS, y=YS, table_top=TABLE_TOP, contact_clearance=CONTACT_CLEARANCE,
        hover_clearance=HOVER_CLEARANCE, high_clearance=HIGH_CLEARANCE, body_clear_y=BODY_CLEAR_Y,
        contact_q=Qc.astype(np.float32), contact_mask=Mc, hover_q=Qh.astype(np.float32), hover_mask=Mh,
        high_q=Qg.astype(np.float32), high_mask=Mg, tilt_deg=tilt.astype(np.float32),
        joints=np.array(ARM.joints), p_hand=P_TIP, planted_root_z=ARM.root_z,
        max_neighbour_jump=worst_jump, max_edge_deviation=worst_dev,
        source="SLSQP on numpy FK (bhl_robust.cloth.arm_fk), grown from the table centre with "
               "neighbour seeding; fingertip = hand-hull bottom-face centre",
    )
    print(f"\nwrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
