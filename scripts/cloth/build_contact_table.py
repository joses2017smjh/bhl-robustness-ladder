#!/usr/bin/env python3
"""Where can the *underside* of the right hand be placed against a table top?

``right_arm_ik_pinch.npz`` places the hand-link origin. A sweep does not touch
anything with that origin: the hand mesh hangs 2-13 cm below it depending on
the pose. This builds the table a sweep controller actually needs.

1. Pick a contact point fixed on the hand: the hand-frame location of the mesh's
   lowest vertex, taken as the median over the reachable sweep poses.
2. For candidate table-top heights, solve damped-least-squares IK that puts that
   point at (x, y, top + CONTACT_CLEARANCE) on a 2 cm robot-frame grid. Keep a
   cell only if the solve lands within 5 mm, no part of the hand mesh dips below
   the table top, and the cell clears the robot's own right thigh (y <= -0.18).
3. Choose the height with the most usable cells, then solve a hover table at
   top + HOVER_CLEARANCE for the approach and retract phases.

Robot frame, root at the origin, +x forward, -y the robot's right; legs in the
pinch squat the controller holds. MuJoCo, same crew builder as the reach table.
Writes ``assets/cloth/right_hand_contact_pinch.npz``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bhl_robust.eval.coop_replay import JOINTS, PINCH_POSE, build_crew  # noqa: E402

CONTACT_CLEARANCE = 0.012   # contact point just above the top: grazes a 2 cm garment's side
HOVER_CLEARANCE = 0.070     # clear of the tallest garment on approach and retract
BODY_CLEAR_Y = -0.18        # the right thigh reaches y = -0.165 in the pinch squat
HEIGHTS = np.round(np.arange(0.28, 0.4201, 0.02), 3)
STEP = 0.02
XS = np.round(np.arange(-0.16, 0.3001, STEP), 3)
YS = np.round(np.arange(-0.46, -0.1799, STEP), 3)

UP = REPO / "external" / "Berkeley-Humanoid-Lite"
CD = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/mjcf_cache")
m, slots, _ = build_crew(UP, CD, 2, ego_camera=False, payload="cube")
d = mujoco.MjData(m)
s = slots[0]
jid = lambda j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s.prefix + j)
reach = np.load(REPO / "assets" / "cloth" / "right_arm_ik_pinch.npz")
ARM = [str(j) for j in reach["joints"]]
adr = {j: int(m.jnt_qposadr[jid(j)]) for j in JOINTS}
dof = [int(m.jnt_dofadr[jid(j)]) for j in ARM]
lo = np.array([m.jnt_range[jid(j)][0] for j in ARM])
hi = np.array([m.jnt_range[jid(j)][1] for j in ARM])
RH = int(s.hands[1])
rz = float(reach["planted_root_z"])
hand_geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == RH
              and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)]
assert hand_geoms, "no hand mesh"
G = hand_geoms[0]
mi = int(m.geom_dataid[G])
VERTS = m.mesh_vert[int(m.mesh_vertadr[mi]):int(m.mesh_vertadr[mi]) + int(m.mesh_vertnum[mi])].copy()


def setq(q):
    d.qpos[:] = 0.0
    d.qpos[s.qpos_adr + 2] = rz
    d.qpos[s.qpos_adr + 3] = 1.0
    for j in JOINTS:
        d.qpos[adr[j]] = PINCH_POSE.get(j, 0.0)
    for j, v in zip(ARM, q):
        d.qpos[adr[j]] = v
    mujoco.mj_forward(m, d)


def hand_mesh_world():
    return d.geom_xpos[G] + VERTS @ d.geom_xmat[G].reshape(3, 3).T


def point_world(p_hand):
    return d.xpos[RH] + d.xmat[RH].reshape(3, 3) @ p_hand


# ---- 1. the contact point, fixed on the hand ---------------------------------
low_pts = []
ix, iy, iz = np.nonzero(reach["mask"])
for a, b, c in zip(ix, iy, iz):
    if not (0.38 <= reach["z"][c] <= 0.48):
        continue
    setq(reach["q"][a, b, c])
    w = hand_mesh_world()
    k = int(np.argmin(w[:, 2]))
    R = d.xmat[RH].reshape(3, 3)
    low_pts.append(R.T @ (w[k] - d.xpos[RH]))
P_HAND = np.median(np.array(low_pts), axis=0)
print(f"contact point in the hand frame (median lowest vertex over {len(low_pts)} poses): {np.round(P_HAND, 4)}")

# Seed cloud: where that point sits for every reachable reach-table config.
seed_q, seed_p = [], []
for a, b, c in zip(ix, iy, iz):
    q = reach["q"][a, b, c]
    setq(q)
    seed_q.append(q)
    seed_p.append(point_world(P_HAND).copy())
seed_q, seed_p = np.array(seed_q), np.array(seed_p)
jacp = np.zeros((3, m.nv))


def solve(target):
    near = np.argsort(np.linalg.norm(seed_p - target, axis=1))[:3]
    best = None
    for k in near:
        q = seed_q[k].copy()
        for _ in range(60):
            setq(q)
            pw = point_world(P_HAND)
            err = target - pw
            if np.linalg.norm(err) < 1e-3:
                break
            mujoco.mj_jac(m, d, jacp, None, pw, RH)
            J = jacp[:, dof]
            q = np.clip(q + J.T @ np.linalg.solve(J @ J.T + 0.05 ** 2 * np.eye(3), err), lo, hi)
        setq(q)
        e = float(np.linalg.norm(target - point_world(P_HAND)))
        if best is None or e < best[1]:
            best = (q.copy(), e, float(hand_mesh_world()[:, 2].min()))
    return best


def table_for(z_point, top, check_floor):
    Q = np.zeros((len(XS), len(YS), 5), np.float32)
    M = np.zeros((len(XS), len(YS)), bool)
    for i, x in enumerate(XS):
        for j, y in enumerate(YS):
            if y > BODY_CLEAR_Y:
                continue
            t = np.array([x, y, z_point])
            if np.min(np.linalg.norm(seed_p - t, axis=1)) > 0.10:
                continue
            q, e, zmin = solve(t)
            if e <= 0.005 and (not check_floor or zmin >= top - 0.004):
                M[i, j], Q[i, j] = True, q
    return Q, M


# ---- 2. contact tables per candidate height ----------------------------------
results = {}
for top in HEIGHTS:
    Q, M = table_for(top + CONTACT_CLEARANCE, top, True)
    results[float(top)] = (Q, M)
    xr = XS[M.any(axis=1)] if M.any() else []
    yr = YS[M.any(axis=0)] if M.any() else []
    span = f"x {min(xr):+.2f}..{max(xr):+.2f}  y {min(yr):+.2f}..{max(yr):+.2f}" if M.any() else "none"
    print(f"table top {top:.2f}: {int(M.sum()):3d} usable contact cells   {span}", flush=True)

best_top = max(results, key=lambda h: int(results[h][1].sum()))
Qc, Mc = results[best_top]
print(f"\nbest table top: {best_top:.2f} m ({int(Mc.sum())} cells)")

# ---- 3. hover table at the chosen height -------------------------------------
Qh, Mh = table_for(best_top + HOVER_CLEARANCE, best_top, False)
both = Mc & Mh
print(f"hover cells {int(Mh.sum())}; cells with both contact and hover: {int(both.sum())}")
print("\nusable sweep cells at the chosen height ('#' contact+hover, '+' contact only); x up = forward")
print("        y: " + " ".join(f"{y:+.2f}"[1:4] if k % 3 == 0 else "   " for k, y in enumerate(YS)))
for i in range(len(XS) - 1, -1, -1):
    row = "".join("  #" if both[i, j] else ("  +" if Mc[i, j] else "  .") for j in range(len(YS)))
    print(f"  x={XS[i]:+.2f} {row}")

out = REPO / "assets" / "cloth" / "right_hand_contact_pinch.npz"
np.savez_compressed(
    out, x=XS, y=YS, table_top=best_top, contact_clearance=CONTACT_CLEARANCE,
    hover_clearance=HOVER_CLEARANCE, body_clear_y=BODY_CLEAR_Y,
    contact_q=Qc, contact_mask=Mc, hover_q=Qh, hover_mask=Mh, joints=np.array(ARM),
    p_hand=P_HAND, planted_root_z=rz, heights_tried=HEIGHTS,
    cells_per_height=np.array([int(results[float(h)][1].sum()) for h in HEIGHTS]),
    source="MuJoCo FK + DLS on a hand-fixed contact point; bhl_robust.eval.coop_replay crew, pinch squat legs",
)
print(f"\nwrote {out} ({out.stat().st_size / 1e3:.0f} kB)")
