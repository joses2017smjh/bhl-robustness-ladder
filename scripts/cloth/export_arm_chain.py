#!/usr/bin/env python3
"""Export the right arm's kinematic chain and hand hull for numpy forward kinematics.

The contact table stores joint angles per cell, and the schedule interpolates
between them. Nothing on the Isaac side could say where the *hand* actually is
between two cells, so a schedule that looked valid cell by cell swung the hand
through the garment on its way in (21300604; see docs/CLOTH_SORT.md). This
writes what ``bhl_robust.cloth.arm_fk`` needs to answer that on a login node:

* the five right-arm joint bodies and the hand link, each as a parent-relative
  position and rotation (all joints are hinges about their local z);
* the hand's visual mesh as a convex hull in the hand-link frame -- the shape a
  sweep pushes with (``assets/cloth/berkeley_humanoid_lite_hand_colliders.usda``
  gives the Isaac hand a 64-vertex hull of the same mesh);
* the right arm's joint limits;
* each body's mass, centre of mass and inertia, for ``arm_fk.inverse_dynamics``
  -- the feedforward that lets a 10 N m/rad, 4 N m arm follow a schedule.

It then checks the numpy FK against MuJoCo on random configurations and refuses
to write if they disagree by more than 1e-6 m. Robot frame: root at
(0, 0, planted_root_z), +x forward, +y left. MuJoCo, same crew builder as the
reach and contact tables. Writes ``assets/cloth/right_arm_chain.npz``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bhl_robust.cloth.arm_fk import ArmChain, fk, hull_points, inverse_dynamics  # noqa: E402
from bhl_robust.eval.coop_replay import JOINTS, PINCH_POSE, build_crew  # noqa: E402

UP = REPO / "external" / "Berkeley-Humanoid-Lite"
CD = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/mjcf_cache")
ARM = (
    "arm_right_shoulder_pitch_joint",
    "arm_right_shoulder_roll_joint",
    "arm_right_shoulder_yaw_joint",
    "arm_right_elbow_pitch_joint",
    "arm_right_elbow_roll_joint",
)
OUT = REPO / "assets" / "cloth" / "right_arm_chain.npz"


def quat_to_mat(q) -> np.ndarray:
    r = np.zeros(9)
    mujoco.mju_quat2Mat(r, np.asarray(q, dtype=float))
    return r.reshape(3, 3)


def main() -> None:
    m, slots, _ = build_crew(UP, CD, 2, ego_camera=False, payload="cube")
    d = mujoco.MjData(m)
    s = slots[0]
    rz = float(np.load(REPO / "assets" / "cloth" / "right_hand_contact_pinch.npz")["planted_root_z"])
    jid = {j: mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, s.prefix + j) for j in JOINTS}
    hand = int(s.hands[1])

    # Chain root -> hand, keeping only the arm bodies (the base is the free root).
    chain, b = [], hand
    while int(m.body_parentid[b]) != 0:
        chain.append(b)
        b = int(m.body_parentid[b])
    base = b
    chain = list(reversed(chain))
    bodies = [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in chain]
    joint_bodies = [int(m.jnt_bodyid[jid[j]]) for j in ARM]
    assert chain[:-1] == joint_bodies and chain[-1] == hand, (bodies, ARM)
    for j in ARM:
        k = jid[j]
        assert int(m.jnt_type[k]) == int(mujoco.mjtJoint.mjJNT_HINGE), j
        assert np.allclose(m.jnt_axis[k], (0, 0, 1)) and np.allclose(m.jnt_pos[k], 0), j

    geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) == hand
             and int(m.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_MESH)]
    assert len(geoms) == 1, geoms
    g = geoms[0]
    mi = int(m.geom_dataid[g])
    verts = m.mesh_vert[int(m.mesh_vertadr[mi]):int(m.mesh_vertadr[mi]) + int(m.mesh_vertnum[mi])]
    v_hand = m.geom_pos[g] + verts @ quat_to_mat(m.geom_quat[g]).T
    hull = v_hand[ConvexHull(v_hand).vertices]

    inertia = []
    for b in chain:
        Rb = quat_to_mat(m.body_iquat[b])
        inertia.append(Rb @ np.diag(m.body_inertia[b]) @ Rb.T)
    arm = ArmChain(
        joints=ARM,
        body_pos=np.array([m.body_pos[b] for b in chain], dtype=float),
        body_rot=np.array([quat_to_mat(m.body_quat[b]) for b in chain], dtype=float),
        hull=np.asarray(hull, dtype=float),
        root_z=rz,
        lower=np.array([m.jnt_range[jid[j]][0] for j in ARM], dtype=float),
        upper=np.array([m.jnt_range[jid[j]][1] for j in ARM], dtype=float),
        mass=np.array([m.body_mass[b] for b in chain], dtype=float),
        com=np.array([m.body_ipos[b] for b in chain], dtype=float),
        inertia=np.array(inertia, dtype=float),
    )

    # Check against MuJoCo on random arm configurations, legs in the pinch squat.
    adr = {j: int(m.jnt_qposadr[jid[j]]) for j in JOINTS}
    rng = np.random.default_rng(0)
    qs = rng.uniform(arm.lower, arm.upper, size=(300, 5))
    worst = 0.0
    for q in qs:
        d.qpos[:] = 0.0
        d.qpos[s.qpos_adr + 2] = rz
        d.qpos[s.qpos_adr + 3] = 1.0
        for j in JOINTS:
            d.qpos[adr[j]] = PINCH_POSE.get(j, 0.0)
        for j, v in zip(ARM, q):
            d.qpos[adr[j]] = v
        mujoco.mj_forward(m, d)
        ref = d.geom_xpos[g] + verts @ d.geom_xmat[g].reshape(3, 3).T
        p, R = fk(arm, q[None])
        ours = p[0] + v_hand @ R[0].T
        worst = max(worst, float(np.abs(ours - ref).max()),
                    float(np.abs(p[0] - d.xpos[hand]).max()))
        assert np.allclose(hull_points(arm, q[None])[0],
                           p[0] + hull @ R[0].T)
    print(f"numpy FK vs MuJoCo over {len(qs)} random arm configurations: max error {worst:.2e} m")
    if worst > 1e-6:
        raise SystemExit("FK disagrees with MuJoCo; not writing")

    # Inverse dynamics against mj_rne: base held still, other joints at rest.
    dof = [int(m.jnt_dofadr[jid[j]]) for j in ARM]
    rne = np.zeros(m.nv)
    worst_tau = 0.0
    for q in qs[:100]:
        qd, qdd = rng.uniform(-3, 3, 5), rng.uniform(-20, 20, 5)
        d.qpos[:] = 0.0
        d.qpos[s.qpos_adr + 2] = rz
        d.qpos[s.qpos_adr + 3] = 1.0
        for j in JOINTS:
            d.qpos[adr[j]] = PINCH_POSE.get(j, 0.0)
        for j, v in zip(ARM, q):
            d.qpos[adr[j]] = v
        d.qvel[:] = 0.0
        d.qacc[:] = 0.0
        d.qvel[dof] = qd
        d.qacc[dof] = qdd
        mujoco.mj_forward(m, d)
        d.qacc[:] = 0.0
        d.qacc[dof] = qdd
        mujoco.mj_rne(m, d, 1, rne)
        ours = inverse_dynamics(arm, q[None], qd[None], qdd[None], gravity=m.opt.gravity)[0]
        worst_tau = max(worst_tau, float(np.abs(ours - rne[dof]).max()))
    print(f"numpy inverse dynamics vs mj_rne over 100 random states: max error {worst_tau:.2e} N m")
    if worst_tau > 1e-6:
        raise SystemExit("inverse dynamics disagrees with MuJoCo; not writing")

    lo, hi = hull.min(axis=0), hull.max(axis=0)
    print(f"base body {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, base)}; chain {bodies}")
    print(f"hand mesh: {len(v_hand)} vertices, hull {len(hull)}; hull AABB centre "
          f"{np.round((lo + hi) / 2, 4)} half-extents {np.round((hi - lo) / 2, 4)} (hand-link frame)")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        OUT, joints=np.array(ARM), body_names=np.array(bodies), body_pos=arm.body_pos,
        body_rot=arm.body_rot, hull=arm.hull, root_z=rz, lower=arm.lower, upper=arm.upper,
        mass=arm.mass, com=arm.com, inertia=arm.inertia, fk_max_error=worst, id_max_error=worst_tau,
        source="MuJoCo crew builder (bhl_robust.eval.coop_replay), right arm of slot 0; "
               "hull = scipy ConvexHull of the hand visual mesh in the hand-link frame",
    )
    print(f"wrote {OUT.relative_to(REPO)} ({OUT.stat().st_size / 1e3:.1f} kB)")


if __name__ == "__main__":
    main()
