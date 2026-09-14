#!/usr/bin/env python3
"""Give the robot's hands the collision geometry the shipped asset leaves out.

The Berkeley Humanoid Lite URDF declares a <visual> on each hand link and no
<collision>; the USD converted from it, and the MuJoCo model built from it,
follow suit. So in every simulator this repo uses, a hand passes through
whatever it touches -- only the forearms, shoulders and legs collide. A sweep
made with the hand therefore cannot move a garment (21300493: 0.07 mm).

This writes a workspace overlay that *sublayers* the untouched upstream USD and
adds one convex collider under each hand link: the convex hull of that hand's
own visual mesh in the link frame, cut to at most 64 vertices so PhysX cooks it
as authored (``arm_fk.reduce_hull``; the fingertip face is kept exactly). No
joint, mass or visual changes; external/ is not modified.

The first overlay used the mesh's bounding box. The hand tapers to a
1.7 x 3.0 cm fingertip, and the box's 5.9 x 7.4 cm bottom face tilted with the
hand; in 21300604 the box pushed and flipped the shirt where the real hand
would not have. The right hand's hull is checked against the MuJoCo hull that
``assets/cloth/right_arm_chain.npz`` stores for the sweep planner, so the
collider Isaac simulates is the shape the planner replays.

Run with Isaac's bundled pxr on the path (slurm/_env.sh setup_pxr):

    PYTHONPATH=<omni.usd.libs> LD_LIBRARY_PATH=... python scripts/cloth/add_hand_colliders.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
from pxr import Gf, Sdf, Usd, UsdGeom, UsdPhysics
from scipy.spatial import ConvexHull

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from bhl_robust.cloth.arm_fk import hull_excess, load_chain, reduce_hull  # noqa: E402

BASE = (REPO / "external/Berkeley-Humanoid-Lite/source/berkeley_humanoid_lite_assets/data/robots/"
        "berkeley_humanoid/berkeley_humanoid_lite/usd/berkeley_humanoid_lite.usd")
OUT = REPO / "assets/cloth/berkeley_humanoid_lite_hand_colliders.usda"
MAX_VERTICES = 64


def outward_triangles(v: np.ndarray) -> list[list[int]]:
    hull = ConvexHull(v)
    tris = []
    for simplex, eq in zip(hull.simplices, hull.equations):
        a, b, c = v[simplex]
        if np.cross(b - a, c - a) @ eq[:3] < 0:
            simplex = simplex[[0, 2, 1]]
        tris.append([int(i) for i in simplex])
    return tris


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    if OUT.exists():
        OUT.unlink()
    layer = Sdf.Layer.CreateNew(str(OUT))
    layer.subLayerPaths.append(os.path.relpath(BASE, OUT.parent))
    stage = Usd.Stage.Open(layer)
    root = stage.GetPrimAtPath("/berkeley_humanoid_lite")
    stage.SetDefaultPrim(root)
    for side in ("right", "left"):
        link = stage.GetPrimAtPath(f"/berkeley_humanoid_lite/arm_{side}_hand_link")
        mesh = stage.GetPrimAtPath(
            f"/berkeley_humanoid_lite/arm_{side}_hand_link/visuals/arm_{side}_hand_link_visual/mesh")
        assert link and mesh, f"{side} hand prims not found"
        m_mesh = UsdGeom.Xformable(mesh).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        m_link = UsdGeom.Xformable(link).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        rel = m_mesh * m_link.GetInverse()
        pts = np.array([list(rel.Transform(Gf.Vec3d(p))) for p in UsdGeom.Mesh(mesh).GetPointsAttr().Get()])
        full = pts[ConvexHull(pts).vertices]
        v = reduce_hull(full, MAX_VERTICES)
        if side == "right":
            mj = load_chain().hull
            gap = float(np.abs(np.r_[full.min(0) - mj.min(0), full.max(0) - mj.max(0)]).max())
            assert gap < 1e-4, f"USD and MuJoCo right-hand hulls disagree by {gap:.2e} m: frames differ"
        tris = outward_triangles(v)
        col = UsdGeom.Mesh.Define(stage, link.GetPath().AppendChild("hand_collider"))
        col.CreatePointsAttr([Gf.Vec3f(*map(float, p)) for p in v])
        col.CreateFaceVertexCountsAttr([3] * len(tris))
        col.CreateFaceVertexIndicesAttr([i for t in tris for i in t])
        col.CreatePurposeAttr(UsdGeom.Tokens.guide)
        UsdPhysics.CollisionAPI.Apply(col.GetPrim())
        UsdPhysics.MeshCollisionAPI.Apply(col.GetPrim()).CreateApproximationAttr(UsdPhysics.Tokens.convexHull)
        print(f"{side} hand collider: {len(v)} of {len(full)} hull vertices, {len(tris)} triangles; "
              f"full hull sticks out of it by at most {hull_excess(full, v) * 1000:.2f} mm; "
              f"lowest point {v[:, 2].min():.4f} vs {full[:, 2].min():.4f} (link frame)")
    layer.Save()
    check = Usd.Stage.Open(str(OUT))
    names = sorted(str(p.GetPath()) for p in check.Traverse(Usd.TraverseInstanceProxies())
                   if p.HasAPI(UsdPhysics.CollisionAPI) and "hand" in str(p.GetPath()))
    print(f"wrote {OUT.relative_to(REPO)}; hand collision prims: {names}")


if __name__ == "__main__":
    main()
