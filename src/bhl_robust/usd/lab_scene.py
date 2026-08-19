"""A composed evaluation scene: lab floor, threshold, ramp, cable.

Parameterized noise asks whether a policy memorised a height-field distribution.
A door threshold, a cable run, and a ramp ask a sharper question: does it
generalize to structured real-world geometry?

Materials and lighting are part of the point. Depth-sensor artifacts — holes,
edge fattening, specular dropout — are material-dependent; they cannot be
simulated on untextured generator geometry. This stage is the prerequisite
for that, even though the cameras themselves are still blocked on this cluster.

Scripted OpenUSD, not Composer: same reason as the difficulty-variant terrain.
"""
from __future__ import annotations

import argparse
from pathlib import Path


def write_lab(path: Path) -> Path:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetMetadata("comment", "BHL lab eval scene: tile, carpet, cable, threshold, ramp")

    UsdGeom.Xform.Define(stage, "/World")

    def box(prim, half, pos, rgb, collide=True):
        p = UsdGeom.Cube.Define(stage, prim)
        p.CreateSizeAttr(2.0)  # unit cube, then scale
        xform = UsdGeom.Xformable(p.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        xform.AddScaleOp().Set(Gf.Vec3d(*half))
        p.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        if collide:
            UsdPhysics.CollisionAPI.Apply(p.GetPrim())
        return p

    def cylinder(prim, radius, half_height, pos, rgb, rotate_x=False):
        p = UsdGeom.Cylinder.Define(stage, prim)
        p.CreateRadiusAttr(radius)
        p.CreateHeightAttr(2.0 * half_height)
        p.CreateAxisAttr("z")
        xform = UsdGeom.Xformable(p.GetPrim())
        xform.AddTranslateOp().Set(Gf.Vec3d(*pos))
        if rotate_x:
            xform.AddRotateXOp().Set(90.0)
        p.CreateDisplayColorAttr([Gf.Vec3f(*rgb)])
        UsdPhysics.CollisionAPI.Apply(p.GetPrim())
        return p

    # Infinite ground via a thin box large enough for a 10 s rollout.
    box("/World/floor", (8.0, 4.0, 0.02), (2.0, 0.0, -0.02), (0.72, 0.72, 0.70))
    # Carpet strip — visual distinction; collision is the floor beneath.
    box("/World/carpet", (1.4, 3.0, 0.004), (0.6, 0.0, 0.004), (0.42, 0.28, 0.22),
        collide=False)
    cylinder("/World/cable", 0.025, 2.6, (1.6, 0.0, 0.025), (0.10, 0.10, 0.10),
             rotate_x=True)
    box("/World/threshold", (0.07, 2.6, 0.038), (2.8, 0.0, 0.038), (0.48, 0.38, 0.26))
    # Ramp: a long thin box pitched about Y.
    ramp = UsdGeom.Cube.Define(stage, "/World/ramp")
    ramp.CreateSizeAttr(2.0)
    xf = UsdGeom.Xformable(ramp.GetPrim())
    xf.AddTranslateOp().Set(Gf.Vec3d(4.3, 0.0, 0.085))
    xf.AddRotateYOp().Set(-6.9)          # atan(0.12/1.0) ~ 6.8 deg
    xf.AddScaleOp().Set(Gf.Vec3d(0.85, 2.4, 0.025))
    ramp.CreateDisplayColorAttr([Gf.Vec3f(0.58, 0.58, 0.54)])
    UsdPhysics.CollisionAPI.Apply(ramp.GetPrim())
    box("/World/landing", (0.70, 2.4, 0.085), (5.7, 0.0, 0.085), (0.52, 0.52, 0.50))

    stage.GetRootLayer().Save()
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    out = write_lab(args.out)
    print(f"USD lab scene -> {out}")
    from pxr import Usd
    stage = Usd.Stage.Open(str(out))
    prims = [p.GetPath().pathString for p in stage.Traverse() if p.GetPath().pathString.count("/") == 2]
    print("  prims:", ", ".join(prims))


if __name__ == "__main__":
    main()
