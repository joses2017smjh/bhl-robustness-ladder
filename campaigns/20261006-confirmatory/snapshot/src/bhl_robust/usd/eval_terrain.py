"""Evaluation terrains as a USD asset with a difficulty variant set.

The MuJoCo harness currently rebuilds a height field from a seed and a scalar
d every time it scores a policy. That is reproducible *if you have this repo*.
A USD asset with a variant set is the same surface, but inspectable, shareable,
and selectable without calling our generator: `d` becomes a variant selection
rather than a function argument.

The mesh is the same field `bhl_robust.eval.terrain_field.build_height_field`
already emits, so a policy scored against the USD and one scored against the
MuJoCo hfield are looking at the same elevation, not two independently sampled
surfaces.

This is deliberately OpenUSD Python (`pxr`), not Composer: the cluster has no
display, and a scripted stage is the artifact that fits the sbatch workflow.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from bhl_robust.eval.terrain_field import GRID, HALF_EXTENT, build_height_field


DIFFICULTIES = (0.00, 0.20, 0.40, 0.60, 0.80, 1.00)


def variant_name(d: float) -> str:
    """USD variant names are identifiers — no dots."""
    return f"d{int(round(d * 100)):03d}"


def _mesh(field_m: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lattice elevation -> triangle mesh in metres, z-up."""
    n = field_m.shape[0]
    xs = np.linspace(-HALF_EXTENT, HALF_EXTENT, n)
    # field is indexed [x, y] (see terrain_field.py).
    pts = np.empty((n * n, 3), dtype=np.float64)
    for i, x in enumerate(xs):
        for j, y in enumerate(xs):
            pts[i * n + j] = (x, y, float(field_m[i, j]))
    faces = []
    for i in range(n - 1):
        for j in range(n - 1):
            a = i * n + j
            b = a + 1
            c = a + n
            d = c + 1
            faces.append((a, c, b))
            faces.append((b, c, d))
    return pts, np.asarray(faces, dtype=np.int32)


def write_stage(path: Path, seed: int = 12345) -> Path:
    from pxr import Gf, Usd, UsdGeom, UsdPhysics

    path.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateNew(str(path))
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    stage.SetMetadata("comment",
                      f"BHL eval terrain; seed={seed}; variants are difficulty d")

    world = UsdGeom.Xform.Define(stage, "/World")
    vs = world.GetPrim().GetVariantSets().AddVariantSet("difficulty")

    # One mesh prim; each variant edits its points in place.
    mesh = UsdGeom.Mesh.Define(stage, "/World/terrain")
    mesh.CreateSubdivisionSchemeAttr().Set("none")
    UsdPhysics.CollisionAPI.Apply(mesh.GetPrim())
    UsdPhysics.MeshCollisionAPI.Apply(mesh.GetPrim()).CreateApproximationAttr().Set("none")

    for d in DIFFICULTIES:
        name = variant_name(d)
        vs.AddVariant(name)
        vs.SetVariantSelection(name)
        with vs.GetVariantEditContext():
            field = build_height_field(d, seed)
            pts, faces = _mesh(field)
            mesh.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
            mesh.CreateFaceVertexCountsAttr([3] * len(faces))
            mesh.CreateFaceVertexIndicesAttr(faces.ravel().tolist())
            mesh.CreateExtentAttr([
                Gf.Vec3f(-HALF_EXTENT, -HALF_EXTENT, float(field.min())),
                Gf.Vec3f(HALF_EXTENT, HALF_EXTENT, float(field.max())),
            ])
            mesh.GetPrim().SetCustomDataByKey("bhl:difficulty", float(d))
            mesh.GetPrim().SetCustomDataByKey("bhl:seed", int(seed))

    vs.SetVariantSelection("d000")
    stage.GetRootLayer().Save()
    return path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=12345,
                    help="must match the MuJoCo harness terrain_seed (12345)")
    args = ap.parse_args()
    out = write_stage(args.out, args.seed)
    print(f"USD terrain -> {out}")
    from pxr import Usd
    stage = Usd.Stage.Open(str(out))
    vs = stage.GetPrimAtPath("/World").GetVariantSet("difficulty")
    print(f"  variants: {vs.GetVariantNames()}")
    print(f"  selected: {vs.GetVariantSelection()}")


if __name__ == "__main__":
    main()
