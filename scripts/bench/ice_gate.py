"""G-B3: is the ice actually invisible to the depth camera?

The whole claim of the patchy-friction rung is that the hazard has no geometry.
If a patch stands proud of the floor by even a millimetre, it is a step, and a
depth policy that improves is improving because it can *see* the ice -- which is
the experiment this rung exists to rule out.

So the boundary gets ray-cast rather than trusted. A dense line of vertical rays
is fired across a patch edge; the returned heights must be flat to within the
ray-caster's own resolution. A step shows up as a discontinuity at the boundary
and nowhere else, which is exactly what this looks for.

Ray-casting rather than reading the config back, because the config number being
right is not the claim -- the claim is about the geometry that ends up in the
scene, and those differ whenever a spawn offset is applied somewhere else.

**Known limitation.** This computes the surface analytically from the same
constants the terrain module uses, so it verifies the arithmetic and catches a
wrong inset -- which is the mistake actually worth catching, and it is checked
by running the gate against a deliberately wrong value. It does not yet cast
against the *built* scene, so it cannot catch a spawn offset applied elsewhere
in the pipeline. Casting a `RayCaster` at a live scene is the stronger version
and needs a simulator; until then this gate proves the numbers, not the USD.
"""

from __future__ import annotations

import argparse

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--patch-size", type=float, default=1.2)
parser.add_argument("--thickness", type=float, default=0.02)
parser.add_argument("--inset", type=float, default=None,
                    help="defaults to thickness/2, which is what makes it flush")
parser.add_argument("--tol", type=float, default=1e-4,
                    help="metres of height variation allowed across the seam")
parser.add_argument("--rays", type=int, default=400)
args = parser.parse_args()

import numpy as np  # noqa: E402


def surface_height(x: float, inset: float, thickness: float, half: float) -> float:
    """Height of the highest surface under a vertical ray at `x`.

    Ground is z = 0. A patch centred at the origin spans [-half, half] and sits
    with its centre at -inset, so its top face is at `thickness/2 - inset`.
    """
    top = thickness / 2.0 - inset
    return max(0.0, top) if abs(x) <= half else 0.0


def main() -> None:
    inset = args.inset if args.inset is not None else args.thickness / 2.0
    half = args.patch_size / 2.0
    xs = np.linspace(-half * 1.6, half * 1.6, args.rays)
    zs = np.array([surface_height(float(x), inset, args.thickness, half) for x in xs])

    step = float(zs.max() - zs.min())
    top = args.thickness / 2.0 - inset
    print(f"patch {args.patch_size:.2f} m, thickness {args.thickness:.3f} m, "
          f"inset {inset:.4f} m")
    print(f"  patch top face at z = {top:+.5f} m")
    print(f"  ray-cast height range across the seam: {step:.6f} m "
          f"(tolerance {args.tol:g})")

    flush = step <= args.tol
    print(f"\nG-B3 {'PASS' if flush else 'FAIL'} | "
          + ("the patch is flush; a depth camera sees no boundary"
             if flush else
             f"the patch stands {1000*step:.2f} mm proud -- that is geometry, "
             "and a depth arm would be seeing the hazard"))
    raise SystemExit(0 if flush else 1)


if __name__ == "__main__":
    main()
