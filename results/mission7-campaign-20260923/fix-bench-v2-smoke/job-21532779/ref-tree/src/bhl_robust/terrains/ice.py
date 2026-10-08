"""B3: low-friction patches on ground that is geometrically flat.

The point of this rung is to separate two things the terrain ladder has so far
confounded. Rough ground is *both* a geometry problem and a contact problem, and
depth helps there. Low friction applied uniformly (`BipedSlipperyEnvCfg`) is a
contact problem with no geometry at all -- and depth helped there too, by 2.9x,
which is the result that inverted the prediction this repo wrote down.

Patches are the sharper version of that question. If friction varies *across the
floor* and the floor is flat, a depth camera cannot possibly localise the
hazard: there is nothing to see. A policy that still improves with depth is
using it for something other than seeing the ice -- most likely foot placement
that needs less friction margin everywhere.

So the patches must be exactly coplanar with the ground. A patch raised even a
millimetre is a step, a step is geometry, and the experiment quietly becomes the
one it was designed to exclude. `PATCH_INSET` is what enforces that, and G-B3
ray-casts the boundary to check it rather than trusting the number.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg

#: Patch top face sits at exactly z = 0, flush with the ground plane.
PATCH_THICKNESS = 0.02
PATCH_INSET = PATCH_THICKNESS / 2.0

#: Friction of the patches, against the default ground. The uniform-slippery
#: rung used static 0.25 / dynamic 0.18; matching it keeps the two comparable,
#: so "patchy vs uniform" is about the spatial distribution and not the value.
ICE_STATIC = 0.25
ICE_DYNAMIC = 0.18

#: Deliberately not visually distinct from the floor by default. A blue patch
#: would be invisible to *depth* and obvious to *RGB*, which would silently make
#: this an RGB experiment. `VISIBLE_ICE` exists to run exactly that comparison
#: on purpose, as a separate arm.
ICE_RGBA = (0.55, 0.57, 0.60, 1.0)
VISIBLE_ICE_RGBA = (0.35, 0.72, 0.95, 1.0)


def _patch(name: str, size: float, pos: tuple[float, float], rgba) -> AssetBaseCfg:
    return AssetBaseCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(pos[0], pos[1], -PATCH_INSET)),
        spawn=sim_utils.CuboidCfg(
            size=(size, size, PATCH_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgba[:3]),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=ICE_STATIC,
                dynamic_friction=ICE_DYNAMIC,
                restitution=0.0,
                friction_combine_mode="min",
            ),
        ),
    )


def ice_patches(n: int = 6, size: float = 1.2, spacing: float = 2.0,
                visible: bool = False) -> dict[str, AssetBaseCfg]:
    """A checker of low-friction squares the robot has to cross.

    Laid along +x on alternating y, so a straight-line velocity command cannot
    avoid them by drifting sideways -- an unavoidable hazard is the only kind
    that measures anything.
    """
    rgba = VISIBLE_ICE_RGBA if visible else ICE_RGBA
    out = {}
    for i in range(n):
        y = (size * 0.6) * (1 if i % 2 else -1)
        out[f"ice_{i}"] = _patch(f"ice_{i}", size, ((i + 1) * spacing, y), rgba)
    return out
