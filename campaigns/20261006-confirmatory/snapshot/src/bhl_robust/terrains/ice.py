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

They also have to be *where the robots are*. Spawned as ``AssetBaseCfg`` under
``{ENV_REGEX_NS}`` they sat at the clone-grid origin; with a terrain generator
Isaac Lab resets each robot onto a terrain origin a median 72 m away
(``21328532``). They are kinematic rigid bodies now, and ``reset_ice_patches``
writes ``default_root_state + env_origins`` at every reset -- the same move
``reset_root_state_uniform`` does for the robot.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm

from bhl_robust.terrains.ice_layout import (
    N_PATCHES,
    PATCH_INSET,
    PATCH_SIZE,
    PATCH_THICKNESS,
    ice_local_offsets,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

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

# Re-export so existing ``from bhl_robust.terrains.ice import PATCH_INSET``
# still works (ice_gate, tests).
__all__ = [
    "ICE_DYNAMIC",
    "ICE_STATIC",
    "PATCH_INSET",
    "PATCH_THICKNESS",
    "ice_patches",
    "install_ice",
    "reset_ice_patches",
]


def _patch(name: str, size: float, pos: tuple[float, float], rgba) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{name}",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(pos[0], pos[1], -PATCH_INSET)),
        spawn=sim_utils.CuboidCfg(
            size=(size, size, PATCH_THICKNESS),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True, kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgba[:3]),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=ICE_STATIC,
                dynamic_friction=ICE_DYNAMIC,
                restitution=0.0,
                friction_combine_mode="min",
            ),
        ),
    )


def ice_patches(n: int = N_PATCHES, size: float = PATCH_SIZE,
                visible: bool = False) -> dict[str, RigidObjectCfg]:
    """A checker of low-friction squares the robot has to cross.

    Packed inside one terrain tile (see ``ice_layout.ice_local_offsets``).
    """
    rgba = VISIBLE_ICE_RGBA if visible else ICE_RGBA
    out = {}
    for i, pos in enumerate(ice_local_offsets(n=n, size=size)):
        out[f"ice_{i}"] = _patch(f"ice_{i}", size, pos, rgba)
    return out


def reset_ice_patches(env: "ManagerBasedEnv", env_ids):
    """Move each env's ice onto that env's terrain origin.

    ``default_root_state`` is the configured local offset. Adding
    ``scene.env_origins`` is what ``reset_root_state_uniform`` does for the
    robot; without it the patches stay at the clone-grid origin.
    """
    origins = env.scene.env_origins[env_ids]
    for name in sorted(k for k in env.scene.keys() if str(k).startswith("ice_")):
        asset = env.scene[name]
        root = asset.data.default_root_state[env_ids].clone()
        pose = root[:, :7].clone()
        pose[:, :3] = root[:, :3] + origins
        asset.write_root_pose_to_sim(pose, env_ids=env_ids)
        asset.write_root_velocity_to_sim(root[:, 7:13], env_ids=env_ids)


def install_ice(cfg, *, visible: bool = False) -> None:
    """Flat tiles, then kinematic patches that follow ``env_origins`` at reset.

    Called from every ice env's ``__post_init__``. The generator is the bumpy
    menu with every sub-terrain replaced by a zero-slope tile, so the patch
    stays flush as the curriculum promotes, and the robots still land on
    terrain origins rather than the clone grid.
    """
    from bhl_robust.terrains.bumpy import ICE_TERRAINS_CFG

    cfg.scene.terrain.terrain_generator = ICE_TERRAINS_CFG
    for name, patch in ice_patches(visible=visible).items():
        setattr(cfg.scene, name, patch)
    cfg.events.reset_ice = EventTerm(func=reset_ice_patches, mode="reset")
