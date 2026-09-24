"""Ice no-ice control for the placed-ice rung (LOC-11, authorized 2026-09-24).

`BipedIceEnvCfg` / `BipedIceDepthEnvCfg` place six low-friction patches under
the robots (static 0.25 / dynamic 0.18, `friction_combine_mode="min"`); the
exposure probe (`21402548`) confirmed the feet spend a third of every episode
on them. The depth arm beat the blind arm there (terrain level 2.92 vs 2.59,
n = 2). Whether that gap is *about ice* is the open question, so this control
keeps everything -- patches, reset placement, terrain generator, curriculum,
observations -- and changes exactly one thing: the patch material becomes the
ground's own material. A depth advantage that survives here has nothing to do
with friction.

The parent cfgs live in files that carry the author's uncommitted edits; this
module subclasses them without editing them. The training launcher records the
working-tree state of those files next to the run.
"""

from __future__ import annotations

from isaaclab.utils import configclass

from bhl_robust.tasks.depth_env_cfg import BipedIceDepthEnvCfg
from bhl_robust.tasks.terrain_env_cfg import BipedIceEnvCfg


def neutralize_ice_patches(cfg) -> dict:
    """Give every `ice_*` patch the terrain's friction; return what was set."""
    ground = cfg.scene.terrain.physics_material
    static, dynamic = float(ground.static_friction), float(ground.dynamic_friction)
    changed = []
    i = 0
    while True:
        patch = getattr(cfg.scene, f"ice_{i}", None)
        if patch is None:
            break
        mat = patch.spawn.physics_material
        mat.static_friction, mat.dynamic_friction = static, dynamic
        # "min" with equal values is neutral; keep it so nothing else differs.
        changed.append(f"ice_{i}")
        i += 1
    if not changed:
        raise RuntimeError("no ice_* patches found on the scene; the control would be the plain ice task")
    return {"patches": changed, "static_friction": static, "dynamic_friction": dynamic}


@configclass
class BipedIceControlEnvCfg(BipedIceEnvCfg):
    """Placed patches at ground friction, blind."""

    def __post_init__(self):
        super().__post_init__()
        self.ice_control = neutralize_ice_patches(self)


@configclass
class BipedIceControlDepthEnvCfg(BipedIceDepthEnvCfg):
    """Placed patches at ground friction, forward depth camera."""

    def __post_init__(self):
        super().__post_init__()
        self.ice_control = neutralize_ice_patches(self)
