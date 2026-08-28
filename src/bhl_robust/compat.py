"""Compatibility shims that let the overlays import on Isaac Lab 3.x.

This repo's standing rule is that `external/` stays pristine and re-pinnable, so
upstream breakages get worked around rather than patched. The same rule applies
here, and it has to, because the 2.x -> 3.x breakage is *in upstream's own
imports*: `berkeley_humanoid_lite`'s env configs import
`AdditiveUniformNoiseCfg`, Isaac Lab 3.x removed that name, and every overlay
that imports an upstream config inherits the failure. Twelve of fifteen modules
failed the port audit on that one symbol.

Nothing here is speculative. Each shim exists because a specific import failed
in `scripts/bench/port_audit.py`, and each is a no-op on 2.x, so importing this
module on the v51 stack changes nothing about the runs that produced every
published number.

Call `apply()` before importing anything from `berkeley_humanoid_lite`.
"""

from __future__ import annotations


def apply() -> list[str]:
    """Install the shims. Returns the names of the ones that were needed."""
    applied: list[str] = []

    # `AdditiveUniformNoiseCfg` was `UniformNoiseCfg` with `operation="add"`,
    # which is already the default; 3.x dropped the redundant alias rather than
    # changing any behaviour. Restoring the name is therefore exact, not an
    # approximation -- the shimmed class produces identical noise.
    import isaaclab.utils.noise as _noise

    if not hasattr(_noise, "AdditiveUniformNoiseCfg"):
        _noise.AdditiveUniformNoiseCfg = _noise.UniformNoiseCfg
        applied.append("AdditiveUniformNoiseCfg -> UniformNoiseCfg")

    if not hasattr(_noise, "AdditiveGaussianNoiseCfg") and hasattr(_noise, "GaussianNoiseCfg"):
        _noise.AdditiveGaussianNoiseCfg = _noise.GaussianNoiseCfg
        applied.append("AdditiveGaussianNoiseCfg -> GaussianNoiseCfg")

    # 3.x moved the physics engine out of the core package: `SimulationCfg.physx`
    # became `SimulationCfg.physics`, typed as a generic `PhysicsCfg`, with the
    # PhysX-specific fields living on `isaaclab_physx`'s `PhysxCfg` subclass.
    # `bounce_threshold_velocity` and `gpu_max_rigid_patch_count` still exist and
    # still mean the same thing; only the attribute path changed.
    #
    # Upstream's `velocity_env_cfg` writes `self.sim.physx.gpu_max_rigid_patch_count`
    # at module scope, so without this every overlay that inherits from it dies
    # on 3.x with `'SimulationCfg' object has no attribute 'physx'` -- which is
    # what all nine redesigned tasks did.
    #
    # The property lazily installs a `PhysxCfg` the first time `.physx` is read,
    # so existing `self.sim.physx.X = v` code works unchanged on both stacks and
    # a config that never touches it keeps 3.x's default of `physics=None`.
    try:
        from isaaclab.sim.simulation_cfg import SimulationCfg
    except Exception:                                            # pragma: no cover
        SimulationCfg = None

    if SimulationCfg is not None and not hasattr(SimulationCfg, "physx"):
        from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg

        def _get_physx(self):
            cur = getattr(self, "physics", None)
            if not isinstance(cur, PhysxCfg):
                cur = PhysxCfg()
                self.physics = cur
            return cur

        def _set_physx(self, value):
            self.physics = value

        SimulationCfg.physx = property(_get_physx, _set_physx)
        applied.append("SimulationCfg.physx -> .physics (PhysxCfg)")

    return applied
