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

    # Two guards, and both are load-bearing.
    #
    # `hasattr(SimulationCfg, "physx")` is False on 2.3.2 as well as on 3.x --
    # `@configclass` gives the field a `default_factory`, so it never becomes a
    # class attribute. Checking `hasattr` therefore fired the shim on the v51
    # stack, which then tried to import a 3.x-only module and took down every
    # v51 task with `ModuleNotFoundError: No module named 'isaaclab_physx'`.
    # The annotation is the honest test of whether the field exists.
    #
    # And the import is guarded regardless, so a stack without isaaclab_physx
    # is left alone instead of raising during `import bhl_robust.tasks`.
    _has_field = "physx" in getattr(SimulationCfg, "__annotations__", {})
    if SimulationCfg is not None and not _has_field:
        try:
            from isaaclab_physx.physics.physx_manager_cfg import PhysxCfg
        except ModuleNotFoundError:
            PhysxCfg = None

        def _get_physx(self):
            cur = getattr(self, "physics", None)
            if not isinstance(cur, PhysxCfg):
                cur = PhysxCfg()
                self.physics = cur
            return cur

        def _set_physx(self, value):
            self.physics = value

        if PhysxCfg is not None:
            SimulationCfg.physx = property(_get_physx, _set_physx)
            applied.append("SimulationCfg.physx -> .physics (PhysxCfg)")

    # 3.x made asset data warp-first: `robot.data.root_quat_w` returns a
    # `ProxyArray` rather than a `torch.Tensor`. ProxyArray is a deprecation
    # bridge -- it forwards indexing, arithmetic and most torch functions, and
    # carries a zero-copy `.torch` view -- so almost everything keeps working.
    # What does not is `isaaclab.utils.math`, whose helpers are torch.jit
    # scripted and reject anything that is not literally a Tensor. The result is
    # that Isaac Lab's own math functions fail on Isaac Lab's own data:
    #
    #   quat_inv() Expected a value of type 'Tensor' for argument 'q'
    #   but instead found type 'ProxyArray'.
    #
    # Coercing at each call site would mean touching every MDP term that ever
    # reads an asset, in this repo and upstream. Wrapping the math module once
    # fixes all of them, costs nothing on 2.x (no ProxyArray exists, so the
    # unwrap is an identity check), and is zero-copy on 3.x.
    try:
        from isaaclab.utils.warp.proxy_array import ProxyArray
    except Exception:                                            # pragma: no cover
        ProxyArray = None

    if ProxyArray is not None:
        import functools
        import isaaclab.utils.math as _math

        def _unwrap(v):
            return v.torch if isinstance(v, ProxyArray) else v

        def _wrap(fn):
            @functools.wraps(fn)
            def inner(*a, **kw):
                return fn(*[_unwrap(x) for x in a],
                          **{k: _unwrap(v) for k, v in kw.items()})
            return inner

        n = 0
        for name in dir(_math):
            fn = getattr(_math, name, None)
            if name.startswith("_") or not callable(fn):
                continue
            # Only the scripted ones are strict about the type, and only they
            # need paying for. Wrapping plain Python helpers too would add a
            # layer to every math call in the hot loop for nothing.
            if type(fn).__name__ != "ScriptFunction":
                continue
            setattr(_math, name, _wrap(fn))
            n += 1
        if n:
            applied.append(f"isaaclab.utils.math: unwrap ProxyArray on {n} scripted fns")

    return applied
