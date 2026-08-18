"""Disturbance curriculum: ramp external push magnitude over training.

Upstream BHL ships a `push_robot` event commented out, with a fixed +/-1.0 m/s
velocity kick. Enabling it at full strength from step zero tends to destroy
early learning -- the policy is knocked over before it can walk, so it never
collects the on-policy data that would teach recovery.

The fix is a curriculum: hold the push at zero while a gait forms, then ramp
linearly to full magnitude. This mirrors how terrain curricula are handled in
Isaac Lab, but keyed on training progress rather than per-env competence,
because a push is applied to all envs uniformly.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def push_velocity_levels(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    term_name: str = "push_robot",
    start_magnitude: float = 0.0,
    end_magnitude: float = 1.5,
    start_step: int = 24_000,
    full_step: int = 96_000,
) -> float:
    """Linearly ramp the `push_robot` velocity range with training progress.

    Args:
        term_name: event term to mutate. Must exist in the env's EventsCfg.
        start_magnitude: push magnitude (m/s) before `start_step`.
        end_magnitude: push magnitude (m/s) at and after `full_step`.
        start_step: env-step count at which the ramp begins. Defaults to roughly
            the first 1000 PPO iterations at 24 steps/env, i.e. long enough for
            a gait to appear before the robot starts getting shoved.
        full_step: env-step count at which the ramp saturates.

    Returns:
        The current push magnitude, which the CurriculumManager logs each
        iteration. That log line is the evidence the ramp actually moved, so it
        is worth keeping even though the value is not consumed.
    """
    step = env.common_step_counter

    if step <= start_step:
        frac = 0.0
    elif step >= full_step:
        frac = 1.0
    else:
        frac = (step - start_step) / max(1, (full_step - start_step))

    magnitude = start_magnitude + frac * (end_magnitude - start_magnitude)

    # Mutate the live event term. get_term_cfg returns the actual cfg object,
    # but set_term_cfg is still called explicitly so this stays correct if
    # Isaac Lab ever switches to returning a copy.
    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["velocity_range"] = {
        "x": (-magnitude, magnitude),
        "y": (-magnitude, magnitude),
    }
    env.event_manager.set_term_cfg(term_name, term_cfg)

    return magnitude


def push_levels_adaptive(
    env,
    env_ids,
    term_name: str = "push_robot",
    step: float = 0.02,
    min_magnitude: float = 0.0,
    max_magnitude: float = 1.0,
    fall_rate_target: float = 0.20,
) -> float:
    """Raise the push only while the policy is actually surviving it.

    The fixed linear ramp in `push_velocity_levels` has a structural flaw that
    pass 1 exposed: the schedule advances on wall-clock training progress
    regardless of whether the policy is coping. Once the magnitude outran the
    robot's ability to recover, the ramp kept climbing and drove a policy that
    had reached reward 24.7 down to ~3.

    This is the same feedback principle Isaac Lab already uses for terrain
    levels (`terrain_levels_vel`), applied to disturbance magnitude: promote
    when the recent fall rate is below target, demote when it is well above.
    The magnitude becomes an outcome of competence rather than of iteration
    count, so the curriculum cannot outrun the policy. It also means the
    converged magnitude is itself a measurement -- roughly "the largest shove
    this machine can learn to reject" -- which a fixed ramp cannot report.

    Returns the current magnitude, logged each iteration by the manager.
    """
    # `terminated` excludes time-outs, so for this task it is the fall signal.
    terminated = env.termination_manager.terminated[env_ids]
    fall_rate = float(terminated.float().mean()) if len(env_ids) else 0.0

    magnitude = getattr(env, "_bhl_push_magnitude", min_magnitude)
    if fall_rate < fall_rate_target:
        magnitude = min(max_magnitude, magnitude + step)
    elif fall_rate > 2.0 * fall_rate_target:
        magnitude = max(min_magnitude, magnitude - step)
    env._bhl_push_magnitude = magnitude

    term_cfg = env.event_manager.get_term_cfg(term_name)
    term_cfg.params["velocity_range"] = {"x": (-magnitude, magnitude), "y": (-magnitude, magnitude)}
    env.event_manager.set_term_cfg(term_name, term_cfg)
    return magnitude
