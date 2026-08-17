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
