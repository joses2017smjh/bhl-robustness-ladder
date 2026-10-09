"""R1HO: R1H physics, commands and rewards, plus command-latched heading.

Only the actor and critic observations change (77/80 -> 79/82). The new
reference uses observable command changes; R1H's reward keeps its original
explicit-mode/resample semantics. Same initial weights for three independent
fine-tuning seeds is conditional repeatability, not three from-scratch runs.
"""
from __future__ import annotations

import torch
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from bhl_robust.eval.heading_observable import torch_latched_features
from bhl_robust.tasks.arms_env_cfg import (
    HumanoidTurnGaitClockHoldCfg, GaitClockActorObservationsCfg,
    _GaitClockPolicyObsCfg, _GaitClockCriticObsCfg,
)


def command_latched_heading(env, command_name: str = "base_velocity"):
    term = env.command_manager.get_term(command_name)
    yaw = term.robot.data.heading_w
    command = term.vel_command_b
    if not hasattr(term, "_h4_reference"):
        term._h4_reference = torch.zeros_like(yaw)
        term._h4_previous_command = torch.zeros_like(command, dtype=torch.float32)
        term._h4_ready = torch.zeros_like(yaw, dtype=torch.bool)
    reset = (env.episode_length_buf == 0) if hasattr(env, "episode_length_buf") else ~term._h4_ready
    features, reference, previous, ready = torch_latched_features(
        yaw, command, term._h4_reference, term._h4_previous_command, term._h4_ready, reset)
    term._h4_reference = reference
    term._h4_previous_command = previous
    term._h4_ready = ready
    return features


@configclass
class _HeadingPolicyObsCfg(_GaitClockPolicyObsCfg):
    command_latched_heading = ObsTerm(func=command_latched_heading)


@configclass
class _HeadingCriticObsCfg(_GaitClockCriticObsCfg):
    command_latched_heading = ObsTerm(func=command_latched_heading)


@configclass
class HeadingObservableObservationsCfg(GaitClockActorObservationsCfg):
    policy: _HeadingPolicyObsCfg = _HeadingPolicyObsCfg()
    critic: _HeadingCriticObsCfg = _HeadingCriticObsCfg()


@configclass
class HumanoidTurnGaitHeadingObservableCfg(HumanoidTurnGaitClockHoldCfg):
    observations: HeadingObservableObservationsCfg = HeadingObservableObservationsCfg()
