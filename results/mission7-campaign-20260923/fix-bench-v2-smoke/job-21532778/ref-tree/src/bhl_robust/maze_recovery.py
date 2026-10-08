"""Tensor-only, simulator-independent state and rewards for maze recovery.

The legacy maze MDP stays intact for replay. These helpers are used only by
the separately registered MazeRecovery tasks. Distances and success remain
privileged training/evaluation signals, never inferred sensor measurements.
"""

from __future__ import annotations

import torch


def tensor(value):
    return value.torch if hasattr(value, "torch") else value


def local_xy(env):
    return tensor(env.scene["robot"].data.root_pos_w)[:, :2] - tensor(env.scene.env_origins)[:, :2]


def _state(env, radius=0.40, hold_steps=8):
    """Update once per physical control step, even across manager consumers.

    Isaac calls terminations before rewards. A reward callback must not count
    a second physical sample toward the dwell requirement. Episode length
    also detects per-environment resets under a common simulation step.
    """
    step = int(env.common_step_counter)
    lengths = tensor(env.episode_length_buf)
    state = getattr(env, "_maze_recovery_state", None)
    if state is not None and state["step"] == step and torch.equal(state["lengths"], lengths):
        return state
    xy = local_xy(env)
    goal = xy.new_tensor((3.0, 0.0))
    dist = torch.linalg.vector_norm(xy - goal, dim=-1)
    if state is None:
        previous = dist.clone()
        hold = torch.zeros_like(lengths, dtype=torch.long)
        reset = torch.ones_like(lengths, dtype=torch.bool)
    else:
        previous = state["dist"]
        hold = state["hold"]
        reset = (lengths <= 1) | (lengths <= state["lengths"])
    robot = env.scene["robot"].data
    gravity = tensor(robot.projected_gravity_b)
    speed = torch.linalg.vector_norm(tensor(robot.root_lin_vel_b)[:, :2], dim=-1)
    tilt_rate = torch.linalg.vector_norm(tensor(robot.root_ang_vel_b)[:, :2], dim=-1)
    # Require an upright settled inspection, including the final dwell sample.
    # cos(tilt)>0.90 is stricter than the environment's 0.78 rad fall threshold.
    settled = (gravity[:, 2] < -0.90) & (speed <= 0.20) & (tilt_rate <= 0.60)
    hold = torch.where((dist <= radius) & settled & ~reset, hold + 1, torch.zeros_like(hold))
    # RewardManager multiplies rates by step_dt. The old per-step displacement
    # was multiplied by dt twice, suppressing the only task progress signal.
    progress = torch.where(reset, torch.zeros_like(dist), (previous - dist) / env.step_dt)
    state = dict(step=step, lengths=lengths.clone(), dist=dist.detach().clone(),
                 hold=hold, progress=progress, settled=settled, success=hold >= hold_steps)
    env._maze_recovery_state = state
    return state


def progress_rate(env):
    return _state(env)["progress"]


def button_reached(env):
    return _state(env)["success"]


def success_bonus(env):
    """A one-off success return, independent of simulation control dt."""
    return button_reached(env).float() / env.step_dt


def failure_penalty(env):
    """Do not apply the inherited fall penalty to a successful termination."""
    terminated = tensor(env.termination_manager.terminated)
    return (terminated & ~button_reached(env)).float()


def command_speed(distance, heading_error, cruise_speed=0.35, stop_radius=0.26):
    """Brake in the target and rotate before translating at large yaw error."""
    speed = torch.clamp((distance - stop_radius) / 0.5, 0.0, 1.0) * cruise_speed
    return speed * torch.clamp(torch.cos(heading_error), 0.0, 1.0)
