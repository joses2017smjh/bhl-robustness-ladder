"""Navigation MDP for B5: walk the corridor and reach the button.

The maze robot is the 12-DoF biped. There is no hand to press the plate, so
success is base proximity to ``BUTTON_AT`` — a whole-body bump, labelled as
one. Commands, progress and dead-end use the same tile-relative coordinates
as ``maze_layout`` (origin = terrain / env origin).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

from bhl_robust.terrains.maze_layout import (
    BUTTON_AT,
    BUTTON_RADIUS,
    CORRIDOR_Y_LIMIT,
    CRUISE_SPEED,
    PATH,
    WAYPOINT_RADIUS,
    WRONG_WAY_X,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def _t(x):
    """ProxyArray on Isaac Lab 3.0, tensor on 2.x."""
    return x.torch if hasattr(x, "torch") else x


def robot_xy_local(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Robot base xy relative to its terrain-tile origin."""
    p = _t(env.scene["robot"].data.root_pos_w)
    origins = _t(env.scene.env_origins)
    return p[:, :2] - origins[:, :2]


def progress_to_button(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Reduction in planar distance to the button since the previous call."""
    xy = robot_xy_local(env)
    goal = torch.tensor(BUTTON_AT[:2], device=xy.device, dtype=xy.dtype)
    dist = (xy - goal).norm(dim=-1)
    prev = getattr(env, "_bhl_maze_prev_dist", None)
    if prev is None or prev.shape != dist.shape:
        prev = dist.clone()
    env._bhl_maze_prev_dist = dist.detach().clone()
    delta = prev - dist
    steps = getattr(env, "episode_length_buf", None)
    if steps is not None:
        delta = torch.where(steps <= 1, torch.zeros_like(delta), delta)
    return delta


def in_dead_end(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Left the corridor or walked to the −x T-junction."""
    xy = robot_xy_local(env)
    return (xy[:, 1].abs() > CORRIDOR_Y_LIMIT) | (xy[:, 0] < WRONG_WAY_X)


def button_reached(
    env: "ManagerBasedRLEnv",
    radius: float = BUTTON_RADIUS,
    hold_steps: int = 8,
) -> torch.Tensor:
    """Base has stayed on the plate. Not a dexterous press — this biped has no arms."""
    xy = robot_xy_local(env)
    goal = torch.tensor(BUTTON_AT[:2], device=xy.device, dtype=xy.dtype)
    now = (xy - goal).norm(dim=-1) <= radius
    buf = getattr(env, "_bhl_maze_button_hold", None)
    if buf is None or buf.shape[0] != env.num_envs:
        buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    buf = torch.where(now, buf + 1, torch.zeros_like(buf))
    steps = getattr(env, "episode_length_buf", None)
    if steps is not None:
        buf = torch.where(steps <= 1, torch.zeros_like(buf), buf)
    env._bhl_maze_button_hold = buf
    return buf >= hold_steps


class MazeWaypointCommand(UniformVelocityCommand):
    """Body-x cruise plus heading P-control onto ``PATH``.

    The observation still sees a 3-vector ``base_velocity`` command, so the
    biped walking rewards stay the ones that already train. Only the *source*
    of that command changes: random SE(2) becomes the button route.
    """

    def __init__(self, cfg: "MazeWaypointCommandCfg", env: "ManagerBasedEnv"):
        super().__init__(cfg, env)
        self._wp_idx = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._path = torch.tensor(PATH, dtype=torch.float32, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        self._wp_idx[ids] = 0
        self._apply_waypoint(ids)

    def _update_command(self):
        self._advance_waypoints()
        self._apply_waypoint(torch.arange(self.num_envs, device=self.device))
        super()._update_command()

    def _advance_waypoints(self) -> None:
        xy = robot_xy_local(self._env)
        idx = self._wp_idx.clamp(max=self._path.shape[0] - 1)
        dist = (self._path[idx] - xy).norm(dim=-1)
        last = self._wp_idx >= (self._path.shape[0] - 1)
        advance = (dist < self.cfg.waypoint_radius) & ~last
        self._wp_idx = torch.where(advance, self._wp_idx + 1, self._wp_idx)

    def _apply_waypoint(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        xy = robot_xy_local(self._env)[env_ids]
        idx = self._wp_idx[env_ids].clamp(max=self._path.shape[0] - 1)
        delta = self._path[idx] - xy
        self.heading_target[env_ids] = torch.atan2(delta[:, 1], delta[:, 0])
        self.vel_command_b[env_ids, 0] = self.cfg.cruise_speed
        self.vel_command_b[env_ids, 1] = 0.0
        self.is_heading_env[env_ids] = True
        self.is_standing_env[env_ids] = False


@configclass
class MazeWaypointCommandCfg(UniformVelocityCommandCfg):
    """UniformVelocityCommand whose heading target is the next maze waypoint."""

    class_type: type = MazeWaypointCommand
    cruise_speed: float = CRUISE_SPEED
    waypoint_radius: float = WAYPOINT_RADIUS
