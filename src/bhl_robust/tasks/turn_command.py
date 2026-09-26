"""A velocity command that actually asks for sustained turns (arm TurnCmd).

Why this exists (2026-09-26, MuJoCo, scripts/bench/turn_test.py): from a
settled stand, the 22-DoF turning arms trained on upstream's command generator
sit in a standing fixed point under a sustained pure yaw-rate command and never
lift a foot. Upstream's `UniformVelocityCommandCfg` with `heading_command=True`
and `rel_heading_envs=1.0` overwrites wz with `clip(0.5 * heading_error)` in
EVERY env on every step, and zeroes all three components in the 2 % standing
envs, so a sustained (0, 0, wz) command almost never occurs in training.

`TurnMixVelocityCommand` keeps upstream's generator and, at every resample,
reassigns each env to one of three modes:

  PURE_TURN  (rel_pure_turn_envs, default 0.25): vx = vy = 0 exactly,
             wz = +/- U(pure_turn_ang_vel_abs) = +/- U(0.3, 1.0) rad/s
  DIRECT     (rel_direct_envs, default 0.25): low-speed walking with a directly
             sampled yaw rate: vx ~ U(-0.5, 0.5), vy ~ U(-0.25, 0.25),
             wz ~ U(-1.0, 1.0)
  UPSTREAM   (the rest, 0.50): exactly what upstream sampled (heading-derived wz)

PURE_TURN and DIRECT envs are removed from heading control by clearing
`is_heading_env`: upstream's `_update_command` recomputes wz only for envs whose
`is_heading_env` is True, so their sampled wz survives every step until the next
resample. `_update_command` is therefore not overridden. The standing draw is
upstream's and still applies to all modes (a standing env is zeroed whatever its
mode), so the overall standing fraction stays at `rel_standing_envs` (0.02).

The command is still (vx, vy, wz) in the base frame, so the observation layout
(75 obs) and the action layout (22) are unchanged.

The mixing is factored into `draw_modes` / `mix_turn_commands`, pure torch
functions that are unit-tested without Isaac Sim.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch

from isaaclab.envs.mdp.commands.commands_cfg import UniformVelocityCommandCfg
from isaaclab.envs.mdp.commands.velocity_command import UniformVelocityCommand
from isaaclab.utils import configclass

PURE_TURN, DIRECT, UPSTREAM = 0, 1, 2


def check_mix(rel_pure: float, rel_direct: float, pure_abs, direct_x, direct_y, direct_wz) -> None:
    """Raise ValueError on a mixing configuration that cannot mean what it says."""
    if not (0.0 <= rel_pure <= 1.0 and 0.0 <= rel_direct <= 1.0 and rel_pure + rel_direct <= 1.0 + 1e-9):
        raise ValueError(f"mode fractions must be in [0,1] and sum <= 1: pure={rel_pure} direct={rel_direct}")
    lo, hi = pure_abs
    if not (0.0 < lo <= hi):
        raise ValueError(f"pure_turn_ang_vel_abs must be 0 < lo <= hi, got {pure_abs}")
    for name, (a, b) in (("direct_lin_vel_x", direct_x), ("direct_lin_vel_y", direct_y), ("direct_ang_vel_z", direct_wz)):
        if a > b:
            raise ValueError(f"{name} must be (lo, hi) with lo <= hi, got {(a, b)}")


def draw_modes(u: torch.Tensor, rel_pure: float, rel_direct: float) -> torch.Tensor:
    """Map uniform draws u in [0, 1) to modes: [0, p) PURE_TURN, [p, p+d) DIRECT, rest UPSTREAM."""
    modes = torch.full(u.shape, UPSTREAM, dtype=torch.long, device=u.device)
    modes[u < rel_pure + rel_direct] = DIRECT
    modes[u < rel_pure] = PURE_TURN
    return modes


def _lerp(bounds, u: torch.Tensor) -> torch.Tensor:
    return bounds[0] + (bounds[1] - bounds[0]) * u


def mix_turn_commands(vel: torch.Tensor, is_heading: torch.Tensor, modes: torch.Tensor, u: torch.Tensor, *,
                      pure_abs, direct_x, direct_y, direct_wz) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply the per-env modes to freshly sampled upstream commands.

    vel (n, 3) and is_heading (n,) are upstream's samples for these envs; modes (n,)
    from `draw_modes`; u (n, 3) uniform [0, 1) draws. Returns new tensors (inputs
    are not modified): PURE_TURN rows become (0, 0, +/-|wz|) with |wz| from
    `pure_abs` (sign from u[:, 0] < 0.5 -> negative, magnitude from u[:, 1]);
    DIRECT rows become (lerp(direct_x, u0), lerp(direct_y, u1), lerp(direct_wz, u2));
    both lose heading control. UPSTREAM rows are returned unchanged.
    """
    vel = vel.clone()
    is_heading = is_heading.clone()
    pure = modes == PURE_TURN
    direct = modes == DIRECT
    sign = torch.where(u[:, 0] < 0.5, -1.0, 1.0).to(vel.dtype)
    pure_wz = sign * _lerp(pure_abs, u[:, 1]).to(vel.dtype)
    zero = torch.zeros_like(vel[:, 0])
    vel[:, 0] = torch.where(pure, zero, torch.where(direct, _lerp(direct_x, u[:, 0]).to(vel.dtype), vel[:, 0]))
    vel[:, 1] = torch.where(pure, zero, torch.where(direct, _lerp(direct_y, u[:, 1]).to(vel.dtype), vel[:, 1]))
    vel[:, 2] = torch.where(pure, pure_wz, torch.where(direct, _lerp(direct_wz, u[:, 2]).to(vel.dtype), vel[:, 2]))
    is_heading[pure | direct] = False
    return vel, is_heading


class TurnMixVelocityCommand(UniformVelocityCommand):
    """Upstream's uniform/heading command with PURE_TURN and DIRECT envs mixed in."""

    cfg: "TurnMixVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        check_mix(cfg.rel_pure_turn_envs, cfg.rel_direct_envs, cfg.pure_turn_ang_vel_abs,
                  cfg.direct_lin_vel_x, cfg.direct_lin_vel_y, cfg.direct_ang_vel_z)
        self.turn_mode = torch.full((self.num_envs,), UPSTREAM, dtype=torch.long, device=self.device)
        # Logged at episode end: the fraction of resetting envs whose last command
        # was a pure turn / a direct-wz command (expected ~0.25 each by default).
        self.metrics["pure_turn_env"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["direct_wz_env"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (super().__str__() + f"\n\tPure-turn probability: {self.cfg.rel_pure_turn_envs}"
                f" (|wz| in {self.cfg.pure_turn_ang_vel_abs})\n\tDirect-wz probability: {self.cfg.rel_direct_envs}")

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        cfg = self.cfg
        modes = draw_modes(torch.rand(ids.numel(), device=self.device), cfg.rel_pure_turn_envs, cfg.rel_direct_envs)
        vel, heading = mix_turn_commands(
            self.vel_command_b[ids], self.is_heading_env[ids], modes, torch.rand(ids.numel(), 3, device=self.device),
            pure_abs=cfg.pure_turn_ang_vel_abs, direct_x=cfg.direct_lin_vel_x, direct_y=cfg.direct_lin_vel_y,
            direct_wz=cfg.direct_ang_vel_z)
        self.vel_command_b[ids] = vel
        self.is_heading_env[ids] = heading
        self.turn_mode[ids] = modes
        self.metrics["pure_turn_env"][ids] = (modes == PURE_TURN).float()
        self.metrics["direct_wz_env"][ids] = (modes == DIRECT).float()


@configclass
class TurnMixVelocityCommandCfg(UniformVelocityCommandCfg):
    """`UniformVelocityCommandCfg` plus the PURE_TURN / DIRECT mode mix (see module docstring)."""

    class_type: type = TurnMixVelocityCommand

    rel_pure_turn_envs: float = 0.25
    """Probability, per resample, that an env gets a pure yaw-rate command (vx = vy = 0)."""

    rel_direct_envs: float = 0.25
    """Probability, per resample, that an env gets low-speed walking with a directly sampled wz."""

    pure_turn_ang_vel_abs: tuple[float, float] = (0.3, 1.0)
    """|wz| range (rad/s) of pure-turn commands; the sign is +/- with equal probability."""

    direct_lin_vel_x: tuple[float, float] = (-0.5, 0.5)
    direct_lin_vel_y: tuple[float, float] = (-0.25, 0.25)
    direct_ang_vel_z: tuple[float, float] = (-1.0, 1.0)
