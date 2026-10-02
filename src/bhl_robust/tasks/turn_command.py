"""A velocity command that actually asks for sustained turns (arm TurnCmd).

Why this exists (2026-09-26, MuJoCo, scripts/bench/turn_test.py): from a
settled stand, most of the 22-DoF turning arms trained on upstream's command
generator sit in a standing fixed point under a sustained pure yaw-rate command
and never lift a foot (10 of 12 checkpoints; see the 2026-09-27 note below for
the two that step). Upstream's `UniformVelocityCommandCfg` with `heading_command=True`
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


# --- Rest-then-turn mix (arm TurnRest, 2026-09-27) ----------------------------
#
# MuJoCo diagnosis (scripts/bench/turn_diagnose.py mujoco, reset seeds 0-2,
# 3 s zero-command settle then (0, 0, +/-0.6) for 6 s; all 12 turning-arm
# checkpoints, results/repo-gpu-20260923/turn-20260927/diagnose/
# mujoco_{existing,hip_track}_s012.json):
#   * 10 of 12 (TurnHip s0-s2, TurnTrack s1-s2, TurnBoth s1-s2, TurnCmd s0-s2)
#     make ZERO lift-offs in all 6 runs, although their actions do respond to wz
#     (||a(cmd) - a(0)|| 1.1-2.4, against ||a|| ~ 47-109 raw; TurnBoth-s0's
#     2.0): an 8-20 deg double-stance twist, never a step.
#   * TurnTrack-s0 steps and turns in the -wz direction only (-231, -259 deg;
#     2/6 runs >= 150 deg); one of those runs starts from a true standstill
#     (0 lift-offs in the last 2 s of the settle). It never steps under +wz,
#     even from reset seed 1, where it is still stepping (7 / 4 lift-offs) at
#     the end of the settle and steps under -wz.
#   * TurnBoth-s0 is the only checkpoint that turns both ways (5/6). It is still
#     stepping in place at the end of the settle (1-9 lift-offs per foot in the
#     last 2 s). Its one failure (-0.6, reset seed 1) is on a reset where it had
#     nearly stopped (1 lift-off per foot), but the +0.6 run from the same settle
#     state stepped and turned 212 deg, so near-rest alone does not account for
#     it: direction matters too.
# Pooled over the 72 runs ("at rest" = 0 lift-offs on both feet in the last 2 s
# of the settle): the robot was at rest at the end of the settle in 50
# and took a step under the turn in 1 of them; it was still stepping in 22 and
# stepped under the turn in 6. Step INITIATION from rest under a pure-turn
# command is therefore rare (1/50), and the TurnCmd mix never asks for it
# explicitly: its pure turns start at a resample, when the robot is usually
# already moving.
#
# REST_TURN (rel_rest_turn_envs): zero command for rest_s ~ U(rest_time_range),
# then (0, 0, +/-|wz|), |wz| ~ U(pure_turn_ang_vel_abs), for the rest of the
# resample interval, no heading control -- the settled-stand test, as a training
# command. PURE_TURN / DIRECT / UPSTREAM are as in TurnMixVelocityCommand.

REST_TURN = 3


def draw_modes_rest(u: torch.Tensor, rel_pure: float, rel_rest: float, rel_direct: float) -> torch.Tensor:
    """u in [0, 1) -> [0, p) PURE_TURN, [p, p+r) REST_TURN, [p+r, p+r+d) DIRECT, rest UPSTREAM."""
    modes = torch.full(u.shape, UPSTREAM, dtype=torch.long, device=u.device)
    modes[u < rel_pure + rel_rest + rel_direct] = DIRECT
    modes[u < rel_pure + rel_rest] = REST_TURN
    modes[u < rel_pure] = PURE_TURN
    return modes


def check_rest_mix(rel_pure: float, rel_rest: float, rel_direct: float, rest_time_range) -> None:
    """Raise ValueError on a rest-mix configuration that cannot mean what it says."""
    fr = (rel_pure, rel_rest, rel_direct)
    if not (all(0.0 <= f <= 1.0 for f in fr) and sum(fr) <= 1.0 + 1e-9):
        raise ValueError(f"mode fractions must be in [0,1] and sum <= 1: pure={rel_pure} rest={rel_rest} direct={rel_direct}")
    lo, hi = rest_time_range
    if not (0.0 <= lo <= hi):
        raise ValueError(f"rest_time_range must be 0 <= lo <= hi, got {rest_time_range}")


def rest_turn_update(vel: torch.Tensor, modes: torch.Tensor, rest_left: torch.Tensor, rest_wz: torch.Tensor,
                     standing: torch.Tensor) -> torch.Tensor:
    """Per-step command of REST_TURN rows (returns a new tensor; inputs untouched).

    REST_TURN rows with rest_left > 0 are (0, 0, 0); REST_TURN rows with
    rest_left <= 0 are (0, 0, rest_wz). Standing rows are zero whatever their
    mode (upstream's rule). Other rows are returned unchanged."""
    vel = vel.clone()
    rest = modes == REST_TURN
    resting = rest & (rest_left > 0.0)
    turning = rest & ~(rest_left > 0.0)
    vel[resting] = 0.0
    vel[turning, 0:2] = 0.0
    vel[turning, 2] = rest_wz[turning].to(vel.dtype)
    vel[standing] = 0.0
    return vel


class TurnRestMixVelocityCommand(TurnMixVelocityCommand):
    """TurnMixVelocityCommand plus REST_TURN envs (zero command, then a sustained pure turn)."""

    cfg: "TurnRestMixVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        check_rest_mix(cfg.rel_pure_turn_envs, cfg.rel_rest_turn_envs, cfg.rel_direct_envs, cfg.rest_time_range)
        self.rest_left = torch.zeros(self.num_envs, device=self.device)
        self.rest_wz = torch.zeros(self.num_envs, device=self.device)
        # Fraction of resetting envs whose last command was rest-then-turn (expected ~rel_rest_turn_envs).
        self.metrics["rest_turn_env"] = torch.zeros(self.num_envs, device=self.device)

    def __str__(self) -> str:
        return (super().__str__() + f"\n\tRest-then-turn probability: {self.cfg.rel_rest_turn_envs}"
                f" (rest {self.cfg.rest_time_range} s)")

    def _resample_command(self, env_ids: Sequence[int]):
        # upstream's sample first (skipping TurnMixVelocityCommand's 3-way draw), then the 4-way mix
        super(TurnMixVelocityCommand, self)._resample_command(env_ids)
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        cfg = self.cfg
        n = ids.numel()
        modes = draw_modes_rest(torch.rand(n, device=self.device), cfg.rel_pure_turn_envs, cfg.rel_rest_turn_envs,
                                cfg.rel_direct_envs)
        u = torch.rand(n, 3, device=self.device)
        vel, heading = mix_turn_commands(
            self.vel_command_b[ids], self.is_heading_env[ids], modes, u,
            pure_abs=cfg.pure_turn_ang_vel_abs, direct_x=cfg.direct_lin_vel_x, direct_y=cfg.direct_lin_vel_y,
            direct_wz=cfg.direct_ang_vel_z)
        rest = modes == REST_TURN
        sign = torch.where(u[:, 0] < 0.5, -1.0, 1.0)
        self.rest_wz[ids] = torch.where(rest, sign * _lerp(cfg.pure_turn_ang_vel_abs, u[:, 1]), 0.0)
        self.rest_left[ids] = torch.where(rest, _lerp(cfg.rest_time_range, torch.rand(n, device=self.device)), 0.0)
        vel[rest] = 0.0
        heading[rest] = False
        self.vel_command_b[ids] = vel
        self.is_heading_env[ids] = heading
        self.turn_mode[ids] = modes
        self.metrics["pure_turn_env"][ids] = (modes == PURE_TURN).float()
        self.metrics["direct_wz_env"][ids] = (modes == DIRECT).float()
        self.metrics["rest_turn_env"][ids] = rest.float()

    def _update_command(self):
        super()._update_command()                        # upstream: heading envs, then standing envs zeroed
        rest = self.turn_mode == REST_TURN
        self.rest_left[rest] -= self._env.step_dt
        self.vel_command_b[:] = rest_turn_update(self.vel_command_b, self.turn_mode, self.rest_left, self.rest_wz,
                                                 self.is_standing_env)


@configclass
class TurnRestMixVelocityCommandCfg(TurnMixVelocityCommandCfg):
    """TurnMixVelocityCommandCfg plus REST_TURN envs. Defaults: 15 % pure, 25 % rest-then-turn,
    20 % direct, 40 % upstream."""

    class_type: type = TurnRestMixVelocityCommand

    rel_pure_turn_envs: float = 0.15
    rel_direct_envs: float = 0.20

    rel_rest_turn_envs: float = 0.25
    """Probability, per resample, of zero command for rest_s ~ U(rest_time_range), then a sustained pure turn."""

    rest_time_range: tuple[float, float] = (1.5, 4.0)
    """Seconds of zero command before the turn (the turn test settles 3.0 s)."""


# --- turning-hold ---
# R1H (Velocity-BHL-Arms-TurnGaitClockHold-v0, 2026-10-02; SLURM_JOBS.md "(C') revised design", frozen before any
# implementation; launcher slurm/repo20260923/gpu_turngait_hold.sbatch). Additive: nothing above is changed, and
# no existing task uses these names.
#
# R1 trains in upstream's heading mode in EVERY env (heading_command=True, rel_heading_envs=1.0): its wz is
# clip(0.5 * wrap(heading_target - yaw), +/-1.5), recomputed every step, so the open-loop wz = 0 of the gate's
# straight walk never occurs in training. TurnHoldMixVelocityCommand is TurnMixVelocityCommand's DIRECT mechanism,
# unchanged, configured by the task as
#   rel_pure_turn_envs = 0.0   no PURE_TURN envs (pure_turn_ang_vel_abs is then never read)
#   rel_direct_envs    = 0.30  per resample, each env is EXPLICIT (DIRECT mode) with probability 0.3; its
#                              is_heading_env is cleared, so its sampled command survives until its next resample
#   direct ranges      = the task's own cfg.ranges (R1: vx (-1.0, 1.0), vy (-0.5, 0.5), wz (-1.5, 1.5))
# plus ONE addition: direct_zero_wz_prob = 0.5, an independent U[0, 1) coin per resampled env; an EXPLICIT env whose
# coin is < 0.5 gets wz = 0.0 exactly, the others keep their DIRECT draw wz ~ U(direct_ang_vel_z). The other 70 %
# keep upstream's heading mode unchanged, and the 2 % standing draw is upstream's (a standing env is zeroed whatever
# its mode).
#
# psi_ref of the heading_hold reward (gait_clock_mdp.heading_hold) is `yaw_at_resample`, written HERE, in
# _resample_command, from the command asset's heading_w. That is how the resample is detected from the command term
# itself: CommandTerm._resample(env_ids) is the only path to a new command, and it calls _resample_command for both
# kinds of resample -- the time-out resample (CommandTerm.compute, at the end of ManagerBasedRLEnv.step, after the
# reward) and the episode reset (CommandTerm.reset, called by ManagerBasedRLEnv._reset_idx AFTER the "reset" events
# have written the new root pose). So psi_ref is the base yaw at the env's last resample, and after an episode reset
# it is the yaw of the new episode's initial pose. (command_counter is not used: CommandTerm.reset zeroes it before
# the reset resample, so an episode reset can leave it at the value it had.)


def check_zero_wz_prob(p: float) -> None:
    """Raise ValueError unless 0 <= p <= 1."""
    if not 0.0 <= float(p) <= 1.0:
        raise ValueError(f"direct_zero_wz_prob must be in [0, 1], got {p}")


def zero_direct_wz(vel: torch.Tensor, modes: torch.Tensor, coin: torch.Tensor,
                   p_zero: float) -> tuple[torch.Tensor, torch.Tensor]:
    """DIRECT rows whose coin is < p_zero get wz = 0.0 exactly; vx, vy and every other row are unchanged.

    vel (n, 3), modes (n,), coin (n,) uniform [0, 1). Returns (new vel, zero mask (n,) bool); inputs untouched."""
    vel = vel.clone()
    zero = (modes == DIRECT) & (coin < p_zero)
    vel[zero, 2] = 0.0
    return vel, zero


class TurnHoldMixVelocityCommand(TurnMixVelocityCommand):
    """TurnMix's DIRECT (explicit) envs with wz = 0 exactly on a share of them, and the base yaw at each resample."""

    cfg: "TurnHoldMixVelocityCommandCfg"

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        check_zero_wz_prob(cfg.direct_zero_wz_prob)
        self.zero_wz_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.yaw_at_resample = torch.zeros(self.num_envs, device=self.device)
        self._first_resample_logged = False
        # Logged at episode end: the fraction of resetting envs whose last command was explicit with wz = 0.
        self.metrics["zero_wz_env"] = torch.zeros(self.num_envs, device=self.device)
        print(f"[turn_command] TurnHoldMixVelocityCommand: explicit {float(cfg.rel_direct_envs)} "
              f"zero_wz_p {float(cfg.direct_zero_wz_prob)} pure {float(cfg.rel_pure_turn_envs)} "
              f"vx {tuple(float(x) for x in cfg.direct_lin_vel_x)} vy {tuple(float(x) for x in cfg.direct_lin_vel_y)} "
              f"wz {tuple(float(x) for x in cfg.direct_ang_vel_z)} heading_command {bool(cfg.heading_command)} "
              f"rel_heading {float(cfg.rel_heading_envs)} stiffness {float(cfg.heading_control_stiffness)} "
              f"rel_standing {float(cfg.rel_standing_envs)}", flush=True)

    def __str__(self) -> str:
        return super().__str__() + f"\n\tExplicit-env zero-wz probability: {self.cfg.direct_zero_wz_prob}"

    @property
    def explicit_env(self) -> torch.Tensor:
        """(num_envs,) bool: the EXPLICIT envs -- DIRECT mode since their last resample, out of heading control."""
        return (self.turn_mode == DIRECT) & ~self.is_heading_env

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)              # upstream's sample, then TurnMix's DIRECT / UPSTREAM draw
        if isinstance(env_ids, slice):
            ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        else:
            ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        if ids.numel() == 0:
            return
        vel, zero = zero_direct_wz(self.vel_command_b[ids], self.turn_mode[ids],
                                   torch.rand(ids.numel(), device=self.device), self.cfg.direct_zero_wz_prob)
        self.vel_command_b[ids] = vel
        self.zero_wz_env[ids] = zero
        self.metrics["zero_wz_env"][ids] = zero.float()
        self.yaw_at_resample[ids] = self.robot.data.heading_w[ids]       # psi_ref of heading_hold
        if not self._first_resample_logged:
            self._first_resample_logged = True
            direct = self.turn_mode[ids] == DIRECT
            n_dir = int(direct.sum())
            print(f"[turn_command] TurnHoldMix first resample: n {ids.numel()} explicit {n_dir / ids.numel():.3f} "
                  f"zero_wz {float(zero.float().mean()):.3f} (of explicit "
                  f"{(int(zero.sum()) / n_dir) if n_dir else float('nan'):.3f}) "
                  f"heading_cleared_iff_explicit {bool(((~self.is_heading_env[ids]) == direct).all())} "
                  f"standing {float(self.is_standing_env[ids].float().mean()):.3f}", flush=True)


@configclass
class TurnHoldMixVelocityCommandCfg(TurnMixVelocityCommandCfg):
    """TurnMixVelocityCommandCfg with R1H's explicit-command mix (see the turning-hold note above). The task builds
    it from its own base_velocity cfg and sets the direct ranges to that cfg's ranges."""

    class_type: type = TurnHoldMixVelocityCommand

    rel_pure_turn_envs: float = 0.0
    rel_direct_envs: float = 0.30

    direct_lin_vel_x: tuple[float, float] = (-1.0, 1.0)
    direct_lin_vel_y: tuple[float, float] = (-0.5, 0.5)
    direct_ang_vel_z: tuple[float, float] = (-1.5, 1.5)

    direct_zero_wz_prob: float = 0.5
    """Probability, per resample of an EXPLICIT (DIRECT) env, that its wz is 0.0 exactly."""
# --- end turning-hold ---
