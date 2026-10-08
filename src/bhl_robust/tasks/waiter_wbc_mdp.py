"""Waiter phase 1 (docs/WAITER_PROGRAM.md): the upper-body command and the hand-load events of the WBC task.

The policy acts on the 12 leg joints only. The 10 arm joints and 2 grippers are PD-driven to `UpperBodyCommand`'s
current target, which this term writes to the articulation every env step; at deploy time the same 12 targets come
from a teleoperator, the phase 2 expert or the phase 3 VLA. The policy observes the current target (arms in rad,
grippers as a 0-1 closure), so it can anticipate what the arms are about to do.

Training distribution (frozen 2026-10-05): a new goal every `resampling_time_range` (1.0-3.0 s), reached by linear
interpolation over `interp_time_range` (0.4-1.2 s). Goal kinds: the default pose (p_default 0.25); each arm joint
uniform in the middle `range_fraction` (70%) of its range (p_uniform 0.5); the same left arm with the right arm
mirroring it (the rest, 0.25). Each gripper goal is closed with probability `p_gripper_closed` (0.5).
"""

from __future__ import annotations

from dataclasses import MISSING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.utils import configclass

from bhl_robust import waiter_asset as W

N_ARM = 10
N_UPPER = 12


def arm_limits_tensor(device) -> tuple[torch.Tensor, torch.Tensor]:
    lo = torch.tensor([W.ARM_LIMITS[j][0] for j in W.ARM_JOINTS], device=device)
    hi = torch.tensor([W.ARM_LIMITS[j][1] for j in W.ARM_JOINTS], device=device)
    return lo, hi


def sample_goals(n: int, device, p_default: float, p_uniform: float, range_fraction: float,
                 p_gripper_closed: float, generator: torch.Generator | None = None) -> torch.Tensor:
    """n goals (n, 12): arms in rad, grippers in rad (0 open, GRIPPER_CLOSED_RAD closed). Pure, so it is testable."""
    lo, hi = arm_limits_tensor(device)
    mid, half = 0.5 * (lo + hi), 0.5 * (hi - lo) * range_fraction
    u = torch.rand(n, 1, device=device, generator=generator)
    arms = mid + (2.0 * torch.rand(n, N_ARM, device=device, generator=generator) - 1.0) * half
    mirrored = arms.clone()
    mirrored[:, 5:] = torch.clamp(-arms[:, :5], lo[5:], hi[5:])     # right = mirror(left): every limit pair is negated
    arms = torch.where(u < p_default, torch.zeros_like(arms),
                       torch.where(u < p_default + p_uniform, arms, mirrored))
    closed = torch.rand(n, 2, device=device, generator=generator) < p_gripper_closed
    grippers = closed.float() * W.GRIPPER_CLOSED_RAD
    return torch.cat([arms, grippers], dim=1)


class UpperBodyCommand(CommandTerm):
    """Smooth random upper-body targets in training; writes PD targets for the 12 upper-body joints each step."""

    cfg: "UpperBodyCommandCfg"

    def __init__(self, cfg: "UpperBodyCommandCfg", env):
        super().__init__(cfg, env)
        self.robot = env.scene[cfg.asset_name]
        self.joint_ids, names = self.robot.find_joints(list(cfg.joint_names), preserve_order=True)
        if list(names) != list(W.UPPER_BODY_JOINTS):
            raise ValueError(f"upper-body joints resolved to {names}, expected {W.UPPER_BODY_JOINTS}")
        z = torch.zeros(self.num_envs, N_UPPER, device=self.device)
        self.start, self.goal, self.cur = z.clone(), z.clone(), z.clone()
        self.t = torch.zeros(self.num_envs, device=self.device)
        self.duration = torch.ones(self.num_envs, device=self.device)
        self.metrics["arm_track_err"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["gripper_track_err"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        enc = self.cur.clone()
        enc[:, N_ARM:] = enc[:, N_ARM:] / W.GRIPPER_CLOSED_RAD
        return enc

    def _update_metrics(self):
        err = (self.robot.data.joint_pos[:, self.joint_ids] - self.cur).abs()
        self.metrics["arm_track_err"] = err[:, :N_ARM].mean(dim=-1)
        self.metrics["gripper_track_err"] = err[:, N_ARM:].mean(dim=-1)

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        self.start[env_ids] = self.cur[env_ids]
        goals = sample_goals(len(env_ids), self.device, self.cfg.p_default, self.cfg.p_uniform,
                             self.cfg.range_fraction, self.cfg.p_gripper_closed)
        if self.cfg.arm_scale != 1.0:
            # phase 1c's disturbance curriculum (`disturbance_curriculum`) shrinks the arm goals toward the default
            # pose (0 rad); grippers are unchanged. At the default 1.0 this branch never runs (phase 1 / 1b unchanged).
            goals[:, :N_ARM] = goals[:, :N_ARM] * self.cfg.arm_scale
        self.goal[env_ids] = goals
        self.t[env_ids] = 0.0
        lo, hi = self.cfg.interp_time_range
        self.duration[env_ids] = math_utils.sample_uniform(lo, hi, (len(env_ids),), self.device)

    def _update_command(self):
        self.t += self._env.step_dt
        a = torch.clamp(self.t / self.duration, 0.0, 1.0).unsqueeze(-1)
        self.cur = self.start + (self.goal - self.start) * a
        self.robot.set_joint_position_target(self.cur, joint_ids=self.joint_ids)

    def reset(self, env_ids=None):
        ids = slice(None) if env_ids is None else env_ids
        self.cur[ids] = 0.0                      # every episode starts from the default pose, grippers open
        out = super().reset(env_ids)              # resamples: start = cur = default
        tgt = self.cur if env_ids is None else self.cur[env_ids]
        self.robot.set_joint_position_target(tgt, joint_ids=self.joint_ids, env_ids=env_ids)
        return out


@configclass
class UpperBodyCommandCfg(CommandTermCfg):
    class_type: type = UpperBodyCommand
    asset_name: str = "robot"
    joint_names: tuple = tuple(W.UPPER_BODY_JOINTS)
    resampling_time_range: tuple[float, float] = (1.0, 3.0)
    interp_time_range: tuple[float, float] = (0.4, 1.2)
    p_default: float = 0.25
    p_uniform: float = 0.5
    range_fraction: float = 0.7
    p_gripper_closed: float = 0.5
    arm_scale: float = 1.0          # phase 1c: set by `disturbance_curriculum`; 1.0 = phase 1 / 1b as frozen
    debug_vis: bool = False


def hand_forces(env, env_ids: torch.Tensor, force_range: tuple[float, float], p_zero: float,
                asset_cfg: SceneEntityCfg = MISSING):
    """apply_external_force_torque on the hand links, with each hand's force zero with probability p_zero."""
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    nb = len(asset_cfg.body_ids)
    forces = math_utils.sample_uniform(*force_range, (len(env_ids), nb, 3), asset.device)
    keep = (torch.rand(len(env_ids), nb, 1, device=asset.device) >= p_zero).float()
    forces = forces * keep
    asset.permanent_wrench_composer.set_forces_and_torques(
        forces=forces, torques=torch.zeros_like(forces), body_ids=asset_cfg.body_ids, env_ids=env_ids)


# ---- phase 1c (2026-10-07): the disturbance curriculum ------------------------------------------------------------
# Phase 1 / 1b switched the arm motion, hand payloads and hand forces on from iteration 0. None of the six resulting
# checkpoints walks (6 s at 0.35 m/s covers 0.07-0.41 m; clock-s2 1.87-2.02 m), and `track_lin_vel_xy_exp` plateaued
# from about iteration 3000 while episodes grew longer: the policy learned to survive the disturbances instead of
# tracking speed. `curricula/push.py` documents the same failure for pushes. Phase 1c holds all three at zero while a
# gait forms, then ramps them to the frozen phase 1 ranges, on training progress (env.common_step_counter).


def disturbance_scale(iteration: float, start_iter: int, end_iter: int) -> float:
    """0 before start_iter, 1 from end_iter, linear in between. Pure, so it is testable."""
    if end_iter <= start_iter:
        raise ValueError(f"end_iter {end_iter} must be > start_iter {start_iter}")
    if iteration <= start_iter:
        return 0.0
    if iteration >= end_iter:
        return 1.0
    return float(iteration - start_iter) / float(end_iter - start_iter)


def scaled_range(full: tuple[float, float], s: float) -> tuple[float, float]:
    return (float(full[0]) * s, float(full[1]) * s)


def disturbance_curriculum(env, env_ids, start_iter: int, end_iter: int, steps_per_iter: int,
                           force_range: tuple[float, float], payload_range: tuple[float, float],
                           command_name: str = "upper_body", force_term: str = "hand_forces",
                           payload_term: str = "hand_payload") -> float:
    """Scale the upper-body arm goals, the hand-force range and the hand-payload range by disturbance_scale.

    The iteration is env.common_step_counter // steps_per_iter (the agent's num_steps_per_env; the launcher checks the
    run's params/agent.yaml). Returns the scale, which the CurriculumManager logs as Curriculum/<term>: the log is the
    evidence that the ramp moved.
    """
    s = disturbance_scale(env.common_step_counter // int(steps_per_iter), start_iter, end_iter)
    env.command_manager.get_term(command_name).cfg.arm_scale = s
    for name, full, key in ((force_term, force_range, "force_range"),
                            (payload_term, payload_range, "mass_distribution_params")):
        term_cfg = env.event_manager.get_term_cfg(name)
        term_cfg.params[key] = scaled_range(full, s)
        env.event_manager.set_term_cfg(name, term_cfg)
    return s
