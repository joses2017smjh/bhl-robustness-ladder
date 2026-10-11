"""Waiter phase 1e (2026-10-10; SLURM_JOBS.md "Predeclared now ... Waiter phase 1e"): stand still under arm loads.

Why. Phase 2 measured the selected WBC s7 standing with WaiterEnv's deploy-time stance hold (gait clock held at
phase 0.5): it leans back 8-12 deg as both arms reach forward (its base origin is at the floor, so the shoulders move
~15 cm back) and turns 6-12 deg clockwise whenever an arm is raised. Upstream's only posture term,
flat_orientation_l2 (weight 1.0), charges a 10 deg lean |g_xy|^2 = 0.03 per step. Phase 1d trained the held stance
phase but kept the +-0.5 m/s pushes and the hand forces on standing envs; its checkpoints shuffled instead of
standing (Q3 3-4/12 against s7's 9/12). Phase 2 therefore runs with a stance lock (a simulation fixture).

What (1e = 1d with ONLY these changes):
  * standing envs (the base-velocity command's `is_standing_env`) get no pushes and no hand forces (their payloads
    and arm goals keep moving): `push_moving_only`, `hand_forces_moving_only`;
  * two standing-only penalties: `StandAnchor` (base displacement and yaw change from where the env started
    standing, normalised by ANCHOR_POS_M and ANCHOR_YAW_DEG, each squared and clamped at ANCHOR_CLAMP) and
    `stand_tilt` (base tilt |g_xy|^2 normalised by sin^2(TILT_DEG), clamped at TILT_CLAMP).
The pure helpers (`anchor_error`, `tilt_error`, `moving_ids`) take tensors or arrays and are tested without Isaac Lab.

Phase 1f (2026-10-11; SLURM_JOBS.md "Predeclared now ... Waiter phase 1f"). All three 1e seeds fell within about a
second of standing in MuJoCo: with the clock held they could not recover from the start-of-episode transient, which
s7 and the 1d seeds (pushed while standing in training) ride out. 1f keeps 1e's two penalties and gives the standing
envs their disturbances back: `push_and_reanchor` pushes every env as phases 1c/1d did, and tells `StandAnchor` to
take a new anchor REANCHOR_DELAY_S after a push. Without that, a pushed robot (it cannot observe how far it was
moved) would sit on the clamped part of the anchor penalty for the rest of the episode and the penalty would stop
rewarding standing still. Hand forces stay on every env (waiter_wbc_mdp.hand_forces, unchanged).

OUTCOME (2026-10-11): phases 1e and 1f are both NEGATIVE, and the transient reading above was wrong. Every 1e and 1f
checkpoint falls within about a second of being told to stand, from a settled stance too, and in training the share
of episodes reaching the time limit dropped from 0.56-0.61 (1c, 1d) to 0.28. The likely cause is these two
penalties themselves: at their clamps they cost 22.5 per second against at most about +4.5 per second of reward for
standing, while falling costs 0.4 once (termination_penalty, weight -10, times the 0.04 s step). The policy cannot
observe its distance from the anchor, so ending the episode is the better deal whenever it has drifted. Do not reuse
`StandAnchor` / `stand_tilt` with negative weights like these; a standing term must not be able to make falling the
cheaper option (bounded positive terms, or a termination cost that dominates).
"""

from __future__ import annotations

import math

ANCHOR_POS_M = 0.05
ANCHOR_YAW_DEG = 5.0
ANCHOR_CLAMP = 9.0
TILT_DEG = 5.0
TILT_CLAMP = 9.0
REANCHOR_DELAY_S = 1.0                 # phase 1f: a pushed standing env takes a new anchor this long after the push
REANCHOR_ATTR = "_waiter_stand_reanchor_step"   # per-env step (env.common_step_counter) at which to re-anchor; -1 = none


def _clamp(x, hi):
    if hasattr(x, "clamp"):                     # torch
        return x.clamp(max=hi)
    import numpy as np
    return np.minimum(x, hi)


def anchor_error(dxy, dyaw, pos_m: float = ANCHOR_POS_M, yaw_deg: float = ANCHOR_YAW_DEG,
                 clamp: float = ANCHOR_CLAMP):
    """(|dxy| / pos_m)^2 + (dyaw / yaw_deg)^2, each clamped at `clamp`. dxy (..., 2), dyaw (...) in rad."""
    pos = (dxy ** 2).sum(-1) / pos_m ** 2
    yaw = (dyaw / math.radians(yaw_deg)) ** 2
    return _clamp(pos, clamp) + _clamp(yaw, clamp)


def tilt_error(g_xy, deg: float = TILT_DEG, clamp: float = TILT_CLAMP):
    """|g_xy|^2 / sin^2(deg), clamped: 1 at a `deg` tilt (g = projected gravity, unit length)."""
    return _clamp((g_xy ** 2).sum(-1) / math.sin(math.radians(deg)) ** 2, clamp)


def moving_ids(env_ids, standing):
    """The env ids among `env_ids` whose `standing` flag is False."""
    return env_ids[~standing[env_ids]]


def wrap_to_pi(a):
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def reanchor_mask(due, now: int, standing):
    """Envs that take a new anchor now: standing, with a pending re-anchor step that has been reached."""
    return standing & (due >= 0) & (due <= now)


def new_anchor_mask(standing, prev, due=None, now: int = 0):
    """Envs whose anchor is (re)set at this step: they just started standing, or (phase 1f) a re-anchor scheduled by a
    push is due. A due entry that fires is cleared in place (-1)."""
    new = standing & ~prev
    if due is not None:
        again = reanchor_mask(due, now, standing)
        due[again] = -1
        new = new | again
    return new


# ---------------------------------------------------------------------------------------------- Isaac Lab terms

def standing_mask(env, command_name: str = "base_velocity"):
    return env.command_manager.get_term(command_name).is_standing_env.bool()


def push_moving_only(env, env_ids, velocity_range: dict, command_name: str = "base_velocity", asset_cfg=None):
    """Upstream push_by_setting_velocity on the envs that are not standing."""
    import torch
    from isaaclab.envs import mdp as il_mdp
    from isaaclab.managers import SceneEntityCfg
    asset_cfg = asset_cfg or SceneEntityCfg("robot")
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    ids = moving_ids(env_ids, standing_mask(env, command_name))
    if len(ids) > 0:
        il_mdp.push_by_setting_velocity(env, ids, velocity_range, asset_cfg)


def push_and_reanchor(env, env_ids, velocity_range: dict, delay_s: float = REANCHOR_DELAY_S, asset_cfg=None):
    """Phase 1f: upstream push_by_setting_velocity on every env in `env_ids` (standing ones included), plus the step at
    which StandAnchor gives each pushed env a new anchor."""
    import torch
    from isaaclab.envs import mdp as il_mdp
    from isaaclab.managers import SceneEntityCfg
    asset_cfg = asset_cfg or SceneEntityCfg("robot")
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    il_mdp.push_by_setting_velocity(env, env_ids, velocity_range, asset_cfg)
    due = getattr(env, REANCHOR_ATTR, None)
    if due is None:
        due = torch.full((env.scene.num_envs,), -1, dtype=torch.long, device=env.device)
        setattr(env, REANCHOR_ATTR, due)
    due[env_ids] = int(env.common_step_counter) + max(1, int(round(delay_s / env.step_dt)))


def hand_forces_moving_only(env, env_ids, force_range: tuple[float, float], p_zero: float, asset_cfg=None,
                            command_name: str = "base_velocity"):
    """waiter_wbc_mdp.hand_forces on the envs that are not standing; standing envs' hand forces are set to zero."""
    import torch
    from bhl_robust.tasks import waiter_wbc_mdp
    asset = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)
    standing = standing_mask(env, command_name)
    moving = moving_ids(env_ids, standing)
    still = env_ids[standing[env_ids]]
    if len(moving) > 0:
        waiter_wbc_mdp.hand_forces(env, moving, force_range, p_zero, asset_cfg)
    if len(still) > 0:
        zeros = torch.zeros(len(still), len(asset_cfg.body_ids), 3, device=asset.device)
        asset.permanent_wrench_composer.set_forces_and_torques(
            forces=zeros, torques=torch.zeros_like(zeros), body_ids=asset_cfg.body_ids, env_ids=still)


def stand_tilt(env, command_name: str = "base_velocity", deg: float = TILT_DEG, clamp: float = TILT_CLAMP,
               asset_cfg=None):
    """Standing envs only: tilt_error of the base's projected gravity (penalty: use a negative weight)."""
    from isaaclab.managers import SceneEntityCfg
    asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
    return tilt_error(asset.data.projected_gravity_b[:, :2], deg, clamp) * standing_mask(env, command_name).float()


import sys as _sys

# Inside the simulation app `isaaclab` is already imported (waiter_env_cfg imports it before this module). Outside it
# (CPU tests) importing isaaclab would try to boot the Isaac Sim kernel and exit, so the class falls back to `object`
# there and is never instantiated.
if "isaaclab" in _sys.modules:
    from isaaclab.managers import ManagerTermBase as _TermBase
else:  # pragma: no cover
    _TermBase = object


class StandAnchor(_TermBase):
    """Standing envs only: anchor_error of the base's displacement and yaw change since the env started standing (the
    anchor is reset when an env resets or turns from moving to standing, and, in phase 1f, REANCHOR_DELAY_S after a
    push from `push_and_reanchor`). A penalty: use a negative weight."""

    def __init__(self, cfg, env):
        import torch
        super().__init__(cfg, env)
        n = env.num_envs
        self.anchor_xy = torch.zeros(n, 2, device=env.device)
        self.anchor_yaw = torch.zeros(n, device=env.device)
        self.prev = torch.zeros(n, dtype=torch.bool, device=env.device)

    def reset(self, env_ids=None):
        if env_ids is None:
            self.prev[:] = False
        else:
            self.prev[env_ids] = False

    def __call__(self, env, command_name: str = "base_velocity", pos_m: float = ANCHOR_POS_M,
                 yaw_deg: float = ANCHOR_YAW_DEG, clamp: float = ANCHOR_CLAMP, asset_cfg=None):
        import isaaclab.utils.math as math_utils
        from isaaclab.managers import SceneEntityCfg
        asset = env.scene[(asset_cfg or SceneEntityCfg("robot")).name]
        standing = standing_mask(env, command_name)
        xy = asset.data.root_pos_w[:, :2]
        yaw = math_utils.euler_xyz_from_quat(asset.data.root_quat_w)[2]
        new = new_anchor_mask(standing, self.prev, getattr(env, REANCHOR_ATTR, None),    # set only by phase 1f's push
                              int(env.common_step_counter))
        self.anchor_xy[new] = xy[new]
        self.anchor_yaw[new] = yaw[new]
        self.prev = standing.clone()
        err = anchor_error(xy - self.anchor_xy, wrap_to_pi(yaw - self.anchor_yaw), pos_m, yaw_deg, clamp)
        return err * standing.float()
