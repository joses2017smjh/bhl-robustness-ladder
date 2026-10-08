"""Gait clock and contact-schedule terms for the R1/R2 turning recipe.

docs/SOLUTIONS_2026-10-01.md section 2. The arm checkpoints step only under their
own training noise: the noise-free gates and the robot see a standing mean policy,
and nothing pays stepping at zero command (`feet_air_time` pays only in single
stance and only when ||cmd[:2]|| > 0.1). These terms pay a fixed contact schedule
at EVERY command, zero included.

  feet_gait          unitree_rl_lab G1 form and constants (period 0.8 s, offsets
                     [0.0, 0.5] for (left, right), stance = phase < 0.55, weight 0.5),
                     with the command gate removed: paying at zero command follows
                     unitree_rl_gym.
  feet_swing_height  unitree_rl_gym G1 `_reward_feet_swing_height` form: sum over feet
                     NOT in contact (||F|| > 1 N) of (z_foot - h)^2, weight -20.
  gait_clock         unitree_rl_lab `gait_phase` observation: sin / cos of 2 pi phase_left.

The phase is Isaac's own episode clock: ((episode_length_buf * step_dt) mod period)
/ period, so it restarts at 0 on every reset. The deploy side
(`bhl_robust.eval.gait_clock`) replays the same clock, reset with the runner.

Only torch is imported at module level (no isaaclab), so the tests load this file
by path, as tests/test_stand_mdp.py does for stand_mdp.py.
"""

from __future__ import annotations

import math

import torch

# ---- FROZEN (predeclared 2026-10-01, docs/SOLUTIONS_2026-10-01.md section 2; do not tune) ----
GAIT_PERIOD_S = 0.8                 # unitree_rl_lab G1 `gait` period
GAIT_OFFSETS = (0.0, 0.5)           # (left, right)
STANCE_THRESHOLD = 0.55             # stance iff (phase mod 1) < 0.55
FEET_GAIT_WEIGHT = 0.5
SWING_HEIGHT_WEIGHT = -20.0         # unitree_rl_gym G1 feet_swing_height scale
# The upstream Berkeley Humanoid Lite env defines no foot-clearance target (no
# clearance / swing-height term or parameter anywhere in its source), so h is the
# predeclared default.
SWING_HEIGHT_TARGET_M = 0.05
SWING_HEIGHT_TARGET_SOURCE = "default 0.05 m (the upstream Berkeley Humanoid Lite env defines no foot-clearance target)"
CONTACT_FORCE_THRESHOLD_N = 1.0     # unitree_rl_gym: contact = ||F_foot|| > 1 N
# z_foot is read as (approximately) the SOLE's height: the `.*_ankle_roll` link origin
# minus 0.060 m, the origin's height above the lowest point of the foot collision box
# at the default joint pose with an upright base (MuJoCo MJCF from the same URDF, box
# 0.072 x 0.22 x 0.04; measured 2026-10-01, before any training). Disclosed in the
# launcher header, with the figures found after it was frozen (2026-10-02, not used to
# change it): that pose tilts the foot, so 0.060 is a box corner, not the flat sole --
# by the URDF geometry the origin sits 0.050 m above a FLAT sole, and TurnBoth-s0
# standing in MuJoCo keeps it at a median 0.051 m (exploration seeds 100-102); Isaac
# printed 0.072 m at the first reward call after a reset (joints scaled 0.5-1.5, not
# a settled stance). So the swing target is an origin height of 0.11 m, about 6 cm of
# flat-sole clearance rather than the nominal 5 cm. The literal link-origin reading
# (foot_height_offset=0.0) is degenerate: h = 0.05 m is the origin's resting height,
# so the term would penalise lifting a foot rather than ask for clearance.
FOOT_ORIGIN_ABOVE_SOLE_M = 0.060
PUSH_VELOCITY_MPS = 0.5             # fixed push magnitude, x and y in [-0.5, 0.5] m/s, from iteration 0
# ------------------------------------------------------------------------------------------

_LOGGED: set[str] = set()


def global_phase(steps: torch.Tensor, step_dt: float, period: float) -> torch.Tensor:
    """unitree_rl_lab's `((episode_length_buf * step_dt) % period) / period`, in [0, 1)."""
    return (steps * step_dt) % period / period


def leg_phases(phase: torch.Tensor, offsets) -> torch.Tensor:
    """(N,) global phase -> (N, len(offsets)) per-leg phases, each (phase + offset) mod 1."""
    return torch.stack([(phase + float(o)) % 1.0 for o in offsets], dim=-1)


def schedule_reward(leg_phase: torch.Tensor, in_contact: torch.Tensor, threshold: float) -> torch.Tensor:
    """Sum over feet of XNOR(stance, in_contact), stance = leg_phase < threshold. (N, F) -> (N,)."""
    stance = leg_phase < threshold
    return (~(stance ^ in_contact.bool())).float().sum(dim=-1)


def swing_height_penalty(z_foot: torch.Tensor, in_contact: torch.Tensor, target_height: float) -> torch.Tensor:
    """Sum over feet NOT in contact of (z_foot - h)^2. (N, F) -> (N,). Positive; the weight is negative."""
    return torch.sum(torch.square(z_foot - target_height) * (~in_contact.bool()).float(), dim=-1)


def _episode_steps(env) -> torch.Tensor:
    # As unitree_rl_lab's gait_phase: a non-RL env has no episode buffer. ManagerBasedRLEnv
    # creates it before loading the managers, so the shape probe at load time sees zeros.
    if not hasattr(env, "episode_length_buf"):
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    return env.episode_length_buf


def gait_clock(env, period: float) -> torch.Tensor:
    """Observation term: [sin(2 pi phase_left), cos(2 pi phase_left)], phase_left = global phase (offset 0)."""
    phase = global_phase(_episode_steps(env), env.step_dt, period)
    out = torch.zeros(phase.shape[0], 2, device=phase.device)
    out[:, 0] = torch.sin(phase * math.pi * 2.0)
    out[:, 1] = torch.cos(phase * math.pi * 2.0)
    return out


def feet_gait(env, period: float, offset, threshold: float, sensor_cfg) -> torch.Tensor:
    """Contact-schedule reward, paid at every command (no command gate).

    Per foot, phase = ((episode_length_buf * step_dt) mod period) / period + offset;
    stance = (phase mod 1) < threshold; reward = sum over feet of XNOR(stance, in_contact),
    in_contact = current_contact_time > 0 (the contact sensor's 1 N threshold)."""
    sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0.0
    if in_contact.shape[1] != len(offset):
        raise ValueError(f"feet_gait: {in_contact.shape[1]} feet resolved but {len(offset)} phase offsets")
    if "feet_gait" not in _LOGGED:
        _LOGGED.add("feet_gait")
        names = [sensor.body_names[i] for i in sensor_cfg.body_ids]
        print(f"[gait_clock_mdp] feet_gait bodies {names} offsets {list(offset)} period {period} "
              f"threshold {threshold} step_dt {env.step_dt}", flush=True)
    phases = leg_phases(global_phase(_episode_steps(env), env.step_dt, period), offset)
    return schedule_reward(phases, in_contact, threshold)


def feet_swing_height(env, target_height: float, sensor_cfg, asset_cfg, foot_height_offset: float = 0.0,
                      force_threshold: float = CONTACT_FORCE_THRESHOLD_N) -> torch.Tensor:
    """unitree_rl_gym G1 feet_swing_height: sum over feet with ||F|| <= force_threshold of
    (z_foot - target_height)^2, z_foot = link-origin height - foot_height_offset (flat plane, ground z = 0).
    foot_height_offset = 0.0 is the literal unitree form."""
    sensor = env.scene.sensors[sensor_cfg.name]
    in_contact = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > force_threshold
    asset = env.scene[asset_cfg.name]
    z_origin = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if "feet_swing_height" not in _LOGGED and bool(in_contact.any()):
        _LOGGED.add("feet_swing_height")
        names = [asset.body_names[i] for i in asset_cfg.body_ids]
        print(f"[gait_clock_mdp] feet_swing_height bodies {names}: link-origin z of feet in contact "
              f"median {float(z_origin[in_contact].median()):.4f} m (n={int(in_contact.sum())}); "
              f"foot_height_offset {foot_height_offset} m, target {target_height} m", flush=True)
    return swing_height_penalty(z_origin - foot_height_offset, in_contact, target_height)


# --- turning-hold ---
# R1H heading_hold (Velocity-BHL-Arms-TurnGaitClockHold-v0; 2026-10-02, SLURM_JOBS.md "(C') revised design"; FROZEN,
# chosen, not tuned). Additive: R1/R2's terms above are unchanged.
#   Active only in the command term's EXPLICIT envs (turn_command.TurnHoldMixVelocityCommand.explicit_env: DIRECT
#   mode, out of heading control) while |wz_cmd| < 0.05 rad/s, and zero everywhere else:
#     reward = exp(-(dpsi / 0.2 rad)^2),  dpsi = wrap(base yaw - psi_ref),  weight 1.0,
#   psi_ref = the env's base yaw at its last command resample (and at episode reset), recorded by the command term
#   itself (TurnHoldMixVelocityCommand.yaw_at_resample, written in its _resample_command; see turn_command.py), and
#   base yaw = the same asset's heading_w (the command term's own robot), read at reward time.
#   Scale: R1's track_ang_vel_z_exp has weight 2.0 and std 0.25 rad/s (exp(-(wz_cmd - wz)^2 / 0.25^2)), so this
#   term's ceiling (1.0 per step, times step_dt like every term) is HALF the yaw-rate term's; it is active in about
#   16 % of envs: 0.30 explicit x P(|wz_cmd| < 0.05 | explicit) = 0.30 x (1 - 0.98 x 0.5 x (1 - 0.1 / 3.0)) = 0.158.
#   Only torch is used (no isaaclab import), like the terms above.
HOLD_STD_RAD = 0.2
HOLD_WZ_THRESHOLD = 0.05            # rad/s
HOLD_WEIGHT = 1.0
HOLD_COMMAND_NAME = "base_velocity"


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """isaaclab.utils.math.wrap_to_pi, verbatim logic: angles wrapped to (-pi, pi]."""
    wrapped = (angles + torch.pi) % (2 * torch.pi)
    return torch.where((wrapped == 0) & (angles > 0), torch.pi, wrapped - torch.pi)


def heading_hold_reward(yaw: torch.Tensor, psi_ref: torch.Tensor, wz_cmd: torch.Tensor, explicit: torch.Tensor,
                        std: float, wz_threshold: float) -> torch.Tensor:
    """(N,) exp(-(wrap(yaw - psi_ref) / std)^2) where explicit and |wz_cmd| < wz_threshold, else 0."""
    active = explicit.bool() & (wz_cmd.abs() < wz_threshold)
    dpsi = wrap_to_pi(yaw - psi_ref)
    return torch.where(active, torch.exp(-torch.square(dpsi / std)), torch.zeros_like(dpsi))


def heading_hold(env, command_name: str, std: float, wz_threshold: float) -> torch.Tensor:
    """Reward term: heading_hold_reward on the command term's explicit envs, its psi_ref and its asset's yaw.
    The command term must be a TurnHoldMixVelocityCommand (explicit_env, yaw_at_resample, vel_command_b, robot)."""
    term = env.command_manager.get_term(command_name)
    missing = [a for a in ("explicit_env", "yaw_at_resample", "vel_command_b", "robot") if not hasattr(term, a)]
    if missing:
        raise TypeError(f"heading_hold: command term {command_name!r} ({type(term).__name__}) lacks {missing}; "
                        f"it needs turn_command.TurnHoldMixVelocityCommand")
    explicit = term.explicit_env
    wz_cmd = term.vel_command_b[:, 2]
    reward = heading_hold_reward(term.robot.data.heading_w, term.yaw_at_resample, wz_cmd, explicit, std, wz_threshold)
    if "heading_hold" not in _LOGGED:
        _LOGGED.add("heading_hold")
        active = explicit & (wz_cmd.abs() < wz_threshold)
        print(f"[gait_clock_mdp] heading_hold command {command_name} ({type(term).__name__}) std {std} "
              f"wz_threshold {wz_threshold}: active in {int(active.sum())} of {int(wz_cmd.shape[0])} envs "
              f"(explicit {int(explicit.sum())})", flush=True)
    return reward
# --- end turning-hold ---
