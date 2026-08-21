"""Left-right mirror for BHL's observation and action, for rsl-rl's symmetry aug.

BHL is bilaterally symmetric and nothing in upstream's training exploits that.
rsl-rl 3.x will either augment each batch with its mirror image or add a mirror
loss, but it cannot know what "mirror" means for a given robot -- the mapping is
supplied here.

Why this robot in particular: a symmetric gait is more torque-efficient than an
asymmetric one, and torque is exactly what an 11.3 kg machine with 6 Nm joints
does not have. The testable claim is that symmetry augmentation buys more on a
weak robot than on a strong one, which this repo's randomization ladder is
already set up to measure.

Observation layout (45), fixed by upstream's `PolicyCfg` term order:

    0:3    velocity command      (vx, vy, wz)
    3:6    base angular velocity (wx, wy, wz)
    6:9    projected gravity     (gx, gy, gz)
    9:21   joint position        (6 left, then 6 right)
    21:33  joint velocity
    33:45  previous action

Reflection is about the sagittal plane, so a quantity flips sign iff it is odd
under y -> -y: the y component of a vector, and the roll/yaw components of a
rotation (its x and z angular components). Pitch does not flip.
"""

from __future__ import annotations

import torch

# Per-leg joint order: hip_roll, hip_yaw, hip_pitch, knee_pitch, ankle_pitch,
# ankle_roll. Roll and yaw joints rotate about axes that reverse under the
# reflection; the three pitch joints do not.
_LEG_SIGN = torch.tensor([-1.0, -1.0, 1.0, 1.0, 1.0, -1.0])
_JOINT_SIGN = torch.cat([_LEG_SIGN, _LEG_SIGN])          # 12
# Swap the two legs: left block <-> right block.
_JOINT_PERM = torch.cat([torch.arange(6, 12), torch.arange(0, 6)])

_CMD_SIGN = torch.tensor([1.0, -1.0, -1.0])              # vx, vy, wz
_ANGVEL_SIGN = torch.tensor([-1.0, 1.0, -1.0])           # wx, wy, wz
_GRAVITY_SIGN = torch.tensor([1.0, -1.0, 1.0])           # gx, gy, gz
_LINVEL_SIGN = torch.tensor([1.0, -1.0, 1.0])            # critic-only tail

OBS_DIM, ACT_DIM = 45, 12


def _mirror_joints(x: torch.Tensor) -> torch.Tensor:
    sign = _JOINT_SIGN.to(x.device, x.dtype)
    perm = _JOINT_PERM.to(x.device)
    return (x * sign)[..., perm]


def mirror_obs(obs: torch.Tensor) -> torch.Tensor:
    """Reflect a 45-dim observation about the robot's sagittal plane."""
    if obs.shape[-1] < OBS_DIM:
        raise ValueError(f"expected at least {OBS_DIM} observation dims, got {obs.shape[-1]}")
    out = obs.clone()
    out[..., 0:3] = obs[..., 0:3] * _CMD_SIGN.to(obs.device, obs.dtype)
    out[..., 3:6] = obs[..., 3:6] * _ANGVEL_SIGN.to(obs.device, obs.dtype)
    out[..., 6:9] = obs[..., 6:9] * _GRAVITY_SIGN.to(obs.device, obs.dtype)
    out[..., 9:21] = _mirror_joints(obs[..., 9:21])
    out[..., 21:33] = _mirror_joints(obs[..., 21:33])
    out[..., 33:45] = _mirror_joints(obs[..., 33:45])
    # The critic group appends base_lin_vel (vx, vy, vz) after the 45.
    if obs.shape[-1] >= OBS_DIM + 3:
        out[..., 45:48] = obs[..., 45:48] * _LINVEL_SIGN.to(obs.device, obs.dtype)
    return out


def mirror_act(act: torch.Tensor) -> torch.Tensor:
    """Reflect a 12-dim joint-position action."""
    return _mirror_joints(act)


def bhl_symmetry(env=None, obs=None, actions=None, obs_type: str = "policy"):
    """rsl-rl `data_augmentation_func` hook.

    rsl-rl calls this with either tensor set to None depending on what it is
    augmenting, and expects the original stacked with its mirror along dim 0.
    """
    obs_out = torch.cat([obs, mirror_obs(obs)], dim=0) if obs is not None else None
    act_out = torch.cat([actions, mirror_act(actions)], dim=0) if actions is not None else None
    return obs_out, act_out
