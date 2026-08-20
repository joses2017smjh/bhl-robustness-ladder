"""MDP terms for two 22-DoF humanoids lifting one object.

The recipe is the one that actually trains, not a collage of every paper:

* Spawn already in a pinch formation (OpenAI Dactyl / DexPBT). Walking up to
  the object is a different task and starves the lift of on-policy contact.
* Dense constellation reach, two-scale tanh (Isaac Lab Franka uses
  ``std=0.1`` because the gripper already starts next to the cube; a
  standing humanoid's hands are ~0.5 m above a floor object, which saturates
  ``std=0.15``. Coarse 0.40 keeps a gradient at spawn; fine 0.12 is the
  pinch. Same coarse/fine split as Isaac Lab's goal-tracking terms).
* Dense lift progress plus a sparse height bonus (Isaac Lab lift weights),
  both multiplied by the pinch kernel. DexPBT stages ``r_pick`` then gates
  ``r_targ`` on ``1_picked``; Isaac Lab gates goal tracking on height. The
  inverse — gate height on pinch — is what stops a toss from looking like a
  lift. Binary-only is too sparse for 6 Nm; progress-only is tossable.
* Competence-gated height, not a wall-clock ramp — the same rule as
  ``push_levels_adaptive``. A schedule that does not look at success will
  outrun a 16 kg, 6 Nm machine.
* Privileged critic (object velocity) — Pinto 2017, every Isaac Lab loco task.

One PPO controls both robots. Tightly coupled pinch is a single physical
system; independent learners spend their samples fighting each other.
"""
from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _hand_midpoint(env: "ManagerBasedRLEnv", robot_cfg: SceneEntityCfg) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    return robot.data.body_pos_w[:, robot_cfg.body_ids, :].mean(dim=1)


def _contact_points(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg) -> tuple[torch.Tensor, torch.Tensor]:
    obj: RigidObject = env.scene[object_cfg.name]
    axis = torch.tensor(env.cfg.contact_axis, device=obj.device, dtype=obj.data.root_pos_w.dtype)
    offset = float(env.cfg.contact_offset)
    centre = obj.data.root_pos_w[:, :3]
    return centre + offset * axis, centre - offset * axis


def constellation_reach(
    env: "ManagerBasedRLEnv",
    std: float,
    robot_a_cfg: SceneEntityCfg,
    robot_b_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1 - tanh(mean hand-midpoint distance to the two pinch points)."""
    c_a, c_b = _contact_points(env, object_cfg)
    d = 0.5 * (
        torch.norm(_hand_midpoint(env, robot_a_cfg) - c_a, dim=-1)
        + torch.norm(_hand_midpoint(env, robot_b_cfg) - c_b, dim=-1)
    )
    # Cached for the gated lift terms and the height curriculum. Both reach
    # kernels write the same *d*; the last one to run wins, and they agree.
    env._bhl_pinch_d = d
    return 1.0 - torch.tanh(d / std)


def _pinch_weight(env: "ManagerBasedRLEnv", std: float = 0.12) -> torch.Tensor:
    """Soft pinch in [0, 1]. 1 if the reach term has not run yet this step."""
    d = getattr(env, "_bhl_pinch_d", None)
    if d is None:
        return torch.ones(env.num_envs, device=env.device)
    return 1.0 - torch.tanh(d / std)


def still_alive(env: "ManagerBasedRLEnv") -> torch.Tensor:
    """Per-step alive bit. Standing spawn with a saturated reach kernel made
    dying in five steps the highest-reward policy; this stops that shortcut
    from beating a real pinch."""
    return (~env.termination_manager.terminated).float()


def object_lift_progress(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """z-gain from spawn, clipped at the current lift target, gated on pinch."""
    obj: RigidObject = env.scene[object_cfg.name]
    spawn = float(env.cfg.object_spawn_z)
    target = float(getattr(env, "_bhl_lift_h", env.cfg.lift_success_z))
    progress = (obj.data.root_pos_w[:, 2] - spawn) / max(target, 1e-3)
    return progress.clamp(0.0, 1.0) * _pinch_weight(env)


def object_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Sparse bonus once the object clears spawn *and* the hands are in a pinch.

    Height alone is a toss (ladder run: lift bonus 3.4, reaching 0.0, *h* at
    the 22 cm cap). Multiplying by the pinch kernel is the DexPBT / Isaac Lab
    gate, inverted: they gate carry on lift, we gate lift on pinch.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    lifted = (obj.data.root_pos_w[:, 2] > (float(env.cfg.object_spawn_z) + minimal_height)).float()
    return lifted * _pinch_weight(env)


def object_xy_drift_l2(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Keep the lift in place. Carry is a later phase; rewarding it now is a toss."""
    obj: RigidObject = env.scene[object_cfg.name]
    spawn_xy = obj.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
    return torch.sum(torch.square(obj.data.root_pos_w[:, :2] - spawn_xy), dim=-1)


def object_lin_vel_l2(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return torch.sum(torch.square(obj.data.root_lin_vel_w), dim=-1)


def object_pos_in_root(
    env: "ManagerBasedRLEnv",
    robot_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, obj.data.root_pos_w[:, :3]
    )
    return pos_b


def object_lin_vel_w(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_lin_vel_w


def either_fallen(
    env: "ManagerBasedRLEnv",
    limit_angle: float,
    robot_a_cfg: SceneEntityCfg = SceneEntityCfg("robot_a"),
    robot_b_cfg: SceneEntityCfg = SceneEntityCfg("robot_b"),
) -> torch.Tensor:
    """Terminate if either robot exceeds the loco tilt limit (0.78 rad)."""
    a: Articulation = env.scene[robot_a_cfg.name]
    b: Articulation = env.scene[robot_b_cfg.name]
    tilt_a = torch.acos((-a.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    tilt_b = torch.acos((-b.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
    return (tilt_a > limit_angle) | (tilt_b > limit_angle)


def lift_height_curriculum(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    term_name: str = "lifting_object",
    step: float = 0.02,
    min_height: float = 0.04,
    max_height: float = 0.22,
    success_rate_target: float = 0.35,
) -> float:
    """Raise the sparse-lift threshold only while the policy is clearing it.

    Same feedback rule as ``push_levels_adaptive``: promote on competence,
    demote if success collapses. The logged value is the measurement.
    """
    obj: RigidObject = env.scene["object"]
    height = getattr(env, "_bhl_lift_h", min_height)
    spawn = float(env.cfg.object_spawn_z)
    if env_ids is None or len(env_ids) == 0:
        idx = slice(None)
        z = obj.data.root_pos_w[:, 2]
    else:
        idx = env_ids
        z = obj.data.root_pos_w[env_ids, 2]
    high_enough = z > (spawn + height)
    d = getattr(env, "_bhl_pinch_d", None)
    if d is None:
        pinched = torch.ones_like(z, dtype=torch.bool)
    else:
        pinched = (d[idx] < 0.20)
    success = float((high_enough & pinched).float().mean())
    if success > success_rate_target:
        height = min(max_height, height + step)
    elif success < 0.5 * success_rate_target:
        height = max(min_height, height - step)
    env._bhl_lift_h = height

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.params["minimal_height"] = height
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return height
