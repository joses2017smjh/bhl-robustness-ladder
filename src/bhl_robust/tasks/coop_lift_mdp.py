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

import math

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


def base_height_mean(
    env: "ManagerBasedRLEnv",
    robot_a: SceneEntityCfg = SceneEntityCfg("robot_a"),
    robot_b: SceneEntityCfg = SceneEntityCfg("robot_b"),
) -> torch.Tensor:
    """Mean base height of the pair, in metres. Diagnostic, not an objective.

    Nothing in this task has ever constrained how low the robots get. Both fall
    tests read orientation -- `either_fallen` on a tilt limit, and
    `flat_orientation_l2` on projected gravity -- so a machine that sinks onto
    its shins with a level torso is scored as upright and paid `still_alive`
    for it. Getting low is also worth a lot: it puts the hands at the height of
    a 28 cm cube, which is three reach terms and the 15.0-weight lift bonus.
    The gradient points down and nothing pushes back.

    In the MuJoCo replay both cube arms drop ~41 cm within 0.2 s, before they
    touch the payload, and hold that pose for the rest of the episode. Whether
    training does the same is unknown, because base height has never been
    recorded -- `base_contact` sits near zero, but a robot folded onto its
    shins never puts its torso down either, so that cannot separate a squat
    from a collapse.

    Registered at weight 0.0 so it is logged as `Episode_Reward/base_height`
    without entering the objective. Measuring the thing first, and deciding
    whether to penalise it second, keeps every published number comparable.
    """
    a = env.scene[robot_a.name].data.root_pos_w[:, 2]
    b = env.scene[robot_b.name].data.root_pos_w[:, 2]
    return 0.5 * (a + b)


def object_lift_progress(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """z-gain from spawn, clipped at the current lift target, gated on pinch."""
    obj: RigidObject = env.scene[object_cfg.name]
    spawn = float(env.cfg.object_spawn_z)
    target = float(getattr(env, "_bhl_lift_h", env.cfg.lift_success_z))
    progress = (obj.data.root_pos_w[:, 2] - spawn) / max(target, 1e-3)
    progress = progress.clamp(0.0, 1.0)
    if getattr(env.cfg, "gate_lift_on_pinch", True):
        progress = progress * _pinch_weight(env)
    return progress


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
    if getattr(env.cfg, "gate_lift_on_pinch", True):
        lifted = lifted * _pinch_weight(env)
    return lifted


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


def object_ang_vel_w(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    return obj.data.root_ang_vel_w


def joint_target_error(
    env: "ManagerBasedRLEnv",
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """PD tracking residual: commanded minus measured joint position.

    On the robot this is motor-current / tracking error, the usual proxy for
    contact when there are no fingers or tactile sensors. COLA / non-prehensile
    lift papers use it so the policy knows it has a clamp from resistance,
    not from a binary contact flag that will not exist on hardware.
    """
    robot: Articulation = env.scene[asset_cfg.name]
    return robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.joint_pos_target[:, asset_cfg.joint_ids]


def opposing_clamp(
    env: "ManagerBasedRLEnv",
    robot_a_cfg: SceneEntityCfg,
    robot_b_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1 when the two hand-midpoints pull on opposite sides of the object.

    Non-prehensile lift needs opposing force, not two hands on the same face.
    Gated on pinch so a far-away opposite pose does not pay.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    centre = obj.data.root_pos_w[:, :3]
    va = centre - _hand_midpoint(env, robot_a_cfg)
    vb = centre - _hand_midpoint(env, robot_b_cfg)
    va = torch.nn.functional.normalize(va, dim=-1, eps=1e-6)
    vb = torch.nn.functional.normalize(vb, dim=-1, eps=1e-6)
    opposite = (-(va * vb).sum(dim=-1)).clamp(0.0, 1.0)
    return opposite * _pinch_weight(env)


def object_tilt_l2(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Roll/pitch of the payload. Synchronous lift keeps this near zero."""
    obj: RigidObject = env.scene[object_cfg.name]
    q = obj.data.root_quat_w
    up_z = 1.0 - 2.0 * (q[:, 1] * q[:, 1] + q[:, 2] * q[:, 2])
    return torch.square(1.0 - up_z)


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
    if getattr(env.cfg, "clock_lift_height", False):
        # Wall-clock ramp: ignore competence. Same shape as the push
        # curriculum that destroyed the policy past 0.7 m/s.
        t = float(getattr(env, "common_step_counter", 0))
        frac = min(1.0, t / 72000.0)
        height = min_height + (max_height - min_height) * frac
        env._bhl_lift_h = height
        term_cfg = env.reward_manager.get_term_cfg(term_name)
        term_cfg.params["minimal_height"] = height
        env.reward_manager.set_term_cfg(term_name, term_cfg)
        return height
    if env_ids is None or len(env_ids) == 0:
        idx = slice(None)
        z = obj.data.root_pos_w[:, 2]
    else:
        idx = env_ids
        z = obj.data.root_pos_w[env_ids, 2]
    high_enough = z > (spawn + height)
    d = getattr(env, "_bhl_pinch_d", None)
    if (not getattr(env.cfg, "gate_lift_on_pinch", True)) or d is None:
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


def stage_lift_on_pinch(
    env: "ManagerBasedRLEnv",
    env_ids: Sequence[int],
    progress_weight: float = 2.0,
    bonus_weight: float = 15.0,
    enter: float = 0.40,
) -> float:
    """DexPBT staging: keep lift weights at 0 until the pinch kernel is on.

    No-op unless ``env.cfg.stage_lift_on_pinch``. Once the batch-mean pinch
    weight clears ``enter``, lift progress and the sparse bonus turn on and
    stay on. That is ``r_pick`` then ``r_targ``, not pick-only for 4,000 iters.
    """
    if not getattr(env.cfg, "stage_lift_on_pinch", False):
        return 1.0
    # ``_pinch_weight`` is 1.0 when the reach term has not run this step.
    # Curriculum can fire on reset before that, which latched staging on
    # at iteration 0 for the overnight ``staged`` arm — it became the
    # control. Missing *d* is "not yet a pinch", not "already pinched".
    if getattr(env, "_bhl_pinch_d", None) is None and not getattr(env, "_bhl_lift_staged", False):
        for name in ("lift_progress", "lifting_object"):
            cfg = env.reward_manager.get_term_cfg(name)
            cfg.weight = 0.0
            env.reward_manager.set_term_cfg(name, cfg)
        return 0.0
    rate = float(_pinch_weight(env).mean())
    staged = bool(getattr(env, "_bhl_lift_staged", False) or rate >= enter)
    env._bhl_lift_staged = staged
    for name, weight in (("lift_progress", progress_weight), ("lifting_object", bonus_weight)):
        cfg = env.reward_manager.get_term_cfg(name)
        cfg.weight = weight if staged else 0.0
        env.reward_manager.set_term_cfg(name, cfg)
    return float(staged)


# --- N-robot crew ----------------------------------------------------------
#
# The two-robot terms above hard-code a pair: two contact points at
# `centre +/- offset * axis`, and a clamp that is the dot product of two
# vectors. Neither generalises by adding arguments, so the crew versions below
# restate them on a ring. At n = 2 they are the same function: two points on a
# ring 180 degrees apart *are* `centre +/- offset * axis`, and the force-closure
# residual of two antiparallel unit vectors *is* `-v_a . v_b`.


def crew_contact_points(env: "ManagerBasedRLEnv", object_cfg: SceneEntityCfg,
                        n: int) -> torch.Tensor:
    """`n` contact points evenly spaced on a horizontal ring around the payload.

    Returns (num_envs, n, 3). Robot *i* is assigned point *i*, and the scene
    places robot *i* at the same bearing, so the assignment is the identity and
    no matching problem appears in the reward.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    centre = obj.data.root_pos_w[:, :3]
    offset = float(env.cfg.contact_offset)
    ang = torch.arange(n, device=centre.device, dtype=centre.dtype) * (2.0 * math.pi / n)
    # Bearing 0 is +x. The pair case sits at +/- 90 degrees, i.e. on +/- y, which
    # is where `contact_axis = (0, 1, 0)` put it.
    ang = ang + math.pi / 2.0
    ring = torch.stack([torch.cos(ang), torch.sin(ang), torch.zeros_like(ang)], dim=-1)
    return centre.unsqueeze(1) + offset * ring.unsqueeze(0)


def crew_reach(
    env: "ManagerBasedRLEnv",
    std: float,
    robot_cfgs: Sequence[SceneEntityCfg],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1 - tanh(mean hand-midpoint distance to each robot's own contact point)."""
    pts = crew_contact_points(env, object_cfg, len(robot_cfgs))
    d = torch.stack(
        [torch.norm(_hand_midpoint(env, c) - pts[:, i], dim=-1)
         for i, c in enumerate(robot_cfgs)],
        dim=-1,
    ).mean(dim=-1)
    # Same cache the pair terms write, so the pinch gate and the height
    # curriculum work unchanged on a crew.
    env._bhl_pinch_d = d
    return 1.0 - torch.tanh(d / std)


def crew_force_closure(
    env: "ManagerBasedRLEnv",
    robot_cfgs: Sequence[SceneEntityCfg],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """1 when the crew's inward pushes cancel, 0 when they all shove one way.

    Each robot contributes a unit vector from its hands toward the payload
    centre. If those sum to zero the payload is squeezed and not accelerated,
    which is the whole content of "opposing" once there are more than two of
    them. Gated on pinch, so standing in a tidy circle far away pays nothing.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    centre = obj.data.root_pos_w[:, :3]
    v = torch.stack(
        [torch.nn.functional.normalize(centre - _hand_midpoint(env, c), dim=-1, eps=1e-6)
         for c in robot_cfgs],
        dim=1,
    )
    residual = torch.norm(v.mean(dim=1), dim=-1)
    return (1.0 - residual).clamp(0.0, 1.0) * _pinch_weight(env)


def crew_spread(
    env: "ManagerBasedRLEnv",
    robot_cfgs: Sequence[SceneEntityCfg],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Penalty: variance of hand-to-payload distance across the crew.

    Force closure is satisfied by a crew that is balanced but loose. This is the
    term that says everyone has to be equally close, which for a non-prehensile
    lift is what stops three robots carrying while the fourth trails.
    """
    obj: RigidObject = env.scene[object_cfg.name]
    centre = obj.data.root_pos_w[:, :3]
    d = torch.stack(
        [torch.norm(centre - _hand_midpoint(env, c), dim=-1) for c in robot_cfgs], dim=-1
    )
    return d.var(dim=-1, unbiased=False)


def any_fallen(
    env: "ManagerBasedRLEnv",
    limit_angle: float,
    robot_names: Sequence[str],
) -> torch.Tensor:
    """Terminate if any crew member exceeds the loco tilt limit."""
    out = None
    for name in robot_names:
        r: Articulation = env.scene[name]
        tilt = torch.acos((-r.data.projected_gravity_b[:, 2]).clamp(-1.0, 1.0))
        hit = tilt > limit_angle
        out = hit if out is None else (out | hit)
    return out
