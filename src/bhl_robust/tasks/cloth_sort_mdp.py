"""Isaac Lab MDP terms for cloth-sort.

Observation / reward / success predicates read the same layout and catalog as
the kinematic env. The 5-D sweep action is expanded into joint-position
targets from the measured pinch pose; it does not learn 22-DoF motor skills.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

try:
    from isaaclab.envs.mdp.actions.action_term import ActionTerm
    from isaaclab.envs.mdp.actions.actions_cfg import ActionTermCfg
except ImportError:  # Isaac Lab 3.x moved some of these
    from isaaclab.managers import ActionTermCfg, ActionTerm  # type: ignore

from bhl_robust.cloth.controller import PINCH_JOINT_POS
from bhl_robust.cloth.garments import BASKET_IDS, GARMENT_BY_NAME, GarmentSpec
from bhl_robust.cloth.kinematics import relative_up_z
from bhl_robust.cloth.layout import (BASKET_X, BASKET_Y, DEFORMABLE_SUCCESS_FRACTION,
                                     TABLE_TOP_Z, basket_aabb)
from bhl_robust.cloth.sweep import ACTION_DIM, ACTION_SCALE
from bhl_robust.tasks.coop_lift_mdp import _t

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _local_pos(env: "ManagerBasedRLEnv", name: str) -> torch.Tensor:
    """Centre of the named asset, env-local. Rigid CoM or deformable nodal mean."""
    obj = env.scene[name]
    data = obj.data
    if hasattr(data, "nodal_pos_w") and data.nodal_pos_w is not None:
        try:
            return _nodal_local(env, name).mean(dim=1)
        except Exception:
            pass
    if hasattr(data, "root_pos_w") and data.root_pos_w is not None:
        p = _t(data.root_pos_w)
        if p.ndim == 1:
            p = p.unsqueeze(0)
        if p.shape[-1] >= 3:
            return p[:, :3] - env.scene.env_origins
    raise RuntimeError(f"{name} has neither nodal_pos_w nor root_pos_w")


def _nodal_local(env: "ManagerBasedRLEnv", name: str) -> torch.Tensor:
    """``(num_envs, n_verts, 3)`` vertex positions relative to the env origin."""
    obj = env.scene[name]
    nodes = _t(obj.data.nodal_pos_w)
    if nodes.ndim == 2:
        nodes = nodes.unsqueeze(0)
    return nodes[..., :3] - env.scene.env_origins.unsqueeze(1)


def _spec_for(name: str) -> GarmentSpec:
    return GARMENT_BY_NAME[name]


# ------------------------------------------------------------------ action

class SweepAction(ActionTerm):
    """Translate one 5-D sweep command into joint-position targets.

    ``process_actions`` latches a new plan. ``apply_actions`` walks the
    approach / sweep / retract phases using the same pinch-plus-offset mapping
    as ``bhl_robust.cloth.controller``, vectorised for ``num_envs``.
    """

    cfg: SweepActionCfg

    def __init__(self, cfg: SweepActionCfg, env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._raw = torch.zeros(env.num_envs, ACTION_DIM, device=env.device)
        self._proc = torch.zeros_like(self._raw)
        self._timer = torch.zeros(env.num_envs, device=env.device)
        self._invalid = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        names = list(self._asset.joint_names)
        self._name_index = {n: i for i, n in enumerate(names)}
        self._default = _t(self._asset.data.default_joint_pos).clone()
        for j, v in PINCH_JOINT_POS.items():
            if j in self._name_index:
                self._default[:, self._name_index[j]] = v

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._proc

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw[:] = actions
        self._proc[:] = actions.clamp(-1.0, 1.0)
        self._timer.zero_()
        dist = _unscale(self._proc[:, 3], *ACTION_SCALE["sweep_distance"])
        speed = _unscale(self._proc[:, 4], *ACTION_SCALE["sweep_speed"])
        self._invalid = (dist > 0.95) | (speed > 1.25)

    def apply_actions(self) -> None:
        dt = float(self._env.step_dt) if hasattr(self._env, "step_dt") else 0.02
        self._timer = self._timer + dt
        targets = self._default.clone()
        angle = self._proc[:, 2] * ACTION_SCALE["sweep_angle"]
        dist = _unscale(self._proc[:, 3], *ACTION_SCALE["sweep_distance"])
        # Progress 0→1 over a nominal 1.5 s sweep window.
        alpha = (self._timer / 1.50).clamp(0.0, 1.0)
        dy = torch.sin(angle) * dist * alpha
        progress = dist * alpha
        idx = self._name_index
        if "arm_right_shoulder_roll_joint" in idx:
            targets[:, idx["arm_right_shoulder_roll_joint"]] = 0.26 + (0.35 * dy).clamp(-0.25, 0.25)
        if "arm_right_shoulder_pitch_joint" in idx:
            targets[:, idx["arm_right_shoulder_pitch_joint"]] = 0.55 + (0.40 * progress).clamp(-0.35, 0.35)
        if "arm_right_shoulder_yaw_joint" in idx:
            targets[:, idx["arm_right_shoulder_yaw_joint"]] = (0.20 * dy).clamp(-0.30, 0.30)
        self._asset.set_joint_position_target(targets)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._timer.zero_()
            self._invalid.zero_()
        else:
            self._timer[env_ids] = 0.0
            self._invalid[env_ids] = False


@configclass
class SweepActionCfg(ActionTermCfg):
    """High-level sweep: ``[dx, dy, angle, distance, speed]`` in ``[-1, 1]``.

    ``class_type`` is a real dataclass default, not an attribute patched on
    after the fact: ``configclass`` freezes field defaults when it builds the
    dataclass, so a later ``SweepActionCfg.class_type = SweepAction`` never
    reaches an *instance* and the action manager calls ``None``. That is what
    21228029 died of. ``SweepAction`` is therefore defined above this.
    """

    class_type: type = SweepAction
    asset_name: str = "robot"


def _unscale(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    return lo + 0.5 * (x + 1.0) * (hi - lo)


# ----------------------------------------------------------- observations

def joint_pos(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _t(env.scene[asset_cfg.name].data.joint_pos)


def joint_vel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _t(env.scene[asset_cfg.name].data.joint_vel)


def base_quat(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _t(env.scene[asset_cfg.name].data.root_quat_w)


def base_ang_vel(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return _t(env.scene[asset_cfg.name].data.root_ang_vel_w)


def hand_pos(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    art: Articulation = env.scene[asset_cfg.name]
    names = art.body_names
    pos = _t(art.data.body_pos_w) - env.scene.env_origins.unsqueeze(1)
    li, ri = names.index("arm_left_hand_link"), names.index("arm_right_hand_link")
    return torch.cat([pos[:, li], pos[:, ri]], dim=-1)


def garment_xy(env: "ManagerBasedRLEnv", asset_name: str = "garment_0") -> torch.Tensor:
    return _local_pos(env, asset_name)[:, :2]


def garment_yaw_proxy(env: "ManagerBasedRLEnv", asset_name: str = "garment_0") -> torch.Tensor:
    """Planar heading from the root quaternion, privileged.

    Surface cloth has no rigid quaternion; those envs report zero yaw rather
    than crashing the observation term.
    """
    obj = env.scene[asset_name]
    data = obj.data
    if not hasattr(data, "root_quat_w") or data.root_quat_w is None:
        return torch.zeros(env.num_envs, 1, device=env.device)
    q = _t(data.root_quat_w)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)).unsqueeze(-1)


def rel_garment_basket(
    env: "ManagerBasedRLEnv",
    asset_name: str = "garment_0",
    garment: str = "shirt_a",
) -> torch.Tensor:
    spec = _spec_for(garment)
    p = _local_pos(env, asset_name)[:, :2]
    c = torch.tensor(
        [BASKET_X, BASKET_Y[spec.target_basket]],
        device=p.device, dtype=p.dtype,
    )
    return c.unsqueeze(0) - p


def last_action(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return env.action_manager.action


# ---------------------------------------------------------------- rewards

def progress_to_basket(
    env: "ManagerBasedRLEnv",
    asset_name: str = "garment_0",
    garment: str = "shirt_a",
) -> torch.Tensor:
    """Reduction in planar distance since the previous call. Privileged."""
    spec = _spec_for(garment)
    p = _local_pos(env, asset_name)[:, :2]
    c = torch.tensor(
        [BASKET_X, BASKET_Y[spec.target_basket]],
        device=p.device, dtype=p.dtype,
    )
    dist = (p - c.unsqueeze(0)).norm(dim=-1)
    prev = getattr(env, "_bhl_cloth_prev_dist", None)
    if prev is None or prev.shape != dist.shape:
        prev = dist.clone()
    env._bhl_cloth_prev_dist = dist.detach().clone()
    delta = prev - dist
    # An env that just reset teleported its garment back to spawn. The
    # distance jump across that boundary is not progress the policy made, and
    # paying for it would reward whichever spawn happens to land nearer the
    # basket. Zero the first step of each episode.
    steps = getattr(env, "episode_length_buf", None)
    if steps is not None:
        delta = torch.where(steps <= 1, torch.zeros_like(delta), delta)
    return delta


def garment_in_correct_basket(
    env: "ManagerBasedRLEnv",
    asset_name: str = "garment_0",
    garment: str = "shirt_a",
) -> torch.Tensor:
    spec = _spec_for(garment)
    return _in_basket(env, asset_name, spec.target_basket)


def garment_in_wrong_basket(
    env: "ManagerBasedRLEnv",
    asset_name: str = "garment_0",
    garment: str = "shirt_a",
) -> torch.Tensor:
    spec = _spec_for(garment)
    wrong = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for bid in BASKET_IDS:
        if bid == spec.target_basket:
            continue
        wrong = wrong | _in_basket(env, asset_name, bid)
    return wrong & ~_in_basket(env, asset_name, spec.target_basket)


def _basket_bounds(basket_id: str, device, dtype) -> tuple[torch.Tensor, torch.Tensor]:
    """Low/high corner of one basket, from the *same* AABB the kinematic env scores.

    Read off `layout.basket_aabb` rather than re-derived from `BASKET_INNER`
    here. The hand-rolled version used a half-extent for z where the layout
    uses the full inner height, so the Isaac ceiling was 0.07 m against the
    kinematic 0.14 m and the two scored the same garment differently.
    """
    box = basket_aabb(basket_id)
    lo = torch.as_tensor(box.low, device=device, dtype=dtype)
    hi = torch.as_tensor(box.high, device=device, dtype=dtype)
    return lo, hi


def _in_basket(env: "ManagerBasedRLEnv", asset_name: str, basket_id: str) -> torch.Tensor:
    p = _local_pos(env, asset_name)[:, :3]
    lo, hi = _basket_bounds(basket_id, p.device, p.dtype)
    # Each comparison is parenthesised: `&` binds tighter than `>=` in Python,
    # so without them this becomes a chained comparison and dies with
    # "bitwise_and_cuda not implemented for 'Float'" -- what 21233802 hit.
    return ((p >= lo) & (p <= hi)).all(dim=-1)


def deformable_in_correct_basket(
    env: "ManagerBasedRLEnv",
    asset_name: str = "garment_0",
    garment: str = "shirt_a",
    threshold: float = DEFORMABLE_SUCCESS_FRACTION,
) -> torch.Tensor:
    """Vertex-fraction test. Do not use the CoM test and call it cloth."""
    spec = _spec_for(garment)
    return _vertex_fraction_in_basket(env, asset_name, spec.target_basket) >= threshold


def deformable_in_wrong_basket(
    env: "ManagerBasedRLEnv",
    asset_name: str = "garment_0",
    garment: str = "shirt_a",
    threshold: float = DEFORMABLE_SUCCESS_FRACTION,
) -> torch.Tensor:
    spec = _spec_for(garment)
    wrong = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
    for bid in BASKET_IDS:
        if bid == spec.target_basket:
            continue
        wrong = wrong | (_vertex_fraction_in_basket(env, asset_name, bid) >= threshold)
    return wrong & ~deformable_in_correct_basket(env, asset_name, garment, threshold)


def _vertex_fraction_in_basket(
    env: "ManagerBasedRLEnv", asset_name: str, basket_id: str,
) -> torch.Tensor:
    pts = _nodal_local(env, asset_name)[..., :3]
    lo, hi = _basket_bounds(basket_id, pts.device, pts.dtype)
    inside = ((pts >= lo) & (pts <= hi)).all(dim=-1)
    return inside.float().mean(dim=-1)


def all_garments_sorted(
    env: "ManagerBasedRLEnv",
    asset_names: tuple[str, ...] = ("garment_0",),
    garments: tuple[str, ...] = ("shirt_a",),
) -> torch.Tensor:
    ok = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
    for name, g in zip(asset_names, garments):
        try:
            env.scene[name]
        except Exception:
            continue
        ok = ok & garment_in_correct_basket(env, name, g)
    return ok


def time_penalty(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def invalid_sweep(env: "ManagerBasedRLEnv") -> torch.Tensor:
    term = env.action_manager.get_term("sweep")
    if hasattr(term, "_invalid"):
        return term._invalid.float()
    return torch.zeros(env.num_envs, device=env.device)


def tilt_from_spawn(
    env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Angle between the robot's current up axis and the one it spawned with.

    Measured *relative to the spawn pose*, not against an absolute body-z
    convention, because this asset's identity orientation is not upright. The
    configured stand-up quaternion is ``(0, 0, 1, 0)`` -- the one the spawn
    photographs (21218517 / 21218627) show standing, matching MuJoCo to 9 mm --
    and it gives ``R[2, 2] = -1``. An absolute ``R[2,2] < 0.70`` test therefore
    calls a photographed-standing robot fallen and terminates every episode on
    step one. That is what 21233866 hit, and it is the same failure mode that
    trained nine v2 arms for 8,000 iterations on one-step episodes (21093953).

    Relative-to-spawn sidesteps the convention argument entirely: it reads 0 at
    reset by construction, whatever the asset's identity frame turns out to
    mean. A yaw jitter at reset leaves it at 0, which is correct -- yaw is not
    tilt.
    """
    robot = env.scene[asset_cfg.name]
    q = _t(robot.data.root_quat_w)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    q0 = _t(robot.data.default_root_state)[:, 3:7]
    if q0.ndim == 1:
        q0 = q0.unsqueeze(0)
    up_z = relative_up_z(q, q0)
    return torch.acos(up_z.clamp(-1.0, 1.0))


#: Same limit the locomotion and coop tasks terminate on.
FALL_LIMIT_RAD = 0.78


def fallen(env: "ManagerBasedRLEnv", asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Tilted more than ``FALL_LIMIT_RAD`` away from the spawn pose."""
    return tilt_from_spawn(env, asset_cfg) > FALL_LIMIT_RAD


# ------------------------------------------------------------ terminations

def success_correct(
    env: "ManagerBasedRLEnv",
    asset_name: str = "garment_0",
    garment: str = "shirt_a",
) -> torch.Tensor:
    return garment_in_correct_basket(env, asset_name, garment)


def success_deformable(
    env: "ManagerBasedRLEnv",
    asset_name: str = "garment_0",
    garment: str = "shirt_a",
) -> torch.Tensor:
    return deformable_in_correct_basket(env, asset_name, garment)


def success_five(
    env: "ManagerBasedRLEnv",
    asset_names: tuple[str, ...] = (
        "garment_0", "garment_1", "garment_2", "garment_3", "garment_4",
    ),
    garments: tuple[str, ...] = (
        "sock_a", "sock_b", "shirt_a", "shirt_b", "jacket",
    ),
) -> torch.Tensor:
    return all_garments_sorted(env, asset_names, garments)


def robot_fallen(env: "ManagerBasedRLEnv") -> torch.Tensor:
    return fallen(env)


def skip_event(env: "ManagerBasedRLEnv", env_ids=None, **kwargs) -> None:
    """No-op event. Deformable garments have no rigid root to ``reset_root``."""
    return
