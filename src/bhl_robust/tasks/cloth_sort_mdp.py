"""Isaac Lab MDP terms for cloth-sort.

Observation / reward / success predicates read the same layout and catalog as
the kinematic env. The 5-D sweep action is expanded into joint-position
targets from the measured pinch pose; it does not learn 22-DoF motor skills.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
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
from bhl_robust.cloth.kinematics import relative_up_z, yaw_atan2_args
from bhl_robust.cloth.layout import (DEFORMABLE_SUCCESS_FRACTION, TABLE_TOP_Z, basket_aabb,
                                     basket_center)
from bhl_robust.cloth.sweep import ACTION_DIM
from bhl_robust.quat_order import quat_order
from bhl_robust.tasks.coop_lift_mdp import _t

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


#: How this Isaac Lab stores quaternions, probed once (bhl_robust.quat_order).
#: Isaac Lab 3.0 stores (x, y, z, w) and 2.x stored (w, x, y, z); every
#: orientation read in this module used to assume the latter, so on 3.0 the fall
#: test counted yaw as tilt and could not see a roll.
QUAT_ORDER = quat_order()


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
    """One 5-D sweep command per RL step, executed as a contact-table schedule.

    ``process_actions`` turns each env's command into a joint schedule for the
    whole primitive (``bhl_robust.cloth.schedule``) -- approach, glide, sweep,
    glide, lift -- from the garment's current position, expressed in the layout
    frame through the robot's *actual* root pose so reset jitter does not shift
    the reach cells. ``apply_actions`` plays one row per physics substep. The env
    runs ``schedule.MACRO_STEP_S`` of physics per RL step.

    It replaces a term that latched a hand-tuned 3-joint mapping every 40 ms,
    never read the garment position, and advanced its timer by the env step
    inside the substep loop.
    """

    cfg: "SweepActionCfg"

    def __init__(self, cfg: "SweepActionCfg", env: "ManagerBasedRLEnv"):
        super().__init__(cfg, env)
        from bhl_robust.cloth.reach import load_contact
        from bhl_robust.cloth.schedule import hold_command, pinch_arm

        self._asset: Articulation = env.scene[cfg.asset_name]
        n = env.num_envs
        self._raw = torch.zeros(n, ACTION_DIM, device=env.device)
        self._proc = torch.zeros_like(self._raw)
        self._invalid = torch.zeros(n, dtype=torch.bool, device=env.device)
        names = list(self._asset.joint_names)
        self._default = _t(self._asset.data.default_joint_pos).clone()
        for j, v in PINCH_JOINT_POS.items():
            if j in names:
                self._default[:, names.index(j)] = v
        table = load_contact()
        self._table = table
        self._arm_idx = torch.tensor([names.index(j) for j in table.joints], device=env.device)
        from bhl_robust.cloth.schedule import MACRO_STEP_S

        self._dt = float(env.cfg.sim.dt)
        self._n_sub = int(round(MACRO_STEP_S / self._dt))
        # latch=False: one RL step is one sweep, so the env must run exactly one
        # schedule per step. latch=True (clips): short RL steps, and a schedule
        # keeps playing across them until it has finished.
        self._latch = bool(getattr(cfg, "latch", False))
        if not self._latch and int(env.cfg.decimation) != self._n_sub:
            raise ValueError(
                f"sweep action needs decimation {self._n_sub} (MACRO_STEP_S / dt) unless latched; "
                f"got {env.cfg.decimation}")
        self._check_arm_actuator()
        # What the drive is sent: the schedule's position target with the
        # feedforward folded in, and its velocity target (schedule.feedforward).
        # Until a plan exists, a gravity-compensated pinch hold.
        _, hold_cmd, _ = hold_command(table.joints, 1)
        cmd = torch.as_tensor(hold_cmd[0], dtype=torch.float32, device=env.device)
        self._sched = cmd.view(1, 1, -1).repeat(n, self._n_sub, 1)
        self._sched_qd = torch.zeros_like(self._sched)
        self._vel = torch.zeros_like(self._default)
        # hold=True: never plan; the arm stays in the pinch hold. The control for
        # "does the free base fall without the arm moving?".
        self._hold = bool(getattr(cfg, "hold", False))
        self._trace_q = np.repeat(pinch_arm(table.joints)[None], self._n_sub, axis=0)
        start = self._n_sub if self._latch else 0
        self._k = torch.full((n,), start, dtype=torch.long, device=env.device)
        # BHL_SWEEP_TRACE=1: record env 0 -- each plan, and every
        # BHL_SWEEP_TRACE_EVERY substeps the commanded and measured arm joints,
        # the hand link, the robot root and the garment -- so tracking error and
        # contact can be read off a run instead of guessed from a clip.
        import os
        self.trace_every = int(os.environ.get("BHL_SWEEP_TRACE_EVERY", "10"))
        self.trace_on = os.environ.get("BHL_SWEEP_TRACE", "0") == "1"
        self.trace_plans: list[dict] = []
        self.trace_samples: list[dict] = []
        self._n_plans = 0
        self._hand_idx = list(self._asset.body_names).index("arm_right_hand_link")

    def _check_arm_actuator(self) -> None:
        """Refuse to run with an arm actuator the schedule's feedforward was not computed for.

        The position offset is torque / Kp and the velocity target cancels the
        drive's damping: both are only right for the gains, limit and rotor
        inertia in ``schedule.ARM_*``. A different actuator would get a silently
        wrong command, which is exactly the kind of quiet mismatch to avoid.
        """
        from bhl_robust.cloth.schedule import ARM_ARMATURE, ARM_EFFORT, ARM_KD, ARM_KP

        groups = [a for a in (getattr(self._asset.cfg, "actuators", None) or {}).values()
                  if any("arm_" in e for e in a.joint_names_expr)]
        if len(groups) != 1 or type(groups[0]).__name__ != "ImplicitActuatorCfg":
            raise ValueError(f"expected one implicit arm actuator group, found {[type(g).__name__ for g in groups]}")
        a = groups[0]
        effort = a.effort_limit_sim if getattr(a, "effort_limit_sim", None) is not None else a.effort_limit
        have = {"stiffness": a.stiffness, "damping": a.damping, "effort": effort, "armature": a.armature}
        want = {"stiffness": ARM_KP, "damping": ARM_KD, "effort": ARM_EFFORT, "armature": ARM_ARMATURE}
        bad = {k: (have[k], want[k]) for k in want
               if not isinstance(have[k], (int, float)) or abs(float(have[k]) - want[k]) > 1e-9}
        if bad:
            raise ValueError(f"arm actuator (have, schedule assumes): {bad}")

    @property
    def action_dim(self) -> int:
        return ACTION_DIM

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._proc

    def _garment_in_layout_frame(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Garment xy and yaw in the layout frame, through the robot's actual root pose."""
        from bhl_robust.cloth.layout import ROBOT_XY

        env = self._env
        g = _local_pos(env, "garment_0")[:, :2]
        root = _t(self._asset.data.root_pos_w)[:, :2] - env.scene.env_origins[:, :2]
        q = _t(self._asset.data.root_quat_w)
        q0 = _t(self._asset.data.default_root_state)[:, 3:7]
        yaw = torch.atan2(*yaw_atan2_args(q, QUAT_ORDER)) - torch.atan2(*yaw_atan2_args(q0, QUAT_ORDER))
        d = g - root
        c, s = torch.cos(-yaw), torch.sin(-yaw)
        out = torch.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1]], dim=1)
        g_yaw = garment_yaw_proxy(env, "garment_0")[:, 0] - yaw
        return out + torch.tensor(ROBOT_XY, device=out.device, dtype=out.dtype), g_yaw

    def process_actions(self, actions: torch.Tensor) -> None:
        from bhl_robust.cloth.schedule import build_schedule

        self._raw[:] = actions
        self._proc[:] = actions.clamp(-1.0, 1.0)
        if self._hold:
            self._k.zero_()
            return
        if self._latch:
            todo = [int(i) for i in torch.nonzero(self._k >= self._n_sub).flatten().tolist()]
        else:
            todo = list(range(self.num_envs))
        if not todo:
            return
        spec = _spec_for(getattr(self._env.cfg, "garment_name", "shirt_a"))
        g_t, yaw_t = self._garment_in_layout_frame()
        g, yaw = g_t.detach().cpu().numpy(), yaw_t.detach().cpu().numpy()
        acts = self._proc.detach().cpu().numpy()
        scale = float(getattr(self.cfg, "residual_scale", 0.0))
        if scale > 0.0:
            # Residual mode: the policy corrects the scripted sweep for this
            # garment. Raw 5-D actions drawn uniformly are a sweep the hand can
            # make about 1 time in 9, so raw PPO would spend its first
            # iterations issuing refused plans.
            from bhl_robust.cloth.scripted import scripted_action
            for i in todo:
                acts[i] = np.clip(scripted_action(g[i], spec, garment_yaw=float(yaw[i])) + scale * acts[i],
                                  -1.0, 1.0)
        for i in todo:
            s = build_schedule(g[i], spec, acts[i], self._dt, self._n_sub, table=self._table,
                               garment_yaw=float(yaw[i]))
            self._sched[i] = torch.as_tensor(s.q_cmd, dtype=torch.float32, device=self.device)
            self._sched_qd[i] = torch.as_tensor(s.qd, dtype=torch.float32, device=self.device)
            self._invalid[i] = not s.valid
            self._k[i] = 0
            if i == 0:
                self._trace_q = s.q
            if i == 0 and self.trace_on:
                self.trace_plans.append({
                    "plan": self._n_plans, "valid": bool(s.valid), "reason": s.reason,
                    "garment_xy": [float(v) for v in g[0]], "garment_yaw": float(yaw[0]),
                    "action": [float(v) for v in acts[0]],
                    "start_xy": None if s.start_xy is None else [float(v) for v in s.start_xy],
                    "end_xy": None if s.end_xy is None else [float(v) for v in s.end_xy],
                    "anchor_in": None if s.anchor_in is None else [float(v) for v in s.anchor_in],
                    "anchor_out": None if s.anchor_out is None else [float(v) for v in s.anchor_out],
                    "route_in": [[float(a), float(b)] for a, b in s.route_in],
                    "route_out": [[float(a), float(b)] for a, b in s.route_out],
                    "sweep_window": None if s.sweep_window is None else [float(v) for v in s.sweep_window],
                    "duration": float(s.duration), "tau_peak": float(s.tau_peak),
                })
        if 0 in todo:
            self._n_plans += 1

    def apply_actions(self) -> None:
        k = self._k.clamp(max=self._n_sub - 1)
        rows = torch.arange(self.num_envs, device=self.device)
        arm = self._sched[rows, k]
        targets = self._default.clone()
        targets[:, self._arm_idx] = arm
        vel = self._vel.clone()
        vel[:, self._arm_idx] = self._sched_qd[rows, k]
        self._asset.set_joint_position_target(targets)
        self._asset.set_joint_velocity_target(vel)
        if self.trace_on and int(self._k[0]) < self._n_sub and int(self._k[0]) % self.trace_every == 0:
            self._record(int(self._k[0]), arm[0], vel[0, self._arm_idx])
        self._k += 1

    def _record(self, k: int, target, vel_target) -> None:
        env, data = self._env, self._asset.data
        o = env.scene.env_origins[0]
        gq = _t(env.scene["garment_0"].data.root_quat_w) if hasattr(env.scene["garment_0"].data, "root_quat_w") else None
        self.trace_samples.append({
            "plan": self._n_plans - 1, "k": k, "t": k * self._dt,
            "q_target": [float(v) for v in target],
            "qd_target": [float(v) for v in vel_target],
            "q_desired": [float(v) for v in self._trace_q[min(k, len(self._trace_q) - 1)]],
            "q": [float(v) for v in _t(data.joint_pos)[0, self._arm_idx]],
            "qd": [float(v) for v in _t(data.joint_vel)[0, self._arm_idx]],
            "hand_pos": [float(v) for v in (_t(data.body_pos_w)[0, self._hand_idx] - o)],
            "hand_quat": [float(v) for v in _t(data.body_quat_w)[0, self._hand_idx]],
            "root_pos": [float(v) for v in (_t(data.root_pos_w)[0] - o)],
            "root_quat": [float(v) for v in _t(data.root_quat_w)[0]],
            "garment_pos": [float(v) for v in _local_pos(env, "garment_0")[0]],
            "garment_quat": None if gq is None else [float(v) for v in gq[0]],
        })

    def reset(self, env_ids=None) -> None:
        start = self._n_sub if self._latch else 0
        if env_ids is None:
            self._k.fill_(start)
            self._invalid.zero_()
        else:
            self._k[env_ids] = start
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
    #: Keep playing a schedule across RL steps until it finishes. For clips,
    #: where the env runs short steps so a camera sees the sweep happen.
    latch: bool = False
    #: 0 = the policy's action is the sweep. > 0 = the sweep is the scripted
    #: sweep plus this times the policy's action (clipped to [-1, 1]).
    residual_scale: float = 0.0
    #: Ignore actions and hold the pinch pose: the arm-still control.
    hold: bool = False


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
    return torch.atan2(*yaw_atan2_args(q, QUAT_ORDER)).unsqueeze(-1)


def rel_garment_basket(
    env: "ManagerBasedRLEnv",
    asset_name: str = "garment_0",
    garment: str = "shirt_a",
) -> torch.Tensor:
    spec = _spec_for(garment)
    p = _local_pos(env, asset_name)[:, :2]
    c = torch.tensor(
        [float(v) for v in basket_center(spec.target_basket)[:2]],
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
        [float(v) for v in basket_center(spec.target_basket)[:2]],
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

    Measured *relative to the spawn pose*, so a yaw jitter at reset reads 0 --
    yaw is not tilt. The quaternions are read in the order this Isaac Lab
    stores them (``QUAT_ORDER``, probed at import).

    History, because both mistakes ended episodes that should not have ended:
    an absolute ``R[2, 2] < 0.70`` test called the photographed-standing spawn
    fallen at reset (21233866), and this relative test replaced it -- but both
    read Isaac Lab 3.0's ``(x, y, z, w)`` quaternions as ``(w, x, y, z)``. Read
    that way the spawn ``(0, 0, 1, 0)`` is a half-turn about y (``R[2, 2] = -1``,
    the real cause of 21233866), and the relative test reports a yaw as tilt and
    a roll as nothing. Every "fell" it produced on Isaac Lab 3.0 -- including
    the free-base cloth runs 21300301 and 21300605 -- needs re-measuring.
    """
    robot = env.scene[asset_cfg.name]
    q = _t(robot.data.root_quat_w)
    if q.ndim == 1:
        q = q.unsqueeze(0)
    q0 = _t(robot.data.default_root_state)[:, 3:7]
    if q0.ndim == 1:
        q0 = q0.unsqueeze(0)
    up_z = relative_up_z(q, q0, order=QUAT_ORDER)
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
