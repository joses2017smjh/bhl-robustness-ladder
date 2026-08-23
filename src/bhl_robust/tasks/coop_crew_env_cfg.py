"""Cooperative lift with a crew of N, on one payload.

`coop_lift_env_cfg` is a pair: two robots named `_a` and `_b`, two contact
points at `centre +/- offset * axis`, and a clamp reward that is the dot product
of two vectors. That is not a limitation of the task, it is a limitation of
having written the terms out by hand -- and it is why the existing "3 robot" and
"4 robot" clips are not crews at all. They are one policy replicated across
independent pairs, each with its own crate, which the replay scores confirm:
crew 2, 3 and 4 return the same peak lift to two decimal places because the
pairs never interact.

This module builds the real thing. Robots stand on a ring around one payload,
each facing the centre, and every per-robot config term is generated rather than
typed, so N is an argument.

What had to change in the reward, and what did not:

* **Reach** is unchanged in form -- distance from each robot's hand midpoint to
  its own contact point -- but the contact points are now N points on a ring.
  At N = 2 a ring of two points 180 degrees apart is exactly the old
  `centre +/- offset * axis`.
* **Clamp** could not survive as a dot product. Its replacement is force
  closure: each robot pushes a unit vector inward, and the reward is
  `1 - |mean|`, which is 1 when the pushes cancel. At N = 2 that is
  algebraically the old `-v_a . v_b`, so the pair result is not perturbed by
  moving to the general term.
* **Spread** is new and only matters above N = 2. Force closure is satisfiable
  by a balanced but loose crew; without a variance penalty three robots carry
  while the fourth trails and the reward does not notice.

The payload scales with the crew. Two robots side-lift a 0.5 kg cube; asking
four robots to do the same thing gives every one of them a quarter of an already
small load, and the experiment stops being about cooperation. Mass and edge
length grow with N so the per-robot share stays near the pair's.
"""

from __future__ import annotations

import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_JOINTS
from berkeley_humanoid_lite.tasks.locomotion.velocity import mdp

from . import coop_lift_mdp as coop
from .coop_lift_env_cfg import (
    _COLLISION,
    _RIGID,
    CoopLiftEnvCfg,
    CoopLiftSceneCfg,
    CurriculumCfg,
    _robot,
)
from .coop_depth_env_cfg import COOP_CAM_POOL, coop_depth_obs, make_coop_depth_camera

# Robots stand this far from the payload centre, the radius the pair uses.
CREW_RADIUS = 0.48


def _cfgclass(name: str, bases: tuple, ns: dict, extra_ann: dict | None = None):
    """Build a configclass whose generated attributes are *real* dataclass fields.

    This is not a style point. `@configclass` copies unannotated class
    attributes onto the instance, so a class built without annotations works
    perfectly under `parse_env_cfg` -- and then silently loses every generated
    term when Hydra round-trips the config through `to_dict()` / `from_dict()`,
    because that path only carries declared fields. The inherited pair fields,
    which *are* annotated on the parent, reassert themselves, and training dies
    on `robot_a` not existing in a scene full of `robot_0..N`.

    Annotating from the value's runtime type covers the generated terms;
    `extra_ann` carries the `None` overrides that delete pair terms, which have
    no value to infer a type from.
    """
    # Dunders are methods and machinery, not fields. Annotating `__post_init__`
    # makes configclass count 25 annotations against 24 members and refuse the
    # class outright.
    ann = {k: type(v) for k, v in ns.items()
           if v is not None and not k.startswith("__")}
    ann.update(extra_ann or {})
    return configclass(type(name, bases, {"__annotations__": ann, **ns}))


def _yaw_quat(theta: float) -> tuple[float, float, float, float]:
    """Quaternion for a yaw that points the robot's +x at the origin."""
    half = (theta + math.pi) / 2.0
    return (math.cos(half), 0.0, 0.0, math.sin(half))


def _bearing(i: int, n: int) -> float:
    """Bearing of crew member `i`. Member 0 sits on +y, as `robot_a` did."""
    return 2.0 * math.pi * i / n + math.pi / 2.0


def _crew_payload(n: int, kind: str) -> RigidObjectCfg:
    """One payload, sized so the per-robot share matches the pair's.

    The pair lifts 0.5 kg of 0.28 m cube. Holding mass fixed while adding
    carriers makes the task monotonically easier with N and the comparison
    meaningless, so mass scales with the crew and the edge grows as the cube
    root of mass, which is what keeps the density of a solid object constant.
    """
    mass = 0.5 * n / 2.0
    scale = (n / 2.0) ** (1.0 / 3.0)
    edge = 0.28 * scale
    common = dict(rigid_props=_RIGID, collision_props=_COLLISION,
                  mass_props=sim_utils.MassPropertiesCfg(mass=mass),
                  visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.48, 0.12)),
                  physics_material=sim_utils.RigidBodyMaterialCfg(
                      static_friction=1.4, dynamic_friction=1.2))
    if kind == "ball":
        spawn = sim_utils.SphereCfg(radius=edge / 2.0, **common)
        z = edge / 2.0
    else:
        spawn = sim_utils.CuboidCfg(size=(edge, edge, edge), **common)
        z = edge / 2.0
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=spawn,
    )


def _names(n: int) -> list[str]:
    return [f"robot_{i}" for i in range(n)]


def make_crew_cfg(n: int, kind: str = "cube", vision: bool = False):
    """Build a crew-of-`n` env config class.

    Returns a `@configclass` type, not an instance, because Isaac Lab's gym
    registration wants an entry point it can instantiate itself.
    """
    if n < 2:
        raise ValueError("a crew is two or more robots")
    names = _names(n)
    cfgs = [SceneEntityCfg(r) for r in names]
    hands = [SceneEntityCfg(r, body_names=[".*_hand_link"]) for r in names]
    edge = 0.28 * (n / 2.0) ** (1.0 / 3.0)

    # --- scene ------------------------------------------------------------
    scene_ns, scene_ann = {}, {}
    for i, r in enumerate(names):
        th = _bearing(i, n)
        scene_ns[r] = _robot(
            "{ENV_REGEX_NS}/" + r,
            (CREW_RADIUS * math.cos(th), CREW_RADIUS * math.sin(th), 0.0),
            _yaw_quat(th),
        )
        scene_ann[r] = ArticulationCfg
        scene_ns[f"contact_{i}"] = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/" + r + "/.*", history_length=3, track_air_time=True)
        if vision:
            scene_ns[f"cam_{i}"] = make_coop_depth_camera(r)
    scene_ns["object"] = _crew_payload(n, kind)
    scene_ann["object"] = RigidObjectCfg
    # The pair's two robots and two contact sensors must go, or the scene
    # carries four robots when the crew is two.
    for dead in ("robot_a", "robot_b", "contact_a", "contact_b"):
        scene_ns[dead] = None
    SceneCls = _cfgclass(f"CoopCrew{n}SceneCfg", (CoopLiftSceneCfg,), scene_ns,
                         {d: type(None) for d in
                          ("robot_a", "robot_b", "contact_a", "contact_b")})

    # --- observations -----------------------------------------------------
    def _policy_terms():
        ns = {}
        for i, r in enumerate(names):
            ns[f"projected_gravity_{i}"] = ObsTerm(
                func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg(r)},
                noise=Unoise(n_min=-0.05, n_max=0.05))
            ns[f"base_ang_vel_{i}"] = ObsTerm(
                func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg(r)},
                noise=Unoise(n_min=-0.3, n_max=0.3))
            ns[f"joint_pos_{i}"] = ObsTerm(
                func=mdp.joint_pos_rel,
                params={"asset_cfg": SceneEntityCfg(r, joint_names=HUMANOID_LITE_JOINTS,
                                                    preserve_order=True)},
                noise=Unoise(n_min=-0.05, n_max=0.05))
            ns[f"joint_vel_{i}"] = ObsTerm(
                func=mdp.joint_vel_rel,
                params={"asset_cfg": SceneEntityCfg(r, joint_names=HUMANOID_LITE_JOINTS,
                                                    preserve_order=True)},
                noise=Unoise(n_min=-2.0, n_max=2.0))
            ns[f"object_pos_{i}"] = ObsTerm(
                func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg(r)})
            ns[f"track_err_{i}"] = ObsTerm(
                func=coop.joint_target_error,
                params={"asset_cfg": SceneEntityCfg(r, joint_names=HUMANOID_LITE_JOINTS,
                                                    preserve_order=True)})
            if vision:
                ns[f"depth_{i}"] = ObsTerm(
                    func=coop_depth_obs,
                    params={"sensor_cfg": SceneEntityCfg(f"cam_{i}"), "pool": COOP_CAM_POOL})
        ns["actions"] = ObsTerm(func=mdp.last_action)
        return ns

    _pol = _policy_terms()
    _pol["__post_init__"] = lambda self: setattr(self, "enable_corruption", True)
    PolicyCls = _cfgclass(f"CoopCrew{n}PolicyCfg", (ObsGroup,), _pol)

    critic_ns = dict(_policy_terms())
    critic_ns["object_lin_vel"] = ObsTerm(func=coop.object_lin_vel_w)
    critic_ns["object_ang_vel"] = ObsTerm(func=coop.object_ang_vel_w)
    for i, r in enumerate(names):
        critic_ns[f"base_lin_vel_{i}"] = ObsTerm(
            func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg(r)})
    critic_ns["__post_init__"] = lambda self: setattr(self, "enable_corruption", False)
    CriticCls = _cfgclass(f"CoopCrew{n}CriticCfg", (ObsGroup,), critic_ns)

    ObsCls = configclass(type(
        f"CoopCrew{n}ObsCfg", (object,),
        {"__annotations__": {"policy": PolicyCls, "critic": CriticCls},
         "policy": PolicyCls(), "critic": CriticCls()}))

    # --- actions ----------------------------------------------------------
    act_ns = {}
    for i, r in enumerate(names):
        act_ns[f"joint_pos_{i}"] = mdp.JointPositionActionCfg(
            asset_name=r, joint_names=HUMANOID_LITE_JOINTS, preserve_order=True,
            scale=0.25, use_default_offset=True)
    act_ns["joint_pos_a"] = None
    act_ns["joint_pos_b"] = None
    from .coop_lift_env_cfg import ActionsCfg as _PairActions
    ActCls = _cfgclass(f"CoopCrew{n}ActionsCfg", (_PairActions,), act_ns,
                       {"joint_pos_a": type(None), "joint_pos_b": type(None)})

    # --- rewards ----------------------------------------------------------
    rew_ns = {
        "reaching_coarse": RewTerm(func=coop.crew_reach, weight=1.0,
                                   params={"std": 0.40, "robot_cfgs": hands}),
        "reaching_fine": RewTerm(func=coop.crew_reach, weight=2.0,
                                 params={"std": 0.12, "robot_cfgs": hands}),
        "opposing_clamp": RewTerm(func=coop.crew_force_closure, weight=2.0,
                                  params={"robot_cfgs": hands}),
        "crew_spread": RewTerm(func=coop.crew_spread, weight=-2.0,
                               params={"robot_cfgs": hands}),
    }
    for i, r in enumerate(names):
        rew_ns[f"flat_{i}"] = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0,
                                      params={"asset_cfg": SceneEntityCfg(r)})
        rew_ns[f"torques_{i}"] = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4,
                                         params={"asset_cfg": SceneEntityCfg(r)})
        rew_ns[f"base_contact_{i}"] = RewTerm(
            func=mdp.undesired_contacts, weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg(f"contact_{i}", body_names="base"),
                    "threshold": 1.0})
    for dead in ("flat_a", "flat_b", "torques_a", "torques_b",
                 "base_contact_a", "base_contact_b"):
        rew_ns[dead] = None
    from .coop_lift_env_cfg import RewardsCfg as _PairRewards
    RewCls = _cfgclass(f"CoopCrew{n}RewardsCfg", (_PairRewards,), rew_ns,
                       {d: type(None) for d in
                        ("flat_a", "flat_b", "torques_a", "torques_b",
                         "base_contact_a", "base_contact_b")})

    # --- terminations + events -------------------------------------------
    from .coop_lift_env_cfg import EventsCfg as _PairEvents
    from .coop_lift_env_cfg import TerminationsCfg as _PairTerms
    TermCls = _cfgclass(
        f"CoopCrew{n}TermCfg", (_PairTerms,),
        {"fallen": DoneTerm(func=coop.any_fallen,
                            params={"limit_angle": 0.78, "robot_names": names})})

    ev_ns = {}
    for i, r in enumerate(names):
        ev_ns[f"reset_joints_{i}"] = EventTerm(
            func=mdp.reset_joints_by_offset, mode="reset",
            params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05),
                    "asset_cfg": SceneEntityCfg(r)})
        th = _bearing(i, n)
        ev_ns[f"reset_root_{i}"] = EventTerm(
            func=mdp.reset_root_state_uniform, mode="reset",
            params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02),
                                   "yaw": (-0.05, 0.05)},
                    "velocity_range": {}, "asset_cfg": SceneEntityCfg(r)})
    for dead in ("reset_joints_a", "reset_joints_b", "reset_root_a", "reset_root_b"):
        ev_ns[dead] = None
    EvCls = _cfgclass(f"CoopCrew{n}EventsCfg", (_PairEvents,), ev_ns,
                      {d: type(None) for d in
                       ("reset_joints_a", "reset_joints_b",
                        "reset_root_a", "reset_root_b")})

    # --- the env ----------------------------------------------------------
    def _post_init(self):
        CoopLiftEnvCfg.__post_init__(self)
        # Contact ring must match the payload the crew is actually holding.
        # The contact ring has to track the payload the crew is actually
        # holding: the pair's 0.16 m is half of a 0.28 m cube, and the cube
        # grows with N.
        self.contact_offset = edge / 2.0 + 0.02
        self.object_spawn_z = edge / 2.0

    return configclass(type(
        f"CoopCrew{n}{kind.capitalize()}{'Depth' if vision else ''}Cfg",
        (CoopLiftEnvCfg,),
        {"__annotations__": {"scene": SceneCls, "observations": ObsCls,
                             "actions": ActCls, "rewards": RewCls,
                             "terminations": TermCls, "events": EvCls,
                             "curriculum": CurriculumCfg},
         "scene": SceneCls(num_envs=1024, env_spacing=4.0),
         "observations": ObsCls(), "actions": ActCls(), "rewards": RewCls(),
         "terminations": TermCls(), "events": EvCls(), "curriculum": CurriculumCfg(),
         "crew_size": n, "object_kind": kind,
         "__post_init__": _post_init}))
