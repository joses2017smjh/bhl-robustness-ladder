"""Emit the crew configs as ordinary Python source.

Why generate a file instead of assembling classes with `type()` at import time:
the dynamic version worked perfectly under `parse_env_cfg` and then failed three
separate ways once anything else touched it. Terms that were not dataclass
fields, so Hydra's to_dict/from_dict round trip dropped them and the inherited
pair fields reasserted. An annotated `__post_init__` that broke configclass's
annotation-versus-member count. Then an import that hung outright.

Isaac Lab's `@configclass` is built for classes written out in a module. Every
hour spent making it accept a class assembled at runtime is an hour spent on the
framework rather than on the robot, so the crew configs are written out instead.
They are then indistinguishable from hand-written configs, which is the shape
Hydra and Isaac Lab are actually tested against. The cost is a generated file in
the tree; the benefit is that its failure mode is a syntax error you can read
rather than a metaclass mismatch you cannot.

Regenerate with:  python scripts/gen_crew_cfg.py
"""
from __future__ import annotations

import math
from pathlib import Path

CREW_RADIUS = 0.48
OUT = Path(__file__).resolve().parents[1] / "src/bhl_robust/tasks/coop_crew_generated.py"

HEADER = '''"""Crews of N humanoids on one payload. GENERATED -- do not edit by hand.

Regenerate with `python scripts/gen_crew_cfg.py`. The rationale for the ring
layout, the force-closure reward and the payload scaling lives in
`coop_crew_env_cfg.py`, which is where a reader should start.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_JOINTS
from berkeley_humanoid_lite.tasks.locomotion.velocity import mdp

from . import coop_lift_mdp as coop
from .coop_depth_env_cfg import COOP_CAM_POOL, coop_depth_obs, make_coop_depth_camera
from .coop_lift_env_cfg import _COLLISION, _RIGID, CoopLiftEnvCfg, CurriculumCfg, _robot
'''


def bearing(i, n):
    return 2.0 * math.pi * i / n + math.pi / 2.0


def yaw_quat(theta):
    h = (theta + math.pi) / 2.0
    return (math.cos(h), 0.0, 0.0, math.sin(h))


def obs_terms(names, indent, vision, critic):
    out = []
    for i, r in enumerate(names):
        out += [
            f'{indent}projected_gravity_{i} = ObsTerm(func=mdp.projected_gravity, params={{"asset_cfg": SceneEntityCfg("{r}")}}, noise=Unoise(n_min=-0.05, n_max=0.05))',
            f'{indent}base_ang_vel_{i} = ObsTerm(func=mdp.base_ang_vel, params={{"asset_cfg": SceneEntityCfg("{r}")}}, noise=Unoise(n_min=-0.3, n_max=0.3))',
            f'{indent}joint_pos_{i} = ObsTerm(func=mdp.joint_pos_rel, params={{"asset_cfg": SceneEntityCfg("{r}", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}}, noise=Unoise(n_min=-0.05, n_max=0.05))',
            f'{indent}joint_vel_{i} = ObsTerm(func=mdp.joint_vel_rel, params={{"asset_cfg": SceneEntityCfg("{r}", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}}, noise=Unoise(n_min=-2.0, n_max=2.0))',
            f'{indent}object_pos_{i} = ObsTerm(func=coop.object_pos_in_root, params={{"robot_cfg": SceneEntityCfg("{r}")}})',
            f'{indent}track_err_{i} = ObsTerm(func=coop.joint_target_error, params={{"asset_cfg": SceneEntityCfg("{r}", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}})',
        ]
        if vision:
            out.append(f'{indent}depth_{i} = ObsTerm(func=coop_depth_obs, params={{"sensor_cfg": SceneEntityCfg("cam_{i}"), "pool": COOP_CAM_POOL}})')
    out.append(f"{indent}actions = ObsTerm(func=mdp.last_action)")
    if critic:
        out.append(f"{indent}object_lin_vel = ObsTerm(func=coop.object_lin_vel_w)")
        out.append(f"{indent}object_ang_vel = ObsTerm(func=coop.object_ang_vel_w)")
        for i, r in enumerate(names):
            out.append(f'{indent}base_lin_vel_{i} = ObsTerm(func=mdp.base_lin_vel, params={{"asset_cfg": SceneEntityCfg("{r}")}})')
    return out


def emit(n, vision):
    tag = f"Crew{n}{'Depth' if vision else ''}"
    edge = 0.28 * (n / 2.0) ** (1.0 / 3.0)
    mass = 0.5 * n / 2.0
    names = [f"robot_{i}" for i in range(n)]
    hands = "[" + ", ".join(f'SceneEntityCfg("{r}", body_names=[".*_hand_link"])' for r in names) + "]"
    L = [f"\n\n# --- crew of {n}{', depth-conditioned' if vision else ''} " + "-" * 24, "@configclass",
         f"class {tag}SceneCfg(InteractiveSceneCfg):",
         '    terrain = TerrainImporterCfg(',
         '        prim_path="/World/ground", terrain_type="plane", terrain_generator=None,',
         '        collision_group=-1,',
         '        physics_material=sim_utils.RigidBodyMaterialCfg(',
         '            friction_combine_mode="multiply", restitution_combine_mode="multiply",',
         '            static_friction=1.0, dynamic_friction=1.0),',
         '        debug_vis=False)']
    for i, r in enumerate(names):
        th = bearing(i, n); q = yaw_quat(th)
        L.append(f'    {r}: ArticulationCfg = _robot("{{ENV_REGEX_NS}}/{r}", ({CREW_RADIUS*math.cos(th):.6f}, {CREW_RADIUS*math.sin(th):.6f}, 0.0), ({q[0]:.6f}, 0.0, 0.0, {q[3]:.6f}))')
    L += ['    object: RigidObjectCfg = RigidObjectCfg(',
          '        prim_path="{ENV_REGEX_NS}/object",',
          f'        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, {edge/2:.6f}), rot=(1.0, 0.0, 0.0, 0.0)),',
          '        spawn=sim_utils.CuboidCfg(',
          f'            size=({edge:.6f}, {edge:.6f}, {edge:.6f}), rigid_props=_RIGID, collision_props=_COLLISION,',
          f'            mass_props=sim_utils.MassPropertiesCfg(mass={mass:.4f}),',
          '            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.48, 0.12)),',
          '            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.2)))']
    for i, r in enumerate(names):
        L.append(f'    contact_{i} = ContactSensorCfg(prim_path="{{ENV_REGEX_NS}}/{r}/.*", history_length=3, track_air_time=True)')
        if vision:
            L.append(f'    cam_{i} = make_coop_depth_camera("{r}")')
    L += ['    light = AssetBaseCfg(prim_path="/World/light",',
          '        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))',
          '    sky_light = AssetBaseCfg(prim_path="/World/skyLight",',
          '        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0))',
          "", "", "@configclass", f"class {tag}ObservationsCfg:", "    @configclass",
          "    class PolicyCfg(ObsGroup):"]
    L += obs_terms(names, "        ", vision, critic=False)
    L += ["", "        def __post_init__(self):", "            self.enable_corruption = True",
          "", "    @configclass", "    class CriticCfg(ObsGroup):"]
    L += obs_terms(names, "        ", vision, critic=True)
    L += ["", "        def __post_init__(self):", "            self.enable_corruption = False",
          "", "    policy: PolicyCfg = PolicyCfg()", "    critic: CriticCfg = CriticCfg()",
          "", "", "@configclass", f"class {tag}ActionsCfg:"]
    for i, r in enumerate(names):
        L.append(f'    joint_pos_{i} = mdp.JointPositionActionCfg(asset_name="{r}", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)')
    L += ["", "", "@configclass", f"class {tag}RewardsCfg:",
          f'    reaching_coarse = RewTerm(func=coop.crew_reach, weight=1.0, params={{"std": 0.40, "robot_cfgs": {hands}}})',
          f'    reaching_fine = RewTerm(func=coop.crew_reach, weight=2.0, params={{"std": 0.12, "robot_cfgs": {hands}}})',
          f'    opposing_clamp = RewTerm(func=coop.crew_force_closure, weight=2.0, params={{"robot_cfgs": {hands}}})',
          f'    crew_spread = RewTerm(func=coop.crew_spread, weight=-2.0, params={{"robot_cfgs": {hands}}})',
          '    lift_progress = RewTerm(func=coop.object_lift_progress, weight=2.0, params={"height": 0.04})',
          '    lifting_object = RewTerm(func=coop.object_is_lifted, weight=15.0, params={"height": 0.04})',
          '    object_tilt = RewTerm(func=coop.object_tilt_l2, weight=-2.0)',
          '    object_xy = RewTerm(func=coop.object_xy_drift_l2, weight=-1.0)',
          '    object_vel = RewTerm(func=coop.object_lin_vel_l2, weight=-0.05)',
          '    still_alive = RewTerm(func=coop.still_alive, weight=1.0)',
          '    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)',
          '    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)']
    for i, r in enumerate(names):
        L += [f'    flat_{i} = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={{"asset_cfg": SceneEntityCfg("{r}")}})',
              f'    torques_{i} = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={{"asset_cfg": SceneEntityCfg("{r}")}})',
              f'    base_contact_{i} = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={{"sensor_cfg": SceneEntityCfg("contact_{i}", body_names="base"), "threshold": 1.0}})']
    L += ["", "", "@configclass", f"class {tag}TerminationsCfg:",
          '    time_out = DoneTerm(func=mdp.time_out, time_out=True)',
          f'    fallen = DoneTerm(func=coop.any_fallen, params={{"limit_angle": 0.78, "robot_names": {names!r}}})',
          "", "", "@configclass", f"class {tag}EventsCfg:"]
    for i, r in enumerate(names):
        L += [f'    reset_joints_{i} = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",',
              f'        params={{"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("{r}")}})',
              f'    reset_root_{i} = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",',
              f'        params={{"pose_range": {{"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}}, "velocity_range": {{}}, "asset_cfg": SceneEntityCfg("{r}")}})']
    L += ['    reset_object = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",',
          '        params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")})',
          "", "", "@configclass", f"class {tag}Cfg(CoopLiftEnvCfg):",
          f'    """Crew of {n} on one payload{", each carrying a depth camera" if vision else ""}."""', "",
          f'    scene: {tag}SceneCfg = {tag}SceneCfg(num_envs=1024, env_spacing=4.0)',
          f'    observations: {tag}ObservationsCfg = {tag}ObservationsCfg()',
          f'    actions: {tag}ActionsCfg = {tag}ActionsCfg()',
          f'    rewards: {tag}RewardsCfg = {tag}RewardsCfg()',
          f'    terminations: {tag}TerminationsCfg = {tag}TerminationsCfg()',
          f'    events: {tag}EventsCfg = {tag}EventsCfg()',
          '    curriculum: CurriculumCfg = CurriculumCfg()',
          f'    crew_size: int = {n}', "",
          "    def __post_init__(self):", "        super().__post_init__()",
          "        # Contact ring tracks the payload this crew actually holds.",
          f'        self.contact_offset = {edge/2 + 0.02:.6f}',
          f'        self.object_spawn_z = {edge/2:.6f}']
    return "\n".join(L)


src = HEADER + "".join(emit(n, v) for n in (3, 4) for v in (False, True))
OUT.write_text(src + "\n")
print(f"wrote {OUT}")
