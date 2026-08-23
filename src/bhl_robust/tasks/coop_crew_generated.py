"""Crews of N humanoids on one payload. GENERATED -- do not edit by hand.

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


# --- crew of 3 ------------------------
@configclass
class Crew3SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False)
    robot_0: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_0", (0.000000, 0.480000, 0.0), (-0.707107, 0.0, 0.0, 0.707107))
    robot_1: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_1", (-0.415692, -0.240000, 0.0), (-0.965926, 0.0, 0.0, -0.258819))
    robot_2: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_2", (0.415692, -0.240000, 0.0), (-0.258819, 0.0, 0.0, -0.965926))
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.160260), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.320520, 0.320520, 0.320520), rigid_props=_RIGID, collision_props=_COLLISION,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.7500),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.48, 0.12)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.2)))
    contact_0 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_0/.*", history_length=3, track_air_time=True)
    contact_1 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_1/.*", history_length=3, track_air_time=True)
    contact_2 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_2/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    sky_light = AssetBaseCfg(prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0))


@configclass
class Crew3ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity_0 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_0 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_0 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_0 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_0 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_0")})
        track_err_0 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        projected_gravity_1 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_1 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_1 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_1 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_1 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_1")})
        track_err_1 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        projected_gravity_2 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_2 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_2 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_2 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_2 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_2")})
        track_err_2 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(ObsGroup):
        projected_gravity_0 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_0 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_0 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_0 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_0 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_0")})
        track_err_0 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        projected_gravity_1 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_1 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_1 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_1 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_1 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_1")})
        track_err_1 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        projected_gravity_2 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_2 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_2 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_2 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_2 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_2")})
        track_err_2 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        actions = ObsTerm(func=mdp.last_action)
        object_lin_vel = ObsTerm(func=coop.object_lin_vel_w)
        object_ang_vel = ObsTerm(func=coop.object_ang_vel_w)
        base_lin_vel_0 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        base_lin_vel_1 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        base_lin_vel_2 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_2")})

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Crew3ActionsCfg:
    joint_pos_0 = mdp.JointPositionActionCfg(asset_name="robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)
    joint_pos_1 = mdp.JointPositionActionCfg(asset_name="robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)
    joint_pos_2 = mdp.JointPositionActionCfg(asset_name="robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)


@configclass
class Crew3RewardsCfg:
    reaching_coarse = RewTerm(func=coop.crew_reach, weight=1.0, params={"std": 0.40, "robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"])]})
    reaching_fine = RewTerm(func=coop.crew_reach, weight=2.0, params={"std": 0.12, "robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"])]})
    opposing_clamp = RewTerm(func=coop.crew_force_closure, weight=2.0, params={"robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"])]})
    crew_spread = RewTerm(func=coop.crew_spread, weight=-2.0, params={"robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"])]})
    lift_progress = RewTerm(func=coop.object_lift_progress, weight=2.0, params={"height": 0.04})
    lifting_object = RewTerm(func=coop.object_is_lifted, weight=15.0, params={"height": 0.04})
    object_tilt = RewTerm(func=coop.object_tilt_l2, weight=-2.0)
    object_xy = RewTerm(func=coop.object_xy_drift_l2, weight=-1.0)
    object_vel = RewTerm(func=coop.object_lin_vel_l2, weight=-0.05)
    still_alive = RewTerm(func=coop.still_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    flat_0 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_0")})
    torques_0 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_0")})
    base_contact_0 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_0", body_names="base"), "threshold": 1.0})
    flat_1 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_1")})
    torques_1 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_1")})
    base_contact_1 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_1", body_names="base"), "threshold": 1.0})
    flat_2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_2")})
    torques_2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_2")})
    base_contact_2 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_2", body_names="base"), "threshold": 1.0})


@configclass
class Crew3TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=coop.any_fallen, params={"limit_angle": 0.78, "robot_names": ['robot_0', 'robot_1', 'robot_2']})


@configclass
class Crew3EventsCfg:
    reset_joints_0 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_0")})
    reset_root_0 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_0")})
    reset_joints_1 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_1")})
    reset_root_1 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_1")})
    reset_joints_2 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_2")})
    reset_root_2 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_2")})
    reset_object = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")})


@configclass
class Crew3Cfg(CoopLiftEnvCfg):
    """Crew of 3 on one payload."""

    scene: Crew3SceneCfg = Crew3SceneCfg(num_envs=1024, env_spacing=4.0)
    observations: Crew3ObservationsCfg = Crew3ObservationsCfg()
    actions: Crew3ActionsCfg = Crew3ActionsCfg()
    rewards: Crew3RewardsCfg = Crew3RewardsCfg()
    terminations: Crew3TerminationsCfg = Crew3TerminationsCfg()
    events: Crew3EventsCfg = Crew3EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    crew_size: int = 3

    def __post_init__(self):
        super().__post_init__()
        # Contact ring tracks the payload this crew actually holds.
        self.contact_offset = 0.180260
        self.object_spawn_z = 0.160260

# --- crew of 3, depth-conditioned ------------------------
@configclass
class Crew3DepthSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False)
    robot_0: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_0", (0.000000, 0.480000, 0.0), (-0.707107, 0.0, 0.0, 0.707107))
    robot_1: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_1", (-0.415692, -0.240000, 0.0), (-0.965926, 0.0, 0.0, -0.258819))
    robot_2: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_2", (0.415692, -0.240000, 0.0), (-0.258819, 0.0, 0.0, -0.965926))
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.160260), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.320520, 0.320520, 0.320520), rigid_props=_RIGID, collision_props=_COLLISION,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.7500),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.48, 0.12)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.2)))
    contact_0 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_0/.*", history_length=3, track_air_time=True)
    cam_0 = make_coop_depth_camera("robot_0")
    contact_1 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_1/.*", history_length=3, track_air_time=True)
    cam_1 = make_coop_depth_camera("robot_1")
    contact_2 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_2/.*", history_length=3, track_air_time=True)
    cam_2 = make_coop_depth_camera("robot_2")
    light = AssetBaseCfg(prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    sky_light = AssetBaseCfg(prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0))


@configclass
class Crew3DepthObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity_0 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_0 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_0 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_0 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_0 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_0")})
        track_err_0 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_0 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_0"), "pool": COOP_CAM_POOL})
        projected_gravity_1 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_1 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_1 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_1 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_1 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_1")})
        track_err_1 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_1 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_1"), "pool": COOP_CAM_POOL})
        projected_gravity_2 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_2 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_2 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_2 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_2 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_2")})
        track_err_2 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_2 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_2"), "pool": COOP_CAM_POOL})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(ObsGroup):
        projected_gravity_0 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_0 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_0 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_0 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_0 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_0")})
        track_err_0 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_0 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_0"), "pool": COOP_CAM_POOL})
        projected_gravity_1 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_1 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_1 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_1 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_1 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_1")})
        track_err_1 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_1 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_1"), "pool": COOP_CAM_POOL})
        projected_gravity_2 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_2 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_2 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_2 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_2 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_2")})
        track_err_2 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_2 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_2"), "pool": COOP_CAM_POOL})
        actions = ObsTerm(func=mdp.last_action)
        object_lin_vel = ObsTerm(func=coop.object_lin_vel_w)
        object_ang_vel = ObsTerm(func=coop.object_ang_vel_w)
        base_lin_vel_0 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        base_lin_vel_1 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        base_lin_vel_2 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_2")})

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Crew3DepthActionsCfg:
    joint_pos_0 = mdp.JointPositionActionCfg(asset_name="robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)
    joint_pos_1 = mdp.JointPositionActionCfg(asset_name="robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)
    joint_pos_2 = mdp.JointPositionActionCfg(asset_name="robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)


@configclass
class Crew3DepthRewardsCfg:
    reaching_coarse = RewTerm(func=coop.crew_reach, weight=1.0, params={"std": 0.40, "robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"])]})
    reaching_fine = RewTerm(func=coop.crew_reach, weight=2.0, params={"std": 0.12, "robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"])]})
    opposing_clamp = RewTerm(func=coop.crew_force_closure, weight=2.0, params={"robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"])]})
    crew_spread = RewTerm(func=coop.crew_spread, weight=-2.0, params={"robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"])]})
    lift_progress = RewTerm(func=coop.object_lift_progress, weight=2.0, params={"height": 0.04})
    lifting_object = RewTerm(func=coop.object_is_lifted, weight=15.0, params={"height": 0.04})
    object_tilt = RewTerm(func=coop.object_tilt_l2, weight=-2.0)
    object_xy = RewTerm(func=coop.object_xy_drift_l2, weight=-1.0)
    object_vel = RewTerm(func=coop.object_lin_vel_l2, weight=-0.05)
    still_alive = RewTerm(func=coop.still_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    flat_0 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_0")})
    torques_0 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_0")})
    base_contact_0 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_0", body_names="base"), "threshold": 1.0})
    flat_1 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_1")})
    torques_1 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_1")})
    base_contact_1 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_1", body_names="base"), "threshold": 1.0})
    flat_2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_2")})
    torques_2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_2")})
    base_contact_2 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_2", body_names="base"), "threshold": 1.0})


@configclass
class Crew3DepthTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=coop.any_fallen, params={"limit_angle": 0.78, "robot_names": ['robot_0', 'robot_1', 'robot_2']})


@configclass
class Crew3DepthEventsCfg:
    reset_joints_0 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_0")})
    reset_root_0 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_0")})
    reset_joints_1 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_1")})
    reset_root_1 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_1")})
    reset_joints_2 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_2")})
    reset_root_2 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_2")})
    reset_object = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")})


@configclass
class Crew3DepthCfg(CoopLiftEnvCfg):
    """Crew of 3 on one payload, each carrying a depth camera."""

    scene: Crew3DepthSceneCfg = Crew3DepthSceneCfg(num_envs=1024, env_spacing=4.0)
    observations: Crew3DepthObservationsCfg = Crew3DepthObservationsCfg()
    actions: Crew3DepthActionsCfg = Crew3DepthActionsCfg()
    rewards: Crew3DepthRewardsCfg = Crew3DepthRewardsCfg()
    terminations: Crew3DepthTerminationsCfg = Crew3DepthTerminationsCfg()
    events: Crew3DepthEventsCfg = Crew3DepthEventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    crew_size: int = 3

    def __post_init__(self):
        super().__post_init__()
        # Contact ring tracks the payload this crew actually holds.
        self.contact_offset = 0.180260
        self.object_spawn_z = 0.160260

# --- crew of 4 ------------------------
@configclass
class Crew4SceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False)
    robot_0: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_0", (0.000000, 0.480000, 0.0), (-0.707107, 0.0, 0.0, 0.707107))
    robot_1: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_1", (-0.480000, 0.000000, 0.0), (-1.000000, 0.0, 0.0, 0.000000))
    robot_2: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_2", (-0.000000, -0.480000, 0.0), (-0.707107, 0.0, 0.0, -0.707107))
    robot_3: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_3", (0.480000, -0.000000, 0.0), (-0.000000, 0.0, 0.0, -1.000000))
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.176389), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.352778, 0.352778, 0.352778), rigid_props=_RIGID, collision_props=_COLLISION,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0000),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.48, 0.12)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.2)))
    contact_0 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_0/.*", history_length=3, track_air_time=True)
    contact_1 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_1/.*", history_length=3, track_air_time=True)
    contact_2 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_2/.*", history_length=3, track_air_time=True)
    contact_3 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_3/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    sky_light = AssetBaseCfg(prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0))


@configclass
class Crew4ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity_0 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_0 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_0 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_0 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_0 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_0")})
        track_err_0 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        projected_gravity_1 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_1 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_1 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_1 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_1 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_1")})
        track_err_1 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        projected_gravity_2 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_2 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_2 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_2 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_2 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_2")})
        track_err_2 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        projected_gravity_3 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_3")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_3 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_3")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_3 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_3 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_3 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_3")})
        track_err_3 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(ObsGroup):
        projected_gravity_0 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_0 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_0 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_0 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_0 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_0")})
        track_err_0 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        projected_gravity_1 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_1 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_1 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_1 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_1 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_1")})
        track_err_1 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        projected_gravity_2 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_2 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_2 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_2 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_2 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_2")})
        track_err_2 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        projected_gravity_3 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_3")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_3 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_3")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_3 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_3 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_3 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_3")})
        track_err_3 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        actions = ObsTerm(func=mdp.last_action)
        object_lin_vel = ObsTerm(func=coop.object_lin_vel_w)
        object_ang_vel = ObsTerm(func=coop.object_ang_vel_w)
        base_lin_vel_0 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        base_lin_vel_1 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        base_lin_vel_2 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_2")})
        base_lin_vel_3 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_3")})

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Crew4ActionsCfg:
    joint_pos_0 = mdp.JointPositionActionCfg(asset_name="robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)
    joint_pos_1 = mdp.JointPositionActionCfg(asset_name="robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)
    joint_pos_2 = mdp.JointPositionActionCfg(asset_name="robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)
    joint_pos_3 = mdp.JointPositionActionCfg(asset_name="robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)


@configclass
class Crew4RewardsCfg:
    reaching_coarse = RewTerm(func=coop.crew_reach, weight=1.0, params={"std": 0.40, "robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"]), SceneEntityCfg("robot_3", body_names=[".*_hand_link"])]})
    reaching_fine = RewTerm(func=coop.crew_reach, weight=2.0, params={"std": 0.12, "robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"]), SceneEntityCfg("robot_3", body_names=[".*_hand_link"])]})
    opposing_clamp = RewTerm(func=coop.crew_force_closure, weight=2.0, params={"robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"]), SceneEntityCfg("robot_3", body_names=[".*_hand_link"])]})
    crew_spread = RewTerm(func=coop.crew_spread, weight=-2.0, params={"robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"]), SceneEntityCfg("robot_3", body_names=[".*_hand_link"])]})
    lift_progress = RewTerm(func=coop.object_lift_progress, weight=2.0, params={"height": 0.04})
    lifting_object = RewTerm(func=coop.object_is_lifted, weight=15.0, params={"height": 0.04})
    object_tilt = RewTerm(func=coop.object_tilt_l2, weight=-2.0)
    object_xy = RewTerm(func=coop.object_xy_drift_l2, weight=-1.0)
    object_vel = RewTerm(func=coop.object_lin_vel_l2, weight=-0.05)
    still_alive = RewTerm(func=coop.still_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    flat_0 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_0")})
    torques_0 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_0")})
    base_contact_0 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_0", body_names="base"), "threshold": 1.0})
    flat_1 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_1")})
    torques_1 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_1")})
    base_contact_1 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_1", body_names="base"), "threshold": 1.0})
    flat_2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_2")})
    torques_2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_2")})
    base_contact_2 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_2", body_names="base"), "threshold": 1.0})
    flat_3 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_3")})
    torques_3 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_3")})
    base_contact_3 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_3", body_names="base"), "threshold": 1.0})


@configclass
class Crew4TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=coop.any_fallen, params={"limit_angle": 0.78, "robot_names": ['robot_0', 'robot_1', 'robot_2', 'robot_3']})


@configclass
class Crew4EventsCfg:
    reset_joints_0 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_0")})
    reset_root_0 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_0")})
    reset_joints_1 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_1")})
    reset_root_1 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_1")})
    reset_joints_2 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_2")})
    reset_root_2 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_2")})
    reset_joints_3 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_3")})
    reset_root_3 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_3")})
    reset_object = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")})


@configclass
class Crew4Cfg(CoopLiftEnvCfg):
    """Crew of 4 on one payload."""

    scene: Crew4SceneCfg = Crew4SceneCfg(num_envs=1024, env_spacing=4.0)
    observations: Crew4ObservationsCfg = Crew4ObservationsCfg()
    actions: Crew4ActionsCfg = Crew4ActionsCfg()
    rewards: Crew4RewardsCfg = Crew4RewardsCfg()
    terminations: Crew4TerminationsCfg = Crew4TerminationsCfg()
    events: Crew4EventsCfg = Crew4EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    crew_size: int = 4

    def __post_init__(self):
        super().__post_init__()
        # Contact ring tracks the payload this crew actually holds.
        self.contact_offset = 0.196389
        self.object_spawn_z = 0.176389

# --- crew of 4, depth-conditioned ------------------------
@configclass
class Crew4DepthSceneCfg(InteractiveSceneCfg):
    terrain = TerrainImporterCfg(
        prim_path="/World/ground", terrain_type="plane", terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply", restitution_combine_mode="multiply",
            static_friction=1.0, dynamic_friction=1.0),
        debug_vis=False)
    robot_0: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_0", (0.000000, 0.480000, 0.0), (-0.707107, 0.0, 0.0, 0.707107))
    robot_1: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_1", (-0.480000, 0.000000, 0.0), (-1.000000, 0.0, 0.0, 0.000000))
    robot_2: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_2", (-0.000000, -0.480000, 0.0), (-0.707107, 0.0, 0.0, -0.707107))
    robot_3: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_3", (0.480000, -0.000000, 0.0), (-0.000000, 0.0, 0.0, -1.000000))
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 0.176389), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=sim_utils.CuboidCfg(
            size=(0.352778, 0.352778, 0.352778), rigid_props=_RIGID, collision_props=_COLLISION,
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0000),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.48, 0.12)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.2)))
    contact_0 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_0/.*", history_length=3, track_air_time=True)
    cam_0 = make_coop_depth_camera("robot_0")
    contact_1 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_1/.*", history_length=3, track_air_time=True)
    cam_1 = make_coop_depth_camera("robot_1")
    contact_2 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_2/.*", history_length=3, track_air_time=True)
    cam_2 = make_coop_depth_camera("robot_2")
    contact_3 = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_3/.*", history_length=3, track_air_time=True)
    cam_3 = make_coop_depth_camera("robot_3")
    light = AssetBaseCfg(prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0))
    sky_light = AssetBaseCfg(prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0))


@configclass
class Crew4DepthObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity_0 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_0 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_0 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_0 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_0 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_0")})
        track_err_0 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_0 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_0"), "pool": COOP_CAM_POOL})
        projected_gravity_1 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_1 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_1 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_1 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_1 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_1")})
        track_err_1 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_1 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_1"), "pool": COOP_CAM_POOL})
        projected_gravity_2 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_2 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_2 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_2 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_2 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_2")})
        track_err_2 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_2 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_2"), "pool": COOP_CAM_POOL})
        projected_gravity_3 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_3")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_3 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_3")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_3 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_3 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_3 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_3")})
        track_err_3 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_3 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_3"), "pool": COOP_CAM_POOL})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True

    @configclass
    class CriticCfg(ObsGroup):
        projected_gravity_0 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_0 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_0")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_0 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_0 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_0 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_0")})
        track_err_0 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_0 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_0"), "pool": COOP_CAM_POOL})
        projected_gravity_1 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_1 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_1")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_1 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_1 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_1 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_1")})
        track_err_1 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_1 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_1"), "pool": COOP_CAM_POOL})
        projected_gravity_2 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_2 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_2")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_2 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_2 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_2 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_2")})
        track_err_2 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_2 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_2"), "pool": COOP_CAM_POOL})
        projected_gravity_3 = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot_3")}, noise=Unoise(n_min=-0.05, n_max=0.05))
        base_ang_vel_3 = ObsTerm(func=mdp.base_ang_vel, params={"asset_cfg": SceneEntityCfg("robot_3")}, noise=Unoise(n_min=-0.3, n_max=0.3))
        joint_pos_3 = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_vel_3 = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)}, noise=Unoise(n_min=-2.0, n_max=2.0))
        object_pos_3 = ObsTerm(func=coop.object_pos_in_root, params={"robot_cfg": SceneEntityCfg("robot_3")})
        track_err_3 = ObsTerm(func=coop.joint_target_error, params={"asset_cfg": SceneEntityCfg("robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)})
        depth_3 = ObsTerm(func=coop_depth_obs, params={"sensor_cfg": SceneEntityCfg("cam_3"), "pool": COOP_CAM_POOL})
        actions = ObsTerm(func=mdp.last_action)
        object_lin_vel = ObsTerm(func=coop.object_lin_vel_w)
        object_ang_vel = ObsTerm(func=coop.object_ang_vel_w)
        base_lin_vel_0 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_0")})
        base_lin_vel_1 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_1")})
        base_lin_vel_2 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_2")})
        base_lin_vel_3 = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_3")})

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Crew4DepthActionsCfg:
    joint_pos_0 = mdp.JointPositionActionCfg(asset_name="robot_0", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)
    joint_pos_1 = mdp.JointPositionActionCfg(asset_name="robot_1", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)
    joint_pos_2 = mdp.JointPositionActionCfg(asset_name="robot_2", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)
    joint_pos_3 = mdp.JointPositionActionCfg(asset_name="robot_3", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True, scale=0.25, use_default_offset=True)


@configclass
class Crew4DepthRewardsCfg:
    reaching_coarse = RewTerm(func=coop.crew_reach, weight=1.0, params={"std": 0.40, "robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"]), SceneEntityCfg("robot_3", body_names=[".*_hand_link"])]})
    reaching_fine = RewTerm(func=coop.crew_reach, weight=2.0, params={"std": 0.12, "robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"]), SceneEntityCfg("robot_3", body_names=[".*_hand_link"])]})
    opposing_clamp = RewTerm(func=coop.crew_force_closure, weight=2.0, params={"robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"]), SceneEntityCfg("robot_3", body_names=[".*_hand_link"])]})
    crew_spread = RewTerm(func=coop.crew_spread, weight=-2.0, params={"robot_cfgs": [SceneEntityCfg("robot_0", body_names=[".*_hand_link"]), SceneEntityCfg("robot_1", body_names=[".*_hand_link"]), SceneEntityCfg("robot_2", body_names=[".*_hand_link"]), SceneEntityCfg("robot_3", body_names=[".*_hand_link"])]})
    lift_progress = RewTerm(func=coop.object_lift_progress, weight=2.0, params={"height": 0.04})
    lifting_object = RewTerm(func=coop.object_is_lifted, weight=15.0, params={"height": 0.04})
    object_tilt = RewTerm(func=coop.object_tilt_l2, weight=-2.0)
    object_xy = RewTerm(func=coop.object_xy_drift_l2, weight=-1.0)
    object_vel = RewTerm(func=coop.object_lin_vel_l2, weight=-0.05)
    still_alive = RewTerm(func=coop.still_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    flat_0 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_0")})
    torques_0 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_0")})
    base_contact_0 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_0", body_names="base"), "threshold": 1.0})
    flat_1 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_1")})
    torques_1 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_1")})
    base_contact_1 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_1", body_names="base"), "threshold": 1.0})
    flat_2 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_2")})
    torques_2 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_2")})
    base_contact_2 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_2", body_names="base"), "threshold": 1.0})
    flat_3 = RewTerm(func=mdp.flat_orientation_l2, weight=-1.0, params={"asset_cfg": SceneEntityCfg("robot_3")})
    torques_3 = RewTerm(func=mdp.joint_torques_l2, weight=-2.0e-4, params={"asset_cfg": SceneEntityCfg("robot_3")})
    base_contact_3 = RewTerm(func=mdp.undesired_contacts, weight=-1.0, params={"sensor_cfg": SceneEntityCfg("contact_3", body_names="base"), "threshold": 1.0})


@configclass
class Crew4DepthTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=coop.any_fallen, params={"limit_angle": 0.78, "robot_names": ['robot_0', 'robot_1', 'robot_2', 'robot_3']})


@configclass
class Crew4DepthEventsCfg:
    reset_joints_0 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_0")})
    reset_root_0 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_0")})
    reset_joints_1 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_1")})
    reset_root_1 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_1")})
    reset_joints_2 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_2")})
    reset_root_2 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_2")})
    reset_joints_3 = EventTerm(func=mdp.reset_joints_by_offset, mode="reset",
        params={"position_range": (-0.02, 0.02), "velocity_range": (-0.05, 0.05), "asset_cfg": SceneEntityCfg("robot_3")})
    reset_root_3 = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.05, 0.05)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("robot_3")})
    reset_object = EventTerm(func=mdp.reset_root_state_uniform, mode="reset",
        params={"pose_range": {"x": (-0.01, 0.01), "y": (-0.01, 0.01)}, "velocity_range": {}, "asset_cfg": SceneEntityCfg("object")})


@configclass
class Crew4DepthCfg(CoopLiftEnvCfg):
    """Crew of 4 on one payload, each carrying a depth camera."""

    scene: Crew4DepthSceneCfg = Crew4DepthSceneCfg(num_envs=1024, env_spacing=4.0)
    observations: Crew4DepthObservationsCfg = Crew4DepthObservationsCfg()
    actions: Crew4DepthActionsCfg = Crew4DepthActionsCfg()
    rewards: Crew4DepthRewardsCfg = Crew4DepthRewardsCfg()
    terminations: Crew4DepthTerminationsCfg = Crew4DepthTerminationsCfg()
    events: Crew4DepthEventsCfg = Crew4DepthEventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    crew_size: int = 4

    def __post_init__(self):
        super().__post_init__()
        # Contact ring tracks the payload this crew actually holds.
        self.contact_offset = 0.196389
        self.object_spawn_z = 0.176389
