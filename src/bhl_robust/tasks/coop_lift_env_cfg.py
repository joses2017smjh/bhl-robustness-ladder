"""Two 22-DoF BHLs, one object, pinch-then-lift.

Three gym ids share this config; only the object geom, the spawn formation,
and the constellation axis change. That is the sample-efficient split: one
code path, three objects, no shared policy until a lift actually appears.

Not a locomotion overlay. Velocity tracking, feet-air-time, and
``joint_deviation_arms`` all fight a squat-and-pinch, so they are absent.
Spawn is the crouch-hold from the kinematics GIF, not the standing default.
The critic sees object twist and both robots; the actor sees proprioception,
object-in-root, and PD tracking residual — asymmetric actor-critic, the
training-time form of teacher-student. Joint limits are URDF walls, not
reward terms, so the 36 cm adduction cap is physics.
"""
from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
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
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

import isaaclab.envs.mdp as mdp
from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import (
    HUMANOID_LITE_CFG,
    HUMANOID_LITE_JOINTS,
)

from . import coop_lift_mdp as coop

_HANDS = ["arm_left_hand_link", "arm_right_hand_link"]
_YAW_M90 = (0.70710678, 0.0, 0.0, -0.70710678)
_YAW_P90 = (0.70710678, 0.0, 0.0, 0.70710678)
_YAW_180 = (0.0, 0.0, 0.0, 1.0)

# Crouch + side-hold from the scripted GIF. Standing spawn left hands ~0.5 m
# above a floor object, which saturates a tanh(·/0.15) kernel. Pelvis drop is
# the sagittal shortening of a 0.12 m thigh + 0.16 m shank at these angles,
# not a guessed number; z of the caller's pos is ignored so every object
# uses the same plant.
_PINCH_ROOT_Z = -0.07
_PINCH_JOINT_POS = {
    **dict(HUMANOID_LITE_CFG.init_state.joint_pos),
    "leg_left_hip_pitch_joint": -0.85,
    "leg_right_hip_pitch_joint": -0.85,
    "leg_left_knee_pitch_joint": 1.45,
    "leg_right_knee_pitch_joint": 1.45,
    "leg_left_ankle_pitch_joint": -0.55,
    "leg_right_ankle_pitch_joint": -0.55,
    "arm_left_shoulder_roll_joint": -0.26,
    "arm_right_shoulder_roll_joint": 0.26,
    "arm_left_shoulder_pitch_joint": -0.55,
    "arm_right_shoulder_pitch_joint": 0.55,
    "arm_left_elbow_pitch_joint": 0.90,
    "arm_right_elbow_pitch_joint": -0.90,
}

def _robot(prim: str, pos: tuple[float, float, float], rot: tuple[float, float, float, float]) -> ArticulationCfg:
    return HUMANOID_LITE_CFG.replace(
        prim_path=prim,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(pos[0], pos[1], _PINCH_ROOT_Z),
            rot=rot,
            joint_pos=_PINCH_JOINT_POS,
            joint_vel={".*": 0.0},
        ),
    )


def _object(spawn: sim_utils.CuboidCfg | sim_utils.SphereCfg, z: float) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/object",
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, z), rot=(1.0, 0.0, 0.0, 0.0)),
        spawn=spawn,
    )


_RIGID = sim_utils.RigidBodyPropertiesCfg(
    disable_gravity=False,
    max_depenetration_velocity=1.0,
    solver_position_iteration_count=16,
    solver_velocity_iteration_count=4,
)
_COLLISION = sim_utils.CollisionPropertiesCfg()


@configclass
class CoopLiftSceneCfg(InteractiveSceneCfg):
    """Two humanoids, one object, flat ground. No table: the object starts on the floor."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )
    robot_a: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_a", (0.0, 0.48, 0.0), _YAW_M90)
    robot_b: ArticulationCfg = _robot("{ENV_REGEX_NS}/robot_b", (0.0, -0.48, 0.0), _YAW_P90)
    object: RigidObjectCfg = _object(
        sim_utils.CuboidCfg(
            size=(0.28, 0.28, 0.28),
            rigid_props=_RIGID,
            collision_props=_COLLISION,
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.95, 0.48, 0.12)),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.4, dynamic_friction=1.2),
        ),
        z=0.14,
    )
    contact_a = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_a/.*", history_length=3, track_air_time=True)
    contact_b = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/robot_b/.*", history_length=3, track_air_time=True)
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity_a = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot_a")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        projected_gravity_b = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot_b")},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        base_ang_vel_a = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot_a")},
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        base_ang_vel_b = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot_b")},
            noise=Unoise(n_min=-0.3, n_max=0.3),
        )
        joint_pos_a = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_a", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_pos_b = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot_b", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-0.05, n_max=0.05),
        )
        joint_vel_a = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_a", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
        )
        joint_vel_b = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot_b", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)},
            noise=Unoise(n_min=-2.0, n_max=2.0),
        )
        object_pos_a = ObsTerm(
            func=coop.object_pos_in_root,
            params={"robot_cfg": SceneEntityCfg("robot_a")},
        )
        object_pos_b = ObsTerm(
            func=coop.object_pos_in_root,
            params={"robot_cfg": SceneEntityCfg("robot_b")},
        )
        track_err_a = ObsTerm(
            func=coop.joint_target_error,
            params={"asset_cfg": SceneEntityCfg("robot_a", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)},
        )
        track_err_b = ObsTerm(
            func=coop.joint_target_error,
            params={"asset_cfg": SceneEntityCfg("robot_b", joint_names=HUMANOID_LITE_JOINTS, preserve_order=True)},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel_a = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_a")})
        base_lin_vel_b = ObsTerm(func=mdp.base_lin_vel, params={"asset_cfg": SceneEntityCfg("robot_b")})
        object_lin_vel = ObsTerm(func=coop.object_lin_vel_w)
        object_ang_vel = ObsTerm(func=coop.object_ang_vel_w)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ActionsCfg:
    joint_pos_a = mdp.JointPositionActionCfg(
        asset_name="robot_a",
        joint_names=HUMANOID_LITE_JOINTS,
        scale=0.25,
        preserve_order=True,
        use_default_offset=True,
    )
    joint_pos_b = mdp.JointPositionActionCfg(
        asset_name="robot_b",
        joint_names=HUMANOID_LITE_JOINTS,
        scale=0.25,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class RewardsCfg:
    """Isaac Lab lift weights, two-scale constellation, pinch-gated height.

    Coarse reach is always on so a 0.5 m spawn gap still has a gradient.
    Fine reach is the pinch. Clamp rewards opposing force through the object
    (the fingerless grasp). Height terms multiply by the pinch kernel —
    without that gate the ladder run paid 3.4 for a toss (reaching 0).
    Sparse lift is 15x heavier than coarse reach, matching Franka lift.
    Tilt penalty is how two robots stay synchronous; a toss spins.
    """

    reaching_coarse = RewTerm(
        func=coop.constellation_reach,
        params={"std": 0.40, "robot_a_cfg": SceneEntityCfg("robot_a", body_names=_HANDS),
                "robot_b_cfg": SceneEntityCfg("robot_b", body_names=_HANDS)},
        weight=1.0,
    )
    reaching_fine = RewTerm(
        func=coop.constellation_reach,
        params={"std": 0.12, "robot_a_cfg": SceneEntityCfg("robot_a", body_names=_HANDS),
                "robot_b_cfg": SceneEntityCfg("robot_b", body_names=_HANDS)},
        weight=1.0,
    )
    opposing_clamp = RewTerm(
        func=coop.opposing_clamp,
        params={
            "robot_a_cfg": SceneEntityCfg("robot_a", body_names=_HANDS),
            "robot_b_cfg": SceneEntityCfg("robot_b", body_names=_HANDS),
        },
        weight=1.5,
    )
    lift_progress = RewTerm(func=coop.object_lift_progress, weight=2.0)
    lifting_object = RewTerm(
        func=coop.object_is_lifted,
        params={"minimal_height": 0.04},
        weight=15.0,
    )
    object_xy = RewTerm(func=coop.object_xy_drift_l2, weight=-0.8)
    object_vel = RewTerm(func=coop.object_lin_vel_l2, weight=-0.05)
    object_tilt = RewTerm(func=coop.object_tilt_l2, weight=-1.5)
    still_alive = RewTerm(func=coop.still_alive, weight=1.0)
    termination_penalty = RewTerm(func=mdp.is_terminated, weight=-10.0)
    flat_a = RewTerm(
        func=mdp.flat_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot_a")},
        weight=-1.0,
    )
    flat_b = RewTerm(
        func=mdp.flat_orientation_l2,
        params={"asset_cfg": SceneEntityCfg("robot_b")},
        weight=-1.0,
    )
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    torques_a = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot_a", joint_names=HUMANOID_LITE_JOINTS)},
        weight=-2.0e-5,
    )
    torques_b = RewTerm(
        func=mdp.joint_torques_l2,
        params={"asset_cfg": SceneEntityCfg("robot_b", joint_names=HUMANOID_LITE_JOINTS)},
        weight=-2.0e-5,
    )
    base_contact_a = RewTerm(
        func=mdp.undesired_contacts,
        params={"sensor_cfg": SceneEntityCfg("contact_a", body_names="base"), "threshold": 1.0},
        weight=-1.0,
    )
    base_contact_b = RewTerm(
        func=mdp.undesired_contacts,
        params={"sensor_cfg": SceneEntityCfg("contact_b", body_names="base"), "threshold": 1.0},
        weight=-1.0,
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    fallen = DoneTerm(func=coop.either_fallen, params={"limit_angle": 0.78})


@configclass
class EventsCfg:
    """Reset to the pinch formation with a small jitter, not a random sprawl.

    Upstream loco uses joint scale (0.5, 1.5), which puts a squat-and-pick
    policy in splits. Offset around the default standing pose instead.
    """

    reset_joints_a = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_a"),
            "position_range": (-0.08, 0.08),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_joints_b = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_b"),
            "position_range": (-0.08, 0.08),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_root_a = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_a"),
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.12, 0.12)},
            "velocity_range": {},
        },
    )
    reset_root_b = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot_b"),
            "pose_range": {"x": (-0.04, 0.04), "y": (-0.04, 0.04), "yaw": (-0.12, 0.12)},
            "velocity_range": {},
        },
    )
    reset_object = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "pose_range": {"x": (-0.03, 0.03), "y": (-0.03, 0.03)},
            "velocity_range": {},
        },
    )


@configclass
class CurriculumCfg:
    lift_height = CurrTerm(
        func=coop.lift_height_curriculum,
        params={
            "term_name": "lifting_object",
            "step": 0.02,
            "min_height": 0.04,
            "max_height": 0.22,
            "success_rate_target": 0.35,
        },
    )


@configclass
class CoopLiftEnvCfg(ManagerBasedRLEnvCfg):
    """Shared dual-humanoid lift. Subclasses only swap the object and formation."""

    scene: CoopLiftSceneCfg = CoopLiftSceneCfg(num_envs=1024, env_spacing=4.0)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventsCfg = EventsCfg()
    curriculum: CurriculumCfg = CurriculumCfg()
    commands = None

    object_kind: str = "cube"
    contact_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    contact_offset: float = 0.16
    object_spawn_z: float = 0.14
    lift_success_z: float = 0.04

    def __post_init__(self):
        self.decimation = 8
        self.episode_length_s = 8.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.disable_contact_processing = False
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 20 * 2**15
        self.sim.physx.bounce_threshold_velocity = 0.01
        # Joint limits live in the URDF. An out-of-range PD target is clipped
        # by the articulation; the 36 cm adduction wall is physics, not a
        # reward term.
        if self.scene.contact_a is not None:
            self.scene.contact_a.update_period = self.sim.dt
        if self.scene.contact_b is not None:
            self.scene.contact_b.update_period = self.sim.dt


@configclass
class CoopLiftCubeCfg(CoopLiftEnvCfg):
    """0.28 m cube, robots facing each other on ±y. Side pinch."""

    object_kind: str = "cube"
    contact_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    contact_offset: float = 0.16
    object_spawn_z: float = 0.14


@configclass
class CoopLiftLadderCfg(CoopLiftEnvCfg):
    """Ladder envelope: 1.5 m × 0.40 m × 0.08 m plank.

    A real 6-foot ladder is about that wide. The 40 cm face is what the
    shoulders can actually close on; a 7 cm rail is not a grasp this
    morphology can make.
    """

    object_kind: str = "ladder"
    contact_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    contact_offset: float = 0.75
    object_spawn_z: float = 0.04

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot_a = _robot("{ENV_REGEX_NS}/robot_a", (-0.85, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        self.scene.robot_b = _robot("{ENV_REGEX_NS}/robot_b", (0.85, 0.0, 0.0), _YAW_180)
        self.scene.object = _object(
            sim_utils.CuboidCfg(
                size=(1.50, 0.40, 0.08),
                rigid_props=_RIGID,
                collision_props=_COLLISION,
                mass_props=sim_utils.MassPropertiesCfg(mass=1.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.45, 0.18)),
                physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.3, dynamic_friction=1.1),
            ),
            z=self.object_spawn_z,
        )
        self.scene.env_spacing = 5.0


@configclass
class CoopLiftBallCfg(CoopLiftEnvCfg):
    """~65 cm yoga ball. Low friction, robots hug from ±y."""

    object_kind: str = "ball"
    contact_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    contact_offset: float = 0.33
    object_spawn_z: float = 0.33

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot_a = _robot("{ENV_REGEX_NS}/robot_a", (0.0, 0.62, 0.0), _YAW_M90)
        self.scene.robot_b = _robot("{ENV_REGEX_NS}/robot_b", (0.0, -0.62, 0.0), _YAW_P90)
        self.scene.object = _object(
            sim_utils.SphereCfg(
                radius=0.33,
                rigid_props=_RIGID,
                collision_props=_COLLISION,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.7),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.18, 0.22)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.55, dynamic_friction=0.45, restitution=0.2
                ),
            ),
            z=self.object_spawn_z,
        )


@configclass
class CoopLiftPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """Same PPO as the 22-DoF loco runs, slightly wider net for 44 actions."""

    num_steps_per_env = 24
    max_iterations = 4000
    save_interval = 100
    experiment_name = "coop_lift"
    empirical_normalization = False
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_hidden_dims=[256, 256, 128],
        critic_hidden_dims=[256, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.01,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
