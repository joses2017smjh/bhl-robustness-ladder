"""The three redesigned tasks, each in blind / depth / RGB.

What changed from `coop_lift_env_cfg`, and why:

* The payload starts on a plinth at `GRASP_Z = 0.30 m` instead of on the floor.
  Measured (`scripts/bench/task_gate.py`): reaching it takes a 15.5 cm squat,
  and it is unreachable from the 41 cm collapse the old policies learned. The
  collapse stops paying without a height penalty having to outweigh a
  15.0-weight lift bonus.
* Every task has a terminal success state. The old lift had none, so there was
  nothing to report a success *rate* over.
* All three vision conditions render through `TiledCameraCfg` on the same
  stack, so the comparison is about the sensor rather than the simulator.

These are v60-only. Isaac Sim 5.1's RTX renderer segfaults on this cluster, so
`ENABLE_CAMERAS=1 BHL_STACK=v60` is not optional for the sighted arms -- and the
blind arm runs there too, or it would not be comparable to them.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
import torch.nn.functional as F
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass

from bhl_robust.reach_band import GRASP_Z
from bhl_robust.tasks import furniture, task_v2_mdp as v2
from bhl_robust.tasks.coop_lift_env_cfg import CoopLiftEnvCfg, _COLLISION, _RIGID, _object, _robot
from bhl_robust.tasks.rgb_env_cfg import CAM_POS, CAM_ROT, CAM_RANGE

CAM_RES = 32

# Layout constants, all of them checked by G-T2 before anything trained on them.
SHELF_X, SHELF_DECK, SHELF_SLOT = 1.2, 0.38, 0.34
NET_X, NET_RIM, NET_MOUTH = 3.5, 0.60, 0.70
WALL_X, WALL_CONTACT = 1.0, 0.50


def _cam(prim: str, data_type: str) -> TiledCameraCfg:
    """One robot's head camera. Same pose for depth and colour."""
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}/base/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=CAM_POS, rot=CAM_ROT, convention="world"),
        data_types=[data_type],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, CAM_RANGE),
        ),
        width=CAM_RES, height=CAM_RES,
    )


# --------------------------------------------------------------- rsl-rl 5.x
# Isaac Lab 3.0 pins rsl-rl-lib 5.0.1, whose runner config is a different shape
# from the 3.0.1 schema every v51 task uses. 5.x reads `cfg["actor"]["class_name"]`
# and `cfg["critic"]`, where 2.x had a single `policy` carrying
# `actor_hidden_dims` and `critic_hidden_dims`. Reusing `CoopLiftPPORunnerCfg`
# here fails with `KeyError: 'class_name'` before a single iteration runs.
#
# So the v2 tasks get their own runner config rather than the v51 one being
# migrated: v51 still trains against rsl-rl 3.0.1 and every published number in
# this repo came from it. Two schemas, two configs, neither pretending to be the
# other.
#
# Network shape is held identical to the v51 baseline -- [256, 256, 128] with
# ELU -- so the v2 results differ from the old coop ones by task and stack, not
# by capacity.
try:
    from isaaclab_rl.rsl_rl import (
        RslRlMLPModelCfg,
        RslRlOnPolicyRunnerCfg,
        RslRlPpoAlgorithmCfg,
    )

    _HIDDEN = [256, 256, 128]

    @configclass
    class TaskV2PPORunnerCfg(RslRlOnPolicyRunnerCfg):
        """PPO for the redesigned tasks, in the rsl-rl 5.x schema."""

        num_steps_per_env = 24
        max_iterations = 8000
        save_interval = 200
        experiment_name = "task_v2"
        empirical_normalization = False
        # The env exposes "policy" and "critic"; map them onto the sets 5.x
        # names. Without this the runner cannot tell which group feeds which
        # network, and the asymmetric actor-critic silently becomes symmetric.
        obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        # The actor needs `distribution_cfg`; that is what makes it stochastic.
        # Without it the runner still asks for a stochastic model and rsl-rl
        # raises `MLPModel.__init__() got an unexpected keyword argument
        # 'stochastic'` -- which reads like a version mismatch and is really a
        # missing field. The critic is deterministic and takes none.
        actor = RslRlMLPModelCfg(
            hidden_dims=_HIDDEN,
            activation="elu",
            obs_normalization=False,
            distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
        )
        critic = RslRlMLPModelCfg(
            hidden_dims=_HIDDEN,
            activation="elu",
            obs_normalization=False,
        )
        algorithm = RslRlPpoAlgorithmCfg(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=0.005,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
        )

    _V2_RUNNER = TaskV2PPORunnerCfg
except ImportError:                                              # v51 stack
    # RslRlMLPModelCfg does not exist on isaaclab_rl 2.x. The v2 tasks are
    # v60-only anyway, so on v51 they register with the old runner and simply
    # are not meant to be trained.
    from bhl_robust.tasks.coop_lift_env_cfg import CoopLiftPPORunnerCfg as _V2_RUNNER


CAM_POOL = 4          # 32x32 -> 8x8, the width section 6's depth arm used


def cam_depth_obs(env, sensor_cfg: SceneEntityCfg, pool: int = CAM_POOL,
                  clip: float = CAM_RANGE):
    """One robot's rendered depth, pooled and scaled to roughly [0, 1]."""
    d = env.scene[sensor_cfg.name].data.output["distance_to_image_plane"]
    if d.ndim == 3:
        d = d.unsqueeze(-1)
    d = d.permute(0, 3, 1, 2).nan_to_num(nan=clip, posinf=clip)
    return (F.avg_pool2d(d, pool).flatten(1) / clip).clamp(0.0, 1.0)


def cam_rgb_obs(env, sensor_cfg: SceneEntityCfg, pool: int = CAM_POOL):
    """One robot's colour view, pooled per channel and scaled to [0, 1].

    Pooled to the same 8x8 grid as the depth arm and kept in three channels, so
    colour carries 3x the numbers depth does at the same spatial resolution.
    That asymmetry is the experiment: if RGB wins, the question is whether it
    won on colour or merely on width.
    """
    c = env.scene[sensor_cfg.name].data.output["rgb"].float()
    c = c.permute(0, 3, 1, 2)[:, :3]
    return (F.avg_pool2d(c, pool).flatten(1) / 255.0).clamp(0.0, 1.0)


@configclass
class _TaskV2Base(CoopLiftEnvCfg):
    """Shared: payload on a plinth, success terminates, longer episode."""

    vision: str = "blind"          # blind | depth | rgb
    target_x: float = SHELF_X

    def __post_init__(self):
        super().__post_init__()
        # A carry plus a placement does not fit in the lift's 8 s.
        self.episode_length_s = 20.0
        self.object_spawn_z = GRASP_Z

    def _add_cameras(self):
        """Mount the cameras AND wire them into the observation.

        Both halves matter. The first cut added the sensors and no observation
        terms, so all three variants reported an identical 194-wide observation
        -- the sighted arms carried cameras nothing ever read, and would have
        trained as three copies of the blind arm while looking like a vision
        experiment. The smoke test's obs column is what caught it.
        """
        if self.vision == "blind":
            return
        depth = self.vision == "depth"
        dt = "distance_to_image_plane" if depth else "rgb"
        self.scene.cam_a = _cam("robot_a", dt)
        self.scene.cam_b = _cam("robot_b", dt)
        fn = cam_depth_obs if depth else cam_rgb_obs
        for side in ("a", "b"):
            setattr(self.observations.policy, f"cam_{side}",
                    ObsTerm(func=fn, params={"sensor_cfg": SceneEntityCfg(f"cam_{side}")}))


@configclass
class CubeToShelfCfg(_TaskV2Base):
    """0.28 m cube from a plinth into a shelf slot with 6 cm of clearance."""

    object_kind: str = "cube"
    contact_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    contact_offset: float = 0.16
    target_x: float = SHELF_X

    def __post_init__(self):
        super().__post_init__()
        self.scene.object = _object(
            sim_utils.CuboidCfg(
                size=(0.28, 0.28, 0.28),
                rigid_props=_RIGID, collision_props=_COLLISION,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.165, 0.471, 0.839)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.4, dynamic_friction=1.2),
            ),
            z=GRASP_Z,
        )
        self.scene.plinth = furniture.plinth(0.14, top=0.26)
        for i, part in enumerate(furniture.shelf(SHELF_SLOT, SHELF_DECK, SHELF_X)):
            setattr(self.scene, f"shelf_{i}", part)
        self.rewards.carry = RewTerm(
            func=v2.carry_progress, params={"target_x": SHELF_X}, weight=3.0)
        self.rewards.placed = RewTerm(
            func=v2.cube_in_slot,
            params={"slot": SHELF_SLOT, "deck_z": SHELF_DECK, "shelf_x": SHELF_X},
            weight=200.0)
        self.terminations.success = DoneTerm(
            func=v2.cube_in_slot,
            params={"slot": SHELF_SLOT, "deck_z": SHELF_DECK, "shelf_x": SHELF_X})
        self._add_cameras()


@configclass
class BallToNetCfg(_TaskV2Base):
    """r = 0.18 m ball carried to a release zone and thrown into a net."""

    object_kind: str = "ball"
    contact_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    contact_offset: float = 0.18
    target_x: float = NET_X

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot_a = _robot("{ENV_REGEX_NS}/robot_a", (0.0, 0.47, 0.0), (0.7071, 0.0, 0.0, -0.7071))
        self.scene.robot_b = _robot("{ENV_REGEX_NS}/robot_b", (0.0, -0.47, 0.0), (0.7071, 0.0, 0.0, 0.7071))
        self.scene.object = _object(
            sim_utils.SphereCfg(
                radius=0.18,
                rigid_props=_RIGID, collision_props=_COLLISION,
                mass_props=sim_utils.MassPropertiesCfg(mass=0.6),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.18, 0.22)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=0.7, dynamic_friction=0.6, restitution=0.3),
            ),
            z=GRASP_Z,
        )
        self.scene.plinth = furniture.plinth(0.18, top=0.22)
        for i, part in enumerate(furniture.net(NET_MOUTH, NET_RIM, NET_X)):
            setattr(self.scene, f"net_{i}", part)
        self.rewards.carry = RewTerm(
            func=v2.carry_progress, params={"target_x": NET_X}, weight=2.0)
        self.rewards.toward = RewTerm(
            func=v2.ball_toward_net, params={"net_x": NET_X}, weight=6.0)
        self.rewards.scored = RewTerm(
            func=v2.ball_in_net,
            params={"mouth": NET_MOUTH, "rim_z": NET_RIM, "net_x": NET_X},
            weight=200.0)
        self.terminations.success = DoneTerm(
            func=v2.ball_in_net,
            params={"mouth": NET_MOUTH, "rim_z": NET_RIM, "net_x": NET_X})
        self._add_cameras()


@configclass
class PlankToWallCfg(_TaskV2Base):
    """1.5 m plank off two supports and leaned against a wall at 50-80 degrees."""

    object_kind: str = "ladder"
    contact_axis: tuple[float, float, float] = (1.0, 0.0, 0.0)
    contact_offset: float = 0.75
    target_x: float = WALL_X

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot_a = _robot("{ENV_REGEX_NS}/robot_a", (-0.85, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        self.scene.robot_b = _robot("{ENV_REGEX_NS}/robot_b", (0.85, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
        self.scene.object = _object(
            sim_utils.CuboidCfg(
                size=(1.50, 0.40, 0.08),
                rigid_props=_RIGID, collision_props=_COLLISION,
                mass_props=sim_utils.MassPropertiesCfg(mass=1.1),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.72, 0.45, 0.18)),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.3, dynamic_friction=1.1),
            ),
            z=GRASP_Z,
        )
        # Two supports, not one plinth: a 1.5 m plank on a 0.26 m pedestal at
        # its centre would see-saw, and the task would be balancing rather than
        # lifting before a robot had touched it.
        h = GRASP_Z - 0.04
        for i, x in enumerate((-0.55, 0.55)):
            setattr(self.scene, f"support_{i}",
                    furniture._box(f"support_{i}", (0.16, 0.30, h), (x, 0.0, h / 2.0)))
        self.scene.wall = furniture.wall(WALL_X)
        self.rewards.leaned = RewTerm(
            func=v2.plank_leaned, params={"wall_x": WALL_X, "contact_z": WALL_CONTACT},
            weight=200.0)
        self.terminations.success = DoneTerm(
            func=v2.plank_leaned, params={"wall_x": WALL_X, "contact_z": WALL_CONTACT})
        self._add_cameras()


def _variants(base, name):
    """blind / depth / rgb subclasses of one task.

    Each class is bound into this module's namespace as well as returned.
    Hydra pickles the env config, and pickle finds a class by looking up
    `module.__qualname__` -- so a class built with `type()` and never assigned
    anywhere fails with

        Can't pickle <class '...CubeToShelfBlindCfg'>: attribute lookup
        CubeToShelfBlindCfg on bhl_robust.tasks.task_v2_env_cfg failed

    which is what killed every v2 training arm while the env smoke passed 9/9,
    because building an env never pickles it.
    """
    out = {}
    for v in ("blind", "depth", "rgb"):
        cls_name = f"{name}{v.capitalize()}Cfg"
        cls = configclass(type(cls_name, (base,), {
            "__doc__": f"{name}, {v} observation.",
            "vision": v,
            "__module__": __name__,
            "__qualname__": cls_name,
        }))
        globals()[cls_name] = cls
        out[v] = cls
    return out


CUBE_VARIANTS = _variants(CubeToShelfCfg, "CubeToShelf")
BALL_VARIANTS = _variants(BallToNetCfg, "BallToNet")
PLANK_VARIANTS = _variants(PlankToWallCfg, "PlankToWall")


@configclass
class BallToNetSoloCfg(BallToNetCfg):
    """The control the ball task needs: one robot, same net, same ball.

    The ball is r = 0.18 m, so its two contact points sit 0.36 m apart -- inside
    a single robot's 0.355 m hand span, measured. That means one robot can
    plausibly bracket it alone, and if it can, the two-robot result is not a
    cooperation result at all. Section 5 learned this the expensive way: the
    cube lift looked cooperative until a three-robot rollout put one robot on a
    crate by itself and it lifted just as well.

    So the solo arm is not an ablation to run if there is time. It is the arm
    that decides whether the paired number means anything, and it should be read
    before the paired number is quoted anywhere.

    Implementation keeps robot_b in the scene but removes it from the action
    manager, so the physics, the observation width and the reward terms are
    untouched and the only difference is that nobody is driving the second
    robot. Deleting it instead would change the observation layout and make the
    two arms incomparable -- which is the mistake that would quietly turn this
    control into a different experiment.
    """

    def __post_init__(self):
        super().__post_init__()
        self.actions.joint_pos_b = None

BALL_SOLO_VARIANTS = _variants(BallToNetSoloCfg, "BallToNetSolo")

# ---------------------------------------------------------------- grippers
# The same three tasks on the 24-DoF asset, so the hands can actually close.
#
# Every manipulation result in this repo was produced by a robot whose hands are
# welded shut (`docs/GRIPPER.md`), which is not a property of the machine -- the
# hardware has two grippers and upstream drives them. These variants are the
# first runs where a policy can perform the grasp the robot really does: lay the
# open hand over the object, close, and let finger and palm retain it
# geometrically rather than by friction.
#
# Separate ids rather than a flag, so the welded-hand arms stay runnable as the
# control. "The same task with and without a working hand" is the comparison
# that prices the asset bug, and it needs both sides.

def _gripper_variant(base, name):
    """One task on the gripper asset, blind/depth/rgb."""
    out = {}
    for v in ("blind", "depth", "rgb"):
        cls_name = f"{name}Gripper{v.capitalize()}Cfg"

        def _post(self, _v=v):
            super(type(self), self).__post_init__()
            from bhl_robust.gripper_asset import (
                HUMANOID_LITE_GRIPPER_JOINT_ORDER, get_gripper_cfg,
            )
            from isaaclab.managers import SceneEntityCfg
            cfg = get_gripper_cfg()
            for side, robot in (("a", self.scene.robot_a), ("b", self.scene.robot_b)):
                robot.spawn = cfg.spawn.replace()
                jp = dict(robot.init_state.joint_pos)
                jp.update({j: 0.0 for j in ("arm_left_gripper_joint",
                                            "arm_right_gripper_joint")})
                robot.init_state = robot.init_state.replace(joint_pos=jp)
                robot.actuators = dict(cfg.actuators)
            # Actions and the joint-indexed observations move 22 -> 24 together;
            # driving 22 of 24 joints would leave the grippers inert and the
            # variant indistinguishable from its control.
            order = HUMANOID_LITE_GRIPPER_JOINT_ORDER
            self.actions.joint_pos_a.joint_names = order
            self.actions.joint_pos_b.joint_names = order
            for grp in (self.observations.policy, self.observations.critic):
                for term in ("joint_pos_a", "joint_vel_a", "track_err_a"):
                    tc = getattr(grp, term, None)
                    if tc is not None and "asset_cfg" in tc.params:
                        tc.params["asset_cfg"] = SceneEntityCfg(
                            "robot_a", joint_names=order, preserve_order=True)
                for term in ("joint_pos_b", "joint_vel_b", "track_err_b"):
                    tc = getattr(grp, term, None)
                    if tc is not None and "asset_cfg" in tc.params:
                        tc.params["asset_cfg"] = SceneEntityCfg(
                            "robot_b", joint_names=order, preserve_order=True)

        cls = configclass(type(cls_name, (base,), {
            "__doc__": f"{name} on the 24-DoF gripper asset, {v} observation.",
            "vision": v,
            "__module__": __name__,
            "__qualname__": cls_name,
            "__post_init__": _post,
        }))
        globals()[cls_name] = cls
        out[v] = cls
    return out


CUBE_GRIPPER_VARIANTS = _gripper_variant(CubeToShelfCfg, "CubeToShelf")
BALL_GRIPPER_VARIANTS = _gripper_variant(BallToNetCfg, "BallToNet")
PLANK_GRIPPER_VARIANTS = _gripper_variant(PlankToWallCfg, "PlankToWall")
