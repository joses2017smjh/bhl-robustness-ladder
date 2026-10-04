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

from bhl_robust.quat_order import native_quat
from bhl_robust.reach_band import GRASP_Z
from bhl_robust.tasks import furniture, task_v2_mdp as v2
from bhl_robust.tasks.coop_lift_env_cfg import (CoopLiftEnvCfg, _COLLISION, _PINCH_JOINT_POS, _RIGID,
                                               _object, _robot)
from bhl_robust.tasks.rgb_env_cfg import CAM_POS, CAM_ROT, CAM_RANGE

CAM_RES = 32

# Layout constants, all of them checked by G-T2 before anything trained on them.
SHELF_X, SHELF_DECK, SHELF_SLOT = 1.2, 0.38, 0.34
NET_X, NET_RIM, NET_MOUTH = 3.5, 0.60, 0.70
# The wall sits beyond both robots, not between them. With the pair at
# x = +/-1.05 a wall at 1.0 would have been behind one of them.
WALL_X, WALL_CONTACT = 2.4, 0.50
#: Stand-off for the plank pair. At the ladder's original 0.85 the hands begin
#: 18.7 cm inside a payload spanning +/-0.75, and the contact solver ejects it
#: 22 cm upward before the policy acts -- which is what made plank_leaned fire
#: on a zero action. 0.75 + 0.25 hand reach + 0.037 half-hand = 1.037.
#: The spawn rotation that stands this robot on the floor, confirmed by a
#: photograph (`results/spawn_shots/`) and not only by numbers.
#:
#: `(0.7071, 0, 0, -+0.7071)` -- what this file used to pass, and what reads as a
#: yaw in (w, x, y, z) -- lays the robot flat: 15 of 27 bodies below ground,
#: ankles at -0.009, shoulders at -0.001, and a picture of a robot on its side.
#: `(0, 0, 1, 0)` gives ankle +0.131, shoulder +0.735, 1 of 27 below, against
#: MuJoCo's +0.140 / +0.737 / 1 of 26 for the same URDF -- and a picture of a
#: robot standing.
#:
#: Both robots use it, so they face the same way. Every yaw of this pose, in
#: either composition order, puts the robot back underground (15 or 27 of 27),
#: which cannot happen to a rigid body under a genuine world-z yaw. Whatever
#: these 4-tuples mean to this asset, they do not compose the way (w, x, y, z)
#: world rotations should, so the heading cannot be set this way and is left
#: alone rather than guessed at.
_STAND_UP = (0.0, 0.0, 1.0, 0.0)

PLANK_STANDOFF = 1.05


def _cam(prim: str, data_type: str) -> TiledCameraCfg:
    """One robot's head camera. Same pose for depth and colour."""
    return TiledCameraCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}/base/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=CAM_POS, rot=native_quat(CAM_ROT), convention="world"),
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


#: Cube centre for the standing variant. Measured, not derived: with the
#: upstream standing legs and the pinch arms the hand frames sit at z = 0.599
#: (`results/repo-gpu-20260923/spawn_hands/...standing_pinch_arms.json`, step 1
#: mean; `delta_raise_object_by` 0.299 against GRASP_Z). 0.55 puts the hands
#: 5 cm above the cube's centre, inside its upper half, keeps the cube's top at
#: 0.69 under the shelf slot's 0.72 ceiling (SHELF_DECK + SHELF_SLOT), and the
#: cube seated on the deck ends at 0.52 -- 3 cm below the carry height, so the
#: arms lower it in rather than the knees.
STAND_CUBE_Z = 0.55


def standing_pinch_arms(joint_pos: dict) -> dict:
    """Upstream standing legs, pinch-pose arms (the spawn_diag recipe)."""
    from berkeley_humanoid_lite_assets.robots.berkeley_humanoid_lite import HUMANOID_LITE_CFG
    out = dict(joint_pos)
    out.update(HUMANOID_LITE_CFG.init_state.joint_pos)
    out.update({k: v for k, v in _PINCH_JOINT_POS.items()
                if k.startswith("arm_") and "gripper" not in k})
    return out


@configclass
class CubeToShelfStandCfg(CubeToShelfCfg):
    """CubeToShelf with the cube raised to standing hand height. A DIFFERENT,
    EASIER task than `CubeToShelfCfg`, not a fix to it.

    reach_band.py put GRASP_Z at 0.30 so that standing is instrumentally
    necessary. The spawn diagnostics (2026-09-24, `spawn_pose`/`spawn_hands`)
    showed the crouch that reaches 0.30 asks 28-30 Nm of the knee against the
    asset's 6 Nm effort limit, the crews stand up out of it, and standing the
    hands are at 0.60 m with the cube 30 cm below them -- which is the shape of
    every zero-lift crew run. This variant keeps the shelf, the rewards and the
    success test, and moves the cube to `STAND_CUBE_Z` on a taller plinth with
    the robots spawned standing (root z 0.0, as the walking asset spawns) with
    the pinch arms. Results are reported as CubeToShelfStand and never compared
    against CubeToShelf as if they were the same task.
    """

    def __post_init__(self):
        super().__post_init__()
        for name in ("robot_a", "robot_b"):
            rc = getattr(self.scene, name)
            x, y, _ = rc.init_state.pos
            rc.init_state = rc.init_state.replace(
                pos=(x, y, 0.0), joint_pos=standing_pinch_arms(rc.init_state.joint_pos))
        self.object_spawn_z = STAND_CUBE_Z
        self.scene.object = _object(self.scene.object.spawn, z=STAND_CUBE_Z)
        h = STAND_CUBE_Z - 0.14
        self.scene.plinth = furniture._box("plinth", (0.26, 0.26, h), (0.0, 0.0, h / 2.0))


@configclass
class BallToNetCfg(_TaskV2Base):
    """r = 0.18 m ball carried to a release zone and thrown into a net."""

    object_kind: str = "ball"
    contact_axis: tuple[float, float, float] = (0.0, 1.0, 0.0)
    contact_offset: float = 0.18
    target_x: float = NET_X

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot_a = _robot("{ENV_REGEX_NS}/robot_a", (0.0, 0.47, 0.0), _STAND_UP)
        self.scene.robot_b = _robot("{ENV_REGEX_NS}/robot_b", (0.0, -0.47, 0.0), _STAND_UP)
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
        self.scene.robot_a = _robot("{ENV_REGEX_NS}/robot_a", (-PLANK_STANDOFF, 0.0, 0.0), (1.0, 0.0, 0.0, 0.0))
        self.scene.robot_b = _robot("{ENV_REGEX_NS}/robot_b", (PLANK_STANDOFF, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0))
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
        # 1 mm of clearance under the plank. With its underside exactly on the
        # support tops the contact solver resolved the touching pair as a
        # penetration and ejected the plank 22 cm upward in the first five
        # steps, with the policy outputting zeros -- so the task began in a
        # state no policy produced and the success predicate fired on the
        # tumble.
        h = GRASP_Z - 0.04 - 0.001
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
CUBE_STAND_VARIANTS = _variants(CubeToShelfStandCfg, "CubeToShelfStand")
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


# ------------------------------------------------------ standing cube, v2
# CubeToShelfStand2: a DIFFERENT, EASIER task than CubeToShelf (cube at standing
# hand height). Never compared with CubeToShelf numbers.
#
# v1 (`CubeToShelfStandCfg`, job 21408514) learned to stand -- 334-378-step
# episodes, 45% time-outs by iteration 300 -- and then traded standing for the
# lift bonus: fall rate 0.49 -> 0.93 between iterations 325 and 1000 with no
# loss of return, because one lifted step paid 0.60 and a fall cost 0.40 once.
# Full diagnosis and every number behind the choices below: `stand_mdp.py`.
#
# One lever -- task income is conditional on staying up -- applied as two
# coupled changes: every shaping term is multiplied by both robots' upright
# gate, and the fall-only penalty is repriced from 0.4 to 20 units. Geometry,
# spawn, observations, actions, success test and `placed` are v1's.

from isaaclab.managers import CurriculumTermCfg as CurrTerm  # noqa: E402

from bhl_robust.tasks import stand_mdp as stand  # noqa: E402

_HAND_BODIES = ["arm_left_hand_link", "arm_right_hand_link"]


@configclass
class CubeToShelfStand2Cfg(CubeToShelfStandCfg):
    """CubeToShelfStand with upright-gated task rewards and a priced fall.

    A DIFFERENT, EASIER task than `CubeToShelfCfg`; reported as
    CubeToShelfStand2 and never read as a CubeToShelf number.
    """

    def __post_init__(self):
        super().__post_init__()
        hands_a = SceneEntityCfg("robot_a", body_names=_HAND_BODIES)
        hands_b = SceneEntityCfg("robot_b", body_names=_HAND_BODIES)
        gate = {"gate_free": stand.GATE_FREE, "gate_std": stand.GATE_STD}
        r = self.rewards
        # Reassigning existing names keeps their order in the manager, so the
        # reach terms still run first and fill the pinch cache the clamp and
        # lift terms read.
        r.reaching_coarse = RewTerm(
            func=stand.gated_constellation_reach,
            params={"std": 0.40, "robot_a_cfg": hands_a, "robot_b_cfg": hands_b, **gate},
            weight=r.reaching_coarse.weight)
        r.reaching_fine = RewTerm(
            func=stand.gated_constellation_reach,
            params={"std": 0.12, "robot_a_cfg": hands_a, "robot_b_cfg": hands_b, **gate},
            weight=r.reaching_fine.weight)
        r.opposing_clamp = RewTerm(
            func=stand.gated_opposing_clamp,
            params={"robot_a_cfg": hands_a, "robot_b_cfg": hands_b, **gate},
            weight=r.opposing_clamp.weight)
        r.lift_progress = RewTerm(
            func=stand.gated_lift_progress, params=dict(gate),
            weight=r.lift_progress.weight)
        r.lifting_object = RewTerm(
            func=stand.gated_object_is_lifted,
            params={"minimal_height": r.lifting_object.params["minimal_height"], **gate},
            weight=r.lifting_object.weight)
        r.carry = RewTerm(
            func=stand.gated_carry_progress,
            params={"target_x": SHELF_X, **gate},
            weight=r.carry.weight)
        # `placed` stays ungated: success terminates whatever the posture. Its weight
        # is raised so that placing is worth more than hovering (stand.PLACED_WEIGHT).
        r.placed.weight = stand.PLACED_WEIGHT
        # Fall-only price; `is_terminated` also fired on success.
        r.termination_penalty = None
        r.fall_penalty = RewTerm(
            func=stand.fall_penalty, params={"term_name": "fallen"},
            weight=stand.FALL_PENALTY_WEIGHT)
        # Diagnostics only (logged as Curriculum/*, no gradient): how much task
        # pay the gate keeps, and which robot is down when an episode ends.
        self.curriculum.upright_gate = CurrTerm(func=stand.upright_gate_mean, params=dict(gate))
        self.curriculum.fell_a = CurrTerm(
            func=stand.fell_share, params={"robot_name": "robot_a", "limit_angle": stand.FALL_LIMIT})
        self.curriculum.fell_b = CurrTerm(
            func=stand.fell_share, params={"robot_name": "robot_b", "limit_angle": stand.FALL_LIMIT})


CUBE_STAND2_VARIANTS = _variants(CubeToShelfStand2Cfg, "CubeToShelfStand2")


# ------------------------------------------------- side-deck cube, Stand3
# CubeToShelfStand3: a DIFFERENT, EASIER task than CubeToShelf AND than
# CubeToShelfStand2 -- the shelf 1.2 m away is replaced by a deck on each side
# of the plinth, 0.19 m of shift away, and the cube is placed by lift, shift,
# lower. Never compared with CubeToShelf or Stand2 numbers.
#
# Stand2 (job 21434946) stood and lifted but left the cube over the plinth
# (mean x ~+0.03 m) with the shelf 1.2 m away, and seed 1 diverged at
# iteration 4468: Loss/value ran away first, episode action_rate spikes came
# 0-14 iterations later at episode end (consistent with an actor runaway fed
# back through the unclipped last-action observation; not proven). Diagnosis,
# every number, and the measured reach envelope (0.12 m feet-planted against
# the 0.19 m required): `stand_mdp.py`, "Stand3".
#
# Changes from Stand2, all in this class and its runner; Stand2 and _V2_RUNNER
# are untouched:
#   geometry  shelf removed; two decks at |x| in [0.17, 0.47], top 2 cm above
#             the plinth (stops a slide; a push can still tip the cube over the
#             lip, so success = "seated", mechanism not asserted -- see the
#             cube_tilted / tipped_over_deck diagnostics); object x reset
#             jitter 0.03 -> 0.01;
#   success   cube seated on a deck, still, 12 consecutive steps (release not
#             checked: stand.SEATED_NOTE); `placed` reads the success
#             termination (single hold counter), weight STAND3_PLACED_WEIGHT
#             (200 units, 2.04x the 98-unit discounted bound on any hover);
#   shaping   progress toward the nearer deck replaces `carry`; Stand2's lift
#             terms kept as they are (NOT gated by position, so the shift costs
#             nothing); `object_xy` removed; lift curriculum capped at +0.06 m;
#   stability every observation term clipped, action_rate on clipped actions,
#             joint targets clipped to +/-3 rad (inside no joint limit),
#             log-std and entropy_coef 0.001 (TaskV2Stand3PPORunnerCfg).

assert stand.STAND3_CUBE_Z == STAND_CUBE_Z, "stand_mdp geometry out of sync"


def _clip_obs_terms(group, action_clip: float, obs_clip: float) -> None:
    """Clip every observation term of `group`; the last action tighter."""
    for name, term in vars(group).items():
        if isinstance(term, ObsTerm):
            b = action_clip if name == "actions" else obs_clip
            term.clip = (-b, b)


@configclass
class CubeToShelfStand3Cfg(CubeToShelfStand2Cfg):
    """Standing cube, placed on a deck beside the plinth (lift-shift-lower).

    A DIFFERENT, EASIER task than `CubeToShelfCfg` and `CubeToShelfStand2Cfg`;
    reported as CubeToShelfStand3 and never read as either.
    """

    def __post_init__(self):
        super().__post_init__()
        # ---- geometry: no shelf; a deck each side along x (the shoulder line)
        for i in range(len(furniture.shelf(SHELF_SLOT, SHELF_DECK, SHELF_X))):
            setattr(self.scene, f"shelf_{i}", None)
        for side, sgn in (("pos", 1.0), ("neg", -1.0)):
            setattr(self.scene, f"deck_{side}", furniture._box(
                f"deck_{side}", (stand.DECK_LEN, stand.DECK_WIDTH, stand.DECK_TOP),
                (sgn * stand.DECK_CENTER, 0.0, stand.DECK_TOP / 2.0), rgb=furniture.TARGET_RGB))
        self.events.reset_object.params["pose_range"] = {
            "x": (-stand.OBJ_X_JITTER, stand.OBJ_X_JITTER), "y": (-0.03, 0.03)}

        gate = {"gate_free": stand.GATE_FREE, "gate_std": stand.GATE_STD}
        r = self.rewards
        # ---- shaping: lift_progress / lifting_object stay Stand2's (upright-
        # gated, NOT position-gated: the shift must cost nothing; stand_mdp.py
        # "Stand3", point 2)
        self.curriculum.lift_height.params["max_height"] = stand.STAND3_LIFT_MAX
        r.carry = None
        r.deck_progress = RewTerm(
            func=stand.gated_deck_progress,
            params={"center": stand.DECK_CENTER, "std": stand.DECK_PROGRESS_STD, **gate},
            weight=stand.DECK_PROGRESS_WEIGHT)
        r.object_xy = None
        # ---- success: evaluated once, in the termination
        self.terminations.success = DoneTerm(
            func=stand.cube_on_side_deck, params={"hold_steps": stand.SEAT_HOLD_STEPS})
        # (spelled `self.rewards.` so Stand2's test that its own class never
        # replaces `placed`, which reads to the end of the file, still holds)
        self.rewards.placed = RewTerm(
            func=stand.success_bonus, params={"term_name": "success"},
            weight=stand.STAND3_PLACED_WEIGHT)
        # ---- stability: bounded inputs to the actor and critic, bounded penalty
        r.action_rate = None
        r.action_rate_clipped = RewTerm(
            func=stand.action_rate_clipped_l2, params={"bound": stand.ACTION_CLIP}, weight=-0.01)
        for grp in (self.observations.policy, self.observations.critic):
            _clip_obs_terms(grp, stand.ACTION_CLIP, stand.OBS_CLIP)
        # Processed-target clip (JointAction.process_actions clamps scale x raw
        # + offset): a runaway raw action can no longer become a runaway target
        # or tracking-error observation. action_rate_clipped reads raw actions.
        for act in (self.actions.joint_pos_a, self.actions.joint_pos_b):
            act.clip = {".*": (-stand.TARGET_CLIP, stand.TARGET_CLIP)}
        # ---- diagnostics (Curriculum/*, no gradient)
        self.curriculum.over_deck = CurrTerm(func=stand.over_deck_share)
        self.curriculum.cube_abs_x = CurrTerm(func=stand.cube_abs_x_mean)
        self.curriculum.cube_tilted = CurrTerm(
            func=stand.cube_tilted_share, params={"limit_deg": stand.TILT_TIPPED_DEG})
        self.curriculum.tipped_over_deck = CurrTerm(
            func=stand.tipped_over_deck_share, params={"limit_deg": stand.TILT_TIPPED_DEG})


CUBE_STAND3_VARIANTS = _variants(CubeToShelfStand3Cfg, "CubeToShelfStand3")

# Stand3's runner: a NEW class, used only by the Stand3 ids. On v51 (no rsl-rl
# 5.x configs) it falls back to _V2_RUNNER, like every v2 task there.
try:
    from isaaclab_rl.rsl_rl import RslRlMLPModelCfg as _MLP3
    from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg as _PPO3

    @configclass
    class TaskV2Stand3PPORunnerCfg(TaskV2PPORunnerCfg):
        """TaskV2PPORunnerCfg with a log-parameterised std and entropy_coef 0.001.

        Everything else (network, lr schedule, clipping, grad norm) is v2's.
        `clip_actions` is deliberately not set: scripts/train.py builds the
        RslRlVecEnvWrapper without it, so it would be dead config; the env-side
        clips in CubeToShelfStand3Cfg do that job.
        """

        actor = _MLP3(
            hidden_dims=_HIDDEN,
            activation="elu",
            obs_normalization=False,
            distribution_cfg=_MLP3.GaussianDistributionCfg(
                init_std=1.0, std_type=stand.STAND3_STD_TYPE),
        )
        algorithm = _PPO3(
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            entropy_coef=stand.STAND3_ENTROPY_COEF,
            desired_kl=0.01,
            max_grad_norm=1.0,
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
        )

    _STAND3_RUNNER = TaskV2Stand3PPORunnerCfg
except (ImportError, NameError):                                 # v51 stack
    _STAND3_RUNNER = _V2_RUNNER


# --------------------------------------------- roll-proof lift cube, Stand4
# CubeToShelfStand4: a DIFFERENT task from CubeToShelfStand3 and CubeToShelf
# (2026-10-01, docs/SOLUTIONS_2026-10-01.md section 4, row C2). Never compared
# with their numbers. Full reasoning and every number: `stand4_mdp.py`.
#
# Stand3 earned `lifting_object` by rolling a supported cube (centre height
# only; replay 21470828). Changes from Stand3, all in this class (Stand3, its
# runner and every other task untouched; runner = Stand3's):
#   hands     both robots spawn from the hand-collider overlay USD (the base
#             USD has no collider on arm_*_hand_link);
#   lift      `lifting_object` (name, weight, `minimal_height` and curriculum
#             kept) also needs the lowest corner >= 0.02 m above every support
#             the cube could rest on AND tilt <= 15 deg;
#   success   Stand3's seated test AND tilt <= 8 deg AND released (no robot
#             link pushing on the cube with >= 1 N: a filtered contact sensor
#             on the cube), held 12 steps; `placed` still reads it (x 5000);
#   logging   lift level, pinch distance, upright gate, tilt, corner
#             clearance, release (env._bhl_s4 per step; Curriculum/* means).

from isaaclab.sensors import ContactSensorCfg  # noqa: E402

from bhl_robust.tasks import stand4_mdp as s4  # noqa: E402


@configclass
class CubeToShelfStand4Cfg(CubeToShelfStand3Cfg):
    """Stand3 with hand colliders, a roll-proof lift term and a flat, released success.

    A DIFFERENT task from `CubeToShelfStand3Cfg` and `CubeToShelfCfg`; reported
    as CubeToShelfStand4 and never read as either.
    """

    def __post_init__(self):
        super().__post_init__()
        # ---- hands: the overlay USD (Stand4 robots only)
        for name in ("robot_a", "robot_b"):
            rc = getattr(self.scene, name)
            rc.spawn = rc.spawn.replace(usd_path=str(s4.HAND_COLLIDER_USD))
        # ---- release sensing: contact reporting on the cube, filtered
        # one-to-many against every collider-bearing robot link
        self.scene.object.spawn = self.scene.object.spawn.replace(activate_contact_sensors=True)
        setattr(self.scene, s4.CUBE_SENSOR, ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/object",
            filter_prim_paths_expr=list(s4.RELEASE_FILTER_EXPRS),
            history_length=0,
            update_period=self.sim.dt,
        ))
        gate = {"gate_free": stand.GATE_FREE, "gate_std": stand.GATE_STD}
        r = self.rewards
        # ---- lift pay: same name, weight and curriculum parameter
        r.lifting_object = RewTerm(
            func=s4.stand4_object_is_lifted,
            params={"minimal_height": r.lifting_object.params["minimal_height"],
                    "clearance": s4.CORNER_CLEARANCE, "tilt_max_deg": s4.LIFT_TILT_MAX_DEG,
                    **gate},
            weight=r.lifting_object.weight)
        # ---- success (Stand3's `placed` reads this termination, unchanged)
        self.terminations.success = DoneTerm(
            func=s4.stand4_cube_placed,
            params={"hold_steps": stand.SEAT_HOLD_STEPS, "tilt_max_deg": s4.SEAT_TILT_MAX_DEG,
                    "release_force_n": s4.RELEASE_FORCE_N, "sensor_name": s4.CUBE_SENSOR})
        # ---- diagnostics (Curriculum/*, no gradient)
        c = self.curriculum
        c.pinch_dist = CurrTerm(func=s4.pinch_distance_mean)
        c.cube_tilt_deg = CurrTerm(func=s4.cube_tilt_mean)
        c.corner_clear = CurrTerm(func=s4.corner_clearance_mean)
        c.roll_ok = CurrTerm(func=s4.roll_ok_share)
        c.lift_paid = CurrTerm(func=s4.lift_paid_share)
        c.released = CurrTerm(func=s4.released_share)
        c.robot_force_max = CurrTerm(func=s4.robot_force_max_mean)


CUBE_STAND4_VARIANTS = _variants(CubeToShelfStand4Cfg, "CubeToShelfStand4")


# --- stand5 ---
# CubeToShelfStand5 (2026-10-03; SLURM_JOBS.md "User approval recorded 2026-10-03 09:55",
# item (S)): CubeToShelfStand4 plus exactly two changes, both in this class (Stand4, its
# runner and every other task untouched; runner = Stand3's, as Stand4). Every reason and
# number: `stand5_mdp.py`.
#   (i)  the actor (policy group) and the critic observe the cube's own z axis in each
#        robot's root frame, object_zaxis_a / object_zaxis_b (3 numbers each), ORACLE like
#        object_pos_a/b: no noise, no scale; clipped +/-OBS_CLIP like every observation
#        term since Stand3 (it never binds); appended last: policy 194 -> 200, critic
#        206 -> 212;
#   (ii) the lift curriculum (`lift_height`: same name, place and parameters) promotes on
#        Stand4's roll-proof lift condition (lowest corner >= 0.02 m above every support
#        under the cube AND tilt <= 15 deg) instead of centre height + pinch.
# Guarded: an exception here is printed and only the Stand5 names are missing (its id then
# fails to register and its smoke fails); every other task imports as before.
try:
    from bhl_robust.tasks import stand5_mdp as s5  # noqa: E402

    @configclass
    class CubeToShelfStand5Cfg(CubeToShelfStand4Cfg):
        """Stand4 with the cube's z axis observed (actor and critic) and a roll-proof lift curriculum.

        Reported as CubeToShelfStand5 and judged only by its own predeclared rule.
        """

        def __post_init__(self):
            super().__post_init__()
            # ---- (i) ORACLE cube orientation, last in both groups (the critic group is
            # its own instance: a term set on the policy group does not reach it)
            for grp in (self.observations.policy, self.observations.critic):
                for name, robot in s5.OBS_TERM_ROBOTS:
                    setattr(grp, name, ObsTerm(
                        func=s5.object_zaxis_in_root, params={"robot_cfg": SceneEntityCfg(robot)},
                        clip=(-stand.OBS_CLIP, stand.OBS_CLIP)))
            # ---- (ii) the lift curriculum: same name (keeps its place), parameters and
            # target term; it promotes on the roll-proof lift condition
            lh = self.curriculum.lift_height
            self.curriculum.lift_height = CurrTerm(
                func=s5.stand5_lift_height_curriculum,
                params={**lh.params, "clearance": s4.CORNER_CLEARANCE,
                        "tilt_max_deg": s4.LIFT_TILT_MAX_DEG})

    CUBE_STAND5_VARIANTS = _variants(CubeToShelfStand5Cfg, "CubeToShelfStand5")
except Exception as _stand5_exc:  # noqa: BLE001
    import sys as _sys
    print(f"[bhl_robust.tasks.task_v2_env_cfg] stand5 NOT defined: {_stand5_exc!r}",
          file=_sys.stderr, flush=True)
# --- end stand5 ---
