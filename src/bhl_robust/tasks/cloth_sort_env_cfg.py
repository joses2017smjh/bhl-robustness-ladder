"""Isaac Lab environments for the rigid-to-deformable cloth-sort ladder.

Stage 1 (this file's default) is cheap rigid proxies and a 5-D sweep action.
Stage 2 / 3 configs exist so the gym ids are real; they do not launch cloth
RL. Deformable construction is cost-gated in ``scripts/cloth/estimate_cost.py``.

The robot is the stock 22-DoF Berkeley Humanoid Lite. No extra arms, no
gripper DoF. Walking-to-the-table is out of scope: the robot spawns already
addressing the table, same reasoning as the coop pinch spawn.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab.envs.mdp as mdp

from bhl_robust.cloth.garments import BASKET_IDS, GARMENTS, GARMENT_BY_NAME, sibling_index
from bhl_robust.cloth.layout import ROBOT_XY, TABLE_TOP_Z, default_spawn_xy
from bhl_robust.cloth.randomization import DomainRandomization
from bhl_robust.tasks import cloth_sort_mdp as cs
from bhl_robust.tasks import furniture
from bhl_robust.tasks.coop_lift_env_cfg import _PINCH_ROOT_Z, _RIGID, _COLLISION, _robot
from bhl_robust.tasks.task_v2_env_cfg import _V2_RUNNER

#: Same 4-tuple task_v2 measured as standing. Cloth-sort is a new task, not a
#: re-run of a FINDINGS number, so it does not inherit the legacy yaw.
_STAND_UP = (0.0, 0.0, 1.0, 0.0)

_DR = DomainRandomization()


def _proxy(spec_name: str, prim: str,
           xy: tuple[float, float] | None = None) -> RigidObjectCfg:
    """Thin rigid rectangle standing in for one garment."""
    spec = GARMENT_BY_NAME[spec_name]
    if xy is None:
        k, n = sibling_index(spec, GARMENTS if spec_name != "shirt_a" else (spec,))
        # Single-garment C0/C1 sits on the table centre, not in a five-item lane.
        if spec_name == "shirt_a" and prim == "garment_0":
            xy = default_spawn_xy(spec, 0, 1)
        else:
            xy = default_spawn_xy(spec, k, n)
    z = TABLE_TOP_Z + 0.5 * spec.proxy_size[2]
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(xy[0], xy[1], z), rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.CuboidCfg(
            size=spec.proxy_size,
            rigid_props=_RIGID,
            collision_props=_COLLISION,
            mass_props=sim_utils.MassPropertiesCfg(mass=spec.mass),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.rgb),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=spec.friction, dynamic_friction=0.85 * spec.friction,
                restitution=0.0,
            ),
        ),
    )


@configclass
class ClothSortSceneCfg(InteractiveSceneCfg):
    """One humanoid, one rigid garment, three baskets, a table."""

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        terrain_generator=None,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=0.7,
            dynamic_friction=0.6,
        ),
        debug_vis=False,
    )
    robot: ArticulationCfg = _robot(
        "{ENV_REGEX_NS}/robot",
        (ROBOT_XY[0], ROBOT_XY[1], _PINCH_ROOT_Z),
        _STAND_UP,
    )
    garment_0: RigidObjectCfg = _proxy("shirt_a", "garment_0")
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )


@configclass
class ClothSortFiveSceneCfg(ClothSortSceneCfg):
    """Five rigid proxies. Evaluation / C5 Mode-B stand-in, not 5-cloth RL."""

    garment_0: RigidObjectCfg = _proxy("sock_a", "garment_0")
    garment_1: RigidObjectCfg = _proxy("sock_b", "garment_1")
    garment_2: RigidObjectCfg = _proxy("shirt_a", "garment_2")
    garment_3: RigidObjectCfg = _proxy("shirt_b", "garment_3")
    garment_4: RigidObjectCfg = _proxy("jacket", "garment_4")


@configclass
class ClothSortObservationsCfg:
    """Oracle / privileged state. Labeled: this is not a perception policy."""

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=cs.joint_pos,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-0.03, n_max=0.03),
        )
        joint_vel = ObsTerm(
            func=cs.joint_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=Unoise(n_min=-1.0, n_max=1.0),
        )
        base_quat = ObsTerm(func=cs.base_quat)
        base_ang_vel = ObsTerm(func=cs.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        hand_pos = ObsTerm(func=cs.hand_pos)
        garment_xy = ObsTerm(func=cs.garment_xy, params={"asset_name": "garment_0"})
        garment_yaw = ObsTerm(func=cs.garment_yaw_proxy, params={"asset_name": "garment_0"})
        rel_basket = ObsTerm(
            func=cs.rel_garment_basket,
            params={"asset_name": "garment_0", "garment": "shirt_a"},
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(PolicyCfg):
        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class ClothSortActionsCfg:
    sweep = cs.SweepActionCfg(asset_name="robot")


@configclass
class ClothSortRewardsCfg:
    progress = RewTerm(
        func=cs.progress_to_basket,
        params={"asset_name": "garment_0", "garment": "shirt_a"},
        weight=1.0,
    )
    success = RewTerm(
        func=cs.garment_in_correct_basket,
        params={"asset_name": "garment_0", "garment": "shirt_a"},
        weight=5.0,
    )
    wrong_basket = RewTerm(
        func=cs.garment_in_wrong_basket,
        params={"asset_name": "garment_0", "garment": "shirt_a"},
        weight=-2.0,
    )
    time = RewTerm(func=cs.time_penalty, weight=-0.05)
    invalid = RewTerm(func=cs.invalid_sweep, weight=-0.5)
    fall = RewTerm(func=cs.fallen, weight=-4.0)


@configclass
class ClothSortTerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    success = DoneTerm(
        func=cs.success_correct,
        params={"asset_name": "garment_0", "garment": "shirt_a"},
    )
    fallen = DoneTerm(func=cs.robot_fallen)


@configclass
class ClothSortEventsCfg:
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (-0.04, 0.04),
            "velocity_range": (0.0, 0.0),
        },
    )
    reset_root = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (-0.02, 0.02), "y": (-0.02, 0.02), "yaw": (-0.08, 0.08)},
            "velocity_range": {},
        },
    )
    reset_garment = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("garment_0"),
            "pose_range": {
                "x": (-_DR.spawn_xy_jitter, _DR.spawn_xy_jitter),
                "y": (-_DR.spawn_xy_jitter, _DR.spawn_xy_jitter),
                "yaw": _DR.yaw_range,
            },
            "velocity_range": {},
        },
    )


@configclass
class ClothSortRigidEnvCfg(ManagerBasedRLEnvCfg):
    """C0 / C1: one rigid proxy, one humanoid, oracle observations."""

    scene: ClothSortSceneCfg = ClothSortSceneCfg(num_envs=64, env_spacing=3.5)
    observations: ClothSortObservationsCfg = ClothSortObservationsCfg()
    actions: ClothSortActionsCfg = ClothSortActionsCfg()
    rewards: ClothSortRewardsCfg = ClothSortRewardsCfg()
    terminations: ClothSortTerminationsCfg = ClothSortTerminationsCfg()
    events: ClothSortEventsCfg = ClothSortEventsCfg()
    #: Catalog name the reward/success terms are bound to.
    garment_name: str = "shirt_a"

    def __post_init__(self):
        self.decimation = 8
        self.episode_length_s = 12.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.scene.robot.init_state.pos = (ROBOT_XY[0], ROBOT_XY[1], _PINCH_ROOT_Z)
        # Same pattern as the maze: attach static colliders after the
        # configclass is built so the scene dataclass does not have to name
        # every basket wall.
        self.scene.table = furniture.cloth_table()
        for bid in BASKET_IDS:
            for i, box in enumerate(furniture.sorting_basket(bid)):
                setattr(self.scene, f"basket_{bid}_{i}", box)


@configclass
class ClothSortRigidFiveEnvCfg(ClothSortRigidEnvCfg):
    """C5 rigid stand-in: five proxies, three baskets. Not five deformables."""

    scene: ClothSortFiveSceneCfg = ClothSortFiveSceneCfg(num_envs=16, env_spacing=4.0)
    active_garment_mode: bool = True

    def __post_init__(self):
        super().__post_init__()
        # garment_0 is sock_a in the five-item scene, not the C0 shirt.
        self.garment_name = "sock_a"
        self.observations.policy.rel_basket.params["garment"] = "sock_a"
        self.observations.critic.rel_basket.params["garment"] = "sock_a"
        self.rewards.progress.params["garment"] = "sock_a"
        self.rewards.success.params["garment"] = "sock_a"
        self.rewards.wrong_basket.params["garment"] = "sock_a"
        self.terminations.success = DoneTerm(func=cs.success_five)


def _newton_cloth_physics():
    """The Franka-cloth Newton/VBD preset. Imported lazily so v51 registration lives."""
    from isaaclab_tasks.manager_based.manipulation.lift_franka_soft.franka_cloth_env_cfg import (
        PhysicsCfg,
    )
    return PhysicsCfg()


def _deformable_garment(spec_name: str, prim: str, resolution: int):
    """Low-resolution surface cloth. Starts at 8×8, never at 961 vertices."""
    from isaaclab.assets.deformable_object import DeformableObjectCfg
    from isaaclab_newton.sim.schemas import NewtonDeformableBodyPropertiesCfg
    from isaaclab_newton.sim.spawners.materials import NewtonSurfaceDeformableBodyMaterialCfg

    spec = GARMENT_BY_NAME[spec_name]
    xy = default_spawn_xy(spec, 0, 1)
    n = int(resolution)
    return DeformableObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/{prim}",
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(xy[0], xy[1], TABLE_TOP_Z + 0.01),
        ),
        spawn=sim_utils.MeshRectangleCfg(
            size=(spec.proxy_size[0], spec.proxy_size[1]),
            resolution=(n, n),
            deformable_props=NewtonDeformableBodyPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=spec.rgb),
            physics_material=NewtonSurfaceDeformableBodyMaterialCfg(
                density=max(spec.mass / max(spec.proxy_size[0] * spec.proxy_size[1] * 0.002, 1e-6), 20.0),
                particle_radius=0.008,
                tri_ke=5e2, tri_ka=5e2, tri_kd=1e-3,
                edge_ke=2.0, edge_kd=1e-3,
            ),
        ),
    )


@configclass
class ClothSortDeformableEnvCfg(ClothSortRigidEnvCfg):
    """Stage 2: one low-resolution deformable, 8 envs.

    Replaces the rigid ``garment_0`` with a Newton surface cloth. If Newton is
    missing this raises rather than silently training on a cube. ``sim.dt``
    is 1/60 to match the Franka cloth scene; that change is documented, not a
    silent fidelity drop. Not a training job — C2/C3 eval and the bench.
    """

    cloth_resolution: int = 8
    n_deformables: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 8
        self.scene.replicate_physics = True
        self.sim.dt = 1.0 / 60.0
        self.decimation = 1
        self.episode_length_s = 8.0
        try:
            self.scene.garment_0 = _deformable_garment(
                self.garment_name, "garment_0", self.cloth_resolution,
            )
            self.sim.physics = _newton_cloth_physics()
        except ImportError as exc:
            raise RuntimeError(
                "ClothSortDeformableEnvCfg needs Isaac Lab Newton "
                "(Isaac Sim 6.0). This is Stage 2, not a rigid env. "
                "Use ClothSort-BHL-Rigid-Oracle-v0 until that stack is up, and "
                "run scripts/cloth/estimate_cost.py before any deformable job."
            ) from exc
        self.rewards.success = RewTerm(
            func=cs.deformable_in_correct_basket,
            params={"asset_name": "garment_0", "garment": self.garment_name},
            weight=5.0,
        )
        self.rewards.wrong_basket = RewTerm(
            func=cs.deformable_in_wrong_basket,
            params={"asset_name": "garment_0", "garment": self.garment_name},
            weight=-2.0,
        )
        self.terminations.success = DoneTerm(
            func=cs.success_deformable,
            params={"asset_name": "garment_0", "garment": self.garment_name},
        )
        self.events.reset_garment = EventTerm(func=cs.skip_event, mode="reset")


@configclass
class ClothSortActiveDeformableEnvCfg(ClothSortRigidFiveEnvCfg):
    """C5 Mode B: one live cloth (garment_0), four rigid proxies.

    That is not five simultaneously active cloth objects. Documented as
    such in docs/CLOTH_SORT.md. Evaluation only.
    """

    cloth_resolution: int = 8
    n_deformables: int = 1
    active_garment_mode: bool = True

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.scene.replicate_physics = True
        self.sim.dt = 1.0 / 60.0
        self.decimation = 1
        try:
            self.scene.garment_0 = _deformable_garment("sock_a", "garment_0", self.cloth_resolution)
            self.sim.physics = _newton_cloth_physics()
        except ImportError as exc:
            raise RuntimeError(
                "Active-garment deformable needs Newton. Mode B is still not "
                "five live cloths even when it constructs."
            ) from exc
        self.garment_name = "sock_a"
        self.rewards.success = RewTerm(
            func=cs.deformable_in_correct_basket,
            params={"asset_name": "garment_0", "garment": "sock_a"},
            weight=5.0,
        )
        self.rewards.wrong_basket = RewTerm(
            func=cs.deformable_in_wrong_basket,
            params={"asset_name": "garment_0", "garment": "sock_a"},
            weight=-2.0,
        )
        self.terminations.success = DoneTerm(
            func=cs.success_deformable,
            params={"asset_name": "garment_0", "garment": "sock_a"},
        )
        self.events.reset_garment = EventTerm(func=cs.skip_event, mode="reset")


# --------------------------------------------------------------- construction

def spawned_cloth_resolution(cfg) -> int | None:
    """Vertices-per-side the scene will *actually* spawn, read off the spawn cfg.

    Returns ``None`` for a rigid proxy, which has no ``resolution``.
    """
    spawn = getattr(getattr(cfg.scene, "garment_0", None), "spawn", None)
    res = getattr(spawn, "resolution", None)
    if res is None:
        return None
    return int(res[0])


def build_cfg(cfg_cls, *, num_envs: int | None = None, device: str | None = None,
              cloth_resolution: int | None = None):
    """Build a cloth-sort env cfg with the cloth resolution genuinely applied.

    ``cloth_resolution`` must reach ``__init__``. The deformable garment is
    spawned inside ``__post_init__``, so assigning the field *after*
    construction relabels the config without changing the mesh -- a bench
    doing that would report 10x10 while simulating 8x8. The read-back below
    is the guard: if the request did not take effect this raises instead of
    writing a mislabeled row.
    """
    kwargs = {}
    fields = getattr(cfg_cls, "__dataclass_fields__", {})
    if cloth_resolution is not None and "cloth_resolution" in fields:
        kwargs["cloth_resolution"] = int(cloth_resolution)
    cfg = cfg_cls(**kwargs)
    if num_envs is not None:
        cfg.scene.num_envs = int(num_envs)
    if device is not None:
        cfg.sim.device = device
    if cloth_resolution is not None:
        got = spawned_cloth_resolution(cfg)
        if got is not None and got != int(cloth_resolution):
            raise RuntimeError(
                f"{cfg_cls.__name__}: asked for a {cloth_resolution}x{cloth_resolution} "
                f"cloth, scene will spawn {got}x{got}. Refusing to bench a mesh "
                f"under the wrong label."
            )
    return cfg


ClothSortPPORunnerCfg = _V2_RUNNER
