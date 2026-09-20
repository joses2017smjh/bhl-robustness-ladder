"""Maze navigation with a lidar and a stereo pair, on the rung that works.

Built on `BipedBumpyEnvCfg` rather than on the v2 manipulation tasks, and that
choice is the design. Locomotion is this project's working half -- the push,
terrain and arms rungs all show real effects -- while every manipulation task
here has scored zero. A new capability rung belongs on the half that walks.

The three sensor conditions are the point:

    blind    proprioception only, the control
    lidar    + 36 sector minima from the RPLIDAR C1 plane
    stereo   + two 16x16 pooled depth images on a 60 mm baseline
    both     lidar and stereo together

Each hazard is chosen so exactly one sensor can see it, which is what makes the
comparison mean anything:

* **floor obstacles at 0.10 m** sit under the 0.34 m lidar plane and inside the
  stereo pair's down-pitched cone. Stereo's hazard.
* **walls and junctions** are at lidar height and, at a corner, outside the
  cameras' 64-pixel forward cone until the robot has already turned. Lidar's.
* **arrow plates** are geometric, not coloured, so they read as depth structure
  to stereo and as one more wall to lidar. Stereo's, and the reason the task
  cannot be solved by wall-following alone.

`ice_gate.py` is the precedent for that discipline: an advantage only counts if
the sensor could physically have seen the thing it is credited with seeing.
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from bhl_robust.sensors_rig import (
    LIDAR_SECTORS, lidar_obs, make_lidar_cfg, make_stereo_cfg,
)
from bhl_robust.tasks.depth_env_cfg import depth_obs
from bhl_robust.tasks import maze_mdp
from bhl_robust.tasks.terrain_env_cfg import BipedBumpyEnvCfg
from bhl_robust.terrains.maze import MAZE_TERRAINS_CFG
from bhl_robust.terrains.maze_layout import CRUISE_SPEED, FLOOR_RGB
# The BHL biped's own ObservationsCfg, not Isaac Lab's upstream one. Upstream's
# carries a `height_scan` term bound to a `height_scanner` sensor this robot's
# config does not create, so inheriting from it registers a term whose sensor
# does not exist and every arm dies at reset -- which is what the first maze
# smoke did, 1 of 4, while the blind arm using the biped config passed.
import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp  # noqa: E402
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg import (
    ObservationsCfg,
    RewardsCfg,
    TerminationsCfg,
)


@configclass
class LidarObsCfg(ObservationsCfg):
    """Proprioception plus lidar sector minima."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        lidar = ObsTerm(
            func=lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "sectors": LIDAR_SECTORS},
            # The C1's stated range accuracy is about 3 cm; on a 12 m scale
            # that is 0.0025 once the term is normalised.
            noise=GaussianNoiseCfg(mean=0.0, std=0.0025),
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class StereoObsCfg(ObservationsCfg):
    """Proprioception plus both eyes, pooled the same way the depth rung pools."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        stereo_l = ObsTerm(
            func=depth_obs,
            params={"sensor_cfg": SceneEntityCfg("stereo_l"), "pool": 4},
            noise=GaussianNoiseCfg(mean=0.0, std=0.0033),
        )
        stereo_r = ObsTerm(
            func=depth_obs,
            params={"sensor_cfg": SceneEntityCfg("stereo_r"), "pool": 4},
            noise=GaussianNoiseCfg(mean=0.0, std=0.0033),
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class BothObsCfg(StereoObsCfg):
    """Lidar and stereo together."""

    @configclass
    class PolicyCfg(StereoObsCfg.PolicyCfg):
        lidar = ObsTerm(
            func=lidar_obs,
            params={"sensor_cfg": SceneEntityCfg("lidar"), "sectors": LIDAR_SECTORS},
            noise=GaussianNoiseCfg(mean=0.0, std=0.0025),
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class MazeCurriculumCfg:
    """No terrain-level promotion: every tile is the same corridor."""


@configclass
class MazeRewardsCfg(RewardsCfg):
    """Walking rewards plus the button route."""

    progress_to_button = RewTerm(func=maze_mdp.progress_to_button, weight=2.0)
    button_reached = RewTerm(func=maze_mdp.button_reached, weight=10.0)
    dead_end = RewTerm(func=maze_mdp.in_dead_end, weight=-2.0)


@configclass
class MazeTerminationsCfg(TerminationsCfg):
    """Fall / timeout plus the button and the dead-end."""

    button_reached = DoneTerm(func=maze_mdp.button_reached)
    dead_end = DoneTerm(func=maze_mdp.in_dead_end)


@configclass
class MazeCommandsCfg:
    """Walk the signed corridor to the button, not a random SE(2) draw."""

    base_velocity: maze_mdp.MazeWaypointCommandCfg = maze_mdp.MazeWaypointCommandCfg(
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        asset_name="robot",
        heading_command=True,
        heading_control_stiffness=0.5,
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        cruise_speed=CRUISE_SPEED,
        ranges=maze_mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(CRUISE_SPEED, CRUISE_SPEED),
            lin_vel_y=(0.0, 0.0),
            ang_vel_z=(-1.0, 1.0),
            heading=(0.0, 0.0),
        ),
    )


@configclass
class MazeBlindEnvCfg(BipedBumpyEnvCfg):
    """The control: the same maze, walked on proprioception alone.

    Without this row a lidar number is a fact about the maze, not about lidar --
    the mistake G-B2 made when it measured its own iteration budget and called
    it a terrain verdict.
    """

    commands: MazeCommandsCfg = MazeCommandsCfg()
    curriculum: MazeCurriculumCfg = MazeCurriculumCfg()
    rewards: MazeRewardsCfg = MazeRewardsCfg()
    terminations: MazeTerminationsCfg = MazeTerminationsCfg()

    def __post_init__(self):
        super().__post_init__()
        # The first maze cloned furniture at ``{ENV_REGEX_NS}``, but generated
        # terrain resets the robots at terrain-tile origins instead.  The
        # walls could therefore be metres away and the ray sensors (which cast
        # only against /World/ground) could not see them.  Fuse every hazard
        # into the generated ground mesh so collision, spawn and sensing share
        # one origin.
        self.scene.terrain.terrain_generator = MAZE_TERRAINS_CFG
        # Parent bumpy cfg clears the Nucleus shingle so height-fields do not
        # stretch a roof texture. A PreviewSurface is what RTX actually
        # colours; without it the fused mesh records as R=G=B (mazenav clips
        # historical run). Ray sensors still cast distance, not albedo.
        self.scene.terrain.visual_material = sim_utils.PreviewSurfaceCfg(
            diffuse_color=FLOOR_RGB,
        )
        # Parent locomotion spawn is ±0.5 m and ±π yaw, which puts robots
        # through the 0.90 m corridor walls and facing the dead end.  Face +x
        # at the tile origin, inside the walls.
        self.events.reset_base.params["pose_range"] = {
            "x": (-0.20, 0.20), "y": (-0.15, 0.15), "yaw": (-0.25, 0.25),
        }
        self.events.reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
            "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0),
        }


@configclass
class MazeLidarEnvCfg(MazeBlindEnvCfg):
    observations: LidarObsCfg = LidarObsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar = make_lidar_cfg()


@configclass
class MazeStereoEnvCfg(MazeBlindEnvCfg):
    observations: StereoObsCfg = StereoObsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.stereo_l = make_stereo_cfg("left", res=64)
        self.scene.stereo_r = make_stereo_cfg("right", res=64)


@configclass
class MazeBothEnvCfg(MazeBlindEnvCfg):
    observations: BothObsCfg = BothObsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.lidar = make_lidar_cfg()
        self.scene.stereo_l = make_stereo_cfg("left", res=64)
        self.scene.stereo_r = make_stereo_cfg("right", res=64)

# --------------------------------------------------------------- pooling sweep
#
# The first four arms found stereo far *worse* than blind -- terrain level
# 0.0005 against 0.5191 -- and dragging `both` down from lidar's 0.7814 to
# 0.0624. Two explanations fit that and they have different consequences:
#
#   a) cameras genuinely do not help on this task
#   b) the stereo term drowned the proprioception it was added to
#
# At pool=4 stereo is 512 of 557 observations, **92%**, against lidar's 36 of 81
# (44%). `depth_obs` warns about exactly this: fed raw, depth would be "99% of
# the input width and the first layer would be almost entirely depth weights".
# So (b) is the more likely reading and it is testable by changing one number.
#
# The arm that decides it is `StereoP16`: 4x4 per eye is 32 of 77 observations,
# **42%**, matched to lidar's 44%. Same information fraction, different sensor.
# If that arm still sits at the floor, cameras do not help here. If it recovers,
# the first result was an encoding artefact and should be reported as one.


@configclass
class MazeStereoP8EnvCfg(MazeStereoEnvCfg):
    """8x8 per eye: 128 of 173 observations, 74%."""

    def __post_init__(self):
        super().__post_init__()
        for grp in (self.observations.policy, self.observations.critic):
            for term in ("stereo_l", "stereo_r"):
                getattr(grp, term).params["pool"] = 8


@configclass
class MazeStereoP16EnvCfg(MazeStereoEnvCfg):
    """4x4 per eye: 32 of 77 observations, 42% -- matched to lidar's 44%.

    This is the arm that separates "cameras do not help" from "cameras were
    drowned out", because it is the only one that holds the information
    *fraction* fixed while changing the sensor.
    """

    def __post_init__(self):
        super().__post_init__()
        for grp in (self.observations.policy, self.observations.critic):
            for term in ("stereo_l", "stereo_r"):
                getattr(grp, term).params["pool"] = 16


@configclass
class MazeBothP16EnvCfg(MazeBothEnvCfg):
    """Lidar plus the pooled stereo: does stereo still poison the lidar arm?"""

    def __post_init__(self):
        super().__post_init__()
        for grp in (self.observations.policy, self.observations.critic):
            for term in ("stereo_l", "stereo_r"):
                getattr(grp, term).params["pool"] = 16
