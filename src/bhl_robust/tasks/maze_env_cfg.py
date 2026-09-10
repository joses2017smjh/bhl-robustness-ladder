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

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from bhl_robust.sensors_rig import (
    LIDAR_SECTORS, lidar_obs, make_lidar_cfg, make_stereo_cfg,
)
from bhl_robust.tasks import furniture
from bhl_robust.tasks.depth_env_cfg import depth_obs
from bhl_robust.tasks.terrain_env_cfg import BipedBumpyEnvCfg
# The BHL biped's own ObservationsCfg, not Isaac Lab's upstream one. Upstream's
# carries a `height_scan` term bound to a `height_scanner` sensor this robot's
# config does not create, so inheriting from it registers a term whose sensor
# does not exist and every arm dies at reset -- which is what the first maze
# smoke did, 1 of 4, while the blind arm using the biped config passed.
import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp  # noqa: E402
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg import (
    ObservationsCfg,
)

# A single corridor with two junctions, signed. Small enough to train and
# large enough that a wrong turn costs the episode.
#   (x, y, length, axis)
MAZE_WALLS = [
    (0.0, +furniture.CORRIDOR_W / 2, 6.0, "x"),
    (0.0, -furniture.CORRIDOR_W / 2, 6.0, "x"),
    (3.0, +1.8, 2.6, "y"),
    (-3.0, -1.8, 2.6, "y"),
]
JUNCTIONS = [((1.5, 0.0, 0.45), "left"), ((-1.5, 0.0, 0.45), "right")]
OBSTACLES = [(0.6, 0.15), (-0.4, -0.20), (2.2, 0.05)]
BUTTON_AT = (3.0, 0.0, 0.55)


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


def _build_maze(scene) -> None:
    """Walls, signs, floor obstacles and the button, on any maze scene."""
    for i, w in enumerate(furniture.corridor_walls(MAZE_WALLS)):
        setattr(scene, f"maze_wall{i}", w)
    for i, (pos, heading) in enumerate(JUNCTIONS):
        setattr(scene, f"maze_arrow{i}", furniture.arrow_marker(f"maze_arrow{i}", pos, heading))
    for i, (x, y) in enumerate(OBSTACLES):
        setattr(scene, f"maze_obs{i}", furniture.small_obstacle(f"maze_obs{i}", (x, y, 0.0)))
    scene.maze_button = furniture.button("maze_button", BUTTON_AT)


@configclass
class MazeBlindEnvCfg(BipedBumpyEnvCfg):
    """The control: the same maze, walked on proprioception alone.

    Without this row a lidar number is a fact about the maze, not about lidar --
    the mistake G-B2 made when it measured its own iteration budget and called
    it a terrain verdict.
    """

    def __post_init__(self):
        super().__post_init__()
        _build_maze(self.scene)


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

