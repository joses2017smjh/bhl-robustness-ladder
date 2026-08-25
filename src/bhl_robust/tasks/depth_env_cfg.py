"""Depth-camera overlays for the BHL locomotion tasks.

The sensor is `RayCasterCamera`, not `TiledCamera`. Isaac Sim 5.1's RTX
renderer segfaults inside `omni.usd.create_hydra_engine` on this cluster's
driver, so every renderer-backed sensor is unavailable; the ray-cast camera
intersects a pinhole ray bundle with the scene meshes in warp instead, and
never asks for a Hydra engine. What it returns is exactly
`distance_to_image_plane` -- range, no radiance -- which is the only channel a
perceptive locomotion policy reads anyway.

Two consequences worth stating, because they are the honest limits of this
substitute:

* Rays are cast against `mesh_prim_paths` only. That is the ground, so the
  robot sees terrain and not itself; self-occlusion is absent by construction.
* There is no material, lighting or sensor-noise model. Depth is geometric and
  exact, so a policy trained on it has not been asked to cope with the noise a
  real D435 produces. The Gaussian term below is the crude stand-in.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg import (
    ObservationsCfg,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from bhl_robust.tasks.terrain_env_cfg import BipedBumpyEnvCfg

# Forward-and-down, at chest height on the base -- the pose a head-mounted
# depth sensor on this robot would actually have. 20 deg down-pitch about +Y in
# world convention: (w, x, y, z) = (cos 10 deg, 0, sin 10 deg, 0) is 20 deg of
# rotation, since a quaternion halves the angle.
CAM_POS = (0.12, 0.0, 0.30)
CAM_ROT = (0.9848, 0.0, 0.1736, 0.0)
CAM_RANGE = 6.0
CAM_FOCAL = 18.0
CAM_APERTURE = 20.955


def make_depth_camera_cfg(res: int = 64, mesh_paths: list[str] | None = None) -> RayCasterCameraCfg:
    """A square forward-looking depth camera on the robot base."""
    return RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        mesh_prim_paths=mesh_paths or ["/World/ground"],
        offset=RayCasterCameraCfg.OffsetCfg(pos=CAM_POS, rot=CAM_ROT, convention="world"),
        data_types=["distance_to_image_plane"],
        # Beyond range a real depth sensor reports its maximum, not NaN. Without
        # this the sky is NaN and the first backward pass is NaN with it.
        depth_clipping_behavior="max",
        max_distance=CAM_RANGE,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=CAM_FOCAL, horizontal_aperture=CAM_APERTURE, width=res, height=res,
        ),
        update_period=0.0,
        debug_vis=False,
    )


def depth_obs(env, sensor_cfg: SceneEntityCfg, pool: int = 4, clip: float = CAM_RANGE) -> torch.Tensor:
    """Depth as a flat observation, average-pooled and scaled to roughly [0, 1].

    Pooling is not cosmetic. A 64x64 image is 4,096 numbers against a 45-dim
    proprioceptive vector; fed raw to upstream's MLP the depth would be 99% of
    the input width and the first layer would be almost entirely depth weights.
    Average pooling to 16x16 is the cheapest honest encoder -- it is what a
    height-scan observation already is, only forward-looking instead of
    underfoot. A learned CNN trunk is the real answer, and is not something
    rsl-rl's default actor provides.
    """
    d = env.scene[sensor_cfg.name].data.output["distance_to_image_plane"]  # (N, H, W, 1)
    d = d.permute(0, 3, 1, 2).nan_to_num(nan=clip, posinf=clip)
    return (F.avg_pool2d(d, pool).flatten(1) / clip).clamp(0.0, 1.0)


@configclass
class DepthObservationsCfg(ObservationsCfg):
    """Upstream's groups with a depth term appended to policy and critic alike."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        depth = ObsTerm(
            func=depth_obs,
            params={"sensor_cfg": SceneEntityCfg("depth_cam"), "pool": 4},
            # Depth is scaled to [0, 1], so 2 cm of range noise is 0.0033.
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
class BipedDepthEnvCfg(BipedBumpyEnvCfg):
    """Rough terrain plus a forward-looking depth camera.

    Identical to `BipedBumpyEnvCfg` -- same terrain menu, same level curriculum,
    same s = 1.0 randomization -- except for the sensor and the observation term
    it feeds, so depth is the isolated variable against the terrain rung the
    README already reports.
    """

    observations: DepthObservationsCfg = DepthObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.depth_cam = make_depth_camera_cfg(res=64)


@configclass
class BipedStairsDepthEnvCfg(BipedDepthEnvCfg):
    """Stairs with the forward depth camera. Same sensor, different geometry."""

    def __post_init__(self):
        super().__post_init__()
        from bhl_robust.terrains.stairs import STAIRS_TERRAINS_CFG
        self.scene.terrain.terrain_generator = STAIRS_TERRAINS_CFG


@configclass
class BipedSlipperyDepthEnvCfg(BipedDepthEnvCfg):
    """Low friction with the forward depth camera.

    Geometry identical to the bumpy depth rung, so any difference between this
    and `BipedDepthEnvCfg` is friction alone -- and the depth camera cannot see
    friction. This arm existing is what makes "depth pays on geometry, not
    material" falsifiable rather than asserted.
    """

    def __post_init__(self):
        super().__post_init__()
        self.events.physics_material.params["static_friction_range"] = (0.25, 0.35)
        self.events.physics_material.params["dynamic_friction_range"] = (0.18, 0.30)
