"""RGB-conditioned locomotion. Isaac Sim 6.0 only.

This module cannot run on the stack that produced every other result in this
repo, and that is the whole point of it existing.

Section 6's finding was that Isaac Sim 5.1's RTX renderer segfaults on this
cluster inside `omni.usd.create_hydra_engine`, so depth came from Warp
ray-casting -- geometry, no materials, no lighting, no colour. Ray-casting has
no colour to return, so RGB was not a modality anyone was declining to use; it
was one the cluster could not produce.

On the 6.0 stack it can. Measured, not assumed: a shaded cube renders at
rgb std 24.4 across 1,902 unique colours with depth correct to the unit, where
5.1 cannot open the stage at all.

So this is the first observation in the project that carries appearance rather
than geometry, and the comparison it enables is the one §6 could not run:
ray-cast depth (exact, free, geometry-only) against rendered RGB (lit, shaded,
and subject to every artefact a real camera has).

Two things to hold onto when reading anything produced here:

* `isaaclab 3.0.0b2` is a **beta**, and its numbers do not belong in the same
  table as a 5.1 number. Report them as a 6.0 section.
* `TiledCamera` costs what ray-casting does not. §6 measured ray-cast depth at
  1.6% of throughput at 4,096 envs; a rendered camera is a different order of
  expense and the env count here reflects that.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F

import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg import (
    ObservationsCfg,
)
import isaaclab.sim as sim_utils
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from bhl_robust.tasks.depth_env_cfg import CAM_POS, CAM_RANGE, CAM_ROT
from bhl_robust.tasks.terrain_env_cfg import BipedBumpyEnvCfg

# Deliberately smaller than the depth camera's 64x64. A rendered tile costs far
# more than a ray-cast one, and the point here is whether colour carries
# anything at all, not how much of it the GPU can push.
RGB_RES = 32
RGB_POOL = 4


def make_rgb_camera_cfg(res: int = RGB_RES) -> TiledCameraCfg:
    """A rendered colour camera in the same pose as §6's ray-cast depth camera.

    Same position, same orientation, same clipping range, so the only thing that
    differs between the two experiments is what the sensor returns.
    """
    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/base/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=CAM_POS, rot=CAM_ROT, convention="world"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, CAM_RANGE),
        ),
        width=res, height=res,
    )


def rgb_obs(env, sensor_cfg: SceneEntityCfg, pool: int = RGB_POOL) -> torch.Tensor:
    """Colour as a flat observation: average-pooled, scaled to [0, 1].

    Pooled for the same reason depth is -- a raw 32x32x3 image is 3,072 numbers
    against 45 of proprioception and would be almost the entire first layer --
    but the pooling costs more here. Depth degrades gracefully under averaging
    because neighbouring pixels are usually the same surface; colour does not,
    because an average of two materials is a third material that is not there.
    A CNN trunk is the honest encoder and is not what rsl-rl's default actor is,
    so a negative result from this arm is a result about pooled colour rather
    than about colour.
    """
    rgb = env.scene[sensor_cfg.name].data.output["rgb"]      # (N, H, W, 3), uint8
    x = rgb.permute(0, 3, 1, 2).float() / 255.0
    return F.avg_pool2d(x, pool).flatten(1).clamp(0.0, 1.0)


@configclass
class RgbObservationsCfg(ObservationsCfg):
    """Upstream's groups with a pooled colour image appended to both."""

    @configclass
    class PolicyCfg(ObservationsCfg.PolicyCfg):
        rgb = ObsTerm(
            func=rgb_obs,
            params={"sensor_cfg": SceneEntityCfg("rgb_cam"), "pool": RGB_POOL},
            # Sensor noise on an 8-bit channel scaled to [0,1] is ~1/255 per LSB;
            # 0.01 is a few counts, which is a quiet camera rather than a clean one.
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )

    @configclass
    class CriticCfg(PolicyCfg):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

        def __post_init__(self):
            self.enable_corruption = False

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class BipedRgbEnvCfg(BipedBumpyEnvCfg):
    """Rough terrain with a rendered colour camera. Requires Isaac Sim 6.0.

    Identical to the terrain rung and to `BipedDepthEnvCfg` in everything except
    the sensor, so three-way comparison -- blind, ray-cast depth, rendered RGB --
    is a comparison of modalities and not of tasks.
    """

    observations: RgbObservationsCfg = RgbObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.rgb_cam = make_rgb_camera_cfg()
        # A rendered tile per environment is not a ray-cast one. 4,096 envs of
        # TiledCamera will not fit where 4,096 ray-casters did.
        self.scene.num_envs = 512
