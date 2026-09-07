"""The added sensor rig: a 2D lidar and a synchronised global-shutter stereo pair.

Both are modelled on parts that exist, because a sensor invented to suit a task
teaches nothing about the robot that would carry it.

**RPLIDAR C1** — 360 degrees in a single plane, 0.05-12 m, 10 Hz, 0.72 deg
angular step (500 samples a revolution). Modelled with `LidarPatternCfg`, which
is a ray-cast and therefore costs what the depth rung already measured: ray-cast
sensing runs at 1.6% of throughput at 4,096 envs, against an RTX camera that
needs a working renderer. That matters here because 5.1's RTX segfaults on this
cluster, so a ray-cast sensor is the one that works on both stacks.

**MMlove global-shutter stereo** — two synchronised cameras on a fixed baseline.
Global shutter is the part worth modelling: a rolling-shutter camera on a walking
biped skews every frame, and this project's own locomotion clips show the base
oscillating several centimetres a step. Simulation has no rolling shutter, so
what is actually modelled is the *baseline* -- the geometry that makes stereo
depth possible -- and the honest note that a rolling-shutter part would be worse
on hardware in a way the sim cannot show.

The 7 inch screen is an output device. It does not enter the observation and is
not modelled; it is listed here so its absence is deliberate rather than
forgotten.
"""

from __future__ import annotations

import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, RayCasterCameraCfg, patterns

# --------------------------------------------------------------------- lidar

#: RPLIDAR C1, from the datasheet.
LIDAR_RANGE = 12.0
LIDAR_RES_DEG = 0.72
LIDAR_HZ = 10.0
#: Mounted on the torso, above the arms' swing so a moving arm is not an
#: obstacle. The real part is 55 mm tall and would sit on the shoulder deck.
LIDAR_POS = (0.0, 0.0, 0.34)

#: 360 / 0.72 = 500 rays. Pooled before it reaches the policy, like depth_obs:
#: 500 raw ranges would be 90% of the observation width and the first layer
#: would be almost entirely lidar weights.
LIDAR_RAYS = int(round(360.0 / LIDAR_RES_DEG))
LIDAR_SECTORS = 36               # 10 degrees a sector


def make_lidar_cfg(mesh_paths: list[str] | None = None) -> RayCasterCfg:
    """A single-plane 360 degree scanner on the torso."""
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        mesh_prim_paths=mesh_paths or ["/World/ground"],
        offset=RayCasterCfg.OffsetCfg(pos=LIDAR_POS),
        # `ray_alignment="base"` keeps the scan plane fixed to the torso, which
        # is what a deck-mounted scanner does. This Isaac Lab drops the older
        # `attach_yaw_only` flag in favour of it.
        ray_alignment="base",
        pattern_cfg=patterns.LidarPatternCfg(
            channels=1,
            vertical_fov_range=(0.0, 0.0),        # one plane, as the part scans
            horizontal_fov_range=(-180.0, 180.0),
            horizontal_res=LIDAR_RES_DEG,
        ),
        max_distance=LIDAR_RANGE,
        update_period=1.0 / LIDAR_HZ,             # 10 Hz, not every physics step
        debug_vis=False,
    )


def lidar_obs(env, sensor_cfg: SceneEntityCfg, sectors: int = LIDAR_SECTORS,
              clip: float = LIDAR_RANGE) -> torch.Tensor:
    """Ranges reduced to per-sector minima, scaled to roughly [0, 1].

    Minimum rather than mean: a sector containing one wall and a lot of open
    space is an obstacle, and averaging would hide it. Scaled by range so the
    term sits on the same scale as the proprioceptive ones.
    """
    sensor = env.scene.sensors[sensor_cfg.name]
    hits = sensor.data.ray_hits_w
    hits = hits.torch if hasattr(hits, "torch") else hits
    pos = sensor.data.pos_w
    pos = pos.torch if hasattr(pos, "torch") else pos
    d = torch.linalg.norm(hits - pos.unsqueeze(1), dim=-1)
    # A ray that hits nothing comes back as inf or as the max; both mean "clear".
    d = torch.nan_to_num(d, nan=clip, posinf=clip).clamp(0.0, clip)
    n = d.shape[1] - (d.shape[1] % sectors)
    return d[:, :n].view(d.shape[0], sectors, -1).min(dim=-1).values / clip


# -------------------------------------------------------------------- stereo

#: MMlove synchronised global-shutter pair. 60 mm baseline is the common module
#: geometry; it sets the depth resolution stereo can resolve at a given range.
STEREO_BASELINE = 0.060
STEREO_POS_L = (0.12, +STEREO_BASELINE / 2, 0.30)
STEREO_POS_R = (0.12, -STEREO_BASELINE / 2, 0.30)
#: Same 20 degree down-pitch as the existing depth rung, so the two are
#: comparable and any difference is the sensor rather than where it points.
STEREO_ROT = (0.9848, 0.0, 0.1736, 0.0)
STEREO_RANGE = 6.0


def make_stereo_cfg(side: str, res: int = 64,
                    mesh_paths: list[str] | None = None) -> RayCasterCameraCfg:
    """One eye of the pair. Call twice; the baseline is the whole point."""
    if side not in ("left", "right"):
        raise ValueError(f"side must be 'left' or 'right', got {side!r}")
    return RayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        mesh_prim_paths=mesh_paths or ["/World/ground"],
        offset=RayCasterCameraCfg.OffsetCfg(
            pos=STEREO_POS_L if side == "left" else STEREO_POS_R,
            rot=STEREO_ROT, convention="world"),
        data_types=["distance_to_image_plane"],
        depth_clipping_behavior="max",
        max_distance=STEREO_RANGE,
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            focal_length=18.0, horizontal_aperture=20.955, width=res, height=res),
        update_period=0.0,
        debug_vis=False,
    )
