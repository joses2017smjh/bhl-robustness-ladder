"""Separate privileged teacher and actual ray-LiDAR student observations.

The 128 body-mounted rays see the terrain mesh; they are simulated measurements,
not stereo image inference. Range noise, ray loss and causal delay are explicit.
No terrain query, absolute pose or true velocity enters the student policy API.
"""
from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, yaw_quat
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg import ObservationsCfg

from bhl_robust.research.perceptive_policy import CausalPointHistory, privileged_height, RAYS
from bhl_robust.tasks.terrain_env_cfg import BipedBumpyEnvCfg


def point_state(env):
    state = getattr(env, "_perceptive_points", None)
    if state is None:
        state = CausalPointHistory(env.num_envs, device=env.device,
                                  delay_steps=int(env.cfg.perception_delay_steps))
        env._perceptive_points = state
    return state


def reset_points(env, env_ids):
    point_state(env).reset(env_ids)


def measured_points(env):
    state = point_state(env)
    tick = int(env.common_step_counter)
    if (state.last == tick).all():
        # Avoid resampling noise when observation groups are queried twice.
        return state.update(torch.zeros(env.num_envs, RAYS, 3, device=env.device),
                            torch.zeros(env.num_envs, RAYS, dtype=torch.bool, device=env.device), tick).flatten(1)
    sensor, robot = env.scene["perceptive_lidar"], env.scene["robot"]
    hits = sensor.data.ray_hits_w
    if hits.shape != (env.num_envs, RAYS, 3):
        raise ValueError("actual LiDAR ray count differs from frozen128")
    # Simulated sensor coordinate conversion. Absolute translation cancels;
    # the student receives only body-relative, gravity-aligned point returns.
    offset = hits - robot.data.root_pos_w[:, None, :]
    q = yaw_quat(robot.data.root_quat_w)[:, None].expand(-1, RAYS, -1)
    local = quat_apply_inverse(q.reshape(-1, 4), offset.reshape(-1, 3)).reshape(-1, RAYS, 3)
    valid = torch.isfinite(local).all(-1) & (torch.linalg.vector_norm(local, dim=-1) < 5.)
    local = torch.where(valid[..., None], local, 0.)
    if env.cfg.perception_noise_m:
        local += torch.randn_like(local) * float(env.cfg.perception_noise_m)
    if env.cfg.perception_dropout:
        valid &= torch.rand_like(valid.float()) >= float(env.cfg.perception_dropout)
    return state.update(local, valid, tick).flatten(1)


def teacher_ground(env):
    return privileged_height(env.scene["robot"].data.root_pos_w[:, 2],
                             env.scene["perceptive_teacher_scan"].data.ray_hits_w[..., 2])


@configclass
class TeacherGroundObs(ObsGroup):
    ground = ObsTerm(func=teacher_ground)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class MeasuredPointsObs(ObsGroup):
    points = ObsTerm(func=measured_points)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class PerceptiveObservationsCfg(ObservationsCfg):
    teacher: TeacherGroundObs = TeacherGroundObs()
    measured: MeasuredPointsObs = MeasuredPointsObs()


@configclass
class PerceptiveTerrainEnvCfg(BipedBumpyEnvCfg):
    observations: PerceptiveObservationsCfg = PerceptiveObservationsCfg()
    perception_noise_m: float = .01
    perception_dropout: float = .10
    perception_delay_steps: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.scene.perceptive_teacher_scan = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/robot/base", offset=RayCasterCfg.OffsetCfg(pos=(0., 0., 20.)),
            ray_alignment="yaw", pattern_cfg=patterns.GridPatternCfg(resolution=.1, size=(1., .6), ordering="yx"),
            mesh_prim_paths=["/World/ground"], update_period=0., debug_vis=False)
        self.scene.perceptive_lidar = RayCasterCfg(
            prim_path="{ENV_REGEX_NS}/robot/base", offset=RayCasterCfg.OffsetCfg(pos=(.1, 0., .38)),
            ray_alignment="base", pattern_cfg=patterns.LidarPatternCfg(channels=8,
                vertical_fov_range=(-75., -12.), horizontal_fov_range=(-60., 60.), horizontal_res=8.),
            mesh_prim_paths=["/World/ground"], max_distance=5., update_period=0., debug_vis=False)
        self.events.perceptive_reset = EventTerm(func=reset_points, mode="reset")
        self.commands.base_velocity.debug_vis = False
        # Bounded, robot-scaled curriculum; the original terrain menu is untouched.
        self.scene.terrain.terrain_generator = self.scene.terrain.terrain_generator.copy()
        generator = self.scene.terrain.terrain_generator
        generator.num_rows, generator.num_cols, generator.use_cache = 5, 4, False
        generator.sub_terrains["rough"].noise_range = (0., .02)
        for name in ("slope_up", "slope_down"):
            generator.sub_terrains[name].slope_range = (0., .10)
        generator.sub_terrains["obstacles"].obstacle_height_range = (0., .025)
        # Fixed difficulty allocation makes student arms comparable; curriculum
        # advancement is not silently carried from one arm to the next.
        self.curriculum.terrain_levels = None
        self.scene.terrain.max_init_terrain_level = 4
