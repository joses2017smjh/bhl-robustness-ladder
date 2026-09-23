"""Versioned repair and a short-to-long curriculum for the failed maze rung."""

from __future__ import annotations

import torch
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

from bhl_robust import maze_recovery as recovery
from bhl_robust.tasks import maze_env_cfg as legacy
from bhl_robust.tasks.maze_mdp import MazeWaypointCommand, MazeWaypointCommandCfg
from bhl_robust.tasks import maze_robust as _robust


class RecoveryWaypointCommand(MazeWaypointCommand):
    """Oracle route teacher with braking; not autonomous visual navigation."""

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        xy = self._estimated_xy()[ids]
        # Near-goal curriculum starts after waypoint 0: never turn around to it.
        self._wp_idx[ids] = (xy[:, 0] >= 1.15).long()
        self._apply_waypoint(ids)

    def _apply_waypoint(self, env_ids):
        super()._apply_waypoint(env_ids)
        if env_ids.numel() == 0:
            return
        xy = self._estimated_xy()[env_ids]
        idx = self._wp_idx[env_ids].clamp(max=self._path.shape[0] - 1)
        delta = self._path[idx] - xy
        distance = delta.norm(dim=-1)
        heading = recovery.tensor(self._env.scene["robot"].data.heading_w)[env_ids]
        # Evaluation-only heading-estimate error (SF-02, maze_recovery_probe --settings);
        # zero unless the probe sets it, so training is unaffected.
        heading = heading + getattr(self, "_sf02_yaw", 0.0)
        error = torch.atan2(torch.sin(self.heading_target[env_ids] - heading),
                            torch.cos(self.heading_target[env_ids] - heading))
        final = idx == self._path.shape[0] - 1
        speed = recovery.command_speed(distance, error, self.cfg.cruise_speed)
        self.vel_command_b[env_ids, 0] = torch.where(final, speed,
            self.cfg.cruise_speed * torch.clamp(torch.cos(error), min=0.0))
        self.is_standing_env[env_ids] = final & (distance <= 0.26)
        true_xy = recovery.local_xy(self._env)[env_ids]
        self.metrics["button_distance"][env_ids] = (true_xy - true_xy.new_tensor((3.0, 0.0))).norm(dim=-1)

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self.metrics["button_distance"] = torch.zeros(self.num_envs, device=self.device)


@configclass
class RecoveryWaypointCommandCfg(MazeWaypointCommandCfg):
    class_type: type = RecoveryWaypointCommand
    cruise_speed: float = 0.35


STAGE_SPAWNS = {"Approach": (2.35, 2.45), "Corridor": (1.65, 1.75), "Full": (-0.20, 0.20)}


def _configure(cfg, stage):
    cfg.commands.base_velocity = RecoveryWaypointCommandCfg(
        resampling_time_range=(1.e9, 1.e9), debug_vis=False, asset_name="robot",
        heading_command=True, heading_control_stiffness=1.0,
        rel_standing_envs=0.0, rel_heading_envs=1.0,
        ranges=MazeWaypointCommandCfg.Ranges(lin_vel_x=(0.35, 0.35),
            lin_vel_y=(0.0, 0.0), ang_vel_z=(-0.8, 0.8), heading=(0.0, 0.0)))
    cfg.rewards.progress_to_button = RewTerm(func=recovery.progress_rate, weight=4.0)
    cfg.rewards.button_reached = RewTerm(func=recovery.success_bonus, weight=12.0)
    cfg.rewards.termination_penalty = RewTerm(func=recovery.failure_penalty, weight=-10.0)
    cfg.terminations.button_reached = DoneTerm(func=recovery.button_reached)
    cfg.events.reset_base.params["pose_range"] = {
        "x": STAGE_SPAWNS[stage], "y": (-0.08, 0.08), "yaw": (-0.15, 0.15)}
    # Keep the two camera features comparable in size to the 36 lidar sectors.
    # These are ray-cast depths, not estimates produced from stereo RGB.
    for group in (cfg.observations.policy, cfg.observations.critic):
        for eye in ("stereo_l", "stereo_r"):
            if hasattr(group, eye):
                getattr(group, eye).params["pool"] = 16
    # Explicit configurable simulated-IMU channels. Values are stress-test
    # assumptions, not a calibration of the user's unspecified USB module.
    cfg.observations.policy.base_ang_vel.noise = GaussianNoiseCfg(mean=0.0, std=0.05)
    cfg.observations.policy.projected_gravity.noise = GaussianNoiseCfg(mean=0.0, std=0.02)


def _make_config(base, stage, arm):
    class StageCfg(base):
        def __post_init__(self):
            super().__post_init__()
            _configure(self, stage)
    StageCfg.__name__ = f"MazeRecovery{stage}{arm}Cfg"
    StageCfg.__qualname__ = StageCfg.__name__
    return configclass(StageCfg)


RECOVERY_CONFIGS = {}
for _stage in STAGE_SPAWNS:
    for _arm, _base in (("Blind", legacy.MazeBlindEnvCfg), ("Lidar", legacy.MazeLidarEnvCfg),
                        ("Stereo", legacy.MazeStereoEnvCfg), ("Both", legacy.MazeBothEnvCfg),
                        ("BothRobust", _robust.MazeBothRobustEnvCfg)):   # SF-04, docs/SENSOR_FUSION.md
        _cfg = _make_config(_base, _stage, _arm)
        globals()[_cfg.__name__] = _cfg
        RECOVERY_CONFIGS[f"Velocity-BHL-MazeRecovery-{_stage}-{_arm}-v0"] = _cfg
