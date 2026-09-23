"""SF-04: the `BothRobust` maze-recovery arm (docs/SENSOR_FUSION.md).

Same observation layout as the `Both` arm, so checkpoints and probes are
shape-compatible, but with the sensor realism the fusion plan asks for, each
resampled per episode by a reset event:

* whole-modality dropout: LiDAR sectors or the stereo pair zeroed for the
  episode (p = 0.2 each), the Sensor-Dropout recipe;
* IMU bias: a constant per-episode gyro bias (std 0.02 rad/s) and gravity
  bias (std 0.02) under the usual white noise;
* IMU delay: 0 or 1 policy step (20 ms at 50 Hz), inside the ~30 ms budget
  SF-03b measured.

The critic keeps the clean `Both` terms plus base linear velocity (asymmetric
actor-critic). Train with BHL_POLICY=recurrent so the actor has the memory a
dropped modality requires.
"""

from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from bhl_robust.robust_sensing import RobustSensingState
from bhl_robust.sensors_rig import LIDAR_SECTORS, lidar_obs
from bhl_robust.tasks.depth_env_cfg import depth_obs
from bhl_robust.tasks.maze_env_cfg import BothObsCfg, MazeBothEnvCfg

P_LIDAR_OFF = 0.2
P_STEREO_OFF = 0.2
GYRO_BIAS_STD = 0.02
GRAVITY_BIAS_STD = 0.02
MAX_DELAY_STEPS = 1


def robust_state(env) -> RobustSensingState:
    st = getattr(env, "_sf04_state", None)
    if st is None:
        gen = torch.Generator(device=env.device)
        gen.manual_seed(int(getattr(env.cfg, "seed", 0) or 0) + 4004)
        st = RobustSensingState(env.num_envs, env.device, p_lidar_off=P_LIDAR_OFF, p_stereo_off=P_STEREO_OFF,
                                gyro_bias_std=GYRO_BIAS_STD, gravity_bias_std=GRAVITY_BIAS_STD,
                                max_delay_steps=MAX_DELAY_STEPS, generator=gen)
        env._sf04_state = st
    return st


def sf04_resample(env, env_ids):
    """Reset-mode event: draw this episode's dropout mask, biases and delay."""
    robust_state(env).resample(env_ids)


def ang_vel_robust(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return robust_state(env).angular_velocity(mdp.base_ang_vel(env, asset_cfg))


def gravity_robust(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    return robust_state(env).gravity(mdp.projected_gravity(env, asset_cfg))


def lidar_robust(env, sensor_cfg: SceneEntityCfg, sectors: int = LIDAR_SECTORS) -> torch.Tensor:
    return robust_state(env).lidar(lidar_obs(env, sensor_cfg, sectors))


def depth_robust(env, sensor_cfg: SceneEntityCfg, pool: int = 4) -> torch.Tensor:
    return robust_state(env).stereo(depth_obs(env, sensor_cfg, pool))


@configclass
class BothRobustObsCfg(BothObsCfg):
    """Both arm layout with dropout, IMU bias and delay on the actor only."""

    @configclass
    class PolicyCfg(BothObsCfg.PolicyCfg):
        base_ang_vel = ObsTerm(func=ang_vel_robust, noise=GaussianNoiseCfg(mean=0.0, std=0.05))
        projected_gravity = ObsTerm(func=gravity_robust, noise=GaussianNoiseCfg(mean=0.0, std=0.02))
        stereo_l = ObsTerm(func=depth_robust, params={"sensor_cfg": SceneEntityCfg("stereo_l"), "pool": 4},
                           noise=GaussianNoiseCfg(mean=0.0, std=0.0033))
        stereo_r = ObsTerm(func=depth_robust, params={"sensor_cfg": SceneEntityCfg("stereo_r"), "pool": 4},
                           noise=GaussianNoiseCfg(mean=0.0, std=0.0033))
        lidar = ObsTerm(func=lidar_robust, params={"sensor_cfg": SceneEntityCfg("lidar"), "sectors": LIDAR_SECTORS},
                        noise=GaussianNoiseCfg(mean=0.0, std=0.0025))

    policy: PolicyCfg = PolicyCfg()
    critic: BothObsCfg.CriticCfg = BothObsCfg.CriticCfg()


@configclass
class MazeBothRobustEnvCfg(MazeBothEnvCfg):
    observations: BothRobustObsCfg = BothRobustObsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.sf04_resample = EventTerm(func=sf04_resample, mode="reset")
