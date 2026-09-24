"""SF-04: the sensor-realism maze-recovery arms (docs/SENSOR_FUSION.md).

All four arms share the `Both` observation layout (45 proprio + 32 stereo +
36 lidar, same term order), so `Both` checkpoints and probes are
shape-compatible and a fine-tune can start from the working `Both` Full run.
Each arm resamples its own perturbations per episode by a reset event; which
perturbations are on is a set of plain fields on the env cfg (`cfg.sf04`,
an `SF04ParamsCfg`), not module constants:

* `BothRobust` — all three: whole-modality dropout (LiDAR sectors or the
  stereo pair zeroed for the episode, p = 0.2 each), a constant per-episode
  gyro/gravity bias (std 0.02) under the usual white noise, and an IMU delay
  of 0 or 1 policy step (20 ms at 50 Hz, inside the ~30 ms budget SF-03b
  measured);
* `BothDelay` — the IMU delay only;
* `BothDrop`  — the modality dropout only;
* `BothBias`  — the IMU bias only.

The parameter table lives in `bhl_robust.robust_sensing.SF04_ARM_PARAMS`
(pure torch, unit-tested). The critic keeps the clean `Both` terms plus base
linear velocity (asymmetric actor-critic). The state is created lazily on
first use and cached on `env._sf04_state`; the evaluation probe overrides
its `force_*` / bias-std attributes directly and resamples, which works for
every arm because the parameters are only read at construction.
"""

from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg

import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from bhl_robust.robust_sensing import SF04_ARM_PARAMS, RobustSensingState, sf04_params
from bhl_robust.sensors_rig import LIDAR_SECTORS, lidar_obs
from bhl_robust.tasks.depth_env_cfg import depth_obs
from bhl_robust.tasks.maze_env_cfg import BothObsCfg, MazeBothEnvCfg


@configclass
class SF04ParamsCfg:
    """Per-episode sensing perturbations; plain float/int fields so Hydra's
    to_dict/from_dict round trip and configclass deep copies carry them."""

    p_lidar_off: float = SF04_ARM_PARAMS["BothRobust"]["p_lidar_off"]
    p_stereo_off: float = SF04_ARM_PARAMS["BothRobust"]["p_stereo_off"]
    gyro_bias_std: float = SF04_ARM_PARAMS["BothRobust"]["gyro_bias_std"]
    gravity_bias_std: float = SF04_ARM_PARAMS["BothRobust"]["gravity_bias_std"]
    max_delay_steps: int = SF04_ARM_PARAMS["BothRobust"]["max_delay_steps"]


def robust_state(env) -> RobustSensingState:
    """Lazy, cached per-env sensing state built from `env.cfg.sf04`."""
    st = getattr(env, "_sf04_state", None)
    if st is None:
        params = sf04_params(getattr(getattr(env, "cfg", None), "sf04", None))
        gen = torch.Generator(device=env.device)
        gen.manual_seed(int(getattr(env.cfg, "seed", 0) or 0) + 4004)
        st = RobustSensingState(env.num_envs, env.device, generator=gen, **params)
        env._sf04_state = st
        print(f"[sf04] sensing state: {params}", flush=True)
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
    """Both arm layout with dropout, IMU bias and delay on the actor only.

    Term order is inherited from BothObsCfg.PolicyCfg unchanged; only the
    functions behind the affected terms differ."""

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
    """All three ingredients (dropout + bias + delay)."""

    observations: BothRobustObsCfg = BothRobustObsCfg()
    sf04: SF04ParamsCfg = SF04ParamsCfg(**SF04_ARM_PARAMS["BothRobust"])

    def __post_init__(self):
        super().__post_init__()
        self.events.sf04_resample = EventTerm(func=sf04_resample, mode="reset")


@configclass
class MazeBothDelayEnvCfg(MazeBothRobustEnvCfg):
    """IMU delay 0-1 policy step only (no dropout, no bias)."""

    sf04: SF04ParamsCfg = SF04ParamsCfg(**SF04_ARM_PARAMS["BothDelay"])


@configclass
class MazeBothDropEnvCfg(MazeBothRobustEnvCfg):
    """LiDAR / stereo whole-modality dropout p 0.2 each only (no delay, no bias)."""

    sf04: SF04ParamsCfg = SF04ParamsCfg(**SF04_ARM_PARAMS["BothDrop"])


@configclass
class MazeBothBiasEnvCfg(MazeBothRobustEnvCfg):
    """Gyro / gravity bias std 0.02 only (no delay, no dropout)."""

    sf04: SF04ParamsCfg = SF04ParamsCfg(**SF04_ARM_PARAMS["BothBias"])


SF04_ARM_CFGS = {
    "BothRobust": MazeBothRobustEnvCfg,
    "BothDelay": MazeBothDelayEnvCfg,
    "BothDrop": MazeBothDropEnvCfg,
    "BothBias": MazeBothBiasEnvCfg,
}
