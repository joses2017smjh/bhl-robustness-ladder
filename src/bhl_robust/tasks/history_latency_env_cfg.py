"""H3 flat-ground information ablation; equal 63-column actor capacity."""
from __future__ import annotations

import torch
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg import (
    BerkeleyHumanoidLiteBipedEnvCfg, ObservationsCfg,
)
from bhl_robust.latency_history import CausalImuHistory


def packet_state(env):
    state = getattr(env, "_h3_imu_packets", None)
    if state is None:
        generator = torch.Generator(device=env.device)
        generator.manual_seed(int(env.cfg.seed or 0) + 330031)
        state = CausalImuHistory(env.num_envs, env.device, mode=env.cfg.h3_mode,
                                max_delay_steps=env.cfg.h3_max_delay_steps,
                                generator=generator, noise=True)
        env._h3_imu_packets = state
    return state


def reset_imu_packets(env, env_ids):
    packet_state(env).reset(env_ids)


def packets(env):
    robot = env.scene["robot"]
    clean = torch.cat([robot.data.root_ang_vel_b, robot.data.projected_gravity_b], dim=-1)
    return packet_state(env).update(clean, int(env.common_step_counter))


def delayed_angular_velocity(env):
    return packets(env)[:, -1, :3]


def delayed_gravity(env):
    return packets(env)[:, -1, 3:]


def past_imu_packets(env):
    return packets(env)[:, :-1].reshape(env.num_envs, 18)


@configclass
class H3PolicyObsCfg(ObservationsCfg.PolicyCfg):
    # Noise is sampled once at capture inside packet_state, then retained with
    # the delayed packet. Applying term noise here would corrupt old samples
    # again, and would change their noise between history columns.
    base_ang_vel = ObsTerm(func=delayed_angular_velocity)
    projected_gravity = ObsTerm(func=delayed_gravity)
    h3_past_imu = ObsTerm(func=past_imu_packets)


@configclass
class H3ObservationsCfg(ObservationsCfg):
    policy: H3PolicyObsCfg = H3PolicyObsCfg()
    # The critic is the untouched clean current teacher critic, not a history
    # actor with privileged observations accidentally inherited into it.
    critic: ObservationsCfg.CriticCfg = ObservationsCfg.CriticCfg()


@configclass
class HistoryLatencyEnvCfg(BerkeleyHumanoidLiteBipedEnvCfg):
    observations: H3ObservationsCfg = H3ObservationsCfg()
    h3_mode: str = "history"
    h3_max_delay_steps: int = 1

    def __post_init__(self):
        super().__post_init__()
        self.events.h3_imu_reset = EventTerm(func=reset_imu_packets, mode="reset")


@configclass
class FeedforwardLatencyEnvCfg(HistoryLatencyEnvCfg):
    h3_mode: str = "feedforward"
