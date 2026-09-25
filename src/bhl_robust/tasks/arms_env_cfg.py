"""22-DoF (arms included) counterparts of the biped robustness tasks.

The biped experiments control 12 leg joints; this variant actuates all 22,
adding shoulders, elbows and the neck. Everything else -- reward terms, the
disturbance protocol, the terrain menu -- is held identical to the biped
overlays, so the arms are the only thing that changed.

Why it is worth running twice: arms are not decoration on a push-recovery task.
A humanoid rejects a lateral shove partly by swinging its arms to move angular
momentum away from the legs, and a 12-DoF biped simply cannot do that. If the
arms help, the push and terrain results should improve at matched settings; if
they do not, that is also informative, because upstream's reward set penalises
arm deviation (`joint_deviation_arms`) and may be suppressing exactly the
strategy that would help.

Upstream's humanoid config repeats the biped's `curriculums` naming bug, so
these bind to `curriculum` for the same reason.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from berkeley_humanoid_lite.tasks.locomotion.velocity.config.humanoid.env_cfg import (
    BerkeleyHumanoidLiteEnvCfg,
    CurriculumsCfg,
    EventsCfg,
)
from berkeley_humanoid_lite.tasks.locomotion.velocity import mdp

from bhl_robust.curricula.push import push_levels_adaptive
from bhl_robust.terrains.bumpy import BUMPY_TERRAINS_CFG

# Matched to the biped arms of the experiment so the two are comparable.
PUSH_INTERVAL_S = (5.0, 9.0)


@configclass
class ArmsPushEventsCfg(EventsCfg):
    """Upstream humanoid events plus the interval push it ships commented out."""

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=PUSH_INTERVAL_S,
        params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}},
    )


@configclass
class ArmsPushCurriculumCfg(CurriculumsCfg):
    """Push magnitude gated on measured fall rate, identical to the biped rule."""

    push_levels = CurrTerm(
        func=push_levels_adaptive,
        params={
            "term_name": "push_robot",
            "step": 0.02,
            "min_magnitude": 0.0,
            "max_magnitude": 1.0,
            "fall_rate_target": 0.20,
        },
    )


@configclass
class ArmsTerrainCurriculumCfg(CurriculumsCfg):
    """Terrain level promotion by distance walked."""

    terrain_levels = CurrTerm(func=mdp.terrain_levels_vel)


@configclass
class HumanoidPushAdaptiveCfg(BerkeleyHumanoidLiteEnvCfg):
    """22 DoF, competence-gated push curriculum."""

    events: ArmsPushEventsCfg = ArmsPushEventsCfg()
    curriculum: ArmsPushCurriculumCfg = ArmsPushCurriculumCfg()


@configclass
class HumanoidBumpyEnvCfg(BerkeleyHumanoidLiteEnvCfg):
    """22 DoF on generated rough terrain with the level curriculum."""

    curriculum: ArmsTerrainCurriculumCfg = ArmsTerrainCurriculumCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.terrain.terrain_type = "generator"
        self.scene.terrain.terrain_generator = BUMPY_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.visual_material = None


# --- Turning gait (2026-09-24) -------------------------------------------
#
# Measured in MuJoCo (scripts/bench/turn_test.py): the shipped 22-DoF gait
# `arms-dr1.0-s0` turns 0.2 deg in 6 s on a 0.6 rad/s yaw-rate command, while
# the 12-DoF biped `dr-default-s0` turns 239 deg through the same replay path.
# The two Isaac configs differ in exactly these places: the humanoid penalises
# hip-yaw/hip-roll and ankle-roll deviation at -1.0 (the biped: -0.2), tracks
# yaw rate through a kernel twice as wide (std 0.5 vs 0.25), and neither config
# rewards stepping under a pure-turn command (the air-time gate reads the
# linear command only). One arm per suspect, and one with everything.
import torch
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor


def feet_air_time_positive_biped_turn(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Upstream `feet_air_time_positive_biped`, gated on the FULL command norm so
    a pure-turn command (vx = vy = 0, wz != 0) still pays for stepping."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward


@configclass
class HumanoidTurnHipCfg(BerkeleyHumanoidLiteEnvCfg):
    """Arm A: the biped's leg-deviation weights (-0.2), nothing else changed."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_deviation_hip.weight = -0.2
        self.rewards.joint_deviation_ankle_roll.weight = -0.2


@configclass
class HumanoidTurnTrackCfg(BerkeleyHumanoidLiteEnvCfg):
    """Arm B: the biped's yaw-rate kernel (std 0.25) at twice the weight."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25


@configclass
class HumanoidTurnBothCfg(BerkeleyHumanoidLiteEnvCfg):
    """Arm C: A + B, plus stepping rewarded under pure-turn commands."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.joint_deviation_hip.weight = -0.2
        self.rewards.joint_deviation_ankle_roll.weight = -0.2
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.25
        self.rewards.feet_air_time.func = feet_air_time_positive_biped_turn
