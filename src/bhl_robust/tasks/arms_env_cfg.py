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


# --- Turning gait, arm TurnCmd (2026-09-26) ----------------------------------
#
# Settled-stand check (turn command after a 3.0 s warm-up, both directions, reset
# seeds 0-2): only TurnBoth-s0 turns both ways (2026-09-27: TurnTrack-s0 turns
# in the -wz direction only); the other 10 of 12 sit in a
# standing fixed point and never lift a foot under a sustained (0, 0, wz)
# command, which upstream's heading-derived generator almost never produces
# (~0.2-1.4 % of steps). TurnCmd = TurnBoth's reward changes + a command mix
# that trains sustained pure turns (25 % of envs) and directly sampled yaw rate
# at low speed (25 %); the other 50 % are upstream's. See turn_command.py.
# Command dimension unchanged -> 75 obs / 22 actions, same export and deploy path.
from bhl_robust.tasks.turn_command import TurnMixVelocityCommandCfg


@configclass
class HumanoidTurnCmdCfg(HumanoidTurnBothCfg):
    """Arm D: TurnBoth + a command distribution with sustained pure turns."""

    def __post_init__(self):
        super().__post_init__()
        old = self.commands.base_velocity
        self.commands.base_velocity = TurnMixVelocityCommandCfg(
            resampling_time_range=old.resampling_time_range,
            debug_vis=old.debug_vis,
            asset_name=old.asset_name,
            heading_command=old.heading_command,
            heading_control_stiffness=old.heading_control_stiffness,
            rel_standing_envs=old.rel_standing_envs,
            rel_heading_envs=old.rel_heading_envs,
            ranges=old.ranges,
            rel_pure_turn_envs=0.25,
            rel_direct_envs=0.25,
            pure_turn_ang_vel_abs=(0.3, 1.0),
            direct_lin_vel_x=(-0.5, 0.5),
            direct_lin_vel_y=(-0.25, 0.25),
            direct_ang_vel_z=(-1.0, 1.0),
        )


# --- Turning gait, arm TurnRest (2026-09-27) ---------------------------------
#
# Diagnosis (scripts/bench/turn_diagnose.py mujoco, all 12 turning-arm
# checkpoints): 10 of 12 never lift a foot under a pure turn although their
# actions respond to wz; TurnTrack-s0 steps and turns in one direction only
# (once from a true standstill); TurnBoth-s0 is the only one that turns both
# ways, while it is still stepping in place. Pooled, a step was started from
# rest in 1 of 50 at-rest runs. See turn_command.py. TurnRest = TurnBoth's reward
# set, unchanged, with a command mix that adds rest-then-turn envs (zero command
# for 1.5-4 s, then a sustained pure turn; turn_command.TurnRestMixVelocityCommand).
# It is trained as a FINE-TUNE of arms-turn-turnboth-s0 (model_5999.pt) by
# slurm/repo20260923/gpu_turngait_v4.sbatch, so the reward set, and hence the
# value function it resumes, match the parent's. Obs 75 / actions 22 unchanged.
from bhl_robust.tasks.turn_command import TurnRestMixVelocityCommandCfg


@configclass
class HumanoidTurnRestCfg(HumanoidTurnBothCfg):
    """Arm E: TurnBoth rewards + pure-turn / rest-then-turn / direct / upstream command mix."""

    def __post_init__(self):
        super().__post_init__()
        old = self.commands.base_velocity
        self.commands.base_velocity = TurnRestMixVelocityCommandCfg(
            resampling_time_range=old.resampling_time_range,
            debug_vis=old.debug_vis,
            asset_name=old.asset_name,
            heading_command=old.heading_command,
            heading_control_stiffness=old.heading_control_stiffness,
            rel_standing_envs=old.rel_standing_envs,
            rel_heading_envs=old.rel_heading_envs,
            ranges=old.ranges,
            rel_pure_turn_envs=0.15,
            rel_rest_turn_envs=0.25,
            rel_direct_envs=0.20,
            rest_time_range=(1.5, 4.0),
            pure_turn_ang_vel_abs=(0.3, 1.0),
            direct_lin_vel_x=(-0.5, 0.5),
            direct_lin_vel_y=(-0.25, 0.25),
            direct_ang_vel_z=(-1.0, 1.0),
        )


# --- Turning gait, arm TurnRestPush (2026-09-27) ------------------------------
#
# TurnRest turned both ways 6/6 on 2/2 seeds but fell in 29/60 and 31/60 matched
# pushes (gate <= 9/60) against 7/60 for its parent TurnBoth-s0; neither trained
# with pushes. TurnRestPush = TurnRest unchanged (rewards + command mix) + the
# interval push with the competence-gated push curriculum, exactly as
# Velocity-BHL-Arms-PushAdaptive-v0. Fine-tuned from arms-turn-turnboth-s0 by
# slurm/repo20260923/gpu_turngait_v5.sbatch, beside a continued-TurnBoth control.
@configclass
class HumanoidTurnRestPushCfg(HumanoidTurnRestCfg):
    """Arm F: TurnRest + interval push with the adaptive push curriculum."""

    events: ArmsPushEventsCfg = ArmsPushEventsCfg()
    curriculum: ArmsPushCurriculumCfg = ArmsPushCurriculumCfg()


# --- Turning gait R1 / R2: gait clock + contact schedule (2026-10-01) ---------
#
# docs/SOLUTIONS_2026-10-01.md section 2: the arm checkpoints step only under their
# own training noise, so the noise-free gates and the robot see a standing mean
# policy, and nothing pays stepping at zero command. Two NEW tasks, trained from
# scratch (no parent checkpoint) by slurm/repo20260923/gpu_turngait_r12.sbatch.
# Each = TurnBoth's rewards and command mix (upstream UniformVelocityCommand) with
# these changes ONLY (frozen constants in gait_clock_mdp.py):
#   + feet_gait          contact schedule, period 0.8 s, offsets [0.0, 0.5] (L, R),
#                        stance < 0.55, weight 0.5, paid at every command incl. zero
#   + feet_swing_height  -20 x sum over swing feet of (z_foot - 0.05)^2
#   feet_air_time        weight 0
#   + push_robot         interval 5-9 s (PUSH_INTERVAL_S), x and y in [-0.5, 0.5] m/s,
#                        fixed magnitude from iteration 0, no push curriculum
# R1 TurnGaitClock: sin/cos(2 pi phase_left) appended to the actor AND critic
#   observations (actor 75 -> 77; deploy via bhl_robust.eval.gait_clock).
# R2 TurnGaitCritic: the clock goes to the critic only; the actor keeps its 75.
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from berkeley_humanoid_lite.tasks.locomotion.velocity.config.humanoid.env_cfg import (
    ObservationsCfg as _HumanoidObservationsCfg,
    RewardsCfg as _HumanoidRewardsCfg,
)

from bhl_robust.tasks import gait_clock_mdp

# (left, right) in this order: preserve_order makes body_ids follow the patterns, so
# feet_gait's offsets [0.0, 0.5] land on (left, right).
_FEET_LR = [".*_left_ankle_roll", ".*_right_ankle_roll"]


@configclass
class TurnGaitRewardsCfg(_HumanoidRewardsCfg):
    """Upstream humanoid rewards + the contact schedule and the swing-height penalty."""

    feet_gait = RewTerm(
        func=gait_clock_mdp.feet_gait,
        weight=gait_clock_mdp.FEET_GAIT_WEIGHT,
        params={
            "period": gait_clock_mdp.GAIT_PERIOD_S,
            "offset": list(gait_clock_mdp.GAIT_OFFSETS),
            "threshold": gait_clock_mdp.STANCE_THRESHOLD,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(_FEET_LR), preserve_order=True),
        },
    )
    feet_swing_height = RewTerm(
        func=gait_clock_mdp.feet_swing_height,
        weight=gait_clock_mdp.SWING_HEIGHT_WEIGHT,
        params={
            "target_height": gait_clock_mdp.SWING_HEIGHT_TARGET_M,
            "foot_height_offset": gait_clock_mdp.FOOT_ORIGIN_ABOVE_SOLE_M,
            "force_threshold": gait_clock_mdp.CONTACT_FORCE_THRESHOLD_N,
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(_FEET_LR), preserve_order=True),
            "asset_cfg": SceneEntityCfg("robot", body_names=list(_FEET_LR), preserve_order=True),
        },
    )


@configclass
class ArmsFixedPushEventsCfg(EventsCfg):
    """Upstream humanoid events + the interval push at a FIXED +/-0.5 m/s (no curriculum)."""

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=PUSH_INTERVAL_S,
        params={"velocity_range": {"x": (-gait_clock_mdp.PUSH_VELOCITY_MPS, gait_clock_mdp.PUSH_VELOCITY_MPS),
                                   "y": (-gait_clock_mdp.PUSH_VELOCITY_MPS, gait_clock_mdp.PUSH_VELOCITY_MPS)}},
    )


@configclass
class _GaitClockPolicyObsCfg(_HumanoidObservationsCfg.PolicyCfg):
    """Upstream actor terms (75) + gait_clock (2), appended last: 77."""

    gait_clock = ObsTerm(func=gait_clock_mdp.gait_clock, params={"period": gait_clock_mdp.GAIT_PERIOD_S})


@configclass
class _GaitClockCriticObsCfg(_HumanoidObservationsCfg.CriticCfg):
    """Upstream critic terms (78) + gait_clock (2), appended last: 80."""

    gait_clock = ObsTerm(func=gait_clock_mdp.gait_clock, params={"period": gait_clock_mdp.GAIT_PERIOD_S})


@configclass
class GaitClockActorObservationsCfg(_HumanoidObservationsCfg):
    """R1: the clock in the actor and the critic."""

    policy: _GaitClockPolicyObsCfg = _GaitClockPolicyObsCfg()
    critic: _GaitClockCriticObsCfg = _GaitClockCriticObsCfg()


@configclass
class GaitClockCriticObservationsCfg(_HumanoidObservationsCfg):
    """R2: the clock in the critic only; the actor keeps upstream's 75."""

    critic: _GaitClockCriticObsCfg = _GaitClockCriticObsCfg()


@configclass
class _HumanoidTurnGaitBaseCfg(HumanoidTurnBothCfg):
    """TurnBoth rewards + feet_gait + feet_swing_height, feet_air_time weight 0, fixed pushes."""

    rewards: TurnGaitRewardsCfg = TurnGaitRewardsCfg()
    events: ArmsFixedPushEventsCfg = ArmsFixedPushEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.rewards.feet_air_time.weight = 0.0


@configclass
class HumanoidTurnGaitClockCfg(_HumanoidTurnGaitBaseCfg):
    """R1 (Velocity-BHL-Arms-TurnGaitClock-v0): gait clock in the actor, 77 actor observations."""

    observations: GaitClockActorObservationsCfg = GaitClockActorObservationsCfg()


@configclass
class HumanoidTurnGaitCriticCfg(_HumanoidTurnGaitBaseCfg):
    """R2 (Velocity-BHL-Arms-TurnGaitCritic-v0): gait clock in the critic only, 75 actor observations."""

    observations: GaitClockCriticObservationsCfg = GaitClockCriticObservationsCfg()
