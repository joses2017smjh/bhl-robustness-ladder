"""Waiter phase 1 task `Velocity-BHL-Waiter-WBC-v0` (docs/WAITER_PROGRAM.md, frozen 2026-10-05).

= R1 (`Velocity-BHL-Arms-TurnGaitClock-v0`, `arms_env_cfg.HumanoidTurnGaitClockCfg`: clock-s2's recipe) with ONLY:
  * the robot: the 24-DoF Waiter asset (grippers, palm colliders, arm 8 Nm / stiffness 20; `waiter_asset`);
  * actions: the 12 leg joints (upstream scale 0.25, default offset); arms and grippers follow the
    `upper_body` command (`waiter_wbc_mdp.UpperBodyCommand`), which writes their PD targets;
  * observations: joint positions and velocities over all 24 joints; the upper-body command (12) after the last
    actions; the gait clock last. Actor 3+3+3+24+24+12+12+2 = 83, critic + base linear velocity = 86;
  * events: + hand payload (startup, add U[0, 0.8] kg to each hand link) + hand forces (interval 1-3 s, each
    component U[-6, 6] N, zero with probability 0.3);
  * rewards: torque / acceleration / joint-limit penalties over the leg joints; the arm joint-deviation terms
    removed (the policy does not move the arms). Every other term, weight and setting is R1's.
"""

from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from berkeley_humanoid_lite.tasks.locomotion.velocity import mdp
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.humanoid.env_cfg import (
    CommandsCfg as _UpstreamCommandsCfg,
    CurriculumsCfg as _UpstreamCurriculumsCfg,
    ObservationsCfg as _UpstreamObservationsCfg,
)

from bhl_robust import waiter_asset as W
from bhl_robust.tasks import gait_clock_mdp, waiter_wbc_mdp
from bhl_robust.tasks.arms_env_cfg import ArmsFixedPushEventsCfg, HumanoidTurnGaitClockCfg

TASK_ID = "Velocity-BHL-Waiter-WBC-v0"
HAND_PAYLOAD_KG = (0.0, 0.8)
HAND_FORCE_N = (-6.0, 6.0)
HAND_FORCE_P_ZERO = 0.3
HAND_FORCE_INTERVAL_S = (1.0, 3.0)
OBS_POLICY = 83
OBS_CRITIC = 86

_ALL = dict(joint_names=W.JOINT_ORDER, preserve_order=True)


@configclass
class WaiterCommandsCfg(_UpstreamCommandsCfg):
    """Upstream's base_velocity (as R1) + the upper-body command."""

    upper_body = waiter_wbc_mdp.UpperBodyCommandCfg()


@configclass
class _WaiterPolicyObsCfg(_UpstreamObservationsCfg.PolicyCfg):
    def __post_init__(self):
        super().__post_init__()
        self.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", **_ALL)}
        self.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", **_ALL)}

    upper_body = ObsTerm(func=mdp.generated_commands, params={"command_name": "upper_body"})
    gait_clock = ObsTerm(func=gait_clock_mdp.gait_clock, params={"period": gait_clock_mdp.GAIT_PERIOD_S})


@configclass
class _WaiterCriticObsCfg(_UpstreamObservationsCfg.CriticCfg):
    def __post_init__(self):
        super().__post_init__()
        self.joint_pos.params = {"asset_cfg": SceneEntityCfg("robot", **_ALL)}
        self.joint_vel.params = {"asset_cfg": SceneEntityCfg("robot", **_ALL)}

    upper_body = ObsTerm(func=mdp.generated_commands, params={"command_name": "upper_body"})
    gait_clock = ObsTerm(func=gait_clock_mdp.gait_clock, params={"period": gait_clock_mdp.GAIT_PERIOD_S})


@configclass
class WaiterObservationsCfg(_UpstreamObservationsCfg):
    policy: _WaiterPolicyObsCfg = _WaiterPolicyObsCfg()
    critic: _WaiterCriticObsCfg = _WaiterCriticObsCfg()


@configclass
class WaiterActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(asset_name="robot", joint_names=W.LEG_JOINTS, scale=0.25,
                                           preserve_order=True, use_default_offset=True)


@configclass
class WaiterEventsCfg(ArmsFixedPushEventsCfg):
    """R1's events (upstream + fixed +/-0.5 m/s pushes) + hand payload + hand forces."""

    hand_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="startup",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["arm_left_hand_link", "arm_right_hand_link"]),
                "mass_distribution_params": HAND_PAYLOAD_KG, "operation": "add"},
    )
    hand_forces = EventTerm(
        func=waiter_wbc_mdp.hand_forces,
        mode="interval",
        interval_range_s=HAND_FORCE_INTERVAL_S,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["arm_left_hand_link", "arm_right_hand_link"]),
                "force_range": HAND_FORCE_N, "p_zero": HAND_FORCE_P_ZERO},
    )


@configclass
class HumanoidWaiterWbcCfg(HumanoidTurnGaitClockCfg):
    """Phase 1 WBC: R1 + the declared changes above (module docstring)."""

    commands: WaiterCommandsCfg = WaiterCommandsCfg()
    observations: WaiterObservationsCfg = WaiterObservationsCfg()
    actions: WaiterActionsCfg = WaiterActionsCfg()
    events: WaiterEventsCfg = WaiterEventsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = W.make_waiter_cfg().replace(prim_path="{ENV_REGEX_NS}/robot")
        legs = SceneEntityCfg("robot", joint_names=W.LEG_JOINTS)
        self.rewards.dof_torques_l2.params["asset_cfg"] = legs
        self.rewards.dof_acc_l2.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=W.LEG_JOINTS)
        self.rewards.dof_pos_limits.params = {"asset_cfg": SceneEntityCfg("robot", joint_names=W.LEG_JOINTS)}
        self.rewards.joint_deviation_shoulder = None
        self.rewards.joint_deviation_elbow = None


# ---- phase 1c (2026-10-07; SLURM_JOBS.md "Predeclared now ... Waiter phase 1c") ------------------------------------
# = phase 1b (this file's HumanoidWaiterWbcCfg + the action-noise std bound, set by the launcher) with ONLY:
#   * a disturbance curriculum (waiter_wbc_mdp.disturbance_curriculum): the upper-body arm goals, the hand-force range
#     and the hand-payload range are all scaled by s = 0 before iteration CURRICULUM_START_ITER, rising linearly to 1
#     at CURRICULUM_END_ITER, then 1 (the frozen phase 1 ranges); grippers, pushes and everything else as phase 1;
#   * the hand payload therefore moves from a startup event to a reset event, so its range can follow s (Isaac Lab
#     re-applies the add on the default mass at every call, so nothing accumulates);
#   * CURRICULUM_ITERS training iterations instead of 6000 (the ramp needs room at both ends).
CURRICULUM_TASK_ID = "Velocity-BHL-Waiter-WBC-Curriculum-v0"
CURRICULUM_START_ITER = 4000
CURRICULUM_END_ITER = 7000
CURRICULUM_ITERS = 10000
STEPS_PER_ITER = 24          # the PPO agent's num_steps_per_env; gpu_waiter_wbc.sbatch checks the run's agent.yaml


@configclass
class WaiterCurriculumEventsCfg(WaiterEventsCfg):
    """Phase 1's events with the hand payload resampled at every reset (range set by the curriculum)."""

    hand_payload = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        mode="reset",
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["arm_left_hand_link", "arm_right_hand_link"]),
                "mass_distribution_params": HAND_PAYLOAD_KG, "operation": "add"},
    )


@configclass
class WaiterCurriculumCfg(_UpstreamCurriculumsCfg):
    disturbance = CurrTerm(
        func=waiter_wbc_mdp.disturbance_curriculum,
        params={"start_iter": CURRICULUM_START_ITER, "end_iter": CURRICULUM_END_ITER, "steps_per_iter": STEPS_PER_ITER,
                "force_range": HAND_FORCE_N, "payload_range": HAND_PAYLOAD_KG},
    )


@configclass
class HumanoidWaiterWbcCurriculumCfg(HumanoidWaiterWbcCfg):
    """Phase 1c: phase 1 + the disturbance curriculum (module notes above)."""

    events: WaiterCurriculumEventsCfg = WaiterCurriculumEventsCfg()
    curriculum: WaiterCurriculumCfg = WaiterCurriculumCfg()


# ---- phase 1d (2026-10-08; SLURM_JOBS.md "Predeclared now ... Waiter phase 1d") ------------------------------------
# = phase 1c (HumanoidWaiterWbcCurriculumCfg, the std bound set by the launcher) with ONLY:
#   * a trained stance: standing envs (zero base command) see the gait clock HELD at gait_clock_mdp.STANCE_PHASE (0.5,
#     both feet in stance) in the actor and critic observations, and feet_gait schedules them there
#     (gait_clock_mdp.gait_clock_stance / feet_gait_stance); the arms, payloads and forces keep moving while they stand;
#   * rel_standing_envs 0.25 (upstream 0.02), so a quarter of the envs practise standing still under arm disturbances.
# Why: the 1c WBC (s7) was never trained to stand; held at a stance phase only at deploy time it stays within 1-2 cm
# with still arms but slides 5-35 cm and turns up to 20 deg while the arms move (phase 2 dev2, 2026-10-08).
STANCE_TASK_ID = "Velocity-BHL-Waiter-WBC-Stance-v0"
STANCE_REL_STANDING_ENVS = 0.25


@configclass
class HumanoidWaiterWbcStanceCfg(HumanoidWaiterWbcCurriculumCfg):
    """Phase 1d: phase 1c + the trained stance (notes above)."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.rel_standing_envs = STANCE_REL_STANDING_ENVS
        for group in (self.observations.policy, self.observations.critic):
            group.gait_clock.func = gait_clock_mdp.gait_clock_stance
            group.gait_clock.params = {"period": gait_clock_mdp.GAIT_PERIOD_S, "command_name": "base_velocity"}
        self.rewards.feet_gait.func = gait_clock_mdp.feet_gait_stance
        self.rewards.feet_gait.params["command_name"] = "base_velocity"


# ---- phase 1e (2026-10-10; SLURM_JOBS.md "Predeclared now ... Waiter phase 1e") ------------------------------------
# = phase 1d (HumanoidWaiterWbcStanceCfg, the std bound set by the launcher) with ONLY:
#   * standing envs get no pushes and no hand forces (payloads and arm goals keep moving):
#     waiter_stand_mdp.push_moving_only / hand_forces_moving_only;
#   * two standing-only penalties: stand_anchor (base displacement / yaw change since the env started standing,
#     normalised by 5 cm / 5 deg, squared, each clamped at 9; weight STAND_ANCHOR_WEIGHT) and stand_tilt (base tilt
#     normalised by 5 deg, squared, clamped at 9; weight STAND_TILT_WEIGHT).
# Why: waiter_stand_mdp module notes (s7 leans back 8-12 deg and turns 6-12 deg while its arms work; 1d shuffled).
from isaaclab.managers import RewardTermCfg as RewTerm  # noqa: E402

from bhl_robust.tasks import waiter_stand_mdp  # noqa: E402
from bhl_robust.tasks.arms_env_cfg import TurnGaitRewardsCfg  # noqa: E402

STAND_TASK_ID = "Velocity-BHL-Waiter-WBC-Stand-v0"
STAND_ANCHOR_WEIGHT = -1.0
STAND_TILT_WEIGHT = -0.5


@configclass
class WaiterStandRewardsCfg(TurnGaitRewardsCfg):
    """R1's rewards (as the waiter tasks edit them in __post_init__) + the two standing-only penalties, last."""

    stand_anchor = RewTerm(func=waiter_stand_mdp.StandAnchor, weight=STAND_ANCHOR_WEIGHT,
                           params={"command_name": "base_velocity", "pos_m": waiter_stand_mdp.ANCHOR_POS_M,
                                   "yaw_deg": waiter_stand_mdp.ANCHOR_YAW_DEG, "clamp": waiter_stand_mdp.ANCHOR_CLAMP})
    stand_tilt = RewTerm(func=waiter_stand_mdp.stand_tilt, weight=STAND_TILT_WEIGHT,
                         params={"command_name": "base_velocity", "deg": waiter_stand_mdp.TILT_DEG,
                                 "clamp": waiter_stand_mdp.TILT_CLAMP})


@configclass
class HumanoidWaiterWbcStandCfg(HumanoidWaiterWbcStanceCfg):
    """Phase 1e: phase 1d + no disturbances while standing + the standing penalties (notes above)."""

    rewards: WaiterStandRewardsCfg = WaiterStandRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.func = waiter_stand_mdp.push_moving_only
        self.events.push_robot.params["command_name"] = "base_velocity"
        self.events.hand_forces.func = waiter_stand_mdp.hand_forces_moving_only
        self.events.hand_forces.params["command_name"] = "base_velocity"


# ---- phase 1f (2026-10-11; SLURM_JOBS.md "Predeclared now ... Waiter phase 1f") ------------------------------------
# = phase 1d (HumanoidWaiterWbcStanceCfg: pushes and hand forces on every env, standing ones included) with ONLY:
#   * phase 1e's two standing-only penalties (WaiterStandRewardsCfg, same weights);
#   * the push event also schedules a new anchor for the anchor penalty 1 s after each push
#     (waiter_stand_mdp.push_and_reanchor; the push itself is upstream's, with the same velocity range).
# Why: the 1e seeds (no disturbances while standing) fell within a second of standing in MuJoCo; s7 and 1d did not.
STANDPUSH_TASK_ID = "Velocity-BHL-Waiter-WBC-StandPush-v0"


@configclass
class HumanoidWaiterWbcStandPushCfg(HumanoidWaiterWbcStanceCfg):
    """Phase 1f: phase 1d + the standing penalties; standing envs keep their pushes and hand forces (notes above)."""

    rewards: WaiterStandRewardsCfg = WaiterStandRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.func = waiter_stand_mdp.push_and_reanchor
        self.events.push_robot.params["delay_s"] = waiter_stand_mdp.REANCHOR_DELAY_S

