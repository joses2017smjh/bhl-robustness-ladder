"""Push-recovery variants of the BHL biped locomotion task.

Two configs:
  * `BipedPushEnvCfg`       - push enabled at fixed full magnitude (ablation)
  * `BipedPushCurriculumCfg`- push magnitude ramped by a curriculum (the method)

The ablation exists so the curriculum's contribution is separable from the mere
presence of pushes. Without it, a win could just mean "pushes help", which is
already known.
"""

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass

from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg import (
    BerkeleyHumanoidLiteBipedEnvCfg,
    CurriculumsCfg,
    EventsCfg,
)
from berkeley_humanoid_lite.tasks.locomotion.velocity import mdp

from bhl_robust.curricula.push import push_velocity_levels, push_levels_adaptive


# Peak push magnitude, m/s applied instantaneously to the base. 1.5 m/s on a
# ~5kg machine is a genuine stagger, not a nudge, but stays inside what the
# ankle actuators can plausibly arrest.
PUSH_PEAK_MPS = 1.5

# Push every 4-7s. The episode is 20s, so a robot sees 3-5 disturbances per
# episode: frequent enough to learn from, sparse enough to re-stabilise between.
PUSH_INTERVAL_S = (4.0, 7.0)


@configclass
class PushEventsCfg(EventsCfg):
    """Upstream events plus the interval push that upstream ships commented out."""

    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=PUSH_INTERVAL_S,
        params={
            "velocity_range": {
                "x": (-PUSH_PEAK_MPS, PUSH_PEAK_MPS),
                "y": (-PUSH_PEAK_MPS, PUSH_PEAK_MPS),
            }
        },
    )


@configclass
class PushCurriculumsCfg(CurriculumsCfg):
    """Ramp the push from zero to full magnitude over training."""

    push_levels = CurrTerm(
        func=push_velocity_levels,
        params={
            "term_name": "push_robot",
            "start_magnitude": 0.0,
            "end_magnitude": PUSH_PEAK_MPS,
            # 4096 envs x 24 steps/iter -> common_step_counter advances 24 per
            # iteration. Hold flat for ~1000 iters, ramp over the next ~3000.
            "start_step": 24_000,
            "full_step": 96_000,
        },
    )


@configclass
class BipedPushEnvCfg(BerkeleyHumanoidLiteBipedEnvCfg):
    """Ablation: pushes on at full strength from the first step."""

    events: PushEventsCfg = PushEventsCfg()


@configclass
class BipedPushCurriculumCfg(BerkeleyHumanoidLiteBipedEnvCfg):
    """The method: push magnitude ramped by curriculum.

    Note the field is `curriculum`, not `curriculums`. Isaac Lab builds its
    CurriculumManager from `cfg.curriculum` (singular); upstream BHL declares
    `curriculums` (plural), which Isaac Lab never reads. That is why upstream's
    CurriculumsCfg is empty and its `terrain_levels_vel` helper is unreachable
    dead code. Binding to the plural name here would silently no-op the entire
    experiment while still training and reporting a plausible-looking number.
    """

    events: PushEventsCfg = PushEventsCfg()
    curriculum: PushCurriculumsCfg = PushCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Start at zero so the curriculum, not this literal, sets the schedule.
        self.events.push_robot.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}


@configclass
class PushAdaptiveCurriculumsCfg(CurriculumsCfg):
    """Magnitude gated on measured fall rate rather than on iteration count."""

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
class BipedPushAdaptiveCfg(BerkeleyHumanoidLiteBipedEnvCfg):
    """Adaptive arm: the curriculum cannot outrun the policy."""

    events: PushEventsCfg = PushEventsCfg()
    curriculum: PushAdaptiveCurriculumsCfg = PushAdaptiveCurriculumsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.events.push_robot.params["velocity_range"] = {"x": (0.0, 0.0), "y": (0.0, 0.0)}
        self.events.push_robot.interval_range_s = (5.0, 9.0)
