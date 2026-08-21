"""Privileged height-scan teacher for the terrain-plateau question.

Section 3 reports the terrain curriculum stalling at level 1.4 of 9 and
attributes it to "6 Nm joints with no height sensing". That is two hypotheses
in one sentence, and they are separable: give the policy perfect terrain
knowledge and see whether the plateau moves.

The scan is a `RayCasterCfg` grid under and ahead of the base -- the standard
privileged observation, and the same warp ray-cast machinery the depth camera
in `depth_env_cfg` uses, so neither needs the RTX renderer.

`ray_alignment="yaw"` (upstream's `attach_yaw_only`) is not cosmetic: with full
base alignment the grid pitches and rolls with the torso, so the same ground
reads differently depending on how the robot happens to be leaning, and the
observation stops being a height map.

Three outcomes, all publishable:

* teacher climbs past 1.4 -- the plateau is a sensing limit, and this quantifies
  what a sensor on BHL would buy.
* teacher also plateaus at 1.4 -- the plateau is torque. Even with perfect
  terrain knowledge an 11.3 kg robot with 6 Nm joints cannot do better.
* student matches teacher -- proprioception already carries the terrain
  information, which is what section 3's "randomization alone reaches d = 0.4"
  was already hinting at.
"""

from __future__ import annotations

import berkeley_humanoid_lite.tasks.locomotion.velocity.mdp as mdp
from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.env_cfg import (
    ObservationsCfg,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCfg, patterns
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
)

from berkeley_humanoid_lite.tasks.locomotion.velocity.config.biped.agents.rsl_rl_ppo_cfg import (
    BerkeleyHumanoidLiteBipedPPORunnerCfg as _BipedPPO,
)
from bhl_robust.tasks.terrain_env_cfg import BipedBumpyEnvCfg

# 1.0 m ahead-and-behind by 0.6 m across at 10 cm spacing: 11 x 7 = 77 points.
# Sized to a stride, not to a room -- this robot's foot lands ~15 cm ahead.
SCAN_PATTERN = patterns.GridPatternCfg(resolution=0.1, size=(1.0, 0.6))
SCAN_HEIGHT = 20.0
SCAN_CLIP = 1.0


def height_scan_obs(env, sensor_cfg: SceneEntityCfg, offset: float = 0.30):
    """Base height above each grid point, clipped -- upstream's `height_scan`.

    Re-implemented here rather than imported because upstream's mdp module does
    not re-export it, and the clip bound matters: unclipped, a ray that misses
    the terrain entirely returns the sensor's 20 m drop height and dominates the
    observation's scale.
    """
    sensor = env.scene[sensor_cfg.name]
    z = sensor.data.pos_w[:, 2].unsqueeze(1) - sensor.data.ray_hits_w[..., 2] - offset
    return z.clip(-SCAN_CLIP, SCAN_CLIP)


def make_height_scanner_cfg() -> RayCasterCfg:
    return RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, SCAN_HEIGHT)),
        ray_alignment="yaw",
        pattern_cfg=SCAN_PATTERN,
        mesh_prim_paths=["/World/ground"],
        update_period=0.0,
        debug_vis=False,
    )


# The three groups are module-level rather than nested inside the observations
# config. Nesting a new group beside the two upstream already declares sends
# isaaclab's configclass machinery into unbounded recursion when the cfg is
# copied by `parse_env_cfg`; flat classes construct identically and do not.
@configclass
class ScanTeacherObsCfg(ObservationsCfg.PolicyCfg):
    """Proprioception plus the privileged height scan."""

    height_scan = ObsTerm(
        func=height_scan_obs,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        noise=Unoise(n_min=-0.02, n_max=0.02),
        clip=(-SCAN_CLIP, SCAN_CLIP),
    )


@configclass
class ScanCriticObsCfg(ScanTeacherObsCfg):
    """The teacher group plus true base velocity, which nothing deploys with."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel)

    def __post_init__(self):
        self.enable_corruption = False


@configclass
class ScanObservationsCfg(ObservationsCfg):
    """Upstream's `policy` group kept intact, with `teacher` published beside it.

    The scan goes in its own group rather than being appended to `policy`, so a
    student distilled from this teacher keeps a byte-identical 45-dim
    proprioceptive observation and stays deployable through the same sim2real
    path as every other policy here.
    """

    teacher: ScanTeacherObsCfg = ScanTeacherObsCfg()
    critic: ScanCriticObsCfg = ScanCriticObsCfg()


@configclass
class BipedScanEnvCfg(BipedBumpyEnvCfg):
    """Terrain task exposing proprioception and a privileged height scan.

    Both groups are published: `policy` is the unchanged 45-dim proprioceptive
    vector, `teacher` is that plus the scan. Which one reaches the actor is a
    property of the *agent* config's `obs_groups`, not of the environment, so a
    single env serves the teacher run and the distillation run without either
    of them redefining the observation.
    """

    observations: ScanObservationsCfg = ScanObservationsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.height_scanner = make_height_scanner_cfg()


@configclass
class ScanTeacherPPORunnerCfg(_BipedPPO):
    """PPO on the privileged group. Everything else matches the terrain rung."""

    experiment_name = "biped"
    obs_groups = {"policy": ["teacher"], "critic": ["critic"]}


@configclass
class ScanStudentDistillCfg(RslRlDistillationRunnerCfg):
    """Distil the scan teacher into a blind, recurrent student.

    Recurrent on purpose. The scan sees ground the feet have not reached yet, so
    a memoryless student cannot reproduce the teacher even in principle -- the
    information it is missing is in the past, in what its own feet have already
    felt. A feedforward student would measure the wrong thing and look like a
    negative result.

    The student's observation set is `policy`: the same 45 dims every other
    policy in this repo deploys with, so the distilled result drops into the
    same sim2sim harness and the same sim2real path unchanged.
    """

    num_steps_per_env = 24
    max_iterations = 3000
    save_interval = 100
    experiment_name = "biped"
    empirical_normalization = False
    obs_groups = {"policy": ["policy"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        # The teacher's trunk must match the shape PPO trained, or the loaded
        # weights will not fit.
        student_hidden_dims=[256, 128, 128],
        teacher_hidden_dims=[256, 128, 128],
        activation="elu",
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        teacher_recurrent=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        learning_rate=1.0e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        loss_type="mse",
    )
