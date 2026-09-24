"""SF-04 follow-up: distil the working `Both` Full s0 MLP teacher into a
recurrent student that sees the degraded sensors (docs/SENSOR_FUSION.md, section 6).

Why this exists
---------------
The BothRobust arm trained from scratch (recurrent actor + LiDAR/stereo dropout
+ IMU bias + 0-1 step IMU delay) fell in 100 % of Corridor episodes
(`21403141`). The published `Both` Full s0 policy walks the whole maze
(`external/.../2026-09-19_16-25-50_wknd-full-both-s0/model_5997.pt`, MLP
[256, 128, 128], obs = 45 proprio + 32 stereo + 36 lidar). Instead of asking
PPO to rediscover the gait under degraded sensing, this path keeps the working
policy as a fixed teacher and trains a student, by behaviour cloning on the
student's own trajectories (DAgger-style: rsl-rl's `Distillation` labels the
states the student visits with the teacher's deterministic action), to
reproduce it from the degraded observations.

What the teacher sees vs what the student sees
----------------------------------------------
Both groups have the **identical 113-dim layout and term order** (the `Both`
layout: velocity_commands 3, base_ang_vel 3, projected_gravity 3, joint_pos 12,
joint_vel 12, actions 12, stereo_l 16, stereo_r 16, lidar 36 -- stereo pooled
to 4x4 per eye by the Full-stage `_configure`), so the teacher's PPO actor
weights load into the teacher unchanged and the student is shape-compatible
with every `Both` probe.

* `teacher` group = `maze_env_cfg.BothObsCfg.PolicyCfg`: the CLEAN `Both`
  policy terms exactly as the teacher was trained on them -- true IMU with the
  upstream white noise (gyro std 0.05, gravity std 0.02, corruption on), both
  eyes and the LiDAR always present, no per-episode bias, no delay. It is
  used only to produce the behaviour-cloning target and is never fed to the
  student.
* `policy` group = `maze_robust.BothRobustObsCfg.PolicyCfg`: the DEGRADED
  terms -- the same channels routed through `RobustSensingState`, which per
  episode (reset event `sf04_resample`) zeroes the LiDAR or the stereo pair
  with p = 0.2 each, adds a constant gyro/gravity bias (std 0.02) under the
  same white noise, and delays the IMU columns by 0 or 1 policy step. This is
  the student's ONLY input, through the `student` observation set.
* `critic` group (`BothObsCfg.CriticCfg`, clean + base_lin_vel) is kept so the
  env also serves the ordinary PPO cfg (`rsl_rl_cfg_entry_point`); the
  distillation runner does not read it.
* The `actions` term of BOTH groups is the environment's last action, i.e. the
  student's action during distillation; the teacher labels student-visited
  states, which is the point of on-policy distillation.

How the student is evaluated
----------------------------
`scripts/bench/maze_recovery_probe.py --runner distillation` builds this
runner cfg, loads the distillation checkpoint and rolls out the STUDENT on its
OWN degraded `policy` group. Under `--settings` the probe forces the per-episode
randomization off (`force_lidar_on/force_stereo_on/force_delay`, bias stds 0)
so every condition -- baseline, IMU delay 1/2 steps, lidar/stereo/both zeroed
-- is explicit and applied to the student's input columns. The teacher group is
computed but unused at evaluation. The student is never scored on clean inputs
it would not have on the robot.

rsl-rl 5.0.1 specifics (verified against the installed source)
--------------------------------------------------------------
* Teacher loading: `Distillation.load()` with `load_cfg=None` detects a PPO
  checkpoint by its `actor_state_dict` key and strict-loads it into the
  teacher (`teacher_loaded = True`, iteration not restored), so
  `scripts/train_distill.py --resume True --load_run <teacher run>
  --checkpoint model_5997.pt` is the whole teacher path. The teacher must be
  an `MLPModel` with `obs_normalization=False` and a scalar Gaussian
  distribution, or the state-dict keys (`mlp.*`, `distribution.std_param`)
  do not match. Checked on CPU with the real checkpoint: strict load OK and
  the teacher's output equals the PPO actor's forward pass.
* The deprecated `RslRlDistillationStudentTeacherRecurrentCfg` (the scan
  precedent) is NOT used: on rsl-rl >= 4 `handle_deprecated_rsl_rl_cfg`
  converts it into an RNN teacher regardless of `teacher_recurrent=False`,
  and the MLP actor weights would not fit. The new-style `student` /
  `teacher` model cfgs express "LSTM student, MLP teacher" directly.
* `scripts/train_distill.py` predates the shim and never calls
  `handle_deprecated_rsl_rl_cfg`, while the 5.x models accept no `**kwargs`;
  the deprecated `stochastic`/`init_noise_std`/... fields would reach
  `RNNModel.__init__` and fail. The runner cfg therefore migrates itself in
  `__post_init__`. The shim is idempotent on an already-migrated instance
  (configclass drops MISSING defaults from the class, so `hasattr` is False
  after the first `del`), so the probe's mandated second call is a no-op.
* Observation sets are named `student` and `teacher` in 5.x
  (`Distillation.construct_algorithm` default sets); a `"policy"` key as in
  the 3.x scan cfg would only work through `resolve_obs_groups`' fallback
  warning, so `obs_groups` maps the sets by their real names.
"""

from __future__ import annotations

import copy
from importlib.metadata import version as _pkg_version

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlMLPModelCfg,
    RslRlRNNModelCfg,
    handle_deprecated_rsl_rl_cfg,
)

from bhl_robust.tasks.maze_env_cfg import BothObsCfg
from bhl_robust.tasks.maze_recovery_env_cfg import RECOVERY_CONFIGS
from bhl_robust.tasks.maze_robust import BothRobustObsCfg

TEACHER_RUN = "2026-09-19_16-25-50_wknd-full-both-s0"
TEACHER_CHECKPOINT = "model_5997.pt"
TEACHER_HIDDEN_DIMS = [256, 128, 128]   # BerkeleyHumanoidLiteBipedPPORunnerCfg actor_hidden_dims
TEACHER_ACTIVATION = "elu"
OBS_DIM = 113                           # 45 proprio + 2 x 16 stereo + 36 lidar

_FULL_BOTHROBUST = RECOVERY_CONFIGS["Velocity-BHL-MazeRecovery-Full-BothRobust-v0"]


# Groups reference already-defined classes only. Defining a new nested group
# class beside the two upstream declares sends the configclass copy machinery
# into unbounded recursion (scan_env_cfg); flat references construct cleanly.
@configclass
class BothRobustDistillObsCfg(BothRobustObsCfg):
    """`policy` = degraded BothRobust terms (student input); `teacher` = clean
    Both terms (behaviour-cloning target); `critic` kept for the PPO cfg."""

    policy: BothRobustObsCfg.PolicyCfg = BothRobustObsCfg.PolicyCfg()
    teacher: BothObsCfg.PolicyCfg = BothObsCfg.PolicyCfg()
    critic: BothObsCfg.CriticCfg = BothObsCfg.CriticCfg()


@configclass
class MazeBothRobustDistillEnvCfg(_FULL_BOTHROBUST):
    """Full-stage BothRobust recovery env (same `_configure(cfg, "Full")`:
    rewards, terminations, spawn, pool 16, IMU noise; reset event
    `sf04_resample` from `MazeBothRobustEnvCfg`) publishing the `teacher`
    group beside `policy` and `critic`."""

    observations: BothRobustDistillObsCfg = BothRobustDistillObsCfg()

    def __post_init__(self):
        super().__post_init__()   # MazeBothRobustEnvCfg (sf04_resample) then _configure(self, "Full")
        assert hasattr(self.events, "sf04_resample"), "reset event sf04_resample missing"
        # `_configure` shapes only the policy and critic groups. Give the
        # teacher group the SAME stereo pooling and IMU noise channels the
        # published Both Full policy trained with, so its 113 inputs are the
        # ones the loaded weights expect. Copies, not shared objects: the probe
        # mutates the policy term's noise std per setting.
        pol, tea = self.observations.policy, self.observations.teacher
        for eye in ("stereo_l", "stereo_r"):
            getattr(tea, eye).params["pool"] = getattr(pol, eye).params["pool"]
        tea.base_ang_vel.noise = copy.deepcopy(pol.base_ang_vel.noise)
        tea.projected_gravity.noise = copy.deepcopy(pol.projected_gravity.noise)


@configclass
class MazeStudentDistillCfg(RslRlDistillationRunnerCfg):
    """LSTM student on the degraded `policy` group, frozen MLP teacher on the
    clean `teacher` group; MSE behaviour cloning, 24-step truncated BPTT."""

    num_steps_per_env = 24
    max_iterations = 2000
    save_interval = 100
    experiment_name = "biped"
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}
    student = RslRlRNNModelCfg(
        hidden_dims=[256, 128, 128],
        activation="elu",
        obs_normalization=False,
        # Stochastic rollouts (rsl-rl samples the student's action in `act`);
        # the deterministic mean is what the probe evaluates.
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=0.1, std_type="scalar"),
        rnn_type="lstm",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )
    teacher = RslRlMLPModelCfg(
        # Must match the PPO actor the checkpoint was trained with
        # (actor_hidden_dims [256, 128, 128], elu, no normalization, scalar
        # std): the loaded weights replace these values, the shapes must fit.
        hidden_dims=TEACHER_HIDDEN_DIMS,
        activation=TEACHER_ACTIVATION,
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=1,
        learning_rate=1.0e-3,
        gradient_length=24,
        max_grad_norm=1.0,
        loss_type="mse",
    )

    def __post_init__(self):
        # Strip the deprecated model fields for the installed rsl-rl so the
        # cfg dict is directly consumable by `DistillationRunner` (see the
        # module docstring). Idempotent: later calls by the probe are no-ops.
        handle_deprecated_rsl_rl_cfg(self, _pkg_version("rsl-rl-lib"))
