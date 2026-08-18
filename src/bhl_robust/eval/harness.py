"""Headless MuJoCo sim2sim evaluation harness.

Upstream's `scripts/sim2sim/play_mujoco.py` cannot be used for measurement: it
spawns a `Se2Gamepad()` input thread, calls `mj_viewer.sync()` (requiring a GUI),
sleeps to hold wall-clock realtime, and loops forever without emitting a metric.

This module keeps the parts that matter for transfer fidelity -- the MJCF, the
PD controller, the sensor-derived observation layout -- and drops the interactive
scaffolding. Crucially it drives the policy through upstream's own `RlController`,
so the observation construction here is bit-identical to the sim2real path. An
evaluator that rebuilt the observation vector independently would be measuring
its own reimplementation, not the transfer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field, asdict
from pathlib import Path

import mujoco
import numpy as np
import torch

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController

# Matches upstream's `bad_orientation` termination (limit_angle=0.78 rad), so a
# fall in MuJoCo means the same thing it meant during Isaac Lab training.
TILT_LIMIT_RAD = 0.78

# The MJCF root frame sits at z~0 with the body geometry offset upward, so
# qpos[2] is NOT a standing height -- it reads ~0.0 while fully upright. An
# absolute height threshold therefore fires on the first step of every episode.
# Falls are detected by tilt (as in training) plus this relative sink, measured
# against the spawn height.
MAX_SINK_M = 0.25


@dataclass
class EpisodeResult:
    """Per-episode metrics. One row of the results CSV."""

    command_vx: float
    command_vy: float
    command_wz: float
    seed: int
    fell: bool
    survival_s: float
    # Tracking errors are averaged over surviving steps only; a policy that
    # falls instantly would otherwise post a flatteringly small error.
    lin_vel_err: float
    yaw_rate_err: float
    distance_m: float
    mean_height_m: float
    mean_tilt_rad: float
    pushes_applied: int = 0
    pushes_survived: int = 0


@dataclass
class EvalConfig:
    """Evaluation protocol. Identical across every policy compared."""

    episode_s: float = 10.0
    settle_s: float = 1.0            # ignore metrics while the robot settles
    seeds: tuple[int, ...] = (0, 1, 2, 3, 4)
    commands: tuple[tuple[float, float, float], ...] = (
        (0.3, 0.0, 0.0),    # walk forward
        (0.5, 0.0, 0.0),    # faster forward
        (-0.2, 0.0, 0.0),   # backward
        (0.0, 0.2, 0.0),    # strafe
        (0.0, 0.0, 0.5),    # turn in place
        (0.3, 0.0, 0.5),    # arc
    )
    # Initial-state perturbation, so seeds actually differ.
    init_joint_noise: float = 0.02
    init_vel_noise: float = 0.05
    # Push protocol (disabled unless push_speed > 0).
    push_speed: float = 0.0
    push_interval_s: float = 3.0
    push_recovery_s: float = 1.5     # must stay upright this long to count


class HeadlessMujocoEnv:
    """MuJoCo stepping with no viewer, no gamepad, and no realtime throttle."""

    def __init__(self, cfg, scene_path: Path):
        self.cfg = cfg
        self.mj_model = mujoco.MjModel.from_xml_path(str(scene_path))
        self.mj_model.opt.timestep = cfg.physics_dt
        self.mj_data = mujoco.MjData(self.mj_model)

        self.num_joints = int(cfg.num_joints)
        self.substeps = int(round(cfg.policy_dt / cfg.physics_dt))
        self.sensordata_dof_size = 3 * self.mj_model.nu

        self.joint_kp = np.asarray(cfg.joint_kp, dtype=np.float32)
        self.joint_kd = np.asarray(cfg.joint_kd, dtype=np.float32)
        self.effort_limits = np.asarray(cfg.effort_limits, dtype=np.float32)
        self.action_indices = np.asarray(cfg.action_indices, dtype=int)
        self.default_joint_positions = np.asarray(cfg.default_joint_positions, dtype=np.float32)

    # -- state accessors, mirroring upstream's sensor layout ----------------

    def _joint_pos(self) -> np.ndarray:
        return np.asarray(self.mj_data.sensordata[0:self.num_joints], dtype=np.float32)

    def _joint_vel(self) -> np.ndarray:
        return np.asarray(self.mj_data.sensordata[self.num_joints:2 * self.num_joints], dtype=np.float32)

    def _base_quat(self) -> np.ndarray:
        i = self.sensordata_dof_size
        return np.asarray(self.mj_data.sensordata[i:i + 4], dtype=np.float32)

    def _base_ang_vel(self) -> np.ndarray:
        i = self.sensordata_dof_size + 4
        return np.asarray(self.mj_data.sensordata[i:i + 3], dtype=np.float32)

    @property
    def base_height(self) -> float:
        return float(self.mj_data.qpos[2])

    @property
    def base_xy(self) -> np.ndarray:
        return np.asarray(self.mj_data.qpos[0:2], dtype=np.float32)

    def base_lin_vel_yaw_frame(self) -> np.ndarray:
        """World linear velocity rotated into the yaw-only body frame.

        Upstream rewards velocity tracking in the yaw frame
        (`track_lin_vel_xy_yaw_frame_exp`), so the evaluation metric has to use
        the same frame or the numbers are not comparable to training.
        """
        v = np.asarray(self.mj_data.qvel[0:3], dtype=np.float32)
        w, x, y, z = (float(c) for c in self.mj_data.qpos[3:7])
        yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
        c, s = math.cos(-yaw), math.sin(-yaw)
        return np.array([c * v[0] - s * v[1], s * v[0] + c * v[1]], dtype=np.float32)

    @property
    def tilt_rad(self) -> float:
        """Angle between the base z-axis and world up."""
        w, x, y, z = (float(c) for c in self.mj_data.qpos[3:7])
        # Third column of the rotation matrix, dotted with world up.
        up_z = 1.0 - 2.0 * (x * x + y * y)
        return float(math.acos(max(-1.0, min(1.0, up_z))))

    # -- dynamics ----------------------------------------------------------

    def reset(self, rng: np.random.Generator, cfg: EvalConfig) -> np.ndarray:
        mujoco.mj_resetData(self.mj_model, self.mj_data)
        self.mj_data.qpos[0:3] = self.cfg.default_base_position
        self.mj_data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        self.mj_data.qpos[7:7 + self.num_joints] = (
            self.default_joint_positions
            + rng.normal(0.0, cfg.init_joint_noise, self.num_joints).astype(np.float32)
        )
        self.mj_data.qvel[:] = 0.0
        self.mj_data.qvel[0:2] = rng.normal(0.0, cfg.init_vel_noise, 2)
        mujoco.mj_forward(self.mj_model, self.mj_data)
        self._spawn_z = float(self.mj_data.qpos[2])
        return self.robot_observations(command=(0.0, 0.0, 0.0))

    @property
    def sink_m(self) -> float:
        """How far the root has dropped below its spawn height."""
        return self._spawn_z - float(self.mj_data.qpos[2])

    def push(self, speed: float, rng: np.random.Generator) -> None:
        """Instantaneous velocity kick, matching training's push_by_setting_velocity."""
        angle = rng.uniform(0.0, 2.0 * math.pi)
        self.mj_data.qvel[0] += speed * math.cos(angle)
        self.mj_data.qvel[1] += speed * math.sin(angle)

    def step(self, target_joint_pos: np.ndarray) -> None:
        targets = np.zeros(self.num_joints, dtype=np.float32)
        targets[self.action_indices] = target_joint_pos
        for _ in range(self.substeps):
            torque = self.joint_kp * (targets - self._joint_pos()) + self.joint_kd * (-self._joint_vel())
            self.mj_data.ctrl[:] = np.clip(torque, -self.effort_limits, self.effort_limits)
            mujoco.mj_step(self.mj_model, self.mj_data)

    def robot_observations(self, command: tuple[float, float, float]) -> np.ndarray:
        """The vector `RlController.update` expects.

        Layout: [quat(4), ang_vel(3), joint_pos(na), joint_vel(na), mode(1), cmd(3)]
        `mode` is unused by RlController but must occupy its slot.
        """
        return np.concatenate([
            self._base_quat(),
            self._base_ang_vel(),
            self._joint_pos()[self.action_indices],
            self._joint_vel()[self.action_indices],
            np.array([3.0], dtype=np.float32),   # RL control mode
            np.asarray(command, dtype=np.float32),
        ]).astype(np.float32)


def run_episode(
    env: HeadlessMujocoEnv,
    controller: RlController,
    command: tuple[float, float, float],
    seed: int,
    cfg: EvalConfig,
) -> EpisodeResult:
    """Roll out one episode and return its metrics."""
    rng = np.random.default_rng(seed)
    obs = env.reset(rng, cfg)

    # A fresh action history per episode; otherwise the previous episode's last
    # action leaks in as the first observation.
    controller.prev_actions[:] = 0.0
    controller.policy_observations[:] = 0.0

    n_steps = int(cfg.episode_s / env.cfg.policy_dt)
    settle_steps = int(cfg.settle_s / env.cfg.policy_dt)
    push_every = int(cfg.push_interval_s / env.cfg.policy_dt) if cfg.push_speed > 0 else 0
    recovery_steps = int(cfg.push_recovery_s / env.cfg.policy_dt)

    lin_errs: list[float] = []
    yaw_errs: list[float] = []
    heights: list[float] = []
    tilts: list[float] = []
    start_xy = env.base_xy.copy()

    fell = False
    survival_steps = n_steps
    pushes_applied = 0
    pushes_survived = 0
    pending_push_step: int | None = None

    for t in range(n_steps):
        if push_every and t > settle_steps and t % push_every == 0:
            env.push(cfg.push_speed, rng)
            pushes_applied += 1
            pending_push_step = t

        actions = controller.update(obs)
        env.step(actions)
        obs = env.robot_observations(command)

        tilt = env.tilt_rad
        if tilt > TILT_LIMIT_RAD or env.sink_m > MAX_SINK_M:
            fell = True
            survival_steps = t
            break

        if pending_push_step is not None and t - pending_push_step >= recovery_steps:
            pushes_survived += 1
            pending_push_step = None

        if t >= settle_steps:
            v = env.base_lin_vel_yaw_frame()
            lin_errs.append(float(np.linalg.norm(v - np.array(command[:2], dtype=np.float32))))
            yaw_errs.append(abs(float(env._base_ang_vel()[2]) - command[2]))
            heights.append(env.base_height)
            tilts.append(tilt)

    # A push still pending when the episode ends counts as survived.
    if pending_push_step is not None and not fell:
        pushes_survived += 1

    mean = lambda xs: float(np.mean(xs)) if xs else float("nan")  # noqa: E731

    return EpisodeResult(
        command_vx=command[0],
        command_vy=command[1],
        command_wz=command[2],
        seed=seed,
        fell=fell,
        survival_s=survival_steps * env.cfg.policy_dt,
        lin_vel_err=mean(lin_errs),
        yaw_rate_err=mean(yaw_errs),
        distance_m=float(np.linalg.norm(env.base_xy - start_xy)),
        mean_height_m=mean(heights),
        mean_tilt_rad=mean(tilts),
        pushes_applied=pushes_applied,
        pushes_survived=pushes_survived,
    )
