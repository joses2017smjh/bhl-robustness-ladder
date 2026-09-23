"""Per-episode sensor-realism state for the SF-04 `BothRobust` maze arm.

Pure torch, no Isaac import, so the logic is unit-testable. One instance lives
on the environment; the reset event resamples the rows of the envs that reset.
All ranges are experiment settings (docs/SENSOR_FUSION.md SF-04), not a
calibration of any IMU or LiDAR.
"""

from __future__ import annotations

import torch


class RobustSensingState:
    def __init__(self, num_envs: int, device, *, p_lidar_off: float = 0.2, p_stereo_off: float = 0.2,
                 gyro_bias_std: float = 0.02, gravity_bias_std: float = 0.02, max_delay_steps: int = 1,
                 generator: torch.Generator | None = None):
        if not 0.0 <= p_lidar_off <= 1.0 or not 0.0 <= p_stereo_off <= 1.0:
            raise ValueError("dropout probabilities must be in [0, 1]")
        if max_delay_steps < 0 or gyro_bias_std < 0 or gravity_bias_std < 0:
            raise ValueError("delay and bias stds must be non-negative")
        self.n, self.device = int(num_envs), device
        self.p_lidar_off, self.p_stereo_off = float(p_lidar_off), float(p_stereo_off)
        self.gyro_bias_std, self.gravity_bias_std = float(gyro_bias_std), float(gravity_bias_std)
        self.max_delay_steps = int(max_delay_steps)
        self.gen = generator
        self.lidar_on = torch.ones(self.n, dtype=torch.bool, device=device)
        self.stereo_on = torch.ones(self.n, dtype=torch.bool, device=device)
        self.gyro_bias = torch.zeros(self.n, 3, device=device)
        self.gravity_bias = torch.zeros(self.n, 3, device=device)
        self.delay = torch.zeros(self.n, dtype=torch.long, device=device)
        self._ang_prev = torch.zeros(self.n, 3, device=device)
        self._grav_prev = torch.zeros(self.n, 3, device=device)
        self._grav_prev[:, 2] = -1.0
        # Evaluation overrides (None = sampled as configured).
        self.force_lidar_on: bool | None = None
        self.force_stereo_on: bool | None = None
        self.force_delay: int | None = None

    def _rand(self, *shape):
        return torch.rand(*shape, generator=self.gen, device=self.device)

    def _randn(self, *shape):
        return torch.randn(*shape, generator=self.gen, device=self.device)

    def resample(self, env_ids) -> None:
        ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long).reshape(-1)
        k = ids.numel()
        if k == 0:
            return
        self.lidar_on[ids] = self._rand(k) >= self.p_lidar_off
        self.stereo_on[ids] = self._rand(k) >= self.p_stereo_off
        self.gyro_bias[ids] = self._randn(k, 3) * self.gyro_bias_std
        self.gravity_bias[ids] = self._randn(k, 3) * self.gravity_bias_std
        self.delay[ids] = torch.randint(0, self.max_delay_steps + 1, (k,), generator=self.gen, device=self.device)
        if self.force_lidar_on is not None:
            self.lidar_on[ids] = bool(self.force_lidar_on)
        if self.force_stereo_on is not None:
            self.stereo_on[ids] = bool(self.force_stereo_on)
        if self.force_delay is not None:
            self.delay[ids] = int(self.force_delay)
        self._ang_prev[ids] = 0.0
        self._grav_prev[ids] = 0.0
        self._grav_prev[ids, 2] = -1.0

    def angular_velocity(self, true_ang_vel: torch.Tensor) -> torch.Tensor:
        cur = true_ang_vel + self.gyro_bias
        out = torch.where(self.delay[:, None] > 0, self._ang_prev, cur)
        self._ang_prev = cur.clone()
        return out

    def gravity(self, true_gravity: torch.Tensor) -> torch.Tensor:
        cur = true_gravity + self.gravity_bias
        out = torch.where(self.delay[:, None] > 0, self._grav_prev, cur)
        self._grav_prev = cur.clone()
        return out

    def lidar(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.lidar_on[:, None].to(values.dtype)

    def stereo(self, values: torch.Tensor) -> torch.Tensor:
        return values * self.stereo_on[:, None].to(values.dtype)
