"""A simulated ten-axis IMU laid out like the user's Hiwonder IM10A, for display.

It samples the MuJoCo robot's own IMU site (gyro, accelerometer, orientation)
from the physics substep hook at a fixed output rate, corrupts the samples with
the manufacturer's published noise figures, runs the repo's 6-axis Mahony
filter on them, and derives magnetometer and barometer channels from the true
pose. It is READ-ONLY: nothing it computes reaches the controller, and it draws
from its own random stream, so an episode with the panel is bit-identical to
the same episode without it.

The noise figures are DATASHEET UPPER BOUNDS, not a calibration of the physical
board (see docs/IMU_INPUT.md and SF-05): replace ``IM10A_DATASHEET`` with the
Allan-variance numbers from a recording of the real unit when one exists.
"""
from __future__ import annotations

import math
from collections import deque

import mujoco
import numpy as np

from bhl_robust.fusion.attitude import MahonyFilter, attitude_from_accel

G = 9.80665

#: Hiwonder IMU module user manual, section 1 (specifications), retrieved 2026-09-29:
#: https://docs.hiwonder.com/projects/IMU-Module/en/latest/docs/1.User_Manual.html
#: Noise at 100 Hz bandwidth; the upper end of each published range is used.
IM10A_DATASHEET = {
    "gyro_noise_dps": 0.07,        # "0.028~0.07 (deg/s)-rms"
    "gyro_bias_dps": 1.0,          # zero drift "+-0.5~1 deg/s" (auto-calibration on the board lowers this)
    "accel_noise_mg": 1.0,         # "0.75~1 mg-rms"
    "accel_bias_mg": 40.0,         # zero drift "+-20~40 mg"
    "mag_resolution_mgauss": 0.0667,  # no noise figure published
    "baro_noise_pa": 0.5,          # "0.5 Pa-RMS" (standard mode)
    "max_rate_hz": 200.0,          # "0.2 Hz-200 Hz", default 10 Hz
    "source": "Hiwonder IMU module user manual s1 (datasheet upper bounds), retrieved 2026-09-29",
}

#: Approximate local geomagnetic field, ENU, microtesla (Corvallis OR: ~51.5 uT total,
#: inclination ~67.5 deg, declination ~15 deg E). Display only; the magnetometer is not used.
FIELD_ENU_UT = np.array([5.1, 19.0, -47.6])
P0_PA = 101325.0


def _pressure_pa(h_m: float) -> float:
    return P0_PA * (1.0 - 2.25577e-5 * h_m) ** 5.25588


def _altitude_m(p_pa: float) -> float:
    return (1.0 - (p_pa / P0_PA) ** (1.0 / 5.25588)) / 2.25577e-5


def rpy_deg(q_wxyz) -> np.ndarray:
    """ZYX roll, pitch, yaw in degrees from a (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q_wxyz)
    roll = math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y))
    pitch = math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x))))
    yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    return np.degrees([roll, pitch, yaw])


def wrap_deg(a: float) -> float:
    return (a + 180.0) % 360.0 - 180.0


class SimIM10A:
    """Display-only IM10A-like IMU on one robot slot of a MultiRunner/ContactRunner."""

    def __init__(self, model, slot, seed: int, physics_dt: float, rate_hz: float = 100.0,
                 params: dict | None = None, window_s: float = 4.0, kp: float = 1.0, ki: float = 0.1):
        self.p = dict(IM10A_DATASHEET if params is None else params)
        if not 0 < rate_hz <= self.p["max_rate_hz"]:
            raise ValueError(f"rate_hz must be in (0, {self.p['max_rate_hz']}]")
        acc_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, slot.prefix + "imu_acc")
        if acc_id < 0:
            raise ValueError(f"missing accelerometer sensor for {slot.prefix}")
        self.acc_adr = int(model.sensor_adr[acc_id])
        self.site_id = int(model.sensor_objid[acc_id])
        self.quat_adr, self.gyro_adr = slot.quat_adr, slot.gyro_adr
        self.every = max(1, int(round(1.0 / (rate_hz * physics_dt))))
        self.dt = self.every * physics_dt
        self.rate_hz = 1.0 / self.dt
        # own stream: never touches the episode's generators
        self.rng = np.random.default_rng([int(seed), 0x1A10A])
        d = self.rng.normal(size=(2, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        self.gyro_bias = math.radians(self.p["gyro_bias_dps"]) * d[0]
        self.accel_bias = self.p["accel_bias_mg"] * 1e-3 * G * d[1]
        self.gyro_sigma = math.radians(self.p["gyro_noise_dps"])
        self.accel_sigma = self.p["accel_noise_mg"] * 1e-3 * G
        self.kp, self.ki = kp, ki
        self.filter = None
        self.h0 = None
        self._substep = 0
        n = max(2, int(round(window_s * self.rate_hz)))
        self.hist = deque(maxlen=n)      # (t, gyro_dps(3), accel(3))
        self.latest: dict | None = None

    # -- wiring ---------------------------------------------------------------
    def attach(self, runner) -> None:
        """Chain onto the runner's substep hook (after any hook already bound)."""
        prev = getattr(runner, "substep_hook", None)

        def hook(data):
            if prev is not None:
                prev(data)
            self.on_substep(data)

        runner.substep_hook = hook

    def on_substep(self, data) -> None:
        self._substep += 1
        if self._substep % self.every:
            return
        self.sample(data, self._substep * self.dt / self.every)

    # -- one output sample ----------------------------------------------------
    def sample(self, data, t: float) -> dict:
        sd = data.sensordata
        gyro_t = np.array(sd[self.gyro_adr:self.gyro_adr + 3], dtype=float)
        acc_t = np.array(sd[self.acc_adr:self.acc_adr + 3], dtype=float)
        q_t = np.array(sd[self.quat_adr:self.quat_adr + 4], dtype=float)
        gyro = gyro_t + self.gyro_bias + self.rng.normal(0.0, self.gyro_sigma, 3)
        acc = acc_t + self.accel_bias + self.rng.normal(0.0, self.accel_sigma, 3)
        rot = np.empty(9)
        mujoco.mju_quat2Mat(rot, q_t)
        rot = rot.reshape(3, 3)
        res_ut = self.p["mag_resolution_mgauss"] * 0.1        # 1 mGauss = 0.1 uT
        mag = np.round(rot.T @ FIELD_ENU_UT / res_ut) * res_ut
        h = float(data.site_xpos[self.site_id][2])
        p = _pressure_pa(h) + self.rng.normal(0.0, self.p["baro_noise_pa"])
        alt = _altitude_m(p)
        if self.filter is None:
            yaw0 = math.radians(rpy_deg(q_t)[2])
            self.filter = MahonyFilter(kp=self.kp, ki=self.ki, q0=attitude_from_accel(acc, yaw=yaw0))
            self.h0 = alt
        q_e = self.filter.update(gyro, acc, self.dt)
        true_rpy, est_rpy = rpy_deg(q_t), rpy_deg(q_e)
        err = np.array([wrap_deg(a - b) for a, b in zip(est_rpy, true_rpy)])
        mag_heading = math.degrees(math.atan2(-(rot.T @ FIELD_ENU_UT)[1], (rot.T @ FIELD_ENU_UT)[0]))
        self.latest = {"t": t, "gyro_dps": np.degrees(gyro), "accel": acc, "mag_ut": mag,
                       "mag_heading_deg": mag_heading, "baro_pa": p, "baro_dalt_m": alt - self.h0,
                       "rpy_est": est_rpy, "rpy_true": true_rpy, "rpy_err": err}
        self.hist.append((t, np.degrees(gyro), acc))
        return self.latest

    def meta(self) -> dict:
        return {"model": "simulated IM10A-like 10-axis IMU (display only; not in the control loop)",
                "rate_hz": round(self.rate_hz, 3), "noise": {k: v for k, v in self.p.items()},
                "gyro_bias_dps": np.degrees(self.gyro_bias).round(4).tolist(),
                "accel_bias_m_s2": self.accel_bias.round(5).tolist(),
                "filter": f"Mahony kp={self.kp} ki={self.ki}, 6-axis (gyro + accel), aligned from the first sample",
                "magnetometer": "true pose x approximate local field (Corvallis, ~51.5 uT), quantised; not used",
                "barometer": "standard atmosphere at the IMU site height + 0.5 Pa rms; not used"}
