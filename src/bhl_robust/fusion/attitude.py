"""Attitude estimation from a gyroscope and an accelerometer.

Two complementary filters, Mahony (explicit gyro-bias state) and Madgwick
(gradient step), in the quaternion convention the rest of this repo uses:
``(w, x, y, z)``, **body-to-world**, as MuJoCo's ``framequat`` sensor reports it
and as the upstream ``RlController`` consumes it. The body-frame gravity the
policy observes is ``rotate_inverse(q, (0, 0, -1))``.

These are the IMU-only forms (no magnetometer): heading is unobservable and
drifts with the gyro; pitch and roll are corrected by the accelerometer's
specific-force direction. Linear acceleration of the body is treated as a
disturbance, which is the standard assumption at walking accelerations.

Nothing here is calibrated to a particular IMU. ``ImuNoise`` is a stress-test
model whose parameters are experiment settings, not measurements.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

GRAVITY_M_S2 = 9.81
_UP = np.array([0.0, 0.0, 1.0])


def quat_normalize(q):
    q = np.asarray(q, dtype=float)
    n = np.linalg.norm(q)
    if not np.isfinite(n) or n == 0.0:
        raise ValueError("quaternion must be finite and non-zero")
    return q / n


def quat_mul(a, b):
    """Hamilton product of two (w, x, y, z) quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw*bw - ax*bx - ay*by - az*bz,
                     aw*bx + ax*bw + ay*bz - az*by,
                     aw*by - ax*bz + ay*bw + az*bx,
                     aw*bz + ax*by - ay*bx + az*bw])


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate(q, v):
    """Rotate a body-frame vector into the world frame."""
    w, x, y, z = q
    u = np.array([x, y, z])
    v = np.asarray(v, dtype=float)
    return v + 2.0*np.cross(u, np.cross(u, v) + w*v)


def rotate_inverse(q, v):
    """Rotate a world-frame vector into the body frame (RlController convention)."""
    return rotate(quat_conj(q), v)


def body_gravity(q):
    """Unit gravity direction in the body frame, as the policy observes it."""
    return rotate_inverse(q, -_UP)


def integrate(q, omega_body, dt):
    """First-order quaternion integration with a body-frame angular rate."""
    dq = 0.5*quat_mul(q, np.array([0.0, *omega_body]))
    return quat_normalize(q + dq*dt)


def angle_between(q_a, q_b):
    """Rotation angle (rad) between two attitudes."""
    d = quat_mul(quat_conj(q_a), q_b)
    return 2.0*np.arctan2(np.linalg.norm(d[1:]), abs(d[0]))


def attitude_from_accel(accel, yaw: float = 0.0):
    """Tilt-only initial alignment from a stationary accelerometer sample.

    Returns the body-to-world quaternion whose body-frame "up" equals the
    measured specific-force direction, with the given world yaw (heading is
    unobservable from the accelerometer). This is how a deployment starts a
    filter: stand still, align, then integrate.
    """
    up = _measured_up(accel)
    if up is None:
        raise ValueError("accelerometer sample must be finite and non-zero")
    # Rotation taking the body up-vector onto world +z.
    c = float(np.clip(up @ _UP, -1.0, 1.0))
    axis = np.cross(up, _UP)
    n = np.linalg.norm(axis)
    if n < 1e-9:
        tilt = np.array([1.0, 0.0, 0.0, 0.0]) if c > 0 else np.array([0.0, 1.0, 0.0, 0.0])
    else:
        angle = np.arctan2(n, c)
        tilt = np.array([np.cos(angle/2), *(np.sin(angle/2)*axis/n)])
    yaw_q = np.array([np.cos(yaw/2), 0.0, 0.0, np.sin(yaw/2)])
    return quat_normalize(quat_mul(yaw_q, tilt))


def _measured_up(accel):
    a = np.asarray(accel, dtype=float)
    n = np.linalg.norm(a)
    if not np.isfinite(n) or n < 1e-6:
        return None
    return a / n


class MahonyFilter:
    """Mahony et al. 2008 nonlinear complementary filter with gyro-bias state.

    ``kp`` weights the accelerometer correction, ``ki`` the bias integrator.
    The correction is ``e = up_measured x up_estimated`` in the body frame.
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.1, q0=None):
        if kp < 0 or ki < 0:
            raise ValueError("gains must be non-negative")
        self.kp, self.ki = float(kp), float(ki)
        self.q = quat_normalize(q0 if q0 is not None else (1.0, 0.0, 0.0, 0.0))
        self.bias = np.zeros(3)

    def update(self, gyro, accel, dt: float):
        gyro = np.asarray(gyro, dtype=float)
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite")
        up_meas = _measured_up(accel)
        error = np.zeros(3)
        if up_meas is not None:
            error = np.cross(up_meas, rotate_inverse(self.q, _UP))
            self.bias = self.bias - self.ki*error*dt
        omega = gyro - self.bias + self.kp*error
        self.q = integrate(self.q, omega, dt)
        return self.q

    @property
    def angular_velocity(self):
        """Bias-corrected rate is what a deployment would publish; the filter
        stores only the bias, so callers subtract it from their own gyro."""
        return -self.bias


class MadgwickFilter:
    """Madgwick 2010 IMU-only gradient-descent filter; ``beta`` is the step size."""

    def __init__(self, beta: float = 0.1, q0=None):
        if beta < 0:
            raise ValueError("beta must be non-negative")
        self.beta = float(beta)
        self.q = quat_normalize(q0 if q0 is not None else (1.0, 0.0, 0.0, 0.0))
        self.bias = np.zeros(3)   # Madgwick's IMU form has no bias state

    def update(self, gyro, accel, dt: float):
        gyro = np.asarray(gyro, dtype=float)
        if dt <= 0 or not np.isfinite(dt):
            raise ValueError("dt must be positive and finite")
        q = self.q
        qdot = 0.5*quat_mul(q, np.array([0.0, *gyro]))
        up_meas = _measured_up(accel)
        if up_meas is not None and self.beta > 0:
            w, x, y, z = q
            # f(q) = R(q)^T e_z - up_measured, with R body-to-world.
            f = np.array([2*(x*z - w*y) - up_meas[0],
                          2*(y*z + w*x) - up_meas[1],
                          1 - 2*(x*x + y*y) - up_meas[2]])
            jac = np.array([[-2*y, 2*z, -2*w, 2*x],
                            [2*x, 2*w, 2*z, 2*y],
                            [0.0, -4*x, -4*y, 0.0]])
            grad = jac.T @ f
            n = np.linalg.norm(grad)
            if n > 0:
                qdot = qdot - self.beta*grad/n
        self.q = quat_normalize(q + qdot*dt)
        return self.q


FILTERS = {"mahony": MahonyFilter, "madgwick": MadgwickFilter}


@dataclass
class ImuNoise:
    """Stress-test IMU corruption: white noise, a constant gyro bias drawn per
    instance, optional bias random walk, and an integer-step delay.

    All parameters are experiment settings. They are **not** a calibration of
    the IM10A or the BNO085; see ``docs/IMU_INPUT.md``.
    """

    gyro_std: float = 0.0           # rad/s, white
    accel_std: float = 0.0          # m/s^2, white
    gyro_bias: float = 0.0          # rad/s, magnitude of a fixed bias (random direction)
    gyro_bias_walk: float = 0.0     # rad/s/sqrt(s)
    delay_steps: int = 0
    seed: int = 0
    _rng: np.random.Generator = field(init=False, repr=False)
    _bias: np.ndarray = field(init=False, repr=False)
    _queue: list = field(init=False, repr=False, default_factory=list)

    def __post_init__(self):
        for name in ("gyro_std", "accel_std", "gyro_bias", "gyro_bias_walk"):
            v = getattr(self, name)
            if not np.isfinite(v) or v < 0:
                raise ValueError(f"{name} must be finite and non-negative")
        if self.delay_steps < 0:
            raise ValueError("delay_steps must be non-negative")
        self._rng = np.random.default_rng(self.seed)
        direction = self._rng.normal(size=3)
        direction /= np.linalg.norm(direction)
        self._bias = self.gyro_bias*direction

    @property
    def bias(self):
        return self._bias.copy()

    def __call__(self, gyro, accel, dt: float):
        """Return the corrupted (gyro, accel) sample delivered *now*."""
        if self.gyro_bias_walk > 0:
            self._bias = self._bias + self._rng.normal(size=3)*self.gyro_bias_walk*np.sqrt(dt)
        g = np.asarray(gyro, dtype=float) + self._bias + self._rng.normal(size=3)*self.gyro_std
        a = np.asarray(accel, dtype=float) + self._rng.normal(size=3)*self.accel_std
        self._queue.append((g, a))
        if len(self._queue) > self.delay_steps:
            return self._queue.pop(0)          # the sample from delay_steps ago
        return self._queue[0]                  # pipeline still filling: stale repeat
