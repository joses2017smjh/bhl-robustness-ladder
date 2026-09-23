import numpy as np
import pytest

from bhl_robust.fusion.attitude import (
    GRAVITY_M_S2, FILTERS, ImuNoise, MahonyFilter, MadgwickFilter, angle_between,
    attitude_from_accel, body_gravity, integrate, quat_conj, quat_mul, quat_normalize,
    rotate, rotate_inverse,
)


def _axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=float)
    axis = axis / np.linalg.norm(axis)
    return np.array([np.cos(angle/2), *(np.sin(angle/2)*axis)])


def _synthetic(seconds, hz, omega_fn, q0=(1.0, 0.0, 0.0, 0.0)):
    """Truth trajectory: body rotates at omega_fn(t); the accelerometer reads +g up
    in the body frame (no linear acceleration)."""
    dt = 1.0/hz
    q = quat_normalize(q0)
    out = []
    for k in range(int(seconds*hz)):
        t = k*dt
        w = np.asarray(omega_fn(t), dtype=float)
        q = integrate(q, w, dt)
        acc = rotate_inverse(q, np.array([0.0, 0.0, GRAVITY_M_S2]))
        out.append((t, q.copy(), w, acc))
    return dt, out


def test_rotation_convention_matches_controller():
    q = _axis_angle((0, 1, 0), np.pi/2)          # body pitched 90 deg nose-down about +y
    g_body = body_gravity(q)
    # Rotating the world -z into a frame pitched +90 deg about y puts gravity on -x... check via
    # explicit conjugation, which is the RlController formula.
    w, x, y, z = q
    expected = np.array([0.0, 0.0, -1.0])
    expected = expected + 2*np.cross(np.array([-x, -y, -z]), np.cross(np.array([-x, -y, -z]), expected) + w*expected)
    assert np.allclose(g_body, expected)
    assert np.allclose(rotate(q, rotate_inverse(q, [1., 2., 3.])), [1., 2., 3.])
    assert np.isclose(np.linalg.norm(g_body), 1.0)
    assert np.isclose(angle_between(q, q), 0.0)
    assert np.isclose(angle_between((1, 0, 0, 0), q), np.pi/2)


@pytest.mark.parametrize("name", list(FILTERS))
def test_static_convergence_from_wrong_initial_attitude(name):
    truth = _axis_angle((1, 0.3, 0), 0.25)
    acc = rotate_inverse(truth, np.array([0.0, 0.0, GRAVITY_M_S2]))
    # Mahony's bias integrator absorbs part of a large initial error and unwinds
    # it with a slow mode (~kp/ki seconds); deployments avoid that by aligning
    # from the accelerometer first (see test below). Here the proportional loop
    # alone is under test, so ki=0.
    f = FILTERS[name](q0=(1.0, 0.0, 0.0, 0.0), **({"ki": 0.0} if name == "mahony" else {}))
    for _ in range(1000):                             # 10 s at 100 Hz
        f.update(np.zeros(3), acc, 0.01)
    err = np.linalg.norm(body_gravity(f.q) - body_gravity(truth))
    assert err < 1e-3, f"{name}: gravity error after 10 s {err}"


def test_initial_alignment_from_accelerometer():
    truth = _axis_angle((0.4, -1.0, 0.0), 0.6)
    acc = rotate_inverse(truth, np.array([0.0, 0.0, GRAVITY_M_S2]))
    q0 = attitude_from_accel(acc)
    assert np.linalg.norm(body_gravity(q0) - body_gravity(truth)) < 1e-9
    # With a correct start the bias integrator has nothing to absorb.
    f = MahonyFilter(kp=1.0, ki=0.1, q0=q0)
    for _ in range(200):
        f.update(np.zeros(3), acc, 0.01)
    assert np.linalg.norm(body_gravity(f.q) - body_gravity(truth)) < 1e-9
    assert np.linalg.norm(f.bias) < 1e-9
    assert np.allclose(attitude_from_accel([0.0, 0.0, 9.81]), (1, 0, 0, 0))
    with pytest.raises(ValueError):
        attitude_from_accel([0.0, 0.0, 0.0])


@pytest.mark.parametrize("name,kw", [("mahony", dict(kp=2.0, ki=0.2)), ("madgwick", dict(beta=0.08))])
def test_tracks_rotation_with_noise_and_bias(name, kw):
    dt, traj = _synthetic(20.0, 100, lambda t: (0.5*np.sin(t), 0.3*np.cos(1.3*t), 0.2))
    noise = ImuNoise(gyro_std=0.01, accel_std=0.05, gyro_bias=0.02, seed=3)
    f = FILTERS[name](q0=traj[0][1], **kw)
    errs = []
    for t, q_true, w, acc in traj:
        g, a = noise(w, acc, dt)
        f.update(g, a, dt)
        if t > 3.0:
            errs.append(np.linalg.norm(body_gravity(f.q) - body_gravity(q_true)))
    rmse = float(np.sqrt(np.mean(np.square(errs))))
    assert rmse < 0.02, f"{name}: gravity RMSE {rmse}"
    if name == "mahony":
        # The integral term should have found most of the constant bias.
        assert np.linalg.norm(f.bias - noise.bias) < 0.5*np.linalg.norm(noise.bias)


def test_heading_is_unobservable_but_gravity_is_not():
    # A pure yaw offset never shows in the accelerometer: gravity error stays 0
    # while the attitude error stays at the yaw offset.
    truth = _axis_angle((0, 0, 1), 0.7)
    acc = rotate_inverse(truth, np.array([0.0, 0.0, GRAVITY_M_S2]))
    f = MahonyFilter(q0=(1.0, 0.0, 0.0, 0.0))
    for _ in range(300):
        f.update(np.zeros(3), acc, 0.01)
    assert np.linalg.norm(body_gravity(f.q) - body_gravity(truth)) < 1e-6
    assert abs(angle_between(f.q, truth) - 0.7) < 1e-3


def test_noise_model_delay_and_validation():
    n = ImuNoise(delay_steps=2, seed=0)
    seen = [n(np.full(3, k), np.zeros(3), 0.01)[0][0] for k in range(5)]
    assert seen == [0.0, 0.0, 0.0, 1.0, 2.0]         # stale repeat, then two steps old
    assert np.allclose(ImuNoise(seed=1).bias, 0.0)
    assert np.isclose(np.linalg.norm(ImuNoise(gyro_bias=0.05, seed=1).bias), 0.05)
    with pytest.raises(ValueError):
        ImuNoise(gyro_std=-1.0)
    with pytest.raises(ValueError):
        MahonyFilter().update(np.zeros(3), np.zeros(3), 0.0)
    with pytest.raises(ValueError):
        quat_normalize((0.0, 0.0, 0.0, 0.0))
    # Zero accelerometer (invalid) falls back to gyro integration only.
    f = MadgwickFilter(q0=_axis_angle((1, 0, 0), 0.1))
    q_before = f.q.copy()
    f.update(np.zeros(3), np.zeros(3), 0.01)
    assert np.allclose(f.q, q_before)
    assert np.allclose(quat_mul(quat_conj(q_before), q_before), (1, 0, 0, 0))
