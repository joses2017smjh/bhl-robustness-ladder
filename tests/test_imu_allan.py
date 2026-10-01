"""Synthetic recovery tests for scripts/sensors/imu_allan.py (IMU noise identification).

Known noise in, parameters out. The conventions under test (docs/IMU_RECORDING.md):
  N  white noise (gyro: angle random walk) = sigma(tau = 1 s) on the slope -1/2 line,
     in (rad/s)/sqrt(Hz). Per-sample std at output rate f = N*sqrt(f), NOT N*sqrt(f/2).
  B  bias instability = min sigma / 0.6643 (IEEE Std 952 flicker floor).
  K  rate random walk = sigma(tau = 3 s) on the slope +1/2 line = ImuNoise.gyro_bias_walk
     (ImuNoise adds N(0,1)*K*sqrt(dt) to its bias per call, so no rate conversion).
Stochastic tolerances come from 20-seed Monte Carlo runs of the same configurations, made
before the seeds below were fixed; the observed spread is quoted next to each check.
Optional dependencies skip cleanly: rosbags (bag reader), scipy (low-pass test), mujoco
(imu_sim import), matplotlib (plot). Without pandas the CSV reader's numpy fallback runs.
"""
from __future__ import annotations

import dataclasses
import importlib.util
import json
import math
import sys
from pathlib import Path

import numpy as np
import pytest

from bhl_robust.fusion.attitude import ImuNoise

REPO = Path(__file__).resolve().parents[1]
SCRIPT = REPO / "scripts" / "sensors" / "imu_allan.py"


def _load_script():
    spec = importlib.util.spec_from_file_location("imu_allan", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ia = _load_script()

DEG = math.pi / 180.0
G0 = 9.80665
MG = 1e-3 * G0
FLICKER = math.sqrt(2.0 * math.log(2.0) / math.pi)        # 0.6643
# IM10A-like gyro levels in rad/s units: per-sample std at 200 Hz = 0.0035*sqrt(200) = 0.0495 deg/s
# (datasheet 0.028-0.07 deg/s rms), bias instability 10 deg/h, rate random walk 0.000261 deg/s/sqrt(s)
N_G, B_G, K_G = 0.0035 * DEG, 10.0 / 3600.0 * DEG, 0.000261 * DEG
N_A = 0.070 * MG                                          # accel, (m/s^2)/sqrt(Hz): 0.99 mg per sample at 200 Hz
GYRO_BIAS = np.array([0.3, -0.5, 0.2]) * DEG
ACC_BIAS = np.array([12.0, -25.0, 18.0]) * MG


# ----------------------------------------------------------------------------- helpers
def _args(*argv):
    return ia.build_parser().parse_args([str(a) for a in argv])


def _tau_grid(n, fs, per_decade=20):
    """The m grid of overlapping_adev (same construction)."""
    m_max = n // 2
    num = max(2, int(math.log10(m_max) * per_decade) + 1)
    ms = np.unique(np.rint(np.logspace(0, math.log10(m_max), num)).astype(np.int64))
    return ms / fs, ms


def _flicker(rng, n, fs, B):
    """1/f rate noise with one-sided PSD B^2/(pi f) on the FFT grid df..fs/2: Allan floor 0.6643 B."""
    M = 1 << int(math.ceil(math.log2(n)))
    f = np.fft.rfftfreq(M, 1.0 / fs)
    S1 = np.zeros_like(f)
    S1[1:] = B ** 2 / (math.pi * f[1:])
    X = np.sqrt(S1 * fs * M / 2) * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / math.sqrt(2)
    X[0] = 0.0
    return np.fft.irfft(X, n=M)[:n], fs / M


def _analytic_b_min(n, fs, N, B, K, df):
    """Minimum over the script's usable tau grid of the ANALYTIC Allan deviation of white N +
    the generated flicker + rate random walk K, divided by 0.6643: what the min method reads."""
    taus, _ = _tau_grid(n, fs)
    taus = taus[(taus >= 10.0 / fs) & (taus <= 0.2 * n / fs)]
    f = np.arange(1, int(round(fs / 2 / df)) + 1) * df
    w = 2.0 * df * B ** 2 / (math.pi * f)               # 2 * S1(f) * df
    av = N ** 2 / taus + K ** 2 * taus / 3.0
    for i, t in enumerate(taus):
        x = math.pi * t * f
        s2 = np.sin(x)
        s2 *= s2
        av[i] += np.dot(w, s2 * s2 / (x * x))
    return math.sqrt(av.min()) / FLICKER


def _rate_noise(rng, n, fs, N, B, K):
    fl, df = _flicker(rng, n, fs, B)
    y = rng.normal(0.0, N * math.sqrt(fs), n) + fl + np.cumsum(rng.normal(size=n) * K * math.sqrt(1.0 / fs))
    return y, df


def _tilted_gravity(roll_deg=1.2, pitch_deg=-0.8):
    r, p = math.radians(roll_deg), math.radians(pitch_deg)
    return G0 * np.array([-math.sin(p), math.sin(r) * math.cos(p), math.cos(r) * math.cos(p)])


def _write_csv(path, cols):
    names = list(cols)
    fmt = ["%.9f" if k in ("t", "t_recv") else "%.17g" for k in names]
    np.savetxt(path, np.column_stack([np.asarray(cols[k], float) for k in names]), delimiter=",",
               header=",".join(names), comments="", fmt=fmt)


def _stationary(G, A, fs, *extra):
    data = ia.from_arrays(np.arange(G.shape[0]) / fs, G, A)
    return ia.analyse_stationary(data, _args("stationary", "in-memory", "--no-plot", *extra))


def _qmul(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def _rotmat(q):
    """Body->world rotation matrices of (n, 4) w,x,y,z quaternions."""
    w, x, y, z = q.T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)], -1),
                     np.stack([2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)], -1),
                     np.stack([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)], -1)], -2)


def _rotation_capture(seed, fs=100.0, flip_gyro_y=False, quat_wxyz=False):
    """The IMU_RECORDING.md capture (b): 13 holds (30 s, 11 x 10 s, 30 s) and 12 slow moves about
    the board axes, with a known accelerometer bias and scale error."""
    rng = np.random.default_rng(seed)
    moves = [("X", 90), ("X", -90), ("X", -90), ("X", 90), ("Y", 90), ("Y", -90), ("Y", -90), ("Y", 90),
             ("Z", 90), ("Z", -90), ("X", 180), ("X", -180)]
    holds = [30.0] + [10.0] * 11 + [30.0]
    q = np.array([1.0, 0.0, 0.0, 0.0])
    W, Q = [], []
    for i, h in enumerate(holds):
        k = int(h * fs)
        W.append(np.zeros((k, 3)))
        Q.append(np.tile(q, (k, 1)))
        if i < len(moves):
            ax, deg = moves[i]
            e = np.eye(3)["XYZ".index(ax)]
            m = int((3.0 if abs(deg) == 90 else 5.0) * fs)
            prof = (1 - np.cos(2 * np.pi * (np.arange(m) + 0.5) / m)) / 2
            w = prof / prof.sum() * math.radians(deg + rng.uniform(-1.5, 1.5)) * fs    # rad/s, sum*dt = angle
            qs = np.empty((m, 4))
            for j, wk in enumerate(w):
                th = wk / fs
                q = _qmul(q, np.array([math.cos(th / 2), *(math.sin(th / 2) * e)]))
                q = q / np.linalg.norm(q)
                qs[j] = q
            W.append(w[:, None] * e)
            Q.append(qs)
    W, Q = np.concatenate(W), np.concatenate(Q)
    F = np.einsum("nji,j->ni", _rotmat(Q), np.array([0.0, 0.0, G0]))   # R^T (0,0,+g): REP-145 specific force
    scale = np.array([0.005, -0.003, 0.002])
    A = F * (1 + scale) + ACC_BIAS + rng.normal(0.0, 1.0 * MG, F.shape)
    G = W + GYRO_BIAS + rng.normal(0.0, 0.05 * DEG, W.shape)
    if flip_gyro_y:
        G[:, 1] *= -1
    Qc = Q if quat_wxyz else Q[:, [1, 2, 3, 0]]     # columns qx,qy,qz,qw (w first = the fault)
    return 1000.0 + np.arange(W.shape[0]) / fs, G, A, Qc, scale


# ----------------------------------------------------------------------------- conventions
def test_reading_conventions_exact_on_the_ieee952_model_curve():
    """No sampling noise: feed fit_allan the IEEE 952 curve AVAR = N^2/tau + (0.6643 B)^2 + K^2 tau/3.
    N must come back as sigma(1 s) of the -1/2 line, B as the flat term / 0.6643 and as min/0.6643,
    K as sigma(3 s) of the +1/2 line."""
    assert ia.FLICKER == pytest.approx(0.6643, abs=1e-4)
    fs, n = 200.0, int(2 * 3600 * 200)
    taus, ms = _tau_grid(n, fs)
    avar = N_G ** 2 / taus + (FLICKER * B_G) ** 2 + K_G ** 2 * taus / 3.0
    f = ia.fit_allan(taus, np.sqrt(avar), ms, n, fs, 0.2)
    assert f["N"] == pytest.approx(N_G, rel=1e-3)
    assert f["fit"]["B"] == pytest.approx(B_G, rel=1e-3)
    assert f["K_visible"] and f["K"] == pytest.approx(K_G, rel=1e-3)
    b_min = math.sqrt(avar[taus <= 0.2 * n / fs].min()) / FLICKER
    assert f["B"] == pytest.approx(b_min, rel=0.01)
    assert f["B"] > B_G                       # white and RRW lift the minimum above the flicker floor
    assert not f["B_is_upper_bound"]


@pytest.mark.parametrize("fs, seed", [(100.0, 11), (200.0, 12)])
def test_white_noise_per_sample_std_is_N_sqrt_f(fs, seed):
    """i.i.d. samples of std sigma_d at rate f have AVAR = sigma_d^2/(f tau), so sigma_d = N*sqrt(f).
    MC, 20 seeds x 10 min: N*sqrt(f)/sigma_d 0.997-1.004; N*sqrt(f/2)/sigma_d 0.705-0.710."""
    n = int(600 * fs)
    y = np.random.default_rng(seed).normal(0.0, N_G * math.sqrt(fs), n)
    taus, adev, ms = ia.overlapping_adev(y, fs)
    f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
    drawn = float(y.std())
    assert f["N"] * math.sqrt(fs) / drawn == pytest.approx(1.0, abs=0.02)
    assert f["N"] * math.sqrt(fs / 2) / drawn < 0.8          # the N*sqrt(f/2) reading is off by 1/sqrt(2)
    assert f["N"] == pytest.approx(N_G, rel=0.03)
    i1 = int(np.argmin(np.abs(taus - 1.0)))
    assert adev[i1] * math.sqrt(taus[i1]) == pytest.approx(N_G, rel=0.10)   # sigma(1 s) on the white line is N


@pytest.mark.parametrize("fs, seed", [(100.0, 21), (200.0, 22)])
def test_flicker_floor_and_rrw_bound_at_im10a_like_levels_30min(fs, seed):
    """White + flicker + rate random walk at the levels above, 30 min. MC, 20 seeds: N 0.994-1.003,
    least-squares B / B 0.93-1.06; K was visible in only 25-35 % of records and the K upper bound
    covered the true K in 95 % (100 Hz) and 100 % (200 Hz) of them."""
    n = int(1800 * fs)
    y, _ = _rate_noise(np.random.default_rng(seed), n, fs, N_G, B_G, K_G)
    taus, adev, ms = ia.overlapping_adev(y, fs)
    f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
    assert f["N"] == pytest.approx(N_G, rel=0.03)
    assert f["fit"]["B"] == pytest.approx(B_G, rel=0.12)
    assert f["K_upper"] >= K_G
    if f["K_visible"]:
        assert f["K"] > 0


@pytest.mark.parametrize("fs, seed", [(100.0, 31), (200.0, 32)])
def test_rate_random_walk_and_min_method_30min(fs, seed):
    """K x5 (the +1/2 line crosses the floor at ~6 s, so 30 min shows it). MC, 20 seeds: K visible
    20/20, K/K_true 0.72-1.17; B_min / (analytic min / 0.6643) 0.965-1.063; N 0.996-1.004."""
    n = int(1800 * fs)
    K = 5 * K_G
    y, df = _rate_noise(np.random.default_rng(seed), n, fs, N_G, B_G, K)
    taus, adev, ms = ia.overlapping_adev(y, fs)
    f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
    assert f["K_visible"]
    assert f["K"] == pytest.approx(K, rel=0.30)
    assert f["B"] == pytest.approx(_analytic_b_min(n, fs, N_G, B_G, K, df), rel=0.10)
    assert f["N"] == pytest.approx(N_G, rel=0.03)


# ----------------------------------------------------------------------------- stationary CLI / JSON
def test_stationary_cli_on_csv_end_to_end(tmp_path):
    """CSV with dropouts, host jitter, a 50 ppm device clock, magnetometer and barometer columns:
    the drop-in keys, unit identities and the ImuNoise block."""
    fs, n = 200.0, 36000
    rng = np.random.default_rng(51)
    G = rng.normal(0.0, N_G * math.sqrt(fs), (n, 3)) + GYRO_BIAS
    A = rng.normal(0.0, N_A * math.sqrt(fs), (n, 3)) + ACC_BIAS + _tilted_gravity()
    keep = np.ones(n, bool)
    keep[rng.choice(np.arange(10, n - 10), size=72, replace=False)] = False
    keep[20000:20040] = False
    idx = np.flatnonzero(keep)
    t = 1.7e9 + idx / (fs * (1 + 50e-6)) + 0.002 + np.abs(rng.normal(0.0, 0.0003, idx.size))
    step_T, mb = 0.0667e-7, np.array([20.0, 5.0, -45.0]) * 1e-6
    M = np.rint((mb + rng.normal(0.0, 0.3e-6, (idx.size, 3))) / step_T) * step_T
    M[np.arange(idx.size) % 4 != 0] = np.nan                       # 50 Hz magnetometer updates
    p = 101325.0 + rng.normal(0.0, 1.2, idx.size)
    p[np.arange(idx.size) % 8 != 0] = np.nan                       # 25 Hz barometer updates
    csv = tmp_path / "still.csv"
    _write_csv(csv, {"t": t, "gx": G[idx, 0], "gy": G[idx, 1], "gz": G[idx, 2], "ax": A[idx, 0], "ay": A[idx, 1],
                     "az": A[idx, 2], "mx": M[:, 0], "my": M[:, 1], "mz": M[:, 2], "p": p})
    out = tmp_path / "out"
    plot = importlib.util.find_spec("matplotlib") is not None
    argv = ["stationary", str(csv), "--nominal-rate", "200", "--out-dir", str(out)] + ([] if plot else ["--no-plot"])
    assert ia.main(argv) == 0
    m = json.loads((out / "imu_measured.json").read_text())
    r = json.loads((out / "imu_allan_report.json").read_text())
    if plot:
        assert (out / "imu_allan.png").stat().st_size > 10_000

    # the eight IM10A_DATASHEET keys, each with a provenance
    assert set(ia.IM10A_DATASHEET) <= set(m) and set(ia.IM10A_DATASHEET) <= set(m["provenance"])
    assert m["max_rate_hz"] == 200.0 and "imu_allan.py" in m["source"]
    # unit identities (module docstring): N in unit/sqrt(Hz), per-sample std = N*sqrt(rate), deg/sqrt(h) = x60
    nd, rate = m["noise_density"], m["rate_hz"]
    assert m["gyro_noise_dps"] == pytest.approx(nd["gyro_dps_rthz"] * math.sqrt(rate), rel=1e-12)
    assert nd["gyro_dps_rthz"] == pytest.approx(nd["gyro_rad_s_rthz"] / DEG, rel=1e-12)
    assert nd["gyro_arw_deg_rth"] == pytest.approx(nd["gyro_dps_rthz"] * 60.0, rel=1e-12)
    assert m["accel_noise_mg"] == pytest.approx(nd["accel_mg_rthz"] * math.sqrt(rate), rel=1e-12)
    assert nd["accel_mg_rthz"] == pytest.approx(nd["accel_m_s2_rthz"] / MG, rel=1e-12)
    assert m["per_sample_std_at"]["100"]["gyro_dps"] == pytest.approx(nd["gyro_dps_rthz"] * 10.0, rel=1e-12)
    assert m["gyro_bias_instability_deg_h"] == pytest.approx(m["gyro_bias_instability_dps"] * 3600.0, rel=1e-12)
    # recovered vs generated
    assert rate == pytest.approx(fs, rel=1e-3)
    assert m["gyro_noise_dps"] == pytest.approx(N_G * math.sqrt(fs) / DEG, rel=0.03)
    assert m["accel_noise_mg"] == pytest.approx(N_A * math.sqrt(fs) / MG, rel=0.03)
    assert m["gyro_bias_dps"] == pytest.approx(np.linalg.norm(G[idx].mean(0)) / DEG, rel=1e-9)
    assert m["gyro_bias_dps"] == pytest.approx(np.linalg.norm(GYRO_BIAS) / DEG, rel=0.005)
    assert "PARTIAL" in m["provenance"]["accel_bias_mg"]
    assert m["accel_bias_mg"] == pytest.approx(abs(np.linalg.norm(A[idx].mean(0)) - G0) / MG, rel=1e-6)
    assert r["timing"]["missing_samples_est"] == n - idx.size
    assert r["samples_interpolated"] == n - idx.size
    assert r["gravity"]["up_axis"] == "+Z"
    assert m["mag_resolution_mgauss"] == pytest.approx(0.0667, rel=0.01)
    assert r["magnetometer"]["field_magnitude_ut"]["mean"] == pytest.approx(np.linalg.norm(mb) * 1e6, rel=0.005)
    assert m["baro_noise_pa"] == pytest.approx(1.2, rel=0.08)
    assert any("record is 3 min" in w for w in m["warnings"])

    # ImuNoise block: exactly the constructor fields, per call at the stated rate, no rate factor on the walk
    fields = {f.name for f in dataclasses.fields(ImuNoise) if f.init} - {"seed"}
    assert set(m["imu_noise"]) == fields
    for key in ("25", "50", "100", "200"):
        ImuNoise(**m["imu_noise_by_rate"][key], seed=0)
    assert m["imu_noise_rate_hz"] == 200.0 and m["imu_noise"] == m["imu_noise_by_rate"]["200"]
    assert m["imu_noise"]["gyro_std"] == pytest.approx(nd["gyro_rad_s_rthz"] * math.sqrt(200.0), rel=1e-12)
    assert m["imu_noise_by_rate"]["25"]["gyro_std"] == pytest.approx(m["imu_noise"]["gyro_std"] / math.sqrt(8.0),
                                                                    rel=1e-12)
    assert m["imu_noise"]["gyro_bias"] == pytest.approx(m["gyro_bias_dps"] * DEG, rel=1e-12)
    assert m["imu_noise_by_rate"]["25"]["gyro_bias_walk"] == m["imu_noise"]["gyro_bias_walk"] \
        == m["gyro_bias_walk_rad_s_rtsec"]
    assert m["imu_noise"]["delay_steps"] == 0 and m["imu_latency_ms"] is None
    # Kalibr IMU noise model uses the same conventions (noise density = N, random walk = K)
    k = m["kalibr_imu"]
    assert k["gyroscope_noise_density"] == pytest.approx(nd["gyro_rad_s_rthz"], rel=1e-12)
    assert k["accelerometer_noise_density"] == pytest.approx(nd["accel_m_s2_rthz"], rel=1e-12)
    assert k["gyroscope_random_walk"] == m["imu_noise"]["gyro_bias_walk"]


def test_repo_imunoise_round_trip():
    """Samples made by the repo's own ImuNoise (100 Hz, 10 min) come back as its constructor values.
    MC, 20 seeds: gyro_std 0.996-1.005, accel_std 0.999-1.007, bias walk visible 20/20 at 0.90-1.09."""
    fs, n = 100.0, 60000
    P = {"gyro_std": 0.0495 * DEG, "accel_std": 0.99 * MG, "gyro_bias": 0.5 * DEG, "gyro_bias_walk": 1e-4}
    noise = ImuNoise(**P, delay_steps=0, seed=61)
    g0, a0 = np.zeros(3), _tilted_gravity()
    G, A = np.empty((n, 3)), np.empty((n, 3))
    for k in range(n):
        G[k], A[k] = noise(g0, a0, 1.0 / fs)
    m, _, _ = _stationary(G, A, fs, "--sim-rate", "100")
    got = m["imu_noise"]
    assert got["gyro_std"] == pytest.approx(P["gyro_std"], rel=0.02)
    assert got["accel_std"] == pytest.approx(P["accel_std"], rel=0.02)
    assert got["gyro_bias"] == pytest.approx(np.linalg.norm(G.mean(0)), rel=1e-9)
    assert m["gyro_bias_walk_kind"].startswith("measured")
    assert got["gyro_bias_walk"] == pytest.approx(P["gyro_bias_walk"], rel=0.30)


def test_device_lowpass_is_flagged_and_white_equivalent_kept():
    """IM10A default Bandwidth 20 Hz at 200 Hz output: correlated samples are flagged and N is the
    white-equivalent level. MC, 20 seeds x 10 min: N/N_true 0.959-1.003 (median 0.981; 0.950 before
    the white-region start was tightened), per-sample std / N*sqrt(fs) 0.47-0.49."""
    signal = pytest.importorskip("scipy.signal")
    fs, n = 200.0, 120000
    b, a = signal.butter(2, 20.0 / (fs / 2))
    rng = np.random.default_rng(71)
    G = signal.lfilter(b, a, rng.normal(0.0, N_G * math.sqrt(fs), (n, 3)), axis=0) + GYRO_BIAS
    A = signal.lfilter(b, a, rng.normal(0.0, N_A * math.sqrt(fs), (n, 3)), axis=0) + ACC_BIAS + _tilted_gravity()
    _, r, _ = _stationary(G, A, fs, "--nominal-rate", "200")
    ratios = [r["allan"]["gyro"][ax]["N"] / N_G for ax in "xyz"]
    assert all(0.94 <= x <= 1.03 for x in ratios) and np.mean(ratios) > 0.96
    assert all(r["allan"]["gyro"][ax]["sample_std_over_white"] < 0.8 for ax in "xyz")
    assert any("correlated" in w for w in r["warnings"])


def test_quantization_duplicates_and_bias_step_are_flagged():
    """IM10A output steps (0.061 deg/s, 0.0005 g per LSB), 1 % repeated samples and one 0.05 deg/s
    gyro bias step (what the vendor auto-calibration does), 3 min at 200 Hz. MC (quantization only,
    20 seeds): N / sqrt(N^2 + q^2/(12 fs)) 0.993-1.011."""
    fs, n = 200.0, 36000
    rng = np.random.default_rng(81)
    qg, qa = 2000.0 / 32768 * DEG, 16.0 / 32768 * G0
    G = rng.normal(0.0, N_G * math.sqrt(fs), (n, 3)) + GYRO_BIAS
    G[n // 2:, 0] += 0.05 * DEG
    A = rng.normal(0.0, N_A * math.sqrt(fs), (n, 3)) + ACC_BIAS + _tilted_gravity()
    G, A = np.rint(G / qg) * qg, np.rint(A / qa) * qa
    rep = rng.choice(np.arange(1, n), size=n // 100, replace=False)
    G[rep], A[rep] = G[rep - 1], A[rep - 1]
    _, r, _ = _stationary(G, A, fs, "--nominal-rate", "200")
    assert r["quantization"]["gx"]["step"] / DEG == pytest.approx(2000.0 / 32768, rel=0.01)
    assert r["quantization"]["az"]["step"] / MG == pytest.approx(16.0 / 32768 * 1000, rel=0.01)
    assert any("quantization-limited" in w for w in r["warnings"])
    assert r["duplicates"]["fraction"] == pytest.approx(0.01, rel=0.2) and r["duplicates"]["suspicious"]
    steps = [s for s in r["drift"]["gyro"]["steps"] if s["axis"] == "x"]
    assert len(steps) == 1 and abs(steps[0]["t_s"] - n / 2 / fs) < 15.0
    assert any("bias step" in w for w in r["warnings"])
    for ax in "yz":
        assert r["allan"]["gyro"][ax]["N"] == pytest.approx(math.sqrt(N_G ** 2 + qg ** 2 / (12 * fs)), rel=0.05)


# ----------------------------------------------------------------------------- rotations / tap
def test_rotation_protocol_recovers_six_position_bias_and_conventions():
    t, G, A, Q, scale = _rotation_capture(91)
    rep = ia.analyse_rotations(ia.from_arrays(t, G, A, Q), _args("rotations", "in-memory"))
    es = rep["expected_sequence"]
    assert len(es["faces"]) == 13 and all(x["ok"] for x in es["faces"])
    assert len(es["moves"]) == 12 and all(x["ok"] for x in es["moves"])
    six = rep["six_position"]
    for i, a in enumerate("XYZ"):
        assert six["axes"][a]["bias_mg"] == pytest.approx(ACC_BIAS[i] / MG, abs=1.5)
        assert six["axes"][a]["scale_error_pct"] == pytest.approx(100 * scale[i], abs=0.25)
    assert six["bias_norm_mg"] == pytest.approx(np.linalg.norm(ACC_BIAS) / MG, abs=2.6)
    c = rep["gyro_accel_consistency"]
    assert c["identity_is_best"] and c["identity_rms_error_deg"] < 1.0
    assert rep["quaternion"]["best"].startswith("xyzw fields, body->world")
    assert not any("disagree" in w or "quaternion matches" in w for w in rep["warnings"])


def test_rotation_check_catches_flipped_gyro_axis_and_w_first_quaternion():
    t, G, A, Q, _ = _rotation_capture(92, flip_gyro_y=True, quat_wxyz=True)
    rep = ia.analyse_rotations(ia.from_arrays(t, G, A, Q), _args("rotations", "in-memory"))
    c = rep["gyro_accel_consistency"]
    assert np.allclose(c["best_gyro_to_accel_matrix"], np.diag([1.0, -1.0, 1.0])) and not c["identity_is_best"]
    bad = sorted(x["expected"] for x in rep["expected_sequence"]["moves"] if not x["ok"])
    assert bad == sorted(["+Y90", "-Y90", "-Y90", "+Y90"])
    assert rep["quaternion"]["best"].startswith("w stored in x slot (wxyz order), body->world")
    assert any("disagree" in w for w in rep["warnings"]) and any("quaternion matches" in w for w in rep["warnings"])


def test_tap_offset_against_a_reference_and_delay_steps(tmp_path):
    """15 taps seen by this IMU (200 Hz, stamps 23 ms late) and a 400 Hz reference whose own latency
    is 4 ms: offset ~23 ms, stamp latency ~27 ms, delay_steps from the absolute value only."""
    rng = np.random.default_rng(101)
    dur, offset = 45.0, 0.023
    taps = 3.0 + np.arange(15) * 2.5 + rng.uniform(-0.3, 0.3, 15)
    amps = rng.uniform(1.5, 4.0, taps.size)
    dirs = rng.normal(size=(taps.size, 3))
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)

    def imu(fs, delay, noise_mg, path):
        t = np.arange(int(dur * fs)) / fs
        a = np.tile(_tilted_gravity(), (t.size, 1)) + rng.normal(0.0, noise_mg * MG, (t.size, 3))
        for T, amp, d in zip(taps, amps, dirs):          # the same physical tap seen by both sensors
            s = t - T
            k = s >= 0
            a[k] += (amp * np.exp(-s[k] / 0.015) * np.sin(2 * np.pi * 60 * s[k]))[:, None] * d
        z = np.zeros_like(t)
        _write_csv(path, {"t": t + delay, "gx": z, "gy": z, "gz": z, "ax": a[:, 0], "ay": a[:, 1], "az": a[:, 2]})

    imu(200.0, offset, 1.0, tmp_path / "main.csv")
    imu(400.0, 0.0, 1.5, tmp_path / "ref.csv")
    rep = ia.analyse_tap(_args("tap", tmp_path / "main.csv", "--ref-csv", tmp_path / "ref.csv", "--ref-latency-ms", 4))
    assert rep["offset_ms"]["n"] >= 14
    assert rep["offset_ms"]["median"] == pytest.approx(23.0, abs=2.0)
    assert rep["latency_ms_estimate"] == pytest.approx(27.0, abs=2.0)
    rel = ia.analyse_tap(_args("tap", tmp_path / "main.csv", "--ref-csv", tmp_path / "ref.csv"))
    assert rel["latency_ms_estimate"] is None and rel["offset_ms"]["median"] == pytest.approx(23.0, abs=2.0)

    fs, n = 200.0, 24000
    G = rng.normal(0.0, N_G * math.sqrt(fs), (n, 3))
    A = rng.normal(0.0, N_A * math.sqrt(fs), (n, 3)) + _tilted_gravity()
    for name, tap in (("abs", rep), ("rel", rel)):
        (tmp_path / f"tap_{name}.json").write_text(json.dumps(ia._jsonable(tap)))
        m, _, _ = _stationary(G, A, fs, "--tap-report", tmp_path / f"tap_{name}.json")
        assert m["imu_tap_offset_ms"] == pytest.approx(tap["offset_ms"]["median"], rel=1e-12)
        if name == "abs":
            lat = tap["latency_ms_estimate"]
            assert m["imu_latency_ms"] == pytest.approx(lat, rel=1e-12)
            for r_ in (25, 200):
                assert m["imu_noise_by_rate"][str(r_)]["delay_steps"] == int(round(lat * r_ / 1000.0))
        else:          # only a relative offset: no absolute latency, delay_steps stays 0
            assert m["imu_latency_ms"] is None and m["imu_noise"]["delay_steps"] == 0


# ----------------------------------------------------------------------------- inputs / optional deps
def test_rosbag2_sqlite3_and_mcap_read_like_the_csv(tmp_path):
    pytest.importorskip("rosbags")
    typesys = pytest.importorskip("rosbags.typesys")
    if not hasattr(typesys, "get_typestore"):
        pytest.skip("rosbags is too old (no typesys.get_typestore)")
    from rosbags.rosbag2 import StoragePlugin, Writer
    ts = typesys.get_typestore(typesys.Stores.ROS2_HUMBLE)
    T = ts.types
    Imu, Hdr, Time = T["sensor_msgs/msg/Imu"], T["std_msgs/msg/Header"], T["builtin_interfaces/msg/Time"]
    Quat, Vec = T["geometry_msgs/msg/Quaternion"], T["geometry_msgs/msg/Vector3"]
    Mag, Pres = T["sensor_msgs/msg/MagneticField"], T["sensor_msgs/msg/FluidPressure"]
    fs, n = 200.0, 12000
    rng = np.random.default_rng(111)
    G = rng.normal(0.0, N_G * math.sqrt(fs), (n, 3)) + GYRO_BIAS
    A = rng.normal(0.0, N_A * math.sqrt(fs), (n, 3)) + ACC_BIAS + _tilted_gravity()
    ns = 1_700_000_000_000_000_000 + np.rint(np.arange(n) * 1e9 / fs).astype(np.int64)    # header stamps
    recv = ns + 3_000_000 + np.rint(np.abs(rng.normal(0.0, 2e5, n))).astype(np.int64)     # ~3 ms later
    step_T, mb = 0.0667e-7, np.array([20.0, 5.0, -45.0]) * 1e-6
    M = np.rint((mb + rng.normal(0.0, 0.3e-6, (n // 4, 3))) / step_T) * step_T
    P = 101325.0 + rng.normal(0.0, 1.2, n // 8)
    _, ref, _ = _stationary(G, A, fs)            # same samples, in memory (report)
    z9 = np.zeros(9)

    def stamp(v):
        return Time(sec=int(v // 1_000_000_000), nanosec=int(v % 1_000_000_000))

    for plugin, cov0 in ((StoragePlugin.SQLITE3, 0.0025), (StoragePlugin.MCAP, -1.0)):
        bag = tmp_path / f"bag_{plugin.name.lower()}"
        ocov = z9.copy()
        ocov[0] = cov0
        with Writer(bag, version=9, storage_plugin=plugin) as w:
            ci = w.add_connection("/imu/data_raw", Imu.__msgtype__, typestore=ts)
            cm = w.add_connection("/wit/mag", Mag.__msgtype__, typestore=ts)
            cp = w.add_connection("/imu/pressure", Pres.__msgtype__, typestore=ts)
            for k in range(n):
                h = Hdr(stamp=stamp(ns[k]), frame_id="imu_link")
                msg = Imu(header=h, orientation=Quat(x=0.0, y=0.0, z=0.0, w=1.0), orientation_covariance=ocov,
                          angular_velocity=Vec(x=G[k, 0], y=G[k, 1], z=G[k, 2]), angular_velocity_covariance=z9,
                          linear_acceleration=Vec(x=A[k, 0], y=A[k, 1], z=A[k, 2]), linear_acceleration_covariance=z9)
                w.write(ci, int(recv[k]), ts.serialize_cdr(msg, Imu.__msgtype__))
                if k % 4 == 0:
                    v = M[k // 4]
                    w.write(cm, int(recv[k]) + 1, ts.serialize_cdr(
                        Mag(header=h, magnetic_field=Vec(x=v[0], y=v[1], z=v[2]), magnetic_field_covariance=z9),
                        Mag.__msgtype__))
                if k % 8 == 0:
                    w.write(cp, int(recv[k]) + 2, ts.serialize_cdr(
                        Pres(header=h, fluid_pressure=P[k // 8], variance=0.0), Pres.__msgtype__))
        out = tmp_path / f"out_{plugin.name.lower()}"
        assert ia.main(["stationary", str(bag), "--out-dir", str(out), "--no-plot"]) == 0
        r = json.loads((out / "imu_allan_report.json").read_text())
        m = json.loads((out / "imu_measured.json").read_text())
        worst = max(abs(r["allan"][s][a]["N"] / ref["allan"][s][a]["N"] - 1) for s in ("gyro", "accel") for a in "xyz")
        assert worst < 1e-6
        assert [r["selected_topics"][k] for k in ("imu", "mag", "baro")] == ["/imu/data_raw", "/wit/mag", "/imu/pressure"]
        assert r["frame_ids"] == ["imu_link"]
        assert (r["orientation_unavailable_frac"] == 1.0) == (cov0 < 0)
        assert r["timing"]["recv_minus_stamp_ms"]["median"] == pytest.approx(3.0 + 0.135, abs=0.1)
        assert m["mag_resolution_mgauss"] == pytest.approx(0.0667, rel=0.01)
        assert m["baro_noise_pa"] == pytest.approx(1.2, rel=0.1)
        csv = tmp_path / f"export_{plugin.name.lower()}.csv"
        assert ia.main(["export-csv", str(bag), "--output", str(csv), "--out-dir", str(out)]) == 0
        header = csv.read_text().split("\n", 1)[0].split(",")
        assert ("qw" in header) == (cov0 >= 0)              # no placeholder quaternion when cov[0] == -1
        assert ia.main(["stationary", str(csv), "--out-dir", str(out / "csv"), "--no-plot"]) == 0
        r2 = json.loads((out / "csv" / "imu_allan_report.json").read_text())
        worst = max(abs(r2["allan"][s][a]["N"] / r["allan"][s][a]["N"] - 1) for s in ("gyro", "accel") for a in "xyz")
        assert worst < 1e-9


def test_datasheet_copy_matches_imu_sim():
    pytest.importorskip("mujoco")                 # bhl_robust.eval.imu_sim imports mujoco
    from bhl_robust.eval.imu_sim import IM10A_DATASHEET
    assert ia.IM10A_DATASHEET == IM10A_DATASHEET


def test_bag_input_without_rosbags_says_how_to_install(tmp_path, monkeypatch):
    bag = tmp_path / "some_bag"
    bag.mkdir()
    (bag / "metadata.yaml").write_text("rosbag2_bagfile_information: {}\n")
    monkeypatch.setitem(sys.modules, "rosbags", None)
    monkeypatch.setitem(sys.modules, "rosbags.highlevel", None)
    with pytest.raises(SystemExit, match="pip install rosbags"):
        ia.main(["stationary", str(bag), "--out-dir", str(tmp_path / "out")])


def test_csv_reader_without_pandas(tmp_path, monkeypatch):
    fs, n = 200.0, 2000
    rng = np.random.default_rng(121)
    cols = {"t": np.arange(n) / fs, **{c: rng.normal(size=n) for c in ("gx", "gy", "gz", "ax", "ay", "az")}}
    cols.update(qx=np.zeros(n), qy=np.zeros(n), qz=np.zeros(n), qw=np.ones(n))
    csv = tmp_path / "x.csv"
    _write_csv(csv, cols)
    monkeypatch.setitem(sys.modules, "pandas", None)
    d = ia.read_csv(csv, gyro_unit="deg/s", accel_unit="g")
    assert np.array_equal(d["imu"]["t"], cols["t"])
    assert np.allclose(d["imu"]["gyro"][:, 1], cols["gy"] * DEG, rtol=1e-15, atol=0)
    assert np.allclose(d["imu"]["accel"][:, 2], cols["az"] * G0, rtol=1e-15, atol=0)
    assert d["imu"]["quat"].shape == (n, 4) and np.all(d["imu"]["quat"][:, 3] == 1.0)


def test_doc_flag_prints_the_conventions(capsys):
    assert ia.main(["--doc"]) == 0
    doc = capsys.readouterr().out
    assert "sigma_d(f) = N * sqrt(f)" in doc and "0.6643" in doc and "K*sqrt(tau/3)" in doc
