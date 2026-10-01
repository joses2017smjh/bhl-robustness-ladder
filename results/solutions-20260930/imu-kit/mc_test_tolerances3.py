#!/usr/bin/env python3
"""Third MC pass for the test configurations: repo ImuNoise round trip (100 Hz, 10 min),
20 Hz 2nd-order low-pass at 200 Hz (10 min), IM10A-LSB quantization (3 min, 200 Hz)."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src")
from bhl_robust.fusion.attitude import ImuNoise
from scipy.signal import butter, lfilter
spec = importlib.util.spec_from_file_location("ia", sys.argv[1]); ia = importlib.util.module_from_spec(spec); spec.loader.exec_module(ia)
DEG, MG = math.pi / 180, 9.80665e-3
def q(v):
    v = np.asarray(v, float)
    return {"min": float(v.min()), "p16": float(np.percentile(v, 16)), "p50": float(np.median(v)),
            "p84": float(np.percentile(v, 84)), "max": float(v.max()), "n": int(v.size)}
args = ia.build_parser().parse_args(["stationary", "x.csv", "--no-plot", "--sim-rate", "100"])
res = {}
t0 = time.time()
# 1. repo ImuNoise round trip
fs, n = 100.0, 60000
P = {"gyro_std": 0.0495 * DEG, "accel_std": 0.99 * MG, "gyro_bias": 0.5 * DEG, "gyro_bias_walk": 1e-4}
rg, ra, rk, vis = [], [], [], []
for s in range(20):
    noise = ImuNoise(**P, delay_steps=0, seed=500 + s)
    g0, a0 = np.zeros(3), np.array([0.0, 0.0, 9.80665])
    G = np.empty((n, 3)); A = np.empty((n, 3))
    for k in range(n):
        G[k], A[k] = noise(g0, a0, 1.0 / fs)
    import contextlib, io
    with contextlib.redirect_stdout(io.StringIO()):
        m, r, _ = ia.analyse_stationary(ia.from_arrays(np.arange(n) / fs, G, A), args)
    rg.append(m["imu_noise"]["gyro_std"] / P["gyro_std"]); ra.append(m["imu_noise"]["accel_std"] / P["accel_std"])
    vis.append(m["gyro_bias_walk_kind"].startswith("measured"))
    rk.append(m["imu_noise"]["gyro_bias_walk"] / P["gyro_bias_walk"])
res["imunoise_100Hz_10min"] = {"gyro_std": q(rg), "accel_std": q(ra), "K_measured_frac": float(np.mean(vis)), "gyro_bias_walk": q(rk)}
print(res["imunoise_100Hz_10min"], f"{time.time() - t0:.0f} s", flush=True)
# 2. low-pass 20 Hz at 200 Hz: white-equivalent N kept, sample std / N sqrt(fs) < 0.8
fs, n = 200.0, 120000
b, a = butter(2, 20.0 / (fs / 2))
rn, rr = [], []
for s in range(20):
    y = lfilter(b, a, np.random.default_rng(600 + s).normal(0, 0.0035 * DEG * math.sqrt(fs), n))
    taus, adev, ms = ia.overlapping_adev(y, fs)
    f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
    rn.append(f["N"] / (0.0035 * DEG)); rr.append(ia.hf_std(y, fs) / (f["N"] * math.sqrt(fs)))
res["lpf20_200Hz_10min"] = {"N_over_true": q(rn), "sample_std_over_white": q(rr)}
print(res["lpf20_200Hz_10min"], flush=True)
# 3. IM10A LSB quantization, 3 min at 200 Hz: N vs sqrt(N^2 + q^2/(12 fs))
fs, n = 200.0, 36000
qg = 2000.0 / 32768 * DEG
rn = []
for s in range(20):
    y = np.rint((np.random.default_rng(700 + s).normal(0, 0.0035 * DEG * math.sqrt(fs), n) + 0.3 * DEG) / qg) * qg
    taus, adev, ms = ia.overlapping_adev(y, fs)
    f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
    rn.append(f["N"] / math.sqrt((0.0035 * DEG) ** 2 + qg ** 2 / (12 * fs)))
res["quant_200Hz_3min"] = {"N_over_quant_model": q(rn)}
print(res["quant_200Hz_3min"], flush=True)
res["elapsed_s"] = time.time() - t0
(HERE / "mc_test_tolerances3.json").write_text(json.dumps(res, indent=2) + "\n")
print(f"done in {res['elapsed_s']:.0f} s")
