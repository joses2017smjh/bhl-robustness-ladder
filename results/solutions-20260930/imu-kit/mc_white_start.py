#!/usr/bin/env python3
"""Regression MC for the white-region start tolerance (WHITE_START_SLOPE_TOL): low-pass, white,
kit regime and K x5 regime, 20 seeds each, same seeds as mc_test_tolerances*.py."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np
from scipy.signal import butter, lfilter
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mc1", HERE / "mc_test_tolerances.py")
sys.argv = [sys.argv[0], str(HERE / "imu_allan.py")]
mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
ia, N0, B0, K0, q = mc.ia, mc.N0, mc.B0, mc.K0, mc.q
res = {}
t0 = time.time()
for tol in (0.1, 0.03, 0.02):
    ia.WHITE_START_SLOPE_TOL = tol
    r = {}
    fs, n = 200.0, 120000
    b, a = butter(2, 20.0 / (fs / 2))
    rn = []
    for s in range(20):
        y = lfilter(b, a, np.random.default_rng(600 + s).normal(0, N0 * math.sqrt(fs), n))
        taus, adev, ms = ia.overlapping_adev(y, fs)
        rn.append(ia.fit_allan(taus, adev, ms, n, fs, 0.2)["N"] / N0)
    r["lpf20_200Hz_10min_N"] = q(rn)
    for fs in (100.0, 200.0):
        n = int(600 * fs)
        rn = []
        for s in range(20):
            y = np.random.default_rng(100 + s).normal(0.0, N0 * math.sqrt(fs), n)
            taus, adev, ms = ia.overlapping_adev(y, fs)
            rn.append(ia.fit_allan(taus, adev, ms, n, fs, 0.2)["N"] / N0)
        r[f"white_{fs:.0f}Hz_N"] = q(rn)
    for label, K, base in (("kit", K0, 200), ("K5", 5 * K0, 300)):
        fs = 200.0
        n = int(1800 * fs)
        rn, rk, rf = [], [], []
        for s in range(10):
            rng = np.random.default_rng(base + s)
            fl, psd = mc.flicker(rng, n, fs, B0)
            y = rng.normal(0.0, N0 * math.sqrt(fs), n) + fl + np.cumsum(rng.normal(size=n) * K * math.sqrt(1 / fs))
            taus, adev, ms = ia.overlapping_adev(y, fs)
            f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
            rn.append(f["N"] / N0); rf.append(f["fit"]["B"] / B0 if f["fit"] else np.nan)
            rk.append(f["K"] / K if f["K"] else np.nan)
        r[f"{label}_200Hz_30min"] = {"N": q(rn), "B_fit": q(rf), "K": q(rk)}
    res[str(tol)] = r
    print(tol, json.dumps(r, indent=None)[:2000], flush=True)
res["elapsed_s"] = time.time() - t0
(HERE / "mc_white_start.json").write_text(json.dumps(res, indent=2) + "\n")
print(f"done in {res['elapsed_s']:.0f} s")
