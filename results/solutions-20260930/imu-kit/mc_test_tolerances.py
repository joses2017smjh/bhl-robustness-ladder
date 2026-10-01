#!/usr/bin/env python3
"""Monte Carlo scatter of N, B, K for the SHORT configurations used by tests/test_imu_allan.py.

Run once (offline) to set the test tolerances from the observed scatter before pinning one
seed per test. Function level (overlapping_adev + fit_allan), no file I/O.
Usage: python mc_test_tolerances.py <path/to/imu_allan.py> [seeds]
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import importlib.util
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("ia", sys.argv[1])
ia = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ia)
SEEDS = int(sys.argv[2]) if len(sys.argv) > 2 else 20

DEG = math.pi / 180
FLICKER = math.sqrt(2 * math.log(2) / math.pi)
N0, B0, K0 = 0.0035 * DEG, 10.0 / 3600 * DEG, 0.000261 * DEG     # kit gyro regime (rad/s units)


def flicker(rng, n, fs, B):
    M = 1 << int(math.ceil(math.log2(n)))
    f = np.fft.rfftfreq(M, 1 / fs)
    S1 = np.zeros_like(f)
    S1[1:] = B ** 2 / (math.pi * f[1:])
    X = np.sqrt(S1 * fs * M / 2) * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / math.sqrt(2)
    X[0] = 0
    return np.fft.irfft(X, n=M)[:n], (f[1:], S1[1:], fs / M)


def analytic_min(N, K, psd, fs, n, frac=0.2):
    f, S1, df = psd
    T = n / fs
    taus = np.unique(np.rint(np.logspace(0, math.log10(n // 2), 120)).astype(int)) / fs
    taus = taus[(taus <= frac * T) & (taus >= 10 / fs)]
    av = N ** 2 / taus + K ** 2 * taus / 3
    for i, t in enumerate(taus):
        x = np.pi * f * t
        av[i] += 2.0 * np.sum(S1 * df * np.sin(x) ** 4 / x ** 2)
    i = int(np.argmin(av))
    return math.sqrt(av[i]) / FLICKER


def q(v):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return None
    return {"min": float(v.min()), "p16": float(np.percentile(v, 16)), "p50": float(np.median(v)),
            "p84": float(np.percentile(v, 84)), "max": float(v.max()), "n": int(v.size)}


def main():
    res = {}
    t0 = time.time()
    # 1. white-noise convention: 10 min, per-sample std drawn as N*sqrt(fs)
    for fs in (100.0, 200.0):
        n = int(600 * fs)
        r_sd, r_N, r_half = [], [], []
        for s in range(SEEDS):
            rng = np.random.default_rng(100 + s)
            y = rng.normal(0.0, N0 * math.sqrt(fs), n)
            taus, adev, ms = ia.overlapping_adev(y, fs)
            f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
            r_sd.append(f["N"] * math.sqrt(fs) / y.std())
            r_N.append(f["N"] / N0)
            r_half.append(f["N"] * math.sqrt(fs / 2) / y.std())
        res[f"white_{fs:.0f}Hz_10min"] = {"Nsqrtfs_over_drawn_sd": q(r_sd), "N_over_true": q(r_N),
                                           "Nsqrt(fs/2)_over_drawn_sd": q(r_half)}
        print(f"white {fs:.0f} Hz: {res[f'white_{fs:.0f}Hz_10min']}", flush=True)
    # 2. kit regime (white + flicker + RRW), 30 min: B_min, B_fit, K bound
    for fs in (100.0, 200.0):
        n = int(1800 * fs)
        rb, rf, up, vis, cover, rk = [], [], [], [], [], []
        bmin = None
        for s in range(SEEDS):
            rng = np.random.default_rng(200 + s)
            fl, psd = flicker(rng, n, fs, B0)
            y = rng.normal(0.0, N0 * math.sqrt(fs), n) + fl + np.cumsum(rng.normal(size=n) * K0 * math.sqrt(1 / fs))
            if bmin is None:
                bmin = analytic_min(N0, K0, psd, fs, n)     # same frequency grid for every seed
            taus, adev, ms = ia.overlapping_adev(y, fs)
            f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
            rb.append(f["B"] / bmin)
            rf.append(f["fit"]["B"] / B0 if f["fit"] else np.nan)
            up.append(f["B_is_upper_bound"])
            vis.append(f["K_visible"])
            cover.append(f["K_upper"] >= K0)
            if f["K_visible"]:
                rk.append(f["K"] / K0)
        res[f"kit_{fs:.0f}Hz_30min"] = {"B_min_over_analytic": q(rb), "B_fit_over_B": q(rf),
                                         "B_upper_bound_frac": float(np.mean(up)), "K_visible_frac": float(np.mean(vis)),
                                         "K_upper_covers_frac": float(np.mean(cover)), "K_over_true_when_visible": q(rk),
                                         "analytic_B_min_over_B": bmin / B0}
        print(f"kit {fs:.0f} Hz: {res[f'kit_{fs:.0f}Hz_30min']}", flush=True)
    # 3. K test regime: K x5 (RRW crosses the floor at a few seconds), 30 min
    for fs in (100.0, 200.0):
        n = int(1800 * fs)
        K5 = 5 * K0
        rk, rg, vis = [], [], []
        for s in range(SEEDS):
            rng = np.random.default_rng(300 + s)
            fl, _ = flicker(rng, n, fs, B0)
            y = rng.normal(0.0, N0 * math.sqrt(fs), n) + fl + np.cumsum(rng.normal(size=n) * K5 * math.sqrt(1 / fs))
            taus, adev, ms = ia.overlapping_adev(y, fs)
            f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
            vis.append(f["K_visible"])
            if f["K_visible"]:
                rk.append(f["K"] / K5)
                rg.append(f["K_graph"] / K5)
        res[f"K5_{fs:.0f}Hz_30min"] = {"K_visible_frac": float(np.mean(vis)), "K_over_true": q(rk), "K_graph_over_true": q(rg)}
        print(f"K x5 {fs:.0f} Hz: {res[f'K5_{fs:.0f}Hz_30min']}", flush=True)
    # 4. ImuNoise-style (no flicker): white gyro_std per sample + bias walk K, 10 min at 200 Hz
    fs, n = 200.0, 120000
    gstd, Kw = 0.0495 * DEG, 1e-4
    rk, rN, vis = [], [], []
    for s in range(SEEDS):
        rng = np.random.default_rng(400 + s)
        y = rng.normal(0.0, gstd, n) + np.cumsum(rng.normal(size=n) * Kw * math.sqrt(1 / fs))
        taus, adev, ms = ia.overlapping_adev(y, fs)
        f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
        rN.append(f["N"] * math.sqrt(fs) / gstd)
        vis.append(f["K_visible"])
        if f["K_visible"]:
            rk.append(f["K"] / Kw)
    res["imunoise_style_200Hz_10min"] = {"gyro_std_over_true": q(rN), "K_visible_frac": float(np.mean(vis)),
                                         "K_over_true": q(rk)}
    print(f"ImuNoise-style: {res['imunoise_style_200Hz_10min']}", flush=True)
    res["seeds"] = SEEDS
    res["elapsed_s"] = time.time() - t0
    (HERE / "mc_test_tolerances.json").write_text(json.dumps(res, indent=2) + "\n")
    print(f"done in {res['elapsed_s']:.0f} s")


if __name__ == "__main__":
    main()
