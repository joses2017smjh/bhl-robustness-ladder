#!/usr/bin/env python3
"""True-vs-recovered table for imu_allan.py at 100 and 200 Hz (function level, no CSV I/O).

Generates gyro and accelerometer rate noise = white (per-sample std N*sqrt(fs)) +
flicker (one-sided PSD B^2/(pi f): Allan floor 0.6643 B) + ImuNoise-style rate random
walk (bias += N(0,1)*K*sqrt(dt) per sample), runs overlapping_adev + fit_allan from
the script under test, and writes results.md / results.json next to this file.

Truth definitions (fixed before any output is read):
  N      generating white density; sigma_d = N*sqrt(fs) was drawn directly
  B      flicker parameter; "B_min" (min/0.6643) is compared with the ANALYTIC minimum
         of the generated process / 0.6643 (white and RRW lift the floor), "B_fit"
         with the flicker parameter
  K      generating diffusion of the random walk
Usage: python synthetic_table.py [path/to/imu_allan.py]
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
TOOL = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "imu_allan.py"
spec = importlib.util.spec_from_file_location("imu_allan_under_test", TOOL)
ia = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ia)

DEG, G0 = math.pi / 180, 9.80665
MG = 1e-3 * G0
FLICKER = math.sqrt(2 * math.log(2) / math.pi)
# IM10A-like magnitudes (datasheet: 0.028-0.07 deg/s and 0.75-1 mg rms at 100 Hz bandwidth)
REGIME = {
    "gyro": {"N": 0.0035 * DEG, "B": 10.0 / 3600 * DEG, "K": 0.000261 * DEG,
             "units": ("deg/s/rtHz", "deg/h", "deg/s/rt s", "deg/s"), "sc": (1 / DEG, 3600 / DEG, 1 / DEG, 1 / DEG)},
    "accel": {"N": 0.070 * MG, "B": 0.05 * MG, "K": 0.00407 * MG,
              "units": ("mg/rtHz", "mg", "mg/rt s", "mg"), "sc": (1 / MG, 1 / MG, 1 / MG, 1 / MG)},
}


def flicker(rng, n, fs, B):
    M = 1 << int(math.ceil(math.log2(n)))
    f = np.fft.rfftfreq(M, 1 / fs)
    S1 = np.zeros_like(f)
    S1[1:] = B ** 2 / (math.pi * f[1:])
    X = np.sqrt(S1 * fs * M / 2) * (rng.normal(size=f.size) + 1j * rng.normal(size=f.size)) / math.sqrt(2)
    X[0] = 0
    return np.fft.irfft(X, n=M)[:n], (f[1:], S1[1:], fs / M)


def flicker_avar(taus, psd):
    f, S1, df = psd
    out = np.empty(taus.size)
    for i, t in enumerate(taus):
        x = np.pi * f * t
        out[i] = 2.0 * np.sum(S1 * df * np.sin(x) ** 4 / x ** 2)
    return out


def analytic_min(N, K, psd, fs, n, frac=0.2):
    T = n / fs
    taus = np.unique(np.rint(np.logspace(0, math.log10(n // 2), 200)).astype(int)) / fs
    taus = taus[(taus <= frac * T) & (taus >= 10 / fs)]
    av = N ** 2 / taus + K ** 2 * taus / 3 + flicker_avar(taus, psd)
    i = int(np.argmin(av))
    return math.sqrt(av[i]) / FLICKER, float(taus[i])


def one(sensor, fs, hours, seed):
    P = REGIME[sensor]
    rng = np.random.default_rng(seed)
    n = int(round(hours * 3600 * fs))
    sd = P["N"] * math.sqrt(fs)
    white = rng.normal(0.0, sd, n)
    fl, psd = flicker(rng, n, fs, P["B"])
    rw = np.cumsum(rng.normal(size=n) * P["K"] * math.sqrt(1.0 / fs))
    y = white + fl + rw
    t0 = time.time()
    taus, adev, ms = ia.overlapping_adev(y, fs)
    f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
    el = time.time() - t0
    bmin_true, tau_true = analytic_min(P["N"], P["K"], psd, fs, n)
    # white-only control on the same draw: the conversion sigma_d = N*sqrt(fs) in isolation
    tw, aw, mw = ia.overlapping_adev(white, fs)
    fw = ia.fit_allan(tw, aw, mw, n, fs, 0.2)
    return {"sensor": sensor, "fs": fs, "hours": hours, "seed": seed, "n": n, "elapsed_s": el,
            "N_true": P["N"], "N": f["N"], "N_slope": f["N_slope"], "N_method": f["N_method"],
            "sd_true": sd, "sd_drawn": float(white.std()), "sd_tool": f["N"] * math.sqrt(fs),
            "sd_alt_half": f["N"] * math.sqrt(fs / 2),
            "N_white_only": fw["N"], "sd_white_only_tool": fw["N"] * math.sqrt(fs),
            "B_true": P["B"], "B_min": f["B"], "B_min_analytic": bmin_true, "tau_B": f["tau_B_s"],
            "tau_B_analytic": tau_true, "B_upper": f["B_is_upper_bound"],
            "B_fit": f["fit"]["B"] if f["fit"] else None,
            "K_true": P["K"], "K": f["K"], "K_graph": f["K_graph"], "K_visible": f["K_visible"],
            "K_upper": f["K_upper"], "K_fit_any": f["fit"]["K"] if f["fit"] else None}


def main():
    rows = []
    t0 = time.time()
    for fs in (100.0, 200.0):
        for hours in (2.0, 0.5):
            for sensor, seed in (("gyro", 1), ("accel", 2)):
                r = one(sensor, fs, hours, seed + int(fs) + int(hours * 10))
                rows.append(r)
                print(f"{sensor:5s} {fs:5.0f} Hz {hours:3.1f} h  N {r['N'] / r['N_true']:.4f}  "
                      f"B_min/an {r['B_min'] / r['B_min_analytic']:.3f}  B_fit/B {(r['B_fit'] or 0) / r['B_true']:.3f}  "
                      f"K {('%.3f' % (r['K'] / r['K_true'])) if r['K'] else 'n/v'}  "
                      f"Kg {('%.3f' % (r['K_graph'] / r['K_true'])) if r['K_graph'] else 'n/v'}  "
                      f"Kup/K {r['K_upper'] / r['K_true']:.2f}  ({r['elapsed_s']:.1f} s)", flush=True)
    el = time.time() - t0
    (HERE / "results.json").write_text(json.dumps({"tool": str(TOOL), "regime_SI": {
        s: {k: v for k, v in p.items() if k in ("N", "B", "K")} for s, p in REGIME.items()},
        "rows": rows, "elapsed_s": el}, indent=2, default=float) + "\n")

    def fmt(x, nd=4):
        return "n/v" if x is None else f"{x:.{nd}g}"

    L = ["# imu_allan.py synthetic recovery: true vs recovered at 100 and 200 Hz", "",
         f"Tool: `{TOOL}`  (function level: `overlapping_adev` + `fit_allan`, one seed per row; "
         f"total {el:.0f} s, numpy {np.__version__}).", "",
         "Process per channel: white (per-sample std drawn as N*sqrt(fs)) + flicker (Allan floor 0.6643 B) "
         "+ ImuNoise-style rate random walk K. Gyro: N 0.0035 deg/s/rtHz, B 10 deg/h, K 0.000261 deg/s/rt s. "
         "Accel: N 0.070 mg/rtHz, B 0.05 mg, K 0.00407 mg/rt s.", "",
         "B truth for the min method = analytic minimum of the generated process / 0.6643 (white and RRW lift the "
         "floor above 0.6643 B); B_fit is compared with the flicker parameter B. K = +1/2 line at tau = 3 s "
         "(least-squares level; K_graph = floor-subtracted graphical read). n/v = not visible "
         "(only the bound K_upper is reported).", "",
         "| fs | T | sensor | N true | N rec (ratio) | sigma_d true = N*sqrt(fs) | N_rec*sqrt(fs) | N_rec*sqrt(fs/2) "
         "| B analytic-min/0.664 | B_min rec (ratio) | tau_B rec / analytic (s) | B param | B_fit (ratio) "
         "| K true | K rec (ratio) | K_graph (ratio) | K_upper |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        u = REGIME[r["sensor"]]["units"]
        sc = REGIME[r["sensor"]]["sc"]
        kr = f"{fmt(r['K'] * sc[2])} ({r['K'] / r['K_true']:.2f})" if r["K"] else "n/v"
        kg = f"{fmt(r['K_graph'] * sc[2])} ({r['K_graph'] / r['K_true']:.2f})" if r["K_graph"] else "n/v"
        L.append(
            f"| {r['fs']:.0f} Hz | {r['hours']:g} h | {r['sensor']} | {fmt(r['N_true'] * sc[0])} {u[0]} "
            f"| {fmt(r['N'] * sc[0])} ({r['N'] / r['N_true']:.4f}) | {fmt(r['sd_true'] * sc[3])} {u[3]} "
            f"| {fmt(r['sd_tool'] * sc[3])} ({r['sd_tool'] / r['sd_true']:.4f}) "
            f"| {fmt(r['sd_alt_half'] * sc[3])} ({r['sd_alt_half'] / r['sd_true']:.3f}) "
            f"| {fmt(r['B_min_analytic'] * sc[1])} {u[1]} | {fmt(r['B_min'] * sc[1])} ({r['B_min'] / r['B_min_analytic']:.3f})"
            f"{' upper bound' if r['B_upper'] else ''} | {r['tau_B']:.0f} / {r['tau_B_analytic']:.0f} "
            f"| {fmt(r['B_true'] * sc[1])} | {fmt((r['B_fit'] or 0) * sc[1])} ({(r['B_fit'] or 0) / r['B_true']:.3f}) "
            f"| {fmt(r['K_true'] * sc[2])} {u[2]} | {kr} | {kg} | {fmt(r['K_upper'] * sc[2])} ({r['K_upper'] / r['K_true']:.2f}) |")
    L += ["", "White-only control (same white draw, nothing else): N_rec*sqrt(fs) / sigma_d drawn:", ""]
    for r in rows:
        L.append(f"- {r['fs']:.0f} Hz {r['hours']:g} h {r['sensor']}: drawn std {r['sd_drawn'] / r['sd_true']:.4f} x "
                 f"sigma_d; N_rec*sqrt(fs) = {r['sd_white_only_tool'] / r['sd_true']:.4f} x sigma_d "
                 f"(N*sqrt(fs/2) would be {r['sd_white_only_tool'] / math.sqrt(2) / r['sd_true']:.3f})")
    (HERE / "results.md").write_text("\n".join(L) + "\n")
    print(f"wrote {HERE / 'results.md'} in {el:.0f} s")


if __name__ == "__main__":
    main()
