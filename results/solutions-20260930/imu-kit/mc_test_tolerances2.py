#!/usr/bin/env python3
"""Second MC pass: B_min vs analytic minimum and N in the K x5 regime; N in the kit regime (30 min)."""
import os
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import importlib.util, json, math, sys, time
from pathlib import Path
import numpy as np
HERE = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("mc1", HERE / "mc_test_tolerances.py")
sys.argv = [sys.argv[0], sys.argv[1]]
mc = importlib.util.module_from_spec(spec); spec.loader.exec_module(mc)
ia, N0, B0, K0 = mc.ia, mc.N0, mc.B0, mc.K0
res = {}
t0 = time.time()
for label, K, base in (("K5", 5 * K0, 300), ("kit", K0, 200)):
    for fs in (100.0, 200.0):
        n = int(1800 * fs)
        rb, rn, up = [], [], []
        bmin = None
        for s in range(20):
            rng = np.random.default_rng(base + s)        # same streams as mc_test_tolerances.py
            fl, psd = mc.flicker(rng, n, fs, B0)
            y = rng.normal(0.0, N0 * math.sqrt(fs), n) + fl + np.cumsum(rng.normal(size=n) * K * math.sqrt(1 / fs))
            if bmin is None:
                bmin = mc.analytic_min(N0, K, psd, fs, n)
            taus, adev, ms = ia.overlapping_adev(y, fs)
            f = ia.fit_allan(taus, adev, ms, n, fs, 0.2)
            rb.append(f["B"] / bmin); rn.append(f["N"] / N0); up.append(f["B_is_upper_bound"])
        res[f"{label}_{fs:.0f}Hz_30min"] = {"B_min_over_analytic": mc.q(rb), "N_over_true": mc.q(rn),
                                             "B_upper_bound_frac": float(np.mean(up)), "analytic_B_min_over_B": bmin / B0}
        print(label, fs, res[f"{label}_{fs:.0f}Hz_30min"], flush=True)
res["elapsed_s"] = time.time() - t0
(HERE / "mc_test_tolerances2.json").write_text(json.dumps(res, indent=2) + "\n")
print(f"done in {res['elapsed_s']:.0f} s")
