import importlib.util, math, time, sys
import numpy as np
spec = importlib.util.spec_from_file_location("ia", sys.argv[1])
ia = importlib.util.module_from_spec(spec); spec.loader.exec_module(ia)
rng = np.random.default_rng(0)
for fs, hours in ((200.0, 2.0), (200.0, 1/6), (100.0, 0.5)):
    n = int(hours*3600*fs)
    y = rng.normal(0, 1e-3, n)
    t0 = time.time(); taus, adev, ms = ia.overlapping_adev(y, fs); t1 = time.time()
    f = ia.fit_allan(taus, adev, ms, n, fs, 0.2); t2 = time.time()
    print(f"fs {fs} h {hours:.3f} n {n}: adev {t1-t0:.2f} s ({taus.size} taus), fit {t2-t1:.2f} s")
