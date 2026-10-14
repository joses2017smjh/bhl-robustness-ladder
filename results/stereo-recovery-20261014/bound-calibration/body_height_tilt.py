"""Body height and tilt change over windows on v1 bounded evaluator truth (prior data only)."""
import json, sys, glob, os
import numpy as np
V = sys.argv[1]
dz_all, tilt_all = {}, {}
for ep in sorted(glob.glob(os.path.join(V, 'collector-inputs/*/bounded_latest-*'))):
    e = json.load(open(os.path.join(ep, 'episode.json')))
    T = np.load(os.path.join(ep, 'evaluator_truth.npz'))['T_W_I']
    body = T @ np.linalg.inv(np.asarray(e['T_body_imu']))
    z = body[:, 2, 3]; up = body[:, :3, 2]
    for W in (5, 25, 75, 125):
        a = np.arange(260, len(z) - W, 2)
        dz_all.setdefault(W, []).append(np.abs(z[a + W] - z[a]))
        tilt_all.setdefault(W, []).append(np.arccos(np.clip(np.einsum('ij,ij->i', up[a], up[a + W]), -1, 1)))
for W in sorted(dz_all):
    dz = np.concatenate(dz_all[W]); t = np.concatenate(tilt_all[W])
    print('T=%.2fs |dz| p99=%.4f max=%.4f m | tilt change p99=%.2f max=%.2f deg' % (W * .04, np.quantile(dz, .99), dz.max(), np.degrees(np.quantile(t, .99)), np.degrees(t.max())))
