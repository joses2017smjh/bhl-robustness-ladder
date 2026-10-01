"""Share of above-threshold steps that are a genuine lift (lowest cube corner >= 2 cm above the deck top 0.43,
i.e. above every support) vs lowest corner at a support-top height (+-1 cm of 0.41 plinth / 0.43 deck)."""
import json, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
SRC = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/repo-gpu-20260923/stand3_replay_2026-09-28/21470828/"
H = 0.14
def R_xyzw(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y*y + z*z); R[..., 0, 1] = 2 * (x*y - z*w); R[..., 0, 2] = 2 * (x*z + y*w)
    R[..., 1, 0] = 2 * (x*y + z*w); R[..., 1, 1] = 1 - 2 * (x*x + z*z); R[..., 1, 2] = 2 * (y*z - x*w)
    R[..., 2, 0] = 2 * (x*z - y*w); R[..., 2, 1] = 2 * (y*z + x*w); R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R
out = {}
for s in ("s0", "s1"):
    z = np.load(SRC + f"{s}.trace.npz")
    v = z["valid"]; p = z["cube_pos"].astype(float); R = R_xyzw(z["cube_quat_raw"].astype(float))
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * H
    lowz = (p[..., None, :] + np.einsum("teij,kj->teki", R, signs))[..., 2].min(-1)
    r = {}
    for thr in (0.59, 0.61):
        m = v & (p[..., 2] > thr)
        lz = lowz[m]
        r[f"z>{thr}"] = {"steps": int(m.sum()),
            "lowest_corner_within_1cm_of_0.41_or_0.43": round(float(((np.abs(lz - 0.41) < 0.01) | (np.abs(lz - 0.43) < 0.01)).mean()), 4),
            "lowest_corner_below_0.40(hanging_off_an_edge)": round(float((lz < 0.40).mean()), 4),
            "lowest_corner_ge_0.45(clear_of_every_support_by_2cm)": round(float((lz >= 0.45).mean()), 4),
            "cube_speed_lt_0.05_share": round(float((z['cube_speed'][m] < 0.05).mean()), 4)}
    # a flat cube must have its lowest corner >= thr - 0.14 to pass: what share of valid steps are a flat lift above 0.61?
    tilt = z["cube_face_tilt_deg"]
    r["valid_steps_flat(<8deg)_and_z>0.61"] = int((v & (tilt < 8) & (p[..., 2] > 0.61)).sum())
    r["valid_steps_lowest_corner_ge_0.45"] = round(float((lowz[v] >= 0.45).mean()), 4)
    out[s] = r
    print(s, json.dumps(r, indent=1))
json.dump(out, open("stand3_lifted_share.json", "w"), indent=1)
