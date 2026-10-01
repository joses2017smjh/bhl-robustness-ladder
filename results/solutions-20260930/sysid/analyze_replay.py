"""Open-loop replay of the v4 actors' own gym command sequences on the physics biped (flat floor): how well does each
model predict the physics heading change over short windows?  Models: gym nominal (1.17, 0.25 s, 1 step), the best
gym parameters INSIDE the randomization box, the best unconstrained (K, tau, latency), and PhysDynamics."""
import json, math, sys
from pathlib import Path
import numpy as np
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src")
from sysid_fit import gym_yaw_fast, DT
from sysid_lib import fwd_speed
from analyze_steps import load
from sysid_env import PhysDynamics

D = Path(__file__).resolve().parent / "data"


def win_err(yp, yh, n):
    """RMSE of heading change over non-overlapping windows of n steps."""
    k = (len(yp) // n) * n
    dp = yp[n:k:n] - yp[0:k - n:n]; dh = yh[n:k:n] - yh[0:k - n:n]
    return float(np.sqrt(np.mean((dp - dh) ** 2)))


def phys_model_yaw(v, w, seed=0):
    m = PhysDynamics(np.random.default_rng(seed), wobble=False, drift=False)
    m.reset()
    out = np.zeros(len(w)); vv = np.zeros(len(w)); yaw = 0.0
    for k in range(len(w)):
        a, b = m.step(v[k], w[k]); yaw += b * DT; out[k] = yaw; vv[k] = a
    return out, vv


res = []
for f in sorted(D.glob("phys_replay_*.npz")):
    r = load(f); m = r["meta"]
    t = r["t"] - DT
    sel = (t >= m["t_on"] - 1e-9) & (t < m["t_off"] - 1e-9)
    v = r["cmd"][sel, 0]; w = r["cmd"][sel, 2]
    i0 = np.flatnonzero(sel)[0]
    yp = r["yaw"][sel] - r["yaw"][i0 - 1]
    rows = {"arm": m["arm"], "maze": m["maze"], "steps": int(sel.sum()), "phys_mean_rate": float((yp[-1]) / (sel.sum() * DT))}
    models = {}
    yg, _, _ = gym_yaw_fast(w, v, 1.17, 0.25, 1); models["gym_nominal"] = yg
    best_in, best_all = None, None
    for lat in range(0, 5):
        for tau in np.arange(0.04, 0.61, 0.01):
            yu, _, _ = gym_yaw_fast(w, v, 1.0, tau, lat)
            for n_ in (25,):
                pass
            # least-squares K on 0.5 s increments
            n = 12
            k = (len(yp) // n) * n
            dp = yp[n:k:n] - yp[0:k - n:n]; du = yu[n:k:n] - yu[0:k - n:n]
            K = float(du @ dp / (du @ du))
            e = win_err(yp, K * yu, n)
            if best_all is None or e < best_all[0]:
                best_all = (e, K, tau, lat)
            Kc = min(max(K, 0.9), 1.4)
            if 0.15 - 1e-9 <= tau <= 0.45 + 1e-9 and lat <= 2:
                e2 = win_err(yp, Kc * yu, n)
                if best_in is None or e2 < best_in[0]:
                    best_in = (e2, Kc, tau, lat)
    models["gym_best_in_range"] = gym_yaw_fast(w, v, best_in[1], best_in[2], best_in[3])[0]
    models["best_unconstrained"] = gym_yaw_fast(w, v, best_all[1], best_all[2], best_all[3])[0]
    models["phys_identified"], vv = phys_model_yaw(v, w)
    rows["best_in_range_params"] = {"K": best_in[1], "tau": round(best_in[2], 3), "lat": best_in[3]}
    rows["best_unconstrained_params"] = {"K": round(best_all[1], 3), "tau": round(best_all[2], 3), "lat": best_all[3]}
    for name, yh in models.items():
        rows[name] = {"rmse_0.2s_rad": round(win_err(yp, yh, 5), 4), "rmse_0.48s_rad": round(win_err(yp, yh, 12), 4),
                      "rmse_2s_rad": round(win_err(yp, yh, 50), 4)}
    # per-step heading change magnitude (the capture-pose smear of maze_explore's biped path is one step of rotation)
    inc = np.diff(np.concatenate([[0.0], yp]))
    rows["per_step_abs_dyaw_deg"] = {"median": round(math.degrees(float(np.median(np.abs(inc)))), 2),
                                    "p90": round(math.degrees(float(np.percentile(np.abs(inc), 90))), 2),
                                    "max": round(math.degrees(float(np.max(np.abs(inc)))), 2)}
    inc3 = yp[3:] - yp[:-3]
    rows["abs_dyaw_over_3_steps_deg"] = {"median": round(math.degrees(float(np.median(np.abs(inc3)))), 2),
                                         "p90": round(math.degrees(float(np.percentile(np.abs(inc3), 90))), 2)}
    vf = fwd_speed(r)[sel]
    rows["speed_phys_mean"] = float(vf.mean()); rows["speed_cmd_mean"] = float(v.mean()); rows["speed_model_mean"] = float(vv.mean())
    res.append(rows)
    print(json.dumps(rows))
(D / "replay_summary.json").write_text(json.dumps(res, indent=1))
