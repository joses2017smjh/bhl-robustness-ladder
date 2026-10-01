"""Yaw-rate step responses: FOPDT fits on the integrated yaw (baseline slope removed), steady-state gain, release,
cross-coupling (forward speed during the turn), base displacement in turn-in-place; and the gym's own model fitted
by the SAME procedure so the parameters compare in one convention."""
import json, math
from pathlib import Path
import numpy as np
from sysid_fit import fit_fopdt_integrated, gym_yaw_fast, DT
from sysid_lib import fwd_speed, lat_speed

D = Path(__file__).resolve().parent / "data"


def load(f):
    z = np.load(f, allow_pickle=True)
    r = {k: z[k] for k in z.files}
    r["meta"] = json.loads(str(z["meta"]))
    return r


def analyse_run(r, fit_window=3.0):
    m = r["meta"]; t = r["t"]; yaw = r["yaw"]; u = m["wz"]
    t_on, t_off = m["t_on"], m["t_off"]
    # sample k is the state at t[k] (= end of step k); the state AT t_on is the sample with t == t_on
    i_on = int(np.argmin(np.abs(t - t_on))); i_off = int(np.argmin(np.abs(t - t_off)))
    pre = (t > t_on - 2.0) & (t <= t_on + 1e-9)
    b = np.polyfit(t[pre], yaw[pre], 1)[0]                       # baseline yaw rate (drift) before the step
    win = (t > t_on + 1e-9) & (t <= t_on + fit_window + 1e-9)
    tt = t[win] - t_on
    y = yaw[win] - yaw[i_on] - b * tt
    fit = fit_fopdt_integrated(tt, y, u)
    ss = (t > t_on + 2.0) & (t <= t_off + 1e-9)
    w_ss = np.polyfit(t[ss], yaw[ss], 1)[0]
    K_ss = (w_ss - b) / u
    # release: deviation from continued turning after t_off
    rel = (t > t_off + 1e-9) & (t <= t_off + 2.0 + 1e-9)
    tr = t[rel] - t_off
    yr = yaw[rel] - yaw[i_off] - w_ss * tr
    fit_rel = fit_fopdt_integrated(tr, yr, -(w_ss - b)) if len(tr) > 5 else None     # "K" here = fraction of w_ss removed
    post = (t > t_off + 1.5) & (t <= t[-1])
    w_post = np.polyfit(t[post], yaw[post], 1)[0] if post.sum() > 5 else float("nan")
    v = fwd_speed(r)
    v_pre = float(np.mean(v[pre])); v_turn = float(np.mean(v[ss]))
    # displacement and turn-circle radius while turning (vx = 0)
    xy = r["xy"][ss]
    disp = float(np.linalg.norm(xy[-1] - xy[0]))
    cx, cy = np.mean(xy, axis=0)
    rad = float(np.mean(np.linalg.norm(xy - [cx, cy], axis=1)))
    return {"vx": m["vx"], "wz": u, "seed": m["seed"], "baseline_rate": b, "K_fit": fit["K"], "tau": fit["tau"], "L": fit["L"],
            "rise63": fit["rise63"], "rmse_rad": fit["rmse"], "K_ss": K_ss, "w_ss": w_ss,
            "release_frac": fit_rel["K"] if fit_rel else None, "release_tau": fit_rel["tau"] if fit_rel else None,
            "release_L": fit_rel["L"] if fit_rel else None, "w_after_release": w_post,
            "v_pre": v_pre, "v_turn": v_turn, "v_ratio": v_turn / v_pre if v_pre > 0.05 else None,
            "turn_disp_m": disp, "turn_circle_r_m": rad, "max_tilt": float(r["tilt"].max()), "fell": float(r["fell"])}


def gym_apparent(w_gain, tau, latency, u=1.0, fit_window=3.0):
    """Fit the same continuous FOPDT to the gym's discrete model (noise-free), same sample convention."""
    n = int(round((fit_window + 1.0) / DT))
    wc = np.full(n, u)
    yaw, _, _ = gym_yaw_fast(wc, np.zeros(n), w_gain, tau, latency)
    tt = (np.arange(n) + 1) * DT
    win = tt <= fit_window + 1e-9
    f = fit_fopdt_integrated(tt[win], yaw[win], u)
    return f


if __name__ == "__main__":
    rows = [analyse_run(load(f)) for f in sorted(D.glob("phys_steps_*.npz"))]
    (D / "steps_fits.json").write_text(json.dumps(rows, indent=1))
    import collections
    g = collections.defaultdict(list)
    for r in rows:
        g[(r["vx"], r["wz"])].append(r)
    keys = ["K_fit", "K_ss", "tau", "L", "rise63", "rmse_rad", "release_frac", "release_tau", "release_L", "v_pre", "v_turn", "v_ratio",
            "turn_disp_m", "turn_circle_r_m", "baseline_rate", "w_after_release", "max_tilt"]
    summ = {}
    print(f"{'vx':>5} {'wz':>5} | " + " ".join(f"{k:>11}" for k in keys))
    for (vx, wz), rr in sorted(g.items()):
        s = {}
        for k in keys:
            vals = np.array([x[k] for x in rr if x[k] is not None], float)
            s[k] = [round(float(vals.mean()), 4), round(float(vals.min()), 4), round(float(vals.max()), 4)] if len(vals) else None
        summ[f"vx{vx}_wz{wz}"] = s
        print(f"{vx:5.2f} {wz:+5.1f} | " + " ".join(f"{s[k][0]:11.3f}" if s[k] else f"{'-':>11}" for k in keys))
    # the gym's range, fitted by the same procedure
    gym = {}
    for wg in (0.9, 1.17, 1.4):
        for tau in (0.15, 0.25, 0.45):
            for lat in (0, 1, 2):
                f = gym_apparent(wg, tau, lat)
                gym[f"K{wg}_tau{tau}_lat{lat}"] = {k: round(v, 4) for k, v in f.items()}
    print("gym apparent (same fit):")
    for k in ("K1.17_tau0.15_lat0", "K1.17_tau0.25_lat1", "K1.17_tau0.45_lat2", "K1.17_tau0.15_lat2", "K1.17_tau0.45_lat0"):
        print(" ", k, gym[k])
    (D / "steps_summary.json").write_text(json.dumps({"physics": summ, "gym_apparent": gym}, indent=1))
