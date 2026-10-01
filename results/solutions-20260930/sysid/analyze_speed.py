"""Forward-speed steps (gain, FOPDT on displacement along the heading), stop response, heading drift while walking
straight, and the periodic yaw wobble (stride frequency, peak-to-peak) -- physics runs data/phys_speed_*.npz."""
import json, math
from pathlib import Path
import numpy as np
from sysid_fit import fit_fopdt_integrated, DT
from sysid_lib import fwd_speed, lat_speed
from analyze_steps import load

D = Path(__file__).resolve().parent / "data"


def along_heading_disp(r, i0):
    """Path length projected on the instantaneous heading, integrated from sample i0 (world velocity . heading)."""
    v = fwd_speed(r)
    return np.cumsum(v[i0:]) * DT


def analyse(r):
    m = r["meta"]; t = r["t"]; u = m["vx"]; t_on, t_off = m["t_on"], m["t_off"]
    i_on = int(np.argmin(np.abs(t - t_on)))
    win = (t > t_on + 1e-9) & (t <= t_on + 3.0 + 1e-9)
    tt = t[win] - t_on
    v = fwd_speed(r)
    s = np.cumsum(v[win]) * DT
    fit = fit_fopdt_integrated(tt, s, u)
    ss = (t > t_on + 3.0) & (t <= t_off + 1e-9)
    v_ss = float(np.mean(v[ss]))
    # stop response: speed decay after t_off, FOPDT on the displacement deficit relative to continued walking
    rel = (t > t_off + 1e-9) & (t <= t_off + 2.0 + 1e-9)
    tr = t[rel] - t_off
    sr = np.cumsum(v[rel]) * DT - v_ss * tr
    fit_stop = fit_fopdt_integrated(tr, sr, -v_ss)
    # heading drift while walking straight (after 3 s of walking), rad/s and deg per metre
    yaw = r["yaw"]
    p = np.polyfit(t[ss], yaw[ss], 1)
    dist = float(np.sum(np.abs(v[ss])) * DT)
    dyaw = float(yaw[ss][-1] - yaw[ss][0])
    # stride wobble: yaw minus its linear trend, spectrum of the yaw rate
    res = yaw[ss] - np.polyval(p, t[ss])
    w = np.gradient(yaw, t)[ss]
    w = w - w.mean()
    F = np.abs(np.fft.rfft(w * np.hanning(len(w))))
    fr = np.fft.rfftfreq(len(w), DT)
    f_peak = float(fr[1:][np.argmax(F[1:])])
    lat = lat_speed(r)[ss]
    return {"vx": u, "seed": m["seed"], "K_v": fit["K"], "tau_v": fit["tau"], "L_v": fit["L"], "rise63_v": fit["rise63"], "rmse_m": fit["rmse"],
            "v_ss": v_ss, "gain_ss": v_ss / u, "stop_frac": fit_stop["K"], "stop_tau": fit_stop["tau"], "stop_L": fit_stop["L"],
            "drift_rad_s": float(p[0]), "drift_deg_per_m": math.degrees(dyaw) / dist if dist > 0 else None, "drift_deg_total": math.degrees(dyaw),
            "dist_m": dist, "wobble_yaw_pp_deg": math.degrees(float(res.max() - res.min())), "wobble_yaw_rms_deg": math.degrees(float(res.std())),
            "yawrate_rms": float(w.std()), "stride_peak_hz": f_peak, "lat_speed_rms": float(lat.std()), "max_tilt": float(r["tilt"].max())}


if __name__ == "__main__":
    rows = [analyse(load(f)) for f in sorted(D.glob("phys_speed_*.npz"))]
    (D / "speed_fits.json").write_text(json.dumps(rows, indent=1))
    keys = ["K_v", "gain_ss", "v_ss", "tau_v", "L_v", "rise63_v", "stop_frac", "stop_tau", "stop_L", "drift_rad_s", "drift_deg_per_m",
            "drift_deg_total", "wobble_yaw_pp_deg", "wobble_yaw_rms_deg", "yawrate_rms", "stride_peak_hz", "lat_speed_rms", "max_tilt"]
    out = {}
    for vx in sorted(set(r["vx"] for r in rows)):
        rr = [r for r in rows if r["vx"] == vx]
        out[str(vx)] = {k: [round(float(np.mean([x[k] for x in rr])), 4), round(float(np.min([x[k] for x in rr])), 4),
                            round(float(np.max([x[k] for x in rr])), 4)] for k in keys}
        print(f"vx={vx}: " + ", ".join(f"{k}={out[str(vx)][k][0]:.3f} [{out[str(vx)][k][1]:.3f},{out[str(vx)][k][2]:.3f}]" for k in keys))
    (D / "speed_summary.json").write_text(json.dumps(out, indent=1))
