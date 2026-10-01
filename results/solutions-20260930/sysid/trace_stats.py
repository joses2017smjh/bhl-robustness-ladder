"""Read-only statistics of the existing NavGym v4 physics-transfer traces (no simulation).

Trace rows are every 5th policy step (0.2 s): {t, xy, yaw, state, cmd:[vx,vy,wz] (POST-brake), ...}.
The brake (team_sensors.brake_command) scales cmd[:2] only; wz passes through unchanged.
"""
import json, math, sys
from pathlib import Path
import numpy as np

ROOT = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/navgym-v4-transfer-20260930")
out = {}
for arm in ("armV4-s5", "armV4-s6", "astar"):
    rows = []
    for f in sorted((ROOT / arm).glob("seed*.json")):
        d = json.loads(f.read_text())
        tr = [r for r in d["trace"] if r["state"] not in ("settle",)]
        t = np.array([r["t"] for r in tr]); vx = np.array([r["cmd"][0] for r in tr]); wz = np.array([r["cmd"][2] for r in tr])
        yaw = np.unwrap(np.array([r["yaw"] for r in tr])); xy = np.array([r["xy"] for r in tr])
        dt = np.diff(t)
        # measured yaw rate and ground speed between 0.2 s samples
        yr = np.diff(yaw) / dt
        sp = np.linalg.norm(np.diff(xy, axis=0), axis=1) / dt
        # forward speed in the heading frame (mid-sample heading)
        hd = 0.5 * (yaw[1:] + yaw[:-1])
        fwd = (np.diff(xy[:, 0]) * np.cos(hd) + np.diff(xy[:, 1]) * np.sin(hd)) / dt
        sat = np.mean(np.abs(wz) >= 0.999)
        s = np.sign(wz); nz = s != 0
        flips = np.sum((s[1:] * s[:-1]) < 0)
        dur = t[-1] - t[0] if len(t) > 1 else 1.0
        # commanded vs measured yaw rate: per interval the cmd at the interval start (sub-sampled: indicative only)
        cmd_mean = wz[:-1]
        rows.append({"seed": d["seed"], "outcome": d["outcome_class"], "elapsed_s": d["elapsed_s"],
                     "sensor_stats": d["sensor_stats"],
                     "braked_frac_of_steps": round(d["sensor_stats"]["braked_robot_steps"] / max(1, d["sensor_stats"]["sampled_robot_steps"]), 3),
                     "wz_sat_frac": round(float(sat), 3), "wz_flip_per_s_at_5Hz_sampling": round(float(flips / dur), 2),
                     "vx_cmd_mean": round(float(vx.mean()), 3),
                     "vx_cmd_frac_eq_0.35": round(float(np.mean(vx >= 0.3495)), 3),
                     "vx_cmd_frac_lt_0.30": round(float(np.mean(vx < 0.30)), 3),
                     "vx_cmd_frac_zero": round(float(np.mean(vx < 1e-3)), 3),
                     "meas_speed_mean": round(float(sp.mean()), 3), "meas_fwd_mean": round(float(fwd.mean()), 3),
                     "meas_abs_yawrate_mean": round(float(np.abs(yr).mean()), 3),
                     "cmd_abs_wz_mean": round(float(np.abs(wz).mean()), 3),
                     "net_yaw_rate_mean": round(float((yaw[-1] - yaw[0]) / dur), 3),
                     "path_m": d["path_length_m"]})
    out[arm] = rows

def agg(rows, k):
    v = np.array([r[k] for r in rows], float)
    return round(float(v.mean()), 3)

for arm, rows in out.items():
    print(f"== {arm}")
    for r in rows:
        print({k: v for k, v in r.items() if k != "sensor_stats"})
    keys = ["braked_frac_of_steps", "wz_sat_frac", "wz_flip_per_s_at_5Hz_sampling", "vx_cmd_mean", "vx_cmd_frac_eq_0.35", "vx_cmd_frac_lt_0.30",
            "vx_cmd_frac_zero", "meas_speed_mean", "meas_fwd_mean", "meas_abs_yawrate_mean", "cmd_abs_wz_mean"]
    print("MEAN", {k: agg(rows, k) for k in keys})
Path(sys.argv[1] if len(sys.argv) > 1 else "trace_stats.json").write_text(json.dumps(out, indent=1))
