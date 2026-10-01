"""Read-only: loop/dither diagnostics of the existing v4 transfer traces (0.2 s rows; cmd is POST-brake).
revisit_frac: fraction of rows whose position is within 0.3 m of a position visited >= 10 s earlier.
low_speed_frac: fraction of rows with ground speed < 0.05 m/s.  The brake stats come from the episode's own sensor_stats."""
import json
from pathlib import Path
import numpy as np

ROOT = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/navgym-v4-transfer-20260930")
rows = []
for arm in ("armV4-s5", "armV4-s6", "astar"):
    for f in sorted((ROOT / arm).glob("seed*.json")):
        d = json.loads(f.read_text())
        tr = [r for r in d["trace"] if r["state"] != "settle"]
        t = np.array([r["t"] for r in tr]); xy = np.array([r["xy"] for r in tr]); vx = np.array([r["cmd"][0] for r in tr])
        sp = np.r_[0, np.linalg.norm(np.diff(xy, axis=0), axis=1) / np.diff(t)]
        rev = 0
        for i in range(len(t)):
            old = t <= t[i] - 10.0
            if old.any() and np.min(np.linalg.norm(xy[old] - xy[i], axis=1)) < 0.3:
                rev += 1
        st = d["sensor_stats"]
        rows.append({"arm": arm, "seed": d["seed"], "outcome": d["outcome_class"], "elapsed_s": d["elapsed_s"], "path_m": d["path_length_m"],
                     "revisit_frac": round(rev / len(t), 3), "low_speed_frac": round(float(np.mean(sp < 0.05)), 3),
                     "braked_frac": round(st["braked_robot_steps"] / st["sampled_robot_steps"], 3),
                     "vx_post_mean_when_lt035": round(float(vx[vx < 0.3495].mean()), 3) if np.any(vx < 0.3495) else None,
                     "vx_post_mean": round(float(vx.mean()), 3), "vx_post_zero_frac": round(float(np.mean(vx < 1e-3)), 3)})
for r in rows:
    print(r)
out = Path(__file__).resolve().parent / "data" / "trace_loops.json"
out.write_text(json.dumps(rows, indent=1))
for arm in ("armV4-s5", "armV4-s6", "astar"):
    for ok in (True, False):
        rr = [r for r in rows if r["arm"] == arm and (r["outcome"] == "goal") == ok]
        if rr:
            print(arm, "success" if ok else "failure", len(rr), {k: round(float(np.mean([r[k] for r in rr])), 3) for k in
                                                                 ("revisit_frac", "low_speed_frac", "braked_frac", "vx_post_mean", "vx_post_zero_frac")})
