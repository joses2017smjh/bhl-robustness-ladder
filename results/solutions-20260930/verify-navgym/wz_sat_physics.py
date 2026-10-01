"""verify-navgym: |wz| saturation and forward-command statistics from the SAVED physics traces
(results/navgym-v4-transfer-20260930/<arm>/seed500xx.json, trace sampled every 5th 0.04 s step;
cmd = command after the team_sensors speed brake, which scales only (vx, vy), so cmd[2] = raw wz).
Reads saved outputs only."""
import json, sys
import numpy as np
RES = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/navgym-v4-transfer-20260930"
out = {}
for arm in ("armV4-s5", "armV4-s6", "astar"):
    rows = []
    for s in range(50000, 50012):
        d = json.load(open(f"{RES}/{arm}/seed{s}.json"))
        tr = [t for t in d["trace"] if t["state"] != "settle"]
        wz = np.array([t["cmd"][2] for t in tr]); vx = np.array([t["cmd"][0] for t in tr])
        sat99 = float(np.mean(np.abs(wz) >= 0.99)); sat95 = float(np.mean(np.abs(wz) >= 0.95))
        # sign flips between consecutive 0.2 s samples (aliased: a lower bound on the per-step flip count)
        sgn = np.sign(wz); flips = int(np.sum(sgn[1:] * sgn[:-1] < 0))
        dur = len(tr) * 0.2
        rows.append({"seed": s, "outcome": d["outcome"], "n_samples": len(tr), "wz_sat_ge_0.99": round(sat99, 3),
                     "wz_sat_ge_0.95": round(sat95, 3), "max_abs_wz": round(float(np.max(np.abs(wz))), 3),
                     "flips_per_s_0.2s_sampled": round(flips / dur, 2), "median_vx_cmd_filtered": round(float(np.median(vx)), 3)})
    out[arm] = rows
    sat = [r["wz_sat_ge_0.99"] for r in rows]
    print(arm, "wz_sat>=0.99 per seed:", sat, "min %.3f max %.3f" % (min(sat), max(sat)),
          "| max|wz|", max(r["max_abs_wz"] for r in rows), "| flips/s (0.2 s aliased)", [r["flips_per_s_0.2s_sampled"] for r in rows])
json.dump(out, open(sys.argv[1], "w"), indent=1)
