"""Dump a plate-frame window from a compact trace (read-only).

usage: dump_compact.py <label> <layout> <door> <side> <t0> <t1> [stride]
"""
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src")
from bhl_robust.mission.layout import generate  # noqa: E402

label, idx, door, side, t0, t1 = sys.argv[1], int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), float(sys.argv[6])
stride = int(sys.argv[7]) if len(sys.argv) > 7 else 1
d = pickle.load(open(Path(__file__).resolve().parent / f"{label}.pkl", "rb"))
ep = next(e for e in d["episodes"] if e["layout_index"] == idx)
lay = generate("validation", idx)
_, direction = lay.door(door)
direction = np.asarray(direction, float)
lateral = np.array([-direction[1], direction[0]])
plate = np.asarray(lay.plate(door, side), float)
dyaw = float(np.arctan2(direction[1], direction[0]))
print(label, "L", idx, "door", door, "side", side, "correct", lay.correct_sides[door], "dir", direction, "plate", plate,
      "cell", lay.cell_m, "hist", [(round(h["time_s"], 2), h.get("door"), h.get("side"), h["phase"]) for h in ep["stage_history"]])
ev_by_t = {}
for (tt, g, b, n, dist) in ep["plate_ev"]:
    key = (round(tt, 2))
    k2 = g.replace("plate_", "") + ":" + b.replace("r0_leg_", "").replace("_ankle_roll", "")
    ev_by_t.setdefault(key, {})
    ev_by_t[key][k2] = max(ev_by_t[key].get(k2, 0.), n)
print("t phase along lat yawerr eff_cmd rec_cmd tilt plate(body:N)")
for i, tt in enumerate(ep["t"]):
    if not (t0 <= tt <= t1) or i % stride:
        continue
    xy = np.asarray(ep["xy"][i])
    al = float((xy - plate) @ direction)
    la = float((xy - plate) @ lateral * side)
    ye = float(np.arctan2(np.sin(ep["yaw"][i] - dyaw), np.cos(ep["yaw"][i] - dyaw)))
    e, r = ep["eff"][i], ep["rec"][i]
    pc = ev_by_t.get(round(tt, 2), {})
    print(f"{tt:6.2f} {ep['phase'][i][:8]:8s} {al:+.3f} {la:+.3f} {ye:+.2f} [{e[0]:+.2f},{e[1]:+.2f},{e[2]:+.2f}] "
          f"[{r[0]:+.2f},{r[1]:+.2f},{r[2]:+.2f}] {ep['tilt'][i]:.3f} " + " ".join(f"{k}:{v:.0f}" for k, v in sorted(pc.items())))
