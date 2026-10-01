"""Dump a per-sample trace (plate frame) for one gate/layout window.

usage: dump_episode.py <gate-key-substring> <layout> <t0> <t1> [stride]
"""
import sys
import numpy as np
from analyze_gates import GATES, load, plate_frame, wrap, generate

key, layout_index, t0, t1 = sys.argv[1], int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
stride = int(sys.argv[5]) if len(sys.argv) > 5 else 1
name = next(n for n in GATES if key in n)
data = load(GATES[name])
ep = next(e for e in data["episodes"] if e["layout_index"] == layout_index)
layout = generate("validation", layout_index)
side = next((h["side"] for h in ep["stage_history"] if "side" in h), layout.correct_sides[0])
door = 0
k = layout.door_indices[door]
a, b = layout.xy(layout.route[k]), layout.xy(layout.route[k + 1])
direction = (b - a) / layout.cell_m
door_yaw = float(np.arctan2(direction[1], direction[0]))
print(name, "layout", layout_index, "cell", layout.cell_m, "door dir", direction, "plate side", side,
      "correct", layout.correct_sides[door], "plate", layout.plate(door, side), "activation",
      "stage", [(round(h["time_s"], 2), h["phase"]) for h in ep["stage_history"]])
print("t  phase  along lat(+=wall side)  yawerr  cmd(eff body)  rec_cmd  v_body  yawrate  tilt  plate-contacts(body:N)")
for i, s in enumerate(ep["samples"]):
    if not (t0 <= s["time_s"] <= t1) or i % stride:
        continue
    al, la, _ = plate_frame(layout, door, side, s["xy"])
    ye = wrap(s["yaw"] - door_yaw)
    pc = {}
    for ev in s["contact_events"]:
        if ev["world_geom"].startswith("plate_"):
            body = ev["body2"] if ev["body1"] == "world" else ev["body1"]
            body = body.replace("r0_leg_", "").replace("_ankle_roll", "")
            key2 = ev["world_geom"][-2:] + ":" + body
            pc[key2] = max(pc.get(key2, 0.), ev["normal_force_N"])
    print(f"{s['time_s']:6.2f} {s['phase'][:8]:8s} {al:+.3f} {la:+.3f} {ye:+.2f} "
          f"[{s['effective_command'][0]:+.2f},{s['effective_command'][1]:+.2f},{s['effective_command'][2]:+.2f}] "
          f"[{s['recorded_command'][0]:+.2f},{s['recorded_command'][1]:+.2f},{s['recorded_command'][2]:+.2f}] "
          f"[{s['velocity_body'][0]:+.2f},{s['velocity_body'][1]:+.2f}] {s['yaw_rate']:+.2f} {s['tilt']:.3f} "
          + " ".join(f"{k2}:{v:.0f}" for k2, v in sorted(pc.items())) + (" W:" + ",".join(s["contacts"]) if s["contacts"] else ""))
