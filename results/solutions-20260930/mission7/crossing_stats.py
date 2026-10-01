"""Per-crossing statistics across all retained replay gates (read-only).

For every crossing: clear / stall / fall, alignment at start, time spent stalled at the
plate edge (|v_along| small while the leading foot is on the plate), plate-edge dwell
before a fall, and the post-stage outcome.
"""
import json
from pathlib import Path

import numpy as np
from analyze_gates import GATES, load, plate_frame, wrap, generate

rows = []
for name, path in GATES.items():
    data = load(path)
    for ep in data["episodes"]:
        idx = ep["layout_index"]
        layout = generate("validation", idx)
        s = ep["samples"]
        hist = ep["stage_history"]
        side_by_door = {h["door"]: h["side"] for h in hist if "side" in h}
        cross_t0 = [h for h in hist if h["phase"] == "cross"]
        for c in cross_t0:
            door = c["door"]
            t0 = c["time_s"]
            t1 = next((h["time_s"] for h in hist if h["phase"] == "recorded" and h["time_s"] >= t0 and h.get("door") == door), None)
            side = side_by_door.get(door, layout.correct_sides[door])
            k = layout.door_indices[door]
            a, b = layout.xy(layout.route[k]), layout.xy(layout.route[k + 1])
            direction = (b - a) / layout.cell_m
            dyaw = float(np.arctan2(direction[1], direction[0]))
            win = [x for x in s if x["time_s"] >= t0 - 1e-9 and (t1 is None or x["time_s"] <= t1 + 1e-9)]
            fall_t = ep["first_fall_s"]
            fell_in_cross = fall_t is not None and fall_t >= t0 and (t1 is None or fall_t <= t1)
            if fell_in_cross:
                win_pre = [x for x in win if x["time_s"] <= fall_t]
            else:
                win_pre = win
            al = np.array([plate_frame(layout, door, side, x["xy"])[0] for x in win_pre])
            ye0 = wrap(win[0]["yaw"] - dyaw)
            # stall: along progress < 2 cm over the trailing 0.8 s while near the plate edge band
            times = np.array([x["time_s"] for x in win_pre])
            edge = (al > -0.36) & (al < -0.14)
            stalled = np.zeros(len(al), bool)
            for i in range(len(al)):
                j = np.searchsorted(times, times[i] - 0.8)
                if times[i] - times[j] >= 0.76 and abs(al[i] - al[j]) < 0.03:
                    stalled[i] = True
            edge_stall_s = float(np.sum(edge & stalled) * 0.04)
            cleared = bool(np.any(al >= 0.349))
            tilt_max_pre = max(x["tilt"] for x in win_pre)
            cmd_lat = float(np.mean([abs(x["effective_command"][1]) for x in win_pre]))
            cmd_yaw = float(np.mean([abs(x["effective_command"][2]) for x in win_pre]))
            post_fall = fall_t is not None and t1 is not None and fall_t > t1
            rows.append(dict(gate=name, layout=idx, door=door, t0=round(t0, 2), dur=None if t1 is None else round(t1 - t0, 2),
                             yaw_err0=round(ye0, 2), sideways=abs(ye0) > 0.8, along0=round(float(al[0]), 3),
                             along_max=round(float(al.max()), 3), cleared=cleared,
                             edge_stall_s=round(edge_stall_s, 2), fell_in_cross=fell_in_cross,
                             fall_after_s=None if not fell_in_cross else round(fall_t - t0, 2),
                             post_stage_fall=post_fall, mean_abs_cmd_lat=round(cmd_lat, 2), mean_abs_cmd_yaw=round(cmd_yaw, 2),
                             tilt_max_pre=round(tilt_max_pre, 3)))

Path(__file__).with_name("crossing_stats.json").write_text(json.dumps(rows, indent=1))
hdr = "gate layout yaw_err0 side? along0 along_max cleared edge_stall_s fell_in_cross fall_after post_fall |lat| |yaw|"
print(hdr)
for r in rows:
    print(f"{r['gate'][:22]:22s} L{r['layout']:2d} d{r['door']} {r['yaw_err0']:+.2f} {'SIDE' if r['sideways'] else 'fwd '} "
          f"{r['along0']:+.3f} {r['along_max']:+.3f} {'CLEAR' if r['cleared'] else 'stall'} {r['edge_stall_s']:4.2f} "
          f"{'FELL' if r['fell_in_cross'] else '    '} {r['fall_after_s']} {'POSTFALL' if r['post_stage_fall'] else ''} "
          f"{r['mean_abs_cmd_lat']:.2f} {r['mean_abs_cmd_yaw']:.2f}")

# pooled over the cross-clear variants
cc = [r for r in rows if not r["gate"].startswith(("guarded", "waitopen"))]
print("\ncross-clear variants: crossings", len(cc))
for label, sel in (("sideways", [r for r in cc if r["sideways"]]), ("forward", [r for r in cc if not r["sideways"]])):
    print(label, "n", len(sel), "cleared", sum(r["cleared"] for r in sel), "fell_in_cross", sum(r["fell_in_cross"] for r in sel),
          "edge-stalled(>=0.4s)", sum(r["edge_stall_s"] >= 0.4 for r in sel),
          "falls among edge-stalled", sum(r["fell_in_cross"] for r in sel if r["edge_stall_s"] >= 0.4))
print("never-cleared crossings", sum(not r["cleared"] for r in cc), "of which fell in cross", sum(r["fell_in_cross"] for r in cc if not r["cleared"]))
print("cleared crossings", sum(r["cleared"] for r in cc), "of which post-stage fall", sum(r["post_stage_fall"] for r in cc if r["cleared"]))
print("never-cleared & survived crossing", sum((not r["cleared"]) and (not r["fell_in_cross"]) for r in cc),
      "of which post-stage fall", sum(r["post_stage_fall"] for r in cc if (not r["cleared"]) and (not r["fell_in_cross"])))
