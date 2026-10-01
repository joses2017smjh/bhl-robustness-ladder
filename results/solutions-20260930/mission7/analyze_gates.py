"""Read-only analysis of Mission 7 exact-replay gate traces.

For each gate and layout: stage history, fall time and phase, plate-frame pose
at key instants, crossing progress, contacts near the fall.
"""
import gzip
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
sys.path.insert(0, str(REPO / "src"))
from bhl_robust.mission.layout import generate  # noqa: E402

C = REPO / "results/mission7-campaign-20260923"
F = REPO / "results/mission7-approach-followup-20260922"
GATES = {
    "guarded(21397732,PASS)": F / "plate-stage-cn-c22-guarded/episodes.json",
    "waitopen2(21400863,PASS)": C / "replay-gate-waitopen2/episodes.json.gz",
    "v2-align(21405537)": C / "replay-gate-v2-align/episodes.json.gz",
    "v3-settle190(21405539)": C / "replay-gate-v3-settle190/episodes.json.gz",
    "v4-settle080(21405541)": C / "replay-gate-v4-settle080/episodes.json.gz",
    "v3-settle190-align(21405543)": C / "replay-gate-v3-settle190-align/episodes.json.gz",
    "v4-settle080-align(21405545)": C / "replay-gate-v4-settle080-align/episodes.json.gz",
}


def load(path):
    op = gzip.open if str(path).endswith(".gz") else open
    with op(path, "rt") as fh:
        return json.load(fh)


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def plate_frame(layout, door, side, xy):
    k = layout.door_indices[door]
    a, b = layout.xy(layout.route[k]), layout.xy(layout.route[k + 1])
    direction = (b - a) / layout.cell_m
    lateral = np.array([-direction[1], direction[0]])
    plate = layout.plate(door, side)
    d = np.asarray(xy) - plate
    return float(d @ direction), float(d @ lateral * side), direction  # lateral>0 = toward wall side


def main(only=None):
    out = {}
    for name, path in GATES.items():
        if only and only not in name:
            continue
        data = load(path)
        rows = []
        for ep in data["episodes"]:
            idx = ep["layout_index"]
            layout = generate("validation", idx)
            s = ep["samples"]
            hist = ep["stage_history"]
            first = next((i for i, x in enumerate(s) if x["tilt"] >= .78), None)
            row = {"layout": idx, "fall": ep["fall"], "first_fall_s": ep["first_fall_s"],
                   "T": round(s[-1]["time_s"], 2),
                   "history": [(round(h["time_s"], 2), h.get("door"), h.get("side"), h["phase"]) for h in hist]}
            # crossing progress per door: along at start/end of cross, max along
            crosses = []
            cur = None
            for h in hist:
                if h["phase"] == "cross":
                    cur = {"door": h["door"], "t0": h["time_s"]}
                if h["phase"] == "recorded" and cur is not None:
                    cur["t1"] = h["time_s"]
                    crosses.append(cur)
                    cur = None
            if cur is not None:
                cur["t1"] = None
                crosses.append(cur)
            side_by_door = {}
            for h in hist:
                if "side" in h and h.get("door") is not None:
                    side_by_door[h["door"]] = h["side"]
            for c in crosses:
                door = c["door"]
                side = side_by_door.get(door, layout.correct_sides[door])
                win = [x for x in s if x["time_s"] >= c["t0"] - 1e-9 and (c["t1"] is None or x["time_s"] <= c["t1"] + 1e-9)]
                alongs = [plate_frame(layout, door, side, x["xy"])[0] for x in win]
                lats = [plate_frame(layout, door, side, x["xy"])[1] for x in win]
                _, _, direction = plate_frame(layout, door, side, win[0]["xy"])
                dyaw = [wrap(x["yaw"] - np.arctan2(direction[1], direction[0])) for x in win]
                c.update(along0=round(alongs[0], 3), along1=round(alongs[-1], 3), along_max=round(max(alongs), 3),
                         lat0=round(lats[0], 3), lat1=round(lats[-1], 3),
                         yaw_err0=round(dyaw[0], 3), yaw_err1=round(dyaw[-1], 3),
                         dur=None if c["t1"] is None else round(c["t1"] - c["t0"], 2),
                         cmd0=[round(v, 3) for v in win[0]["effective_command"]],
                         cmd_mid=[round(v, 3) for v in win[len(win) // 2]["effective_command"]])
            row["crosses"] = crosses
            if first is not None:
                fs = s[first]
                phase = fs["phase"]
                # locate nearest door for plate frame
                best = None
                for door in (0, 1):
                    for side in (-1, 1):
                        a, l, _ = plate_frame(layout, door, side, fs["xy"])
                        dist = np.hypot(a, l)
                        if best is None or dist < best[0]:
                            best = (dist, door, side, a, l)
                window = s[max(0, first - 75):first + 1]
                plate_ev = [ev for x in window for ev in x["contact_events"] if ev["world_geom"].startswith("plate_")]
                geoms = sorted({ev["world_geom"] for ev in plate_ev})
                bodies = sorted({ev.get("body2") if ev.get("body1") == "world" else ev.get("body1") for ev in plate_ev})
                walls = sorted({c for x in window for c in x["contacts"]})
                row["fall_detail"] = {
                    "phase": phase, "xy": [round(v, 3) for v in fs["xy"]],
                    "nearest_plate": {"door": best[1], "side": best[2], "along": round(best[3], 3),
                                      "lat": round(best[4], 3), "dist": round(best[0], 3)},
                    "correct_side": layout.correct_sides[best[1]],
                    "cmd_last3s": [(round(x["time_s"], 2), x["phase"], [round(v, 2) for v in x["effective_command"]],
                                    [round(v, 2) for v in x["velocity_body"]], round(x["tilt"], 3))
                                   for x in s[max(0, first - 75):first + 1:5]],
                    "plate_geoms_last3s": geoms, "plate_bodies_last3s": bodies, "walls_last3s": walls,
                    "peak_plate_N_last3s": round(max((ev["normal_force_N"] for ev in plate_ev), default=0.), 1),
                }
            rows.append(row)
        out[name] = rows
    return out


if __name__ == "__main__":
    res = main(sys.argv[1] if len(sys.argv) > 1 else None)
    dest = Path(__file__).with_name("gate_analysis.json")
    dest.write_text(json.dumps(res, indent=1))
    for name, rows in res.items():
        print("=" * 20, name, "upright", sum(not r["fall"] for r in rows), "/", len(rows))
        for r in rows:
            print(f"L{r['layout']:2d} fall={r['fall']!s:5} t_fall={r['first_fall_s']} T={r['T']}")
            for c in r["crosses"]:
                print("    cross", c)
            if "fall_detail" in r:
                fd = r["fall_detail"]
                print("    FALL phase", fd["phase"], "xy", fd["xy"], "plate", fd["nearest_plate"], "correct", fd["correct_side"],
                      "plates", fd["plate_geoms_last3s"], "bodies", fd["plate_bodies_last3s"], "walls", fd["walls_last3s"],
                      "peakN", fd["peak_plate_N_last3s"])
