"""Independent re-derivation (read-only) of the Mission 7 crossing facts from the compact traces.

Geometry comes straight from bhl_robust.mission.layout (plate = xy(route[k]) + lateral*side*.42 - direction*.12,
direction = layout.door(i)[1]); 'along' = (base_xy - plate) . direction, 'lat' = (base_xy - plate) . lateral * side.
Cleared = along >= 0.35 (the --cross-clear value).  Sideways = |yaw - door_yaw| > 0.8 rad at cross start.
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src")
from bhl_robust.mission.layout import generate  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = ["gref_21397732", "waitopen2_21400863", "v2align_21405537", "v3s190_21405539", "v4s080_21405541",
          "v3s190a_21405543", "v4s080a_21405545"]
CROSS_CLEAR = ["v2align_21405537", "v3s190_21405539", "v4s080_21405541", "v3s190a_21405543", "v4s080a_21405545"]


def wrap(a):
    return float(np.arctan2(np.sin(a), np.cos(a)))


def frame(layout, door, side):
    _, direction = layout.door(door)
    direction = np.asarray(direction, float)
    lateral = np.array([-direction[1], direction[0]])
    plate = np.asarray(layout.plate(door, side), float)
    return plate, direction, lateral


def main():
    out = {}
    for label in LABELS:
        d = pickle.load(open(HERE / f"{label}.pkl", "rb"))
        rows = []
        for ep in d["episodes"]:
            idx = ep["layout_index"]
            lay = generate("validation", idx)
            t = np.array(ep["t"])
            xy = np.array(ep["xy"])
            yaw = np.array(ep["yaw"])
            tilt = np.array(ep["tilt"])
            eff = np.array(ep["eff"])
            hist = ep["stage_history"]
            fall_t = ep["first_fall_s"]
            # sanity: first_fall_s equals first tilt >= .78
            ff = t[np.argmax(tilt >= .78)] if np.any(tilt >= .78) else None
            assert (ff is None and fall_t is None) or abs(ff - fall_t) < 1e-9, (label, idx, ff, fall_t)
            row = dict(layout=idx, T=round(float(t[-1]), 2), fall=ep["fall"], first_fall_s=fall_t,
                       history=[(round(h["time_s"], 2), h.get("door"), h.get("side"), h["phase"]) for h in hist],
                       crossings=[])
            approaches = [h for h in hist if h["phase"] == "approach"]
            row["takeover_s"] = [round(h["time_s"], 2) for h in approaches]
            row["end_minus_first_takeover_s"] = None if not approaches else round(float(t[-1]) - approaches[0]["time_s"], 2)
            # phase at end of episode / time in approach without settle
            last_phase = hist[-1]["phase"] if hist else None
            row["last_phase"] = last_phase
            if approaches and last_phase == "approach":
                a0 = approaches[-1]["time_s"]
                m = t >= a0 - 1e-9
                seg = xy[m]
                row["stuck_in_approach_s"] = round(float(t[-1]) - a0, 2)
                row["approach_path_len_m"] = round(float(np.sum(np.linalg.norm(np.diff(seg, axis=0), axis=1))), 2)
                row["approach_net_disp_m"] = round(float(np.linalg.norm(seg[-1] - seg[0])), 3)
                door, side = approaches[-1]["door"], approaches[-1]["side"]
                plate, direction, lateral = frame(lay, door, side)
                al = (seg - plate) @ direction
                la = (seg - plate) @ lateral * side
                pre = plate - direction * .30
                row["approach_along_range"] = [round(float(al.min()), 3), round(float(al.max()), 3)]
                row["approach_lat_range"] = [round(float(la.min()), 3), round(float(la.max()), 3)]
                row["approach_dist_to_prepoint_min"] = round(float(np.min(np.linalg.norm(seg - pre, axis=1))), 3)
                # displacement over the final 20 s of the approach
                m20 = t >= t[-1] - 20.
                row["approach_last20s_path_m"] = round(float(np.sum(np.linalg.norm(np.diff(xy[m20], axis=0), axis=1))), 3)
                row["approach_last20s_net_m"] = round(float(np.linalg.norm(xy[m20][-1] - xy[m20][0])), 3)
                ev = [e for e in ep["plate_ev"] if e[0] >= a0 - 1e-9]
                bygeom = {}
                for (tt, g, b, n, dist) in ev:
                    k = f"{g}:{b}"
                    v = bygeom.setdefault(k, [0, 0., 1e9, -1e9])
                    v[0] += 1
                    v[1] = max(v[1], n)
                    v[2] = min(v[2], tt)
                    v[3] = max(v[3], tt)
                row["approach_plate_contacts"] = {k: [v[0], round(v[1], 1), round(v[2], 2), round(v[3], 2)]
                                                  for k, v in sorted(bygeom.items())}
                row["approach_eff_cmd_last"] = [round(v, 3) for v in eff[-1]]
            # crossings
            for i, h in enumerate(hist):
                if h["phase"] != "cross":
                    continue
                door = h["door"]
                t0 = h["time_s"]
                t1 = next((g["time_s"] for g in hist[i + 1:] if g["phase"] == "recorded" and g.get("door") == door), None)
                side = next(g["side"] for g in hist[:i][::-1] if g.get("door") == door and "side" in g)
                plate, direction, lateral = frame(lay, door, side)
                dyaw = float(np.arctan2(direction[1], direction[0]))
                m = (t >= t0 - 1e-9) & ((t <= t1 + 1e-9) if t1 is not None else True)
                fell_in = fall_t is not None and fall_t >= t0 - 1e-9 and (t1 is None or fall_t <= t1 + 1e-9)
                mpre = m & ((t <= fall_t + 1e-9) if fell_in else True)
                al = (xy[mpre] - plate) @ direction
                la = (xy[mpre] - plate) @ lateral * side
                al_all = (xy[m] - plate) @ direction
                ye0 = wrap(yaw[m][0] - dyaw)
                ev = [e for e in ep["plate_ev"] if t0 - 1e-9 <= e[0] <= (t1 if t1 is not None else 1e9) + 1e-9]
                bodies = {}
                for (tt, g, b, n, dist) in ev:
                    k = f"{g}:{b}"
                    bodies[k] = max(bodies.get(k, 0.), n)
                post_fall = fall_t is not None and t1 is not None and fall_t > t1 + 1e-9
                cmd = eff[mpre]
                row["crossings"].append(dict(
                    door=door, side=side, correct_side=lay.correct_sides[door],
                    door_dir=[float(v) for v in direction], t0=round(t0, 2), t1=None if t1 is None else round(t1, 2),
                    dur=None if t1 is None else round(t1 - t0, 2),
                    along0=round(float(al[0]), 3), along_end=round(float(al_all[-1]), 3),
                    along_end_prefall=round(float(al[-1]), 3),
                    along_max=round(float(al.max()), 3), disp_end=round(float(al_all[-1] - al[0]), 3),
                    disp_max=round(float(al.max() - al[0]), 3),
                    lat0=round(float(la[0]), 3), lat_end=round(float(la[-1]), 3),
                    yaw_err0=round(ye0, 3), sideways=abs(ye0) > .8,
                    cleared=bool(np.any(al >= .35)), cleared_349=bool(np.any(al >= .349)),
                    fell_in_cross=fell_in, fall_after_cross_start=None if not fell_in else round(fall_t - t0, 2),
                    post_stage_fall=post_fall,
                    cmd_mean_abs=[round(float(v), 3) for v in np.mean(np.abs(cmd), axis=0)],
                    plate_contact_peakN=bodies))
            # whole-episode proximity to every plate (base position)
            prox = {}
            for door in (0, 1):
                for side in (-1, 1):
                    plate, direction, lateral = frame(lay, door, side)
                    al = (xy - plate) @ direction
                    la = (xy - plate) @ lateral * side
                    dist = np.hypot(al, la)
                    inside = dist <= .24
                    # traversal: base inside the disc footprint at some time, and along goes from <= -0.24 to >= +0.24
                    # within one contiguous visit of the |lat| <= 0.24 band
                    trav = False
                    band = np.abs(la) <= .24
                    j = 0
                    n = len(al)
                    while j < n:
                        if band[j]:
                            k = j
                            while k < n and band[k]:
                                k += 1
                            seg = al[j:k]
                            if seg.min() <= -.24 and seg.max() >= .24:
                                trav = True
                            j = k
                        else:
                            j += 1
                    prox[f"plate_{door}_{side}"] = dict(min_dist=round(float(dist.min()), 3),
                                                        t_min=round(float(t[np.argmin(dist)]), 2),
                                                        base_inside_s=round(float(inside.sum() * .04), 2),
                                                        base_traversed=trav,
                                                        along_at_end=round(float(al[-1]), 3), lat_at_end=round(float(la[-1]), 3))
            row["plate_proximity"] = prox
            # plate contact summary over the whole episode by geom
            pc = {}
            for (tt, g, b, n, dist) in ep["plate_ev"]:
                v = pc.setdefault(g, dict(n=0, peakN=0., t_first=1e9, t_last=-1e9, bodies=set()))
                v["n"] += 1
                v["peakN"] = max(v["peakN"], n)
                v["t_first"] = min(v["t_first"], tt)
                v["t_last"] = max(v["t_last"], tt)
                v["bodies"].add(b)
            row["plate_contacts_by_geom"] = {g: dict(n=v["n"], peakN=round(v["peakN"], 1), t_first=round(v["t_first"], 2),
                                                     t_last=round(v["t_last"], 2), bodies=sorted(v["bodies"]))
                                             for g, v in sorted(pc.items())}
            rows.append(row)
        out[label] = rows
    (HERE / "verify_crossings.json").write_text(json.dumps(out, indent=1, default=str))
    return out


if __name__ == "__main__":
    out = main()
    for label, rows in out.items():
        print("=" * 10, label, "upright", sum(not r["fall"] for r in rows), "/", len(rows))
        for r in rows:
            print(f" L{r['layout']:2d} T={r['T']} fall={r['first_fall_s']} takeover={r['takeover_s']} "
                  f"end-takeover={r['end_minus_first_takeover_s']} last_phase={r['last_phase']}")
            if "stuck_in_approach_s" in r:
                print("    STUCK-IN-APPROACH", r["stuck_in_approach_s"], "path", r["approach_path_len_m"], "net", r["approach_net_disp_m"],
                      "along", r["approach_along_range"], "lat", r["approach_lat_range"], "min d(pre)", r["approach_dist_to_prepoint_min"],
                      "last20s path/net", r["approach_last20s_path_m"], r["approach_last20s_net_m"], "cmd_last", r["approach_eff_cmd_last"])
                print("    approach plate contacts (n, peakN, t_first, t_last):", r["approach_plate_contacts"])
            for c in r["crossings"]:
                print(f"    cross d{c['door']} s{c['side']:+d} dir={c['door_dir']} t0={c['t0']} t1={c['t1']} yaw0={c['yaw_err0']:+.2f} "
                      f"{'SIDE' if c['sideways'] else 'fwd '} along0={c['along0']:+.3f} end={c['along_end']:+.3f} "
                      f"max={c['along_max']:+.3f} disp_end={c['disp_end']:+.3f} disp_max={c['disp_max']:+.3f} "
                      f"{'CLEAR' if c['cleared'] else 'noclear'} {'FELL@+%.2f' % c['fall_after_cross_start'] if c['fell_in_cross'] else ''} "
                      f"{'POSTFALL' if c['post_stage_fall'] else ''} cmd|.|={c['cmd_mean_abs']} plateN={ {k: round(v) for k, v in c['plate_contact_peakN'].items()} }")
            trav = {k: v for k, v in r["plate_proximity"].items() if v["base_traversed"] or v["base_inside_s"] > 0}
            print("    base-inside/traversed plates:", trav)
