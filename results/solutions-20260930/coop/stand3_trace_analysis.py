"""Characterise the Stand3 replay traces (job 21470828): cube height over the deck, hand
proximity, cube roll, base displacement / lean. numpy only; reads the committed traces."""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np

SRC = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/repo-gpu-20260923/"
           "stand3_replay_2026-09-28/21470828")
OUT = Path(__file__).resolve().parent
DECK_EDGE, OFF_FLOOR, SEAT_Z, SEAT_TOL, SEAT_SPEED = 0.17, 0.34, 0.57, 0.02, 0.05
SEAT_XLO, SEAT_XHI, SEAT_YH = 0.19, 0.45, 0.10
SPAWN_Z = 0.55
# final Curriculum/lift_height (last-200 mean) from the event files (tb_stand3.json)
tb = json.load(open(OUT / "tb_stand3.json"))
LIFT_H = {k: tb[k]["Curriculum/lift_height"]["win200_mean_from"]["7800"] for k in ("s0", "s1")}


def pct(a, qs=(5, 25, 50, 75, 95)):
    a = np.asarray(a, float)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return None
    return {f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs} | {"n": int(a.size)}


res = {}
for s in ("s0", "s1"):
    z = np.load(SRC / f"{s}.trace.npz")
    rep = json.load(open(SRC / f"replay_{s}.json"))
    valid = z["valid"]
    pos = z["cube_pos"]
    x, y, zz = pos[..., 0], pos[..., 1], pos[..., 2]
    spd = z["cube_speed"]
    ax = np.abs(x)
    over = valid & (ax > DECK_EDGE) & (zz > OFF_FLOOR)
    still = spd < SEAT_SPEED
    held_above = over & still & (zz - SEAT_Z >= SEAT_TOL)
    seated = valid & (ax > SEAT_XLO) & (ax < SEAT_XHI) & (np.abs(y) < SEAT_YH) & (np.abs(zz - SEAT_Z) < SEAT_TOL) & still
    moving_over = over & ~still
    thr = SPAWN_Z + LIFT_H[s]
    hf = z["hand_force"]
    sdf = z["hand_sdf"]                       # hand-link origin distance to the cube surface (m)
    # which env side is "deck": sign of x when over deck
    r = {"lift_pay_threshold_z": round(thr, 4), "seat_window_z": [SEAT_Z - SEAT_TOL, SEAT_Z + SEAT_TOL],
         "hand_force_nonzero_share_all_valid": round(float((hf[valid] > 1e-6).mean()), 6),
         "hand_force_max_N": round(float(np.nanmax(hf[valid])), 6),
         "over_deck_steps": int(over.sum()), "held_above_steps": int(held_above.sum()),
         "held_above_z": pct(zz[held_above]),
         "held_above_z_minus_seat_top_cm": pct(100 * (zz[held_above] - (SEAT_Z + SEAT_TOL))),
         "held_above_share_at_or_above_pay_threshold": round(float((zz[held_above] > thr).mean()), 4) if held_above.any() else None,
         "held_above_share_between_seat_top_and_pay_threshold": round(float(((zz[held_above] >= SEAT_Z + SEAT_TOL) & (zz[held_above] <= thr)).mean()), 4) if held_above.any() else None,
         "moving_over_deck_z": pct(zz[moving_over]),
         "over_deck_abs_x": pct(ax[over]),
         "held_above_abs_x": pct(ax[held_above]),
         "held_above_min_hand_sdf_m": pct(sdf[held_above].min(axis=-1)),
         "held_above_hands_within_5cm": pct((sdf[held_above] < 0.05).sum(-1), qs=(5, 50, 95)),
         "held_above_hands_inside_cube_share": round(float((sdf[held_above] <= 0).any(-1).mean()), 4) if held_above.any() else None,
         "cube_face_tilt_deg_over_deck": pct(z["cube_face_tilt_deg"][over]),
         "cube_up_axis_not_z_share_over_deck": round(float((z["cube_up_axis"][over] != 2).mean()), 4) if over.any() else None,
         "cube_up_axis_not_z_share_all_offfloor": round(float((z["cube_up_axis"][valid & (zz > OFF_FLOOR)] != 2).mean()), 4),
         }
    # the trained config spawns the cube 180 deg about x (raw quaternion), so the up axis reads the
    # cube's -z; cube_up_axis is an index of the body axis most aligned with world z (sign-agnostic?)
    ua = z["cube_up_axis"][valid & (zz > OFF_FLOOR)]
    r["cube_up_axis_counts_offfloor"] = {int(k): int(v) for k, v in zip(*np.unique(ua, return_counts=True))}

    # ----- per-episode profiles
    ep_idx = z["ep_idx"]
    eps = []
    T, E = valid.shape
    for e in range(E):
        for k in np.unique(ep_idx[:, e][valid[:, e]]):
            m = valid[:, e] & (ep_idx[:, e] == k)
            idx = np.where(m)[0]
            if idx.size == 0:
                continue
            ez, ex, espd = zz[idx, e], ax[idx, e], spd[idx, e]
            carried = ez > OFF_FLOOR
            cmax = float(ex[carried].max()) if carried.any() else 0.0
            ov = over[idx, e]
            ha = held_above[idx, e]
            first_over = int(np.argmax(ov)) if ov.any() else None
            # z while over deck, the minimum reached (did it ever go down into the window?)
            zmin_over = float(ez[ov].min()) if ov.any() else None
            zmed_over = float(np.median(ez[ov])) if ov.any() else None
            eps.append({"env": e, "ep": int(k), "len": int(idx.size), "carried_max_abs_x": round(cmax, 4),
                        "reached": cmax >= SEAT_XLO, "over_steps": int(ov.sum()), "held_above_steps": int(ha.sum()),
                        "first_over_step": first_over, "z_min_over": zmin_over, "z_med_over": zmed_over,
                        "ended_fallen": bool(z["term_fallen"][idx[-1], e]), "ended_success": bool(z["term_success"][idx[-1], e]),
                        "z_at_carried_max": float(ez[carried][ex[carried].argmax()]) if carried.any() else None,
                        "face_changed": bool((z["cube_up_axis"][idx, e] != z["cube_up_axis"][idx[0], e]).any()),
                        "max_face_tilt_deg": float(np.nanmax(z["cube_face_tilt_deg"][idx, e]))})
    reached = [q for q in eps if q["reached"]]
    r["episodes"] = len(eps)
    r["reached_episodes"] = len(reached)
    r["reached_z_min_over_deck"] = pct([q["z_min_over"] for q in reached if q["z_min_over"] is not None])
    r["reached_z_median_over_deck"] = pct([q["z_med_over"] for q in reached if q["z_med_over"] is not None])
    r["reached_over_deck_steps"] = pct([q["over_steps"] for q in reached], qs=(10, 50, 90))
    r["reached_first_over_step"] = pct([q["first_over_step"] for q in reached if q["first_over_step"] is not None], qs=(10, 50, 90))
    r["reached_that_dipped_into_seat_window"] = int(sum(1 for q in reached if q["z_min_over"] is not None and q["z_min_over"] < SEAT_Z + SEAT_TOL))
    r["all_eps_face_changed_share"] = round(float(np.mean([q["face_changed"] for q in eps])), 4)
    r["all_eps_max_face_tilt_deg"] = pct([q["max_face_tilt_deg"] for q in eps])
    r["fallen_eps"] = int(sum(q["ended_fallen"] for q in eps))
    r["fallen_eps_carried_max_abs_x"] = pct([q["carried_max_abs_x"] for q in eps if q["ended_fallen"]])
    r["success_eps"] = int(sum(q["ended_success"] for q in eps))
    r["carried_max_abs_x_all"] = pct([q["carried_max_abs_x"] for q in eps])

    # ----- base displacement / lean toward the deck at the carried max (from replay JSON)
    eprows = rep["episodes"]
    def col(rows, who, key):
        return [q[f"robot_{who}"]["at_carried_max"][key] for q in rows if q.get(f"robot_{who}") and q[f"robot_{who}"].get("at_carried_max")]
    for tag, rows in (("reached", [q for q in eprows if q["reached_required"]]),
                      ("not_reached", [q for q in eprows if not q["reached_required"]])):
        r[f"{tag}_base_dx_toward_deck_m"] = {w: pct(col(rows, w, "base_dx_toward_deck_m")) for w in ("a", "b")}
        r[f"{tag}_lean_x_toward_deck_deg"] = {w: pct(col(rows, w, "lean_x_toward_deck_deg")) for w in ("a", "b")}
        r[f"{tag}_yaw_rel_deg"] = {w: pct(col(rows, w, "yaw_rel_deg")) for w in ("a", "b")}
    # pair-sum of base shift toward the deck vs cube shift
    both = [(q["max_abs_x_carried"], q["robot_a"]["at_carried_max"]["base_dx_toward_deck_m"],
             q["robot_b"]["at_carried_max"]["base_dx_toward_deck_m"]) for q in eprows
            if q.get("robot_a", {}).get("at_carried_max") and q.get("robot_b", {}).get("at_carried_max")]
    if both:
        arr = np.array(both)
        r["cube_shift_vs_mean_base_shift"] = {"cube_p50": round(float(np.median(arr[:, 0])), 4),
                                               "mean_base_shift_p50": round(float(np.median(arr[:, 1:].mean(1))), 4),
                                               "share_of_cube_shift_from_base_p50": round(float(np.median(arr[:, 1:].mean(1) / np.maximum(arr[:, 0], 1e-3))), 3)}
    res[s] = r
    print("=====", s)
    print(json.dumps({k: v for k, v in r.items()}, indent=1)[:6000])

(OUT / "stand3_trace_analysis.json").write_text(json.dumps(res, indent=1))
