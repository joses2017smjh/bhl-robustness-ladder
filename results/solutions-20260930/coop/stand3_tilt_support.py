"""Is the 'held above' cube actually resting tilted on the deck/plinth edge?  Lowest-corner
height and position from the logged pose (xyzw on v60), vs deck top 0.43 / plinth top 0.41."""
import json, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from pathlib import Path
SRC = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/repo-gpu-20260923/stand3_replay_2026-09-28/21470828")
H = 0.14; DECK_TOP = 0.43; PLINTH_TOP = 0.41; PLINTH_HALF = 0.13; DECK_EDGE = 0.17
def R_xyzw(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y * y + z * z); R[..., 0, 1] = 2 * (x * y - z * w); R[..., 0, 2] = 2 * (x * z + y * w)
    R[..., 1, 0] = 2 * (x * y + z * w); R[..., 1, 1] = 1 - 2 * (x * x + z * z); R[..., 1, 2] = 2 * (y * z - x * w)
    R[..., 2, 0] = 2 * (x * z - y * w); R[..., 2, 1] = 2 * (y * z + x * w); R[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return R
def pct(a, qs=(5, 25, 50, 75, 95)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return None if a.size == 0 else {**{f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs}, "n": int(a.size)}
if __name__ != "__main__":
    raise SystemExit
out = {}
for s in ("s0", "s1"):
    z = np.load(SRC / f"{s}.trace.npz")
    v = z["valid"]; p = z["cube_pos"]; q = z["cube_quat_raw"].astype(float); spd = z["cube_speed"]
    R = R_xyzw(q)
    # check the logged face tilt is reproduced (xyzw assumption)
    mz = np.abs(R[..., 2, :]).max(-1)
    tilt_chk = np.degrees(np.arccos(np.clip(mz, -1, 1)))
    err = np.nanmax(np.abs(tilt_chk[v] - z["cube_face_tilt_deg"][v]))
    # 8 corners, world
    signs = np.array([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], float) * H
    corners = p[..., None, :] + np.einsum("teij,kj->teki", R, signs)
    low_i = corners[..., 2].argmin(-1)
    low = np.take_along_axis(corners, low_i[..., None, None].repeat(3, -1), axis=-2)[..., 0, :]
    ax = np.abs(p[..., 0]); zc = p[..., 2]
    over = v & (ax > DECK_EDGE) & (zc > 0.34)
    held = over & (spd < 0.05) & (zc >= 0.59)
    # rotation about world y (the pinch axis): angle of body axes in the x-z plane
    # angle between body z (or x) and world z projected: use atan2 of R[0,2], R[2,2] mod 90
    roll_y = np.degrees(np.arctan2(R[..., 0, 2], R[..., 2, 2]))
    roll_y_mod = np.abs(((roll_y + 45) % 90) - 45)       # distance to nearest face-flat about y
    # rotation about world x (would tip toward a robot)
    roll_x = np.degrees(np.arctan2(R[..., 1, 2], R[..., 2, 2]))
    roll_x_mod = np.abs(((roll_x + 45) % 90) - 45)
    lowx = np.abs(low[..., 0])
    r = {"face_tilt_recompute_max_err_deg": round(float(err), 4),
         "held_steps": int(held.sum()),
         "held_lowest_corner_z": pct(low[..., 2][held]),
         "held_lowest_corner_abs_x": pct(lowx[held]),
         "held_lowest_corner_over": {"plinth(|x|<0.13)": round(float((lowx[held] < PLINTH_HALF).mean()), 4),
                                     "gap(0.13-0.17)": round(float(((lowx[held] >= PLINTH_HALF) & (lowx[held] < DECK_EDGE)).mean()), 4),
                                     "deck(>=0.17)": round(float((lowx[held] >= DECK_EDGE).mean()), 4)},
         "held_lowest_corner_minus_support_top_cm": None,
         "held_tilt_about_pinch_axis_deg": pct(roll_y_mod[held]),
         "held_tilt_about_x_deg": pct(roll_x_mod[held]),
         "held_centre_z": pct(zc[held]),
         "flat_seated_centre_would_be": 0.57,
         }
    sup = np.where(lowx >= DECK_EDGE, DECK_TOP, np.where(lowx < PLINTH_HALF, PLINTH_TOP, np.nan))
    gap = (sup[held] - 0.0)
    d = (low[..., 2] - sup)[held]
    r["held_lowest_corner_minus_support_top_cm"] = pct(100 * d)
    r["held_share_resting(lowest corner within 1 cm of a support top)"] = round(float((np.abs(d) < 0.01).mean()), 4)
    r["held_share_airborne(lowest corner > 1 cm above support or over gap)"] = round(float(((d > 0.01) | ~np.isfinite(d)).mean()), 4)
    # a flat cube would be seatable: share of held steps with tilt < 8 deg (seat window reachable)
    r["held_share_tilt_lt_8deg"] = round(float((z["cube_face_tilt_deg"][held] < 8).mean()), 4)
    out[s] = r
    print(s, json.dumps(r, indent=1))
json.dump(out, open("stand3_tilt_support.json", "w"), indent=1)
