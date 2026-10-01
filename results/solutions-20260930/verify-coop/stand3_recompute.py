"""Independent recomputation of the Stand3 'lift reward farmed by rolling' claim from the 21470828 traces.
lifting_object (task_v2_env_cfg.py:582 -> stand_mdp.gated_object_is_lifted -> coop_lift_mdp.object_is_lifted)
 = 1[cube centre z > object_spawn_z(0.55) + minimal_height] x (1 - tanh(d_pinch/0.12)) x upright_gate.
minimal_height is set by lift_height_curriculum: step 0.02 from 0.04, Stand3 cap 0.06 -> only 0.04 or 0.06.
This script evaluates the HEIGHT clause only (the trace has no pinch distance or gate value)."""
import json, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
SRC = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/repo-gpu-20260923/stand3_replay_2026-09-28/21470828/"
H = 0.14; SPAWN = 0.55; PL_HALF = 0.13; PL_TOP = 0.41; DK_LO, DK_HI, DK_Y, DK_TOP = 0.17, 0.47, 0.15, 0.43

def R_xyzw(q):
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3))
    R[..., 0, 0] = 1 - 2 * (y*y + z*z); R[..., 0, 1] = 2 * (x*y - z*w); R[..., 0, 2] = 2 * (x*z + y*w)
    R[..., 1, 0] = 2 * (x*y + z*w); R[..., 1, 1] = 1 - 2 * (x*x + z*z); R[..., 1, 2] = 2 * (y*z - x*w)
    R[..., 2, 0] = 2 * (x*z - y*w); R[..., 2, 1] = 2 * (y*z + x*w); R[..., 2, 2] = 1 - 2 * (x*x + y*y)
    return R

def pct(a, qs=(5, 50, 95)):
    a = np.asarray(a, float); a = a[np.isfinite(a)]
    return None if a.size == 0 else {**{f"p{q}": round(float(np.percentile(a, q)), 4) for q in qs}, "n": int(a.size)}

out = {}
for s, inv_thr in (("s0", 0.6055), ("s1", 0.6083)):
    z = np.load(SRC + f"{s}.trace.npz")
    v = z["valid"]; p = z["cube_pos"].astype(float); R = R_xyzw(z["cube_quat_raw"].astype(float))
    tilt = z["cube_face_tilt_deg"].astype(float)
    signs = np.array([[a, b, c] for a in (-1, 1) for b in (-1, 1) for c in (-1, 1)], float) * H
    C = p[..., None, :] + np.einsum("teij,kj->teki", R, signs)          # (T,E,8,3) world corners
    cx, cy, cz = np.abs(C[..., 0]), np.abs(C[..., 1]), C[..., 2]
    sup = np.where((cx <= PL_HALF) & (cy <= PL_HALF), PL_TOP,
          np.where((cx >= DK_LO) & (cx <= DK_HI) & (cy <= DK_Y), DK_TOP, 0.0))
    clr = (cz - sup).min(-1)                    # min over corners of (corner z - support top under that corner)
    lowz = cz.min(-1)
    # centre rise that a pure re-orientation would give with the lowest corner resting on the support under it
    rise_orient = (p[..., 2] - lowz)            # centre height above the lowest corner
    r = {"valid_steps": int(v.sum())}
    # robots' line (pinch axis) from base positions at valid steps
    ba, bb = z["base_pos_a"].astype(float), z["base_pos_b"].astype(float)
    d = (bb - ba)[v]; d[:, 2] = 0
    r["robot_a_to_b_unit_xy_median"] = [round(float(x), 3) for x in np.median(d[:, :2] / np.linalg.norm(d[:, :2], axis=1, keepdims=True), axis=0)]
    for name, thr in (("h0.04_thr0.59", 0.59), ("investigator_thr", inv_thr), ("h0.06_thr0.61", 0.61)):
        m = v & (p[..., 2] > thr)
        rr = {"thr_z": thr, "share_of_valid": round(float(m[v].mean()), 4), "steps": int(m.sum())}
        if m.any():
            rr["tilt_deg"] = pct(tilt[m])
            rr["share_tilt_gt_15"] = round(float((tilt[m] > 15).mean()), 4)
            rr["share_resting(<1cm clearance)"] = round(float((clr[m] < 0.01).mean()), 4)
            rr["share_airborne(>2cm clearance)"] = round(float((clr[m] > 0.02).mean()), 4)
            rr["lowest_corner_z"] = pct(lowz[m])
            rr["share_lowest_corner_on_deck_top(0.42-0.44)"] = round(float(((lowz[m] > 0.42) & (lowz[m] < 0.44)).mean()), 4)
            rr["share_lowest_corner_on_plinth_top(0.40-0.42)"] = round(float(((lowz[m] > 0.40) & (lowz[m] <= 0.42)).mean()), 4)
            # would a FLAT cube (tilt<8) clear this threshold?  (actual lift)
            rr["share_flat_lt8deg"] = round(float((tilt[m] < 8).mean()), 4)
            rr["centre_minus_lowest_corner_m"] = pct(rise_orient[m])
        r[name] = rr
    # per-episode: face changed? max tilt; rotation axis at max tilt relative to robot line
    ep = z["ep_idx"]; T, E = v.shape
    ups = []; n_ep = 0; n_face = 0; n_tilt30 = 0; ax_pinch = []
    upax = z["cube_up_axis"]
    for e in range(E):
        for k in np.unique(ep[:, e][v[:, e]]):
            idx = np.where(v[:, e] & (ep[:, e] == k))[0]
            if idx.size == 0: continue
            n_ep += 1
            if (upax[idx, e] != upax[idx[0], e]).any(): n_face += 1
            if np.nanmax(tilt[idx, e]) > 30: n_tilt30 += 1
            # axis of rotation at the max-tilt step: the up-most body axis u (sign to +z); tilt axis = z x u (horizontal)
            j = idx[np.nanargmax(tilt[idx, e])]
            Rm = R[j, e]; col = np.argmax(np.abs(Rm[2, :])); u = Rm[:, col] * np.sign(Rm[2, col])
            axis = np.cross([0, 0, 1.0], u); nrm = np.linalg.norm(axis[:2])
            dl = (bb[j, e] - ba[j, e]); dl[2] = 0; dl /= max(np.linalg.norm(dl), 1e-9)
            if nrm > 1e-6:
                ax_pinch.append(abs(float(np.dot(axis[:2] / nrm, dl[:2]))))   # |cos| between tilt axis and robot line
    r["episodes"] = n_ep
    r["episodes_face_changed"] = n_face
    r["episodes_max_tilt_gt30"] = n_tilt30
    a = np.array(ax_pinch)
    r["at_max_tilt_|cos(tilt_axis, robot_a->b line)|"] = pct(a, qs=(10, 50, 90))
    r["share_eps_tilt_axis_within_30deg_of_robot_line"] = round(float((a > np.cos(np.radians(30))).mean()), 4)
    out[s] = r
    print("=====", s); print(json.dumps(r, indent=1))
json.dump(out, open("/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/verify-coop/stand3_recompute.json", "w"), indent=1)
