"""Read-only: (a) recorded door-activation times vs stage timing; (b) post-hand-back falls: replay-vs-recording
heading offset and world-angle error of the recorded command, at hand-back and in the 2 s before the fall."""
import json, pickle
import numpy as np
SRC = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/mission7-replay-smoke-20260921/fullroute/legacy-doors.json"
src = json.load(open(SRC))
eps = src["episodes"]
ref = {i: e for i, e in enumerate(eps)}
def wrap(a): return float(np.arctan2(np.sin(a), np.cos(a)))
print("source episodes", len(eps), "falls", sum(1 for e in eps if e.get("failure") == "fall" or e.get("fell")))
LAB = ["gref_21397732", "v2align_21405537", "v3s190_21405539", "v4s080_21405541", "v3s190a_21405543", "v4s080a_21405545"]
for lab in LAB:
    d = pickle.load(open(f"{lab}.pkl", "rb"))
    for ep in d["episodes"]:
        idx = ep["layout_index"]
        r = ref[idx]
        act = r["activation_s"]
        hist = ep["stage_history"]
        appr = [h["time_s"] for h in hist if h["phase"] == "approach"]
        cross = [h["time_s"] for h in hist if h["phase"] == "cross"]
        hb = [h["time_s"] for h in hist if h["phase"] == "recorded"]
        if lab == "gref_21397732" or lab == "v2align_21405537":
            print(f"{lab[:8]} L{idx:2d} activation_s={act} approach={[round(x,2) for x in appr]} cross={[round(x,2) for x in cross]} "
                  f"handback={[round(x,2) for x in hb]} door0_open_at_cross_start={bool(cross) and act[0] is not None and act[0] <= cross[0]+1e-8} "
                  f"door0_open_at_takeover={bool(appr) and act[0] is not None and act[0] <= appr[0]+1e-8}")
        ff = ep["first_fall_s"]
        if ff is None or not hb or ff <= hb[0]:
            continue
        # post-stage fall
        t = np.array(ep["t"]); yaw = np.array(ep["yaw"]); rec = np.array(ep["rec"])
        tr = r["diagnostic_trace"]
        tt = np.array([x["time_s"] for x in tr]); ryaw = np.array([x["yaw"] for x in tr])
        assert len(tt) == len(t) and np.allclose(tt, t)
        i_hb = int(np.searchsorted(t, hb[0] - 1e-9)); i_f = int(np.searchsorted(t, ff - 1e-9))
        def angerr(i0, i1):
            out = []
            for i in range(i0, i1):
                c = rec[i, :2]
                if np.linalg.norm(c) < .1: continue
                a1 = np.arctan2(c[1], c[0]) + yaw[i]; a2 = np.arctan2(c[1], c[0]) + ryaw[i]
                out.append(abs(wrap(a1 - a2)))
            return None if not out else round(float(np.degrees(np.median(out))), 1)
        print(f"POSTFALL {lab[:8]} L{idx:2d} handback={hb[0]:.2f} fall={ff:.2f} dyaw@hb={np.degrees(wrap(yaw[i_hb]-ryaw[i_hb])):+.1f}deg "
              f"median cmd angle err: first4s={angerr(i_hb, min(i_f, i_hb+100))} last2s_before_fall={angerr(max(i_hb, i_f-50), i_f)}")
