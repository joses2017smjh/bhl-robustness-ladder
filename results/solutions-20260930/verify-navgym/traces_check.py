"""verify-navgym: shuttle count on 50009, a revisit-fraction attempt on the failure traces, the settle
duration, and the gym-collision -> physics-outcome cross-tab. Reads saved JSONs only."""
import json, sys
import numpy as np
RES = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/results/navgym-v4-transfer-20260930"
GYM = json.load(open("/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/navgym/gym_replay.json"))
out = {}
def load(a, s): return json.load(open(f"{RES}/{a}/seed{s}.json"))
# settle duration
d = load("armV4-s6", 50009)
settle = [t["t"] for t in d["trace"] if t["state"] == "settle"]
print("settle samples:", len(settle), "last settle t:", settle[-1] if settle else None)
out["settle_last_t"] = settle[-1] if settle else None
# s6 shuttle on 50009: one-way traverses of the bottom corridor between x<1.0 (start cell) and x>6.0 (pocket side)
xy = np.array([t["xy"] for t in d["trace"]]); tt = np.array([t["t"] for t in d["trace"]])
state, legs, times = None, 0, []
for (x, y), t in zip(xy, tt):
    z = "W" if x < 1.0 else ("E" if x > 6.0 else None)
    if z and z != state:
        if state is not None:
            legs += 1; times.append((z, t))
        state = z
print("s6 50009: one-way corridor traverses (x<1.0 <-> x>6.0):", legs, "arrivals:", times, "| ymax", xy[:, 1].max(), "| reached pocket (5,1) [y>0.7, x>6.3]:", bool(np.any((xy[:, 0] > 6.3) & (xy[:, 1] > 0.7))))
out["s6_50009_corridor_traverses"] = legs
# s5 50009 physics extent
d5 = load("armV4-s5", 50009); xy5 = np.array([t["xy"] for t in d5["trace"]])
print("s5 50009 physics: xmax %.2f, ymax %.2f, ever x>6.3: %s" % (xy5[:, 0].max(), xy5[:, 1].max(), bool(np.any(xy5[:, 0] > 6.3))))
# revisit fraction attempts on the 7 failures (definitions are mine; the report's definition is not recorded)
fails = {"armV4-s5": [50000, 50002, 50009, 50010], "armV4-s6": [50001, 50002, 50009]}
rv = {}
for a, seeds in fails.items():
    for s in seeds:
        dd = load(a, s)
        tr = [t for t in dd["trace"] if t["state"] != "settle"]
        p = np.array([t["xy"] for t in tr]); T = np.array([t["t"] for t in tr])
        # A: fraction of samples within 0.3 m of any sample >= 10 s earlier
        A = np.mean([bool(np.any((np.hypot(*(p[:i] - p[i]).T) < 0.3) & (T[:i] <= T[i] - 10.0))) if i else False for i in range(len(p))])
        # B: fraction of samples in a maze cell that was entered before and left (cell re-entry)
        cells = [(int(round(x / 1.4)), int(round(y / 1.4))) for x, y in p]
        entries, prev, flags = {}, None, []
        for c in cells:
            if c != prev:
                entries[c] = entries.get(c, 0) + 1
            flags.append(entries[c] > 1)          # this sample is in a cell entered more than once so far
            prev = c
        B = float(np.mean(flags))
        # C: 1 - (distinct 0.2 m grid cells visited / samples moved > 0.02 m)
        g = {(int(np.floor(x / 0.2)), int(np.floor(y / 0.2))) for x, y in p}
        C = 1 - len(g) / max(1, len(p))
        rv[f"{a}/{s}"] = {"A_within0.3m_of_>=10s_earlier": round(float(A), 3), "B_cell_reentry": round(B, 3), "C_1-distinct0.2cells/samples": round(C, 3),
                          "path_m": dd["path_length_m"], "stall_longest_s": dd["stall"]["longest_s"], "braked_steps": dd["sensor_stats"]["braked_robot_steps"]}
        print(a, s, rv[f"{a}/{s}"])
out["revisit_attempts"] = rv
# gym collisions (random-dyn draws) vs physics outcome
xt = []
for a, rows in GYM.items():
    for r in rows:
        nc = sum(o == "collision" for o in r["outcomes"])
        if nc:
            dd = load(a, r["maze_seed"])
            xt.append({"arm": a, "seed": r["maze_seed"], "gym_collisions_of_5": nc, "gym_nominal": r["nominal_yaw0"], "physics": dd["outcome_class"],
                       "physics_completion_s": dd["completion_s"], "braked_steps_of_4500": dd["sensor_stats"]["braked_robot_steps"]})
            print("gym-collision cross-tab:", xt[-1])
out["gym_collision_vs_physics"] = xt
json.dump(out, open(sys.argv[1], "w"), indent=1)
