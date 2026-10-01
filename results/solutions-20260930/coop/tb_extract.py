import sys, json, os
os.environ.setdefault("OPENBLAS_NUM_THREADS","1")
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import numpy as np
RUNS = {"s0": "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2/2026-09-27_17-56-20_v2-cubetoshelfstand3-blind-s0",
        "s1": "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2/2026-09-27_20-03-28_v2-cubetoshelfstand3-blind-s1"}
out = {}
for k, d in RUNS.items():
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    want = [t for t in tags if any(s in t for s in ("success", "lift_height", "over_deck", "cube_abs_x", "placed", "lifting_object", "lift_progress", "deck_progress", "fallen", "time_out", "mean_episode_length", "stage_lift", "opposing_clamp", "reaching_fine", "upright_gate", "mean_reward", "action_std", "Policy/mean_noise_std"))]
    res = {"all_tags": tags}
    for t in want:
        ev = ea.Scalars(t)
        steps = np.array([e.step for e in ev]); vals = np.array([e.value for e in ev])
        # windowed means
        wins = {}
        for a in (0, 200, 500, 1000, 1500, 2000, 3000, 4000, 5000, 6000, 7000, 7800):
            m = (steps >= a) & (steps < a + 200)
            if m.any():
                wins[a] = round(float(vals[m].mean()), 4)
        res[t] = {"n": len(ev), "first": [int(steps[0]), float(vals[0])], "last": [int(steps[-1]), float(vals[-1])],
                  "max": [int(steps[vals.argmax()]), float(vals.max())], "win200_mean_from": wins}
    out[k] = res
json.dump(out, open("tb_stand3.json", "w"), indent=1)
for k, r in out.items():
    print("=====", k)
    for t, v in r.items():
        if t == "all_tags": continue
        print(t, v["win200_mean_from"], "max", v["max"], "last", v["last"])
