"""Is Curriculum/lift_height a mixture of the two discrete levels {0.04, 0.06}? (Stand3 cap 0.06, step 0.02)."""
import json, os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
B = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite/logs/rsl_rl/task_v2/"
RUNS = {"s0": B + "2026-09-27_17-56-20_v2-cubetoshelfstand3-blind-s0", "s1": B + "2026-09-27_20-03-28_v2-cubetoshelfstand3-blind-s1"}
out = {}
for k, d in RUNS.items():
    ea = EventAccumulator(d, size_guidance={"scalars": 0}); ea.Reload()
    r = {}
    for tag in ("Curriculum/lift_height", "Episode_Reward/lifting_object", "Episode_Termination/success", "Curriculum/tipped_over_deck", "Curriculum/over_deck"):
        if tag not in ea.Tags()["scalars"]:
            continue
        ev = ea.Scalars(tag)
        st = np.array([e.step for e in ev]); v = np.array([e.value for e in ev], float)
        last = st >= st.max() - 199
        r[tag] = {"last200_mean": round(float(v[last].mean()), 5), "max": round(float(v.max()), 5), "min": round(float(v.min()), 5)}
        if tag == "Curriculum/lift_height":
            frac = (v - 0.04) / 0.02           # share at the 0.06 level if a 2-level mixture
            k24 = frac * 24
            r[tag]["values_are_k_over_24_mixtures_share"] = round(float((np.abs(k24 - np.round(k24)) < 0.02).mean()), 4)
            r[tag]["last200_share_at_0.06_level"] = round(float(frac[last].mean()), 4)
            r[tag]["last200_iters_exactly_0.06"] = int((np.abs(v[last] - 0.06) < 1e-6).sum())
            r[tag]["last200_iters_exactly_0.04"] = int((np.abs(v[last] - 0.04) < 1e-6).sum())
            r[tag]["first_iter_reaching_0.06"] = int(st[np.argmax(v >= 0.06 - 1e-6)]) if (v >= 0.06 - 1e-6).any() else None
            r[tag]["n_distinct_last200"] = int(len(np.unique(np.round(v[last], 6))))
            r[tag]["sample_last10"] = [round(float(x), 6) for x in v[-10:]]
    out[k] = r
    print(k, json.dumps(r, indent=1))
json.dump(out, open("/nfs/hpc/share/sanchej7/Humanoid_Lite/solutions-20260930/verify-coop/tb_lift_levels.json", "w"), indent=1)
