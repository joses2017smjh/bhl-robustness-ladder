"""Dump every scalar of the turning-arm TensorBoard runs to one pickle (read-only on the repo)."""
import glob
import os
import pickle

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

ROOT = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tb_scalars.pkl")

runs = sorted(glob.glob(os.path.join(ROOT, "*arms-turn-*")) + glob.glob(os.path.join(ROOT, "*arms-dr1.0-s0")))
data = {}
for d in runs:
    name = os.path.basename(d).split("_", 2)[-1]
    if "smoke" in name:
        continue
    ea = EventAccumulator(d, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags()["scalars"]
    data[name] = {t: [(e.step, e.value) for e in ea.Scalars(t)] for t in tags}
    print(name, len(tags), len(data[name].get("Train/mean_reward", [])))
with open(OUT, "wb") as f:
    pickle.dump(data, f)
print("wrote", OUT)
