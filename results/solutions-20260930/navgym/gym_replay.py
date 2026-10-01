"""Diagnostic: the NavGym v4 gate-passing actors on the SAME 12 maze layouts the physics transfer
used (maze seeds 50000-50011, 6x6, 1 extra opening), inside the 2-D gym. Already-scored set: this is
a diagnosis of the recorded NEGATIVE, it changes no verdict."""
import json, math, sys
import numpy as np
import onnxruntime as ort
sys.path.insert(0, "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder/src")
from bhl_robust.navgym import env as ng

REPO = "/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder"
ACTORS = {a: f"{REPO}/results/navgym-v4-20260928/{a}/actor.onnx" for a in ("armV4-s5", "armV4-s6")}


def run(sess, keys, env, obs, max_steps=None):
    steps = max_steps or env.max_steps
    info = {}
    wz_sat, flips, prev_w = 0, 0, 0.0
    for t in range(steps):
        feeds = {k: np.asarray(obs[k], dtype=np.float32)[None] for k in keys}
        a = np.clip(sess.run(None, feeds)[0][0], -1, 1)
        wz_sat += abs(a[1]) >= 0.99
        flips += (a[1] * prev_w) < 0
        prev_w = a[1]
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    return info.get("outcome", "time_out" if not term else "?"), t + 1, wz_sat / (t + 1), flips / ((t + 1) * ng.DT)


out = {}
for name, path in ACTORS.items():
    so = ort.SessionOptions(); so.intra_op_num_threads = 1
    sess = ort.InferenceSession(path, so)
    keys = [i.name for i in sess.get_inputs()]
    rows = []
    for k in range(12):
        seed = 50000 + k
        res = []
        # (a) five randomized-dynamics draws, random initial heading (the gym's own protocol)
        for r in range(5):
            env = ng.MazeNavEnv(sizes=((6, 6),), randomize_dynamics=True, seed_base=seed, seed_span=1, version=2)
            obs, _ = env.reset(seed=seed * 10 + r)
            res.append(run(sess, keys, env, obs))
        # (b) nominal dynamics, heading 0 like the physics start, physics time limit (180 s = 4500 steps)
        env = ng.MazeNavEnv(sizes=((6, 6),), randomize_dynamics=False, seed_base=seed, seed_span=1, version=2,
                            max_steps=4500)
        obs, _ = env.reset(seed=seed)
        env.yaw = 0.0
        env._scan()
        obs = env._obs()
        nominal = run(sess, keys, env, obs)
        rows.append({"maze_seed": seed, "random_dyn_goals": sum(o == "goal" for o, *_ in res), "outcomes": [o for o, *_ in res],
                     "nominal_yaw0": nominal[0], "nominal_steps": nominal[1],
                     "wz_sat": round(float(np.mean([x[2] for x in res])), 2), "flips_per_s": round(float(np.mean([x[3] for x in res])), 2)})
        print(name, rows[-1], flush=True)
    out[name] = rows
json.dump(out, open(sys.argv[1], "w"), indent=1)
