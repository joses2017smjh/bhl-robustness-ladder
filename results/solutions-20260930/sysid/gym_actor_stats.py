"""Full-rate (every 0.04 s step) action statistics of the NavGym v4 actors IN THE GYM, on training-range
6x6 mazes (maze seeds < 10 000 only). Deterministic clipped mean, exactly as maze_explore.py --policy runs it."""
import json, math, os, sys, time
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
from pathlib import Path
REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
sys.path.insert(0, str(REPO / "src"))
import numpy as np
import onnxruntime as ort
from bhl_robust.navgym.env import MazeNavEnv, DT

OUT = Path(__file__).resolve().parent / "data"
OUT.mkdir(exist_ok=True)


def load_actor(path):
    so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    keys = tuple(i.name for i in sess.get_inputs())
    def act(obs):
        feeds = {k: np.asarray(obs[k], dtype=np.float32)[None] for k in keys}
        return np.clip(sess.run(None, feeds)[0][0], -1.0, 1.0).astype(np.float32)
    return act


def run_lengths(sign):
    """lengths of runs of constant nonzero sign"""
    out, cur, n = [], None, 0
    for s in sign:
        if s == cur:
            n += 1
        else:
            if cur is not None and cur != 0:
                out.append(n)
            cur, n = s, 1
    if cur is not None and cur != 0:
        out.append(n)
    return out


def episode(env, act, maze_seed, dyn_seed):
    env.seed_base, env.seed_span = int(maze_seed), 1
    obs, info = env.reset(seed=int(dyn_seed))
    A, P = [], []
    while True:
        a = act(obs)
        obs, r, term, trunc, inf = env.step(a)
        A.append(a.copy()); P.append((env.x, env.y, env.yaw, env.v, env.w))
        if term or trunc:
            return inf["outcome"], np.array(A), np.array(P), env.dyn


if __name__ == "__main__":
    arms = sys.argv[1].split(",") if len(sys.argv) > 1 else ["armV4-s5", "armV4-s6"]
    seeds = [int(s) for s in sys.argv[2].split(",")] if len(sys.argv) > 2 else list(range(9000, 9006))
    res = {}
    for arm in arms:
        act = load_actor(REPO / f"results/navgym-v4-20260928/{arm}/actor.onnx")
        env = MazeNavEnv(sizes=((6, 6),), extra_openings=1, version=2, gamma=0.998)
        rows = []
        allA = []
        for s in seeds:
            assert s < 10_000
            t0 = time.time()
            outcome, A, P, dyn = episode(env, act, s, s)
            wz = A[:, 1]; sg = np.sign(np.where(np.abs(wz) < 1e-6, 0, wz))
            flips = int(np.sum(sg[1:] * sg[:-1] < 0))
            rl = run_lengths(sg)
            rows.append({"maze_seed": s, "outcome": outcome, "steps": len(A), "wall_s": round(time.time() - t0, 1),
                         "dyn": {k: round(float(getattr(dyn, k)), 3) for k in ("w_gain", "v_gain", "tau", "drift", "latency", "v_noise", "w_noise")},
                         "wz_sat_frac": round(float(np.mean(np.abs(wz) >= 0.999)), 3),
                         "flips_per_s": round(flips / (len(A) * DT), 2), "p_flip_per_step": round(flips / max(1, len(A) - 1), 3),
                         "median_dwell_steps": float(np.median(rl)) if rl else None, "mean_dwell_steps": round(float(np.mean(rl)), 2) if rl else None,
                         "vx_sat_frac": round(float(np.mean(A[:, 0] >= 0.999)), 3), "vx_zero_frac": round(float(np.mean(A[:, 0] <= -0.999)), 3),
                         "abs_w_mean": round(float(np.mean(np.abs(P[:, 4]))), 3)})
            allA.append(A)
            print(arm, rows[-1], flush=True)
            np.savez(OUT / f"gym_actor_{arm}_maze{s}.npz", A=A, P=P)
        A = np.concatenate(allA)
        sg = np.sign(A[:, 1])
        rl = run_lengths(sg)
        hist = np.bincount(np.minimum(rl, 30))
        res[arm] = {"episodes": rows, "pooled": {"steps": int(len(A)), "wz_sat_frac": round(float(np.mean(np.abs(A[:, 1]) >= 0.999)), 3),
                    "p_flip_per_step": round(float(np.sum(sg[1:] * sg[:-1] < 0) / (len(A) - 1)), 3),
                    "dwell_hist_steps(1..30+)": hist.tolist(), "mean_dwell_steps": round(float(np.mean(rl)), 2),
                    "vx_sat_frac": round(float(np.mean(A[:, 0] >= 0.999)), 3)}}
        print(arm, "POOLED", res[arm]["pooled"], flush=True)
    (OUT / f"gym_actor_stats_{'_'.join(arms)}.json").write_text(json.dumps(res, indent=1))
