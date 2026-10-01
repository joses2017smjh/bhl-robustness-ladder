"""Predictive check: the v4 actors in the gym with physics-identified pieces swapped in (sysid_env.SysidNavEnv), on
TRAINING-RANGE 6x6 mazes only (maze seeds < 10 000), deterministic clipped mean as deployed.

usage: eval_sysid_gym.py <config> <arm> <maze_start> <n> [out_tag]
"""
import json, math, os, sys, time
for _k in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_k, "1")
from pathlib import Path
REPO = Path("/nfs/hpc/share/sanchej7/Humanoid_Lite/bhl-robustness-ladder")
sys.path.insert(0, str(REPO / "src"))
import numpy as np
from gym_actor_stats import load_actor, run_lengths
from sysid_env import SysidNavEnv, IdentifiedDynamics, PhysDynamics, LowPassW, MovingAvgW, RateLimitW
from bhl_robust.navgym.env import DT

OUT = Path(__file__).resolve().parent / "data" / "gymcheck"
OUT.mkdir(parents=True, exist_ok=True)


# ---- the identified physics model (filled from the sysid fits; see report). Kept in ONE place.
def w_in_phys(w_cmd, hist):
    return w_cmd            # replaced below once the bang-bang data are analysed


PHYS = dict(Kw=1.05, tau_w=0.04, Lw=0, Kv=1.0, tau_v=0.13, Lv=0, drift=0.0, w_noise=0.0, v_noise=0.0)


def make_config(name):
    """name -> SysidNavEnv kwargs. Components: dyn (identified dynamics), brake, obs10hz (+mismatch), fix (capture fix),
    h0 (heading 0), settle (25-step settle), and deployment filters lpfXX / maN / rlX."""
    parts = name.split("+")
    kw = {}
    for p in parts:
        if p == "base":
            pass
        elif p == "dyn":
            kw["dyn_model"] = lambda rng: PhysDynamics(rng)
        elif p == "dynlin":      # fast linear yaw only (no standing dead zone, no v map)
            kw["dyn_model"] = lambda rng: PhysDynamics(rng, standing_deadzone=False, v_map=False, chatter_v=1.0)
        elif p == "brake":
            kw["brake"] = True
        elif p == "brakeR":      # relaxed brake: full stop at 0.30 m, free above 0.50 m
            kw["brake"] = True; kw["brake_lo"] = 0.30; kw["brake_span"] = 0.20
        elif p == "obs10":
            kw["obs10hz"] = True
        elif p == "fix":
            kw["capture_fix"] = True
        elif p == "h0":
            kw["heading0"] = True
        elif p == "settle":
            kw["settle_steps"] = 25
        elif p.startswith("lpf"):
            kw["cmd_filter"] = LowPassW(tau_f=float(p[3:]))
        elif p.startswith("ma"):
            kw["cmd_filter"] = MovingAvgW(n=int(p[2:]))
        elif p.startswith("rl"):
            kw["cmd_filter"] = RateLimitW(rate=float(p[2:]))
        else:
            raise ValueError(p)
    return kw


PHYS_DYN = dict(PHYS)


def run(config, arm, maze_start, n, tag=""):
    act = load_actor(REPO / f"results/navgym-v4-20260928/{arm}/actor.onnx")
    env = SysidNavEnv(sizes=((6, 6),), extra_openings=1, gamma=0.998, **make_config(config))
    rows = []
    t0 = time.time()
    for s in range(maze_start, maze_start + n):
        assert s < 10_000, "training-range maze seeds only"
        env.seed_base, env.seed_span = s, 1
        obs, info = env.reset(seed=s)
        A = []
        xy = []
        while True:
            a = act(obs)
            obs, r, term, trunc, inf = env.step(a)
            A.append(a); xy.append((env.x, env.y))
            if term or trunc:
                break
        A = np.array(A); xy = np.array(xy)
        sg = np.sign(A[:, 1]); flips = int(np.sum(sg[1:] * sg[:-1] < 0))
        # loops: fraction of time the robot is within 0.3 m of a position it occupied >= 10 s earlier
        rows.append({"maze_seed": s, "outcome": inf["outcome"], "steps": int(env.t), "max_steps": int(env.max_steps),
                     "path_m": float(np.sum(np.linalg.norm(np.diff(xy, axis=0), axis=1))), "route_m": float(env.route_m),
                     "flips_per_s": flips / (len(A) * DT), "vx_zero_frac": float(np.mean(A[:, 0] <= -0.9)),
                     "braked_frac": env.stats["braked"] / max(1, env.stats["steps"]),
                     "mean_brake_scale": env.stats["brake_scale_sum"] / max(1, env.stats["steps"]) if env.brake else None})
    oc = [r["outcome"] for r in rows]
    summ = {"config": config, "arm": arm, "maze_seeds": [maze_start, maze_start + n - 1], "n": n,
            "success": oc.count("goal"), "collision": oc.count("collision"), "time_out": oc.count("time_out"),
            "median_steps_success": float(np.median([r["steps"] for r in rows if r["outcome"] == "goal"])) if "goal" in oc else None,
            "mean_path_over_route_success": float(np.mean([r["path_m"] / r["route_m"] for r in rows if r["outcome"] == "goal"])) if "goal" in oc else None,
            "mean_braked_frac": float(np.mean([r["braked_frac"] for r in rows])), "mean_flips_per_s": float(np.mean([r["flips_per_s"] for r in rows])),
            "wall_s": round(time.time() - t0, 1), "phys_dyn": "PhysDynamics (sysid_env.py)" if any(x.startswith("dyn") for x in config.split("+")) else None}
    (OUT / f"{config}__{arm}__{maze_start}_{n}{tag}.json").write_text(json.dumps({"summary": summ, "episodes": rows}, indent=1, default=str))
    print(json.dumps(summ, default=str), flush=True)
    return summ


if __name__ == "__main__":
    config, arm, ms, n = sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4])
    tag = sys.argv[5] if len(sys.argv) > 5 else ""
    run(config, arm, ms, n, tag)
