"""Diagnose why NavGym v2 actors time out on held-out 5x5 mazes.

DIAGNOSIS ONLY. The episodes are the trainer's published held-out evaluation
episodes (maze seed 10 000 + k, randomized dynamics from reset(seed=k),
k = 0..47), i.e. the seeds of the v2 final evaluation. They are therefore NOT
used to score anything designed from this diagnosis: NavGym v3 is scored on
fresh maze seeds 20 000..20 047 (``heldout_env(..., maze_base=20_000)``).

Actors are rebuilt from the PPO zips (final = ppo_final.zip, best =
best/ppo_best.zip, ckpt:<steps> = ckpt/ppo_<steps>_steps.zip) and driven
deterministically by the Gaussian MEAN, clipped to [-1, 1] -- exactly what
``model.predict(deterministic=True)`` and the exported ONNX actor do.

Every episode runs ONCE with a flat limit of ``--extend`` x its own
route-scaled v2 limit (the limit draws no randomness, so the trajectory is the
v2 evaluation trajectory step for step); the outcome is judged at the original
limit, and the extension only says whether/when a time-out would have reached
the goal.

Time-out classes (declared here, applied to the final WINDOW = the last 500
steps = 20 s before the original limit; the first rule that matches wins):

  stalled               path length in the window < 0.30 m (mean speed < 1.5 cm/s)
  stuck_at_wall         >= 50 % of window steps with centre clearance < ROBOT_RADIUS + 0.08 m
                        AND net geodesic progress in the window < 0.50 m
  looping               net geodesic progress in the window < 0.50 m (moving, not closing in)
  slow_but_progressing  net geodesic progress in the window >= 0.50 m

Action statistics (per episode, over the steps up to the original limit):
fraction of steps whose raw mean is outside [-1, 1] per channel (saturated),
median |raw mean|, fraction at full forward speed / at zero forward speed /
at full turn rate, wz sign flips per second, mean |w| (rad/s), the fraction of
steps turning in place (|v| < 0.05 m/s, |w| > 0.5 rad/s), mean geodesic
progress rate (m/s) against the minimum rate the route-scaled limit implies
(route / (limit * DT)), and tortuosity (path length / geodesic progress).

``--dose`` adds a short probe of every checkpoint of the run (the first
``--dose-steps`` steps of ``--dose-episodes`` held-out episodes): its per-channel
policy std from the zip and the fraction of saturated means, beside the
periodic evaluation's 5x5 time-out rate at the same step (eval_history.json,
held-out seeds 10 000..10 023). Std grows monotonically with training time in
v2, so this is a within-run correlation, confounded with training progress.

Usage (one actor set per invocation; ~1.2 ms per step on the login node):
  python scripts/bench/navgym_diagnose.py --run results/navgym-v2-20260926/armA-s0 \\
      --actors final best --out <scratch>/diag_armA-s0.json [--dose]
"""
from __future__ import annotations

import argparse
import io
import json
import math
import sys
import time
import zipfile
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

from bhl_robust.navgym.env import (DT, OBS_KEYS_V2, ROBOT_RADIUS, V_MAX, W_MAX, box_clearance,  # noqa: E402
                                   heldout_env, route_time_limit)

WINDOW = 500                    # steps (20 s) before the original limit
STALL_PATH_M = 0.30
WALL_CLEAR_M = ROBOT_RADIUS + 0.08
WALL_FRAC = 0.5
PROGRESS_M = 0.50
REVISIT_M, REVISIT_GAP = 0.30, 250
CLASSES = ("stalled", "stuck_at_wall", "looping", "slow_but_progressing")


# ------------------------------------------------------------------ actors
def load_policy(zip_path: Path, version: int = 2):
    """The SB3 policy of a navgym_train.py zip, rebuilt with the trainer's architecture."""
    from stable_baselines3.common.policies import MultiInputActorCriticPolicy
    from bhl_robust.navgym.env import MazeNavEnv
    from navgym_train import NavExtractor
    env = MazeNavEnv(sizes=((5, 5),), version=version)
    pol = MultiInputActorCriticPolicy(env.observation_space, env.action_space, lr_schedule=lambda _: 0.0,
                                      features_extractor_class=NavExtractor, features_extractor_kwargs={"features_dim": 256},
                                      net_arch={"pi": [128, 128], "vf": [128, 128]}, share_features_extractor=True)
    with zipfile.ZipFile(zip_path) as zf:
        sd = torch.load(io.BytesIO(zf.read("policy.pth")), map_location="cpu", weights_only=False)
    pol.load_state_dict(sd)
    pol.eval()
    return pol


class Actor:
    """Deterministic actor: raw Gaussian mean (pre-clip) of a policy."""

    def __init__(self, policy, keys=OBS_KEYS_V2):
        from navgym_train import ActorOnnx
        self.net = ActorOnnx(policy, keys).eval()
        self.keys = tuple(keys)
        self.std = [float(s) for s in torch.exp(policy.log_std.detach())]

    def __call__(self, obs: dict) -> np.ndarray:
        with torch.no_grad():
            return self.net(*(torch.as_tensor(np.asarray(obs[k], dtype=np.float32))[None] for k in self.keys))[0].numpy().astype(np.float64)


class OnnxActor:
    """The exported ONNX actor (actor.onnx / actor_best.onnx): the deployable artifact. Its
    raw mean differs from the torch policy by ~1e-5 (float32); --backend onnx measures how
    sensitive episode outcomes are to that."""

    def __init__(self, path: Path, std):
        import onnxruntime as ort
        so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(str(path), so)
        self.keys = tuple(i.name for i in self.sess.get_inputs())
        self.std = std

    def __call__(self, obs: dict) -> np.ndarray:
        return self.sess.run(None, {k: np.asarray(obs[k], np.float32)[None] for k in self.keys})[0][0].astype(np.float64)


def actor_zip(run: Path, name: str) -> Path:
    if name == "final":
        return run / "ppo_final.zip"
    if name == "best":
        return run / "best" / "ppo_best.zip"
    if name.startswith("ckpt:"):
        return run / "ckpt" / f"ppo_{int(name.split(':', 1)[1])}_steps.zip"
    raise ValueError(f"unknown actor {name!r} (final | best | ckpt:<steps>)")


# ------------------------------------------------------------------ episodes
def classify_timeout(path_w: float, progress_w: float, wall_frac_w: float) -> str:
    """The declared time-out classes (module docstring), first match wins."""
    if path_w < STALL_PATH_M:
        return "stalled"
    if wall_frac_w >= WALL_FRAC and progress_w < PROGRESS_M:
        return "stuck_at_wall"
    if progress_w < PROGRESS_M:
        return "looping"
    return "slow_but_progressing"


def run_episode(actor: Actor, size: int, k: int, extend: float, maze_base: int = 10_000, max_steps_cap: int | None = None) -> dict:
    env0, _, info0 = heldout_env((size, size), k, version=2, gamma=1.0, maze_base=maze_base)
    limit = int(info0["max_steps"])
    assert limit == route_time_limit(info0["route_m"])
    ext = int(math.ceil(extend * limit)) if max_steps_cap is None else min(int(math.ceil(extend * limit)), max_steps_cap)
    env, obs, info = heldout_env((size, size), k, version=2, gamma=1.0, max_steps=ext, maze_base=maze_base)
    route = float(info["route_m"])
    X, MU, GEO, CL, V, W = [], [], [], [], [], []
    X.append((env.x, env.y)); GEO.append(env.potential); CL.append(float(box_clearance(env.boxes, env.x, env.y)))
    outcome_ext, t_end = None, None
    while True:
        mu = actor(obs)
        obs, _, term, trunc, st = env.step(np.clip(mu, -1.0, 1.0).astype(np.float32))
        MU.append(mu); V.append(env.v); W.append(env.w)
        X.append((env.x, env.y)); GEO.append(env.potential); CL.append(float(box_clearance(env.boxes, env.x, env.y)))
        if term or trunc:
            outcome_ext, t_end = st["outcome"], int(env.t)
            break
    MU = np.asarray(MU); X = np.asarray(X); GEO = np.asarray(GEO); CL = np.asarray(CL); V = np.asarray(V); W = np.asarray(W)
    # outcome at the ORIGINAL limit
    if outcome_ext in ("goal", "collision") and t_end <= limit:
        outcome = outcome_ext
    else:
        outcome = "time_out"
    T = min(t_end, limit)                        # steps judged
    mu, xs, geo = MU[:T], X[:T + 1], GEO[:T + 1]
    a = np.clip(mu, -1, 1)
    seg = np.linalg.norm(np.diff(xs, axis=0), axis=1)
    path = float(seg.sum())
    progress = float(geo[0] - geo[-1])
    w_sign = np.sign(a[:, 1])
    flips = int(np.sum((w_sign[1:] * w_sign[:-1]) < 0))
    ep = {"k": k, "maze_seed": int(env.episode_seed), "outcome": outcome, "steps": int(T), "limit_steps": limit,
          "route_m": round(route, 3), "min_rate_mps": round(route / (limit * DT), 4),
          "progress_rate_mps": round(progress / (T * DT), 4), "progress_frac": round(progress / max(route, 1e-9), 3),
          "geo_left_m": round(float(geo[-1]), 3), "path_m": round(path, 2), "tortuosity": round(path / max(progress, 0.05), 2),
          "sat_v": round(float(np.mean(np.abs(mu[:, 0]) > 1.0)), 3), "sat_w": round(float(np.mean(np.abs(mu[:, 1]) > 1.0)), 3),
          "med_abs_mu": [round(float(np.median(np.abs(mu[:, 0]))), 2), round(float(np.median(np.abs(mu[:, 1]))), 2)],
          "frac_full_speed": round(float(np.mean(a[:, 0] >= 0.99)), 3), "frac_zero_speed": round(float(np.mean(a[:, 0] <= -0.99)), 3),
          "frac_full_turn": round(float(np.mean(np.abs(a[:, 1]) >= 0.99)), 3), "w_flips_per_s": round(flips / (T * DT), 3),
          "mean_abs_w": round(float(np.mean(np.abs(W[:T]))), 3), "mean_v": round(float(np.mean(V[:T])), 4),
          "frac_turn_in_place": round(float(np.mean((np.abs(V[:T]) < 0.05) & (np.abs(W[:T]) > 0.5))), 3),
          "min_clearance_m": round(float(CL[:T + 1].min()), 3),
          "extended": {"limit_steps": ext, "outcome": outcome_ext, "steps": t_end}}
    d = env.dyn
    ep["dyn"] = {"w_gain": round(d.w_gain, 3), "v_gain": round(d.v_gain, 3), "tau": round(d.tau, 3), "drift": round(d.drift, 4), "latency": d.latency}
    if outcome == "time_out":
        w0 = max(0, T - WINDOW)
        seg_w = seg[w0:T]
        path_w = float(seg_w.sum())
        progress_w = float(geo[w0] - geo[T])
        wall_frac_w = float(np.mean(CL[w0 + 1:T + 1] < WALL_CLEAR_M))
        # revisits: window positions within REVISIT_M of a position >= REVISIT_GAP steps earlier
        rev = 0
        for i in range(w0 + 1, T + 1):
            if i > REVISIT_GAP and np.min(np.linalg.norm(xs[:i - REVISIT_GAP] - xs[i], axis=1)) < REVISIT_M:
                rev += 1
        best = np.minimum.accumulate(geo)
        improved = np.nonzero(best[:-1] - best[1:] > 0)[0]
        # steps since the best-ever geodesic last improved by >= 0.25 m
        last_big = 0
        ref = geo[0]
        for i in range(1, T + 1):
            if ref - geo[i] >= 0.25:
                ref, last_big = geo[i], i
        ep["timeout"] = {"class": classify_timeout(path_w, progress_w, wall_frac_w), "window_steps": int(T - w0),
                         "window_path_m": round(path_w, 3), "window_progress_m": round(progress_w, 3),
                         "window_wall_frac": round(wall_frac_w, 3), "window_revisit_frac": round(rev / max(1, T - w0), 3),
                         "window_sat_w": round(float(np.mean(np.abs(mu[w0:T, 1]) > 1.0)), 3),
                         "window_zero_speed_frac": round(float(np.mean(a[w0:T, 0] <= -0.99)), 3),
                         "window_mean_speed_mps": round(path_w / ((T - w0) * DT), 4),
                         "window_w_flips_per_s": round(float(np.sum((w_sign[w0 + 1:T] * w_sign[w0:T - 1]) < 0)) / ((T - w0) * DT), 3),
                         "steps_since_quarter_m_progress": int(T - last_big), "n_best_improvements": int(improved.size),
                         "reached_in_extension": outcome_ext == "goal",
                         "extension_steps_to_goal_over_limit": round(t_end / limit, 3) if outcome_ext == "goal" else None}
    return ep


def summarize(eps: list) -> dict:
    n = len(eps)
    by = {o: [e for e in eps if e["outcome"] == o] for o in ("goal", "collision", "time_out")}
    out = {"n": n, "success": len(by["goal"]) / n, "collision": len(by["collision"]) / n, "time_out": len(by["time_out"]) / n}
    tos = by["time_out"]
    out["time_out_classes"] = {c: sum(e["timeout"]["class"] == c for e in tos) for c in CLASSES}
    out["time_out_seeds"] = [e["k"] for e in tos]
    out["time_out_reached_in_extension"] = sum(e["timeout"]["reached_in_extension"] for e in tos)
    keys = ("sat_v", "sat_w", "frac_full_speed", "frac_zero_speed", "frac_full_turn", "w_flips_per_s", "mean_abs_w", "mean_v",
            "frac_turn_in_place", "progress_rate_mps", "min_rate_mps", "tortuosity", "progress_frac", "route_m")
    out["by_outcome_median"] = {o: ({k: float(np.median([e[k] for e in R])) for k in keys} if R else None) for o, R in by.items()}
    out["all_median"] = {k: float(np.median([e[k] for e in eps])) for k in keys}
    out["median_abs_mu_all"] = [float(np.median([e["med_abs_mu"][0] for e in eps])), float(np.median([e["med_abs_mu"][1] for e in eps]))]
    return out


def dose_probe(run: Path, n_eps: int, n_steps: int) -> list:
    hist = {r["timesteps"]: r for r in json.loads((run / "eval_history.json").read_text())}
    rows = []
    for z in sorted((run / "ckpt").glob("ppo_*_steps.zip"), key=lambda p: int(p.stem.split("_")[1])):
        steps = int(z.stem.split("_")[1])
        act = Actor(load_policy(z))
        sat_v, sat_w, flips, absmu = [], [], [], []
        for k in range(n_eps):
            env, obs, _ = heldout_env((5, 5), k, version=2, gamma=1.0)
            mus = []
            for _ in range(n_steps):
                mu = act(obs)
                mus.append(mu)
                obs, _, te, tr, _ = env.step(np.clip(mu, -1, 1).astype(np.float32))
                if te or tr:
                    break
            mus = np.asarray(mus)
            sat_v.append(np.mean(np.abs(mus[:, 0]) > 1)); sat_w.append(np.mean(np.abs(mus[:, 1]) > 1))
            s = np.sign(np.clip(mus[:, 1], -1, 1))
            flips.append(np.sum(s[1:] * s[:-1] < 0) / (len(mus) * DT))
            absmu.append(np.median(np.abs(mus), axis=0))
        h = hist.get(steps)
        rows.append({"timesteps": steps, "std": [round(s, 3) for s in act.std], "sat_v": round(float(np.mean(sat_v)), 3),
                     "sat_w": round(float(np.mean(sat_w)), 3), "w_flips_per_s": round(float(np.mean(flips)), 3),
                     "median_abs_mu": [round(float(x), 2) for x in np.median(np.asarray(absmu), axis=0)],
                     "eval_5x5": h["eval"]["5x5"] if h else None, "eval_6x6": h["eval"]["6x6"] if h else None})
        print(f"[dose] {steps / 1e6:4.0f}M std {rows[-1]['std']} sat_w {rows[-1]['sat_w']} flips {rows[-1]['w_flips_per_s']} "
              f"5x5 t/o {h['eval']['5x5']['time_out'] if h else None}", flush=True)
    if len(rows) >= 4:
        from scipy.stats import spearmanr
        st = [float(np.mean(r["std"])) for r in rows if r["eval_5x5"]]
        to = [r["eval_5x5"]["time_out"] for r in rows if r["eval_5x5"]]
        sw = [r["sat_w"] for r in rows if r["eval_5x5"]]
        rho, p = spearmanr(st, to)
        rho2, p2 = spearmanr(st, sw)
        rows.append({"spearman_std_vs_5x5_timeout": [round(float(rho), 3), round(float(p), 4)],
                     "spearman_std_vs_sat_w": [round(float(rho2), 3), round(float(p2), 4)],
                     "note": "std rises monotonically with training time in v2: these correlations are confounded with training progress"})
    return rows


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", type=Path, required=True, help="a navgym_train.py output dir (v2 env)")
    ap.add_argument("--actors", nargs="*", default=["final", "best"], help="final | best | ckpt:<steps>")
    ap.add_argument("--size", type=int, default=5)
    ap.add_argument("--seeds", type=int, default=48, help="held-out k = 0..N-1 (maze seed maze_base + k)")
    ap.add_argument("--maze-base", type=int, default=10_000, help="10 000 = the published v2 eval seeds (diagnosis only)")
    ap.add_argument("--extend", type=float, default=2.0, help="run every episode to this multiple of its route-scaled limit")
    ap.add_argument("--dose", action="store_true", help="also probe every ckpt/ zip (std vs saturation vs periodic 5x5 time-outs)")
    ap.add_argument("--dose-episodes", type=int, default=6)
    ap.add_argument("--dose-steps", type=int, default=400)
    ap.add_argument("--check-onnx", action="store_true", default=True, help="check the rebuilt final/best actor against the exported ONNX")
    ap.add_argument("--backend", choices=("torch", "onnx"), default="torch",
                    help="torch = the policy rebuilt from the zip; onnx = the exported actor (final/best only)")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    torch.set_num_threads(1)
    t0 = time.time()
    res = {"tool": "navgym_diagnose", "run": str(args.run), "backend": args.backend, "size": f"{args.size}x{args.size}",
           "seeds": (f"maze seed {args.maze_base} + k, k = 0..{args.seeds - 1}; dynamics reset(seed=k)"
                     + (" -- the PUBLISHED v2 held-out eval seeds: diagnosis only, never a scored set for anything designed from this"
                        if args.maze_base == 10_000 else "")),
           "deterministic": "raw Gaussian mean clipped to [-1, 1] (= model.predict(deterministic=True) = the ONNX actor)",
           "extend": args.extend,
           "rules": {"window_steps": WINDOW, "stalled": f"window path < {STALL_PATH_M} m",
                     "stuck_at_wall": f"window frac(clearance < {WALL_CLEAR_M:.2f} m) >= {WALL_FRAC} and window progress < {PROGRESS_M} m",
                     "looping": f"window geodesic progress < {PROGRESS_M} m", "slow_but_progressing": f"window geodesic progress >= {PROGRESS_M} m",
                     "order": list(CLASSES)},
           "actors": {}}
    summ = json.loads((args.run / "summary.json").read_text()) if (args.run / "summary.json").is_file() else {}
    for name in args.actors:
        z = actor_zip(args.run, name)
        pol = load_policy(z)
        act = Actor(pol)
        entry = {"zip": str(z), "std": act.std}
        onnx = {"final": args.run / "actor.onnx", "best": args.run / "actor_best.onnx"}.get(name)
        if args.check_onnx and onnx is not None and onnx.is_file():
            import onnxruntime as ort
            so = ort.SessionOptions(); so.intra_op_num_threads = 1; so.inter_op_num_threads = 1
            sess = ort.InferenceSession(str(onnx), so)
            _, obs, _ = heldout_env((args.size, args.size), 0, version=2, gamma=1.0)
            ref = sess.run(None, {k: np.asarray(obs[k], np.float32)[None] for k in act.keys})[0][0]
            entry["onnx_max_abs_diff"] = float(np.max(np.abs(ref - act(obs))))
        if args.backend == "onnx":
            if onnx is None or not onnx.is_file():
                raise SystemExit(f"--backend onnx needs an exported actor for {name}")
            act = OnnxActor(onnx, act.std)
            entry["backend"] = f"onnx:{onnx}"
        else:
            entry["backend"] = "torch (rebuilt from the zip)"
        te = time.time()
        eps = [run_episode(act, args.size, k, args.extend, args.maze_base) for k in range(args.seeds)]
        entry["summary"] = summarize(eps)
        entry["episodes"] = eps
        entry["wall_s"] = round(time.time() - te, 1)
        # agreement with the trainer's own 48-maze evaluation (same episodes, same limit)
        pub = (summ.get("final_eval_heldout_48") if name == "final" else (summ.get("best") or {}).get("final_eval_heldout_48")) or {}
        key = f"{args.size}x{args.size}"
        if key in pub and args.maze_base == 10_000 and args.seeds == 48:
            entry["published"] = pub[key]
            entry["matches_published"] = all(abs(entry["summary"][m] - pub[key][m]) < 1e-9 for m in ("success", "collision", "time_out"))
        res["actors"][name] = entry
        s = entry["summary"]
        print(f"[{name}] std {[round(x, 2) for x in act.std]} | success {s['success']:.3f} coll {s['collision']:.3f} t/o {s['time_out']:.3f} "
              f"| classes {s['time_out_classes']} | t/o reached in x{args.extend} {s['time_out_reached_in_extension']}/{len(s['time_out_seeds'])} "
              f"| sat_w {s['all_median']['sat_w']:.2f} |mu| {s['median_abs_mu_all']} | matches published {entry.get('matches_published')} "
              f"| {entry['wall_s']} s", flush=True)
    if args.dose:
        res["dose_response"] = dose_probe(args.run, args.dose_episodes, args.dose_steps)
    res["wall_s"] = round(time.time() - t0, 1)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1) + "\n")
    print(f"NAVGYM DIAGNOSE wrote {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
