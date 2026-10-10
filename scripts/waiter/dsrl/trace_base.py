#!/usr/bin/env python3
"""Dev diagnostic: run the frozen VLA on DEV layouts and log, at every decision, where the tumblers, the tray and
the robot are, so that failed episodes can be sorted by what went wrong (failure_modes.py reads the JSON).

The poses come from the env's oracle() and are used for this log only; the policy sees the head image, the state and
the instruction as always. Arms: base (i.i.d. N(0, I) noise, the deployed policy), zero (noise = 0), or latent_const
(one random DSRL latent per episode, held for the whole episode and logged: does the latent decide which tumbler is
taken?). --max-decisions ends every episode early (the pick-up happens in the first seconds).
Seeds: dev range 1000-1999 only. Not a scored run.

usage (through dsrl_gpu.sbatch, which adds --env-python):
  dsrl_gpu.sbatch trace --policy smolvla --policy-path <ckpt> --env waiter --env-kwargs '<json>' \
      --seeds 1300-1347 --exec-horizon 10 --n-envs 16 --out <trace.json>
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chunk_vec_env import ChunkVecEnv  # noqa: E402
from rollout import check_seeds, choose_noise, parse_seeds  # noqa: E402
from smolvla_noise import NoiseMap  # noqa: E402
from train_dsrl import add_common_args, atomic_json, build_policy, env_spec  # noqa: E402


def tilt_deg(R) -> float:
    return math.degrees(math.acos(max(-1.0, min(1.0, float(np.asarray(R)[2, 2])))))


def snapshot(o: dict) -> dict:
    """The few oracle numbers the analysis needs (lists, JSON-friendly)."""
    out = {"t": round(float(o["t"]), 2), "base": [round(float(v), 4) for v in o["base_pos"][:2]] +
           [round(float(o["base_yaw"]), 4)], "locked": bool(o["base_locked"])}
    for name in ("tumbler_red", "tumbler_blue", "tray"):                  # [x, y, z, tilt deg, yaw rad]
        pos, R = o[name]
        R = np.asarray(R)
        out[name] = [round(float(v), 4) for v in pos] + [round(tilt_deg(R), 1),
                                                         round(math.atan2(float(R[1, 0]), float(R[0, 0])), 4)]
    for side in ("left", "right"):
        out[f"pivot_{side}"] = [round(float(v), 4) for v in o[f"pivot_{side}"]]
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--arm", choices=("base", "zero", "latent_const"), default="base")
    ap.add_argument("--max-decisions", type=int, default=0, help="end each episode after this many decisions (0 = off)")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    check_seeds(seeds, {"dev"}, "trace")
    torch.manual_seed(args.noise_seed)
    policy = build_policy(args.policy, args.policy_path, args.device, args.feature_mode, args.exec_horizon)
    H = int(args.exec_horizon or policy.n_action_steps)
    T, D = policy.noise_shape
    noise_map = NoiseMap(T, D, args.noise_mode, args.noise_k, args.active_dims)
    b = float(args.action_magnitude)
    venv = ChunkVecEnv(env_spec(args), args.n_envs, python=args.env_python)
    queue, done_eps = list(seeds), {}
    slot, obs = {}, {}

    def start(slots):
        todo = {}
        for i in slots:
            if queue:
                s = queue.pop(0)
                todo[i] = (s, None)
        obs.update(venv.reset(todo) if todo else {})
        for i, (s, _) in todo.items():
            o = venv.oracle(i)
            slot[i] = {"seed": int(s), "instruction": obs[i]["instruction"], "target_colour": o["target_colour"],
                       "target_side": o["target_side"], "success": False, "fell": False, "steps": 0,
                       "stage_max": 0, "trace": [dict(snapshot(o), stage=0)]}
            if args.arm == "latent_const":           # one latent per layout, reproducible from the two seeds
                g = np.random.default_rng([args.noise_seed, int(s)])
                slot[i]["latent"] = (np.clip(g.standard_normal(noise_map.latent_dim), -b, b) / b).round(4).tolist()
        return list(todo)

    t0 = time.time()
    active = start(range(venv.n))
    while active:
        ol = [obs[i] for i in active]
        ctx = policy.encode(ol)
        if args.arm == "latent_const":
            noise = noise_map(torch.as_tensor(np.asarray([slot[i]["latent"] for i in active], np.float32) * b))
        else:
            noise, _ = choose_noise(args.arm, list(range(len(active))), [slot[i]["seed"] for i in active],
                                    [len(slot[i]["trace"]) - 1 for i in active], None, policy, noise_map, b, None,
                                    args.noise_seed)
        chunks = policy.denoise(ctx, noise.to(args.device))[:, :H].numpy()
        venv.step_async({i: chunks[j] for j, i in enumerate(active)})
        res = venv.step_wait()
        finished = []
        for i in active:
            r, ep = res[i], slot[i]
            ep["steps"] += r["n"]
            ep["stage_max"] = max(ep["stage_max"], r["stage_max"])
            ep["success"] |= r["success"]
            ep["fell"] |= r["fell"]
            ep["trace"].append(dict(snapshot(venv.oracle(i)), stage=int(r["stage_last"])))
            obs[i] = r["obs"]
            if r["done"] or (args.max_decisions and len(ep["trace"]) - 1 >= args.max_decisions):
                ep["cut_short"] = not r["done"]
                done_eps[ep["seed"]] = ep
                finished.append(i)
                print(f"[trace] {len(done_eps)}/{len(seeds)} seed={ep['seed']} success={ep['success']} "
                      f"stage<={ep['stage_max']} steps={ep['steps']}", flush=True)
        active = [i for i in active if i not in finished] + start(finished)
    venv.close()
    out = {"purpose": "dev", "arm": args.arm, "exec_horizon": H, "noise_seed": args.noise_seed, "seeds": seeds,
           "action_magnitude": b, "noise_map": noise_map.describe(), "max_decisions": args.max_decisions,
           "policy_path": args.policy_path, "env_kwargs": json.loads(args.env_kwargs),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename, "wall_s": time.time() - t0,
           "successes": sum(e["success"] for e in done_eps.values()),
           "episodes": [done_eps[s] for s in seeds]}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, out)
    print(f"[trace] done {out['successes']}/{len(seeds)} -> {args.out}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
