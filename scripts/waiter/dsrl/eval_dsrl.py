#!/usr/bin/env python3
"""Evaluate the frozen base policy vs the DSRL-steered policy on a seed list; JSON out.

Arms (one episode per seed per arm, same layouts, paired by seed):
  base          the frozen VLA as deployed: i.i.d. N(0, I) noise of shape (chunk, 32) per chunk, drawn from a
                generator seeded by (--noise-seed, layout seed, decision index) -> reproducible
  zero          noise = 0 (the flow's "mode"; a free control: does ANY fixed noise beat random noise?)
  latent_random N(0, I) latent clipped to the SAC box through the run's NoiseMap (the DSRL warm-up distribution)
  dsrl          the SAC actor's deterministic latent (tanh(mean)) -> NoiseMap -> noise
  dsrl_stoch    a sample from the SAC actor
Paired statistics: success both / only-A / only-B / neither, exact one-sided McNemar p, bootstrap CI.

Seeds: dev range 1000-1999 only, unless --scored-phase4-eval, which allows ONLY the never-used phase 4
evaluation range 6000-6099 (to be run once, after the phase 4 gate is frozen). TOY envs/policies are labelled.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chunk_vec_env import ChunkVecEnv  # noqa: E402
from rollout import ARMS, ObsBuilder, arm_summary, check_seeds, paired_summary, parse_seeds, run_episodes  # noqa: E402
from sac_noise import NoiseSAC  # noqa: E402
from smolvla_noise import NoiseMap  # noqa: E402
from train_dsrl import add_common_args, atomic_json, build_policy, env_spec  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--run-dir", type=Path, help="train_dsrl.py output dir (config.json + ckpt/sac.pt)")
    ap.add_argument("--sac-ckpt", type=Path, help="default: <run-dir>/ckpt/sac.pt")
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--arms", default="base,zero,dsrl")
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--noise-seed", type=int, default=0)
    ap.add_argument("--obs-encoder", choices=("vlm", "cnn"), default="vlm")
    ap.add_argument("--scored-phase4-eval", action="store_true",
                    help="allow ONLY seeds 6000-6099 (the single scored phase 4 evaluation)")
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    seeds = parse_seeds(args.seeds)
    check_seeds(seeds, {"p4_eval"} if args.scored_phase4_eval else {"dev"}, "evaluation")
    arms = [a.strip() for a in args.arms.split(",") if a.strip()]
    unknown = [a for a in arms if a not in ARMS]
    if unknown:
        raise SystemExit(f"unknown arms {unknown}; choose from {ARMS}")

    run_cfg = None
    if args.run_dir:
        run_cfg = json.loads((args.run_dir / "config.json").read_text())
        nm, ra = run_cfg["noise_map"], run_cfg["args"]
        args.exec_horizon = int(run_cfg["exec_horizon"])
        args.noise_mode, args.noise_k, args.active_dims = nm["mode"], nm["k"], nm["active_dims"]
        args.action_magnitude = float(run_cfg["sac"]["action_magnitude"])
        args.obs_encoder = run_cfg["obs_encoder"]
        args.feature_mode = ra.get("feature_mode", args.feature_mode)
        if ra.get("policy") != args.policy:
            print(f"[eval] WARNING: run trained with --policy {ra.get('policy')}, evaluating {args.policy}", flush=True)
    if any(a.startswith("dsrl") for a in arms) and not (args.run_dir or args.sac_ckpt):
        raise SystemExit("dsrl arms need --run-dir or --sac-ckpt")

    torch.manual_seed(args.noise_seed)
    policy = build_policy(args.policy, args.policy_path, args.device, args.feature_mode, args.exec_horizon)
    H = int(args.exec_horizon or policy.n_action_steps)
    T, D = policy.noise_shape
    noise_map = NoiseMap(T, D, args.noise_mode, args.noise_k, args.active_dims)
    b = float(args.action_magnitude)
    sac, obs_builder = None, ObsBuilder(args.obs_encoder, policy.feature_dim)
    if any(a.startswith("dsrl") for a in arms):
        ck = args.sac_ckpt or (args.run_dir / "ckpt" / "sac.pt")
        sac = NoiseSAC.load(ck, device=args.device)
        if sac.cfg.act_dim != noise_map.latent_dim or sac.cfg.vec_dim != obs_builder.vec_dim:
            raise SystemExit(f"checkpoint dims ({sac.cfg.act_dim}, {sac.cfg.vec_dim}) != "
                             f"({noise_map.latent_dim}, {obs_builder.vec_dim})")
    toy = args.env == "stub" or args.policy == "stub"
    labels = ["TOY"] if toy else []
    if run_cfg and "SHAPED" in run_cfg.get("labels", []):
        labels.append("TRAINED_WITH_SHAPING")
    venv = ChunkVecEnv(env_spec(args), args.n_envs, python=args.env_python)
    out = {"labels": labels, "argv": sys.argv, "seeds": seeds, "exec_horizon": H, "policy": policy.describe(),
           "noise_map": noise_map.describe(), "action_magnitude": b, "noise_seed": args.noise_seed,
           "run_dir": str(args.run_dir) if args.run_dir else None, "scored_phase4_eval": args.scored_phase4_eval,
           "arms": {}, "paired": {}, "episodes": {}}
    for arm in arms:
        t0 = time.time()
        eps = run_episodes(venv, policy, seeds, arm, H, noise_map, b, obs_builder,
                           sac if arm.startswith("dsrl") else None, noise_seed=args.noise_seed, device=args.device)
        out["episodes"][arm] = eps
        out["arms"][arm] = {**arm_summary(eps), "wall_s": time.time() - t0}
        print(f"[eval] {' '.join(labels)} {arm}: {out['arms'][arm]}", flush=True)
    succ = {a: [e["success"] for e in out["episodes"][a]] for a in arms}
    for t_arm, c_arm in (("dsrl", "base"), ("dsrl", "zero"), ("zero", "base"), ("dsrl_stoch", "base"),
                         ("latent_random", "base")):
        if t_arm in succ and c_arm in succ:
            out["paired"][f"{t_arm}_vs_{c_arm}"] = paired_summary(succ[t_arm], succ[c_arm])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, out)
    print(json.dumps({"arms": out["arms"], "paired": out["paired"]}, indent=2), flush=True)
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
