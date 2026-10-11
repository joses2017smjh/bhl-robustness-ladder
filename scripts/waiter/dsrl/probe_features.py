#!/usr/bin/env python3
"""Dev diagnostic: can the SAC observation tell which tumbler is the target?

For each DEV layout the env is reset (no steps), the frozen VLA's prefix features are computed exactly as DSRL's SAC
observation (``--feature-mode``, default pooled: mean image tokens, mean language tokens, state token, state), and
small probes are trained to read four facts from them:
  colour      the instruction's tumbler colour (red / blue)                    language only
  side        the instruction's table side (left / right)                      language only
  red_left    the red tumbler stands on the robot's left                        vision only
  target_left the tumbler the instruction names stands on the robot's left      vision AND language (the pick)
Probes: logistic regression and a one-hidden-layer MLP, on a train split, accuracy on a held-out split.
Labels come from the env's oracle and are used here only. Seeds: dev range 1000-1999. Not a scored run.

usage (through dsrl_gpu.sbatch, which adds --env-python):
  dsrl_gpu.sbatch probe --policy smolvla --policy-path <ckpt> --env waiter --env-kwargs '<json>' \
      --seeds 1000-1599 --n-envs 8 --out <probe.json>
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from chunk_vec_env import ChunkVecEnv  # noqa: E402
from rollout import check_seeds, parse_seeds, pool4  # noqa: E402
from train_dsrl import add_common_args, atomic_json, build_policy, env_spec  # noqa: E402

LABELS = ("colour", "side", "red_left", "target_left")


def fit_probe(x_tr, y_tr, x_te, y_te, hidden: int = 0, steps: int = 1500, seed: int = 0) -> dict:
    """Train on (x_tr, y_tr), report accuracy on both splits. hidden = 0: logistic regression."""
    torch.manual_seed(seed)
    mu, sd = x_tr.mean(0, keepdim=True), x_tr.std(0, keepdim=True).clamp_min(1e-6)
    x_tr, x_te = (x_tr - mu) / sd, (x_te - mu) / sd
    d = x_tr.shape[1]
    net = torch.nn.Linear(d, 1) if hidden == 0 else torch.nn.Sequential(
        torch.nn.Linear(d, hidden), torch.nn.ReLU(), torch.nn.Linear(hidden, 1))
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
    for _ in range(steps):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(net(x_tr).squeeze(1), y_tr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = lambda x, y: float(((net(x).squeeze(1) > 0).float() == y).float().mean())  # noqa: E731
        return {"train_acc": acc(x_tr, y_tr), "test_acc": acc(x_te, y_te)}


def probe_all(feats: torch.Tensor, labels: dict, n_train: int) -> dict:
    out = {}
    for name in LABELS:
        y = torch.as_tensor(labels[name], dtype=torch.float32)
        base = float(max(y[n_train:].mean(), 1 - y[n_train:].mean()))
        out[name] = {"majority": base,
                     "linear": fit_probe(feats[:n_train], y[:n_train], feats[n_train:], y[n_train:], 0),
                     "mlp256": fit_probe(feats[:n_train], y[:n_train], feats[n_train:], y[n_train:], 256)}
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--seeds", required=True)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--train-frac", type=float, default=0.7)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()
    seeds = parse_seeds(args.seeds)
    check_seeds(seeds, {"dev"}, "probe")
    policy = build_policy(args.policy, args.policy_path, args.device, args.feature_mode, args.exec_horizon)
    venv = ChunkVecEnv(env_spec(args), args.n_envs, python=args.env_python)
    feats, imgs, labels = [], [], {k: [] for k in LABELS}
    t0 = time.time()
    for k in range(0, len(seeds), venv.n):
        batch = seeds[k:k + venv.n]
        obs = venv.reset({i: (s, None) for i, s in enumerate(batch)})
        ol = [obs[i] for i in range(len(batch))]
        feats.append(policy.encode(ol).features.float().cpu())
        imgs.append(np.stack([pool4(np.asarray(o["image"], np.uint8)) for o in ol]))
        for i in range(len(batch)):
            o = venv.oracle(i)
            red_left = float(o["tumbler_red"][0][1]) > 0.0
            labels["colour"].append(o["target_colour"] == "red")
            labels["side"].append(o["target_side"] == "left")
            labels["red_left"].append(red_left)
            labels["target_left"].append(red_left == (o["target_colour"] == "red"))
        print(f"[probe] {min(k + venv.n, len(seeds))}/{len(seeds)} resets", flush=True)
    venv.close()
    feats = torch.cat(feats)
    img = torch.as_tensor(np.concatenate(imgs)).float().flatten(1) / 255.0
    n_train = int(round(args.train_frac * len(seeds)))
    hidden = (feats.shape[1] - policy.state_dim) // 3 if args.feature_mode == "pooled" else 0
    parts = {"all": feats}
    if hidden:                                           # pooled = [image mean, language mean, state token, state]
        parts.update(image_tokens=feats[:, :hidden], language_tokens=feats[:, hidden:2 * hidden],
                     state_token=feats[:, 2 * hidden:3 * hidden])
    res = {"purpose": "dev", "seeds": seeds, "n_train": n_train, "n_test": len(seeds) - n_train,
           "feature_mode": args.feature_mode, "feature_dim": int(feats.shape[1]), "policy_path": args.policy_path,
           "label_means": {k: float(np.mean(v)) for k, v in labels.items()},
           "probes": {name: probe_all(x, labels, n_train) for name, x in parts.items()},
           "probes_image_56x56_pixels": probe_all(img, labels, n_train),
           "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename, "wall_s": time.time() - t0}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(args.out, res)
    for part, r in {**res["probes"], "56x56 pixels": res["probes_image_56x56_pixels"]}.items():
        print(f"[probe] {part:16s} " + "  ".join(
            f"{k} lin {v['linear']['test_acc']:.2f} mlp {v['mlp256']['test_acc']:.2f}" for k, v in r.items()), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
