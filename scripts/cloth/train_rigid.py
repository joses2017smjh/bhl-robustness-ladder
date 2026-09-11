#!/usr/bin/env python3
"""C1: learn 5-D sweep residuals on the kinematic rigid proxy.

Isaac PPO at 1024–2048 envs is the intended Stage 1 trainer once a GPU cell
has measured rigid-proxy throughput. This script is the login-node counterpart:
REINFORCE on a linear residual around the scripted sweep, so C1 has a measured
number that is not a fabricated Isaac result.

Deformable training is refused here. Use ``estimate_cost.py`` first.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.cloth.cost import report_cost
from bhl_robust.cloth.env import make_env
from bhl_robust.cloth.metrics import EpisodeMetrics, RunMetrics
from bhl_robust.cloth.randomization import DomainRandomization
from bhl_robust.cloth.scripted import scripted_action
from bhl_robust.cloth.sweep import ACTION_DIM


def _train(episodes: int, seed: int, lr: float) -> tuple[np.ndarray, RunMetrics]:
    rng = np.random.default_rng(seed)
    # Residual around the scripted action. Starts at zero, so step 0 is C0.
    theta = np.zeros(ACTION_DIM)
    env = make_env("C1", seed=seed, domain_rand=DomainRandomization())
    metrics = RunMetrics(physics="kinematic_rigid", policy="learned_residual", num_envs=1)
    t0 = time.perf_counter()
    steps = 0
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        em = EpisodeMetrics(n_garments=env.n_garments)
        noise = rng.normal(0.0, 0.15, size=ACTION_DIM)
        done = False
        ret = 0.0
        while not done:
            base = scripted_action(env.garment_xy(env._selected), env.selected_spec())
            act = np.clip(base + theta + noise, -1.0, 1.0)
            _, rew, done, info = env.step(act)
            ret += rew
            steps += 1
            em.n_sweeps += 1
            em.invalid_trajectory += int(info.invalid)
            em.displacement_per_sweep.append(info.displacement)
            if info.all_correct and em.steps_to_success is None:
                em.steps_to_success = em.n_sweeps
        # REINFORCE on the residual. Baseline is 0; the scripted policy already
        # gets most of the return, so the gradient is small unless it fails.
        theta += lr * (ret - 2.0) * noise
        theta = np.clip(theta, -0.6, 0.6)
        em.success = env.all_correct
        em.correct_count = int(env.all_correct)
        em.wrong_count = sum(s.sorted_wrong for s in env._states)
        em.final_distance = env._distance_to_target(0)
        metrics.episodes.append(em)
    metrics.env_steps_per_s = steps / max(time.perf_counter() - t0, 1e-9)
    return theta, metrics


def _eval(theta: np.ndarray, episodes: int, seed: int) -> RunMetrics:
    env = make_env("C1", seed=seed + 10_000, domain_rand=DomainRandomization())
    metrics = RunMetrics(physics="kinematic_rigid", policy="learned_residual", num_envs=1)
    for ep in range(episodes):
        env.reset(seed=seed + 10_000 + ep)
        em = EpisodeMetrics(n_garments=env.n_garments)
        done = False
        while not done:
            base = scripted_action(env.garment_xy(env._selected), env.selected_spec())
            act = np.clip(base + theta, -1.0, 1.0)
            _, _, done, info = env.step(act)
            em.n_sweeps += 1
            em.invalid_trajectory += int(info.invalid)
            if info.all_correct and em.steps_to_success is None:
                em.steps_to_success = em.n_sweeps
        em.success = env.all_correct
        em.correct_count = int(env.all_correct)
        em.wrong_count = sum(s.sorted_wrong for s in env._states)
        em.final_distance = env._distance_to_target(0)
        metrics.episodes.append(em)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="kinematic", choices=("kinematic", "isaac"))
    p.add_argument("--episodes", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.backend == "isaac":
        raise SystemExit(
            "Isaac C1 training: sbatch slurm/95d_cloth_c1.sbatch after a rigid "
            "throughput number exists. Do not assume 2048 envs are free."
        )

    # Rigid kinematic job: the cloth cost gate does not apply, but report it.
    print(json.dumps(report_cost(
        num_envs=1, iterations=args.episodes, steps_per_iter=4, physics="rigid",
    ).as_dict(), indent=2))

    theta, train = _train(args.episodes, args.seed, args.lr)
    ev = _eval(theta, args.eval_episodes, args.seed)
    payload = {
        "train": train.as_dict(),
        "eval": ev.as_dict(),
        "theta": theta.tolist(),
        "backend": "kinematic",
        "rung": "C1",
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        np.save(args.out.with_suffix(".theta.npy"), theta)


if __name__ == "__main__":
    main()
