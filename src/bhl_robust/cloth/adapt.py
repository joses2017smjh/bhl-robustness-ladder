"""C4 behaviour cloning: linear policy from scripted (obs, action) pairs.

Preferred order in docs/CLOTH_SORT.md is zero-shot, then BC, then residual,
then small RL. This is the BC step. It does not launch deformable RL.
"""

from __future__ import annotations

import numpy as np

from bhl_robust.cloth.env import make_env
from bhl_robust.cloth.metrics import EpisodeMetrics, RunMetrics
from bhl_robust.cloth.scripted import scripted_action
from bhl_robust.cloth.sweep import ACTION_DIM


def collect_scripted(episodes: int = 64, seed: int = 0, rung: str = "C0"):
    """Oracle observations paired with the scripted 5-D action."""
    env = make_env(rung, seed=seed)
    xs, ys = [], []
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        done = False
        while not done:
            act = scripted_action(env.garment_xy(env._selected), env.selected_spec())
            xs.append(obs.copy())
            ys.append(act.copy())
            obs, _, done, _ = env.step(act)
    return np.stack(xs), np.stack(ys)


def fit_linear(obs: np.ndarray, actions: np.ndarray) -> np.ndarray:
    """Least-squares ``action = [obs, 1] @ W``. W is ``(obs_dim+1, 5)``."""
    a = np.asarray(actions, dtype=float)
    if a.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM}-D actions, got {a.shape}")
    x = np.asarray(obs, dtype=float)
    phi = np.hstack([x, np.ones((x.shape[0], 1))])
    w, *_ = np.linalg.lstsq(phi, a, rcond=None)
    return w


def linear_action(obs: np.ndarray, weights: np.ndarray) -> np.ndarray:
    phi = np.concatenate([np.asarray(obs, dtype=float).ravel(), [1.0]])
    return np.clip(phi @ weights, -1.0, 1.0)


def evaluate_linear(weights: np.ndarray, episodes: int, seed: int, rung: str = "C1") -> RunMetrics:
    env = make_env(rung, seed=seed)
    metrics = RunMetrics(physics="kinematic_rigid", policy="bc_linear", num_envs=1)
    for ep in range(episodes):
        obs = env.reset(seed=seed + ep)
        em = EpisodeMetrics(n_garments=env.n_garments)
        done = False
        while not done:
            act = linear_action(obs, weights)
            obs, _, done, info = env.step(act)
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
