#!/usr/bin/env python3
"""C0 / C2 / C5 scripted baseline. Default backend is the kinematic env.

Isaac backends need a GPU and SimulationApp; they are opt-in so a login node
can still score the planner. Deformable Isaac evaluation is refused unless a
cost report is accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.cloth.cost import report_cost
from bhl_robust.cloth.env import KinematicClothSortEnv, make_env
from bhl_robust.cloth.garments import GARMENTS
from bhl_robust.cloth.metrics import EpisodeMetrics, RunMetrics
from bhl_robust.cloth.scripted import scripted_action


def run_kinematic(rung: str, episodes: int, seed: int) -> RunMetrics:
    env = make_env(rung, seed=seed)
    metrics = RunMetrics(physics="kinematic_rigid", policy="scripted", num_envs=1)
    t0 = time.perf_counter()
    env_steps = 0
    for ep in range(episodes):
        env.reset(seed=seed + ep)
        em = EpisodeMetrics(n_garments=env.n_garments)
        done = False
        while not done:
            act = scripted_action(env.garment_xy(env._selected), env.selected_spec())
            _, _, done, info = env.step(act)
            env_steps += 1
            em.n_sweeps += 1
            em.invalid_trajectory += int(info.invalid)
            em.displacement_per_sweep.append(info.displacement)
            if info.all_correct and em.steps_to_success is None:
                em.steps_to_success = em.n_sweeps
        em.success = env.all_correct
        em.correct_count = sum(s.sorted_correct for s in env._states)
        em.wrong_count = sum(s.sorted_wrong for s in env._states)
        em.final_distance = env._distance_to_target(0)
        em.env_steps = em.n_sweeps
        metrics.episodes.append(em)
    elapsed = time.perf_counter() - t0
    metrics.env_steps_per_s = env_steps / max(elapsed, 1e-9)
    return metrics


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rung", default="C0", choices=("C0", "C2", "C5"))
    p.add_argument("--backend", default="kinematic", choices=("kinematic", "isaac"))
    p.add_argument("--episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.backend == "isaac":
        physics = "deformable" if args.rung in ("C2",) else "rigid"
        rep = report_cost(
            num_envs=8, iterations=1, steps_per_iter=args.episodes,
            physics=physics,
        )
        print(json.dumps(rep.as_dict(), indent=2))
        if physics == "deformable" and not rep.accepted:
            raise SystemExit(rep.reason)
        raise SystemExit(
            "Isaac backend: sbatch slurm/95c_cloth_smoke.sbatch, then "
            "python scripts/cloth/eval_isaac.py --rung C0 --headless. "
            "Deformable C2 is cost-gated inside that script."
        )

    metrics = run_kinematic(args.rung, args.episodes, args.seed)
    payload = metrics.as_dict()
    payload["rung"] = args.rung
    payload["backend"] = "kinematic"
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
