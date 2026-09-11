#!/usr/bin/env python3
"""Isaac scripted eval for C0 / C2 / C5.

Must run under SimulationApp (sbatch). Deformable cells are cost-gated.
Success is the geometric predicate, not a rendered clip.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--rung", default="C0", choices=("C0", "C2", "C5"))
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--episodes", type=int, default=8)
parser.add_argument("--max_steps", type=int, default=80)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.cloth.cost import report_cost  # noqa: E402
from bhl_robust.cloth.garments import GARMENT_BY_NAME  # noqa: E402
from bhl_robust.cloth.metrics import EpisodeMetrics, RunMetrics  # noqa: E402
from bhl_robust.cloth.scripted import scripted_action  # noqa: E402
from bhl_robust.tasks.cloth_sort_env_cfg import build_cfg  # noqa: E402
from bhl_robust.tasks.cloth_sort_mdp import _local_pos  # noqa: E402


TASK = {
    "C0": "ClothSort-BHL-Rigid-Oracle-v0",
    "C2": "ClothSort-BHL-Deformable-Oracle-v0",
    "C5": "ClothSort-BHL-RigidFive-Oracle-v0",
}
GARMENT = {
    "C0": "shirt_a",
    "C2": "shirt_a",
    "C5": "sock_a",
}


def _xy(env, name: str = "garment_0") -> np.ndarray:
    p = _local_pos(env.unwrapped, name)[0, :2]
    return p.detach().cpu().numpy()


def _term_fired(tm, name: str) -> bool:
    """Did the named termination term fire this step?

    Read through the public ``get_term``. ``_term_dones`` is a
    ``(num_envs, n_terms)`` tensor here, not a dict keyed by name, so the old
    ``"success" in tm._term_dones`` raised ``RuntimeError`` and a bare
    ``except: pass`` turned that into a silent ``False``. ``success_rate``
    and ``fall_rate`` could therefore never report anything but 0.0 -- a
    headline number that was structurally incapable of moving.

    Raises rather than defaulting if the term is absent: silence is what made
    the previous version wrong.
    """
    active = list(tm.active_terms)
    if name not in active:
        raise KeyError(f"termination term {name!r} is not active; have {active}")
    return bool(tm.get_term(name)[0].item())


def main() -> None:
    tid = TASK[args_cli.rung]
    physics = "deformable" if args_cli.rung == "C2" else "rigid"
    n_env = args_cli.num_envs
    if physics == "deformable":
        n_env = min(n_env, 8)
        rep = report_cost(
            num_envs=n_env, iterations=args_cli.episodes,
            steps_per_iter=args_cli.max_steps, physics="deformable", max_hours=4.0,
        )
        print(json.dumps(rep.as_dict(), indent=2))
        if not rep.accepted:
            raise SystemExit(rep.reason)

    cfg = build_cfg(
        gym.spec(tid).kwargs["env_cfg_entry_point"],
        num_envs=n_env, device=app_launcher.device,
    )
    env = gym.make(tid, cfg=cfg, disable_env_checker=True)
    spec = GARMENT_BY_NAME[GARMENT[args_cli.rung]]
    metrics = RunMetrics(
        physics="isaac_deformable" if physics == "deformable" else "isaac_rigid",
        policy="scripted",
        num_envs=n_env,
        n_deformables=1 if physics == "deformable" else 0,
    )
    t0 = time.perf_counter()
    env_steps = 0
    for ep in range(args_cli.episodes):
        env.reset()
        em = EpisodeMetrics(n_garments=5 if args_cli.rung == "C5" else 1)
        done = False
        step = 0
        while not done and step < args_cli.max_steps:
            act_np = scripted_action(_xy(env), spec)
            act = torch.tensor(act_np, dtype=torch.float32, device=env.unwrapped.device)
            act = act.unsqueeze(0).repeat(n_env, 1)
            _, _, term, trunc, _ = env.step(act)
            env_steps += n_env
            step += 1
            em.n_sweeps += 1
            # The sweep term flags a plan it refused. Without this the Isaac
            # eval published invalid_trajectory_rate = 0 whatever happened --
            # the same structurally pinned metric success and fall_rate were.
            em.invalid_trajectory += int(bool(
                env.unwrapped.action_manager.get_term("sweep")._invalid[0].item()))
            done = bool(term[0] or trunc[0])
            if done and em.steps_to_success is None:
                em.steps_to_success = step
        em.env_steps = step
        tm = env.unwrapped.termination_manager
        em.success = _term_fired(tm, "success")
        em.fell = _term_fired(tm, "fallen")
        metrics.episodes.append(em)
    elapsed = time.perf_counter() - t0
    metrics.env_steps_per_s = env_steps / max(elapsed, 1e-9)
    payload = metrics.as_dict()
    payload["rung"] = args_cli.rung
    payload["task"] = tid
    print(json.dumps(payload, indent=2))
    if args_cli.out:
        Path(args_cli.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args_cli.out).write_text(json.dumps(payload, indent=2))
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
