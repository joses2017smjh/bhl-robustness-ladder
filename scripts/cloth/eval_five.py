#!/usr/bin/env python3
"""C5: five-garment evaluation.

Default is the kinematic stand-in with active-garment mode (Mode B): one
garment is displaced per sweep, the others stay put. That is not five
simultaneous deformables. Mode A (all active) is the ``--all-active`` flag
and still kinematic unless ``--backend isaac`` is used.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.cloth.ladder import ACTIVE_GARMENT_WARNING

sys.path.insert(0, str(Path(__file__).resolve().parent))
from eval_scripted import run_kinematic  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--episodes", type=int, default=32)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--all-active", action="store_true",
                   help="Mode A: every garment can move. Still kinematic here.")
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    from bhl_robust.cloth.env import make_env
    # run_kinematic uses make_env("C5") which defaults to active-garment mode.
    if args.all_active:
        # Temporarily evaluate with Mode A by constructing the env ourselves
        # and reusing the scripted loop from eval_scripted via a local copy.
        from bhl_robust.cloth.env import KinematicClothSortEnv
        from bhl_robust.cloth.garments import GARMENTS
        from bhl_robust.cloth.metrics import EpisodeMetrics, RunMetrics
        from bhl_robust.cloth.randomization import IDENTITY
        from bhl_robust.cloth.scripted import scripted_action
        import time
        env = KinematicClothSortEnv(
            GARMENTS, domain_rand=IDENTITY, seed=args.seed, active_garment_mode=False,
        )
        metrics = RunMetrics(physics="kinematic_rigid", policy="scripted", num_envs=1)
        t0 = time.perf_counter()
        steps = 0
        for ep in range(args.episodes):
            env.reset(seed=args.seed + ep)
            em = EpisodeMetrics(n_garments=env.n_garments)
            done = False
            while not done:
                act = scripted_action(env.garment_xy(env._selected), env.selected_spec())
                _, _, done, info = env.step(act)
                steps += 1
                em.n_sweeps += 1
                em.invalid_trajectory += int(info.invalid)
                if info.all_correct and em.steps_to_success is None:
                    em.steps_to_success = em.n_sweeps
            em.success = env.all_correct
            em.correct_count = sum(s.sorted_correct for s in env._states)
            em.wrong_count = sum(s.sorted_wrong for s in env._states)
            em.final_distance = env._distance_to_target(0)
            metrics.episodes.append(em)
        metrics.env_steps_per_s = steps / max(time.perf_counter() - t0, 1e-9)
        payload = metrics.as_dict()
        payload["active_garment_mode"] = False
        payload["note"] = "Mode A kinematic: all five proxies can move. Not five deformables."
    else:
        metrics = run_kinematic("C5", args.episodes, args.seed)
        payload = metrics.as_dict()
        payload["active_garment_mode"] = True
        payload["note"] = ACTIVE_GARMENT_WARNING

    payload["rung"] = "C5"
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
