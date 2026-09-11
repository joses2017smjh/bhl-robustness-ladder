#!/usr/bin/env python3
"""Print the wall-clock cost of a proposed cloth-sort training job.

Must be run before any deformable training submission. Exits 2 if the job
is refused. G-C1 numbers are the default deformable rate; pass a measured
env-steps/s once Stage 2 has its own bench.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.cloth.cost import STANDARD_ITERATIONS, STANDARD_NUM_ENVS, STANDARD_STEPS_PER_ITER, report_cost


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--num-envs", type=int, default=STANDARD_NUM_ENVS)
    p.add_argument("--iterations", type=int, default=STANDARD_ITERATIONS)
    p.add_argument("--steps-per-iter", type=int, default=STANDARD_STEPS_PER_ITER)
    p.add_argument("--measured-sps", type=float, default=None)
    p.add_argument("--physics", default="deformable", choices=("deformable", "rigid"))
    p.add_argument("--max-hours", type=float, default=12.0)
    p.add_argument("--i-accept-the-cost", action="store_true")
    args = p.parse_args()

    rep = report_cost(
        num_envs=args.num_envs,
        iterations=args.iterations,
        steps_per_iter=args.steps_per_iter,
        measured_env_steps_per_s=args.measured_sps,
        max_hours=args.max_hours,
        i_accept_the_cost=args.i_accept_the_cost,
        physics=args.physics,
    )
    print(json.dumps(rep.as_dict(), indent=2))
    if not rep.accepted:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
