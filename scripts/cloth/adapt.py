#!/usr/bin/env python3
"""C4: behaviour-clone a linear sweep policy from the scripted baseline.

Cost-gated. The default backend is kinematic. Isaac deformable fine-tuning
is refused unless estimate_cost.py accepts the job.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.cloth.adapt import collect_scripted, evaluate_linear, fit_linear
from bhl_robust.cloth.cost import report_cost


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="kinematic", choices=("kinematic", "isaac"))
    p.add_argument("--collect", type=int, default=64)
    p.add_argument("--eval-episodes", type=int, default=64)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path, default=None)
    args = p.parse_args()

    if args.backend == "isaac":
        rep = report_cost(
            num_envs=8, iterations=200, steps_per_iter=24, physics="deformable",
        )
        print(json.dumps(rep.as_dict(), indent=2))
        if not rep.accepted:
            raise SystemExit(rep.reason)
        raise SystemExit(
            "Isaac C4 is not wired as a full RL job. Collect kinematic BC first; "
            "only then consider a cost-gated residual on ClothSort-BHL-Deformable-Oracle-v0."
        )

    x, y = collect_scripted(args.collect, args.seed, "C0")
    w = fit_linear(x, y)
    ev = evaluate_linear(w, args.eval_episodes, args.seed + 10_000, "C1")
    payload = {
        "rung": "C4",
        "backend": "kinematic",
        "policy": "bc_linear",
        "n_demo_steps": int(x.shape[0]),
        "eval": ev.as_dict(),
    }
    print(json.dumps(payload, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(payload, indent=2))
        np.save(args.out.with_suffix(".W.npy"), w)


if __name__ == "__main__":
    main()
