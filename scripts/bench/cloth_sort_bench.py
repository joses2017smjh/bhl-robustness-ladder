#!/usr/bin/env python3
"""Throughput bench for cloth-sort configurations.

Writes CSV + Markdown. Does not claim an optimisation works without a row.
The default path is a kinematic smoke (login node). Isaac combinations
(Franka / BHL × 0/1/5 deformables × resolution × env count) are submitted
via slurm/95b_cloth_sort_bench.sbatch and append to the same files.

Stop increasing ``--envs`` when throughput falls or the job OOMs. G-C1
already showed the 961-vertex Franka scene falling from 182 to 71 env-steps/s
and overflowing at 512.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from bhl_robust.cloth.env import make_env
from bhl_robust.cloth.metrics import append_rows_csv
from bhl_robust.cloth.mesh import RESOLUTIONS, grid_counts
from bhl_robust.cloth.scripted import scripted_action


def bench_kinematic(rung: str, steps: int, seed: int) -> dict:
    env = make_env(rung, seed=seed)
    env.reset(seed=seed)
    t0 = time.perf_counter()
    done = False
    n = 0
    while n < steps:
        if done:
            env.reset(seed=seed + n)
            done = False
        act = scripted_action(env.garment_xy(env._selected), env.selected_spec())
        _, _, done, _ = env.step(act)
        n += 1
    elapsed = time.perf_counter() - t0
    return {
        "robot": "bhl_kinematic",
        "n_deformables": 0,
        "cloth_resolution": "n/a",
        "n_vertices": 0,
        "num_envs": 1,
        "steps": n,
        "elapsed_s": elapsed,
        "env_steps_per_s": n / max(elapsed, 1e-9),
        "per_env_steps_per_s": n / max(elapsed, 1e-9),
        "gpu_mem_mb": None,
        "rung": rung,
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--backend", default="kinematic", choices=("kinematic", "isaac"))
    p.add_argument("--steps", type=int, default=400)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out-csv", type=Path, default=_REPO / "results" / "cloth_sort_bench.csv")
    p.add_argument("--out-md", type=Path, default=_REPO / "results" / "cloth_sort_bench.md")
    args = p.parse_args()

    if args.backend == "isaac":
        raise SystemExit(
            "Isaac bench is slurm/95b_cloth_sort_bench.sbatch → "
            "scripts/bench/cloth_sort_isaac_bench.py. "
            "Do not run it on a login node and do not skip the cost gate."
        )

    rows = [bench_kinematic(r, args.steps, args.seed) for r in ("C0", "C5")]
    # Record the mesh sizes the Isaac bench *will* sweep, so the table exists
    # before anyone claims a resolution was measured.
    for name, n in RESOLUTIONS:
        v, t = grid_counts(n)
        rows.append({
            "robot": "planned_isaac",
            "n_deformables": 1,
            "cloth_resolution": name,
            "n_vertices": v,
            "num_envs": None,
            "steps": 0,
            "elapsed_s": None,
            "env_steps_per_s": None,
            "per_env_steps_per_s": None,
            "gpu_mem_mb": None,
            "rung": f"mesh_{n}x{n}",
        })

    # Shared file with the Isaac bench, which has different columns. Union the
    # schema instead of appending under a foreign header; see append_rows_csv.
    append_rows_csv(args.out_csv, rows)

    lines = [
        "# Cloth-sort throughput",
        "",
        "Kinematic rows are measured. `planned_isaac` rows name the mesh",
        "resolutions Stage 2 will sweep; they are not measurements.",
        "",
        "| robot | deformables | resolution | vertices | envs | env-steps/s |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for r in rows:
        sps = r["env_steps_per_s"]
        sps_s = f"{sps:.1f}" if isinstance(sps, float) else "not yet measured"
        lines.append(
            f"| {r['robot']} | {r['n_deformables']} | {r['cloth_resolution']} | "
            f"{r['n_vertices']} | {r['num_envs'] if r['num_envs'] is not None else '—'} | {sps_s} |"
        )
    args.out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
