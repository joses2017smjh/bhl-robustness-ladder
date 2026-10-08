#!/usr/bin/env python3
"""Isaac throughput for cloth-sort. Stop when the curve falls or the job dies.

Combinations (cheap first):

* BHL rigid, 0 deformables, envs 8 → 32 → 64
* BHL, 1 deformable, 8×8 then 10×12, 8 envs only unless 8×8 stays above
  the G-C1 8-env rate

Never starts at 961 vertices. Never jumps to 128 cloth envs. Writes CSV+MD.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--out-csv", type=str, default=None)
parser.add_argument("--out-md", type=str, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.cloth.cost import report_cost  # noqa: E402
from bhl_robust.cloth.mesh import isaac_grid_counts  # noqa: E402
from bhl_robust.cloth.metrics import append_rows_csv  # noqa: E402
from bhl_robust.tasks.cloth_sort_env_cfg import build_cfg, spawned_cloth_resolution  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]


def _clear() -> None:
    try:
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
    except Exception:
        pass


def _gpu_mb() -> float | None:
    try:
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1e6)
    except Exception:
        return None
    return None


def _bench(tid: str, n_env: int, steps: int, res: int | None) -> dict:
    _clear()
    cfg = build_cfg(
        gym.spec(tid).kwargs["env_cfg_entry_point"],
        num_envs=n_env, device=app_launcher.device, cloth_resolution=res,
    )
    # Report what the scene will spawn, not what was asked for.
    actual = spawned_cloth_resolution(cfg)
    env = gym.make(tid, cfg=cfg, disable_env_checker=True)
    env.reset()
    act = torch.zeros((n_env, env.unwrapped.action_space.shape[-1]), device=env.unwrapped.device)
    for _ in range(3):
        env.step(act)
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(act)
    elapsed = time.perf_counter() - t0
    n_verts = isaac_grid_counts(actual)[0] if actual else 0
    row = {
        "robot": "bhl",
        "task": tid,
        "n_deformables": 1 if actual else 0,
        "cloth_resolution": f"{actual}x{actual}" if actual else "n/a",
        "n_vertices": n_verts,
        "num_envs": n_env,
        "steps": steps,
        "elapsed_s": elapsed,
        "env_steps_per_s": (n_env * steps) / max(elapsed, 1e-9),
        "per_env_steps_per_s": steps / max(elapsed, 1e-9),
        "gpu_mem_mb": _gpu_mb(),
        "ok": True,
        "error": None,
    }
    env.close()
    return row


def main() -> None:
    rows = []
    # Rigid scaling. Stop if throughput falls > 25% from the first successful cell.
    prev = None
    for n in (8, 32, 64):
        try:
            row = _bench("ClothSort-BHL-Rigid-Oracle-v0", n, args_cli.steps, None)
        except Exception as e:
            rows.append({
                "robot": "bhl", "task": "ClothSort-BHL-Rigid-Oracle-v0",
                "n_deformables": 0, "cloth_resolution": "n/a", "n_vertices": 0,
                "num_envs": n, "ok": False, "error": f"{type(e).__name__}: {e}"[:160],
            })
            break
        rows.append(row)
        print(f"rigid envs={n} sps={row['env_steps_per_s']:.1f}")
        if prev is not None and row["env_steps_per_s"] < 0.75 * prev:
            print("throughput fell; stopping rigid scale-up")
            break
        prev = row["env_steps_per_s"]

    for n_side in (8, 10):
        n_env = 8
        n_verts, _ = isaac_grid_counts(n_side)
        rep = report_cost(
            num_envs=n_env, iterations=1, steps_per_iter=args_cli.steps,
            physics="deformable", max_hours=2.0,
        )
        if not rep.accepted:
            rows.append({
                "robot": "bhl", "n_deformables": 1,
                "cloth_resolution": f"{n_side}x{n_side}", "n_vertices": n_verts,
                "num_envs": n_env, "ok": False, "error": rep.reason,
            })
            break
        try:
            row = _bench(
                "ClothSort-BHL-Deformable-Oracle-v0", n_env, args_cli.steps, n_side,
            )
        except Exception as e:
            rows.append({
                "robot": "bhl", "n_deformables": 1,
                "cloth_resolution": f"{n_side}x{n_side}", "n_vertices": n_verts,
                "num_envs": n_env, "ok": False,
                "error": f"{type(e).__name__}: {e}"[:160],
            })
            print(f"deformable {n_side}x{n_side} FAILED")
            break
        rows.append(row)
        print(f"deformable {n_side}x{n_side} envs={n_env} sps={row['env_steps_per_s']:.1f}")

    out_csv = Path(args_cli.out_csv or os.environ.get(
        "BENCH_OUT", str(_REPO / "results" / "cloth_sort_bench.csv"),
    ))
    if out_csv.suffix != ".csv":
        out_csv = out_csv.with_suffix(".csv")
    out_md = Path(args_cli.out_md or str(_REPO / "results" / "cloth_sort_bench.md"))
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    # Union the schema rather than appending under a foreign header; see
    # append_rows_csv. The kinematic and Isaac benches share this file and
    # do not share columns.
    append_rows_csv(out_csv, rows)
    lines = [
        "# Cloth-sort throughput",
        "",
        "Isaac rows below are measured. Kinematic rows live in the same CSV.",
        "",
        "| robot | deformables | resolution | vertices | envs | env-steps/s | ok |",
        "|---|---:|---|---:|---:|---:|---|",
    ]
    for r in rows:
        sps = r.get("env_steps_per_s")
        sps_s = f"{sps:.1f}" if isinstance(sps, float) else "FAIL"
        lines.append(
            f"| {r.get('robot')} | {r.get('n_deformables')} | {r.get('cloth_resolution')} | "
            f"{r.get('n_vertices')} | {r.get('num_envs')} | {sps_s} | {r.get('ok')} |"
        )
    out_md.write_text("\n".join(lines) + "\n")
    print(json.dumps(rows, indent=2, default=str))


if __name__ == "__main__":
    main()
    simulation_app.close()
