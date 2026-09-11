#!/usr/bin/env python3
"""Does each cloth-sort gym id construct, reset and step?

Mirrors scripts/bench/task_v2_smoke.py. Rigid cells first. Deformable is
opt-in and cost-gated: a smoke that cannot print a CostReport is not a smoke.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--include-deformable", action="store_true")
parser.add_argument("--cloth-res", type=int, default=8)
parser.add_argument("--out", type=str, default=None)
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
from bhl_robust.tasks.cloth_sort_env_cfg import build_cfg, spawned_cloth_resolution  # noqa: E402
from bhl_robust.cloth.robot import EXPECTED_ARMS, EXPECTED_JOINTS, assert_articulation  # noqa: E402


def _clear_sim() -> None:
    try:
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
    except Exception:
        pass


RIGID = (
    "ClothSort-BHL-Rigid-Oracle-v0",
    "ClothSort-BHL-RigidFive-Oracle-v0",
)
SOFT = (
    "ClothSort-BHL-Deformable-Oracle-v0",
    "ClothSort-BHL-ActiveCloth-Oracle-v0",
)


def _gpu_mb() -> float | None:
    try:
        if torch.cuda.is_available():
            return float(torch.cuda.max_memory_allocated() / 1e6)
    except Exception:
        return None
    return None


def _run(tid: str, n_env: int, steps: int, cloth_res: int | None = None) -> dict:
    cfg = build_cfg(
        gym.spec(tid).kwargs["env_cfg_entry_point"],
        num_envs=n_env, device=app_launcher.device, cloth_resolution=cloth_res,
    )
    actual_res = spawned_cloth_resolution(cfg)
    t_build0 = time.perf_counter()
    env = gym.make(tid, cfg=cfg, disable_env_checker=True)
    env.reset()
    build_s = time.perf_counter() - t_build0
    art = env.unwrapped.scene["robot"]
    assert_articulation(list(art.joint_names))
    n_arm = sum(1 for n in art.joint_names if n.startswith("arm_"))
    if n_arm != 10:
        raise AssertionError(f"{tid} arm joint count {n_arm}, expected 10")
    act = torch.zeros((n_env, env.unwrapped.action_space.shape[-1]), device=env.unwrapped.device)
    # Time the stepping only. Scene construction is tens of times the cost of
    # six steps, so folding it in would report a "throughput" that is mostly
    # USD loading. This is a liveness check either way -- the real number
    # comes from scripts/bench/cloth_sort_isaac_bench.py, which warms up first.
    t0 = time.perf_counter()
    for _ in range(steps):
        env.step(act)
    elapsed = time.perf_counter() - t0
    n_steps = n_env * steps
    row = {
        "task": tid,
        "num_envs": n_env,
        "steps": steps,
        "ok": True,
        "error": None,
        "action_dim": int(env.unwrapped.action_space.shape[-1]),
        "n_joints": len(art.joint_names),
        "n_arms": EXPECTED_ARMS,
        "step_elapsed_s": elapsed,
        "build_s": build_s,
        # Unwarmed, six steps. A liveness figure, not the bench number.
        "step_env_steps_per_s": n_steps / max(elapsed, 1e-9),
        "gpu_mem_mb": _gpu_mb(),
        "cloth_resolution": actual_res,
        "n_vertices": isaac_grid_counts(actual_res)[0] if actual_res else 0,
    }
    env.close()
    return row


def main() -> None:
    rows = []
    bad = 0
    for tid in RIGID:
        _clear_sim()
        try:
            rows.append(_run(tid, args_cli.num_envs, args_cli.steps))
        except Exception as e:
            bad += 1
            rows.append({
                "task": tid, "ok": False, "error": f"{type(e).__name__}: {e}"[:200],
            })
            if bad == 1:
                import traceback
                traceback.print_exc()

    if args_cli.include_deformable:
        n_soft = min(args_cli.num_envs, 8)
        rep = report_cost(
            num_envs=n_soft, iterations=1, steps_per_iter=args_cli.steps,
            physics="deformable", max_hours=2.0,
        )
        print("cost_gate", json.dumps(rep.as_dict()))
        if not rep.accepted:
            rows.append({"task": "deformable", "ok": False, "error": rep.reason})
        else:
            for tid in SOFT:
                _clear_sim()
                try:
                    rows.append(_run(tid, n_soft, args_cli.steps, args_cli.cloth_res))
                except Exception as e:
                    bad += 1
                    rows.append({
                        "task": tid, "ok": False,
                        "error": f"{type(e).__name__}: {e}"[:200],
                    })
                    if bad == 1:
                        import traceback
                        traceback.print_exc()

    print(f"\n{'task':42} {'ok':5}  note")
    for r in rows:
        note = r.get("error") or (
            f"step_sps={r.get('step_env_steps_per_s', 0):.1f} "
            f"build={r.get('build_s', 0):.1f}s dim={r.get('action_dim')}"
        )
        print(f"{r['task']:42} {str(r.get('ok')):5}  {note}")
    print(f"\nexpected joints={EXPECTED_JOINTS} arms={EXPECTED_ARMS}")
    out = args_cli.out or os.environ.get("BENCH_OUT")
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(rows, indent=2))
    if any(not r.get("ok") for r in rows):
        raise SystemExit(1)
    print("built and stepped")


if __name__ == "__main__":
    main()
    simulation_app.close()
