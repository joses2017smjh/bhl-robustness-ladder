"""G-C1: can a Newton cloth scene run at RL scale on this cluster?

The folding repo's stack is CPU-only at one env per process, which is what rules
out RL there. 6.0 replaces PhysX particle cloth with Newton (VBD, Style3D) on
GPU, so the question is not whether cloth simulates but how many environments it
simulates at once. That number decides the whole task design: at 1,024 envs this
is an RL problem, at 16 it is a scripted demonstration with a rendered camera.

Measured, not assumed, because a wrong guess here costs a task built on it.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--envs", type=int, nargs="+", default=[16, 64, 256, 1024])
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--task", type=str,
                    default="Isaac-Lift-Cloth-Franka-IK-Abs-v0",
                    help="an Isaac Lab cloth task to measure the solver on")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402


def main() -> None:
    ids = [i for i in gym.registry if "loth" in i or "soft" in i.lower()]
    print(f"cloth/soft task ids visible: {ids[:8]}")
    task = args_cli.task if args_cli.task in gym.registry else (ids[0] if ids else None)
    if task is None:
        print("no cloth task registered; Newton cloth may need isaaclab_contrib")
        simulation_app.close()
        raise SystemExit(1)
    print(f"measuring: {task}\n")
    print(f"{'envs':>6} {'steps/s':>10} {'env-steps/s':>13}")
    for n in args_cli.envs:
        try:
            cfg = gym.spec(task).kwargs["env_cfg_entry_point"]()
            cfg.scene.num_envs = n
            env = gym.make(task, cfg=cfg, disable_env_checker=True)
            env.reset()
            act = torch.zeros((n, env.unwrapped.action_space.shape[-1]),
                              device=env.unwrapped.device)
            for _ in range(10):
                env.step(act)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args_cli.steps):
                env.step(act)
            torch.cuda.synchronize()
            dt = time.time() - t0
            sps = args_cli.steps / dt
            print(f"{n:6d} {sps:10.2f} {sps*n:13.0f}")
            env.close()
        except Exception as e:                                   # noqa: BLE001
            print(f"{n:6d} {'FAIL':>10}  {type(e).__name__}: {str(e)[:60]}")
            break
    simulation_app.close()


if __name__ == "__main__":
    main()
