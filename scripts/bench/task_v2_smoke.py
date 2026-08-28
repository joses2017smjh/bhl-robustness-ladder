"""Build and step each redesigned task, and report what the policy will see.

Nine configs that import are not nine configs that construct. This builds every
one at a handful of envs, steps it, and prints the observation width and whether
the success term is wired -- the two things that are silently wrong most often.
A task whose `terminations.success` is missing trains perfectly happily and can
never finish, which is the failure the redesign exists to remove.

Run under `BHL_STACK=v60 ENABLE_CAMERAS=1`; the sighted arms need RTX.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=6)
parser.add_argument("--only", type=str, default=None,
                    help="substring filter over task ids")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402  registers the ids

def _clear_sim() -> None:
    """Drop the SimulationContext between tasks.

    Isaac Lab allows one context per process. Without this, the first task to
    fail leaves its context standing and every later task reports "Simulation
    context already exists" -- so a run that tested one task would look like a
    run that tested nine, with eight identical and meaningless errors.
    """
    try:
        from isaaclab.sim import SimulationContext
        SimulationContext.clear_instance()
    except Exception:                                            # noqa: BLE001
        pass


TASKS = [f"TaskV2-BHL-{t}-{v}-v0"
         for t in ("CubeToShelf", "BallToNet", "PlankToWall")
         for v in ("Blind", "Depth", "Rgb")]


def main() -> None:
    rows, bad = [], 0
    for tid in TASKS:
        if args_cli.only and args_cli.only not in tid:
            continue
        try:
            cfg = gym.spec(tid).kwargs["env_cfg_entry_point"]()
            cfg.scene.num_envs = args_cli.num_envs
            cfg.sim.device = app_launcher.device
            env = gym.make(tid, cfg=cfg, disable_env_checker=True)
            obs, _ = env.reset()
            width = int(obs["policy"].shape[-1]) if isinstance(obs, dict) else int(obs.shape[-1])
            has_success = "success" in env.unwrapped.termination_manager.active_terms
            act = torch.zeros((args_cli.num_envs, env.unwrapped.action_space.shape[-1]),
                              device=env.unwrapped.device)
            for _ in range(args_cli.steps):
                env.step(act)
            rows.append((tid, width, has_success, "ok"))
            env.close()
        except Exception as e:                                   # noqa: BLE001
            bad += 1
            rows.append((tid, -1, False, f"{type(e).__name__}: {e}"[:88]))
            if bad == 1:
                # Full traceback for the first failure only. Nine identical
                # truncated one-liners say nothing that one traceback does not,
                # and the truncation hid the actual call site last time.
                import traceback
                print(f"\n--- first failure, {tid} ---")
                traceback.print_exc()
                print("--- end ---\n")
        finally:
            _clear_sim()

    print(f"\n{'task':38} {'obs':>5} {'success term':>13}  note")
    for tid, w, s, note in rows:
        print(f"{tid:38} {w:5d} {str(s):>13}  {note}")
    missing = [r[0] for r in rows if r[3] == "ok" and not r[2]]
    if missing:
        print(f"\nno success termination on: {missing}")
    print(f"\n{len(rows) - bad}/{len(rows)} built and stepped")
    simulation_app.close()
    raise SystemExit(1 if (bad or missing) else 0)


if __name__ == "__main__":
    main()
