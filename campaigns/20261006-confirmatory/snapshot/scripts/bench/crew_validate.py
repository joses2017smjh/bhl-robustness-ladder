"""Gate: does ONE N-robot crew build, step, and have the shape it claims?

One variant per invocation, deliberately. Building a second
`ManagerBasedRLEnv` in a process that has already built and closed one hangs
Isaac Sim indefinitely -- crew 3 passed in six minutes and crew 4 then produced
no output for the next fifty-four before the wall clock killed it. The same
thing happened earlier in this project to the third task in a three-task smoke
script, which passed on its own immediately afterwards. `env.close()` does not
give the stage back.

Run before any crew training. The failure this is guarding against is not a
crash -- a crash is easy. It is the one section 5 already documents twice: a
config that looks changed and is not, so an arm trains as a copy of its control
and the result is a plausible number that means nothing. Here that would be a
scene holding four robots and two payloads, or an action space still 44 wide.

Exits non-zero on any failure, so a training array can depend on it with
`afterok` rather than trusting a human to read the log.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser()
p.add_argument("--task", required=True)
p.add_argument("--crew", type=int, required=True)
AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import torch, gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import berkeley_humanoid_lite.tasks  # noqa: F401,E402
import bhl_robust.tasks  # noqa: F401,E402

NJ = 22
ok = True
task, n = a.task, a.crew
try:
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=8)
    env = gym.make(task, cfg=cfg)
    obs, _ = env.reset()
    act = torch.zeros(env.action_space.shape, device="cuda:0")
    for _ in range(3):
        obs, rew, term, trunc, _ = env.step(act)
    u = env.unwrapped
    n_act = env.action_space.shape[-1]
    shapes = {k: tuple(v.shape) for k, v in obs.items()}
    robots = [k for k in u.scene.keys() if k.startswith("robot_")]
    checks = [
        (n_act == n * NJ, f"action width {n_act} == {n}x{NJ}"),
        (len(robots) == n, f"{len(robots)} robots in scene == {n}"),
        ("object" in u.scene.keys(), "exactly one payload"),
        (torch.isfinite(rew).all().item(), "reward finite"),
    ]
    for good, msg in checks:
        print(("  PASS " if good else "  FAIL ") + msg, flush=True)
        ok = ok and good
    print(f"SMOKE OK | {task:<40} obs={shapes}", flush=True)
    env.close()
except Exception as e:
    ok = False
    import traceback; traceback.print_exc()
    print(f"SMOKE FAIL | {task} | {type(e).__name__}: {e}", flush=True)

print(f"SMOKE SUMMARY | {task} | {'ALL PASS' if ok else 'FAILURES ABOVE'}", flush=True)
# SimulationApp.close() terminates the interpreter, so the verdict has to be
# written where the shell can read it -- a raise after close() never lands.
import os  # noqa: E402
with open(os.environ["GATE_OUT"], "a") as f:
    f.write(f"{task} {'PASS' if ok else 'FAIL'}\n")
app.close()
