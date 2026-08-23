"""Gate: do the N-robot crews build, step, and have the shapes they claim?

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
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args(); a.headless = True
app = AppLauncher(a).app

import torch, gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import berkeley_humanoid_lite.tasks  # noqa: F401,E402
import bhl_robust.tasks  # noqa: F401,E402

NJ = 22
ok = True
for n, vis in ((3, False), (4, False), (3, True), (4, True)):
    task = f"CoopLift-BHL-Cube-Crew{n}{'-Depth' if vis else ''}-v0"
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
        # One payload, N robots, N actions-worth of joints.
        checks = [
            (n_act == n * NJ, f"action width {n_act} == {n}x{NJ}"),
            (len(robots) == n, f"{len(robots)} robots in scene == {n}"),
            ("object" in u.scene.keys(), "exactly one payload"),
            (torch.isfinite(rew).all().item(), "reward finite"),
        ]
        for good, msg in checks:
            print(("  PASS " if good else "  FAIL ") + msg)
            ok = ok and good
        print(f"SMOKE OK | {task:<38} obs={shapes}")
        env.close()
    except Exception as e:
        ok = False
        import traceback; traceback.print_exc()
        print(f"SMOKE FAIL | {task} | {type(e).__name__}: {e}", flush=True)
print(f"SMOKE SUMMARY | {'ALL PASS' if ok else 'FAILURES ABOVE'}", flush=True)
# SimulationApp.close() terminates the interpreter, so the exit status has to be
# written somewhere the shell can read it -- a raise after close() never lands.
import os
with open(os.environ.get("GATE_OUT", "/tmp/crew_gate.txt"), "w") as f:
    f.write("ALL PASS\n" if ok else "FAILURES\n")
app.close()
