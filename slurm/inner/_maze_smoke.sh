#!/bin/bash
set -euo pipefail
cd "$REPO"; export PYTHONPATH="$REPO/src:${PYTHONPATH:-}"
"$PY" - <<'PY'
"""Do the four maze arms construct, reset and step, and are the sensors live?

A gate that does not run the thing it gates is decoration -- the v2 env smoke
passed 9/9 through every one of the training failures because it never executed
the training path. This one steps each arm and prints the observation width and
the sensors' actual readings, so a sensor wired but returning constants is
visible rather than silent.
"""
import argparse
from isaaclab.app import AppLauncher
p = argparse.ArgumentParser(); AppLauncher.add_app_launcher_args(p)
a = p.parse_args([]); a.headless = True; a.enable_cameras = True
app = AppLauncher(a)

import gymnasium as gym, torch
import bhl_robust.tasks  # noqa: F401

TASKS = ["Velocity-BHL-Maze-Blind-v0", "Velocity-BHL-Maze-Lidar-v0",
         "Velocity-BHL-Maze-Stereo-v0", "Velocity-BHL-Maze-Both-v0"]
ok = 0
for task in TASKS:
    print(f"\n=== {task} ===")
    try:
        cfg = gym.spec(task).kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = 8
        env = gym.make(task, cfg=cfg, disable_env_checker=True)
        obs, _ = env.reset()
        u = env.unwrapped
        w = obs["policy"].shape[-1] if hasattr(obs["policy"], "shape") else "?"
        act = torch.zeros((8, u.action_space.shape[-1]), device=u.device)
        for _ in range(5):
            obs, *_ = env.step(act)
        print(f"  obs width {w}   stepped 5x ok")
        for s in ("lidar", "stereo_l", "stereo_r"):
            if s in u.scene.sensors:
                d = u.scene.sensors[s].data
                arr = getattr(d, "ray_hits_w", None)
                if arr is None:
                    arr = d.output["distance_to_image_plane"]
                arr = arr.torch if hasattr(arr, "torch") else arr
                fin = torch.isfinite(arr)
                print(f"  {s:9} shape {tuple(arr.shape)}  finite {fin.float().mean():.2f}  "
                      f"range [{arr[fin].min():.2f}, {arr[fin].max():.2f}]")
        env.close(); ok += 1
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"  FAILED: {exc!r}")
print(f"\n=== maze smoke: {ok}/{len(TASKS)} ===")
app.app.close()
PY
