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
wall_checks = 0
for task in TASKS:
    print(f"\n=== {task} ===")
    try:
        cfg = gym.spec(task).kwargs["env_cfg_entry_point"]()
        cfg.scene.num_envs = 8
        env = gym.make(task, cfg=cfg, disable_env_checker=True)
        obs, _ = env.reset()
        u = env.unwrapped
        w = obs["policy"].shape[-1] if hasattr(obs["policy"], "shape") else "?"
        def T(x):
            return x.torch if hasattr(x, "torch") else x
        root = T(u.scene["robot"].data.root_pos_w)
        local = root[:, :2] - T(u.scene.env_origins)[:, :2]
        from bhl_robust.terrains.maze_layout import BUTTON_AT, CORRIDOR_Y_LIMIT
        y_max = float(local[:, 1].abs().max())
        button = torch.tensor(BUTTON_AT[:2], device=local.device, dtype=local.dtype)
        dist = float((local - button).norm(dim=-1).mean())
        print(f"  spawn |y|_max {y_max:.3f} m  (limit {CORRIDOR_Y_LIMIT:.2f})  "
              f"mean button dist {dist:.2f} m")
        if y_max > CORRIDOR_Y_LIMIT:
            raise RuntimeError(f"spawned outside corridor: |y|={y_max:.3f}")
        cmd_name = getattr(u.cfg.commands.base_velocity, "class_type").__name__
        if cmd_name != "MazeWaypointCommand":
            raise RuntimeError(f"maze command is {cmd_name}, not MazeWaypointCommand")
        names = [n for n in ("progress_to_button", "button_reached", "dead_end")
                 if hasattr(u.cfg.rewards, n)]
        missing = [n for n in ("progress_to_button", "button_reached", "dead_end") if n not in names]
        if missing:
            raise RuntimeError(f"maze rewards missing {missing}")
        if not hasattr(u.cfg.terminations, "button_reached"):
            raise RuntimeError("maze terminations missing button_reached")
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
                if s == "lidar":
                    hits = arr
                    pos = d.pos_w
                    pos = pos.torch if hasattr(pos, "torch") else pos
                    ranges = torch.linalg.norm(hits - pos[:, None, :], dim=-1)
                    valid = torch.isfinite(ranges)
                    near = float(ranges[valid].min()) if bool(valid.any()) else float("inf")
                    frac = float(valid.float().mean())
                    print(f"  lidar wall gate: hits {frac:.3f}, nearest {near:.3f} m")
                    # At each terrain origin the corridor side walls are 0.45 m
                    # from the scanner.  The old scene-grid clones produced no
                    # nearby /World/ground hit at all.
                    if frac < 0.20 or not (0.30 < near < 0.65):
                        raise RuntimeError(
                            f"lidar does not see the terrain-mesh corridor: "
                            f"hit_fraction={frac:.3f}, nearest={near:.3f}"
                        )
                    wall_checks += 1
        env.close(); ok += 1
    except Exception as exc:
        import traceback; traceback.print_exc()
        print(f"  FAILED: {exc!r}")
print(f"\n=== maze smoke: {ok}/{len(TASKS)} ===")
if ok != len(TASKS) or wall_checks != 2:
    raise SystemExit(f"maze smoke failed: tasks={ok}/{len(TASKS)}, wall_checks={wall_checks}/2")
app.app.close()
PY
