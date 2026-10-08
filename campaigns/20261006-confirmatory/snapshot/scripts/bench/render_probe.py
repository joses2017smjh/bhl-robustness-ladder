#!/usr/bin/env python3
"""Why do Isaac renders of trained tasks show no robot?

The spawn photographs (`slurm/99s_spawn_shot.sbatch`) show a robot. Every clip
recorded through `train_play --video` since -- the maze arms, the cloth scene,
the task-env spawn render -- shows the scene and, for the maze, the velocity
arrows riding on the root, but no body. Those two pipelines differ in two ways:
a camera *sensor* against the *viewport* render product, and a robot that barely
moved against one that physics has moved a long way.

This measures instead of guessing. For one task per process, with fabric on or
off, it resets, steps zero actions, and at a few steps records:

* the articulation root and the lowest / highest body, env-local, from physics;
* the same base link's position as USD reports it, which is what a renderer
  reading USD would draw -- a large gap with fabric on means stale transforms;
* how many mesh prims sit under the robot, how many compute as invisible, and
  their purposes (an all-`guide` or all-invisible visual set renders nothing);
* a viewport frame (`env.render()`) and a camera-sensor frame aimed at the root.

No policy and no checkpoint.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", required=True)
parser.add_argument("--steps", type=str, default="0,10,50")
parser.add_argument("--out-dir", type=str, required=True)
parser.add_argument("--disable_fabric", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
args_cli.enable_cameras = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import CameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402

TAG = "usd" if args_cli.disable_fabric else "fabric"
OUT = Path(args_cli.out_dir)
OUT.mkdir(parents=True, exist_ok=True)
EYE = (2.0, 2.0, 1.2)


def _cpu(x) -> torch.Tensor:
    return torch.as_tensor(x[:]).detach().float().cpu()


def _save(arr, path: Path) -> str:
    try:
        from PIL import Image
        a = np.asarray(arr)
        if a.dtype != np.uint8:
            a = np.clip(a, 0, 255).astype(np.uint8)
        Image.fromarray(a[..., :3]).save(path)
        return str(path)
    except Exception as exc:  # keep measuring even if a frame cannot be written
        return f"not saved: {exc!r}"


def _usd_report(prim_path: str) -> dict:
    from pxr import Usd, UsdGeom
    import omni.usd
    stage = omni.usd.get_context().get_stage()
    root = stage.GetPrimAtPath(prim_path)
    if not root or not root.IsValid():
        return {"prim_path": prim_path, "valid": False}
    meshes = [p for p in Usd.PrimRange(root, Usd.TraverseInstanceProxies())
              if p.IsA(UsdGeom.Gprim)]
    vis = Counter(str(UsdGeom.Imageable(p).ComputeVisibility()) for p in meshes)
    purpose = Counter(str(UsdGeom.Imageable(p).ComputePurpose()) for p in meshes)
    base = stage.GetPrimAtPath(prim_path + "/base")
    base_t = None
    if base and base.IsValid():
        m = UsdGeom.Xformable(base).ComputeLocalToWorldTransform(Usd.TimeCode.Default())
        base_t = [round(float(v), 4) for v in m.ExtractTranslation()]
    return {"prim_path": prim_path, "valid": True, "loaded": bool(root.IsLoaded()),
            "gprims": len(meshes), "visibility": dict(vis), "purpose": dict(purpose),
            "usd_base_world": base_t}


def main() -> None:
    env_cfg = parse_env_cfg(args_cli.task, device="cuda:0", num_envs=1,
                            use_fabric=not args_cli.disable_fabric)
    env_cfg.scene.probe_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/probe_cam", height=360, width=640, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
        offset=CameraCfg.OffsetCfg(pos=EYE, rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    scene = env.unwrapped.scene
    art_name = next(iter(scene.articulations))
    art = scene.articulations[art_name]
    prim_path = art.cfg.prim_path.replace("env_.*", "env_0")
    try:
        env.unwrapped.cfg.viewer.origin_type = "asset_root"
        env.unwrapped.cfg.viewer.asset_name = art_name
    except Exception:
        pass
    env.reset()
    act = torch.zeros((1, env.unwrapped.action_space.shape[-1]), device=env.unwrapped.device)
    steps = sorted(int(s) for s in args_cli.steps.split(","))
    records, k = [], 0
    for target in steps:
        while k < target:
            env.step(act)
            k += 1
        origin = _cpu(scene.env_origins)[0]
        root = _cpu(art.data.root_pos_w)[0, :3] - origin
        body = _cpu(art.data.body_pos_w)[0] - origin
        cam = scene["probe_cam"]
        w_root = _cpu(art.data.root_pos_w)[0, :3]
        eye = (w_root + torch.tensor(EYE)).unsqueeze(0).to(env.unwrapped.device)
        cam.set_world_poses_from_view(eye, w_root.unsqueeze(0).to(env.unwrapped.device))
        env.step(act)
        k += 1
        cam_rgb = _cpu(cam.data.output["rgb"])[0].numpy()
        view_rgb = env.render()
        stem = f"{args_cli.task}_{TAG}_s{k:03d}"
        rec = {
            "step": k, "task": args_cli.task, "mode": TAG, "articulation": art_name,
            "root_local": [round(float(v), 4) for v in root],
            "body_z_min": round(float(body[:, 2].min()), 4),
            "body_z_max": round(float(body[:, 2].max()), 4),
            "env_origin": [round(float(v), 4) for v in origin],
            "physics_base_world": [round(float(v), 4) for v in _cpu(art.data.body_pos_w)[0, 0]],
            "usd": _usd_report(prim_path),
            "camera_png": _save(cam_rgb, OUT / f"{stem}_camera.png"),
            "viewport_png": _save(view_rgb, OUT / f"{stem}_viewport.png") if view_rgb is not None else "render() returned None",
        }
        records.append(rec)
        print(json.dumps(rec))
    (OUT / f"{args_cli.task}_{TAG}.json").write_text(json.dumps(records, indent=2))
    env.close()
    print("render probe done")


if __name__ == "__main__":
    main()
    simulation_app.close()
