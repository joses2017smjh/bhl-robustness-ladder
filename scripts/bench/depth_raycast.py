"""Depth sensing on a cluster whose RTX renderer will not start.

Isaac Sim 5.1's RTX renderer segfaults at `omni.usd.create_hydra_engine` on this
cluster's driver, which takes `TiledCamera` -- and with it every RGB/depth
annotator -- off the table. `RayCasterCamera` is a second, independent depth
path: it intersects a pinhole ray bundle against the scene's meshes in warp on
the GPU. No Hydra engine, no render product, no `--enable_cameras`.

The two are not interchangeable. RTX gives radiance; a ray-cast camera gives
range and nothing else. For perceptive locomotion -- where the policy consumes
`distance_to_image_plane` and never a colour channel -- that difference does not
matter, and the ray-cast path is the faster of the two anyway.

Reports steps/s with and without the sensor so the depth tax is isolated rather
than confounded with the physics cost.
"""
import argparse, os, time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--task", default="Velocity-BHL-Biped-Bumpy-v0")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--res", type=int, default=64, help="square depth resolution")
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--no_camera", action="store_true")
parser.add_argument("--save_npz", default="", help="dump the first env's depth stack here")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
# Deliberately NOT setting args.enable_cameras: that selects the rendering
# experience file, which starts the RTX renderer and segfaults. The ray-cast
# camera works in the plain headless app.

simulation_app = AppLauncher(args).app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab.sensors import RayCasterCameraCfg  # noqa: F401,E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import berkeley_humanoid_lite.tasks  # noqa: F401,E402
import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.tasks.depth_env_cfg import make_depth_camera_cfg  # noqa: E402

env_cfg = parse_env_cfg(args.task, device="cuda:0", num_envs=args.num_envs)
if not args.no_camera:
    env_cfg.scene.depth_cam = make_depth_camera_cfg(res=args.res)

env = gym.make(args.task, cfg=env_cfg)
env.reset()
act = torch.zeros(env.action_space.shape, device="cuda:0")

for _ in range(10):          # warm up: warp kernel JIT, BVH build, first alloc
    env.step(act)
torch.cuda.synchronize()

frames = []
t0 = time.perf_counter()
for _ in range(args.steps):
    env.step(act)
    if args.save_npz and not args.no_camera:
        d = env.unwrapped.scene["depth_cam"].data.output["distance_to_image_plane"]
        frames.append(d[0, ..., 0].clone())
torch.cuda.synchronize()
dt = time.perf_counter() - t0

sps = args.num_envs * args.steps / dt
mode = "physics only" if args.no_camera else f"raycast depth {args.res}x{args.res}"
line = (f"RESULT | {mode:26s} | envs={args.num_envs:5d} | "
        f"{sps:9.0f} env-steps/s | {dt/args.steps*1000:6.1f} ms/step")

if not args.no_camera:
    d = env.unwrapped.scene["depth_cam"].data.output["distance_to_image_plane"]
    finite = torch.isfinite(d)
    line += (f"\nDEPTH  | tensor {tuple(d.shape)} | finite={finite.float().mean().item():.3f}"
             f" | range [{d[finite].min().item():.2f}, {d[finite].max().item():.2f}] m")

if args.save_npz and frames:
    import numpy as np
    os.makedirs(os.path.dirname(args.save_npz) or ".", exist_ok=True)
    np.savez_compressed(args.save_npz, depth=torch.stack(frames).cpu().numpy())
    line += f"\nSAVED  | {len(frames)} frames -> {args.save_npz}"

# Isaac Sim's shutdown can hard-exit before Python flushes stdout, so the
# number is appended to a file before anything is torn down.
with open(os.environ.get("BENCH_OUT", "/tmp/bench_results.txt"), "a") as f:
    f.write(line + "\n")
print(line, flush=True)

env.close()
simulation_app.close()
