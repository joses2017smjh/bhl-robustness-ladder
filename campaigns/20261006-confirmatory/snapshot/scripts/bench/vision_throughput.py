"""Measure tiled depth-camera rendering throughput on this cluster's GPUs.

The gate for vision-in-the-loop locomotion is not "does it run" but "does it run
fast enough to train". A locomotion policy needs on the order of 10^8 env-steps;
if rendering drops throughput below roughly 10k steps/s the experiment stops
being a two-week project.

Reports steps/s with and without cameras, so the rendering tax is isolated
rather than confounded with the physics cost.
"""
import argparse, time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--res", type=int, default=64, help="square depth resolution")
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--no_camera", action="store_true")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
if not args.no_camera:
    args.enable_cameras = True

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import berkeley_humanoid_lite.tasks  # noqa: F401,E402

TASK = "Velocity-Berkeley-Humanoid-Lite-Biped-v0"
env_cfg = parse_env_cfg(TASK, device="cuda:0", num_envs=args.num_envs)

if not args.no_camera:
    # Forward-and-down looking depth camera on the base, the pose a perceptive
    # locomotion policy would actually use.
    env_cfg.scene.depth_cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/base/front_cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.12, 0.0, 0.62),
                                        rot=(0.8924, 0.0, 0.4512, 0.0),
                                        convention="world"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=18.0, focus_distance=400.0,
            horizontal_aperture=20.955, clipping_range=(0.05, 6.0),
        ),
        width=args.res, height=args.res,
    )

env = gym.make(TASK, cfg=env_cfg)
env.reset()
act = torch.zeros(env.action_space.shape, device="cuda:0")

for _ in range(10):          # warm up: shader compile, first-frame allocation
    env.step(act)
torch.cuda.synchronize()

t0 = time.perf_counter()
for _ in range(args.steps):
    env.step(act)
torch.cuda.synchronize()
dt = time.perf_counter() - t0

sps = args.num_envs * args.steps / dt
mode = "physics only" if args.no_camera else f"depth {args.res}x{args.res}"
line = (f"RESULT | {mode:18s} | envs={args.num_envs:5d} | "
        f"{sps:9.0f} env-steps/s | {dt/args.steps*1000:6.1f} ms/step")

# Isaac Sim's shutdown can hard-exit the process before Python flushes stdout,
# so the number is appended to a file before anything is torn down.
import os
with open(os.environ.get("BENCH_OUT", "/tmp/bench_results.txt"), "a") as f:
    f.write(line + "\n")
print(line, flush=True)

env.close()
simulation_app.close()
