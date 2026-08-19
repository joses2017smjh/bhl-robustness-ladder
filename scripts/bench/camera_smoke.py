"""Minimal tiled-camera test: does depth rendering work AT ALL on this GPU?

Deliberately independent of the BHL environment, so a crash here means the
hardware/driver/Isaac Sim combination cannot render, while a pass here plus a
crash in the full env means the env-side camera config is wrong.
"""
import argparse, os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=32)
parser.add_argument("--num", type=int, default=4)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True
args.enable_cameras = True

simulation_app = AppLauncher(args).app

import torch  # noqa: E402
import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import AssetBaseCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors import TiledCameraCfg  # noqa: E402
from isaaclab.utils import configclass  # noqa: E402

OUT = os.environ.get("BENCH_OUT", "/tmp/bench_results.txt")


@configclass
class Cfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground",
                          spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light",
                         spawn=sim_utils.DistantLightCfg(intensity=3000.0))
    cam = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/cam",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.0, 0.0, 1.0),
                                        rot=(0.7071, 0.0, 0.7071, 0.0),
                                        convention="world"),
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0, focus_distance=400.0,
                                         horizontal_aperture=20.955,
                                         clipping_range=(0.05, 6.0)),
        width=args.res, height=args.res,
    )


sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device="cuda:0"))
scene = InteractiveScene(Cfg(num_envs=args.num, env_spacing=2.0))
sim.reset()

for _ in range(5):
    sim.step()
    scene.update(0.005)

d = scene["cam"].data.output["distance_to_image_plane"]
line = (f"CAMERA OK | {args.num} cams @ {args.res}x{args.res} | tensor {tuple(d.shape)} "
        f"| finite={torch.isfinite(d).float().mean().item():.3f} "
        f"| range [{d[torch.isfinite(d)].min().item():.2f}, {d[torch.isfinite(d)].max().item():.2f}] m")
with open(OUT, "a") as f:
    f.write(line + "\n")
print(line, flush=True)

simulation_app.close()
