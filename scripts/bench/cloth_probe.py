"""G-C1: how many cloth environments does Newton sustain on this cluster?

Self-contained on purpose. The first version measured Isaac Lab's own
`Isaac-Lift-Cloth-Franka-v0`, which pulls a table, a Franka and a sky texture
from the Omniverse content server and died on one missing asset -- a dependency
the sorting task does not have, since it brings its own garment meshes. Chasing
someone else's demo assets was measuring the wrong thing anyway.

This spawns N cloth grids and steps them, and nothing else. The number it
produces decides the sorting task's whole shape: at ~1,000 envs it is an RL
problem, at ~16 it is a scripted demonstration with a rendered camera, and the
folding repo's CPU path is one env per process, which is what ruled RL out
there.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--envs", type=int, nargs="+", default=[16, 64, 256, 1024])
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--cloth-usd", type=str, default=None,
                    help="defaults to warp's square_cloth.usd, which ships locally")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import glob  # noqa: E402
import time  # noqa: E402

import torch  # noqa: E402


def _local_cloth() -> str | None:
    if args_cli.cloth_usd:
        return args_cli.cloth_usd
    for pat in (
        "/nfs/hpc/share/sanchej7/Humanoid_Lite/venv-isaac60/lib/python3.12/"
        "site-packages/warp/examples/assets/square_cloth.usd",
        "/nfs/hpc/share/sanchej7/Humanoid_Lite/venv-isaac60/lib/python3.12/"
        "site-packages/isaacsim/extscache/omni.warp.core-*/warp/examples/assets/square_cloth.usd",
    ):
        hits = sorted(glob.glob(pat))
        if hits:
            return hits[0]
    return None


def main() -> None:
    cloth = _local_cloth()
    print(f"cloth asset: {cloth}")
    if cloth is None:
        print("no local cloth USD found")
        simulation_app.close()
        raise SystemExit(1)

    import isaaclab.sim as sim_utils
    from isaaclab.sim import SimulationCfg, SimulationContext

    print(f"\n{'envs':>6} {'steps/s':>10} {'env-steps/s':>13}  note")
    for n in args_cli.envs:
        try:
            SimulationContext.clear_instance()
            sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0,
                                                  device=app_launcher.device))
            # A grid of cloth prims, one per notional env, with nothing else in
            # the scene -- the question is solver throughput, not scene realism.
            side = int(n ** 0.5) + 1
            for i in range(n):
                x, y = (i % side) * 1.5, (i // side) * 1.5
                sim_utils.UsdFileCfg(usd_path=cloth).func(
                    f"/World/cloth_{i}", sim_utils.UsdFileCfg(usd_path=cloth),
                    translation=(x, y, 1.0))
            sim.reset()
            for _ in range(10):
                sim.step()
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args_cli.steps):
                sim.step()
            torch.cuda.synchronize()
            dt = time.time() - t0
            sps = args_cli.steps / dt
            print(f"{n:6d} {sps:10.2f} {sps * n:13.0f}  ok")
        except Exception as e:                                   # noqa: BLE001
            print(f"{n:6d} {'FAIL':>10} {'':>13}  {type(e).__name__}: {str(e)[:60]}")
            break
    simulation_app.close()


if __name__ == "__main__":
    main()
