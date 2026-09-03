"""G-C1: cloth throughput on Isaac Lab's own Newton scene, assets localised.

Three earlier probes produced no number. The last one spawned a mesh with
`UsdFileCfg` and measured an empty simulation -- verification showed "physics
schemas on the cloth prim: NONE", because loading a mesh does not make it cloth.
Newton needs a `DeformableObjectCfg` whose spawner carries
`NewtonDeformableBodyPropertiesCfg`, and a `CoupledMJWarpVBDSolverCfg` on the
env. That composition lives behind Isaac Lab's `PresetCfg` machinery.

So rather than rebuild it, this measures Isaac Lab's own maintained cloth
scene -- which is genuinely a VBD cloth sim -- after replacing every remotely
hosted asset with a local primitive. The scene 404'd on a table and a sky
texture from the Omniverse content server, neither of which affects solver
throughput, and neither of which the sorting task would ever have used.

Substitution is by walking the scene config for any `usd_path` that is a URL,
rather than by naming the two assets that happened to fail today.
"""

from __future__ import annotations

import argparse

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--envs", type=int, nargs="+", default=[8, 32, 128, 512])
parser.add_argument("--steps", type=int, default=40)
parser.add_argument("--task", type=str, default="Isaac-Lift-Cloth-Franka-v0")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.sim as sim_utils  # noqa: E402


def _localise(scene_cfg) -> tuple[list[str], list[str]]:
    """Drop only the scene entries whose remote USD does not resolve.

    Three attempts died on this function being too clever. Substituting a cuboid
    with `rigid_props` made the prim a Newton physics body and tripped
    FrameView. Substituting a visual-only cuboid still guessed at the prim's
    role. Deleting *every* remote asset removed the Franka itself, and the
    scene's force-torque sites reference it:

        Site 'ft_2' with body_pattern '.../Robot/panda_link0' matched no
        prototype bodies

    So: HEAD each remote URL and drop only the ones that 404. The table was
    genuinely missing; the robot was never the problem. This makes no judgement
    about what a prim is for -- only about whether it exists -- and it reports
    both lists, so a scene gutted by accident is visible rather than silent.
    """
    import urllib.request

    dropped, kept = [], []
    for name in dir(scene_cfg):
        if name.startswith("_"):
            continue
        item = getattr(scene_cfg, name, None)
        spawn = getattr(item, "spawn", None)
        path = getattr(spawn, "usd_path", None)
        if not isinstance(path, str) or not path.startswith(("http://", "https://")):
            continue
        try:
            req = urllib.request.Request(path, method="HEAD")
            with urllib.request.urlopen(req, timeout=20) as r:
                ok = 200 <= r.status < 400
        except Exception:                                        # noqa: BLE001
            ok = False
        if ok:
            kept.append(name)
        else:
            setattr(scene_cfg, name, None)
            dropped.append(name)
    return dropped, kept


def main() -> None:
    if args_cli.task not in gym.registry:
        print(f"{args_cli.task} not registered")
        simulation_app.close()
        raise SystemExit(1)

    from isaaclab_tasks.utils import parse_env_cfg

    print(f"{'envs':>6} {'steps/s':>10} {'env-steps/s':>13}  note")
    for n in args_cli.envs:
        try:
            cfg = parse_env_cfg(args_cli.task, device=app_launcher.device, num_envs=n)
            dropped, kept = _localise(cfg.scene)
            env = gym.make(args_cli.task, cfg=cfg, disable_env_checker=True)
            env.reset()
            act = torch.zeros((n, env.unwrapped.action_space.shape[-1]),
                              device=env.unwrapped.device)
            for _ in range(5):
                env.step(act)
            torch.cuda.synchronize()
            t0 = time.time()
            for _ in range(args_cli.steps):
                env.step(act)
            torch.cuda.synchronize()
            dt = time.time() - t0
            sps = args_cli.steps / dt
            print(f"{n:6d} {sps:10.2f} {sps * n:13.0f}  dropped={dropped} kept={kept}")
            env.close()
        except Exception as e:                                   # noqa: BLE001
            print(f"{n:6d} {'FAIL':>10} {'':>13}  {type(e).__name__}: {str(e)[:70]}")
            import traceback
            traceback.print_exc()
            break
    simulation_app.close()


if __name__ == "__main__":
    main()
