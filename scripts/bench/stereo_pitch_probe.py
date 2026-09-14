"""Where does the maze stereo pair actually look? Measured, not read off the config.

`STEREO_ROT = (0.9848, 0, 0.1736, 0)` is 20 degrees of down-pitch written
(w, x, y, z). Isaac Lab 3.0 reads offsets (x, y, z, w), so on the v60 stack
the same tuple is a half-turn: the prediction is that every maze stereo arm
trained with its cameras pitched 20 degrees *up* and upside down, seeing
almost nothing inside the 6 m range.

A prediction about a sensor is only worth re-running nine GPU-days for if the
sensor says so itself. So this builds the stereo arm, adds a third camera with
the old raw tuple beside the corrected pair, and reports for each camera:

  pitch     forward axis relative to the robot base, in degrees (down < 0)
  up_z      the camera's up axis in the base frame (+1 upright, -1 flipped)
  hits      fraction of pixels that return terrain inside max range
  rows      hit fraction down the image, top row first

Measured relative to the base so a robot leaning at reset cannot pass for a
camera pose. Verdict, and exit code, in one line the batch script greps --
Isaac can exit 0 on a crash, so the line is the evidence, not the status:

  STEREO-PITCH PASS   corrected pair looks down and sees terrain; raw looks up
  STEREO-PITCH NOBUG  raw already looked down -- the premise is wrong, re-run nothing
  STEREO-PITCH FAIL   anything else
"""

from __future__ import annotations

import argparse
import math

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task", default="Velocity-BHL-Maze-StereoP16-v0")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=3)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from isaaclab.utils.math import quat_apply, quat_apply_inverse  # noqa: E402

import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.quat_order import quat_order  # noqa: E402
from bhl_robust.sensors_rig import STEREO_ROT, make_stereo_cfg  # noqa: E402


def T(x):
    """ProxyArray on 3.0, tensor on 2.x."""
    return x.torch if hasattr(x, "torch") else x


def measure(env, name: str) -> dict:
    u = env.unwrapped
    cam = u.scene.sensors[name]
    robot = u.scene["robot"]
    n = u.num_envs
    qc = T(cam.data.quat_w_world)
    qb = T(robot.data.root_quat_w)
    ex = torch.tensor([1.0, 0.0, 0.0], device=qc.device).expand(n, 3)
    ez = torch.tensor([0.0, 0.0, 1.0], device=qc.device).expand(n, 3)
    fwd_b = quat_apply_inverse(qb, quat_apply(qc, ex))
    up_b = quat_apply_inverse(qb, quat_apply(qc, ez))
    d = T(cam.data.output["distance_to_image_plane"])[..., 0]
    hit = torch.isfinite(d) & (d < cam.cfg.max_distance - 1e-3)
    rows = hit.float().mean(dim=(0, 2))
    return {
        "pitch": math.degrees(math.asin(float(fwd_b[:, 2].mean().clamp(-1, 1)))),
        "up_z": float(up_b[:, 2].mean()),
        "hits": float(hit.float().mean()),
        "rows": [float(rows[i]) for i in torch.linspace(0, len(rows) - 1, 5).long()],
    }


def main() -> None:
    cfg = gym.spec(args_cli.task).kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = args_cli.num_envs
    raw = make_stereo_cfg("left", res=64)
    raw.offset.rot = tuple(STEREO_ROT)      # the tuple exactly as v60 got it before
    cfg.scene.stereo_raw = raw

    env = gym.make(args_cli.task, cfg=cfg, disable_env_checker=True)
    u = env.unwrapped
    env.reset()
    act = torch.zeros((u.num_envs, u.action_space.shape[-1]), device=u.device)
    for _ in range(args_cli.steps):
        env.step(act)

    grav_z = float(T(u.scene["robot"].data.projected_gravity_b)[:, 2].mean())
    print(f"quat order probed: {quat_order()}   base projected-gravity z {grav_z:+.3f} "
          f"(-1 upright)   configured corrected rot {tuple(u.scene.sensors['stereo_l'].cfg.offset.rot)}")
    res = {}
    for name in ("stereo_l", "stereo_r", "stereo_raw"):
        m = res[name] = measure(env, name)
        print(f"  {name:10} pitch {m['pitch']:+6.1f} deg   up_z {m['up_z']:+.2f}   "
              f"hits {m['hits']:.3f}   rows(top->bottom) "
              + " ".join(f"{r:.2f}" for r in m["rows"]))

    fixed_ok = all(-25.0 < res[c]["pitch"] < -15.0 and res[c]["up_z"] > 0.8 and res[c]["hits"] > 0.3
                   for c in ("stereo_l", "stereo_r"))
    raw_up = res["stereo_raw"]["pitch"] > 10.0
    raw_down = res["stereo_raw"]["pitch"] < -10.0
    if fixed_ok and raw_up:
        verdict = "PASS"
    elif fixed_ok and raw_down:
        verdict = "NOBUG"
    else:
        verdict = "FAIL"
    print(f"STEREO-PITCH {verdict}  corrected pitch {res['stereo_l']['pitch']:+.1f}/"
          f"{res['stereo_r']['pitch']:+.1f} hits {res['stereo_l']['hits']:.3f}  "
          f"raw pitch {res['stereo_raw']['pitch']:+.1f} hits {res['stereo_raw']['hits']:.3f}",
          flush=True)
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
