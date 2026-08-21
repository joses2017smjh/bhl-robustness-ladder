"""Check the ray-cast depth against closed-form geometry before trusting it.

Warp ray-casting has a specific failure mode: `mesh_prim_paths` names the meshes
that get loaded, and naming them wrong produces an all-NaN depth image rather
than an error. A training run on that silently learns from a constant, and the
reward curve looks plausible the whole time. So the sensor is validated against
a surface whose depth image can be written down.

On flat ground the answer is exact. With the camera at height h, optical axis
pitched theta below horizontal, and normalised image coordinate y_n = (v-cy)/fy
increasing downward:

    Z(u, v) = h / (sin(theta) + y_n * cos(theta))

which is `distance_to_image_plane` -- the camera-frame z of the hit, not the
slant range. Three things are checked: that the image is finite, that it matches
the closed form, and that it stops matching when the ground stops being flat.
That last one matters because an all-constant sensor would pass the first two on
a plane.
"""
import argparse, math, os

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--res", type=int, default=64)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--tol", type=float, default=0.03, help="relative tolerance")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

simulation_app = AppLauncher(args).app

import numpy as np  # noqa: E402
import torch  # noqa: E402
import gymnasium as gym  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402
import berkeley_humanoid_lite.tasks  # noqa: F401,E402
import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.tasks.depth_env_cfg import (  # noqa: E402
    CAM_APERTURE, CAM_FOCAL, CAM_RANGE, make_depth_camera_cfg,
)

OUT = os.environ.get("BENCH_OUT", "/tmp/bench_results.txt")
lines: list[str] = []


def log(s: str) -> None:
    lines.append(s)
    print(s, flush=True)


# The overlay's camera rotation is a quaternion; recover the pitch it encodes
# rather than restating 20 degrees, so this check fails if the pose is edited.
from bhl_robust.tasks.depth_env_cfg import CAM_ROT  # noqa: E402
THETA = 2.0 * math.asin(CAM_ROT[2])
log(f"CHECK  | camera pitch from cfg quaternion: {math.degrees(THETA):.2f} deg below horizontal")


def analytic(h: float, res: int) -> np.ndarray:
    """Closed-form depth image of a flat plane at z = 0."""
    f_px = res * CAM_FOCAL / CAM_APERTURE
    c = (res - 1) / 2.0
    v = np.arange(res)[:, None].repeat(res, axis=1)
    y_n = (v - c) / f_px
    denom = math.sin(THETA) + y_n * math.cos(THETA)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = h / denom
    z[denom <= 0] = np.inf          # rays above the horizon never hit
    return np.clip(z, 0.0, CAM_RANGE)


def measure(task: str, difficulty_note: str):
    cfg = parse_env_cfg(task, device="cuda:0", num_envs=args.num_envs)
    cfg.scene.depth_cam = make_depth_camera_cfg(res=args.res)
    env = gym.make(task, cfg=cfg)
    env.reset()
    act = torch.zeros(env.action_space.shape, device="cuda:0")
    # One step so the sensor buffers fill; more would let the robot start
    # toppling and the base would no longer be level.
    env.step(act)
    sensor = env.unwrapped.scene["depth_cam"]
    d = sensor.data.output["distance_to_image_plane"][:, ..., 0].cpu().numpy()
    h = sensor.data.pos_w[:, 2].cpu().numpy()
    env.close()
    log(f"CHECK  | {difficulty_note}: camera height {h.mean():.3f} m, "
        f"finite {np.isfinite(d).mean():.3f}")
    return d, float(h.mean())


# --- 1. flat ground: must match the closed form -----------------------------
flat, h = measure("Velocity-Berkeley-Humanoid-Lite-Biped-v0", "flat plane")
if not np.isfinite(flat).all():
    log("FAIL   | non-finite depth on flat ground -- check mesh_prim_paths")

exp = analytic(h, args.res)
img = flat[0]
# Row order is a convention, not a guarantee; report which one matches rather
# than asserting one and calling a flip a failure.
cands = {"as-returned": img, "row-flipped": img[::-1]}
best_name, best_err = None, np.inf
for name, cand in cands.items():
    m = np.isfinite(exp) & (exp < CAM_RANGE * 0.999) & (cand < CAM_RANGE * 0.999)
    if m.sum() < 16:
        continue
    err = float(np.abs(cand[m] - exp[m]).mean() / np.abs(exp[m]).mean())
    log(f"CHECK  | {name:12s}: mean relative error {err:.4f} over {int(m.sum())} px")
    if err < best_err:
        best_name, best_err = name, err

verdict = "PASS" if best_err <= args.tol else "FAIL"
log(f"{verdict}   | flat-ground depth matches closed form to {best_err:.4f} "
    f"({best_name}, tol {args.tol})")

# --- 2. rough ground: must NOT match it -------------------------------------
rough, h2 = measure("Velocity-BHL-Biped-Bumpy-v0", "generated terrain")
exp2 = analytic(h2, args.res)
m2 = np.isfinite(exp2) & (exp2 < CAM_RANGE * 0.999) & (rough[0] < CAM_RANGE * 0.999)
img2 = rough[0] if best_name == "as-returned" else rough[0][::-1]
if m2.sum() >= 16:
    err2 = float(np.abs(img2[m2] - exp2[m2]).mean() / np.abs(exp2[m2]).mean())
    log(f"CHECK  | rough-ground departure from the flat model: {err2:.4f}")
    log(("PASS   | " if err2 > best_err * 3 else "FAIL   | ")
        + "terrain moves the depth image (a constant sensor would not)")
# Per-env spread is the other thing a stuck sensor cannot fake: different
# environments sit on different terrain tiles and must disagree.
spread = float(np.nanstd(rough.reshape(rough.shape[0], -1).mean(axis=1)))
log(f"CHECK  | across-env spread of mean depth on terrain: {spread:.4f} m")

with open(OUT, "a") as f:
    f.write("\n".join(lines) + "\n")
simulation_app.close()
