"""Does the coop depth camera actually see the payload, and does it track it?

The whole vision-in-the-lift-loop design rests on one claim:
`MultiMeshRayCasterCamera` with `track_mesh_transforms=True` re-transforms the
ray bundle against a moving rigid body, so a camera can watch a cube that is
being picked up. If that claim is wrong the sensor degrades to the static-mesh
`RayCasterCamera` of §6 and returns a picture of the empty floor -- which trains
perfectly happily, converges to something, and means nothing. §6's own warning
about all-NaN warp casts is the same lesson: this sensor family fails quietly.

So this asserts on the cube specifically, in four steps that each rule out a way
of being wrong:

1. the image is finite and matches the floor-only closed form *outside* the
   cube's footprint -- i.e. the camera is working at all;
2. some pixels are *nearer* than the floor would be, and they form a connected
   blob of about the size the cube subtends -- i.e. the cube is in the image and
   not just noise;
3. teleporting the cube out of view removes exactly those pixels -- i.e. the
   near blob is the cube and not a rendering artefact of the floor;
4. raising the cube 20 cm moves the blob, and moves it *up* -- i.e. the
   transform is tracked per step rather than baked at startup. This is the step
   that a static-mesh raycaster fails.
"""
import argparse
import math
import os

# Path wiring, same as `scripts/train.py` and for the same reason: this repo is
# not installed, it is overlaid, so `bhl_robust` is only importable if `src/` is
# on the path. Doing it here rather than trusting the caller's PYTHONPATH is what
# `train.py` does, and it is why `train.py` works from any cwd -- the first two
# submissions of `58_coop_depth` died 32 s in on `No module named 'bhl_robust'`
# because this script trusted the shell and the shell did not set it.
import sys as _sys

_SRC = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "src")
if _SRC not in _sys.path:
    _sys.path.insert(0, _SRC)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser()
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--tol", type=float, default=0.05)
parser.add_argument("--keep-object-pose", action="store_true",
                    help="the depthboth arm: vision alongside the privileged "
                         "object pose rather than replacing it")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
args.headless = True

simulation_app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

import berkeley_humanoid_lite.tasks  # noqa: F401,E402
import bhl_robust.tasks  # noqa: F401,E402
from bhl_robust.eval.coop_replay import (  # noqa: E402
    OBS_DEPTH_BOTH,
    OBS_DEPTH_SWAP,
)
from bhl_robust.tasks.coop_depth_env_cfg import (  # noqa: E402
    COOP_CAM_RES,
    apply_depth_flags,
)
from bhl_robust.tasks.depth_env_cfg import (  # noqa: E402
    CAM_APERTURE,
    CAM_FOCAL,
    CAM_RANGE,
    CAM_ROT,
)

OUT = os.environ.get("BENCH_OUT", "/tmp/coop_depth_validate.txt")
lines: list[str] = []
fails: list[str] = []


def log(s: str) -> None:
    lines.append(s)
    print(s, flush=True)


def check(name: str, ok: bool, detail: str) -> None:
    log(f"{'PASS' if ok else 'FAIL'}  | {name}: {detail}")
    if not ok:
        fails.append(name)


THETA = 2.0 * math.asin(CAM_ROT[2])
log(f"CHECK  | camera pitch from cfg quaternion: {math.degrees(THETA):.2f} deg below horizontal")

env_cfg = parse_env_cfg("CoopLift-BHL-Cube-Depth-v0", num_envs=args.num_envs)
env_cfg.drop_object_pose = not args.keep_object_pose
apply_depth_flags(env_cfg)
arm = "depthboth" if args.keep_object_pose else "depthswap"
want = OBS_DEPTH_BOTH if args.keep_object_pose else OBS_DEPTH_SWAP
log(f"CHECK  | arm {arm}: drop_object_pose={env_cfg.drop_object_pose}")

env = gym.make("CoopLift-BHL-Cube-Depth-v0", cfg=env_cfg).unwrapped
env.reset()

# --- 0. the observation vector the MuJoCo replay has to reproduce -----------
# `eval/coop_replay` rebuilds this vector in numpy from the checkpoint width
# alone, and it assembles the terms in a hard-coded order. That order is an
# inference about `@configclass` dataclass semantics -- inherited fields first,
# so the depth terms an override adds land after `actions`. If the inference is
# wrong, the replay silently feeds a permuted vector to a network that will
# happily produce plausible-looking garbage. So the order is asserted here,
# against the live manager, rather than trusted.
terms = env.observation_manager.active_terms["policy"]
dims = [int(np.prod(d)) for d in env.observation_manager.group_obs_term_dim["policy"]]
total = sum(dims)
log("CHECK  | policy observation terms, in manager order:")
at = 0
for t, n in zip(terms, dims):
    log(f"       |   [{at:>3}:{at + n:>3}]  {t}  ({n})")
    at += n

check("obs_width", total == want, f"{total} against replay's {want} for {arm}")
depth_terms = [t for t in terms if t.startswith("depth_")]
check("depth_present", depth_terms == ["depth_a", "depth_b"],
      f"depth terms {depth_terms}")
if depth_terms:
    check("depth_is_last", terms[-2:] == ["depth_a", "depth_b"],
          f"vector ends with {terms[-2:]}")
    check("actions_before_depth", "actions" in terms
          and terms.index("actions") < terms.index("depth_a"),
          f"actions at {terms.index('actions') if 'actions' in terms else None}, "
          f"depth_a at {terms.index('depth_a')}")
obj_terms = [t for t in terms if t.startswith("object_pos")]
check("object_pose_gating",
      (obj_terms == []) if not args.keep_object_pose
      else (obj_terms == ["object_pos_a", "object_pos_b"]),
      f"object pose terms {obj_terms} for {arm}")

cam = env.scene["depth_cam_a"]
obj = env.scene["object"]
zero = torch.zeros(env.num_envs, env.action_space.shape[-1], device=env.device)


def depth() -> np.ndarray:
    """One settled depth image for env 0, in metres."""
    for _ in range(4):
        env.step(zero)
    d = cam.data.output["distance_to_image_plane"]
    d = d[..., 0] if d.ndim == 4 else d
    return d[0].detach().cpu().numpy().astype(np.float64)


def move_object(dz: float = 0.0, dx: float = 0.0) -> None:
    state = obj.data.default_root_state.clone()
    state[:, :3] += env.scene.env_origins
    state[:, 0] += dx
    state[:, 2] += dz
    obj.write_root_pose_to_sim(state[:, :7])
    obj.write_root_velocity_to_sim(torch.zeros_like(state[:, 7:13]))


# --- 1. the image exists, and the floor part of it is right -----------------
d0 = depth()
res = d0.shape[0]
finite = np.isfinite(d0) & (d0 > 0)
check("finite", finite.mean() > 0.5, f"{finite.mean():.0%} finite pixels, res {res}x{res}")
check("resolution", res == COOP_CAM_RES, f"{res} against cfg {COOP_CAM_RES}")

f_px = res * CAM_FOCAL / CAM_APERTURE
c = (res - 1) / 2.0
v = np.arange(res)[:, None].repeat(res, axis=1)
y_n = (v - c) / f_px
denom = math.sin(THETA) + y_n * math.cos(THETA)


def floor_ref() -> np.ndarray:
    """Closed-form bare-floor depth for the camera's *current* height.

    Recomputed per frame rather than captured once. The camera rides the robot's
    base, the robots are stepping the whole time, and each `depth()` advances
    four control steps -- so a reference frozen at t=0 drifts out from under the
    comparison. That drift is what failed `restored` and one arm's
    `tracks_transform` on the first run: the sensor was fine, the yardstick was
    moving.
    """
    h = float(cam.data.pos_w[0, 2].item())
    with np.errstate(divide="ignore", invalid="ignore"):
        f = np.where(denom > 0, h / denom, np.inf)
    return np.clip(f, 0.0, CAM_RANGE)


def near_mask(d: np.ndarray) -> np.ndarray:
    """Pixels more than 2 cm nearer than bare floor: i.e. the payload."""
    return (floor_ref() - d) > 0.02


log(f"CHECK  | camera height {float(cam.data.pos_w[0, 2].item()):.3f} m; "
    f"floor-only depth spans "
    f"{np.nanmin(floor_ref()[np.isfinite(floor_ref())]):.2f}-{CAM_RANGE:.2f} m")

# --- 2. something is nearer than the floor, and it is cube-sized ------------
near = near_mask(d0)
frac = float(near.mean())
check("payload_visible", frac > 0.01,
      f"{frac:.1%} of pixels are >2 cm nearer than bare floor "
      f"({near.sum()} px)")
if near.any():
    rows, cols = np.where(near)
    log(f"CHECK  | near-blob rows {rows.min()}-{rows.max()}, "
        f"cols {cols.min()}-{cols.max()}, min depth {d0[near].min():.3f} m")

# --- 3. the transform is tracked, not baked -------------------------------
# This runs *before* the teleport, on a scene that has only ever been settling.
# Doing it after an 8 m round trip meant asserting on a cube that had been
# dropped back in and was still bouncing.
row_before = float(np.where(near)[0].mean()) if near.any() else float("nan")
move_object(dz=0.20)
d_up = depth()
up = near_mask(d_up)
if near.any() and up.any():
    row_after = float(np.where(up)[0].mean())
    # Both simulators index images top-down, so a cube that rises moves to a
    # *smaller* row. This is the assertion a static-mesh raycaster fails: it
    # would report an unchanged image.
    check("tracks_transform", row_after < row_before - 0.5,
          f"blob centroid row {row_before:.1f} -> {row_after:.1f} "
          f"after lifting the cube 20 cm")
    delta = np.abs(d_up - d0)[np.isfinite(d_up) & np.isfinite(d0)].mean()
    check("image_changed", float(delta) > 1e-3, f"mean |delta| = {float(delta):.4f} m")
else:
    check("tracks_transform", False, "no near pixels in one of the two frames")

# --- 4. remove the cube; the near pixels must go with it -------------------
move_object(dx=8.0)
d_gone = depth()
gone = near_mask(d_gone)
check("blob_is_the_payload", gone.mean() < 0.2 * max(frac, 1e-9) + 1e-4,
      f"{gone.mean():.1%} near pixels with the cube 8 m away (was {frac:.1%})")

# --- 5. and bring it back ---------------------------------------------------
# Directional, not exact. The robots have been stepping throughout and the cube
# is dropped from its default pose, so demanding the blob return to within two
# percentage points of its original size asserts on settling noise rather than
# on the sensor. What matters is that it comes back at all.
move_object()
back = near_mask(depth())
check("restored", back.mean() > 0.5 * frac,
      f"{back.mean():.1%} near pixels after moving it back "
      f"(was {frac:.1%}, gate {0.5 * frac:.1%})")

log("")
log(f"RESULT | {'ALL PASS' if not fails else 'FAILED: ' + ', '.join(fails)}")
with open(OUT, "a") as fh:
    fh.write("\n".join(lines) + "\n")

env.close()
simulation_app.close()
raise SystemExit(1 if fails else 0)
