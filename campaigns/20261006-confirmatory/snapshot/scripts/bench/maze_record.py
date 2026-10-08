"""Record one maze-recovery episode per seed, with the termination that ended it.

v60 (Isaac Sim 6.0 / Lab 3.0.0b2). Built on the rollout of
scripts/bench/maze_recovery_probe.py: the task's env with the training
observation noise kept ON (the probe's --keep-corruption semantics), the
checkpoint through OnPolicyRunner, num_envs 1, run until the FIRST termination
(or --max-steps), and every termination term that fired is written down.

Two pictures are taken of every step, because on this stack they disagree
(docs/ISAAC_RENDER.md section 10, jobs 21247917 / 21299608):

* ``viewport``  -- gym.wrappers.RecordVideo around the env built with
  render_mode="rgb_array", the way scripts/train.py records --video. On v60
  that is the Kit perspective camera through isaaclab's VideoRecorder. With
  fabric on it draws the robot at its stale USD pose (the spawn), so the
  published finding is that this clip shows the maze and no walking robot.
  It is recorded anyway, as ``<name>.viewport.mp4``: it re-measures the
  finding on this exact task and checkpoint, and if it does show the robot
  the JSON says so (frame count and file) and it can be promoted by hand.
* ``camera_sensor`` -- a CameraCfg sensor that follows the robot's root, the
  recorder train_play.py uses for every published Isaac clip. This is the
  clip used for the GIF: ``<name>.mp4`` (real time, 1/step_dt fps) written
  with imageio; PNG frames also go to --frames-root so the launcher can
  assemble them with ffmpeg if the writer fails.

Searches. ``--search name:want:seed:max_seeds[:imu_delay_steps]`` (repeatable,
colon-separated because Apptainer --env and sbatch --export split on commas)
runs episodes at seed, seed+1, ... until an episode with the wanted outcome
(``success``, ``failure`` or ``any``) is found or max_seeds is spent. Every
episode is kept as ``<label>-s<seed>[-d<delay>].{json,mp4,viewport.mp4}``;
the first matching episode is copied to ``<name>.{json,mp4,viewport.mp4}``.
A later search first looks through episodes already recorded by an earlier
one, so a failure met while looking for the success is reused, not re-rolled.
``failure`` means a termination other than button_reached. Reaching
--max-steps with no termination is ``step_cap`` and satisfies neither.

The optional imu_delay_steps field delays the policy's IMU columns
(base_ang_vel, projected_gravity) by that many control steps, the
maze_recovery_probe SF-01 setting, for a same-policy same-seed controlled
failure (docs/SENSOR_FUSION.md: 2 steps = 40 ms -> 0.02 success).

Kit swallows stderr and its close exits 0, so tracebacks go to stdout and to
``<out-dir>/<label>.error.txt``; the launcher scores the JSON files, never the
exit code. ``--selftest`` exercises the Isaac-free helpers and exits.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]

SUCCESS_TERM = "button_reached"
# The policy's lidar / stereo terms are scaled by these ranges (sensors_rig.py); panels undo it.
LIDAR_RANGE_M = 12.0
STEREO_RANGE_M = 6.0
WANTS = ("success", "failure", "any")

# Camera framings, metres. World-frame offsets; the sensor follows the root
# (a fixed wide shot makes the 0.5 m robot ~50 px), the viewport frames env 0's
# corridor from the -y side and above: spawn x in -0.2..0.2, button at (3, 0),
# walls at y = +-0.45 (maze_layout: CORRIDOR_W 0.90, WALL_H 0.60).
#
# The sensor is a rear-follow shot INSIDE the corridor, not a side view: any
# eye beyond the y = +-0.45 walls has its line of sight to the feet cross the
# 0.60 m wall (for a side eye 2 m out and 1.5 m up the crossing is at z ~0.4 m,
# so the lower ~0.25 m of the robot is hidden, more as it drifts toward that
# wall). Behind the robot at its own y, 2 m back and 1.5 m up, nothing stands
# between the lens and the body; at focal 18 mm (36 deg vertical FOV) the feet
# are ~14 deg below the frame centre and the button plate, seen from the spawn,
# ~14 deg above it, with both walls at ~+-13 deg converging toward the plate.
SENSOR_EYE = (-2.0, 0.0, 1.5)
SENSOR_LOOKAT = (0.5, 0.0, 0.1)
VIEWPORT_EYE = (1.4, -3.5, 2.5)
VIEWPORT_LOOKAT = (1.4, 0.0, 0.2)


# ----------------------------------------------------------------------------
# Isaac-free helpers (covered by --selftest)
# ----------------------------------------------------------------------------
def parse_vec(raw: str | None, default):
    """'x:y:z' (commas accepted) -> tuple of 3 floats."""
    if not raw:
        return tuple(default)
    parts = [float(v) for v in raw.replace(",", ":").split(":")]
    if len(parts) != 3:
        raise ValueError(f"expected three numbers separated by ':', got {raw!r}")
    return tuple(parts)


def parse_search(spec: str) -> dict:
    """'name:want:seed:max_seeds[:imu_delay_steps]' -> dict."""
    parts = spec.split(":")
    if len(parts) not in (4, 5):
        raise ValueError(f"--search wants name:want:seed:max_seeds[:imu_delay_steps], got {spec!r}")
    name, want, seed, max_seeds = parts[:4]
    delay = int(parts[4]) if len(parts) == 5 else 0
    if want not in WANTS:
        raise ValueError(f"want must be one of {WANTS}, got {want!r}")
    if not name or "/" in name:
        raise ValueError(f"bad search name {name!r}")
    seed, max_seeds = int(seed), int(max_seeds)
    if max_seeds < 1 or delay < 0:
        raise ValueError(f"max_seeds >= 1 and imu_delay_steps >= 0 required in {spec!r}")
    return {"name": name, "want": want, "seed": seed, "max_seeds": max_seeds, "imu_delay_steps": delay}


def classify(success: bool, done: bool, fired: list[str], timed_out: bool) -> str:
    """One word for the launcher: success | <term> | time_out | step_cap."""
    if done and success:
        return "success"
    if done:
        others = [t for t in fired if t not in (SUCCESS_TERM, "time_out")]
        if others:
            return "+".join(others)
        return "time_out" if (timed_out or "time_out" in fired) else "unknown_termination"
    return "step_cap"


def is_failure(outcome: str) -> bool:
    return outcome not in ("success", "step_cap")


def matches(want: str, outcome: str) -> bool:
    if want == "success":
        return outcome == "success"
    if want == "failure":
        return is_failure(outcome)
    return outcome != "step_cap"


def sha256(path) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head(repo: Path) -> str | None:
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True,
                              text=True, timeout=20, check=True).stdout.strip()
    except Exception:                                            # noqa: BLE001
        return None


def episode_stem(label: str, seed: int, delay: int) -> str:
    return f"{label}-s{seed}" + (f"-d{delay}" if delay else "")


def selftest() -> int:
    s = parse_search("both-s0-success:success:100:4")
    assert s == {"name": "both-s0-success", "want": "success", "seed": 100, "max_seeds": 4, "imu_delay_steps": 0}, s
    assert parse_search("x:failure:100:2:2")["imu_delay_steps"] == 2
    for bad in ("a:b", "a:nope:1:1", "a:success:1:0", "a/b:success:1:1"):
        try:
            parse_search(bad)
        except ValueError:
            pass
        else:
            raise AssertionError(f"accepted {bad!r}")
    assert classify(True, True, ["button_reached"], False) == "success"
    assert classify(False, True, ["base_orientation"], False) == "base_orientation"
    assert classify(False, True, ["time_out"], True) == "time_out"
    assert classify(False, True, ["dead_end", "time_out"], True) == "dead_end"
    assert classify(False, False, [], False) == "step_cap"
    assert is_failure("base_orientation") and is_failure("time_out") and is_failure("dead_end")
    assert not is_failure("success") and not is_failure("step_cap")
    assert matches("failure", "dead_end") and not matches("failure", "step_cap")
    assert matches("success", "success") and not matches("success", "dead_end")
    assert matches("any", "success") and matches("any", "time_out") and not matches("any", "step_cap")
    assert parse_vec("1:2:3", (0, 0, 0)) == (1.0, 2.0, 3.0) and parse_vec(None, SENSOR_EYE) == SENSOR_EYE
    assert episode_stem("both-s0", 100, 0) == "both-s0-s100" and episode_stem("both-s0", 100, 2) == "both-s0-s100-d2"
    print("maze_record selftest OK")
    return 0


# ----------------------------------------------------------------------------
# Isaac side
# ----------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", default="Velocity-BHL-MazeRecovery-Full-Both-v0")
    p.add_argument("--checkpoint", required=False)
    p.add_argument("--label", default=None, help="episode file prefix (default: derived from the task)")
    p.add_argument("--search", action="append", default=[],
                   help="name:want:seed:max_seeds[:imu_delay_steps]; repeatable (default both-s0-success:success:100:1)")
    p.add_argument("--out-dir", default=str(REPO / "results/repo-gpu-20260923/maze-record"))
    p.add_argument("--frames-root", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "maze-record-frames"),
                   help="PNG frames of the camera-sensor clip go here (node-local by default)")
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=360)
    p.add_argument("--sensor-eye", default=None, help="x:y:z offset from the robot root (default %s)" % (SENSOR_EYE,))
    p.add_argument("--sensor-lookat", default=None, help="x:y:z offset from the robot root (default %s)" % (SENSOR_LOOKAT,))
    p.add_argument("--viewport-eye", default=None, help="x:y:z offset from env 0's origin (default %s)" % (VIEWPORT_EYE,))
    p.add_argument("--viewport-lookat", default=None, help="x:y:z offset from env 0's origin (default %s)" % (VIEWPORT_LOOKAT,))
    p.add_argument("--no-viewport", action="store_true", help="skip the RecordVideo viewport clip")
    p.add_argument("--viewport-episodes", type=int, default=1,
                   help="record the viewport clip for the first N episodes only (0 = all); it is a diagnostic of the "
                        "stale-pose finding and costs a Kit app update per frame")
    p.add_argument("--no-overlays", action="store_true", help="do not spawn the coloured maze clip overlays")
    p.add_argument("--drop-corruption", action="store_true",
                   help="force enable_corruption=False (the published noise-off scoring); default keeps the training noise")
    p.add_argument("--selftest", action="store_true", help="run the Isaac-free helper checks and exit")
    pan = p.add_argument_group("three-panel recording (top view + the policy's stereo and lidar inputs)")
    pan.add_argument("--panels", action="store_true",
                     help="add a fixed overhead camera and dump per-step lidar/stereo arrays for compose_panels.py")
    pan.add_argument("--top-eye", default="1.5:-0.25:3.9", help="x:y:z offset of the overhead eye from env 0's origin")
    pan.add_argument("--top-lookat", default="1.5:0.0:0.0", help="x:y:z offset of its target from env 0's origin")
    pan.add_argument("--top-width", type=int, default=1280)
    pan.add_argument("--top-height", type=int, default=720)
    pan.add_argument("--render-quality", choices=("stock", "clean", "taa", "rtl", "pathtrace"), default="stock",
                     help="stock = the kit's RealTimePathTracing at 1 spp, which relies on the NGX denoiser that is "
                          "missing here (grain 36); clean/taa = stochastic terms off (+ TAA): measured 34, not enough; "
                          "rtl = the classic RaytracedLighting mode (no per-pixel path sampling); "
                          "pathtrace = offline PathTracing at 16 spp through the OptiX denoiser (no NGX needed)")
    pan.add_argument("--light-scale", type=float, default=1.0,
                     help="multiply every scene light's intensity (path tracing over-exposes the stock lights: the "
                          "tan floor and blue walls came out white in 21408634)")
    pan.add_argument("--floor-checker", action="store_true",
                     help="visual-only 0.5 m checker tiles on the corridor floor (the fused ground mesh has no albedo); "
                          "collision off and not in any ray-caster's mesh list, so the MDP is untouched")
    pan.add_argument("--target-luma", type=float, default=122.0,
                     help="auto-exposure trim after --light-scale: scale /World/light and /World/skyLight until the overhead "
                          "frame's mean luminance is within 10 of this (0-255; a tan floor at correct exposure is ~120-130); "
                          "0 disables")
    pan.add_argument("--warmup-frames", type=int, default=6,
                     help="render-only frames after each reset before recording; the last two give the grain metrics")
    return p


class ImuDelay:
    """Policy-side FIFO delay of the IMU columns (maze_recovery_probe SF-01)."""

    def __init__(self, slices: dict, steps: int):
        self.slices, self.steps, self.queue = slices, int(steps), []

    def __call__(self, obs):
        import torch
        if self.steps <= 0 or not self.slices:
            return obs
        pol = obs["policy"]
        cur = torch.cat([pol[:, s] for s in self.slices.values()], dim=1).clone()
        self.queue.append(cur)
        old = self.queue.pop(0) if len(self.queue) > self.steps else self.queue[0]
        pol = pol.clone()
        c = 0
        for s in self.slices.values():
            w = s.stop - s.start
            pol[:, s] = old[:, c:c + w]
            c += w
        obs = obs.clone() if hasattr(obs, "clone") else dict(obs)
        obs["policy"] = pol
        return obs


def imu_slices(u) -> dict:
    import torch
    om = u.observation_manager
    names, dims = om.active_terms["policy"], om.group_obs_term_dim["policy"]
    out, col = {}, 0
    for n, d in zip(names, dims):
        w = int(torch.tensor(d).prod()) if not isinstance(d, int) else d
        if n in ("base_ang_vel", "projected_gravity"):
            out[n] = slice(col, col + w)
        col += w
    return out


class FrameSink:
    """Streams camera-sensor frames to an mp4 (imageio) and to PNGs."""

    def __init__(self, mp4: Path, png_dir: Path, fps: float):
        self.mp4, self.png_dir, self.fps = mp4, png_dir, fps
        self.n, self.writer, self.writer_error = 0, None, None
        if png_dir.exists():
            shutil.rmtree(png_dir)
        png_dir.mkdir(parents=True, exist_ok=True)
        mp4.parent.mkdir(parents=True, exist_ok=True)
        self.codecs = ["libx264", "mpeg4"]      # imageio's bundled ffmpeg has libx264; the cluster's own does not
        self.codec = None
        self._open()

    def _open(self):
        while self.codecs:
            codec = self.codecs.pop(0)
            try:
                import imageio.v2 as imageio
                self.writer = imageio.get_writer(str(self.mp4), fps=round(self.fps), codec=codec, quality=8,
                                                 pixelformat="yuv420p", macro_block_size=1)
                self.codec = codec
                return
            except Exception as exc:                             # noqa: BLE001
                self.writer_error = repr(exc)
        self.writer = None
        print(f"[sensor] imageio writer unavailable ({self.writer_error}); PNG frames only", flush=True)

    def add(self, rgb):
        from PIL import Image
        img = Image.fromarray(rgb)
        img.save(self.png_dir / f"frame_{self.n:04d}.png")
        if self.writer is not None:
            try:
                self.writer.append_data(rgb)
            except Exception as exc:                             # noqa: BLE001
                self.writer_error = repr(exc)
                self.writer = None
                if self.n == 0 and self.codecs:                  # encoder rejected on the first frame: try the next codec
                    self._open()
                    if self.writer is not None:
                        try:
                            self.writer.append_data(rgb)
                        except Exception as exc2:                # noqa: BLE001
                            self.writer_error = repr(exc2)
                            self.writer = None
        self.n += 1

    def close(self) -> dict:
        if self.writer is not None:
            try:
                self.writer.close()
            except Exception as exc:                             # noqa: BLE001
                self.writer_error = repr(exc)
        ok = self.mp4.is_file() and self.mp4.stat().st_size > 0
        return {"frames": self.n, "mp4": str(self.mp4) if ok else None, "codec": self.codec if ok else None,
                "png_dir": str(self.png_dir), "writer_error": self.writer_error}


def render_profile(quality: str) -> tuple[dict, str | None]:
    """carb settings (dotted keys, no underscores) and the antialiasing mode for a --render-quality."""
    stochastic_off = {
        "rtx.directLighting.sampledLighting.enabled": False,
        "rtx.directLighting.sampledLighting.autoEnable": False,
        "rtx.ambientOcclusion.enabled": False,
        "rtx.indirectDiffuse.enabled": False,
        "rtx.reflections.enabled": False,
        "rtx.raytracing.subpixel.mode": 0,
    }
    if quality == "clean":
        return dict(stochastic_off), None
    if quality == "taa":
        return dict(stochastic_off), "TAA"
    if quality == "rtl":
        return {"rtx.rendermode": "RaytracedLighting", **stochastic_off}, "TAA"
    if quality == "pathtrace":
        return {"rtx.rendermode": "PathTracing", "rtx.pathtracing.spp": 16, "rtx.pathtracing.totalSpp": 16,
                "rtx.pathtracing.clampSpp": 16, "rtx.pathtracing.optixDenoiser.enabled": True,
                "rtx.pathtracing.optixDenoiser.blendFactor": 0.0}, None
    return {}, None


def apply_carb_now(settings: dict, aa: str | None) -> dict:
    """Set the profile straight into carb, right after the app is up and before
    the stage/render products exist -- RenderCfg.carb_settings is applied too,
    but job 21408633 showed the kit's rendering preset winning over it."""
    import carb
    st = carb.settings.get_settings()
    applied = {}
    for k, v in settings.items():
        path = "/" + k.replace(".", "/")
        st.set(path, v)
        applied[path] = st.get(path)
    if aa is not None:
        st.set("/rtx/post/aa/op", {"Off": 0, "TAA": 1, "FXAA": 2, "DLSS": 3, "DLAA": 4}[aa])
        applied["/rtx/post/aa/op"] = st.get("/rtx/post/aa/op")
    return applied


def main() -> int:
    args, _unknown = build_parser().parse_known_args()
    if args.selftest:
        return selftest()

    from isaaclab.app import AppLauncher
    parser = build_parser()
    AppLauncher.add_app_launcher_args(parser)
    args = parser.parse_args()
    if not args.checkpoint:
        parser.error("--checkpoint is required")
    searches = [parse_search(s) for s in (args.search or ["both-s0-success:success:100:1"])]
    args.enable_cameras = True          # the CameraCfg sensor needs the RTX renderer even without the viewport clip
    app = AppLauncher(args)
    if args.panels and args.render_quality != "stock":
        try:
            prof, aa = render_profile(args.render_quality)
            print(f"[panels] carb settings applied at app start: {json.dumps(apply_carb_now(prof, aa))}", flush=True)
        except Exception as exc:                                    # noqa: BLE001
            print(f"[panels] direct carb application failed ({exc!r}); RenderCfg path only", flush=True)
    try:
        rc = run(args, searches)
    except BaseException:                                       # noqa: BLE001
        import traceback
        tb = traceback.format_exc()
        print("RECORD-ERROR\n" + tb, flush=True)                  # stdout: Kit swallows stderr
        try:
            Path(args.out_dir).mkdir(parents=True, exist_ok=True)
            (Path(args.out_dir) / f"{args.label or 'maze-record'}.error.txt").write_text(tb)
        except Exception:                                        # noqa: BLE001
            pass
        raise
    finally:
        app.app.close()
    return rc


def run(args, searches) -> int:
    import gymnasium as gym
    import numpy as np
    import torch
    import isaaclab.sim as sim_utils
    from isaaclab.sensors import CameraCfg
    import bhl_robust.tasks  # noqa: F401
    from bhl_robust import maze_recovery as recovery

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    label = args.label or args.task.replace("Velocity-BHL-MazeRecovery-", "").replace("-v0", "").lower()
    sensor_eye = parse_vec(args.sensor_eye, SENSOR_EYE)
    sensor_lookat = parse_vec(args.sensor_lookat, SENSOR_LOOKAT)
    viewport_eye = parse_vec(args.viewport_eye, VIEWPORT_EYE)
    viewport_lookat = parse_vec(args.viewport_lookat, VIEWPORT_LOOKAT)
    ckpt_sha = sha256(args.checkpoint)
    if ckpt_sha is None:
        raise FileNotFoundError(f"checkpoint missing: {args.checkpoint}")

    spec = gym.spec(args.task)
    cfg = spec.kwargs["env_cfg_entry_point"]()
    cfg.scene.num_envs = 1
    cfg.seed = searches[0]["seed"]
    if args.drop_corruption:
        cfg.observations.policy.enable_corruption = False
    observation_corruption = bool(cfg.observations.policy.enable_corruption)
    # Viewer: frame env 0 (honoured by the interactive viewport controller; the
    # v60 VideoRecorder copies eye/lookat as raw world coordinates, fixed below
    # once env 0's origin is known).
    cfg.viewer.origin_type = "env"
    cfg.viewer.env_index = 0
    cfg.viewer.eye = tuple(viewport_eye)
    cfg.viewer.lookat = tuple(viewport_lookat)
    cfg.viewer.resolution = (args.width, args.height)
    vr = getattr(cfg, "video_recorder", None)
    if vr is not None:
        # RecordVideo buffers every frame in RAM for moviepy: 600 x 1280x720 is ~1.6 GB.
        vr.window_width, vr.window_height = args.width, args.height
    # The camera-sensor recorder train_play.py uses for every published Isaac
    # clip (docs/ISAAC_RENDER.md section 10): mounted per env, aimed at the
    # robot's root every step, world-frame offsets so the view does not spin.
    cfg.scene.clip_cam = CameraCfg(
        prim_path="{ENV_REGEX_NS}/clip_cam", height=args.height, width=args.width, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 2.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
    )

    top_eye = parse_vec(args.top_eye, (1.5, -0.25, 3.9))
    top_lookat = parse_vec(args.top_lookat, (1.5, 0.0, 0.0))
    render_settings = {}
    if args.panels:
        cfg.scene.top_cam = CameraCfg(
            prim_path="{ENV_REGEX_NS}/top_cam", height=args.top_height, width=args.top_width, data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(focal_length=18.0),
            offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 4.0), rot=(1.0, 0.0, 0.0, 0.0), convention="world"),
        )
        if args.render_quality != "stock":
            # The kit's default mode on this stack is RealTimePathTracing at 1 spp, which
            # hands its clean-up to the NGX denoiser -- and NGX fails to initialise on
            # these nodes (job 21408633's readback). Every Isaac clip's speckle is that raw
            # 1-spp image. `render_profile` holds the alternatives; they are applied both
            # here (RenderCfg) and straight into carb at app start (`apply_carb_now`).
            render_settings, aa = render_profile(args.render_quality)
            try:
                cfg.sim.render.carb_settings = dict(render_settings)
                if aa is not None:
                    cfg.sim.render.antialiasing_mode = aa
                    render_settings["antialiasing_mode"] = aa
            except Exception as exc:                                 # noqa: BLE001
                print(f"[panels] RenderCfg override unavailable ({exc!r})", flush=True)
                render_settings = {**render_settings, "rendercfg_error": repr(exc)}
    lights_scaled = {}
    if args.panels and args.light_scale != 1.0:
        for name_, val_ in list(vars(cfg.scene).items()):
            sp = getattr(val_, "spawn", None)
            if sp is not None and hasattr(sp, "intensity") and "Light" in type(sp).__name__:
                before = float(sp.intensity)
                sp.intensity = before * args.light_scale
                lights_scaled[name_] = [before, float(sp.intensity)]
        print(f"[panels] scene lights scaled x{args.light_scale:g}: {json.dumps(lights_scaled)}", flush=True)
    render_mode = None if args.no_viewport else "rgb_array"
    env = gym.make(args.task, cfg=cfg, render_mode=render_mode)
    u = env.unwrapped
    overlays = 0
    checker_tiles = 0
    if args.panels and args.floor_checker:
        try:
            from bhl_robust.terrains.maze_layout import FLOOR_RGB
            origins_ = recovery.tensor(u.scene.env_origins)
            o = origins_[0].detach().cpu().tolist()
            dark = tuple(c * 0.82 for c in FLOOR_RGB)
            tile, half_x, half_y = 0.5, 3.5, 2.5
            nx, ny = int(2 * half_x / tile), int(2 * half_y / tile)
            for ix in range(nx):
                for iy in range(ny):
                    cx_ = -half_x + (ix + 0.5) * tile
                    cy_ = -half_y + (iy + 0.5) * tile
                    tcfg = sim_utils.CuboidCfg(
                        size=(tile, tile, 0.004),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=FLOOR_RGB if (ix + iy) % 2 == 0 else dark),
                        collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
                    )
                    tcfg.func(f"/World/maze_panels_floor/t{ix}_{iy}", tcfg, translation=(o[0] + cx_, o[1] + cy_, o[2] + 0.002))
                    checker_tiles += 1
            print(f"[panels] spawned {checker_tiles} visual-only floor tiles", flush=True)
        except Exception as exc:                                     # noqa: BLE001
            print(f"[panels] floor checker skipped ({exc!r})", flush=True)
    if not args.no_overlays:
        try:
            from bhl_robust.terrains.maze_viz import spawn_maze_clip_color
            overlays = spawn_maze_clip_color(u)
            print(f"[clip] spawned {overlays} maze clip-colour overlays", flush=True)
        except Exception as exc:                                 # noqa: BLE001
            print(f"[clip] overlays skipped ({exc!r})", flush=True)

    # Viewport clip: gym.wrappers.RecordVideo, as scripts/train.py wraps for
    # --video. Driven by hand (start/stop per episode) instead of a step
    # trigger so several episodes can be recorded in one Kit boot.
    rv = None
    viewport_error = None
    viewport_dir = out_dir / f".{label}-viewport-tmp"

    class TolerantRecordVideo(gym.wrappers.RecordVideo):
        """A viewport render error must not take the camera-sensor clip down with it."""
        capture_error = None

        def _capture_frame(self):
            try:
                super()._capture_frame()
            except Exception as exc:                             # noqa: BLE001
                self.capture_error = repr(exc)
                self.recording, self.recorded_frames = False, []
                print(f"[viewport] capture disabled after error: {exc!r}", flush=True)

    if render_mode == "rgb_array":
        try:
            rv = TolerantRecordVideo(env, video_folder=str(viewport_dir), step_trigger=lambda step: False,
                                     video_length=args.max_steps + 5, name_prefix=label, disable_logger=True)
            env = rv
        except Exception as exc:                                 # noqa: BLE001
            viewport_error = repr(exc)
            print(f"[viewport] RecordVideo unavailable ({exc!r}); camera-sensor clip only", flush=True)
    env.reset()

    # Policy: the task's rsl_rl cfg through OnPolicyRunner (maze_recovery_probe).
    from importlib.metadata import version
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from rsl_rl.runners import OnPolicyRunner
    agent = spec.kwargs["rsl_rl_cfg_entry_point"]()
    if os.environ.get("BHL_POLICY", "").strip() == "recurrent":
        from isaaclab_rl.rsl_rl import RslRlPpoActorCriticRecurrentCfg
        old_pol = agent.policy
        agent.policy = RslRlPpoActorCriticRecurrentCfg(
            init_noise_std=old_pol.init_noise_std, actor_hidden_dims=old_pol.actor_hidden_dims,
            critic_hidden_dims=old_pol.critic_hidden_dims, activation=old_pol.activation,
            rnn_type="lstm", rnn_hidden_dim=256, rnn_num_layers=1)
        print("[overlay] policy: ActorCriticRecurrent (lstm, 256)", flush=True)
    try:
        from isaaclab_rl.rsl_rl import handle_deprecated_rsl_rl_cfg
        agent = handle_deprecated_rsl_rl_cfg(agent, version("rsl-rl-lib"))
    except ImportError:
        pass
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent.to_dict(), log_dir=None, device=u.device)
    runner.load(args.checkpoint)
    policy = runner.get_inference_policy(device=u.device)
    slices = imu_slices(u)

    clip_cam = u.scene["clip_cam"]
    robot = u.scene["robot"]
    dev = u.device

    def as_torch(x):
        # v60 wraps some buffers (recovery.tensor's `.torch`); train_play reads the rest with torch.as_tensor(x[:]).
        return x.torch if hasattr(x, "torch") else torch.as_tensor(x[:])
    eye_off = torch.tensor(sensor_eye, device=dev, dtype=torch.float32)
    look_off = torch.tensor(sensor_lookat, device=dev, dtype=torch.float32)
    origin0 = recovery.tensor(u.scene.env_origins)[0].detach().float().cpu().tolist()
    vp_eye = [o + d for o, d in zip(origin0, viewport_eye)]
    vp_lookat = [o + d for o, d in zip(origin0, viewport_lookat)]
    viewport_camera = "default"
    if rv is not None:
        # Kit's perspective camera is aimed lazily, on the first frame, from the
        # capture cfg's world-frame eye/lookat; point it at env 0's corridor.
        try:
            cap = u.video_recorder._capture
            cap.cfg.eye, cap.cfg.lookat = tuple(vp_eye), tuple(vp_lookat)
            viewport_camera = "env0_corridor"
        except Exception as exc:                                 # noqa: BLE001
            viewport_camera = f"default ({exc!r})"
        try:
            u.sim.set_camera_view(tuple(vp_eye), tuple(vp_lookat))
        except Exception as exc:                                 # noqa: BLE001
            print(f"[viewport] sim.set_camera_view skipped ({exc!r})", flush=True)
    step_dt, physics_dt = float(u.step_dt), float(u.physics_dt)
    fps = 1.0 / step_dt
    base = {"task": args.task, "checkpoint": args.checkpoint, "checkpoint_sha256": ckpt_sha, "num_envs": 1,
            "max_steps": args.max_steps, "sim_dt": physics_dt, "step_dt": step_dt,
            "decimation": int(getattr(u.cfg, "decimation", round(step_dt / physics_dt))),
            "observation_corruption": observation_corruption, "stack": os.environ.get("BHL_STACK"),
            "hostname": socket.gethostname(), "git_commit": git_head(REPO), "clip_overlays": overlays,
            "capture_path_used_for_gif": "camera_sensor",
            "capture_note": ("viewport = gym RecordVideo on the Kit perspective camera (draws the stale USD pose "
                             "under fabric on this stack, docs/ISAAC_RENDER.md s10); camera_sensor = CameraCfg "
                             "following the robot root, the recorder behind every published Isaac clip"),
            "sensor_eye_offset_m": list(sensor_eye), "sensor_lookat_offset_m": list(sensor_lookat),
            "viewport_eye_world_m": vp_eye, "viewport_lookat_world_m": vp_lookat, "viewport_camera": viewport_camera,
            "viewport_error": viewport_error, "env0_origin_m": origin0, "frame_size": [args.width, args.height],
            "video_fps": fps, "playback_speed": 1.0,
            "panels_enabled": bool(args.panels), "render_quality": args.render_quality if args.panels else None,
            "light_scale": args.light_scale if args.panels else None, "lights_scaled": lights_scaled,
            "floor_checker_tiles": checker_tiles,
            "render_settings": render_settings, "top_eye_offset_m": list(top_eye), "top_lookat_offset_m": list(top_lookat)}
    print(json.dumps({k: v for k, v in base.items() if k != "capture_note"}), flush=True)
    top_cam = u.scene["top_cam"] if args.panels else None
    if args.panels:
        try:
            import carb
            st_ = carb.settings.get_settings()
            base["rtx_effective"] = {k: st_.get(k) for k in (
                "/rtx/rendermode", "/rtx/post/aa/op", "/rtx/post/dlss/execMode", "/rtx/directLighting/sampledLighting/enabled",
                "/rtx/directLighting/sampledLighting/samplesPerPixel", "/rtx/ambientOcclusion/enabled", "/rtx/indirectDiffuse/enabled",
                "/rtx/pathtracing/spp", "/rtx/pathtracing/totalSpp", "/rtx/pathtracing/optixDenoiser/enabled",
                "/rtx-transient/dldenoiser/enabled")}
            print(f"[panels] effective rtx settings: {json.dumps(base['rtx_effective'])}", flush=True)
        except Exception as exc:                                     # noqa: BLE001
            print(f"[panels] could not read carb settings ({exc!r})", flush=True)
    from bhl_robust.quat_order import unpack_wxyz

    def term_slices() -> dict:
        om = u.observation_manager
        names, dims = om.active_terms["policy"], om.group_obs_term_dim["policy"]
        out, col = {}, 0
        for n_, d_ in zip(names, dims):
            w_ = int(torch.tensor(d_).prod()) if not isinstance(d_, int) else d_
            out[n_] = (slice(col, col + w_), tuple(d_) if not isinstance(d_, int) else (d_,))
            col += w_
        return out

    pol_slices = term_slices() if args.panels else {}

    def grain(a: np.ndarray) -> float:
        a = a.astype(np.float32)
        pd = np.pad(a, ((1, 1), (1, 1), (0, 0)), mode="edge")
        b = sum(pd[i:i + a.shape[0], j:j + a.shape[1]] for i in range(3) for j in range(3)) / 9.0
        return float(np.mean(np.abs(a - b)))

    def read_top() -> np.ndarray:
        rgb = as_torch(top_cam.data.output["rgb"])[0, ..., :3]
        return np.ascontiguousarray(rgb.detach().cpu().numpy().astype(np.uint8))

    episodes: list[dict] = []

    def sim_time():
        for obj, attr in ((u.sim, "current_time"), (getattr(u.sim, "physics_manager", None), "current_time")):
            v = getattr(obj, attr, None) if obj is not None else None
            if v is not None:
                try:
                    return float(v)
                except Exception:                                # noqa: BLE001
                    pass
        return None

    def record_episode(seed: int, delay_steps: int) -> dict:
        stem = episode_stem(label, seed, delay_steps)
        print(f"=== episode {stem} | seed {seed} | imu delay {delay_steps} steps | {time.strftime('%H:%M:%S')} ===", flush=True)
        sink = FrameSink(out_dir / f"{stem}.mp4", Path(args.frames_root) / stem, fps)
        vp_name = f"{stem}.viewport"
        use_viewport = rv is not None and (args.viewport_episodes <= 0 or len(episodes) < args.viewport_episodes)
        if use_viewport:
            try:
                rv.start_recording(vp_name)
            except Exception as exc:                             # noqa: BLE001
                print(f"[viewport] start_recording failed ({exc!r})", flush=True)
        t_sim0 = sim_time()
        t_wall0 = time.time()
        # The rollout steps under torch.inference_mode(), which turns the command
        # term's waypoint index into an inference tensor; the reset must run in
        # the same mode or its in-place update is rejected (job 21402843).
        with torch.inference_mode():
            u.reset(seed=seed)
            obs = env.get_observations()
            obs = obs[0] if isinstance(obs, tuple) else obs
        mod = getattr(policy, "__self__", None)
        if mod is not None and hasattr(mod, "reset"):
            try:
                mod.reset()                                      # recurrent policies: clear the hidden state
            except Exception:                                    # noqa: BLE001
                pass
        delay = ImuDelay(slices, delay_steps)
        pan = None
        if args.panels:
            top_png = Path(args.frames_root) / f"{stem}-top"
            if top_png.exists():
                shutil.rmtree(top_png)
            top_png.mkdir(parents=True, exist_ok=True)
            pan = {"top_png_dir": str(top_png), "stereo_raw": [], "stereo_policy": [], "lidar_sector": [],
                   "lidar_hits": [], "root_xy": [], "root_yaw": [], "top_frames": 0, "warmup": [], "errors": []}
            try:
                o0 = recovery.tensor(u.scene.env_origins)[0].detach().float()
                eye = (o0 + o0.new_tensor(top_eye)).unsqueeze(0)
                tgt = (o0 + o0.new_tensor(top_lookat)).unsqueeze(0)
                top_cam.set_world_poses_from_view(eye, tgt)
                for _ in range(max(2, args.warmup_frames)):
                    u.sim.render()
                    top_cam.update(dt=0.0, force_recompute=True)
                    pan["warmup"].append(read_top())
                    pan["warmup"] = pan["warmup"][-2:]
                # Auto-exposure trim: the configured light scale is a guess, and a miss costs a
                # queue cycle. Scale the two upstream light prims by target/measured (clamped)
                # until the mean luminance sits near the target, then re-measure the grain.
                pan["exposure"] = {"target_luma": args.target_luma, "steps": []}
                if args.target_luma > 0:
                    try:
                        import omni.usd
                        stage_ = omni.usd.get_context().get_stage()
                        prims_ = [stage_.GetPrimAtPath(pp) for pp in ("/World/light", "/World/skyLight")]
                        prims_ = [pr for pr in prims_ if pr and pr.IsValid()]

                        def _iattr(pr):
                            for nm in ("inputs:intensity", "intensity"):
                                at = pr.GetAttribute(nm)
                                if at and at.IsValid():
                                    return at
                            return None
                        attrs_ = [a_ for a_ in (_iattr(pr) for pr in prims_) if a_ is not None]
                        if not attrs_:
                            raise RuntimeError("no light intensity attributes found")
                        for _it in range(4):
                            fr = pan["warmup"][-1].astype(np.float32)
                            luma = float((0.299 * fr[..., 0] + 0.587 * fr[..., 1] + 0.114 * fr[..., 2]).mean())
                            factor = float(np.clip(args.target_luma / max(luma, 1.0), 0.5, 2.0))
                            pan["exposure"]["steps"].append({"mean_luma": round(luma, 1), "factor": round(factor, 3),
                                                             "intensities": [float(a_.Get()) for a_ in attrs_]})
                            if abs(luma - args.target_luma) <= 10.0:
                                break
                            for a_ in attrs_:
                                a_.Set(float(a_.Get()) * factor)
                            for _ in range(2):
                                u.sim.render()
                                top_cam.update(dt=0.0, force_recompute=True)
                                pan["warmup"].append(read_top())
                                pan["warmup"] = pan["warmup"][-2:]
                        pan["exposure"]["final_intensities"] = [float(a_.Get()) for a_ in attrs_]
                        print(f"[panels] exposure trim: {json.dumps(pan['exposure'])}", flush=True)
                    except Exception as exc:                         # noqa: BLE001
                        pan["exposure"]["error"] = repr(exc)
                        print(f"[panels] exposure trim skipped ({exc!r}); configured light scale stands", flush=True)
                a, b = pan["warmup"]
                pan["grain_spatial"] = grain(b)
                pan["grain_temporal_mad"] = float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32))))
                print(f"[panels] {stem}: render quality {args.render_quality}: spatial grain {pan['grain_spatial']:.2f} "
                      f"(stock clip 36.1, MuJoCo 0.4), temporal MAD {pan['grain_temporal_mad']:.2f} (stock 47.3)", flush=True)
            except Exception as exc:                                 # noqa: BLE001
                pan["errors"].append(f"warmup: {exc!r}")
                print(f"[panels] warm-up failed ({exc!r})", flush=True)
        start = recovery.local_xy(u)[0].clone()
        goal = start.new_tensor((3.0, 0.0))
        min_dist = float((start - goal).norm())
        max_disp = 0.0
        steps = done = 0
        success = False
        fired: list[str] = []
        timed_out = False
        final_dist = min_dist
        for _ in range(args.max_steps):
            with torch.inference_mode():
                # Aim the sensor at the root BEFORE stepping (train_play), from the
                # pre-step pose; world-frame offsets keep the view from spinning.
                root = as_torch(robot.data.root_pos_w)[:, :3].to(dev).float()
                try:
                    clip_cam.set_world_poses_from_view(root + eye_off, root + look_off)
                except Exception as exc:                         # noqa: BLE001
                    if steps == 0:
                        print(f"[sensor] set_world_poses_from_view failed ({exc!r})", flush=True)
                if pan is not None:
                    try:
                        pol = obs["policy"] if isinstance(obs, dict) or hasattr(obs, "keys") else obs
                        pol = pol.torch if hasattr(pol, "torch") else pol
                        sl, sr = pol_slices["stereo_l"], pol_slices["stereo_r"]
                        pooled = torch.stack([pol[0, sl[0]].reshape(-1), pol[0, sr[0]].reshape(-1)]).float()
                        side_ = int(round(pooled.shape[1] ** 0.5))          # the pooled eye is square (64/pool)
                        pooled = pooled.reshape(2, side_, side_) if side_ * side_ == pooled.shape[1] else pooled.unsqueeze(1)
                        pan["stereo_policy"].append((pooled * STEREO_RANGE_M).cpu().numpy().astype(np.float32))
                        lsl = pol_slices["lidar"]
                        pan["lidar_sector"].append((pol[0, lsl[0]].float() * LIDAR_RANGE_M).cpu().numpy().astype(np.float32))
                        raw = []
                        for eye_name in ("stereo_l", "stereo_r"):
                            dimg = as_torch(u.scene[eye_name].data.output["distance_to_image_plane"])[0]
                            raw.append(dimg.reshape(dimg.shape[0], dimg.shape[1]).float().cpu().numpy())
                        pan["stereo_raw"].append(np.stack(raw).astype(np.float32))
                        lid = u.scene["lidar"]
                        hits = as_torch(lid.data.ray_hits_w)[0, :, :2].float()
                        lpos = as_torch(lid.data.pos_w)[0, :2].float()
                        qw_, qx_, qy_, qz_ = unpack_wxyz(as_torch(robot.data.root_quat_w)[0:1].float())
                        yaw = float(torch.atan2(2 * (qw_[0] * qz_[0] + qx_[0] * qy_[0]),
                                                1 - 2 * (qy_[0] ** 2 + qz_[0] ** 2)))
                        rel = hits - lpos
                        rel = torch.nan_to_num(rel, nan=LIDAR_RANGE_M, posinf=LIDAR_RANGE_M, neginf=-LIDAR_RANGE_M)
                        c, s_ = float(np.cos(-yaw)), float(np.sin(-yaw))
                        body = torch.stack([c * rel[:, 0] - s_ * rel[:, 1], s_ * rel[:, 0] + c * rel[:, 1]], dim=1)
                        pan["lidar_hits"].append(body.cpu().numpy().astype(np.float32))
                        rxy = as_torch(robot.data.root_pos_w)[0, :2].float().cpu().numpy()
                        pan["root_xy"].append(rxy.astype(np.float32))
                        pan["root_yaw"].append(np.float32(yaw))
                    except Exception as exc:                         # noqa: BLE001
                        if not pan["errors"]:
                            print(f"[panels] sensor read failed ({exc!r})", flush=True)
                        pan["errors"].append(f"step {steps}: {exc!r}")
                action = policy(delay(obs))
                obs, _, dones, _ = env.step(action)
            steps += 1
            if pan is not None:
                try:
                    from PIL import Image
                    Image.fromarray(read_top()).save(Path(pan["top_png_dir"]) / f"frame_{pan['top_frames']:04d}.png")
                    pan["top_frames"] += 1
                except Exception as exc:                             # noqa: BLE001
                    if pan["top_frames"] == 0 and steps == 1:
                        print(f"[panels] top frame read failed ({exc!r})", flush=True)
            try:
                rgb = as_torch(clip_cam.data.output["rgb"])[0, ..., :3]
                sink.add(np.ascontiguousarray(rgb.detach().cpu().numpy().astype(np.uint8)))
            except Exception as exc:                             # noqa: BLE001
                if sink.n == 0 and steps == 1:
                    print(f"[sensor] frame read failed ({exc!r})", flush=True)
            # Pre-reset terminal state: the recovery cache is from the terminal
            # transition, unlike root_pos_w which is already reset (probe).
            state = getattr(u, "_maze_recovery_state", None)
            d = bool(recovery.tensor(dones)[0])
            if state is not None:
                dist = float(state["dist"][0])
                won = bool(state["success"][0])
            else:
                dist = float((recovery.local_xy(u)[0] - goal).norm())
                won = bool(recovery.tensor(u.termination_manager.get_term(SUCCESS_TERM))[0])
            min_dist = min(min_dist, dist)
            final_dist = dist
            if not d:
                max_disp = max(max_disp, float((recovery.local_xy(u)[0] - start).norm()))
            if d:
                done = 1
                tm = u.termination_manager
                for name in tm.active_terms:
                    try:
                        if bool(recovery.tensor(tm.get_term(name))[0]):
                            fired.append(name)
                    except Exception:                            # noqa: BLE001
                        pass
                timed_out = bool(recovery.tensor(tm.time_outs)[0])
                if timed_out and "time_out" not in fired:
                    fired.append("time_out")
                success = won or SUCCESS_TERM in fired
                break
        sensor = sink.close()
        panels_rec = None
        if pan is not None:
            npz = out_dir / f"{stem}.panels.npz"
            try:
                m = min(len(pan["stereo_raw"]), len(pan["stereo_policy"]), len(pan["lidar_sector"]), len(pan["lidar_hits"]))
                np.savez_compressed(npz, stereo_raw_m=np.stack(pan["stereo_raw"][:m]) if m else np.zeros((0, 2, 1, 1), np.float32),
                                    stereo_policy_m=np.stack(pan["stereo_policy"][:m]) if m else np.zeros((0, 2, 1, 1), np.float32),
                                    lidar_sector_policy_m=np.stack(pan["lidar_sector"][:m]) if m else np.zeros((0, 36), np.float32),
                                    lidar_hits_body_xy=np.stack(pan["lidar_hits"][:m]) if m else np.zeros((0, 1, 2), np.float32),
                                    root_xy=np.stack(pan["root_xy"][:m]) if m else np.zeros((0, 2), np.float32),
                                    root_yaw=np.asarray(pan["root_yaw"][:m], np.float32), step_dt=step_dt)
                panels_rec = {"npz": str(npz), "top_png_dir": pan["top_png_dir"], "top_frames": pan["top_frames"],
                              "sensor_rows": m, "grain_spatial": pan.get("grain_spatial"), "exposure": pan.get("exposure"),
                              "grain_temporal_mad": pan.get("grain_temporal_mad"),
                              "grain_baseline": {"stock_clip_spatial": 36.1, "stock_clip_temporal_mad": 47.3, "mujoco_spatial": 0.4},
                              "needs_denoise": (pan.get("grain_spatial") is None) or (pan["grain_spatial"] > 12.0),
                              "errors": pan["errors"][:5], "n_errors": len(pan["errors"])}
            except Exception as exc:                                 # noqa: BLE001
                panels_rec = {"error": repr(exc), "top_png_dir": pan["top_png_dir"], "top_frames": pan["top_frames"]}
            print(f"[panels] {stem}: {json.dumps({k: v for k, v in panels_rec.items() if k != 'errors'})}", flush=True)
        t_sim1 = sim_time()
        wall_s = time.time() - t_wall0
        viewport = {"recorded": use_viewport, "frames": 0, "mp4": None,
                    "error": getattr(rv, "capture_error", None) if rv is not None else viewport_error}
        if use_viewport:
            viewport["frames"] = len(rv.recorded_frames)
            try:
                if rv.recording:
                    rv.stop_recording()
                src = viewport_dir / f"{vp_name}.mp4"
                if src.is_file():
                    dst = out_dir / f"{stem}.viewport.mp4"
                    shutil.move(str(src), str(dst))
                    viewport["mp4"] = str(dst)
            except Exception as exc:                             # noqa: BLE001
                viewport["error"] = repr(exc)
                print(f"[viewport] stop_recording failed ({exc!r})", flush=True)
        outcome = classify(success, bool(done), fired, timed_out)
        rec = dict(base, seed=seed, imu_delay_steps=delay_steps, outcome=outcome, success=success,
                   terminated=bool(done), terminations_fired=fired, steps=steps, sim_seconds=steps * step_dt,
                   wall_seconds=round(wall_s, 2),
                   # step_dt means the viewport's per-frame Kit update did not step physics; larger means it did.
                   sim_time_per_step=(None if t_sim0 is None or t_sim1 is None or steps == 0 else (t_sim1 - t_sim0) / steps),
                   final_button_distance_m=final_dist, min_button_distance_m=min_dist,
                   max_displacement_from_spawn_m=max_disp, camera_sensor=sensor, viewport=viewport,
                   mp4=sensor["mp4"], viewport_mp4=viewport["mp4"], episode=stem, panels=panels_rec)
        (out_dir / f"{stem}.json").write_text(json.dumps(rec, indent=2) + "\n")
        print(json.dumps({k: rec[k] for k in ("episode", "seed", "imu_delay_steps", "outcome", "terminations_fired",
                                              "steps", "sim_seconds", "final_button_distance_m", "min_button_distance_m")}
                         | {"sensor_frames": sensor["frames"], "viewport_frames": viewport["frames"]}), flush=True)
        episodes.append(rec)
        return rec

    def publish(search: dict, rec: dict) -> None:
        name = search["name"]
        out = dict(rec, search=search, selected_from_episode=rec["episode"])
        for key, suffix in (("mp4", ".mp4"), ("viewport_mp4", ".viewport.mp4")):
            src = rec.get(key)
            dst = out_dir / f"{name}{suffix}"
            if src and Path(src).is_file():
                shutil.copy2(src, dst)
                out[key] = str(dst)
            else:
                out[key] = None
        (out_dir / f"{name}.json").write_text(json.dumps(out, indent=2) + "\n")
        print(f"SEARCH {name}: {rec['outcome']} from {rec['episode']} -> {name}.json", flush=True)

    results = {}
    for search in searches:
        found = None
        for rec in episodes:
            if rec["imu_delay_steps"] == search["imu_delay_steps"] and matches(search["want"], rec["outcome"]):
                found = rec
                break
        tried = []
        if found is None:
            for k in range(search["max_seeds"]):
                seed = search["seed"] + k
                prior = next((r for r in episodes if r["seed"] == seed and r["imu_delay_steps"] == search["imu_delay_steps"]), None)
                rec = prior or record_episode(seed, search["imu_delay_steps"])
                tried.append({"seed": seed, "outcome": rec["outcome"]})
                if matches(search["want"], rec["outcome"]):
                    found = rec
                    break
        if found is not None:
            publish(search, found)
            results[search["name"]] = {"outcome": found["outcome"], "episode": found["episode"]}
        else:
            (out_dir / f"{search['name']}.json").write_text(json.dumps(
                {"search": search, "outcome": None, "found": False, "tried": tried, **base}, indent=2) + "\n")
            results[search["name"]] = {"outcome": None, "tried": tried}
            print(f"SEARCH {search['name']}: no {search['want']} in {len(tried)} seed(s)", flush=True)
    summary = {"label": label, "searches": results, "episodes": [
        {k: r[k] for k in ("episode", "seed", "imu_delay_steps", "outcome", "steps", "terminations_fired")}
        | {"sensor_frames": r["camera_sensor"]["frames"], "viewport_frames": r["viewport"]["frames"]} for r in episodes]}
    (out_dir / f"{label}.episodes.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("MAZE-RECORD SUMMARY " + json.dumps(summary), flush=True)
    try:
        shutil.rmtree(viewport_dir, ignore_errors=True)
    except Exception:                                            # noqa: BLE001
        pass
    env.close()
    print("MAZE_RECORD_DONE", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
