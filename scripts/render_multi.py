"""Render N policies racing side by side in one shared MuJoCo scene.

The paired GIFs elsewhere in this repo are two separate rollouts composited
after the fact. This is the other thing: one world, one solver, one clock.

Three things this can add to that shot, each because the plain version was
misreadable:

* `--variant humanoid` runs the 22-DoF model. The lab-floor clip was rendered
  on the biped, so the robots had no arms in a section whose neighbouring
  result is that arms are what carry this machine over rough ground.
* `--depth-of LABEL` draws that robot's own egocentric depth along the bottom,
  as a live image plus a scrolling waterfall of the centre column. A still
  depth panel shows what the robot sees; the waterfall shows what it *saw*,
  which is the question worth asking of a clip where something trips.
* `--hero LABEL` paints one robot in the orange/black livery instead of a flat
  tint, for when the interesting thing is a single robot's gait rather than
  which of four fell.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from bhl_robust.eval.depth import FAR_M, NEAR_M, _turbo
from bhl_robust.eval.livery import JOINT_RGBA, SHELL_RGBA
from bhl_robust.eval.mjcf_assets import EGO_CAM_NAME
from bhl_robust.eval.multi_robot import PALETTE, MultiRunner, build_multi
from bhl_robust.eval.video import EpisodeRecorder, FONT

TILT_LIMIT = 0.78

# A fallen robot keeps its hue but loses most of its value, so the eye reads the
# frame as "three colours and one shadow" without a caption. Tinting it flat red
# instead would collide with the red fall border.
_DEAD_SCALE = 0.30


def _legend(frame: np.ndarray, labels: list[str],
            hero: int | None = None) -> np.ndarray:
    """Paint a colour-coded strip along the bottom of a rendered frame.

    The hero's chip is drawn in its livery colours rather than in its palette
    tint. A legend that names a robot by a colour the robot is not wearing is
    worse than no legend.
    """
    h, w, _ = frame.shape
    bar_h = 36
    out = frame.copy()
    out[-bar_h:, :, :] = (18, 18, 22)
    n = len(labels)
    slot_w = w // max(n, 1)
    for i, lab in enumerate(labels):
        x0 = i * slot_w + 8
        x1 = min(x0 + 18, w - 4)
        if i == hero:
            mid = (x0 + x1) // 2
            out[-bar_h + 8:-8, x0:mid, :] = tuple(int(255 * c) for c in SHELL_RGBA[:3])
            out[-bar_h + 8:-8, mid:x1, :] = tuple(int(255 * c) for c in JOINT_RGBA[:3])
        else:
            out[-bar_h + 8:-8, x0:x1, :] = tuple(
                int(255 * c) for c in PALETTE[i % len(PALETTE)][:3])
        # Tiny 5x7 bitmap is overkill; a coloured chip plus the caption
        # burned in by ffmpeg is enough to read. Keep the chip here so the
        # GIF still identifies robots if the text pass is skipped.
    return out


def kill_colour(model: mujoco.MjModel, prefix: str) -> None:
    """Darken every geom of one robot, in place, when it falls."""
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or ""
        if b != 0 and name.startswith(prefix):
            model.geom_rgba[g, :3] *= _DEAD_SCALE


class DepthBand:
    """The bottom strip: live depth image, colour ramp, and a time waterfall.

    The waterfall is the centre column of the depth image, one pixel per
    captured frame, scrolling left. That column is the ground straight ahead of
    the robot, so a bump crossing the field of view draws a diagonal streak that
    arrives at the right-hand edge shortly before the foot does -- which is the
    only way a clip can show that an obstacle was visible *before* it mattered.
    """

    def __init__(self, width: int, height: int, res: int):
        self.w, self.h, self.res = width, height, res
        self.k = max(1, (height - 26) // res)
        self.img_px = res * self.k
        self.pad = 20
        self.ramp_w = 12
        wf_x0 = self.pad + self.img_px + 16 + self.ramp_w + 18
        self.wf_x0 = wf_x0
        self.wf_w = max(16, width - wf_x0 - self.pad)
        # Start full-far so the strip reads as "no history yet" rather than as a
        # wall 25 cm in front of the robot.
        self.wf = np.full((res, self.wf_w), FAR_M, dtype=np.float32)

    def _norm(self, d):
        return (np.asarray(d) - NEAR_M) / (FAR_M - NEAR_M)

    def render(self, depth: np.ndarray) -> np.ndarray:
        band = np.full((self.h, self.w, 3), 16, dtype=np.uint8)

        img = _turbo(self._norm(depth))
        img = np.repeat(np.repeat(img, self.k, axis=0), self.k, axis=1)
        y0 = (self.h - img.shape[0]) // 2
        band[y0:y0 + img.shape[0], self.pad:self.pad + img.shape[1]] = img

        rx = self.pad + self.img_px + 16
        ramp = _turbo(np.linspace(1.0, 0.0, img.shape[0])[:, None].repeat(self.ramp_w, axis=1))
        band[y0:y0 + img.shape[0], rx:rx + self.ramp_w] = ramp

        self.wf = np.roll(self.wf, -1, axis=1)
        self.wf[:, -1] = depth[:, self.res // 2]
        strip = _turbo(self._norm(self.wf))
        strip = np.repeat(strip, self.k, axis=0)
        band[y0:y0 + strip.shape[0], self.wf_x0:self.wf_x0 + self.wf_w] = strip

        band[y0 - 1, self.pad:self.pad + self.img_px] = (70, 70, 78)
        band[y0 - 1, self.wf_x0:self.wf_x0 + self.wf_w] = (70, 70, 78)
        return band


class LegendRecorder(EpisodeRecorder):
    """EpisodeRecorder with a colour-chip strip and an optional depth band."""

    def __init__(self, model: mujoco.MjModel, path, fps: float, width: int = 960,
                 height: int = 540, caption: str = "", track_body: str = "",
                 labels: list[str] | None = None, hero: int | None = None,
                 depth_cam: int | None = None,
                 depth_res: int = 64, depth_band: int = 176):
        self._band = depth_band if depth_cam is not None else 0
        # The parent sizes both the ffmpeg input and its own renderer from
        # `height`. The band is drawn, not rendered, so ffmpeg gets the padded
        # height and the scene renderer is rebuilt at the unpadded one.
        super().__init__(model, path, fps=fps, width=width,
                         height=height + self._band, caption=caption,
                         track_body=track_body)
        self.labels = labels or []
        self.hero = hero
        self._scene_h = height
        self._depth_cam = depth_cam
        self._dep = None
        self._strip = None
        if self._band:
            self.renderer.close()
            self.renderer = mujoco.Renderer(model, height=self._scene_h, width=width)
            self._dep = mujoco.Renderer(model, height=depth_res, width=depth_res)
            self._dep.enable_depth_rendering()
            self._strip = DepthBand(width, self._band, depth_res)

    def capture(self, data: mujoco.MjData, flash: str | None = None) -> None:
        if self._track_id >= 0:
            self.camera.lookat[:] = data.xpos[self._track_id]
        self.renderer.update_scene(data, camera=self.camera)
        scene = self.renderer.render()

        if self._band:
            frame = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            frame[:self._scene_h] = scene
            self._dep.update_scene(data, camera=self._depth_cam)
            frame[self._scene_h:] = self._strip.render(self._dep.render())
            frame[self._scene_h - 2:self._scene_h] = (40, 40, 46)
        else:
            frame = scene.copy()

        if self.labels:
            frame = _legend(frame, self.labels, self.hero)
        if flash:
            colour = {"push": (255, 190, 0), "fall": (220, 40, 40)}.get(flash)
            if colour is not None:
                b = 10
                frame[:b, :, :] = colour
                frame[-b:, :, :] = colour
                frame[:, :b, :] = colour
                frame[:, -b:, :] = colour
        try:
            self._proc.stdin.write(frame.astype(np.uint8).tobytes())
            self.n_frames += 1
        except BrokenPipeError:
            pass

    def close(self) -> str | None:
        if self._dep is not None:
            try:
                self._dep.close()
            except Exception:
                pass
        return super().close()


def mp4_to_gif(mp4: Path, gif: Path, seconds: float = 9.0, width: int = 880,
               fps: int = 10) -> None:
    """Palette-resampled GIF that GitHub will actually inline."""
    filt = (
        f"fps={fps},scale={width}:-2:flags=lanczos,"
        f"trim=duration={seconds:.2f},setpts=PTS-STARTPTS,split[s0][s1];"
        "[s0]palettegen=max_colors=128[pal];"
        "[s1][pal]paletteuse=dither=bayer:bayer_scale=3"
    )
    gif.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
         "-filter_complex", filt, "-loop", "0", str(gif)],
        check=True,
    )


def burn_caption(mp4: Path, labels: list[str]) -> None:
    """Burn colour-keyed names into the bottom bar (in place)."""
    if not Path(FONT).is_file():
        return
    chips = "     ".join(
        lab.replace(":", r"\:").replace("'", "") for lab in labels
    )
    tmp = mp4.with_suffix(".caption.mp4")
    vf = (
        f"drawtext=fontfile={FONT}:text='{chips}':"
        f"x=36:y=h-28:fontsize=16:fontcolor=white"
    )
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(mp4),
         "-vf", vf, "-c:v", "libx264", "-pix_fmt", "yuv420p",
         "-preset", "veryfast", "-crf", "23", str(tmp)],
        check=True,
    )
    tmp.replace(mp4)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deploy", nargs="+", required=True, help="deploy.yaml per robot")
    p.add_argument("--labels", nargs="+", required=True)
    p.add_argument("--upstream", type=Path, required=True)
    p.add_argument("--cache-dir", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--seconds", type=float, default=12.0)
    p.add_argument("--vx", type=float, default=0.3)
    p.add_argument("--push", type=float, default=0.0)
    p.add_argument("--push-every", type=float, default=3.5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--world", choices=("flat", "lab"), default="flat")
    p.add_argument("--variant", choices=("biped", "humanoid"), default="biped")
    p.add_argument("--depth-of", default=None,
                   help="label whose egocentric depth is drawn along the bottom")
    p.add_argument("--depth-res", type=int, default=64)
    p.add_argument("--hero", default=None,
                   help="label painted in the orange/black livery")
    p.add_argument("--gif", type=Path, default=None)
    p.add_argument("--gif-seconds", type=float, default=None)
    p.add_argument("--gif-fps", type=int, default=10)
    p.add_argument("--gif-width", type=int, default=880)
    args = p.parse_args()

    n = len(args.deploy)
    if len(args.labels) != n:
        raise SystemExit(f"{n} deploy files but {len(args.labels)} labels")

    def slot_of(label, what):
        if label is None:
            return None
        if label not in args.labels:
            raise SystemExit(f"--{what} {label!r} is not one of {args.labels}")
        return args.labels.index(label)

    hero = slot_of(args.hero, "hero")
    depth_slot = slot_of(args.depth_of, "depth-of")

    cfgs = [OmegaConf.load(d) for d in args.deploy]
    model, slots = build_multi(
        args.upstream, args.cache_dir, n, args.labels, variant=args.variant,
        world=args.world, ego_camera=depth_slot is not None, hero=hero,
    )

    depth_cam = None
    if depth_slot is not None:
        cam = f"r{depth_slot}_{EGO_CAM_NAME}"
        depth_cam = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam)
        if depth_cam < 0:
            raise SystemExit(f"no camera {cam!r} in the composed model")

    ctrls = []
    for c in cfgs:
        rc = RlController(c)
        rc.load_policy()
        ctrls.append(rc)

    run = MultiRunner(model, slots, cfgs, ctrls)
    rng = np.random.default_rng(args.seed)
    run.reset(rng)

    cap = f"{n} policies, one world   cmd vx={args.vx:+.2f}"
    cap += "   22 DoF" if args.variant == "humanoid" else "   12 DoF"
    if args.world == "lab":
        cap += "   lab floor"
    if args.push > 0:
        cap += f"   push {args.push:.2f} m/s"
    if depth_slot is not None:
        cap += f"   depth: {args.labels[depth_slot]}"

    rec = LegendRecorder(
        model, args.out, fps=1.0 / cfgs[0].policy_dt,
        width=1280, height=640, caption=cap, track_body="",
        labels=args.labels, hero=hero, depth_cam=depth_cam,
        depth_res=args.depth_res,
    )
    rec.camera.distance = 5.4 if args.world == "lab" else 4.8
    rec.camera.elevation = -16.0
    rec.camera.azimuth = 128.0
    rec._track_id = -1
    rec.camera.lookat[:] = [0.8, 0.0, 0.32]

    command = (args.vx, 0.0, 0.0)
    steps = int(args.seconds / cfgs[0].policy_dt)
    push_every = int(args.push_every / cfgs[0].policy_dt) if args.push > 0 else 0
    settle = int(1.0 / cfgs[0].policy_dt)
    # Two independent markers, because they mean opposite things. A push is an
    # event: amber, for eight frames, gone. A fall is a state: red, and it stays
    # red, matching the paired clips where the border comes on and does not go
    # off. Sharing one counter meant every fall was drawn in the push colour.
    push_flash = 0
    any_fell = False
    fell_at = [None] * n
    peak = np.array([float(run.d.qpos[s.qpos_adr]) for s in run.slots])

    for t in range(steps):
        if push_every and t > settle and t % push_every == 0:
            run.push_all(args.push, rng)
            push_flash = 8

        targets = []
        for i in range(n):
            obs = run.observe(i, command)
            targets.append(ctrls[i].update(obs))
        run.step(targets)

        for i in range(n):
            peak[i] = max(peak[i], float(run.d.qpos[run.slots[i].qpos_adr]))
            if fell_at[i] is None and run.tilt(i) > TILT_LIMIT:
                fell_at[i] = t * cfgs[0].policy_dt
                kill_colour(model, f"r{i}_")
                any_fell = True

        # Track the pack so the shot stays on the robots as they walk away. Only
        # robots still upright vote: once one is face down it stops advancing,
        # and averaging it in drags the camera off the ones still walking.
        alive = [float(run.d.qpos[s.qpos_adr])
                 for i, s in enumerate(run.slots) if fell_at[i] is None]
        xs = alive if alive else [float(run.d.qpos[s.qpos_adr]) for s in run.slots]
        rec.camera.lookat[0] = float(np.mean(xs)) + 0.5
        rec.camera.lookat[1] = 0.0

        mark = "push" if push_flash > 0 else ("fall" if any_fell else None)
        rec.capture(run.d, mark)
        push_flash = max(0, push_flash - 1)

    err = rec.close()
    print(f"video -> {args.out}  ({rec.n_frames} frames)" if not err else f"ffmpeg: {err}")
    for i, s in enumerate(slots):
        status = "FELL @ %.1fs" % fell_at[i] if fell_at[i] is not None else "upright"
        print(f"  {s.label:22s} {status:16s} peak x={peak[i]:+.2f} m")

    burn_caption(args.out, args.labels)
    gif = args.gif or args.out.with_suffix(".gif")
    mp4_to_gif(args.out, gif, seconds=args.gif_seconds or min(args.seconds, 9.0),
               width=args.gif_width, fps=args.gif_fps)
    print(f"gif   -> {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
