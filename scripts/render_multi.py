"""Render N policies racing side by side in one shared MuJoCo scene.

The paired GIFs elsewhere in this repo are two separate rollouts composited
after the fact. This is the other thing: one world, one solver, one clock.
"""
from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import mujoco
import numpy as np
from omegaconf import OmegaConf

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController
from bhl_robust.eval.multi_robot import PALETTE, MultiRunner, build_multi
from bhl_robust.eval.video import EpisodeRecorder, FONT

TILT_LIMIT = 0.78


def _legend(frame: np.ndarray, labels: list[str]) -> np.ndarray:
    """Paint a colour-coded strip along the bottom of a rendered frame."""
    h, w, _ = frame.shape
    bar_h = 36
    out = frame.copy()
    out[-bar_h:, :, :] = (18, 18, 22)
    n = len(labels)
    slot_w = w // max(n, 1)
    for i, lab in enumerate(labels):
        x0 = i * slot_w + 8
        x1 = min(x0 + 18, w - 4)
        rgb = tuple(int(255 * c) for c in PALETTE[i % len(PALETTE)][:3])
        out[-bar_h + 8:-8, x0:x1, :] = rgb
        # Tiny 5x7 bitmap is overkill; a coloured chip plus the caption
        # burned in by ffmpeg is enough to read. Keep the chip here so the
        # GIF still identifies robots if the text pass is skipped.
    return out


class LegendRecorder(EpisodeRecorder):
    """EpisodeRecorder that stamps a colour chip strip onto every frame."""

    def __init__(self, *args, labels: list[str] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self.labels = labels or []

    def capture(self, data: mujoco.MjData, flash: str | None = None) -> None:
        if self._track_id >= 0:
            self.camera.lookat[:] = data.xpos[self._track_id]
        self.renderer.update_scene(data, camera=self.camera)
        frame = self.renderer.render()
        if self.labels:
            frame = _legend(frame, self.labels)
        if flash:
            colour = {"push": (255, 190, 0), "fall": (220, 40, 40)}.get(flash)
            if colour is not None:
                b = 10
                frame = frame.copy()
                frame[:b, :, :] = colour
                frame[-b:, :, :] = colour
                frame[:, :b, :] = colour
                frame[:, -b:, :] = colour
        try:
            self._proc.stdin.write(frame.astype(np.uint8).tobytes())
            self.n_frames += 1
        except BrokenPipeError:
            pass


def mp4_to_gif(mp4: Path, gif: Path, seconds: float = 9.0, width: int = 880) -> None:
    """Palette-resampled GIF that GitHub will actually inline."""
    filt = (
        f"fps=10,scale={width}:-2:flags=lanczos,"
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
    p.add_argument("--gif", type=Path, default=None)
    args = p.parse_args()

    n = len(args.deploy)
    cfgs = [OmegaConf.load(d) for d in args.deploy]
    model, slots = build_multi(
        args.upstream, args.cache_dir, n, args.labels, world=args.world,
    )
    ctrls = []
    for c in cfgs:
        rc = RlController(c)
        rc.load_policy()
        ctrls.append(rc)

    run = MultiRunner(model, slots, cfgs, ctrls)
    rng = np.random.default_rng(args.seed)
    run.reset(rng)

    cap = f"{n} policies, one world   cmd vx={args.vx:+.2f}"
    if args.world == "lab":
        cap += "   lab floor"
    if args.push > 0:
        cap += f"   push {args.push:.2f} m/s"
    rec = LegendRecorder(
        model, args.out, fps=1.0 / cfgs[0].policy_dt,
        width=1280, height=640, caption=cap, track_body="",
        labels=args.labels,
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
    flash = 0
    fell_at = [None] * n

    for t in range(steps):
        if push_every and t > settle and t % push_every == 0:
            run.push_all(args.push, rng)
            flash = 8

        targets = []
        for i in range(n):
            obs = run.observe(i, command)
            a = ctrls[i].update(obs)
            targets.append(a)
        run.step(targets)

        for i in range(n):
            if fell_at[i] is None and run.tilt(i) > TILT_LIMIT:
                fell_at[i] = t * cfgs[0].policy_dt

        # Track the pack so the shot stays on the robots as they walk away.
        xs = [float(run.d.qpos[s.qpos_adr]) for s in run.slots]
        rec.camera.lookat[0] = float(np.mean(xs)) + 0.5
        rec.camera.lookat[1] = 0.0

        rec.capture(run.d, "push" if flash > 0 else None)
        flash = max(0, flash - 1)

    err = rec.close()
    print(f"video -> {args.out}  ({rec.n_frames} frames)" if not err else f"ffmpeg: {err}")
    for i, s in enumerate(slots):
        x = run.d.qpos[s.qpos_adr]
        status = "FELL @ %.1fs" % fell_at[i] if fell_at[i] is not None else "upright"
        print(f"  {s.label:22s} {status:16s} x={x:+.2f} m")

    burn_caption(args.out, args.labels)
    gif = args.gif or args.out.with_suffix(".gif")
    mp4_to_gif(args.out, gif, seconds=min(args.seconds, 9.0))
    print(f"gif   -> {gif}  ({gif.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
