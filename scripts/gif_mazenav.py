"""Assemble docs/gifs/isaac/mazenav_seed0.gif from 97c camera-sensor frames.

Four seed-0 navigation policies (blind / lidar / stereo / both) in a 2x2.
These trained 6,000 iterations on MazeWaypointCommand. Last-50 TensorBoard:
button_reached = 0 in every seed of every arm; ~94% of episodes time out.
The clip is evidence of the walk-to-timeout gait, not of a button press.

Usage: gif_mazenav.py [frames_root] [out.gif] [--every 2] [--width 420]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

p = argparse.ArgumentParser(description=__doc__)
p.add_argument(
    "frames_root",
    nargs="?",
    default="results/clips/frames",
    help="directory that contains mazenav_{blind,lidar,stereo,both}/",
)
p.add_argument("out", nargs="?", default="docs/gifs/isaac/mazenav_seed0.gif")
p.add_argument("--every", type=int, default=4)
p.add_argument("--width", type=int, default=380)
a = p.parse_args()

FONT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


ARMS = (
    ("blind", "blind"),
    ("lidar", "lidar"),
    ("stereo", "stereo"),
    ("both", "both"),
)
root = Path(a.frames_root)
dirs = []
for label, _ in ARMS:
    d = root / f"mazenav_{label}"
    n = len(list(d.glob("frame_*.png"))) if d.is_dir() else 0
    if n < 50:
        raise SystemExit(f"{d} has {n} frames (need >= 50)")
    dirs.append((label, d, n))

n_use = min(n for _, _, n in dirs)
cell = a.width
H_TITLE, H_CAP = 36, 22
W = cell * 2
H = H_TITLE + (cell + H_CAP) * 2
f_title, f_cap = font(BOLD, 14), font(BOLD, 12)

frames = []
for i in range(0, n_use, a.every):
    canvas = Image.new("RGB", (W, H), (252, 252, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (8, 8),
        "Maze navigation seed 0 — walk to timeout, button 0/12 (n=3, four arms)",
        font=f_title,
        fill=(11, 11, 11),
    )
    for k, (label, d, _) in enumerate(dirs):
        col, row = k % 2, k // 2
        shot = Image.open(d / f"frame_{i:04d}.png").convert("RGB")
        side = min(shot.size)
        left = (shot.width - side) // 2
        shot = (
            shot.crop((left, 0, left + side, side))
            .resize((cell, cell))
            .filter(ImageFilter.GaussianBlur(1.6))
        )
        x, y = col * cell, H_TITLE + row * (cell + H_CAP)
        canvas.paste(shot, (x, y))
        draw.text((x + 8, y + cell + 3), label, font=f_cap, fill=(42, 120, 214))
    frames.append(
        canvas.quantize(colors=128, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    )

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
frames[0].save(
    out,
    save_all=True,
    append_images=frames[1:],
    duration=40 * a.every,
    loop=0,
    optimize=True,
)
print(f"{len(frames)} frames -> {out} ({out.stat().st_size / 1e6:.1f} MB)")
