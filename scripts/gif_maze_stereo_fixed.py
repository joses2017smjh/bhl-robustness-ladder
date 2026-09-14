"""Assemble docs/gifs/isaac/maze_stereo_fixed.gif from the 97b clip frames.

Top: the corrected 16x16 stereo policy (mazefix-stereo-s0) in Isaac, through the
camera-sensor recorder. Bottom: the left stereo eye as every B5 run on v60 had it
before 3f7b679 (the raw (w, x, y, z) tuple read as (x, y, z, w): 20 degrees up,
upside down) beside the corrected eye this policy trained on. Depth is drawn
bright-near, black at the 6 m range limit -- black is "nothing there".

Usage: gif_maze_stereo_fixed.py <frames_dir> <out.gif> [--every 2] [--width 560]
"""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

p = argparse.ArgumentParser(description=__doc__)
p.add_argument("frames")
p.add_argument("out")
p.add_argument("--every", type=int, default=2)
p.add_argument("--width", type=int, default=560)  # total; the robot view is width - 200
a = p.parse_args()

FONT = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf"
BOLD = "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf"


def font(path, size):
    try:
        return ImageFont.truetype(path, size)
    except OSError:
        return ImageFont.load_default()


VIEW = a.width - 200            # square crop around the followed robot
PANEL = 132
H_TITLE = 30
W, H = VIEW + 200, H_TITLE + max(VIEW, 2 * (PANEL + 44))
f_title, f_cap, f_small = font(BOLD, 13), font(BOLD, 12), font(FONT, 10)

d = Path(a.frames)
n = len(list(d.glob("frame_*.png")))
frames = []
for i in range(0, n, a.every):
    canvas = Image.new("RGB", (W, H), (252, 252, 251))
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), "Maze stereo, 16×16 an eye, cameras fixed: terrain level 0.88 (was 0.02)",
              font=f_title, fill=(11, 11, 11))
    shot = Image.open(d / f"frame_{i:04d}.png").convert("RGB")
    side = min(shot.size)
    left = (shot.width - side) // 2
    # The bumpy terrain is per-pixel noise, which GIF cannot compress; a light
    # blur keeps the robot readable and the file an order of magnitude smaller.
    shot = shot.crop((left, 0, left + side, side)).resize((VIEW, VIEW)).filter(ImageFilter.GaussianBlur(1.2))
    canvas.paste(shot, (0, H_TITLE))
    x = VIEW + (200 - PANEL) // 2
    for k, (name, cap, sub, colour) in enumerate((
        ("stereo_raw", "before the fix", "+20° up, upside down", (235, 104, 52)),
        ("stereo_l", "after the fix", "−20° down, on the ground", (42, 120, 214)),
    )):
        y = H_TITLE + 6 + k * (PANEL + 44)
        eye = Image.open(d / f"{name}_{i:04d}.png").convert("L").resize((PANEL, PANEL), Image.NEAREST)
        canvas.paste(eye.convert("RGB"), (x, y))
        draw.rectangle([x - 2, y - 2, x + PANEL + 1, y + PANEL + 1], outline=colour, width=2)
        draw.text((x, y + PANEL + 4), cap, font=f_cap, fill=colour)
        draw.text((x, y + PANEL + 19), sub, font=f_small, fill=(82, 81, 78))
    frames.append(canvas.quantize(colors=64, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE))

frames[0].save(a.out, save_all=True, append_images=frames[1:], duration=40 * a.every, loop=0, optimize=True)
print(f"{len(frames)} frames -> {a.out} ({Path(a.out).stat().st_size / 1e6:.1f} MB)")
