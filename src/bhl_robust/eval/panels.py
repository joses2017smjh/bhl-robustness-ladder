"""Composite frames for the maze clips.

One picture per control step: a top view of the maze (main), the robot's own
paired depth (what its stereo rig returns: ray depth on this repo's two
harnesses, never RGB stereo matching), its lidar scan reduced to the sectors
the controller reads, optionally a robot-eye render, and a label bar that says
which of those the controller actually consumes. Pure numpy + PIL, so a login
node can build the frames from recorded data and the tests can run without a
simulator.

Conventions. Lidar sector 0 starts at -180 degrees (both harnesses:
``team_sensors.ray_pattern`` uses ``linspace(-pi, pi, ...)``, the Isaac
``LidarPatternCfg`` has ``horizontal_fov_range=(-180, 180)``), angles
increase counter-clockwise seen from above, body +x is forward. The scan is
drawn robot-centred with forward up and the robot's left on the left.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

FONT_CANDIDATES = (
    "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
)
BG = (18, 20, 24)
PANEL_BG = (30, 33, 40)
TEXT = (235, 235, 235)
DIM = (150, 155, 165)
ROBOT = (255, 200, 40)
SCAN = (80, 200, 255)
SCAN_FILL = (40, 110, 150)
STALE = (200, 60, 60)
GRID = (60, 66, 78)
MAX_GIF_BYTES = 5 * 1024 * 1024        # tests/test_result_artifacts.py's committable limit


def load_font(size: int):
    for cand in FONT_CANDIDATES:
        if os.path.isfile(cand):
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _title(draw: ImageDraw.ImageDraw, w: int, text: str, size: int = 14, colour=TEXT) -> int:
    font = load_font(size)
    draw.rectangle((0, 0, w, size + 8), fill=(40, 44, 52))
    draw.text((6, 4), text, font=font, fill=colour)
    return size + 8


def depth_colours(depth_m: np.ndarray, max_range: float) -> np.ndarray:
    """(H, W) metres -> (H, W, 3) uint8: near is warm and bright, far is dark."""
    d = np.asarray(depth_m, dtype=np.float32)
    d = np.nan_to_num(d, nan=max_range, posinf=max_range, neginf=0.0)
    t = np.clip(d / max_range, 0.0, 1.0)
    # three-stop ramp: near (1.0-t high) yellow -> orange -> deep blue far
    near = np.array([255, 220, 90], dtype=np.float32)
    mid = np.array([230, 110, 40], dtype=np.float32)
    far = np.array([20, 30, 70], dtype=np.float32)
    s = 1.0 - t
    out = np.where((s > 0.5)[..., None], mid + (near - mid) * ((s - 0.5) / 0.5)[..., None],
                   far + (mid - far) * (s / 0.5)[..., None])
    return out.astype(np.uint8)


def depth_pair_panel(pair, max_range: float, size: tuple[int, int], title: str,
                     pooled=None, stale: bool = False, subtitle: str | None = None) -> Image.Image:
    """Left/right depth images side by side; `pooled` (2, h, w) is what the
    controller reads, drawn small under them when it differs from the raw."""
    w, h = size
    img = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(img)
    top = _title(draw, w, title)
    pair = None if pair is None else np.asarray(pair, dtype=np.float32)
    if pair is None or stale:
        draw.text((8, top + 8), "no packet (stale)", font=load_font(13), fill=STALE)
        return img
    n = pair.shape[0]
    gap = 6
    avail_h = h - top - 8 - (0 if pooled is None else 36)
    tile_w = (w - gap * (n + 1)) // n
    tile_h = max(8, min(avail_h, tile_w))
    for i in range(n):
        rgb = depth_colours(pair[i], max_range)
        tile = Image.fromarray(rgb).resize((tile_w, tile_h), Image.NEAREST)
        x0 = gap + i * (tile_w + gap)
        img.paste(tile, (x0, top + 4))
        draw.text((x0 + 3, top + 4), "L" if i == 0 else "R", font=load_font(12), fill=TEXT)
    y = top + 4 + tile_h + 2
    if pooled is not None:
        pooled = np.asarray(pooled, dtype=np.float32)
        ph = 26
        pw = min(tile_w, ph * pooled.shape[2] // max(1, pooled.shape[1]))
        for i in range(pooled.shape[0]):
            tile = Image.fromarray(depth_colours(pooled[i], max_range)).resize((pw, ph), Image.NEAREST)
            img.paste(tile, (gap + i * (tile_w + gap), y + 2))
        draw.text((gap + n * (tile_w + gap) - 2 - 90, y + 6), f"policy sees {pooled.shape[1]}x{pooled.shape[2]}",
                  font=load_font(11), fill=DIM)
        y += ph + 6
    if subtitle:
        draw.text((8, min(y, h - 16)), subtitle, font=load_font(11), fill=DIM)
    return img


def lidar_panel(sector_m, max_range: float, size: tuple[int, int], title: str, window_m: float = 6.0,
                rays_xy=None, stale: bool = False, brake: dict | None = None, subtitle: str | None = None) -> Image.Image:
    """Robot-centred scan, forward up. `sector_m` are per-sector minimum ranges
    (metres, sector 0 at -180 deg, counter-clockwise). `rays_xy` are optional
    raw hit points in the body frame, (N, 2), drawn as dots behind the sectors."""
    w, h = size
    img = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(img)
    top = _title(draw, w, title)
    cx, cy = w // 2, top + (h - top) // 2
    radius = min(w, h - top) // 2 - 10
    scale = radius / window_m

    def to_px(xb, yb):
        # body +x forward -> up; body +y (left) -> screen left
        return cx - yb * scale, cy - xb * scale

    for r in (1.0, 2.0, 4.0, 6.0):
        if r <= window_m:
            rr = r * scale
            draw.ellipse((cx - rr, cy - rr, cx + rr, cy + rr), outline=GRID)
            draw.text((cx + rr - 18, cy - 12), f"{r:g}m", font=load_font(10), fill=DIM)
    if sector_m is None or stale:
        draw.text((8, top + 8), "no packet (stale)", font=load_font(13), fill=STALE)
    else:
        s = np.nan_to_num(np.asarray(sector_m, dtype=np.float32), nan=max_range, posinf=max_range)
        n = len(s)
        edges = np.linspace(-np.pi, np.pi, n + 1)
        if rays_xy is not None:
            for xb, yb in np.asarray(rays_xy, dtype=np.float32):
                if np.hypot(xb, yb) <= window_m:
                    px, py = to_px(xb, yb)
                    draw.point((px, py), fill=SCAN)
        poly = []
        for i in range(n):
            r = float(min(s[i], window_m))
            a0, a1 = edges[i], edges[i + 1]
            for a in (a0, a1):
                poly.append(to_px(r * np.cos(a), r * np.sin(a)))
        draw.polygon(poly, outline=SCAN, fill=SCAN_FILL)
        # sectors at the maximum range are "clear": mark them faintly
        for i in range(n):
            if s[i] >= max_range - 1e-6:
                a = 0.5 * (edges[i] + edges[i + 1])
                r = window_m
                px, py = to_px(r * np.cos(a), r * np.sin(a))
                draw.line((cx, cy, px, py), fill=GRID)
    # the robot: a small triangle pointing forward (up)
    draw.polygon([(cx, cy - 9), (cx - 6, cy + 6), (cx + 6, cy + 6)], fill=ROBOT)
    if brake:
        txt = []
        if brake.get("stale_stop"):
            txt.append("STOP: stale sensors")
        elif brake.get("range_scale", 1.0) < 0.999:
            txt.append(f"brake x{float(brake['range_scale']):.2f}")
        if brake.get("imu_scale", 1.0) < 0.999:
            txt.append(f"imu x{float(brake['imu_scale']):.2f}")
        if txt:
            draw.text((8, h - 18), "  ".join(txt), font=load_font(11), fill=ROBOT)
    if subtitle:
        draw.text((8, top + 4), subtitle, font=load_font(11), fill=DIM)
    return img


def image_panel(rgb, size: tuple[int, int], title: str) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(img)
    top = _title(draw, w, title)
    if rgb is not None:
        pic = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8)))
        pic = pic.resize((w, max(1, h - top)), Image.BILINEAR)
        img.paste(pic, (0, top))
    return img


def compose_frame(main_rgb, panels: list[Image.Image], header: str, footer: str,
                  side_w: int = 320, bar_h: int = 34) -> np.ndarray:
    """Main view on the left, the panels stacked on the right, a header line
    above and a footer line below. Returns an (H, W, 3) uint8 array whose
    sides are even (yuv420p)."""
    main = Image.fromarray(np.ascontiguousarray(np.asarray(main_rgb)[..., :3].astype(np.uint8)))
    mw, mh = main.size
    side_h = sum(p.size[1] for p in panels)
    body_h = max(mh, side_h)
    W, H = mw + side_w, body_h + 2 * bar_h
    W += W % 2
    H += H % 2
    canvas = Image.new("RGB", (W, H), BG)
    canvas.paste(main, (0, bar_h))
    y = bar_h
    for p in panels:
        if p.size[0] != side_w:
            p = p.resize((side_w, p.size[1]))
        canvas.paste(p, (mw, y))
        y += p.size[1]
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), header, font=load_font(17), fill=TEXT)
    draw.text((8, H - bar_h + 8), footer, font=load_font(13), fill=DIM)
    return np.asarray(canvas)


def write_gif(src_mp4: Path, out_gif: Path, *, fps: int = 8, speed: float = 1.0, width: int = 860,
              max_colors: int = 128, max_bytes: int = MAX_GIF_BYTES) -> dict:
    """ffmpeg palette GIF from an mp4; shrinks (colours, fps, width) until it
    fits the committable budget. Returns what it ended up using."""
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    attempts = [(max_colors, fps, width), (96, fps, width), (96, max(5, fps - 2), width),
                (64, max(5, fps - 2), width), (64, 5, int(width * 0.85)), (48, 5, int(width * 0.75))]
    last = None
    for colors, f, wpx in attempts:
        rate = "" if speed == 1.0 else f"setpts=PTS/{speed:g},"
        vf = f"{rate}fps={f},scale={wpx}:-2:flags=lanczos,split[a][b];[a]palettegen=max_colors={colors}:stats_mode=diff[p];" \
             f"[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(src_mp4), "-filter_complex", vf,
                        "-loop", "0", str(out_gif)], check=True)
        size = out_gif.stat().st_size
        last = {"gif": str(out_gif), "bytes": size, "mb": round(size / 1e6, 2), "fps": f, "width": wpx,
                "max_colors": colors, "speed": speed, "within_budget": size <= max_bytes}
        if size <= max_bytes:
            break
    return last
