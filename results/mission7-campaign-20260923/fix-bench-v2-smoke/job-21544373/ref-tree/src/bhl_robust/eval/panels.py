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
import tempfile
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


_FFMPEG: str | None = None


def ffmpeg_exe() -> str:
    """The system ffmpeg when it starts, else imageio-ffmpeg's bundled binary.

    dgxh-2's ffmpeg dies loading libvmaf (job 21408633), while the bundled one
    that imageio writes the mp4s with runs everywhere the venv does."""
    global _FFMPEG
    if _FFMPEG is None:
        try:
            subprocess.run(["ffmpeg", "-version"], check=True, capture_output=True, timeout=30)
            _FFMPEG = "ffmpeg"
        except Exception:                                            # noqa: BLE001
            import imageio_ffmpeg
            _FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()
    return _FFMPEG


def load_font(size: int):
    for cand in FONT_CANDIDATES:
        if os.path.isfile(cand):
            try:
                return ImageFont.truetype(cand, size)
            except OSError:
                continue
    return ImageFont.load_default()


def _title(draw: ImageDraw.ImageDraw, w: int, text: str, size: int = 13, colour=TEXT) -> int:
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
    avail_h = h - top - 8 - (0 if pooled is None else 36) - (16 if subtitle else 0)
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
        label, font = f"policy sees {pooled.shape[1]}x{pooled.shape[2]}", load_font(11)
        draw.text((w - gap - 2 - int(draw.textlength(label, font=font)), y + 6), label, font=font, fill=DIM)
        y += ph + 6
    if subtitle:
        draw.text((8, min(y + 2, h - 15)), subtitle, font=load_font(11), fill=DIM)
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

    for r in (0.5, 1.0, 2.0, 4.0, 6.0):
        if r <= window_m and (r >= 1.0 or window_m <= 3.0):
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


def image_panel(rgb, size: tuple[int, int], title: str, subtitle: str | None = None) -> Image.Image:
    w, h = size
    img = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(img)
    top = _title(draw, w, title)
    sub_h = 16 if subtitle else 0
    if rgb is not None:
        pic = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8)))
        pic = pic.resize((w, max(1, h - top - sub_h)), Image.BILINEAR)
        img.paste(pic, (0, top))
    if subtitle:
        draw.text((8, h - 15), subtitle, font=load_font(11), fill=DIM)
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
              max_colors: int = 128, max_bytes: int = MAX_GIF_BYTES, badge: str | None = None) -> dict:
    """ffmpeg palette GIF from an mp4; shrinks (colours, fps, width) until it
    fits the committable budget. `badge` (default: the playback speed when it
    is not 1x) is drawn bottom-right so a sped-up GIF says so on its face.
    Returns what it ended up using."""
    out_gif = Path(out_gif)
    out_gif.parent.mkdir(parents=True, exist_ok=True)
    if badge is None and speed != 1.0:
        badge = f"GIF {speed:g}x"
    # The badge is a PIL-drawn PNG overlaid by ffmpeg: the bundled imageio-ffmpeg build has no
    # drawtext filter (job 21442351), while overlay is in every build.
    tmp = tempfile.TemporaryDirectory()
    badge_in, badge_f, tail = [], "", "[v]"
    if badge:
        font = load_font(16)
        tw, th = (lambda b: (b[2] - b[0], b[3] - b[1]))(ImageDraw.Draw(Image.new("RGB", (1, 1))).textbbox((0, 0), badge, font=font))
        img = Image.new("RGBA", (tw + 12, th + 12), (0x1E, 0x21, 0x28, 217))
        ImageDraw.Draw(img).text((6, 6), badge, font=font, fill=(255, 255, 255, 255), anchor="lt")
        img.save(Path(tmp.name) / "badge.png")
        badge_in, tail = ["-i", str(Path(tmp.name) / "badge.png")], "[vb]"
        badge_f = "[v][1:v]overlay=W-w-4:H-h-2[vb];"
    attempts = [(max_colors, fps, width), (96, fps, width), (96, max(5, fps - 2), width),
                (64, max(5, fps - 2), width), (64, 5, int(width * 0.85)), (48, 5, int(width * 0.75))]
    last = None
    for colors, f, wpx in attempts:
        rate = "" if speed == 1.0 else f"setpts=PTS/{speed:g},"
        vf = f"[0:v]{rate}fps={f},scale={wpx}:-2:flags=lanczos[v];{badge_f}{tail}split[a][b];" \
             f"[a]palettegen=max_colors={colors}:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle"
        subprocess.run([ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(src_mp4), *badge_in, "-filter_complex", vf,
                        "-loop", "0", str(out_gif)], check=True)
        size = out_gif.stat().st_size
        last = {"gif": str(out_gif), "bytes": size, "mb": round(size / 1e6, 2), "fps": f, "width": wpx, "ffmpeg": ffmpeg_exe(),
                "max_colors": colors, "speed": speed, "badge": badge, "within_budget": size <= max_bytes}
        if size <= max_bytes:
            break
    tmp.cleanup()
    return last


AXIS_COLOURS = ((235, 90, 80), (90, 200, 110), (90, 150, 255))     # x red, y green, z blue


def _trace_box(draw, box, series, t_now, window_s, lo, hi, label, unit, font, zero=True):
    """Three time series (t, (3,)) in a box; the newest sample at the right edge."""
    x0, y0, x1, y1 = box
    draw.rectangle(box, outline=GRID)
    if zero and lo < 0 < hi:
        yz = y1 - (0 - lo) / (hi - lo) * (y1 - y0)
        draw.line((x0, yz, x1, yz), fill=GRID)
    draw.text((x0 + 4, y0 + 2), label, font=font, fill=TEXT)
    draw.text((x1 - 4 - draw.textlength(f"{hi:g}", font=font), y0 + 2), f"{hi:g}", font=font, fill=DIM)
    draw.text((x1 - 4 - draw.textlength(f"{lo:g} {unit}", font=font), y1 - 14), f"{lo:g} {unit}", font=font, fill=DIM)
    if len(series) < 2:
        return
    ts = np.array([s[0] for s in series])
    vals = np.array([s[1] for s in series])
    xs = x1 - (t_now - ts) / window_s * (x1 - x0)
    for k in range(3):
        v = np.clip(vals[:, k], lo, hi)
        ys = y1 - (v - lo) / (hi - lo) * (y1 - y0)
        pts = [(float(a), float(b)) for a, b in zip(xs, ys) if a >= x0]
        if len(pts) > 1:
            draw.line(pts, fill=AXIS_COLOURS[k], width=2)


def imu_strip(hist, latest, size: tuple[int, int], title: str, window_s: float = 4.0,
              gyro_range: float = 100.0, accel_range: tuple[float, float] = (-5.0, 25.0)) -> Image.Image:
    """Gyro (deg/s) and specific force (m/s^2) traces, x/y/z, over the last `window_s`."""
    w, h = size
    img = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(img)
    top = _title(draw, w, title)
    font = load_font(12)
    if latest is None:
        draw.text((8, top + 8), "no IMU sample yet", font=load_font(13), fill=STALE)
        return img
    t_now = latest["t"]
    gap, pad = 12, 8
    bw = (w - 2 * pad - gap) // 2
    y0, y1 = top + 6, h - 22
    gy = [(t, g) for t, g, _ in hist]
    ac = [(t, a) for t, _, a in hist]
    _trace_box(draw, (pad, y0, pad + bw, y1), gy, t_now, window_s, -gyro_range, gyro_range,
               "gyro (deg/s)", "deg/s", font)
    _trace_box(draw, (pad + bw + gap, y0, pad + 2 * bw + gap, y1), ac, t_now, window_s,
               accel_range[0], accel_range[1], "accelerometer, specific force (m/s^2)", "m/s^2", font)
    legend_x = pad
    for k, name in enumerate(("x forward", "y left", "z up")):
        draw.rectangle((legend_x, h - 16, legend_x + 10, h - 8), fill=AXIS_COLOURS[k])
        draw.text((legend_x + 14, h - 19), name, font=font, fill=DIM)
        legend_x += 90
    g, a = latest["gyro_dps"], latest["accel"]
    draw.text((legend_x + 10, h - 19), f"now: gyro [{g[0]:+6.1f} {g[1]:+6.1f} {g[2]:+6.1f}]  "
              f"accel [{a[0]:+5.2f} {a[1]:+5.2f} {a[2]:+5.2f}]  last {window_s:g} s", font=font, fill=DIM)
    return img


def imu_readout_panel(latest, size: tuple[int, int], title: str, notes: list[str]) -> Image.Image:
    """Attitude estimate vs truth, plus the magnetometer and barometer channels."""
    w, h = size
    img = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(img)
    top = _title(draw, w, title)
    font, small = load_font(13), load_font(11)
    if latest is None:
        draw.text((8, top + 8), "no IMU sample yet", font=font, fill=STALE)
        return img
    y = top + 6
    draw.text((8, y), "attitude     filter    truth    error", font=small, fill=DIM)
    y += 16
    for k, name in enumerate(("roll", "pitch", "yaw")):
        e, t, err = latest["rpy_est"][k], latest["rpy_true"][k], latest["rpy_err"][k]
        draw.text((8, y), f"{name:<6} {e:+8.1f}° {t:+8.1f}° {err:+6.1f}°", font=font, fill=TEXT)
        y += 17
    y += 4
    m = latest["mag_ut"]
    draw.text((8, y), f"mag  |B| {np.linalg.norm(m):5.1f} µT  hdg {latest['mag_heading_deg']:+6.1f}°",
              font=small, fill=DIM)
    y += 15
    draw.text((8, y), f"baro {latest['baro_pa'] / 100:8.2f} hPa  Δalt {latest['baro_dalt_m']:+5.2f} m",
              font=small, fill=DIM)
    y += 17
    for line in notes:
        if y > h - 14:
            break
        draw.text((8, y), line, font=small, fill=DIM)
        y += 14
    return img


def stereo_rgb_panel(left, right, size: tuple[int, int], title: str, subtitle: str | None = None) -> Image.Image:
    """Two RGB eye renders side by side."""
    w, h = size
    img = Image.new("RGB", (w, h), PANEL_BG)
    draw = ImageDraw.Draw(img)
    top = _title(draw, w, title)
    gap = 6
    tile_w = (w - 3 * gap) // 2
    avail = h - top - 8 - (16 if subtitle else 0)
    for i, rgb in enumerate((left, right)):
        if rgb is None:
            continue
        pic = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8)))
        th = min(avail, int(tile_w * pic.size[1] / pic.size[0]))
        pic = pic.resize((tile_w, th), Image.BILINEAR)
        x0 = gap + i * (tile_w + gap)
        img.paste(pic, (x0, top + 4))
        draw.text((x0 + 3, top + 4), "L" if i == 0 else "R", font=load_font(12), fill=TEXT)
    if subtitle:
        draw.text((8, h - 15), subtitle, font=load_font(11), fill=DIM)
    return img
