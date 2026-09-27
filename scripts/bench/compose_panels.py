"""Compose the Isaac maze three-panel clip from what maze_record.py --panels
saved: overhead PNG frames + a .npz of per-step sensor arrays + the episode
JSON. Isaac-free (numpy + PIL + the repo's panels module), so the launcher can
run it outside the container, after an optional denoise pass on the frames.

  python scripts/bench/compose_panels.py --episode <stem>.json [--top-dir DIR] --out <stem>.panels.mp4
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

from maze_record import FrameSink                  # noqa: E402
from bhl_robust.eval import panels                 # noqa: E402

LIDAR_RANGE = 12.0
STEREO_RANGE = 6.0


def compose(episode: dict, top_dir: Path, out_mp4: Path, png_dir: Path, quality_note: str = "",
            crop_frac: float = 1.0) -> dict:
    """`crop_frac` keeps that central fraction of the overhead frame's height
    (the corridor is 0.9 m tall in a 2.6 m tall view)."""
    from PIL import Image
    npz = np.load(episode["panels"]["npz"])
    frames = sorted(top_dir.glob("frame_*.png"))
    # maze_record reads the sensors BEFORE env.step and saves the overhead frame AFTER it, so
    # top frame k and sensor row k + 1 show the same state. On the terminating step the env
    # auto-resets inside env.step, so that step's overhead frame shows the spawn pose: dropped.
    n_top = len(frames) - (1 if episode.get("terminated") else 0)
    n = min(n_top, int(npz["lidar_sector_policy_m"].shape[0]) - 1)
    if n <= 0:
        raise RuntimeError(f"nothing to compose: {len(frames)} top frames, {npz['lidar_sector_policy_m'].shape[0]} sensor rows")
    fps = float(episode["video_fps"])
    sink = FrameSink(out_mp4, png_dir, fps)
    arm = episode["task"].replace("Velocity-BHL-MazeRecovery-", "").replace("-v0", "")
    seed = episode["seed"]
    step_dt = float(episode["step_dt"])
    outcome = episode.get("outcome", "?")
    last = None
    for k in range(n):
        top = np.asarray(Image.open(frames[k]).convert("RGB"))
        if 0.0 < crop_frac < 1.0:
            hh = top.shape[0]
            keep = int(hh * crop_frac) // 2 * 2
            y0 = (hh - keep) // 2
            top = top[y0:y0 + keep]
        j = k + 1                                    # the sensor row of the state this frame shows
        raw = npz["stereo_raw_m"][j]                 # (2, H, W)
        pooled = npz["stereo_policy_m"][j]           # (2, h, w)
        if pooled.ndim == 2:                         # flat terms: make them square images
            side = int(round(pooled.shape[1] ** 0.5))
            pooled = pooled.reshape(2, side, side) if side * side == pooled.shape[1] else pooled[:, None, :]
        sectors = npz["lidar_sector_policy_m"][j]    # (36,)
        hits = npz["lidar_hits_body_xy"][j]          # (R, 2); a miss was recorded as the range sentinel
        hits = hits[np.hypot(hits[:, 0], hits[:, 1]) < LIDAR_RANGE - 0.1]
        # the two panels add up to the 720 px overhead view
        dh = min(300, top.shape[0] // 2)
        dp = panels.depth_pair_panel(raw, STEREO_RANGE, (320, dh), "stereo: ray depth 64x64, L / R (no RGB)",
                                     pooled=pooled, subtitle="policy input = the pooled pair (+ training noise)")
        # 3 m window: the corridor walls sit 0.45 m out and would be a few pixels at 6 m
        lp = panels.lidar_panel(sectors, LIDAR_RANGE, (320, max(200, top.shape[0] - dh)),
                                "lidar: 36 sector minima of 500 rays", window_m=3.0, rays_xy=hits,
                                subtitle="forward up | 3 m of 12 m range | raw hits behind")
        t = j * step_dt
        end_t = episode.get("sim_seconds")
        at = f" at {end_t:.2f} s" if isinstance(end_t, (int, float)) else ""
        status = (f"REACHED THE BUTTON{at}" if (k == n - 1 and outcome == "success")
                  else (outcome.upper().replace("_", " ") + at if k == n - 1 else "walking"))
        header = (f"Isaac Sim B5 maze, Full stage | policy {arm} (lidar + stereo) | seed {seed} | t = {t:5.2f} s"
                  f" | {status} | 1x")
        footer = ("policy consumes: 36 lidar sectors + 2 x pooled ray depth + IMU/proprioception, training noise on"
                  f" | heading from an oracle waypoint teacher | RTX {quality_note}".rstrip(" |"))
        frame = panels.compose_frame(top, [dp, lp], header, footer, side_w=320)
        sink.add(np.ascontiguousarray(frame))
        last = frame
    if last is not None:
        for _ in range(int(1.2 * fps)):
            sink.add(np.ascontiguousarray(last))
    return sink.close()


def recover_top_frames(composed_dir: Path, top_dir: Path, n: int, side_w: int = 320, bar_h: int = 34) -> int:
    """Crop the overhead view back out of the first `n` composed frames (lossless PNG, laid
    out by panels.compose_frame: header bar, overhead view at x = 0, panels at the right)."""
    from PIL import Image
    top_dir.mkdir(parents=True, exist_ok=True)
    src = sorted(composed_dir.glob("frame_*.png"))[:n]
    if len(src) < n:
        raise RuntimeError(f"{composed_dir}: {len(src)} composed frames, need {n}")
    for k, f in enumerate(src):
        a = np.asarray(Image.open(f).convert("RGB"))
        Image.fromarray(a[bar_h:a.shape[0] - bar_h, :a.shape[1] - side_w]).save(top_dir / f"frame_{k:04d}.png")
    return len(src)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--top-dir", type=Path, default=None, help="overhead PNG frames (default: the episode's)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--png-dir", type=Path, default=None)
    ap.add_argument("--quality-note", default="")
    ap.add_argument("--crop-frac", type=float, default=1.0)
    ap.add_argument("--top-from-composed", type=Path, default=None,
                    help="recover the overhead frames from an earlier composition's PNG frames (the node-local "
                         "top PNGs are deleted after a job); writes them to --top-dir, which is then required")
    args = ap.parse_args()
    ep = json.loads(args.episode.read_text())
    top_dir = args.top_dir or Path(ep["panels"]["top_png_dir"])
    png_dir = args.png_dir or (args.out.parent / f".{args.out.stem}-frames")
    if args.top_from_composed is not None:
        if args.top_dir is None or png_dir.resolve() == args.top_from_composed.resolve():
            ap.error("--top-from-composed needs its own --top-dir and a --png-dir that is not the source")
        recover_top_frames(args.top_from_composed, top_dir, int(ep["panels"]["top_frames"]))
    res = compose(ep, top_dir, args.out, png_dir, args.quality_note, args.crop_frac)
    print(json.dumps(res))
    return 0 if res.get("mp4") or res.get("frames") else 1


if __name__ == "__main__":
    sys.exit(main())
