"""Build side-by-side success/failure GIFs for the README.

GitHub markdown will not play an MP4 committed to the repo, but it renders GIFs
inline, so the README's clips have to be GIFs.

Each pair is composited into ONE file rather than two images in a table: two
separate GIFs drift out of sync on every loop, and the whole point of the pair
is that both robots are seeing identical conditions at the same instant.

The shorter clip (always the failure, since the episode ends when the robot
falls) is extended by freezing its last frame, so the fall stays on screen
instead of the pair collapsing to the length of the failure.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W = 430           # per-panel width; 2*W + divider stays under GitHub's column
FPS = 10
MAX_S = 9.0       # keeps each GIF a few MB rather than tens


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def pair_gif(left: Path, right: Path, out: Path, left_label: str, right_label: str):
    d = min(MAX_S, max(duration(left), duration(right)))

    def panel(idx: int, label: str, colour: str) -> str:
        safe = label.replace(":", r"\:").replace("'", "")
        return (
            f"[{idx}:v]scale={W}:-2,"
            f"tpad=stop_mode=clone:stop_duration={MAX_S},"
            f"trim=duration={d:.2f},setpts=PTS-STARTPTS,"
            f"drawtext=fontfile={FONT}:text='{safe}':x=(w-tw)/2:y=h-38:"
            f"fontsize=19:fontcolor=white:box=1:boxcolor={colour}@0.85:boxborderw=9"
            f"[p{idx}]"
        )

    filt = (
        panel(0, left_label, "0x1f7a4d") + ";" +
        panel(1, right_label, "0x9c2222") + ";" +
        "[p0][p1]hstack=inputs=2,fps=" + str(FPS) + ",split[s0][s1];"
        "[s0]palettegen=max_colors=128[pal];"
        "[s1][pal]paletteuse=dither=bayer:bayer_scale=3"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
         "-filter_complex", filt, "-loop", "0", str(out)], check=True)
    mb = out.stat().st_size / 1e6
    print(f"  {out.name}  {mb:.1f} MB  ({d:.1f}s)")
    return mb


def existing_clip(path: Path) -> Path | None:
    """Rollouts name the file OK or FELL after the fact. Accept either."""
    if path.exists():
        return path
    name = path.name
    swapped = re.sub(r"__OK\.mp4$", "__FELL.mp4", name)
    if swapped == name:
        swapped = re.sub(r"__FELL\.mp4$", "__OK.mp4", name)
    alt = path.with_name(swapped)
    return alt if alt.exists() else None


if __name__ == "__main__":
    repo = Path(__file__).resolve().parents[1]
    V, D, A, OUT = repo / "results/video", repo / "results/demo", repo / "results/demo_arms", repo / "docs/gifs"

    pairs = [
        # Experiment 1 - domain randomization, flat sim2sim, same strafe command
        (V / "dr-default-s0__vx+0.0_vy+0.2_wz+0.0__OK.mp4",
         V / "dr-off-s0__vx+0.0_vy+0.2_wz+0.0__FELL.mp4",
         OUT / "dr_pair.gif",
         "s=1.0 randomized  -  WALKS", "s=0 no randomization  -  FALLS"),

        # Experiment 2 - push recovery, identical 0.5 m/s shoves
        (D / "push-adaptive__vx+0.3_vy+0.0_wz+0.0__OK.mp4",
         D / "nopush-baseline__vx+0.3_vy+0.0_wz+0.0__FELL.mp4",
         OUT / "push_pair.gif",
         "trained on pushes  -  RECOVERS", "no push training  -  FALLS"),

        # Experiment 3 - rough terrain at d=0.80
        (D / "terrain-trained__vx+0.3_vy+0.0_wz+0.0_d0.80__OK.mp4",
         D / "flatDR-baseline__vx+0.3_vy+0.0_wz+0.0_d0.80__FELL.mp4",
         OUT / "terrain_pair.gif",
         "trained on terrain  -  WALKS", "flat-trained  -  FALLS"),

        # 22-DoF counterparts, same three conditions, filmed by slurm/37_arms_gifs.
        (A / "dr" / "arms-dr1.0-s0__vx+0.0_vy+0.2_wz+0.0__OK.mp4",
         A / "dr" / "arms-dr0.0-s0__vx+0.0_vy+0.2_wz+0.0__FELL.mp4",
         OUT / "arms_dr_pair.gif",
         "22-DoF  s=1.0  -  WALKS", "22-DoF  s=0  -  FALLS"),
        (A / "push" / "arms-push-s0__vx+0.3_vy+0.0_wz+0.0__OK.mp4",
         A / "push" / "arms-dr1.0-s0__vx+0.3_vy+0.0_wz+0.0__FELL.mp4",
         OUT / "arms_push_pair.gif",
         "22-DoF  push-trained  -  RECOVERS", "22-DoF  no push training  -  FALLS"),
        (A / "terrain" / "arms-terrain-s0__vx+0.3_vy+0.0_wz+0.0_d0.80__OK.mp4",
         A / "terrain" / "arms-dr1.0-s0__vx+0.3_vy+0.0_wz+0.0_d0.80__FELL.mp4",
         OUT / "arms_terrain_pair.gif",
         "22-DoF  terrain-trained  -  WALKS", "22-DoF  flat-trained  -  FALLS"),
    ]

    total = 0.0
    for left, right, out, ll, rl in pairs:
        left, right = existing_clip(left), existing_clip(right)
        if left is None or right is None:
            print(f"  SKIP {out.name}: missing clip", file=sys.stderr)
            continue
        total += pair_gif(left, right, out, ll, rl)
    print(f"total {total:.1f} MB")
