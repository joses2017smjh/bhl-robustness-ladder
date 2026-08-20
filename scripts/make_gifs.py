"""Build side-by-side success/failure GIFs for the README.

GitHub markdown will not play an MP4 committed to the repo, but it renders GIFs
inline, so the README's clips have to be GIFs.

Each pair is composited into ONE file rather than two images in a table: two
separate GIFs drift out of sync on every loop, and the whole point of the pair
is that both robots are seeing identical conditions at the same instant.

The shorter clip (always the failure, since the episode ends when the robot
falls) is extended by freezing its last frame, so the fall stays on screen
instead of the pair collapsing to the length of the failure. A red outline is
drawn from the fall onward — the same marker the evaluator burns into the MP4,
thickened after downscale so it still reads at GIF size.
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
W = 430           # per-panel width; 2*W + divider stays under GitHub's column
FPS = 10
MAX_S = 9.0       # biped pairs; keeps each GIF a few MB rather than tens
ARMS_S = 12.0     # 22-DoF: full 10 s episode plus a hold on the fallen frame


def duration(path: Path) -> float:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True).stdout
    return float(json.loads(out)["format"]["duration"])


def pair_gif(
    left: Path,
    right: Path,
    out: Path,
    left_label: str,
    right_label: str,
    max_s: float = MAX_S,
    outline_right: bool = False,
):
    d_left, d_right = duration(left), duration(right)
    # Pad to the budget so a fall freezes on screen instead of the GIF
    # ending when the episode does.
    d = max_s
    # Evaluator holds 15 extra frames (~0.3 s) on the fall; outline from there.
    fall_t = max(0.05, d_right - 0.40) if outline_right else None

    def panel(idx: int, label: str, colour: str, outline_t: float | None) -> str:
        safe = label.replace(":", r"\:").replace("'", "")
        box = ""
        if outline_t is not None:
            box = (
                f"drawbox=x=0:y=0:w=iw:h=ih:c=0xdc2828:t=8:"
                f"enable='gte(t,{outline_t:.2f})',"
            )
        return (
            f"[{idx}:v]scale={W}:-2,{box}"
            f"tpad=stop_mode=clone:stop_duration={max_s},"
            f"trim=duration={d:.2f},setpts=PTS-STARTPTS,"
            f"drawtext=fontfile={FONT}:text='{safe}':x=(w-tw)/2:y=h-38:"
            f"fontsize=19:fontcolor=white:box=1:boxcolor={colour}@0.85:boxborderw=9"
            f"[p{idx}]"
        )

    filt = (
        panel(0, left_label, "0x1f7a4d", None) + ";" +
        panel(1, right_label, "0x9c2222" if outline_right else "0x1f7a4d", fall_t) + ";" +
        "[p0][p1]hstack=inputs=2,fps=" + str(FPS) + ",split[s0][s1];"
        "[s0]palettegen=max_colors=128[pal];"
        "[s1][pal]paletteuse=dither=bayer:bayer_scale=3"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
         "-filter_complex", filt, "-loop", "0", str(out)], check=True)
    mb = out.stat().st_size / 1e6
    print(f"  {out.name}  {mb:.1f} MB  ({d:.1f}s, sources {d_left:.1f}/{d_right:.1f}"
          f"{'' if fall_t is None else f', outline @ {fall_t:.1f}s'})")
    return mb


def existing_clip(path: Path, *, allow_swap: bool = True) -> Path | None:
    """Rollouts name the file OK or FELL after the fact."""
    if path.exists():
        return path
    if not allow_swap:
        return None
    name = path.name
    swapped = re.sub(r"__OK\.mp4$", "__FELL.mp4", name)
    if swapped == name:
        swapped = re.sub(r"__FELL\.mp4$", "__OK.mp4", name)
    alt = path.with_name(swapped)
    return alt if alt.exists() else None


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--only", choices=["biped", "arms"], default=None)
    args = p.parse_args(argv)

    repo = Path(__file__).resolve().parents[1]
    V, D, A, OUT = repo / "results/video", repo / "results/demo", repo / "results/demo_arms", repo / "docs/gifs"

    # right_fell: do not silently substitute an OK clip — that is how the
    # 22-DoF pairs lost the freeze and the red outline.
    biped = [
        (V / "dr-default-s0__vx+0.0_vy+0.2_wz+0.0__OK.mp4",
         V / "dr-off-s0__vx+0.0_vy+0.2_wz+0.0__FELL.mp4",
         OUT / "dr_pair.gif",
         "s=1.0 randomized  -  WALKS", "s=0 no randomization  -  FALLS",
         MAX_S, True),
        (D / "push-adaptive__vx+0.3_vy+0.0_wz+0.0__OK.mp4",
         D / "nopush-baseline__vx+0.3_vy+0.0_wz+0.0__FELL.mp4",
         OUT / "push_pair.gif",
         "trained on pushes  -  RECOVERS", "no push training  -  FALLS",
         MAX_S, True),
        (D / "terrain-trained__vx+0.3_vy+0.0_wz+0.0_d0.80__OK.mp4",
         D / "flatDR-baseline__vx+0.3_vy+0.0_wz+0.0_d0.80__FELL.mp4",
         OUT / "terrain_pair.gif",
         "trained on terrain  -  WALKS", "flat-trained  -  FALLS",
         MAX_S, True),
    ]
    # 22-DoF: use the command where the weaker policy actually fell. The biped
    # commands (flat strafe, 0.3 m/s forward, terrain forward) do not knock
    # these policies over, so swapping FELL→OK produced two walking panels.
    # Flat s=0 never falls in 10 s (n=60); that pair stays a walk vs walk.
    arms = [
        (A / "dr" / "arms-dr1.0-s0__vx+0.0_vy+0.2_wz+0.0__OK.mp4",
         A / "dr" / "arms-dr0.0-s0__vx+0.0_vy+0.2_wz+0.0__OK.mp4",
         OUT / "arms_dr_pair.gif",
         "22-DoF  s=1.0  -  WALKS", "22-DoF  s=0  -  WALKS",
         ARMS_S, False),
        (A / "push" / "arms-push-s0__vx+0.3_vy+0.0_wz+0.5__OK.mp4",
         A / "push" / "arms-dr1.0-s0__vx+0.3_vy+0.0_wz+0.5__FELL.mp4",
         OUT / "arms_push_pair.gif",
         "22-DoF  push-trained  -  RECOVERS", "22-DoF  no push training  -  FALLS",
         ARMS_S, True),
        (A / "terrain" / "arms-terrain-s0__vx+0.0_vy+0.2_wz+0.0_d0.80__OK.mp4",
         A / "terrain" / "arms-dr1.0-s0__vx+0.0_vy+0.2_wz+0.0_d0.80__FELL.mp4",
         OUT / "arms_terrain_pair.gif",
         "22-DoF  terrain-trained  -  WALKS", "22-DoF  flat-trained  -  FALLS",
         ARMS_S, True),
    ]

    if args.only == "biped":
        pairs = biped
    elif args.only == "arms":
        pairs = arms
    else:
        pairs = biped + arms

    total = 0.0
    for left, right, out, ll, rl, max_s, right_fell in pairs:
        left = existing_clip(left, allow_swap=not str(left).endswith("__FELL.mp4"))
        right = existing_clip(right, allow_swap=not right_fell)
        if left is None or right is None:
            print(f"  SKIP {out.name}: missing clip", file=sys.stderr)
            continue
        if right_fell and "__FELL" not in right.name:
            print(f"  SKIP {out.name}: right panel did not fall ({right.name})",
                  file=sys.stderr)
            continue
        total += pair_gif(left, right, out, ll, rl, max_s=max_s,
                          outline_right=right_fell)
    print(f"total {total:.1f} MB")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
