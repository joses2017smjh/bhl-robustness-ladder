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

Colour convention: the FAILING / negative panel carries the red label and the
red outline, the succeeding panel the green label. `outline_right` marks the
right panel as the failure (the locomotion pairs below), `outline_left` the left
one (the Mission 7 pairs, whose baseline is on the left). Until 2026-09-23 the
Mission 7 launchers passed `outline_right=True` with the success on the right,
which coloured the success red; `pair_sidecar` now records which panel is red.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"
GREEN, RED = "0x1f7a4d", "0x9c2222"


def font_file() -> str:
    """The DejaVu path the clips were made with, else whatever fontconfig has.

    Compute nodes carry the DejaVu path; the login node does not, and drawtext
    aborts on a missing fontfile.
    """
    if Path(FONT).is_file():
        return FONT
    try:
        alt = subprocess.run(["fc-match", "-f", "%{file}", "sans:bold"],
                             capture_output=True, text=True, timeout=30).stdout.strip()
    except Exception:                                            # noqa: BLE001
        alt = ""
    return alt if alt and Path(alt).is_file() else FONT


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
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
    outline_left: bool = False,
    speed: float = 1.0,
    fps: int = FPS,
    outline_from_s: float | None = None,
    max_colors: int = 128,
):
    """Composite two clips side by side into one looping GIF.

    outline_right / outline_left: which panel is the failure. That panel gets
    the red label and a red outline from `outline_from_s` (source-clip seconds;
    default 0.4 s before its end, where the evaluator holds the fall) onward.
    speed > 1 plays both sources faster (setpts), so `max_s` is GIF seconds.
    """
    if outline_right and outline_left:
        raise ValueError("only one panel can be the failure")
    d_left, d_right = duration(left), duration(right)
    # Pad to the budget so a fall freezes on screen instead of the GIF
    # ending when the episode does.
    d = max_s
    # Evaluator holds 15 extra frames (~0.3 s) on the fall; outline from there.
    fall_t = None
    if outline_right or outline_left:
        d_fail = d_right if outline_right else d_left
        src_t = max(0.05, d_fail - 0.40) if outline_from_s is None else max(0.0, outline_from_s)
        fall_t = src_t / speed                    # drawbox runs after setpts
    font = font_file()

    def panel(idx: int, label: str, colour: str, outline_t: float | None) -> str:
        safe = label.replace(":", r"\:").replace("'", "")
        box = ""
        if outline_t is not None:
            box = (
                f"drawbox=x=0:y=0:w=iw:h=ih:c=0xdc2828:t=8:"
                f"enable='gte(t,{outline_t:.2f})',"
            )
        rate = "" if speed == 1.0 else f"setpts=PTS/{speed:g},"
        return (
            f"[{idx}:v]scale={W}:-2,{rate}{box}"
            f"tpad=stop_mode=clone:stop_duration={max_s},"
            f"trim=duration={d:.2f},setpts=PTS-STARTPTS,"
            f"drawtext=fontfile={font}:text='{safe}':x=(w-tw)/2:y=h-38:"
            f"fontsize=19:fontcolor=white:box=1:boxcolor={colour}@0.85:boxborderw=9"
            f"[p{idx}]"
        )

    filt = (
        panel(0, left_label, RED if outline_left else GREEN, fall_t if outline_left else None) + ";" +
        panel(1, right_label, RED if outline_right else GREEN, fall_t if outline_right else None) + ";" +
        "[p0][p1]hstack=inputs=2,fps=" + str(fps) + ",split[s0][s1];"
        f"[s0]palettegen=max_colors={max_colors}[pal];"
        "[s1][pal]paletteuse=dither=bayer:bayer_scale=3"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-y", "-loglevel", "error", "-i", str(left), "-i", str(right),
         "-filter_complex", filt, "-loop", "0", str(out)], check=True)
    mb = out.stat().st_size / 1e6
    print(f"  {out.name}  {mb:.1f} MB  ({d:.1f}s @ {speed:g}x, sources {d_left:.1f}/{d_right:.1f}"
          f"{'' if fall_t is None else f', outline @ {fall_t:.1f}s gif time'})")
    return mb


def pair_sidecar(
    gif: Path,
    left_mp4: Path,
    right_mp4: Path,
    repo: Path,
    *,
    left_label: str,
    right_label: str,
    red_panel: str,
    playback_speed: float,
    gif_budget_s: float,
    camera: str,
    carry_from: Path | None = None,
    extra: dict | None = None,
) -> dict:
    """Write docs/gifs/<name>.json for a pair GIF (the shape the Mission 7 sidecars use).

    Reads each MP4's own sidecar (written by the render scripts) for outcome,
    configuration/controller and evidence hashes. `carry_from` is a previous
    sidecar whose caption / labels / scope / evidence strings are carried over
    verbatim (they hold the scientific caveats) and whose GIF hash is recorded
    as superseded.
    """
    repo = Path(repo).resolve()

    def rel(path: Path) -> str:
        path = Path(path).resolve()
        return str(path.relative_to(repo)) if path.is_relative_to(repo) else str(path)

    sides = {}
    for key, mp4, label in (("left", left_mp4, left_label), ("right", right_mp4, right_label)):
        meta = json.loads(Path(mp4).with_suffix(".json").read_text())
        side = {"label": label, "outcome": meta["outcome_of_this_run"], "camera": meta.get("camera"),
                "robot_visibility": meta.get("robot_visibility")}
        for k in ("controller", "reset_sequence", "configuration", "re_encoded"):
            if k in meta:
                side[k] = meta[k]
        sides[key] = (meta, side)
    (lm, left), (rm, right) = sides["left"], sides["right"]
    prev = json.loads(carry_from.read_text()) if carry_from and Path(carry_from).is_file() else {}
    record = {
        "output": rel(gif), "output_sha256": sha256(gif),
        "source_clip": [rel(m) for m in (left_mp4, right_mp4)],
        "source_sha256": [sha256(left_mp4), sha256(right_mp4)],
        "evidence": [lm.get("evidence"), rm.get("evidence")],
        "evidence_sha256": [lm.get("evidence_sha256"), rm.get("evidence_sha256")],
        "left": left, "right": right,
        "expected_success": {"left": red_panel != "left", "right": red_panel != "right"},
        "completion_s": {k: (v["outcome"]["elapsed_s"] if v["outcome"]["success"] else None) for k, v in (("left", left), ("right", right))},
        "red_outline_panel": red_panel, "green_panel": "right" if red_panel == "left" else "left",
        "playback_speed": playback_speed, "gif_budget_s": gif_budget_s, "camera": camera,
    }
    if lm.get("reproduces_evaluated_episode") is not None:
        record["reproduces_evaluated_episode"] = bool(lm.get("reproduces_evaluated_episode") and rm.get("reproduces_evaluated_episode"))
    for k in ("caption", "labels", "scope"):
        if k in prev:
            record[k] = prev[k]
    if prev:
        legacy = "red_outline_panel" not in prev        # pre-2026-09-23 sidecar: tracking camera, outline_right=True
        record["supersedes"] = {"output_sha256": prev.get("output_sha256"), "source_sha256": prev.get("source_sha256"),
                               "camera": prev.get("camera", "tracking"),
                               "red_outline_panel": prev.get("red_outline_panel", "right" if legacy else None),
                               "note": ("previous GIF: tracking camera behind the maze walls, success panel outlined red" if legacy
                                        else f"previous GIF: {prev.get('camera')} camera, red outline on the {prev.get('red_outline_panel')} panel")}
    if extra:
        record.update(extra)
    record["git_commit"] = subprocess.check_output(["git", "-C", str(repo), "rev-parse", "HEAD"], text=True).strip()
    Path(gif).with_suffix(".json").write_text(json.dumps(record, indent=2) + "\n")
    return record


def ffmpeg_visibility(mp4: Path, *, sample_fps: int = 5, crop: float = 0.4, size=(160, 90)) -> dict:
    """Per-second luminance statistics of a clip, decoded through ffmpeg.

    The robot sits at the tracked lookat (frame centre), so the central crop
    is the robot region: `central_std` is its spatial contrast and
    `central_diff` the mean absolute frame-to-frame change. Wall-only frames
    from the old tracking camera scored central_std 3.5-9 and central_diff
    0-0.3; this is a printed cross-check for the submitter, while the exact
    verdict comes from the recorder's segmentation probe in the MP4 sidecar.
    """
    import numpy as np
    w, h = size
    raw = subprocess.run(
        ["ffmpeg", "-v", "error", "-i", str(mp4), "-vf", f"fps={sample_fps},scale={w}:{h}",
         "-f", "rawvideo", "-pix_fmt", "gray", "-"], capture_output=True, check=True).stdout
    frames = np.frombuffer(raw, np.uint8).reshape(-1, h, w).astype(np.float32)
    lo, hi = (1 - crop) / 2, (1 + crop) / 2
    centre = frames[:, int(h * lo):int(h * hi), int(w * lo):int(w * hi)]
    rows = []
    for s in range(0, len(frames), sample_fps):
        blk, cb = frames[s:s + sample_fps], centre[s:s + sample_fps]
        rows.append({"t_s": s // sample_fps, "mean_luminance": round(float(blk.mean()), 2),
                     "central_std": round(float(cb.std(axis=(1, 2)).mean()), 2),
                     "central_diff": round(float(np.abs(np.diff(cb, axis=0)).mean()) if len(cb) > 1 else 0.0, 3)})
    return {"clip": str(mp4), "per_second": rows,
            "median_central_std": float(np.median([r["central_std"] for r in rows])),
            "median_central_diff": float(np.median([r["central_diff"] for r in rows])),
            "old_tracking_reference": {"central_std": "3.5-9", "central_diff": "0-0.3"}}


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
         # Corrected 2026-09-23: this is the one command of six where the
         # ordering favours push training; over n=60 the push-trained policy
         # falls MORE (0.15) than the DR-only control (0.10).
         "push-trained: 0.15 falls (n=60)", "DR only: 0.10 falls (n=60)",
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
