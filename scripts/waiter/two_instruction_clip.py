#!/usr/bin/env python3
"""Side-by-side display clip of one layout under two instructions (display only, not a gate).

Made for the presentation (2026-10-15; SLURM_JOBS.md "Presentation clip"): two RecordingWaiterEnv recordings of the
same policy on the same dev layout, one with the layout's own instruction and one with the target colour and table
swapped (clip_env swap_target). From each recorded frame it takes the follow view, the head camera and the stage
list, and lays the two episodes out next to each other; the shorter one holds its last frame (with its result card).
The sidecar JSON carries both episode records and, if given, the trace counts over all recorded layouts.

usage: two_instruction_clip.py --left <mp4> --right <mp4> [--trace-left <json> --trace-right <json>]
                               --out docs/gifs/<name> [--speed 4]
"""
from __future__ import annotations

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(HERE / "dsrl")]

from bhl_robust.eval.panels import BG, DIM, TEXT, load_font, write_gif  # noqa: E402

VIEW = (560, 420)               # clip_env.RecordingWaiterEnv defaults (view, SIDE_W, BAR_H)
SIDE_W, BAR_H = 250, 32
HS = SIDE_W - 16                # head camera size in the recorded frame
STAGE_ROWS, ROW_H = 6, 16
PANEL_W, VIEW_H, INSET = 410, 308, 104
TITLE_H, INSTR_H, GAP, BANNER_H, FOOT_H = 24, 58, 10, 34, 42
GOOD, BAD = (110, 210, 120), (235, 110, 100)


def crops(frame: np.ndarray, instruction: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(follow view, head camera, stage list) from one RecordingWaiterEnv frame."""
    w, h = VIEW
    view = frame[BAR_H:BAR_H + h, :w]
    head = frame[BAR_H + 22:BAR_H + 22 + HS, w + 8:w + 8 + HS]
    lines = len(textwrap.wrap(f"“{instruction}”", 30)[:4])
    y0 = BAR_H + 22 + HS + 8 + ROW_H * lines + 6
    stages = frame[y0:y0 + STAGE_ROWS * ROW_H, w + 8:w + 8 + SIDE_W - 16]
    return view, head, stages


def panel(frame: np.ndarray, rec: dict, title: str) -> Image.Image:
    view, head, stages = crops(frame, rec["instruction"])
    img = Image.new("RGB", (PANEL_W, TITLE_H + INSTR_H + VIEW_H + STAGE_ROWS * ROW_H + 6), BG)
    d = ImageDraw.Draw(img)
    d.text((4, 4), title, font=load_font(14), fill=DIM)
    y = TITLE_H
    for line in textwrap.wrap(f"“{rec['instruction']}”", 48)[:3]:
        d.text((4, y), line, font=load_font(14), fill=TEXT)
        y += 18
    y = TITLE_H + INSTR_H
    img.paste(Image.fromarray(view).resize((PANEL_W, VIEW_H), Image.BILINEAR), (0, y))
    inset = Image.fromarray(head).resize((INSET, INSET), Image.BILINEAR)
    d.rectangle((PANEL_W - INSET - 6, y + 4, PANEL_W - 4, y + INSET + 6), outline=(235, 235, 235), width=2)
    img.paste(inset, (PANEL_W - INSET - 5, y + 5))
    d.rectangle((PANEL_W - INSET - 6, y + INSET + 6, PANEL_W - 4, y + INSET + 22), fill=(20, 22, 28))
    d.text((PANEL_W - INSET - 2, y + INSET + 8), "head camera", font=load_font(11), fill=(235, 235, 235))
    img.paste(Image.fromarray(stages), (4, y + VIEW_H + 4))
    return img


def frames_of(mp4: Path) -> list[np.ndarray]:
    import imageio.v2 as imageio
    r = imageio.get_reader(str(mp4))
    out = [np.asarray(f) for f in r]
    r.close()
    return out


def count(trace: dict | None) -> str | None:
    if not trace:
        return None
    eps = trace.get("episodes") or trace.get("rows") or []
    ok = sum(bool(e.get("success")) for e in eps)
    return f"{ok}/{len(eps)}"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--left", type=Path, required=True)
    ap.add_argument("--right", type=Path, required=True)
    ap.add_argument("--trace-left", type=Path)
    ap.add_argument("--trace-right", type=Path)
    ap.add_argument("--banner", default="Phase 3b SmolVLA: the same layout, two instructions")
    ap.add_argument("--footer", action="append", default=[])
    ap.add_argument("--speed", type=float, default=4.0)
    ap.add_argument("--out", type=Path, required=True, help="output stem: <out>.gif, <out>.mp4, <out>.json")
    a = ap.parse_args(argv)
    recs = [json.loads(p.with_suffix(".json").read_text()) for p in (a.left, a.right)]
    if recs[0]["layout"]["seed"] != recs[1]["layout"]["seed"]:
        raise SystemExit("REFUSED: the two recordings are on different layouts")
    if any(recs[0]["layout"][k] != recs[1]["layout"][k] for k in ("tumblers", "tray", "delivery")):
        raise SystemExit("REFUSED: the two recordings differ in the physical layout")
    traces = [json.loads(p.read_text()) if p else None for p in (a.trace_left, a.trace_right)]
    titles = ["A: the layout's own instruction", "B: colour and table both swapped"]
    fa, fb = frames_of(a.left), frames_of(a.right)
    n = max(len(fa), len(fb))
    W = 2 * PANEL_W + GAP
    probe = panel(fa[0], recs[0], titles[0])
    H = BANNER_H + probe.height + FOOT_H
    H += H % 2
    import imageio.v2 as imageio
    fps = 1.0 / 0.08                                          # waiter_env.STEP_DT: one recorded frame per env step
    a.out.parent.mkdir(parents=True, exist_ok=True)
    mp4 = a.out.with_suffix(".mp4")
    wr = imageio.get_writer(str(mp4), fps=fps, codec="libx264", quality=8, pixelformat="yuv420p", macro_block_size=1)
    foot = a.footer or ["Simulation | dev layout, display only, not a gate | stance lock while standing"]
    for k in range(n):
        img = Image.new("RGB", (W, H), BG)
        d = ImageDraw.Draw(img)
        d.text((8, 8), f"{a.banner} (dev layout {recs[0]['layout']['seed']})", font=load_font(16), fill=TEXT)
        for i, (fr, rec) in enumerate(((fa, recs[0]), (fb, recs[1]))):
            img.paste(panel(fr[min(k, len(fr) - 1)], rec, titles[i]), (i * (PANEL_W + GAP), BANNER_H))
        y = H - FOOT_H + 4
        for line in foot[:2]:
            d.text((8, y), line, font=load_font(12), fill=DIM)
            y += 17
        wr.append_data(np.asarray(img))
    wr.close()
    gif = write_gif(mp4, a.out.with_suffix(".gif"), fps=6, speed=a.speed, width=W, max_colors=64, stats_mode="full")
    record = {"display_only": True, "made": "scripts/waiter/two_instruction_clip.py", "frames": n, "fps": fps,
              "left": recs[0], "right": recs[1], "dev_counts": {"layout_instruction": count(traces[0]),
                                                                "swapped_instruction": count(traces[1])},
              "traces": [str(p) if p else None for p in (a.trace_left, a.trace_right)], "mp4": str(mp4), "gif": gif}
    a.out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"wrote {a.out.with_suffix('.gif')} ({gif['mb']} MB, within budget: {gif['within_budget']}); "
          f"counts {record['dev_counts']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
