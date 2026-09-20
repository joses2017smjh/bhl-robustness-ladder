#!/usr/bin/env python3
"""Make labelled GIFs from measured simulator recordings, never synthetic footage.

Requires ffmpeg with drawtext and fontconfig. Run from any directory; existing
outputs are refused. The sidecar records sources, scores and SHA-256 hashes.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import subprocess

ROOT = Path(__file__).resolve().parents[1]
CASES = (
    ("inspection-maze", "weekend-inspection", True,
     "COMPLETED - two ordered inspections and exit",
     "MuJoCo | learned gait + oracle waypoints + sensor brake | 1x", 1.0),
    ("inspection-maze-failure", "weekend-inspection-failure", False,
     "REJECTED - wrong branch enters the dead end",
     "Negative control | intentional route error, not a policy crash | 1x", 1.0),
    ("team3-airlock", "weekend-team3", True,
     "COMPLETED - three robots inspect, wait, cross and rendezvous",
     "Shared MuJoCo world | learned gait + oracle team supervisor | 1.1x", 1.1),
)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    font = subprocess.check_output(["fc-match", "-f", "%{file}", "sans"], text=True).strip()
    if not Path(font).is_file():
        raise FileNotFoundError("A fontconfig-resolved sans font is required")
    campaign = ROOT / "results/weekend-20260919"
    outputs = ROOT / "docs/gifs"
    outputs.mkdir(parents=True, exist_ok=True)
    for source, name, expected_success, title, scope, speed in CASES:
        clip = campaign / f"{source}.mp4"
        evidence = campaign / ("team3-video.json" if source == "team3-airlock" else f"{source}-video.json")
        row = json.loads(evidence.read_text())["episodes"][0]
        if row["success"] is not expected_success:
            raise ValueError(f"Recording does not match caption: {evidence}")
        out = outputs / f"{name}.gif"
        sidecar = outputs / f"{name}.json"
        if out.exists() or sidecar.exists():
            raise FileExistsError(f"Refusing to overwrite {out} or its evidence")
        # Native footage, resized and captioned. No interpolation or relighting.
        vf = (f"setpts=PTS/{speed},fps=10,scale=720:-2:flags=lanczos,"
              "pad=iw:ih+58:0:0:color=0x12171d,"
              f"drawtext=fontfile='{font}':text='{title}':x=12:y=h-49:fontsize=17:fontcolor=white,"
              f"drawtext=fontfile='{font}':text='{scope}':x=12:y=h-24:fontsize=12:fontcolor=0xbac8d6,"
              "split[a][b];[a]palettegen=max_colors=96[p];[b][p]paletteuse=dither=bayer")
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-n", "-threads", "2", "-i", str(clip),
                        "-filter_complex_threads", "1", "-filter_complex", vf, "-loop", "0", str(out)], check=True)
        record = {"source_clip": str(clip.relative_to(ROOT)), "source_sha256": sha(clip),
                  "evidence": str(evidence.relative_to(ROOT)), "evidence_sha256": sha(evidence),
                  "output": str(out.relative_to(ROOT)), "output_sha256": sha(out),
                  "expected_success": expected_success, "failure": row.get("failure"),
                  "completion_s": row.get("completion_s"), "playback_speed": speed,
                  "caption": title, "scope": scope}
        sidecar.write_text(json.dumps(record, indent=2) + "\n")
        print(f"{out.relative_to(ROOT)}: {out.stat().st_size:,} bytes")


if __name__ == "__main__":
    main()
