"""Offscreen rendering of evaluation episodes to MP4.

Deliberately driven by the same rollout code that produces the metrics, so a
clip always shows the exact episode a results row describes. Upstream's
Isaac Sim playback path (`play.py`) cannot be used: rsl-rl 3.x changed
`env.get_observations()`'s arity and it crashes after loading the policy.

Frames are piped raw into ffmpeg rather than buffered, because a 10s episode at
25 fps and 960x540 is ~390MB in RAM and there is no reason to hold it.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import mujoco
import numpy as np

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


class EpisodeRecorder:
    """Renders a tracking camera view and streams it to an MP4."""

    def __init__(
        self,
        model: mujoco.MjModel,
        path: Path,
        fps: float,
        width: int = 960,
        height: int = 540,
        caption: str = "",
        track_body: str = "base",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height

        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = 2.2
        self.camera.elevation = -12.0
        self.camera.azimuth = 135.0
        self._track_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body)

        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{height}", "-r", f"{fps:g}",
            "-i", "pipe:0",
        ]
        if caption and Path(FONT).is_file():
            safe = caption.replace(":", r"\:").replace("'", "")
            cmd += ["-vf", (
                f"drawtext=fontfile={FONT}:text='{safe}':x=24:y=24:"
                f"fontsize=24:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10"
            )]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-crf", "23", str(self.path)]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.n_frames = 0

    def capture(self, data: mujoco.MjData, flash: str | None = None) -> None:
        """Render one frame. `flash` draws a coloured border: push or fall."""
        if self._track_id >= 0:
            self.camera.lookat[:] = data.xpos[self._track_id]
        self.renderer.update_scene(data, camera=self.camera)
        frame = self.renderer.render()

        if flash:
            # Per-frame markers are drawn into the array; ffmpeg's drawtext can
            # only burn in text that is fixed for the whole clip.
            colour = {"push": (255, 190, 0), "fall": (220, 40, 40)}.get(flash)
            if colour is not None:
                b = 10
                frame = frame.copy()
                frame[:b, :, :] = colour
                frame[-b:, :, :] = colour
                frame[:, :b, :] = colour
                frame[:, -b:, :] = colour

        try:
            self._proc.stdin.write(frame.astype(np.uint8).tobytes())
            self.n_frames += 1
        except BrokenPipeError:
            pass

    def close(self) -> str | None:
        """Finish encoding. Returns ffmpeg's stderr if it failed."""
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        self._proc.wait()
        try:
            self.renderer.close()
        except Exception:
            # MuJoCo's EGL context teardown raises a spurious EGLError on this
            # driver; the frames are already encoded by this point.
            pass
        if self._proc.returncode != 0:
            return self._proc.stderr.read().decode(errors="replace")[-500:]
        return None


def ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None
