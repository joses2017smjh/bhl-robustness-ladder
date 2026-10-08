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
import functools
import subprocess
from pathlib import Path

import mujoco
import numpy as np

FONT = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"


@functools.lru_cache(maxsize=1)
def _h264_encoder() -> list[str]:
    """Pick an H.264 encoder this ffmpeg build actually has.

    The clips in docs/gifs were made where `libx264` was present; this cluster's
    ffmpeg is built without it and fails with `Unknown encoder 'libx264'` after
    the whole episode has been simulated -- the render is thrown away at the
    write. Ask the binary instead of assuming, and fall back through the
    encoders that produce an equivalent file.
    """
    try:
        out = subprocess.run(["ffmpeg", "-hide_banner", "-encoders"],
                             capture_output=True, text=True, timeout=30).stdout
    except Exception:                                            # noqa: BLE001
        out = ""
    for enc, extra in (("libx264", ["-preset", "veryfast", "-crf", "23"]),
                       ("libopenh264", ["-b:v", "4M"]),
                       ("mpeg4", ["-q:v", "3"])):
        if f" {enc} " in out:
            return ["-c:v", enc, "-pix_fmt", "yuv420p", *extra]
    return ["-pix_fmt", "yuv420p"]        # let ffmpeg choose


# Camera presets: (distance m, elevation deg, azimuth deg). MuJoCo's free camera
# looks along the azimuth (azimuth 0 = along +x) and sits `distance` behind the
# lookat point, so a negative elevation places it above the robot.
#
#   tracking  the historical default: close, nearly level, fixed azimuth. In the
#             Mission 7 mazes (1.1 m walls, 1.5-1.7 m cells) this camera ends up
#             behind a wall and the clip shows grey wall.
#   overhead  steep and far enough that the sight line clears the walls AND
#             the raised door panels: the Mission 7 doors are 1.6 m wide
#             lintels spanning z = 1.45-2.55 m over the doorway, and at -65 deg
#             (camera 4.1 m up) the ray to a robot standing under one crosses
#             the panel band, hiding the robot for ~3.6 s of the doors crossing
#             (ray probe, 2026-09-23). At -80 deg the camera sits 4.4 m up and
#             every robot geom stays in unblocked view through all four
#             Mission 7 clips (min 14 % of geoms, 0 blank per-second probes);
#             at a wall 0.85 m away (half a cell) the sight line is 4.8 m up.
#   chase     behind the robot, looking along its heading; the azimuth follows
#             the base yaw through a low-pass filter so steps do not judder it.
CAMERA_PRESETS: dict[str, tuple[float, float, float]] = {
    "tracking": (2.2, -12.0, 135.0),
    "overhead": (4.5, -80.0, 90.0),
    "chase": (3.2, -25.0, 0.0),
}


def yaw_of_quat(q) -> float:
    """Yaw (rad) of a MuJoCo (w, x, y, z) quaternion."""
    w, x, y, z = (float(v) for v in q)
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def smooth_azimuth(previous: float | None, target: float, alpha: float) -> float:
    """Wrap-aware exponential smoothing of an azimuth in degrees."""
    if previous is None:
        return float(target)
    delta = (float(target) - previous + 180.0) % 360.0 - 180.0
    return (previous + alpha * delta) % 360.0


class EpisodeRecorder:
    """Renders a camera view that follows one body and streams it to an MP4."""

    def __init__(
        self,
        model: mujoco.MjModel,
        path: Path,
        fps: float,
        width: int = 960,
        height: int = 540,
        caption: str = "",
        track_body: str = "base",
        camera: str = "tracking",
        camera_distance: float | None = None,
        camera_elevation: float | None = None,
        camera_azimuth: float | None = None,
        visibility_every: int = 0,
        chase_smoothing: float = 0.15,
    ):
        """`camera` picks a preset from CAMERA_PRESETS (default unchanged);
        the camera_* arguments override single preset values. With
        `visibility_every` = N > 0 every N-th frame is also rendered as a
        segmentation image and the fraction of pixels that belong to the
        tracked body's kinematic tree is appended to `self.visibility` as
        (sim time s, fraction) -- an exact occlusion check for the sidecar."""
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.width, self.height = width, height
        if camera not in CAMERA_PRESETS:
            raise ValueError(f"unknown camera preset {camera!r}; choose from {sorted(CAMERA_PRESETS)}")
        self.camera_name = camera
        distance, elevation, azimuth = CAMERA_PRESETS[camera]

        self.renderer = mujoco.Renderer(model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.distance = distance if camera_distance is None else float(camera_distance)
        self.camera.elevation = elevation if camera_elevation is None else float(camera_elevation)
        self.camera.azimuth = azimuth if camera_azimuth is None else float(camera_azimuth)
        self._chase = camera == "chase"
        self._chase_alpha = float(chase_smoothing)
        self._chase_azimuth: float | None = None if camera_azimuth is None else float(camera_azimuth)
        self._track_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body)

        self.visibility_every = int(visibility_every)
        self.visibility: list[tuple[float, float]] = []
        self._robot_geoms = np.zeros(model.ngeom, dtype=bool)
        if self._track_id >= 0:
            root = model.body_rootid[self._track_id]
            self._robot_geoms = model.body_rootid[model.geom_bodyid] == root

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
        cmd += [*_h264_encoder(), str(self.path)]

        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.n_frames = 0

    def capture(self, data: mujoco.MjData, flash: str | None = None) -> None:
        """Render one frame. `flash` draws a coloured border: push or fall."""
        if self._track_id >= 0:
            self.camera.lookat[:] = data.xpos[self._track_id]
            if self._chase:
                heading = float(np.degrees(yaw_of_quat(data.xquat[self._track_id])))
                self._chase_azimuth = smooth_azimuth(self._chase_azimuth, heading, self._chase_alpha)
                self.camera.azimuth = self._chase_azimuth
        if self.visibility_every > 0 and self.n_frames % self.visibility_every == 0:
            self.visibility.append((float(data.time), self._robot_fraction(data)))
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

    def _robot_fraction(self, data: mujoco.MjData) -> float:
        """Fraction of the frame covered by the tracked robot (segmentation pass).

        The segmentation flag is switched off again before the RGB pass so it
        never leaks into a written frame.
        """
        try:
            self.renderer.enable_segmentation_rendering()
            self.renderer.update_scene(data, camera=self.camera)
            seg = self.renderer.render()
        finally:
            self.renderer.disable_segmentation_rendering()
        seg = np.asarray(seg)
        geom_ids = seg[..., 0]
        # channel 1 is the object type; sites are rendered too and carry site ids
        valid = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (geom_ids >= 0) & (geom_ids < len(self._robot_geoms))
        hit = np.zeros_like(valid)
        hit[valid] = self._robot_geoms[geom_ids[valid]]
        return float(hit.mean())

    def visibility_summary(self) -> dict | None:
        """Min / mean robot-pixel fraction and the share of probes above 0.1 %."""
        if not self.visibility:
            return None
        fractions = np.array([f for _, f in self.visibility])
        return {
            "camera": self.camera_name,
            "probes": int(len(fractions)),
            "probe_every_frames": self.visibility_every,
            "robot_pixel_fraction_min": float(fractions.min()),
            "robot_pixel_fraction_mean": float(fractions.mean()),
            "share_of_probes_with_robot_visible": float((fractions >= 1e-3).mean()),
            "visible_threshold_fraction": 1e-3,
        }

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
