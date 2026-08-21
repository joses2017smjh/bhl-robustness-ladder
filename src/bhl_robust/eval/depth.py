"""Egocentric depth from MuJoCo's offscreen buffer, paired with the RGB view.

This is the *evaluation* half of the depth work. The training half lives in
`bhl_robust.tasks.depth_env_cfg` and ray-casts against the terrain mesh in warp,
because Isaac Sim 5.1's RTX renderer will not start on this cluster. MuJoCo has
no such problem: its offscreen renderer is plain OpenGL through EGL, the same
context that already produces every clip in this repo, and `mjr_readPixels`
hands back the depth buffer alongside the colour one.

That makes the two paths genuinely independent -- different renderer, different
projection code, different simulator -- which is the same argument the rest of
this project makes for scoring PhysX policies in MuJoCo. A depth map that agrees
across both is a depth map, not an artefact of one implementation.

The recorder deliberately implements the same `capture` / `close` interface as
`EpisodeRecorder`, so `run_episode` drives it without knowing the difference.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.eval.mjcf_assets import EGO_CAM_NAME
from bhl_robust.eval.video import FONT

# Depth is colourised over a fixed window so a colour means the same distance in
# every frame. The window is NOT the sensor's 0.05-6.0 m range: at a 20 deg
# down-pitch almost every pixel lands within 3 m, so stretching the ramp to 6 m
# spends most of its resolution on sky and flattens the ground into one blue.
NEAR_M, FAR_M = 0.25, 3.5


def _turbo(x: np.ndarray) -> np.ndarray:
    """Cheap Turbo-like colormap on x in [0, 1] -> uint8 RGB.

    A colormap, not greyscale: 8-bit greyscale spends its whole range on
    luminance, and the 10-30 cm band that actually decides a foot placement is
    then a handful of indistinguishable dark values. Hue separates it.
    """
    x = np.clip(x, 0.0, 1.0)
    r = np.clip(1.5 - np.abs(4.0 * x - 3.0), 0.0, 1.0)
    g = np.clip(1.5 - np.abs(4.0 * x - 2.0), 0.0, 1.0)
    b = np.clip(1.5 - np.abs(4.0 * x - 1.0), 0.0, 1.0)
    return (np.stack([r, g, b], axis=-1) * 255.0).astype(np.uint8)


class DepthPairRecorder:
    """Third-person RGB on the left, the robot's own depth image on the right."""

    def __init__(
        self,
        model: mujoco.MjModel,
        path: Path,
        fps: float,
        panel: int = 480,
        depth_res: int = 64,
        caption: str = "",
        track_body: str = "base",
    ):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.panel = panel
        self.depth_res = depth_res

        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAM_NAME)
        if cam_id < 0:
            raise RuntimeError(
                f"model has no camera {EGO_CAM_NAME!r}; call prepare_mjcf(..., ego_camera=True)"
            )
        self._ego_cam = cam_id

        # Two renderers rather than one toggled between colour and depth: MuJoCo
        # allocates its framebuffer at construction, and the depth view is
        # rendered at the sensor's real resolution (64x64) and upscaled, so the
        # clip shows the pixels the policy actually receives rather than a
        # 480x480 render pretending to be a depth sensor.
        self.rgb = mujoco.Renderer(model, height=panel, width=panel)
        self.dep = mujoco.Renderer(model, height=depth_res, width=depth_res)
        self.dep.enable_depth_rendering()

        self.chase = mujoco.MjvCamera()
        self.chase.distance = 2.2
        self.chase.elevation = -12.0
        self.chase.azimuth = 135.0
        self._track_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, track_body)

        width = panel * 2 + 8
        cmd = [
            "ffmpeg", "-y", "-loglevel", "error",
            "-f", "rawvideo", "-pix_fmt", "rgb24",
            "-s", f"{width}x{panel}", "-r", f"{fps:g}",
            "-i", "pipe:0",
        ]
        if caption and Path(FONT).is_file():
            safe = caption.replace(":", r"\:").replace("'", "")
            cmd += ["-vf", (
                f"drawtext=fontfile={FONT}:text='{safe}':x=24:y=24:"
                f"fontsize=22:fontcolor=white:box=1:boxcolor=black@0.55:boxborderw=10"
            )]
        cmd += ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-preset", "veryfast",
                "-crf", "23", str(self.path)]
        self._proc = subprocess.Popen(cmd, stdin=subprocess.PIPE,
                                      stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        self.n_frames = 0
        self.depth_stack: list[np.ndarray] = []

    def capture(self, data: mujoco.MjData, flash: str | None = None) -> None:
        if self._track_id >= 0:
            self.chase.lookat[:] = data.xpos[self._track_id]
        self.rgb.update_scene(data, camera=self.chase)
        left = self.rgb.render()

        self.dep.update_scene(data, camera=self._ego_cam)
        depth = self.dep.render()                       # float32 metres
        self.depth_stack.append(depth.copy())

        norm = (depth - NEAR_M) / (FAR_M - NEAR_M)
        right = _turbo(norm)
        # Nearest-neighbour upscale: bilinear would invent gradients across the
        # depth discontinuities that are the whole content of the image.
        k = self.panel // self.depth_res
        right = np.repeat(np.repeat(right, k, axis=0), k, axis=1)
        if right.shape[0] != self.panel:
            right = right[: self.panel, : self.panel]

        frame = np.zeros((self.panel, self.panel * 2 + 8, 3), dtype=np.uint8)
        frame[:, : self.panel] = left
        frame[:, self.panel + 8:] = right

        if flash:
            colour = {"push": (255, 190, 0), "fall": (220, 40, 40)}.get(flash)
            if colour is not None:
                b = 8
                frame[:b, :, :] = colour
                frame[-b:, :, :] = colour
                frame[:, :b, :] = colour
                frame[:, -b:, :] = colour

        try:
            self._proc.stdin.write(frame.tobytes())
            self.n_frames += 1
        except BrokenPipeError:
            pass

    def close(self) -> str | None:
        try:
            self._proc.stdin.close()
        except Exception:
            pass
        self._proc.wait()
        for r in (self.rgb, self.dep):
            try:
                r.close()
            except Exception:
                # MuJoCo's EGL teardown raises a spurious EGLError on this
                # driver; the frames are already encoded by this point.
                pass
        if self._proc.returncode != 0:
            return self._proc.stderr.read().decode(errors="replace")[-500:]
        return None
