"""Display-only renders from the robot's stereo rig: RGB and rendered depth per
eye at a useful resolution, and dense optical flow on the left eye.

None of this reaches the controller. The speed brake reads the 8x8 ray-cast
depth pair in ``team_sensors`` (and the map is built from the lidar); these
renders sit at the same mount, baseline, 20 deg down-pitch and 60.5 deg
vertical field of view so the viewer sees what that rig would see.
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

from bhl_robust.eval.team_sensors import DEPTH_RANGE, STEREO_BASELINE

RIG_PITCH_DEG = 20.0      # team_sensors ray pattern: 20 deg down
RIG_VFOV_DEG = 60.5       # team_sensors: vfov_deg


class RigCameras:
    """Left/right RGB + depth renders from a body-frame stereo mount."""

    def __init__(self, model, slot, stereo_center, width: int = 160, height: int = 120,
                 baseline: float = STEREO_BASELINE, pitch_deg: float = RIG_PITCH_DEG,
                 vfov_deg: float = RIG_VFOV_DEG):
        self.model, self.slot = model, slot
        self.center = np.asarray(stereo_center, dtype=float)
        self.baseline, self.pitch = baseline, math.radians(pitch_deg)
        self.vfov = vfov_deg
        self.rgb = mujoco.Renderer(model, height=height, width=width)
        self.depth = mujoco.Renderer(model, height=height, width=width)
        self.depth.enable_depth_rendering()
        self.cam = mujoco.MjvCamera()
        self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        # hide the robot's own body from its eyes, as the ray sensors do (geom group 5 is not drawn)
        self.opt = mujoco.MjvOption()
        self.own = np.array([(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "")
                             .startswith(slot.prefix) for g in range(model.ngeom)])

    def _place(self, d, side: int):
        R = d.xmat[self.slot.body_id].reshape(3, 3)
        p = d.xpos[self.slot.body_id]
        eye = p + R @ (self.center + np.array([0.0, side * self.baseline / 2, 0.0]))
        f = R @ np.array([math.cos(-self.pitch), 0.0, math.sin(-self.pitch)])
        f = f / np.linalg.norm(f)
        self.cam.lookat[:] = eye + 1.0 * f
        self.cam.distance = 1.0
        self.cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
        self.cam.elevation = float(np.degrees(np.arcsin(np.clip(f[2], -1.0, 1.0))))

    def render(self, d):
        """-> (rgb_l, rgb_r, depth_l, depth_r); depth in metres, clipped to DEPTH_RANGE."""
        vis = self.model.vis.global_
        fovy0 = float(vis.fovy)
        groups = self.model.geom_group.copy()
        out = []
        try:
            vis.fovy = self.vfov                                  # visual setting only; restored below
            self.model.geom_group[self.own] = 5
            self.opt.geomgroup[5] = 0
            for side in (+1, -1):                                 # left eye is +y
                self._place(d, side)
                self.rgb.update_scene(d, camera=self.cam, scene_option=self.opt)
                rgb = self.rgb.render().copy()
                self.depth.update_scene(d, camera=self.cam, scene_option=self.opt)
                dep = np.clip(self.depth.render().copy(), 0.0, DEPTH_RANGE)
                out.append((rgb, dep))
        finally:
            vis.fovy = fovy0
            self.model.geom_group[:] = groups
        (rl, dl), (rr, dr) = out
        return rl, rr, dl, dr

    def close(self):
        self.rgb.close()
        self.depth.close()


class FlowView:
    """Dense Farneback optical flow between consecutive left-eye frames, as an HSV image."""

    def __init__(self):
        import cv2                                                # noqa: F401  (fail early if missing)
        self.prev = None
        self.last_stats = None

    def update(self, rgb, dt: float):
        import cv2
        gray = cv2.cvtColor(np.ascontiguousarray(rgb[..., :3]), cv2.COLOR_RGB2GRAY)
        if self.prev is None or self.prev.shape != gray.shape:
            self.prev = gray
            return None
        flow = cv2.calcOpticalFlowFarneback(self.prev, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        self.prev = gray
        mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])
        hsv = np.zeros((*gray.shape, 3), np.uint8)
        hsv[..., 0] = (ang * 90 / np.pi).astype(np.uint8)          # direction -> hue
        hsv[..., 1] = 255
        scale = max(4.0, float(np.percentile(mag, 99)))
        hsv[..., 2] = np.clip(mag / scale * 255, 0, 255).astype(np.uint8)
        img = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        # mean horizontal flow: + = image moving right, i.e. the robot yawing left
        self.last_stats = {"median_px_per_s": float(np.median(mag)) / dt, "mean_dx_px_per_s": float(flow[..., 0].mean()) / dt}
        return img
