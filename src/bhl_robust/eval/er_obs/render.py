"""ER-OBS-1 frames: fixed-camera RGB, segmentation and C1 (cube invisible) renders through the UNCHANGED harness's
frame_hook, with the ORACLE labels of the same snapshot.

Everything here only reads the simulation: the hook never writes to `runner.d` and never calls mj_forward (that
would rewrite qacc_warmstart and the rollout would diverge from the unchanged harness). `episode_identity` proves
it per episode: the rendered rollout and a read-only rollout of the same seed agree bitwise.

C1 uses a render-only deep copy of the compiled model whose cube geom has rgba alpha 0: MuJoCo's mjv_updateScene
then omits the geom, so the cube has no body, shadow, reflection or segmentation. Setting alpha 0 on the scene geom
after update_scene instead leaves the cube's shadow (measured 2026-10-03, render probe 1), so it is not used. Each
C1 frame passes `c1_ok`: the cube geom is absent from the C1 scene (exact), and the frame lies within the renderer's
own repeat noise of the main renderer's frame with the cube's scene geom moved 100 m away (<= 2 levels, <= 100 px;
on NVIDIA EGL two renders of an identical scene in one context differ by 1 level in up to ~20 px, job 21532252).
The renderer is node-dependent (NVIDIA hardware EGL, or Mesa llvmpipe where the NVIDIA device is not usable, e.g.
MIG slices) and recorded in the manifest.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.eval.er_obs import C1_NOISE_MAX_LEVELS, C1_NOISE_MAX_PX, VISIBLE_MIN_PIXELS
from bhl_robust.eval.er_obs import labels as L

#: Frozen camera (CLAUSES_AS_APPLIED): free camera in front of the pair, 20 deg to robot a's side.
CAMERA = {"type": "free (fixed)", "lookat": [0.0, 0.10, 0.30], "distance": 2.6, "azimuth": -110.0,
          "elevation": -30.0}
WIDTH, HEIGHT = 960, 540
#: Render flags set explicitly on EVERY render (mjvScene flags persist across update_scene).
RENDER_FLAGS = {"reflection": 0, "shadow": 1}
REMOVED_Z = -100.0


def make_camera(spec: dict = CAMERA) -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = spec["lookat"]
    cam.distance = float(spec["distance"])
    cam.azimuth = float(spec["azimuth"])
    cam.elevation = float(spec["elevation"])
    return cam


def png_bytes(arr: np.ndarray) -> bytes:
    """PNG with no ancillary metadata (PIL writes no text chunks unless asked); checked by `png_chunks`."""
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(np.ascontiguousarray(arr)).save(buf, format="PNG")
    return buf.getvalue()


def png_chunks(data: bytes) -> list[str]:
    """Chunk types of a PNG byte string (used to refuse any frame carrying text metadata)."""
    if data[:8] != b"\x89PNG\r\n\x1a\n":
        raise ValueError("not a PNG")
    out, i = [], 8
    while i + 8 <= len(data):
        n = int.from_bytes(data[i:i + 4], "big")
        out.append(data[i + 4:i + 8].decode("latin-1"))
        i += 12 + n
    return out


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def c1_ok(n_cube_geoms_c1: int, max_diff: int, n_diff_px: int) -> bool:
    """The C1 gates: the cube geom is absent from the C1 scene (exact) AND the C1 frame is within the renderer's
    repeat noise of the cube-removed render (max abs difference <= 2 levels, <= 100 differing pixels)."""
    return bool(int(n_cube_geoms_c1) == 0 and int(max_diff) <= C1_NOISE_MAX_LEVELS
                and int(n_diff_px) <= C1_NOISE_MAX_PX)


def _diff(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=2)
    return int(d.max()), int((d > 0).sum())


def gl_info() -> dict:
    """GL vendor/renderer of the current context (node-dependent: NVIDIA EGL or Mesa llvmpipe)."""
    try:
        from OpenGL import GL
        return {k: GL.glGetString(getattr(GL, k)).decode() for k in ("GL_VENDOR", "GL_RENDERER", "GL_VERSION")}
    except Exception as e:                                             # noqa: BLE001
        return {"error": type(e).__name__}


class FrameRenderer:
    """Main RGB, segmentation, C1 and the cube-removed reference of one snapshot."""

    def __init__(self, model, cube_geom: int, width: int = WIDTH, height: int = HEIGHT, camera: dict = CAMERA):
        self.cube_geom = int(cube_geom)
        self.width, self.height = int(width), int(height)
        self.inv_model = copy.deepcopy(model)                   # render-only; never stepped
        self.inv_model.geom_rgba[self.cube_geom, 3] = 0.0
        self.r = mujoco.Renderer(model, height=self.height, width=self.width)
        self.r_inv = mujoco.Renderer(self.inv_model, height=self.height, width=self.width)
        self.cam = make_camera(camera)
        self.camera = dict(camera)
        # warm-up renders on blank data, discarded (no first-render effect reaches a frame)
        for rr, mm in ((self.r, model), (self.r_inv, self.inv_model)):
            self._render(rr, mujoco.MjData(mm))
        self.gl = gl_info()

    def _cube_ids(self, scn) -> list[int]:
        geom = int(mujoco.mjtObj.mjOBJ_GEOM)
        return [i for i in range(scn.ngeom) if scn.geoms[i].objtype == geom and scn.geoms[i].objid == self.cube_geom]

    def _render(self, rr, data, *, seg: bool = False, remove_cube: bool = False):
        if seg:
            rr.enable_segmentation_rendering()
        try:
            rr.update_scene(data, camera=self.cam)
            rr.scene.flags[int(mujoco.mjtRndFlag.mjRND_REFLECTION)] = RENDER_FLAGS["reflection"]
            rr.scene.flags[int(mujoco.mjtRndFlag.mjRND_SHADOW)] = RENDER_FLAGS["shadow"]
            n_cube = len(self._cube_ids(rr.scene))
            if remove_cube:
                for i in self._cube_ids(rr.scene):
                    rr.scene.geoms[i].pos[:] = [0.0, 0.0, REMOVED_Z]
            img = np.asarray(rr.render()).copy()
        finally:
            if seg:
                rr.disable_segmentation_rendering()
        return img, n_cube

    def frame(self, data) -> dict:
        rgb, n_main = self._render(self.r, data)
        seg, _ = self._render(self.r, data, seg=True)
        c1, n_inv = self._render(self.r_inv, data)
        removed, _ = self._render(self.r, data, remove_cube=True)
        removed2, _ = self._render(self.r, data, remove_cube=True)          # the renderer's own repeat noise
        geom = int(mujoco.mjtObj.mjOBJ_GEOM)
        mask = (seg[..., 1] == geom) & (seg[..., 0] == self.cube_geom)
        mx, npx = _diff(c1, removed)
        rmx, rnpx = _diff(removed, removed2)
        return {"rgb": rgb, "c1": c1, "mask": mask,
                "c1_check": {"n_scene_cube_geoms_main": int(n_main), "n_scene_cube_geoms_c1": int(n_inv),
                             "diff_px_vs_cube_removed": npx, "max_diff_vs_cube_removed": mx,
                             "identical_to_cube_removed": bool(n_inv == 0 and mx == 0),
                             "removed_repeat_diff": {"max": rmx, "n_px": rnpx},
                             "ok": c1_ok(n_inv, mx, npx)}}

    def close(self):
        for rr in (self.r, self.r_inv):
            try:
                rr.close()
            except Exception:                                          # noqa: BLE001
                pass                       # MuJoCo's EGL teardown can raise a spurious EGLError (video.py)


def mask_summary(mask: np.ndarray) -> dict:
    """Cube pixel count, bbox and an interior pixel (the median pixel of the mask, by row then column)."""
    ys, xs = np.nonzero(mask)
    n = int(len(ys))
    if not n:
        return {"cube_pixels": 0, "cube_visible": False, "bbox_rc": None, "interior_rc": None}
    order = np.lexsort((xs, ys))
    mid = order[n // 2]
    return {"cube_pixels": n, "cube_visible": bool(n >= VISIBLE_MIN_PIXELS),
            "bbox_rc": [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())],
            "interior_rc": [int(ys[mid]), int(xs[mid])]}


class FrameHook:
    """frame_hook for `scripted_carry.run_place_episode`: one frame per simulated second.

    Also mirrors the harness's rest height exactly as `run_place_episode` sets it (the cube z at the top of the first
    policy step with t >= t_settle) and checks, on every later step, that the `lift` the harness passes equals
    (cube z at the top of that step) - (mirrored rest height) to 0.0, so C2 uses the harness's own rest height."""

    def __init__(self, renderer: FrameRenderer, model, pair, floor_geom: int, seed: int, out_dir: Path,
                 t_settle: float, steps_per_frame: int, write: bool = True):
        self.rd, self.model, self.pair, self.floor = renderer, model, pair, int(floor_geom)
        self.seed, self.out_dir = int(seed), Path(out_dir)
        self.t_settle, self.spf, self.write = float(t_settle), int(steps_per_frame), bool(write)
        self.prev_z = None
        self.rest_z = None
        self.rest_step = None
        self.lift_checked = 0
        self.lift_mismatch_max = 0.0
        self.frames: list[dict] = []
        self.qpos: dict[int, np.ndarray] = {}

    def __call__(self, *, step, t, runner, script, pairs, lift, **_):
        d = runner.d
        k_pair = pairs.index(self.pair) if self.pair in pairs else 0
        z_now = float(d.xpos[self.pair.cube_body][2])
        if self.rest_z is None and t >= self.t_settle - 1e-9:
            self.rest_z, self.rest_step = self.prev_z, int(step)
        if self.rest_z is not None and self.prev_z is not None:
            self.lift_checked += 1
            self.lift_mismatch_max = max(self.lift_mismatch_max,
                                         abs(float(lift[k_pair]) - (self.prev_z - self.rest_z)))
        self.prev_z = z_now
        if (step + 1) % self.spf:
            return
        k = (step + 1) // self.spf
        self.qpos[k] = d.qpos.copy()
        st = L.frame_state(self.model, d, runner, self.pair, self.floor)
        imgs = self.rd.frame(d)
        ms = mask_summary(imgs["mask"])
        fid = f"s{self.seed}_t{k:02d}"
        rec = {"id": fid, "seed": self.seed, "k": k, "step": int(step), "sim_time_s": round(float(d.time), 6),
               "harness_t_s": round(float(t), 6), "phase": script.phase(t),
               "labels": st["labels"], "seated_flat_strict_body_z": st["seated_flat_strict_body_z"],
               "body_z_disagrees": st["body_z_disagrees"], "state": st["state"],
               "cube_pixels": ms["cube_pixels"], "cube_visible": ms["cube_visible"], "mask_bbox_rc": ms["bbox_rc"],
               "mask_interior_rc": ms["interior_rc"], "c1_check": imgs["c1_check"],
               "_z": st["_z"], "_offset": st["_offset"]}
        files = {"rgb": imgs["rgb"], "c1": imgs["c1"], "cube_mask": imgs["mask"].astype(np.uint8) * 255}
        rec["files"], rec["sha256"] = {}, {}
        for name, arr in files.items():
            b = png_bytes(arr)
            rel = f"frames/s{self.seed}/t{k:02d}_{name}.png"
            if self.write:
                p = self.out_dir / rel
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(b)
            rec["files"][name], rec["sha256"][name] = rel, sha256_bytes(b)
            rec.setdefault("png_chunks", {})[name] = png_chunks(b)
        self.frames.append(rec)

    def finalize(self) -> list[dict]:
        """C2 predicates with the harness's rest height (known only after settling), applied to every frame."""
        out = []
        for rec in self.frames:
            rec = dict(rec)
            z, off = rec.pop("_z"), rec.pop("_offset")
            if self.rest_z is None:
                rec["c2"] = None
            else:
                dz = z - self.rest_z
                rec["c2"] = {"dz_from_rest_m": round(dz, 6), "lifted_centre": L.c2_lifted(dz),
                             "placed_no_orientation": L.c2_placed(dz, off)}
            out.append(rec)
        return out


class QposReader:
    """Read-only frame_hook: copies qpos at the same capture steps (the identity check's reference rollout)."""

    def __init__(self, steps_per_frame: int):
        self.spf = int(steps_per_frame)
        self.qpos: dict[int, np.ndarray] = {}

    def __call__(self, *, step, runner, **_):
        if (step + 1) % self.spf == 0:
            self.qpos[(step + 1) // self.spf] = runner.d.qpos.copy()


def episode_identity(ep_render: dict, ep_read: dict, q_render: dict, q_read: dict) -> dict:
    """The rendered rollout equals the read-only rollout: bitwise qpos at every capture step and the same returned
    episode dict (scores, series, trace)."""
    same_keys = sorted(q_render) == sorted(q_read)
    qpos_equal = same_keys and all(np.array_equal(q_render[k], q_read[k]) for k in q_render)
    ep_equal = json.dumps(ep_render, sort_keys=True) == json.dumps(ep_read, sort_keys=True)
    return {"capture_steps": len(q_render), "qpos_bitwise_equal": bool(qpos_equal),
            "episode_dict_equal": bool(ep_equal), "ok": bool(qpos_equal and ep_equal)}
