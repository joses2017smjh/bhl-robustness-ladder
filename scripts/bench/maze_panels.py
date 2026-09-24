"""Three-panel clip of the MuJoCo inspection maze: top view + the robot's own
paired depth + its lidar scan (+ a robot-eye render).

Re-simulates one `inspection_maze.py` episode (same deploy config, seed, route,
sensor mode, speed) with a per-step recorder hook, so the clip is an episode of
the published harness, not a replay of the 25-step trace. The textured world
(`world_xml(textured=True)`) changes materials, lights and the skybox only.

Honesty of the panels (drawn into the footer): the gait is the frozen learned
PPO policy, the route is oracle waypoints, and the lidar / paired ray depth
feed ONLY the speed brake (`team_sensors.brake_command`); the depth is ray
depth, not RGB stereo matching. The mission's numbers are the JSON's.

Outputs (in --out-dir): <name>.mp4 (real time, 1/policy_dt fps), <name>.json
(episode result minus the trace + render settings + the README recording's
completion time for the reproduction check), and the GIF + sidecar under
docs/gifs/ when --gif is given. `--no-render` runs the whole pipeline with a
blank main view (no OpenGL), for a login-node check.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

import inspection_maze as im                                  # noqa: E402
from maze_record import FrameSink                             # noqa: E402
from bhl_robust.eval import panels                            # noqa: E402
from bhl_robust.eval.inspection_maze import world_xml         # noqa: E402
from bhl_robust.eval.multi_robot import _WORLDS, build_multi  # noqa: E402
from bhl_robust.eval.team_sensors import (DEPTH_RANGE, LIDAR_RANGE, STEREO_CENTER)  # noqa: E402

README_COMPLETION_S = 17.36     # results/weekend-20260919/inspection-maze-video.json, seed 0 ordered reactive


def parse_vec(raw, default):
    if not raw:
        return tuple(default)
    v = [float(x) for x in raw.replace(",", ":").split(":")]
    if len(v) != len(default):
        raise ValueError(f"expected {len(default)} numbers, got {raw!r}")
    return tuple(v)


def sha256(path: Path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PanelRecorder:
    """The frame hook: renders, reads the sensor packet the brake used, composes."""

    def __init__(self, args, model, slot, policy_dt: float, out_mp4: Path, png_dir: Path):
        import mujoco
        self.args, self.model, self.slot = args, model, slot
        self.render = not args.no_render
        self.w, self.h = args.width, args.height
        self.ew, self.eh = parse_vec(args.eye_size, (320, 180))
        self.ew, self.eh = int(self.ew), int(self.eh)
        self.side_w = 320
        self.fps = 1.0 / policy_dt
        self.sink = FrameSink(out_mp4, png_dir, self.fps)
        self.frames = 0
        self.last_frame = None
        if self.render:
            self.top = mujoco.Renderer(model, height=self.h, width=self.w)
            self.eye = mujoco.Renderer(model, height=self.eh, width=self.ew)
            self.cam = mujoco.MjvCamera()
            self.cam.lookat[:] = parse_vec(args.lookat, (2.4, 0.2, 0.0))
            self.cam.distance, self.cam.azimuth, self.cam.elevation = args.distance, args.azimuth, args.elevation
            self.eye_cam = mujoco.MjvCamera()
            self.eye_cam.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.pitch = np.deg2rad(20.0)          # the rig's down-pitch (team_sensors.ray_pattern)

    def _eye_view(self, d):
        R = d.xmat[self.slot.body_id].reshape(3, 3)
        p = d.xpos[self.slot.body_id]
        eye = p + R @ STEREO_CENTER
        f = R @ np.array([np.cos(-self.pitch), 0.0, np.sin(-self.pitch)])
        f = f / np.linalg.norm(f)
        self.eye_cam.lookat[:] = eye + 2.0 * f
        self.eye_cam.distance = 2.0
        self.eye_cam.azimuth = float(np.degrees(np.arctan2(f[1], f[0])))
        self.eye_cam.elevation = float(np.degrees(np.arcsin(np.clip(f[2], -1.0, 1.0))))
        self.eye.update_scene(d, camera=self.eye_cam)
        return self.eye.render()

    def __call__(self, *, step, now, runner, sensors, mission, xy, yaw, command, raw):
        if self.args.stride > 1 and step % self.args.stride:
            return
        if self.render:
            self.top.update_scene(runner.d, camera=self.cam)
            top_rgb = self.top.render()
            eye_rgb = self._eye_view(runner.d)
        else:
            top_rgb = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
            eye_rgb = None
        pkt = sensors.latest[0]
        stale = not (pkt and pkt.get("extero_fresh"))
        lidar = np.asarray(pkt["lidar_sector_m"]) if pkt else None
        depth = np.asarray(pkt["paired_idealized_depth_m"]) if pkt else None
        brake = pkt.get("brake") if pkt else None
        # the three panels add up to the main view's height (170 + 170 + rest)
        eye_panel = panels.image_panel(eye_rgb, (self.side_w, 170), "robot-eye view (render, not an input)")
        depth_panel = panels.depth_pair_panel(depth, DEPTH_RANGE, (self.side_w, 170),
                                              "stereo rig: paired ray depth 8x8 (no RGB)",
                                              stale=stale, subtitle="10 Hz packets; feeds the speed brake only")
        lidar_panel = panels.lidar_panel(lidar, LIDAR_RANGE, (self.side_w, max(160, self.h - 340)),
                                         "lidar: 36 sector minima (108 rays)", window_m=6.0, stale=stale, brake=brake,
                                         subtitle="forward is up; feeds the speed brake only")
        done = mission.completed_at is not None
        status = "COMPLETED" if done else mission.active_label.replace("_", " ")
        header = (f"MuJoCo inspection maze | seed {self.args.seed} | {self.args.route} route | sensor mode {self.args.sensor_mode}"
                  f" | t = {now:5.2f} s | {status} | stations {len(mission.station_times)}/2 | 1x")
        footer = ("gait: learned PPO, frozen | route: oracle waypoints | speed brake: lidar + paired ray depth + IMU"
                  " | textured world = visual only")
        frame = panels.compose_frame(top_rgb, [eye_panel, depth_panel, lidar_panel], header, footer, side_w=self.side_w)
        self.last_frame = frame
        self.sink.add(np.ascontiguousarray(frame))
        self.frames += 1

    def hold(self, seconds: float, banner: str):
        """Repeat the last frame with a banner (the mission ends between hooks)."""
        if self.last_frame is None:
            return
        from PIL import Image, ImageDraw
        img = Image.fromarray(self.last_frame.copy())
        draw = ImageDraw.Draw(img)
        font = panels.load_font(30)
        tw = draw.textlength(banner, font=font)
        x, y = (img.size[0] - self.side_w - tw) / 2, 60
        draw.rectangle((x - 14, y - 8, x + tw + 14, y + 40), fill=(20, 110, 60))
        draw.text((x, y), banner, font=font, fill=(255, 255, 255))
        arr = np.asarray(img)
        for _ in range(int(seconds * self.fps / max(1, self.args.stride))):
            self.sink.add(np.ascontiguousarray(arr))
            self.frames += 1

    def close(self):
        if self.render:
            self.top.close()
            self.eye.close()
        return self.sink.close()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--route", choices=("ordered", "wrong_branch"), default="ordered")
    ap.add_argument("--sensor-mode", choices=("record", "reactive", "reactive_dropout"), default="reactive")
    ap.add_argument("--seconds", type=float, default=60.0)
    ap.add_argument("--speed", type=float, default=0.4)
    ap.add_argument("--out-dir", type=Path, default=REPO / "results/repo-gpu-20260923/maze-panels")
    ap.add_argument("--name", default="inspection-maze-panels")
    ap.add_argument("--frames-root", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "maze-panels-frames"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--eye-size", default="320:180")
    ap.add_argument("--lookat", default="2.4:0.2:0.0")
    ap.add_argument("--distance", type=float, default=6.4)
    ap.add_argument("--azimuth", type=float, default=90.0)
    ap.add_argument("--elevation", type=float, default=-66.0)
    ap.add_argument("--stride", type=int, default=1, help="record every Nth policy step")
    ap.add_argument("--plain-world", action="store_true", help="use the untextured README world")
    ap.add_argument("--no-render", action="store_true", help="blank main view; no OpenGL needed (pipeline check)")
    ap.add_argument("--gif", type=Path, default=None, help="also write this GIF (+ .json sidecar)")
    ap.add_argument("--gif-speed", type=float, default=2.0)
    ap.add_argument("--gif-fps", type=int, default=8)
    ap.add_argument("--gif-width", type=int, default=860)
    args = ap.parse_args()

    import mujoco
    from omegaconf import OmegaConf
    base = im.build_parser().parse_args([
        "--deploy", str(args.deploy), "--upstream", str(args.upstream), "--cache-dir", str(args.cache_dir),
        "--output", str(args.out_dir / f"{args.name}.unused.json"), "--seeds", "1", "--seed-start", str(args.seed),
        "--seconds", str(args.seconds), "--speed", str(args.speed), "--sensor-modes", args.sensor_mode,
        "--routes", args.route])
    cfg = OmegaConf.load(args.deploy)
    if cfg.num_actions != 22 or cfg.num_observations != 75:
        raise SystemExit("requires the full 22-DoF/75-observation humanoid locomotion checkpoint")
    policy = im.CpuPolicy(cfg.policy_checkpoint_path)
    _WORLDS["inspection_maze"] = world_xml(textured=not args.plain_world)
    model, slots = build_multi(args.upstream, args.cache_dir, 1, ["inspector"], variant="humanoid", world="inspection_maze")

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_mp4 = args.out_dir / f"{args.name}.mp4"
    rec = PanelRecorder(args, model, slots[0], float(cfg.policy_dt), out_mp4, Path(args.frames_root) / args.name)
    t0 = time.time()
    row = im.episode(base, model, slots, cfg, policy, args.seed, args.sensor_mode, args.route, frame_hook=rec)
    banner = (f"COMPLETED  {row['completion_s']:.2f} s" if row["success"]
              else f"{(row['failure'] or 'incomplete').upper()}  {row['elapsed_s']:.2f} s")
    rec.hold(1.2, banner)
    sink = rec.close()
    result = {k: v for k, v in row.items() if k != "trace"}
    result.update({
        "name": args.name, "mp4": sink["mp4"], "frames": sink["frames"], "png_dir": sink["png_dir"],
        "writer_error": sink["writer_error"], "video_fps": rec.fps / max(1, args.stride), "playback_speed": 1.0,
        "render": {"textured_world": not args.plain_world, "no_render": args.no_render, "main": [args.width, args.height],
                   "camera": {"lookat": parse_vec(args.lookat, (2.4, 0.2, 0.0)), "distance": args.distance,
                              "azimuth": args.azimuth, "elevation": args.elevation},
                   "eye_view": {"size": [rec.ew, rec.eh], "mount": list(map(float, STEREO_CENTER)), "pitch_deg": -20.0}},
        "panels": {"lidar": "36 sector minima of 108 rays, 12 m range, brake input", "depth": "paired idealized ray depth 8x8, 6 m, brake input",
                   "eye": "render for the viewer; not an input"},
        "control": "oracle_map_waypoints+frozen_isaac_gait+sensor_brake",
        "readme_recording_completion_s": README_COMPLETION_S,
        "matches_readme_recording": bool(row["success"] and abs(float(row["completion_s"]) - README_COMPLETION_S) < 0.05),
        "deploy": str(args.deploy), "checkpoint": str(cfg.policy_checkpoint_path),
        "checkpoint_sha256": sha256(Path(cfg.policy_checkpoint_path)) if Path(cfg.policy_checkpoint_path).is_file() else None,
        "mujoco": mujoco.__version__, "wall_seconds": round(time.time() - t0, 1),
    })
    if sink["mp4"] is None and sink["frames"] > 0:
        # assemble from the PNGs (imageio writer unavailable on this node)
        import subprocess
        png = Path(sink["png_dir"])
        for enc, extra in (("libx264", ["-preset", "veryfast", "-crf", "20"]), ("mpeg4", ["-q:v", "3"])):
            r = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", str(round(result["video_fps"])),
                                "-i", str(png / "frame_%04d.png"), "-c:v", enc, *extra, "-pix_fmt", "yuv420p", str(out_mp4)])
            if r.returncode == 0 and out_mp4.is_file():
                result["mp4"] = str(out_mp4); result["mp4_assembled_with"] = enc; break
    if result.get("mp4"):
        result["mp4_sha256"] = sha256(Path(result["mp4"]))
    if args.gif and result.get("mp4"):
        g = panels.write_gif(Path(result["mp4"]), args.gif, fps=args.gif_fps, speed=args.gif_speed, width=args.gif_width)
        result["gif"] = g
        sidecar = {
            "output": os.path.relpath(args.gif, REPO), "output_sha256": sha256(args.gif), "output_mb": g["mb"],
            "source_clip": os.path.relpath(result["mp4"], REPO), "source_sha256": result["mp4_sha256"],
            "evidence": os.path.relpath(args.out_dir / f"{args.name}.json", REPO),
            "episode": {k: result[k] for k in ("seed", "route", "sensor_mode", "success", "failure", "completion_s",
                                                "stations_completed", "wall_contact_steps", "path_length_m")},
            "matches_readme_recording": result["matches_readme_recording"],
            "readme_recording_completion_s": README_COMPLETION_S,
            "playback_speed": args.gif_speed, "gif": g,
            "scope": "MuJoCo 3.3.5 | frozen learned gait + oracle waypoints + sensor speed brake | lidar and paired ray depth shown are the brake's inputs | textured world is visual only",
            "capture": "top view: MuJoCo offscreen renderer, free camera; robot-eye: free camera at the stereo mount; depth/lidar panels: the packet the brake consumed at that step",
            "checkpoint": result["checkpoint"], "checkpoint_sha256": result["checkpoint_sha256"],
        }
        args.gif.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    (args.out_dir / f"{args.name}.json").write_text(json.dumps(result, indent=2) + "\n")
    try:
        (args.out_dir / f"{args.name}.unused.json").unlink()
    except FileNotFoundError:
        pass
    print(json.dumps({k: result.get(k) for k in ("name", "success", "failure", "completion_s", "frames", "mp4",
                                                  "matches_readme_recording", "gif", "wall_seconds")}), flush=True)
    print(f"MAZE-PANELS RESULT: {'PASS' if result['success'] and result.get('mp4') else 'FAIL'}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
