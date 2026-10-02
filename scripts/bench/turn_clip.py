"""Side-by-side clip of two gaits given the same turn command (display only, not a gate).

Made for the turning ablation of 2026-10-01 (job 21506543): R1 puts the gait clock in the actor, R2 in the
critic only. Each side re-simulates turn_test.py's v2 turn runs on one reset seed: 3.0 s standing, then
(0, 0, +0.6) rad/s for 6 s; a fresh reset, then (0, 0, -0.6) for 6 s. `simulate` repeats run_command's
steps line for line (one robot, flat world, ContactRunner, CpuPolicy, make_controller,
rng = default_rng(seed)); the renderer only reads the state. The sidecar puts the yaw each run reaches here
next to the scored turn-test JSON's value and says whether they match within MATCH_TOL_DEG, so a clip that is
not the scored run says so.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

from bhl_robust.eval.panels import BG, DIM, ROBOT, TEXT, load_font, write_gif   # noqa: E402
from turn_test import V2_TURN_WARM_S, V2_TURN_WZ, yaw_of                         # noqa: E402

MATCH_TOL_DEG = 0.5
TURN_MIN_DEG = 150.0          # turn_test's default --turn-min-deg (drawn on the result card; nothing is gated here)
OK = (110, 210, 120)
BAD = (235, 95, 85)
HEAD_H, FOOT_H, BANNER_H = 58, 46, 34


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def scored_yaw(turn_json: dict, name: str):
    """The scored yaw (deg) of run `name` (e.g. 'turn+_s0') in a turn_test v2 JSON, or None."""
    for r in turn_json.get("turns", []):
        if r.get("name") == name:
            return r.get("yaw_deg")
    return None


def matches(yaw_deg: float, scored_deg, tol: float = MATCH_TOL_DEG) -> bool:
    return scored_deg is not None and abs(float(yaw_deg) - float(scored_deg)) <= tol


def card_for(yaw_deg: float, wz: float, fell) -> tuple[str, bool]:
    """Result card text and colour: green iff the run turned >= TURN_MIN_DEG in the commanded direction
    without falling (turn_test's v2 per-run criterion)."""
    if fell is not None:
        return f"FELL at {fell:.1f} s", False
    ok = yaw_deg * math.copysign(1.0, wz) >= TURN_MIN_DEG
    return f"turned {yaw_deg:+.0f}\N{DEGREE SIGN}  (scored test needs {TURN_MIN_DEG:.0f}\N{DEGREE SIGN})", ok


def simulate(deploy: Path, upstream: Path, cache: Path, variant: str, cmd, seconds: float, warm: float, seed: int,
             size: int, show_from: float, cam: tuple[float, float, float]):
    """turn_test.run_command plus a read-only free camera. Returns (result, frames): result has run_command's
    keys; frames are (rgb, t_s, commanded wz, yaw turned since reset in deg), one per policy step from
    `show_from` on."""
    import mujoco
    from omegaconf import OmegaConf
    from bhl_robust.eval.gait_clock import make_controller
    from bhl_robust.eval.multi_robot import build_multi
    from team_airlock import ContactRunner, CpuPolicy
    cfg = OmegaConf.load(deploy)
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    model, slots = build_multi(upstream, cache / variant, 1, ["t"], variant=variant, world="flat")
    ctrl = make_controller(cfg)
    ctrl.policy = policy
    runner = ContactRunner(model, slots, [cfg], [ctrl])
    rng = np.random.default_rng(seed)
    runner.reset(rng)
    slot = slots[0]
    owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "").startswith(slot.prefix)
                       else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners)
    dt = float(cfg.policy_dt)
    yaw0 = yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7])
    xy0 = runner.d.xpos[slot.body_id, :2].copy()

    renderer = mujoco.Renderer(model, height=size, width=size)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.elevation = cam[0], cam[1]
    camera.azimuth = math.degrees(yaw0) + 180.0 + cam[2]       # facing the robot's front, cam[2] deg off its axis
    camera.lookat[:] = [xy0[0], xy0[1], 0.42]

    fell, yaws = None, []
    frames = []
    for step in range(int((warm + seconds) / dt)):
        now = step * dt
        c = np.zeros(3) if now < warm else np.asarray(cmd, dtype=float)
        obs = runner.observe(0, c)
        runner.step([ctrl.update(obs)])
        yaws.append(yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]))
        if now + 1e-9 >= show_from:
            camera.lookat[:2] = runner.d.xpos[slot.body_id, :2]
            renderer.update_scene(runner.d, camera=camera)
            turned = math.degrees(float(np.unwrap(np.array(yaws))[-1] - yaw0))
            frames.append((renderer.render().copy(), now + dt, float(c[2]), turned))
        if runner.tilt(0) >= 0.78:
            fell = now
            break
    renderer.close()
    dyaw = float(np.unwrap(np.array(yaws))[-1] - yaw0)
    xy1 = runner.d.xpos[slot.body_id, :2]
    return ({"cmd": [float(v) for v in cmd], "fell_at_s": fell, "yaw_deg": round(math.degrees(dyaw), 1),
             "displacement_m": round(float(np.linalg.norm(xy1 - xy0)), 3), "seconds": seconds}, frames)


def _dial(draw: ImageDraw.ImageDraw, cx: int, cy: int, r: int, turned_deg: float) -> None:
    """Top-view heading dial: grey = heading at reset, orange = heading now (a left turn goes anticlockwise)."""
    draw.ellipse((cx - r, cy - r, cx + r, cy + r), outline=DIM, width=2, fill=(24, 27, 33))
    draw.line((cx, cy, cx, cy - r + 4), fill=(120, 124, 132), width=3)
    th = math.radians(turned_deg)
    draw.line((cx, cy, cx - (r - 4) * math.sin(th), cy - (r - 4) * math.cos(th)), fill=ROBOT, width=4)
    draw.ellipse((cx - 4, cy - 4, cx + 4, cy + 4), fill=ROBOT)


def side_panel(rgb: np.ndarray, title: str, subtitle: str, turned_deg: float, wz: float, card: str | None,
               card_ok: bool | None) -> Image.Image:
    """One side: title bar, the rendered view, a heading dial with the yaw turned since reset (or 'standing'
    before the command), and an optional result card."""
    view = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8)))
    w, h = view.size
    img = Image.new("RGB", (w, h + HEAD_H), BG)
    draw = ImageDraw.Draw(img)
    draw.text((10, 7), title, font=load_font(17), fill=TEXT)
    draw.text((10, 32), subtitle, font=load_font(13), fill=DIM)
    img.paste(view, (0, HEAD_H))
    r = 40
    cx, cy = w - r - 14, HEAD_H + r + 14
    _dial(draw, cx, cy, r, turned_deg if wz != 0.0 else 0.0)
    shown = f"{turned_deg:+.0f}\N{DEGREE SIGN}" if wz != 0.0 else "standing"
    font = load_font(18)
    draw.text((cx - draw.textlength(shown, font=font) / 2, cy + r + 6), shown, font=font, fill=TEXT)
    if card:
        font = load_font(18)
        tw = draw.textlength(card, font=font)
        x0, y0 = (w - tw) / 2 - 12, HEAD_H + h - 52
        draw.rectangle((x0, y0, x0 + tw + 24, y0 + 36), fill=(18, 20, 24), outline=OK if card_ok else BAD, width=3)
        draw.text((x0 + 12, y0 + 8), card, font=font, fill=OK if card_ok else BAD)
    return img


def compose(left: Image.Image, right: Image.Image, banner: str, footer: list[str]) -> np.ndarray:
    """Banner across the top, the two sides with a 6 px gutter, up to two footer lines; even sides (yuv420p)."""
    gutter = 6
    W = left.size[0] + gutter + right.size[0]
    H = BANNER_H + max(left.size[1], right.size[1]) + FOOT_H
    W += W % 2
    H += H % 2
    canvas = Image.new("RGB", (W, H), (12, 13, 16))
    draw = ImageDraw.Draw(canvas)
    draw.text((10, 8), banner, font=load_font(16), fill=TEXT)
    canvas.paste(left, (0, BANNER_H))
    canvas.paste(right, (left.size[0] + gutter, BANNER_H))
    for i, line in enumerate(footer[:2]):
        draw.text((10, H - FOOT_H + 5 + 19 * i), line, font=load_font(13), fill=DIM)
    return np.asarray(canvas)


def open_writer(mp4: Path, fps: int):
    """imageio's bundled ffmpeg has libx264; fall back to mpeg4 as maze_record.FrameSink does."""
    import imageio.v2 as imageio
    last = None
    for codec in ("libx264", "mpeg4"):
        try:
            return imageio.get_writer(str(mp4), fps=fps, codec=codec, quality=8, pixelformat="yuv420p",
                                      macro_block_size=1), codec
        except Exception as exc:                                 # noqa: BLE001
            last = exc
    raise RuntimeError(f"no mp4 writer: {last!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    for side in ("left", "right"):
        ap.add_argument(f"--{side}-deploy", type=Path, required=True)
        ap.add_argument(f"--{side}-scored", type=Path, required=True, help="that checkpoint's turn_test v2 JSON")
        ap.add_argument(f"--{side}-title", required=True)
        ap.add_argument(f"--{side}-subtitle", required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--variant", choices=("biped", "humanoid"), default="humanoid")
    ap.add_argument("--seed", type=int, default=0, help="reset seed (turn_test v2 scored seeds 0, 1, 2)")
    ap.add_argument("--seconds", type=float, default=6.0)
    ap.add_argument("--show-from", type=float, default=2.0, help="first sim second shown of each run")
    ap.add_argument("--hold-s", type=float, default=1.2, help="result card hold at the end of each run")
    ap.add_argument("--size", type=int, default=480)
    ap.add_argument("--cam", type=float, nargs=3, default=(2.1, -20.0, 30.0), metavar=("DIST", "ELEV", "OFF_AXIS"))
    ap.add_argument("--footer-line", action="append", default=[], help="up to two lines under the clip")
    ap.add_argument("--mp4", type=Path, required=True)
    ap.add_argument("--gif", type=Path, required=True)
    ap.add_argument("--gif-fps", type=int, default=10)
    ap.add_argument("--gif-width", type=int, default=880)
    args = ap.parse_args()

    from omegaconf import OmegaConf
    sidecar = args.gif.with_suffix(".json")
    for p in (args.gif, sidecar, args.mp4):
        if p.exists():
            print(f"REFUSING: {p} exists (never overwrite a published clip)")
            return 3
    sides = []
    for side in ("left", "right"):
        dep = getattr(args, f"{side}_deploy")
        cfg = OmegaConf.load(dep)
        sides.append({"side": side, "deploy": dep, "run_dir": dep.resolve().parent.parent.name,
                      "policy_dt": float(cfg.policy_dt), "onnx": Path(str(cfg.policy_checkpoint_path)),
                      "scored_path": getattr(args, f"{side}_scored"),
                      "scored": json.loads(getattr(args, f"{side}_scored").read_text()),
                      "title": getattr(args, f"{side}_title"), "subtitle": getattr(args, f"{side}_subtitle")})
    if sides[0]["policy_dt"] != sides[1]["policy_dt"]:
        print(f"REFUSING: different policy_dt {sides[0]['policy_dt']} vs {sides[1]['policy_dt']}")
        return 2
    fps = round(1.0 / sides[0]["policy_dt"])

    args.mp4.parent.mkdir(parents=True, exist_ok=True)
    writer, codec = open_writer(args.mp4, fps)
    runs, n_frames = [], 0
    for k, sign in enumerate((1.0, -1.0)):
        name = f"turn{'+' if sign > 0 else '-'}_s{args.seed}"
        wz = sign * V2_TURN_WZ
        direction = f"left (+{V2_TURN_WZ:g} rad/s)" if sign > 0 else f"right (-{V2_TURN_WZ:g} rad/s)"
        sims = []
        for s in sides:
            res, frames = simulate(s["deploy"], args.upstream, args.cache_dir, args.variant, (0.0, 0.0, wz),
                                   args.seconds, V2_TURN_WARM_S, args.seed, args.size, args.show_from, tuple(args.cam))
            sc = scored_yaw(s["scored"], name)
            runs.append({"side": s["side"], "run_dir": s["run_dir"], "name": name, **res, "scored_yaw_deg": sc,
                         "matches_scored": matches(res["yaw_deg"], sc)})
            print(f"  {s['side']:5s} {s['run_dir']}: {name} yaw {res['yaw_deg']} deg (scored {sc}), "
                  f"fell {res['fell_at_s']}, {len(frames)} frames", flush=True)
            sims.append((res, frames))
        n = max(len(f) for _, f in sims)
        for i in range(n + int(round(args.hold_s * fps))):
            panels = []
            for (res, frames), s in zip(sims, sides):
                rgb, _t, cwz, turned = frames[min(i, len(frames) - 1)]
                card, ok = card_for(res["yaw_deg"], wz, res["fell_at_s"]) if i >= len(frames) else (None, None)
                panels.append(side_panel(rgb, s["title"], s["subtitle"], turned, cwz, card, ok))
            longest = max((f for _, f in sims), key=len)
            t_now = longest[min(i, len(longest) - 1)][1]
            banner = (f"Run {k + 1}/2: stand {V2_TURN_WARM_S:.0f} s, then turn {direction} for {args.seconds:.0f} s"
                      f"   (same command to both)      t = {t_now:4.1f} s")
            writer.append_data(compose(panels[0], panels[1], banner, args.footer_line))
            n_frames += 1
        del sims
    writer.close()

    gif = write_gif(args.mp4, args.gif, fps=args.gif_fps, width=args.gif_width)
    record = {
        "what": "turning ablation clip: the same turn_test v2 command given to two LEARNED gaits, side by side",
        "label": "LEARNED gaits (PPO, frozen at evaluation); MuJoCo CPU physics, rendered; display only, not a gate",
        "protocol": {"turn_wz_rad_s": V2_TURN_WZ, "warm_s": V2_TURN_WARM_S, "seconds": args.seconds,
                     "reset_seed": args.seed, "variant": args.variant, "shown_from_s": args.show_from,
                     "turn_min_deg_drawn": TURN_MIN_DEG, "match_tol_deg": MATCH_TOL_DEG, "camera": list(args.cam)},
        "sides": [{"side": s["side"], "title": s["title"], "subtitle": s["subtitle"], "run_dir": s["run_dir"],
                   "deploy": str(s["deploy"]), "onnx_sha256": sha256(s["onnx"]),
                   "scored_json": os.path.relpath(s["scored_path"], REPO)} for s in sides],
        "runs": runs,
        "reproduces_scored_runs": all(r["matches_scored"] for r in runs),
        "mp4": os.path.relpath(args.mp4, REPO), "mp4_sha256": sha256(args.mp4), "mp4_codec": codec,
        "mp4_frames": n_frames, "mp4_fps": fps, "gif": gif,
        "git_head": subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"], capture_output=True,
                                   text=True).stdout.strip(),
        "node": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
        "written_at": time.strftime("%Y-%m-%d %H:%M:%S %Z"),
    }
    sidecar.write_text(json.dumps(record, indent=2) + "\n")
    print(f"gif -> {args.gif} ({gif['mb']} MB, within budget {gif['within_budget']}); sidecar -> {sidecar}")
    print(f"TURN-CLIP RESULT: reproduces_scored_runs={record['reproduces_scored_runs']} "
          f"{[(r['side'], r['name'], r['yaw_deg'], r['scored_yaw_deg']) for r in runs]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
