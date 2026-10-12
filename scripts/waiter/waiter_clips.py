#!/usr/bin/env python3
"""Side-by-side display clips of two Waiter leg controllers (display only, not a gate).

Made for the phase 1g result (SLURM_JOBS.md 2026-10-12): each side re-simulates one scored gate episode step for
step, with a read-only free camera:
  stand  gate Q4 (stand_manip_gate.py): trajectory P4_pick_then_hook, 16 s, one reset seed; the same steps as
         stand_manip_gate.episode (0.5 kg per hand, command 0, clock held at step 10, free base);
  walk   the qualify walk (turn_test.run_command, as cpu_turn_qualify's v2x walks): 1 s standing, then 0.35 m/s
         forward for 6 s, one reset seed.
The sidecar JSON puts each side's numbers next to the scored gate JSON's and says whether they match, so a clip that
is not the scored run says so.

usage: waiter_clips.py stand|walk --left "TITLE|DEPLOY|SCORED_JSON" --right "..." --seed N --out docs/gifs/<name>
       (SCORED_JSON: the Q4 JSON for stand, the qualify __v2x.json for walk)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import socket
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path[:0] = [str(REPO / "src"), str(REPO / "scripts/bench"), str(HERE)]

import q2_gate as Q                                                     # noqa: E402
import stand_manip_gate as SM                                           # noqa: E402
from bhl_robust.eval.panels import BG, DIM, TEXT, load_font, write_gif  # noqa: E402
from turn_clip import compose, open_writer                              # noqa: E402
from turn_test import V2_WALK_CMD, V2_WALK_WARM_S, yaw_of               # noqa: E402

GOOD, WARN, BAD = (30, 110, 52), (150, 105, 20), (150, 34, 30)
HEAD_H = 58
WALK_S, WALK_MIN_M = 6.0, 1.5
STAND_TRAJ = "P4_pick_then_hook"


def _camera(mujoco, xy, yaw0, dist, elev, az_off, ahead=0.0):
    cam = mujoco.MjvCamera()
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.distance, cam.elevation = dist, elev
    cam.azimuth = math.degrees(yaw0) + 180.0 + az_off          # 180 = facing the robot's front
    cam.lookat[:] = [xy[0] + ahead * math.cos(yaw0), xy[1] + ahead * math.sin(yaw0), 0.42]
    return cam


def stand_run(deploy: Path, upstream: Path, cache: Path, seed: int, size, every: int):
    """stand_manip_gate.episode, step for step, plus frames (rgb, t, readout)."""
    import mujoco
    cfg, model, slots, ctrl, runner = Q.make(deploy, upstream, cache)
    traj, dur = SM.TRAJECTORIES[STAND_TRAJ]
    runner.reset(np.random.default_rng(seed))
    ctrl.reset_phase()
    slot = slots[0]
    dt = float(cfg.policy_dt)
    quat0 = runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    cam = _camera(mujoco, runner.d.qpos[slot.qpos_adr:slot.qpos_adr + 2].copy(), yaw_of(quat0), 2.0, -14.0, 55.0)
    held, p0, y0, pt0 = None, None, None, None
    disp = yaw = pitch = 0.0
    fell, frames = None, []
    n_steps = int(round(dur / dt))
    for k in range(int(round(dur / dt))):
        t = k * dt
        ctrl.set_upper_body(traj(t))
        if held is None and ctrl.clock_step % 20 == SM.HOLD_STEP:
            held = SM.HOLD_STEP
        if held is not None:
            ctrl.clock_step = held
        runner.step([ctrl.update(runner.observe(0, np.zeros(3)))])
        quat = runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]
        if runner.tilt(0) >= Q.TILT_FALL:
            fell = round(t, 2)
        elif t >= SM.T0_S:
            xy = runner.d.qpos[slot.qpos_adr:slot.qpos_adr + 2]
            if p0 is None:
                p0, y0, pt0 = xy.copy(), Q.yaw_of(quat), SM.pitch_of(quat)
            disp = max(disp, float(np.linalg.norm(xy - p0)))
            yaw = max(yaw, abs(math.degrees(math.remainder(Q.yaw_of(quat) - y0, 2 * math.pi))))
            pitch = max(pitch, abs(math.degrees(SM.pitch_of(quat) - pt0)))
        if k % every == 0 or fell is not None or k == n_steps - 1:
            renderer.update_scene(runner.d, camera=cam)
            line = (f"{t + dt:4.1f} s | moved {100 * disp:3.1f} cm | yaw {yaw:3.1f}\N{DEGREE SIGN} | "
                    f"pitch {pitch:3.1f}\N{DEGREE SIGN}") if t >= SM.T0_S else f"{t + dt:4.1f} s | told to stand"
            frames.append((renderer.render().copy(), t + dt, line))
        if fell is not None:
            break
    renderer.close()
    if fell is not None:
        res = {"seed": seed, "fell_at_s": fell, "disp_m": None, "yaw_deg": None, "pitch_deg": None, "ok": False}
        card, colour = f"FELL at {fell:.1f} s", BAD
    else:
        ok = disp <= SM.MAX_DISP_M and yaw <= SM.MAX_YAW_DEG and pitch <= SM.MAX_PITCH_DEG
        res = {"seed": seed, "fell_at_s": None, "disp_m": round(disp, 4), "yaw_deg": round(yaw, 2),
               "pitch_deg": round(pitch, 2), "ok": bool(ok)}
        card = f"no fall: {100 * disp:.1f} cm, {yaw:.1f}\N{DEGREE SIGN}, {pitch:.1f}\N{DEGREE SIGN}"
        colour = GOOD if ok else WARN
    return res, frames, card, colour


def walk_run(deploy: Path, upstream: Path, cache: Path, seed: int, size, every: int):
    """turn_test.run_command (the v2x walk), step for step, plus frames."""
    import mujoco
    from omegaconf import OmegaConf
    from bhl_robust.eval.gait_clock import make_controller
    from bhl_robust.eval.multi_robot import build_multi
    from team_airlock import ContactRunner, CpuPolicy
    variant, cmd, warm, seconds = "waiter", V2_WALK_CMD, V2_WALK_WARM_S, WALK_S
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
    renderer = mujoco.Renderer(model, height=size[1], width=size[0])
    cam = _camera(mujoco, xy0, yaw0, 3.0, -12.0, 90.0, ahead=1.0)     # side view, centred 1 m ahead of the start
    frames, n_steps = [], int((warm + seconds) / dt)
    fell, yaws = None, []
    for step in range(int((warm + seconds) / dt)):
        now = step * dt
        c = np.zeros(3) if now < warm else np.asarray(cmd, dtype=float)
        obs = runner.observe(0, c)
        runner.step([ctrl.update(obs)])
        yaws.append(yaw_of(runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]))
        if step % every == 0 or step == n_steps - 1 or runner.tilt(0) >= 0.78:
            renderer.update_scene(runner.d, camera=cam)
            moved = float(np.linalg.norm(runner.d.xpos[slot.body_id, :2] - xy0))
            line = (f"{now + dt:3.1f} s | standing (command 0)" if now < warm else
                    f"{now + dt:3.1f} s | {cmd[0]:.2f} m/s forward | travelled {moved:4.2f} m")
            frames.append((renderer.render().copy(), now + dt, line))
        if runner.tilt(0) >= 0.78:
            fell = now
            break
    renderer.close()
    dyaw = float(np.unwrap(np.array(yaws))[-1] - yaw0)
    xy1 = runner.d.xpos[slot.body_id, :2]
    res = {"cmd": [float(v) for v in cmd], "fell_at_s": fell, "yaw_deg": round(math.degrees(dyaw), 1),
           "displacement_m": round(float(np.linalg.norm(xy1 - xy0)), 3), "seconds": seconds}
    if fell is not None:
        card, colour = f"FELL at {fell:.1f} s", BAD
    else:
        d = res["displacement_m"]
        card = f"walked {d:.2f} m (needs {WALK_MIN_M:.1f} m)"
        colour = GOOD if d >= WALK_MIN_M else BAD
    return res, frames, card, colour


def side(rgb, title: str, subtitle: str, line: str, card: str | None, colour) -> Image.Image:
    view = Image.fromarray(np.ascontiguousarray(np.asarray(rgb)[..., :3].astype(np.uint8)))
    w, h = view.size
    img = Image.new("RGB", (w, h + HEAD_H + 24), BG)
    draw = ImageDraw.Draw(img)
    draw.text((10, 7), title, font=load_font(17), fill=TEXT)
    draw.text((10, 32), subtitle, font=load_font(13), fill=DIM)
    img.paste(view, (0, HEAD_H))
    draw.text((10, HEAD_H + h + 4), line, font=load_font(13), fill=TEXT)
    if card:
        font = load_font(16)
        tw = draw.textlength(card, font=font)
        x0, y0 = max(4, (w - tw) / 2 - 10), HEAD_H + h - 46
        draw.rectangle((x0, y0, x0 + tw + 20, y0 + 32), fill=colour, outline=(255, 255, 255), width=2)
        draw.text((x0 + 10, y0 + 7), card, font=font, fill=(255, 255, 255))
    return img


def scored_match(kind: str, scored: dict, res: dict, seed: int) -> dict:
    """The scored gate's numbers for this episode and whether the clip reproduces them."""
    if kind == "stand":
        row = next((e for e in scored.get("episodes", []) if e.get("trajectory") == STAND_TRAJ and e.get("seed") == seed), None)
        keys = ("fell_at_s", "disp_m", "yaw_deg", "pitch_deg")
        same = row is not None and all(row.get(k) == res.get(k) for k in keys)
    else:
        row = next((w for w in scored.get("walks", []) if w.get("seed") == seed), None)
        same = row is not None and row.get("displacement_m") == res["displacement_m"] and row.get("yaw_deg") == res["yaw_deg"]
    return {"scored": row, "clip": res, "matches_scored_run": bool(same)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("kind", choices=("stand", "walk"))
    ap.add_argument("--left", required=True, help="TITLE|DEPLOY|SCORED_JSON")
    ap.add_argument("--right", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--banner", required=True)
    ap.add_argument("--footer", action="append", default=[])
    ap.add_argument("--upstream", type=Path, default=REPO / "external/Berkeley-Humanoid-Lite")
    ap.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("TMPDIR", "/tmp")) / "waiter-clips")
    ap.add_argument("--size", type=int, nargs=2, default=(400, 330))
    ap.add_argument("--every", type=int, default=2, help="render every n-th policy step (0.04 s each)")
    ap.add_argument("--speed", type=float, default=1.0, help="GIF playback speed")
    ap.add_argument("--out", type=Path, required=True, help="output stem: <out>.gif, <out>.mp4, <out>.json")
    a = ap.parse_args(argv)
    run = stand_run if a.kind == "stand" else walk_run
    sides, record = [], {"kind": a.kind, "seed": a.seed, "display_only": True, "host": socket.gethostname(),
                         "slurm_job": os.environ.get("SLURM_JOB_ID"), "sides": []}
    for spec in (a.left, a.right):
        title, deploy, scored = spec.split("|")
        res, frames, card, colour = run(Path(deploy), a.upstream, a.cache_dir, a.seed, tuple(a.size), a.every)
        m = scored_match(a.kind, json.loads(Path(scored).read_text()), res, a.seed)
        sub = (f"gate Q4, trajectory {STAND_TRAJ}, reset seed {a.seed}" if a.kind == "stand"
               else f"qualify walk, reset seed {a.seed}")
        sub += "" if m["matches_scored_run"] else "  (NOT the scored run)"
        sides.append((title, sub, frames, card, colour))
        record["sides"].append({"title": title, "deploy": deploy, "scored_json": scored, **m})
        print(f"[clip] {title}: {res} | matches scored run: {m['matches_scored_run']}", flush=True)
    fps = 1.0 / (0.04 * a.every)                                      # real time: one frame per rendered step
    n = max(len(s[2]) for s in sides) + int(round(2 * fps))          # hold the end cards for 2 s
    a.out.parent.mkdir(parents=True, exist_ok=True)
    mp4 = a.out.with_suffix(".mp4")
    writer, codec = open_writer(mp4, fps)
    for i in range(n):
        panels = []
        for title, sub, frames, card, colour in sides:
            rgb, _, line = frames[min(i, len(frames) - 1)]
            panels.append(side(rgb, title, sub, line, card if i >= len(frames) - 1 else None, colour))
        writer.append_data(compose(panels[0], panels[1], a.banner, a.footer))
    writer.close()
    gif = write_gif(mp4, a.out.with_suffix(".gif"), fps=min(int(fps), 12), speed=a.speed, width=820,
                    stats_mode="full")                  # "diff" drops the red of the static result cards
    record.update({"mp4": str(mp4), "codec": codec, "fps": fps, "frames": n, "gif": gif})
    a.out.with_suffix(".json").write_text(json.dumps(record, indent=1) + "\n")
    print(f"[clip] wrote {gif['gif']} ({gif['mb']} MB, within budget {gif['within_budget']})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
