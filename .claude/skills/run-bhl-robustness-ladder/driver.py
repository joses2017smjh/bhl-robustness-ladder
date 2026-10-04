"""Drive a frozen BHL gait policy in MuJoCo through a plan of commands, report what it did, optionally screenshot it.

Agent tooling for the run-bhl-robustness-ladder skill (not a gate, not a benchmark). The simulation is the one the
scored turn test uses (scripts/bench/turn_test.py run_command): one robot, flat world, build_multi + ContactRunner +
CpuPolicy + make_controller, rng = default_rng(seed), fall = tilt >= 0.78 rad. The renderer only reads the state, so
`--plan stand:3,turn:0.6:6 --seed 0` reproduces the scored v2 turn run turn+_s0 (use --expect-yaw to check it).

    python driver.py policies [--variant humanoid|biped]
    python driver.py rollout --policy arms-turngait-clock-s2 --plan stand:3,turn:0.6:6 --seed 0 \
        [--png sheet.png] [--json out.json] [--expect-yaw 196.8]

Plan segments (comma-separated; seconds last): stand:T | walk:VX:T | turn:WZ:T | cmd:VX:VY:WZ:T.
Rendering needs MUJOCO_GL=egl on a GPU node (run.sh gpu ...); CPU nodes have neither EGL nor OSMesa.
Exit codes: 0 ok, 1 the robot fell or --expect-yaw missed, 2 bad arguments / nothing to run.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import tempfile
from pathlib import Path

import numpy as np

SKILL = Path(__file__).resolve().parent
REPO = SKILL.parents[2]
UPSTREAM = REPO / "external" / "Berkeley-Humanoid-Lite"
RUNS = UPSTREAM / "logs" / "rsl_rl"
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "bench"))     # team_airlock, turn_test (as turn_test.py itself does)

FALL_TILT = 0.78


def yaw_of(q):
    return math.atan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2))


def parse_plan(text: str) -> list[dict]:
    """'stand:3,walk:0.35:4,turn:-0.6:5,cmd:0.2:0:0.3:2' -> [{name, cmd (vx, vy, wz), seconds}, ...]."""
    plan = []
    for raw in text.split(","):
        parts = raw.strip().split(":")
        kind, nums = parts[0], [float(p) for p in parts[1:]]
        if kind == "stand" and len(nums) == 1:
            cmd = (0.0, 0.0, 0.0)
        elif kind == "walk" and len(nums) == 2:
            cmd = (nums[0], 0.0, 0.0)
        elif kind == "turn" and len(nums) == 2:
            cmd = (0.0, 0.0, nums[0])
        elif kind == "cmd" and len(nums) == 4:
            cmd = tuple(nums[:3])
        else:
            raise ValueError(f"bad plan segment {raw!r} (stand:T | walk:VX:T | turn:WZ:T | cmd:VX:VY:WZ:T)")
        if nums[-1] <= 0:
            raise ValueError(f"segment {raw!r}: seconds must be > 0")
        plan.append({"name": raw.strip(), "cmd": cmd, "seconds": nums[-1]})
    return plan


def find_deploy(policy: str, variant: str) -> Path:
    """The newest logs/rsl_rl/<variant>/*_<policy>/exported/deploy.yaml."""
    hits = sorted((RUNS / variant).glob(f"*_{policy}/exported/deploy.yaml"))
    if not hits:
        raise FileNotFoundError(f"no exported deploy.yaml for run name {policy!r} under {RUNS / variant}")
    return hits[-1]


def list_policies(variant: str) -> list[str]:
    """'<run name>  (<run dir>)' per exported policy; --policy takes the run name (the dir minus its timestamp)."""
    dirs = [p.parent.parent.name for p in sorted((RUNS / variant).glob("*/exported/deploy.yaml"))]
    return [f"{d.split('_', 2)[-1]}  ({d})" for d in dirs]


def rollout(deploy: Path, variant: str, plan: list[dict], seed: int, shots: bool, size: int, cache: Path):
    """Run the plan. Returns (segments, frames): one report per segment; frames = [(label, rgb)] at reset and at
    each segment's end when `shots`."""
    import mujoco
    from omegaconf import OmegaConf
    from bhl_robust.eval.gait_clock import make_controller
    from bhl_robust.eval.multi_robot import build_multi
    from team_airlock import ContactRunner, CpuPolicy

    cfg = OmegaConf.load(deploy)
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    model, slots = build_multi(UPSTREAM, cache / variant, 1, ["t"], variant=variant, world="flat")
    ctrl = make_controller(cfg)
    ctrl.policy = policy
    runner = ContactRunner(model, slots, [cfg], [ctrl])
    runner.reset(np.random.default_rng(seed))
    slot = slots[0]
    owners = np.array([0 if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g])) or "")
                       .startswith(slot.prefix) else -1 for g in range(model.ngeom)])
    runner.configure_contacts(owners)
    dt = float(cfg.policy_dt)
    quat = lambda: runner.d.qpos[slot.qpos_adr + 3:slot.qpos_adr + 7]   # noqa: E731
    yaw0 = yaw_of(quat())
    xy0 = runner.d.xpos[slot.body_id, :2].copy()

    renderer = camera = None
    frames = []
    if shots:
        renderer = mujoco.Renderer(model, height=size, width=size)
        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        camera.distance, camera.elevation = 2.6, -20.0
        camera.azimuth = math.degrees(yaw0) + 180.0 + 30.0     # facing the robot's front, 30 deg off its axis
        camera.lookat[:] = [xy0[0], xy0[1], 0.42]

    def shoot(label):
        if renderer is not None:
            camera.lookat[:2] = runner.d.xpos[slot.body_id, :2]
            renderer.update_scene(runner.d, camera=camera)
            frames.append((label, renderer.render().copy()))

    shoot("t=0.0 s  reset")
    ends = np.cumsum([s["seconds"] for s in plan])
    total = float(ends[-1])
    yaws, fell, seg_i = [], None, 0
    seg_start = {"yaw": 0.0, "xy": xy0.copy(), "t": 0.0}
    segments = []

    def close_segment(i, t_end):
        unwrapped = float(np.unwrap(np.array(yaws))[-1] - yaw0) if yaws else 0.0
        xy = runner.d.xpos[slot.body_id, :2]
        segments.append({"segment": plan[i]["name"], "cmd": list(plan[i]["cmd"]), "t_start_s": round(seg_start["t"], 3),
                         "t_end_s": round(t_end, 3),
                         "yaw_in_segment_deg": round(math.degrees(unwrapped - seg_start["yaw"]), 1),
                         "yaw_since_reset_deg": round(math.degrees(unwrapped), 1),
                         "displacement_in_segment_m": round(float(np.linalg.norm(xy - seg_start["xy"])), 3),
                         "displacement_since_reset_m": round(float(np.linalg.norm(xy - xy0)), 3)})
        seg_start.update(yaw=unwrapped, xy=xy.copy(), t=t_end)

    # Same step loop as turn_test.run_command: int(total / dt) steps; the command at `now` is the first segment
    # whose end is > now (run_command: zeros while now < warm).
    for step in range(int(total / dt)):
        now = step * dt
        while seg_i < len(plan) - 1 and not now < ends[seg_i]:
            close_segment(seg_i, float(ends[seg_i]))
            shoot(f"t={ends[seg_i]:.1f} s  end of {plan[seg_i]['name']}  yaw {segments[-1]['yaw_since_reset_deg']:+.0f}")
            seg_i += 1
        obs = runner.observe(0, np.asarray(plan[seg_i]["cmd"], dtype=float))
        runner.step([ctrl.update(obs)])
        yaws.append(yaw_of(quat()))
        if runner.tilt(0) >= FALL_TILT:
            fell = now
            break
    t_last = fell if fell is not None else total
    close_segment(seg_i, t_last)
    label = f"FELL at {fell:.2f} s" if fell is not None else f"end of {plan[seg_i]['name']}"
    shoot(f"t={t_last:.1f} s  {label}  yaw {segments[-1]['yaw_since_reset_deg']:+.0f}")
    if renderer is not None:
        renderer.close()   # explicit: otherwise EGL's teardown at exit prints an ignored EGLError traceback
    for s in segments:
        s["fell_at_s"] = fell
    return segments, frames


def contact_sheet(frames, path: Path, title: str):
    from PIL import Image, ImageDraw
    size = frames[0][1].shape[0]
    cols = min(len(frames), 4)
    rows = math.ceil(len(frames) / cols)
    head, cap = 28, 22
    sheet = Image.new("RGB", (cols * size, head + rows * (size + cap)), (16, 18, 22))
    draw = ImageDraw.Draw(sheet)
    draw.text((8, 8), title, fill=(235, 235, 235))
    for k, (label, rgb) in enumerate(frames):
        x, y = (k % cols) * size, head + (k // cols) * (size + cap)
        sheet.paste(Image.fromarray(np.ascontiguousarray(rgb[..., :3])), (x, y))
        draw.text((x + 6, y + size + 5), label, fill=(235, 235, 235))
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="what", required=True)
    pl = sub.add_parser("policies", help="list run names that have an exported deploy.yaml")
    pl.add_argument("--variant", choices=("humanoid", "biped"), default="humanoid")
    ro = sub.add_parser("rollout", help="drive one policy through a plan")
    who = ro.add_mutually_exclusive_group(required=True)
    who.add_argument("--policy", help="run name, e.g. arms-turngait-clock-s2 (newest *_<name>/exported/deploy.yaml)")
    who.add_argument("--deploy", type=Path, help="an exported deploy.yaml")
    ro.add_argument("--variant", choices=("humanoid", "biped"), default="humanoid",
                    help="humanoid = 22-DoF with arms; biped = legs only (must match the policy)")
    ro.add_argument("--plan", default="stand:3,turn:0.6:6")
    ro.add_argument("--seed", type=int, default=0)
    ro.add_argument("--png", type=Path, help="write a contact sheet: reset + every segment's end (needs EGL)")
    ro.add_argument("--size", type=int, default=320)
    ro.add_argument("--json", type=Path, help="write the report here")
    ro.add_argument("--expect-yaw", type=float, help="exit 1 unless the final yaw since reset is within --tol deg")
    ro.add_argument("--tol", type=float, default=0.5)
    ro.add_argument("--cache", type=Path, default=Path(tempfile.gettempdir()) / f"bhl-driver-{os.getuid()}")
    args = ap.parse_args()

    if args.what == "policies":
        for name in list_policies(args.variant):
            print(name)
        return 0
    try:
        plan = parse_plan(args.plan)
        deploy = args.deploy or find_deploy(args.policy, args.variant)
    except (ValueError, FileNotFoundError) as exc:
        print(f"DRIVER_ERROR {exc}")
        return 2
    segments, frames = rollout(deploy, args.variant, plan, args.seed, args.png is not None, args.size, args.cache)
    for s in segments:
        print(json.dumps(s))
    final = segments[-1]
    report = {"deploy": str(deploy), "variant": args.variant, "seed": args.seed, "plan": args.plan,
              "segments": segments, "fell_at_s": final["fell_at_s"],
              "yaw_since_reset_deg": final["yaw_since_reset_deg"],
              "label": "LEARNED gait (frozen ONNX policy), MuJoCo CPU physics; agent tooling, not a gate"}
    if args.png:
        contact_sheet(frames, args.png, f"{deploy.parent.parent.name}  seed {args.seed}  plan {args.plan}")
        report["png"] = str(args.png)
        print(f"DRIVER_PNG {args.png}")
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(report, indent=1))
    ok = final["fell_at_s"] is None
    if args.expect_yaw is not None:
        hit = abs(final["yaw_since_reset_deg"] - args.expect_yaw) <= args.tol
        print(f"DRIVER_EXPECT_YAW {'MATCH' if hit else 'MISMATCH'} got {final['yaw_since_reset_deg']} "
              f"expected {args.expect_yaw} +- {args.tol}")
        ok = ok and hit
    print("DRIVER_OK" if ok else ("DRIVER_FELL" if final["fell_at_s"] is not None else "DRIVER_MISMATCH"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
