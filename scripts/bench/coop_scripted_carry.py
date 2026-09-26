"""Scripted-arm cooperative cube carry in MuJoCo: score seeds, or render one.

    learned gait (frozen) + scripted arms + oracle cube pose

Two modes.

* score (default): run the predeclared seeds (0-9) for a crew of 2 (one pair)
  or 4 (two pairs, two cubes, one world), write one JSON with every episode,
  the per-pair summary under `scripted_carry.SUCCESS_RULE`, and print
  `SCRIPTED_CARRY_VERDICT crew=N verdict=PASS|NEGATIVE|INCOMPLETE`, computed
  from the JSON. The JSON is rewritten after every episode, so a wall-time
  kill keeps what finished.
* render (`--render-from SCORE.json`): read a score JSON; if it did not PASS,
  print `SCRIPTED_CARRY_RENDER=SKIPPED (NEGATIVE)` and write nothing. If it
  passed, re-simulate the median-by-completion-time successful seed with an
  offscreen renderer (needs OpenGL: MUJOCO_GL=egl on a GPU node), write the
  mp4, a GIF within the 5 MiB budget (`panels.write_gif`) and a JSON sidecar
  with source hashes, labels, metrics and playback speed. `--no-render` runs
  the same pipeline with blank frames for a login-node check.

Everything the script decides is in `scripted_carry.CarryParams` and
`KEYFRAMES_LEFT`; both are copied into every output.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

from bhl_robust.eval import panels                                    # noqa: E402
from bhl_robust.eval import scripted_carry as sc                      # noqa: E402

TASK = "coop_scripted_carry_v1"


def sha256(path) -> str | None:
    path = Path(path)
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_head() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(REPO), "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, timeout=20).stdout.strip() or None
    except Exception:                                                  # noqa: BLE001
        return None


def parse_seeds(raw: str) -> list[int]:
    out = []
    for part in raw.split(","):
        if "-" in part:
            a, b = part.split("-")
            out += list(range(int(a), int(b) + 1))
        elif part:
            out.append(int(part))
    return out


def load(args):
    from omegaconf import OmegaConf
    from team_airlock import CpuPolicy

    cfg = OmegaConf.load(args.deploy)
    if cfg.num_actions != 22 or cfg.num_joints != 22 or cfg.num_observations != 75:
        raise SystemExit("requires the full 22-DoF, 75-observation humanoid locomotion policy")
    policy = CpuPolicy(cfg.policy_checkpoint_path)
    p = sc.CarryParams()
    model, slots, pairs = sc.build_carry(args.upstream, args.cache_dir, args.crew // 2, p)
    return cfg, policy, p, model, slots, pairs


def provenance(args, cfg, p) -> dict:
    import mujoco
    return {
        "task": TASK, "label": sc.LABEL, "label_detail": sc.LABEL_DETAIL,
        "learned": "22-DoF locomotion gait (legs + each robot's outer arm), frozen",
        "scripted": "each robot's grasping arm: joint keyframes reach/squeeze/lift/hold",
        "oracle": "cube pose (carry gate + stop) and robot base poses (pair sync) from the simulator",
        "modelling_choices": [
            "hand pads: one box collision geom per hand = hand-mesh AABB (upstream hands are visual-only)",
            f"grasping-arm PD kp {p.grasp_kp} (deploy.yaml 10); kd and the 4 Nm arm effort cap unchanged",
            "grasping arm spawned in the script's rest pose",
            "the cube may rotate in the hands; the rule scores its centre height",
        ],
        "rule": sc.SUCCESS_RULE, "params": sc.params_dict(p), "keyframes_left": sc.KEYFRAMES_LEFT,
        "crew": args.crew, "pairs": args.crew // 2,
        "simulator": f"MuJoCo {mujoco.__version__}", "physics_dt": float(cfg.physics_dt),
        "policy_dt": float(cfg.policy_dt),
        "deploy": str(Path(args.deploy).resolve()), "checkpoint": str(cfg.policy_checkpoint_path),
        "checkpoint_sha256": sha256(cfg.policy_checkpoint_path),
        "module_sha256": sha256(REPO / "src/bhl_robust/eval/scripted_carry.py"),
        "script_sha256": sha256(Path(__file__)), "git_head": git_head(),
    }


def verdict_of(summary: dict) -> str:
    if not summary["complete"]:
        return "INCOMPLETE"
    return "PASS" if summary["pass"] else "NEGATIVE"


# ------------------------------------------------------------------ score

def score(args) -> int:
    cfg, policy, p, model, slots, pairs = load(args)
    rule = dict(sc.SUCCESS_RULE)
    seeds = parse_seeds(args.seeds)
    if args.seconds is not None:
        # a shortened episode is a pipeline check, never a scored run
        rule["episode_s"] = float(args.seconds)
    payload = provenance(args, cfg, p)
    payload["scored_run"] = args.seconds is None and seeds == list(sc.SUCCESS_RULE["seeds"])
    payload["episodes"] = []
    args.out.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        t0 = time.time()
        ep = sc.run_episode(model, slots, pairs, cfg, policy, seed, p, rule=rule)
        ep["wall_s"] = round(time.time() - t0, 1)
        payload["episodes"].append(ep)
        for r in ep["pairs"]:
            print(json.dumps({"crew": args.crew, "seed": seed, "pair": r["pair"],
                              "success": r["success"], "first_failed_check": r["first_failed_check"],
                              "lift_peak_m": r["lift_peak_m"], "lift_hold_s": r["lift_hold_s"],
                              "carry_m": r["carry_m"], "max_tilt_rad": r["max_tilt_rad"],
                              "floor": r["cube_floor_contact"], "completion_s": r["completion_s"],
                              "wall_s": ep["wall_s"]}), flush=True)
        payload["summary"] = sc.summarize(payload["episodes"], args.crew // 2, rule)
        payload["summary"]["median_seed"] = sc.median_seed(payload["episodes"])
        payload["summary"]["verdict"] = verdict_of(payload["summary"])
        if not payload["scored_run"]:
            payload["summary"]["verdict"] = "PIPELINE_CHECK"
            payload["summary"]["pass"] = False
        args.out.write_text(json.dumps(payload, indent=1, allow_nan=False) + "\n")
    s = payload["summary"]
    counts = " ".join(f"pair{pp['pair']}={pp['successes']}/{pp['episodes']}" for pp in s["per_pair"])
    print(f"SCRIPTED_CARRY_VERDICT crew={args.crew} verdict={s['verdict']} {counts} "
          f"median_seed={s['median_seed']} json={args.out}", flush=True)
    return 0


# ------------------------------------------------------------------ render

class Recorder:
    """Frame hook: overhead-oblique view that follows the cube(s), plus a
    text panel with the live numbers and the labels."""

    def __init__(self, args, model, pairs, policy_dt: float, out_mp4: Path, png_dir: Path):
        from maze_record import FrameSink
        import mujoco

        self.args, self.model, self.pairs = args, model, pairs
        self.render = not args.no_render
        self.w, self.h = args.width, args.height
        self.side_w = 330
        self.fps = 1.0 / policy_dt
        self.sink = FrameSink(out_mp4, png_dir, self.fps / max(1, args.stride))
        self.frames = 0
        self.look = None
        if self.render:
            self.r = mujoco.Renderer(model, height=self.h, width=self.w)
            self.cam = mujoco.MjvCamera()
            self.cam.distance = args.distance
            self.cam.azimuth, self.cam.elevation = args.azimuth, args.elevation
        self.last = None

    def __call__(self, *, step, t, runner, script, pairs, lift, horiz, carry_on, carry_done, commands):
        if self.args.stride > 1 and step % self.args.stride:
            return
        centre = np.mean([runner.d.xpos[pr.cube_body] for pr in pairs], axis=0)
        target = np.array([centre[0], centre[1] - 0.15, 0.35])
        self.look = target if self.look is None else 0.9 * self.look + 0.1 * target
        if self.render:
            self.cam.lookat[:] = self.look
            self.r.update_scene(runner.d, camera=self.cam)
            main = self.r.render()
        else:
            main = np.full((self.h, self.w, 3), 40, dtype=np.uint8)
        panel = self._panel(t, script.phase(t), runner, pairs, lift, horiz, carry_on, carry_done)
        header = (f"MuJoCo | {self.args.crew} BHL humanoids, {len(pairs)} cube(s) 0.28 m / 0.5 kg | "
                  f"seed {self.args.seed_used} | t = {t:5.2f} s | {script.phase(t)} | 1x")
        footer = sc.LABEL + " | hand pads added | scripted arm kp 30"
        frame = panels.compose_frame(main, [panel], header, footer, side_w=self.side_w)
        self.last = frame
        self.sink.add(np.ascontiguousarray(frame))
        self.frames += 1

    def _panel(self, t, phase, runner, pairs, lift, horiz, carry_on, carry_done):
        from PIL import Image, ImageDraw
        img = Image.new("RGB", (self.side_w, self.h), panels.PANEL_BG)
        d = ImageDraw.Draw(img)
        top = panels._title(d, self.side_w, "live numbers (simulator)")
        f, fs = panels.load_font(14), panels.load_font(12)
        y = top + 8
        for k, pr in enumerate(pairs):
            F = runner.cube_forces(pr)
            state = "carrying" if carry_on[k] else ("stopped" if carry_done[k] else phase)
            lines = [f"pair {k}: {state}",
                     f"  cube lift   {100 * lift[k]:6.1f} cm",
                     f"  carried     {horiz[k]:6.2f} m",
                     f"  squeeze b/a {F['b']:4.1f} / {F['a']:4.1f} N",
                     f"  robot tilt  {max(runner.tilt(pr.robot_a), runner.tilt(pr.robot_b)):5.2f} rad"]
            for ln in lines:
                d.text((10, y), ln, font=f, fill=panels.TEXT)
                y += 20
            y += 8
        rule = sc.SUCCESS_RULE
        for ln in ["rule (predeclared):", f"  lift >= {100 * rule['lift_peak_m']:.0f} cm,",
                   f"  >= {100 * rule['lift_hold_m']:.0f} cm for {rule['lift_hold_s']:.0f} s,",
                   f"  carried >= {rule['carry_m']:.1f} m, no fall,", "  no floor contact, 30 s",
                   "", "learned: gait (legs + outer arm)", "scripted: grasping arm",
                   "oracle: cube + robot poses", "contact: MuJoCo, no welds"]:
            d.text((10, y), ln, font=fs, fill=panels.DIM)
            y += 17
        return img

    def hold(self, seconds: float, banner: str):
        if self.last is None:
            return
        from PIL import Image, ImageDraw
        img = Image.fromarray(self.last.copy())
        d = ImageDraw.Draw(img)
        font = panels.load_font(28)
        tw = d.textlength(banner, font=font)
        x, y = (img.size[0] - self.side_w - tw) / 2, 56
        d.rectangle((x - 14, y - 8, x + tw + 14, y + 38), fill=(20, 110, 60))
        d.text((x, y), banner, font=font, fill=(255, 255, 255))
        arr = np.asarray(img)
        for _ in range(int(seconds * self.fps / max(1, self.args.stride))):
            self.sink.add(np.ascontiguousarray(arr))

    def close(self):
        if self.render:
            self.r.close()
        return self.sink.close()


def render(args) -> int:
    scored = json.loads(Path(args.render_from).read_text())
    s = scored.get("summary", {})
    if args.pipeline_check_seed is not None:
        # login-node check of the render path only: blank frames, no GIF, any verdict
        if not args.no_render:
            raise SystemExit("--pipeline-check-seed requires --no-render")
        s = dict(s, median_seed=args.pipeline_check_seed, verdict="PIPELINE_CHECK")
        scored = dict(scored, scored_run=True)
    elif not (scored.get("scored_run") and s.get("verdict") == "PASS" and s.get("median_seed") is not None):
        print(f"SCRIPTED_CARRY_RENDER=SKIPPED verdict={s.get('verdict')} crew={scored.get('crew')} "
              f"(no success clip is rendered for a run that did not pass)", flush=True)
        return 0
    if int(scored["crew"]) != args.crew:
        raise SystemExit(f"--crew {args.crew} but {args.render_from} is crew {scored['crew']}")
    seed = int(s["median_seed"])
    args.seed_used = seed
    want = next((e for e in scored["episodes"] if e["seed"] == seed), scored["episodes"][0])
    cfg, policy, p, model, slots, pairs = load(args)
    if sc.params_dict(p) != scored["params"] or sc.KEYFRAMES_LEFT != scored["keyframes_left"]:
        raise SystemExit("scripted_carry parameters changed since the scored run; re-score first")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"carry_scripted_{args.crew}"
    mp4 = args.out_dir / f"{stem}.mp4"
    rec = Recorder(args, model, pairs, float(cfg.policy_dt), mp4, Path(args.frames_root) / stem)
    t0 = time.time()
    ep = sc.run_episode(model, slots, pairs, cfg, policy, seed, p, frame_hook=rec)
    ok = all(r["success"] for r in ep["pairs"])
    done_t = max((r["completion_s"] or 0.0) for r in ep["pairs"])
    rec.hold(1.5, f"CARRIED  {min(r['carry_m'] for r in ep['pairs']):.2f} m  (seed {seed})" if ok
             else f"NOT REPRODUCED (seed {seed})")
    sink = rec.close()
    if sink["mp4"] is None and sink["frames"] > 0:
        png = Path(sink["png_dir"])
        for enc, extra in (("libx264", ["-preset", "veryfast", "-crf", "20"]), ("mpeg4", ["-q:v", "3"])):
            r = subprocess.run([panels.ffmpeg_exe(), "-y", "-loglevel", "error", "-framerate",
                                str(round(rec.fps / max(1, args.stride))), "-i", str(png / "frame_%04d.png"),
                                "-c:v", enc, *extra, "-pix_fmt", "yuv420p", str(mp4)])
            if r.returncode == 0 and mp4.is_file():
                sink["mp4"] = str(mp4)
                break
    matches = [{"pair": a["pair"], "success": a["success"] == b["success"],
                "completion_s": a["completion_s"], "scored_completion_s": b["completion_s"]}
               for a, b in zip(ep["pairs"], want["pairs"])]
    result = {
        "name": stem, "seed": seed, "crew": args.crew, "reproduced_success": ok,
        "matches_scored_episode": matches, "episode": {k: v for k, v in ep.items() if k != "trace"},
        "frames": sink["frames"], "mp4": sink["mp4"], "mp4_sha256": sha256(mp4) if sink["mp4"] else None,
        "no_render": args.no_render, "video_fps": rec.fps / max(1, args.stride),
        # FrameSink encodes at round(fps): at stride 2 that is 12 fps for 12.5 frames per sim second
        "encoded_fps": round(rec.fps / max(1, args.stride)),
        "playback_speed_mp4": round(round(rec.fps / max(1, args.stride)) / (rec.fps / max(1, args.stride)), 4),
        "camera": {"type": "free, follows mean cube xy", "distance": args.distance,
                   "azimuth": args.azimuth, "elevation": args.elevation, "size": [args.width, args.height]},
        "scored_json": os.path.relpath(Path(args.render_from).resolve(), REPO),
        "scored_json_sha256": sha256(args.render_from),
        "label": sc.LABEL, "label_detail": sc.LABEL_DETAIL, "wall_seconds": round(time.time() - t0, 1),
    }
    if args.gif and result["mp4"] and ok and not args.no_render:
        g = panels.write_gif(mp4, args.gif, fps=args.gif_fps, speed=args.gif_speed, width=args.gif_width)
        result["gif"] = g
        sidecar = {
            "output": os.path.relpath(Path(args.gif).resolve(), REPO), "output_sha256": sha256(args.gif),
            "output_mb": g["mb"], "within_budget": g["within_budget"],
            "source_clip": os.path.relpath(mp4.resolve(), REPO), "source_sha256": result["mp4_sha256"],
            "evidence": result["scored_json"], "evidence_sha256": result["scored_json_sha256"],
            "caption": (f"{args.crew} BHL humanoids lift a 0.28 m / 0.5 kg cube together and carry it "
                        f"(seed {seed}, median completion of the successful seeds). "
                        f"{sc.LABEL}. Hand pads added; grasping arm scripted at kp 30."),
            "label": sc.LABEL, "label_detail": sc.LABEL_DETAIL,
            "modelling_choices": scored["modelling_choices"],
            "success_rule": scored["rule"], "summary": s,
            "episode": [{k: r[k] for k in ("pair", "success", "lift_peak_m", "lift_hold_s", "carry_m",
                                           "max_tilt_rad", "completion_s")} for r in ep["pairs"]],
            "playback_speed": args.gif_speed,
            "sim_seconds_per_gif_second": round(args.gif_speed * result["playback_speed_mp4"], 4),
            "gif": g,
            "checkpoint": scored["checkpoint"], "checkpoint_sha256": scored["checkpoint_sha256"],
            "module_sha256": scored["module_sha256"],
        }
        Path(args.gif).with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    elif args.gif:
        result["gif"] = None
        result["gif_skipped"] = ("--no-render" if args.no_render else
                                 "re-simulated episode did not succeed" if not ok else "no mp4")
    (args.out_dir / f"{stem}.json").write_text(json.dumps(result, indent=1) + "\n")
    gif_path = result["gif"]["gif"] if result.get("gif") else None
    print(f"SCRIPTED_CARRY_RENDER crew={args.crew} seed={seed} reproduced={ok} "
          f"completion_s={done_t:.2f} frames={sink['frames']} gif={gif_path}", flush=True)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--deploy", type=Path, required=True)
    ap.add_argument("--upstream", type=Path, required=True)
    ap.add_argument("--cache-dir", type=Path, required=True)
    ap.add_argument("--crew", type=int, choices=(2, 4), default=2)
    ap.add_argument("--seeds", default="0-9", help="e.g. 0-9 or 100,101")
    ap.add_argument("--seconds", type=float, default=None,
                    help="shorten episodes (pipeline check only; the verdict becomes PIPELINE_CHECK)")
    ap.add_argument("--out", type=Path, help="score mode: JSON output")
    ap.add_argument("--render-from", type=Path, help="render mode: a score JSON")
    ap.add_argument("--out-dir", type=Path, help="render mode: mp4 + result JSON directory")
    ap.add_argument("--gif", type=Path)
    ap.add_argument("--frames-root", default=os.path.join(os.environ.get("TMPDIR", "/tmp"), "scripted-carry-frames"))
    ap.add_argument("--width", type=int, default=960)
    ap.add_argument("--height", type=int, default=540)
    ap.add_argument("--distance", type=float, default=3.4)
    ap.add_argument("--azimuth", type=float, default=135.0)
    ap.add_argument("--elevation", type=float, default=-38.0)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--no-render", action="store_true")
    ap.add_argument("--pipeline-check-seed", type=int, default=None,
                    help="with --no-render: exercise the render path on this seed regardless of verdict")
    ap.add_argument("--gif-speed", type=float, default=2.0)
    ap.add_argument("--gif-fps", type=int, default=8)
    ap.add_argument("--gif-width", type=int, default=860)
    args = ap.parse_args()
    if args.render_from:
        if not args.out_dir:
            ap.error("--render-from needs --out-dir")
        return render(args)
    if not args.out:
        ap.error("score mode needs --out")
    return score(args)


if __name__ == "__main__":
    sys.exit(main())
