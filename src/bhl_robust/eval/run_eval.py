"""Evaluate an exported policy in MuJoCo and write per-episode metrics to CSV.

This is the measurement instrument for the whole project. Every policy --
every DR rung, every push arm, every terrain variant -- is scored by this same
protocol so the numbers are comparable.

Usage:
    python -m bhl_robust.eval.run_eval \
        --deploy-cfg <run>/exported/deploy.yaml \
        --upstream   <path to Berkeley-Humanoid-Lite> \
        --cache-dir  <writable scratch> \
        --out        results/raw/<run>.csv \
        --label      dr-off-s0
"""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import asdict
from pathlib import Path

from omegaconf import OmegaConf

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController

from bhl_robust.eval.harness import (
    EvalConfig,
    HeadlessMujocoEnv,
    run_episode,
)
from bhl_robust.eval.mjcf_assets import prepare_mjcf
from bhl_robust.eval.video import EpisodeRecorder, ffmpeg_available


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deploy-cfg", required=True, type=Path,
                   help="deploy YAML written by play.py (points at the ONNX)")
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--label", required=True, help="run identifier recorded in every row")
    p.add_argument("--variant", default="biped", choices=["biped", "humanoid"])
    p.add_argument("--episode-s", type=float, default=10.0)
    p.add_argument("--seeds", type=int, default=5)
    p.add_argument("--push-speed", type=float, default=0.0,
                   help=">0 enables the push protocol (experiment 1 scoring)")
    p.add_argument("--onnx", type=Path, default=None,
                   help="override the checkpoint path inside the deploy YAML")
    p.add_argument("--video-dir", type=Path, default=None,
                   help="render MP4s here (requires MUJOCO_GL=egl and a GPU)")
    p.add_argument("--video-seed", type=int, default=0,
                   help="only this seed is filmed, so one clip per command")
    args = p.parse_args(argv)

    cfg = OmegaConf.load(args.deploy_cfg)
    if args.onnx is not None:
        cfg.policy_checkpoint_path = str(args.onnx)

    onnx_path = Path(cfg.policy_checkpoint_path)
    if not onnx_path.is_file():
        print(f"ERROR: policy not found: {onnx_path}", file=sys.stderr)
        return 2

    scene = prepare_mjcf(args.upstream, args.cache_dir, args.variant)

    eval_cfg = EvalConfig(
        episode_s=args.episode_s,
        seeds=tuple(range(args.seeds)),
        push_speed=args.push_speed,
    )

    env = HeadlessMujocoEnv(cfg, scene)
    controller = RlController(cfg)
    controller.load_policy()

    if args.video_dir and not ffmpeg_available():
        print("ERROR: --video-dir given but ffmpeg is not on PATH", file=sys.stderr)
        return 2

    rows: list[dict] = []
    for command in eval_cfg.commands:
        for seed in eval_cfg.seeds:
            recorder = None
            if args.video_dir is not None and seed == args.video_seed:
                tag = f"vx{command[0]:+.1f}_vy{command[1]:+.1f}_wz{command[2]:+.1f}"
                caption = f"{args.label}   cmd vx={command[0]:+.2f} vy={command[1]:+.2f} wz={command[2]:+.2f}"
                if eval_cfg.push_speed > 0:
                    caption += f"   push {eval_cfg.push_speed:.2f} m/s"
                recorder = EpisodeRecorder(
                    env.mj_model,
                    args.video_dir / f"{args.label}__{tag}.mp4",
                    fps=1.0 / env.cfg.policy_dt,
                    caption=caption,
                )

            res = run_episode(env, controller, command, seed, eval_cfg, recorder=recorder)

            if recorder is not None:
                err = recorder.close()
                outcome = "FELL" if res.fell else "OK"
                final = recorder.path.with_name(
                    recorder.path.stem + f"__{outcome}" + recorder.path.suffix)
                if err:
                    print(f"  ffmpeg failed: {err}", file=sys.stderr)
                else:
                    recorder.path.rename(final)
                    print(f"  video -> {final.name} ({recorder.n_frames} frames)")

            row = {"label": args.label, **asdict(res)}
            rows.append(row)
            print(
                f"{args.label} cmd=({command[0]:+.1f},{command[1]:+.1f},{command[2]:+.1f}) "
                f"seed={seed} fell={int(res.fell)} surv={res.survival_s:5.2f}s "
                f"lin_err={res.lin_vel_err:.3f} dist={res.distance_m:.2f}",
                flush=True,
            )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    n_fell = sum(r["fell"] for r in rows)
    print(f"\n{args.label}: {len(rows)} episodes, {n_fell} falls "
          f"({100.0 * n_fell / len(rows):.1f}% fall rate) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
