"""Render one rollout as third-person RGB beside the robot's own depth image.

The depth here comes from MuJoCo's offscreen buffer, not from Isaac Lab -- a
different renderer, a different projection, a different simulator. That is the
same argument the rest of this repo makes for scoring PhysX policies in MuJoCo:
a depth map that survives the change of implementation is a depth map.

Driven by the same `run_episode` that produces the metrics, so the clip shows an
episode a results row describes rather than a separate demo.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from omegaconf import OmegaConf

from berkeley_humanoid_lite_lowlevel.policy.rl_controller import RlController

from bhl_robust.eval.depth import FAR_M, NEAR_M, DepthPairRecorder
from bhl_robust.eval.harness import EvalConfig, HeadlessMujocoEnv, run_episode
from bhl_robust.eval.mjcf_assets import prepare_mjcf
from bhl_robust.eval.video import ffmpeg_available


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--deploy-cfg", required=True, type=Path)
    p.add_argument("--upstream", required=True, type=Path)
    p.add_argument("--cache-dir", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path, help="output MP4")
    p.add_argument("--label", required=True)
    p.add_argument("--variant", default="biped", choices=["biped", "humanoid"])
    p.add_argument("--episode-s", type=float, default=10.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--terrain-difficulty", type=float, default=0.6)
    p.add_argument("--terrain-seed", type=int, default=12345)
    p.add_argument("--command", type=float, nargs=3, default=[0.3, 0.0, 0.0])
    p.add_argument("--depth-res", type=int, default=64)
    p.add_argument("--panel", type=int, default=448, help="must be a multiple of --depth-res")
    args = p.parse_args(argv)

    if not ffmpeg_available():
        print("ERROR: ffmpeg is not on PATH", file=sys.stderr)
        return 2

    cfg = OmegaConf.load(args.deploy_cfg)
    if not Path(cfg.policy_checkpoint_path).is_file():
        print(f"ERROR: policy not found: {cfg.policy_checkpoint_path}", file=sys.stderr)
        return 2

    rough = args.terrain_difficulty > 0.0
    # ego_camera=True is what puts the depth camera in the model; it caches to a
    # separate directory so the scored runs keep loading the unmodified MJCF.
    scene = prepare_mjcf(args.upstream, args.cache_dir, args.variant,
                         terrain=rough, ego_camera=True)

    eval_cfg = EvalConfig(
        episode_s=args.episode_s,
        seeds=(args.seed,),
        terrain_difficulty=args.terrain_difficulty,
        terrain_seed=args.terrain_seed,
    )
    env = HeadlessMujocoEnv(cfg, scene, terrain_difficulty=args.terrain_difficulty,
                            terrain_seed=args.terrain_seed)
    controller = RlController(cfg)
    controller.load_policy()

    caption = (f"{args.label}   depth {args.depth_res}x{args.depth_res}, "
               f"{NEAR_M:.2f}-{FAR_M:.1f} m")
    if rough:
        caption += f"   terrain d={args.terrain_difficulty:.2f}"

    rec = DepthPairRecorder(env.mj_model, args.out, fps=1.0 / env.cfg.policy_dt,
                            panel=args.panel, depth_res=args.depth_res, caption=caption)
    res = run_episode(env, controller, tuple(args.command), args.seed, eval_cfg, recorder=rec)
    err = rec.close()
    if err:
        print(f"ffmpeg failed: {err}", file=sys.stderr)
        return 3

    import numpy as np
    stack = np.stack(rec.depth_stack)
    print(f"{args.label}: {rec.n_frames} frames -> {args.out}")
    print(f"  fell={int(res.fell)} survived={res.survival_s:.2f}s distance={res.distance_m:.2f} m")
    print(f"  depth {stack.shape} range [{stack.min():.2f}, {stack.max():.2f}] m")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
