#!/usr/bin/env python3
"""Run the existing official LeHome fold evaluator; persist its returned metrics.

This entrypoint does not copy modules into the sibling checkout. It calls the
official evaluation loop and scorer, supplying the already-tested Storm camera
shim. GPU PhysX is mandatory. Short smokes verify plumbing, not folding success.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import traceback

import numpy as np


def write_result(path, value):
    tmp = path.with_suffix(".tmp.json")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--lehome-repo", type=Path, required=True)
    p.add_argument("--assets", type=Path, required=True)
    p.add_argument("--dataset-root", type=Path, required=True)
    p.add_argument("--policy-path", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--garment-type", choices=("top_short", "top_long", "pant_short", "pant_long"), default="pant_short")
    p.add_argument("--episodes", type=int, default=2)
    p.add_argument("--max-steps", type=int, default=600)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    args.out = args.out.resolve()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.out.exists():
        raise FileExistsError(f"refusing to overwrite evaluation {args.out}")
    if min(args.episodes, args.max_steps) < 1:
        raise ValueError("episodes and max-steps must be positive")
    policy_path = args.policy_path.resolve()
    if policy_path.is_file() and policy_path.suffix == ".json":
        policy_path = Path(json.loads(policy_path.read_text())["path"])
    if not (policy_path / "model.safetensors").is_file():
        raise FileNotFoundError(f"policy checkpoint missing: {policy_path}")
    lehome = args.lehome_repo / "external" / "lehome-challenge"
    sys.path[:0] = [str(lehome), str(lehome / "source" / "lehome"), str(args.lehome_repo / "src")]
    os.chdir(lehome)
    os.environ["LEHOME_DISABLE_KEYBOARD"] = "1"
    from lehome_fold import storm_eval
    from lehome_fold.storm_obs import StormObserver, StormObsConfig

    storm_eval.stub_keyboard()
    category = {"top_short": "Top_Short", "top_long": "Top_Long",
                "pant_short": "Pant_Short", "pant_long": "Pant_Long"}[args.garment_type]
    garment_list = args.assets / "objects" / "Challenge_Garment" / "Release" / category / f"{category}.txt"
    names = [v.strip() for v in garment_list.read_text().splitlines() if v.strip()]
    class CheckedStormObserver(StormObserver):
        """Give each new garment a unique USD layer identifier and fresh cameras."""
        def __init__(self, config):
            super().__init__(config)
            self.root_workdir = Path(self.workdir)
            self.garment_counter = 0
            self.rendered_frames = 0

        def retarget(self, garment_dir):
            if garment_dir != self.cfg.garment_dir:
                self.garment_counter += 1
                # USD attributes/cameras can keep the old layer alive after
                # _stage=None; deleting its FILE does not unregister its ID.
                self._points = None
                self._cams.clear()
                self._link_ops.clear()
                self._rec = None
                self._last_frames = {}
                self.workdir = str(self.root_workdir / f"garment_{self.garment_counter:02d}")
                Path(self.workdir).mkdir(parents=True, exist_ok=True)
            return super().retarget(garment_dir)

        def render(self):
            frames = super().render()
            if set(frames) != {"top_rgb", "left_rgb", "right_rgb"}:
                raise RuntimeError("missing Storm camera output")
            for name, frame in frames.items():
                if frame.shape != (480, 640, 3) or frame.dtype != np.uint8 or np.std(frame) < 1:
                    raise RuntimeError(f"invalid/blank Storm RGB for {name}")
            self.rendered_frames += 1
            return frames

    observer = CheckedStormObserver(StormObsConfig(
        assets=str(args.assets),
        garment_dir=storm_eval.garment_dir_for(str(args.assets), names[0]),
        workdir=str(args.out.parent / f"{args.out.stem}_frames"), extra={}))
    original_enable = storm_eval.enable

    def strict_enable(env_module, current_observer, *, device="cuda", verbose=True):
        original_observations = original_enable(env_module, current_observer,
                                               device=device, verbose=verbose)

        def checked_observations(env):
            points, links = storm_eval._particles(env), storm_eval._link_poses(env)
            if points is not None and links is not None:
                # The sibling shim catches renderer exceptions and serves
                # stale frames. Any such failure invalidates policy evaluation.
                current_observer.update(points, links)
                current_observer.render()
            return original_observations(env)

        env_module.GarmentEnv._get_observations = checked_observations
        return original_observations

    storm_eval.enable = strict_enable
    storm_eval.enable_when_imported("lehome.tasks.bedroom.garment_bi_v2", observer, device="cuda")

    from isaaclab.app import AppLauncher
    from scripts.utils.parser import setup_eval_parser
    from scripts.utils.common import launch_app_from_args
    parser = setup_eval_parser()
    AppLauncher.add_app_launcher_args(parser)
    # --enable_cameras must be absent: it instantiates the broken RTX delegate
    # before the camera shim has a chance to run.
    official = parser.parse_args([
        "--policy_type", "lerobot", "--policy_path", str(policy_path),
        "--dataset_root", str(args.dataset_root), "--garment_type", args.garment_type,
        "--garment_cfg_base_path", str(args.assets / "objects" / "Challenge_Garment"),
        "--num_episodes", str(args.episodes), "--max_steps", str(args.max_steps),
        "--seed", str(args.seed), "--device", "cuda:0", "--headless"])
    app = launch_app_from_args(official)
    import lehome.tasks.bedroom  # noqa: F401
    from scripts.utils import evaluation

    records = []
    original = evaluation.run_evaluation_loop
    result = {"task": "bimanual_so101_garment_folding", "policy": str(policy_path),
              "garment_type": args.garment_type, "seed": args.seed,
              "renderer": "Storm RGB (synthetic depth unused by RGB policy)",
              "physics": "Isaac Sim 5.1 GPU PhysX particle cloth",
              "scorer": "unmodified official LeHome run_evaluation_loop returned metrics",
              "expected_episodes": len(names) * args.episodes, "max_steps": args.max_steps,
              "completed": False, "garments": records}

    def measured(env, policy, **kwargs):
        config = kwargs.pop("args")
        name = kwargs["garment_name"]
        health = {"steps": 0, "max_particle_step_m": 0.0, "finite": True}
        initial_render_count = observer.rendered_frames
        step_fn = env.step
        previous = None

        def checked_step(*step_args, **step_kwargs):
            nonlocal previous
            out = step_fn(*step_args, **step_kwargs)
            vertices = storm_eval._particles(env)
            if vertices is None or not np.isfinite(vertices).all():
                health["finite"] = False
                raise RuntimeError("nonfinite or unreadable cloth particles")
            for side in ("left_arm", "right_arm"):
                data = getattr(env, side).data
                if not bool(data.joint_pos.isfinite().all()) or not bool(data.joint_vel.isfinite().all()):
                    health["finite"] = False
                    raise RuntimeError("nonfinite robot articulation")
            if previous is not None and vertices.shape == previous.shape:
                health["max_particle_step_m"] = max(health["max_particle_step_m"],
                    float(np.linalg.norm(vertices - previous, axis=1).max()))
            previous = vertices.copy()
            health["steps"] += 1
            return out

        env.step = checked_step
        try:
            metrics = original(env, policy, config, **kwargs)
        finally:
            env.step = step_fn
        if len(metrics) != args.episodes:
            raise RuntimeError(f"truncated official metrics for {name}")
        health["rendered_observations"] = observer.rendered_frames - initial_render_count
        if health["rendered_observations"] < health["steps"]:
            raise RuntimeError(f"not every step received a fresh camera observation: {health}")
        if args.max_steps >= 10 and health["max_particle_step_m"] < 1e-5:
            raise RuntimeError(f"cloth did not measurably simulate for {name}: {health}")
        records.append({"garment": name, "split": "unseen" if "_Unseen_" in name else "seen",
                        "physics_health": health, "episodes": metrics})
        result["rendered_observations"] = observer.rendered_frames
        write_result(args.out, result)
        return metrics

    evaluation.run_evaluation_loop = measured
    try:
        evaluation.eval(official, app)
        if [r["garment"] for r in records] != names:
            raise RuntimeError("official evaluator returned before the garment list was complete")
        result["completed"] = True
        for split in ("seen", "unseen"):
            episodes = [e for r in records if r["split"] == split for e in r["episodes"]]
            result[split] = {"episodes": len(episodes), "successes": sum(bool(e["success"]) for e in episodes),
                             "success_rate": float(np.mean([bool(e["success"]) for e in episodes])) if episodes else None}
        write_result(args.out, result)
        print(f"[fold] COMPLETE {args.out}", flush=True)
        return 0
    except Exception as exc:
        result["error"] = repr(exc)
        write_result(args.out, result)
        raise


if __name__ == "__main__":
    code = 1
    try:
        code = main()
    except Exception:
        traceback.print_exc()
    # Kit shutdown previously held GPUs for 39 min after writing results.
    # Metrics are atomically durable before exiting; exceptions remain failures.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
