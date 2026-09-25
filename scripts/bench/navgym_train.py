"""Train a maze-navigation policy on NavGym with Stable-Baselines3 PPO.

Curriculum on maze size (3x3 -> 4x4 -> 5x5 -> 6x6, promoted when the rolling
training success rate clears a bar), periodic held-out evaluation (seeds
>= 10 000, never trained on) to TensorBoard and JSON, checkpoints, and at the
end an ONNX export of the deterministic actor -- the deployable artifact that
`scripts/bench/maze_explore.py --policy` runs on the physics robot.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = HERE.parents[1]

import gymnasium as gym                                            # noqa: E402
from stable_baselines3 import PPO                                  # noqa: E402
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback  # noqa: E402
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor  # noqa: E402
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv  # noqa: E402

from bhl_robust.navgym.env import MazeNavEnv, MAP_CROP           # noqa: E402

CURRICULUM = [((3, 3),), ((3, 3), (4, 4)), ((4, 4), (5, 5)), ((5, 5), (6, 6))]
EVAL_SIZES = ((4, 4), (5, 5), (6, 6))


class NavExtractor(BaseFeaturesExtractor):
    """Small CNN on the 3x24x24 egocentric map, MLP on lidar + goal, concatenated."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),        # 12x12
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),        # 6x6
            nn.Flatten(), nn.Linear(32 * 6 * 6, 128), nn.ReLU())
        n_vec = observation_space["lidar"].shape[0] + observation_space["goal"].shape[0]
        self.mlp = nn.Sequential(nn.Linear(n_vec, 128), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(256, features_dim), nn.ReLU())

    def forward(self, obs):
        vec = torch.cat([obs["lidar"], obs["goal"]], dim=1)
        return self.out(torch.cat([self.cnn(obs["map"]), self.mlp(vec)], dim=1))


def make_env(rank: int, seed: int, sizes, randomize: bool = True):
    def _init():
        env = MazeNavEnv(sizes=sizes, randomize_dynamics=randomize)
        env.reset(seed=seed + rank)
        return env
    return _init


def evaluate(model, sizes, seeds, deterministic: bool = True) -> dict:
    """Held-out mazes: one episode per seed per size; success = goal reached."""
    out = {}
    for sz in sizes:
        env = MazeNavEnv(sizes=(sz,), randomize_dynamics=True, seed_base=10_000, seed_span=1)
        succ, steps, coll = 0, [], 0
        for k, sd in enumerate(seeds):
            env.seed_base = 10_000 + sd
            obs, info = env.reset(seed=sd)
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=deterministic)
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
            succ += info["outcome"] == "goal"
            coll += info["outcome"] == "collision"
            steps.append(env.t)
        out[f"{sz[0]}x{sz[1]}"] = {"success": succ / len(seeds), "collision": coll / len(seeds), "mean_steps": float(np.mean(steps)), "n": len(seeds)}
    return out


class CurriculumAndEval(BaseCallback):
    def __init__(self, out: Path, eval_every: int, n_eval: int = 24, promote_at: float = 0.8):
        super().__init__()
        self.out, self.eval_every, self.n_eval, self.promote_at = out, eval_every, n_eval, promote_at
        self.stage, self.next_eval = 0, eval_every
        self.recent: list[bool] = []
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            oc = info.get("outcome")
            if oc is not None:
                self.recent.append(oc == "goal")
        self.recent = self.recent[-400:]
        if len(self.recent) >= 200 and np.mean(self.recent) >= self.promote_at and self.stage < len(CURRICULUM) - 1:
            self.stage += 1
            self.recent = []
            self.training_env.env_method("set_sizes", CURRICULUM[self.stage])
            self.logger.record("curriculum/stage", self.stage)
            print(f"[curriculum] promoted to stage {self.stage}: sizes {CURRICULUM[self.stage]} at {self.num_timesteps} steps", flush=True)
        if self.num_timesteps >= self.next_eval:
            self.next_eval += self.eval_every
            ev = evaluate(self.model, EVAL_SIZES, list(range(self.n_eval)))
            rec = {"timesteps": self.num_timesteps, "stage": self.stage, "train_success_recent": float(np.mean(self.recent)) if self.recent else None,
                   "eval": ev, "wall_s": round(time.time() - self._t0, 1)}
            self.history.append(rec)
            for k, v in ev.items():
                self.logger.record(f"eval_heldout/{k}_success", v["success"])
            (self.out / "eval_history.json").write_text(json.dumps(self.history, indent=2) + "\n")
            print(f"[eval] {json.dumps(rec)}", flush=True)
        return True

    def _on_training_start(self) -> None:
        self._t0 = time.time()


class ActorOnnx(nn.Module):
    """Deterministic actor (mean action) for export: three inputs, one output."""

    def __init__(self, policy):
        super().__init__()
        self.policy = policy

    def forward(self, lidar, map_, goal):
        obs = {"lidar": lidar, "map": map_, "goal": goal}
        feats = self.policy.extract_features(obs, self.policy.pi_features_extractor)
        latent = self.policy.mlp_extractor.forward_actor(feats)
        return torch.tanh(self.policy.action_net(latent)) if False else self.policy.action_net(latent)


def export_onnx(model, path: Path):
    actor = ActorOnnx(model.policy).eval()
    dummy = (torch.zeros(1, 36), torch.zeros(1, 3, MAP_CROP, MAP_CROP), torch.zeros(1, 5))
    torch.onnx.export(actor, dummy, str(path), input_names=["lidar", "map", "goal"], output_names=["action"],
                      dynamic_axes={"lidar": {0: "n"}, "map": {0: "n"}, "goal": {0: "n"}, "action": {0: "n"}}, opset_version=17)
    # check against the torch policy on random inputs
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so)
    rng = np.random.default_rng(0)
    feeds = {"lidar": rng.uniform(0, 1, (4, 36)).astype(np.float32), "map": rng.integers(0, 2, (4, 3, MAP_CROP, MAP_CROP)).astype(np.float32),
             "goal": rng.uniform(-1, 1, (4, 5)).astype(np.float32)}
    out = sess.run(None, feeds)[0]
    with torch.no_grad():
        ref = actor(*(torch.as_tensor(feeds[k]) for k in ("lidar", "map", "goal"))).numpy()
    err = float(np.abs(out - ref).max())
    return {"onnx": str(path), "max_abs_diff_vs_torch": err}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total-steps", type=int, default=60_000_000)
    ap.add_argument("--n-envs", type=int, default=16)
    ap.add_argument("--n-steps", type=int, default=256)
    ap.add_argument("--eval-every", type=int, default=1_000_000)
    ap.add_argument("--checkpoint-every", type=int, default=2_000_000)
    ap.add_argument("--subproc", action="store_true", default=True)
    ap.add_argument("--dummy-vec", action="store_true", help="single-process vector env (debug)")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, args.n_envs)))
    fns = [make_env(i, args.seed * 1000, CURRICULUM[0]) for i in range(args.n_envs)]
    venv = DummyVecEnv(fns) if args.dummy_vec else SubprocVecEnv(fns, start_method="forkserver")
    model = PPO("MultiInputPolicy", venv, n_steps=args.n_steps, batch_size=1024, n_epochs=4, learning_rate=3e-4, gamma=0.995, gae_lambda=0.95,
                clip_range=0.2, ent_coef=0.003, vf_coef=0.5, max_grad_norm=1.0, seed=args.seed, device="cpu", verbose=0,
                tensorboard_log=str(args.out / "tb"),
                policy_kwargs={"features_extractor_class": NavExtractor, "features_extractor_kwargs": {"features_dim": 256},
                               "net_arch": {"pi": [128, 128], "vf": [128, 128]}, "share_features_extractor": True})
    (args.out / "config.json").write_text(json.dumps({**vars(args), "out": str(args.out), "curriculum": CURRICULUM, "eval_sizes": EVAL_SIZES,
                                                       "ppo": {"n_steps": args.n_steps, "batch_size": 1024, "n_epochs": 4, "lr": 3e-4, "gamma": 0.995,
                                                               "gae_lambda": 0.95, "clip": 0.2, "ent_coef": 0.003}}, indent=2, default=str) + "\n")
    cb = [CurriculumAndEval(args.out, args.eval_every), CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.n_envs), save_path=str(args.out / "ckpt"), name_prefix="ppo")]
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=False)
    model.save(str(args.out / "ppo_final.zip"))
    final = evaluate(model, EVAL_SIZES, list(range(48)))
    onnx_info = export_onnx(model, args.out / "actor.onnx")
    summary = {"total_steps": args.total_steps, "wall_hours": round((time.time() - t0) / 3600, 2), "final_eval_heldout_48": final,
               "curriculum_stage": cb[0].stage, "onnx": onnx_info}
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("NAVGYM SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
