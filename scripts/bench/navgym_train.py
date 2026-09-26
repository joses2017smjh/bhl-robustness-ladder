"""Train a maze-navigation policy on NavGym with Stable-Baselines3 PPO.

Curriculum on maze size (3x3 -> 4x4 -> 5x5 -> 6x6, promoted when the rolling
training success rate clears a bar), periodic held-out evaluation (seeds
>= 10 000, never trained on) to TensorBoard and JSON, checkpoints, and at the
end an ONNX export of the deterministic actor -- the deployable artifact that
`scripts/bench/maze_explore.py --policy` runs on the physics robot.

Defaults reproduce the v1 run (results/navgym-20260924): --env-version 1 --ppo v1.
NavGym v2 (2026-09-26) is selected explicitly:

  --env-version 2   MazeNavEnv(version=2): round footprint, route-scaled time
                    limit, -5 crash / -0.002 step / continuous potential shaping
                    with the learner's gamma, extra "near" lidar key.
  --ppo v2          target_kl 0.02, linear LR 3e-4 -> 3e-5, ent_coef 0.01,
                    log_std_init -0.5, a std floor (log_std clamped >= log 0.2
                    after every update), gamma 0.998.
  --ppo v1          the v1 settings (gamma 0.995, LR 3e-4 constant, ent 0.003,
                    no KL target, log_std_init 0, no floor) -- the ablation arm.

Every run: VecMonitor (rollout/ep_rew_mean, ep_len_mean), held-out success /
collision / time-out rates to TensorBoard and eval_history.json, the best
checkpoint by held-out 6x6 success (tie-break 5x5) kept in best/, and at the
end the final AND the best actor evaluated on 48 held-out mazes per size and
exported to ONNX (actor.onnx, actor_best.onnx), each checked against torch.
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
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv, VecMonitor  # noqa: E402

from bhl_robust.navgym.env import MazeNavEnv, MAP_CROP, heldout_env  # noqa: E402

CURRICULUM = [((3, 3),), ((3, 3), (4, 4)), ((4, 4), (5, 5)), ((5, 5), (6, 6))]
EVAL_SIZES = ((4, 4), (5, 5), (6, 6))
V2_PERIODIC_EVAL_SIZES = ((5, 5), (6, 6))      # v2 episodes are up to 3x longer; 4x4 only in the final eval
STD_FLOOR = 0.2

PPO_SETTINGS = {
    # v1: exactly the settings of results/navgym-20260924
    "v1": {"n_steps": 256, "batch_size": 1024, "n_epochs": 4, "lr": 3e-4, "lr_final": None, "gamma": 0.995, "gae_lambda": 0.95,
           "clip": 0.2, "ent_coef": 0.003, "target_kl": None, "log_std_init": 0.0, "std_floor": None},
    # v2: the post-mortem fixes for the std collapse / KL blow-up / saturated means
    "v2": {"n_steps": 256, "batch_size": 1024, "n_epochs": 4, "lr": 3e-4, "lr_final": 3e-5, "gamma": 0.998, "gae_lambda": 0.95,
           "clip": 0.2, "ent_coef": 0.01, "target_kl": 0.02, "log_std_init": -0.5, "std_floor": STD_FLOOR},
}


class NavExtractor(BaseFeaturesExtractor):
    """Small CNN on the 3x24x24 egocentric map, MLP on lidar + goal, concatenated."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),        # 12x12
            nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),        # 6x6
            nn.Flatten(), nn.Linear(32 * 6 * 6, 128), nn.ReLU())
        # vector keys in a fixed order, from the space: v1 (lidar, goal) builds exactly the v1 layers
        self.vec_keys = [k for k in ("lidar", "near", "goal") if k in observation_space.spaces]
        n_vec = sum(observation_space[k].shape[0] for k in self.vec_keys)
        self.mlp = nn.Sequential(nn.Linear(n_vec, 128), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(256, features_dim), nn.ReLU())

    def forward(self, obs):
        vec = torch.cat([obs[k] for k in getattr(self, "vec_keys", ("lidar", "goal"))], dim=1)
        return self.out(torch.cat([self.cnn(obs["map"]), self.mlp(vec)], dim=1))


def make_env(rank: int, seed: int, sizes, randomize: bool = True, version: int = 1, gamma: float = 0.998):
    def _init():
        if version == 1:
            env = MazeNavEnv(sizes=sizes, randomize_dynamics=randomize)
        else:
            env = MazeNavEnv(sizes=sizes, randomize_dynamics=randomize, version=version, gamma=gamma)
        env.reset(seed=seed + rank)
        return env
    return _init


def evaluate(model, sizes, seeds, deterministic: bool = True, version: int = 1, gamma: float = 0.998) -> dict:
    """Held-out mazes: one episode per seed per size (maze seed 10 000 + seed, dynamics from
    reset(seed=seed)); success = goal reached. v2 episodes run to the v2 route-scaled limit."""
    out = {}
    for sz in sizes:
        env = MazeNavEnv(sizes=(sz,), randomize_dynamics=True, seed_base=10_000, seed_span=1) if version == 1 else None
        succ, steps, coll, tout = 0, [], 0, 0
        for k, sd in enumerate(seeds):
            if version == 1:
                env.seed_base = 10_000 + sd
                obs, info = env.reset(seed=sd)
            else:
                env, obs, info = heldout_env(sz, sd, version=version, gamma=gamma)
            done = False
            while not done:
                a, _ = model.predict(obs, deterministic=deterministic)
                obs, r, term, trunc, info = env.step(a)
                done = term or trunc
            succ += info["outcome"] == "goal"
            coll += info["outcome"] == "collision"
            tout += info["outcome"] == "time_out"
            steps.append(env.t)
        out[f"{sz[0]}x{sz[1]}"] = {"success": succ / len(seeds), "collision": coll / len(seeds), "time_out": tout / len(seeds),
                                   "mean_steps": float(np.mean(steps)), "n": len(seeds)}
    return out


class HeldoutPool:
    """The same held-out episodes as `evaluate` (v2 env), stepped in parallel on a
    SubprocVecEnv of evaluation workers with batched deterministic actions; a worker
    that finishes takes the next episode. Used for v2 so the long v2 episodes do not
    stall training; v1 keeps the serial `evaluate` it was run with."""

    def __init__(self, n_workers: int, version: int, gamma: float):
        fns = [make_env(i, 0, ((4, 4),), version=version, gamma=gamma) for i in range(n_workers)]
        self.venv = SubprocVecEnv(fns, start_method="forkserver")
        self.n = n_workers

    def evaluate(self, model, sizes, seeds) -> dict:
        jobs = [(tuple(sz), int(sd)) for sz in sizes for sd in seeds]
        res = {}
        nxt = 0
        cur: list = [None] * self.n
        obs = None

        def assign(i):
            nonlocal nxt
            o = self.venv.env_method("heldout_reset", jobs[nxt][0], jobs[nxt][1], indices=[i])[0]
            cur[i] = jobs[nxt]
            nxt += 1
            return o

        first = {}
        for i in range(min(self.n, len(jobs))):
            first[i] = assign(i)
        # a batch observation: idle workers (if any) just repeat worker 0's first observation
        keys = list(first[0].keys())
        obs = {k: np.stack([first.get(i, first[0])[k] for i in range(self.n)]) for k in keys}
        while any(c is not None for c in cur):
            acts, _ = model.predict(obs, deterministic=True)
            obs, _, dones, infos = self.venv.step(acts)
            for i in range(self.n):
                if cur[i] is not None and dones[i]:
                    res[cur[i]] = (infos[i]["outcome"], int(infos[i]["t"]))
                    cur[i] = None
                    if nxt < len(jobs):
                        o = assign(i)
                        for k in keys:
                            obs[k][i] = o[k]
        out = {}
        for sz in sizes:
            R = [res[(tuple(sz), int(sd))] for sd in seeds]
            n = len(R)
            out[f"{sz[0]}x{sz[1]}"] = {"success": sum(o == "goal" for o, _ in R) / n, "collision": sum(o == "collision" for o, _ in R) / n,
                                       "time_out": sum(o == "time_out" for o, _ in R) / n, "mean_steps": float(np.mean([t for _, t in R])), "n": n}
        return out

    def close(self):
        self.venv.close()


def _best_key(ev: dict) -> tuple:
    """Best-checkpoint order: held-out 6x6 success, tie-break 5x5 success."""
    return (ev.get("6x6", {}).get("success", -1.0), ev.get("5x5", {}).get("success", -1.0))


class StdFloor(BaseCallback):
    """Clamp the Gaussian policy's log_std to >= log(floor) after every PPO update (the
    rollout-start hook runs right after train()). Records the pre-clamp std -- the value
    SB3 logged as train/std for that update -- so the run's std is judged unclamped."""

    def __init__(self, floor: float):
        super().__init__()
        self.log_floor = float(np.log(floor))
        self.last_std_pre_clamp = None
        self.clamps = 0

    def _clamp(self) -> None:
        ls = self.model.policy.log_std
        with torch.no_grad():
            self.last_std_pre_clamp = float(torch.exp(ls).mean())
            low = ls < self.log_floor
            if bool(low.any()):
                self.clamps += 1
                ls.clamp_(min=self.log_floor)
        self.logger.record("train/std_pre_floor", self.last_std_pre_clamp)
        self.logger.record("train/std_floor_clamps", self.clamps)

    def _on_rollout_start(self) -> None:
        if self.model.num_timesteps > 0:
            self._clamp()

    def _on_training_end(self) -> None:
        self._clamp()                      # the final update is not followed by a rollout

    def _on_step(self) -> bool:
        return True


class CurriculumAndEval(BaseCallback):
    def __init__(self, out: Path, eval_every: int, n_eval: int = 24, promote_at: float = 0.8, version: int = 1, gamma: float = 0.998,
                 eval_sizes=EVAL_SIZES, pool: HeldoutPool | None = None):
        super().__init__()
        self.pool = pool
        self.out, self.eval_every, self.n_eval, self.promote_at = out, eval_every, n_eval, promote_at
        self.version, self.gamma, self.eval_sizes = version, gamma, tuple(eval_sizes)
        self.stage, self.next_eval = 0, eval_every
        self.recent: list[bool] = []
        self.history: list[dict] = []
        self.best = None                   # {"timesteps", "eval", "key"}

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
            te = time.time()
            if self.pool is not None:
                ev = self.pool.evaluate(self.model, self.eval_sizes, list(range(self.n_eval)))
            else:
                ev = evaluate(self.model, self.eval_sizes, list(range(self.n_eval)), version=self.version, gamma=self.gamma)
            rec = {"timesteps": self.num_timesteps, "stage": self.stage, "train_success_recent": float(np.mean(self.recent)) if self.recent else None,
                   "eval": ev, "wall_s": round(time.time() - self._t0, 1), "eval_wall_s": round(time.time() - te, 1)}
            for k, v in ev.items():
                self.logger.record(f"eval_heldout/{k}_success", v["success"])
                self.logger.record(f"eval_heldout/{k}_collision", v["collision"])
                self.logger.record(f"eval_heldout/{k}_time_out", v["time_out"])
            key = _best_key(ev)
            if self.best is None or key > tuple(self.best["key"]):
                (self.out / "best").mkdir(parents=True, exist_ok=True)
                self.model.save(str(self.out / "best" / "ppo_best.zip"))
                self.best = {"timesteps": self.num_timesteps, "eval": ev, "key": list(key)}
                (self.out / "best" / "best.json").write_text(json.dumps(self.best, indent=2) + "\n")
                rec["new_best"] = True
            self.logger.record("eval_heldout/best_6x6_success", self.best["key"][0])
            self.history.append(rec)
            (self.out / "eval_history.json").write_text(json.dumps(self.history, indent=2) + "\n")
            print(f"[eval] {json.dumps(rec)}", flush=True)
        return True

    def _on_training_start(self) -> None:
        self._t0 = time.time()


class ActorOnnx(nn.Module):
    """Deterministic actor (mean action) for export: one input per observation key, one output.
    v1: (lidar, map, goal); v2: (lidar, near, map, goal)."""

    def __init__(self, policy, keys=("lidar", "map", "goal")):
        super().__init__()
        self.policy = policy
        self.keys = tuple(keys)

    def forward(self, *xs):
        obs = dict(zip(self.keys, xs))
        feats = self.policy.extract_features(obs, self.policy.pi_features_extractor)
        latent = self.policy.mlp_extractor.forward_actor(feats)
        return self.policy.action_net(latent)


def onnx_keys(observation_space) -> tuple:
    """ONNX input names/order for an observation space: the v1 order, plus "near" after "lidar" in v2."""
    return tuple(k for k in ("lidar", "near", "map", "goal") if k in observation_space.spaces)


def export_onnx(model, path: Path):
    keys = onnx_keys(model.observation_space)
    shapes = {k: tuple(model.observation_space[k].shape) for k in keys}
    actor = ActorOnnx(model.policy, keys).eval()
    dummy = tuple(torch.zeros((1,) + shapes[k]) for k in keys)
    torch.onnx.export(actor, dummy, str(path), input_names=list(keys), output_names=["action"],
                      dynamic_axes={**{k: {0: "n"} for k in keys}, "action": {0: "n"}}, opset_version=17)
    # check against the torch policy on random inputs
    import onnxruntime as ort
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    so.inter_op_num_threads = 1
    sess = ort.InferenceSession(str(path), so)
    rng = np.random.default_rng(0)
    feeds = {}
    for k in keys:
        if k == "map":
            feeds[k] = rng.integers(0, 2, (4,) + shapes[k]).astype(np.float32)
        elif k == "goal":
            feeds[k] = rng.uniform(-1, 1, (4,) + shapes[k]).astype(np.float32)
        else:
            feeds[k] = rng.uniform(0, 1, (4,) + shapes[k]).astype(np.float32)
    out = sess.run(None, feeds)[0]
    with torch.no_grad():
        ref = actor(*(torch.as_tensor(feeds[k]) for k in keys)).numpy()
    err = float(np.abs(out - ref).max())
    return {"onnx": str(path), "inputs": list(keys), "max_abs_diff_vs_torch": err}


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
    ap.add_argument("--env-version", type=int, choices=(1, 2), default=1, help="MazeNavEnv version (1 = the v1 run; 2 = NavGym v2)")
    ap.add_argument("--ppo", choices=tuple(PPO_SETTINGS), default="v1", help="PPO settings (v1 = the v1 run; v2 = the post-mortem fixes)")
    ap.add_argument("--n-eval", type=int, default=24, help="held-out seeds per size in the periodic evaluation")
    ap.add_argument("--n-final-eval", type=int, default=48)
    ap.add_argument("--eval-workers", type=int, default=None,
                    help="v2 only: parallel held-out evaluation workers (default = --n-envs; 0 = serial). v1 is always serial.")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(max(1, min(4, args.n_envs)))
    P = dict(PPO_SETTINGS[args.ppo])
    P["n_steps"] = args.n_steps
    gamma = P["gamma"]
    version = args.env_version
    # Shaping discount for the v2 env: 1.0, i.e. reward = metres of geodesic progress (the v1 form),
    # NOT the learner's gamma. With the learner's gamma (Ng-invariant form) a robot standing still far
    # from the goal is paid (1 - gamma) * geodesic per step and the goal bonus, discounted over the
    # ~2000-step routes, is worth only ~0.1 more than stalling (review of 2026-09-26). The progress form
    # pays nothing for stalling, cannot be farmed by cycling under a discounted objective, and gives
    # ~+6.6 discounted for a 20 m route against ~-1 for stalling and -5 for a crash.
    env_gamma = 1.0 if version == 2 else gamma
    fns = [make_env(i, args.seed * 1000, CURRICULUM[0], version=version, gamma=env_gamma) for i in range(args.n_envs)]
    venv = DummyVecEnv(fns) if args.dummy_vec else SubprocVecEnv(fns, start_method="forkserver")
    venv = VecMonitor(venv)                   # rollout/ep_rew_mean and ep_len_mean (logging only)
    if P["lr_final"] is None:
        lr = P["lr"]
    else:
        lr0, lr1 = P["lr"], P["lr_final"]
        lr = lambda progress_remaining: lr1 + (lr0 - lr1) * progress_remaining   # noqa: E731  linear lr0 -> lr1
    policy_kwargs = {"features_extractor_class": NavExtractor, "features_extractor_kwargs": {"features_dim": 256},
                     "net_arch": {"pi": [128, 128], "vf": [128, 128]}, "share_features_extractor": True}
    if P["log_std_init"] != 0.0:
        policy_kwargs["log_std_init"] = P["log_std_init"]
    model = PPO("MultiInputPolicy", venv, n_steps=P["n_steps"], batch_size=P["batch_size"], n_epochs=P["n_epochs"], learning_rate=lr, gamma=gamma,
                gae_lambda=P["gae_lambda"], clip_range=P["clip"], ent_coef=P["ent_coef"], vf_coef=0.5, max_grad_norm=1.0, target_kl=P["target_kl"],
                seed=args.seed, device="cpu", verbose=0, tensorboard_log=str(args.out / "tb"), policy_kwargs=policy_kwargs)
    eval_sizes = EVAL_SIZES if version == 1 else V2_PERIODIC_EVAL_SIZES
    (args.out / "config.json").write_text(json.dumps({**vars(args), "out": str(args.out), "curriculum": CURRICULUM, "eval_sizes": eval_sizes,
                                                       "final_eval_sizes": EVAL_SIZES, "env_version": version, "env_shaping_gamma": env_gamma if version == 2 else None,
                                                       "ppo": P}, indent=2, default=str) + "\n")
    n_eval_workers = 0 if version == 1 else (args.n_envs if args.eval_workers is None else args.eval_workers)
    pool = HeldoutPool(n_eval_workers, version, env_gamma) if n_eval_workers > 0 and not args.dummy_vec else None

    def heldout(m, n_seeds):
        if pool is not None:
            return pool.evaluate(m, EVAL_SIZES, list(range(n_seeds)))
        return evaluate(m, EVAL_SIZES, list(range(n_seeds)), version=version, gamma=env_gamma)

    cur = CurriculumAndEval(args.out, args.eval_every, n_eval=args.n_eval, version=version, gamma=env_gamma, eval_sizes=eval_sizes, pool=pool)
    cb = [cur, CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.n_envs), save_path=str(args.out / "ckpt"), name_prefix="ppo")]
    floor = None
    if P["std_floor"] is not None:
        floor = StdFloor(P["std_floor"])
        cb.append(floor)
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=False)
    model.save(str(args.out / "ppo_final.zip"))
    std_final = float(torch.exp(model.policy.log_std).mean())
    final = heldout(model, args.n_final_eval)
    onnx_info = export_onnx(model, args.out / "actor.onnx")
    best_info = None
    if (args.out / "best" / "ppo_best.zip").is_file():
        best_model = PPO.load(str(args.out / "best" / "ppo_best.zip"), device="cpu")
        best_info = {**cur.best, "note": "selected on held-out seeds 0-23 (a subset of the 48): its 48-maze numbers are selection-biased",
                     "final_eval_heldout_48": heldout(best_model, args.n_final_eval),
                     "onnx": export_onnx(best_model, args.out / "actor_best.onnx")}
    summary = {"total_steps": args.total_steps, "wall_hours": round((time.time() - t0) / 3600, 2), "final_eval_heldout_48": final,
               "curriculum_stage": cb[0].stage, "onnx": onnx_info, "env_version": version, "ppo": args.ppo,
               "train_std_last_update": (floor.last_std_pre_clamp if floor is not None else std_final), "policy_std_final": std_final,
               "std_floor_clamps": floor.clamps if floor is not None else None, "best": best_info,
               "eval_mode": "parallel_pool" if pool is not None else "serial"}
    if pool is not None:
        pool.close()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("NAVGYM SUMMARY " + json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
