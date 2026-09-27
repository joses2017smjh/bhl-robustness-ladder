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

NavGym v3 (2026-09-27) = the v2 env + --ppo v3:

  --ppo v3          v2 PPO plus ONE change: a std CEILING of 1.0 (log_std clamped
                    to [log 0.2, log 1.0] after every update, the floor's own hook).
                    Why (scripts/bench/navgym_diagnose.py on the v2 Arm A runs):
                    with ent_coef 0.01 and actions clipped to [-1, 1], the
                    pre-clamp std grew monotonically to 7-10 (vx) / 4.5 (wz); the
                    Gaussian means followed (median |mean| 25 / 6-12), so the
                    deterministic actor is bang-bang: median 91-96 % of wz means
                    outside [-1, 1], wz sign flips 15-20 per second. 11 of the 15
                    5x5 time-outs of the final actors are a dither in place
                    (< 3 cm/s over the last 20 s, never near a wall), the other 4
                    move but wander (net 0.04-0.07 m/s; the limit needs ~0.10);
                    the forward command is at
                    zero in a median 72 % of time-out steps vs 3-6 % on successes,
                    and 0/15 reach the goal with twice the time. The v2
                    checkpoints at std <= 1.15 (2 M, 4 M steps, both seeds) had wz
                    saturation 0.10-0.23, 1.4-2.7 flips/s and periodic 5x5
                    time-outs 0.00-0.125 (but more collisions).
  --final-eval-fresh-base 20000
                    also evaluate the final and the best actor on the FRESH held-out
                    set (maze seed 20 000 + k, dynamics reset(seed=20 000 + k)),
                    written as "final_eval_fresh_48"; the periodic evaluation and the
                    best-checkpoint selection stay on seeds 10 000..10 023.
  verdict-v3 ROOT   (`navgym_train.py verdict-v3 <root> --seeds 2 3 4`) applies the
                    predeclared v3 rule to <root>/armV3-s<seed>/summary.json.

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

from bhl_robust.navgym.env import FRESH_BASE, HELDOUT_BASE, MazeNavEnv, MAP_CROP, heldout_env  # noqa: E402

CURRICULUM = [((3, 3),), ((3, 3), (4, 4)), ((4, 4), (5, 5)), ((5, 5), (6, 6))]
EVAL_SIZES = ((4, 4), (5, 5), (6, 6))
V2_PERIODIC_EVAL_SIZES = ((5, 5), (6, 6))      # v2 episodes are up to 3x longer; 4x4 only in the final eval
STD_FLOOR = 0.2
STD_CEILING_V3 = 1.0

PPO_SETTINGS = {
    # v1: exactly the settings of results/navgym-20260924
    "v1": {"n_steps": 256, "batch_size": 1024, "n_epochs": 4, "lr": 3e-4, "lr_final": None, "gamma": 0.995, "gae_lambda": 0.95,
           "clip": 0.2, "ent_coef": 0.003, "target_kl": None, "log_std_init": 0.0, "std_floor": None},
    # v2: the post-mortem fixes for the std collapse / KL blow-up / saturated means
    "v2": {"n_steps": 256, "batch_size": 1024, "n_epochs": 4, "lr": 3e-4, "lr_final": 3e-5, "gamma": 0.998, "gae_lambda": 0.95,
           "clip": 0.2, "ent_coef": 0.01, "target_kl": 0.02, "log_std_init": -0.5, "std_floor": STD_FLOOR},
    # v3: v2 + a std ceiling (the diagnosis of 2026-09-27: std growth -> saturated bang-bang means -> dither time-outs)
    "v3": {"n_steps": 256, "batch_size": 1024, "n_epochs": 4, "lr": 3e-4, "lr_final": 3e-5, "gamma": 0.998, "gae_lambda": 0.95,
           "clip": 0.2, "ent_coef": 0.01, "target_kl": 0.02, "log_std_init": -0.5, "std_floor": STD_FLOOR, "std_ceiling": STD_CEILING_V3},
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


def evaluate(model, sizes, seeds, deterministic: bool = True, version: int = 1, gamma: float = 0.998, maze_base: int = HELDOUT_BASE) -> dict:
    """Held-out mazes: one episode per seed per size (maze seed 10 000 + seed, dynamics from
    reset(seed=seed)); success = goal reached. v2 episodes run to the v2 route-scaled limit."""
    out = {}
    for sz in sizes:
        if version == 1 and maze_base != HELDOUT_BASE:
            raise ValueError("the v1 serial evaluation only knows the original held-out set")
        env = MazeNavEnv(sizes=(sz,), randomize_dynamics=True, seed_base=10_000, seed_span=1) if version == 1 else None
        succ, steps, coll, tout = 0, [], 0, 0
        for k, sd in enumerate(seeds):
            if version == 1:
                env.seed_base = 10_000 + sd
                obs, info = env.reset(seed=sd)
            else:
                env, obs, info = heldout_env(sz, sd, version=version, gamma=gamma, maze_base=maze_base)
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

    def evaluate(self, model, sizes, seeds, maze_base: int = HELDOUT_BASE) -> dict:
        """`maze_base` 10 000 (default) is the published held-out set; FRESH_BASE (20 000) the v3 scored set."""
        jobs = [(tuple(sz), int(sd)) for sz in sizes for sd in seeds]
        res = {}
        nxt = 0
        cur: list = [None] * self.n
        obs = None

        def assign(i):
            nonlocal nxt
            if maze_base == HELDOUT_BASE:
                o = self.venv.env_method("heldout_reset", jobs[nxt][0], jobs[nxt][1], indices=[i])[0]
            else:
                o = self.venv.env_method("heldout_reset", jobs[nxt][0], jobs[nxt][1], maze_base, indices=[i])[0]
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
    SB3 logged as train/std for that update -- so the run's std is judged unclamped.
    `ceiling` (v3 only; None = v2 exactly) also clamps log_std <= log(ceiling) in the same hook."""

    def __init__(self, floor: float, ceiling: float | None = None):
        super().__init__()
        self.log_floor = float(np.log(floor))
        self.log_ceiling = None if ceiling is None else float(np.log(ceiling))
        self.last_std_pre_clamp = None
        self.last_std_pre_clamp_per_dim = None
        self.clamps = 0
        self.ceiling_clamps = 0

    def _clamp(self) -> None:
        ls = self.model.policy.log_std
        with torch.no_grad():
            self.last_std_pre_clamp = float(torch.exp(ls).mean())
            self.last_std_pre_clamp_per_dim = [float(x) for x in torch.exp(ls)]
            low = ls < self.log_floor
            if bool(low.any()):
                self.clamps += 1
                ls.clamp_(min=self.log_floor)
            if self.log_ceiling is not None and bool((ls > self.log_ceiling).any()):
                self.ceiling_clamps += 1
                ls.clamp_(max=self.log_ceiling)
        self.logger.record("train/std_pre_floor", self.last_std_pre_clamp)
        self.logger.record("train/std_floor_clamps", self.clamps)
        if self.log_ceiling is not None:
            self.logger.record("train/std_ceiling_clamps", self.ceiling_clamps)

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
    ap.add_argument("--final-eval-fresh-base", type=int, default=None,
                    help="also evaluate the final and best actors on the fresh held-out set maze seed BASE + k (v3: 20000); "
                         "default None = only the original set (10 000 + k), as every earlier run")
    args = ap.parse_args()
    if args.final_eval_fresh_base is not None and args.final_eval_fresh_base < 10_000 + 1_000:
        ap.error("--final-eval-fresh-base must be clear of the training seeds (< 10 000) and the original held-out set (10 000 + k)")
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

    def heldout(m, n_seeds, maze_base=HELDOUT_BASE):
        if pool is not None:
            return pool.evaluate(m, EVAL_SIZES, list(range(n_seeds)), maze_base=maze_base)
        return evaluate(m, EVAL_SIZES, list(range(n_seeds)), version=version, gamma=env_gamma, maze_base=maze_base)

    cur = CurriculumAndEval(args.out, args.eval_every, n_eval=args.n_eval, version=version, gamma=env_gamma, eval_sizes=eval_sizes, pool=pool)
    cb = [cur, CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.n_envs), save_path=str(args.out / "ckpt"), name_prefix="ppo")]
    floor = None
    if P["std_floor"] is not None:
        floor = StdFloor(P["std_floor"], P.get("std_ceiling"))
        cb.append(floor)
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=False)
    model.save(str(args.out / "ppo_final.zip"))
    std_final = float(torch.exp(model.policy.log_std).mean())
    final = heldout(model, args.n_final_eval)
    fresh = args.final_eval_fresh_base
    final_fresh = heldout(model, args.n_final_eval, maze_base=fresh) if fresh is not None else None
    onnx_info = export_onnx(model, args.out / "actor.onnx")
    best_info = None
    if (args.out / "best" / "ppo_best.zip").is_file():
        best_model = PPO.load(str(args.out / "best" / "ppo_best.zip"), device="cpu")
        best_info = {**cur.best, "note": "selected on held-out seeds 0-23 (a subset of the 48): its 48-maze numbers are selection-biased",
                     "final_eval_heldout_48": heldout(best_model, args.n_final_eval),
                     "onnx": export_onnx(best_model, args.out / "actor_best.onnx")}
        if fresh is not None:
            best_info["final_eval_fresh_48"] = heldout(best_model, args.n_final_eval, maze_base=fresh)
    summary = {"total_steps": args.total_steps, "wall_hours": round((time.time() - t0) / 3600, 2), "final_eval_heldout_48": final,
               "curriculum_stage": cb[0].stage, "onnx": onnx_info, "env_version": version, "ppo": args.ppo,
               "train_std_last_update": (floor.last_std_pre_clamp if floor is not None else std_final), "policy_std_final": std_final,
               "std_floor_clamps": floor.clamps if floor is not None else None, "best": best_info,
               "eval_mode": "parallel_pool" if pool is not None else "serial"}
    if fresh is not None or P.get("std_ceiling") is not None:
        # v3 additions (absent from v1/v2 summaries, which stay as published)
        summary["num_timesteps"] = int(model.num_timesteps)
        summary["reached_last_step"] = bool(model.num_timesteps >= args.total_steps)
        summary["std_pre_clamp_per_dim_last_update"] = floor.last_std_pre_clamp_per_dim if floor is not None else None
        summary["std_ceiling"] = P.get("std_ceiling")
        summary["std_ceiling_clamps"] = floor.ceiling_clamps if floor is not None else None
        summary["policy_std_final_per_dim"] = [float(x) for x in torch.exp(model.policy.log_std.detach())]
    if fresh is not None:
        summary["final_eval_fresh_48"] = final_fresh
        summary["fresh_set"] = {"maze_seeds": f"{fresh} + k, k = 0..{args.n_final_eval - 1}", "dynamics": f"reset(seed={fresh} + k)",
                                "note": "never used for training, periodic evaluation, best-checkpoint selection or diagnosis"}
    if pool is not None:
        pool.close()
    (args.out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print("NAVGYM SUMMARY " + json.dumps(summary), flush=True)
    return 0


# ---------------------------------------------------------------- v3 verdict
V3_RULE = {"5x5_success_min": 0.80, "6x6_success_min": 0.70, "6x6_collision_max": 0.15, "n": 48, "total_steps": 30_000_000,
           "seeds": (2, 3, 4), "min_seeds_passing": 2, "set": "final_eval_fresh_48", "fresh_base": 20_000}


def judge_v3(summary: dict | None, rule: dict = V3_RULE) -> dict:
    """Per-seed checks of the predeclared v3 rule on one summary.json (None = no summary: the run did not finish)."""
    if summary is None:
        return {"meets": False, "checks": {"summary_exists": False}, "numbers": None}
    ev = summary.get(rule["set"]) or {}
    a, b = ev.get("5x5"), ev.get("6x6")
    fs = summary.get("fresh_set") or {}
    checks = {"summary_exists": True,
              "config_v3": summary.get("ppo") == "v3" and summary.get("env_version") == 2 and summary.get("std_ceiling") == STD_CEILING_V3,
              "reached_last_step": bool(summary.get("reached_last_step")) and int(summary.get("num_timesteps", 0)) >= rule["total_steps"]
              and int(summary.get("total_steps", 0)) == rule["total_steps"],
              "fresh_set_is_20000": str(fs.get("maze_seeds", "")).startswith(f"{rule['fresh_base']} + k"),
              "n48": bool(a and b and a["n"] == rule["n"] and b["n"] == rule["n"]),
              "5x5_success": bool(a and a["success"] >= rule["5x5_success_min"]),
              "6x6_success": bool(b and b["success"] >= rule["6x6_success_min"]),
              "6x6_collision": bool(b and b["collision"] <= rule["6x6_collision_max"])}
    nums = None
    if a and b:
        ho = summary.get("final_eval_heldout_48") or {}
        nums = {"fresh": {k: ev[k] for k in ("4x4", "5x5", "6x6") if k in ev},
                "heldout_10000_for_comparability": {k: ho[k] for k in ("4x4", "5x5", "6x6") if k in ho},
                "num_timesteps": summary.get("num_timesteps"), "std_pre_clamp_last": summary.get("std_pre_clamp_per_dim_last_update"),
                "std_ceiling_clamps": summary.get("std_ceiling_clamps"), "std_floor_clamps": summary.get("std_floor_clamps")}
        best = summary.get("best") or {}
        if best:
            nums["best_checkpoint_not_gated"] = {"timesteps": best.get("timesteps"), "fresh": best.get("final_eval_fresh_48"),
                                                 "heldout_10000_selection_biased": best.get("final_eval_heldout_48")}
    return {"meets": all(checks.values()), "checks": checks, "numbers": nums}


def verdict_v3(root: Path, seeds=V3_RULE["seeds"], rule: dict = V3_RULE, final: bool = False) -> dict:
    """The arm verdict. A seed without summary.json is PENDING while the array runs; with
    `final` (all tasks have ended) it is a run that did not reach its last step, i.e. a miss."""
    per = {}
    for sd in seeds:
        p = root / f"armV3-s{sd}" / "summary.json"
        per[str(sd)] = judge_v3(json.loads(p.read_text()) if p.is_file() else None, rule)
    n_pass = sum(v["meets"] for v in per.values())
    missing = 0 if final else sum(not v["checks"].get("summary_exists") for v in per.values())
    if n_pass >= rule["min_seeds_passing"]:
        verdict = "PASS"
    elif n_pass + missing >= rule["min_seeds_passing"]:
        verdict = "PENDING"
    else:
        verdict = "NEGATIVE"
    return {"rule": {**rule, "seeds": list(rule["seeds"]),
                     "text": ("PASS iff >= 2 of 3 fresh seeds (2, 3, 4) meet every clause on the FRESH held-out set (maze seeds "
                              "20000-20047, final actor, deterministic): 5x5 success >= 0.80, 6x6 success >= 0.70, 6x6 collision "
                              "<= 0.15, n = 48 per size, and the run reached its last step (num_timesteps >= 30 000 000). "
                              "Seeds 10000-10047 are reported for comparability, not gated.")},
            "final": bool(final), "seeds_passing": n_pass, "verdict": verdict, "per_seed": per}


def verdict_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="navgym_train.py verdict-v3")
    ap.add_argument("root", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(V3_RULE["seeds"]))
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/verdict_v3.json")
    ap.add_argument("--final", action="store_true", help="every array task has ended: a missing summary counts as a miss")
    a = ap.parse_args(argv)
    v = verdict_v3(a.root, tuple(a.seeds), final=a.final)
    out = a.out or a.root / "verdict_v3.json"
    out.write_text(json.dumps(v, indent=2) + "\n")
    rd = json.loads(out.read_text())            # the printed verdict is recomputed from the file just written
    parts = []
    for sd, r in rd["per_seed"].items():
        n = r["numbers"]
        if n is None:
            parts.append(f"s{sd}: no summary")
        else:
            f5, f6 = n["fresh"]["5x5"], n["fresh"]["6x6"]
            h = n["heldout_10000_for_comparability"]
            cmp_ = (f" [10000-set 5x5 {h['5x5']['success']:.3f} 6x6 {h['6x6']['success']:.3f}]" if "5x5" in h and "6x6" in h else "")
            parts.append(f"s{sd}: {'MEETS' if r['meets'] else 'misses'} fresh 5x5 {f5['success']:.3f} 6x6 {f6['success']:.3f} "
                         f"(coll {f6['collision']:.3f}; last step {r['checks']['reached_last_step']}){cmp_}")
    print(f"NAVGYM-V3 VERDICT: {rd['verdict']} ({rd['seeds_passing']}/{len(rd['per_seed'])} seeds meet the bar) | " + " | ".join(parts), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verdict-v3":
        sys.exit(verdict_main(sys.argv[2:]))
    sys.exit(main())
