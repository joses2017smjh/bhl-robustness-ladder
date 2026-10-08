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

NavGym v4 (2026-09-28) = v2 exactly (--env-version 2 --ppo v2) + ONE change that prices stalling:

  --idle-cost 0.008 MazeNavEnv(version=2, idle_cost=0.008) in the TRAINING envs: an extra -0.008 on
                    every step whose commanded forward action a[0] <= -0.9 (v_cmd <= 5 % of V_MAX)
                    while the straight-line goal distance at the start of the step is > 0.5 m. An
                    endless idle stall is then worth (-0.002 - 0.008) / (1 - 0.998) = -5.0 (the crash)
                    instead of -1.0 (SB3 bootstraps truncations). Chosen over a raised step cost
                    (-0.010) by a pilot on training-range mazes (slurm/repo20260923/cpu_navgym_v4.sbatch
                    header). Default None = off: no env keyword, no config key, no summary key.
                    The held-out evaluations are unaffected (the idle cost changes no outcome).
                    Logged per rollout as stall/idle_charged_frac; summary.json "stall_price".
  verdict-v4 ROOT   applies the predeclared v4 rule (maze seeds 40000-40047) to <root>/armV4-s<seed>/.
  reading-v2ctl ROOT
                    the report-only reading of the v2 matched-seed control (v2 recipe on seeds 2-4)
                    against the v3 twins (cpu_navgym_v2ctl.sbatch header).

NavGym v5 (2026-10-01, N2) = the v4 configuration exactly + env version 3 (bhl_robust/navgym/env.py docstring):

  --env-version 3   v2's dynamics and rewards + a visitation channel (stacked onto the ego map: "map_visit"),
                    a coarse 0.6 m wide map ("map_coarse"), a yaw-change penalty and the deployment speed
                    brake; the policy gets NavExtractorV5 (a second map CNN). With --ppo v2 --idle-cost 0.008.
                    summary.json adds "v5_config" (the configuration as built) and "yaw_stats_not_gated".
  verdict-v5 ROOT   the predeclared v5 gym gate (maze seeds 61000-61047) on <root>/armV5-s<seed>/. Only
                    --final writes <root>/verdict_v5.json (once); without --final it is a status check that
                    prints, or writes only to an --out file not named verdict_v5.json.
  verdict-v5-transfer OUT --gym-verdict ROOT/verdict_v5.json
                    the predeclared physics-transfer rule (maze seeds 63000-63011) on the gate-passing actors;
                    reads only a FINAL gym verdict; only --final (every episode run has ended) writes
                    <out>/verdict.json (once).

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
from bhl_robust.navgym.env import V2_STEP_COST, V4_IDLE_A0_MAX, V4_IDLE_COST, V4_IDLE_FAR_M  # noqa: E402
from bhl_robust.navgym import env as navenv                       # noqa: E402  (NavGym v5 constants, read by v5_config)

CURRICULUM = [((3, 3),), ((3, 3), (4, 4)), ((4, 4), (5, 5)), ((5, 5), (6, 6))]
EVAL_SIZES = ((4, 4), (5, 5), (6, 6))
V2_PERIODIC_EVAL_SIZES = ((5, 5), (6, 6))      # v2 episodes are up to 3x longer; 4x4 only in the final eval
STD_FLOOR = 0.2
STD_CEILING_V3 = 1.0
# NavGym v4's stall price, as summary.json records it (the v4 verdict requires exactly this)
V4_STALL_PRICE = {"lever": "idle_cost", "idle_cost": V4_IDLE_COST, "idle_a0_max": V4_IDLE_A0_MAX, "idle_far_m": V4_IDLE_FAR_M,
                  "step_cost": V2_STEP_COST}


def stall_price(idle_cost: float) -> dict:
    return {"lever": "idle_cost", "idle_cost": float(idle_cost), "idle_a0_max": V4_IDLE_A0_MAX, "idle_far_m": V4_IDLE_FAR_M,
            "step_cost": V2_STEP_COST}

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


def make_env(rank: int, seed: int, sizes, randomize: bool = True, version: int = 1, gamma: float = 0.998, env_kwargs: dict | None = None):
    """`env_kwargs` (v2 only; None = the v2 env exactly) are extra MazeNavEnv keywords, e.g. the v4 idle_cost."""
    def _init():
        if version == 1:
            env = MazeNavEnv(sizes=sizes, randomize_dynamics=randomize)
        elif env_kwargs:
            env = MazeNavEnv(sizes=sizes, randomize_dynamics=randomize, version=version, gamma=gamma, **env_kwargs)
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
        self.final_infos = {}               # each job's last info (read by the v5 yaw statistics; changes no result)
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
                    self.final_infos[cur[i]] = {k: v for k, v in infos[i].items() if k != "terminal_observation"}
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


class IdleMonitor(BaseCallback):
    """v4 only (added when --idle-cost is set): fraction of training steps the env charged the idle cost
    (info["idle_charged"]), logged once per rollout as stall/idle_charged_frac. Reads infos only."""

    def __init__(self):
        super().__init__()
        self.n = self.k = 0
        self.total_n = self.total_k = 0

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "idle_charged" in info:
                self.n += 1
                self.k += bool(info["idle_charged"])
        return True

    def _on_rollout_end(self) -> None:
        if self.n:
            self.logger.record("stall/idle_charged_frac", self.k / self.n)
        self.total_n += self.n
        self.total_k += self.k
        self.n = self.k = 0


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
    """ONNX input names/order for an observation space: the v1 order, plus "near" after "lidar" in v2; v5 (env
    version 3): lidar, near, map_visit, map_coarse, goal (= env.OBS_KEYS_V3)."""
    return tuple(k for k in ("lidar", "near", "map", "map_visit", "map_coarse", "goal") if k in observation_space.spaces)


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


def build_parser() -> argparse.ArgumentParser:
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
    ap.add_argument("--env-version", type=int, choices=(1, 2, 3), default=1,
                    help="MazeNavEnv version (1 = the v1 run; 2 = NavGym v2; 3 = NavGym v5: v2 + visitation and coarse map "
                         "channels, yaw-change penalty, speed brake, with the v5 extractor)")
    ap.add_argument("--ppo", choices=tuple(PPO_SETTINGS), default="v1", help="PPO settings (v1 = the v1 run; v2 = the post-mortem fixes)")
    ap.add_argument("--n-eval", type=int, default=24, help="held-out seeds per size in the periodic evaluation")
    ap.add_argument("--n-final-eval", type=int, default=48)
    ap.add_argument("--eval-workers", type=int, default=None,
                    help="v2 only: parallel held-out evaluation workers (default = --n-envs; 0 = serial). v1 is always serial.")
    ap.add_argument("--final-eval-fresh-base", type=int, default=None,
                    help="also evaluate the final and best actors on the fresh held-out set maze seed BASE + k (v3: 20000); "
                         "default None = only the original set (10 000 + k), as every earlier run")
    ap.add_argument("--idle-cost", type=float, default=None,
                    help="v4 stall price (env v2 only; v4: 0.008): subtract this on training steps with forward command ~0 "
                         "(a[0] <= -0.9) more than 0.5 m from the goal; default None = off, as every earlier run")
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if args.final_eval_fresh_base is not None and args.final_eval_fresh_base < 10_000 + 1_000:
        ap.error("--final-eval-fresh-base must be clear of the training seeds (< 10 000) and the original held-out set (10 000 + k)")
    if args.idle_cost is not None and (args.env_version not in (2, 3) or not args.idle_cost > 0):
        ap.error("--idle-cost needs --env-version 2 or 3 and a value > 0")
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
    env_gamma = 1.0 if version in (2, 3) else gamma      # v5 (env version 3) keeps v2's progress-form shaping
    env_kwargs = {"idle_cost": args.idle_cost} if args.idle_cost is not None else None
    fns = [make_env(i, args.seed * 1000, CURRICULUM[0], version=version, gamma=env_gamma, env_kwargs=env_kwargs) for i in range(args.n_envs)]
    venv = DummyVecEnv(fns) if args.dummy_vec else SubprocVecEnv(fns, start_method="forkserver")
    venv = VecMonitor(venv)                   # rollout/ep_rew_mean and ep_len_mean (logging only)
    if P["lr_final"] is None:
        lr = P["lr"]
    else:
        lr0, lr1 = P["lr"], P["lr_final"]
        lr = lambda progress_remaining: lr1 + (lr0 - lr1) * progress_remaining   # noqa: E731  linear lr0 -> lr1
    policy_kwargs = {"features_extractor_class": NavExtractorV5 if version == 3 else NavExtractor, "features_extractor_kwargs": {"features_dim": 256},
                     "net_arch": {"pi": [128, 128], "vf": [128, 128]}, "share_features_extractor": True}
    if P["log_std_init"] != 0.0:
        policy_kwargs["log_std_init"] = P["log_std_init"]
    model = PPO("MultiInputPolicy", venv, n_steps=P["n_steps"], batch_size=P["batch_size"], n_epochs=P["n_epochs"], learning_rate=lr, gamma=gamma,
                gae_lambda=P["gae_lambda"], clip_range=P["clip"], ent_coef=P["ent_coef"], vf_coef=0.5, max_grad_norm=1.0, target_kl=P["target_kl"],
                seed=args.seed, device="cpu", verbose=0, tensorboard_log=str(args.out / "tb"), policy_kwargs=policy_kwargs)
    eval_sizes = EVAL_SIZES if version == 1 else V2_PERIODIC_EVAL_SIZES
    cfg_args = dict(vars(args))
    if cfg_args.get("idle_cost") is None:
        cfg_args.pop("idle_cost", None)         # off: config.json exactly as before the flag existed
    (args.out / "config.json").write_text(json.dumps({**cfg_args, "out": str(args.out), "curriculum": CURRICULUM, "eval_sizes": eval_sizes,
                                                       "final_eval_sizes": EVAL_SIZES, "env_version": version, "env_shaping_gamma": env_gamma if version in (2, 3) else None,
                                                       "ppo": P, **({"stall_price": stall_price(args.idle_cost)} if args.idle_cost is not None else {}),
                                                       **({"v5_config": v5_config(model.observation_space, type(model.policy.features_extractor).__name__)} if version == 3 else {})},
                                                      indent=2, default=str) + "\n")
    n_eval_workers = 0 if version == 1 else (args.n_envs if args.eval_workers is None else args.eval_workers)
    pool = HeldoutPool(n_eval_workers, version, env_gamma) if n_eval_workers > 0 and not args.dummy_vec else None

    yaw_reports = {}                       # v5 only: yaw statistics of the final evaluations (reported, not gated)

    def heldout(m, n_seeds, maze_base=HELDOUT_BASE, tag=None):
        if pool is not None:
            ev = pool.evaluate(m, EVAL_SIZES, list(range(n_seeds)), maze_base=maze_base)
            if version == 3 and tag is not None:
                yaw_reports[tag] = yaw_stats(pool.final_infos)
            return ev
        return evaluate(m, EVAL_SIZES, list(range(n_seeds)), version=version, gamma=env_gamma, maze_base=maze_base)

    cur = CurriculumAndEval(args.out, args.eval_every, n_eval=args.n_eval, version=version, gamma=env_gamma, eval_sizes=eval_sizes, pool=pool)
    cb = [cur, CheckpointCallback(save_freq=max(1, args.checkpoint_every // args.n_envs), save_path=str(args.out / "ckpt"), name_prefix="ppo")]
    floor = None
    if P["std_floor"] is not None:
        floor = StdFloor(P["std_floor"], P.get("std_ceiling"))
        cb.append(floor)
    idle_mon = None
    if args.idle_cost is not None:
        idle_mon = IdleMonitor()
        cb.append(idle_mon)
    t0 = time.time()
    model.learn(total_timesteps=args.total_steps, callback=cb, progress_bar=False)
    model.save(str(args.out / "ppo_final.zip"))
    std_final = float(torch.exp(model.policy.log_std).mean())
    final = heldout(model, args.n_final_eval, tag="final_actor_heldout_10000")
    fresh = args.final_eval_fresh_base
    final_fresh = heldout(model, args.n_final_eval, maze_base=fresh, tag="final_actor_fresh") if fresh is not None else None
    onnx_info = export_onnx(model, args.out / "actor.onnx")
    best_info = None
    if (args.out / "best" / "ppo_best.zip").is_file():
        best_model = PPO.load(str(args.out / "best" / "ppo_best.zip"), device="cpu")
        best_info = {**cur.best, "note": "selected on held-out seeds 0-23 (a subset of the 48): its 48-maze numbers are selection-biased",
                     "final_eval_heldout_48": heldout(best_model, args.n_final_eval, tag="best_actor_heldout_10000"),
                     "onnx": export_onnx(best_model, args.out / "actor_best.onnx")}
        if fresh is not None:
            best_info["final_eval_fresh_48"] = heldout(best_model, args.n_final_eval, maze_base=fresh, tag="best_actor_fresh")
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
    if args.idle_cost is not None:
        # v4 addition (absent from every run without --idle-cost)
        summary["stall_price"] = stall_price(args.idle_cost)
        summary["idle_charged_frac_training"] = (idle_mon.total_k / idle_mon.total_n) if idle_mon.total_n else None
    if version == 3:
        # v5 additions (absent from every v1/v2 run): the declared configuration as built, and the yaw statistics
        summary["v5_config"] = v5_config(model.observation_space, type(model.policy.features_extractor).__name__)
        summary["yaw_stats_not_gated"] = {"note": V5_YAW_NOTE, **yaw_reports}
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


# ---------------------------------------------------------------- v2 matched-seed control (2026-09-28): REPORT-ONLY reading
V2CTL_READING = {"seeds": (2, 3, 4), "own_dir": "armV2C-s{}", "twin_dir": "armV3-s{}", "set": "final_eval_heldout_48",
                 "never_used_set": "final_eval_fresh_48", "never_used_base": 40_000, "total_steps": 30_000_000, "n": 48,
                 "match_rel_tol": 1e-6, "b_seed": 2, "b_not_caused_below": 0.30, "b_caused_at": 0.60,
                 "c_raised_by": 0.10, "c_within": 0.05, "c_min_seeds": 2}
V2CTL_TEXT = ("REPORT-ONLY reading (not a gate), predeclared 2026-09-28 before any v2ctl run. v2ctl = the v2 recipe (--ppo v2 "
              "--env-version 2) on training seeds 2, 3, 4, each the twin of the v3 run with the same seed. "
              "(a) determinism: if the v2ctl run's rollout/ep_rew_mean equals its v3 twin's (relative tolerance 1e-6) at every "
              "point logged at a step strictly below that twin's first logged std-ceiling clamp, the pair is a SINGLE-FACTOR "
              "comparison, else a SEED-MATCHED REPLICATE (first divergence reported). (b) s2: v2ctl-s2 final 5x5 success on the "
              "10000-set < 0.30 -> the s2 collapse is not caused by the ceiling; >= 0.60 -> the ceiling caused or deepened it; "
              "else inconclusive. (c) on the 10000-set: >= 2/3 v2ctl seeds with 6x6 collision at least 0.10 lower than the v3 "
              "twin -> the ceiling raised collisions; >= 2/3 within 0.05 -> it did not; else inconclusive. The never-used set "
              "(maze seeds 40000-40047) and a 5-seed v2 estimate (Arm A s0, s1 + v2ctl s2-s4, 10000-set) are reported, not read.")


def _tb_scalars(run: Path, tags) -> dict:
    """{tag: [(step, value), ...] or None} for TensorBoard scalars of a navgym_train.py run (the newest tb/PPO_<n>
    directory, i.e. the last start of the run), each event file read once."""
    dirs = sorted((d for d in (run / "tb").glob("PPO_*") if d.is_dir()), key=lambda d: int(d.name.split("_")[-1]))
    if not dirs:
        return {t: None for t in tags}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
    ea = EventAccumulator(str(dirs[-1]), size_guidance={"scalars": 0})
    ea.Reload()
    have = set(ea.Tags().get("scalars", []))
    return {t: ([(int(e.step), float(e.value)) for e in ea.Scalars(t)] if t in have else None) for t in tags}


def determinism_check(own: list | None, twin: list | None, twin_clamps: list | None, rel_tol: float = 1e-6) -> dict:
    """Reading (a): own vs twin rollout/ep_rew_mean at every step strictly below the twin's first ceiling clamp.
    The clamp is applied at a rollout start and logged with that rollout's dump, so the first logged clamp step S is
    the first point whose rollout can differ; every point < S comes from identical computation if training is deterministic."""
    if own is None or twin is None or twin_clamps is None:
        return {"result": "no result", "why": "missing TensorBoard data"}
    first = next((s for s, v in twin_clamps if v > 0), None)
    if first is None:
        return {"result": "no result", "why": "the twin never clamped"}
    a = {s: v for s, v in own if s < first}
    b = {s: v for s, v in twin if s < first}
    steps = sorted(set(a) | set(b))
    div = None
    for s in steps:
        if s not in a or s not in b or abs(a[s] - b[s]) > rel_tol * max(1.0, abs(b[s])):
            div = s
            break
    same = div is None and len(steps) > 0
    return {"result": "single-factor comparison" if same else "seed-matched replicate", "first_ceiling_clamp_step": first,
            "points_compared": len(steps), "first_divergence_step": div,
            "max_abs_diff": max((abs(a[s] - b[s]) for s in steps if s in a and s in b), default=None)}


def _v2ctl_config_ok(s: dict, rule: dict = V2CTL_READING) -> dict:
    ev, fr = s.get(rule["set"]) or {}, s.get(rule["never_used_set"]) or {}
    fs = s.get("fresh_set") or {}
    return {"config_v2": s.get("ppo") == "v2" and s.get("env_version") == 2 and s.get("std_ceiling") is None,
            "reached_last_step": bool(s.get("reached_last_step")) and int(s.get("num_timesteps", 0)) >= rule["total_steps"]
            and int(s.get("total_steps", 0)) == rule["total_steps"],
            "n48": all((ev.get(k) or {}).get("n") == rule["n"] for k in ("5x5", "6x6")),
            "never_used_set_is_40000": str(fs.get("maze_seeds", "")).startswith(f"{rule['never_used_base']} + k")
            and all((fr.get(k) or {}).get("n") == rule["n"] for k in ("5x5", "6x6"))}


def reading_v2ctl(root: Path, v3_root: Path, v2_root: Path | None = None, rule: dict = V2CTL_READING, tb: bool = True) -> dict:
    """The predeclared report-only reading of the v2 matched-seed control (V2CTL_TEXT)."""
    load = lambda p: json.loads(p.read_text()) if p.is_file() else None   # noqa: E731
    per = {}
    for sd in rule["seeds"]:
        own_dir, twin_dir = root / rule["own_dir"].format(sd), v3_root / rule["twin_dir"].format(sd)
        s, t = load(own_dir / "summary.json"), load(twin_dir / "summary.json")
        r = {"summary_exists": s is not None, "twin_summary_exists": t is not None}
        if s is not None:
            r["checks"] = _v2ctl_config_ok(s, rule)
            r["valid"] = all(r["checks"].values())
            r["heldout_10000"] = {k: s[rule["set"]][k] for k in ("4x4", "5x5", "6x6") if k in s.get(rule["set"], {})}
            r["never_used_40000_not_read"] = {k: v for k, v in (s.get(rule["never_used_set"]) or {}).items()}
        if t is not None:
            r["twin_heldout_10000"] = {k: t[rule["set"]][k] for k in ("4x4", "5x5", "6x6") if k in t.get(rule["set"], {})}
        if tb:
            mine = _tb_scalars(own_dir, ("rollout/ep_rew_mean",))
            twin = _tb_scalars(twin_dir, ("rollout/ep_rew_mean", "train/std_ceiling_clamps"))
            r["a_determinism"] = determinism_check(mine["rollout/ep_rew_mean"], twin["rollout/ep_rew_mean"],
                                                   twin["train/std_ceiling_clamps"], rule["match_rel_tol"])
        if r.get("valid") and t is not None:
            d = t[rule["set"]]["6x6"]["collision"] - s[rule["set"]]["6x6"]["collision"]
            r["c_twin_minus_own_6x6_collision"] = d
            r["c_class"] = "lower by >= 0.10" if d >= rule["c_raised_by"] else ("within 0.05" if abs(d) <= rule["c_within"] else "neither")
        per[str(sd)] = r
    # (b)
    b = per[str(rule["b_seed"])]
    if b.get("valid"):
        x = b["heldout_10000"]["5x5"]["success"]
        b_read = ("the collapse is not caused by the ceiling" if x < rule["b_not_caused_below"] else
                  "the ceiling caused or deepened it" if x >= rule["b_caused_at"] else "inconclusive")
        b_out = {"v2ctl_s2_5x5_success_10000": x, "reading": b_read}
    else:
        b_out = {"reading": "no result (v2ctl-s2 has no valid summary)"}
    # (c)
    n_low = sum(r.get("c_class") == "lower by >= 0.10" for r in per.values())
    n_within = sum(r.get("c_class") == "within 0.05" for r in per.values())
    n_pairs = sum("c_class" in r for r in per.values())
    c_read = ("the ceiling raised collisions" if n_low >= rule["c_min_seeds"] else
              "the ceiling did not raise collisions" if n_within >= rule["c_min_seeds"] else
              "inconclusive" if n_pairs >= rule["c_min_seeds"] else f"no result ({n_pairs} valid pairs)")
    # 5-seed v2 estimate (10000-set, final actors; not a reading)
    est = {}
    if v2_root is not None:
        for sd in (0, 1):
            s = load(v2_root / f"armA-s{sd}" / "summary.json")
            if s is not None:
                est[f"armA-s{sd}"] = {k: s["final_eval_heldout_48"][k] for k in ("5x5", "6x6")}
    for sd in rule["seeds"]:
        if per[str(sd)].get("valid"):
            est[f"v2ctl-s{sd}"] = {k: per[str(sd)]["heldout_10000"][k] for k in ("5x5", "6x6")}
    pooled = None
    if est:
        pooled = {"n_seeds": len(est),
                  **{f"{sz}_{m}_mean": float(np.mean([v[sz][m] for v in est.values()])) for sz in ("5x5", "6x6") for m in ("success", "collision", "time_out")}}
    return {"rule": {**rule, "seeds": list(rule["seeds"]), "text": V2CTL_TEXT}, "label": "learned actors (deterministic mean); pose and goal are oracle inputs",
            "per_seed": per, "a": {str(sd): per[str(sd)].get("a_determinism") for sd in rule["seeds"]}, "b": b_out,
            "c": {"n_lower_by_0.10": n_low, "n_within_0.05": n_within, "n_valid_pairs": n_pairs, "reading": c_read},
            "v2_five_seed_estimate_10000_not_read": {"per_seed": est, "pooled": pooled}}


def reading_v2ctl_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="navgym_train.py reading-v2ctl")
    ap.add_argument("root", type=Path)
    ap.add_argument("--v3-root", type=Path, default=REPO / "results" / "navgym-v3-20260927")
    ap.add_argument("--v2-root", type=Path, default=REPO / "results" / "navgym-v2-20260926")
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/reading_v2ctl.json (never written into the v3 or v2 dirs)")
    a = ap.parse_args(argv)
    out = a.out or a.root / "reading_v2ctl.json"
    if out.resolve().parent in (a.v3_root.resolve(), a.v2_root.resolve()):
        ap.error("the reading is never written into the v3 or v2 result directories")
    v = reading_v2ctl(a.root, a.v3_root, a.v2_root)
    out.write_text(json.dumps(v, indent=2) + "\n")
    rd = json.loads(out.read_text())            # the printed reading is recomputed from the file just written
    parts = []
    for sd, r in rd["per_seed"].items():
        ad = r.get("a_determinism") or {}
        if not r.get("valid"):
            parts.append(f"s{sd}: {'no valid summary' if r.get('summary_exists') else 'no summary'} (a: {ad.get('result')})")
            continue
        h, tw = r["heldout_10000"], r.get("twin_heldout_10000") or {}
        nu = r.get("never_used_40000_not_read") or {}
        parts.append(f"s{sd}: 10000-set 5x5 {h['5x5']['success']:.3f} 6x6 {h['6x6']['success']:.3f} coll {h['6x6']['collision']:.3f}"
                     + (f" vs v3 twin coll {tw['6x6']['collision']:.3f}" if "6x6" in tw else "")
                     + (f" | 40000-set 5x5 {nu['5x5']['success']:.3f} 6x6 {nu['6x6']['success']:.3f} coll {nu['6x6']['collision']:.3f}" if "6x6" in nu else "")
                     + f" | (a) {ad.get('result')}")
    print(f"NAVGYM-V2CTL READING (report-only, not a gate; learned actors, oracle pose+goal): (b) {rd['b']['reading']} | "
          f"(c) {rd['c']['reading']} ({rd['c']['n_lower_by_0.10']} lower by >= 0.10, {rd['c']['n_within_0.05']} within 0.05) | "
          + " | ".join(parts), flush=True)
    return 0


# ---------------------------------------------------------------- v4 verdict (2026-09-28)
V4_RULE = {"5x5_success_min": 0.80, "6x6_success_min": 0.70, "6x6_collision_max": 0.15, "n": 48, "total_steps": 30_000_000,
           "seeds": (5, 6, 7), "min_seeds_passing": 2, "set": "final_eval_fresh_48", "fresh_base": 40_000}
V4_TEXT = ("PASS iff >= 2 of 3 NEW training seeds (5, 6, 7) meet every clause with their FINAL actor (deterministic) on the "
           "NEVER-USED set (maze seeds 40000-40047, dynamics reset(seed=40000 + k), v2 time limit): 5x5 success >= 0.80, 6x6 "
           "success >= 0.70, 6x6 collision <= 0.15, n = 48 per size, the run reached its last step (num_timesteps >= 30 000 000), "
           "and it is the declared v4 configuration (--ppo v2, --env-version 2, no std ceiling, the frozen stall price). "
           "Anything else is NEGATIVE. Seeds 10000-10047 and the best checkpoint are reported, not gated.")


def judge_v4(summary: dict | None, rule: dict = V4_RULE) -> dict:
    """Per-seed checks of the predeclared v4 rule on one summary.json (None = no summary: the run did not finish)."""
    if summary is None:
        return {"meets": False, "checks": {"summary_exists": False}, "numbers": None}
    ev = summary.get(rule["set"]) or {}
    a, b = ev.get("5x5"), ev.get("6x6")
    fs = summary.get("fresh_set") or {}
    checks = {"summary_exists": True,
              "config_v4": summary.get("ppo") == "v2" and summary.get("env_version") == 2 and summary.get("std_ceiling") is None
              and summary.get("stall_price") == V4_STALL_PRICE,
              "reached_last_step": bool(summary.get("reached_last_step")) and int(summary.get("num_timesteps", 0)) >= rule["total_steps"]
              and int(summary.get("total_steps", 0)) == rule["total_steps"],
              "never_used_set_is_40000": str(fs.get("maze_seeds", "")).startswith(f"{rule['fresh_base']} + k"),
              "n48": bool(a and b and a["n"] == rule["n"] and b["n"] == rule["n"]),
              "5x5_success": bool(a and a["success"] >= rule["5x5_success_min"]),
              "6x6_success": bool(b and b["success"] >= rule["6x6_success_min"]),
              "6x6_collision": bool(b and b["collision"] <= rule["6x6_collision_max"])}
    nums = None
    if a and b:
        ho = summary.get("final_eval_heldout_48") or {}
        nums = {"never_used_40000": {k: ev[k] for k in ("4x4", "5x5", "6x6") if k in ev},
                "heldout_10000_for_comparability": {k: ho[k] for k in ("4x4", "5x5", "6x6") if k in ho},
                "num_timesteps": summary.get("num_timesteps"), "stall_price": summary.get("stall_price"),
                "std_pre_clamp_last": summary.get("std_pre_clamp_per_dim_last_update"), "std_floor_clamps": summary.get("std_floor_clamps")}
        best = summary.get("best") or {}
        if best:
            nums["best_checkpoint_not_gated"] = {"timesteps": best.get("timesteps"), "never_used_40000": best.get("final_eval_fresh_48"),
                                                 "heldout_10000_selection_biased": best.get("final_eval_heldout_48")}
    return {"meets": all(checks.values()), "checks": checks, "numbers": nums}


def verdict_v4(root: Path, seeds=V4_RULE["seeds"], rule: dict = V4_RULE, final: bool = False) -> dict:
    """The v4 arm verdict (as verdict_v3): a missing summary is PENDING while the array runs, a miss with `final`."""
    per = {}
    for sd in seeds:
        p = root / f"armV4-s{sd}" / "summary.json"
        per[str(sd)] = judge_v4(json.loads(p.read_text()) if p.is_file() else None, rule)
    n_pass = sum(v["meets"] for v in per.values())
    missing = 0 if final else sum(not v["checks"].get("summary_exists") for v in per.values())
    if n_pass >= rule["min_seeds_passing"]:
        verdict = "PASS"
    elif n_pass + missing >= rule["min_seeds_passing"]:
        verdict = "PENDING"
    else:
        verdict = "NEGATIVE"
    return {"rule": {**rule, "seeds": list(rule["seeds"]), "stall_price": V4_STALL_PRICE, "text": V4_TEXT},
            "label": "learned actors (deterministic mean); pose and goal are oracle inputs",
            "final": bool(final), "seeds_passing": n_pass, "verdict": verdict, "per_seed": per}


def verdict_v4_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="navgym_train.py verdict-v4")
    ap.add_argument("root", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(V4_RULE["seeds"]))
    ap.add_argument("--out", type=Path, default=None, help="default: <root>/verdict_v4.json")
    ap.add_argument("--final", action="store_true", help="every array task has ended: a missing summary counts as a miss")
    a = ap.parse_args(argv)
    v = verdict_v4(a.root, tuple(a.seeds), final=a.final)
    out = a.out or a.root / "verdict_v4.json"
    out.write_text(json.dumps(v, indent=2) + "\n")
    rd = json.loads(out.read_text())            # the printed verdict is recomputed from the file just written
    parts = []
    for sd, r in rd["per_seed"].items():
        n = r["numbers"]
        if n is None:
            parts.append(f"s{sd}: " + ("summary without a 5x5 + 6x6 never-used evaluation (a miss)" if r["checks"].get("summary_exists")
                                       else "no summary"))
        else:
            f5, f6 = n["never_used_40000"]["5x5"], n["never_used_40000"]["6x6"]
            h = n["heldout_10000_for_comparability"]
            cmp_ = (f" [10000-set 5x5 {h['5x5']['success']:.3f} 6x6 {h['6x6']['success']:.3f}]" if "5x5" in h and "6x6" in h else "")
            bad = [k for k, ok in r["checks"].items() if not ok]
            parts.append(f"s{sd}: {'MEETS' if r['meets'] else 'misses'} 40000-set 5x5 {f5['success']:.3f} 6x6 {f6['success']:.3f} "
                         f"(coll {f6['collision']:.3f}; last step {r['checks']['reached_last_step']}"
                         + (f"; failed {bad}" if bad else "") + f"){cmp_}")
    print(f"NAVGYM-V4 VERDICT: {rd['verdict']} ({rd['seeds_passing']}/{len(rd['per_seed'])} seeds meet the bar; learned actors, "
          f"oracle pose+goal) | " + " | ".join(parts), flush=True)
    return 0


# ---------------------------------------------------------------- NavGym v5 (2026-10-01): N2, fully learned navigation
# v5 = the v4 configuration exactly (--env-version 3 keeps env v2's dynamics and rewards; --ppo v2; --idle-cost 0.008;
# 16 envs; 30 M steps; the v4 curriculum and periodic evaluation) PLUS, as env version 3 (bhl_robust/navgym/env.py), the
# visitation channel, the coarse wide map, the yaw-change penalty and the deployment speed brake -- and, because the
# observation has a second map input, the extractor below. Nothing else changes.
class NavExtractorV5(BaseFeaturesExtractor):
    """NavGym v5 (env version 3): NavExtractor's map CNN with its input widened to the 4-channel "map_visit" (occupied,
    free, unknown, visitation), an identical second CNN with its own weights for the 4-channel "map_coarse", and the same
    MLP on lidar + near + goal; the three 128-d embeddings (384) feed the same 256-d output layer. NavExtractor itself
    (v1-v4) is unchanged."""

    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256):
        super().__init__(observation_space, features_dim)

        def cnn(c):
            return nn.Sequential(
                nn.Conv2d(c, 16, 3, stride=1, padding=1), nn.ReLU(),
                nn.Conv2d(16, 32, 3, stride=2, padding=1), nn.ReLU(),        # 12x12
                nn.Conv2d(32, 32, 3, stride=2, padding=1), nn.ReLU(),        # 6x6
                nn.Flatten(), nn.Linear(32 * 6 * 6, 128), nn.ReLU())
        self.cnn = cnn(observation_space["map_visit"].shape[0])
        self.cnn_coarse = cnn(observation_space["map_coarse"].shape[0])
        self.vec_keys = [k for k in ("lidar", "near", "goal") if k in observation_space.spaces]
        n_vec = sum(observation_space[k].shape[0] for k in self.vec_keys)
        self.mlp = nn.Sequential(nn.Linear(n_vec, 128), nn.ReLU())
        self.out = nn.Sequential(nn.Linear(3 * 128, features_dim), nn.ReLU())

    def forward(self, obs):
        vec = torch.cat([obs[k] for k in self.vec_keys], dim=1)
        return self.out(torch.cat([self.cnn(obs["map_visit"]), self.cnn_coarse(obs["map_coarse"]), self.mlp(vec)], dim=1))


# The declared v5 configuration, FROZEN before any v5 training. summary.json["v5_config"] is rebuilt at run time from the
# env constants and the trained model (v5_config below); the gym gate requires it to equal this.
V5_CONFIG = {
    "env_version": 3,
    "obs_keys": ["lidar", "near", "map_visit", "map_coarse", "goal"],
    "map_visit_shape": [4, 24, 24], "map_coarse_shape": [4, 24, 24],
    "visitation": {"res_m": 0.2, "tau_s": 60.0, "radius_m": 0.22, "rule": "every step r *= exp(-dt / tau), then cells with centre within radius = 1"},
    "coarse_map": {"block": 3, "res_m": 0.6, "crop": 24, "pool": "block mean of occupied, free, unknown, visitation; outside = unknown"},
    "yaw_change_cost": 0.005,
    "brake": {"lo_m": 0.42, "span_m": 0.48, "half_angle_deg": 25.0, "period_s": 0.1},
    "extractor": "NavExtractorV5",
}
V5_YAW_NOTE = ("REPORTED, NOT GATED. Per maze size, pooled over the evaluation episodes of that set: yaw_flips_per_s = the number "
               "of steps whose clipped yaw action a_yaw has the opposite sign to the previous step's (sign product < 0) divided "
               "by the episodes' total duration (steps x 0.04 s); wz_saturation_share = the share of steps with |a_yaw| >= 0.99; "
               "braked_share = the share of steps the speed brake cut a non-zero forward command.")


def v5_config(observation_space, extractor_name: str) -> dict:
    """The v5 configuration as built: observation keys and shapes from the space, constants from the env module."""
    return {
        "env_version": 3,
        "obs_keys": list(onnx_keys(observation_space)),
        "map_visit_shape": list(observation_space["map_visit"].shape), "map_coarse_shape": list(observation_space["map_coarse"].shape),
        "visitation": {"res_m": navenv.MAP_RES, "tau_s": navenv.V3_VISIT_TAU_S, "radius_m": navenv.V3_VISIT_RADIUS,
                       "rule": V5_CONFIG["visitation"]["rule"]},
        "coarse_map": {"block": navenv.V3_COARSE_BLOCK, "res_m": navenv.V3_COARSE_RES, "crop": navenv.V3_COARSE_CROP,
                       "pool": V5_CONFIG["coarse_map"]["pool"]},
        "yaw_change_cost": navenv.V3_YAW_CHANGE_COST,
        "brake": {"lo_m": navenv.V3_BRAKE_LO, "span_m": navenv.V3_BRAKE_SPAN, "half_angle_deg": navenv.V3_BRAKE_HALF_ANGLE_DEG,
                  "period_s": navenv.V3_BRAKE_PERIOD_S},
        "extractor": str(extractor_name),
    }


def yaw_stats(final_infos: dict) -> dict:
    """V5_YAW_NOTE, from {(size, seed): the episode's last info} (HeldoutPool.final_infos of one evaluation)."""
    out = {}
    for (sz, _sd), inf in sorted(final_infos.items()):
        o = out.setdefault(f"{sz[0]}x{sz[1]}", {"episodes": 0, "steps": 0, "yaw_flips": 0, "yaw_sat_steps": 0, "braked_steps": 0})
        o["episodes"] += 1
        o["steps"] += int(inf["t"])
        for k in ("yaw_flips", "yaw_sat_steps", "braked_steps"):
            o[k] += int(inf[k])
    for o in out.values():
        o["yaw_flips_per_s"] = o["yaw_flips"] / (o["steps"] * navenv.DT) if o["steps"] else None
        o["wz_saturation_share"] = o["yaw_sat_steps"] / o["steps"] if o["steps"] else None
        o["braked_share"] = o["braked_steps"] / o["steps"] if o["steps"] else None
    return out


# ---------------------------------------------------------------- v5 gym gate (FROZEN 2026-10-01, before any v5 training)
V5_RULE = {"5x5_success_min": 0.80, "6x6_success_min": 0.70, "6x6_collision_max": 0.15, "n": 48, "total_steps": 30_000_000,
           "seeds": (8, 9, 10), "min_seeds_passing": 2, "set": "final_eval_fresh_48", "fresh_base": 61_000}
V5_TEXT = ("Gym gate: V4_RULE's clauses unchanged (5x5 success >= 0.80, 6x6 success >= 0.70, 6x6 collision <= 0.15, n = 48 "
           "per size, the run reached its last step (num_timesteps >= 30 000 000), and it is the declared v5 configuration), "
           "judged on each seed's FINAL actor (deterministic) on the NEVER-USED set maze seeds 61000-61047 (dynamics "
           "reset(seed = 61000 + k)). v5 PASSES iff >= 2 of the 3 seeds (8, 9, 10) meet every clause; otherwise NEGATIVE. "
           "Seeds 10000-10047 and the best checkpoint are reported, not gated.")


def judge_v5(summary: dict | None, rule: dict = V5_RULE) -> dict:
    """Per-seed checks of the predeclared v5 gym gate on one summary.json (None = no summary: the run did not finish)."""
    if summary is None:
        return {"meets": False, "checks": {"summary_exists": False}, "numbers": None}
    ev = summary.get(rule["set"]) or {}
    a, b = ev.get("5x5"), ev.get("6x6")
    fs = summary.get("fresh_set") or {}
    checks = {"summary_exists": True,
              "config_v5": summary.get("ppo") == "v2" and summary.get("env_version") == 3 and summary.get("std_ceiling") is None
              and summary.get("stall_price") == V4_STALL_PRICE and summary.get("v5_config") == V5_CONFIG,
              "reached_last_step": bool(summary.get("reached_last_step")) and int(summary.get("num_timesteps", 0)) >= rule["total_steps"]
              and int(summary.get("total_steps", 0)) == rule["total_steps"],
              "never_used_set_is_61000": str(fs.get("maze_seeds", "")).startswith(f"{rule['fresh_base']} + k")
              and str(fs.get("dynamics", "")) == f"reset(seed={rule['fresh_base']} + k)",
              "n48": bool(a and b and a["n"] == rule["n"] and b["n"] == rule["n"]),
              "5x5_success": bool(a and a["success"] >= rule["5x5_success_min"]),
              "6x6_success": bool(b and b["success"] >= rule["6x6_success_min"]),
              "6x6_collision": bool(b and b["collision"] <= rule["6x6_collision_max"])}
    nums = None
    if a and b:
        ho = summary.get("final_eval_heldout_48") or {}
        nums = {"never_used_61000": {k: ev[k] for k in ("4x4", "5x5", "6x6") if k in ev},
                "heldout_10000_for_comparability": {k: ho[k] for k in ("4x4", "5x5", "6x6") if k in ho},
                "num_timesteps": summary.get("num_timesteps"), "stall_price": summary.get("stall_price"),
                "std_pre_clamp_last": summary.get("std_pre_clamp_per_dim_last_update"), "std_floor_clamps": summary.get("std_floor_clamps"),
                "yaw_stats_not_gated": summary.get("yaw_stats_not_gated")}
        best = summary.get("best") or {}
        if best:
            nums["best_checkpoint_not_gated"] = {"timesteps": best.get("timesteps"), "never_used_61000": best.get("final_eval_fresh_48"),
                                                 "heldout_10000_selection_biased": best.get("final_eval_heldout_48")}
    return {"meets": all(checks.values()), "checks": checks, "numbers": nums}


def verdict_v5(root: Path, seeds=V5_RULE["seeds"], rule: dict = V5_RULE, final: bool = False) -> dict:
    """The v5 arm verdict (as verdict_v4): a missing summary is PENDING while the array runs, a miss with `final`."""
    per = {}
    for sd in seeds:
        p = root / f"armV5-s{sd}" / "summary.json"
        per[str(sd)] = judge_v5(json.loads(p.read_text()) if p.is_file() else None, rule)
    n_pass = sum(v["meets"] for v in per.values())
    missing = 0 if final else sum(not v["checks"].get("summary_exists") for v in per.values())
    if n_pass >= rule["min_seeds_passing"]:
        verdict = "PASS"
    elif n_pass + missing >= rule["min_seeds_passing"]:
        verdict = "PENDING"
    else:
        verdict = "NEGATIVE"
    return {"rule": {**rule, "seeds": list(rule["seeds"]), "stall_price": V4_STALL_PRICE, "v5_config": V5_CONFIG, "text": V5_TEXT},
            "label": "LEARNED actors (deterministic mean); pose and goal are ORACLE inputs",
            "final": bool(final), "seeds_passing": n_pass, "gate_passing_seeds": [int(sd) for sd, v in per.items() if v["meets"]],
            "verdict": verdict, "per_seed": per}


def _write_once(out: Path, obj: dict) -> tuple:
    """Write a verdict JSON unless the file exists (never overwritten); return (the verdict as recorded, written?)."""
    if out.exists():
        return json.loads(out.read_text()), False
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(obj, indent=2) + "\n")
    return json.loads(out.read_text()), True


# The FINAL gym verdict's file name, which the transfer launcher reads: only a --final verdict is ever written under it.
V5_FINAL_VERDICT_NAME = "verdict_v5.json"


def verdict_v5_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="navgym_train.py verdict-v5")
    ap.add_argument("root", type=Path)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(V5_RULE["seeds"]))
    ap.add_argument("--out", type=Path, default=None,
                    help=f"with --final, default <root>/{V5_FINAL_VERDICT_NAME} (never overwritten). Without --final (a status "
                         f"check) the verdict is only printed, unless --out names a file, never one called {V5_FINAL_VERDICT_NAME}")
    ap.add_argument("--final", action="store_true", help="every array task has ended: a missing summary counts as a miss")
    a = ap.parse_args(argv)
    out = a.out or (a.root / V5_FINAL_VERDICT_NAME if a.final else None)
    if not a.final and out is not None and out.name == V5_FINAL_VERDICT_NAME:
        print(f"NOT WRITTEN: {out}: the name {V5_FINAL_VERDICT_NAME} is reserved for the FINAL verdict (--final, once every array "
              "task has ended), which the transfer launcher reads; for a status check omit --out or pass another file", flush=True)
        return 2
    v = verdict_v5(a.root, tuple(a.seeds), final=a.final)
    if out is None:
        rd = v
        print(f"status only (no --final): printed, not written; {V5_FINAL_VERDICT_NAME} holds only the final verdict", flush=True)
    else:
        rd, written = _write_once(out, v)
        print(("wrote " if written else "REFUSING to overwrite; the recorded verdict is re-printed: ") + str(out), flush=True)
    parts = []
    for sd, r in rd["per_seed"].items():
        n = r["numbers"]
        if n is None:
            parts.append(f"s{sd}: " + ("summary without a 5x5 + 6x6 never-used evaluation (a miss)" if r["checks"].get("summary_exists")
                                       else "no summary"))
        else:
            f5, f6 = n["never_used_61000"]["5x5"], n["never_used_61000"]["6x6"]
            bad = [k for k, ok in r["checks"].items() if not ok]
            parts.append(f"s{sd}: {'MEETS' if r['meets'] else 'misses'} 61000-set 5x5 {f5['success']:.3f} 6x6 {f6['success']:.3f} "
                         f"(coll {f6['collision']:.3f}; last step {r['checks']['reached_last_step']}" + (f"; failed {bad}" if bad else "") + ")")
    print(f"NAVGYM-V5 VERDICT: {rd['verdict']} ({rd['seeds_passing']}/{len(rd['per_seed'])} seeds meet the bar; final={rd['final']}; "
          f"LEARNED actors, ORACLE pose+goal) | " + " | ".join(parts), flush=True)
    return 0


# ---------------------------------------------------------------- v5 physics transfer (FROZEN 2026-10-01)
V5_TRANSFER_RULE = {
    "seeds": list(range(63_000, 63_012)), "n": 12, "min_goals": 10, "max_falls": 0, "time_limit_s": 180.0,
    "maze": {"n": 6, "m": 6, "extra_openings": 1},
    "text": ("Physics transfer, only if the gym gate PASSES, only for gate-passing final actors: MuJoCo 3.3.5, the frozen "
             "learned biped gait, LEARNED v5 (vx, wz) commands, ORACLE pose and goal, --policy-capture-pose; hard 6x6, 1 extra "
             "opening, maze seeds 63000-63011, 180 s. PASS iff EACH gate-passing final actor reaches >= 10/12 goals with 0 "
             "falls; otherwise NEGATIVE; INCOMPLETE if a summary is missing or not 12 episodes. A* on the same mazes "
             "reported, not gated."),
}
V5_TRANSFER_LABELS = {"learned": "frozen biped gait (PPO, dr-default-s0) + NavGym v5 final actor (vx, wz) commands",
                      "scripted": "A* reference only (not gated)", "oracle": "pose and goal coordinate"}


def v5_gate_passing_actors(gym_verdict: dict | None) -> list:
    """The final actors the transfer runs: only after a FINAL gym PASS, the seeds that met every clause (else none)."""
    if not gym_verdict or gym_verdict.get("verdict") != "PASS" or gym_verdict.get("final") is not True:
        return []
    return [f"armV5-s{sd}" for sd, r in sorted(gym_verdict.get("per_seed", {}).items(), key=lambda kv: int(kv[0])) if r.get("meets")]


def judge_v5_transfer(actors, rows: dict, summaries: dict, astar_rows: dict | None = None, rule: dict = V5_TRANSFER_RULE) -> dict:
    """Pure physics-transfer verdict. `actors`: the gate-passing final actors (v5_gate_passing_actors); rows[a] = {maze
    seed: per-seed JSON of maze_explore.py}; summaries[a] = its summary.json or None. INCOMPLETE when there is no actor,
    a summary is missing or not 12 episodes, the per-seed set is not 63000-63011, or an episode is not the declared run
    (that actor's actor.onnx, capture-pose integration, the v5 visitation memory, hard 6x6 with 1 extra opening, 180 s)."""
    problems, per = [], {}
    if not actors:
        problems.append("no gate-passing actor: the transfer runs only after a final gym PASS")
    for a in actors:
        r, s = rows.get(a) or {}, summaries.get(a)
        if s is None:
            problems.append(f"{a}: no summary.json")
        else:
            if s.get("n") != rule["n"]:
                problems.append(f"{a}: summary.json has n = {s.get('n')}, not {rule['n']} episodes")
            st = s.get("settings") or {}
            if st.get("time_limit") != rule["time_limit_s"] or st.get("policy_capture_pose") is not True or \
                    (st.get("n"), st.get("m"), st.get("extra_openings")) != (rule["maze"]["n"], rule["maze"]["m"], rule["maze"]["extra_openings"]):
                problems.append(f"{a}: summary settings are not the declared run ({ {k: st.get(k) for k in ('n', 'm', 'extra_openings', 'time_limit', 'policy_capture_pose')} })")
        missing = [sd for sd in rule["seeds"] if sd not in r]
        extra = sorted(set(r) - set(rule["seeds"]))
        if missing:
            problems.append(f"{a}: missing maze seeds {missing}")
        if extra:
            problems.append(f"{a}: unexpected maze seeds {extra}")
        for sd in sorted(r):
            e = r[sd]
            if not str(e.get("policy") or "").endswith(f"{a}/actor.onnx"):
                problems.append(f"{a} seed {sd}: not this actor's final actor.onnx ({e.get('policy')})")
            mi = e.get("policy_map_integration") or {}
            if mi.get("mode") != "capture_pose" or not mi.get("updates_at_capture_pose"):
                problems.append(f"{a} seed {sd}: capture-pose integration not recorded")
            pv = e.get("policy_visitation") or {}
            if not pv.get("updates"):
                problems.append(f"{a} seed {sd}: v5 visitation memory not recorded")
            mz = e.get("maze") or {}
            if (mz.get("n"), mz.get("m"), mz.get("extra_openings")) != (rule["maze"]["n"], rule["maze"]["m"], rule["maze"]["extra_openings"]):
                problems.append(f"{a} seed {sd}: not a hard 6x6 maze with 1 extra opening")
        goals = sum(bool(e.get("success")) for e in r.values())
        per[a] = {"goals": goals, "n": len(r), "falls": sum(e.get("outcome") == "fall" for e in r.values()),
                  "time_outs": sum(e.get("outcome") == "time_out" for e in r.values()),
                  "clean": sum(bool(e.get("clean_success")) for e in r.values()),
                  "failed_seeds": [sd for sd in rule["seeds"] if sd in r and not r[sd].get("success")],
                  "summary_success": None if s is None else s.get("success")}
        if s is not None and s.get("success") != goals:
            problems.append(f"{a}: summary.json success {s.get('success')} != per-seed goals {goals}")
    ref = None
    if astar_rows is not None:
        ref = {"goals": sum(bool(e.get("success")) for e in astar_rows.values()), "n": len(astar_rows), "label": "SCRIPTED A* reference, not gated"}
    base = {"rule": rule["text"], "labels": V5_TRANSFER_LABELS, "actors": list(actors), "per_actor": per, "astar_reference": ref}
    if problems:
        return {"verdict": "INCOMPLETE", "problems": problems, **base}
    meets = {a: per[a]["goals"] >= rule["min_goals"] and per[a]["falls"] <= rule["max_falls"] for a in actors}
    return {"verdict": "PASS" if all(meets.values()) else "NEGATIVE", "meets": meets, **base}


def _load_rows(d: Path) -> dict:
    """{maze seed: per-seed JSON} from <d>/seed<k>.json (maze_explore.py's files)."""
    import re
    d = Path(d)
    if not d.is_dir():
        return {}
    return {int(m.group(1)): json.loads(f.read_text()) for f in sorted(d.iterdir()) if (m := re.fullmatch(r"seed(\d+)\.json", f.name))}


def verdict_v5_transfer_main(argv) -> int:
    ap = argparse.ArgumentParser(prog="navgym_train.py verdict-v5-transfer")
    ap.add_argument("out", type=Path, help="the transfer output directory (one sub-directory per actor, plus astar/)")
    ap.add_argument("--gym-verdict", type=Path, required=True, help="the FINAL gym verdict (verdict_v5.json)")
    ap.add_argument("--list-actors", action="store_true", help="only print the gate-passing actors, one per line (for the launcher)")
    ap.add_argument("--final", action="store_true",
                    help="every episode run has ended (the launcher passes it once they have all exited): only then is "
                         "<out>/verdict.json written (once, never overwritten); without it the verdict is only printed")
    a = ap.parse_args(argv)
    gv = json.loads(a.gym_verdict.read_text()) if a.gym_verdict.is_file() else None
    if gv is None or gv.get("final") is not True:
        state = "missing" if gv is None else f"verdict {gv.get('verdict')}, final={gv.get('final')}"
        print(f"NOT A FINAL GYM VERDICT: {a.gym_verdict} ({state}): the transfer reads only the final verdict that "
              "cpu_navgym_v5.sbatch NAVGYM_MODE=verdict writes (--final); nothing listed, judged or written", file=sys.stderr, flush=True)
        return 4
    actors = v5_gate_passing_actors(gv)
    if a.list_actors:
        print("\n".join(actors))
        return 0
    rows = {x: _load_rows(a.out / x) for x in actors}
    summ = {x: (json.loads((a.out / x / "summary.json").read_text()) if (a.out / x / "summary.json").is_file() else None) for x in actors}
    v = judge_v5_transfer(actors, rows, summ, _load_rows(a.out / "astar") if (a.out / "astar").is_dir() else None)
    v["gym_verdict"] = {"path": str(a.gym_verdict), "verdict": (gv or {}).get("verdict"), "final": (gv or {}).get("final")}
    if a.final:
        rd, written = _write_once(a.out / "verdict.json", v)
        print(("wrote " if written else "REFUSING to overwrite; the recorded verdict is re-printed: ") + str(a.out / "verdict.json"), flush=True)
    else:
        rd = v
        print("status only (no --final): printed, not written; the launcher writes verdict.json with --final once every "
              "episode run has ended", flush=True)
    detail =" | ".join(f"{x}: {p['goals']}/{p['n']} (falls {p['falls']}, time-outs {p['time_outs']})" for x, p in rd["per_actor"].items())
    ref = rd.get("astar_reference") or {}
    print(f"NAVGYM-V5-TRANSFER VERDICT: {rd['verdict']} | {detail} | A* reference (SCRIPTED, not gated): {ref.get('goals')}/{ref.get('n')} | "
          "LEARNED gait + LEARNED NavGym v5 commands; ORACLE pose + goal"
          + (" | problems: " + "; ".join(rd["problems"]) if rd.get("problems") else ""), flush=True)
    return 0


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "verdict-v3":
        sys.exit(verdict_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "reading-v2ctl":
        sys.exit(reading_v2ctl_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "verdict-v4":
        sys.exit(verdict_v4_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "verdict-v5":
        sys.exit(verdict_v5_main(sys.argv[2:]))
    if len(sys.argv) > 1 and sys.argv[1] == "verdict-v5-transfer":
        sys.exit(verdict_v5_transfer_main(sys.argv[2:]))
    sys.exit(main())
