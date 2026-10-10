#!/usr/bin/env python3
"""DSRL-SAC training: a frozen flow VLA (SmolVLA or the TOY stub) steered by SAC over its initial noise.

Loop (one GPU process + N CPU env workers):
  every env waits at a decision point -> one batched prefix pass of the frozen VLA (features + KV cache)
  -> SAC actor picks a latent per env -> NoiseMap -> 10-step denoise -> each worker executes the first H
  actions of its chunk (stops at done) -> while the workers step, the GPU runs UTD x N SAC updates.
One SAC transition per executed chunk; reward per chunk (see --reward); discount gamma ** n_executed.

Rewards (env reward is the contract's sparse success, 1.0 on the success step):
  --reward sparse   : r = 1 on the chunk where success is reached, else 0; success and fall are terminal;
                      the 60 s time limit is a truncation (bootstrapped).  [DEFAULT, unshaped]
  --reward minus1   : r = -1 per chunk until success (0 on the success chunk), as in the official pi0 DSRL
                      code; a fall is terminal with r = -1/(1 - gamma_chunk) (= -1 forever) so falling is
                      never cheaper than continuing.  [unshaped up to that fall convention]
  SHAPING (off by default, every use labelled SHAPED in the outputs):
  --shaping stage   : potential-based  F = gamma^n * Phi(s') - Phi(s), Phi = coef * info["stage"] /
                      --stage-count (Ng et al. 1999: leaves the optimal policy unchanged); Phi = 0 at terminals.
  --fall-penalty X  : extra -X on a fall (not potential-based).

Seeds: training layouts must be in the dev range 1000-1999 unless --allow-phase4-training-range (5000-5999,
the real phase 4 run). Evaluation seeds here are dev-range only (the scored 6000-6099 run is eval_dsrl.py).
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from collections import deque
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
from chunk_vec_env import ChunkVecEnv  # noqa: E402
from rollout import (ObsBuilder, arm_summary, check_seeds, paired_summary, parse_seeds,  # noqa: E402
                     run_episodes, take)
from sac_noise import NoiseSAC, ReplayBuffer, SACConfig  # noqa: E402
from smolvla_noise import NoiseMap  # noqa: E402


def build_policy(kind, path, device, feature_mode="pooled", exec_horizon=None):
    if kind == "stub":
        from stub_policy import StubFlowPolicy
        return StubFlowPolicy(device=device, n_action_steps=exec_horizon or 10)
    from smolvla_noise import SmolVLANoisePolicy
    if kind == "smolvla":
        return SmolVLANoisePolicy.from_pretrained(path, device=device, feature_mode=feature_mode)
    if kind == "smolvla-untuned":
        return SmolVLANoisePolicy.untuned_waiter(path, device=device, feature_mode=feature_mode)
    raise ValueError(kind)


def env_spec(args):
    spec = {"kind": args.env, "kwargs": json.loads(args.env_kwargs),
            "sys_path": [str(REPO / "src"), str(HERE)]}
    if args.env == "import":
        spec["target"] = args.env_target
    return spec


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2) + "\n")
    os.replace(tmp, path)


def add_common_args(ap):
    ap.add_argument("--policy", choices=("stub", "smolvla", "smolvla-untuned"), default="stub")
    ap.add_argument("--policy-path", help="SmolVLA pretrained_model dir (smolvla) or smolvla_base dir (untuned)")
    ap.add_argument("--feature-mode", default="pooled", choices=("pooled", "mean", "state"))
    ap.add_argument("--env", choices=("stub", "waiter", "import"), default="stub")
    ap.add_argument("--env-target", help="module:Class for --env import")
    ap.add_argument("--env-kwargs", default="{}", help="JSON kwargs for the env constructor")
    ap.add_argument("--env-python", default=None,
                    help="interpreter for env worker processes (default: in-process, sequential)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--exec-horizon", type=int, default=None,
                    help="actions executed per chunk = env steps per SAC decision (default: policy n_action_steps)")
    ap.add_argument("--noise-mode", choices=("repeat", "dct", "full"), default="repeat")
    ap.add_argument("--noise-k", type=int, default=1)
    ap.add_argument("--active-dims", type=int, default=None)
    ap.add_argument("--action-magnitude", type=float, default=1.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--train-seeds", default="1000-1899")
    ap.add_argument("--allow-phase4-training-range", action="store_true")
    ap.add_argument("--eval-seeds", default="1900-1949")
    ap.add_argument("--eval-every", type=int, default=0, help="decisions between dev evals (0 = start/end only)")
    ap.add_argument("--n-eval-envs", type=int, default=0, help="0 = same as --n-envs")
    ap.add_argument("--no-eval", action="store_true")
    ap.add_argument("--obs-encoder", choices=("vlm", "cnn"), default="vlm")
    ap.add_argument("--reward", choices=("sparse", "minus1"), default="sparse")
    ap.add_argument("--shaping", choices=("none", "stage"), default="none")
    ap.add_argument("--shaping-coef", type=float, default=0.1)
    ap.add_argument("--stage-count", type=float, default=1.0)
    ap.add_argument("--fall-penalty", type=float, default=0.0)
    ap.add_argument("--gamma", type=float, default=0.999, help="per env step (12.5 Hz)")
    ap.add_argument("--utd", type=float, default=20.0, help="SAC updates per new chunk transition")
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=3)
    ap.add_argument("--n-critics", type=int, default=2)
    ap.add_argument("--critic-reduction", choices=("min", "mean"), default="min")
    ap.add_argument("--actor-lr", type=float, default=3e-4)
    ap.add_argument("--critic-lr", type=float, default=3e-4)
    ap.add_argument("--target-entropy", default="0.0", help="float, or 'auto' (= -latent_dim)")
    ap.add_argument("--init-alpha", type=float, default=1.0, help="initial SAC temperature")
    ap.add_argument("--actor-delay", type=int, default=1, help="critic updates per actor (and temperature) update")
    ap.add_argument("--no-backup-entropy", action="store_true",
                    help="leave the entropy bonus out of the critic target (with a 0/1 sparse reward and a long "
                         "horizon the bonus otherwise sets the Q scale: toy run A had Q ~ 40-230 for rewards <= 1)")
    ap.add_argument("--activation", default="tanh")
    ap.add_argument("--buffer-size", type=int, default=200_000)
    ap.add_argument("--warmup-transitions", type=int, default=500,
                    help="chunk transitions collected with N(0,1) latents (clipped) before SAC acts/updates")
    ap.add_argument("--total-decisions", type=int, default=5000, help="SAC decisions (chunk transitions)")
    ap.add_argument("--total-env-steps", type=int, default=0)
    ap.add_argument("--max-wall-min", type=float, default=0.0, help="checkpoint and stop after this many minutes")
    ap.add_argument("--ckpt-every", type=int, default=2000)
    ap.add_argument("--log-every", type=int, default=10, help="rounds between train_log lines")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    t_start = time.time()
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    (out / "workers").mkdir(exist_ok=True)

    # ---- seeds (frozen ranges)
    toy = args.env == "stub" or args.policy == "stub"
    train_seeds = parse_seeds(args.train_seeds)
    eval_seeds = [] if args.no_eval else parse_seeds(args.eval_seeds)
    allowed = {"dev"} | ({"p4_train"} if args.allow_phase4_training_range else set())
    check_seeds(train_seeds, allowed, "training")
    check_seeds(eval_seeds, {"dev"}, "in-training evaluation")
    if set(train_seeds) & set(eval_seeds):
        raise SystemExit("training and evaluation seeds overlap")

    torch.manual_seed(args.seed)
    np_rng = np.random.default_rng(args.seed)
    seed_rng = np.random.default_rng(args.seed + 1)
    device = args.device

    policy = build_policy(args.policy, args.policy_path, device, args.feature_mode, args.exec_horizon)
    H = int(args.exec_horizon or policy.n_action_steps)
    if not 1 <= H <= policy.chunk_size:
        raise SystemExit(f"exec horizon {H} outside 1..{policy.chunk_size}")
    T, D = policy.noise_shape
    noise_map = NoiseMap(T, D, args.noise_mode, args.noise_k, args.active_dims)
    b = float(args.action_magnitude)
    obs_builder = ObsBuilder(args.obs_encoder, policy.feature_dim)
    te = None if args.target_entropy == "auto" else float(args.target_entropy)
    cfg = SACConfig(vec_dim=obs_builder.vec_dim, act_dim=noise_map.latent_dim, img_hw=obs_builder.img_hw,
                    action_magnitude=b, hidden=args.hidden, layers=args.layers, n_critics=args.n_critics,
                    critic_reduction=args.critic_reduction, actor_lr=args.actor_lr, critic_lr=args.critic_lr,
                    target_entropy=te, batch_size=args.batch_size, activation=args.activation,
                    init_alpha=args.init_alpha, backup_entropy=not args.no_backup_entropy,
                    actor_delay=args.actor_delay)
    sac = NoiseSAC(cfg, device=device, seed=args.seed)
    buf = ReplayBuffer(args.buffer_size, cfg.vec_dim, cfg.act_dim, cfg.img_hw, seed=args.seed)
    gamma_chunk = args.gamma ** H
    labels = (["TOY"] if toy else []) + (["SHAPED"] if (args.shaping != "none" or args.fall_penalty) else [])

    st = {"decisions": 0, "env_steps": 0, "episodes": 0, "transitions": 0, "updates": 0, "rounds": 0,
          "wall_s_before_resume": 0.0}
    ckpt = out / "ckpt"
    if args.resume and (ckpt / "state.json").exists():
        sac.load_state_dict(torch.load(ckpt / "sac.pt", map_location=device, weights_only=False))
        buf.load(ckpt / "buffer.npz")
        saved = json.loads((ckpt / "state.json").read_text())
        st.update(saved["counters"])
        st["wall_s_before_resume"] = saved["counters"].get("wall_s", 0.0)
        seed_rng.bit_generator.state = saved["seed_rng"]
        np_rng.bit_generator.state = saved["np_rng"]
        print(f"[dsrl] resumed at {st}", flush=True)

    config = {"labels": labels, "argv": sys.argv, "args": {k: str(v) for k, v in vars(args).items()},
              "policy": policy.describe(), "noise_map": noise_map.describe(), "sac": cfg.__dict__,
              "exec_horizon": H, "gamma_per_step": args.gamma, "gamma_per_full_chunk": gamma_chunk,
              "obs_encoder": args.obs_encoder, "sac_vec_dim": cfg.vec_dim,
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "host": os.uname().nodename,
              "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() and "cuda" in device else None,
              "torch": torch.__version__, "numpy": np.__version__}
    atomic_json(out / "config.json", config)
    print(f"[dsrl] {' '.join(labels)} policy={policy.describe()} latent_dim={noise_map.latent_dim} H={H}", flush=True)

    venv = ChunkVecEnv(env_spec(args), args.n_envs, python=args.env_python, log_dir=str(out / "workers"))
    config["workers"] = venv.worker_info
    atomic_json(out / "config.json", config)
    eval_venv = None
    evals = []

    def do_eval(tag, arms):
        nonlocal eval_venv
        if not eval_seeds:
            return
        if eval_venv is None:
            eval_venv = ChunkVecEnv(env_spec(args), args.n_eval_envs or args.n_envs, python=args.env_python,
                                    log_dir=str(out / "workers"))
        res = {"tag": tag, "decisions": st["decisions"], "env_steps": st["env_steps"], "labels": labels, "arms": {}}
        eps = {}
        for arm in arms:
            t0 = time.time()
            eps[arm] = run_episodes(eval_venv, policy, eval_seeds, arm, H, noise_map, b, obs_builder,
                                    sac if arm.startswith("dsrl") else None, noise_seed=args.seed, device=device)
            res["arms"][arm] = {**arm_summary(eps[arm]), "wall_s": time.time() - t0}
        if "dsrl" in eps and "base" in eps:
            res["dsrl_vs_base"] = paired_summary([e["success"] for e in eps["dsrl"]], [e["success"] for e in eps["base"]])
        res["episodes"] = eps
        evals.append({k: v for k, v in res.items() if k != "episodes"})
        atomic_json(out / f"eval_{tag}.json", res)
        print(f"[dsrl] eval {tag}: " + ", ".join(f"{a}={r['success_rate']:.3f}" for a, r in res["arms"].items()), flush=True)

    def save_ckpt():
        tmp = out / "ckpt.new"
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        sac.save(tmp / "sac.pt")
        buf.save(tmp / "buffer.npz")
        atomic_json(tmp / "state.json", {"counters": {**st, "wall_s": wall()}, "seed_rng": seed_rng.bit_generator.state,
                                         "np_rng": np_rng.bit_generator.state, "labels": labels})
        old = out / "ckpt.old"
        if ckpt.exists():
            os.replace(ckpt, old)
        os.replace(tmp, ckpt)
        if old.exists():
            shutil.rmtree(old)

    def wall():
        return st["wall_s_before_resume"] + time.time() - t_start

    if st["decisions"] == 0:
        do_eval("start", ["base", "zero", "dsrl"])

    # ---- episode bookkeeping
    def new_ep(seed):
        return {"seed": int(seed), "success": False, "fell": False, "steps": 0, "decisions": 0, "stage_max": 0,
                "return_train": 0.0, "warmup_frac": 0.0}

    slots = {i: int(seed_rng.choice(train_seeds)) for i in range(args.n_envs)}
    obs = venv.reset({i: (s, None) for i, s in slots.items()})
    ep = {i: new_ep(s) for i, s in slots.items()}
    phi = {i: 0.0 for i in slots}
    pending = {}
    recent = deque(maxlen=50)
    stats_acc, timers = [], {"encode": 0.0, "denoise": 0.0, "env_wait": 0.0, "update": 0.0}
    u_acc = []                                       # |latent| of the acted decisions since the last log line
    ep_log = open(out / "episodes.jsonl", "a")
    tr_log = open(out / "train_log.jsonl", "a")
    last_ckpt, last_eval = st["decisions"], st["decisions"]

    def stop():
        return (st["decisions"] >= args.total_decisions
                or (args.total_env_steps and st["env_steps"] >= args.total_env_steps)
                or (args.max_wall_min and wall() > 60 * args.max_wall_min))

    while not stop():
        active = list(range(args.n_envs))
        ol = [obs[i] for i in active]
        t0 = time.time()
        ctx = policy.encode(ol)
        so = obs_builder(ol, ctx.features)
        timers["encode"] += time.time() - t0
        for j, i in enumerate(active):            # the current obs completes last round's transitions
            if i in pending:
                tr = pending.pop(i)
                buf.add(tr["obs"], tr["act"], tr["rew"], tr["disc"], tr["term"], take(so, j))
        warm = st["transitions"] < args.warmup_transitions
        if warm:
            u = (np.clip(np_rng.standard_normal((len(active), noise_map.latent_dim)), -b, b) / b).astype(np.float32)
        else:
            u = sac.act(so)
        u_acc.append(np.abs(u))
        t0 = time.time()
        chunks = policy.denoise(ctx, noise_map(torch.as_tensor(u * b)).to(device))[:, :H].numpy()
        timers["denoise"] += time.time() - t0
        venv.step_async({i: chunks[j] for j, i in enumerate(active)})
        # SAC updates overlap with the CPU workers executing the chunks
        t0 = time.time()
        if not warm and buf.size >= args.batch_size:
            for _ in range(int(round(args.utd * len(active)))):
                stats_acc.append(sac.update(buf))
                st["updates"] += 1
        timers["update"] += time.time() - t0
        t0 = time.time()
        res = venv.step_wait()
        timers["env_wait"] += time.time() - t0
        st["rounds"] += 1
        finals, resets = [], {}
        for j, i in enumerate(active):
            r = res[i]
            term = bool(r["success"] or r["fell"])
            if args.reward == "sparse":
                rew = 1.0 if r["success"] else 0.0
            else:
                rew = 0.0 if r["success"] else (-1.0 / (1.0 - gamma_chunk) if r["fell"] else -1.0)
            disc = float(args.gamma ** r["n"])
            if args.shaping == "stage":
                phi_next = 0.0 if term else args.shaping_coef * r["stage_last"] / args.stage_count
                rew += disc * phi_next - phi[i]
                phi[i] = phi_next
            if r["fell"]:
                rew -= args.fall_penalty
            tr = {"obs": take(so, j), "act": u[j], "rew": rew, "disc": disc, "term": term}
            e = ep[i]
            e["steps"] += r["n"]
            e["decisions"] += 1
            e["stage_max"] = max(e["stage_max"], r["stage_max"])
            e["success"] |= r["success"]
            e["fell"] |= r["fell"]
            e["return_train"] += rew
            e["warmup_frac"] += float(warm)
            st["decisions"] += 1
            st["transitions"] += 1
            st["env_steps"] += r["n"]
            if r["done"]:
                if term:
                    buf.add(tr["obs"], tr["act"], tr["rew"], tr["disc"], True, obs_builder.zeros())
                else:
                    finals.append((tr, r["obs"]))
                e["warmup_frac"] /= max(1, e["decisions"])
                e.update(episode=st["episodes"], t=r["t"], decisions_total=st["decisions"], wall_s=wall())
                ep_log.write(json.dumps(e) + "\n")
                st["episodes"] += 1
                recent.append(e["success"])
                s = int(seed_rng.choice(train_seeds))
                resets[i] = (s, None)
                ep[i], phi[i] = new_ep(s), 0.0
            else:
                pending[i] = tr
                obs[i] = r["obs"]
        if finals:                                 # time-limit truncation: bootstrap from the final obs
            fo = [o for _, o in finals]
            fso = obs_builder(fo, policy.encode(fo).features)
            for k, (tr, _) in enumerate(finals):
                buf.add(tr["obs"], tr["act"], tr["rew"], tr["disc"], False, take(fso, k))
        if resets:
            obs.update(venv.reset(resets))
            ep_log.flush()
        if st["rounds"] % args.log_every == 0:
            agg = {}
            for k in ("critic_loss", "q_mean", "actor_loss", "entropy", "alpha"):
                vals = [s[k] for s in stats_acc if k in s]
                if vals:
                    agg[k] = float(np.mean(vals))
            stats_acc.clear()
            ua = np.concatenate(u_acc)               # 1.0 = the latent sits on the box boundary (tanh saturated)
            agg.update(u_abs_mean=float(ua.mean()), u_sat_frac=float((ua > 0.98).mean()))
            u_acc.clear()
            line = {**{k: st[k] for k in ("decisions", "env_steps", "episodes", "updates")},
                    "warm": warm, "success_last50": float(np.mean(recent)) if recent else None,
                    "buffer": buf.size, "wall_s": wall(), "env_steps_per_s": st["env_steps"] / max(1e-9, wall()),
                    "timers_s": {k: round(v, 2) for k, v in timers.items()}, **agg}
            tr_log.write(json.dumps(line) + "\n")
            tr_log.flush()
            print(f"[dsrl] dec={st['decisions']} ep={st['episodes']} succ50={line['success_last50']} "
                  f"upd={st['updates']} {agg}", flush=True)
        if st["decisions"] - last_ckpt >= args.ckpt_every:
            save_ckpt()
            last_ckpt = st["decisions"]
        if args.eval_every and st["decisions"] - last_eval >= args.eval_every:
            do_eval(f"dec{st['decisions']}", ["dsrl"])
            last_eval = st["decisions"]

    save_ckpt()
    do_eval("final", ["base", "zero", "latent_random", "dsrl", "dsrl_stoch"])
    summary = {"labels": labels, "counters": {**st, "wall_s": wall()}, "timers_s": timers,
               "train_success_last50": float(np.mean(recent)) if recent else None, "evals": evals,
               "config": "config.json"}
    atomic_json(out / "summary.json", summary)
    ep_log.close()
    tr_log.close()
    venv.close()
    if eval_venv is not None:
        eval_venv.close()
    print(f"[dsrl] done {json.dumps(summary['counters'])}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
