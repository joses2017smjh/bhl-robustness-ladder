#!/usr/bin/env python3
"""Waiter phase 3: fine-tune SmolVLA (lerobot/smolvla_base) on contract npz demonstrations.

Inputs: one head RGB image (224x224, simulated), the 30-D state and the instruction; output: chunks of the 15-D
contract action (base velocity 3, arm targets 10, gripper closures 2) at 12.5 Hz.

Splits: held-out LAYOUTS (seed % val_seed_mod == val_seed_rem, or --val-seeds) and held-out instruction TEMPLATES
(templates.json "held_out", never trained on). Normalisation statistics come from training episodes only.
Checkpoints are atomic and resumable (``--resume auto`` picks up ``latest.json``); training batches are a pure
function of (seed, step), so a resumed run draws the same batches an uninterrupted one would have.

Validation loss is a flow-matching loss with fixed per-sample noise/time (comparable across checkpoints); the
open-loop action error is the sampled chunk vs the demonstration. Neither is a task success rate: only closed-loop
rollouts (rollout_vla.py) measure that.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import random
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import waiter_vla_data as wd  # noqa: E402


def atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_argument_group("data")
    g.add_argument("--data", type=Path, required=True, help="directory with manifest.json + episode npz files")
    g.add_argument("--templates", type=Path, help="templates.json (default: <data>/templates.json if present)")
    g.add_argument("--held-out-templates", help="comma list overriding templates.json held_out")
    g.add_argument("--val-seed-mod", type=int, default=10)
    g.add_argument("--val-seed-rem", type=int, default=9)
    g.add_argument("--val-seeds", default="", help="extra held-out layout seeds, e.g. 3090-3099,3150")
    g.add_argument("--include-failed", action="store_true", help="also train on unsuccessful demonstrations")
    g.add_argument("--allow-dev-seeds", action="store_true", help="accept dev-range (1000-1999) episodes")
    g.add_argument("--lenient", action="store_true", help="do not enforce contract action ranges")
    g.add_argument("--no-manifest", action="store_true", help="glob *.npz when manifest.json is missing")
    g.add_argument("--limit-episodes", type=int)
    g.add_argument("--cache", type=Path, help="memmap cache dir (default $TMPDIR/waiter-vla-cache)")
    g.add_argument("--std-floor", type=float, default=1e-2)
    g.add_argument("--sampling", default="frame",
                   help="frame | episode | early:<K>:<P> (a fraction P of each batch from the first K frames of "
                        "random episodes; see waiter_vla_data.batch_indices)")
    g = p.add_argument_group("model")
    g.add_argument("--base", default="lerobot/smolvla_base", help="HF repo id (offline cache) or local dir")
    g.add_argument("--chunk-size", type=int, default=50, help="actions predicted per call (50 = 4.0 s at 12.5 Hz)")
    g.add_argument("--n-action-steps", type=int, default=10,
                   help="actions executed per call at rollout (stored in the checkpoint config; 10 = 0.8 s)")
    g.add_argument("--train-mode", choices=("expert", "expert+vision", "full"), default="expert")
    g = p.add_argument_group("optimisation")
    g.add_argument("--steps", type=int, default=20000)
    g.add_argument("--batch-size", type=int, default=32)
    g.add_argument("--lr", type=float, default=1e-4)
    g.add_argument("--decay-lr", type=float, default=2.5e-6)
    g.add_argument("--warmup", type=int, default=1000)
    g.add_argument("--weight-decay", type=float, default=1e-10)
    g.add_argument("--grad-clip", type=float, default=10.0)
    g.add_argument("--amp", choices=("none", "bf16"), default="none")
    g.add_argument("--bf16-trainable", action="store_true",
                   help="train the expert in its constructed bf16 (upstream lerobot behaviour) instead of fp32 master weights")
    g.add_argument("--seed", type=int, default=0)
    g.add_argument("--num-workers", type=int, default=4)
    g = p.add_argument_group("bookkeeping")
    g.add_argument("--out", type=Path, required=True)
    g.add_argument("--save-every", type=int, default=1000)
    g.add_argument("--log-every", type=int, default=50)
    g.add_argument("--keep-last", type=int, default=2)
    g.add_argument("--no-trainer-state", action="store_true",
                   help="do not write trainer.pt (optimizer state; 3.6 GB in --train-mode full): the run cannot be "
                        "resumed, its checkpoints are model-only")
    g.add_argument("--val-per-episode", type=int, default=8)
    g.add_argument("--val-max-samples", type=int, default=512)
    g.add_argument("--openloop-samples", type=int, default=32, help="per split, for sampled-chunk action error")
    g.add_argument("--resume", default="auto", help="auto | none | <checkpoint dir>")
    g.add_argument("--max-minutes", type=float, help="checkpoint and exit cleanly after this wall time")
    g.add_argument("--audit-only", action="store_true", help="build caches/splits/stats and exit (no GPU)")
    args = p.parse_args(argv)
    if min(args.steps, args.batch_size, args.save_every, args.log_every, args.val_per_episode, args.chunk_size) < 1:
        p.error("steps, batch size, intervals, val-per-episode and chunk size must be positive")
    if not 1 <= args.n_action_steps <= args.chunk_size:
        p.error("need 1 <= n-action-steps <= chunk-size")
    if args.sampling not in ("frame", "episode"):
        wd.parse_early_mode(args.sampling)          # raises on a malformed mode
    if args.keep_last < 1:
        p.error("--keep-last must be >= 1")
    return args


# ---------------------------------------------------------------------------------------------------------------
def build_data(args, log):
    templates = None
    tpath = args.templates or (args.data / "templates.json")
    if tpath.is_file():
        templates = wd.TemplateSet.load(tpath)
        if args.held_out_templates is not None:
            held = [t for t in args.held_out_templates.split(",") if t]
            templates = wd.TemplateSet(templates.templates, templates.slots, held)
    elif args.templates is not None:
        raise FileNotFoundError(tpath)
    held = frozenset(templates.held_out) if templates else frozenset()
    spec = wd.SplitSpec(args.val_seed_mod, args.val_seed_rem, wd.parse_seed_list(args.val_seeds), held)
    cache = args.cache or Path(os.environ.get("TMPDIR", "/tmp")) / "waiter-vla-cache"
    dataset = wd.build_dataset(args.data, cache, spec, templates=templates, include_failed=args.include_failed,
                               allow_dev_seeds=args.allow_dev_seeds, strict=not args.lenient,
                               require_manifest=not args.no_manifest, limit=args.limit_episodes, log=log)
    return dataset, templates, (str(tpath) if templates else None)


def check_token_lengths(dataset, cfg) -> dict:
    """Truncation would silently cut the end of an instruction ('... to the left table'): refuse it."""
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(cfg.vlm_model_name)
    lengths = {}
    for e in dataset.episodes:
        text = e.meta["instruction"]
        lengths[text] = len(tok(text + "\n")["input_ids"])
    worst = max(lengths.values())
    if worst > cfg.tokenizer_max_length:
        bad = [t for t, n in lengths.items() if n > cfg.tokenizer_max_length]
        raise ValueError(f"{len(bad)} instructions exceed tokenizer_max_length={cfg.tokenizer_max_length}: {bad[:3]}")
    return {"max_tokens": worst, "limit": cfg.tokenizer_max_length, "distinct_instructions": len(lengths)}


def lr_at(step: int, args) -> float:
    if step < args.warmup:
        return args.lr * (step + 1) / args.warmup
    span = max(1, args.steps - args.warmup)
    progress = min(1.0, (step - args.warmup) / span)
    return args.decay_lr + 0.5 * (args.lr - args.decay_lr) * (1 + math.cos(math.pi * progress))


class StepBatchSampler:
    """Yields one list of (episode, frame) keys per optimisation step, from start to total (exclusive)."""

    def __init__(self, lengths, batch_size, seed, start, total, mode):
        self.lengths, self.batch_size, self.seed = list(lengths), batch_size, seed
        self.start, self.total, self.mode = start, total, mode

    def __iter__(self):
        for step in range(self.start, self.total):
            yield wd.batch_indices(step, self.seed, self.batch_size, self.lengths, self.mode)

    def __len__(self):
        return max(0, self.total - self.start)


def collate(items):
    import torch

    out = {k: torch.stack([it[k] for it in items]) for k in ("image", "state", "action", "action_is_pad")}
    out["stage"] = torch.tensor([it["stage"] for it in items])
    out["task"] = [it["task"] for it in items]
    out["episode"] = [it["episode"] for it in items]
    out["frame"] = [it["frame"] for it in items]
    return out


# ---------------------------------------------------------------------------------------------------------------
def main(argv=None):
    args = parse_args(argv)
    args.out.mkdir(parents=True, exist_ok=True)
    log_file = open(args.out / "train.log", "a", buffering=1)

    def log(msg):
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        log_file.write(line + "\n")

    began = time.monotonic()
    dataset, templates, template_path = build_data(args, log)
    summary = dataset.summary()
    data_hash = dataset.data_hash()
    stats = wd.dataset_norm_stats(dataset, std_floor=args.std_floor)
    stats_hash = hashlib.sha256(json.dumps(stats, sort_keys=True).encode()).hexdigest()
    split_counts = {k: (v["episodes"], v["frames"]) for k, v in summary["splits"].items()}
    log(f"data hash {data_hash[:16]} splits (episodes, frames) {split_counts} excluded {len(summary['excluded'])}")
    for key in (wd.STATE_KEY, wd.ACTION_KEY):
        log(f"stats {key}: floored dims {stats[key]['floored_dims']} (std < {args.std_floor})")
    synthetic = any(e.meta["synthetic"] for e in dataset.episodes)
    atomic_json(args.out / "data_split.json", {**summary, "data_hash": data_hash, "templates": template_path,
                                               "synthetic": synthetic, "data_root": str(args.data.resolve())})
    atomic_json(args.out / "norm_stats.json", {**stats, "sha256": stats_hash, "data_hash": data_hash})
    if args.audit_only:
        log("audit only: done")
        return 0

    import torch

    from smolvla_io import (cast_trainable_fp32, chunk_loss, dtype_report, load_finetuned, load_policy, make_config,
                            make_processors, resolve_base, save_checkpoint, set_trainable, to_batch)

    if not torch.cuda.is_available():
        raise RuntimeError("training requires a CUDA allocation")
    try:  # cu128 wheel has no SM70 kernels: fail before loading anything heavy
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise RuntimeError(f"CUDA kernel preflight failed on {torch.cuda.get_device_name()}") from exc
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.benchmark = False
    device = "cuda"

    # --- resume or fresh -------------------------------------------------------------------------------------
    resume_dir = None
    if args.resume == "auto":
        latest = args.out / "latest.json"
        if latest.is_file():
            resume_dir = Path(json.loads(latest.read_text())["path"])
    elif args.resume != "none":
        resume_dir = Path(args.resume)
    base_dir = resolve_base(args.base)
    run_config = {"base": str(base_dir), "base_model_sha256": None, "chunk_size": args.chunk_size,
                  "train_mode": args.train_mode, "data_hash": data_hash, "stats_sha256": stats_hash,
                  "seed": args.seed, "batch_size": args.batch_size, "steps": args.steps, "lr": args.lr,
                  "sampling": args.sampling, "amp": args.amp, "fp32_trainable": not args.bf16_trainable}
    if resume_dir is not None:
        if not (resume_dir / "trainer.pt").is_file():
            raise SystemExit(f"{resume_dir} has no trainer.pt (a --no-trainer-state run): it cannot be resumed")
        trainer = torch.load(resume_dir / "trainer.pt", map_location="cpu", weights_only=False)
        for key in ("chunk_size", "train_mode", "data_hash", "stats_sha256", "seed", "batch_size", "sampling",
                    "fp32_trainable"):
            if trainer["run_config"][key] != run_config[key]:
                raise ValueError(f"resume mismatch on {key}: {trainer['run_config'][key]} != {run_config[key]}")
        # The checkpoint config carries the freeze flags set by make_config, so construction re-applies them.
        policy, pre, post, cfg, meta = load_finetuned(resume_dir, device=device)
        weights_report = meta["weights"]
        log(f"resuming from {resume_dir} at step {trainer['step']}")
    else:
        trainer = None
        cfg = make_config(base_dir, chunk_size=args.chunk_size, n_action_steps=args.n_action_steps,
                          train_mode=args.train_mode, device=device)
        policy, weights_report = load_policy(base_dir, cfg)
        pre, post = make_processors(cfg, stats)
    trainable = set_trainable(policy, args.train_mode)
    trainable["cast_to_fp32"] = 0 if args.bf16_trainable else cast_trainable_fp32(policy)
    trainable["dtypes"] = dtype_report(policy)
    tokens = check_token_lengths(dataset, cfg)
    log(f"weights {weights_report} trainable {trainable} tokens {tokens}")
    params = [p for p in policy.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=args.lr, betas=(0.9, 0.95), eps=1e-8, weight_decay=args.weight_decay)
    step, best, history = 0, float("inf"), []
    if trainer is not None:
        optimizer.load_state_dict(trainer["optimizer"])
        step, best, history = trainer["step"], trainer["best"], trainer["history"]
        torch.set_rng_state(trainer["torch_rng"])
        torch.cuda.set_rng_state_all(trainer["cuda_rng"])
        random.setstate(trainer["py_rng"])
    policy.to(device)
    policy.train()

    train_idx = dataset.indices("train")
    train_eps = [dataset.episodes[i] for i in train_idx]
    amp_ctx = (lambda: torch.autocast("cuda", dtype=torch.bfloat16)) if args.amp == "bf16" else None

    # --- validation sets (fixed) ------------------------------------------------------------------------------
    val_sets = {}
    for split in ("val_layout", "val_template"):
        eps = [dataset.episodes[i] for i in dataset.indices(split)]
        if eps:
            keys = wd.validation_samples([e.length for e in eps], args.val_per_episode)[: args.val_max_samples]
            val_sets[split] = (eps, keys)
    probe_keys = wd.validation_samples([e.length for e in train_eps], max(1, args.val_per_episode // 2))
    probe_keys = [probe_keys[i] for i in np.random.default_rng(1234).permutation(len(probe_keys))[:256]]
    val_sets["train_probe"] = (train_eps, probe_keys)
    max_dim = cfg.max_action_dim

    def fixed_noise_time(n_keys, offset):
        """Per-sample flow noise and time (SmolVLA's Beta(1.5, 1) time law), fixed by sample position so every
        checkpoint is scored on identical draws regardless of batch composition."""
        noise = torch.empty(n_keys, args.chunk_size, max_dim)
        times = torch.empty(n_keys)
        for j in range(n_keys):
            noise[j] = torch.randn(args.chunk_size, max_dim, generator=torch.Generator().manual_seed(20261005 + offset + j))
            times[j] = float(np.random.default_rng(20261005 + offset + j).beta(1.5, 1.0)) * 0.999 + 0.001
        return noise, times

    def materialize(eps, keys):
        items = [make_item(eps, e, f) for e, f in keys]
        return collate(items)

    def make_item(eps, e, f):
        s = wd.make_sample(eps[e], f, args.chunk_size)
        return {"image": torch.from_numpy(s["image"]), "state": torch.from_numpy(s["state"]),
                "action": torch.from_numpy(s["action"]), "action_is_pad": torch.from_numpy(s["action_is_pad"]),
                "stage": s["stage"], "task": s["task"], "episode": e, "frame": f}

    a_std = torch.tensor(stats[wd.ACTION_KEY]["std"])

    @torch.no_grad()
    def validate():
        policy.eval()
        results = {}
        for split_i, (split, (eps, keys)) in enumerate(sorted(val_sets.items())):
            noise, times = fixed_noise_time(len(keys), 100000 * split_i)
            per_sample, stages = [], []
            for start in range(0, len(keys), args.batch_size):
                b = materialize(eps, keys[start:start + args.batch_size])
                proc = pre(to_batch(b["image"], b["state"], b["task"], device, actions=b["action"]))
                sl = slice(start, start + len(b["task"]))
                if amp_ctx:
                    with amp_ctx():
                        _, ps = chunk_loss(policy, proc, b["action_is_pad"], noise=noise[sl].to(device),
                                           time=times[sl].to(device))
                else:
                    _, ps = chunk_loss(policy, proc, b["action_is_pad"], noise=noise[sl].to(device),
                                       time=times[sl].to(device))
                per_sample.append(ps.float().cpu())
                stages.append(b["stage"])
            ps = torch.cat(per_sample)
            st = torch.cat(stages)
            res = {"flow_loss": float(ps.mean()), "samples": len(keys),
                   "flow_loss_by_stage": {int(s): float(ps[st == s].mean()) for s in torch.unique(st)}}
            # open-loop sampled chunks vs demonstration (unnormalised contract units)
            ol_keys = keys[:: max(1, len(keys) // max(1, args.openloop_samples))][: args.openloop_samples]
            if ol_keys:
                b = materialize(eps, ol_keys)
                proc = pre(to_batch(b["image"], b["state"], b["task"], device))
                ol_noise, _ = fixed_noise_time(len(ol_keys), 100000 * split_i + 50000)
                pred = policy.predict_action_chunk(proc, noise=ol_noise.to(device))
                pred = post(pred).float().cpu()
                gt = b["action"].float()
                valid = (~b["action_is_pad"]).float()[..., None]
                res["openloop"] = {}
                for horizon, h in (("executed", args.n_action_steps), ("chunk", args.chunk_size)):
                    err = (pred[:, :h] - gt[:, :h]) * valid[:, :h]
                    n = valid[:, :h].sum().clamp_min(1)
                    rmse = torch.sqrt((err ** 2).sum(dim=(0, 1)) / n)
                    res["openloop"][horizon] = {
                        "base_rmse": float(torch.sqrt((rmse[:3] ** 2).mean())),
                        "arm_rmse_rad": float(torch.sqrt((rmse[3:13] ** 2).mean())),
                        "gripper_mae": float(err[..., 13:15].abs().sum() / (2 * n)),
                        "normalised_rmse": float(torch.sqrt(((rmse / a_std) ** 2).mean())),
                    }
            results[split] = res
        policy.train()
        return results

    def score(val):
        for split in ("val_layout", "val_template", "train_probe"):
            if split in val:
                return val[split]["flow_loss"], split
        raise AssertionError

    # --- checkpointing ----------------------------------------------------------------------------------------
    script_hashes = {p.name: file_sha(p) for p in sorted(HERE.glob("*.py"))}
    env_info = {"host": socket.gethostname(), "slurm_job": os.environ.get("SLURM_JOB_ID"),
                "gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda,
                "python": platform.python_version(), "scripts_sha256": script_hashes}
    try:
        import lerobot
        env_info["lerobot"] = getattr(lerobot, "__version__", "unknown")
    except Exception:  # pragma: no cover
        pass

    def save(val, train_loss):
        nonlocal best
        directory = args.out / f"step_{step:06d}"
        staging = args.out / f".step_{step:06d}.pending"
        if directory.exists():
            raise FileExistsError(f"refusing to overwrite {directory}")
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir()
        metric, metric_split = score(val)
        is_best = metric <= best
        best = min(best, metric)
        save_checkpoint(policy, pre, post, staging, {
            "step": step, "validation": val, "selection_metric": {"split": metric_split, "flow_loss": metric},
            "train_loss": train_loss, "run_config": run_config, "data_hash": data_hash, "synthetic": synthetic,
            "n_action_steps": args.n_action_steps, "chunk_size": args.chunk_size, "control_hz": wd.CONTROL_HZ,
            "image_key": wd.IMAGE_KEY, "env": env_info,
            "note": "validation losses are not task success; closed-loop rollouts measure success"})
        if not args.no_trainer_state:
            torch.save({"step": step, "best": best, "history": history, "optimizer": optimizer.state_dict(),
                        "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                        "py_rng": random.getstate(), "run_config": run_config}, staging / "trainer.pt")
        os.replace(staging, directory)
        atomic_json(args.out / "latest.json", {"path": str(directory.resolve()), "step": step})
        if is_best:
            atomic_json(args.out / "best.json", {"path": str(directory.resolve()), "step": step,
                                                 "metric": {"split": metric_split, "flow_loss": metric}})
        # Retention: the newest keep_last step dirs plus the best one (best.json points at it; never copied).
        best_name = Path(json.loads((args.out / "best.json").read_text())["path"]).name
        steps_saved = sorted(p for p in args.out.glob("step_*") if p.is_dir())
        for old in steps_saved[: -args.keep_last]:
            if old.name != best_name:
                shutil.rmtree(old)
        return is_best

    metrics_path = args.out / "metrics.jsonl"

    def record(row):
        with open(metrics_path, "a") as f:
            f.write(json.dumps(row) + "\n")

    if step == 0:
        val = validate()
        history.append({"step": 0, "validation": val})
        record({"step": 0, "validation": val})
        log(f"step 0 (pretrained, before fine-tune) validation {json.dumps(val)}")
        save(val, None)

    # --- training loop -----------------------------------------------------------------------------------------
    dataset_t = wd.make_chunk_dataset(train_eps, args.chunk_size)
    sampler = StepBatchSampler([e.length for e in train_eps], args.batch_size, args.seed, step, args.steps,
                               args.sampling)
    loader = torch.utils.data.DataLoader(dataset_t, batch_sampler=sampler, num_workers=args.num_workers,
                                         collate_fn=collate, pin_memory=True,
                                         prefetch_factor=4 if args.num_workers > 0 else None)
    t_last, wait, n_seen, last_loss = time.monotonic(), 0.0, 0, None
    t_fetch = time.monotonic()
    stopped_early = False
    for batch in loader:
        wait += time.monotonic() - t_fetch
        lr = lr_at(step, args)
        for group in optimizer.param_groups:
            group["lr"] = lr
        proc = pre(to_batch(batch["image"], batch["state"], batch["task"], device, actions=batch["action"]))
        if amp_ctx:
            with amp_ctx():
                loss, _ = chunk_loss(policy, proc, batch["action_is_pad"])
        else:
            loss, _ = chunk_loss(policy, proc, batch["action_is_pad"])
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = torch.nn.utils.clip_grad_norm_(params, args.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        n_seen += len(batch["task"])
        last_loss = float(loss)
        if step % args.log_every == 0 or step == args.steps:
            now = time.monotonic()
            row = {"step": step, "loss": last_loss, "lr": lr, "grad_norm": float(gnorm),
                   "samples_per_s": n_seen / max(1e-9, now - t_last), "data_wait_frac": wait / max(1e-9, now - t_last),
                   "gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9}
            record(row)
            log(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            t_last, wait, n_seen = now, 0.0, 0
        out_of_time = args.max_minutes is not None and (time.monotonic() - began) / 60 > args.max_minutes
        if step % args.save_every == 0 or step == args.steps or out_of_time:
            val = validate()
            history.append({"step": step, "train_loss": last_loss, "validation": val})
            record({"step": step, "validation": val})
            is_best = save(val, last_loss)
            log(f"saved step {step} best={is_best} validation {json.dumps(val)}")
            t_last, wait, n_seen = time.monotonic(), 0.0, 0  # throughput excludes validation/saving
        if out_of_time and step < args.steps:
            stopped_early = True
            log(f"wall-time budget {args.max_minutes} min reached at step {step}; resume with --resume auto")
            break
        t_fetch = time.monotonic()
    status = {"steps_done": step, "steps_requested": args.steps, "complete": step >= args.steps,
              "stopped_early": stopped_early, "elapsed_s": time.monotonic() - began, "data_hash": data_hash,
              "best_metric": best, "synthetic": synthetic, "env": env_info}
    atomic_json(args.out / ("completed.json" if step >= args.steps else "interrupted.json"), status)
    log(f"done: {status['steps_done']}/{args.steps} steps")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
