#!/usr/bin/env python3
"""Train SmolVLA with garment-held-out validation and resumable checkpoints.

Reads the existing LeHome capture archives without modifying the sibling repo.
Full RGB frames are cached as uncompressed memory maps: no resize/domain change,
and no 50+ GB concatenate peak. The cache is reconstructible local scratch.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import time

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from bhl_robust.cloth.folding_data import capture_manifest, split_captures, stratified_success_split


def atomic_json(path, obj):
    path = Path(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


def prepare(rows, cache):
    prepared = []
    cache.mkdir(parents=True, exist_ok=True)
    for row in rows:
        key = hashlib.sha256(json.dumps(row, sort_keys=True).encode()).hexdigest()[:20]
        directory = cache / key
        directory.mkdir(exist_ok=True)
        if not (directory / "ready.json").is_file():
            with np.load(row["path"], allow_pickle=False) as source:
                for name in ("images", "state", "action"):
                    value = source[name]
                    if name == "images" and (value.shape != (row["frames"], 3, 480, 640, 3)
                                             or value.dtype != np.uint8):
                        raise ValueError(f"invalid RGB capture: {row['path']}: {value.shape}")
                    tmp = directory / f"{name}.tmp.npy"
                    np.save(tmp, value, allow_pickle=False)
                    os.replace(tmp, directory / f"{name}.npy")
            atomic_json(directory / "ready.json", row)
        prepared.append({**row, "cache": str(directory)})
        print(f"[fold] cache {row['garment']} episode {row['episode']}", flush=True)
    return prepared


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--policy-path", required=True, type=Path)
    p.add_argument("--captures", required=True)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--cache", type=Path)
    p.add_argument("--steps", type=int, default=1500)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--save-every", type=int, default=100)
    p.add_argument("--val-per-episode", type=int, default=8)
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--unfreeze", choices=("vision", "vision+action"), default="vision+action")
    p.add_argument("--resume", type=Path)
    p.add_argument("--audit-only", action="store_true")
    p.add_argument("--smoke", action="store_true", help="use two train captures and one held-out capture; pipeline check only")
    p.add_argument("--validation-mode", choices=("fixed-ids", "stratified-success"), default="fixed-ids")
    p.add_argument("--class-balanced", action="store_true", help="sample garment class, then episode, then frame uniformly")
    args = p.parse_args()
    if min(args.steps, args.batch_size, args.save_every, args.val_per_episode) < 1:
        p.error("steps, batch size, checkpoint interval and validation samples must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    split_fn = stratified_success_split if args.validation_mode == "stratified-success" else split_captures
    manifest = split_fn(capture_manifest(args.captures))
    if args.class_balanced:
        manifest["sampling"] = "uniform class, uniform episode, uniform frame"
    if args.smoke:
        manifest["train"] = manifest["train"][:2]
        manifest["validation"] = manifest["validation"][:1]
        manifest["smoke_only"] = True
    serialized = json.dumps(manifest, sort_keys=True)
    data_hash = hashlib.sha256(serialized.encode()).hexdigest()
    atomic_json(args.out / "data_split.json", {**manifest, "sha256": data_hash})
    counts = {k: len(manifest[k]) for k in ("train", "validation", "excluded_failed_replays")}
    print(f"[fold] split {counts}, sha256={data_hash}", flush=True)
    if args.audit_only:
        return 0
    if args.cache is None:
        p.error("--cache is required for training (approximately 30 GB of disposable scratch)")

    import torch
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    if not torch.cuda.is_available():
        raise RuntimeError("fold policy training requires a CUDA allocation")
    # The installed cu128 wheel excludes V100/SM70. Test an actual CUDA kernel
    # before decompressing gigabytes of captures or loading the policy.
    try:
        torch.ones(1, device="cuda").add_(1)
        torch.cuda.synchronize()
    except RuntimeError as exc:
        raise RuntimeError(f"Torch GPU kernel preflight failed on {torch.cuda.get_device_name()}; "
                           "use a GPU supported by this wheel or an isolated compatible stack") from exc
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    rows = prepare(manifest["train"] + manifest["validation"], args.cache)
    arrays = [{k: np.load(Path(row["cache"]) / f"{k}.npy", mmap_mode="r")
               for k in ("images", "state", "action")} for row in rows]
    n_train = len(manifest["train"])
    class_indices = {}
    for index, row in enumerate(rows[:n_train]):
        class_indices.setdefault(row["garment"].split("_Seen_")[0], []).append(index)
    classes = sorted(class_indices)
    source = args.resume or args.policy_path
    cfg = PreTrainedConfig.from_pretrained(str(source), cli_overrides={})
    cfg.pretrained_path = str(source)
    policy = SmolVLAPolicy.from_pretrained(str(source)).cuda()
    pre, _ = make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(source))
    prefixes = ("model.vlm_with_expert.lm_expert", "model.action_in_proj", "model.action_out_proj",
                "model.action_time_mlp", "model.state_proj")
    for name, par in policy.named_parameters():
        vision = any(k in name.lower() for k in ("vision", "image", "patch", "visual"))
        par.requires_grad_(vision or (args.unfreeze == "vision+action" and name.startswith(prefixes)))
    params = [v for v in policy.parameters() if v.requires_grad]
    if not params:
        raise RuntimeError("no trainable parameters")
    optimizer = torch.optim.AdamW(params, lr=args.lr)
    step, best, history = 0, float("inf"), []
    if args.resume:
        state = torch.load(args.resume / "trainer.pt", map_location="cpu", weights_only=False)
        if state["data_hash"] != data_hash or state["unfreeze"] != args.unfreeze:
            raise ValueError("resume data split or trainable parameter set changed")
        optimizer.load_state_dict(state["optimizer"])
        for opt_state in optimizer.state.values():
            for key, value in opt_state.items():
                if torch.is_tensor(value):
                    opt_state[key] = value.cuda()
        step, best, history = state["step"], state["best"], state["history"]
        rng.bit_generator.state = state["numpy_rng"]
        torch.set_rng_state(state["torch_rng"])
        torch.cuda.set_rng_state_all(state["cuda_rng"])

    keys = ("observation.images.top_rgb", "observation.images.left_rgb", "observation.images.right_rgb")

    def batch(samples):
        image_batch = np.stack([arrays[e]["images"][f] for e, f in samples])
        b = {key: torch.from_numpy(image_batch[:, j]).permute(0, 3, 1, 2).cuda().float() / 255
             for j, key in enumerate(keys)}
        b["observation.state"] = torch.from_numpy(np.stack([arrays[e]["state"][f] for e, f in samples])).cuda()
        b["action"] = torch.from_numpy(np.stack([arrays[e]["action"][f] for e, f in samples])).cuda()
        b["task"] = ["fold the garment"] * len(samples)
        return pre(b)

    def loss_on(samples):
        out = policy.forward(batch(samples))
        return out[0] if isinstance(out, tuple) else out["loss"]

    val_samples = [(e, int(f)) for e in range(n_train, len(rows))
                   for f in np.linspace(0, rows[e]["frames"] - 1, args.val_per_episode)]

    def validate():
        policy.eval()
        losses = []
        # Fixed flow noise makes checkpoints comparable; preserve training RNG.
        with torch.random.fork_rng(devices=[torch.cuda.current_device()]), torch.no_grad():
            torch.manual_seed(20260919)
            for start in range(0, len(val_samples), args.batch_size):
                samples = val_samples[start:start + args.batch_size]
                losses.extend([float(loss_on(samples))] * len(samples))
        policy.train()
        return float(np.mean(losses))

    def save(val):
        nonlocal best
        directory = args.out / f"step_{step:06d}"
        staging = args.out / f".step_{step:06d}.pending"
        if directory.exists():
            raise FileExistsError(f"refusing to overwrite checkpoint {directory}")
        staging.mkdir(exist_ok=True)
        policy.save_pretrained(staging)
        for path in args.policy_path.iterdir():
            if path.name.startswith("policy_preprocessor") or path.name.startswith("policy_postprocessor"):
                shutil.copy2(path, staging / path.name)
        best = min(best, val)
        torch.save({"step": step, "best": best, "history": history,
                    "optimizer": optimizer.state_dict(), "numpy_rng": rng.bit_generator.state,
                    "torch_rng": torch.get_rng_state(), "cuda_rng": torch.cuda.get_rng_state_all(),
                    "data_hash": data_hash, "unfreeze": args.unfreeze}, staging / "trainer.pt")
        atomic_json(staging / "fold_training.json", {"step": step, "validation_loss": val,
                    "best_validation_loss": best, "data_hash": data_hash, "base": str(args.policy_path),
                    "unfreeze": args.unfreeze, "success_metric": "not evaluated; loss is not a fold rate"})
        os.replace(staging, directory)
        atomic_json(args.out / "latest.json", {"path": str(directory.resolve()), "step": step})
        if val <= best:
            atomic_json(args.out / "best.json", {"path": str(directory.resolve()), "step": step})

    if step == 0:
        val = validate()
        history.append({"step": 0, "validation_loss": val})
        save(val)
    began = time.monotonic()
    policy.train()
    while step < args.steps:
        # Balanced episodes, no class dominance from episode length.
        if args.class_balanced:
            episode_ids = [int(rng.choice(class_indices[classes[int(c)]]))
                           for c in rng.integers(len(classes), size=args.batch_size)]
        else:
            episode_ids = rng.integers(n_train, size=args.batch_size)
        samples = [(int(e), int(rng.integers(rows[int(e)]["frames"]))) for e in episode_ids]
        loss = loss_on(samples)
        if not torch.isfinite(loss):
            raise RuntimeError(f"nonfinite training loss at step {step}")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0, error_if_nonfinite=True)
        optimizer.step()
        step += 1
        if step % 10 == 0:
            print(f"[fold] step={step}/{args.steps} loss={float(loss):.5f}", flush=True)
        if step % args.save_every == 0 or step == args.steps:
            val = validate()
            if not np.isfinite(val):
                raise RuntimeError("nonfinite validation loss")
            history.append({"step": step, "train_loss": float(loss), "validation_loss": val})
            save(val)
            print(f"[fold] saved step={step} validation={val:.5f}", flush=True)
    atomic_json(args.out / "completed.json", {"steps": step, "elapsed_s": time.monotonic() - began,
                "data_hash": data_hash, "history": history})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
