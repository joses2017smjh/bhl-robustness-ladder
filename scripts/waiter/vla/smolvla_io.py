"""lerobot 0.4.3 SmolVLA glue for the waiter (shared by train_smolvla.py and rollout_vla.py).

Facts this module relies on (read from venv-lehome-train, lerobot 0.4.3):
- ``lerobot/smolvla_base`` stores all 450 M parameters (VLM truncated to 16 layers + action expert) in its
  ``model.safetensors``. Its config says ``load_vlm_weights: true``, which would make the constructor download
  SmolVLM2-500M weights first; we force it False and load everything from the safetensors, then verify that every
  parameter of the constructed model was present in the file (``verify_weight_keys``).
- State/action are zero-padded to ``max_state_dim``/``max_action_dim`` = 32 inside the policy, so the waiter's
  30-D state and 15-D action fit the pretrained ``state_proj``/``action_in_proj`` shapes unchanged.
- Images: float [0,1] (B,3,H,W) -> ``resize_with_pad`` to 512x512 -> [-1,1] for SigLIP. The 224x224 head
  camera is upsampled; 512 keeps the pretrained 64-token-per-image layout (224 itself is not a valid grid for the
  pixel-shuffle connector: 224/16 = 14 is not divisible by 4).
- ``SmolVLAPolicy.forward`` looks for the padding mask under the misspelt key ``actions_id_pad`` (dataset batches
  carry ``action_is_pad``), so in 0.4.3 the episode-end mask is silently never applied. ``chunk_loss`` applies it.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from waiter_vla_data import ACTION_DIM, ACTION_KEY, IMAGE_KEY, STATE_DIM, STATE_KEY

BASE_REPO = "lerobot/smolvla_base"
TRAIN_MODES = ("expert", "expert+vision", "full")


def resolve_base(path_or_repo: str) -> Path:
    p = Path(path_or_repo)
    if p.is_dir():
        return p
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(path_or_repo, local_files_only=True))


def make_config(base_dir: Path, *, chunk_size: int, n_action_steps: int, train_mode: str = "expert",
                device: str = "cuda", num_steps: int | None = None):
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401  (registers the config subclass)
    from lerobot.configs.policies import PreTrainedConfig
    from lerobot.configs.types import FeatureType, PolicyFeature

    if train_mode not in TRAIN_MODES:
        raise ValueError(f"train_mode must be one of {TRAIN_MODES}")
    if not 1 <= n_action_steps <= chunk_size:
        raise ValueError("need 1 <= n_action_steps <= chunk_size")
    cfg = PreTrainedConfig.from_pretrained(str(base_dir))
    if cfg.type != "smolvla":
        raise ValueError(f"{base_dir} is a {cfg.type} policy, not smolvla")
    cfg.load_vlm_weights = False
    cfg.input_features = {
        IMAGE_KEY: PolicyFeature(type=FeatureType.VISUAL, shape=(3, 224, 224)),
        STATE_KEY: PolicyFeature(type=FeatureType.STATE, shape=(STATE_DIM,)),
    }
    cfg.output_features = {ACTION_KEY: PolicyFeature(type=FeatureType.ACTION, shape=(ACTION_DIM,))}
    cfg.chunk_size = int(chunk_size)
    cfg.n_action_steps = int(n_action_steps)
    cfg.empty_cameras = 0
    cfg.device = device
    if num_steps is not None:
        cfg.num_steps = int(num_steps)
    cfg.freeze_vision_encoder = train_mode == "expert" or train_mode == "expert+vision"
    cfg.train_expert_only = train_mode != "full"
    cfg.train_state_proj = True
    cfg.pretrained_path = str(base_dir)
    if STATE_DIM > cfg.max_state_dim or ACTION_DIM > cfg.max_action_dim:
        raise ValueError("waiter state/action exceed SmolVLA padding dims")
    return cfg


def verify_weight_keys(policy, model_file: Path) -> dict:
    """Every parameter/buffer of the constructed model must come from the checkpoint file (no random init)."""
    from safetensors import safe_open

    with safe_open(str(model_file), framework="pt") as f:
        file_keys = set(f.keys())
    model_keys = set(policy.state_dict().keys())
    missing = sorted(model_keys - file_keys)
    unexpected = sorted(file_keys - model_keys)
    if missing:
        raise RuntimeError(f"{len(missing)} model tensors absent from {model_file} (would stay randomly "
                           f"initialised), e.g. {missing[:5]}")
    return {"file_tensors": len(file_keys), "model_tensors": len(model_keys), "unexpected": unexpected[:20],
            "n_unexpected": len(unexpected)}


_SAFETENSORS_DTYPES = {"F32": "float32", "BF16": "bfloat16", "F16": "float16", "F64": "float64"}


def match_file_dtypes(policy, model_file: Path) -> int:
    """Give every floating parameter the dtype it has in the checkpoint file, then reload it exactly.

    ``load_state_dict`` copies into the dtypes chosen at construction, and construction makes the action expert
    bf16 (it copies SmolVLM2's text config). A checkpoint written with fp32 master weights would otherwise be
    silently rounded to bf16 on load (resume or rollout). Returns the number of tensors whose dtype changed.
    """
    import torch
    from safetensors import safe_open
    from safetensors.torch import load_model

    with safe_open(str(model_file), framework="pt") as f:
        file_dtypes = {k: _SAFETENSORS_DTYPES.get(f.get_slice(k).get_dtype()) for k in f.keys()}
    changed = 0
    for name, par in policy.named_parameters():
        want = file_dtypes.get(name)
        if want and par.dtype.is_floating_point and str(par.dtype) != f"torch.{want}":
            par.data = par.data.to(getattr(torch, want))
            changed += 1
    if changed:
        device = next(policy.parameters()).device
        load_model(policy, str(model_file), strict=False, device=str(device))
    return changed


def load_policy(weights_dir: Path, cfg):
    from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy

    policy = SmolVLAPolicy.from_pretrained(str(weights_dir), config=cfg, strict=False)
    model_file = Path(weights_dir) / "model.safetensors"
    report = verify_weight_keys(policy, model_file)
    report["dtype_matched_tensors"] = match_file_dtypes(policy, model_file)
    return policy, report


def cast_trainable_fp32(policy) -> int:
    """fp32 master weights for every trainable parameter (the expert is constructed in bf16: pure-bf16 AdamW with
    lr <= 1e-4 rounds most updates away, bf16 having an 8-bit mantissa). The forward casts activations to each
    layer's weight dtype, so a fp32 expert next to the frozen bf16 VLM is supported by lerobot's code."""
    import torch

    n = 0
    for par in policy.parameters():
        if par.requires_grad and par.dtype != torch.float32:
            par.data = par.data.float()
            n += 1
    return n


def dtype_report(policy) -> dict:
    out: dict = {}
    for par in policy.parameters():
        key = f"{'trainable' if par.requires_grad else 'frozen'}_{str(par.dtype).replace('torch.', '')}"
        out[key] = out.get(key, 0) + par.numel()
    return out


def set_trainable(policy, train_mode: str) -> dict:
    """Apply the trainable-parameter set. 'expert' = SmolVLA's published fine-tune recipe (VLM frozen; action
    expert + state/action/time projections trained); 'expert+vision' additionally trains the SigLIP encoder and
    connector (the cloth project's choice for a rendered-image domain gap); 'full' trains everything lerobot allows.
    """
    if train_mode == "expert+vision":
        vlm = policy.model.vlm_with_expert.get_vlm_model()
        for module in (vlm.vision_model, vlm.connector):
            for par in module.parameters():
                par.requires_grad_(True)
    total = sum(p.numel() for p in policy.parameters())
    trainable = sum(p.numel() for p in policy.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("no trainable parameters")
    return {"train_mode": train_mode, "params_total": total, "params_trainable": trainable}


def stats_to_tensors(stats: dict) -> dict:
    import torch

    return {key: {k: torch.tensor(np.asarray(v, dtype=np.float32)) for k, v in s.items() if k in ("mean", "std")}
            for key, s in stats.items()}


def make_processors(cfg, stats: dict):
    from lerobot.policies.smolvla.processor_smolvla import make_smolvla_pre_post_processors

    return make_smolvla_pre_post_processors(cfg, dataset_stats=stats_to_tensors(stats))


def load_processors(cfg, checkpoint: Path):
    from lerobot.policies.factory import make_pre_post_processors

    return make_pre_post_processors(policy_cfg=cfg, pretrained_path=str(checkpoint))


def to_batch(images, states, tasks, device, actions=None):
    """Raw contract observations -> the dict the SmolVLA preprocessor expects.

    images: (B,224,224,3) uint8 (numpy or torch); states: (B,30) float32; tasks: list[str].
    """
    import torch

    images = torch.as_tensor(images)
    if images.dtype != torch.uint8 or images.ndim != 4 or images.shape[-1] != 3:
        raise ValueError(f"images must be (B,H,W,3) uint8, got {tuple(images.shape)} {images.dtype}")
    batch = {
        IMAGE_KEY: images.to(device, non_blocking=True).permute(0, 3, 1, 2).float().div_(255.0),
        STATE_KEY: torch.as_tensor(states, dtype=torch.float32).to(device, non_blocking=True),
        "task": list(tasks),
    }
    if actions is not None:
        batch[ACTION_KEY] = torch.as_tensor(actions, dtype=torch.float32).to(device, non_blocking=True)
    return batch


def chunk_loss(policy, processed: dict, is_pad, *, noise=None, time=None):
    """Flow-matching loss with the episode-end mask lerobot 0.4.3 drops.

    Returns (loss_for_backprop, per_sample_real_dim_loss). The backprop loss averages over all 32 padded action
    dims (as lerobot does: the pretrained expert learns to keep the zero-padded dims on the noise path, which keeps
    inference consistent) but only over in-episode chunk steps. The per-sample loss is over the 15 real dims.
    """
    _, info = policy.forward(processed, noise=noise, time=time)
    losses = info["losses_after_forward"]  # (B, chunk, max_action_dim), keeps the graph
    valid = (~is_pad.to(losses.device)).float()  # (B, chunk)
    denom = valid.sum().clamp_min(1.0)
    loss = (losses.mean(dim=-1) * valid).sum() / denom
    real = losses[..., :ACTION_DIM].mean(dim=-1)
    per_sample = (real * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
    return loss, per_sample


def save_checkpoint(policy, pre, post, directory: Path, extra: dict) -> None:
    policy.save_pretrained(directory)
    pre.save_pretrained(directory)
    post.save_pretrained(directory)
    (directory / "waiter_vla.json").write_text(json.dumps(extra, indent=2, allow_nan=False) + "\n")


def load_finetuned(checkpoint: Path, *, device: str = "cuda", n_action_steps: int | None = None,
                   num_steps: int | None = None):
    """Load a checkpoint written by train_smolvla.py (directory, or a latest/best .json pointer)."""
    import lerobot.policies.smolvla.configuration_smolvla  # noqa: F401
    from lerobot.configs.policies import PreTrainedConfig

    checkpoint = Path(checkpoint)
    if checkpoint.is_file() and checkpoint.suffix == ".json":
        checkpoint = Path(json.loads(checkpoint.read_text())["path"])
    if not (checkpoint / "model.safetensors").is_file():
        raise FileNotFoundError(f"no model.safetensors in {checkpoint}")
    cfg = PreTrainedConfig.from_pretrained(str(checkpoint))
    cfg.device = device
    cfg.load_vlm_weights = False
    if n_action_steps is not None:
        if not 1 <= n_action_steps <= cfg.chunk_size:
            raise ValueError(f"n_action_steps must be in 1..{cfg.chunk_size}")
        cfg.n_action_steps = int(n_action_steps)
    if num_steps is not None:
        cfg.num_steps = int(num_steps)
    policy, report = load_policy(checkpoint, cfg)
    pre, post = load_processors(cfg, checkpoint)
    meta_path = checkpoint / "waiter_vla.json"
    meta = json.loads(meta_path.read_text()) if meta_path.is_file() else {}
    return policy, pre, post, cfg, {"checkpoint": str(checkpoint), "weights": report, **meta}
