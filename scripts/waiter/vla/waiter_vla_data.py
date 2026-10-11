"""Waiter phase 3 data layer: contract npz episodes -> SmolVLA training samples.

Everything here follows the frozen contract in ``docs/WAITER_INTERFACES.md``:

- one ``.npz`` per episode with ``image`` (T,224,224,3) uint8, ``state`` (T,30) f32, ``action`` (T,15) f32 (the
  action applied after observing step t), ``instruction`` str, ``seed`` int, ``success`` bool, ``stage`` (T,) int8,
  plus a ``manifest.json`` listing episodes with their seeds, success and sha256;
- 12.5 Hz high-level control, action = base velocity (3) + arm targets (10) + gripper closures (2).

The numpy core (validation, manifest, template matching, splits, cache, statistics, action chunks, deterministic
sampling) has no torch/lerobot dependency so it is unit-testable in any interpreter. ``ChunkDataset`` imports torch
lazily.

Seed hygiene: training data may only come from the phase 2 demonstration range 3000-3999 (and, for development,
1000-1999). Seeds in the scored ranges are refused outright. Synthetic episodes (``synthetic`` field true, written
only by ``make_synthetic_episodes.py``) may use made-up seeds outside every contract range.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

# ---------------------------------------------------------------------------------------------------------------
# Contract constants
# ---------------------------------------------------------------------------------------------------------------
IMAGE_SHAPE = (224, 224, 3)
STATE_DIM = 30
ACTION_DIM = 15
CONTROL_HZ = 12.5
CONTROL_DT = 0.08
EPISODE_LIMIT_S = 60.0
BASE_LIMITS = np.array([0.4, 0.3, 0.6], dtype=np.float32)  # |vx|, |vy|, |wz|
GRIPPER_CLOSED_RAD = 1.20
ACTION_SLICES = {"base": slice(0, 3), "arms": slice(3, 13), "grippers": slice(13, 15)}
STATE_SLICES = {"joints": slice(0, 24), "gravity": slice(24, 27), "ang_vel": slice(27, 30)}

# SmolVLA / lerobot feature keys used everywhere in phase 3.
IMAGE_KEY = "observation.images.head"
STATE_KEY = "observation.state"
ACTION_KEY = "action"

# Frozen seed ranges (inclusive).
SEED_RANGES = {
    "dev": (1000, 1999),
    "phase2_gate": (2000, 2099),
    "phase2_demos": (3000, 3999),
    "phase3_scored": (4000, 4099),
    "phase4_train": (5000, 5999),
    "phase4_scored": (6000, 6099),
}
TRAINING_RANGES = ("phase2_demos",)  # + "dev" with allow_dev_seeds
NEVER_TRAIN_RANGES = ("phase2_gate", "phase3_scored", "phase4_train", "phase4_scored")


class EpisodeError(ValueError):
    """An episode violates the data contract."""


def seed_range_name(seed: int) -> str | None:
    for name, (lo, hi) in SEED_RANGES.items():
        if lo <= seed <= hi:
            return name
    return None


def check_training_seed(seed: int, *, synthetic: bool, allow_dev_seeds: bool) -> None:
    """Raise if an episode with this seed must not be used for training."""
    name = seed_range_name(seed)
    if name in NEVER_TRAIN_RANGES:
        raise EpisodeError(f"seed {seed} is in the frozen '{name}' range; it can never be training data")
    if synthetic:
        if name is not None and name != "dev":
            raise EpisodeError(f"synthetic episode uses contract seed {seed} ({name}); use made-up ids")
        return
    allowed = set(TRAINING_RANGES) | ({"dev"} if allow_dev_seeds else set())
    if name not in allowed:
        raise EpisodeError(f"seed {seed} ({name}) is outside the training ranges {sorted(allowed)}")


# ---------------------------------------------------------------------------------------------------------------
# Episode loading and validation
# ---------------------------------------------------------------------------------------------------------------
def _scalar(value, kind: str):
    arr = np.asarray(value)
    if arr.ndim != 0:
        if arr.size != 1:
            raise EpisodeError(f"field expected to be a scalar, got shape {arr.shape}")
        arr = arr.reshape(())
    item = arr.item()
    if kind == "str":
        if isinstance(item, bytes):
            item = item.decode("utf-8")
        if not isinstance(item, str):
            raise EpisodeError(f"expected a string, got {type(item).__name__}")
        return item
    if kind == "int":
        if isinstance(item, bool) or not isinstance(item, (int, np.integer)):
            raise EpisodeError(f"expected an integer, got {type(item).__name__}")
        return int(item)
    if kind == "bool":
        if not isinstance(item, (bool, np.bool_)):
            raise EpisodeError(f"expected a bool, got {type(item).__name__}")
        return bool(item)
    raise AssertionError(kind)


def load_episode(path: str | Path, *, strict: bool = True) -> dict:
    """Load and validate one contract episode. Raises EpisodeError on any violation.

    ``strict`` also enforces the action value ranges of the contract (base velocity limits, gripper in [0, 1]);
    the stored action is the action *applied*, which the environment clips, so out-of-range values indicate a
    recording bug.
    """
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            keys = set(z.files)
            required = {"image", "state", "action", "instruction", "seed", "success", "stage"}
            missing = required - keys
            if missing:
                raise EpisodeError(f"{path.name}: missing fields {sorted(missing)}")
            ep = {
                "image": z["image"],
                "state": z["state"],
                "action": z["action"],
                "stage": z["stage"],
                "instruction": _scalar(z["instruction"], "str"),
                "seed": _scalar(z["seed"], "int"),
                "success": _scalar(z["success"], "bool"),
                "synthetic": _scalar(z["synthetic"], "bool") if "synthetic" in keys else False,
                "template_id": _scalar(z["template_id"], "str") if "template_id" in keys else None,
                "instruction_id": _scalar(z["instruction_id"], "int") if "instruction_id" in keys else None,
                "extra_fields": sorted(keys - required - {"synthetic", "template_id", "instruction_id"}),
            }
    except EpisodeError:
        raise
    except Exception as exc:  # corrupt zip, pickled object arrays, ...
        raise EpisodeError(f"{path.name}: unreadable episode ({exc!r})") from exc
    problems = validate_arrays(ep, strict=strict)
    if problems:
        raise EpisodeError(f"{path.name}: " + "; ".join(problems))
    return ep


def validate_arrays(ep: dict, *, strict: bool = True) -> list[str]:
    problems = []
    image, state, action, stage = ep["image"], ep["state"], ep["action"], ep["stage"]
    if image.ndim != 4 or image.shape[1:] != IMAGE_SHAPE:
        problems.append(f"image shape {image.shape} != (T,{IMAGE_SHAPE})")
    if image.dtype != np.uint8:
        problems.append(f"image dtype {image.dtype} != uint8")
    t = image.shape[0] if image.ndim >= 1 else -1
    if t < 1:
        problems.append("empty episode")
    for name, arr, dim, dtype in (("state", state, STATE_DIM, np.float32), ("action", action, ACTION_DIM, np.float32)):
        if arr.shape != (t, dim):
            problems.append(f"{name} shape {arr.shape} != ({t},{dim})")
        if arr.dtype != dtype:
            problems.append(f"{name} dtype {arr.dtype} != float32")
        elif arr.size and not np.isfinite(arr).all():
            problems.append(f"{name} has non-finite values")
    if stage.shape != (t,):
        problems.append(f"stage shape {stage.shape} != ({t},)")
    if stage.dtype != np.int8:
        problems.append(f"stage dtype {stage.dtype} != int8")
    if not ep["instruction"].strip():
        problems.append("empty instruction")
    if strict and not problems:
        tol = 1e-4
        base = np.abs(action[:, ACTION_SLICES["base"]])
        if (base > BASE_LIMITS + tol).any():
            problems.append(f"base velocity exceeds contract limits (max |a| {base.max(axis=0).tolist()})")
        grip = action[:, ACTION_SLICES["grippers"]]
        if grip.min() < -tol or grip.max() > 1 + tol:
            problems.append(f"gripper closure outside [0,1] ({grip.min():.4f}..{grip.max():.4f})")
    return problems


def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            block = f.read(chunk)
            if not block:
                break
            h.update(block)
    return h.hexdigest()


# ---------------------------------------------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class EpisodeEntry:
    path: Path
    seed: int
    success: bool
    sha256: str | None


_PATH_KEYS = ("file", "path", "npz", "name", "filename")


def read_manifest(root: str | Path, *, require_manifest: bool = True) -> list[EpisodeEntry]:
    """Read ``manifest.json`` in ``root`` (contract) or, if allowed, fall back to globbing ``*.npz``.

    Accepted manifest shapes: ``{"episodes": [...]}`` or a bare list; each entry needs a path-like key (one of
    file/path/npz/name/filename, relative to the manifest), ``seed``, ``success`` and ``sha256``.
    """
    root = Path(root)
    manifest = root / "manifest.json"
    if not manifest.is_file():
        if require_manifest:
            raise FileNotFoundError(f"{manifest} missing (the contract requires it; --no-manifest to glob)")
        return [EpisodeEntry(p, -1, False, None) for p in sorted(root.glob("*.npz"))]
    data = json.loads(manifest.read_text())
    rows = data.get("episodes") if isinstance(data, dict) else data
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"{manifest}: no episode list")
    entries = []
    for i, row in enumerate(rows):
        key = next((k for k in _PATH_KEYS if k in row), None)
        if key is None:
            raise ValueError(f"{manifest}: entry {i} has no path key {_PATH_KEYS}")
        for need in ("seed", "success"):
            if need not in row:
                raise ValueError(f"{manifest}: entry {i} lacks '{need}'")
        path = Path(row[key])
        entries.append(EpisodeEntry(path if path.is_absolute() else root / path, int(row["seed"]),
                                    bool(row["success"]), row.get("sha256")))
    paths = [e.path for e in entries]
    if len(set(paths)) != len(paths):
        raise ValueError(f"{manifest}: duplicate episode paths")
    return entries


# ---------------------------------------------------------------------------------------------------------------
# Instruction templates
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class TemplateSet:
    """Instruction templates declared by phase 2.

    JSON format::

        {"templates": {"T0": "pick up the {colour} tumbler, put it on the tray and bring the tray to the {side} table",
                       ...},
         "slots": {"colour": ["red", "blue"], "side": ["left", "right"]},   # optional; else any text
         "held_out": ["T3"]}                                                # templates never trained on
    """

    templates: dict[str, str]
    slots: dict[str, list[str]] = field(default_factory=dict)
    held_out: list[str] = field(default_factory=list)

    def __post_init__(self):
        unknown = set(self.held_out) - set(self.templates)
        if unknown:
            raise ValueError(f"held-out templates not declared: {sorted(unknown)}")
        if self.templates and set(self.held_out) == set(self.templates):
            raise ValueError("every template is held out; nothing left to train on")
        self._regex = {tid: self._compile(t) for tid, t in self.templates.items()}

    @classmethod
    def load(cls, path: str | Path) -> "TemplateSet":
        data = json.loads(Path(path).read_text())
        return cls(dict(data["templates"]), dict(data.get("slots", {})), list(data.get("held_out", [])))

    def _compile(self, template: str) -> re.Pattern:
        template = _norm_text(template)  # normalise once: fragments keep their boundary spaces
        parts, pos, seen = [], 0, set()
        for m in re.finditer(r"\{(\w+)\}", template):
            parts.append(re.escape(template[pos:m.start()]))
            name = m.group(1)
            values = self.slots.get(name)
            body = "|".join(re.escape(_norm_text(v)) for v in sorted(values, key=len, reverse=True)) if values else ".+?"
            parts.append(f"(?P={name})" if name in seen else f"(?P<{name}>{body})")
            seen.add(name)
            pos = m.end()
        parts.append(re.escape(template[pos:]))
        return re.compile("^" + "".join(parts) + "$")

    def match(self, instruction: str) -> str:
        return self.parse(instruction)[0]

    def parse(self, instruction: str) -> tuple[str, dict]:
        """-> (template id, slot values), e.g. ('T0', {'colour': 'red', 'side': 'left'})."""
        text = _norm_text(instruction)
        hits = [(tid, m) for tid, rx in self._regex.items() if (m := rx.match(text))]
        if len(hits) != 1:
            raise EpisodeError(f"instruction {instruction!r} matches {len(hits)} templates {[h[0] for h in hits]}")
        return hits[0][0], hits[0][1].groupdict()

    def counterfactual(self, instruction: str, slot: str) -> str:
        """Same template, ``slot`` replaced by the next declared value (e.g. left -> right): a language-use control."""
        tid, values = self.parse(instruction)
        options = [_norm_text(v) for v in self.slots.get(slot, [])]
        if slot not in values or len(options) < 2:
            raise ValueError(f"template {tid} has no swappable slot {slot!r} (declared values {options})")
        values[slot] = options[(options.index(values[slot]) + 1) % len(options)]
        return _norm_text(self.templates[tid]).format(**values)


def _norm_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


# ---------------------------------------------------------------------------------------------------------------
# Splits: held-out layouts (by seed) and held-out instruction templates
# ---------------------------------------------------------------------------------------------------------------
SPLITS = ("train", "val_layout", "val_template")


@dataclass
class SplitSpec:
    """Assign an episode to train / val_layout (held-out seed => held-out layout) / val_template.

    A seed is a validation layout if ``seed % val_seed_mod == val_seed_rem`` or it is listed in ``val_seeds``.
    An episode whose instruction template is held out goes to ``val_template`` whatever its seed: held-out
    templates are never seen in training.
    """

    val_seed_mod: int = 10
    val_seed_rem: int = 9
    val_seeds: frozenset = frozenset()
    held_out_templates: frozenset = frozenset()

    def assign(self, seed: int, template_id: str | None) -> str:
        if template_id is not None and template_id in self.held_out_templates:
            return "val_template"
        if (self.val_seed_mod > 0 and seed % self.val_seed_mod == self.val_seed_rem) or seed in self.val_seeds:
            return "val_layout"
        return "train"

    def to_json(self) -> dict:
        return {"val_seed_mod": self.val_seed_mod, "val_seed_rem": self.val_seed_rem,
                "val_seeds": sorted(self.val_seeds), "held_out_templates": sorted(self.held_out_templates)}


def parse_seed_list(text: str | None) -> frozenset:
    """'3090-3099,3150' -> frozenset of ints."""
    out = set()
    for part in (text or "").split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            lo, hi = (int(v) for v in part.split("-", 1))
            if hi < lo:
                raise ValueError(f"bad seed range {part}")
            out.update(range(lo, hi + 1))
        else:
            out.add(int(part))
    return frozenset(out)


# ---------------------------------------------------------------------------------------------------------------
# Uncompressed memmap cache (content addressed)
# ---------------------------------------------------------------------------------------------------------------
CACHE_ARRAYS = ("image", "state", "action", "stage")
CACHE_VERSION = 1


def _atomic_json(path: Path, obj) -> None:
    tmp = path.with_suffix(path.suffix + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=2, allow_nan=False) + "\n")
    os.replace(tmp, path)


@dataclass
class CachedEpisode:
    directory: Path
    meta: dict
    _arrays: dict | None = field(default=None, repr=False, compare=False)

    @property
    def length(self) -> int:
        return int(self.meta["frames"])

    def arrays(self) -> dict:
        # Opened lazily so forked/spawned DataLoader workers each get their own memmap handles.
        if self._arrays is None:
            self._arrays = {k: np.load(self.directory / f"{k}.npy", mmap_mode="r") for k in CACHE_ARRAYS}
        return self._arrays

    def __getstate__(self):
        return {"directory": self.directory, "meta": self.meta, "_arrays": None}


def cache_episode(entry: EpisodeEntry, cache_root: str | Path, *, strict: bool = True,
                  verify_sha256: bool = True) -> CachedEpisode:
    """Validate one npz and store its arrays as uncompressed .npy for memmap access.

    The cache key is the file's sha256 (verified against the manifest when present), so a changed source can
    never be served from a stale cache entry.
    """
    cache_root = Path(cache_root)
    digest = sha256_file(entry.path) if (verify_sha256 or entry.sha256 is None) else entry.sha256
    if entry.sha256 is not None and digest != entry.sha256:
        raise EpisodeError(f"{entry.path.name}: sha256 {digest[:12]} != manifest {entry.sha256[:12]}")
    directory = cache_root / f"v{CACHE_VERSION}-{digest[:32]}"
    ready = directory / "meta.json"
    if ready.is_file():
        meta = json.loads(ready.read_text())
        return CachedEpisode(directory, meta)
    ep = load_episode(entry.path, strict=strict)
    if entry.seed >= 0 and ep["seed"] != entry.seed:
        raise EpisodeError(f"{entry.path.name}: seed {ep['seed']} != manifest seed {entry.seed}")
    if entry.sha256 is not None and ep["success"] != entry.success:
        raise EpisodeError(f"{entry.path.name}: success {ep['success']} != manifest {entry.success}")
    directory.mkdir(parents=True, exist_ok=True)
    for key in CACHE_ARRAYS:
        tmp = directory / f"{key}.tmp{os.getpid()}.npy"
        np.save(tmp, np.ascontiguousarray(ep[key]), allow_pickle=False)
        os.replace(tmp, directory / f"{key}.npy")
    meta = {k: ep[k] for k in ("instruction", "seed", "success", "synthetic", "template_id", "instruction_id")}
    meta.update({"frames": int(ep["image"].shape[0]), "source": str(entry.path), "sha256": digest,
                 "cache_version": CACHE_VERSION, "stage_values": sorted(int(v) for v in np.unique(ep["stage"]))})
    _atomic_json(ready, meta)
    return CachedEpisode(directory, meta)


# ---------------------------------------------------------------------------------------------------------------
# Dataset assembly
# ---------------------------------------------------------------------------------------------------------------
@dataclass
class WaiterDataset:
    episodes: list[CachedEpisode]
    split_of: list[str]
    template_of: list[str | None]
    excluded: list[dict]
    split_spec: SplitSpec

    def indices(self, split: str) -> list[int]:
        return [i for i, s in enumerate(self.split_of) if s == split]

    def summary(self) -> dict:
        out = {"split_spec": self.split_spec.to_json(), "excluded": self.excluded, "splits": {}}
        for split in SPLITS:
            idx = self.indices(split)
            out["splits"][split] = {
                "episodes": len(idx),
                "frames": int(sum(self.episodes[i].length for i in idx)),
                "seeds": sorted(int(self.episodes[i].meta["seed"]) for i in idx),
                "templates": sorted({t for i in idx if (t := self.template_of[i]) is not None}),
                "sha256": [self.episodes[i].meta["sha256"] for i in idx],
            }
        return out

    def data_hash(self) -> str:
        """Hash of exactly what defines training/validation data (episode content + split assignment)."""
        rows = sorted((e.meta["sha256"], s) for e, s in zip(self.episodes, self.split_of))
        blob = json.dumps({"rows": rows, "split": self.split_spec.to_json()}, sort_keys=True)
        return hashlib.sha256(blob.encode()).hexdigest()


def build_dataset(root: str | Path, cache_root: str | Path, split_spec: SplitSpec, *,
                  templates: TemplateSet | None = None, include_failed: bool = False,
                  allow_dev_seeds: bool = False, strict: bool = True, require_manifest: bool = True,
                  limit: int | None = None, log=print) -> WaiterDataset:
    entries = read_manifest(root, require_manifest=require_manifest)
    if limit is not None:
        entries = entries[:limit]
    episodes, split_of, template_of, excluded = [], [], [], []
    seen_seeds: dict[int, str] = {}
    for entry in entries:
        cached = cache_episode(entry, cache_root, strict=strict)
        meta = cached.meta
        check_training_seed(meta["seed"], synthetic=meta["synthetic"], allow_dev_seeds=allow_dev_seeds)
        if not meta["success"] and not include_failed:
            excluded.append({"source": meta["source"], "seed": meta["seed"], "reason": "failed demonstration"})
            continue
        template = meta.get("template_id")
        if templates is not None:
            matched = templates.match(meta["instruction"])
            if template is not None and template != matched:
                raise EpisodeError(f"{meta['source']}: template_id {template} but instruction matches {matched}")
            template = matched
        split = split_spec.assign(meta["seed"], template)
        # One layout must never sit on both sides of the split (a seed may have several instructions).
        prev = seen_seeds.get(meta["seed"])
        if prev is not None and {prev, split} == {"train", "val_layout"}:
            raise EpisodeError(f"seed {meta['seed']} assigned to both train and val_layout")
        seen_seeds.setdefault(meta["seed"], split)
        episodes.append(cached)
        split_of.append(split)
        template_of.append(template)
        log(f"[data] {Path(meta['source']).name} seed={meta['seed']} T={cached.length} template={template} -> {split}")
    if not any(s == "train" for s in split_of):
        raise ValueError("no training episodes after filtering and splitting")
    return WaiterDataset(episodes, split_of, template_of, excluded, split_spec)


# ---------------------------------------------------------------------------------------------------------------
# Normalisation statistics
# ---------------------------------------------------------------------------------------------------------------
def compute_norm_stats(arrays: Iterable[np.ndarray], *, std_floor: float = 1e-2) -> dict:
    """Per-dimension statistics over all frames of the given (T, D) arrays.

    ``std`` is the value SmolVLA's MEAN_STD normaliser divides by: the population std floored at ``std_floor``.
    The floor matters for dimensions that are (near) constant in the demonstrations, e.g. projected gravity and
    base velocity with a fixed base: without it a 1e-6 std would map a tiny deviation at rollout to a huge
    normalised input. ``raw_std`` keeps the unfloored value for audit.
    """
    data = [np.asarray(a, dtype=np.float64) for a in arrays]
    if not data:
        raise ValueError("no arrays for statistics")
    dims = {a.shape[1] for a in data}
    if len(dims) != 1:
        raise ValueError(f"inconsistent dimensions {dims}")
    stacked = np.concatenate(data, axis=0)
    if stacked.shape[0] < 2:
        raise ValueError("need at least two frames for statistics")
    mean = stacked.mean(axis=0)
    raw_std = stacked.std(axis=0)
    std = np.maximum(raw_std, std_floor)
    return {
        "mean": mean.tolist(), "std": std.tolist(), "raw_std": raw_std.tolist(),
        "min": stacked.min(axis=0).tolist(), "max": stacked.max(axis=0).tolist(),
        "q01": np.quantile(stacked, 0.01, axis=0).tolist(), "q99": np.quantile(stacked, 0.99, axis=0).tolist(),
        "count": int(stacked.shape[0]), "std_floor": float(std_floor),
        "floored_dims": [int(i) for i in np.nonzero(raw_std < std_floor)[0]],
    }


def dataset_norm_stats(dataset: WaiterDataset, *, std_floor: float = 1e-2) -> dict:
    """Statistics over TRAINING episodes only (validation never leaks into normalisation)."""
    train = [dataset.episodes[i] for i in dataset.indices("train")]
    return {
        STATE_KEY: compute_norm_stats((e.arrays()["state"] for e in train), std_floor=std_floor),
        ACTION_KEY: compute_norm_stats((e.arrays()["action"] for e in train), std_floor=std_floor),
    }


def normalize(x: np.ndarray, stats: dict) -> np.ndarray:
    return (np.asarray(x, np.float64) - np.asarray(stats["mean"])) / np.asarray(stats["std"])


def unnormalize(x: np.ndarray, stats: dict) -> np.ndarray:
    return np.asarray(x, np.float64) * np.asarray(stats["std"]) + np.asarray(stats["mean"])


# ---------------------------------------------------------------------------------------------------------------
# Action chunks and sampling
# ---------------------------------------------------------------------------------------------------------------
def action_chunk(actions: np.ndarray, t: int, chunk: int) -> tuple[np.ndarray, np.ndarray]:
    """Actions t..t+chunk-1; past the episode end the last action is repeated and flagged as padding.

    Returns (chunk, D) float32 actions and a (chunk,) bool ``is_pad`` mask (True = beyond the episode).
    """
    n = actions.shape[0]
    if not 0 <= t < n:
        raise IndexError(f"frame {t} outside episode of length {n}")
    idx = np.arange(t, t + chunk)
    is_pad = idx >= n
    out = np.asarray(actions[np.minimum(idx, n - 1)], dtype=np.float32)
    return out, is_pad


def batch_indices(step: int, seed: int, batch_size: int, lengths: Sequence[int], mode: str = "frame") -> list:
    """Training samples for one optimisation step as (episode_index, frame) pairs.

    A pure function of (seed, step): resuming at step k reproduces exactly the batches an uninterrupted run would
    have drawn, without saving sampler state, and DataLoader prefetching cannot desynchronise it.
    mode 'frame' = uniform over all frames; 'episode' = uniform episode, then uniform frame.
    """
    rng = np.random.default_rng([int(seed), int(step), 0x5A17])
    lengths = np.asarray(lengths, dtype=np.int64)
    if mode == "frame":
        offsets = np.concatenate([[0], np.cumsum(lengths)])
        flat = rng.integers(offsets[-1], size=batch_size)
        eps = np.searchsorted(offsets, flat, side="right") - 1
        return [(int(e), int(f - offsets[e])) for e, f in zip(eps, flat)]
    if mode == "episode":
        eps = rng.integers(len(lengths), size=batch_size)
        return [(int(e), int(rng.integers(lengths[e]))) for e in eps]
    raise ValueError(f"unknown sampling mode {mode}")


def validation_samples(lengths: Sequence[int], per_episode: int) -> list:
    """Evenly spaced frames per episode (first and last included): deterministic validation set."""
    out = []
    for e, n in enumerate(lengths):
        frames = np.unique(np.linspace(0, n - 1, min(per_episode, n)).round().astype(int))
        out.extend((e, int(f)) for f in frames)
    return out


def make_sample(episode: CachedEpisode, frame: int, chunk: int) -> dict:
    arr = episode.arrays()
    actions, is_pad = action_chunk(arr["action"], frame, chunk)
    return {
        "image": np.array(arr["image"][frame], copy=True),  # (224,224,3) uint8; copy: memmaps are read-only
        "state": np.array(arr["state"][frame], dtype=np.float32, copy=True),
        "action": actions,
        "action_is_pad": is_pad,
        "stage": int(arr["stage"][frame]),
        "task": episode.meta["instruction"],
    }


def make_chunk_dataset(episodes: Sequence[CachedEpisode], chunk: int):
    """torch Dataset indexed by (episode_index, frame) pairs; use with a batch_sampler of such pairs."""
    import torch

    class ChunkDataset(torch.utils.data.Dataset):
        def __init__(self):
            self.episodes = list(episodes)
            self.chunk = chunk

        def __len__(self):
            return sum(e.length for e in self.episodes)

        def __getitem__(self, key):
            e, f = key
            s = make_sample(self.episodes[e], f, self.chunk)
            return {
                "image": torch.from_numpy(s["image"]),
                "state": torch.from_numpy(s["state"]),
                "action": torch.from_numpy(s["action"]),
                "action_is_pad": torch.from_numpy(s["action_is_pad"]),
                "stage": s["stage"],
                "task": s["task"],
                "episode": e,
                "frame": f,
            }

    return ChunkDataset()
