"""Shared DSRL pieces: seed-range guards, SAC observations, noise arms, episode rollouts, paired stats."""
from __future__ import annotations

import math
import zlib

import numpy as np
import torch

# Frozen seed ranges (docs/WAITER_INTERFACES.md). Nothing outside its range, ever.
SEED_RANGES = {"dev": (1000, 1999), "p2_gate": (2000, 2099), "p2_demo": (3000, 3999),
               "p3_eval": (4000, 4099), "p4_train": (5000, 5999), "p4_eval": (6000, 6099),
               "p3b_eval": (7000, 7099), "p4b_eval": (8000, 8099)}
SCORED_P4_RANGES = ("p4_eval", "p4b_eval")     # what eval_dsrl.py --scored-phase4-eval may run, one range per run


def parse_seeds(text: str) -> list[int]:
    out = []
    for part in str(text).split(","):
        part = part.strip()
        if not part:
            continue
        if "-" in part:
            a, b = part.split("-", 1)
            if int(b) < int(a):
                raise ValueError(f"empty seed range {part!r}")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return out


def range_of(seed: int) -> str | None:
    for name, (lo, hi) in SEED_RANGES.items():
        if lo <= seed <= hi:
            return name
    return None


def check_seeds(seeds, allowed: set[str], what: str):
    bad = sorted({(s, range_of(s)) for s in seeds if range_of(s) not in allowed}, key=lambda x: x[0])
    if bad:
        raise SystemExit(f"refusing {what} seeds outside {sorted(allowed)}: {bad[:5]}{' ...' if len(bad) > 5 else ''}")


# ------------------------------------------------------------------------------------------ observations
def _instr_hash(text: str, dim: int = 16) -> np.ndarray:
    v = np.zeros(dim, np.float32)
    for w in text.lower().replace(",", " ").split():
        v[zlib.crc32(w.encode()) % dim] += 1.0
    return v / max(1.0, float(np.linalg.norm(v)))


def pool4(img: np.ndarray) -> np.ndarray:
    """224x224x3 uint8 -> 56x56x3 uint8 (4x4 mean)."""
    return img.reshape(56, 4, 56, 4, 3).mean((1, 3)).astype(np.uint8)


class ObsBuilder:
    """Builds the SAC observation for a batch. ``vlm``: the frozen policy's prefix features (no grad).
    ``cnn``: vec = [state (30), instruction hash (16)], img = 56x56 pooled head image."""

    def __init__(self, mode: str, feature_dim: int):
        if mode not in ("vlm", "cnn"):
            raise ValueError(mode)
        self.mode = mode
        self.vec_dim = feature_dim if mode == "vlm" else 30 + 16
        self.img_hw = 0 if mode == "vlm" else 56

    def __call__(self, obs_list, ctx_features: torch.Tensor | None):
        if self.mode == "vlm":
            return {"vec": ctx_features.float().cpu().numpy()}
        vec = np.stack([np.concatenate([np.asarray(o["state"], np.float32), _instr_hash(o["instruction"])])
                        for o in obs_list])
        img = np.stack([pool4(np.asarray(o["image"], np.uint8)) for o in obs_list])
        return {"vec": vec, "img": img}

    def zeros(self):
        z = {"vec": np.zeros(self.vec_dim, np.float32)}
        if self.img_hw:
            z["img"] = np.zeros((self.img_hw, self.img_hw, 3), np.uint8)
        return z


def take(obs_batch: dict, j: int) -> dict:
    return {k: v[j] for k, v in obs_batch.items()}


# ------------------------------------------------------------------------------------------ noise arms
def episode_noise(noise_seed: int, seed: int, decision: int, shape) -> torch.Tensor:
    """Reproducible i.i.d. N(0, I) noise per (run noise seed, layout seed, decision index): paired arms."""
    g = torch.Generator().manual_seed((int(noise_seed) * 1_000_003 + int(seed)) * 4096 + int(decision))
    return torch.randn(*shape, generator=g)


ARMS = ("base", "zero", "latent_random", "dsrl", "dsrl_stoch")


def choose_noise(arm, idxs, seeds, decisions, sac_obs, policy, noise_map, b, sac=None, noise_seed=0, rng=None):
    """Returns (noise (B, chunk, 32), latent u (B, d) or None)."""
    T, D = policy.noise_shape
    B = len(idxs)
    if arm == "base":
        return torch.stack([episode_noise(noise_seed, seeds[i], decisions[i], (T, D)) for i in idxs]), None
    if arm == "zero":
        return torch.zeros(B, T, D), None
    if arm == "latent_random":   # DSRL warm-up distribution: N(0, I) latent clipped to the box, normalised
        g = rng if rng is not None else np.random.default_rng(noise_seed)
        u = np.clip(g.standard_normal((B, noise_map.latent_dim)), -b, b) / b
    elif arm in ("dsrl", "dsrl_stoch"):
        u = sac.act(sac_obs, deterministic=(arm == "dsrl"))
    else:
        raise ValueError(arm)
    u = np.asarray(u, np.float32)
    return noise_map(torch.as_tensor(u * b)), u


# ------------------------------------------------------------------------------------------ evaluation
def run_episodes(venv, policy, seeds, arm, H, noise_map, b, obs_builder=None, sac=None, noise_seed=0,
                 instruction_ids=None, device="cpu"):
    """One episode per seed with ``arm``; the venv's n slots work through the seed list."""
    results, queue = {}, list(enumerate(seeds))
    slot_seed, slot_k, slot_dec, slot_ep = {}, {}, {}, {}
    obs = {}
    rng = np.random.default_rng(noise_seed)          # latent_random: one stream for the run, a new latent per decision

    def start(slots):
        todo = {}
        for i in slots:
            if not queue:
                continue
            k, s = queue.pop(0)
            slot_seed[i], slot_k[i], slot_dec[i] = s, k, 0
            slot_ep[i] = {"seed": int(s), "arm": arm, "success": False, "fell": False, "steps": 0,
                          "decisions": 0, "stage_max": 0, "t": 0.0}
            todo[i] = (s, None if instruction_ids is None else instruction_ids[k])
        obs.update(venv.reset(todo) if todo else {})
        return list(todo)

    active = start(range(venv.n))
    while active:
        ol = [obs[i] for i in active]
        ctx = policy.encode(ol)
        sac_obs = obs_builder(ol, ctx.features) if (sac is not None and obs_builder is not None) else None
        noise, _ = choose_noise(arm, list(range(len(active))), [slot_seed[i] for i in active],
                                [slot_dec[i] for i in active], sac_obs, policy, noise_map, b, sac, noise_seed, rng)
        chunks = policy.denoise(ctx, noise.to(device))[:, :H].numpy()
        venv.step_async({i: chunks[j] for j, i in enumerate(active)})
        res = venv.step_wait()
        finished = []
        for i in active:
            r, ep = res[i], slot_ep[i]
            ep["steps"] += r["n"]
            ep["decisions"] += 1
            ep["stage_max"] = max(ep["stage_max"], r["stage_max"])
            ep["success"] |= r["success"]
            ep["fell"] |= r["fell"]
            ep["t"] = r["t"]
            slot_dec[i] += 1
            obs[i] = r["obs"]
            if r["done"]:
                results[slot_k[i]] = ep
                finished.append(i)
        restarted = start(finished)
        active = [i for i in active if i not in finished] + restarted
    return [results[k] for k in sorted(results)]


# ------------------------------------------------------------------------------------------ statistics
def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (c - h, c + h)


def mcnemar_exact(b, c):
    """b = #(treatment success, control fail), c = #(treatment fail, control success).
    Returns (one-sided p for treatment > control, two-sided p)."""
    n = b + c
    if n == 0:
        return 1.0, 1.0
    tail = sum(math.comb(n, i) for i in range(b, n + 1)) / 2 ** n
    two = min(1.0, 2 * min(tail, sum(math.comb(n, i) for i in range(0, b + 1)) / 2 ** n))
    return tail, two


def paired_summary(treat: list[bool], ctrl: list[bool], n_boot=10000, seed=0):
    t, c = np.asarray(treat, bool), np.asarray(ctrl, bool)
    b, cc = int((t & ~c).sum()), int((~t & c).sum())
    one, two = mcnemar_exact(b, cc)
    rng = np.random.default_rng(seed)
    d = t.astype(float) - c.astype(float)
    boots = [d[rng.integers(0, len(d), len(d))].mean() for _ in range(n_boot)] if len(d) else [float("nan")]
    return {"n": int(len(d)), "treat_rate": float(t.mean()) if len(t) else None,
            "ctrl_rate": float(c.mean()) if len(c) else None, "diff": float(d.mean()) if len(d) else None,
            "diff_ci95_bootstrap": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))],
            "both": int((t & c).sum()), "treat_only": b, "ctrl_only": cc, "neither": int((~t & ~c).sum()),
            "mcnemar_p_one_sided": one, "mcnemar_p_two_sided": two}


def arm_summary(eps):
    n = len(eps)
    k = sum(e["success"] for e in eps)
    return {"n": n, "success": k, "success_rate": k / n if n else None, "wilson95": list(wilson(k, n)),
            "fell": sum(e["fell"] for e in eps),
            "mean_steps": float(np.mean([e["steps"] for e in eps])) if n else None,
            "mean_stage_max": float(np.mean([e["stage_max"] for e in eps])) if n else None}
