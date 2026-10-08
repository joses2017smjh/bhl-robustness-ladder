"""Episode-level capture audit and split for the separate SO-101 folding track.

No Isaac, torch, or pickle dependency. Failed replay states do not make valid
supervision for their original demonstration's future action chunk.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def capture_manifest(pattern: str) -> list[dict]:
    import glob

    rows = []
    for name in sorted(glob.glob(pattern)):
        path = Path(name).resolve()
        with np.load(path, allow_pickle=False) as data:
            garment = str(data["garment"].item())
            episode = int(data["episode"].item())
            success = bool(data["success"].item())
            state, action = data["state"], data["action"]
            if not re.fullmatch(r"(?:Top|Pant)_(?:Short|Long)_Seen_\d+", garment):
                raise ValueError(f"training capture is not a recognized Seen garment: {garment}")
            if state.ndim != 2 or state.shape[1] != 12:
                raise ValueError(f"invalid SO-101 state shape in {path}: {state.shape}")
            if action.shape != (len(state), 50, 12):
                raise ValueError(f"capture must contain full 50-step chunks: {path}: {action.shape}")
            if not np.isfinite(state).all() or not np.isfinite(action).all():
                raise ValueError(f"nonfinite capture: {path}")
        rows.append({"path": str(path), "garment": garment, "episode": episode,
                     "success": success, "frames": len(state),
                     "bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns})
    if not rows:
        raise ValueError(f"no captures: {pattern}")
    keys = [(r["garment"], r["episode"]) for r in rows]
    if len(set(keys)) != len(keys):
        raise ValueError("duplicate garment/episode captures would bias training")
    return rows


def split_captures(rows: list[dict], holdout_ids: tuple[int, ...] = (8, 9)) -> dict:
    """Hold out entire Seen garment identities, retaining official Unseen for evaluation."""
    if not holdout_ids or any(i < 0 or i > 9 for i in holdout_ids):
        raise ValueError("holdout ids must be nonempty and within Seen_0..9")
    train, val, excluded = [], [], []
    for row in rows:
        if not row["success"]:
            excluded.append(row)
        elif int(row["garment"].rsplit("_", 1)[1]) in holdout_ids:
            val.append(row)
        else:
            train.append(row)
    if not train or not val:
        raise ValueError("need successful replays in both train and held-out garments")
    assert not ({r["garment"] for r in train} & {r["garment"] for r in val})
    return {"train": train, "validation": val, "excluded_failed_replays": excluded,
            "holdout_ids": list(holdout_ids), "split_unit": "garment_identity",
            "note": "Success-filtered replay BC, not DAgger; residual replay drift remains possible."}


def stratified_success_split(rows: list[dict]) -> dict:
    """Reserve the highest numbered successful garment in each of four classes.

    This uses success labels to ensure usable supervised validation, so it does
    not estimate population success rates. Official full-episode evaluation does.
    """
    categories = ("Top_Short", "Top_Long", "Pant_Short", "Pant_Long")
    heldout = []
    for category in categories:
        garments = {r["garment"] for r in rows if r["success"]
                    and r["garment"].startswith(category + "_Seen_")}
        if len(garments) < 2:
            raise ValueError(f"{category}: need successful replays on >=2 distinct garments "
                             "for nonempty class training and identity-held-out validation")
        heldout.append(max(garments, key=lambda value: int(value.rsplit("_", 1)[1])))
    train = [r for r in rows if r["success"] and r["garment"] not in heldout]
    validation = [r for r in rows if r["success"] and r["garment"] in heldout]
    return {"train": train, "validation": validation,
            "excluded_failed_replays": [r for r in rows if not r["success"]],
            "heldout_garments": heldout, "split_unit": "garment_identity",
            "validation_mode": "stratified-success",
            "note": "Garment identities held out from this adaptation, not base pretraining. "
                    "Success-filtered supervision is not an unbiased success-rate estimate."}
