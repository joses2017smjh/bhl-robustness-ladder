"""Frozen, episode-level health screening calibrated against measured outcomes.

This is an offline association experiment, not an online safety controller or
calibrated probability. An episode's tracking fraction is available only at its
end. The second reset seed is never read by fit().
"""
from __future__ import annotations

import hashlib
import json
import math

CALIBRATION_SEED = 730000
EVALUATION_SEED = 730001
THRESHOLDS = tuple(x / 20 for x in range(21)) + (1.01,)
FEATURE = "tracking_fraction_unique_estimator_updates"
SCHEMA = "bhl-outcome-health-v1"


def protocol():
    return {"schema": SCHEMA + "-protocol", "calibration_seed": CALIBRATION_SEED,
            "evaluation_seed": EVALUATION_SEED, "feature": FEATURE,
            "adverse_outcome": "not success (no clean full-horizon goal)",
            "thresholds": list(THRESHOLDS), "accept_rule": "finite tracking fraction >= threshold",
            "selection": "maximum balanced accuracy, then fewer false accepts, then more accepted episodes, then higher threshold",
            "minimum_each_outcome": 2, "fit_separately_by": "native estimator/controller arm",
            "baseline_threshold": .9,
            "scope": "offline episode-end screen; held-out reset seed in reused route families; no probability or closed-loop safety claim"}


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, allow_nan=False,
                                     separators=(",", ":")).encode()).hexdigest()


def validate_rows(rows, seed):
    from .navigation_sensor_controls import ARMS, CONDITIONS
    if not isinstance(rows, list) or not rows:
        raise ValueError("nonempty complete cohort required")
    ids = set()
    for row in rows:
        identity = tuple(row.get(k) for k in ("method", "case", "condition", "seed"))
        if identity in ids or row.get("seed") != seed or row.get("method") not in ARMS:
            raise ValueError("duplicate, wrong-seed or unknown-arm episode")
        if row.get("case") not in ("straight", "occluders") or row.get("condition") not in CONDITIONS:
            raise ValueError("unknown route or condition")
        if type(row.get("success")) is not bool or type(row.get("full_horizon")) is not bool:
            raise ValueError("explicit measured outcome and horizon required")
        value = row.get(FEATURE)
        if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float))
                                  or not math.isfinite(value) or not 0 <= value <= 1):
            raise ValueError("invalid measured tracking fraction")
        if row["success"] and (not row["full_horizon"] or any(row.get(k) is not False for k in ("fell", "collision", "nonfinite"))):
            raise ValueError("clean outcome inconsistent with safety/horizon evidence")
        ids.add(identity)
    for arm in {r["method"] for r in rows}:
        expected = {(arm, case, condition, seed) for case in ("straight", "occluders") for condition in CONDITIONS}
        if {i for i in ids if i[0] == arm} != expected:
            raise ValueError("every included arm requires both routes and all conditions")
    return rows


def score(rows, threshold):
    accepts = [r.get(FEATURE) is not None and r[FEATURE] >= threshold for r in rows]
    good = [r["success"] for r in rows]
    tp = sum(a and g for a, g in zip(accepts, good))
    fp = sum(a and not g for a, g in zip(accepts, good))
    tn = sum(not a and not g for a, g in zip(accepts, good))
    fn = sum(not a and g for a, g in zip(accepts, good))
    return {"episodes": len(rows), "accepted_clean": tp, "accepted_adverse": fp,
            "rejected_adverse": tn, "rejected_clean": fn, "accepted_episodes": tp + fp,
            "adverse_acceptance_rate": fp / (fp + tn) if fp + tn else None,
            "clean_acceptance_rate": tp / (tp + fn) if tp + fn else None,
            "observed_adverse_fraction_among_accepted": fp / (tp + fp) if tp + fp else None,
            "balanced_accuracy": .5 * (tp / (tp + fn) + tn / (tn + fp)) if tp + fn and tn + fp else None}


def fit(rows):
    validate_rows(rows, CALIBRATION_SEED)
    models = {}
    for arm in sorted({r["method"] for r in rows}):
        group = [r for r in rows if r["method"] == arm]
        positive = sum(r["success"] for r in group)
        if min(positive, len(group) - positive) < 2:
            models[arm] = {"status": "INSUFFICIENT_OUTCOME_VARIATION", "threshold": None,
                           "clean": positive, "adverse": len(group) - positive}
            continue
        candidates = [(threshold, score(group, threshold)) for threshold in THRESHOLDS]
        threshold, selected = max(candidates, key=lambda p: (p[1]["balanced_accuracy"],
            -p[1]["accepted_adverse"], p[1]["accepted_episodes"], p[0]))
        models[arm] = {"status": "FIT_ON_CALIBRATION_ONLY", "threshold": threshold,
                       "calibration": selected, "candidate_scores": candidates}
    result = {"schema": SCHEMA + "-model", "protocol": protocol(), "calibration_sha256": digest(rows),
              "calibration_episodes": len(rows), "models": models}
    result["model_sha256"] = digest(result)
    return result


def evaluate(model, rows):
    validate_rows(rows, EVALUATION_SEED)
    unsigned = {k: v for k, v in model.items() if k != "model_sha256"}
    if model.get("model_sha256") != digest(unsigned) or model.get("protocol") != protocol():
        raise ValueError("frozen calibration model/protocol checksum mismatch")
    if set(model["models"]) != {r["method"] for r in rows}:
        raise ValueError("same declared eligible arms required across split")
    result = {"schema": SCHEMA + "-evaluation", "model_sha256": model["model_sha256"],
              "evaluation_sha256": digest(rows), "protocol": protocol(), "arms": {}}
    for arm, spec in model["models"].items():
        group = [r for r in rows if r["method"] == arm]
        result["arms"][arm] = {"fit_status": spec["status"], "threshold": spec["threshold"],
            "frozen_gate": score(group, spec["threshold"]) if spec["threshold"] is not None else None,
            "engineering_baseline_0_9": score(group, .9), "accept_all": score(group, 0)}
    result["uncertainty_limit"] = "One held-out reset seed and two reused route families; dependent episodes do not support population confidence intervals."
    return result
