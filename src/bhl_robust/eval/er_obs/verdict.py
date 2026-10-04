"""ER-OBS-1 verdicts: pure functions over the frames manifest and the recorded call outcomes.

ACCURACY (`accuracy_verdict`) and IN-LOOP-ELIGIBLE (`latency_verdict`) are separate verdicts. Rules verbatim in
`bhl_robust.eval.er_obs.RULE_TEXT`; clauses as applied in `CLAUSES_AS_APPLIED`. Verdict JSONs are written once
(`write_once` refuses an existing path) and the verdict lines are read back from the written JSON.

Failed calls (changed after the 2026-10-03 review, before any real call): on MAIN frames a failed call is scored as
WRONG (the negation of the label; failures can only hurt a PASS). On C1 frames, the control that must score LOW, a
failed call is scored as CORRECT (the label itself), so failures can only make C1 harder to hold; the C1 booleans
come from the record's `booleans` (parsed on their own, whatever cube_point holds; the point is never scored on C1),
and the C1 failure rate is reported.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import numpy as np

from bhl_robust.eval.er_obs import (BA_PASS_MIN, BOOLEANS, C1_BA_MAX, LABEL, LATENCY_CALLS, LATENCY_P95_MAX_S,
                                    MIN_CLASS_FRAMES, POINT_HIT_PASS_MIN, VARIANTS)


def balanced_accuracy(labels, preds, *, failed: str = "wrong") -> dict:
    """(TPR + TNR) / 2. A prediction of None is a failed call, scored as `failed`: "wrong" (the negation of the label;
    main frames) or "correct" (the label itself; C1 frames). None (undefined) when a class is empty."""
    if failed not in ("wrong", "correct"):
        raise ValueError(f"failed must be 'wrong' or 'correct', not {failed!r}")
    tp = fn = tn = fp = n_failed = 0
    for y, p in zip(labels, preds, strict=True):
        y = bool(y)
        if p is None:
            n_failed += 1
            p = (not y) if failed == "wrong" else y
        else:
            p = bool(p)
        if y:
            tp += p
            fn += not p
        else:
            tn += not p
            fp += p
    pos, neg = tp + fn, tn + fp
    tpr = tp / pos if pos else None
    tnr = tn / neg if neg else None
    ba = (tpr + tnr) / 2.0 if (pos and neg) else None
    return {"ba": ba, "tpr": tpr, "tnr": tnr, "tp": tp, "fn": fn, "tn": tn, "fp": fp, "n_pos": pos, "n_neg": neg,
            "n_failed": n_failed, "failed_scored_as": failed}


def point_pixel(point_yx, height: int, width: int) -> tuple[int, int]:
    """[y, x] normalized 0-1000 -> (row, col): floor(v * size / 1000), clipped to the last pixel."""
    y, x = float(point_yx[0]), float(point_yx[1])
    row = min(int(math.floor(y * height / 1000.0)), height - 1)
    col = min(int(math.floor(x * width / 1000.0)), width - 1)
    return max(row, 0), max(col, 0)


def point_hit(point_yx, mask) -> bool:
    """True iff the answer's point lands on a cube pixel of the segmentation mask (H x W bool)."""
    if point_yx is None:
        return False
    m = np.asarray(mask, dtype=bool)
    r, c = point_pixel(point_yx, m.shape[0], m.shape[1])
    return bool(m[r, c])


def p95_nearest_rank(latencies) -> float:
    """Nearest-rank 95th percentile: the ceil(0.95 n)-th smallest value (the 48th of 50). None = failed = +inf."""
    vals = sorted(math.inf if v is None else float(v) for v in latencies)
    if not vals:
        return math.inf
    return vals[math.ceil(0.95 * len(vals)) - 1]


def class_counts(frames) -> dict:
    return {b: {"true": sum(1 for f in frames if f["labels"][b]), "false": sum(1 for f in frames if not f["labels"][b])}
            for b in BOOLEANS}


def short_classes(counts: dict, minimum: int = MIN_CLASS_FRAMES) -> list[str]:
    return [f"{b}={v}" for b, c in counts.items() for v in ("true", "false") if c[v] < minimum]


def pred_main(rec, b):
    """Main frames: the predicted boolean of an ok outcome (full schema check, cube_point included), or None for a
    failed call (scored as wrong)."""
    if rec is None or rec.get("kind") != "ok":
        return None
    return bool(rec["answer"][b])


def pred_c1(rec, b):
    """C1 frames: the boolean parsed on its own (whatever cube_point holds), or None for a failed call (no four
    parsed booleans; scored as correct)."""
    if rec is None or not isinstance(rec.get("booleans"), dict):
        return None
    return bool(rec["booleans"][b])


def _failures(frames, outcomes, variant, pred) -> dict:
    """Failed calls of one variant over the frames with a recorded outcome (a missing outcome makes the run
    INCOMPLETE and is counted there)."""
    recs = [outcomes.get((f["id"], variant)) for f in frames]
    have = [r for r in recs if r is not None]
    failed = [r for r in have if pred(r, BOOLEANS[0]) is None]
    kinds = {}
    for r in failed:
        kinds[r.get("kind")] = kinds.get(r.get("kind"), 0) + 1
    return {"failed": len(failed), "frames_with_outcome": len(have),
            "rate": (len(failed) / len(have)) if have else None, "failed_kinds": kinds}


def accuracy_verdict(frames: list[dict], outcomes: dict, hits: dict, *, expected_frames: int) -> dict:
    """frames: the manifest's frame records (labels, cube_visible, c2, body-z fields); outcomes: (frame_id, variant)
    -> recorded call outcome; hits: frame_id -> point hit (bool) for cube-visible main frames with an ok answer.

    INCOMPLETE (precedence) iff the manifest does not hold `expected_frames` frames, any of the 8 label classes has
    fewer than 20 frames, or any frame lacks a recorded main or C1 outcome. Else PASS iff every main BA >= 0.90
    (failed calls scored wrong) AND the point hit rate >= 0.95 AND every C1 BA <= 0.60 (failed calls scored correct);
    else NEGATIVE."""
    ids = [f["id"] for f in frames]
    counts = class_counts(frames)
    short = short_classes(counts)
    missing = [f"{i}/{v}" for i in ids for v in VARIANTS if outcomes.get((i, v)) is None]
    main, c1, c1_answered = {}, {}, {}
    for b in BOOLEANS:
        y = [f["labels"][b] for f in frames]
        main[b] = balanced_accuracy(y, [pred_main(outcomes.get((i, "main")), b) for i in ids], failed="wrong")
        p1 = [pred_c1(outcomes.get((i, "c1")), b) for i in ids]
        c1[b] = balanced_accuracy(y, p1, failed="correct")
        ya = [yy for yy, pp in zip(y, p1) if pp is not None]
        c1_answered[b] = balanced_accuracy(ya, [pp for pp in p1 if pp is not None])
    main_fail = _failures(frames, outcomes, "main", pred_main)
    c1_fail = _failures(frames, outcomes, "c1", pred_c1)
    visible = [f["id"] for f in frames if f["cube_visible"]]
    n_hit = sum(1 for i in visible if hits.get(i) is True)
    hit_rate = (n_hit / len(visible)) if visible else None
    kinds = {v: {} for v in VARIANTS}
    for (i, v), rec in outcomes.items():
        if rec is not None:
            kinds[v][rec.get("kind")] = kinds[v].get(rec.get("kind"), 0) + 1
    # C2: the status-quo rules against the strict labels, same frames (reported, never gating)
    c2_frames = [f for f in frames if f.get("c2") is not None]
    c2 = {"lifted_centre_vs_lifted_clear": balanced_accuracy([f["labels"]["lifted_clear"] for f in c2_frames],
                                                             [f["c2"]["lifted_centre"] for f in c2_frames]),
          "placed_no_orientation_vs_seated_flat": balanced_accuracy(
              [f["labels"]["seated_flat"] for f in c2_frames], [f["c2"]["placed_no_orientation"] for f in c2_frames]),
          "frames": len(c2_frames)}
    body_z = [f["id"] for f in frames if f.get("body_z_disagrees")]
    checks = {
        "main_ba_all_ge_0.90": all(main[b]["ba"] is not None and main[b]["ba"] >= BA_PASS_MIN for b in BOOLEANS),
        "point_hit_rate_ge_0.95": hit_rate is not None and hit_rate >= POINT_HIT_PASS_MIN,
        "c1_ba_all_le_0.60": all(c1[b]["ba"] is not None and c1[b]["ba"] <= C1_BA_MAX for b in BOOLEANS),
    }
    incomplete_reasons = []
    if len(frames) != expected_frames:
        incomplete_reasons.append(f"{len(frames)} frames, {expected_frames} predeclared")
    if short:
        incomplete_reasons.append("classes with fewer than 20 frames: " + ", ".join(short))
    if missing:
        incomplete_reasons.append(f"{len(missing)} frame/variant(s) without a recorded call outcome")
    if incomplete_reasons:
        verdict = "INCOMPLETE"
    else:
        verdict = "PASS" if all(checks.values()) else "NEGATIVE"
    return {
        "verdict": verdict, "checks": checks, "incomplete_reasons": incomplete_reasons,
        "frames": len(frames), "expected_frames": expected_frames, "class_counts": counts,
        "main_balanced_accuracy": main, "c1_balanced_accuracy": c1,
        "scoring": {"main": "a failed call is scored WRONG (the negation of the label) on every boolean and the point",
                    "c1": "a failed call (no four parsed booleans) is scored CORRECT on every boolean; the C1 booleans "
                          "are parsed on their own, whatever cube_point holds (the point is never scored on C1)"},
        "main_failures": main_fail, "c1_failures": c1_fail,
        "c1_balanced_accuracy_answered_only": c1_answered,
        "c1_answered_only_note": "C1 BA over the C1 frames with four parsed booleans only: reported, never gating",
        "point": {"cube_visible_frames": len(visible), "hits": n_hit, "hit_rate": hit_rate},
        "outcome_kinds": kinds, "missing_outcomes": missing[:50], "n_missing_outcomes": len(missing),
        "c2_status_quo": c2,
        "body_z_disagreement": {"frames": len(body_z), "ids": body_z,
                                "note": "seated_flat (any face down) vs Stand4's strict body-z tilt"},
        "label": LABEL,
    }


def latency_verdict(records: list[dict], *, expected_calls: int = LATENCY_CALLS) -> dict:
    """IN-LOOP-ELIGIBLE iff nearest-rank p95 <= 1.0 s over the 50 calls (failed calls = +inf); INCOMPLETE if fewer
    than 50 calls have a recorded outcome."""
    done = [r for r in records if r is not None and r.get("kind") is not None]
    lat = [r.get("latency_s") if r.get("kind") == "ok" else None for r in done]
    p95 = p95_nearest_rank(lat)
    if len(done) < expected_calls:
        verdict = "INCOMPLETE"
    else:
        verdict = "IN-LOOP-ELIGIBLE" if p95 <= LATENCY_P95_MAX_S else "NOT-IN-LOOP-ELIGIBLE"
    ok = sorted(v for v in lat if v is not None)
    kinds = {}
    for r in done:
        kinds[r["kind"]] = kinds.get(r["kind"], 0) + 1
    return {"verdict": verdict, "calls": len(done), "expected_calls": expected_calls,
            "p95_nearest_rank_s": None if math.isinf(p95) else round(p95, 4), "p95_is_inf": math.isinf(p95),
            "failed_calls": len(done) - len(ok), "outcome_kinds": kinds,
            "ok_latency_s": {"min": ok[0] if ok else None, "median": ok[len(ok) // 2] if ok else None,
                             "max": ok[-1] if ok else None},
            "rule": "nearest-rank p95 (ceil(0.95 n)-th smallest) of 50 synchronous calls <= 1.0 s; failed = +inf",
            "label": LABEL}


def write_once(path, obj) -> Path:
    """Write a verdict JSON exactly once (refuses an existing path; atomic create)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    data = (json.dumps(obj, indent=1, allow_nan=False, ensure_ascii=False) + "\n").encode("utf-8")
    try:
        fd = os.open(p, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o644)
    except FileExistsError:
        raise FileExistsError(f"{p} exists; verdicts are written once and never overwritten") from None
    with os.fdopen(fd, "wb") as f:
        f.write(data)
    return p


def accuracy_line(v: dict) -> str:
    """One line from a written accuracy verdict JSON."""
    m, c = v["main_balanced_accuracy"], v["c1_balanced_accuracy"]

    def f(x):
        return "n/a" if x is None else f"{x:.3f}"
    head = v.get("verdict_label") or v["verdict"]
    return (f"ER-OBS-1 ACCURACY: {head} | main BA " + " ".join(f"{b}={f(m[b]['ba'])}" for b in BOOLEANS)
            + f" | point hit {f(v['point']['hit_rate'])} ({v['point']['hits']}/{v['point']['cube_visible_frames']})"
            + " | C1 BA " + " ".join(f"{b}={f(c[b]['ba'])}" for b in BOOLEANS)
            + f" (failed calls scored correct; C1 failure rate {f(v['c1_failures']['rate'])})"
            + f" | C2 lifted {f(v['c2_status_quo']['lifted_centre_vs_lifted_clear']['ba'])}"
            + f" placed {f(v['c2_status_quo']['placed_no_orientation_vs_seated_flat']['ba'])}"
            + f" | body-z disagreements {v['body_z_disagreement']['frames']}"
            + (f" | INCOMPLETE: {'; '.join(v['incomplete_reasons'])}" if v["incomplete_reasons"] else "")
            + f" | {v['label']}")


def latency_line(v: dict) -> str:
    head = v.get("verdict_label") or v["verdict"]
    p95 = "inf" if v["p95_is_inf"] else f"{v['p95_nearest_rank_s']:.3f} s"
    return (f"ER-OBS-1 LATENCY: {head} | p95 {p95} over {v['calls']}/{v['expected_calls']} calls "
            f"({v['failed_calls']} failed) | {v['label']}")
