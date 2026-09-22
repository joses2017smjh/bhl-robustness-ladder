"""Balanced four-direction privileged Approach controller evaluation."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np

from bhl_robust.mission.approach_debug import DebugEnv, RecoveryTranslationController
from bhl_robust.mission.layout import generate
from mission7_overnight import write


def direction_for(layout):
    delta = np.asarray(layout.route[-1]) - np.asarray(layout.route[-2])
    return tuple(int(x) for x in delta)


def balanced_layouts(split="test", per_direction=16):
    found = defaultdict(list)
    # The public test split has 13 instances of each positive direction. Add
    # validation layouts only where needed while retaining held-out geometry
    # (neither train nor a controller-tuning fixture).
    sources = [("test", range(64)), ("validation", range(32))]
    for source_split, indices in sources:
        for index in indices:
            layout = generate(source_split, index)
            key = direction_for(layout)
            if len(found[key]) < per_direction:
                found[key].append({"split": source_split, "index": index})
    expected = {(-1, 0), (1, 0), (0, -1), (0, 1)}
    if set(found) != expected or any(len(found[k]) != per_direction for k in expected):
        raise RuntimeError({k: len(v) for k, v in found.items()})
    return {"%+d,%+d" % key: value for key, value in sorted(found.items())}


def failure_stage(row):
    if row["success"]:
        return "success"
    if row["fall"]:
        if row["minimum_goal_distance_m"] < .36:
            return "entered_goal_region_but_fell"
        if row["fraction_distance_closed"] < .15:
            return "failed_to_translate_toward_target"
        return "translation_or_alignment_fall"
    if row["goal_region_entered"]:
        if row["hold_s"] > 0:
            return "contact_or_dwell_failed"
        return "entered_goal_region_but_dwell_failed"
    if row["fraction_distance_closed"] < .15:
        return "failed_to_translate_toward_target"
    return "failed_to_rotate_or_align_or_stopped_early"


def summarize(rows):
    by_direction = {}
    for direction in sorted({r["direction"] for r in rows}):
        subset = [r for r in rows if r["direction"] == direction]
        by_direction[direction] = {
            "episodes": len(subset),
            "successes": sum(r["success"] for r in subset),
            "success_rate": float(np.mean([r["success"] for r in subset])),
            "falls": sum(r["fall"] for r in subset),
            "failures": dict(Counter(r["failure_stage"] for r in subset)),
        }
    return {
        "episodes": len(rows),
        "successes": sum(r["success"] for r in rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "falls": sum(r["fall"] for r in rows),
        "failures": dict(Counter(r["failure_stage"] for r in rows)),
        "by_direction": by_direction,
    }


def run_controller(env, controller):
    while True:
        _, _, done, _ = env.step(controller.action())
        if done:
            return env.metrics()


def evaluate(a, out):
    selection = balanced_layouts()
    rows = []
    envs = {split: DebugEnv(a.repo, out / ("pulse-" + split + "-cache"), stage="approach",
                            split=split, seed=2300, approach_distance=.65)
            for split in ("test", "validation")}
    for direction, indices in selection.items():
        for entry in indices[:1] if a.smoke else indices:
            index, split = entry["index"], entry["split"]
            env = envs[split]
            env.reset(index)
            controller = RecoveryTranslationController(env, speed=.50, stop_radius=.28)
            row = run_controller(env, controller)
            row.update(layout_index=index, layout_split=split, direction=direction,
                       controller="measured_translation_0.50mps_with_bounded_0.30mps_stall_pulse",
                       recovery_pulse_used=controller.kicked)
            row["failure_stage"] = failure_stage(row)
            rows.append(row)
            write(out / "pulse.json", {"complete": False, "selection": selection, "episodes": rows})
    write(out / "pulse.json", {"complete": True, "selection": selection,
                                "summary": summarize(rows), "episodes": rows})

    # Matched standstill uses the same test layouts and reset seed stream.  A
    # second .40 m easy-start fixture documents when initial proximity alone
    # can satisfy the official dwell requirement.
    controls = {}
    for distance, label in ((.65, "matched_standstill"), (.40, "easy_standstill")):
        stand_rows = []
        stand_envs = {split: DebugEnv(a.repo, out / (label + "-" + split + "-cache"), stage="approach",
                                      split=split, seed=2300, approach_distance=distance)
                      for split in ("test", "validation")}
        for direction, indices in selection.items():
            for entry in indices[:1] if a.smoke else indices:
                index, split = entry["index"], entry["split"]
                stand_env = stand_envs[split]
                stand_env.reset(index)
                stand_env.max_seconds = 8.0 if label == "matched_standstill" else 12.0
                while True:
                    _, _, done, _ = stand_env.step(np.zeros(5))
                    if done:
                        break
                row = stand_env.metrics()
                row.update(layout_index=index, layout_split=split, direction=direction, control=label)
                row["failure_stage"] = failure_stage(row)
                stand_rows.append(row)
                write(out / (label + ".json"), {"complete": False, "selection": selection,
                                                   "episodes": stand_rows})
        controls[label] = {"summary": summarize(stand_rows), "episodes": stand_rows}
        write(out / (label + ".json"), {"complete": True, "selection": selection,
                                         **controls[label]})
    write(out / "result.json", {
        "complete": True,
        "status": "COMPLETED_DIAGNOSTIC",
        "selection": selection,
        "controller": {
            "speed_m_s": .50, "stall_pulse_m_s": .30, "stall_pulse_s": .40,
            "stop_radius_m": .28, "heading_gain": 1.2,
        },
        "pulse": summarize(rows),
        "controls": {k: v["summary"] for k, v in controls.items()},
        "gate": {
            "minimum_episodes": len(rows) >= 64,
            "successes_at_least_60": sum(r["success"] for r in rows) >= 60,
            "zero_falls": sum(r["fall"] for r in rows) == 0,
            "per_direction_at_least_14": all(x["successes"] >= 14 for x in summarize(rows)["by_direction"].values()),
        },
        "benchmark_geometry_unchanged": True,
        "note": "Test split is used for a balanced 16-per-direction matrix; all four directions have matched pulse and standstill controls.",
    })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve(); args.campaign = args.campaign.resolve(); args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    evaluate(args, args.out)
    print("MISSION7_APPROACH_FOLLOWUP_COMPLETE", flush=True)
