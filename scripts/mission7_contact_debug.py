"""Per-physics-step contact diagnostics for the retained Mission7 plate falls."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.mission.approach_debug import DebugEnv, physical_sample, wrap
from mission7_overnight import write


def _episode_trace(a):
    source = json.loads((a.campaign / "fullroute/legacy-doors.json").read_text())
    if not source.get("complete"):
        raise RuntimeError("legacy door route is not complete")
    return source


def _contact_summary(events):
    plates = [e for e in events if e["world_geom"].startswith("plate_")]
    nonplates = [e for e in events if not e["world_geom"].startswith("plate_")]
    return {
        "contact_events": len(events),
        "plate_contact_events": len(plates),
        "other_obstacle_contact_events": len(nonplates),
        "plate_geoms": sorted({e["world_geom"] for e in plates}),
        "other_obstacle_geoms": sorted({e["world_geom"] for e in nonplates}),
        "peak_plate_normal_force_N": max((e["normal_force_N"] for e in plates), default=0.0),
        "peak_plate_tangent_force_N": max((e["tangent_force_N"] for e in plates), default=0.0),
        "minimum_plate_distance_m": min((e["distance_m"] for e in plates), default=None),
        "first_plate_contact_s": min((e["time_s"] for e in plates), default=None),
        "last_plate_contact_s": max((e["time_s"] for e in plates), default=None),
        "peak_other_normal_force_N": max((e["normal_force_N"] for e in nonplates), default=0.0),
    }


def probe(a, out):
    source = _episode_trace(a)
    selected = [
        (index, episode) for index, episode in enumerate(source["episodes"])
        if episode.get("fall")
    ]
    if not selected:
        raise RuntimeError("no retained fall episodes in legacy route")
    if a.smoke:
        selected = selected[:1]

    env = DebugEnv(a.repo, out / "cache", stage="doors", split="validation", seed=1000)
    rows = []
    next_reset = 0
    for index, episode in selected:
        # Reset every earlier layout to preserve the original RNG stream and
        # therefore the exact initial pose and sensor seed.
        for reset_index in range(next_reset, index + 1):
            env.reset(reset_index)
        next_reset = index + 1
        runner = env.runner
        samples = []
        maximum_pose_difference = 0.
        for reference in episode["diagnostic_trace"]:
            env.state.open = [
                t is not None and runner.d.time + 1e-8 >= t
                for t in episode["activation_s"]
            ]
            env._gates()
            command = np.asarray(reference["command"], dtype=float)
            runner.contact_trace = []
            runner.step([env.controller.update(runner.observe(0, command))])
            sample = physical_sample(env, command, reference["phase"])
            sample["contact_events"] = list(runner.contact_trace)
            sample["plate_contacts"] = sorted({
                e["world_geom"] for e in runner.contact_trace
                if e["world_geom"].startswith("plate_")
            })
            samples.append(sample)
            maximum_pose_difference = max(
                maximum_pose_difference,
                float(np.linalg.norm(np.asarray(sample["xy"]) - np.asarray(reference["xy"]))),
                abs(sample["tilt"] - reference["tilt"]),
                abs(wrap(sample["yaw"] - reference["yaw"])),
            )
            if not np.isfinite(runner.d.qpos).all() or runner.d.warning.number.sum():
                raise FloatingPointError("contact probe physics warning")
            if not a.smoke and abs(sample["time_s"] - reference["time_s"]) >= 1e-7:
                raise AssertionError("probe replay time diverged")
        events = [e for sample in samples for e in sample["contact_events"]]
        summary = _contact_summary(events)
        fall_sample = next((s for s in samples if s["tilt"] >= .78), None)
        rows.append({
            "layout_index": index,
            "layout_seed": episode["layout"]["seed"],
            "original_elapsed_s": episode["elapsed_s"],
            "replay_elapsed_s": samples[-1]["time_s"],
            "original_failure": episode["failure"],
            "fall_time_s": None if fall_sample is None else fall_sample["time_s"],
            "fall_command": None if fall_sample is None else fall_sample["command"],
            "fall_phase": None if fall_sample is None else fall_sample["phase"],
            "fall_xy": None if fall_sample is None else fall_sample["xy"],
            "fall_tilt": None if fall_sample is None else fall_sample["tilt"],
            "maximum_pose_difference": maximum_pose_difference,
            "matched_original": maximum_pose_difference < 1e-6,
            "contact_summary": summary,
            "samples": samples,
        })
        write(out / "episodes.json", {"complete": False, "episodes": rows})

    all_events = [
        event for row in rows for sample in row["samples"] for event in sample["contact_events"]
    ]
    plate_events = [e for e in all_events if e["world_geom"].startswith("plate_")]
    report = {
        "complete": True,
        "status": "COMPLETED_DIAGNOSTIC",
        "episodes": len(rows),
        "falls_replayed": sum(r["fall_time_s"] is not None for r in rows),
        "unchanged_replay_matches": all(r["matched_original"] for r in rows),
        "selected_layouts": [r["layout_index"] for r in rows],
        "aggregate": {
            "contact_events": len(all_events),
            "plate_contact_events": len(plate_events),
            "peak_plate_normal_force_N": max((e["normal_force_N"] for e in plate_events), default=0.0),
            "peak_plate_tangent_force_N": max((e["tangent_force_N"] for e in plate_events), default=0.0),
            "minimum_plate_distance_m": min((e["distance_m"] for e in plate_events), default=None),
            "plate_geom_counts": dict(Counter(e["world_geom"] for e in plate_events)),
        },
        "episodes_detail": [
            {k: v for k, v in row.items() if k != "samples"} for row in rows
        ],
        "interpretation": (
            "Open-loop commands and recorded gate-opening times replayed on unchanged geometry. "
            "Contact forces and distances are descriptive evidence; the subsequent controller "
            "mitigation must pass fresh route episodes before any causal claim is made."
        ),
    }
    if not report["unchanged_replay_matches"]:
        report["status"] = "INVALID_REPLAY"
        report["interpretation_allowed"] = False
        report["reason"] = "The detailed instrumentation did not reproduce the recorded unchanged trajectory."
    else:
        report["interpretation_allowed"] = True
    write(out / "episodes.json", {"complete": True, "episodes": rows})
    write(out / "result.json", report)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.campaign = args.campaign.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    probe(args, args.out)
    print("MISSION7_CONTACT_PROBE_COMPLETE", flush=True)
