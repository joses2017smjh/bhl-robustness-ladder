"""Contact-brake pressure-plate replay and paired route evaluation."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path

import numpy as np
import mujoco

from bhl_robust.mission.approach_debug import DebugEnv, PlateSafeRouteController, physical_sample
from bhl_robust.mission.approach_debug import command_action
from mission7_overnight import write


def selected_failures(campaign):
    source = json.loads((campaign / "fullroute/legacy-doors.json").read_text())
    if not source.get("complete"):
        raise RuntimeError("legacy route input incomplete")
    return source, [(i, e) for i, e in enumerate(source["episodes"]) if e.get("fall")]


def replay(a, out):
    source, selected = selected_failures(a.campaign)
    if a.smoke:
        selected = selected[:1]
    env = DebugEnv(a.repo, out / "replay-cache", stage="doors", split="validation", seed=1000)
    rows = []
    next_reset = 0
    for index, episode in selected:
        for reset_index in range(next_reset, index + 1):
            env.reset(reset_index)
        next_reset = index + 1
        runner = env.runner
        brake_until = 0.
        previous_open = tuple(env.state.open)
        samples = []
        for reference in episode["diagnostic_trace"]:
            env.state.open = [
                t is not None and runner.d.time + 1e-8 >= t
                for t in episode["activation_s"]
            ]
            env._gates()
            now = float(runner.d.time)
            changed = tuple(env.state.open) != previous_open
            if changed:
                brake_until = max(brake_until, now + 1.0)
            previous_open = tuple(env.state.open)
            recorded = np.asarray(reference["command"], dtype=float)
            # Begin braking before the first contact impulse.  This replay
            # intervention tests whether the measured penetration/impulse is
            # avoidable by timing alone; route evaluation below still has to
            # acquire the plate physically.
            nearest_plate = min(
                (np.linalg.norm(runner.d.xpos[env.slot.body_id, :2] -
                                 env.layout.plate(door, env.layout.correct_sides[door]))
                 for door in range(2) if not env.state.open[door]),
                default=float("inf"),
            )
            if nearest_plate < .45:
                brake_until = max(brake_until, now + .80)
            effective = np.zeros(3) if now < brake_until else recorded
            runner.contact_trace = []
            runner.step([env.controller.update(runner.observe(0, effective))])
            events = list(runner.contact_trace)
            plate_events = [e for e in events if e["world_geom"].startswith("plate_")]
            if plate_events:
                brake_until = max(brake_until, float(runner.d.time) + 1.0)
            sample = physical_sample(env, effective, reference["phase"])
            sample.update(recorded_command=recorded.tolist(),
                          effective_command=effective.tolist(),
                          plate_event_count=len(plate_events),
                          peak_plate_normal_force_N=max((e["normal_force_N"] for e in plate_events), default=0.),
                          peak_plate_tangent_force_N=max((e["tangent_force_N"] for e in plate_events), default=0.),
                          minimum_plate_distance_m=min((e["distance_m"] for e in plate_events), default=None),
                          brake_until_s=brake_until)
            samples.append(sample)
            if not np.isfinite(runner.d.qpos).all() or runner.d.warning.number.sum():
                raise FloatingPointError("plate-safe replay physics warning")
        falls = [s for s in samples if s["tilt"] >= .78]
        rows.append({
            "layout_index": index,
            "layout_seed": episode["layout"]["seed"],
            "original_elapsed_s": episode["elapsed_s"],
            "replay_elapsed_s": samples[-1]["time_s"],
            "fall": bool(falls),
            "first_fall_s": falls[0]["time_s"] if falls else None,
            "maximum_tilt": max(s["tilt"] for s in samples),
            "plate_contact_samples": sum(s["plate_event_count"] > 0 for s in samples),
            "peak_plate_normal_force_N": max(s["peak_plate_normal_force_N"] for s in samples),
            "peak_plate_tangent_force_N": max(s["peak_plate_tangent_force_N"] for s in samples),
            "minimum_plate_distance_m": min((s["minimum_plate_distance_m"] for s in samples
                                              if s["minimum_plate_distance_m"] is not None), default=None),
            "samples": samples,
        })
        write(out / "replay.json", {"complete": False, "episodes": rows})
    write(out / "replay.json", {"complete": True, "episodes": rows,
                                 "summary": {"episodes": len(rows), "falls": sum(r["fall"] for r in rows),
                                             "no_fall": sum(not r["fall"] for r in rows)}})


def summarize(rows):
    return {
        "episodes": len(rows),
        "successes": sum(r["success"] for r in rows),
        "success_rate": float(np.mean([r["success"] for r in rows])),
        "falls": sum(r["fall"] for r in rows),
        "failures": dict(Counter(r["failure"] or "success" for r in rows)),
    }


def routes(a, out):
    rows = []
    summaries = {}
    for stage in ("doors", "transport"):
        env = DebugEnv(a.repo, out / (stage + "-cache"), stage=stage, split="validation", seed=1000)
        stage_rows = []
        for index in range(1 if a.smoke else 16):
            env.reset(index)
            controller = PlateSafeRouteController(env, contact_hold_s=1.0)
            while True:
                action = controller.action()
                _, _, done, _ = env.step(action)
                if done:
                    break
            row = env.metrics()
            row.update(layout_index=index, controller="plate_safe_contact_brake_1.0s",
                       plate_contacts_seen=sorted(set(controller.plate_contacts_seen)))
            stage_rows.append(row); rows.append({"stage": stage, **row})
            write(out / (stage + ".json"), {"complete": False, "episodes": stage_rows})
        summaries[stage] = summarize(stage_rows)
        write(out / (stage + ".json"), {"complete": True, "summary": summaries[stage], "episodes": stage_rows})
    write(out / "routes.json", {"complete": True, "summaries": summaries, "episodes": rows})
    return summaries


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve(); args.campaign = args.campaign.resolve(); args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    replay_out = args.out / "replay"
    replay_out.mkdir(exist_ok=True)
    replay(args, replay_out)
    summaries = routes(args, args.out)
    write(args.out / "result.json", {
        "complete": True, "status": "COMPLETED_DIAGNOSTIC",
        "replay": json.loads((replay_out / "replay.json").read_text())["summary"],
        "routes": summaries, "geometry_unchanged": True,
        "plate_collision_enabled": True,
        "controller": "MeasuredRouteController plus 1.0 s standstill after physical plate contact or gate activation",
        "gate": {"replay_10_no_falls": json.loads((replay_out / "replay.json").read_text())["summary"]["no_fall"] >= 10,
                 "doors_16": summaries["doors"]["episodes"] >= 16,
                 "transport_16": summaries["transport"]["episodes"] >= 16},
    })
    print("MISSION7_PLATE_SAFE_COMPLETE", flush=True)
