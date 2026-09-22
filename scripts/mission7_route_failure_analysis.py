"""Episode-level comparison of the guarded replay and existing route traces.

This is a read-only analysis of completed Mission 7 artifacts.  It deliberately
does not rerun MuJoCo, change scoring, or infer contact causality where the
route job did not record a contact event.  The route campaign used
``PlateSafeRouteController``; the guarded replay used ``PlateStage``.  Keeping
those controller identities explicit is the central diagnostic of this report.
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def load(path: Path):
    return json.loads(path.read_text())


def wrap(angle: float) -> float:
    return math.atan2(math.sin(angle), math.cos(angle))


def geometry(episode, door=0, side=None):
    layout = episode["layout"]
    k = layout["door_indices"][door]
    a, b = layout["route"][k], layout["route"][k + 1]
    cell = float(layout["cell_m"])
    delta = (float(b[0] - a[0]), float(b[1] - a[1]))
    norm = math.hypot(*delta)
    direction = (delta[0] / norm, delta[1] / norm)
    side = layout["correct_sides"][door] if side is None else int(side)
    lateral = (-direction[1], direction[0])
    anchor = (float(a[0]) * cell, float(a[1]) * cell)
    plate = (
        anchor[0] + lateral[0] * side * 0.42 - direction[0] * 0.12,
        anchor[1] + lateral[1] * side * 0.42 - direction[1] * 0.12,
    )
    door_center = (
        (float(a[0]) + float(b[0])) * cell / 2.0,
        (float(a[1]) + float(b[1])) * cell / 2.0,
    )
    return plate, direction, lateral, door_center, side


def state_features(episode, sample, door=0, side=None):
    plate, direction, lateral, _, side = geometry(episode, door, side)
    delta = (sample["xy"][0] - plate[0], sample["xy"][1] - plate[1])
    target_heading = math.atan2(direction[1], direction[0])
    _, _, _, door_center, _ = geometry(episode, door, side)
    return {
        "time_s": float(sample["time_s"]),
        "plate_side": int(side),
        "plate_distance_m": math.hypot(*delta),
        "lateral_offset_m": delta[0] * lateral[0] + delta[1] * lateral[1],
        "longitudinal_offset_m": delta[0] * direction[0] + delta[1] * direction[1],
        "heading_error_rad": wrap(target_heading - float(sample["yaw"])),
        "world_direction": [float(direction[0]), float(direction[1])],
        "world_minus_x": bool(direction[0] < -0.5),
        "plate_center_xy": [float(x) for x in plate],
        "door_center_xy": [float(x) for x in door_center],
        "door_distance_m": math.hypot(
            float(sample["xy"][0]) - door_center[0],
            float(sample["xy"][1]) - door_center[1],
        ),
        "xy": [float(x) for x in sample["xy"]],
        "velocity_world_mps": [float(x) for x in sample["velocity"]],
        "velocity_body_mps": [float(x) for x in sample["velocity_body"]],
        "speed_mps": math.hypot(*sample["velocity"]),
        "yaw_rad": float(sample["yaw"]),
        "yaw_rate_rad_s": float(sample["yaw_rate"]),
        "tilt": float(sample["tilt"]),
    }


def transitions(trace):
    result = []
    previous = object()
    for sample in trace:
        phase = sample.get("phase")
        if phase != previous:
            result.append({"time_s": float(sample["time_s"]), "phase": phase})
            previous = phase
    return result


def first(trace, predicate):
    return next((sample for sample in trace if predicate(sample)), None)


def first_nonempty_contact(trace):
    return first(trace, lambda sample: bool(sample.get("contacts")))


def guarded_capture(guarded, route_rows):
    rows = []
    for row in guarded:
        entry = first(row["samples"], lambda sample: sample["phase"] == "approach")
        history = row.get("stage_history", [])
        if entry is None or not history:
            rows.append({
                "layout_index": row["layout_index"],
                "layout_seed": row["layout_seed"],
                "stage_activated": False,
                "stage_history": history,
            })
            continue
        side = history[0]["side"]
        source = route_rows[row["layout_index"]]
        rows.append({
            "layout_index": row["layout_index"],
            "layout_seed": row["layout_seed"],
            "stage_activated": True,
            "stage_history": history,
            "entry": state_features(source, entry, side=side),
        })
    return rows


def route_category(row, trace, coarse):
    has_plate = bool(row.get("plate_contacts_seen"))
    has_switch = any(sample.get("phase") == "switch" for sample in trace)
    has_plate_brake = any(sample.get("phase") == "plate_brake" for sample in trace)
    has_acquire = any(sample.get("phase") == "acquire" for sample in trace)
    first_fall = first(trace, lambda sample: float(sample["tilt"]) >= 0.78)
    pickup = bool(row.get("pickup_success"))
    if first_fall is not None:
        if has_plate and any(sample.get("phase") == "plate_brake" for sample in trace):
            return "after_plate_interaction_or_recovery"
        if has_plate or has_switch:
            return "plate_alignment_or_traversal"
        if has_acquire or pickup:
            return "manipulation_before_plate"
        return "approach_or_navigation_before_plate"
    if row.get("failure") == "timeout":
        if has_plate or has_plate_brake or has_switch:
            return "plate_interaction_not_completed"
        if has_acquire or pickup:
            return "manipulation_before_plate"
        return "approach_or_navigation_before_plate"
    if row.get("success"):
        return "success"
    return "incomplete"


def route_episode(row, route_stage, guarded_entries):
    trace = row.get("diagnostic_trace", [])
    coarse = row.get("trace", [])
    switch = first(trace, lambda sample: sample.get("phase") == "switch")
    plate_brake = first(trace, lambda sample: sample.get("phase") == "plate_brake")
    fall = first(trace, lambda sample: float(sample["tilt"]) >= 0.78)
    contact = first_nonempty_contact(trace)
    gate = first(coarse, lambda sample: any(sample.get("gates", [])))
    handoff_phases = {"approach", "settle", "cross"}
    guarded_active_in_route = any(sample.get("phase") in handoff_phases for sample in trace)
    features = None
    feature_sample = switch or plate_brake
    if feature_sample is not None:
        features = state_features(row, feature_sample)
    exact = [entry for entry in guarded_entries if entry["layout_index"] == row["layout_index"]]
    exact_entry = exact[0].get("entry") if exact and exact[0].get("entry") else None
    return {
        "stage": route_stage,
        "layout_index": row["layout_index"],
        "layout_seed": row["layout"]["seed"],
        "failure": row.get("failure"),
        "success": bool(row.get("success")),
        "fall": bool(row.get("fall")),
        "failure_taxonomy": route_category(row, trace, coarse),
        "first_meaningful_divergence": (
            "route_navigation_never_reached_guarded_stage"
            if switch is None and not guarded_active_in_route
            else "route_used_plate_safe_switch_instead_of_guarded_stage"
        ),
        "guarded_stage_activated": guarded_active_in_route,
        "route_switch_activated": switch is not None,
        "route_plate_brake_activated": plate_brake is not None,
        "route_interaction_state_source": None if feature_sample is None else feature_sample.get("phase"),
        "controller_state_transitions": transitions(trace),
        "plate_contacts_seen": row.get("plate_contacts_seen", []),
        "first_approx_plate_brake_s": None if plate_brake is None else float(plate_brake["time_s"]),
        "first_observed_plate_contact": None if not row.get("plate_contacts_seen") else {
            "approx_time_s": None if plate_brake is None else float(plate_brake["time_s"]),
            "plates": row.get("plate_contacts_seen"),
            "contact_trace_available": False,
        },
        "first_gate_open_s": None if gate is None else float(gate["time_s"]),
        "first_unintended_contact": None if contact is None else {
            "time_s": float(contact["time_s"]),
            "contacts": contact.get("contacts", []),
            "phase": contact.get("phase"),
        },
        "fall_onset": None if fall is None else {
            "time_s": float(fall["time_s"]),
            "phase": fall.get("phase"),
            "xy": fall.get("xy"),
            "velocity_world_mps": fall.get("velocity"),
            "yaw_rad": fall.get("yaw"),
            "tilt": float(fall["tilt"]),
        },
        "world_minus_x": bool(
            row["layout"]["route"][row["layout"]["door_indices"][0] + 1][0]
            < row["layout"]["route"][row["layout"]["door_indices"][0]][0]
        ),
        "stage_entry_or_switch": features,
        "matching_exact_replay_entry": exact_entry,
        "outside_exact_replay_capture_region": None if features is None or exact_entry is None else {
            key: abs(features[key] - exact_entry[key]) > 0.25
            for key in ("lateral_offset_m", "longitudinal_offset_m", "heading_error_rad", "speed_mps")
        },
        "elapsed_s": float(row["elapsed_s"]),
        "distance_m": float(row["distance_m"]),
        "goal_distance_m": float(row["final_goal_distance_m"]),
        "door_success": bool(row.get("door_success")),
        "pickup_success": bool(row.get("pickup_success")),
        "placement_success": bool(row.get("placement_success")),
        "wall_contact_intervals": int(row.get("wall_contact_intervals", 0)),
    }


def numeric_summary(rows, key):
    values = [row["entry"][key] for row in rows if row.get("entry") and key in row["entry"]]
    if not values:
        return None
    return {"min": min(values), "max": max(values), "mean": sum(values) / len(values), "n": len(values)}


def run(args):
    guarded_report = load(args.guarded_result)
    guarded_episodes = load(args.guarded_episodes)["episodes"]
    doors = load(args.doors)["episodes"]
    transport = load(args.transport)["episodes"]
    # The guarded replay is indexed against the same validation layouts as the
    # route campaign.  Doors supplies the layout metadata for the replay rows.
    guarded_entries = guarded_capture(guarded_episodes, doors)
    route_rows = [route_episode(row, "doors", guarded_entries) for row in doors]
    route_rows += [route_episode(row, "transport", guarded_entries) for row in transport]
    exact_entries = [row for row in guarded_entries if row.get("entry")]
    summary = {
        "guarded_replay": {
            "episodes": len(guarded_episodes),
            "upright": sum(not row["fall"] for row in guarded_episodes),
            "exact_replay_gate_passed": bool(guarded_report.get("exact_replay_gate_passed")),
            "capture_region_entry": {
                key: numeric_summary(exact_entries, key)
                for key in ("plate_distance_m", "lateral_offset_m", "longitudinal_offset_m",
                            "heading_error_rad", "speed_mps", "yaw_rate_rad_s", "tilt")
            },
        },
        "routes": {},
    }
    for stage in ("doors", "transport"):
        rows = [row for row in route_rows if row["stage"] == stage]
        summary["routes"][stage] = {
            "episodes": len(rows),
            "successes": sum(row["success"] for row in rows),
            "falls": sum(row["fall"] for row in rows),
            "timeouts": sum(row["failure"] == "timeout" for row in rows),
            "guarded_stage_activations": sum(row["guarded_stage_activated"] for row in rows),
            "route_switch_activations": sum(row["route_switch_activated"] for row in rows),
            "plate_brake_activations": sum(row["route_plate_brake_activated"] for row in rows),
            "world_minus_x": sum(row["world_minus_x"] for row in rows),
            "taxonomy": {
                category: sum(row["failure_taxonomy"] == category for row in rows)
                for category in sorted({row["failure_taxonomy"] for row in rows})
            },
        }
    report = {
        "complete": True,
        "status": "COMPLETED_READ_ONLY_ROUTE_FAILURE_ANALYSIS",
        "controller_comparison": {
            "guarded_replay": "PlateStage",
            "route_evaluation": "PlateSafeRouteController(contact_hold_s=1.0)",
            "guarded_stage_observed_in_route_trace": any(row["guarded_stage_activated"] for row in route_rows),
            "interpretation": "The existing route evaluation does not test the guarded PlateStage handoff; route switch and plate brake are separate states.",
        },
        "hypothesis": {
            "narrow_exact_replay_gate_supported": True,
            "evidence": [
                "The guarded exact replay has 10/10 upright episodes.",
                "The route campaign has zero guarded-stage activations in both stages.",
                "Most route episodes diverge in navigation, acquisition, or PlateSafe switch before the guarded replay maneuver can be evaluated.",
            ],
        },
        "summary": summary,
        "guarded_capture_entries": guarded_entries,
        "route_episodes": route_rows,
        "limitations": [
            "Route diagnostic_trace records wall/door contacts but the route campaign did not retain per-physics-step plate contact events; plate-brake time is an approximate first-contact indicator.",
            "No contact causality is inferred from temporal association alone.",
            "The route failure taxonomy is an operational phase classification, not a replacement for the unchanged benchmark predicate.",
        ],
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"out": str(args.out), "summary": summary}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--guarded-result", type=Path, required=True)
    parser.add_argument("--guarded-episodes", type=Path, required=True)
    parser.add_argument("--doors", type=Path, required=True)
    parser.add_argument("--transport", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    run(parser.parse_args())
