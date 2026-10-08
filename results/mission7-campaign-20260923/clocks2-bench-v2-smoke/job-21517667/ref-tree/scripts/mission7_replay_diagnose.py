"""Instrument the unchanged ten-episode Mission7 replay before intervention."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.mission.approach_debug import DebugEnv, physical_sample, wrap
from mission7_overnight import write


def _source(campaign):
    source = json.loads((campaign / "fullroute/legacy-doors.json").read_text())
    if not source.get("complete"):
        raise RuntimeError("legacy door route is not complete")
    return source


def _name(model, kind, index):
    return mujoco.mj_id2name(model, kind, int(index)) or ""


def _plate_edge_distance(env, base_xy, geom_name):
    _, door, side = geom_name.split("_")
    door, side = int(door), int(side)
    center, direction = env.layout.door(door)
    center = env.layout.plate(door, side)
    lateral = np.array([-direction[1], direction[0]], dtype=float)
    delta = np.asarray(base_xy, dtype=float) - center
    longitudinal = abs(float(np.dot(delta, direction)))
    transverse = abs(float(np.dot(delta, lateral)))
    if side == env.layout.correct_sides[door]:
        return float(np.linalg.norm(delta) - .24)
    outside = np.maximum([longitudinal - .24, transverse - .24], 0.)
    if np.any(outside > 0.):
        return float(np.linalg.norm(outside))
    return -float(min(.24 - longitudinal, .24 - transverse))


def clearance_snapshot(env):
    """Read-only geometry distances at a sampled replay state.

    Center-distance culling keeps this diagnostic bounded while retaining every
    pair that could be within 2.5 m; contact events themselves are recorded at
    every 0.5 ms physics step by the runner.
    """
    runner, model, data = env.runner, env.model, env.runner.d
    robot = np.flatnonzero(runner.own).tolist()
    obstacles = sorted(runner.walls | set(runner.plates))
    plates = sorted(runner.plates)
    fromto = np.zeros(6)
    nearest_obstacle = (float("inf"), "", "")
    nearest_plate = (float("inf"), "", "")
    robot_xyz = data.geom_xpos[robot]
    obstacle_xyz = data.geom_xpos[obstacles]
    center_distance = np.linalg.norm(robot_xyz[:, None, :] - obstacle_xyz[None, :, :], axis=2)
    robot_radius = np.asarray(model.geom_rbound[robot], dtype=float)[:, None]
    obstacle_radius = np.asarray(model.geom_rbound[obstacles], dtype=float)[None, :]
    candidates = np.argwhere(center_distance <= 2.5 + robot_radius + obstacle_radius)
    if not len(candidates):
        candidates = np.asarray([[int(np.argmin(center_distance) // len(obstacles)),
                                  int(np.argmin(center_distance) % len(obstacles))]])
    for robot_index, obstacle_index in candidates:
        robot_geom, obstacle_geom = robot[int(robot_index)], obstacles[int(obstacle_index)]
        distance = float(mujoco.mj_geomDistance(
            model, data, int(robot_geom), int(obstacle_geom), 10., fromto))
        row = (distance, _name(model, mujoco.mjtObj.mjOBJ_GEOM, robot_geom),
               _name(model, mujoco.mjtObj.mjOBJ_GEOM, obstacle_geom))
        if distance < nearest_obstacle[0]:
            nearest_obstacle = row
        if obstacle_geom in runner.plates and distance < nearest_plate[0]:
            nearest_plate = row
    base_xy = data.xpos[runner.slots[0].body_id, :2]
    edge_rows = [(_plate_edge_distance(env, base_xy, _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)),
                  _name(model, mujoco.mjtObj.mjOBJ_GEOM, geom)) for geom in plates]
    correct_edges = [row for row in edge_rows if int(row[1].split("_")[1]) < 2 and
                     int(row[1].split("_")[2]) == env.layout.correct_sides[int(row[1].split("_")[1])]]
    return {
        "robot_obstacle_clearance_m": None if not np.isfinite(nearest_obstacle[0]) else nearest_obstacle[0],
        "nearest_obstacle_robot_geom": nearest_obstacle[1],
        "nearest_obstacle_geom": nearest_obstacle[2],
        "robot_plate_surface_clearance_m": None if not np.isfinite(nearest_plate[0]) else nearest_plate[0],
        "nearest_plate_robot_geom": nearest_plate[1],
        "nearest_plate_geom": nearest_plate[2],
        "robot_plate_edge_clearance_m": min((row[0] for row in edge_rows), default=None),
        "nearest_plate_edge_geom": min(edge_rows, default=(None, ""))[1] if edge_rows else None,
        "correct_plate_edge_clearance_m": min((row[0] for row in correct_edges), default=None),
        "base_xy": base_xy.astype(float).tolist(),
    }


def _first(events, predicate):
    return min((event for event in events if predicate(event)),
               key=lambda event: event["time_s"], default=None)


def _event_detail(event, samples, env):
    if event is None:
        return None
    sample = min(samples, key=lambda row: abs(row["time_s"] - event["time_s"]))
    plate_name = event["world_geom"] if event["world_geom"].startswith("plate_") else None
    lateral_offset = None
    longitudinal_offset = None
    if plate_name:
        _, door, side = plate_name.split("_")
        center, direction = env.layout.door(int(door))
        center = env.layout.plate(int(door), int(side))
        delta = np.asarray(event.get("base_xy", sample["xy"])) - center
        lateral = np.array([-direction[1], direction[0]], dtype=float)
        lateral_offset = float(np.dot(delta, lateral))
        longitudinal_offset = float(np.dot(delta, direction))
    return {
        "time_s": event["time_s"],
        "geom1": event["geom1"], "geom2": event["geom2"],
        "world_geom": event["world_geom"], "body1": event.get("body1", ""),
        "body2": event.get("body2", ""), "world_body": event.get("world_body", ""),
        "distance_m": event["distance_m"],
        "normal_force_N": event["normal_force_N"],
        "tangent_force_N": event["tangent_force_N"],
        "base_xy": event.get("base_xy", sample["xy"]),
        "base_yaw": event.get("base_yaw", sample["yaw"]),
        "lateral_offset_m": lateral_offset,
        "longitudinal_offset_m": longitudinal_offset,
        "nearest_sample": {
            "time_s": sample["time_s"], "command": sample["command"],
            "velocity_world": sample["velocity"], "velocity_body": sample["velocity_body"],
            "yaw_rate": sample["yaw_rate"], "phase": sample["phase"],
        },
    }


def _window(samples, center, radius=.8):
    rows = [row for row in samples if center is not None and abs(row["time_s"] - center) <= radius]
    return [{key: row[key] for key in ("time_s", "xy", "velocity", "velocity_body", "yaw",
                                       "yaw_rate", "tilt", "command", "command_world", "phase")}
            for row in rows]


def _episode(env, index, source_episode, clearance_stride=5):
    runner = env.runner
    samples = []
    maximum_pose_difference = 0.
    for reference in source_episode["diagnostic_trace"]:
        env.state.open = [t is not None and runner.d.time + 1e-8 >= t
                          for t in source_episode["activation_s"]]
        env._gates()
        command = np.asarray(reference["command"], dtype=float)
        runner.contact_trace = []
        runner.step([env.controller.update(runner.observe(0, command))])
        sample = physical_sample(env, command, reference["phase"])
        sample["contact_events"] = list(runner.contact_trace)
        # Exact contact events are captured at every physics substep. Full
        # geometry distances are sampled every 200 ms and on contact/fall
        # boundaries to keep the ten-episode diagnostic tractable.
        if len(samples) % clearance_stride == 0 or sample["contact_events"] or sample["tilt"] >= .78:
            sample["clearance"] = clearance_snapshot(env)
        else:
            sample["clearance"] = None
        samples.append(sample)
        maximum_pose_difference = max(
            maximum_pose_difference,
            float(np.linalg.norm(np.asarray(sample["xy"]) - np.asarray(reference["xy"]))),
            abs(sample["tilt"] - reference["tilt"]),
            abs(wrap(sample["yaw"] - reference["yaw"])),
        )
        if not np.isfinite(runner.d.qpos).all() or runner.d.warning.number.sum():
            raise FloatingPointError("diagnostic replay physics warning")
        if abs(sample["time_s"] - reference["time_s"]) >= 1e-7:
            raise AssertionError("diagnostic replay time diverged")
    events = [event for sample in samples for event in sample["contact_events"]]
    plate = [event for event in events if event["world_geom"].startswith("plate_")]
    obstacle = [event for event in events if not event["world_geom"].startswith("plate_")]
    fall = next((row for row in samples if row["tilt"] >= .78), None)
    first_plate = _first(plate, lambda _: True)
    first_obstacle = _first(obstacle, lambda _: True)
    last_plate = max((event["time_s"] for event in plate), default=None)
    fall_time = None if fall is None else fall["time_s"]
    if fall_time is None:
        stage = "no_fall"
    elif first_plate is None or fall_time < first_plate["time_s"]:
        stage = "before_plate_entry"
    elif last_plate is not None and fall_time <= last_plate + .08:
        stage = "during_plate_traversal"
    else:
        stage = "after_plate_exit"
    minima = {
        key: min((row["clearance"][key] for row in samples
                  if row["clearance"] is not None and row["clearance"][key] is not None), default=None)
        for key in ("robot_obstacle_clearance_m", "robot_plate_surface_clearance_m",
                    "robot_plate_edge_clearance_m", "correct_plate_edge_clearance_m")
    }
    return {
        "layout_index": index, "layout_seed": source_episode["layout"]["seed"],
        "original_elapsed_s": source_episode["elapsed_s"],
        "replay_elapsed_s": samples[-1]["time_s"], "original_failure": source_episode["failure"],
        "fall_time_s": fall_time, "fall_phase": None if fall is None else fall["phase"],
        "fall_xy": None if fall is None else fall["xy"], "fall_tilt": None if fall is None else fall["tilt"],
        "maximum_pose_difference": maximum_pose_difference,
        "matched_original": maximum_pose_difference < 1e-6,
        "minimum_clearances_m": minima,
        "first_undesired_contact": _event_detail(first_obstacle, samples, env),
        "first_plate_contact": _event_detail(first_plate, samples, env),
        "last_plate_contact_s": last_plate,
        "failure_stage_relative_to_plate": stage,
        "fall_after_first_plate_contact_s": None if fall_time is None or first_plate is None else fall_time-first_plate["time_s"],
        "command_motion_window": _window(samples, fall_time),
        "contact_counts": dict(Counter(event["world_geom"] for event in events)),
        "samples": samples,
    }


def diagnose(args, out):
    source = _source(args.campaign)
    selected = [(index, episode) for index, episode in enumerate(source["episodes"])
                if episode.get("fall")]
    if args.smoke:
        selected = selected[:1]
    env = DebugEnv(args.repo, out / "cache", stage="doors", split="validation", seed=1000)
    rows, next_reset = [], 0
    for index, episode in selected:
        for reset_index in range(next_reset, index + 1):
            env.reset(reset_index)
        next_reset = index + 1
        rows.append(_episode(env, index, episode, clearance_stride=20 if args.smoke else 5))
        write(out / "episodes.json", {"complete": False, "episodes": rows})
    def aggregate_min(key):
        values = [row["minimum_clearances_m"][key] for row in rows
                  if row["minimum_clearances_m"][key] is not None]
        return min(values) if values else None

    report = {
        "complete": True, "status": "COMPLETED_DIAGNOSTIC",
        "episodes": len(rows), "falls_replayed": sum(row["fall_time_s"] is not None for row in rows),
        "unchanged_replay_matches": all(row["matched_original"] for row in rows),
        "selected_layouts": [row["layout_index"] for row in rows],
        "aggregate": {
            "minimum_robot_obstacle_clearance_m": aggregate_min("robot_obstacle_clearance_m"),
            "minimum_robot_plate_surface_clearance_m": aggregate_min("robot_plate_surface_clearance_m"),
            "minimum_robot_plate_edge_clearance_m": aggregate_min("robot_plate_edge_clearance_m"),
            "failure_stages": dict(Counter(row["failure_stage_relative_to_plate"] for row in rows)),
            "first_contact_geoms": dict(Counter((row["first_plate_contact"] or {}).get("world_geom", "none") for row in rows)),
        },
        "episodes_detail": [{key: value for key, value in row.items() if key != "samples"} for row in rows],
        "geometry_unchanged": True,
        "interpretation": "Read-only clearance and contact instrumentation on the exact recorded command stream; no controller, geometry, fall predicate, or activation rule was changed.",
        "interpretation_allowed": all(row["matched_original"] for row in rows),
    }
    if not report["interpretation_allowed"]:
        report["status"] = "INVALID_REPLAY"
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
    args.repo, args.campaign, args.out = args.repo.resolve(), args.campaign.resolve(), args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    diagnose(args, args.out)
    print("MISSION7_REPLAY_DIAGNOSTIC_COMPLETE", flush=True)
