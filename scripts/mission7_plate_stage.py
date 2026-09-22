"""Staged plate-crossing intervention on the exact ten Mission7 replays.

The unchanged replay and its recorded gate-opening times remain the baseline.
This intervention changes only the command stream near each unopened plate:
approach a pre-plate pose, settle with zero translation, then make a short
straight crossing with yaw correction frozen. Correct-side staging is retained;
wrong-side staging is allowed only when the correct plate is more than 1.0 m
away. Geometry, activation schedule, fall predicate, and deterministic reset
stream are unchanged.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import mujoco
import numpy as np

from bhl_robust.mission.approach_debug import DebugEnv, command_action, physical_sample, wrap
from mission7_overnight import write


def _source(campaign):
    data = json.loads((campaign / "fullroute/legacy-doors.json").read_text())
    if not data.get("complete"):
        raise RuntimeError("legacy door route is not complete")
    return data


def _world_command(env, vector, speed=0.30):
    vector = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        return np.zeros(3)
    yaw = float(np.arctan2(
        2 * (env.runner.d.qpos[env.slot.qpos_adr + 3] * env.runner.d.qpos[env.slot.qpos_adr + 6]
           + env.runner.d.qpos[env.slot.qpos_adr + 4] * env.runner.d.qpos[env.slot.qpos_adr + 5]),
        1 - 2 * (env.runner.d.qpos[env.slot.qpos_adr + 5] ** 2
                 + env.runner.d.qpos[env.slot.qpos_adr + 6] ** 2)))
    world = vector / norm * float(speed)
    body = np.array([[np.cos(yaw), np.sin(yaw)],
                     [-np.sin(yaw), np.cos(yaw)]]) @ world
    return np.array([np.clip(body[0], -.4, .4), np.clip(body[1], -.35, .35), 0.])


def _plate_state(env, door, side):
    center, direction = env.layout.door(door)
    plate = env.layout.plate(door, side)
    direction = np.asarray(direction, dtype=float)
    return np.asarray(plate), direction


class PlateStage:
    """One bounded staged maneuver per unopened correct plate."""

    def __init__(self, env, approach_radius=.78, settle_s=.40, cross_s=1.20):
        self.env = env
        self.approach_radius = float(approach_radius)
        self.settle_s = float(settle_s)
        self.cross_s = float(cross_s)
        # A wrong-side plate is only an intervention target when the route is
        # clearly not entering the intended plate.  The threshold separates
        # the layout-13 wrong-side trace from layout-4's earlier near miss.
        self.wrong_side_min_correct_distance = 1.0
        self.door = None
        self.side = None
        self.phase = "recorded"
        self.phase_until = 0.
        self.done = set()
        self.history = []

    def _start(self, door, side, now):
        self.door = door
        self.side = side
        self.phase = "approach"
        self.phase_until = float(now)
        self.history.append({"time_s": float(now), "door": int(door),
                             "side": int(side), "phase": self.phase})

    def command(self, recorded):
        env, runner, slot = self.env, self.env.runner, self.env.slot
        now = float(runner.d.time)
        xy = runner.d.xpos[slot.body_id, :2].copy()
        if self.phase == "recorded":
            candidates = []
            for door, opened in enumerate(env.state.open):
                if opened or door in self.done:
                    continue
                correct = env.layout.correct_sides[door]
                correct_distance = float(np.linalg.norm(
                    xy - _plate_state(env, door, correct)[0]))
                candidates.append((door, correct))
                wrong = -correct
                if correct_distance > self.wrong_side_min_correct_distance:
                    candidates.append((door, wrong))
            nearest = min(
                ((float(np.linalg.norm(xy - _plate_state(env, door, side)[0])),
                  int(side != env.layout.correct_sides[door]), door, side)
                 for door, side in candidates),
                default=(float("inf"), 1, None, None),
            )
            if nearest[0] <= self.approach_radius:
                self._start(nearest[2], nearest[3], now)
            else:
                return np.asarray(recorded, dtype=float), "recorded"
        plate, direction = _plate_state(env, self.door, self.side)
        if self.phase == "approach":
            pre = plate - direction * .30
            if np.linalg.norm(xy - pre) <= .13:
                self.phase = "settle"
                self.phase_until = now + self.settle_s
                self.history.append({"time_s": now, "door": int(self.door),
                                     "side": int(self.side), "phase": self.phase})
                return np.zeros(3), self.phase
            return _world_command(env, pre - xy), self.phase
        if self.phase == "settle":
            if now < self.phase_until:
                return np.zeros(3), self.phase
            self.phase = "cross"
            self.phase_until = now + self.cross_s
            self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase})
        if self.phase == "cross":
            # The short crossing deliberately carries no yaw correction.
            if now < self.phase_until:
                return _world_command(env, direction), self.phase
            self.done.add(self.door)
            self.history.append({"time_s": now, "door": int(self.door),
                                 "side": int(self.side), "phase": "recorded"})
            self.door = None
            self.side = None
            self.phase = "recorded"
            return np.asarray(recorded, dtype=float), "recorded"
        raise AssertionError(self.phase)


def _episode(env, index, source_episode, baseline):
    runner = env.runner
    stage = PlateStage(env)
    samples = []
    maximum_recorded_pose_difference = 0.
    for reference in source_episode["diagnostic_trace"]:
        env.state.open = [t is not None and runner.d.time + 1e-8 >= t
                          for t in source_episode["activation_s"]]
        env._gates()
        recorded = np.asarray(reference["command"], dtype=float)
        effective, phase = stage.command(recorded)
        runner.contact_trace = []
        runner.step([env.controller.update(runner.observe(0, effective))])
        sample = physical_sample(env, effective, phase)
        sample["recorded_command"] = recorded.tolist()
        sample["effective_command"] = effective.tolist()
        sample["contact_events"] = list(runner.contact_trace)
        samples.append(sample)
        maximum_recorded_pose_difference = max(
            maximum_recorded_pose_difference,
            float(np.linalg.norm(np.asarray(sample["xy"]) - np.asarray(reference["xy"]))),
            abs(sample["tilt"] - reference["tilt"]),
            abs(wrap(sample["yaw"] - reference["yaw"])),
        )
        if abs(sample["time_s"] - reference["time_s"]) >= 1e-7:
            raise AssertionError("intervention replay time diverged")
        if not np.isfinite(runner.d.qpos).all() or runner.d.warning.number.sum():
            raise FloatingPointError("plate-stage replay physics warning")
    falls = [sample for sample in samples if sample["tilt"] >= .78]
    contacts = [event for sample in samples for event in sample["contact_events"]
                if event["world_geom"].startswith("plate_")]
    return {
        "layout_index": index,
        "layout_seed": source_episode["layout"]["seed"],
        "original_elapsed_s": source_episode["elapsed_s"],
        "replay_elapsed_s": samples[-1]["time_s"],
        "fall": bool(falls),
        "first_fall_s": None if not falls else falls[0]["time_s"],
        "maximum_tilt": max(sample["tilt"] for sample in samples),
        "maximum_recorded_pose_difference": maximum_recorded_pose_difference,
        "plate_contacts": len(contacts),
        "peak_plate_normal_force_N": max((event["normal_force_N"] for event in contacts), default=0.),
        "peak_plate_tangent_force_N": max((event["tangent_force_N"] for event in contacts), default=0.),
        "minimum_plate_distance_m": min((event["distance_m"] for event in contacts), default=None),
        "stage_history": stage.history,
        "baseline": {key: baseline.get(key) for key in ("fall_time_s", "fall_phase", "minimum_clearances_m",
                                                         "failure_stage_relative_to_plate")},
        "samples": samples,
    }


def run(args, out):
    source = _source(args.campaign)
    baseline_path = args.baseline / "result.json"
    baseline = json.loads(baseline_path.read_text())
    if not baseline.get("unchanged_replay_matches"):
        raise RuntimeError("authoritative baseline is not exact; intervention is not interpretable")
    baseline_rows = {row["layout_index"]: row for row in baseline["episodes_detail"]}
    selected = [(index, episode) for index, episode in enumerate(source["episodes"]) if episode.get("fall")]
    if args.smoke:
        selected = selected[:1]
    env = DebugEnv(args.repo, out / "cache", stage="doors", split="validation", seed=1000)
    rows, next_reset = [], 0
    for index, episode in selected:
        for reset_index in range(next_reset, index + 1):
            env.reset(reset_index)
        next_reset = index + 1
        rows.append(_episode(env, index, episode, baseline_rows[index]))
        write(out / "episodes.json", {"complete": False, "episodes": rows})
    report = {
        "complete": True,
        "status": "COMPLETED_INTERVENTION" if not args.smoke else "COMPLETED_SMOKE",
        "intervention": "staged plate maneuver: pre-plate approach, 0.40 s settle, 1.20 s straight crossing with yaw correction frozen; wrong-side staging only when the correct plate is over 1.0 m away",
        "geometry_unchanged": True,
        "activation_schedule_unchanged": True,
        "fall_predicate_unchanged": True,
        "baseline_result": str(baseline_path),
        "episodes": len(rows),
        "falls": sum(row["fall"] for row in rows),
        "upright": sum(not row["fall"] for row in rows),
        "exact_replay_gate_passed": len(rows) == 10 and sum(not row["fall"] for row in rows) == 10,
        "episode_comparison": [{key: value for key, value in row.items() if key != "samples"} for row in rows],
    }
    write(out / "episodes.json", {"complete": True, "episodes": rows})
    write(out / "result.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.campaign = args.campaign.resolve()
    args.baseline = args.baseline.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    run(args, args.out)
    print("MISSION7_PLATE_STAGE_COMPLETE", flush=True)
