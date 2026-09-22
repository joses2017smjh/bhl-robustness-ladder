"""Small route-to-PlateStage handoff probe on selected validation layouts.

This probe changes one factor relative to the completed route evaluation:
the unchanged PlateSafeRouteController hands command authority to the
unchanged guarded PlateStage.  The default ``switch`` mode retains the
original switch-only handoff.  The ``early`` mode changes only when the
existing PlateStage activation predicate is polled: it is polled every route
step, so the stage can acquire its existing .78 m capture region before
physical plate contact.  Route navigation, plate geometry, stage internals,
termination, and the fall predicate remain unchanged.  The selected layout
indices are intentionally small and are supplied explicitly by the submitter.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bhl_robust.mission.approach_debug import (DebugEnv, PlateSafeRouteController,
                                               wrap, yaw_of)
from mission7_plate_stage import PlateStage


def physical_to_action(command, base_action):
    scales = np.asarray([0.4, 0.35, 0.4], dtype=float)
    action = np.asarray(base_action, dtype=float).copy()
    action[:3] = np.arctanh(np.clip(np.asarray(command, dtype=float) / scales, -0.999999, 0.999999))
    return action


class RouteHandoffController:
    """PlateSafe routing with an explicitly scoped stage handoff condition."""

    def __init__(self, env, handoff_mode="switch"):
        self.env = env
        self.handoff_mode = handoff_mode
        self.route = PlateSafeRouteController(env, contact_hold_s=1.0)
        self.stage = PlateStage(env)
        self.handoff_seen = False
        self.phase_history = []
        self.route_phase_history = []
        self.handoff_condition_first_true_time = None
        self.handoff_condition_reason = None
        self.stage_activation_time = None
        self.stage_entry = None
        self.stage_activation_times = []
        self.stage_entries = []
        self.plate_contact_events = []

    def _entry_state(self, previous_route_phase, route_action):
        """Record the state at the instant the unchanged stage is acquired."""
        env, runner, slot = self.env, self.env.runner, self.env.slot
        door, side = int(self.stage.door), int(self.stage.side)
        _, direction = env.layout.door(door)
        direction = np.asarray(direction, dtype=float)
        direction /= max(np.linalg.norm(direction), 1e-9)
        plate = np.asarray(env.layout.plate(door, side), dtype=float)
        xy = runner.d.xpos[slot.body_id, :2].astype(float)
        yaw = yaw_of(env)
        velocity = runner.d.qvel[slot.qvel_adr:slot.qvel_adr + 2].astype(float)
        rotation = np.array([[np.cos(yaw), np.sin(yaw)],
                             [-np.sin(yaw), np.cos(yaw)]])
        lateral = np.array([-direction[1], direction[0]])
        delta = xy - plate
        return {
            "time_s": float(runner.d.time),
            "door": door,
            "side": side,
            "plate_xy": plate.tolist(),
            "door_direction_world": direction.tolist(),
            "world_direction": ("+x" if direction[0] > .5 else "-x" if direction[0] < -.5
                                 else "+y" if direction[1] > .5 else "-y"),
            "base_xy": xy.tolist(),
            "base_yaw_rad": float(yaw),
            "plate_distance_m": float(np.linalg.norm(delta)),
            "lateral_offset_m": float(np.dot(delta, lateral)),
            "forward_distance_m": float(np.dot(delta, direction)),
            "heading_error_rad": float(wrap(yaw - np.arctan2(direction[1], direction[0]))),
            "velocity_world_mps": velocity.tolist(),
            "velocity_body_mps": (rotation @ velocity).tolist(),
            "linear_speed_mps": float(np.linalg.norm(velocity)),
            "angular_velocity_radps": float(runner.d.qvel[slot.qvel_adr + 5]),
            "previous_route_phase": previous_route_phase,
            "route_waypoint": int(self.route.waypoint),
            "route_action": np.asarray(route_action, dtype=float).tolist(),
        }

    def action(self):
        base_action = self.route.action()
        recorded = np.tanh(base_action[:3]) * np.asarray([0.4, 0.35, 0.4])
        now = float(self.env.runner.d.time)
        previous_route_phase = self.env.phase
        self.route_phase_history.append({
            "time_s": now,
            "phase": previous_route_phase,
            "waypoint": int(self.route.waypoint),
            "stage_phase_before": self.stage.phase,
        })
        poll_stage = (self.handoff_mode == "early" or self.env.phase == "switch"
                      or self.stage.phase != "recorded")
        if poll_stage:
            stage_phase_before = self.stage.phase
            if not self.handoff_seen:
                self.handoff_seen = stage_phase_before != "recorded"
            command, phase = self.stage.command(recorded)
            if stage_phase_before == "recorded" and self.stage.phase != "recorded":
                self.handoff_seen = True
                self.handoff_condition_first_true_time = now
                self.handoff_condition_reason = (
                    "continuous_existing_plate_stage_predicate"
                    if self.handoff_mode == "early" else "route_switch_phase")
                self.stage_activation_time = now
                self.stage_entry = self._entry_state(previous_route_phase, base_action)
                self.stage_activation_times.append(now)
                self.stage_entries.append(self.stage_entry)
                self.phase_history.append({
                    "time_s": now,
                    "phase": "stage_" + self.stage.phase,
                    "event": "activation",
                })
            if phase != "recorded":
                self.env.phase = "stage_" + phase
                if not self.phase_history or self.phase_history[-1].get("phase") != self.env.phase:
                    self.phase_history.append({"time_s": now, "phase": self.env.phase,
                                               "event": "transition"})
                return physical_to_action(command, base_action)
        return base_action

    def observe_contacts(self, events):
        for event in events:
            if event["world_geom"].startswith("plate_"):
                self.plate_contact_events.append(event)


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    stages = (args.stage,) if args.stage != "both" else ("doors", "transport")
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    rows = []
    for stage in stages:
        for index in indices:
            env = DebugEnv(args.repo, args.out / f"{stage}-{index}-cache",
                           stage=stage, split="validation", seed=1000)
            env.reset(index)
            controller = RouteHandoffController(env, handoff_mode=args.handoff)
            while True:
                env.runner.contact_trace = []
                _, _, done, _ = env.step(controller.action())
                controller.observe_contacts(env.runner.contact_trace)
                if done:
                    break
            row = env.metrics()
            activation_time = (controller.stage_activation_times[0]
                               if controller.stage_activation_times else None)
            stage_entry = controller.stage_entries[0] if controller.stage_entries else None
            contact_time = (controller.plate_contact_events[0]["time_s"]
                            if controller.plate_contact_events else None)
            target_contact_time = None
            if stage_entry is not None:
                target_name = f"plate_{stage_entry['door']}_{stage_entry['side']}"
                target_events = [event for event in controller.plate_contact_events
                                 if event["world_geom"] == target_name]
                if target_events:
                    target_contact_time = target_events[0]["time_s"]
            fall_rows = [trace for trace in row["diagnostic_trace"] if trace["tilt"] >= .78]
            fall = fall_rows[0] if fall_rows else None
            if row["success"]:
                failure_phase = "none"
            elif activation_time is None:
                failure_phase = "before PlateStage"
            elif controller.stage.phase != "recorded":
                failure_phase = "during PlateStage"
            else:
                failure_phase = "after PlateStage"
            stage_completion_time = next(
                (item["time_s"] for item in reversed(controller.stage.history)
                 if item["phase"] == "recorded"),
                None,
            )
            row.update(
                layout_index=index,
                stage=stage,
                handoff_mode=args.handoff,
                controller=("PlateSafeRouteController plus unchanged guarded PlateStage "
                            f"on {args.handoff} handoff"),
                route_plate_contacts_seen=sorted(set(controller.route.plate_contacts_seen)),
                guarded_stage_activated=controller.handoff_seen,
                guarded_stage_history=controller.stage.history,
                guarded_stage_phase_history=controller.phase_history,
                guarded_stage_completed=bool(controller.stage.done),
                route_phase_history=controller.route_phase_history,
                handoff_condition_first_true_time=controller.handoff_condition_first_true_time,
                handoff_condition_reason=controller.handoff_condition_reason,
                plate_stage_activation_time=activation_time,
                first_plate_contact_time=contact_time,
                first_target_plate_contact_time=target_contact_time,
                activation_before_first_plate_contact=(
                    activation_time is not None and contact_time is not None
                    and contact_time > activation_time),
                activation_to_first_plate_contact_margin_s=(
                    None if activation_time is None or contact_time is None
                    else float(contact_time - activation_time)),
                activation_to_target_plate_contact_margin_s=(
                    None if activation_time is None or target_contact_time is None
                    else float(target_contact_time - activation_time)),
                plate_contact_events=controller.plate_contact_events,
                stage_entry_state=stage_entry,
                stage_activation_times=controller.stage_activation_times,
                stage_entry_states=controller.stage_entries,
                stage_completion_time=stage_completion_time,
                crossing_completion=bool(controller.stage.done),
                fall_time_s=None if fall is None else fall["time_s"],
                fall_phase=None if fall is None else fall["phase"],
                failure_phase=failure_phase,
            )
            rows.append(row)
            (args.out / f"{stage}-{index}.json").write_text(json.dumps(row, indent=2) + "\n")
    result = {
        "complete": True,
        "status": "COMPLETED_TARGETED_HANDOFF_PROBE",
        "controller": ("PlateSafeRouteController plus unchanged guarded PlateStage "
                       f"on {args.handoff} handoff"),
        "selected_indices": indices,
        "stages": list(stages),
        "geometry_unchanged": True,
        "activation_semantics_unchanged": True,
        "fall_predicate_unchanged": True,
        "plate_stage_internal_behavior_unchanged": True,
        "handoff_mode": args.handoff,
        "route_episode_count": len(rows),
        "guarded_stage_activations": sum(row["guarded_stage_activated"] for row in rows),
        "guarded_stage_completions": sum(row["guarded_stage_completed"] for row in rows),
        "episodes": rows,
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "episodes": len(rows),
        "guarded_stage_activations": result["guarded_stage_activations"],
        "guarded_stage_completions": result["guarded_stage_completions"],
        "out": str(args.out),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("doors", "transport", "both"), required=True)
    parser.add_argument("--indices", required=True, help="comma-separated validation layout indices")
    parser.add_argument("--handoff", choices=("switch", "early"), default="switch")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.out = args.out.resolve()
    if not args.out.is_relative_to(args.repo):
        parser.error("output must remain inside repository")
    run(args)
