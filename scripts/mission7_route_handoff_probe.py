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
import sys

import numpy as np

from bhl_robust.mission.approach_debug import (DebugEnv, PlateSafeRouteController,
                                               wrap, yaw_of)
from mission7_plate_stage import PlateStage


# Body-frame command scales (vx, vy, wz) shared by every conversion below.
COMMAND_SCALES = np.asarray([0.4, 0.35, 0.4], dtype=float)

# Forward speed the rejoin pulse imposes.  Note this is the same magnitude the
# route controller already drives at, which is why the pulse must be checked
# for effect and not merely for activation.
REJOIN_PULSE_MPS = .30

# Smallest commanded-velocity difference counted as a real change (m/s).
COMMAND_EPSILON_MPS = 1e-3


def action_to_physical(action):
    return np.tanh(np.asarray(action, dtype=float)[:3]) * COMMAND_SCALES


def physical_to_action(command, base_action):
    action = np.asarray(base_action, dtype=float).copy()
    action[:3] = np.arctanh(np.clip(np.asarray(command, dtype=float) / COMMAND_SCALES,
                                    -0.999999, 0.999999))
    return action


class RouteHandoffController:
    """PlateSafe routing with an explicitly scoped stage handoff condition."""

    def __init__(self, env, handoff_mode="switch", rejoin_diagnostic=False,
                 rejoin_fix="none"):
        self.env = env
        self.handoff_mode = handoff_mode
        self.rejoin_diagnostic = bool(rejoin_diagnostic)
        self.rejoin_fix = rejoin_fix
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
        self.route_action_events = []
        self.decision_events = []
        self.stage_exit_events = []
        self.post_step_events = []
        self.rejoin_fix_until = 0.
        self.rejoin_fix_used = False
        self.rejoin_fix_events = []
        # An intervention that fires is not an intervention that did anything.
        # 21399503 fired a .30 m/s "forward restart pulse" while the route was
        # already commanding .30 m/s forward, so the run looked clean and the
        # forward drive never changed.  Record the command with and without the
        # intervention so a no-op is visible instead of inferred.
        self.rejoin_fix_command_samples = []
        self.rejoin_fix_command_delta = 0.
        self.rejoin_fix_forward_delta = 0.
        self.post_stage_exit_time = None
        self.post_stage_exit_waypoint = None
        self.post_stage_last_target_waypoint = None
        self.post_stage_best_target_distance = None
        self.post_stage_last_target_progress_time = None
        # Doors/1 holds station at waypoint 8 for 134 s while the route commands
        # .30 m/s forward with no brake active.  The env already reports wall and
        # door contacts on every physics step and observe_contacts threw all of
        # them away, keeping only plate_*.  Retaining a compact summary is what
        # separates "physically blocked" from "locomotion produced no gait".
        self._last_world_contacts = {}
        self.post_stage_steps = 0
        self.post_stage_contact_steps = 0
        self.post_stage_geom_steps = {}
        self.post_stage_min_distance_m = None
        self.post_stage_first_contact_time_s = None
        if self.rejoin_diagnostic:
            route_action = self.route.action

            def traced_route_action():
                before = self._route_snapshot()
                action = route_action()
                after = self._route_snapshot()
                self.route_action_events.append({
                    "time_s": float(self.env.runner.d.time),
                    "before": before,
                    "after": after,
                    "route_action": np.asarray(action, dtype=float).tolist(),
                })
                return action

            # Observe the existing route call without changing its arguments,
            # return value, or state transitions.
            self.route.action = traced_route_action

    def _route_target(self):
        """Read-only reconstruction of the target selected by route code."""
        env, route, state = self.env, self.route, self.env.state
        xy = env.runner.d.xpos[env.slot.body_id, :2].copy()
        waypoint = int(route.waypoint)
        target = None
        target_kind = None
        door_index = None
        interaction_index = None
        if env.stage == "transport" and not state.acquired:
            limit = env.layout.object_index
            if waypoint < limit:
                target_kind, target = "waypoint", env.layout.xy(env.layout.route[waypoint])
            else:
                target_kind, target = "parcel", env.runner.d.xpos[env.parcel_body, :2].copy()
                interaction_index = "acquire"
        else:
            if waypoint < len(env.layout.route) - 1:
                target_kind, target = "waypoint", env.layout.xy(env.layout.route[waypoint])
            for door, k in enumerate(env.layout.door_indices):
                if waypoint >= k and not state.open[door]:
                    door_index = int(door)
                    interaction_index = "activate"
                    center = env.layout.xy(env.layout.route[k])
                    if np.linalg.norm(xy - center) < .5:
                        target_kind = "plate"
                        target = env.layout.plate(door, env.layout.correct_sides[door])
                    else:
                        target_kind, target = "door_center", center
                    break
            if (env.stage == "transport" and waypoint == len(env.layout.route) - 1
                    and all(state.crossed)):
                goal = env.layout.xy(env.layout.route[-1])
                mount = env.runner.d.xmat[env.slot.body_id].reshape(3, 3) @ [.32, 0, .48]
                target_kind, target = "release", goal - mount[:2]
                interaction_index = "release"
        if target is None:
            target = env.layout.xy(env.layout.route[-1])
            target_kind = "goal"
        target = np.asarray(target, dtype=float)
        delta = target - xy
        yaw = yaw_of(env)
        target_heading = float(np.arctan2(delta[1], delta[0])) if np.linalg.norm(delta) > 1e-9 else yaw
        return {
            "target_kind": target_kind,
            "target_xy": target.tolist(),
            "target_waypoint": waypoint,
            "door_index": door_index,
            "interaction_index": interaction_index,
            "route_segment_index": waypoint,
            "planner_goal": target_kind,
            "expected_next_state": ("stage_" + self.stage.phase
                                     if self.stage.phase != "recorded" else env.phase),
            "target_distance_m": float(np.linalg.norm(delta)),
            "target_heading_error_rad": float(wrap(target_heading - yaw)),
            "target_behind": bool(np.dot(delta, np.array([np.cos(yaw), np.sin(yaw)])) < 0.),
        }

    def _route_snapshot(self):
        """Read-only state needed to diagnose stage-to-route transition."""
        env, route, state = self.env, self.route, self.env.state
        slot = env.slot
        velocity = env.runner.d.qvel[slot.qvel_adr:slot.qvel_adr + 2].astype(float)
        return {
            "time_s": float(env.runner.d.time),
            "active_controller": ("PlateStage:" + self.stage.phase
                                   if self.stage.phase != "recorded"
                                   else "PlateSafeRouteController"),
            "env_phase": env.phase,
            "stage_phase": self.stage.phase,
            "stage_door": None if self.stage.door is None else int(self.stage.door),
            "stage_done": sorted(int(x) for x in self.stage.done),
            "base_xy": env.runner.d.xpos[slot.body_id, :2].astype(float).tolist(),
            "base_yaw_rad": float(yaw_of(env)),
            "velocity_world_mps": velocity.tolist(),
            "linear_speed_mps": float(np.linalg.norm(velocity)),
            "angular_velocity_radps": float(env.runner.d.qvel[slot.qvel_adr + 5]),
            "open": list(state.open),
            "crossed": list(state.crossed),
            "acquired": bool(state.acquired),
            "carrying": bool(state.carrying),
            "failure": state.failure,
            "completed_s": state.completed_s,
            "route_waypoint": int(route.waypoint),
            "route_previous_open": list(route.previous_open),
            "route_previous_carry": bool(route.previous_carry),
            "route_brake_until_s": float(route.brake_until),
            "route_contact_brake_until_s": float(route.contact_brake_until),
            "route_plate_contacts_seen": sorted(set(route.plate_contacts_seen)),
            "route_reset_reinitialize_hook_invoked": False,
            **self._route_target(),
        }

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
        stage_phase_before_action = self.stage.phase
        route_snapshot_before = self._route_snapshot() if self.rejoin_diagnostic else None
        base_action = self.route.action()
        recorded = action_to_physical(base_action)
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
                effective_action = physical_to_action(command, base_action)
            else:
                effective_action = base_action
        else:
            effective_action = base_action
        # Stage exit is tracked unconditionally: it is one timestamp, and
        # gating it on --rejoin-diagnostic meant the intervention could not run
        # without also enabling the tracing that monkey-patches route.action and
        # writes two full state snapshots per step.  Only the heavy record below
        # is diagnostic-only.
        if (stage_phase_before_action != "recorded"
                and self.stage.phase == "recorded"):
            self.post_stage_exit_time = now
            self.post_stage_exit_waypoint = int(self.route.waypoint)
            self.post_stage_last_target_waypoint = None
            self.post_stage_best_target_distance = None
            self.post_stage_last_target_progress_time = None
        if (self.rejoin_diagnostic and stage_phase_before_action != "recorded"
                and self.stage.phase == "recorded"):
            self.stage_exit_events.append({
                "time_s": now,
                "active_controller_before": "PlateStage:" + stage_phase_before_action,
                "active_controller_after": "PlateSafeRouteController",
                "route_state_at_exit": self._route_snapshot(),
                "stage_history": list(self.stage.history),
            })
        target = self._route_target()
        if (self.post_stage_exit_time is not None
                and target["target_kind"] == "waypoint"):
            target_waypoint = int(target["target_waypoint"])
            target_distance = float(target["target_distance_m"])
            if target_waypoint != self.post_stage_last_target_waypoint:
                self.post_stage_last_target_waypoint = target_waypoint
                self.post_stage_best_target_distance = target_distance
                self.post_stage_last_target_progress_time = now
            elif (self.post_stage_best_target_distance is None
                  or target_distance < self.post_stage_best_target_distance - .02):
                self.post_stage_best_target_distance = target_distance
                self.post_stage_last_target_progress_time = now
        if (self.rejoin_fix == "forward_pulse"
                and self.post_stage_exit_time is not None and not self.rejoin_fix_used
                and self.rejoin_fix_until <= now
                and self.post_stage_last_target_progress_time is not None
                and now - self.post_stage_last_target_progress_time >= .8
                and self.route.waypoint > self.post_stage_exit_waypoint
                and self.env.phase == "advance"
                and target["target_kind"] == "waypoint"):
            self.rejoin_fix_used = True
            self.rejoin_fix_until = now + .4
            self.rejoin_fix_events.append({
                "time_s": now,
                "event": "start_forward_restart_pulse",
                "route_waypoint": int(self.route.waypoint),
                "target": target,
                "stalled_for_s": float(now - self.post_stage_last_target_progress_time),
                "best_target_distance_m": self.post_stage_best_target_distance,
            })
        if self.rejoin_fix_until > now:
            without = action_to_physical(effective_action)
            imposed = np.asarray([REJOIN_PULSE_MPS, 0., 0.], dtype=float)
            delta = imposed - without
            self.rejoin_fix_command_delta = max(
                self.rejoin_fix_command_delta, float(np.linalg.norm(delta)))
            self.rejoin_fix_forward_delta = max(
                self.rejoin_fix_forward_delta, float(abs(delta[0])))
            self.rejoin_fix_command_samples.append({
                "time_s": now,
                "command_without_intervention_mps": without.tolist(),
                "command_with_intervention_mps": imposed.tolist(),
                "command_delta_mps": float(np.linalg.norm(delta)),
                "forward_delta_mps": float(abs(delta[0])),
            })
            self.env.phase = "rejoin_recover"
            effective_action = physical_to_action(imposed, base_action)
        if self.rejoin_diagnostic:
            route_snapshot_after = self._route_snapshot()
            self.decision_events.append({
                "time_s": now,
                "active_controller_before": route_snapshot_before["active_controller"],
                "active_controller_after": route_snapshot_after["active_controller"],
                "stage_phase_before": stage_phase_before_action,
                "stage_phase_after": self.stage.phase,
                "route_state_before": route_snapshot_before,
                "route_state_after": route_snapshot_after,
                "effective_action": np.asarray(effective_action, dtype=float).tolist(),
            })
        return effective_action

    def observe_contacts(self, events):
        world = {}
        for event in events:
            if event["world_geom"].startswith("plate_"):
                self.plate_contact_events.append(event)
            name = event["world_geom"]
            distance = float(event["distance_m"])
            if name not in world or distance < world[name]:
                world[name] = distance
        self._last_world_contacts = world

    def observe_step(self, action):
        now = float(self.env.runner.d.time)
        if self.post_stage_exit_time is not None:
            self.post_stage_steps += 1
            world = self._last_world_contacts
            if world:
                self.post_stage_contact_steps += 1
                if self.post_stage_first_contact_time_s is None:
                    self.post_stage_first_contact_time_s = now
                for name, distance in world.items():
                    self.post_stage_geom_steps[name] = (
                        self.post_stage_geom_steps.get(name, 0) + 1)
                    if (self.post_stage_min_distance_m is None
                            or distance < self.post_stage_min_distance_m):
                        self.post_stage_min_distance_m = distance
        # Clearing the pulse window must not depend on the tracing below, or an
        # intervention run without --rejoin-diagnostic would latch on forever.
        if self.rejoin_fix_until and now >= self.rejoin_fix_until:
            self.rejoin_fix_events.append({
                "time_s": now,
                "event": "end_forward_restart_pulse",
                "route_waypoint": int(self.route.waypoint),
            })
            self.rejoin_fix_until = 0.
        if not self.rejoin_diagnostic:
            return
        sample = self.env.diagnostic_trace[-1] if self.env.diagnostic_trace else None
        self.post_step_events.append({
            "time_s": float(self.env.runner.d.time),
            "state": self._route_snapshot(),
            "effective_action": np.asarray(action, dtype=float).tolist(),
            "physical_sample": sample,
        })


# Per-step traces, measured on the 21399503 episode record: plate_contact_events
# 8.1 MB, rejoin_diagnostic 5.1 MB, diagnostic_trace 1.9 MB out of 15.6 MB.
# result.json used to embed every episode row whole, duplicating the per-episode
# files beside it and reaching 84 MB.  The traces stay in <stage>-<index>.json;
# result.json keeps the verdict, which is what the docs cite and what git keeps.
HEAVY_EPISODE_KEYS = (
    "plate_contact_events", "rejoin_diagnostic", "diagnostic_trace",
    "decision_rewards", "route_phase_history", "trace",
    "guarded_stage_history", "guarded_stage_phase_history",
    "stage_entry_states", "world_contact_samples",
)


def compact_episode(row, record_name):
    summary = {key: value for key, value in row.items()
               if key not in HEAVY_EPISODE_KEYS}
    summary["full_episode_record"] = record_name
    summary["omitted_trace_keys"] = [key for key in HEAVY_EPISODE_KEYS if key in row]
    return summary


def run(args):
    args.out.mkdir(parents=True, exist_ok=True)
    stages = (args.stage,) if args.stage != "both" else ("doors", "transport")
    indices = [int(value) for value in args.indices.split(",") if value.strip()]
    rows = []
    summaries = []
    for stage in stages:
        for index in indices:
            env = DebugEnv(args.repo, args.out / f"{stage}-{index}-cache",
                           stage=stage, split="validation", seed=1000)
            env.reset(index)
            controller = RouteHandoffController(
                env, handoff_mode=args.handoff, rejoin_diagnostic=args.rejoin_diagnostic,
                rejoin_fix=args.rejoin_fix)
            while True:
                env.runner.contact_trace = []
                effective_action = controller.action()
                _, _, done, _ = env.step(effective_action)
                controller.observe_contacts(env.runner.contact_trace)
                controller.observe_step(effective_action)
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
            rejoin_diagnostic = None
            if args.rejoin_diagnostic and controller.stage_exit_events:
                first_exit = controller.stage_exit_events[0]
                window_start = float(first_exit["time_s"]) - 1.0
                rejoin_diagnostic = {
                    "window_start_s": window_start,
                    "window_end_s": float(env.runner.d.time),
                    "stage_exit_events": controller.stage_exit_events,
                    "controller_transition_trace": [
                        event for event in controller.decision_events
                        if event["time_s"] >= window_start
                    ],
                    "route_action_trace": [
                        event for event in controller.route_action_events
                        if event["time_s"] >= window_start
                    ],
                    "post_stage_state_trace": [
                        event for event in controller.post_step_events
                        if event["time_s"] >= window_start
                    ],
                }
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
                post_stage_contact_summary={
                    "stage_exit_time_s": controller.post_stage_exit_time,
                    "stage_exit_waypoint": controller.post_stage_exit_waypoint,
                    "steps_observed": controller.post_stage_steps,
                    "steps_with_world_contact": controller.post_stage_contact_steps,
                    "fraction_steps_in_contact": (
                        controller.post_stage_contact_steps / controller.post_stage_steps
                        if controller.post_stage_steps else None),
                    "distinct_world_geoms": sorted(controller.post_stage_geom_steps),
                    "world_geom_step_counts": controller.post_stage_geom_steps,
                    "min_contact_distance_m": controller.post_stage_min_distance_m,
                    "first_contact_time_s": controller.post_stage_first_contact_time_s,
                },
                rejoin_fix=args.rejoin_fix,
                rejoin_fix_events=controller.rejoin_fix_events,
                rejoin_fix_fired=bool(controller.rejoin_fix_events),
                rejoin_fix_command_samples=controller.rejoin_fix_command_samples,
                rejoin_fix_command_delta_mps=controller.rejoin_fix_command_delta,
                rejoin_fix_forward_delta_mps=controller.rejoin_fix_forward_delta,
                rejoin_fix_changed_forward_command=bool(
                    controller.rejoin_fix_forward_delta > COMMAND_EPSILON_MPS),
                rejoin_diagnostic=rejoin_diagnostic,
            )
            rows.append(row)
            record_name = f"{stage}-{index}.json"
            (args.out / record_name).write_text(json.dumps(row, indent=2) + "\n")
            summaries.append(compact_episode(row, record_name))
    requested = args.rejoin_fix != "none"
    fired = any(row["rejoin_fix_fired"] for row in rows)
    changed_forward = any(row["rejoin_fix_changed_forward_command"] for row in rows)
    changed_any = any(row["rejoin_fix_command_delta_mps"] > COMMAND_EPSILON_MPS
                      for row in rows)
    if not requested:
        intervention_status = "NOT_REQUESTED"
    elif not fired:
        intervention_status = "REQUESTED_BUT_NEVER_FIRED"
    elif not changed_forward:
        intervention_status = "FIRED_BUT_DID_NOT_CHANGE_FORWARD_COMMAND"
    else:
        intervention_status = "APPLIED"
    ineffective = requested and intervention_status != "APPLIED"
    result = {
        "complete": True,
        "status": ("COMPLETED_TARGETED_HANDOFF_PROBE" if not ineffective
                   else "INEFFECTIVE_INTERVENTION"),
        "controller": ("PlateSafeRouteController plus unchanged guarded PlateStage "
                       f"on {args.handoff} handoff"),
        "selected_indices": indices,
        "stages": list(stages),
        "geometry_unchanged": True,
        "activation_semantics_unchanged": True,
        "fall_predicate_unchanged": True,
        "plate_stage_internal_behavior_unchanged": True,
        "handoff_mode": args.handoff,
        "rejoin_fix": args.rejoin_fix,
        "intervention_requested": requested,
        "intervention_fired": fired,
        "intervention_changed_command": changed_any,
        "intervention_changed_forward_command": changed_forward,
        "intervention_status": intervention_status,
        "rejoin_diagnostic": args.rejoin_diagnostic,
        "route_episode_count": len(rows),
        "guarded_stage_activations": sum(row["guarded_stage_activated"] for row in rows),
        "guarded_stage_completions": sum(row["guarded_stage_completed"] for row in rows),
        "episodes_are_summaries": True,
        "episodes": summaries,
    }
    (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "episodes": len(rows),
        "guarded_stage_activations": result["guarded_stage_activations"],
        "guarded_stage_completions": result["guarded_stage_completions"],
        "intervention_status": intervention_status,
        "out": str(args.out),
    }, sort_keys=True), flush=True)
    # A requested intervention that never fired, or that reproduced the command
    # the route was already issuing, is not a result about the intervention.
    # Job 21399502 exited 0 with "rejoin_fix": "forward_pulse" and no pulse, and
    # 21399503 exited 0 with a pulse whose forward speed equalled the route's
    # own.  Both read as clean completions.  Fail loudly instead; the evidence
    # above is already written, so nothing is lost by exiting non-zero.
    if ineffective:
        message = (f"requested intervention {args.rejoin_fix!r} was ineffective: "
                   f"{intervention_status}")
        if args.allow_inactive_intervention:
            print(f"warning: {message} (allowed by --allow-inactive-intervention)",
                  file=sys.stderr, flush=True)
        else:
            raise SystemExit(f"error: {message}\n"
                             f"  evidence retained at {args.out}\n"
                             f"  pass --allow-inactive-intervention to record this "
                             f"as an intentional null result")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--stage", choices=("doors", "transport", "both"), required=True)
    parser.add_argument("--indices", required=True, help="comma-separated validation layout indices")
    parser.add_argument("--handoff", choices=("switch", "early"), default="switch")
    parser.add_argument("--rejoin-diagnostic", action="store_true")
    parser.add_argument("--rejoin-fix", choices=("none", "forward_pulse"), default="none")
    parser.add_argument("--allow-inactive-intervention", action="store_true",
                        help="record a requested-but-ineffective intervention as a "
                             "null result instead of failing the run")
    parser.add_argument("--preflight", action="store_true",
                        help="validate interpreter, imports and arguments, then exit "
                             "without running any episode")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.out = args.out.resolve()
    if not args.out.is_relative_to(args.repo):
        parser.error("output must remain inside repository")
    if args.preflight:
        # Reaching this line already proves the three things that killed
        # 21396684, 21397985, 21399179, 21399201 and 21398501: this interpreter
        # has numpy and mujoco, the source snapshot is import-closed (the
        # module-level imports above resolved), argparse accepted every
        # forwarded argument, and the output path validated.  Seconds here, in
        # place of hours of queue wait followed by an immediate crash.
        import mujoco
        print(json.dumps({
            "status": "PREFLIGHT_OK",
            "python": sys.executable,
            "numpy": np.__version__,
            "mujoco": mujoco.__version__,
            "stage": args.stage,
            "indices": args.indices,
            "handoff": args.handoff,
            "rejoin_fix": args.rejoin_fix,
            "rejoin_diagnostic": args.rejoin_diagnostic,
            "out": str(args.out),
        }, sort_keys=True), flush=True)
        raise SystemExit(0)
    run(args)
