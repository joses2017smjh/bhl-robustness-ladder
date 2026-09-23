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
import gc
import hashlib
import json
from pathlib import Path
import sys

import mujoco

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

# Gait-policy observation layout (deploy.yaml: num_observations 75, history 0):
# command(3) ang_vel(3) gravity(3) joint_pos(22) joint_vel(22) prev_actions(22).
OBS_CMD = slice(0, 3)
OBS_JPOS = slice(9, 31)
OBS_JVEL = slice(31, 53)
OBS_PREV = slice(53, 75)

# Mechanism classification thresholds, declared before Campaign A.  Raw
# statistics are always reported beside the label so an episode can be
# reclassified without rerunning it.
#
# Progress is tested first.  On the first instrumented Doors/1 recovery the
# post-stage window walked 4.7 m at 0.36 m/s with a mean |target - joint| of
# 1.26 rad: with kp 10 and a 4 N.m effort limit this gait drives joints through
# the PD as a force command, and targets are never reached.  An absolute
# tracking threshold therefore labels normal walking as poor tracking.  Poor
# tracking is instead judged against the same episode's own walking reference
# window, taken while it approached the plate.
FROZEN_TARGET_RANGE_RAD = .02     # absolute fallback when no walking reference exists
FROZEN_TARGET_RATIO = .05         # stall target range / walking-reference range below this: frozen
POOR_TRACKING_RATIO = 2.0         # stall tracking error / walking-reference error above this
STALL_PROGRESS_M = .30            # base displacement over the window below this: no progress
SLIP_PATH_FRACTION = .50          # stance-foot slip / base path above this: feet slipping
STALL_WINDOW_S = 20.              # final window analysed for a post-stage timeout


def summarize_chain(chain, t0, t1, defaults, effort, reference=None):
    """Statistics of the command-to-motion chain over ticks with t0 <= t <= t1.

    ``reference`` is the same episode's walking-reference summary; tracking is
    judged as a ratio against it rather than against an absolute threshold.
    """
    t = np.asarray(chain["t"], dtype=float)
    keep = (t >= t0) & (t <= t1)
    if keep.sum() < 25:
        return {"window_s": [t0, t1], "ticks": int(keep.sum()), "mechanism": "window_too_short"}
    idx = np.flatnonzero(keep)
    tgt = np.asarray(chain["tgt_out"], dtype=float)[idx]
    jpos_rel = np.asarray(chain["jpos_in"], dtype=float)[idx]
    jpos_abs = jpos_rel + np.asarray(defaults, dtype=float)
    raw = np.asarray(chain["raw_out"], dtype=float)[idx]
    prev = np.asarray(chain["prev_in"], dtype=float)[idx]
    ctrl = np.asarray(chain["ctrl_pre"], dtype=float)[idx]
    base = np.asarray(chain["base"], dtype=float)[idx]
    fl = np.asarray(chain["foot_l"], dtype=float)[idx]
    fr = np.asarray(chain["foot_r"], dtype=float)[idx]
    # target at tick k is tracked by the joint state observed at tick k+1
    track = np.abs(tgt[:-1] - jpos_abs[1:])
    slip = 0.
    for foot in (fl, fr):
        both = (foot[1:, 0] > .5) & (foot[:-1, 0] > .5)
        slip += float(np.linalg.norm(foot[1:, 1:3] - foot[:-1, 1:3], axis=1)[both].sum())
    path = float(np.linalg.norm(np.diff(base[:, :2], axis=0), axis=1).sum())
    progress = float(np.linalg.norm(base[-1, :2] - base[0, :2]))
    stats = {
        "window_s": [float(t[idx[0]]), float(t[idx[-1]])],
        "ticks": int(len(idx)),
        "target_range_mean_rad": float((tgt.max(0) - tgt.min(0)).mean()),
        "target_std_mean_rad": float(tgt.std(0).mean()),
        "raw_out_std_mean": float(raw.std(0).mean()),
        "prev_in_step_change_mean": float(np.linalg.norm(np.diff(prev, axis=0), axis=1).mean()),
        "tracking_err_mean_rad": float(track.mean()),
        "tracking_err_max_rad": float(track.max()),
        "ctrl_saturation_frac": float((np.abs(ctrl) >= .99 * np.asarray(effort, dtype=float)).mean()),
        "ctrl_abs_mean_nm": float(np.abs(ctrl).mean()),
        "foot_contact_any_frac": float(((fl[:, 0] > .5) | (fr[:, 0] > .5)).mean()),
        "double_support_frac": float(((fl[:, 0] > .5) & (fr[:, 0] > .5)).mean()),
        "stance_slip_m": slip,
        "base_path_m": path,
        "base_progress_m": progress,
        "base_speed_mean_mps": float(np.linalg.norm(base[:, 2:4], axis=1).mean()),
        "cmd_in_mean_mps": np.asarray(chain["cmd_in"], dtype=float)[idx].mean(0).tolist(),
    }
    ref_err = (reference or {}).get("tracking_err_mean_rad")
    stats["tracking_err_ratio_vs_reference"] = (
        None if not ref_err else float(stats["tracking_err_mean_rad"] / ref_err))
    stats["slip_path_fraction"] = float(slip / path) if path > 1e-6 else None
    ref_range = (reference or {}).get("target_range_mean_rad")
    stats["target_range_ratio_vs_reference"] = (
        None if not ref_range else float(stats["target_range_mean_rad"] / ref_range))
    frozen = (stats["target_range_ratio_vs_reference"] < FROZEN_TARGET_RATIO
              if stats["target_range_ratio_vs_reference"] is not None
              else stats["target_range_mean_rad"] < FROZEN_TARGET_RANGE_RAD)
    if progress >= STALL_PROGRESS_M:
        mechanism = "progressing"
    elif frozen:
        mechanism = "frozen_targets"
    elif (stats["tracking_err_ratio_vs_reference"] is not None
          and stats["tracking_err_ratio_vs_reference"] > POOR_TRACKING_RATIO):
        mechanism = "poor_joint_tracking"
    elif stats["slip_path_fraction"] is not None and stats["slip_path_fraction"] > SLIP_PATH_FRACTION:
        mechanism = "cycling_without_progress_slip"
    else:
        mechanism = "cycling_without_progress_in_place"
    stats["mechanism"] = mechanism
    return stats


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
                 rejoin_fix="none", chain_trace=False, stage_lateral_m=None,
                 stage_activate=False, exit_ramp_s=0.):
        self.env = env
        self.handoff_mode = handoff_mode
        self.rejoin_diagnostic = bool(rejoin_diagnostic)
        self.rejoin_fix = rejoin_fix
        self.chain_trace = bool(chain_trace)
        self.stage_lateral_m = stage_lateral_m
        # F2: the stage asserts the activate flag while it holds the robot on
        # the plate, and waits (bounded) for the door before crossing.  The
        # route controller only raises activate in its own 'switch' phase, so a
        # stage-placed press otherwise never registers (doors/15: 5,180 plate
        # contacts, no activation, crossing into a closed door).
        self.stage_activate = bool(stage_activate)
        # F3: after the crossing, hold a forward-only 0.30 m/s command along
        # the door direction for exit_ramp_s before the route resumes.  Ten of
        # 29 Campaign A episodes fell 0.2-1.3 s after exit while the route
        # commanded up to 0.35 m/s lateral, the gait's weakest axis.
        self.exit_ramp_s = float(exit_ramp_s)
        self.exit_ramp_until = 0.
        self.exit_ramp_direction = None
        self.exit_ramp_events = []
        self.route = PlateSafeRouteController(env, contact_hold_s=1.0)
        self.stage = PlateStage(env, stage_lateral_m=stage_lateral_m,
                                wait_open_s=2.0 if stage_activate else 0.)
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
        # The stall condition is tracked for every run, intervention or not: it
        # is the branch point Campaign B reruns to, and its fingerprint is what
        # a paired comparison must match before the arms are compared.
        self.stall_first_time = None
        self.stall_fingerprint = None
        self.stall_event = None
        # prev_actions reset delivery: the next gait update's observation must
        # carry zeros in its prev_actions slice, and only that update can show it.
        self._verify_prev_input = False
        self.prev_reset_delivery = None
        # Command-to-motion chain, one row per 25 Hz gait update.
        self.chain = {key: [] for key in ("t", "cmd_in", "prev_in", "jpos_in", "jvel_in",
                                          "raw_out", "tgt_out", "ctrl_pre", "foot_l",
                                          "foot_r", "base", "phase")}
        model, slot = env.model, env.slot
        self.floor_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor")
        self.foot = {}
        for side in ("left", "right"):
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY,
                                     f"{slot.prefix}leg_{side}_ankle_roll")
            geoms = {int(g) for g in np.flatnonzero(model.geom_bodyid == body)}
            self.foot[side] = (int(body), geoms)
        controller_update = env.controller.update

        def traced_update(robot_observations):
            pre = self._chain_pre() if self.chain_trace else None
            targets = controller_update(robot_observations)
            controller = self.env.controller
            if self._verify_prev_input:
                delivered = controller.policy_observations[0, OBS_PREV].copy()
                self.prev_reset_delivery = {
                    "time_s": float(self.env.runner.d.time),
                    "policy_input_prev_actions": delivered.tolist(),
                    "policy_input_prev_actions_norm": float(np.linalg.norm(delivered)),
                    "delivered_zero_input": bool(np.abs(delivered).max() < 1e-9),
                }
                self._verify_prev_input = False
            if pre is not None:
                obs = controller.policy_observations[0]
                self.chain["t"].append(pre["t"])
                self.chain["cmd_in"].append(np.round(obs[OBS_CMD], 4).tolist())
                self.chain["prev_in"].append(np.round(obs[OBS_PREV], 4).tolist())
                self.chain["jpos_in"].append(np.round(obs[OBS_JPOS], 4).tolist())
                self.chain["jvel_in"].append(np.round(obs[OBS_JVEL], 4).tolist())
                self.chain["raw_out"].append(np.round(controller.policy_actions[0], 4).tolist())
                self.chain["tgt_out"].append(np.round(np.asarray(targets, dtype=float), 4).tolist())
                self.chain["ctrl_pre"].append(pre["ctrl"])
                self.chain["foot_l"].append(pre["foot_l"])
                self.chain["foot_r"].append(pre["foot_r"])
                self.chain["base"].append(pre["base"])
                self.chain["phase"].append(self.env.phase)
            return targets

        # Wrap the gait controller rather than the environment: every one of the
        # five 25 Hz updates inside an env.step is seen, with the exact 75-vector
        # the ONNX policy consumed.  The controller is rebuilt on env.reset, so
        # this is installed after reset, per episode.
        env.controller.update = traced_update
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

    def _foot_state(self, side):
        d, m = self.env.runner.d, self.env.model
        body, geoms = self.foot[side]
        contact = 0.
        for i in range(d.ncon):
            c = d.contact[i]
            g1, g2 = int(c.geom1), int(c.geom2)
            if (g1 == self.floor_geom and g2 in geoms) or (g2 == self.floor_geom and g1 in geoms):
                contact = 1.
                break
        xyz = d.xpos[body]
        return [contact, round(float(xyz[0]), 4), round(float(xyz[1]), 4), round(float(xyz[2]), 4)]

    def _chain_pre(self):
        """Physical state at the instant a gait update is called (= end of the
        previous 80-substep window): applied torque, feet, base."""
        d, s = self.env.runner.d, self.env.slot
        base = d.xpos[s.body_id, :2]
        vel = d.qvel[s.qvel_adr:s.qvel_adr + 2]
        return {
            "t": float(d.time),
            "ctrl": np.round(d.ctrl[s.ctrl], 3).tolist(),
            "foot_l": self._foot_state("left"),
            "foot_r": self._foot_state("right"),
            "base": [round(float(base[0]), 4), round(float(base[1]), 4),
                     round(float(vel[0]), 4), round(float(vel[1]), 4),
                     round(float(yaw_of(self.env)), 4)],
        }

    def _fingerprint(self):
        d = self.env.runner.d
        blob = np.concatenate([[d.time], d.qpos, d.qvel,
                               self.env.controller.prev_actions]).astype(np.float64)
        return {
            "sha256_16": hashlib.sha256(blob.tobytes()).hexdigest()[:16],
            "time_s": float(d.time),
            "base_xy": d.xpos[self.env.slot.body_id, :2].astype(float).tolist(),
            "prev_actions_norm": float(np.linalg.norm(self.env.controller.prev_actions)),
            "route_waypoint": int(self.route.waypoint),
        }

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
                if self.stage_activate:
                    # Same value the route uses in its switch phase.
                    effective_action[3] = 2.
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
            if self.exit_ramp_s > 0.:
                _, direction = self.env.layout.door(
                    int(self.stage.history[-1]["door"]))
                self.exit_ramp_direction = np.asarray(direction, dtype=float)
                self.exit_ramp_until = now + self.exit_ramp_s
                self.exit_ramp_events.append({"time_s": now, "event": "exit_ramp_start",
                                              "until_s": self.exit_ramp_until})
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
        stall_now = (
            self.post_stage_exit_time is not None
            and self.post_stage_last_target_progress_time is not None
            and now - self.post_stage_last_target_progress_time >= .8
            and self.route.waypoint > self.post_stage_exit_waypoint
            and self.env.phase == "advance"
            and target["target_kind"] == "waypoint")
        if stall_now and self.stall_first_time is None:
            self.stall_first_time = now
            self.stall_fingerprint = self._fingerprint()
            self.stall_event = {
                "time_s": now,
                "route_waypoint": int(self.route.waypoint),
                "target": target,
                "stalled_for_s": float(now - self.post_stage_last_target_progress_time),
                "best_target_distance_m": self.post_stage_best_target_distance,
                "fingerprint": self.stall_fingerprint,
            }
        if (self.rejoin_fix != "none" and stall_now and not self.rejoin_fix_used
                and self.rejoin_fix_until <= now):
            self.rejoin_fix_used = True
            event = {**self.stall_event, "time_s": now,
                     "fingerprint_at_fire": self._fingerprint()}
            if self.rejoin_fix == "forward_pulse":
                self.rejoin_fix_until = now + .4
                event["event"] = "start_forward_restart_pulse"
            elif self.rejoin_fix == "prev_actions_reset":
                controller = self.env.controller
                before = controller.prev_actions.copy()
                controller.prev_actions[:] = 0.
                self._verify_prev_input = True
                event.update({
                    "event": "prev_actions_reset",
                    "prev_actions_before": before.tolist(),
                    "prev_actions_before_norm": float(np.linalg.norm(before)),
                    "prev_actions_after_norm": float(np.linalg.norm(controller.prev_actions)),
                    "target_changed": bool(np.linalg.norm(before) > 1e-9),
                })
            self.rejoin_fix_events.append(event)
        if self.exit_ramp_until > now and self.exit_ramp_direction is not None:
            yaw = yaw_of(self.env)
            body = np.array([[np.cos(yaw), np.sin(yaw)],
                             [-np.sin(yaw), np.cos(yaw)]]) @ (self.exit_ramp_direction * .30)
            self.env.phase = "exit_ramp"
            effective_action = physical_to_action([np.clip(body[0], 0., .4), 0., 0.], base_action)
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
    "stage_entry_states", "world_contact_samples", "chain",
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
                rejoin_fix=args.rejoin_fix, chain_trace=args.chain_trace,
                stage_lateral_m=args.stage_lateral, stage_activate=args.stage_activate,
                exit_ramp_s=args.exit_ramp)
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
            end_time = float(env.runner.d.time)
            chain_post_stage = None
            chain_stall_window = None
            chain_reference = None
            if args.chain_trace:
                defaults = env.controller.default_joint_positions
                effort = env.runner.eff
                # Walking reference: from settle-out until just before the stage
                # took over (or the whole episode if it never did).
                ref_end = ((activation_time - .5) if activation_time is not None
                           else end_time)
                chain_reference = summarize_chain(controller.chain, 1.0, ref_end,
                                                  defaults, effort)
            if args.chain_trace and controller.post_stage_exit_time is not None:
                chain_post_stage = summarize_chain(
                    controller.chain, controller.post_stage_exit_time + 1.0, end_time,
                    defaults, effort, reference=chain_reference)
                if failure_phase == "after PlateStage" and row.get("timeout"):
                    chain_stall_window = summarize_chain(
                        controller.chain, max(controller.post_stage_exit_time + 1.0,
                                              end_time - STALL_WINDOW_S),
                        end_time, defaults, effort, reference=chain_reference)
            exposure = ("exposed" if controller.stall_first_time is not None
                        else "not_exposed")
            fix_event = controller.rejoin_fix_events[0] if controller.rejoin_fix_events else None
            if args.rejoin_fix == "none":
                delivery = "NOT_REQUESTED"
            elif exposure == "not_exposed":
                delivery = "NOT_EXPOSED"
            elif fix_event is None:
                delivery = "EXPOSED_BUT_NOT_EXECUTED"
            elif args.rejoin_fix == "forward_pulse":
                delivery = ("APPLIED" if controller.rejoin_fix_forward_delta > COMMAND_EPSILON_MPS
                            else "EXECUTED_BUT_TARGET_UNCHANGED")
            elif args.rejoin_fix == "prev_actions_reset":
                if not fix_event.get("target_changed"):
                    delivery = "EXECUTED_BUT_TARGET_UNCHANGED"
                elif not (controller.prev_reset_delivery or {}).get("delivered_zero_input"):
                    delivery = "EXECUTED_BUT_NOT_DELIVERED_TO_POLICY"
                else:
                    delivery = "APPLIED"
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
                stage_lateral_m=args.stage_lateral,
                stage_activate=args.stage_activate,
                exit_ramp_s=args.exit_ramp,
                exit_ramp_events=controller.exit_ramp_events,
                controller=("PlateSafeRouteController plus guarded PlateStage "
                            f"(lateral {'plate centre' if args.stage_lateral is None else f'{args.stage_lateral:.2f} m'}) "
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
                stall_branch_time_s=controller.stall_first_time,
                stall_event=controller.stall_event,
                stall_fingerprint=controller.stall_fingerprint,
                intervention_exposure=exposure,
                intervention_delivery=delivery,
                prev_actions_reset_delivery=controller.prev_reset_delivery,
                chain_walking_reference_summary=chain_reference,
                chain_post_stage_summary=chain_post_stage,
                chain_stall_window_summary=chain_stall_window,
                mechanism=((chain_stall_window or chain_post_stage or {}).get("mechanism")),
                chain=controller.chain if args.chain_trace else None,
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
            record_name = f"{stage}-{index}.json"
            (args.out / record_name).write_text(json.dumps(row, indent=2) + "\n")
            summary = compact_episode(row, record_name)
            summaries.append(summary)
            # Keep only the compact summary in memory.  Job 21400755 (Transport
            # 8-15, chain trace + rejoin diagnostic) ran out of its 12 GB after
            # five episodes because every full row -- chain columns, per-step
            # route snapshots, contact events -- stayed referenced here until
            # the end.  Every aggregate below reads fields the summary keeps.
            rows.append(summary)
            del row, controller
            gc.collect()
    requested = args.rejoin_fix != "none"
    fired = any(row["rejoin_fix_fired"] for row in rows)
    changed_forward = any(row["rejoin_fix_changed_forward_command"] for row in rows)
    changed_any = any(row["rejoin_fix_command_delta_mps"] > COMMAND_EPSILON_MPS
                      for row in rows)
    exposed = [row for row in rows if row["intervention_exposure"] == "exposed"]
    deliveries = sorted({row["intervention_delivery"] for row in rows})
    # Never reaching the stall is a route outcome, not a runner defect: those
    # episodes stay in the route denominator and are marked not exposed.  The
    # run fails only when an EXPOSED episode did not receive its intervention.
    if not requested:
        intervention_status = "NOT_REQUESTED"
    elif not exposed:
        intervention_status = "REQUESTED_BUT_NO_EPISODE_EXPOSED"
    elif all(row["intervention_delivery"] == "APPLIED" for row in exposed):
        intervention_status = "APPLIED"
    else:
        bad = sorted({row["intervention_delivery"] for row in exposed
                      if row["intervention_delivery"] != "APPLIED"})
        intervention_status = "EXPOSED_EPISODES_NOT_APPLIED:" + ",".join(bad)
    ineffective = requested and bool(exposed) and intervention_status != "APPLIED"
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
        "stage_lateral_m": args.stage_lateral,
        "stage_activate": args.stage_activate,
        "exit_ramp_s": args.exit_ramp,
        "rejoin_fix": args.rejoin_fix,
        "intervention_requested": requested,
        "intervention_fired": fired,
        "intervention_changed_command": changed_any,
        "intervention_changed_forward_command": changed_forward,
        "intervention_status": intervention_status,
        "intervention_deliveries": deliveries,
        "episodes_exposed": len(exposed),
        "chain_trace": args.chain_trace,
        "mechanism_counts": {m: sum(1 for row in rows if row["mechanism"] == m)
                             for m in sorted({row["mechanism"] for row in rows}, key=str)},
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
    parser.add_argument("--rejoin-fix", choices=("none", "forward_pulse", "prev_actions_reset"),
                        default="none")
    parser.add_argument("--chain-trace", action="store_true",
                        help="record the command-to-motion chain at every 25 Hz gait update")
    parser.add_argument("--stage-lateral", type=float, default=None,
                        help="PlateStage body-centre lateral offset from the door centreline (m); "
                             "default None = plate centre (0.42 m), the exact-replay behaviour")
    parser.add_argument("--stage-activate", action="store_true",
                        help="stage asserts the activate flag on the plate and waits up to 2 s "
                             "for the door before crossing")
    parser.add_argument("--exit-ramp", type=float, default=0.,
                        help="seconds of forward-only 0.30 m/s along the door direction after "
                             "the crossing before the route resumes (0 = off)")
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
