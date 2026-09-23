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


def _yaw(env):
    q = env.runner.d.qpos[env.slot.qpos_adr + 3:env.slot.qpos_adr + 7]
    return float(np.arctan2(2 * (q[0] * q[3] + q[1] * q[2]), 1 - 2 * (q[2] ** 2 + q[3] ** 2)))


def _plate_state(env, door, side):
    center, direction = env.layout.door(door)
    plate = env.layout.plate(door, side)
    direction = np.asarray(direction, dtype=float)
    return np.asarray(plate), direction


class PlateStage:
    """One bounded staged maneuver per unopened correct plate."""

    def __init__(self, env, approach_radius=.78, settle_s=.40, cross_s=1.20,
                 stage_lateral_m=None, wait_open_s=0., press_hold=False,
                 creep_mps=.15, creep_max_m=.45, pre_point_m=.30, cross_clear_m=None,
                 cross_max_s=4.0, cross_kick=False, align_yaw=False, align_tol_rad=.20,
                 align_max_s=1.5):
        self.env = env
        self.approach_radius = float(approach_radius)
        self.settle_s = float(settle_s)
        self.cross_s = float(cross_s)
        # Where the body centres for the crossing, as a lateral offset from the
        # door centreline toward the plate side.  None keeps the exact replay
        # behaviour: the plate centre itself, 0.42 m off the centreline, which
        # puts the body 0.29-0.39 m from the corridor wall against a 0.316 m
        # half-body.  The plate is 0.24 m in radius and is pressed by a foot,
        # not by the body centre, so a smaller offset keeps the body clear of
        # the wall while the wall-side foot still lands on the plate.  Capture
        # (approach_radius from the plate) and the fall predicate are unchanged.
        self.stage_lateral_m = None if stage_lateral_m is None else float(stage_lateral_m)
        # After the fixed settle, optionally keep standing on the plate for up
        # to wait_open_s more until the door is open, instead of crossing into
        # a closed door.  0 keeps the exact replay behaviour.
        self.wait_open_s = float(wait_open_s)
        self.wait_open_until = None
        # Press-and-hold: after the settle, creep along the door direction at
        # creep_mps until the plate registers a press (door open) or creep_max_m
        # has been covered, hold still with activate asserted for up to
        # wait_open_s, then cross.  The default stage waits at the pre-point
        # 0.30 m before the plate, where nothing presses it: 8 of 9 F1+F2+F3
        # in-stage terminations were crossings into a still-closed door.
        self.press_hold = bool(press_hold)
        self.creep_mps = float(creep_mps)
        self.creep_max_m = float(creep_max_m)
        self.creep_origin = None
        # Where the settle point sits before the plate along the door direction.
        # 0.30 m is the replay value; the baseline trace shows the feet already
        # on the plate edge there, and approach-phase trips (doors/13, 5, 7)
        # walking onto the raised plate from the side.
        self.pre_point_m = float(pre_point_m)
        # Cross until the body is this far PAST the plate centre along the door
        # direction (None = the replay's fixed 1.20 s).  From the 0.40 s
        # standstill the gait covers only ~0.13 m in 1.20 s, so the replay
        # crossing ends with the base 3 cm short of the plate edge and the feet
        # on it -- which is where the route resumes with full lateral and every
        # one of the eight post-exit falls in Campaign A tripped on the plate.
        self.cross_clear_m = None if cross_clear_m is None else float(cross_clear_m)
        self.cross_max_s = float(cross_max_s)
        self.cross_start_s = None
        # After the settle standstill the gait restarts for ~0.2 m at the
        # 0.30 m/s command and then stands on the plate edge: the feedback fixed
        # point.  A stage-owned one-shot prev_actions := 0 at the start of the
        # crossing is the intervention with matched evidence of un-sticking it
        # (2/2 genuine stalls).  Recorded in history; the replay gate must pass.
        self.cross_kick = bool(cross_kick)
        self.kick_events = []
        # The crossing is a world-frame vector with no yaw correction, so a
        # robot that reached the pre-point sideways crosses sideways: in the
        # three V2 replay falls the body command was [0.01, 0.30, 0.0] for two
        # seconds before the plate-edge trip.  Turn in place toward the door
        # direction during the settle (bounded), so the crossing is forward.
        self.align_yaw = bool(align_yaw)
        self.align_tol_rad = float(align_tol_rad)
        self.align_max_s = float(align_max_s)
        self.align_until = None
        self.align_events = []
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

    def _kick(self, now):
        if not self.cross_kick:
            return
        controller = self.env.controller
        before = float(np.linalg.norm(controller.prev_actions))
        controller.prev_actions[:] = 0.
        self.kick_events.append({"time_s": float(now), "door": int(self.door),
                                 "prev_actions_before_norm": before, "event": "cross_kick"})
        self.history.append({"time_s": float(now), "door": int(self.door), "phase": "cross_kick",
                             "prev_actions_before_norm": before})

    def _start(self, door, side, now):
        self.door = door
        self.side = side
        self.align_until = None
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
        if self.stage_lateral_m is not None:
            # Anchor on the same cell the plate is anchored on.  layout.plate()
            # is xy(route[k]) + lateral*side*.42 - direction*.12; the first
            # version of this offset used layout.door()'s centre, half a cell
            # further along, and staged the robot ~0.75 m past the plate.
            k = env.layout.door_indices[self.door]
            anchor = np.asarray(env.layout.xy(env.layout.route[k]), dtype=float)
            lateral = np.array([-direction[1], direction[0]])
            plate = anchor + lateral * self.side * self.stage_lateral_m - direction * .12
        if self.phase == "approach":
            pre = plate - direction * self.pre_point_m
            if np.linalg.norm(xy - pre) <= .13:
                self.phase = "settle"
                self.phase_until = now + self.settle_s
                self.history.append({"time_s": now, "door": int(self.door),
                                     "side": int(self.side), "phase": self.phase})
                return np.zeros(3), self.phase
            return _world_command(env, pre - xy), self.phase
        if self.phase == "settle":
            if self.align_yaw:
                if self.align_until is None:
                    self.align_until = now + self.align_max_s
                err = wrap(np.arctan2(direction[1], direction[0]) - _yaw(env))
                if abs(err) > self.align_tol_rad and now < self.align_until:
                    return np.array([0., 0., float(np.clip(1.2 * err, -.35, .35))]), self.phase
                if not self.align_events or self.align_events[-1]["door"] != int(self.door):
                    self.align_events.append({"time_s": now, "door": int(self.door), "yaw_error_rad": float(err),
                                              "aligned": bool(abs(err) <= self.align_tol_rad)})
                if now < self.phase_until:
                    return np.zeros(3), self.phase
            elif now < self.phase_until:
                return np.zeros(3), self.phase
            if self.press_hold and not env.state.open[self.door]:
                self.phase = "creep"
                self.creep_origin = xy.copy()
                self.history.append({"time_s": now, "door": int(self.door),
                                     "side": int(self.side), "phase": self.phase})
            elif self.wait_open_s > 0. and not env.state.open[self.door]:
                if self.wait_open_until is None:
                    self.wait_open_until = now + self.wait_open_s
                    self.history.append({"time_s": now, "door": int(self.door),
                                         "side": int(self.side), "phase": "wait_open"})
                if now < self.wait_open_until:
                    return np.zeros(3), self.phase
            self.wait_open_until = None
            self.phase = "cross"
            self.phase_until = now + self.cross_s
            self.cross_start_s = now
            self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase})
            self._kick(now)
        if self.phase == "creep":
            travelled = float(np.linalg.norm(xy - self.creep_origin))
            if env.state.open[self.door] or travelled >= self.creep_max_m:
                self.phase = "hold"
                self.phase_until = now + self.wait_open_s
                self.history.append({"time_s": now, "door": int(self.door),
                                     "side": int(self.side), "phase": self.phase,
                                     "creep_m": travelled, "open": bool(env.state.open[self.door])})
                return np.zeros(3), self.phase
            return _world_command(env, direction, speed=self.creep_mps), self.phase
        if self.phase == "hold":
            if not env.state.open[self.door] and now < self.phase_until:
                return np.zeros(3), self.phase
            self.phase = "cross"
            self.phase_until = now + self.cross_s
            self.cross_start_s = now
            self.history.append({"time_s": now, "door": int(self.door), "phase": self.phase,
                                 "open": bool(env.state.open[self.door])})
            self._kick(now)
        if self.phase == "cross":
            # The short crossing deliberately carries no yaw correction.
            if self.cross_clear_m is not None:
                along = float((xy - _plate_state(env, self.door, self.side)[0]) @ direction)
                keep = along < self.cross_clear_m and now < self.cross_start_s + self.cross_max_s
            else:
                keep = now < self.phase_until
            if keep:
                return _world_command(env, direction), self.phase
            self.done.add(self.door)
            self.history.append({"time_s": now, "door": int(self.door),
                                 "side": int(self.side), "phase": "recorded"})
            self.door = None
            self.side = None
            self.phase = "recorded"
            return np.asarray(recorded, dtype=float), "recorded"
        raise AssertionError(self.phase)


def _episode(env, index, source_episode, baseline, stage_lateral_m=None, wait_open_s=0., press_hold=False,
             pre_point_m=.30, cross_clear_m=None, settle_s=.40, cross_kick=False, align_yaw=False):
    runner = env.runner
    stage = PlateStage(env, stage_lateral_m=stage_lateral_m, wait_open_s=wait_open_s, press_hold=press_hold,
                       pre_point_m=pre_point_m, cross_clear_m=cross_clear_m, settle_s=settle_s, cross_kick=cross_kick,
                       align_yaw=align_yaw)
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
    only = getattr(args, "only", "")
    if only:
        keep = {int(x) for x in only.split(",") if x.strip()}
        selected = [(i, e) for i, e in selected if i in keep]
    env = DebugEnv(args.repo, out / "cache", stage="doors", split="validation", seed=1000)
    rows, next_reset = [], 0
    for index, episode in selected:
        for reset_index in range(next_reset, index + 1):
            env.reset(reset_index)
        next_reset = index + 1
        rows.append(_episode(env, index, episode, baseline_rows[index],
                             stage_lateral_m=getattr(args, 'stage_lateral', None),
                             wait_open_s=getattr(args, 'wait_open', 0.),
                             press_hold=getattr(args, 'press_hold', False),
                             pre_point_m=getattr(args, 'pre_point', .30),
                             cross_clear_m=getattr(args, 'cross_clear', None),
                             settle_s=getattr(args, 'settle_s', .40),
                             cross_kick=getattr(args, 'cross_kick', False),
                             align_yaw=getattr(args, 'align_yaw', False)))
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
    parser.add_argument("--stage-lateral", type=float, default=None,
                        help="PlateStage body-centre lateral offset (m); default None = plate "
                             "centre, the exact 10/10 replay behaviour")
    parser.add_argument("--wait-open", type=float, default=0.,
                        help="PlateStage bounded wait on the plate for the door before crossing (s); 0 = replay behaviour")
    parser.add_argument("--press-hold", action="store_true",
                        help="creep onto the plate after the settle and hold until the door opens")
    parser.add_argument("--pre-point", type=float, default=.30,
                        help="settle point distance before the plate along the door direction (m)")
    parser.add_argument("--cross-clear", type=float, default=None,
                        help="cross until the body is this far past the plate centre (m); default fixed 1.20 s")
    parser.add_argument("--settle-s", type=float, default=.40, help="settle standstill before the crossing (s); replay 0.40")
    parser.add_argument("--cross-kick", action="store_true", help="zero prev_actions once at the start of the crossing")
    parser.add_argument("--align-yaw", action="store_true", help="turn in place toward the door direction during the settle")
    parser.add_argument("--only", default="", help="comma-separated layout indices to replay (local checks); default all ten")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args()
    args.repo = args.repo.resolve()
    args.campaign = args.campaign.resolve()
    args.baseline = args.baseline.resolve()
    args.out = args.out.resolve()
    if args.preflight:
        import sys
        print(json.dumps({"status": "PREFLIGHT_OK", "python": sys.executable, "mujoco": mujoco.__version__,
                          "stage_lateral_m": args.stage_lateral, "wait_open_s": args.wait_open,
                          "pre_point_m": args.pre_point, "cross_clear_m": args.cross_clear,
                          "settle_s": args.settle_s, "cross_kick": args.cross_kick, "align_yaw": args.align_yaw, "campaign": str(args.campaign),
                          "baseline": str(args.baseline), "out": str(args.out)}, sort_keys=True), flush=True)
        raise SystemExit(0)
    args.out.mkdir(parents=True, exist_ok=True)
    run(args, args.out)
    print("MISSION7_PLATE_STAGE_COMPLETE", flush=True)
