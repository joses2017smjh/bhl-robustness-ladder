"""Mission 7 plate bench: 64 staged crossings with the TurnBoth-s0 stage gait (M2).

M2 of docs/SOLUTIONS_2026-10-01.md section 3, authorized 2026-10-01 as a
106-episode Mission 7 line (64 bench + 10 exact replay + 32 route gate).
Launcher: slurm/repo20260923/cpu_m7_plate_bench.sbatch (its header carries the
same frozen rule).  LEARNED gaits (the shipped arms-dr1.0-s0 gait and
TurnBoth-s0), SCRIPTED stage (scripts/mission7_plate_stage.py PlateStage with
stage_gait="turnboth", every other parameter at its replay default), ORACLE
layout and plate pose (as the existing stage uses them).

One crossing per TRAINING-split layout L = 0..31 (generate("train", L)) and
door d = 0/1:
  entry heading relative to the door direction  0, +90, -90, 180 deg  for L mod 4 = 0, 1, 2, 3
  plate   round (the correct side, cylinder)  for floor(L/4) mod 2 = 0, square (the wrong side, box) for 1
  entry   standstill (settled at the pre-point) for d XOR (floor(L/8) mod 2) = 0, continuous walking into
          the takeover for 1
a 4 x 2 x 2 grid of 16 cells x 4 crossings, 2 door-0 and 2 door-1 crossings per cell, 16 per heading.

PREDECLARED RULE (frozen; a chain, each step only after the previous PASSES):
- Bench: PASS iff >= 62/64 crossings clear (base along >= +0.35 m past the plate centre along the door
  direction, within 10 s of takeover, by mission7_gates' clear definition), 0 falls (tilt >= 0.78), and
  >= 15/16 clears per heading. Otherwise FAIL; INCOMPLETE if any of the 64 is missing.
- Then the unchanged exact ten-fall replay once with --stage-gait turnboth, through the existing replay-gate
  machinery (complete, 10 episodes, 10 upright, 0 falls).
- Then the route gate as coded by M1: Doors and Transport each >= 16/16 successes on validation layouts 0-15
  (32 episodes).
- M3 (turn while stepping with the shipped gait + a stall watchdog) is predeclared as conditional on a bench
  FAIL: not implemented.

Bench protocol (declared 2026-10-02 before any bench episode; nothing here is tuned on a result):
- Each crossing gets its own DebugEnv(stage="doors", split="train", seed=7000 + 2L + d), reset to layout L
  (joint and spawn noise only).  The base is then placed at the centre of route[k-1] (k = the door's route
  index: the route's previous waypoint, inside the maze and clear of every plate and wall) with the entry
  yaw = door direction + heading.
- The shipped gait settles 3.0 s at zero command (turn_test v2's settled-stand protocol), then runs up to the
  stage's pre-point P (the plate centre minus 0.30 m along the door direction) on a straight world-frame line
  with the stage's own approach law (_world_command, 0.30 m/s) and the route's heading hold (wz =
  clip(1.2 x yaw error, +-0.35)) on the entry yaw.  Arrival = within 0.13 m of P (the stage's own
  approach-to-settle tolerance), checked every decision step.
- walking entry: the stage takes over at arrival, while the robot walks.  standstill entry: zero command for
  3.0 s at arrival, then the stage takes over.  Takeover = PlateStage._start(d, side) on the bench's plate,
  so the square plate is staged too (the stage's own predicate would not pick it within 1.0 m of the round
  one).  From then on PlateStage.command drives the robot until its hand-back; then 2.0 s at zero command on
  the restored shipped gait.
- The stage is polled every 5 gait updates (0.2 s, MissionEnv.repeat, as in the route probe); every command
  passes through the route's action transform (tanh(arctanh(c / [.4, .35, .4])) x [.4, .35, .4]).
- Physics as in the exact replay: runner.step on the unchanged model, doors closed, no MissionState
  termination.  Falls (tilt >= 0.78, any phase, run-up included) end the crossing and count.  Wall/door
  contacts and plate presses (normal force >= 1 N) are recorded and reported, not gated; the route gate
  applies Mission 7's own collision rule.
- A run-up that has not arrived after 20 s, or a stage that has not handed back 60 s after takeover, ends
  the crossing without a clear (a miss, never a pass).
- Run-up geometry, disclosed before the bench: in 17 of the 64 crossings (RUNUP_OVER_OPPOSITE_PLATE) the
  straight run-up line from route[k-1] to P passes 0.114-0.131 m from the centre of the door's OPPOSITE plate
  (radius / half-side 0.24 m), i.e. over it, on the shipped gait before the takeover; none passes within
  0.24 m of the target plate's centre.  A fall there counts as a bench fall (falls at any phase count).  Each
  crossing records both distances (geometry["runup_line_to_plate_centre_m"]) and verdict.json lists the 17
  and every fall's phase, reported and not gated.
- Scoring: mission7_gates.crossing_outcomes on the crossing's own trace (a real clear: along >= 0.35 m before
  hand-back, no fall at or before hand-back), plus the first such sample within 10 s of takeover.

Smoke mode runs named crossings on exploration layouts L >= 32 only (never a scored layout); a smoke is not
part of the 106 episodes and its verdict is never PASS.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import hashlib
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

import mission7_gates as gates
from bhl_robust.mission.layout import SPLITS, generate


RULE = (
    "Bench: PASS iff >= 62/64 crossings clear (base along >= +0.35 m past the plate centre along the door "
    "direction, within 10 s of takeover, by mission7_gates' clear definition), 0 falls (tilt >= 0.78), and "
    ">= 15/16 clears per heading. Otherwise FAIL; INCOMPLETE if any of the 64 is missing.")
CHAIN = (
    "a chain, each step only after the previous PASSES: bench -> the unchanged exact ten-fall replay once with "
    "--stage-gait turnboth (complete, 10 episodes, 10 upright, 0 falls) -> the route gate as coded by M1 (Doors "
    "and Transport each >= 16/16 successes on validation layouts 0-15, 32 episodes); M3 (turn while stepping "
    "with the shipped gait + a stall watchdog) is predeclared as conditional on a bench FAIL")
LABELS = {"gaits": "LEARNED (shipped arms-dr1.0-s0 outside the stage; TurnBoth-s0 during the stage)",
          "stage": "SCRIPTED (PlateStage, stage_gait=turnboth)",
          "layout_and_plate_pose": "ORACLE (as the existing stage uses them)"}

BENCH_SPLIT = "train"
BENCH_LAYOUTS = tuple(range(32))
BENCH_DOORS = (0, 1)
HEADINGS_DEG = (0, 90, -90, 180)          # by L mod 4
PLATES = ("round", "square")              # by floor(L/4) mod 2
ENTRIES = ("standstill", "walking")       # by d XOR (floor(L/8) mod 2)
TOTAL = 64
REQUIRED_CLEARS = 62
REQUIRED_PER_HEADING = 15
PER_HEADING = 16
CLEAR_WINDOW_S = 10.0
CLEAR_ALONG_M = gates.CLEAR_ALONG_M       # 0.35
FALL_TILT = gates.FALL_TILT               # 0.78
TIME_EPS_S = 1e-9

# Protocol constants (module docstring).
STAGE_KWARGS = {"stage_gait": "turnboth"}   # every other PlateStage parameter at its replay default
PRE_POINT_M = .30                           # PlateStage's default pre_point_m (the stage computes P with it)
SETTLE_S = 3.0
RUNUP_SPEED = .30
YAW_HOLD_GAIN = 1.2
YAW_HOLD_MAX = .35
ARRIVE_M = .13
RUNUP_MAX_S = 20.0
STAGE_MAX_S = 60.0
POST_HANDBACK_S = 2.0
DECISION_TICKS = 5
SEED_BASE = 7000
COMMAND_SCALES = np.array([.4, .35, .4])
SMOKE_MIN_LAYOUT = 32
PLATE_HALF_M = .24                          # plate radius (round) / half-side (square), layout.world_xml
# Disclosed geometry (module docstring; asserted in tests/test_mission7_plate_bench.py): the grid crossings
# whose straight run-up line passes within PLATE_HALF_M of the opposite plate's centre.  Reported, not gated.
RUNUP_OVER_OPPOSITE_PLATE = ((3, 0), (3, 1), (4, 1), (5, 0), (5, 1), (7, 0), (7, 1), (13, 0), (16, 0), (18, 1),
                             (22, 1), (25, 0), (26, 0), (28, 1), (29, 0), (29, 1), (30, 0))
# provenance.json's physics sources besides src/bhl_robust/mission/*.py and scripts/mission7*.py: the replay
# snapshot's set (scripts/submit_mission7_plate_stage.py) minus its sbatch, plus this bench's launcher.  The
# launcher's clean-tree check covers the same files.
BENCH_SOURCES = ("src/bhl_robust/__init__.py", "src/bhl_robust/sensor_io.py", "src/bhl_robust/eval/__init__.py",
                 "src/bhl_robust/eval/multi_robot.py", "src/bhl_robust/eval/mjcf_assets.py",
                 "src/bhl_robust/eval/livery.py", "src/bhl_robust/eval/team_sensors.py",
                 "slurm/repo20260923/cpu_m7_plate_bench.sbatch")


# ---- pure: grid, geometry, scoring, verdict ---------------------------------------------------------

def cell_of(layout_index, door):
    """(heading_deg, plate, entry) of crossing (L, d) by the declared grid."""
    layout_index, door = int(layout_index), int(door)
    return (HEADINGS_DEG[layout_index % 4], PLATES[(layout_index // 4) % 2],
            ENTRIES[door ^ ((layout_index // 8) % 2)])


def grid(layouts=BENCH_LAYOUTS):
    rows = []
    for layout_index in layouts:
        for door in BENCH_DOORS:
            heading, plate, entry = cell_of(layout_index, door)
            rows.append({"layout": int(layout_index), "door": int(door), "heading_deg": heading,
                         "plate": plate, "entry": entry})
    return rows


def plate_side(layout, door, plate):
    """The staged side: the correct side for the round plate, the other one for the square plate."""
    correct = int(layout.correct_sides[door])
    return correct if plate == "round" else -correct


def segment_distance(point, a, b):
    """Distance from ``point`` to the segment a-b (pure)."""
    point, a, b = (np.asarray(v, dtype=float) for v in (point, a, b))
    ab = b - a
    t = float(np.clip((point - a) @ ab / (ab @ ab), 0., 1.)) if ab @ ab > 0 else 0.
    return float(np.linalg.norm(point - (a + t * ab)))


def crossing_geometry(layout, door, plate, heading_deg, pre_point_m=PRE_POINT_M):
    side = plate_side(layout, door, plate)
    _, direction = layout.door(door)
    direction = np.asarray(direction, dtype=float)
    plate_xy = np.asarray(layout.plate(door, side), dtype=float)
    pre = plate_xy - direction * pre_point_m          # PlateStage: plate - direction * pre_point_m
    k = int(layout.door_indices[door])
    start = np.asarray(layout.xy(layout.route[k - 1]), dtype=float)
    door_yaw = float(np.arctan2(direction[1], direction[0]))
    entry_yaw = float(np.arctan2(np.sin(door_yaw + np.radians(heading_deg)),
                                 np.cos(door_yaw + np.radians(heading_deg))))
    # Reported, not gated: how close the straight run-up line comes to each plate of this door.
    runup_line = {"target": segment_distance(plate_xy, start, pre),
                  "opposite": segment_distance(layout.plate(door, -side), start, pre)}
    return {"side": side, "direction": direction.tolist(), "plate_xy": plate_xy.tolist(), "pre_point": pre.tolist(),
            "start": start.tolist(), "door_yaw_rad": door_yaw, "entry_yaw_rad": entry_yaw,
            "run_up_m": float(np.linalg.norm(pre - start)),
            "runup_line_to_plate_centre_m": runup_line,
            "runup_over_opposite_plate": bool(runup_line["opposite"] < PLATE_HALF_M),
            "runup_over_target_plate": bool(runup_line["target"] < PLATE_HALF_M)}


def first_clear_time(row, layout, crossing):
    """First trace time at which ``crossing`` (a mission7_gates.crossing_outcomes entry) is cleared.

    The same window and fall cut as crossing_outcomes: samples from the ``cross`` entry to the hand-back,
    up to the first fall when the fall is inside the window.  None when it never clears.
    """
    samples = row["samples"]
    t = np.array([s["time_s"] for s in samples], dtype=float)
    xy = np.array([s["xy"] for s in samples], dtype=float).reshape(-1, 2)
    _, direction = layout.door(crossing["door"])
    plate = np.asarray(layout.plate(crossing["door"], crossing["side"]), dtype=float)
    window = t >= crossing["start_s"] - gates.TIME_EPS_S
    if crossing["handback_s"] is not None:
        window &= t <= crossing["handback_s"] + gates.TIME_EPS_S
    if crossing["fell_in_crossing"]:
        window &= t <= crossing["first_fall_s"] + gates.TIME_EPS_S
    along = (xy - plate) @ np.asarray(direction, dtype=float)
    hit = np.flatnonzero(window & (along >= CLEAR_ALONG_M))
    return float(t[hit[0]]) if hit.size else None


def score_crossing(row, layout, door, takeover_s):
    """Clear within 10 s of takeover by mission7_gates' clear definition, and the fall flag (pure).

    ``row`` holds ``layout_index``, ``layout_seed``, ``stage_history``, ``samples`` and the header fields
    crossing_outcomes cross-checks.  A crossing that never reached ``cross`` or never handed back is not a
    clear; a bench row with more than one staged crossing raises.
    """
    outcomes = gates.crossing_outcomes(row, layout) if row["samples"] else []
    if len(outcomes) > 1:
        raise ValueError(f"layout {row['layout_index']}: {len(outcomes)} staged crossings in one bench crossing")
    fall_s = next((s["time_s"] for s in row["samples"] if s["tilt"] >= FALL_TILT), None)
    result = {"crossing": outcomes[0] if outcomes else None, "fell": fall_s is not None, "first_fall_s": fall_s,
              "takeover_s": takeover_s, "first_clear_s": None, "clear_after_takeover_s": None,
              "real_clear": False, "clear": False}
    if not outcomes or takeover_s is None:
        return result
    crossing = outcomes[0]
    if crossing["door"] != door:
        raise ValueError(f"layout {row['layout_index']}: staged door {crossing['door']}, bench door {door}")
    t_clear = first_clear_time(row, layout, crossing)
    if (t_clear is not None) != crossing["cleared"]:
        raise ValueError("first_clear_time disagrees with mission7_gates.crossing_outcomes")
    result.update(first_clear_s=t_clear, real_clear=bool(crossing["real_clear"]),
                  clear_after_takeover_s=None if t_clear is None else float(t_clear - takeover_s))
    result["clear"] = bool(crossing["real_clear"] and t_clear is not None
                           and t_clear - takeover_s <= CLEAR_WINDOW_S + TIME_EPS_S)
    return result


def bench_verdict(results, expected=None):
    """The frozen bench rule over per-crossing results (pure).

    Each result carries ``layout``, ``door``, ``heading_deg``, ``plate``, ``entry``, ``complete``, ``clear``
    and ``fell``.  INCOMPLETE when any of the expected crossings is missing or incomplete, duplicated, not
    in the grid, or labelled other than the grid labels it; otherwise PASS iff clears >= 62 of 64, 0 falls
    and >= 15/16 clears in every heading; else FAIL.
    """
    expected = grid() if expected is None else expected
    want = {(c["layout"], c["door"]): c for c in expected}
    seen, problems = {}, []
    for r in results:
        key = (int(r["layout"]), int(r["door"]))
        if key in seen:
            problems.append(f"duplicate crossing {key}")
        seen[key] = r
        if key not in want:
            problems.append(f"crossing {key} is not in the bench grid")
        elif any(r.get(f) != want[key][f] for f in ("heading_deg", "plate", "entry")):
            problems.append(f"crossing {key} is labelled {[r.get(f) for f in ('heading_deg', 'plate', 'entry')]}, "
                            f"the grid says {[want[key][f] for f in ('heading_deg', 'plate', 'entry')]}")
    missing = sorted(key for key in want if key not in seen or not seen[key].get("complete"))
    usable = [seen[key] for key in want if key in seen and seen[key].get("complete")]
    clears = sum(bool(r["clear"]) for r in usable)
    falls = sum(bool(r["fell"]) for r in usable)
    per_heading = {str(h): {"clears": sum(bool(r["clear"]) for r in usable if r["heading_deg"] == h),
                            "crossings": sum(1 for r in usable if r["heading_deg"] == h)} for h in HEADINGS_DEG}
    per_cell = {}
    for c in expected:
        name = f"{c['heading_deg']}/{c['plate']}/{c['entry']}"
        cell = per_cell.setdefault(name, {"clears": 0, "crossings": 0, "doors": [0, 0]})
        r = seen.get((c["layout"], c["door"]))
        if r is not None and r.get("complete"):
            cell["crossings"] += 1
            cell["doors"][c["door"]] += 1
            cell["clears"] += int(bool(r["clear"]))
    if missing or problems or len(want) != TOTAL:
        verdict = "INCOMPLETE"
    elif (clears >= REQUIRED_CLEARS and falls == 0
          and all(v["clears"] >= REQUIRED_PER_HEADING for v in per_heading.values())):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "rule": RULE, "clears": clears, "falls": falls, "crossings_scored": len(usable),
            "expected": len(want), "per_heading": per_heading, "per_cell": per_cell,
            "missing": [list(k) for k in missing], "problems": problems,
            "misses": sorted([r["layout"], r["door"]] for r in usable if not r["clear"]),
            "fallen": sorted([r["layout"], r["door"]] for r in usable if r["fell"])}


def parse_crossings(text):
    out = []
    for item in text.split(","):
        if item.strip():
            layout_index, door = item.split(":")
            out.append((int(layout_index), int(door)))
    return out


# ---- simulation -----------------------------------------------------------------------------------

def _place(env, xy, yaw):
    import mujoco
    s, d = env.slot, env.runner.d
    d.qpos[s.qpos_adr:s.qpos_adr + 2] = xy
    d.qpos[s.qpos_adr + 3:s.qpos_adr + 7] = [np.cos(yaw / 2), 0., 0., np.sin(yaw / 2)]
    d.qvel[s.qvel_adr:s.qvel_adr + 6] = 0.
    mujoco.mj_forward(env.model, d)


def _route_transform(command):
    """The route's action path: physical_to_action then MissionEnv.step's tanh x scales."""
    from bhl_robust.mission.approach_debug import command_action
    return np.tanh(command_action(command)[:3]) * COMMAND_SCALES


def run_crossing(repo, cache, spec, stage_kwargs=STAGE_KWARGS):
    """One bench crossing; returns the full row (trace included)."""
    from bhl_robust.mission.approach_debug import DebugEnv, physical_sample, wrap
    from mission7_plate_stage import PlateStage, _world_command, _yaw
    layout_index, door = int(spec["layout"]), int(spec["door"])
    env = DebugEnv(repo, cache, stage="doors", split=BENCH_SPLIT, seed=SEED_BASE + 2 * layout_index + door)
    env.reset(layout_index)
    layout, runner = env.layout, env.runner
    geometry = crossing_geometry(layout, door, spec["plate"], spec["heading_deg"])
    entry_yaw, pre = geometry["entry_yaw_rad"], np.asarray(geometry["pre_point"])
    _place(env, np.asarray(geometry["start"]), entry_yaw)
    stage = PlateStage(env, **stage_kwargs)
    samples, events = [], []
    state, state_s = "bench_settle", float(runner.d.time)
    takeover_s = handback_s = None
    end_reason = None
    fell = False
    while end_reason is None:
        now = float(runner.d.time)
        xy = runner.d.xpos[env.slot.body_id, :2].copy()
        command = np.zeros(3)
        if state == "bench_settle" and now + TIME_EPS_S >= state_s + SETTLE_S:
            state, state_s = "bench_runup", now
            events.append({"time_s": now, "event": "runup_start"})
        if state == "bench_runup":
            if np.linalg.norm(xy - pre) <= ARRIVE_M:
                events.append({"time_s": now, "event": "arrived", "distance_m": float(np.linalg.norm(xy - pre))})
                if spec["entry"] == "walking":
                    state = "takeover"
                else:
                    state, state_s = "bench_standstill", now
            elif now - state_s >= RUNUP_MAX_S:
                end_reason = "runup_timeout"
                break
            else:
                command = _world_command(env, pre - xy, speed=RUNUP_SPEED)
                command[2] = float(np.clip(YAW_HOLD_GAIN * wrap(entry_yaw - _yaw(env)), -YAW_HOLD_MAX, YAW_HOLD_MAX))
        if state == "bench_standstill" and now + TIME_EPS_S >= state_s + SETTLE_S:
            state = "takeover"
        if state == "takeover":
            stage._start(door, geometry["side"], now)
            takeover_s, state = now, "stage"
            events.append({"time_s": now, "event": "takeover",
                           "speed_mps": float(np.linalg.norm(runner.d.qvel[env.slot.qvel_adr:env.slot.qvel_adr + 2])),
                           "distance_to_pre_point_m": float(np.linalg.norm(xy - pre)),
                           "yaw_rad": _yaw(env)})
        if state == "stage":
            if now - takeover_s >= STAGE_MAX_S:
                end_reason = "stage_timeout"
                break
            command, phase = stage.command(np.zeros(3))
            if phase == "recorded":
                handback_s, state, state_s = now, "bench_post", now
                events.append({"time_s": now, "event": "handback"})
        if state == "bench_post":
            if now + TIME_EPS_S >= state_s + POST_HANDBACK_S:
                end_reason = "complete"
                break
            command = np.zeros(3)
        label = "stage_" + stage.phase if state == "stage" else state
        applied = _route_transform(command)
        for _ in range(DECISION_TICKS):
            targets = env.controller.update(runner.observe(0, applied))
            runner.step([targets])
            if not np.isfinite(runner.d.qpos).all() or runner.d.warning.number.sum():
                raise FloatingPointError("plate-bench physics warning")
            sample = physical_sample(env, applied, label)
            sample["plate_presses"] = sorted([int(p), bool(c)] for p, c in runner.button_contacts)
            samples.append(sample)
            if sample["tilt"] >= FALL_TILT:
                fell, end_reason = True, "fall"
                break
    tilts = [s["tilt"] for s in samples]
    fall_s = next((s["time_s"] for s in samples if s["tilt"] >= FALL_TILT), None)
    row = {"layout_index": layout_index, "layout_seed": int(layout.seed),
           "replay_elapsed_s": samples[-1]["time_s"] if samples else None,
           "first_fall_s": fall_s, "maximum_tilt": max(tilts) if tilts else None,
           "stage_history": stage.history, "samples": samples}
    scored = score_crossing(row, layout, door, takeover_s)
    target = [door, spec["plate"] == "round"]
    stage_samples = [s for s in samples if s["phase"].startswith("stage_")]
    presses = [s for s in samples if s["plate_presses"]]
    wall = [s for s in samples if s["contacts"]]
    return {
        **{key: spec[key] for key in ("layout", "door", "heading_deg", "plate", "entry")},
        "complete": True, "end_reason": end_reason, "geometry": geometry,
        "layout_seed": int(layout.seed), "cell_m": float(layout.cell_m), "env_seed": SEED_BASE + 2 * layout_index + door,
        "takeover_s": takeover_s, "handback_s": handback_s, "events": events,
        "clear": scored["clear"], "real_clear": scored["real_clear"], "first_clear_s": scored["first_clear_s"],
        "clear_after_takeover_s": scored["clear_after_takeover_s"], "crossing": scored["crossing"],
        "fell": fell or scored["fell"], "first_fall_s": fall_s,
        "fall_phase": next((s["phase"] for s in samples if s["tilt"] >= FALL_TILT), None),
        "maximum_tilt": row["maximum_tilt"],
        "plate_activation": {   # reported, not gated
            "target_plate": {"door": door, "side": geometry["side"], "shape": spec["plate"]},
            "target_pressed_during_stage": any(target in s["plate_presses"] for s in stage_samples),
            "first_target_press_s": next((s["time_s"] for s in presses if target in s["plate_presses"]), None),
            "correct_plate_pressed_during_stage": any(any(p[1] for p in s["plate_presses"]) for s in stage_samples),
            "wrong_plate_pressed_during_stage": any(any(not p[1] for p in s["plate_presses"]) for s in stage_samples),
            "presses_before_takeover": sorted({tuple(p) for s in presses if not s["phase"].startswith("stage_")
                                               and (takeover_s is None or s["time_s"] <= takeover_s)
                                               for p in s["plate_presses"]}),
        },
        "wall_contacts": {   # reported, not gated
            "samples": len(wall), "stage_samples": sum(1 for s in wall if s["phase"].startswith("stage_")),
            "first_s": wall[0]["time_s"] if wall else None,
            "geoms": sorted({g for s in wall for g in s["contacts"]}),
        },
        "stage_history": stage.history, "gait_events": stage.gait_events,
        "stage_kwargs": stage_kwargs,
        "trace_header": {key: row[key] for key in ("layout_index", "layout_seed", "replay_elapsed_s",
                                                    "first_fall_s", "maximum_tilt")},
        "samples": samples,
    }


def crossing_path(out, layout_index, door):
    return out / "crossings" / f"L{int(layout_index):03d}-d{int(door)}.json.gz"


def write_crossing(out, result):
    path = crossing_path(out, result["layout"], result["door"])
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as stream:
        json.dump(result, stream)
    os.replace(tmp, path)
    return path


def read_results(out):
    """Every crossing on disk, samples dropped, re-scored from its own trace."""
    results = []
    for path in sorted((out / "crossings").glob("L*-d*.json.gz")):
        with gzip.open(path, "rt") as stream:
            result = json.load(stream)
        layout = generate(BENCH_SPLIT, int(result["layout"]))
        row = {**result["trace_header"], "stage_history": result["stage_history"], "samples": result["samples"]}
        rescored = score_crossing(row, layout, int(result["door"]), result["takeover_s"])
        if (rescored["clear"], rescored["fell"]) != (result["clear"], result["fell"]):
            raise ValueError(f"{path}: stored clear/fall {result['clear']}/{result['fell']} != re-scored "
                             f"{rescored['clear']}/{rescored['fell']}")
        results.append({key: value for key, value in result.items() if key != "samples"})
    return results


def compact_result(result):
    """The per-crossing line kept in verdict.json (the gzipped traces are not committed)."""
    takeover = next((e for e in result.get("events", []) if e["event"] == "takeover"), {})
    crossing = result.get("crossing") or {}
    return {**{key: result.get(key) for key in (
        "layout", "door", "heading_deg", "plate", "entry", "cell_m", "end_reason", "clear", "real_clear",
        "clear_after_takeover_s", "fell", "fall_phase", "maximum_tilt", "takeover_s", "handback_s")},
        "takeover_speed_mps": takeover.get("speed_mps"),
        "along_start_m": crossing.get("along_start_m"), "along_max_m": crossing.get("along_max_m"),
        "phases": [h["phase"] for h in result.get("stage_history", [])],
        "turn_timed_out": any(h.get("turn_timed_out") for h in result.get("stage_history", [])),
        "target_pressed_during_stage": result["plate_activation"]["target_pressed_during_stage"],
        "wall_contact_stage_samples": result["wall_contacts"]["stage_samples"],
        "runup_over_opposite_plate": (result.get("geometry") or {}).get("runup_over_opposite_plate")}


def runup_report(results):
    """Reported, not gated: run-ups over the opposite plate, and the phase of every fall."""
    over = sorted([r["layout"], r["door"]] for r in results
                  if (r.get("geometry") or {}).get("runup_over_opposite_plate"))
    fallen = [r for r in results if r["fell"]]
    phases = {}
    for r in fallen:
        phases[str(r.get("fall_phase"))] = phases.get(str(r.get("fall_phase")), 0) + 1
    return {"runup_over_opposite_plate_not_gated": {
                "crossings": over, "count": len(over),
                "fallen": sorted([r["layout"], r["door"]] for r in fallen if [r["layout"], r["door"]] in over),
                "fallen_before_takeover": sorted([r["layout"], r["door"]] for r in fallen
                                                 if [r["layout"], r["door"]] in over and r.get("takeover_s") is None)},
            "falls_by_phase_not_gated": dict(sorted(phases.items()))}


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def source_files(repo):
    """The physics sources provenance.json hashes (relative paths, sorted, no duplicates).

    BENCH_SOURCES + src/bhl_robust/mission/*.py + scripts/mission7*.py: every file the replay snapshot freezes
    except its own sbatch, so the release script can check that the replay runs the code the bench validated.
    """
    repo = Path(repo)
    names = set(BENCH_SOURCES)
    names.update(str(p.relative_to(repo)) for p in (repo / "src/bhl_robust/mission").glob("*.py"))
    names.update(str(p.relative_to(repo)) for p in (repo / "scripts").glob("mission7*.py"))
    return sorted(names)


def provenance(repo, mode, crossings):
    sources = source_files(repo)
    upstream = repo / "external/Berkeley-Humanoid-Lite"
    from mission7_plate_stage import SHIPPED_EXPORT, TURNBOTH_EXPORT
    weights = {"shipped": upstream / SHIPPED_EXPORT / "policy.onnx", "turnboth": upstream / TURNBOTH_EXPORT / "policy.onnx"}

    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unavailable: {exc}"
    import mujoco
    return {"mode": mode, "crossings": [list(c) for c in crossings], "rule": RULE, "chain": CHAIN, "labels": LABELS,
            "stage_kwargs": STAGE_KWARGS,
            "protocol": {"split": BENCH_SPLIT, "settle_s": SETTLE_S, "runup_speed_mps": RUNUP_SPEED,
                         "yaw_hold": [YAW_HOLD_GAIN, YAW_HOLD_MAX], "arrive_m": ARRIVE_M, "runup_max_s": RUNUP_MAX_S,
                         "stage_max_s": STAGE_MAX_S, "post_handback_s": POST_HANDBACK_S,
                         "decision_ticks": DECISION_TICKS, "seed_base": SEED_BASE, "pre_point_m": PRE_POINT_M,
                         "clear_window_s": CLEAR_WINDOW_S, "clear_along_m": CLEAR_ALONG_M, "fall_tilt": FALL_TILT},
            "git_head": git("rev-parse", "HEAD"),
            "git_status_sources": git("status", "--porcelain", "--", *sources),
            "sources_sha256": {name: _sha256(repo / name) for name in sources if (repo / name).is_file()},
            "weights_sha256": {name: _sha256(path) for name, path in weights.items()},
            "host": platform.node(), "python": sys.executable, "mujoco": mujoco.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("scored", "smoke"), required=True)
    parser.add_argument("--crossings", default="",
                        help="smoke only: comma-separated L:d on exploration train layouts L >= 32")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    args.repo, args.out = args.repo.resolve(), args.out.resolve()
    if args.mode == "scored":
        if args.crossings:
            parser.error("the scored bench runs all 64 grid crossings; --crossings is smoke-only")
        specs = grid()
    else:
        pairs = parse_crossings(args.crossings)
        if not pairs:
            parser.error("--mode smoke needs --crossings")
        bad = [p for p in pairs if not SMOKE_MIN_LAYOUT <= p[0] < SPLITS[BENCH_SPLIT][1] or p[1] not in BENCH_DOORS]
        if bad:
            parser.error(f"smoke crossings must use exploration train layouts {SMOKE_MIN_LAYOUT}-"
                         f"{SPLITS[BENCH_SPLIT][1] - 1} and doors 0/1, never a scored layout: {bad}")
        specs = grid(sorted({p[0] for p in pairs}))
        specs = [s for s in specs if (s["layout"], s["door"]) in set(pairs)]
    if args.preflight:
        from mission7_plate_stage import turnboth_check
        import mujoco
        check = turnboth_check(args.repo / "external/Berkeley-Humanoid-Lite")
        print(json.dumps({"status": "PREFLIGHT_OK", "mode": args.mode, "crossings": len(specs),
                          "python": sys.executable, "mujoco": mujoco.__version__, "out": str(args.out),
                          "stage_gait_check": check}, sort_keys=True), flush=True)
        return 0
    if args.out.exists():
        parser.error(f"output exists; a bench result is never overwritten: {args.out}")
    (args.out / "crossings").mkdir(parents=True)
    (args.out / "provenance.json").write_text(json.dumps(
        provenance(args.repo, args.mode, [(s["layout"], s["door"]) for s in specs]), indent=2) + "\n")
    for spec in specs:
        result = run_crossing(args.repo, args.out / "cache" / f"L{spec['layout']:03d}-d{spec['door']}", spec)
        write_crossing(args.out, result)
        print(json.dumps({"layout": spec["layout"], "door": spec["door"], "heading_deg": spec["heading_deg"],
                          "plate": spec["plate"], "entry": spec["entry"], "end": result["end_reason"],
                          "clear": result["clear"], "fell": result["fell"],
                          "clear_after_takeover_s": result["clear_after_takeover_s"]}), flush=True)
    results = read_results(args.out)
    verdict = bench_verdict(results, expected=grid() if args.mode == "scored" else specs)
    if args.mode == "smoke":   # a smoke is never a bench verdict
        verdict["verdict"] = "SMOKE_" + ("INCOMPLETE" if verdict["verdict"] == "INCOMPLETE" else "RUN")
        verdict["smoke"] = True
    verdict.update(chain=CHAIN, labels=LABELS, stage_kwargs=STAGE_KWARGS, mode=args.mode,
                   plate_activation_not_gated={
                       "round_target_pressed_during_stage": sum(
                           r["plate_activation"]["target_pressed_during_stage"] for r in results if r["plate"] == "round"),
                       "square_target_pressed_during_stage": sum(
                           r["plate_activation"]["target_pressed_during_stage"] for r in results if r["plate"] == "square"),
                       "crossings": len(results)},
                   wall_contacts_not_gated={"crossings_with_stage_contact": sum(
                       r["wall_contacts"]["stage_samples"] > 0 for r in results)},
                   end_reasons={reason: sum(r["end_reason"] == reason for r in results)
                                for reason in sorted({r["end_reason"] for r in results}, key=str)},
                   **runup_report(results),
                   crossings=[compact_result(r) for r in results],
                   written_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    path = args.out / "verdict.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"M7 PLATE BENCH {args.mode}: {verdict['verdict']} clears {verdict['clears']}/{verdict['expected']} "
          f"falls {verdict['falls']} per heading "
          f"{ {h: v['clears'] for h, v in verdict['per_heading'].items()} } -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
