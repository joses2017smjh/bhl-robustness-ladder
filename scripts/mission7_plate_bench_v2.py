"""Mission 7 plate bench v2: run-up-free and gait-agnostic (--stage-gait turnboth | m3); planned for M3 only.

Introduced 2026-10-02 AFTER the M2 bench FAILED (job 21507962, v1 = scripts/mission7_plate_bench.py, unchanged):
8/64 clears, 8 falls, and 26/64 crossings ended in v1's own run-up (the shipped gait walking from route[k-1] to the
pre-point holding the entry yaw) before the stage acted, so v1's >= 62/64 was unreachable for any stage gait.  M2's
FAIL stands.  v2 keeps v1's grid, seeds, physics, stage polling and clear definition, and removes the run-up.
Launcher: slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch (its header carries the same frozen rule, the grid, the
dropped set and every M3 constant).  LEARNED gait(s) (the shipped arms-dr1.0-s0 gait; TurnBoth-s0 during the stage
only with --stage-gait turnboth), SCRIPTED stage (scripts/mission7_plate_stage.py PlateStage, stage_gait=<gait>,
every other parameter at its replay default), ORACLE layout and plate pose.

Grid (identical to v1, mission7_plate_bench.grid()): one crossing per TRAINING-split layout L = 0..31 and door d = 0/1;
entry heading 0, +90, -90, 180 deg relative to the door direction for L mod 4 = 0..3; round (correct) plate for
floor(L/4) mod 2 = 0, square (wrong) plate for 1; standstill entry for d XOR (floor(L/8) mod 2) = 0, walking for 1.

Entry protocol (declared before any v2 episode; nothing here is tuned on a result):
- P = plate centre - 0.30 m along the door direction (PlateStage's pre-point); entry yaw = door direction + heading.
- Spawn: DebugEnv(stage doors, split train, seed 7000 + 2L + d) reset to layout L (the env's own standing pose:
  default joints + N(0, 0.02) rad noise, base at the model's qpos0 height, which puts the lowest foot geom 0.039 m
  above the floor, 0.009 m above a plate top), then the base is moved in x, y and yaw only (velocity zeroed), as v1.
- standstill: spawn at P with the entry yaw, 3.0 s at zero command on the shipped gait, then the stage takes over.
- walking: spawn at S = P - 0.30 m along the entry-yaw unit vector, 3.0 s at zero command, then walk FORWARD along
  that heading at 0.30 m/s (route heading hold clip(1.2 x yaw error, +-0.35) on the entry yaw) for the nominal
  1.0 s; the stage takes over at arrival (within 0.13 m of P, as v1); not arrived after 3.0 s of walking = a miss.
- A walking crossing is run ONLY IF its spawn pose and its straight path S -> P are geometrically clear: no wall or
  closed door within ROBOT_PLANAR_RADIUS_M of the segment, and no foot box (FK of the default pose) on any plate at
  S.  The dropped set is computed from layout geometry alone (dropped_crossings()) and frozen in DROPPED.
- Takeover = PlateStage._start(d, side) on the bench's plate (the square plate too), then PlateStage.command drives
  until its hand-back, then 2.0 s at zero command.  The stage is polled every 5 gait updates (0.2 s) and every command
  passes through the route's action transform, as v1.  Physics as v1: runner.step, doors closed, no MissionState
  termination.  Falls (tilt >= 0.78) count in EVERY phase after the spawn (settle, walk, stage, post) and end the
  crossing.  A stage that has not handed back 60 s after takeover is a miss.  Wall/door contacts, spawn contacts and
  plate presses are recorded and reported, not gated.
- Scoring: score_crossing() = mission7_plate_bench.score_crossing, unchanged (mission7_gates' real clear, first clear
  within 10 s of takeover); check_rescore_m2() shows it reproduces the 38 M2 crossings that reached the stage.

Smoke mode runs named crossings on exploration train layouts L >= 250 only, never a scored layout; it is never a
verdict.

--- m7-clocks2 (2026-10-02, opt-in): --stage-gait clocks2 (arms-turngait-clock-s2) or --stage-gait export
--stage-gait-export <exported dir> runs the same bench, rule, grid and protocol with PlateStage's export stage gait:
the stage CONTROLLER is swapped at takeover (scripts/mission7_plate_stage.py, M2's turnboth law unchanged).  Launcher:
slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch; the turnboth and m3 paths are unchanged.
"""
from __future__ import annotations

import argparse
import datetime as dt
import gzip
import json
import os
import platform
import subprocess
import sys
from pathlib import Path

import numpy as np

import mission7_gates as gates
import mission7_plate_bench as v1
from bhl_robust.mission.layout import SPLITS, generate, wall_segments

RULE_V2 = (
    "Bench v2: with N crossings run (64 minus the dropped set) and N_h per heading, bench v2 PASSES iff clears >= "
    "N - 2, 0 falls, and clears >= N_h - 1 for every heading. Otherwise FAIL; INCOMPLETE if any declared crossing is "
    "missing. The clear definition is unchanged: mission7_gates' real clear (base along >= +0.35 m past the plate "
    "centre along the door direction), within 10 s of takeover.")
CHAIN_V2 = (
    "a chain, each step only after the previous PASSES: bench v2 with --stage-gait m3 -> the unchanged exact "
    "ten-fall replay once with --stage-gait m3 (complete, 10 episodes, 10 upright, 0 falls; release arm "
    "m3-shipped-step) -> the route gate as coded by M1 (Doors and Transport each >= 16/16 successes on validation "
    "layouts 0-15, 32 episodes)")
DISCLOSURE = (
    "bench v2 was introduced after the M2 bench FAIL (21507962) revealed the run-up confound (26/64 crossings ended "
    "in v1's own run-up before takeover); M2's FAIL stands and v1 is unchanged")
STAGE_GAITS_V2 = ("turnboth", "m3")
# --- m7-clocks2 --- the opt-in export stage gaits (= mission7_plate_stage.EXPORT_STAGE_GAITS, which this module does
# not import at load time: it pulls in MuJoCo), their chain and their disclosure.  RULE_V2 above is theirs verbatim.
EXPORT_STAGE_GAITS_V2 = ("clocks2", "export")
CHAIN_V2_CLOCKS2 = (
    "a chain, each step only after the previous PASSES: bench v2 with --stage-gait clocks2 -> the unchanged exact "
    "ten-fall replay once with --stage-gait clocks2 (complete, 10 episodes, 10 upright, 0 falls; release arm "
    "m7-clocks2) -> the route gate as coded by M1 (Doors and Transport each >= 16/16 successes on validation "
    "layouts 0-15, 32 episodes)")
CHAIN_V2_EXPORT = (
    "bench v2 with --stage-gait export only: no replay or route release arm is wired for a generic export; any "
    "chain step after a PASS needs its own arm and the user's approval")
DISCLOSURE_CLOCKS2 = (
    "arms-turngait-clock-s2 was qualified at +-0.6 rad/s and is untested at the interface's 0.40 rad/s; a 180-deg "
    "entry needs (pi - 0.15)/0.40 = 7.48 s of turning + the 0.40-0.60 s settle as polled + 0.65 m / 0.30 m/s = 2.17 s "
    "of crossing = 10.05-10.25 s (about 10.1-10.3 s) against the 10 s clear window, so the 180-deg clause (>= 7/8) is "
    "likely to fail on timing alone unless the gait turns faster than commanded")
V2_CLOCKS2_LAUNCHER = "slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch"
GAIT_CLOCK_SOURCE = "src/bhl_robust/eval/gait_clock.py"   # the clock controller the export gaits build

BENCH_SPLIT = v1.BENCH_SPLIT              # "train"
TOTAL_GRID = v1.TOTAL                     # 64
MAX_MISSES = 2                            # clears >= N - 2
MAX_MISSES_PER_HEADING = 1                # clears >= N_h - 1
CLEAR_WINDOW_S = v1.CLEAR_WINDOW_S        # 10.0
FALL_TILT = gates.FALL_TILT               # 0.78
TIME_EPS_S = 1e-9

PRE_POINT_M = v1.PRE_POINT_M              # 0.30
SETTLE_S = 3.0
WALK_SPEED = .30
WALK_NOMINAL_S = 1.0
WALK_DISTANCE_M = WALK_SPEED * WALK_NOMINAL_S   # 0.30: S = P - 0.30 m along the entry heading
WALK_MAX_S = 3.0                          # 3 x the nominal walk; not arrived = a miss
YAW_HOLD_GAIN = v1.YAW_HOLD_GAIN          # 1.2 (the route's heading hold)
YAW_HOLD_MAX = v1.YAW_HOLD_MAX            # 0.35
ARRIVE_M = v1.ARRIVE_M                    # 0.13
STAGE_MAX_S = v1.STAGE_MAX_S              # 60
POST_HANDBACK_S = v1.POST_HANDBACK_S      # 2.0
DECISION_TICKS = v1.DECISION_TICKS        # 5
SEED_BASE = v1.SEED_BASE                  # 7000
SMOKE_MIN_LAYOUT = 250                    # exploration train layouts only

# Geometry of the drop rule.  ROBOT_PLANAR_RADIUS_M: the robot's measured collision envelope at rest, 0.632 m elbow
# to elbow (SLURM_JOBS.md, Approach -x arms 2026-09-23); FK of the exact default pose gives 0.295 m (srun 21508399).
# FOOT_BOX_*: both ankle-roll collision boxes in the base frame at the default pose (FK, srun 21508399), left foot
# y in [0.021, 0.093], right foot mirrored.  Plates: radius / half-side 0.24 (layout.world_xml); the square plate is
# axis-aligned in the world.  Walls: layout.wall_segments half-sizes; doors: closed, world_xml half-sizes.
ROBOT_PLANAR_RADIUS_M = .316
FOOT_BOX_X = (-.080, .143)
FOOT_BOX_Y = (.021, .093)
PLATE_HALF_M = v1.PLATE_HALF_M            # 0.24
DOOR_HALF_THICKNESS_M = .045

# Frozen from dropped_crossings() before any v2 episode (srun 21508529; asserted in tests/test_mission7_plate_bench_v2.py):
# 23 of the 32 walking crossings, so N = 41 (per heading 0: 12, +90: 11, -90: 10, 180: 8).  All 8 walking 180-deg
# crossings drop by construction (S = P + 0.30 m along the door direction = the plate centre).
DROPPED = ((3, 1), (4, 1), (5, 1), (6, 1), (7, 1), (9, 0), (10, 0), (11, 0), (12, 0), (13, 0), (14, 0), (15, 0), (16, 1),
           (19, 1), (21, 1), (22, 1), (23, 1), (24, 0), (26, 0), (27, 0), (29, 0), (30, 0), (31, 0))
# Disclosed, not dropped (the spec places every standstill base directly at P): standstill spawns whose mj_forward at
# the spawn instant shows a robot geom inside a wall (dist < 0; all in 1.5 m cells, where P lies 0.29 m from a wall
# face against the 0.294 m elbow reach of the default pose).  Depths 1.3-9.3 mm (L4 d0 6.6, L5 d0 6.9, L11 d1 1.3,
# L22 d0 4.2, L24 d1 7.0, L30 d1 9.3 mm; srun 21508529, joint noise as the bench seeds it); every crossing records its
# own spawn contacts.
STANDSTILL_SPAWN_WALL_OVERLAP = ((4, 0), (5, 0), (11, 1), (22, 0), (24, 1), (30, 1))

V2_LAUNCHER = "slurm/repo20260923/cpu_m7_plate_bench_v2.sbatch"
M2_BENCH_DIR = "results/mission7-campaign-20260923/m2-plate-bench"

# per-crossing clear computation: v1's, unchanged (re-exported so the re-score check names what v2 uses)
score_crossing = v1.score_crossing
cell_of = v1.cell_of


# ---- pure: geometry, drop rule, verdict ---------------------------------------------------------------

def obstacle_boxes(layout):
    """Walls (layout.wall_segments) and both closed doors as (centre xy, half sizes xy), axis-aligned."""
    boxes = [(np.asarray(c, dtype=float), np.asarray(s, dtype=float)) for c, s in wall_segments(layout)]
    for i in range(2):
        centre, direction = layout.door(i)
        half = (DOOR_HALF_THICKNESS_M, layout.cell_m / 2) if direction[0] else (layout.cell_m / 2, DOOR_HALF_THICKNESS_M)
        boxes.append((np.asarray(centre, dtype=float), np.asarray(half, dtype=float)))
    return boxes


def point_box_distance(point, centre, half):
    gap = np.maximum(np.abs(np.asarray(point, dtype=float) - centre) - half, 0.)
    return float(np.hypot(gap[0], gap[1]))


def segment_box_distance(a, b, centre, half, iterations=80):
    """Distance (to < 1e-12 m) from segment a-b to an axis-aligned box; the distance is convex along the segment."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    lo, hi = 0., 1.
    for _ in range(iterations):
        m1, m2 = lo + (hi - lo) / 3, hi - (hi - lo) / 3
        if point_box_distance(a + (b - a) * m1, centre, half) <= point_box_distance(a + (b - a) * m2, centre, half):
            hi = m2
        else:
            lo = m1
    return min(point_box_distance(a + (b - a) * t, centre, half) for t in (0., 1., (lo + hi) / 2))


def foot_polygons(xy, yaw):
    """Both foot boxes (4 corners each, world xy) for a base at xy with this yaw (default pose)."""
    c, s = np.cos(yaw), np.sin(yaw)
    rotation = np.array([[c, -s], [s, c]])
    out = []
    for sign in (1., -1.):
        y0, y1 = sorted((sign * FOOT_BOX_Y[0], sign * FOOT_BOX_Y[1]))
        corners = np.array([[FOOT_BOX_X[0], y0], [FOOT_BOX_X[1], y0], [FOOT_BOX_X[1], y1], [FOOT_BOX_X[0], y1]])
        out.append(np.asarray(xy, dtype=float) + corners @ rotation.T)
    return out


def _cross2(u, v):
    return float(u[0] * v[1] - u[1] * v[0])


def _point_in_convex(point, poly):
    signs = [_cross2(poly[(i + 1) % 4] - poly[i], point - poly[i]) for i in range(4)]
    return all(v >= -1e-12 for v in signs) or all(v <= 1e-12 for v in signs)


def _point_segment_distance(point, a, b):
    return v1.segment_distance(point, a, b)


def polygon_hits_disc(poly, centre, radius):
    centre = np.asarray(centre, dtype=float)
    if _point_in_convex(centre, poly):
        return True
    return min(_point_segment_distance(centre, poly[i], poly[(i + 1) % 4]) for i in range(4)) < radius


def polygon_hits_box(poly, centre, half):
    """Separating-axis test between a convex quadrilateral and an axis-aligned box."""
    box = np.asarray(centre, dtype=float) + np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]]) * np.asarray(half, dtype=float)
    axes = [np.array([1., 0.]), np.array([0., 1.])]
    axes += [np.array([-(poly[(i + 1) % 4] - poly[i])[1], (poly[(i + 1) % 4] - poly[i])[0]]) for i in range(4)]
    for axis in axes:
        if not np.any(axis):
            continue
        p, q = poly @ axis, box @ axis
        if p.max() < q.min() or q.max() < p.min():
            return False
    return True


def plate_footprints(layout):
    """Every plate as (door, side, shape, centre): round (correct side) disc, square (wrong side) box."""
    out = []
    for door in range(2):
        for side in (-1, 1):
            shape = "round" if side == layout.correct_sides[door] else "square"
            out.append((door, side, shape, np.asarray(layout.plate(door, side), dtype=float)))
    return out


def feet_on_plates(layout, xy, yaw):
    hits = []
    for door, side, shape, centre in plate_footprints(layout):
        for poly in foot_polygons(xy, yaw):
            hit = (polygon_hits_disc(poly, centre, PLATE_HALF_M) if shape == "round"
                   else polygon_hits_box(poly, centre, (PLATE_HALF_M, PLATE_HALF_M)))
            if hit:
                hits.append([door, side])
                break
    return hits


def crossing_geometry(layout, door, plate, heading_deg, entry):
    """P, S, entry yaw and the drop-rule quantities of one crossing (pure)."""
    side = v1.plate_side(layout, door, plate)
    _, direction = layout.door(door)
    direction = np.asarray(direction, dtype=float)
    plate_xy = np.asarray(layout.plate(door, side), dtype=float)
    pre = plate_xy - direction * PRE_POINT_M
    door_yaw = float(np.arctan2(direction[1], direction[0]))
    entry_yaw = float(np.arctan2(np.sin(door_yaw + np.radians(heading_deg)), np.cos(door_yaw + np.radians(heading_deg))))
    heading = np.array([np.cos(entry_yaw), np.sin(entry_yaw)])
    walk_start = pre - heading * WALK_DISTANCE_M
    boxes = obstacle_boxes(layout)
    path_clearance = min(segment_box_distance(walk_start, pre, c, h) for c, h in boxes)
    p_clearance = min(point_box_distance(pre, c, h) for c, h in boxes)
    s_feet = feet_on_plates(layout, walk_start, entry_yaw)
    walking_clear = bool(path_clearance >= ROBOT_PLANAR_RADIUS_M and not s_feet)
    spawn = walk_start if entry == "walking" else pre
    return {"side": int(side), "direction": direction.tolist(), "plate_xy": plate_xy.tolist(), "pre_point": pre.tolist(),
            "walk_start": walk_start.tolist(), "spawn": spawn.tolist(), "door_yaw_rad": door_yaw,
            "entry_yaw_rad": entry_yaw, "entry": entry,
            "walk_path_wall_clearance_m": path_clearance, "pre_point_wall_clearance_m": p_clearance,
            "walk_start_feet_on_plates": s_feet, "walking_geometrically_clear": walking_clear}


def dropped_crossings(layouts=v1.BENCH_LAYOUTS):
    """Walking crossings whose spawn pose or straight 1 s path is not geometrically clear (layout geometry only)."""
    out = []
    for spec in v1.grid(layouts):
        if spec["entry"] != "walking":
            continue
        g = crossing_geometry(generate(BENCH_SPLIT, spec["layout"]), spec["door"], spec["plate"],
                              spec["heading_deg"], spec["entry"])
        if not g["walking_geometrically_clear"]:
            out.append((spec["layout"], spec["door"]))
    return tuple(out)


def declared_crossings(dropped=None):
    dropped = set(DROPPED if dropped is None else dropped)
    return [s for s in v1.grid() if (s["layout"], s["door"]) not in dropped]


def bench_v2_verdict(results, declared=None):
    """The frozen v2 rule over per-crossing results (pure).

    ``declared`` is the list of crossings run (default: the grid minus DROPPED).  INCOMPLETE when a declared crossing
    is missing or incomplete, duplicated, not declared (a dropped or foreign crossing), or labelled other than the
    grid labels it, or when nothing is declared; otherwise PASS iff clears >= N - 2, 0 falls and clears >= N_h - 1
    in every heading; else FAIL.
    """
    declared = declared_crossings() if declared is None else declared
    want = {(c["layout"], c["door"]): c for c in declared}
    seen, problems = {}, []
    for r in results:
        key = (int(r["layout"]), int(r["door"]))
        if key in seen:
            problems.append(f"duplicate crossing {key}")
        seen[key] = r
        if key not in want:
            problems.append(f"crossing {key} is not a declared crossing")
        elif any(r.get(f) != want[key][f] for f in ("heading_deg", "plate", "entry")):
            problems.append(f"crossing {key} is labelled {[r.get(f) for f in ('heading_deg', 'plate', 'entry')]}, "
                            f"the grid says {[want[key][f] for f in ('heading_deg', 'plate', 'entry')]}")
    missing = sorted(key for key in want if key not in seen or not seen[key].get("complete"))
    usable = [seen[key] for key in want if key in seen and seen[key].get("complete")]
    n = len(want)
    clears = sum(bool(r["clear"]) for r in usable)
    falls = sum(bool(r["fell"]) for r in usable)
    per_heading = {}
    for h in v1.HEADINGS_DEG:
        n_h = sum(1 for c in want.values() if c["heading_deg"] == h)
        per_heading[str(h)] = {"declared": n_h, "required": max(n_h - MAX_MISSES_PER_HEADING, 0),
                               "crossings": sum(1 for r in usable if r["heading_deg"] == h),
                               "clears": sum(bool(r["clear"]) for r in usable if r["heading_deg"] == h)}
    per_cell = {}
    for c in v1.grid():
        name = f"{c['heading_deg']}/{c['plate']}/{c['entry']}"
        cell = per_cell.setdefault(name, {"declared": 0, "clears": 0, "crossings": 0})
        if (c["layout"], c["door"]) in want:
            cell["declared"] += 1
            r = seen.get((c["layout"], c["door"]))
            if r is not None and r.get("complete"):
                cell["crossings"] += 1
                cell["clears"] += int(bool(r["clear"]))
    if missing or problems or n == 0:
        verdict = "INCOMPLETE"
    elif (clears >= n - MAX_MISSES and falls == 0
          and all(v["clears"] >= v["required"] for v in per_heading.values())):
        verdict = "PASS"
    else:
        verdict = "FAIL"
    return {"verdict": verdict, "rule": RULE_V2, "N": n, "required_clears": max(n - MAX_MISSES, 0), "clears": clears,
            "falls": falls, "crossings_scored": len(usable), "per_heading": per_heading, "per_cell": per_cell,
            "missing": [list(k) for k in missing], "problems": problems,
            "misses": sorted([r["layout"], r["door"]] for r in usable if not r["clear"]),
            "fallen": sorted([r["layout"], r["door"]] for r in usable if r["fell"])}


def parse_smoke_specs(text):
    """Smoke crossings (pure): 'L:d' with the grid's own labels, or 'L:d:heading:plate:entry' with explicit labels.

    Smoke only, never scored: exploration train layouts >= SMOKE_MIN_LAYOUT, and a walking entry must pass the drop
    rule.  Explicit labels exist because no walking crossing of the grid formula on layouts 250-255 passes it.
    """
    specs = []
    for item in (x.strip() for x in text.split(",")):
        if not item:
            continue
        parts = item.split(":")
        if len(parts) not in (2, 5):
            raise ValueError(f"smoke crossing {item!r} is not L:d or L:d:heading:plate:entry")
        layout_index, door = int(parts[0]), int(parts[1])
        if not SMOKE_MIN_LAYOUT <= layout_index < SPLITS[BENCH_SPLIT][1] or door not in v1.BENCH_DOORS:
            raise ValueError(f"smoke crossings must use exploration train layouts {SMOKE_MIN_LAYOUT}-"
                             f"{SPLITS[BENCH_SPLIT][1] - 1} and doors 0/1, never a scored layout: {item!r}")
        heading, plate, entry = (cell_of(layout_index, door) if len(parts) == 2
                                 else (int(parts[2]), parts[3], parts[4]))
        if heading not in v1.HEADINGS_DEG or plate not in v1.PLATES or entry not in v1.ENTRIES:
            raise ValueError(f"smoke crossing {item!r} has labels outside the grid's")
        if entry == "walking" and not crossing_geometry(generate(BENCH_SPLIT, layout_index), door, plate, heading,
                                                        entry)["walking_geometrically_clear"]:
            raise ValueError(f"smoke walking crossing {item!r} fails the drop rule")
        specs.append({"layout": layout_index, "door": door, "heading_deg": heading, "plate": plate, "entry": entry})
    if not specs:
        raise ValueError("--mode smoke needs --crossings")
    if len({(s["layout"], s["door"]) for s in specs}) != len(specs):
        raise ValueError("duplicate smoke crossing")
    return specs


def check_rescore_m2(m2_dir):
    """Re-score every M2 crossing that reached the stage with v2's per-crossing clear computation (evidence)."""
    m2_dir = Path(m2_dir)
    compact = {(c["layout"], c["door"]): c for c in json.loads((m2_dir / "verdict.json").read_text())["crossings"]}
    rows = []
    for path in sorted((m2_dir / "crossings").glob("L*-d*.json.gz")):
        with gzip.open(path, "rt") as stream:
            result = json.load(stream)
        if result.get("takeover_s") is None:
            continue
        layout = generate(BENCH_SPLIT, int(result["layout"]))
        row = {**result["trace_header"], "stage_history": result["stage_history"], "samples": result["samples"]}
        s = score_crossing(row, layout, int(result["door"]), result["takeover_s"])
        key = (int(result["layout"]), int(result["door"]))
        line = {"layout": key[0], "door": key[1], "recorded_clear": bool(result["clear"]),
                "verdict_json_clear": compact.get(key, {}).get("clear"), "rescored_clear": bool(s["clear"]),
                "recorded_real_clear": bool(result["real_clear"]), "rescored_real_clear": bool(s["real_clear"]),
                "recorded_clear_after_takeover_s": result["clear_after_takeover_s"],
                "rescored_clear_after_takeover_s": s["clear_after_takeover_s"]}
        line["match"] = (line["recorded_clear"] == line["rescored_clear"] == line["verdict_json_clear"]
                         and line["recorded_real_clear"] == line["rescored_real_clear"]
                         and line["recorded_clear_after_takeover_s"] == line["rescored_clear_after_takeover_s"])
        rows.append(line)
    return {"m2_dir": str(m2_dir), "reached_stage": len(rows), "matches": sum(r["match"] for r in rows),
            "all_match": bool(rows) and all(r["match"] for r in rows),
            "recorded_clears": sum(r["recorded_clear"] for r in rows),
            "rescored_clears": sum(r["rescored_clear"] for r in rows),
            "clear_definition": gates.CLEAR_DEFINITION, "rows": rows}


# ---- simulation -----------------------------------------------------------------------------------------

def _spawn_contacts(env):
    """Robot-vs-world contacts with dist < 0 right after the spawn placement (reported, not gated)."""
    import mujoco
    r = env.runner
    out = []
    for i in range(r.d.ncon):
        c = r.d.contact[i]
        a, b = int(c.geom1), int(c.geom2)
        if r.own[a] == r.own[b] or c.dist >= 0:
            continue
        world = b if r.own[a] else a
        out.append({"world_geom": mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_GEOM, world) or str(world),
                    "dist_m": float(c.dist)})
    return out


def run_crossing(repo, cache, spec, stage_gait):
    """One v2 crossing; returns the full row (trace included)."""
    from bhl_robust.mission.approach_debug import DebugEnv, physical_sample, wrap
    from mission7_plate_stage import PlateStage, _yaw
    layout_index, door = int(spec["layout"]), int(spec["door"])
    env = DebugEnv(repo, cache, stage="doors", split=BENCH_SPLIT, seed=SEED_BASE + 2 * layout_index + door)
    env.reset(layout_index)
    layout, runner = env.layout, env.runner
    geometry = crossing_geometry(layout, door, spec["plate"], spec["heading_deg"], spec["entry"])
    entry_yaw, pre = geometry["entry_yaw_rad"], np.asarray(geometry["pre_point"])
    v1._place(env, np.asarray(geometry["spawn"]), entry_yaw)
    spawn = {"xy": geometry["spawn"], "yaw_rad": entry_yaw,
             "base_z_m": float(runner.d.qpos[env.slot.qpos_adr + 2]), "contacts": _spawn_contacts(env)}
    stage = PlateStage(env, stage_gait=stage_gait)
    walking = spec["entry"] == "walking"
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
            if walking:
                state, state_s = "bench_walk", now
                events.append({"time_s": now, "event": "walk_start", "distance_to_pre_point_m": float(np.linalg.norm(xy - pre))})
            else:
                state = "takeover"
        if state == "bench_walk":
            if np.linalg.norm(xy - pre) <= ARRIVE_M:
                events.append({"time_s": now, "event": "arrived", "distance_m": float(np.linalg.norm(xy - pre)),
                               "walk_s": now - state_s})
                state = "takeover"
            elif now - state_s >= WALK_MAX_S - TIME_EPS_S:
                end_reason = "walk_timeout"
                break
            else:
                command = np.array([WALK_SPEED, 0., float(np.clip(YAW_HOLD_GAIN * wrap(entry_yaw - _yaw(env)),
                                                                  -YAW_HOLD_MAX, YAW_HOLD_MAX))])
        if state == "takeover":
            stage._start(door, geometry["side"], now)
            takeover_s, state = now, "stage"
            events.append({"time_s": now, "event": "takeover",
                           "speed_mps": float(np.linalg.norm(runner.d.qvel[env.slot.qvel_adr:env.slot.qvel_adr + 2])),
                           "distance_to_pre_point_m": float(np.linalg.norm(xy - pre)), "yaw_rad": _yaw(env)})
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
        applied = v1._route_transform(command)
        for _ in range(DECISION_TICKS):
            targets = env.controller.update(runner.observe(0, applied))
            runner.step([targets])
            if not np.isfinite(runner.d.qpos).all() or runner.d.warning.number.sum():
                raise FloatingPointError("plate-bench-v2 physics warning")
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
    result = {
        **{key: spec[key] for key in ("layout", "door", "heading_deg", "plate", "entry")},
        "bench": "v2", "stage_gait": stage_gait,
        "complete": True, "end_reason": end_reason, "geometry": geometry, "spawn": spawn,
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
        "stage_history": stage.history, "gait_events": getattr(stage, "gait_events", []),
        "m3_events": getattr(stage, "m3_events", None),
        "policy_is_shipped_at_end": env.controller.policy is env.gait,
        "stage_kwargs": {"stage_gait": stage_gait},
        "trace_header": {key: row[key] for key in ("layout_index", "layout_seed", "replay_elapsed_s",
                                                    "first_fall_s", "maximum_tilt")},
        "samples": samples,
    }
    return result


def compact_result(result):
    """v1's per-crossing line plus the v2 fields (spawn contacts, walk, watchdog)."""
    line = v1.compact_result(result)
    line.pop("runup_over_opposite_plate", None)
    events = result.get("m3_events") or []
    arrived = next((e for e in result.get("events", []) if e["event"] == "arrived"), {})
    line.update(stage_gait=result.get("stage_gait"),
                spawn_wall_contacts=[c["world_geom"] for c in (result.get("spawn") or {}).get("contacts", [])],
                walk_s=arrived.get("walk_s"),
                watchdog_recoveries=sum(e.get("event") == "stall_recover" for e in events),
                watchdog_handback=any(e.get("event") == "stall_handback" for e in events))
    if result.get("stage_gait") in EXPORT_STAGE_GAITS_V2:   # --- m7-clocks2 --- the controller swaps, compact
        line["controller_swaps"] = [{key: e.get(key) for key in (
            "event", "time_s", "controller", "obs_width", "clock_step", "stage_controller_clock_step")}
            for e in result.get("gait_events") or []]
    return line


def source_files(repo, stage_gait=None):
    """v1's physics-source set (the replay snapshot's files minus its sbatch, plus v1's launcher) plus this launcher.

    --- m7-clocks2 --- an export stage gait also hashes the clock controller module and its own launcher (its replay
    snapshot copies gait_clock.py too); every other gait's set is unchanged.
    """
    names = set(v1.source_files(repo)) | {V2_LAUNCHER}
    if stage_gait in EXPORT_STAGE_GAITS_V2:
        names |= {GAIT_CLOCK_SOURCE, V2_CLOCKS2_LAUNCHER}
    return sorted(names)


def chain_for(stage_gait):
    """The predeclared chain text of a bench-v2 run (m3's unchanged)."""
    return {"clocks2": CHAIN_V2_CLOCKS2, "export": CHAIN_V2_EXPORT}.get(stage_gait, CHAIN_V2)


def provenance(repo, mode, stage_gait, crossings):
    import hashlib
    sources = source_files(repo, stage_gait)
    upstream = repo / "external/Berkeley-Humanoid-Lite"
    from mission7_plate_stage import SHIPPED_EXPORT, TURNBOTH_EXPORT, m3_constants
    weights = {"shipped": upstream / SHIPPED_EXPORT / "policy.onnx"}
    if stage_gait == "turnboth":
        weights["turnboth"] = upstream / TURNBOTH_EXPORT / "policy.onnx"
    export_info = None
    if stage_gait in EXPORT_STAGE_GAITS_V2:   # --- m7-clocks2 --- the export, checked, and its weights
        from mission7_plate_stage import export_check, export_dir_for
        export_info = export_check(upstream, *export_dir_for(upstream, stage_gait))
        weights[stage_gait] = Path(export_info["policy"])

    def sha(path):
        return hashlib.sha256(Path(path).read_bytes()).hexdigest()

    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(repo), *args], capture_output=True, text=True,
                                  timeout=60).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            return f"unavailable: {exc}"
    import mujoco
    prov = {"bench": "v2", "mode": mode, "stage_gait": stage_gait, "crossings": [list(c) for c in crossings],
            "rule": RULE_V2, "chain": CHAIN_V2, "disclosure": DISCLOSURE, "labels": labels(stage_gait),
            "dropped": [list(c) for c in DROPPED], "standstill_spawn_wall_overlap": [list(c) for c in
                                                                                     STANDSTILL_SPAWN_WALL_OVERLAP],
            "stage_constants": m3_constants() if stage_gait == "m3" else {"stage_gait": stage_gait},
            "protocol": {"split": BENCH_SPLIT, "settle_s": SETTLE_S, "walk_speed_mps": WALK_SPEED,
                         "walk_distance_m": WALK_DISTANCE_M, "walk_max_s": WALK_MAX_S,
                         "yaw_hold": [YAW_HOLD_GAIN, YAW_HOLD_MAX], "arrive_m": ARRIVE_M, "stage_max_s": STAGE_MAX_S,
                         "post_handback_s": POST_HANDBACK_S, "decision_ticks": DECISION_TICKS, "seed_base": SEED_BASE,
                         "pre_point_m": PRE_POINT_M, "clear_window_s": CLEAR_WINDOW_S,
                         "clear_along_m": gates.CLEAR_ALONG_M, "fall_tilt": FALL_TILT,
                         "robot_planar_radius_m": ROBOT_PLANAR_RADIUS_M, "foot_box_x": FOOT_BOX_X,
                         "foot_box_y": FOOT_BOX_Y},
            "git_head": git("rev-parse", "HEAD"),
            "git_status_sources": git("status", "--porcelain", "--", *sources),
            "sources_sha256": {name: sha(repo / name) for name in sources if (repo / name).is_file()},
            "weights_sha256": {name: sha(path) for name, path in weights.items()},
            "host": platform.node(), "python": sys.executable, "mujoco": mujoco.__version__,
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "started_utc": dt.datetime.now(dt.timezone.utc).isoformat()}
    if export_info is not None:   # --- m7-clocks2 --- (turnboth and m3 records are unchanged)
        prov.update(chain=chain_for(stage_gait), stage_constants=export_info, stage_gait_export=export_info["export"])
        if stage_gait == "clocks2":
            prov["disclosure_stage_gait"] = DISCLOSURE_CLOCKS2
    return prov


def labels(stage_gait):
    gait = ("LEARNED (the shipped arms-dr1.0-s0 gait throughout; never swapped)" if stage_gait == "m3" else
            "LEARNED (shipped arms-dr1.0-s0 outside the stage; TurnBoth-s0 during the stage)")
    if stage_gait in EXPORT_STAGE_GAITS_V2:   # --- m7-clocks2 ---
        gait = ("LEARNED (shipped arms-dr1.0-s0 outside the stage; during the stage the controller is swapped to "
                + ("arms-turngait-clock-s2, a 77-observation gait-clock policy (Turning R1's qualified seed)"
                   if stage_gait == "clocks2" else "the --stage-gait-export policy") + ")")
    return {"gaits": gait, "stage": f"SCRIPTED (PlateStage, stage_gait={stage_gait})",
            "layout_and_plate_pose": "ORACLE (as the existing stage uses them)"}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("scored", "smoke", "rescore-m2"), required=True)
    parser.add_argument("--stage-gait", choices=STAGE_GAITS_V2 + EXPORT_STAGE_GAITS_V2, default=None)
    parser.add_argument("--crossings", default="",
                        help="smoke only: comma-separated L:d on exploration train layouts L >= 250")
    parser.add_argument("--m2-dir", type=Path, default=None, help="rescore-m2 only (default the M2 bench directory)")
    # --- m7-clocks2 --- the generic export stage gait (clocks2 is a pinned preset and takes neither flag)
    parser.add_argument("--stage-gait-export", type=Path, default=None,
                        help="--stage-gait export only: the exported directory swapped in as the stage controller")
    parser.add_argument("--stage-gait-export-sha256", default=None,
                        help="--stage-gait export only: refuse the export unless its policy.onnx has this sha256")
    parser.add_argument("--preflight", action="store_true")
    args = parser.parse_args(argv)
    args.repo, args.out = args.repo.resolve(), args.out.resolve()
    if (args.stage_gait == "export") != (args.stage_gait_export is not None):
        parser.error("--stage-gait-export <exported dir> is required with --stage-gait export, and only with it")
    if args.stage_gait_export_sha256 is not None and args.stage_gait != "export":
        parser.error("--stage-gait-export-sha256 pins a --stage-gait export only")
    if args.stage_gait == "export":
        from mission7_plate_stage import select_export_gait
        args.stage_gait_export = args.stage_gait_export.resolve()
        select_export_gait(args.stage_gait_export, args.stage_gait_export_sha256)
    if args.mode == "rescore-m2":
        out = args.out / "rescore_m2.json"
        if out.exists():
            parser.error(f"output exists; evidence is never overwritten: {out}")
        report = check_rescore_m2(args.m2_dir or (args.repo / M2_BENCH_DIR))
        args.out.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(report, indent=2) + "\n")
        print(f"M2 RE-SCORE under bench v2: {report['matches']}/{report['reached_stage']} crossings reproduce their "
              f"recorded clear (recorded clears {report['recorded_clears']}, re-scored {report['rescored_clears']}) "
              f"-> {out}", flush=True)
        return 0 if report["all_match"] and report["reached_stage"] == 38 else 1
    if args.stage_gait is None:
        parser.error("--stage-gait turnboth|m3 is required for the bench")
    if args.mode == "scored":
        if args.crossings:
            parser.error("the scored bench runs every declared crossing; --crossings is smoke-only")
        if dropped_crossings() != DROPPED:
            parser.error("the frozen DROPPED set no longer matches dropped_crossings(); nothing runs")
        specs = declared_crossings()
    else:
        try:
            specs = parse_smoke_specs(args.crossings)
        except ValueError as exc:
            parser.error(str(exc))
    if args.preflight:
        import mujoco
        from mission7_plate_stage import m3_constants, turnboth_check
        check = (turnboth_check(args.repo / "external/Berkeley-Humanoid-Lite") if args.stage_gait == "turnboth"
                 else m3_constants())
        if args.stage_gait in EXPORT_STAGE_GAITS_V2:   # --- m7-clocks2 --- the export check, before any episode
            from mission7_plate_stage import export_check, export_dir_for
            upstream = args.repo / "external/Berkeley-Humanoid-Lite"
            check = export_check(upstream, *export_dir_for(upstream, args.stage_gait))
        print(json.dumps({"status": "PREFLIGHT_OK", "bench": "v2", "mode": args.mode, "stage_gait": args.stage_gait,
                          "crossings": len(specs), "dropped": len(DROPPED), "python": sys.executable,
                          "mujoco": mujoco.__version__, "out": str(args.out), "stage_gait_check": check},
                         sort_keys=True), flush=True)
        return 0
    if args.out.exists():
        parser.error(f"output exists; a bench result is never overwritten: {args.out}")
    (args.out / "crossings").mkdir(parents=True)
    (args.out / "provenance.json").write_text(json.dumps(
        provenance(args.repo, args.mode, args.stage_gait, [(s["layout"], s["door"]) for s in specs]), indent=2) + "\n")
    for spec in specs:
        result = run_crossing(args.repo, args.out / "cache" / f"L{spec['layout']:03d}-d{spec['door']}", spec,
                              args.stage_gait)
        v1.write_crossing(args.out, result)
        print(json.dumps({"layout": spec["layout"], "door": spec["door"], "heading_deg": spec["heading_deg"],
                          "plate": spec["plate"], "entry": spec["entry"], "end": result["end_reason"],
                          "clear": result["clear"], "fell": result["fell"],
                          "clear_after_takeover_s": result["clear_after_takeover_s"]}), flush=True)
    results = v1.read_results(args.out)
    verdict = bench_v2_verdict(results, declared=declared_crossings() if args.mode == "scored" else specs)
    if args.mode == "smoke":   # a smoke is never a bench verdict
        verdict["verdict"] = "SMOKE_" + ("INCOMPLETE" if verdict["verdict"] == "INCOMPLETE" else "RUN")
        verdict["smoke"] = True
    verdict.update(bench="v2", stage_gait=args.stage_gait, mode=args.mode, chain=chain_for(args.stage_gait),
                   disclosure=DISCLOSURE,
                   labels=labels(args.stage_gait), dropped=[list(c) for c in DROPPED],
                   standstill_spawn_wall_overlap_not_gated=[list(c) for c in STANDSTILL_SPAWN_WALL_OVERLAP],
                   plate_activation_not_gated={
                       "round_target_pressed_during_stage": sum(
                           r["plate_activation"]["target_pressed_during_stage"] for r in results if r["plate"] == "round"),
                       "square_target_pressed_during_stage": sum(
                           r["plate_activation"]["target_pressed_during_stage"] for r in results if r["plate"] == "square"),
                       "crossings": len(results)},
                   wall_contacts_not_gated={"crossings_with_stage_contact": sum(
                       r["wall_contacts"]["stage_samples"] > 0 for r in results)},
                   falls_by_phase_not_gated={p: sum(r["fall_phase"] == p for r in results if r["fell"])
                                             for p in sorted({str(r["fall_phase"]) for r in results if r["fell"]})},
                   end_reasons={reason: sum(r["end_reason"] == reason for r in results)
                                for reason in sorted({r["end_reason"] for r in results}, key=str)},
                   crossings=[compact_result(r) for r in results],
                   written_utc=dt.datetime.now(dt.timezone.utc).isoformat())
    if args.stage_gait in EXPORT_STAGE_GAITS_V2:   # --- m7-clocks2 --- (turnboth and m3 verdicts are unchanged)
        verdict["stage_gait_export"] = str(args.stage_gait_export) if args.stage_gait == "export" else "clocks2 preset"
        if args.stage_gait == "clocks2":
            verdict["disclosure_stage_gait"] = DISCLOSURE_CLOCKS2
    path = args.out / "verdict.json"
    if path.exists():
        raise FileExistsError(path)
    path.write_text(json.dumps(verdict, indent=2) + "\n")
    print(f"M7 PLATE BENCH V2 {args.mode} ({args.stage_gait}): {verdict['verdict']} clears {verdict['clears']}/"
          f"{verdict['N']} (need {verdict['required_clears']}) falls {verdict['falls']} per heading "
          f"{ {h: (v['clears'], v['declared']) for h, v in verdict['per_heading'].items()} } -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
