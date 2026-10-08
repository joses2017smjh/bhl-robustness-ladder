"""Mission 7 crossing diagnosis (design D, m7-crossdiag): why clock-s2 stage crossings stall at the plate edge.

Frozen design: SLURM_JOBS.md, "User approval recorded 2026-10-03 09:55", item (D) "Mission 7 crossing diagnosis
(investigation, no gate)", with the coordinator's "Correction to design (D)" (layouts 170-249) and "Coordinator re-freeze
of (D)'s measurement and classifier, 2026-10-03 11:40" (contact normals, leading foot, obstruction evidence, precedence,
falls, specificity, INCOMPLETE) and "Coordinator correction to (D)'s re-frozen classifier, 2026-10-04 10:45" (a blocked
swing's touchdown must be on the floor side for both obstruction evidences).  Launcher:
slurm/repo20260923/cpu_m7_crossing_diag.sbatch (its header carries the frozen threshold block below verbatim;
tests/test_m7_crossing_diag.py checks that the block, the header and the constants agree).

WHAT RUNS.  Bench-v2 crossings exactly as bench v2 runs them with --stage-gait clocks2 (job 21517668): every crossing is
scripts/mission7_plate_bench_v2.py run_crossing(repo, cache, spec, "clocks2"), imported and called unchanged (spawn
construction, 3.0 s settle, walking/standstill entries, PlateStage with M2's turnboth law on the swapped-in clock-s2
controller, 0.2 s polling through the route's action transform, falls in every phase, clear definition, scoring).  No
frozen source is edited.  The only addition is a READ-ONLY recorder: while a crossing runs,
bhl_robust.mission.approach_debug.physical_sample (which run_crossing imports function-locally and calls once per policy
step, right after runner.step) is wrapped; the wrapper returns the original's sample unchanged and then reads MjData
(xpos, geom_xpos, geom_xmat, d.contact; mj_contactForce into its own array) to append one diagnosis record.  Nothing is
written to MjData, the controllers or the stage; the smoke checks that a wrapped crossing's bench record is
byte-identical to an unwrapped one.
  Labels: LEARNED gaits (shipped arms-dr1.0-s0 outside the stage; LEARNED clock-s2 = arms-turngait-clock-s2 as the stage
  gait) | SCRIPTED stage (PlateStage, stage_gait=clocks2) | ORACLE layout and plate pose.  DIAGNOSIS ONLY: no gate, no
  bench verdict; nothing here changes a bench rule.

LAYOUTS.  Exploration train layouts 170-249 only (never bench crossings, train layouts 0-31 or validation layouts), at
most 80 crossings.  CORRECTION (coordinator decision 2026-10-03, recorded in the ledger as a correction): 250-289 as
first frozen does not exist beyond 255; corrected to 170-249 before any diagnosis episode.  (bhl_robust.mission.layout.
SPLITS["train"] = (0, 256): generate("train", L) exists only for L <= 255, so 250-289 would have yielded only 6
standstill crossings, L250-L255 d1.)  No Mission 7 crossing record exists for any layout in 170-249 (checked before any
diagnosis episode; other workstreams' smokes are kept to layouts 32-169), so no run crossing has a public outcome.  The
crossings bench v2's construction yields there: mission7_plate_bench.grid() labels, walking crossings that fail bench
v2's drop rule dropped exactly as mission7_plate_bench_v2.dropped_crossings() drops them.
CAP RULE (deterministic, frozen before any diagnosis episode): the kept crossings are taken in grid order (layout
ascending, door 0 before door 1) and the first 80 are run; the rest are listed as capped out, never run.  On 170-249:
160 grid crossings, 61 walking ones dropped, 99 kept; the run is the first 80, L170 d1 through L234 d0 (65 layouts;
0 deg 24, +90 21, -90 19, 180 16; 64 standstill, 16 walking); capped out: L234 d1, L235 d1, L236 d0, L236 d1, L237 d1,
L238 d1, L239 d1, L240 d0, L241 d0, L242 d0, L242 d1, L243 d0, L244 d0, L245 d0, L246 d0, L247 d0, L248 d1, L249 d0,
L249 d1.  The smoke cannot use the instructed layouts 290-299 (they do not exist): it runs SMOKE_CROSSINGS on the
exploration train layouts 32, 33 and 38, outside 170-249, which run mode refuses.

STALL (design D; the definition of solutions-20260930/campaign-m7-180-gemini/m7-180-analysis/edge_dwell.py, verbatim):
along = (base xy - plate centre) . door direction (near plate edge -0.24 m on the centre line, pre-point -0.30 m, clear
+0.35 m); edge dwell = time from the stage's first "cross" history entry to the first bench sample (in any later phase)
with along > -0.20 m; a STALL is a crossing with a "cross" entry whose edge dwell is >= 2.0 s (1e-6 s tolerance) or that
never passes -0.20 m.  A crossing with no "cross" entry (fall or time-out before it) is "no_cross", never a stall.
Window W = from the "cross" entry to the earliest of its first pass of -0.20 m, the crossing's end (the stage's next
"turn_back"/"recorded" entry) and the trace's end.  Every crossing with a "cross" entry gets the classifier's criteria on
W (and, as a control, on its first 2.0 s); stalls get a class.

FEET (per policy step, diagnosis only).  The sole contact geom of each foot is the one collision BOX on its ankle-roll
link (r0_leg_{left,right}_ankle_roll; half sizes 0.036 x 0.11 x 0.02 m).  From its 8 world corners: toe_along_m = the
most advanced corner along the door direction; toe_to_edge_m = max over the box's outline (its 12 edges projected on the
floor, 41 evenly spaced points each, corners included) of (point along - the near-edge along at that point's lateral
offset: -sqrt(0.24^2 - lat^2) on the round plate, -0.24 on the square one; points beside the plate ignored; None if all
are beside it), so a toe face crossing the round plate's centre line is measured where the curved edge is nearest;
toe_z_m / sole_min_z_m (reported, not used: a box sinks into the round plate);
inside_plate = all 8 corners inside the target plate's footprint shrunk by 0.01 m (the smoke's round-plate check).
Contacts of that box, normal force by mj_contactForce, classified by world geom and, for plates, by the contact NORMAL
(MuJoCo's frame[0:3], sign-corrected to point from the world geom into the foot; its world z component is n_up_z): floor;
target plate TOP (support) or target plate EDGE FACE (obstruction); other plates top / edge; walls/doors/goal posts; own
body; other.  The contact point's height is recorded but never used: box-on-cylinder contacts sink (smoke 21532133, L33
d1 step 283: a foot standing mid-plate has one contact at z 0.0175 and its lowest corner at z 0.005).

FROZEN THRESHOLDS (re-frozen 2026-10-03 per the coordinator re-freeze; corrected 2026-10-04 10:45; before any episode on 170-249)
  STALL_DWELL_S = 2.0                 design D: edge dwell >= 2.0 s is a stall (1e-6 s tolerance)
  ONPLATE_ALONG_M = -0.2              design D / edge_dwell.py: past the edge = base along > -0.20 m
  TOP_NORMAL_MIN_Z = 0.7071           a plate contact is TOP (support) iff its normal, from the plate into the foot, has
                                      world z >= 0.7071 (within 45 deg of vertical); otherwise it is the EDGE FACE
  SUPPORT_MIN_N = 8.0                 a foot is loaded iff floor + plate-top + other-plate-top normal force >= 8.0 N (5% of
                                      the robot's 160.2 N weight: 16.33 kg in the compiled model)
  SWING_MIN_STEPS = 2                 a swing = >= 2 consecutive unloaded policy steps (>= 0.08 s); its touchdown = the next
                                      loaded step
  LEADING_TOL_M = 0.01                a swing is the LEADING foot's iff at touchdown its toe along >= the other foot's toe
                                      along - 0.01 m (at or ahead of it)
  EDGE_FORCE_MIN_N = 1.0              obstruction (a): target-plate edge-face normal force >= 1.0 N at any step from lift-off
                                      to touchdown (1 N = the bench's own plate-press force threshold)
  EDGE_STOP_BEFORE_M = 0.02           obstruction (b): touchdown toe_to_edge >= -0.02 m ...
  EDGE_STOP_AFTER_M = 0.01            ... and <= +0.01 m, on the floor side: floor force >= 8.0 N and target-plate-top force
                                      < 8.0 N at touchdown
  BLOCKED_MIN_SWINGS = 3              blocked step-up iff >= 3 blocked swings (leading AND obstructed) touch down inside W
  GAIT_PERIOD_STEPS = 20              one clock-s2 gait period (0.8 s) in policy steps (0.04 s)
  SLOW_MIN_PERIODS = 2                slow progress needs >= 2 full gait periods in W
  ADVANCE_MIN_PER_PERIOD_M = 0.004    a period advances iff base along gains >= 0.004 m over it (differences one gait
                                      period apart cancel the gait's fore-aft sway)
  ADVANCING_PERIOD_FRACTION = 0.75    keeps advancing iff >= 75% of W's full periods advance
  SLOW_MAX_FRACTION = 0.5             too slowly iff mean along speed over W < 0.5 x mean commanded vx over W
  WALL_DOMINANT_FRACTION = 0.5        wall-blocked iff >= 50% of W's policy steps have a wall/door/goal-post contact (the
                                      bench sample's own "contacts" field)
  PRECEDENCE = wall_blocked > blocked_step_up > slow_progress > other   (coordinator re-freeze; conservative for F1)
  RULES (coordinator re-freeze 2026-10-03 11:40; the pure functions upward_normal, contact_key, is_loaded, is_leading,
  is_blocked_swing, stall_criteria, classify, decide and decision_lines encode them; the tests check each):
  - Plate contacts are classified by the contact NORMAL only: MuJoCo's frame[0:3] points from geom1 to geom2, so it is
    negated when the plate is geom2; n_up_z = the world z of the normal from the plate into the foot; TOP iff n_up_z
    >= 0.7071, else EDGE FACE; all TOP force (target and other plates) is support; contact heights are never used.
  - A BLOCKED swing is a swing of the LEADING foot (toe along = its sole box's most advanced corner along the door
    direction; at touchdown >= the other foot's toe along - 0.01 m) with obstruction evidence (a) or (b) above AND a
    FLOOR-SIDE touchdown: floor force >= 8.0 N and target-plate-top force < 8.0 N at touchdown, i.e. the foot did not
    get over the edge (coordinator correction 2026-10-04 10:45: a swing that brushes the edge face and lands on the
    plate top got over; the smoke audit found 4 of 11 such swings); no swing-height clause (clock-s2's flat-ground
    swings are themselves low; heights are reported only).
  - Each stall gets one class on its window W by the precedence wall_blocked > blocked_step_up > slow_progress > other;
    every criterion and the overlaps are reported.
  - Falls: a crossing that fell (bench v2: tilt >= 0.78 in any phase) is excluded from the stall count and reported
    separately; the blocked-step-up fraction is printed both ways (non-fall; including fall stalls); the decision uses
    the non-fall stalls.
  - SPECIFICITY: among the crossings that CLEAR (bench v2's clear), the share whose own window W (their own edge dwell)
    meets the blocked-step-up criterion (>= 3 blocked swings, regardless of precedence); printed on the decision line.
  - F1 is TRAINED iff non-fall stalls > 0 AND 3 x blocked >= non-fall stalls AND clears > 0 AND meeting clears x non-fall
    stalls < blocked x clears (exact integers: the non-fall fraction >= 1/3 and the clear share strictly lower than
    it).  0 non-fall stalls -> F1 NOT trained; 0 clears -> the specificity check cannot hold -> F1 NOT trained.  The
    decision line always prints the non-fall fraction and the clear share, never "undefined" or a True/False flag.
  - A partial or aborted run (a declared crossing missing or failed, an imported source changed during the run, the
    module failing, the job terminated) prints INCOMPLETE and decides nothing; so does a log without a DECISION line.
  - Smoke check (machinery only): a foot standing inside a ROUND target plate (all 8 sole corners inside the footprint
    shrunk by 0.01 m, >= 8.0 N of target-plate force) reads as loaded in >= 95% of >= 10 such foot-steps.
END FROZEN THRESHOLDS

CLASSES (on the stall's window W; precedence and rules as frozen above).
  wall_blocked:    a wall/door/goal-post contact in >= 50% of W's policy steps.
  blocked_step_up: >= 3 blocked swings touch down in W.  A BLOCKED swing is the LEADING foot's, touches down on the
                   floor side (floor >= 8.0 N, target-plate top < 8.0 N) and shows obstruction: (a) target-plate
                   edge-face force >= 1.0 N from lift-off to touchdown, or (b) the toe stopped at the edge at touchdown,
                   -0.02 <= toe_to_edge_m <= +0.01 (inclusive; None never).
  slow_progress:   >= 2 full gait periods in W, >= 75% of them advance >= 0.004 m, and mean along speed over W
                   < 0.5 x mean commanded vx over W (which must be > 0).
  other:           none of the above.
The decision over the classes is the frozen block's RULES (decide(), decision_lines()).
"""
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import gzip
import hashlib
import json
import os
import platform
import re
import sys
import traceback
from pathlib import Path

import numpy as np

_SCRIPTS = Path(__file__).resolve().parents[1]
if str(_SCRIPTS) not in sys.path:   # mission7_* live in scripts/ (the launcher also puts it on PYTHONPATH)
    sys.path.insert(0, str(_SCRIPTS))

import mission7_plate_bench as v1  # noqa: E402
import mission7_plate_bench_v2 as v2  # noqa: E402
from bhl_robust.mission.layout import SPLITS, generate  # noqa: E402

# ---- frozen constants (the docstring's FROZEN THRESHOLDS block; checked by the tests) -------------------------------
STALL_DWELL_S = 2.0
ONPLATE_ALONG_M = -0.2
TOP_NORMAL_MIN_Z = 0.7071
SUPPORT_MIN_N = 8.0
SWING_MIN_STEPS = 2
LEADING_TOL_M = 0.01
EDGE_FORCE_MIN_N = 1.0
EDGE_STOP_BEFORE_M = 0.02
EDGE_STOP_AFTER_M = 0.01
BLOCKED_MIN_SWINGS = 3
GAIT_PERIOD_STEPS = 20
SLOW_MIN_PERIODS = 2
ADVANCE_MIN_PER_PERIOD_M = 0.004
ADVANCING_PERIOD_FRACTION = 0.75
SLOW_MAX_FRACTION = 0.5
WALL_DOMINANT_FRACTION = 0.5
PRECEDENCE = ("wall_blocked", "blocked_step_up", "slow_progress", "other")
FROZEN_THRESHOLD_NAMES = (
    "STALL_DWELL_S", "ONPLATE_ALONG_M", "TOP_NORMAL_MIN_Z", "SUPPORT_MIN_N", "SWING_MIN_STEPS", "LEADING_TOL_M",
    "EDGE_FORCE_MIN_N", "EDGE_STOP_BEFORE_M", "EDGE_STOP_AFTER_M", "BLOCKED_MIN_SWINGS", "GAIT_PERIOD_STEPS",
    "SLOW_MIN_PERIODS", "ADVANCE_MIN_PER_PERIOD_M", "ADVANCING_PERIOD_FRACTION", "SLOW_MAX_FRACTION",
    "WALL_DOMINANT_FRACTION")

# Fixed facts (not classifier thresholds).
DWELL_EPS_S = 1e-6                    # the >= 2.0 s comparison (bench times are accumulated floats)
TIME_EPS_S = 1e-9                     # edge_dwell.py's sample-window tolerance
GEOM_EPS_M = 1e-9                     # inclusive band edges
PLATE_HALF_M = v2.PLATE_HALF_M        # 0.24: radius (round) / half-side (square)
CLEAR_ALONG_M = v1.CLEAR_ALONG_M      # 0.35
INSIDE_MARGIN_M = 0.01                # inside_plate: the footprint shrunk by 1 cm (the smoke's round-plate check only)
OUTLINE_SAMPLES = 41                  # toe_to_edge: points per box edge (corners included; the edge midpoint exact)
STAGE_GAIT = "clocks2"
FIRST_WINDOW_S = 2.0                  # the control window: the first 2.0 s of every crossing
ROBOT_WEIGHT_N = 160.2                # 16.33 kg x 9.81 (compiled model, srun 21531556); the smoke re-measures it
FOOT_BODIES = {"left": "leg_left_ankle_roll", "right": "leg_right_ankle_roll"}
FORCE_KEYS = ("floor_N", "plate_top_N", "plate_edge_N", "other_plate_N", "other_plate_edge_N", "wall_N", "self_N",
              "other_N")
# Smoke-only machinery thresholds (never applied to a diagnosis crossing).
SMOKE_ROUND_PLATE_MIN_FOOT_STEPS = 10
SMOKE_ROUND_PLATE_MIN_LOADED_SHARE = 0.95
SMOKE_FLOOR_NORMAL_MIN_Z = 0.99

# Layout sets (module docstring).
BENCH_LAYOUTS = range(0, 32)                    # bench crossings / train layouts 0-31: never
RUN_LAYOUTS = range(170, 250)                   # design D as corrected: exploration train layouts 170-249 only
FIRST_FROZEN_RUN_LAYOUTS = range(250, 290)      # design D as first frozen (does not exist beyond 255; history only)
MAX_CROSSINGS = 80
CAP_RULE = ("the kept crossings in grid order (layout ascending, door 0 before door 1), after bench v2's drop rule; "
            "the first 80 are run, the rest are listed as capped out and never run")
SMOKE_CROSSINGS = ((32, 0), (33, 1), (38, 0))   # 0 deg round standstill, +90 round walking, -90 square standstill
SMOKE_LAYOUTS = tuple(sorted({layout for layout, _ in SMOKE_CROSSINGS}))
DIAG_SPLIT = "train"

LABELS = {"gaits": "LEARNED (shipped arms-dr1.0-s0 outside the stage; during the stage the controller is swapped to "
                   "arms-turngait-clock-s2 = clock-s2, Turning R1's qualified seed, a 77-observation gait-clock policy)",
          "stage": "SCRIPTED (PlateStage, stage_gait=clocks2, M2's turnboth law; bench v2's run_crossing unchanged)",
          "layout_and_plate_pose": "ORACLE (as the existing stage uses them)",
          "purpose": "DIAGNOSIS ONLY (investigation, no gate; nothing here is a bench verdict)"}
DESIGN = ("SLURM_JOBS.md, 'User approval recorded 2026-10-03 09:55', item (D) Mission 7 crossing diagnosis; 'Correction "
          "to design (D)' (layouts 170-249); 'Coordinator re-freeze of (D)'s measurement and classifier, 2026-10-03 "
          "11:40'")
BENCH_PROVENANCE = "results/mission7-campaign-20260923/clocks2-plate-bench-v2/provenance.json"
OWN_FILES = ("scripts/bench/m7_crossing_diag.py", "slurm/repo20260923/cpu_m7_crossing_diag.sbatch",
             "tests/test_m7_crossing_diag.py")
SOURCE_DRIFT_ENV = "M7_DIAG_ALLOW_SOURCE_DRIFT"
DISCLOSURES = (
    "layouts: 250-289 as first frozen does not exist beyond 255; corrected to 170-249 before any diagnosis episode "
    "(coordinator decision 2026-10-03, recorded in the ledger as a correction; SPLITS['train'] = (0, 256), so 250-289 "
    "would have yielded only 6 standstill crossings, L250-L255 d1)",
    "layouts: no Mission 7 crossing record exists for any layout in 170-249, so no run crossing has a public outcome; "
    "bench v2's construction yields 160 grid crossings there, its drop rule drops 61 walking ones, and of the 99 kept the "
    "first 80 in grid order (layout ascending, door 0 before door 1) are run: L170 d1 through L234 d0; 19 are capped out",
    "smoke: the instructed smoke layouts 290-299 do not exist; the smoke runs exploration train layouts 32, 33 and 38 "
    "(outside 170-249), which run mode refuses",
    "measurement and classifier re-frozen by the coordinator (2026-10-03 11:40) after an independent review of the "
    "smokes and synthetic checks only (no stall outcome): plate contacts classified by the contact normal (box-on-"
    "cylinder contacts sink, so contact heights misclassified round-plate support), and a blocked swing must be the "
    "leading foot's and show obstruction evidence (the earlier both-feet, low-swing test flagged false positives)",
    "classifier corrected by the coordinator (2026-10-04 10:45) after the implementer's audit of its own smoke records "
    "(layouts 32/33/38 only; no stall outcome): 4 of the 11 swings the re-frozen rule called blocked had touched down on "
    "the target plate top after brushing the edge face, so both obstruction evidences now also require a floor-side "
    "touchdown (the design's 'fails to get over the 3 cm edge')",
    "decision: precedence wall-blocked > blocked step-up > slow progress > other; falls excluded from the stall count and "
    "reported separately; specificity check on the clears; 0 non-fall stalls -> F1 NOT trained; INCOMPLETE decides "
    "nothing",
    "physics: the bench ran on haswell&el8 (cn-c21); this launcher runs on el9 nodes as instructed, so trajectories are "
    "not bit-comparable across CPU types (the layouts differ anyway)",
    "labels: the stage gait is LEARNED clock-s2, the stage SCRIPTED, layout and plate pose ORACLE",
)


def frozen_block(doc=None):
    """The FROZEN THRESHOLDS block of the docstring, line by line (the launcher header carries it verbatim)."""
    lines = (doc or __doc__).splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith("FROZEN THRESHOLDS"))
    end = next(i for i, line in enumerate(lines) if line.startswith("END FROZEN THRESHOLDS"))
    return lines[start:end + 1]


def frozen_block_values(lines):
    """{NAME: value text} of the block's 'NAME = value' lines (pure)."""
    out = {}
    for line in lines:
        match = re.match(r"^\s*([A-Z_]+) = (\S+)", line.lstrip("# "))
        if match:
            out[match.group(1)] = match.group(2)
    return out


def frozen_thresholds():
    return {name: globals()[name] for name in FROZEN_THRESHOLD_NAMES} | {"PRECEDENCE": list(PRECEDENCE)}


# ---- pure: layout guard and crossing enumeration --------------------------------------------------------------------

def check_layouts(mode, layouts, split=DIAG_SPLIT):
    """Refuse anything but the declared exploration layouts of ``mode`` (pure).  Returns the sorted layouts."""
    if mode not in ("run", "smoke"):
        raise ValueError(f"unknown mode {mode!r} (run|smoke)")
    if split != DIAG_SPLIT:
        raise ValueError(f"split {split!r} refused: exploration TRAIN layouts only; validation and test layouts are "
                         "never touched")
    layouts = [int(x) for x in layouts]
    if not layouts:
        raise ValueError("no layouts")
    if len(set(layouts)) != len(layouts):
        raise ValueError(f"duplicate layouts {layouts}")
    for layout in layouts:
        if layout in BENCH_LAYOUTS:
            raise ValueError(f"layout {layout} refused: train layouts 0-31 are the bench's (never touched)")
        if mode == "run" and layout not in RUN_LAYOUTS:
            raise ValueError(f"layout {layout} refused: run mode uses design D's exploration layouts "
                             f"{RUN_LAYOUTS.start}-{RUN_LAYOUTS.stop - 1} only")
        if mode == "smoke":
            if layout in RUN_LAYOUTS:
                raise ValueError(f"layout {layout} refused: {RUN_LAYOUTS.start}-{RUN_LAYOUTS.stop - 1} are reserved "
                                 "for the diagnosis run")
            if layout not in SMOKE_LAYOUTS:
                raise ValueError(f"layout {layout} refused: smoke layouts are {list(SMOKE_LAYOUTS)} only")
    return sorted(layouts)


def check_crossings(mode, crossings, split=DIAG_SPLIT):
    """(layout, door) pairs: guarded layouts, doors 0/1, no duplicates (pure)."""
    pairs = [(int(layout), int(door)) for layout, door in crossings]
    if not pairs:
        raise ValueError("no crossings")
    if len(set(pairs)) != len(pairs):
        raise ValueError(f"duplicate crossings {pairs}")
    if any(door not in v1.BENCH_DOORS for _, door in pairs):
        raise ValueError(f"doors must be 0/1: {pairs}")
    check_layouts(mode, sorted({layout for layout, _ in pairs}), split)
    return pairs


def existing_layouts(layouts):
    """(layouts that exist in the train split, layouts that do not) (pure)."""
    n = SPLITS[DIAG_SPLIT][1]
    return [x for x in layouts if 0 <= x < n], [x for x in layouts if not 0 <= x < n]


def crossing_id(spec):
    return "L%dd%d" % (int(spec["layout"]), int(spec["door"]))


def enumerate_crossings(mode, layouts=None, crossings=None, cap=MAX_CROSSINGS):
    """The crossings bench v2's construction yields on the declared layouts (grid labels, v2's drop rule, the cap).

    Run mode: ``layouts`` (default RUN_LAYOUTS) through the run guard; layouts missing from the train split are listed,
    not run.  Smoke mode: ``crossings`` (default SMOKE_CROSSINGS) through the smoke guard, grid labels, a walking
    crossing must pass the drop rule.  Returns the bookkeeping summary.json reports, ``specs`` in run order.
    """
    if mode == "run":
        layouts = check_layouts("run", list(RUN_LAYOUTS) if layouts is None else layouts)
        present, absent = existing_layouts(layouts)
        dropped = set(v2.dropped_crossings(tuple(present))) if present else set()
        grid = v1.grid(present)
        kept = [s for s in grid if (s["layout"], s["door"]) not in dropped]
        return {"mode": mode, "layouts_declared": layouts, "layouts_nonexistent": absent,
                "train_split_size": SPLITS[DIAG_SPLIT][1], "grid_crossings": len(grid),
                "dropped": sorted([list(c) for c in dropped]), "kept_before_cap": len(kept),
                "capped_out": [[s["layout"], s["door"]] for s in kept[cap:]], "cap": cap, "cap_rule": CAP_RULE,
                "specs": kept[:cap]}
    if mode != "smoke":
        raise ValueError(f"unknown mode {mode!r}")
    pairs = check_crossings("smoke", SMOKE_CROSSINGS if crossings is None else crossings)
    specs = []
    for layout_index, door in pairs:
        heading, plate, entry = v1.cell_of(layout_index, door)
        if entry == "walking" and (layout_index, door) in set(v2.dropped_crossings((layout_index,))):
            raise ValueError(f"smoke walking crossing L{layout_index} d{door} fails bench v2's drop rule")
        specs.append({"layout": layout_index, "door": door, "heading_deg": heading, "plate": plate, "entry": entry})
    return {"mode": mode, "layouts_declared": sorted({s["layout"] for s in specs}), "layouts_nonexistent": [],
            "train_split_size": SPLITS[DIAG_SPLIT][1], "grid_crossings": len(specs), "dropped": [],
            "kept_before_cap": len(specs), "capped_out": [[s["layout"], s["door"]] for s in specs[cap:]], "cap": cap,
            "cap_rule": CAP_RULE, "specs": specs[:cap]}


# ---- pure: plate and foot geometry, contact classification ----------------------------------------------------------

_SIGNS = np.array([[sx, sy, sz] for sx in (-1., 1.) for sy in (-1., 1.) for sz in (-1., 1.)])


def box_corners(centre, rotation, half):
    """The 8 world corners of a box geom (centre, 3x3 world rotation, half sizes)."""
    return (np.asarray(centre, dtype=float)
            + (_SIGNS * np.asarray(half, dtype=float)) @ np.asarray(rotation, dtype=float).T)


def near_edge_along(lateral, shape, half=PLATE_HALF_M):
    """Along (from the plate centre, door direction) of the plate's NEAR edge at a lateral offset; None beside it.

    Round: -sqrt(r^2 - lat^2) for |lat| < r.  Square (axis-aligned, as every door direction is): -half for |lat| <= half.
    """
    lateral = abs(float(lateral))
    if shape == "round":
        return None if lateral >= half else -float(np.sqrt(half * half - lateral * lateral))
    if shape == "square":
        return None if lateral > half else -float(half)
    raise ValueError(f"unknown plate shape {shape!r}")


def near_edge_along_array(lateral, shape, half=PLATE_HALF_M):
    """near_edge_along over an array; NaN beside the plate (pure)."""
    lat = np.abs(np.asarray(lateral, dtype=float))
    if shape == "round":
        out = np.full(lat.shape, np.nan)
        inside = lat < half
        out[inside] = -np.sqrt(half * half - lat[inside] ** 2)
        return out
    if shape == "square":
        return np.where(lat <= half, -float(half), np.nan)
    raise ValueError(f"unknown plate shape {shape!r}")


_BOX_EDGES = np.array([(i, j) for i in range(8) for j in range(i + 1, 8) if int(np.sum(_SIGNS[i] != _SIGNS[j])) == 1])
_OUTLINE_S = np.linspace(0., 1., OUTLINE_SAMPLES)


def outline_toe_to_edge(along, lateral, shape):
    """max over the box outline (12 edges x OUTLINE_SAMPLES points) of (along - near-edge along at that lateral offset);
    None if every point is beside the plate (pure; plate frame coordinates of the 8 corners)."""
    along, lateral = np.asarray(along, dtype=float), np.asarray(lateral, dtype=float)
    i, j = _BOX_EDGES[:, 0], _BOX_EDGES[:, 1]
    point_along = along[i][:, None] + (along[j] - along[i])[:, None] * _OUTLINE_S[None, :]
    point_lateral = lateral[i][:, None] + (lateral[j] - lateral[i])[:, None] * _OUTLINE_S[None, :]
    margin = point_along - near_edge_along_array(point_lateral, shape)
    return None if np.all(np.isnan(margin)) else float(np.nanmax(margin))


def inside_footprint(along, lateral, shape, margin=INSIDE_MARGIN_M, half=PLATE_HALF_M):
    """Every planar point (plate frame) inside the plate's footprint shrunk by ``margin`` (pure)."""
    along, lateral = np.asarray(along, dtype=float), np.asarray(lateral, dtype=float)
    limit = half - margin + GEOM_EPS_M   # inclusive
    if shape == "round":
        return bool(np.all(np.hypot(along, lateral) <= limit))
    if shape == "square":
        return bool(np.all((np.abs(along) <= limit) & (np.abs(lateral) <= limit)))
    raise ValueError(f"unknown plate shape {shape!r}")


def foot_metrics(corners, plate_xy, direction, shape):
    """Toe/heel along, toe_to_edge, heights and inside_plate of one foot box (pure)."""
    corners = np.asarray(corners, dtype=float)
    d = np.asarray(direction, dtype=float)
    lateral_dir = np.array([-d[1], d[0]])
    rel = corners[:, :2] - np.asarray(plate_xy, dtype=float)
    along, lateral = rel @ d, rel @ lateral_dir
    order = np.argsort(-along, kind="stable")
    return {"toe_along_m": float(along.max()), "heel_along_m": float(along.min()),
            "toe_lateral_m": float(lateral[order[0]]),
            "toe_to_edge_m": outline_toe_to_edge(along, lateral, shape),
            "toe_z_m": float(corners[order[:4], 2].min()), "sole_min_z_m": float(corners[:, 2].min()),
            "inside_plate": inside_footprint(along, lateral, shape)}


def world_geom_class(name, target_plate):
    """floor | target_plate | other_plate | wall | other, from a world geom's name (pure)."""
    if name == "floor":
        return "floor"
    if name.startswith("plate_"):
        return "target_plate" if name == target_plate else "other_plate"
    if name.startswith(("wall_", "door_", "goal_post")):
        return "wall"
    return "other"


def upward_normal(frame_normal, world_is_geom1):
    """MuJoCo's contact normal (frame[0:3]) points from geom1 to geom2; return it pointing from the world geom into the
    foot (pure)."""
    n = np.asarray(frame_normal, dtype=float)
    return n if world_is_geom1 else -n


def contact_key(world_class, n_up_z):
    """The force bucket of one foot contact (pure): plate contacts split by their normal at TOP_NORMAL_MIN_Z."""
    if world_class == "target_plate":
        return "plate_top_N" if n_up_z >= TOP_NORMAL_MIN_Z else "plate_edge_N"
    if world_class == "other_plate":
        return "other_plate_N" if n_up_z >= TOP_NORMAL_MIN_Z else "other_plate_edge_N"
    return {"floor": "floor_N", "wall": "wall_N", "self": "self_N"}.get(world_class, "other_N")


def support_force(foot):
    return float(foot["floor_N"] + foot["plate_top_N"] + foot["other_plate_N"])


def is_loaded(foot):
    return bool(support_force(foot) >= SUPPORT_MIN_N)


# ---- pure: edge dwell, window, swings, features, criteria, classification -----------------------------------------

def edge_dwell(times, along, history, takeover_s):
    """edge_dwell.py's dwell, the stall flag and the window W (pure).  ``along`` = base along per bench sample."""
    t = np.asarray(times, dtype=float)
    along = np.asarray(along, dtype=float)
    cross = [h for h in history if h.get("phase") == "cross"]
    if takeover_s is None or not cross or not t.size:
        return {"no_cross": True, "stall": False}
    cs = float(cross[0]["time_s"])
    tb = next((float(h["time_s"]) for h in history if h.get("phase") in ("turn_back", "recorded")
               and h["time_s"] >= cs), None)
    after = t >= cs - TIME_EPS_S
    if not after.any():
        return {"no_cross": True, "stall": False, "note": "no sample after the cross entry"}
    on = np.flatnonzero(after & (along > ONPLATE_ALONG_M))
    clear = np.flatnonzero(after & (along >= CLEAR_ALONG_M))
    t_on = float(t[on[0]]) if on.size else None
    dwell = None if t_on is None else t_on - cs
    stall = t_on is None or dwell >= STALL_DWELL_S - DWELL_EPS_S
    end = min([x for x in (t_on, tb) if x is not None], default=float(t[-1]))
    i0 = int(np.flatnonzero(after)[0])
    within = np.flatnonzero(after & (t <= end + TIME_EPS_S))
    i1 = int(within[-1]) if within.size else i0
    start_index = int(np.argmin(np.abs(t - cs)))
    return {"no_cross": False, "stall": bool(stall), "cross_start_s": cs, "cross_end_s": tb,
            "cross_start_after_takeover_s": cs - float(takeover_s), "along_cross_start_m": float(along[start_index]),
            "t_onplate_s": t_on, "dwell_s": dwell, "never_past": t_on is None,
            "onplate_in_cross": None if t_on is None or tb is None else bool(t_on <= tb + TIME_EPS_S),
            "t_first_along_clear_s": float(t[clear[0]]) if clear.size else None,
            "window": {"start_s": float(t[i0]), "end_s": float(t[i1]), "i0": i0, "i1": i1, "steps": i1 - i0 + 1}}


def unloaded_runs(loaded, min_steps=SWING_MIN_STEPS):
    """[(first, last)] of maximal runs of >= min_steps consecutive unloaded steps (pure)."""
    runs, start = [], None
    for i, flag in enumerate(list(loaded) + [True]):
        if not flag and start is None:
            start = i
        elif flag and start is not None:
            if i - start >= min_steps:
                runs.append((start, i - 1))
            start = None
    return runs


def is_leading(swing):
    """The swinging foot's toe at or ahead of the other foot's toe at touchdown (pure)."""
    return bool(swing["touchdown_toe_along_m"] >= swing["other_toe_along_m"] - LEADING_TOL_M - GEOM_EPS_M)


def has_edge_face_obstruction(swing):
    return bool(swing["edge_face_max_N"] >= EDGE_FORCE_MIN_N)


def has_toe_stopped_at_edge(swing):
    """Touchdown with the toe in the frozen band at the edge, on the floor side and not on the plate top (pure)."""
    tte = swing.get("touchdown_toe_to_edge_m")
    return bool(tte is not None
                and -EDGE_STOP_BEFORE_M - GEOM_EPS_M <= tte <= EDGE_STOP_AFTER_M + GEOM_EPS_M
                and swing["touchdown_floor_N"] >= SUPPORT_MIN_N
                and swing["touchdown_plate_top_N"] < SUPPORT_MIN_N)


def touchdown_on_floor_side(swing):
    """The swing touched down on the floor side, not on the target plate top (pure; correction 2026-10-04 10:45)."""
    return bool(swing["touchdown_floor_N"] >= SUPPORT_MIN_N and swing["touchdown_plate_top_N"] < SUPPORT_MIN_N)


def is_blocked_swing(swing):
    """The frozen blocked swing: the leading foot's, touching down on the floor side, with obstruction evidence (pure;
    module docstring)."""
    return bool(is_leading(swing) and touchdown_on_floor_side(swing)
                and (has_edge_face_obstruction(swing) or has_toe_stopped_at_edge(swing)))


def foot_swings(steps, side):
    """Every swing of one foot over the whole trace (pure over diagnosis records), with its touchdown and evidence."""
    other_side = "right" if side == "left" else "left"
    feet = [s["feet"][side] for s in steps]
    others = [s["feet"][other_side] for s in steps]
    out = []
    for a, b in unloaded_runs([is_loaded(f) for f in feet]):
        c = b + 1
        if c >= len(steps):   # still unloaded when the trace ends: no touchdown
            continue
        reach = [f["toe_to_edge_m"] for f in feet[a:c + 1] if f["toe_to_edge_m"] is not None]
        swing = {"foot": side, "start_s": steps[a]["t"], "end_s": steps[b]["t"], "touchdown_s": steps[c]["t"],
                 "touchdown_index": c, "steps": b - a + 1, "touchdown_phase": steps[c]["phase"],
                 "touchdown_toe_along_m": feet[c]["toe_along_m"], "other_toe_along_m": others[c]["toe_along_m"],
                 "touchdown_toe_to_edge_m": feet[c]["toe_to_edge_m"],
                 "touchdown_floor_N": feet[c]["floor_N"], "touchdown_plate_top_N": feet[c]["plate_top_N"],
                 "edge_face_max_N": max(f["plate_edge_N"] for f in feet[a:c + 1]),
                 "max_toe_to_edge_m": max(reach) if reach else None,
                 "max_toe_z_m": max(f["toe_z_m"] for f in feet[a:b + 1])}   # reported only
        swing["leading"] = is_leading(swing)
        swing["touchdown_floor_side"] = touchdown_on_floor_side(swing)
        swing["edge_face_obstruction"] = has_edge_face_obstruction(swing)
        swing["toe_stopped_at_edge"] = has_toe_stopped_at_edge(swing)
        swing["blocked"] = is_blocked_swing(swing)
        out.append(swing)
    return out


def window_features(steps, i0, i1):
    """Classifier features over steps[i0..i1] (pure; a swing counts when its touchdown lies inside the window)."""
    i1 = max(i0, i1)
    sub = steps[i0:i1 + 1]
    swings = {side: [s for s in foot_swings(steps, side) if i0 <= s["touchdown_index"] <= i1] for side in FOOT_BODIES}
    every = [s for v in swings.values() for s in v]
    t = [s["t"] for s in sub]
    along = [s["base_along_m"] for s in sub]
    periods = [along[k + GAIT_PERIOD_STEPS] - along[k]
               for k in range(0, len(sub) - GAIT_PERIOD_STEPS, GAIT_PERIOD_STEPS)]
    duration = t[-1] - t[0] if len(t) > 1 else 0.
    return {
        "window_steps": len(sub), "window_s": duration,
        "n_swings": len(every),
        "n_leading_swings": sum(s["leading"] for s in every),
        "n_edge_face_swings": sum(s["edge_face_obstruction"] for s in every),
        "n_toe_stop_swings": sum(s["toe_stopped_at_edge"] for s in every),
        "n_obstructed_swings": sum(s["edge_face_obstruction"] or s["toe_stopped_at_edge"] for s in every),
        "n_blocked_swings": sum(s["blocked"] for s in every),
        "per_foot": {side: {"swings": len(v), "leading": sum(s["leading"] for s in v),
                            "edge_face": sum(s["edge_face_obstruction"] for s in v),
                            "toe_stop": sum(s["toe_stopped_at_edge"] for s in v),
                            "blocked": sum(s["blocked"] for s in v)} for side, v in swings.items()},
        "max_toe_z_per_swing_m": {side: [round(s["max_toe_z_m"], 4) for s in v] for side, v in swings.items()},
        "touchdown_toe_to_edge_m": {side: [None if s["touchdown_toe_to_edge_m"] is None
                                           else round(s["touchdown_toe_to_edge_m"], 4) for s in v]
                                    for side, v in swings.items()},
        "touchdowns_on_plate_top": {side: sum(s["touchdown_plate_top_N"] >= SUPPORT_MIN_N for s in v)
                                    for side, v in swings.items()},
        "plate_top_loaded_steps": {side: sum(s["feet"][side]["plate_top_N"] >= SUPPORT_MIN_N for s in sub)
                                   for side in FOOT_BODIES},
        "edge_face_force_steps": {side: sum(s["feet"][side]["plate_edge_N"] >= EDGE_FORCE_MIN_N for s in sub)
                                  for side in FOOT_BODIES},
        "period_advances_m": periods,
        "mean_along_speed_mps": (along[-1] - along[0]) / duration if duration > 0 else 0.,
        "mean_cmd_vx_mps": float(np.mean([s["cmd"][0] for s in sub])) if sub else 0.,
        "mean_meas_vx_body_mps": float(np.mean([s["v_body"][0] for s in sub])) if sub else 0.,
        "mean_cmd_wz_rps": float(np.mean([s["cmd"][2] for s in sub])) if sub else 0.,
        "mean_meas_wz_rps": float(np.mean([s["wz"] for s in sub])) if sub else 0.,
        "wall_steps": sum(bool(s["wall_instant"]) for s in sub),
        "along_start_m": along[0] if along else None, "along_end_m": along[-1] if along else None,
        "swings": swings,
    }


def stall_criteria(features):
    """Each class's own criterion (pure; module docstring)."""
    periods = list(features["period_advances_m"])
    advancing = sum(a >= ADVANCE_MIN_PER_PERIOD_M - GEOM_EPS_M for a in periods)
    cmd = float(features["mean_cmd_vx_mps"])
    steps = int(features["window_steps"])
    return {
        "wall_blocked": bool(steps > 0 and int(features["wall_steps"]) >= WALL_DOMINANT_FRACTION * steps - 1e-12),
        "blocked_step_up": int(features["n_blocked_swings"]) >= BLOCKED_MIN_SWINGS,
        "slow_progress": bool(len(periods) >= SLOW_MIN_PERIODS
                              and advancing >= ADVANCING_PERIOD_FRACTION * len(periods) - 1e-12
                              and cmd > 0 and float(features["mean_along_speed_mps"]) < SLOW_MAX_FRACTION * cmd),
    }


def classify(features):
    """The primary class of a stall by PRECEDENCE (pure)."""
    criteria = stall_criteria(features)
    return next((name for name in PRECEDENCE[:-1] if criteria[name]), "other")


def _ratio(numerator, denominator):
    return {"numerator": numerator, "denominator": denominator,
            "value": (numerator / denominator) if denominator else None,
            "text": f"{numerator}/{denominator} = {numerator / denominator:.6f}" if denominator
            else f"{numerator}/0 (none)"}


def decide(rows, expected, failures=(), drift_changed=False):
    """The frozen decision over the run's rows (pure; module docstring DECISION)."""
    failures = list(failures)
    complete = not failures and not drift_changed and len(rows) == expected and expected > 0
    stalls = [r for r in rows if r.get("stall")]
    nonfall = [r for r in stalls if not r.get("fell")]
    fall_stalls = [r for r in stalls if r.get("fell")]
    clears = [r for r in rows if r.get("clear")]
    blocked = sum(r["cls"] == "blocked_step_up" for r in nonfall)
    blocked_all = sum(r["cls"] == "blocked_step_up" for r in stalls)
    meeting = sum(bool((r.get("criteria") or {}).get("blocked_step_up")) for r in clears)
    out = {"status": "COMPLETE" if complete else "INCOMPLETE", "crossings_expected": expected,
           "crossings_complete": len(rows), "failures": failures, "source_changed_during_run": bool(drift_changed),
           "falls": sum(bool(r.get("fell")) for r in rows), "stalls_all": len(stalls), "stalls_non_fall": len(nonfall),
           "stalls_fell": len(fall_stalls),
           "class_counts_non_fall": {name: sum(r["cls"] == name for r in nonfall) for name in PRECEDENCE},
           "class_counts_fall_stalls": {name: sum(r["cls"] == name for r in fall_stalls) for name in PRECEDENCE},
           "blocked_fraction_non_fall": _ratio(blocked, len(nonfall)),
           "blocked_fraction_including_falls": _ratio(blocked_all, len(stalls)),
           "specificity_clears_meeting_criterion": _ratio(meeting, len(clears))}
    if not complete:
        out.update(f1_trained=None, reason=(f"{len(rows)} of {expected} crossings complete"
                                            + (f"; failures {[f['crossing'] for f in failures]}" if failures else "")
                                            + ("; an imported source changed during the run" if drift_changed else "")
                                            + "; decides nothing"))
    elif not nonfall:
        out.update(f1_trained=False, reason="0 non-fall stalls")
    elif 3 * blocked < len(nonfall):
        out.update(f1_trained=False, reason=f"non-fall blocked fraction {blocked}/{len(nonfall)} < 1/3")
    elif not clears:
        out.update(f1_trained=False, reason="0 clears: the specificity check cannot hold")
    elif not meeting * len(nonfall) < blocked * len(clears):
        out.update(f1_trained=False, reason=(f"specificity fails: clear share {meeting}/{len(clears)} is not lower "
                                             f"than the non-fall fraction {blocked}/{len(nonfall)}"))
    else:
        out.update(f1_trained=True, reason=(f"non-fall fraction {blocked}/{len(nonfall)} >= 1/3 and clear share "
                                            f"{meeting}/{len(clears)} is lower"))
    return out


def decision_lines(decision, label="M7_CROSSING_DIAG", smoke=False):
    """The printed decision lines (pure).  A complete run's DECISION line always carries the non-fall blocked-step-up
    fraction and the specificity clear share; no line prints 'undefined' or a True/False flag."""
    d = decision
    tag = " (SMOKE: machinery only, never a decision)" if smoke else ""
    fraction = d["blocked_fraction_non_fall"]["text"]
    share = d["specificity_clears_meeting_criterion"]["text"]
    lines = [f"{label} STALLS non-fall {d['stalls_non_fall']} (fall stalls {d['stalls_fell']}, excluded; crossings that "
             f"fell {d['falls']}) class_counts_non_fall {json.dumps(d['class_counts_non_fall'])} "
             f"class_counts_fall_stalls {json.dumps(d['class_counts_fall_stalls'])}{tag}",
             f"{label} BLOCKED_STEP_UP_FRACTION non-fall {fraction}; including fall "
             f"stalls {d['blocked_fraction_including_falls']['text']}{tag}",
             f"{label} SPECIFICITY clears meeting the blocked-step-up criterion on their own edge dwell {share}{tag}"]
    if d["status"] != "COMPLETE":
        lines.append(f"{label} DECISION INCOMPLETE ({d['reason']}){tag}")
    else:
        lines.append(f"{label} DECISION F1 {'TRAINED' if d['f1_trained'] else 'NOT TRAINED'} ({d['reason']}); "
                     f"non-fall blocked-step-up fraction {fraction}; clear share meeting the criterion {share}{tag}")
    return lines


# ---- the read-only recorder around approach_debug.physical_sample ---------------------------------------------------

class FootRecorder:
    """Per-policy-step diagnosis records of one crossing; reads MjData only."""

    def __init__(self, geometry, plate_shape, door):
        self.plate_xy = np.asarray(geometry["plate_xy"], dtype=float)
        self.direction = np.asarray(geometry["direction"], dtype=float)
        if sorted(np.abs(self.direction).round(9).tolist()) != [0., 1.]:
            raise ValueError(f"door direction {self.direction.tolist()} is not axis-aligned")
        self.door_yaw = float(np.arctan2(self.direction[1], self.direction[0]))
        self.shape = plate_shape
        self.target_plate = f"plate_{int(door)}_{int(geometry['side'])}"
        self.steps = []
        self.feet = None
        self.model_id = None
        self.names = None

    def resolve(self, env):
        """The two sole boxes (one collision box per ankle-roll link), resolved once per model."""
        import mujoco
        model = env.model
        if self.model_id == id(model):
            return
        prefix = env.slot.prefix
        feet = {}
        for side, body_name in FOOT_BODIES.items():
            body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, prefix + body_name)
            if body < 0:
                raise RuntimeError(f"no body {prefix + body_name}")
            boxes = [g for g in range(model.ngeom) if int(model.geom_bodyid[g]) == body
                     and int(model.geom_type[g]) == int(mujoco.mjtGeom.mjGEOM_BOX)
                     and (int(model.geom_contype[g]) or int(model.geom_conaffinity[g]))]
            if len(boxes) != 1:
                raise RuntimeError(f"{prefix + body_name}: {len(boxes)} collision boxes, expected 1")
            feet[side] = {"body": int(body), "geom": int(boxes[0]),
                          "half": np.asarray(model.geom_size[boxes[0]], dtype=float).copy()}
        self.feet = feet
        self.names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, g) or "" for g in range(model.ngeom)]
        self.model_id = id(model)

    def record(self, env, command, phase, sample):
        """One diagnosis record after physical_sample returned ``sample`` (which is not modified)."""
        import mujoco
        self.resolve(env)
        runner = env.runner
        d, model = runner.d, env.model
        geom_side = {f["geom"]: side for side, f in self.feet.items()}
        feet = {side: dict.fromkeys(FORCE_KEYS, 0.) | {"contacts": []} for side in self.feet}
        force = np.zeros(6)
        for i in range(int(d.ncon)):
            contact = d.contact[i]
            g1, g2 = int(contact.geom1), int(contact.geom2)
            for mine, other in ((g1, g2), (g2, g1)):
                side = geom_side.get(mine)
                if side is None:
                    continue
                normal = 0.
                if int(contact.efc_address) >= 0:
                    mujoco.mj_contactForce(model, d, i, force)
                    normal = float(force[0])
                name = self.names[other] or f"geom{other}"
                world = "self" if bool(runner.own[other]) else world_geom_class(name, self.target_plate)
                n_up_z = float(upward_normal(contact.frame[:3], world_is_geom1=(other == g1))[2])
                feet[side][contact_key(world, n_up_z)] += normal
                feet[side]["contacts"].append([name, round(float(contact.pos[2]), 5), round(normal, 3),
                                               round(n_up_z, 4), round(float(contact.dist), 5)])
        for side, f in self.feet.items():
            rotation = np.asarray(d.geom_xmat[f["geom"]], dtype=float).reshape(3, 3)
            metrics = foot_metrics(box_corners(d.geom_xpos[f["geom"]], rotation, f["half"]), self.plate_xy,
                                   self.direction, self.shape)
            out = feet[side]
            for key in FORCE_KEYS:
                out[key] = round(float(out[key]), 3)
            out.update({key: (value if value is None or isinstance(value, bool) else round(value, 6))
                        for key, value in metrics.items()})
            out["ankle_xyz"] = [round(float(x), 6) for x in d.xpos[f["body"]]]
            out["box_xyz"] = [round(float(x), 6) for x in d.geom_xpos[f["geom"]]]
            out["loaded"] = is_loaded(out)
        rel = np.asarray(sample["xy"], dtype=float) - self.plate_xy
        yaw = float(sample["yaw"])
        self.steps.append({
            "t": float(sample["time_s"]), "phase": str(phase),
            "base_along_m": round(float(rel @ self.direction), 6),
            "base_lateral_m": round(float(rel @ np.array([-self.direction[1], self.direction[0]])), 6),
            "yaw_err_rad": round(float(np.arctan2(np.sin(self.door_yaw - yaw), np.cos(self.door_yaw - yaw))), 6),
            "cmd": [round(float(x), 6) for x in np.asarray(command, dtype=float)[:3]],
            "v_body": [round(float(x), 6) for x in sample["velocity_body"]],
            "v_along_mps": round(float(np.asarray(sample["velocity"], dtype=float) @ self.direction), 6),
            "wz": round(float(sample["yaw_rate"]), 6),
            "tilt": round(float(sample["tilt"]), 6),
            "wall_instant": list(sample["contacts"]),
            "wall_any_substep": bool(getattr(runner, "wall_contact", False)),
            "feet": feet,
        })


@contextlib.contextmanager
def recording(recorder):
    """Wrap approach_debug.physical_sample for the duration; the original is restored in finally."""
    import bhl_robust.mission.approach_debug as approach_debug
    original = approach_debug.physical_sample
    if getattr(original, "_m7_crossing_diag", False):
        raise RuntimeError("physical_sample is already wrapped")

    def recording_physical_sample(env, command, phase):
        sample = original(env, command, phase)
        recorder.record(env, command, phase, sample)
        return sample

    recording_physical_sample._m7_crossing_diag = True
    approach_debug.physical_sample = recording_physical_sample
    try:
        yield recorder
    finally:
        approach_debug.physical_sample = original


def run_plain(repo, cache, spec):
    """bench v2's crossing, unwrapped (the identity reference)."""
    return v2.run_crossing(repo, cache, spec, STAGE_GAIT)


def run_instrumented(repo, cache, spec):
    """(bench v2 result, diagnosis records): v2.run_crossing unchanged, with the recorder installed."""
    geometry = v2.crossing_geometry(generate(DIAG_SPLIT, int(spec["layout"])), int(spec["door"]), spec["plate"],
                                    spec["heading_deg"], spec["entry"])
    recorder = FootRecorder(geometry, spec["plate"], spec["door"])
    with recording(recorder):
        result = v2.run_crossing(repo, cache, spec, STAGE_GAIT)
    samples = result["samples"]
    if len(recorder.steps) != len(samples) or any(abs(r["t"] - s["time_s"]) > 1e-12
                                                   for r, s in zip(recorder.steps, samples)):
        raise RuntimeError(f"diagnosis records ({len(recorder.steps)}) are not 1:1 with the bench samples "
                           f"({len(samples)})")
    if result["geometry"] != geometry:
        raise RuntimeError("the bench's crossing geometry differs from the recorder's")
    return result, recorder.steps


# ---- per-crossing analysis ------------------------------------------------------------------------------------------

def _compact(features):
    return {key: value for key, value in features.items() if key != "swings"}


def analyse_crossing(result, steps):
    """The summary row of one crossing (pure over the bench result and the diagnosis records)."""
    g = result["geometry"]
    plate_xy, direction = np.asarray(g["plate_xy"], dtype=float), np.asarray(g["direction"], dtype=float)
    samples = result["samples"]
    t = [s["time_s"] for s in samples]
    along = (np.asarray([s["xy"] for s in samples], dtype=float).reshape(-1, 2) - plate_xy) @ direction
    dwell = edge_dwell(t, along, result["stage_history"], result["takeover_s"])
    row = {key: result.get(key) for key in ("layout", "door", "heading_deg", "plate", "entry", "end_reason", "clear",
                                            "real_clear", "clear_after_takeover_s", "fell", "fall_phase", "takeover_s",
                                            "handback_s", "maximum_tilt", "cell_m")}
    row["wall_contact_stage_samples"] = result["wall_contacts"]["stage_samples"]
    row["turn_timed_out"] = any(bool(h.get("turn_timed_out")) for h in result["stage_history"])
    row["edge"] = {key: value for key, value in dwell.items() if key != "window"}
    if dwell["no_cross"]:
        row.update(stall=False, cls=None, control_cls=None, note="no cross phase (never a stall)")
        return row
    w = dwell["window"]
    features = window_features(steps, w["i0"], w["i1"])
    limit = dwell["cross_start_s"] + FIRST_WINDOW_S
    if dwell["cross_end_s"] is not None:
        limit = min(limit, dwell["cross_end_s"])
    first_end = max([i for i, s in enumerate(steps) if s["t"] <= limit + TIME_EPS_S], default=w["i0"])
    first = window_features(steps, w["i0"], max(w["i0"], first_end))
    cls = classify(features)
    row.update(stall=dwell["stall"], window=w,
               cls=cls if dwell["stall"] else None, control_cls=None if dwell["stall"] else cls,
               criteria=stall_criteria(features), criteria_first_2s=stall_criteria(first),
               control_cls_first_2s=classify(first), features=_compact(features), features_first_2s=_compact(first),
               swings=features["swings"],
               t_clear_after_takeover_s=(None if dwell["t_first_along_clear_s"] is None
                                         else dwell["t_first_along_clear_s"] - float(result["takeover_s"])))
    return row


def summarise(rows):
    """Counts, overlaps and controls (pure).  The decision itself is decide()'s."""
    stalls = [r for r in rows if r.get("stall")]
    nonfall = [r for r in stalls if not r.get("fell")]
    controls = [r for r in rows if not r.get("stall") and r.get("control_cls") is not None]
    clears = [r for r in rows if r.get("clear") and "features" in r]
    return {
        "crossings": len(rows), "clears": sum(bool(r.get("clear")) for r in rows),
        "falls": sum(bool(r.get("fell")) for r in rows),
        "no_cross": sum(bool(r["edge"].get("no_cross")) for r in rows),
        "stalls_all": len(stalls), "stalls_non_fall": len(nonfall), "stalls_fell": len(stalls) - len(nonfall),
        "stalls_that_cleared": sum(bool(r.get("clear")) for r in stalls),
        "stalls_never_past": sum(bool(r["edge"].get("never_past")) for r in stalls),
        "stall_classes": {crossing_id(r): r["cls"] for r in stalls},
        "fell_stall_ids": [crossing_id(r) for r in stalls if r.get("fell")],
        "criteria_overlaps_non_fall_stalls": {
            "per_criterion": {name: sum(bool(r["criteria"][name]) for r in nonfall) for name in PRECEDENCE[:-1]},
            "combinations": _combination_counts(nonfall)},
        "controls": {
            "non_stall_crossings_with_a_cross": len(controls),
            "control_classes_on_W": {name: sum(r["control_cls"] == name for r in controls) for name in PRECEDENCE},
            "control_classes_first_2s": {name: sum(r["control_cls_first_2s"] == name for r in controls)
                                         for name in PRECEDENCE},
            "clears_blocked_swings_on_W": {crossing_id(r): r["features"]["n_blocked_swings"] for r in clears},
            "clears_blocked_swings_first_2s": {crossing_id(r): r["features_first_2s"]["n_blocked_swings"]
                                               for r in clears},
            "stalls_blocked_swings_first_2s": {crossing_id(r): r["features_first_2s"]["n_blocked_swings"]
                                               for r in stalls},
        },
    }


def _combination_counts(rows):
    out = {}
    for r in rows:
        key = "+".join(name for name in PRECEDENCE[:-1] if r["criteria"][name]) or "none"
        out[key] = out.get(key, 0) + 1
    return dict(sorted(out.items()))


# ---- provenance ------------------------------------------------------------------------------------------------------

def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def imported_files(root):
    """Files of every imported module under ``root`` (resolved; sorted, relative).  Modules whose __file__ is not an
    absolute path to a file (e.g. torch._classes / torch._ops report "_classes.py" / "_ops.py") are skipped: resolving
    a relative name would anchor it at the working directory."""
    root = Path(root).resolve()
    out = set()
    for module in list(sys.modules.values()):
        path = getattr(module, "__file__", None)
        if not isinstance(path, str) or not os.path.isabs(path) or not os.path.isfile(path):
            continue
        try:
            out.add(str(Path(path).resolve().relative_to(root)))
        except ValueError:
            continue
    return sorted(out)


def source_check(repo, bench_provenance=None):
    """sha256 of the bench's sources, this workstream's files and every imported repo module, compared with the
    clocks2 bench v2 provenance (21517668).  ``relevant_drift`` = imported files whose hash differs from the bench's."""
    repo = Path(repo).resolve()
    bench = json.loads((Path(bench_provenance) if bench_provenance else repo / BENCH_PROVENANCE).read_text())
    recorded = bench["sources_sha256"]
    imported = imported_files(repo)
    names = sorted(set(recorded) | set(OWN_FILES) | set(imported))
    current = {name: sha256(repo / name) for name in names if (repo / name).is_file()}
    differing = sorted(name for name, digest in recorded.items() if current.get(name) != digest)
    upstream = repo / "external/Berkeley-Humanoid-Lite"
    upstream_files = imported_files(upstream) if upstream.exists() else []
    return {"bench_provenance": BENCH_PROVENANCE, "bench_git_head": bench.get("git_head"),
            "bench_slurm_job_id": bench.get("slurm_job_id"), "sources_sha256": current,
            "bench_sources_checked": len(recorded), "bench_sources_differing": differing,
            "identical_to_bench": not differing,
            "relevant_drift": sorted(name for name in differing if name in imported),
            "imported_repo_files": imported,
            "imported_not_in_bench_provenance": sorted(set(imported) - set(recorded) - set(OWN_FILES)),
            "imported_upstream_sha256": {name: sha256(upstream.resolve() / name) for name in upstream_files},
            "bench_weights_sha256": bench.get("weights_sha256")}


def sources_changed(start, end):
    """What changed in the imported sources between the start and the end source_check() of a run (pure).

    changed: an imported file hashed at both times whose sha256 differs, or that vanished.  relevant_drift_changed: the
    imported files that differ from the bench's are not the same set (this also catches a module first imported during
    the run whose bench-recorded hash differs).  imported_during_run: modules first imported after the start check;
    recorded with their end hash, not by themselves a change (no start hash exists to compare).  ``any`` makes the run
    INCOMPLETE.
    """
    before, after = start["sources_sha256"], end["sources_sha256"]
    watched = set(start["imported_repo_files"]) | set(end["imported_repo_files"])
    changed = sorted(name for name in watched if name in before and after.get(name) != before[name])
    drift = sorted(end["relevant_drift"]) != sorted(start["relevant_drift"])
    return {"changed": changed, "relevant_drift_changed": drift,
            "imported_during_run": sorted(set(end["imported_repo_files"]) - set(start["imported_repo_files"])),
            "any": bool(changed or drift)}


def weights(repo):
    """Both policies' sha256 (clock-s2 through the stage's own pinned export check)."""
    from mission7_plate_stage import SHIPPED_EXPORT, export_check, export_dir_for
    upstream = Path(repo) / "external/Berkeley-Humanoid-Lite"
    info = export_check(upstream, *export_dir_for(upstream, STAGE_GAIT))
    return {"shipped": sha256(upstream / SHIPPED_EXPORT / "policy.onnx"), "clocks2": info["policy_sha256"],
            "clocks2_deploy_yaml": info["deploy_yaml_sha256"], "export_check": info}


def git_head(repo):
    import subprocess
    try:
        return subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True,
                              timeout=60).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        return f"unavailable: {exc}"


def import_simulation_stack():
    """Import every module a crossing uses before any episode, so the source check sees them all."""
    import bhl_robust.eval.gait_clock  # noqa: F401
    import bhl_robust.mission.approach_debug  # noqa: F401
    import bhl_robust.mission.env  # noqa: F401
    import mission7_plate_stage  # noqa: F401


# ---- smoke machinery checks and statistics --------------------------------------------------------------------------

def measured_robot_weight(env):
    import mujoco
    model = env.model
    mass = sum(float(model.body_mass[b]) for b in range(model.nbody)
               if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or "").startswith(env.slot.prefix))
    return mass * float(-model.opt.gravity[2])


def identity_check(repo, out, spec):
    """The first smoke crossing run unwrapped, then wrapped: the bench records must be byte-identical."""
    name = f"L{int(spec['layout']):03d}-d{int(spec['door'])}"
    plain = run_plain(repo, out / "cache-identity-plain" / name, spec)
    wrapped, steps = run_instrumented(repo, out / "cache" / name, spec)
    a, b = json.dumps(plain, sort_keys=True), json.dumps(wrapped, sort_keys=True)
    return ({"crossing": [spec["layout"], spec["door"]], "equal": a == b,
             "plain_sha256": hashlib.sha256(a.encode()).hexdigest(),
             "wrapped_sha256": hashlib.sha256(b.encode()).hexdigest(), "samples": len(plain["samples"])},
            wrapped, steps)


def buckets_match(foot, target_plate, tolerance_n=0.05):
    """A foot record's force buckets equal a recount of its own contacts by world geom and normal (pure)."""
    recount = dict.fromkeys(FORCE_KEYS, 0.)
    for name, _z, force, n_up_z, _dist in foot["contacts"]:
        world = "self" if not name or name.startswith("geom") else world_geom_class(name, target_plate)
        recount[contact_key(world, n_up_z)] += force
    return all(abs(recount[key] - foot[key]) <= tolerance_n for key in FORCE_KEYS if key not in ("self_N", "other_N"))


def round_plate_standing(results):
    """Foot-steps standing inside a ROUND target plate (footprint shrunk by 1 cm, >= 8.0 N of target-plate contact force
    in total, any normal) and whether they read as loaded (pure over the diagnosis records)."""
    total, loaded, examples = 0, 0, []
    for spec, _, steps in results:
        if spec["plate"] != "round":
            continue
        for i, s in enumerate(steps):
            for side in FOOT_BODIES:
                f = s["feet"][side]
                if f["inside_plate"] and f["plate_top_N"] + f["plate_edge_N"] >= SUPPORT_MIN_N:
                    total += 1
                    loaded += bool(f["loaded"])
                    if not f["loaded"] and len(examples) < 5:
                        examples.append({"crossing": crossing_id(spec), "step": i, "foot": side, "foot_record": f})
    return {"foot_steps": total, "loaded": loaded, "share_loaded": (loaded / total) if total else None,
            "unloaded_examples": examples}


def _quartiles(values):
    return [round(float(q), 4) for q in np.percentile(values, (25, 50, 75))] if values else None


def swing_statistics(results):
    """Per crossing and foot: swings over the stage phases with their leading / obstruction / blocked counts and toe
    heights, for all of the foot's swings and for its LEADING swings only (reported only; no height clause is used)."""
    out = {}
    for spec, _, steps in results:
        per = {}
        for side in FOOT_BODIES:
            stage = [s for s in foot_swings(steps, side) if s["touchdown_phase"].startswith("stage_")]
            heights = [s["max_toe_z_m"] for s in stage]
            leading = [s["max_toe_z_m"] for s in stage if s["leading"]]
            per[side] = {"stage_swings": len(stage), "leading": len(leading),
                         "edge_face_obstruction": sum(s["edge_face_obstruction"] for s in stage),
                         "toe_stopped_at_edge": sum(s["toe_stopped_at_edge"] for s in stage),
                         "blocked_leading_and_obstructed": sum(s["blocked"] for s in stage),
                         "obstructed_but_trailing": sum((s["edge_face_obstruction"] or s["toe_stopped_at_edge"])
                                                        and not s["leading"] for s in stage),
                         "max_toe_z_m_quartiles": _quartiles(heights),
                         "share_max_toe_z_below_4cm": (sum(h < 0.04 for h in heights) / len(heights)) if heights
                         else None,
                         "leading_max_toe_z_m_quartiles": _quartiles(leading),
                         "leading_share_max_toe_z_below_4cm": (sum(h < 0.04 for h in leading) / len(leading))
                         if leading else None}
        out[crossing_id(spec)] = per
    return out


def machinery_checks(results, rows, identity, sources, robot_weight_n):
    """Smoke-only checks; any failure makes the smoke exit non-zero."""
    checks, problems = {}, []

    def check(name, ok, detail=None):
        checks[name] = {"ok": bool(ok), "detail": detail}
        if not ok:
            problems.append(name)
    check("identity_wrapped_equals_plain", identity and identity["equal"], identity)
    check("sources_identical_to_bench_v2", sources["identical_to_bench"], sources["bench_sources_differing"])
    check("robot_weight_matches_the_frozen_value", robot_weight_n is not None
          and abs(robot_weight_n - ROBOT_WEIGHT_N) < 0.5, robot_weight_n)
    for (spec, result, steps), row in zip(results, rows):
        name = crossing_id(spec)
        check(f"{name}_complete", result["complete"] and result["end_reason"] in (
            "complete", "fall", "walk_timeout", "stage_timeout"), result["end_reason"])
        check(f"{name}_records_1to1", len(steps) == len(result["samples"]), [len(steps), len(result["samples"])])
        settle = [s for s in steps if s["phase"] == "bench_settle" and s["t"] >= 2.0]
        loads = [sum(support_force(s["feet"][side]) for side in FOOT_BODIES) for s in settle]
        median = float(np.median(loads)) if loads else None
        check(f"{name}_feet_carry_the_robot_in_the_settle", median is not None
              and 0.5 * ROBOT_WEIGHT_N <= median <= 1.5 * ROBOT_WEIGHT_N,
              {"median_support_N": median, "weight_N": ROBOT_WEIGHT_N})
        check(f"{name}_foot_metrics_finite", all(np.isfinite([s["feet"][side][k] for side in FOOT_BODIES for k in (
            "toe_z_m", "sole_min_z_m", "toe_along_m", "heel_along_m")]).all() for s in steps))
        events = result.get("gait_events") or []
        check(f"{name}_clock_s2_swapped_in", result["takeover_s"] is None or bool(
            events and events[0].get("event") == "swap_to_clocks2"
            and events[0].get("controller") == "GaitClockRlController"), [e.get("event") for e in events])
        check(f"{name}_analysed", "edge" in row and (bool(row["edge"].get("no_cross")) or "features" in row),
              {k: row.get(k) for k in ("stall", "cls", "control_cls")})
    trace_swings = {crossing_id(spec): {side: len(foot_swings(steps, side)) for side in FOOT_BODIES}
                    for spec, _, steps in results}
    check("swing_detection_finds_swings_on_both_feet", all(
        sum(v[side] for v in trace_swings.values()) > 0 for side in FOOT_BODIES), trace_swings)
    floor_normals = [c[3] for _, _, steps in results for s in steps for side in FOOT_BODIES
                     for c in s["feet"][side]["contacts"] if c[0] == "floor"]
    check("floor_contact_normals_point_up", bool(floor_normals) and min(floor_normals) >= SMOKE_FLOOR_NORMAL_MIN_Z,
          {"n": len(floor_normals), "min_n_up_z": min(floor_normals) if floor_normals else None})
    mismatches = [(crossing_id(spec), i, side) for spec, result, steps in results
                  for i, s in enumerate(steps) for side in FOOT_BODIES
                  if not buckets_match(s["feet"][side], f"plate_{int(spec['door'])}_{int(result['geometry']['side'])}")]
    check("force_buckets_recount_from_the_contacts_by_normal", not mismatches, mismatches[:5])
    standing = round_plate_standing(results)
    check("foot_standing_inside_a_round_plate_reads_loaded",
          standing["foot_steps"] >= SMOKE_ROUND_PLATE_MIN_FOOT_STEPS
          and standing["share_loaded"] >= SMOKE_ROUND_PLATE_MIN_LOADED_SHARE, standing)
    floor_soles = [s["feet"][side]["sole_min_z_m"] for _, _, steps in results for s in steps
                   if s["phase"] == "bench_settle" and s["t"] >= 2.0 for side in FOOT_BODIES
                   if s["feet"][side]["floor_N"] >= SUPPORT_MIN_N and s["feet"][side]["plate_top_N"] == 0.
                   and s["feet"][side]["plate_edge_N"] == 0.]
    check("floor_loaded_soles_sit_at_z0", bool(floor_soles) and max(abs(z) for z in floor_soles) < 0.01,
          {"n": len(floor_soles), "max_abs_z_m": max(abs(z) for z in floor_soles) if floor_soles else None})
    return {"ok": not problems, "problems": problems, "checks": checks}


# ---- report -----------------------------------------------------------------------------------------------------------

def _layout_text(layouts):
    """'170-249' for a contiguous run, else the comma list (pure)."""
    layouts = sorted(layouts)
    if layouts and layouts == list(range(layouts[0], layouts[-1] + 1)):
        return f"{layouts[0]}-{layouts[-1]}"
    return ", ".join(str(x) for x in layouts)


def report_markdown(summary):
    s = summary["summary"]
    e = summary["enumeration"]
    d = summary["decision"]
    smoke = summary["mode"] == "smoke"
    title = "Mission 7 crossing diagnosis" + (" (SMOKE: machinery only, not a result)" if smoke else "")
    label = "M7_CROSSING_DIAG" + ("_SMOKE" if smoke else "")
    lines = [f"# {title}", "", f"Design: {DESIGN}. Investigation, no gate.", "",
             "Labels: " + "; ".join(f"{k}: {v}" for k, v in LABELS.items()) + ".", "",
             "## Result", "",
             f"- Crossings run: {s['crossings']} (clears {s['clears']}, falls {s['falls']}, no cross phase "
             f"{s['no_cross']}).",
             f"- Stalls (edge dwell >= 2.0 s or never past -0.20 m): {s['stalls_all']}; non-fall {s['stalls_non_fall']}, "
             f"fell {s['stalls_fell']} (excluded); {s['stalls_never_past']} never past; {s['stalls_that_cleared']} "
             "still cleared.",
             "- Criteria overlaps among the non-fall stalls: "
             + json.dumps(s["criteria_overlaps_non_fall_stalls"]) + "."]
    lines += [f"- `{line}`" for line in decision_lines(d, label, smoke)]
    lines += ["", "## Layouts", "",
              f"- Declared {_layout_text(e['layouts_declared'])} ({len(e['layouts_declared'])} layouts); absent from "
              f"the train split (size {e['train_split_size']}): {len(e['layouts_nonexistent'])}.",
              f"- Grid crossings {e['grid_crossings']}, dropped by bench v2's drop rule {len(e['dropped'])}, capped out "
              f"{len(e['capped_out'])} ({e['cap_rule']}), run {len(summary['rows'])}.", "",
              "## Per crossing", "",
              "| crossing | heading | plate | entry | end | fell | clear (s after takeover) | edge dwell s | stall | "
              "class | criteria (wall/blocked/slow) | blocked swings W / first 2 s | swings W (leading) | "
              "wall steps / W steps | mean along speed / cmd vx |",
              "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in summary["rows"]:
        f = r.get("features") or {}
        f2 = r.get("features_first_2s") or {}
        c = r.get("criteria") or {}
        dwell = r["edge"].get("dwell_s")
        clear_s = r.get("clear_after_takeover_s")
        cls = r.get("cls") or (f"(control: {r['control_cls']})" if r.get("control_cls") else "-")
        dwell_text = "never" if r["edge"].get("never_past") else ("-" if dwell is None else f"{dwell:.2f}")
        speed = "-" if not f else f"{f['mean_along_speed_mps']:.3f} / {f['mean_cmd_vx_mps']:.3f}"
        crit = "-" if not c else "/".join("y" if c[k] else "n" for k in ("wall_blocked", "blocked_step_up",
                                                                         "slow_progress"))
        lines.append(
            f"| {crossing_id(r)} | {r['heading_deg']} | {r['plate']} | {r['entry']} | {r['end_reason']} | "
            f"{'yes' if r.get('fell') else 'no'} | {'yes' if r['clear'] else 'no'} "
            f"({'-' if clear_s is None else f'{clear_s:.2f}'}) | {dwell_text} | {'yes' if r.get('stall') else 'no'} | "
            f"{cls} | {crit} | {f.get('n_blocked_swings', '-')} / {f2.get('n_blocked_swings', '-')} | "
            f"{f.get('n_swings', '-')} ({f.get('n_leading_swings', '-')}) | {f.get('wall_steps', '-')} / "
            f"{f.get('window_steps', '-')} | {speed} |")
    if smoke:
        lines += ["", "## Smoke machinery", "", "```", json.dumps({k: summary.get(k) for k in (
            "round_plate_standing", "swing_statistics")}, indent=1, default=str), "```"]
    lines += ["", "## Frozen thresholds", "", "```"] + frozen_block() + ["```", "", "## Disclosures", ""]
    lines += [f"- {item}" for item in DISCLOSURES]
    src = summary["sources"]
    change = summary.get("source_change_during_run") or {}
    lines += ["", f"Sources identical to the clocks2 bench v2 ({src['bench_slurm_job_id']}, {src['bench_git_head']}): "
              f"{src['identical_to_bench']} (differing: {src['bench_sources_differing']}; drift allowed: "
              f"{summary['source_drift_allowed']}).",
              f"Imported sources changed during the run: {change.get('changed')} (relevant drift changed: "
              f"{change.get('relevant_drift_changed')}; first imported during the run: "
              f"{change.get('imported_during_run')}).", ""]
    return "\n".join(lines)


# ---- CLI ----------------------------------------------------------------------------------------------------------------

def _write_json(path, data):
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, indent=1) + "\n")
    os.replace(tmp, path)


def _write_crossing(out, spec, result, steps, row):
    path = out / "crossings" / f"L{int(spec['layout']):03d}-d{int(spec['door'])}.json.gz"
    if path.exists():
        raise FileExistsError(path)
    tmp = path.with_suffix(".tmp")
    with gzip.open(tmp, "wt") as stream:
        json.dump({"bench_v2_result": result, "diag_steps": steps, "analysis": row}, stream)
    os.replace(tmp, path)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("run", "smoke"), required=True)
    parser.add_argument("--preflight", action="store_true", help="guards, enumeration, source and weight checks only")
    args = parser.parse_args(argv)
    repo, out = args.repo.resolve(), args.out.resolve()
    if not (repo / OWN_FILES[0]).is_file():
        parser.error(f"{repo} is not a checkout with {OWN_FILES[0]}")
    enumeration = enumerate_crossings(args.mode)
    specs = enumeration.pop("specs")
    import_simulation_stack()
    sources = source_check(repo)
    drift_allowed = os.environ.get(SOURCE_DRIFT_ENV) == "1"
    if sources["relevant_drift"] and not drift_allowed:
        parser.error(f"imported Mission 7 sources differ from the clocks2 bench v2's: {sources['relevant_drift']}; run "
                     f"from a worktree of the bench's code, or set {SOURCE_DRIFT_ENV}=1 (recorded in the output)")
    weight_info = weights(repo)
    bench_weights = sources["bench_weights_sha256"] or {}
    if (weight_info["clocks2"], weight_info["shipped"]) != (bench_weights.get("clocks2"), bench_weights.get("shipped")):
        parser.error("policy weights differ from the clocks2 bench v2's")
    ids = [crossing_id(s) for s in specs]
    if args.preflight:
        import mujoco
        print(json.dumps({"status": "PREFLIGHT_OK", "mode": args.mode, "crossings": len(specs), "crossing_ids": ids,
                          "layouts_nonexistent": len(enumeration["layouts_nonexistent"]),
                          "dropped": len(enumeration["dropped"]), "identical_to_bench": sources["identical_to_bench"],
                          "relevant_drift": sources["relevant_drift"], "source_drift_allowed": drift_allowed,
                          "weights": {k: weight_info[k] for k in ("shipped", "clocks2")}, "mujoco": mujoco.__version__,
                          "python": sys.executable, "out": str(out), "repo": str(repo)}, sort_keys=True), flush=True)
        return 0
    if out.exists():
        parser.error(f"output exists; a diagnosis result is never overwritten: {out}")
    if not specs:
        parser.error("no crossing to run")
    (out / "crossings").mkdir(parents=True)
    import mujoco
    started = dt.datetime.now(dt.timezone.utc).isoformat()
    _write_json(out / "provenance.json", {
        "design": DESIGN, "mode": args.mode, "labels": LABELS, "thresholds": frozen_thresholds(),
        "enumeration": enumeration, "crossings": [[s["layout"], s["door"]] for s in specs], "repo": str(repo),
        "git_head": git_head(repo), "sources": sources, "source_drift_allowed": drift_allowed, "weights": weight_info,
        "host": platform.node(), "python": sys.executable, "mujoco": mujoco.__version__,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": started, "disclosures": list(DISCLOSURES)})
    print(f"m7 crossing diagnosis ({args.mode}): {len(specs)} crossings {ids} -> {out}", flush=True)
    identity, results, rows, failures, weight_n = None, [], [], [], None
    for k, spec in enumerate(specs):
        try:
            if args.mode == "smoke" and k == 0:
                identity, result, steps = identity_check(repo, out, spec)
                print(json.dumps({"identity": identity}), flush=True)
            else:
                result, steps = run_instrumented(repo, out / "cache" / f"L{spec['layout']:03d}-d{spec['door']}", spec)
            row = analyse_crossing(result, steps)
            _write_crossing(out, spec, result, steps, row)
        except Exception as exc:   # recorded; the run then cannot be COMPLETE and decides nothing
            failures.append({"crossing": crossing_id(spec), "error": repr(exc), "traceback": traceback.format_exc()})
            print(json.dumps({"crossing": crossing_id(spec), "FAILED": repr(exc)}), flush=True)
            continue
        results.append((spec, result, steps))
        rows.append(row)
        print(json.dumps({"crossing": crossing_id(spec), "heading": spec["heading_deg"], "plate": spec["plate"],
                          "entry": spec["entry"], "end": result["end_reason"], "clear": result["clear"],
                          "fell": result["fell"], "dwell_s": row["edge"].get("dwell_s"),
                          "never_past": row["edge"].get("never_past"), "stall": row.get("stall"),
                          "class": row.get("cls"), "control_class": row.get("control_cls"),
                          "blocked_swings": (row.get("features") or {}).get("n_blocked_swings")}), flush=True)
    if args.mode == "smoke":
        from bhl_robust.mission.approach_debug import DebugEnv
        probe = DebugEnv(repo, out / "cache-weight", stage="doors", split=DIAG_SPLIT, seed=0)
        probe.reset(specs[0]["layout"])   # the model only: nothing is stepped
        weight_n = measured_robot_weight(probe)
    sources_end = source_check(repo)
    change = sources_changed(sources, sources_end)
    decision = decide(rows, len(specs), failures, change["any"])
    summary = {"design": DESIGN, "mode": args.mode, "labels": LABELS, "investigation_no_gate": True,
               "smoke_machinery_only": args.mode == "smoke", "thresholds": frozen_thresholds(),
               "enumeration": enumeration, "summary": summarise(rows), "decision": decision, "rows": rows,
               "failures": failures, "sources": sources_end, "source_change_during_run": change,
               "source_drift_allowed": drift_allowed,
               "weights": {k: weight_info[k] for k in ("shipped", "clocks2", "clocks2_deploy_yaml")},
               "repo": str(repo), "git_head": git_head(repo), "host": platform.node(),
               "slurm_job_id": os.environ.get("SLURM_JOB_ID"), "started_utc": started,
               "written_utc": dt.datetime.now(dt.timezone.utc).isoformat(), "disclosures": list(DISCLOSURES)}
    status = 0 if decision["status"] == "COMPLETE" else 1
    if args.mode == "smoke":
        summary.update(identity=identity, robot_weight_n=weight_n, round_plate_standing=round_plate_standing(results),
                       swing_statistics=swing_statistics(results),
                       machinery=machinery_checks(results, rows, identity, sources_end, weight_n))
        status = 0 if summary["machinery"]["ok"] and decision["status"] == "COMPLETE" else 1
    _write_json(out / "summary.json", summary)
    (out / "report.md").write_text(report_markdown(summary) + "\n")
    label = "M7_CROSSING_DIAG" + ("_SMOKE" if args.mode == "smoke" else "")
    for line in decision_lines(decision, label, smoke=args.mode == "smoke"):
        print(line, flush=True)
    if args.mode == "smoke":
        print(json.dumps({"round_plate_standing": {k: summary["round_plate_standing"][k] for k in (
            "foot_steps", "loaded", "share_loaded")}, "swing_statistics": summary["swing_statistics"]}), flush=True)
        print(json.dumps({"machinery_ok": summary["machinery"]["ok"], "problems": summary["machinery"]["problems"]}),
              flush=True)
    return status


if __name__ == "__main__":
    raise SystemExit(main())
