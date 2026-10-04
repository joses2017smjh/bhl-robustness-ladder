"""Mission 7 crossing diagnosis (design D, m7-crossdiag): classifier, decision, guards, foot logging, launcher.

No Mission 7 episode runs here.  The classifier, the decision and the stall definition are tested on synthetic traces at
their boundaries; the foot recorder on tiny MuJoCo stubs (two sole boxes, a floor and a round plate: standing on the
floor, on the plate top, sunk into the plate, pressed against its edge face); the physical_sample plumbing through a
stand-in for bench v2's run_crossing; the launcher on fake trees with stub executables.  The edge-dwell reproduction of
the clocks2 bench-v2 traces runs only where those (gitignored) records exist.
"""
from __future__ import annotations

import copy
import gzip
import json
import os
import re
import shutil
import signal
import subprocess
import sys
import time
import types
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[1]
for _path in (_REPO / "src", _REPO / "scripts", _REPO / "scripts" / "bench"):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import m7_crossing_diag as m  # noqa: E402
import mission7_plate_bench as v1  # noqa: E402
import mission7_plate_bench_v2 as v2  # noqa: E402

LAUNCHER = _REPO / "slurm/repo20260923/cpu_m7_crossing_diag.sbatch"
CLOCKS2_LAUNCHER = _REPO / "slurm/repo20260923/cpu_m7_plate_bench_v2_clocks2.sbatch"
MODULE = _REPO / "scripts/bench/m7_crossing_diag.py"
BENCH_RECORDS = Path(os.environ.get("M7_DIAG_BENCH_RECORDS", _REPO / "results/mission7-campaign-20260923/"
                                                                       "clocks2-plate-bench-v2/crossings"))
# edge_dwell.py's dwell (s, rounded to 3 decimals) on the 41 clocks2 bench-v2 crossings (None = never past -0.20 m),
# solutions-20260930/campaign-m7-180-gemini/m7-180-analysis/clocks2_edge_dwell.json; L22 d0 and L25 d0 had no "cross".
ANALYSIS_DWELL = {
    "L0d0": 3.28, "L0d1": 2.0, "L1d0": 0.92, "L1d1": 1.0, "L2d0": 0.52, "L2d1": 0.64, "L3d0": 0.24, "L4d0": 3.36,
    "L5d0": 3.92, "L6d0": 1.2, "L7d0": 2.08, "L8d0": 1.56, "L8d1": 0.84, "L9d1": 2.84, "L10d1": 1.08, "L11d1": 0.72,
    "L12d1": 1.04, "L13d1": None, "L14d1": 0.92, "L15d1": 3.16, "L16d0": 0.72, "L17d0": 1.84, "L17d1": 1.76,
    "L18d0": 2.64, "L18d1": 1.0, "L19d0": 0.0, "L20d0": 1.2, "L20d1": None, "L21d0": 2.68, "L23d0": 0.36,
    "L24d1": 1.32, "L25d1": 1.6, "L26d1": 3.36, "L27d1": 0.8, "L28d0": None, "L28d1": 0.72, "L29d1": 1.12,
    "L30d1": 1.08, "L31d1": 0.8}
ANALYSIS_NO_CROSS = {"L22d0", "L25d0"}
ANALYSIS_NON180_STALLS = {"L0d0", "L0d1", "L4d0", "L5d0", "L9d1", "L13d1", "L18d0", "L20d1", "L21d0", "L26d1", "L28d0"}
CORRECTION = "250-289 as first frozen does not exist beyond 255; corrected to 170-249 before any diagnosis episode"
CAPPED_OUT = [(234, 1), (235, 1), (236, 0), (236, 1), (237, 1), (238, 1), (239, 1), (240, 0), (241, 0), (242, 0), (242, 1),
              (243, 0), (244, 0), (245, 0), (246, 0), (247, 0), (248, 1), (249, 0), (249, 1)]


# ---- frozen design facts ---------------------------------------------------------------------------------------------

def test_design_d_constants_and_layout_sets():
    assert (m.STALL_DWELL_S, m.ONPLATE_ALONG_M, m.MAX_CROSSINGS, m.STAGE_GAIT) == (2.0, -0.2, 80, "clocks2")
    assert (m.TOP_NORMAL_MIN_Z, m.SUPPORT_MIN_N, m.SWING_MIN_STEPS, m.LEADING_TOL_M) == (0.7071, 8.0, 2, 0.01)
    assert (m.EDGE_FORCE_MIN_N, m.EDGE_STOP_BEFORE_M, m.EDGE_STOP_AFTER_M, m.BLOCKED_MIN_SWINGS) == (1.0, 0.02, 0.01, 3)
    assert m.RUN_LAYOUTS == range(170, 250) and m.BENCH_LAYOUTS == range(0, 32)
    assert m.FIRST_FROZEN_RUN_LAYOUTS == range(250, 290)
    assert "grid order" in m.CAP_RULE and "first 80" in m.CAP_RULE
    assert m.SMOKE_CROSSINGS == ((32, 0), (33, 1), (38, 0)) and m.SMOKE_LAYOUTS == (32, 33, 38)
    assert not set(m.SMOKE_LAYOUTS) & set(m.RUN_LAYOUTS) and not set(m.SMOKE_LAYOUTS) & set(m.BENCH_LAYOUTS)
    assert m.PRECEDENCE == ("wall_blocked", "blocked_step_up", "slow_progress", "other")
    assert m.PLATE_HALF_M == 0.24 and m.CLEAR_ALONG_M == 0.35
    assert "LEARNED" in m.LABELS["gaits"] and "clock-s2" in m.LABELS["gaits"]
    assert m.LABELS["stage"].startswith("SCRIPTED") and m.LABELS["layout_and_plate_pose"].startswith("ORACLE")
    assert any(CORRECTION in d for d in m.DISCLOSURES)
    assert any("no run crossing has a public outcome" in d for d in m.DISCLOSURES)
    assert any("re-frozen by the coordinator" in d for d in m.DISCLOSURES)
    assert any("corrected by the coordinator (2026-10-04 10:45)" in d and "floor-side touchdown" in d
               for d in m.DISCLOSURES)
    header = " ".join(line.lstrip("# ") for line in LAUNCHER.read_text().splitlines() if line.startswith("#"))
    assert "Coordinator correction to (D)'s re-frozen classifier, 2026-10-04 10:45" in header
    assert "FLOOR-SIDE touchdown" in header
    assert "Coordinator re-freeze" in m.DESIGN and "Correction" in m.DESIGN
    for gone in ("PLATE_TOP_CONTACT_MIN_Z_M", "EDGE_BAND_BEFORE_M", "SWING_CLEAR_MARGIN_M", "is_edge_blocked"):
        assert not hasattr(m, gone), gone


def test_the_docstring_block_is_the_constants():
    values = m.frozen_block_values(m.frozen_block())
    assert set(values) == set(m.FROZEN_THRESHOLD_NAMES) | {"PRECEDENCE"}
    for name in m.FROZEN_THRESHOLD_NAMES:
        assert values[name] == str(getattr(m, name)), name
    assert m.frozen_thresholds()["PRECEDENCE"] == list(m.PRECEDENCE)
    block = " ".join(" ".join(m.frozen_block()).split())
    assert "wall_blocked > blocked_step_up > slow_progress > other" in block and "re-frozen" in block
    # every decision of the coordinator re-freeze is a RULE inside the block (which the launcher carries verbatim), and
    # the numbers the rules quote are the constants
    for phrase in ("classified by the contact NORMAL only",
                   "frame[0:3] points from geom1 to geom2, so it is negated when the plate is geom2",
                   f"TOP iff n_up_z >= {m.TOP_NORMAL_MIN_Z}, else EDGE FACE",
                   "all TOP force (target and other plates) is support", "contact heights are never used",
                   "a swing of the LEADING foot", f"at touchdown >= the other foot's toe along - {m.LEADING_TOL_M} m",
                   "obstruction evidence (a) or (b)", "no swing-height clause",
                   "by the precedence wall_blocked > blocked_step_up > slow_progress > other",
                   f"tilt >= {v2.FALL_TILT} in any phase) is excluded from the stall count and reported separately",
                   "printed both ways (non-fall; including fall stalls); the decision uses the non-fall stalls",
                   "SPECIFICITY: among the crossings that CLEAR (bench v2's clear)",
                   f"(>= {m.BLOCKED_MIN_SWINGS} blocked swings, regardless of precedence); printed on the decision line",
                   "F1 is TRAINED iff non-fall stalls > 0 AND 3 x blocked >= non-fall stalls AND clears > 0 AND meeting "
                   "clears x non-fall stalls < blocked x clears",
                   "0 non-fall stalls -> F1 NOT trained", "0 clears -> the specificity check cannot hold -> F1 NOT trained",
                   'never "undefined" or a True/False flag', "prints INCOMPLETE and decides nothing",
                   "so does a log without a DECISION line",
                   f"shrunk by {m.INSIDE_MARGIN_M} m, >= {m.SUPPORT_MIN_N} N of target-plate force) reads as loaded in "
                   f">= {m.SMOKE_ROUND_PLATE_MIN_LOADED_SHARE:.0%} of >= {m.SMOKE_ROUND_PLATE_MIN_FOOT_STEPS} such "
                   "foot-steps"):
        assert phrase in block, phrase
    assert "\nDECISION (" not in m.__doc__   # the rules live in the block only (no second paraphrase to diverge)


# ---- the layout guard and the enumeration ----------------------------------------------------------------------------

@pytest.mark.parametrize("mode, layouts, ok", [
    ("run", list(range(170, 250)), True), ("run", [170], True), ("run", [249], True), ("run", [200, 171], True),
    ("run", [169], False), ("run", [250], False), ("run", [255], False), ("run", [289], False), ("run", [290], False),
    ("run", [0], False), ("run", [31], False), ("run", [32], False), ("run", [38], False), ("run", [170, 250], False),
    ("run", [170, 170], False), ("run", [], False),
    ("smoke", [32, 33, 38], True), ("smoke", [32], True),
    ("smoke", [170], False), ("smoke", [249], False), ("smoke", [250], False), ("smoke", [289], False),
    ("smoke", [34], False), ("smoke", [0], False), ("smoke", [31], False), ("smoke", [290], False),
    ("bench", [170], False)])
def test_layout_guard(mode, layouts, ok):
    if ok:
        assert m.check_layouts(mode, layouts) == sorted(layouts)
    else:
        with pytest.raises(ValueError):
            m.check_layouts(mode, layouts)


@pytest.mark.parametrize("split", ["validation", "test", "TRAIN"])
def test_layout_guard_refuses_every_split_but_train(split):
    with pytest.raises(ValueError, match="never touched"):
        m.check_layouts("run", [170], split=split)
    with pytest.raises(ValueError):
        m.check_crossings("smoke", [(32, 0)], split=split)


def test_crossing_guard():
    assert m.check_crossings("smoke", [(32, 0), (33, 1)]) == [(32, 0), (33, 1)]
    for bad in ([(32, 2)], [(32, 0), (32, 0)], [(170, 1)], [(250, 1)], [(5, 0)], []):
        with pytest.raises(ValueError):
            m.check_crossings("smoke", bad)
    with pytest.raises(ValueError):
        m.check_crossings("run", [(32, 0)])


def _kept(layouts):
    dropped = set(v2.dropped_crossings(tuple(layouts)))
    return [s for s in v1.grid(layouts) if (s["layout"], s["door"]) not in dropped], dropped


def test_run_enumeration_is_bench_v2s_construction_on_170_249_capped_at_80_in_grid_order():
    e = m.enumerate_crossings("run")
    kept, dropped = _kept(range(170, 250))
    assert e["layouts_declared"] == list(range(170, 250)) and e["layouts_nonexistent"] == []
    assert e["train_split_size"] == 256 and e["grid_crossings"] == 160 and len(dropped) == 61
    assert sorted(map(tuple, e["dropped"])) == sorted(dropped)
    assert all(s["entry"] == "walking" for s in v1.grid(range(170, 250)) if (s["layout"], s["door"]) in dropped)
    assert e["kept_before_cap"] == 99 == len(kept) and e["cap"] == 80 and e["cap_rule"] == m.CAP_RULE
    # the cap rule: grid order (layout ascending, door 0 before door 1), the first 80 run, the rest capped out
    assert e["specs"] == kept[:80]
    assert [(s["layout"], s["door"]) for s in e["specs"]] == sorted((s["layout"], s["door"]) for s in e["specs"])
    assert [tuple(c) for c in e["capped_out"]] == [(s["layout"], s["door"]) for s in kept[80:]] == CAPPED_OUT
    assert e["specs"][0] == {"layout": 170, "door": 1, "heading_deg": -90, "plate": "round", "entry": "standstill"}
    assert e["specs"][-1] == {"layout": 234, "door": 0, "heading_deg": -90, "plate": "round", "entry": "walking"}
    assert len({s["layout"] for s in e["specs"]}) == 65
    assert Counter(s["heading_deg"] for s in e["specs"]) == {0: 24, 90: 21, -90: 19, 180: 16}
    assert Counter(s["entry"] for s in e["specs"]) == {"standstill": 64, "walking": 16}
    assert all(s == dict(zip(("layout", "door"), (s["layout"], s["door"])), **dict(zip(
        ("heading_deg", "plate", "entry"), v1.cell_of(s["layout"], s["door"])))) for s in e["specs"])
    # deterministic: the input order of the layouts does not matter; a smaller cap keeps the same prefix
    assert m.enumerate_crossings("run", layouts=list(reversed(range(170, 250))))["specs"] == e["specs"]
    capped = m.enumerate_crossings("run", cap=2)
    assert capped["specs"] == kept[:2] and len(capped["capped_out"]) == 97
    for bad in ([31], [250], [169], [32]):
        with pytest.raises(ValueError):
            m.enumerate_crossings("run", layouts=bad)
    # history: the first-frozen 250-289 exists only up to 255 and would have yielded 6 standstill crossings
    present, absent = m.existing_layouts(list(m.FIRST_FROZEN_RUN_LAYOUTS))
    assert present == list(range(250, 256)) and absent == list(range(256, 290))
    assert [(s["layout"], s["door"], s["entry"]) for s in _kept(range(250, 256))[0]] == [
        (L, 1, "standstill") for L in range(250, 256)]


def test_smoke_enumeration_uses_grid_labels_and_the_drop_rule():
    e = m.enumerate_crossings("smoke")
    assert [(s["layout"], s["door"], s["heading_deg"], s["plate"], s["entry"]) for s in e["specs"]] == [
        (32, 0, 0, "round", "standstill"), (33, 1, 90, "round", "walking"), (38, 0, -90, "square", "standstill")]
    assert (33, 1) not in v2.dropped_crossings((33,))
    for bad in ([(250, 1)], [(170, 1)]):
        with pytest.raises(ValueError):
            m.enumerate_crossings("smoke", crossings=bad)


# ---- geometry and contact classification ------------------------------------------------------------------------------

def test_near_edge_along_round_and_square():
    assert m.near_edge_along(0., "round") == -0.24 and m.near_edge_along(0., "square") == -0.24
    assert m.near_edge_along(0.1, "round") == pytest.approx(-np.sqrt(0.24 ** 2 - 0.01))
    assert m.near_edge_along(-0.1, "round") == m.near_edge_along(0.1, "round")
    assert m.near_edge_along(0.24, "round") is None and m.near_edge_along(0.2399, "round") is not None
    assert m.near_edge_along(0.24, "square") == -0.24 and m.near_edge_along(0.2401, "square") is None
    with pytest.raises(ValueError):
        m.near_edge_along(0., "hexagon")


def test_inside_footprint_boundaries():
    assert m.inside_footprint([0.23, 0.], [0., 0.], "round") and not m.inside_footprint([0.2301], [0.], "round")
    assert m.inside_footprint([0.16], [0.16], "round") and not m.inside_footprint([0.17], [0.17], "round")
    assert m.inside_footprint([0.23], [-0.23], "square") and not m.inside_footprint([0.23], [0.2301], "square")
    with pytest.raises(ValueError):
        m.inside_footprint([0.], [0.], "hexagon")


def test_foot_metrics_of_an_axis_aligned_box():
    # long axis along +x (the door direction), toe at x = -0.25, lateral 0.034..0.106, bottom at z = 0
    rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])   # local y -> world x, local x -> world -y
    corners = m.box_corners([-0.36, 0.07, 0.02], rotation, [0.036, 0.11, 0.02])
    assert corners[:, 0].max() == pytest.approx(-0.25) and corners[:, 2].min() == pytest.approx(0.)
    f = m.foot_metrics(corners, [0., 0.], [1., 0.], "round")
    expected = max(-0.25 - m.near_edge_along(y, "round") for y in (0.034, 0.106))
    assert f["toe_to_edge_m"] == pytest.approx(expected) and f["toe_along_m"] == pytest.approx(-0.25)
    assert f["heel_along_m"] == pytest.approx(-0.47) and f["toe_z_m"] == pytest.approx(0.)
    assert f["inside_plate"] is False
    assert m.foot_metrics(corners, [0., 0.], [1., 0.], "square")["toe_to_edge_m"] == pytest.approx(-0.01)
    assert m.foot_metrics(corners + [0., 0.5, 0.], [0., 0.], [1., 0.], "round")["toe_to_edge_m"] is None
    assert m.foot_metrics(corners, [0., 0.], [0., -1.], "square")["toe_along_m"] == pytest.approx(-0.034)
    centred = m.foot_metrics(m.box_corners([0., 0.05, 0.05], rotation, [0.036, 0.11, 0.02]), [0., 0.], [1., 0.],
                             "round")
    assert centred["inside_plate"] is True and centred["toe_to_edge_m"] > 0.3


def test_toe_to_edge_uses_the_outline_where_the_round_edge_bulges():
    rotation = np.array([[0., 1., 0.], [-1., 0., 0.], [0., 0., 1.]])
    corners = m.box_corners([-0.36, 0., 0.02], rotation, [0.036, 0.11, 0.02])   # toe face across the centre line
    exact = m.foot_metrics(corners, [0., 0.], [1., 0.], "round")["toe_to_edge_m"]
    assert exact == pytest.approx(-0.25 + 0.24)                                 # the edge's nearest point, lat 0
    corner_based = max(-0.25 - m.near_edge_along(y, "round") for y in (-0.036, 0.036))
    assert corner_based == pytest.approx(-0.01272, abs=1e-5) and exact > corner_based
    assert m.foot_metrics(corners, [0., 0.], [1., 0.], "square")["toe_to_edge_m"] == pytest.approx(-0.01)
    lateral = np.array([0.2, 0.3, 0.2, 0.3, 0.2, 0.3, 0.2, 0.3])
    along = np.array([-0.3, -0.3, -0.25, -0.25, -0.3, -0.3, -0.25, -0.25])
    assert m.outline_toe_to_edge(along, lateral, "square") == pytest.approx(-0.01)   # partly beside the plate
    assert m.outline_toe_to_edge(along, lateral + 0.1, "square") is None
    assert list(np.isnan(m.near_edge_along_array([0.24, 0.2399], "round"))) == [True, False]
    assert m.near_edge_along_array([0.1], "round")[0] == pytest.approx(m.near_edge_along(0.1, "round"))


def test_world_geom_classes_normals_and_buckets():
    assert m.world_geom_class("floor", "plate_0_1") == "floor"
    assert m.world_geom_class("plate_0_1", "plate_0_1") == "target_plate"
    assert m.world_geom_class("plate_0_-1", "plate_0_1") == "other_plate"
    assert {m.world_geom_class(n, "plate_0_1") for n in ("wall_m7_3", "door_1", "goal_post_-1")} == {"wall"}
    assert m.world_geom_class("drop_zone", "plate_0_1") == "other"
    assert list(m.upward_normal([0., 0., 1.], world_is_geom1=True)) == [0., 0., 1.]
    assert list(m.upward_normal([0., 0., 1.], world_is_geom1=False)) == [0., 0., -1.]
    assert m.contact_key("target_plate", 0.7071) == "plate_top_N"
    assert m.contact_key("target_plate", 0.70709) == "plate_edge_N"
    assert m.contact_key("target_plate", 1.) == "plate_top_N" and m.contact_key("target_plate", 0.) == "plate_edge_N"
    assert m.contact_key("target_plate", -1.) == "plate_edge_N"
    assert m.contact_key("other_plate", 0.9) == "other_plate_N"
    assert m.contact_key("other_plate", 0.1) == "other_plate_edge_N"
    assert m.contact_key("floor", 0.) == "floor_N" and m.contact_key("self", 0.3) == "self_N"
    assert m.contact_key("wall", 0.) == "wall_N" and m.contact_key("other", 0.) == "other_N"


def _foot(floor=0., top=0., edge=0., other=0., tte=-0.2, toe_along=None, toe_z=0.02, inside=False, contacts=()):
    f = dict.fromkeys(m.FORCE_KEYS, 0.)
    f.update(floor_N=floor, plate_top_N=top, plate_edge_N=edge, other_plate_N=other, toe_to_edge_m=tte,
             toe_along_m=(tte - 0.24 if toe_along is None and tte is not None else toe_along if toe_along is not None
                          else -0.5),
             heel_along_m=-0.7, toe_lateral_m=0., toe_z_m=toe_z, sole_min_z_m=0., inside_plate=inside,
             contacts=list(contacts))
    f["loaded"] = m.is_loaded(f)
    return f


def test_support_counts_floor_plate_top_and_other_plate_top_only():
    assert m.is_loaded(_foot(floor=8.0)) and m.is_loaded(_foot(floor=4.0, top=2.0, other=2.0))
    assert not m.is_loaded(_foot(floor=7.999)) and not m.is_loaded(_foot(edge=50.))
    f = _foot()
    f["other_plate_edge_N"] = 50.
    assert not m.is_loaded(f)


def test_buckets_match_recounts_the_contacts_by_normal():
    contacts = [["floor", 0.0, 30., 1., -0.001], ["plate_0_1", 0.0175, 60., 0.98, -0.025],
                ["plate_0_1", 0.01, 5., 0.05, -0.002], ["plate_1_1", 0.03, 7., 0.9, -0.001]]
    f = _foot(floor=30., top=60., edge=5., other=7., contacts=contacts)
    assert m.buckets_match(f, "plate_0_1")
    assert not m.buckets_match(dict(f, plate_top_N=0., plate_edge_N=65.), "plate_0_1")   # the old z-based bucketing


# ---- the swing tests at their boundaries -----------------------------------------------------------------------------

def _swing(td_along=-0.245, other_along=-0.245, edge=0., tte=-0.005, floor=60., top=0.):
    return {"touchdown_toe_along_m": td_along, "other_toe_along_m": other_along, "edge_face_max_N": edge,
            "touchdown_toe_to_edge_m": tte, "touchdown_floor_N": floor, "touchdown_plate_top_N": top}


@pytest.mark.parametrize("swing, leading", [
    (_swing(), True), (_swing(td_along=0.10, other_along=0.11), True), (_swing(td_along=0.10, other_along=0.1101), False),
    (_swing(td_along=0.30, other_along=0.10), True), (_swing(td_along=-0.30, other_along=-0.245), False)])
def test_leading_foot_boundary(swing, leading):
    assert m.is_leading(swing) is leading


@pytest.mark.parametrize("swing, edge, stop", [
    (_swing(edge=1.0, tte=-0.10), True, False), (_swing(edge=0.999, tte=-0.10), False, False),
    (_swing(tte=-0.02), False, True), (_swing(tte=-0.0200001), False, False),
    (_swing(tte=0.01), False, True), (_swing(tte=0.0100001), False, False), (_swing(tte=None), False, False),
    (_swing(floor=8.0, top=7.999), False, True), (_swing(floor=7.999, top=0.), False, False),
    (_swing(floor=60., top=8.0), False, False)])
def test_obstruction_evidence_boundaries(swing, edge, stop):
    assert m.has_edge_face_obstruction(swing) is edge and m.has_toe_stopped_at_edge(swing) is stop


@pytest.mark.parametrize("swing, blocked", [
    (_swing(), True),                                                    # leading, toe stopped at the edge on the floor
    (_swing(edge=5., tte=-0.10), True),                                  # leading, edge-face force
    (_swing(tte=-0.10), False),                                          # leading, no obstruction evidence
    (_swing(td_along=-0.245, other_along=-0.20), False),                 # trailing although stopped at the edge
    (_swing(td_along=-0.30, other_along=-0.20, edge=5.), False),         # trailing with edge-face force
    (_swing(tte=0.10, floor=0., top=80.), False),                        # stepped onto the plate
    # correction 2026-10-04: edge-face force counts only with a floor-side touchdown
    (_swing(edge=5., tte=0.05, floor=0., top=80.), False),               # brushed the edge face, landed on the top
    (_swing(edge=5., tte=-0.10, floor=60., top=8.0), False),             # floor AND top loaded: not floor side
    (_swing(edge=5., tte=-0.10, floor=8.0, top=7.999), True),            # floor side at its boundary
    (_swing(edge=5., tte=-0.10, floor=7.999, top=0.), False)])           # unloaded floor: not floor side
def test_blocked_swing_is_leading_and_obstructed(swing, blocked):
    assert m.is_blocked_swing(swing) is blocked


@pytest.mark.parametrize("floor, top, side", [
    (8.0, 0., True), (7.999, 0., False), (60., 7.999, True), (60., 8.0, False), (0., 80., False), (0., 0., False)])
def test_touchdown_on_floor_side_boundaries(floor, top, side):
    assert m.touchdown_on_floor_side(_swing(floor=floor, top=top)) is side


# ---- the classifier and the decision at their boundaries -------------------------------------------------------------

def _features(blocked=0, periods=(), speed=0., cmd=0.3, wall=0, steps=100):
    return {"n_blocked_swings": blocked, "period_advances_m": list(periods), "mean_along_speed_mps": speed,
            "mean_cmd_vx_mps": cmd, "wall_steps": wall, "window_steps": steps}


@pytest.mark.parametrize("features, cls", [
    (_features(blocked=3), "blocked_step_up"), (_features(blocked=2), "other"), (_features(blocked=7), "blocked_step_up"),
    (_features(periods=(.01, .01), speed=.02), "slow_progress"),
    (_features(periods=(.01,), speed=.02), "other"),
    (_features(periods=(.01, .01, .01, -.01), speed=.02), "slow_progress"),
    (_features(periods=(.01, .01, -.01, -.01), speed=.02), "other"),
    (_features(periods=(.004, .004), speed=.02), "slow_progress"),
    (_features(periods=(.0039, .01), speed=.02), "other"),
    (_features(periods=(.01, .01), speed=.15, cmd=.3), "other"),
    (_features(periods=(.01, .01), speed=.1499, cmd=.3), "slow_progress"),
    (_features(periods=(.01, .01), speed=.02, cmd=0.), "other"),
    (_features(wall=50, steps=100), "wall_blocked"), (_features(wall=49, steps=100), "other"),
    (_features(wall=0, steps=0), "other"),
    # precedence: wall_blocked > blocked_step_up > slow_progress > other
    (_features(blocked=3, periods=(.01, .01), speed=.02, wall=50), "wall_blocked"),
    (_features(blocked=3, periods=(.01, .01), speed=.02, wall=49), "blocked_step_up"),
    (_features(blocked=2, periods=(.01, .01), speed=.02, wall=49), "slow_progress")])
def test_classify_boundaries_and_precedence(features, cls):
    assert m.classify(features) == cls
    criteria = m.stall_criteria(features)
    assert set(criteria) == set(m.PRECEDENCE[:-1]) and all(isinstance(v, bool) for v in criteria.values())


def _row(stall=False, fell=False, cls=None, clear=False, meets=False):
    return {"stall": stall, "fell": fell, "cls": cls, "clear": clear, "criteria": {"blocked_step_up": meets}}


def _rows(blocked=0, other=0, fall_blocked=0, clears=0, meeting=0):
    rows = [_row(stall=True, cls="blocked_step_up", meets=True) for _ in range(blocked)]
    rows += [_row(stall=True, cls="other") for _ in range(other)]
    rows += [_row(stall=True, fell=True, cls="blocked_step_up", meets=True) for _ in range(fall_blocked)]
    rows += [_row(clear=True, meets=k < meeting) for k in range(clears)]
    return rows


@pytest.mark.parametrize("rows, trained, reason", [
    (_rows(blocked=1, other=2, clears=10), True, "non-fall fraction 1/3 >= 1/3"),
    (_rows(blocked=1, other=3, clears=10), False, "non-fall blocked fraction 1/4 < 1/3"),
    (_rows(blocked=2, other=4, clears=9, meeting=2), True, "non-fall fraction 2/6 >= 1/3"),     # 2/9 < 2/6
    (_rows(blocked=2, other=4, clears=9, meeting=3), False, "specificity fails"),                # 3/9 == 2/6
    (_rows(blocked=1, other=2, clears=3, meeting=1), False, "specificity fails"),                # 1/3 == 1/3
    (_rows(blocked=1, other=2), False, "0 clears"),
    (_rows(fall_blocked=2, clears=5), False, "0 non-fall stalls"),
    (_rows(clears=5), False, "0 non-fall stalls"),
    (_rows(blocked=1, other=3, fall_blocked=3, clears=10), False, "1/4 < 1/3")])                 # falls do not count
def test_decision_rule(rows, trained, reason):
    d = m.decide(rows, expected=len(rows))
    assert d["status"] == "COMPLETE" and d["f1_trained"] is trained and reason in d["reason"]
    lines = m.decision_lines(d)
    assert len(lines) == 4 and all("undefined" not in line for line in lines)
    assert lines[-1].startswith(f"M7_CROSSING_DIAG DECISION F1 {'TRAINED' if trained else 'NOT TRAINED'} (")
    # the decision line itself always carries the non-fall fraction and the specificity clear share (no flags)
    fraction, share = d["blocked_fraction_non_fall"]["text"], d["specificity_clears_meeting_criterion"]["text"]
    assert lines[-1].endswith(f"); non-fall blocked-step-up fraction {fraction}; clear share meeting the criterion "
                              f"{share}")
    assert "True" not in lines[-1] and "False" not in lines[-1]


def test_the_fraction_is_shown_both_ways_and_falls_are_reported_separately():
    d = m.decide(_rows(blocked=1, other=3, fall_blocked=3, clears=10, meeting=1), expected=17)
    assert d["blocked_fraction_non_fall"]["text"] == "1/4 = 0.250000"
    assert d["blocked_fraction_including_falls"]["text"] == "4/7 = 0.571429"
    assert (d["stalls_non_fall"], d["stalls_fell"], d["falls"]) == (4, 3, 3)
    assert d["specificity_clears_meeting_criterion"]["text"] == "1/10 = 0.100000"
    assert d["class_counts_non_fall"] == {"wall_blocked": 0, "blocked_step_up": 1, "slow_progress": 0, "other": 3}
    lines = m.decision_lines(d)
    assert lines[1] == ("M7_CROSSING_DIAG BLOCKED_STEP_UP_FRACTION non-fall 1/4 = 0.250000; including fall stalls "
                        "4/7 = 0.571429")
    assert lines[2] == ("M7_CROSSING_DIAG SPECIFICITY clears meeting the blocked-step-up criterion on their own edge "
                        "dwell 1/10 = 0.100000")
    zero = m.decision_lines(m.decide(_rows(clears=4), expected=4))
    assert "non-fall 0/0 (none)" in zero[1] and zero[-1] == (
        "M7_CROSSING_DIAG DECISION F1 NOT TRAINED (0 non-fall stalls); non-fall blocked-step-up fraction 0/0 (none); "
        "clear share meeting the criterion 0/4 = 0.000000")
    assert lines[-1] == ("M7_CROSSING_DIAG DECISION F1 NOT TRAINED (non-fall blocked fraction 1/4 < 1/3); non-fall "
                         "blocked-step-up fraction 1/4 = 0.250000; clear share meeting the criterion 1/10 = 0.100000")
    no_clears = m.decision_lines(m.decide(_rows(blocked=2, other=1), expected=3))
    assert no_clears[-1] == ("M7_CROSSING_DIAG DECISION F1 NOT TRAINED (0 clears: the specificity check cannot hold); "
                             "non-fall blocked-step-up fraction 2/3 = 0.666667; clear share meeting the criterion 0/0 "
                             "(none)")


@pytest.mark.parametrize("kwargs", [{"expected": 81}, {"failures": [{"crossing": "L170d1", "error": "x"}]},
                                    {"drift_changed": True}])
def test_a_partial_or_aborted_run_is_incomplete_and_decides_nothing(kwargs):
    rows = _rows(blocked=3, clears=10)
    d = m.decide(rows, **({"expected": len(rows)} | kwargs))
    assert d["status"] == "INCOMPLETE" and d["f1_trained"] is None and "decides nothing" in d["reason"]
    lines = m.decision_lines(d)
    assert lines[-1].startswith("M7_CROSSING_DIAG DECISION INCOMPLETE (") and "TRAINED" not in lines[-1]
    smoke = m.decision_lines(d, "M7_CROSSING_DIAG_SMOKE", smoke=True)
    assert all(line.endswith("(SMOKE: machinery only, never a decision)") for line in smoke)


# ---- the stall definition (edge_dwell.py) --------------------------------------------------------------------------

def _history(cross=1.0, end=5.0):
    h = [{"time_s": 0.6, "phase": "approach"}, {"time_s": 0.8, "phase": "settle"}, {"time_s": cross, "phase": "cross"}]
    return h + ([{"time_s": end, "phase": "turn_back"}] if end is not None else [])


def _times(n=200):
    return [round(0.04 * i, 10) for i in range(n)]


@pytest.mark.parametrize("pass_at, stall", [(3.0, True), (2.99, True), (2.96, False), (None, True), (1.0, False)])
def test_stall_is_dwell_at_least_2s_or_never_past(pass_at, stall):
    t = _times()
    along = [-0.30 if pass_at is None or x < pass_at - 1e-9 else -0.19 for x in t]
    d = m.edge_dwell(t, along, _history(cross=1.0), takeover_s=0.6)
    assert d["stall"] is stall and d["no_cross"] is False
    if pass_at is None:
        assert d["never_past"] and d["dwell_s"] is None and d["window"]["end_s"] == pytest.approx(5.0)
    else:
        first = min(x for x in t if x >= pass_at - 1e-9)
        assert d["dwell_s"] == pytest.approx(first - 1.0) and d["window"]["end_s"] == pytest.approx(first)


def test_dwell_of_exactly_2s_is_a_stall_despite_float_accumulation():
    t = [0.04 * i for i in range(200)]   # accumulated, not rounded: 3.0 is not exact
    i = next(k for k, x in enumerate(t) if x >= 2.9999)
    along = [-0.30] * i + [-0.19] * (200 - i)
    d = m.edge_dwell(t, along, _history(cross=t[25]), takeover_s=0.6)
    assert d["dwell_s"] == pytest.approx(2.0, abs=1e-9) and d["stall"] is True


def test_along_exactly_minus_020_is_not_past_and_the_window_stops_at_the_crossing_end():
    t = _times()
    d = m.edge_dwell(t, [-0.2] * len(t), _history(cross=1.0, end=5.0), takeover_s=0.6)
    assert d["never_past"] and d["stall"] and d["window"]["end_s"] == pytest.approx(5.0)
    d = m.edge_dwell(t, [-0.3 if x < 6.0 else 0. for x in t], _history(cross=1.0, end=5.0), takeover_s=0.6)
    assert d["dwell_s"] == pytest.approx(5.0) and d["onplate_in_cross"] is False and d["stall"]
    assert d["window"]["end_s"] == pytest.approx(5.0)
    d = m.edge_dwell(t, [-0.3] * len(t), _history(cross=1.0, end=None), takeover_s=0.6)
    assert d["window"]["end_s"] == pytest.approx(t[-1]) and d["cross_end_s"] is None


def test_no_cross_entry_is_never_a_stall():
    t = _times()
    assert m.edge_dwell(t, [-0.3] * len(t), [{"time_s": 0.6, "phase": "approach"}], 0.6) == {"no_cross": True,
                                                                                               "stall": False}
    assert m.edge_dwell(t, [-0.3] * len(t), _history(), None)["no_cross"]


def _bench_record_dwell(record):
    g = record["geometry"]
    t = [s["time_s"] for s in record["samples"]]
    along = (np.asarray([s["xy"] for s in record["samples"]]) - np.asarray(g["plate_xy"])) @ np.asarray(g["direction"])
    return m.edge_dwell(t, along, record["stage_history"], record["takeover_s"])


@pytest.mark.skipif(not BENCH_RECORDS.is_dir(), reason="the clocks2 bench-v2 crossing records are gitignored")
def test_edge_dwell_reproduces_the_analysis_on_the_clocks2_bench_traces():
    """Read-only: the recorded traces of 21517668 (no episode runs); the stall definition, not the classifier."""
    seen, stalls = set(), set()
    for path in sorted(BENCH_RECORDS.glob("L*-d*.json.gz")):
        with gzip.open(path, "rt") as stream:
            record = json.load(stream)
        key = m.crossing_id(record)
        seen.add(key)
        d = _bench_record_dwell(record)
        if key in ANALYSIS_NO_CROSS:
            assert d["no_cross"], key
            continue
        dwell = None if d["dwell_s"] is None else round(d["dwell_s"], 3)
        assert dwell == ANALYSIS_DWELL[key], key
        if record["heading_deg"] != 180 and not record["clear"] and d["stall"]:
            stalls.add(key)
    assert seen == set(ANALYSIS_DWELL) | ANALYSIS_NO_CROSS
    assert stalls == ANALYSIS_NON180_STALLS


# ---- swings and features on synthetic diagnosis records --------------------------------------------------------------

def test_unloaded_runs():
    assert m.unloaded_runs([True, False, False, True, False, True, False, False, False]) == [(1, 2), (6, 8)]
    assert m.unloaded_runs([False, True]) == [] and m.unloaded_runs([]) == []
    assert m.unloaded_runs([False, False, True], min_steps=3) == []


def _step(t, along, feet, cmd=(0.3, 0., 0.), wall=(), phase="stage_cross"):
    return {"t": t, "phase": phase, "base_along_m": along, "base_lateral_m": 0., "yaw_err_rad": 0.,
            "cmd": list(cmd), "v_body": [0., 0.], "v_along_mps": 0., "wz": 0., "tilt": 0.,
            "wall_instant": list(wall), "wall_any_substep": bool(wall), "feet": feet}


def _treading(n, left="edge", right="edge", along=lambda i: -0.30, wall=lambda i: ()):
    """Both feet alternate 6 loaded / 6 unloaded steps (offset by 6, so at one foot's touchdown the other lifts off).
    Per foot: 'edge' = every touchdown stops the toe 5 mm before the edge on the floor; 'plate' = touchdowns onto the
    plate top, toe 10 cm past the edge; 'face' = touchdowns 10 cm before the edge after edge-face force mid-swing;
    'none' = touchdowns 10 cm before the edge, no force; 'faceplate' = edge-face force mid-swing, then a touchdown onto
    the plate top (got over)."""
    loaded = {"edge": {"floor": 80., "tte": -0.005}, "plate": {"top": 80., "tte": 0.10},
              "face": {"floor": 80., "tte": -0.10}, "none": {"floor": 80., "tte": -0.10},
              "faceplate": {"top": 80., "tte": 0.10}}
    swinging = {"edge": {"tte": -0.03}, "plate": {"tte": 0.05}, "face": {"tte": -0.12}, "none": {"tte": -0.12},
                "faceplate": {"tte": 0.05}}
    steps = []
    for i in range(n):
        feet = {}
        for side, shift, mode in (("left", 0, left), ("right", 6, right)):
            k = (i + shift) % 12
            feet[side] = (_foot(**loaded[mode]) if k < 6
                          else _foot(**swinging[mode], edge=5. if mode in ("face", "faceplate") and k == 9 else 0.))
        steps.append(_step(round(0.04 * i, 10), along(i), feet, wall=wall(i)))
    return steps


def test_foot_swings_leading_and_obstruction_on_synthetic_treading():
    steps = _treading(60)
    left, right = m.foot_swings(steps, "left"), m.foot_swings(steps, "right")
    assert [s["touchdown_index"] for s in left] == [12, 24, 36, 48]   # the run still open at the end is dropped
    assert [s["touchdown_index"] for s in right] == [6, 18, 30, 42, 54]
    assert all(s["leading"] and s["toe_stopped_at_edge"] and s["blocked"] for s in left + right)
    f = m.window_features(steps, 10, 40)
    assert f["n_blocked_swings"] == 5 and f["per_foot"]["left"]["blocked"] == 3 and f["per_foot"]["right"]["blocked"] == 2
    assert m.classify(f) == "blocked_step_up"
    # the left toe stops at the edge, but behind the right foot already on the plate: trailing, so it never counts
    trailing = m.window_features(_treading(60, left="edge", right="plate"), 10, 40)
    assert trailing["per_foot"]["left"] == {"swings": 3, "leading": 0, "edge_face": 0, "toe_stop": 3, "blocked": 0}
    assert trailing["per_foot"]["right"]["blocked"] == 0 and trailing["n_blocked_swings"] == 0
    assert m.classify(trailing) == "other"
    face = m.window_features(_treading(60, left="face", right="face"), 10, 40)
    assert face["n_edge_face_swings"] == 5 and face["n_toe_stop_swings"] == 0 and face["n_blocked_swings"] == 5
    for mode in ("plate", "none"):
        clean = m.window_features(_treading(60, left=mode, right=mode), 10, 40)
        assert clean["n_swings"] == 5 and clean["n_obstructed_swings"] == 0 and clean["n_blocked_swings"] == 0
    plate = m.window_features(_treading(60, left="plate", right="plate"), 10, 40)
    assert plate["touchdowns_on_plate_top"] == {"left": 3, "right": 2}
    # correction 2026-10-04: edge-face force, then a touchdown on the plate top, got over: never blocked
    over = m.window_features(_treading(60, left="faceplate", right="faceplate"), 10, 40)
    assert over["n_edge_face_swings"] == 5 and over["n_blocked_swings"] == 0 and m.classify(over) == "other"
    assert not any(s["touchdown_floor_side"] for side in ("left", "right")
                   for s in m.foot_swings(_treading(60, left="faceplate", right="faceplate"), side))


def test_window_features_progress_and_wall():
    slow = _treading(80, left="none", right="none", along=lambda i: -0.30 + 0.0008 * i)   # 0.02 m/s, 0.016 m/period
    f = m.window_features(slow, 0, 79)
    assert len(f["period_advances_m"]) == 3 and all(a == pytest.approx(0.016) for a in f["period_advances_m"])
    assert f["mean_along_speed_mps"] == pytest.approx(0.02) and f["mean_cmd_vx_mps"] == pytest.approx(0.3)
    assert m.classify(f) == "slow_progress"
    walled = _treading(80, wall=lambda i: ("wall_m7_1",) if i % 2 == 0 else ())   # blocked swings AND wall: wall wins
    f = m.window_features(walled, 0, 79)
    assert f["wall_steps"] == 40 and f["window_steps"] == 80 and f["n_blocked_swings"] >= 3
    assert m.stall_criteria(f) == {"wall_blocked": True, "blocked_step_up": True, "slow_progress": False}
    assert m.classify(f) == "wall_blocked"
    assert m.classify(m.window_features(_treading(80, left="none", right="none"), 0, 79)) == "other"


def _bench_result(steps, cross_s, end_s, plate="round", clear=False, fell=False, takeover=0.6):
    """A bench-v2-shaped record whose base track is the steps' along (plate at the origin, door direction +x)."""
    history = [{"time_s": takeover, "phase": "approach"}, {"time_s": cross_s, "phase": "cross"},
               {"time_s": end_s, "phase": "turn_back"}, {"time_s": end_s + 1., "phase": "recorded"}]
    samples = [{"time_s": s["t"], "xy": [s["base_along_m"], 0.], "tilt": 0.} for s in steps]
    return {"layout": 999, "door": 1, "heading_deg": 0, "plate": plate, "entry": "standstill", "end_reason": "complete",
            "clear": clear, "real_clear": clear, "clear_after_takeover_s": None, "fell": fell, "fall_phase": None,
            "takeover_s": takeover, "handback_s": end_s + 1., "maximum_tilt": 0.1, "cell_m": 1.6,
            "geometry": {"plate_xy": [0., 0.], "direction": [1., 0.], "side": 1},
            "wall_contacts": {"stage_samples": 0}, "stage_history": history, "samples": samples}


def test_analyse_crossing_and_the_decision_end_to_end():
    stuck = _treading(200)   # never past -0.20 m inside a 1.0 -> 5.0 s crossing, blocked at the edge
    stall = m.analyse_crossing(_bench_result(stuck, 1.0, 5.0), stuck)
    assert stall["stall"] and stall["edge"]["never_past"] and stall["cls"] == "blocked_step_up"
    assert stall["control_cls"] is None and "swings" not in stall["features"]
    assert stall["window"]["start_s"] == pytest.approx(1.0) and stall["window"]["end_s"] == pytest.approx(5.0)
    quick = _treading(200, left="plate", right="plate", along=lambda i: -0.30 if i < 50 else 0.40)   # dwell 1.0 s
    clear = m.analyse_crossing(_bench_result(quick, 1.0, 5.0, clear=True), quick)
    assert not clear["stall"] and clear["cls"] is None and clear["control_cls"] in m.PRECEDENCE
    assert clear["edge"]["dwell_s"] == pytest.approx(1.0) and clear["t_clear_after_takeover_s"] == pytest.approx(1.4)
    assert clear["features_first_2s"]["window_steps"] == 51 and clear["criteria"]["blocked_step_up"] is False
    fallen = m.analyse_crossing(_bench_result(stuck, 1.0, 5.0, fell=True), stuck)
    rows = [stall, clear, fallen]
    json.dumps(rows)
    s = m.summarise(rows)
    assert (s["stalls_all"], s["stalls_non_fall"], s["stalls_fell"], s["clears"]) == (2, 1, 1, 1)
    d = m.decide(rows, expected=3)
    assert d["f1_trained"] is True and d["blocked_fraction_non_fall"]["text"] == "1/1 = 1.000000"
    assert d["specificity_clears_meeting_criterion"]["text"] == "0/1 = 0.000000"
    assert s["criteria_overlaps_non_fall_stalls"]["combinations"] == {"blocked_step_up": 1}


def test_analyse_crossing_without_a_cross_entry():
    steps = _treading(50)
    result = _bench_result(steps, 1.0, 1.5)
    result["stage_history"] = [{"time_s": 0.6, "phase": "approach"}]
    row = m.analyse_crossing(result, steps)
    assert row["stall"] is False and row["cls"] is None and row["edge"]["no_cross"]


def test_smoke_statistics_on_synthetic_records():
    on_plate = _foot(top=60., edge=0., tte=0.3, inside=True)
    sunk_old = _foot(top=0., edge=60., tte=0.3, inside=True)   # what the z-based bucketing produced on a sunk contact
    steps = [_step(0.04 * i, -0.1, {"left": on_plate, "right": _foot(floor=80.)}) for i in range(12)]
    spec_round = {"layout": 32, "door": 0, "plate": "round"}
    good = m.round_plate_standing([(spec_round, None, steps)])
    assert (good["foot_steps"], good["loaded"], good["share_loaded"]) == (12, 12, 1.0)
    bad = m.round_plate_standing([(spec_round, None, [_step(0., -0.1, {"left": sunk_old, "right": _foot()})])])
    assert bad["foot_steps"] == 1 and bad["loaded"] == 0 and bad["unloaded_examples"]
    assert m.round_plate_standing([({"layout": 38, "door": 0, "plate": "square"}, None, steps)])["foot_steps"] == 0
    stats = m.swing_statistics([({"layout": 32, "door": 0, "plate": "round"}, None, _treading(60))])
    left = stats["L32d0"]["left"]
    assert left["stage_swings"] == 4 and left["leading"] == 4 and left["blocked_leading_and_obstructed"] == 4
    assert left["obstructed_but_trailing"] == 0 and left["share_max_toe_z_below_4cm"] == 1.0
    assert left["max_toe_z_m_quartiles"] == left["leading_max_toe_z_m_quartiles"] == [0.02, 0.02, 0.02]
    assert left["leading_share_max_toe_z_below_4cm"] == 1.0
    behind = m.swing_statistics([({"layout": 33, "door": 1, "plate": "round"}, None,
                                  _treading(60, left="edge", right="plate"))])["L33d1"]["left"]
    assert behind["toe_stopped_at_edge"] == 4 and behind["leading"] == 0 and behind["obstructed_but_trailing"] == 4
    assert behind["blocked_leading_and_obstructed"] == 0
    assert behind["leading_max_toe_z_m_quartiles"] is None and behind["leading_share_max_toe_z_below_4cm"] is None
    assert behind["max_toe_z_m_quartiles"] == [0.02, 0.02, 0.02]


# ---- the recorder on MuJoCo stubs ------------------------------------------------------------------------------------

def _stub_xml(left="-0.36 0.07 -0.0001", right="0 -0.07 0.0499"):
    """Floor, a round plate (r 0.24, top 0.03) at the origin, a wall far away, two sole boxes (long axis along +x)."""
    return f"""<mujoco><option timestep="0.0005"/><worldbody>
<geom name="floor" type="plane" size="5 5 .05"/>
<geom name="plate_0_1" type="cylinder" size=".24 .015" pos="0 0 .015"/>
<geom name="wall_m7_0" type="box" size=".04 1 .55" pos="3 0 .55"/>
<body name="r0_leg_left_ankle_roll" pos="{left}"><freejoint/>
  <geom type="box" size=".036 .11 .02" pos="0 0 .02" quat=".7071068 0 0 -.7071068" mass="2"/>
  <geom type="box" size=".01 .01 .01" pos="0 0 .1" contype="0" conaffinity="0" mass=".01"/></body>
<body name="r0_leg_right_ankle_roll" pos="{right}"><freejoint/>
  <geom type="box" size=".036 .11 .02" pos="0 0 .0" quat=".7071068 0 0 -.7071068" mass="2"/></body>
</worldbody></mujoco>"""


STUB_XML = _stub_xml()


def _stub_env(xml=STUB_XML):
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    own = np.array([(mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(b)) or "").startswith("r0_")
                    for b in model.geom_bodyid])
    runner = types.SimpleNamespace(d=data, own=own, wall_contact=False)
    return types.SimpleNamespace(model=model, runner=runner, slot=types.SimpleNamespace(prefix="r0_"))


def _sample(t=1.0):
    return {"time_s": t, "xy": [-0.5, 0.], "velocity": [0.1, 0.02], "velocity_body": [0.1, 0.02], "yaw": 0.,
            "yaw_rate": 0.05, "tilt": 0.01, "contacts": ["wall_m7_0"], "command": [0.3, 0., 0.], "phase": "stage_cross"}


def _record(xml=STUB_XML):
    env = _stub_env(xml)
    rec = m.FootRecorder({"plate_xy": [0., 0.], "direction": [1., 0.], "side": 1}, "round", 0)
    sample = _sample()
    before = copy.deepcopy(sample)
    qpos = env.runner.d.qpos.copy()
    rec.record(env, np.array([0.3, 0., 0.1]), "stage_cross", sample)
    assert sample == before and np.array_equal(env.runner.d.qpos, qpos)   # read-only
    assert len(rec.steps) == 1
    return rec.steps[0]


def test_recorder_floor_and_plate_top_normals_point_up_and_feet_read_loaded():
    step = _record()
    left, right = step["feet"]["left"], step["feet"]["right"]
    # left: on the floor, toe 1 cm before the centre-line edge (lateral 0.034..0.106 on the round plate)
    assert left["floor_N"] >= m.SUPPORT_MIN_N and left["plate_top_N"] == 0. and left["loaded"] is True
    assert all(c[0] == "floor" and c[3] > 0.999 for c in left["contacts"])
    expected = max(-0.25 - m.near_edge_along(y, "round") for y in (0.034, 0.106))
    assert left["toe_to_edge_m"] == pytest.approx(expected, abs=2e-4) and left["inside_plate"] is False
    assert left["toe_along_m"] == pytest.approx(-0.25, abs=2e-4) and abs(left["sole_min_z_m"]) < 1e-3
    assert left["ankle_xyz"] == pytest.approx([-0.36, 0.07, -0.0001], abs=1e-6)   # the link, not the box
    assert left["box_xyz"] == pytest.approx([-0.36, 0.07, 0.0199], abs=1e-6)
    # right: standing on the plate top, inside it
    assert right["plate_top_N"] >= m.SUPPORT_MIN_N and right["plate_edge_N"] == 0. and right["floor_N"] == 0.
    assert right["loaded"] is True and right["inside_plate"] is True and right["toe_to_edge_m"] > 0.3
    assert all(c[0] == "plate_0_1" and c[3] >= m.TOP_NORMAL_MIN_Z for c in right["contacts"])
    assert m.buckets_match(left, "plate_0_1") and m.buckets_match(right, "plate_0_1")
    assert step["base_along_m"] == pytest.approx(-0.5) and step["cmd"] == [0.3, 0., 0.1]
    assert step["v_body"] == [0.1, 0.02] and step["v_along_mps"] == pytest.approx(0.1) and step["wz"] == 0.05
    assert step["phase"] == "stage_cross" and step["wall_instant"] == ["wall_m7_0"] and step["t"] == 1.0
    json.dumps(step)


def test_recorder_a_foot_sunk_into_the_round_plate_is_support_not_edge():
    """The review's case (smoke 21532133 L33 d1 step 283): one contact at z ~ 0.0175 under a foot standing mid-plate."""
    right = _record(_stub_xml(right="0 -0.07 0.025"))["feet"]["right"]   # sole bottom at 0.005: 2.5 cm into the plate
    assert right["sole_min_z_m"] == pytest.approx(0.005, abs=1e-3)
    plate = [c for c in right["contacts"] if c[0] == "plate_0_1"]
    assert plate and all(c[1] < 0.025 for c in plate) and all(c[3] >= m.TOP_NORMAL_MIN_Z for c in plate)
    assert right["plate_top_N"] >= m.SUPPORT_MIN_N and right["plate_edge_N"] == 0. and right["loaded"] is True


def test_recorder_a_toe_pressed_against_the_edge_face_is_edge_not_support():
    left = _record(_stub_xml(left="-0.348 0 -0.0001"))["feet"]["left"]   # toe 2 mm into the cylinder's side
    plate = [c for c in left["contacts"] if c[0] == "plate_0_1"]
    assert plate and all(abs(c[3]) < m.TOP_NORMAL_MIN_Z for c in plate)
    assert left["plate_edge_N"] > 0. and left["plate_top_N"] == 0. and left["floor_N"] >= m.SUPPORT_MIN_N
    assert left["toe_to_edge_m"] == pytest.approx(0.002, abs=2e-4) and m.buckets_match(left, "plate_0_1")


# The plate on a static body defined AFTER the feet: its geom id is higher than the sole boxes', so for this box-box pair
# MuJoCo makes the sole geom1 and the plate geom2 (in the real model the plates are world geoms, always geom1).
GEOM2_PLATE_XML = """<mujoco><option timestep="0.0005"/><worldbody>
<geom name="floor" type="plane" size="5 5 .05"/>
<body name="r0_leg_left_ankle_roll" pos="-1 0.5 -0.0001"><freejoint/>
  <geom type="box" size=".036 .11 .02" pos="0 0 .02" quat=".7071068 0 0 -.7071068" mass="2"/></body>
<body name="r0_leg_right_ankle_roll" pos="0 -0.07 0.0499"><freejoint/>
  <geom type="box" size=".036 .11 .02" pos="0 0 .0" quat=".7071068 0 0 -.7071068" mass="2"/></body>
<body name="late_static_plate"><geom name="plate_0_1" type="box" size=".24 .24 .015" pos="0 0 .015"/></body>
</worldbody></mujoco>"""


def test_recorder_negates_the_normal_when_the_plate_is_geom2():
    """The sign correction on a real MuJoCo contact: MuJoCo's normal points from geom1 (the sole) down into geom2 (the
    plate); the recorder must turn it into an upward normal, so the foot standing on the plate top reads loaded."""
    mujoco = pytest.importorskip("mujoco")
    env = _stub_env(GEOM2_PLATE_XML)
    d = env.runner.d
    plate = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_GEOM, "plate_0_1")
    pairs = [(int(d.contact[i].geom1), int(d.contact[i].geom2), float(d.contact[i].frame[2])) for i in range(int(d.ncon))]
    on_plate = [p for p in pairs if plate in p[:2]]
    assert on_plate and all(g2 == plate for _, g2, _ in on_plate)   # the precondition: the plate is geom2 here
    assert all(nz < -0.99 for _, _, nz in on_plate)                  # MuJoCo's raw normal: from the sole down
    rec = m.FootRecorder({"plate_xy": [0., 0.], "direction": [1., 0.], "side": 1}, "square", 0)
    rec.record(env, np.zeros(3), "stage_cross", _sample())
    right, left = rec.steps[0]["feet"]["right"], rec.steps[0]["feet"]["left"]
    plate_contacts = [c for c in right["contacts"] if c[0] == "plate_0_1"]
    assert plate_contacts and all(c[3] > 0.99 for c in plate_contacts)
    assert right["plate_top_N"] >= m.SUPPORT_MIN_N and right["plate_edge_N"] == 0. and right["loaded"] is True
    assert left["floor_N"] >= m.SUPPORT_MIN_N and left["loaded"] is True
    assert m.buckets_match(right, "plate_0_1") and m.buckets_match(left, "plate_0_1")


def test_recorder_refuses_a_model_without_exactly_one_sole_box():
    mujoco = pytest.importorskip("mujoco")
    xml = STUB_XML.replace('<geom type="box" size=".036 .11 .02" pos="0 0 .0"', '<geom type="sphere" size=".03"')
    env = _stub_env()
    env.model = mujoco.MjModel.from_xml_string(xml)
    rec = m.FootRecorder({"plate_xy": [0., 0.], "direction": [1., 0.], "side": 1}, "round", 0)
    with pytest.raises(RuntimeError, match="collision boxes"):
        rec.resolve(env)
    with pytest.raises(ValueError, match="axis-aligned"):
        m.FootRecorder({"plate_xy": [0., 0.], "direction": [.6, .8], "side": 1}, "round", 0)


def test_recording_wraps_physical_sample_and_always_restores_it(monkeypatch):
    import bhl_robust.mission.approach_debug as approach_debug
    calls = []

    def original(env, command, phase):
        calls.append(phase)
        return {"phase": phase}
    monkeypatch.setattr(approach_debug, "physical_sample", original)
    recorder = types.SimpleNamespace(steps=[])
    recorder.record = lambda env, command, phase, sample: recorder.steps.append((phase, sample))
    with m.recording(recorder):
        from bhl_robust.mission.approach_debug import physical_sample   # what run_crossing does, function-locally
        assert physical_sample is not original and physical_sample._m7_crossing_diag
        out = physical_sample("env", [0., 0., 0.], "stage_turn")
        assert out == {"phase": "stage_turn"} and recorder.steps[0][1] is out
        with pytest.raises(RuntimeError, match="already wrapped"):
            with m.recording(recorder):
                pass
    assert approach_debug.physical_sample is original and calls == ["stage_turn"]
    with pytest.raises(KeyError):
        with m.recording(recorder):
            raise KeyError("boom")
    assert approach_debug.physical_sample is original


def _fake_run_crossing(env, extra_sample=False, record=None):
    def run_crossing(repo, cache, spec, stage_gait):
        from bhl_robust.mission.approach_debug import physical_sample   # function-local, as bench v2's
        if record is not None:
            record.append((Path(cache).name, stage_gait))
        samples = []
        for _ in range(3):
            sample = physical_sample(env, np.array([0.3, 0., 0.]), "stage_cross")
            sample["plate_presses"] = []   # bench v2 adds this after physical_sample returns
            samples.append(sample)
        if extra_sample:
            samples.append(dict(samples[-1], time_s=99.))
        g = v2.crossing_geometry(m.generate("train", spec["layout"]), spec["door"], spec["plate"], spec["heading_deg"],
                                 spec["entry"])
        return {"geometry": g, "samples": samples}
    return run_crossing


def test_run_instrumented_records_one_step_per_bench_sample(monkeypatch, tmp_path):
    import bhl_robust.mission.approach_debug as approach_debug
    env = _stub_env()
    clock = iter(range(100))
    original = lambda env_, command, phase: dict(_sample(t=float(next(clock)) * 0.04), phase=phase)  # noqa: E731
    monkeypatch.setattr(approach_debug, "physical_sample", original)
    calls = []
    monkeypatch.setattr(v2, "run_crossing", _fake_run_crossing(env, record=calls))
    [spec] = [s for s in m.enumerate_crossings("smoke")["specs"] if s["layout"] == 38]
    result, steps = m.run_instrumented(tmp_path, tmp_path / "cache" / "L038-d0", spec)
    assert calls == [("L038-d0", "clocks2")]
    assert len(steps) == len(result["samples"]) == 3
    assert [s["t"] for s in steps] == [s["time_s"] for s in result["samples"]]
    assert all(set(s["feet"]) == {"left", "right"} for s in steps) and "plate_presses" not in steps[0]
    assert approach_debug.physical_sample is original
    monkeypatch.setattr(v2, "run_crossing", _fake_run_crossing(env, extra_sample=True))
    with pytest.raises(RuntimeError, match="not 1:1"):
        m.run_instrumented(tmp_path, tmp_path / "cache2", spec)
    assert approach_debug.physical_sample is original


# ---- provenance -------------------------------------------------------------------------------------------------------

def test_source_check_flags_drift_against_a_bench_provenance(tmp_path, monkeypatch):
    names = ["scripts/mission7_plate_bench_v2.py", "scripts/mission7_plate_bench.py", "src/bhl_robust/mission/layout.py",
             "scripts/mission7_diagnostic.py"]
    recorded = {name: m.sha256(_REPO / name) for name in names}
    recorded["scripts/mission7_plate_bench.py"] = "0" * 64        # imported and drifted
    recorded["scripts/mission7_diagnostic.py"] = "f" * 64          # not imported: reported, not relevant
    provenance = tmp_path / "provenance.json"
    provenance.write_text(json.dumps({"sources_sha256": recorded, "weights_sha256": {"clocks2": "x"},
                                      "git_head": "abc", "slurm_job_id": "1"}))
    imported = ["scripts/bench/m7_crossing_diag.py", "scripts/mission7_plate_bench.py",
                "scripts/mission7_plate_bench_v2.py", "src/bhl_robust/mission/layout.py"]
    monkeypatch.setattr(m, "imported_files", lambda root: imported if Path(root).resolve() == _REPO.resolve() else [])
    s = m.source_check(_REPO, provenance)
    assert s["bench_sources_differing"] == ["scripts/mission7_diagnostic.py", "scripts/mission7_plate_bench.py"]
    assert s["relevant_drift"] == ["scripts/mission7_plate_bench.py"] and s["identical_to_bench"] is False
    assert "scripts/bench/m7_crossing_diag.py" in s["sources_sha256"]
    assert "scripts/bench/m7_crossing_diag.py" in s["imported_repo_files"]
    assert s["sources_sha256"]["scripts/mission7_plate_bench_v2.py"] == m.sha256(_REPO / "scripts/mission7_plate_bench_v2.py")


def test_sources_changed_flags_changes_but_not_a_module_first_imported_during_the_run():
    start = {"sources_sha256": {"a.py": "1", "b.py": "2", "bench_only.py": "3"}, "imported_repo_files": ["a.py", "b.py"],
             "relevant_drift": []}
    assert m.sources_changed(start, copy.deepcopy(start)) == {
        "changed": [], "relevant_drift_changed": False, "imported_during_run": [], "any": False}
    # first imported during the run, with no start hash (outside the bench's provenance): recorded, not a change
    late = copy.deepcopy(start)
    late["sources_sha256"]["late.py"] = "9"
    late["imported_repo_files"].append("late.py")
    c = m.sources_changed(start, late)
    assert c["imported_during_run"] == ["late.py"] and c["changed"] == [] and c["any"] is False
    # first imported during the run but hashed at the start (the bench recorded it): compared
    late_bench = copy.deepcopy(start)
    late_bench["imported_repo_files"].append("bench_only.py")
    assert m.sources_changed(start, late_bench)["any"] is False
    late_bench["sources_sha256"]["bench_only.py"] = "x"
    assert m.sources_changed(start, late_bench)["changed"] == ["bench_only.py"]
    # an imported file edited or removed during the run, or the relevant drift changing: INCOMPLETE
    edited = copy.deepcopy(start)
    edited["sources_sha256"]["a.py"] = "z"
    gone = copy.deepcopy(start)
    del gone["sources_sha256"]["b.py"]
    drift = copy.deepcopy(start)
    drift["relevant_drift"] = ["a.py"]
    assert m.sources_changed(start, edited)["changed"] == ["a.py"] and m.sources_changed(start, gone)["changed"] == ["b.py"]
    assert m.sources_changed(start, drift) == {"changed": [], "relevant_drift_changed": True, "imported_during_run": [],
                                               "any": True}
    rows = _rows(blocked=1, other=2, clears=5)
    for end in (edited, gone, drift):
        d = m.decide(rows, expected=len(rows), drift_changed=m.sources_changed(start, end)["any"])
        assert d["status"] == "INCOMPLETE" and "an imported source changed during the run" in d["reason"]
    assert m.decide(rows, expected=len(rows), drift_changed=m.sources_changed(start, late)["any"])["status"] == "COMPLETE"


def test_report_markdown_carries_the_decision_line_and_the_source_change():
    stuck = _treading(200)
    rows = [m.analyse_crossing(_bench_result(stuck, 1.0, 5.0), stuck)]
    d = m.decide(rows, expected=1)
    summary = {"mode": "run", "summary": m.summarise(rows), "decision": d, "rows": rows,
               "enumeration": {"layouts_declared": [170], "layouts_nonexistent": [], "train_split_size": 256,
                               "grid_crossings": 2, "dropped": [], "capped_out": [], "cap_rule": m.CAP_RULE},
               "sources": {"bench_slurm_job_id": "21517668", "bench_git_head": "9de55d3", "identical_to_bench": True,
                           "bench_sources_differing": []},
               "source_drift_allowed": False,
               "source_change_during_run": {"changed": [], "relevant_drift_changed": False,
                                            "imported_during_run": ["late.py"], "any": False}}
    text = m.report_markdown(summary)
    assert f"- `{m.decision_lines(d)[-1]}`" in text and "first imported during the run: ['late.py']" in text
    assert "| L999d1 | 0 | round | standstill | complete |" in text and "SMOKE" not in text
    assert all(line in text for line in m.frozen_block())


def test_imported_files_skips_relative_and_missing_module_files(monkeypatch, tmp_path):
    (tmp_path / "pkg").mkdir()
    real = tmp_path / "pkg" / "real.py"
    real.write_text("")
    fakes = {"m7diag_fake_real": str(real), "m7diag_fake_relative": "_classes.py",   # torch._classes reports this
             "m7diag_fake_missing": str(tmp_path / "pkg" / "gone.py"), "m7diag_fake_none": None}
    for name, path in fakes.items():
        module = types.ModuleType(name)
        module.__file__ = path
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.chdir(tmp_path)   # a relative name would otherwise resolve under the working directory
    (tmp_path / "_classes.py").write_text("")
    assert m.imported_files(tmp_path) == ["pkg/real.py"]


def test_layout_text():
    assert m._layout_text(range(170, 250)) == "170-249" and m._layout_text([33, 32, 38]) == "32, 33, 38"
    assert m._layout_text([]) == ""


# ---- the launcher ------------------------------------------------------------------------------------------------------

def _header():
    return [line for line in LAUNCHER.read_text().splitlines() if line.startswith("#")]


def test_launcher_header_carries_the_frozen_block_decision_resources_labels_and_disclosures():
    text = LAUNCHER.read_text()
    header = [line.rstrip() for line in _header()]
    for line in m.frozen_block():
        assert ("# " + line).rstrip() in header, line
    values = m.frozen_block_values([line[2:] for line in header if line.startswith("#   ")])
    for name in m.FROZEN_THRESHOLD_NAMES:
        assert values[name] == str(getattr(m, name)), name
    for line in ("#SBATCH --account=eecs", "#SBATCH --partition=share", "#SBATCH --constraint=el9",
                 "#SBATCH --cpus-per-task=2", "#SBATCH --mem=8G", "#SBATCH --time=04:00:00",
                 "#SBATCH --output=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out",
                 "#SBATCH --error=/nfs/hpc/share/sanchej7/Humanoid_Lite/logs/%x-%j.out"):
        assert line in text.splitlines()
    joined = " ".join(line.lstrip("# ") for line in header)
    for phrase in ("INVESTIGATION, NO GATE", "LEARNED clock-s2 stage gait", "SCRIPTED stage", "ORACLE layout and",
                   CORRECTION, "CAP RULE (deterministic, frozen before any diagnosis episode)", "the first 80 are run",
                   "L170 d1 through L234 d0", "layouts 32, 33 and 38", "Coordinator re-freeze of (D)'s measurement",
                   "classified by the contact NORMAL", "precedence wall_blocked > blocked_step_up > slow_progress > other",
                   "the LEADING foot", "F1 is TRAINED iff non-fall stalls > 0 AND 3 x blocked >= non-fall stalls AND "
                   "clears > 0", "0 clears -> the specificity check cannot hold -> F1 NOT trained",
                   "prints INCOMPLETE and decides nothing", "M7_CROSSING_DIAG DECISION F1 TRAINED",
                   "M7_CROSSING_DIAG SPECIFICITY", "M7_CROSSING_DIAG BLOCKED_STEP_UP_FRACTION non-fall",
                   "a foot standing inside a round plate reads as loaded", "haswell&el8",
                   "M7_DIAG_ALLOW_SOURCE_DRIFT=1", "byte-identical", "User approval recorded",
                   "run_crossing(repo, cache, spec, \"clocks2\")",
                   "M7_CROSSING_DIAG DECISION F1 TRAINED (<reason>); non-fall blocked-step-up fraction <b>/<n> = <v>; "
                   "clear share meeting the criterion <m>/<c> = <v>",
                   "Reading: only a \"M7_CROSSING_DIAG DECISION F1 TRAINED\" or \"M7_CROSSING_DIAG DECISION F1 NOT "
                   "TRAINED\" line followed by \"M7_CROSSING_DIAG_RUN_COMPLETE\" (job exit 0) decides",
                   "a log without a DECISION line (e.g. the job was killed) decides nothing"):
        assert phrase in joined, phrase
    assert "\n# DECISION (" not in text and "\n# - Falls:" not in text   # the rules are only the block's (verbatim)
    capped = ", ".join(f"L{layout} d{door}" for layout, door in CAPPED_OUT)
    assert f"capped out: {capped}." in joined
    doc = " ".join(m.__doc__.split())
    assert CORRECTION in doc and f"capped out: {capped}." in doc and "L170 d1 through L234 d0" in doc
    assert "records exist only" not in joined and "records exist only" not in doc   # the stale sentence is gone
    assert ('export PYTHONPATH="$REPO/src:$REPO/scripts" OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 '
            'PYTHONUNBUFFERED=1') in text
    subprocess.run(["bash", "-n", str(LAUNCHER)], check=True)


def _function(path, name):
    match = re.search(rf"^{name}\(\) \{{\n.*?^\}}\n", path.read_text(), re.S | re.M)
    assert match, (path, name)
    return match.group(0)


def test_the_smoke_limit_check_is_the_clocks2_launchers():
    assert _function(LAUNCHER, "smoke_limits_ok") == _function(CLOCKS2_LAUNCHER, "smoke_limits_ok")


def _clean_env(tmp_path, **extra):
    env = {k: v for k, v in os.environ.items() if not k.startswith(("SLURM_", "M7_DIAG_", "PYTHON"))}
    env.update(HOME=str(tmp_path), GIT_CONFIG_NOSYSTEM="1", **{k: str(v) for k, v in extra.items()})
    return env


def _stubs(tmp_path, python_rc=0, squeue="30:00", scontrol_command=None, python_sleep=False, preflight_sleep=False):
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    calls = tmp_path / "python-calls.txt"
    preflight = "exec /bin/sleep 60" if preflight_sleep else 'echo \'{"status": "PREFLIGHT_OK"}\'; exit 0'
    (bin_dir / "python").write_text(
        "#!/bin/bash\n"
        f'echo "$*" >> "{calls}"\n'
        f'case " $* " in *" --preflight "*) {preflight} ;; esac\n'
        + ("exec /bin/sleep 60\n" if python_sleep else "")
        + f"exit {python_rc}\n")
    (bin_dir / "squeue").write_text(f'#!/bin/bash\nprintf "%s\\n" "{squeue}"\n')
    (bin_dir / "sleep").write_text("#!/bin/bash\nexit 0\n")
    if scontrol_command is not None:
        (bin_dir / "scontrol").write_text(
            f'#!/bin/bash\necho "JobId=4242 JobName=x Command={scontrol_command} WorkDir=/tmp StdOut=/dev/null"\n')
    for stub in bin_dir.iterdir():
        stub.chmod(0o755)
    return bin_dir, calls


def _fake_tree(tmp_path, commit=True, skip=()):
    """A git checkout holding this workstream's three files (the real ones), nothing else."""
    root = tmp_path / "repo"
    for rel in m.OWN_FILES:
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(_REPO / rel, root / rel)
    git = ["git", "-C", str(root), "-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(git + ["init", "-q"], check=True)
    if commit:
        subprocess.run(git + ["add", *[rel for rel in m.OWN_FILES if rel not in skip]], check=True)
        subprocess.run(git + ["commit", "-q", "-m", "fake"], check=True)
    return root


def _launch(script, mode, env, cwd):
    return subprocess.run(["bash", str(script), mode], env=env, cwd=cwd, capture_output=True, text=True, timeout=120)


def test_real_mode_prefix_on_a_fake_tree_reaches_the_run(tmp_path):
    root = _fake_tree(tmp_path)
    bin_dir, calls = _stubs(tmp_path)
    out = tmp_path / "results" / "crossing-diag-20261003"
    env = _clean_env(tmp_path, PATH=f"{bin_dir}:{os.environ['PATH']}", PYTHON=bin_dir / "python", M7_DIAG_OUT=out)
    run = _launch(root / "slurm/repo20260923/cpu_m7_crossing_diag.sbatch", "run", env, tmp_path)   # REPO from its own path
    assert run.returncode == 0, (run.stdout, run.stderr)
    lines = calls.read_text().splitlines()
    assert lines == [f"scripts/bench/m7_crossing_diag.py --repo {root.resolve()} --out {out} --mode run --preflight",
                     f"scripts/bench/m7_crossing_diag.py --repo {root.resolve()} --out {out} --mode run"]
    assert f"M7_CROSSING_DIAG_RUN_COMPLETE -> {out}" in run.stdout and f"repo={root.resolve()}" in run.stdout
    assert "INCOMPLETE" not in run.stdout and not out.exists()   # the module (stubbed here) creates it


def test_real_mode_dry_run_stops_after_the_preflight(tmp_path):
    root = _fake_tree(tmp_path)
    bin_dir, calls = _stubs(tmp_path)
    env = _clean_env(tmp_path, PATH=f"{bin_dir}:{os.environ['PATH']}", PYTHON=bin_dir / "python",
                     M7_DIAG_OUT=tmp_path / "out", M7_DIAG_DRY_RUN=1, M7_DIAG_REPO=root)
    run = _launch(LAUNCHER, "run", env, tmp_path)   # the real launcher, REPO from M7_DIAG_REPO
    assert run.returncode == 0, (run.stdout, run.stderr)
    assert len(calls.read_text().splitlines()) == 1 and "--preflight" in calls.read_text()
    assert "DRY_RUN: stopping before any episode" in run.stdout


@pytest.mark.parametrize("case", ["out_exists", "uncommitted_change", "not_committed", "python_fails"])
def test_real_mode_refusals_and_a_failing_run_prints_incomplete(tmp_path, case):
    root = _fake_tree(tmp_path, skip=("tests/test_m7_crossing_diag.py",) if case == "not_committed" else ())
    bin_dir, calls = _stubs(tmp_path, python_rc=3 if case == "python_fails" else 0)
    out = tmp_path / "out"
    if case == "out_exists":
        out.mkdir()
    if case == "uncommitted_change":
        with open(root / "scripts/bench/m7_crossing_diag.py", "a") as stream:
            stream.write("# edit\n")
    env = _clean_env(tmp_path, PATH=f"{bin_dir}:{os.environ['PATH']}", PYTHON=bin_dir / "python", M7_DIAG_OUT=out,
                     M7_DIAG_REPO=root)
    run = _launch(LAUNCHER, "run", env, tmp_path)
    assert run.returncode != 0 and "M7_CROSSING_DIAG_RUN_COMPLETE" not in run.stdout
    if case == "python_fails":
        assert run.returncode == 3 and len(calls.read_text().splitlines()) == 2
        assert "M7_CROSSING_DIAG DECISION INCOMPLETE (the diagnosis exited 3; decides nothing)" in run.stdout
    else:
        assert run.returncode == 2 and not calls.exists(), (run.stdout, run.stderr)


@pytest.mark.parametrize("during", ["preflight", "run"])
def test_a_terminated_run_prints_incomplete(tmp_path, during):
    root = _fake_tree(tmp_path)
    bin_dir, calls = _stubs(tmp_path, python_sleep=during == "run", preflight_sleep=during == "preflight")
    env = _clean_env(tmp_path, PATH=f"{bin_dir}:{os.environ['PATH']}", PYTHON=bin_dir / "python",
                     M7_DIAG_OUT=tmp_path / "out", M7_DIAG_REPO=root)
    proc = subprocess.Popen(["bash", str(LAUNCHER), "run"], env=env, cwd=tmp_path, stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, text=True, start_new_session=True)
    want = 1 if during == "preflight" else 2
    deadline = time.time() + 60
    while time.time() < deadline and not (calls.exists() and len(calls.read_text().splitlines()) == want):
        time.sleep(0.2)
    time.sleep(0.5)
    os.killpg(proc.pid, signal.SIGTERM)   # Slurm's time limit / scancel signal the whole job
    stdout, _ = proc.communicate(timeout=60)
    assert len(calls.read_text().splitlines()) == want
    assert "M7_CROSSING_DIAG DECISION INCOMPLETE (the job was terminated; decides nothing)" in stdout
    assert "M7_CROSSING_DIAG_RUN_COMPLETE" not in stdout and proc.returncode != 0
    text = LAUNCHER.read_text()   # the trap is armed before the preflight
    assert text.index("trap 'echo \"M7_CROSSING_DIAG DECISION INCOMPLETE") < text.index(
        '"$PY" "$MODULE" --repo "$REPO" --out "$OUT" --mode run --preflight')


def test_repo_resolution_from_the_script_slurm_recorded_and_fail_closed(tmp_path):
    root = _fake_tree(tmp_path)
    spool = tmp_path / "spool"
    spool.mkdir()
    shutil.copy(LAUNCHER, spool / "slurm_script")   # sbatch runs a spooled copy
    bin_dir, calls = _stubs(tmp_path, scontrol_command=root / "slurm/repo20260923/cpu_m7_crossing_diag.sbatch")
    env = _clean_env(tmp_path, PATH=f"{bin_dir}:{os.environ['PATH']}", PYTHON=bin_dir / "python",
                     M7_DIAG_OUT=tmp_path / "out", M7_DIAG_DRY_RUN=1, SLURM_JOB_ID=4242)
    run = _launch(spool / "slurm_script", "run", env, tmp_path)
    assert run.returncode == 0 and f"repo={root.resolve()}" in run.stdout, (run.stdout, run.stderr)
    env.pop("SLURM_JOB_ID")
    run = _launch(spool / "slurm_script", "run", env, tmp_path)
    assert run.returncode == 2 and "set M7_DIAG_REPO" in run.stderr
    env.update(M7_DIAG_REPO=str(tmp_path / "nowhere"))
    assert _launch(spool / "slurm_script", "run", env, tmp_path).returncode == 2
    assert _launch(LAUNCHER, "bench", env | {"M7_DIAG_REPO": str(root)}, tmp_path).returncode == 2


@pytest.mark.parametrize("name, limit, smoke_dir, rc", [
    ("m7-crossing-diag-smoke", "30:00", "x-smoke/job-4242", 0),
    ("m7-crossing-diag-smoke", "1:00:00", "x-smoke", 0),
    ("m7-crossing-diag-smoke", "1:00:01", "x-smoke/job-4242", 2),
    ("m7-crossing-diag", "30:00", "x-smoke/job-4242", 2),
    ("m7-crossing-diag-smoke", "30:00", "x-smokes/job-4242", 2),
    ("m7-crossing-diag-smoke", "30:00", "EXISTS-smoke", 2)])
def test_smoke_mode_limits_output_dir_and_exit_codes(tmp_path, name, limit, smoke_dir, rc):
    root = _fake_tree(tmp_path)
    bin_dir, calls = _stubs(tmp_path, squeue=limit)
    target = tmp_path / smoke_dir
    if smoke_dir.startswith("EXISTS"):
        target.mkdir()
    env = _clean_env(tmp_path, PATH=f"{bin_dir}:{os.environ['PATH']}", PYTHON=bin_dir / "python", M7_DIAG_REPO=root,
                     SLURM_JOB_NAME=name, SLURM_JOB_ID=4242, M7_DIAG_SMOKE_DIR=target)
    run = _launch(LAUNCHER, "smoke", env, tmp_path)
    assert run.returncode == rc, (run.stdout, run.stderr)
    if rc == 0:
        assert calls.read_text().splitlines() == [
            f"scripts/bench/m7_crossing_diag.py --repo {root.resolve()} --out {target} --mode smoke --preflight",
            f"scripts/bench/m7_crossing_diag.py --repo {root.resolve()} --out {target} --mode smoke"]
        assert "M7_CROSSING_DIAG_SMOKE_PASS" in run.stdout
    else:
        assert not calls.exists()


def test_a_failing_smoke_exits_non_zero_without_the_pass_line(tmp_path):
    root = _fake_tree(tmp_path)
    bin_dir, calls = _stubs(tmp_path, python_rc=1)
    env = _clean_env(tmp_path, PATH=f"{bin_dir}:{os.environ['PATH']}", PYTHON=bin_dir / "python", M7_DIAG_REPO=root,
                     SLURM_JOB_NAME="m7-crossing-diag-smoke", SLURM_JOB_ID=4242,
                     M7_DIAG_SMOKE_DIR=tmp_path / "d-smoke" / "job-4242")
    run = _launch(LAUNCHER, "smoke", env, tmp_path)
    assert run.returncode == 1 and "M7_CROSSING_DIAG_SMOKE_PASS" not in run.stdout
    assert len(calls.read_text().splitlines()) == 2
