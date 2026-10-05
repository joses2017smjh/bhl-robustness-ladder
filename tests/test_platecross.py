"""Mission 7 learned crossing (workstream m7-platecross, 2026-10-02) without Isaac Sim.

Covers: the plate geometry against Mission 7's own layout (bhl_robust.mission.layout.world_xml), the declared
field (lattice, counts, clear centre, closed meshes), the Isaac Lab sub-terrain contract and the generator's
world frame, the scene check that the smoke's probe applies to the imported mesh (and what it catches), the
env cfg (R1 unchanged but scene.terrain; AST), the guarded task registration, the env.yaml checks against the
parent, the frozen selection rule at its boundaries (none qualified, ties, a v2 FAIL never counts even when
QUALIFIED, a missing JSON or a changed v2 protocol -> INCOMPLETE), the
write-once selection file, and the launcher header / array table / log guard / smoke isolation.
"""

from __future__ import annotations

import ast
import copy
import importlib.util
import json
import math
import re
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
TERRAIN = REPO / "src/bhl_robust/tasks/platecross_terrain.py"
ENV_CFG = REPO / "src/bhl_robust/tasks/platecross_env_cfg.py"
TASKS_INIT = REPO / "src/bhl_robust/tasks/__init__.py"
LAYOUT = REPO / "src/bhl_robust/mission/layout.py"
SELECT = REPO / "scripts/bench/platecross_select.py"
SBATCH = REPO / "slurm/repo20260923/gpu_platecross.sbatch"
R12 = REPO / "slurm/repo20260923/gpu_turngait_r12.sbatch"
LOGROOT = REPO / "external/Berkeley-Humanoid-Lite/logs/rsl_rl/humanoid"
R1_RES = REPO / "results/repo-gpu-20260923/turngait-r12-20261001"

# The task's text, word for word.
FROZEN_RULE = ("the three fine-tuned final checkpoints go through the unchanged turn qualification (turn_test v2 + "
               "cpu_turn_qualify); the qualified seed with the lowest push-fall rate (tie: lowest seed index) is the "
               "SINGLE stage gait run on bench v2 under bench v2's rule (N = 41; >= 39/41 clears, 0 falls, >= "
               "11/10/9/7 per heading) with the generic stage-gait override; no qualified seed -> NEGATIVE (no bench). "
               "Labels: LEARNED gait (fine-tuned from clock-s2 on plates); MuJoCo gates; bench v2's 180-deg timing "
               "caveat applies.")


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


pt = _load("platecross_terrain_under_test", TERRAIN)
sel = _load("platecross_select_under_test", SELECT)
layout = _load("mission_layout_under_test", LAYOUT)


# =============================================================================== Mission 7's plates

def _mission_plates(split="train", index=0):
    root = ET.fromstring(layout.world_xml(layout.generate(split, index)))
    return [g for g in root.iter("geom") if g.get("name", "").startswith("plate_")]


def test_plate_constants_are_mission7s_world_xml():
    shapes = set()
    for index in range(12):
        plates = _mission_plates("train", index)
        assert len(plates) == 4                                          # 2 doors x 2 sides
        for g in plates:
            size = tuple(float(x) for x in g.get("size").split())
            pos = tuple(float(x) for x in g.get("pos").split())
            assert g.get("euler") is None and g.get("quat") is None and g.get("axisangle") is None
            assert g.get("contype") is None and g.get("conaffinity") is None   # MuJoCo default: collides
            assert math.isclose(pos[2], pt.MJCF_PLATE_Z) and math.isclose(pos[2], pt.PLATE_HEIGHT_M / 2)
            if g.get("type") == "cylinder":
                shapes.add("round")
                assert size == pt.MJCF_ROUND_SIZE
                assert math.isclose(size[0], pt.ROUND_RADIUS_M) and math.isclose(2 * size[1], pt.PLATE_HEIGHT_M)
            else:
                assert g.get("type") == "box"
                shapes.add("square")
                assert size == pt.MJCF_SQUARE_SIZE
                assert math.isclose(2 * size[0], pt.SQUARE_SIDE_M) and math.isclose(2 * size[1], pt.SQUARE_SIDE_M)
                assert math.isclose(2 * size[2], pt.PLATE_HEIGHT_M)
    assert shapes == {"round", "square"}


def test_cited_source_lines_hold_the_plates():
    lines = LAYOUT.read_text().splitlines()
    assert """'type="cylinder" size=".24 .015"' if correct else 'type="box" size=".24 .24 .015"'""" in lines[151]
    assert 'pos="{x} {y} .015"' in lines[152]
    assert "def plate(self, i, side):" in lines[64]
    src = TERRAIN.read_text()
    assert "src/bhl_robust/mission/layout.py, world_xml(), lines" in src and "152-153" in src


# =============================================================================== the declared field

def test_declared_numbers():
    assert (pt.PITCH_M, pt.LATTICE_N, pt.TILE_M, pt.DISC_SECTIONS) == (1.2, 7, 8.4, 64)
    assert math.isclose(pt.TILE_M, pt.LATTICE_N * pt.PITCH_M)
    assert (pt.NUM_ROWS, pt.NUM_COLS, pt.BORDER_M, pt.TERRAIN_SEED, pt.CLEAR_CENTRE) == (10, 10, 20.0, 0, True)
    assert pt.PLATES_PER_TILE == {"round": 24, "square": 24}


def test_plate_layout_lattice_checkerboard_and_clear_centre():
    lay = pt.plate_layout()
    assert len(lay) == 48
    assert sum(s == "round" for s, _, _ in lay) == 24 and sum(s == "square" for s, _, _ in lay) == 24
    xy = np.array([[x, y] for _, x, y in lay])
    k = xy / pt.PITCH_M
    assert np.allclose(k, np.round(k), atol=1e-9) and np.abs(np.round(k)).max() == 3       # 7 x 7, centred
    assert not any(abs(x) < 1e-9 and abs(y) < 1e-9 for _, x, y in lay)                      # centre empty
    for s, x, y in lay:
        i, j = round(x / pt.PITCH_M), round(y / pt.PITCH_M)
        assert s == ("round" if (i + j) % 2 == 0 else "square")
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2) + np.eye(len(xy)) * 99
    assert math.isclose(d.min(), pt.PITCH_M)
    # nearest plate edge to the spawn point: 1.2 - 0.24 = 0.96 m (Chebyshev), past the +/-0.5 m spawn range
    edge = min(max(abs(x), abs(y)) - 0.5 * pt.SQUARE_SIDE_M for _, x, y in lay)
    assert math.isclose(edge, 0.96)
    # density as declared in the header: 0.68 plates/m^2 and 14.0 % of the floor
    area = 24 * 0.5 * 64 * 0.24 ** 2 * math.sin(2 * math.pi / 64) + 24 * 0.48 ** 2               # 64-gons + squares
    assert round(48 / pt.TILE_M ** 2, 2) == 0.68 and round(100 * area / pt.TILE_M ** 2, 1) == 14.0
    with pytest.raises(ValueError):
        pt.plate_layout((8.0, 8.0))                                                         # not an odd multiple


def test_plate_meshes_are_closed_with_outward_normals():
    trimesh = pytest.importorskip("trimesh")
    v, f = pt.disc_arrays(1.0, 2.0)
    m = trimesh.Trimesh(v, f, process=False)
    poly = 0.5 * 64 * 0.24 ** 2 * math.sin(2 * math.pi / 64)
    assert m.is_watertight and math.isclose(m.volume, poly * 0.03, rel_tol=1e-9)             # positive: outward
    assert np.allclose(np.hypot(v[2:, 0] - 1.0, v[2:, 1] - 2.0), 0.24)                      # rim on r = 0.24
    assert math.isclose(v[:, 2].min(), 0.0) and math.isclose(v[:, 2].max(), 0.03)
    v, f = pt.box_arrays(-1.0, 0.5)
    m = trimesh.Trimesh(v, f, process=False)
    assert m.is_watertight and math.isclose(m.volume, 0.48 * 0.48 * 0.03, rel_tol=1e-9)
    assert np.allclose(m.bounds, [[-1.24, 0.26, 0.0], [-0.76, 0.74, 0.03]])


def test_plates_terrain_follows_the_isaac_sub_terrain_contract(capsys):
    pytest.importorskip("trimesh")
    cfg = SimpleNamespace(size=(8.4, 8.4), round_radius_m=0.24, square_side_m=0.48, height_m=0.03, pitch_m=1.2,
                          disc_sections=64, clear_centre=True)
    pt._ANNOUNCED = False
    meshes, origin = pt.plates_terrain(0.37, cfg)
    assert len(meshes) == 1 and np.allclose(origin, [4.2, 4.2, 0.0])
    b = meshes[0].bounds
    assert np.allclose(b, [[0.0, 0.0, 0.0], [8.4, 8.4, 0.03]])
    first = capsys.readouterr().out
    assert first.strip() == pt.marker_line((8.4, 8.4), 0.24, 0.48, 0.03, 1.2, 64, True)
    pt.plates_terrain(0.9, cfg)
    assert capsys.readouterr().out == ""                                                    # printed once
    found = pt.find_plates(meshes[0].vertices, meshes[0].faces)
    assert len(found) == 48
    got = sorted((p["shape"], round(p["x"] - 4.2, 6), round(p["y"] - 4.2, 6)) for p in found)
    want = sorted((s, round(x, 6), round(y, 6)) for s, x, y in pt.plate_layout())
    assert got == want


def test_marker_line_text():
    assert pt.marker_line((8.4, 8.4), 0.24, 0.48, 0.03, 1.2, 64, True) == (
        "[platecross_terrain] plates tile 8.4 x 8.4 m: 24 round (r 0.24 m, 64-gon) + 24 square (0.48 m), "
        "height 0.03 m, pitch 1.2 m, centre clear")


# =============================================================================== the scene check

def _world_mesh(rows, cols, size=(8.4, 8.4), border=True, float32=True, tweak=None):
    """Isaac Lab's TerrainGenerator assembly: every tile centred (-size/2), moved to ((r + .5) sx, (c + .5) sy),
    the whole terrain shifted by -(rows sx, cols sy) / 2, plus a 1 m deep border; points stored as float32 (USD)."""
    tv, tf = pt.tile_arrays(size)
    if tweak:
        tv, tf = tweak(tv.copy(), tf.copy())
    verts, faces, n = [], [], 0
    for r in range(rows):
        for c in range(cols):
            off = np.array([(r + 0.5) * size[0] - 0.5 * size[0] - 0.5 * rows * size[0],
                            (c + 0.5) * size[1] - 0.5 * size[1] - 0.5 * cols * size[1], 0.0])
            verts.append(tv + off)
            faces.append(tf + n)
            n += len(tv)
    if border:
        bv, bf = pt.box_arrays(0.0, 0.0, side=rows * size[0] + 40.0, height=1.0)
        bv[:, 2] -= 1.0                                                                     # z in [-1, 0]
        verts.append(bv)
        faces.append(bf + n)
    v = np.vstack(verts)
    if float32:
        v = v.astype(np.float32).astype(np.float64)
    return v, np.vstack(faces)


def test_scene_check_accepts_the_declared_field_at_full_size():
    v, f = _world_mesh(pt.NUM_ROWS, pt.NUM_COLS)
    found = pt.find_plates(v, f)
    summary, problems = pt.check_scene_plates(found, pt.expected_world_plates())
    assert problems == [], problems
    assert summary["found"] == {"round": 2400, "square": 2400, "other": 0}
    assert summary["max_position_error_m"] < 1e-4 and summary["missing"] == summary["extra"] == 0
    assert math.isclose(summary["round"]["rmax_m"][0], 0.24, abs_tol=1e-4)
    assert math.isclose(summary["square"]["extent_m"][1], 0.48, abs_tol=1e-4)
    origins = pt.tile_centres()
    assert math.isclose(pt.spawn_clearance(origins, found), 0.96, abs_tol=1e-4)


def test_tile_centres_are_the_generator_origins():
    c = pt.tile_centres(10, 10, (8.4, 8.4))
    assert c.shape == (100, 2) and np.allclose(c[0], [-37.8, -37.8]) and np.allclose(c[-1], [37.8, 37.8])
    assert np.allclose(c[1], [-37.8, -29.4])                                                # row-major: (r, c)


def _drop_one_plate(v, f):
    keep = np.ones(len(f), bool)
    first_plate_face = 2                                                                    # after the floor
    keep[first_plate_face:first_plate_face + 4 * 64] = False                                # the first disc
    return v, f[keep]


@pytest.mark.parametrize("tweak, needle", [
    (_drop_one_plate, "round plates in the scene"),
    (lambda v, f: (v * np.array([1.0, 1.0, 4.0 / 3.0]), f), "height off"),                 # 4 cm plates
    (lambda v, f: (v + np.array([0.01, 0.0, 0.0]), f), "no scene plate within"),             # moved 1 cm
    (lambda v, f: (v * np.array([1.05, 1.05, 1.0]), f), "of another shape"),                 # 5 % larger
])
def test_scene_check_catches_a_wrong_field(tweak, needle):
    v, f = _world_mesh(2, 3, tweak=tweak)
    _, problems = pt.check_scene_plates(pt.find_plates(v, f), pt.expected_world_plates(2, 3))
    assert any(needle in p for p in problems), problems


def test_scene_check_rejects_an_empty_scene():
    v, f = _world_mesh(1, 1, tweak=lambda v, f: (v[:4], f[:2]))
    _, problems = pt.check_scene_plates(pt.find_plates(v, f), pt.expected_world_plates(1, 1))
    assert any("no plates found" in p for p in problems)


# =============================================================================== env cfg and registration

def _classes(src):
    tree = ast.parse(src)
    return tree, {n.name: n for n in tree.body if isinstance(n, ast.ClassDef)}


def test_env_cfg_changes_only_the_terrain():
    src = ENV_CFG.read_text()
    tree, classes = _classes(src)
    cfg = classes["HumanoidPlateCrossCfg"]
    assert [ast.unparse(b) for b in cfg.bases] == ["HumanoidTurnGaitClockCfg"]
    assert "from bhl_robust.tasks.arms_env_cfg import HumanoidTurnGaitClockCfg" in src
    body = [n for n in cfg.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
    assert len(body) == 1 and isinstance(body[0], ast.FunctionDef) and body[0].name == "__post_init__"
    stmts = body[0].body
    assert ast.unparse(stmts[0]) == "super().__post_init__()"
    assigned = {}
    for s in stmts[1:]:
        assert isinstance(s, ast.Assign) and len(s.targets) == 1
        t = ast.unparse(s.targets[0])
        assert t.startswith("self.scene.terrain."), t
        assigned[t] = ast.unparse(s.value)
    assert assigned == {"self.scene.terrain.terrain_type": "'generator'",
                        "self.scene.terrain.terrain_generator": "PLATES_TERRAINS_CFG",
                        "self.scene.terrain.max_init_terrain_level": "None",
                        "self.scene.terrain.visual_material": "None"}
    # nothing else in the module touches R1's classes or any other config section
    for node in ast.walk(tree):
        if isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
            tgt = ast.unparse(node.targets[0] if isinstance(node, ast.Assign) else node.target)
            assert "HumanoidTurnGaitClockCfg" not in tgt and "arms_env_cfg" not in tgt and "gait_clock_mdp" not in tgt
            for word in ("rewards", "observations", "events", "commands", "terminations", "actions", "curriculum", "sim."):
                assert word not in tgt, tgt


def test_env_cfg_terrain_generator_uses_the_declared_constants():
    src = ENV_CFG.read_text()
    tree, classes = _classes(src)
    fields = {ast.unparse(n.target): ast.unparse(n.value) for n in classes["PlatesTerrainCfg"].body
              if isinstance(n, ast.AnnAssign)}
    assert fields == {"function": "_pt.plates_terrain", "round_radius_m": "_pt.ROUND_RADIUS_M",
                      "square_side_m": "_pt.SQUARE_SIDE_M", "height_m": "_pt.PLATE_HEIGHT_M", "pitch_m": "_pt.PITCH_M",
                      "disc_sections": "_pt.DISC_SECTIONS", "clear_centre": "_pt.CLEAR_CENTRE"}
    assert [ast.unparse(b) for b in classes["PlatesTerrainCfg"].bases] == ["SubTerrainBaseCfg"]
    gen = next(n for n in tree.body if isinstance(n, ast.Assign) and ast.unparse(n.targets[0]) == "PLATES_TERRAINS_CFG")
    kw = {k.arg: ast.unparse(k.value) for k in gen.value.keywords}
    assert kw == {"size": "(_pt.TILE_M, _pt.TILE_M)", "border_width": "_pt.BORDER_M", "num_rows": "_pt.NUM_ROWS",
                  "num_cols": "_pt.NUM_COLS", "curriculum": "False", "seed": "_pt.TERRAIN_SEED", "use_cache": "False",
                  "sub_terrains": "{'plates': PlatesTerrainCfg(proportion=1.0)}"}
    assert 'TASK_ID = "Velocity-BHL-Arms-PlateCross-v0"' in src and sel.TASK_ID == "Velocity-BHL-Arms-PlateCross-v0"
    # the terrain module stays Isaac-free (the tests and the probe's analysis import it without Isaac)
    imports = [ast.unparse(n) for n in ast.parse(TERRAIN.read_text()).body if isinstance(n, (ast.Import, ast.ImportFrom))]
    assert imports == ["from __future__ import annotations", "import math", "import sys", "import numpy as np"]


def test_task_registration_is_one_guarded_block():
    src = TASKS_INIT.read_text()
    start, end = src.index("# --- m7-platecross ---"), src.index("# --- end m7-platecross ---")
    block = src[start:end]
    # F1 "PlateCross v2" (2026-10-04) registers its own id in its own guarded block, appended at the end of the file
    rest = src[:start] + src[end:]
    if "# --- m7-platecross2 ---" in rest:
        s2, e2 = rest.index("# --- m7-platecross2 ---"), rest.index("# --- end m7-platecross2 ---")
        rest = rest[:s2] + rest[e2:]
    assert src.count("# --- m7-platecross ---") == 1 and "PlateCross" not in rest
    assert "try:" in block and "except Exception as _exc:" in block and "NOT registered" in block
    assert 'id="Velocity-BHL-Arms-PlateCross-v0"' in block
    assert '"env_cfg_entry_point": _platecross.HumanoidPlateCrossCfg' in block
    assert '"rsl_rl_cfg_entry_point": _ARM_PPO_CFG' in block                                # R1's runner
    # R1's own registration is untouched
    assert '("Velocity-BHL-Arms-TurnGaitClock-v0", arms_env_cfg.HumanoidTurnGaitClockCfg),' in src[:start]
    ast.parse(src)


# =============================================================================== env.yaml checks

def _parent_env():
    runs = sorted(LOGROOT.glob("*_arms-turngait-clock-s2"))
    if not runs or not (runs[-1] / "params/env.yaml").is_file():
        pytest.skip("arms-turngait-clock-s2's env.yaml is not here")
    return sel.load_env_yaml(runs[-1] / "params/env.yaml")


def _platecross_terrain_dump():
    """The scene.terrain fields as Isaac Lab's dump_yaml would write PlateCross's (only the checked keys matter)."""
    return {"terrain_type": "generator", "max_init_terrain_level": None, "visual_material": None,
            "terrain_generator": {
                "class_type": "isaaclab.terrains.terrain_generator:TerrainGenerator", "seed": 0, "curriculum": False,
                "size": [8.4, 8.4], "border_width": 20.0, "border_height": 1.0, "num_rows": 10, "num_cols": 10,
                "color_scheme": "none", "horizontal_scale": 0.1, "vertical_scale": 0.005, "slope_threshold": 0.75,
                "difficulty_range": [0.0, 1.0], "use_cache": False, "cache_dir": "/tmp/isaaclab/terrains",
                "sub_terrains": {"plates": {
                    "function": "bhl_robust.tasks.platecross_terrain:plates_terrain", "proportion": 1.0,
                    "size": [8.4, 8.4], "flat_patch_sampling": None, "round_radius_m": 0.24, "square_side_m": 0.48,
                    "height_m": 0.03, "pitch_m": 1.2, "disc_sections": 64, "clear_centre": True}}}}


def _child(parent, seed=0):
    env = copy.deepcopy(parent)
    env["seed"] = seed
    env["scene"]["terrain"].update(_platecross_terrain_dump())
    return env


def test_check_env_accepts_the_declared_change_only():
    parent = _parent_env()
    assert sel.check_env(_child(parent), parent) == []
    assert sel.check_env(_child(parent, seed=2), parent) == []
    gc = _load("gait_clock_under_test_pc", REPO / "src/bhl_robust/eval/gait_clock.py")
    assert gc.check_recipe(_child(parent), "R1") == []                                        # R12's recipe check
    # the parent itself is not the plates terrain
    assert any("terrain_type 'plane'" in p for p in sel.check_env(parent, parent))


@pytest.mark.parametrize("mutate, needle", [
    (lambda e: e["rewards"]["feet_gait"].__setitem__("weight", 1.0), "rewards.feet_gait.weight"),
    (lambda e: e["events"]["push_robot"].__setitem__("interval_range_s", [10.0, 15.0]), "events.push_robot"),
    (lambda e: e["scene"].__setitem__("num_envs", 64), "scene.num_envs"),
    (lambda e: e["scene"]["terrain"]["physics_material"].__setitem__("static_friction", 0.5), "physics_material"),
    (lambda e: e["sim"].__setitem__("dt", 0.004), "sim.dt"),
    (lambda e: e["scene"]["terrain"]["terrain_generator"]["sub_terrains"]["plates"].__setitem__("pitch_m", 1.0),
     "plates.pitch_m"),
    (lambda e: e["scene"]["terrain"]["terrain_generator"]["sub_terrains"]["plates"].__setitem__("round_radius_m", 0.25),
     "plates.round_radius_m"),
    (lambda e: e["scene"]["terrain"]["terrain_generator"].__setitem__("curriculum", True), "curriculum"),
    (lambda e: e["scene"]["terrain"]["terrain_generator"].__setitem__("num_rows", 5), "num_rows"),
    (lambda e: e["scene"]["terrain"].__setitem__("max_init_terrain_level", 5), "max_init_terrain_level"),
])
def test_check_env_catches_any_other_change(mutate, needle):
    parent = _parent_env()
    env = _child(parent)
    mutate(env)
    probs = sel.check_env(env, parent)
    assert any(needle in p for p in probs), probs


def test_check_env_smoke_allows_the_env_count_only():
    parent = _parent_env()
    env = _child(parent)
    env["scene"]["num_envs"] = 64
    env["scene"]["terrain"]["num_envs"] = 64
    assert sel.check_env(env, parent, smoke=True) == []
    assert sel.check_env(env, parent, smoke=False) != []


# =============================================================================== the selection rule

def _row(qualify, falls, n=60, complete=True, v2="PASS"):
    """A decide() row: `qualify` is cpu_turn_qualify's verdict (True = QUALIFIED), `v2` turn_test v2's."""
    return {"complete": complete, "why": "" if complete else "missing", "v2_verdict": v2,
            "qualify_verdict": "QUALIFIED" if qualify else "NOT QUALIFIED", "falls": falls, "n": n}


R12_VERDICT = REPO / "scripts/bench/turngait_r12_verdict.py"


def test_frozen_rule_text():
    assert sel.FROZEN_RULE + " " + sel.FROZEN_LABELS == FROZEN_RULE
    assert sel.NEGATIVE_LINE == "NEGATIVE: no qualified seed, no bench"
    assert sel.SEEDS == (0, 1, 2) and sel.N_PUSH == 60 and sel.final_ckpt() == "model_8998.pt"
    assert sel.final_ckpt(3) == "model_6001.pt" and sel.PARENT_RUN == "arms-turngait-clock-s2"


def test_reading_is_the_joint_turn_qualification():
    # the frozen rule's "unchanged turn qualification (turn_test v2 + cpu_turn_qualify)", counted per seed as the
    # R1 / v5 joint rule counts it: v2 PASS AND QUALIFIED (never the qualify verdict alone)
    assert "both PASSES turn_test v2" in sel.READING and "AND is QUALIFIED by cpu_turn_qualify's unchanged rule" in sel.READING
    assert "a seed that fails v2 does not count even if cpu_turn_qualify says QUALIFIED" in sel.READING
    for text in (SBATCH.read_text(), SELECT.read_text()):
        assert "not a selection gate" not in text and "does not gate" not in text
    r12 = _load("turngait_r12_verdict_under_test", R12_VERDICT)
    assert sel.V2_RULE == r12.V2_RULE                                                       # the unchanged v2
    for v2 in ("PASS", "FAIL", None, "INCOMPLETE"):
        for q in ("QUALIFIED", "NOT QUALIFIED", None, "INCOMPLETE"):
            assert sel.seed_qualified(v2, q) == r12.seed_counts(True, v2, q) == (v2 == "PASS" and q == "QUALIFIED")


def test_decide_boundaries():
    d = sel.decide
    assert d({0: _row(False, 3), 1: _row(False, 0), 2: _row(False, 9)}) == {
        "verdict": "NEGATIVE", "seed": None, "reason": "NEGATIVE: no qualified seed, no bench"}
    assert d({0: _row(False, 1), 1: _row(True, 9), 2: _row(False, 0)})["seed"] == 1           # the only qualified
    assert d({0: _row(True, 9), 1: _row(True, 8), 2: _row(True, 9)})["seed"] == 1             # lowest rate
    r = d({0: _row(True, 5), 1: _row(True, 5), 2: _row(True, 7)})
    assert r["verdict"] == "SELECTED" and r["seed"] == 0 and "tie" in r["reason"]              # tie -> lowest index
    assert d({0: _row(False, 0), 1: _row(True, 4), 2: _row(True, 4)})["seed"] == 1
    assert d({0: _row(True, 0), 1: _row(True, 0), 2: _row(True, 0)})["seed"] == 0
    # a NOT QUALIFIED seed never wins on a lower push rate
    assert d({0: _row(False, 0), 1: _row(False, 0), 2: _row(True, 9)})["seed"] == 2
    # a seed that FAILS v2 never counts, even QUALIFIED with the lowest push rate (the joint qualification)
    r = d({0: _row(False, 0), 1: _row(True, 1, v2="FAIL"), 2: _row(True, 4)})
    assert r["verdict"] == "SELECTED" and r["seed"] == 2 and "s1 (v2 FAIL, QUALIFIED)" in r["reason"]
    assert d({0: _row(True, 5, v2="FAIL"), 1: _row(True, 5), 2: _row(True, 5)})["seed"] == 1  # tie among the qualified
    assert d({0: _row(True, 9), 1: _row(True, 0, v2="FAIL"), 2: _row(True, 0, v2="FAIL")})["seed"] == 0
    # ... so the only QUALIFIED seed failing v2 -> NEGATIVE, as does every seed failing v2
    assert d({0: _row(False, 1), 1: _row(True, 0, v2="FAIL"), 2: _row(False, 2)})["verdict"] == "NEGATIVE"
    assert d({s: _row(True, 0, v2="FAIL") for s in sel.SEEDS})["verdict"] == "NEGATIVE"
    # v2 PASS without QUALIFIED never counts either
    assert d({s: _row(False, 0, v2="PASS") for s in sel.SEEDS})["verdict"] == "NEGATIVE"
    # INCOMPLETE beats everything: a missing or incomplete seed, even with a qualified one in hand
    assert d({0: _row(True, 1), 1: _row(True, 2)})["verdict"] == "INCOMPLETE"
    assert d({0: _row(True, 1), 1: _row(True, 2), 2: _row(True, 0, complete=False)})["verdict"] == "INCOMPLETE"
    assert d({0: _row(False, 1), 1: _row(False, 2), 2: _row(False, 0, complete=False)})["verdict"] == "INCOMPLETE"
    assert d({0: _row(True, 1), 1: _row(True, 2), 2: _row(True, 3), 3: _row(True, 0)})["verdict"] == "INCOMPLETE"
    assert d({})["verdict"] == "INCOMPLETE"
    # ... and so does a complete row without a valid v2 or qualify verdict
    for bad in ({"v2_verdict": None}, {"v2_verdict": "INCOMPLETE"}, {"qualify_verdict": "INCOMPLETE"},
                {"qualify_verdict": None}):
        r = d({0: _row(True, 1), 1: _row(True, 2), 2: {**_row(True, 0), **bad}})
        assert r["verdict"] == "INCOMPLETE" and "no valid v2 / qualify verdict" in r["reason"], r


def _make_seed(res, logs, seed, qualify="QUALIFIED", falls=5, v2="PASS", n=60, iters=3000, **over):
    run = sel.run_name(seed)
    final = sel.final_ckpt(iters)
    rd = logs / f"2026-10-03_00-00-0{seed}_{run}"
    (rd / "exported").mkdir(parents=True, exist_ok=True)
    (rd / final).write_bytes(b"ckpt")
    deploy = rd / "exported" / "deploy.yaml"
    deploy.write_text("policy_dt: 0.04\nnum_observations: 77\ngait_clock:\n  period_s: 0.8\n")
    (rd / "exported" / "policy.onnx").write_bytes(b"onnx-" + bytes([seed]))
    for sub in ("training", "turn-test-v2", "qualify"):
        (res / sub).mkdir(parents=True, exist_ok=True)
    rec = {"run": run, "task": sel.TASK_ID, "seed": seed, "final_ckpt": final, "run_dir": str(rd), "guard": [],
           "env_check": "PLATECROSS ENV: OK x", "recipe_check": "GAIT-CLOCK RECIPE: OK R1 x",
           **{k: True for k in sel.RECORD_FLAGS}}
    rec.update(over.get("record", {}))
    v2doc = {"protocol": "v2", "verdict": v2, "deploy": str(deploy), "rule": copy.deepcopy(sel.V2_RULE)}
    v2doc.update(over.get("v2doc", {}))
    q = {"run": run, "deploy": str(deploy), "verdict": qualify, "detail": "d",
         "clauses": {"push": {"falls": falls, "n": n, "rate": round(falls / n, 4)}}}
    q.update(over.get("qdoc", {}))
    for kind, path, doc in (("training", res / "training" / f"{run}.json", rec),
                            ("v2", res / "turn-test-v2" / f"{run}.json", v2doc),
                            ("qualify", res / "qualify" / f"{run}__qualify.json", q)):
        if kind not in over.get("skip", ()):
            path.write_text(json.dumps(doc))
    return rd


def _select(res, capsys, *extra):
    assert sel.main(["select", "--res", str(res), *extra]) == 0
    return capsys.readouterr().out


def test_selection_writes_once_with_the_export_and_its_provenance(tmp_path, capsys):
    res, logs = tmp_path / "res", tmp_path / "logs"
    _make_seed(res, logs, 0, qualify="NOT QUALIFIED", falls=0)
    _make_seed(res, logs, 1, falls=1, v2="FAIL")              # QUALIFIED with the lowest rate, but fails v2: excluded
    rd2 = _make_seed(res, logs, 2, falls=4)
    out = _select(res, capsys)
    assert "PLATECROSS SELECTION: SELECTED arms-platecross-clocks2-s2 " in out and "(tie with" not in out
    assert "s1 (v2 FAIL, QUALIFIED)" in out and "s0 (v2 PASS, NOT QUALIFIED)" in out
    doc = json.loads((res / "selection.json").read_text())
    s = doc["selected"]
    assert doc["verdict"] == "SELECTED" and s["seed"] == 2 and s["push"] == {"falls": 4, "n": 60, "rate": 0.0667}
    assert s["deploy_yaml"] == str(rd2 / "exported/deploy.yaml") and s["policy_onnx"] == str(rd2 / "exported/policy.onnx")
    assert s["sha256"]["policy_onnx"] == sel.sha256(rd2 / "exported/policy.onnx") and s["gait_clock_block"] is True
    assert s["num_observations_line"] == "num_observations: 77"
    assert (s["v2_verdict"], s["qualify_verdict"]) == ("PASS", "QUALIFIED")
    per = doc["per_seed"]
    assert (per["s1"]["v2_verdict"], per["s1"]["qualify_verdict"], per["s1"]["qualified"], per["s1"]["falls"]) == (
        "FAIL", "QUALIFIED", False, 1)
    assert (per["s0"]["v2_verdict"], per["s0"]["qualify_verdict"], per["s0"]["qualified"]) == ("PASS", "NOT QUALIFIED", False)
    assert per["s2"]["qualified"] is True
    assert doc["rule"] == sel.FROZEN_RULE and doc["labels"] == sel.FROZEN_LABELS and doc["reading"] == sel.READING
    assert doc["v2_rule"] == sel.V2_RULE and doc["disclosed"] == sel.DISCLOSED and sel.SWING_DISCLOSURE in doc["disclosed"]
    assert "COORDINATOR" in doc["bench"]
    # results change afterwards: the recorded selection stays
    _make_seed(res, logs, 1, falls=0)
    out = _select(res, capsys)
    assert "not overwritten" in out and json.loads((res / "selection.json").read_text()) == doc


def test_selection_negative_when_the_only_qualified_seed_fails_v2(tmp_path, capsys):
    res, logs = tmp_path / "res", tmp_path / "logs"
    _make_seed(res, logs, 0, qualify="NOT QUALIFIED", falls=1)
    _make_seed(res, logs, 1, falls=0, v2="FAIL")
    _make_seed(res, logs, 2, qualify="NOT QUALIFIED", falls=2)
    out = _select(res, capsys)
    assert "PLATECROSS SELECTION: NEGATIVE: no qualified seed, no bench" in out
    assert "s1 v2 FAIL / QUALIFIED (push 0/60)" in out and "s0 v2 PASS / NOT QUALIFIED (push 1/60)" in out
    doc = json.loads((res / "selection.json").read_text())
    assert doc["verdict"] == "NEGATIVE" and doc["selected"] is None and doc["bench"] == sel.NEGATIVE_LINE
    assert doc["per_seed"]["s1"]["qualify_verdict"] == "QUALIFIED" and doc["per_seed"]["s1"]["qualified"] is False


def test_selection_negative_is_written_with_no_bench(tmp_path, capsys):
    res, logs = tmp_path / "res", tmp_path / "logs"
    for s in sel.SEEDS:
        _make_seed(res, logs, s, qualify="NOT QUALIFIED", falls=s)
    out = _select(res, capsys)
    assert "PLATECROSS SELECTION: NEGATIVE: no qualified seed, no bench" in out
    doc = json.loads((res / "selection.json").read_text())
    assert doc["verdict"] == "NEGATIVE" and doc["selected"] is None and doc["bench"] == sel.NEGATIVE_LINE


@pytest.mark.parametrize("seed2_over, why", [
    ({"skip": ("qualify",)}, "no qualify JSON"),
    ({"skip": ("v2",)}, "no v2 JSON"),
    ({"skip": ("training",)}, "no training JSON"),
    ({"qdoc": {"run": "arms-platecross-clocks2-s1"}}, "another run"),
    ({"qdoc": {"deploy": "/elsewhere/deploy.yaml"}}, "another deploy"),
    ({"v2doc": {"deploy": "/elsewhere/deploy.yaml"}}, "another deploy"),
    ({"v2doc": {"protocol": "v2x"}}, "not a turn_test v2"),
    ({"v2doc": {"verdict": "INCOMPLETE"}}, "not a turn_test v2"),
    ({"v2doc": {"rule": None}}, "not the unchanged turn_test v2 protocol (no rule block)"),
    ({"v2doc": {"rule": {**sel.V2_RULE, "turn_min_deg": 120.0}}}, "not the unchanged turn_test v2 protocol (turn_min_deg)"),
    ({"v2doc": {"rule": {**sel.V2_RULE, "walk_seed": 1}}}, "not the unchanged turn_test v2 protocol (walk_seed)"),
    ({"qdoc": {"verdict": "INCOMPLETE"}}, "qualify verdict"),
    ({"n": 59}, "need n = 60"),
    ({"qdoc": {"clauses": {"push": {"falls": True, "n": 60}}}}, "no push falls"),
    ({"record": {"guard": ["policy-obs-not-77"]}}, "record-guard"),
    ({"record": {"parent_loaded_in_log": False}}, "parent_loaded_in_log"),
    ({"record": {"env_check": "PLATECROSS ENV: FAIL (x)"}}, "env-check"),
    ({"record": {"final_ckpt": "model_6001.pt"}}, "final_ckpt"),
])
def test_selection_incomplete_writes_nothing(tmp_path, capsys, seed2_over, why):
    res, logs = tmp_path / "res", tmp_path / "logs"
    _make_seed(res, logs, 0, falls=3)
    _make_seed(res, logs, 1, falls=2)
    _make_seed(res, logs, 2, **seed2_over)
    out = _select(res, capsys)
    assert "PLATECROSS SELECTION: INCOMPLETE" in out and why in out, out
    assert not (res / "selection.json").exists()


def test_selection_needs_the_final_checkpoint_in_the_run_dir(tmp_path, capsys):
    res, logs = tmp_path / "res", tmp_path / "logs"
    for s in sel.SEEDS:
        rd = _make_seed(res, logs, s)
    (rd / "model_8998.pt").unlink()
    assert "run dir has no model_8998.pt" in _select(res, capsys)


def test_write_once_never_overwrites(tmp_path):
    p = tmp_path / "x.json"
    assert sel.write_once(p, {"a": 1}, "t") is True
    assert sel.write_once(p, {"a": 2}, "t") is False
    assert json.loads(p.read_text()) == {"a": 1} and sorted(x.name for x in tmp_path.iterdir()) == ["x.json"]


def _record_args(out, **over):
    a = {"out": str(out), "run": "arms-platecross-clocks2-s0", "task": sel.TASK_ID, "seed": "0",
         "final-ckpt": "model_8998.pt", "run-dir": "/r", "job": "1", "parent-dir": "/p", "parent-ckpt": "model_5999.pt",
         "parent-sha256": "ab", "ckpt-ok": "true", "task-ok": "true", "parent-ok": "true", "plates-ok": "true",
         "gait-ok": "true", "push-ok": "true", "guard": "", "env-check": "PLATECROSS ENV: OK y",
         "recipe-check": "GAIT-CLOCK RECIPE: OK R1 y"}
    a.update(over)
    return ["write-record"] + [f"--{k}={v}" for k, v in a.items()]


def _check_record(path, capsys):
    capsys.readouterr()
    sel.main(["check-record", "--record", str(path), "--run", "arms-platecross-clocks2-s0", "--final-ckpt",
              "model_8998.pt", "--run-dir", "/r"])
    return capsys.readouterr().out.splitlines()[-1]


def test_record_cli_roundtrip(tmp_path, capsys):
    out = tmp_path / "rec.json"
    sel.main(_record_args(out))
    assert _check_record(out, capsys) == ""                                                 # no problems
    rec = json.loads(out.read_text())
    assert rec["guard"] == [] and rec["parent_ckpt_sha256"] == "ab" and all(rec[k] is True for k in sel.RECORD_FLAGS)
    sel.main(_record_args(out, **{"plates-ok": "false"}))
    assert "not overwritten" in capsys.readouterr().out and json.loads(out.read_text()) == rec   # written once
    bad = tmp_path / "bad.json"
    sel.main(_record_args(bad, **{"plates-ok": "false", "guard": " no-plates-terrain-marker", "env-check": "x",
                                  "recipe-check": "y"}))
    line = _check_record(bad, capsys)
    for needle in ("record-plates_marker_in_log-not-true", "record-guard:no-plates-terrain-marker",
                   "record-env-check-not-ok", "record-recipe-check-not-ok"):
        assert needle in line
    assert "unreadable-training-record" in _check_record(tmp_path / "missing.json", capsys)
    other = tmp_path / "other.json"
    sel.main(_record_args(other, **{"run-dir": "/elsewhere"}))
    assert "record-is-for-another-run-dir" in _check_record(other, capsys)


def _rsl_checkpoint(path, seed, width=77):
    """An rsl_rl-shaped checkpoint (actor MLP width -> 16 -> 16 -> 22, ELU) and the actor as a torch module."""
    torch = pytest.importorskip("torch")
    torch.manual_seed(seed)
    actor = torch.nn.Sequential(torch.nn.Linear(width, 16), torch.nn.ELU(), torch.nn.Linear(16, 16), torch.nn.ELU(),
                                torch.nn.Linear(16, 22))
    sd = {f"actor.{k}": v for k, v in actor.state_dict().items()}
    sd.update({"critic.0.weight": torch.zeros(1, 80), "critic.0.bias": torch.zeros(1), "std": torch.ones(22)})
    torch.save({"model_state_dict": sd, "iter": 8998}, path)
    return actor


def _onnx(actor, path, width=77):
    import inspect
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnxruntime")
    kw = {"dynamo": False} if "dynamo" in inspect.signature(torch.onnx.export).parameters else {}
    try:
        torch.onnx.export(actor, torch.zeros(1, width), str(path), input_names=["obs"], output_names=["actions"],
                          opset_version=11, **kw)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"torch.onnx.export unavailable here: {exc!r}")


def test_check_export_matches_only_the_final_checkpoint(tmp_path, capsys):
    final = _rsl_checkpoint(tmp_path / "model_8998.pt", seed=1)
    _rsl_checkpoint(tmp_path / "model_5999.pt", seed=2)
    _onnx(final, tmp_path / "policy.onnx")
    assert sel.export_actor_error(tmp_path / "policy.onnx", tmp_path / "model_8998.pt") < 1e-5
    assert sel.export_actor_error(tmp_path / "policy.onnx", tmp_path / "model_5999.pt") > 1e-3
    sel.main(["check-export", "--onnx", str(tmp_path / "policy.onnx"), "--ckpt", str(tmp_path / "model_8998.pt"),
              "--parent", str(tmp_path / "model_5999.pt")])
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("PLATECROSS EXPORT: OK")
    sel.main(["check-export", "--onnx", str(tmp_path / "policy.onnx"), "--ckpt", str(tmp_path / "model_5999.pt"),
              "--parent", str(tmp_path / "model_8998.pt")])
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("PLATECROSS EXPORT: FAIL")
    sel.main(["check-export", "--onnx", str(tmp_path / "missing.onnx"), "--ckpt", str(tmp_path / "model_8998.pt"),
              "--parent", str(tmp_path / "model_5999.pt")])
    assert capsys.readouterr().out.strip().splitlines()[-1].startswith("PLATECROSS EXPORT: FAIL")


def test_smoke_standins_reproduce_r1s_selection(tmp_path, capsys):
    if not (R1_RES / "qualify/arms-turngait-clock-s2__qualify.json").is_file():
        pytest.skip("R1's qualify JSONs are not here")
    logs = tmp_path / "logs"
    rd = _make_seed(tmp_path / "scratch", logs, 0, iters=3)
    rec = tmp_path / "rec.json"
    rec.write_text((tmp_path / "scratch/training/arms-platecross-clocks2-s0.json").read_text().replace(
        '"arms-platecross-clocks2-s0"', '"arms-platecross-clocks2-s0-smoke"'))
    assert sel.main(["smoke-standins", "--dest", str(tmp_path / "x"), "--source-res", str(R1_RES),
                     "--training-record", str(rec)]) == 1                                   # not a -smoke dir
    dest = tmp_path / "platecross-smoke" / "standin"
    assert sel.main(["smoke-standins", "--dest", str(dest), "--source-res", str(R1_RES),
                     "--training-record", str(rec)]) == 0
    out = _select(dest, capsys, "--suffix=-smoke", "--iters", "3")
    assert "PLATECROSS SELECTION: SELECTED arms-platecross-clocks2-s2-smoke " in out
    s = json.loads((dest / "selection.json").read_text())["selected"]
    assert s["seed"] == 2 and s["push"]["falls"] == 9 and s["deploy_yaml"] == str(rd / "exported/deploy.yaml")
    # the SYNTHETIC joint check: stand-in s1 keeps its real v2 FAIL, its qualify JSON relabelled QUALIFIED 0/60;
    # the qualify verdict alone would pick s1 (0/60 < 9/60), the joint reading still picks s2
    joint = tmp_path / "platecross-smoke" / "joint"
    assert sel.main(["smoke-standins", "--joint-check", "--dest", str(joint), "--source-res", str(R1_RES),
                     "--training-record", str(rec)]) == 0
    q1 = json.loads((joint / "qualify/arms-platecross-clocks2-s1-smoke__qualify.json").read_text())
    assert q1["verdict"] == "QUALIFIED" and q1["clauses"]["push"]["falls"] == 0 and "SYNTHETIC" in q1["standin"]
    rows = {k: sel.read_seed(joint, k, suffix="-smoke", iters=3) for k in sel.SEEDS}
    assert sel.decide({k: dict(r, v2_verdict="PASS") for k, r in rows.items()})["seed"] == 1     # were v2 not a gate
    out = _select(joint, capsys, "--suffix=-smoke", "--iters", "3")
    assert "PLATECROSS SELECTION: SELECTED arms-platecross-clocks2-s2-smoke " in out
    doc = json.loads((joint / "selection.json").read_text())
    p1 = doc["per_seed"]["s1"]
    assert doc["selected"]["seed"] == 2
    assert (p1["v2_verdict"], p1["qualify_verdict"], p1["qualified"], p1["falls"]) == ("FAIL", "QUALIFIED", False, 0)


# =============================================================================== launcher

def _header():
    lines = []
    for ln in SBATCH.read_text().splitlines()[1:]:
        if not ln.startswith("#"):
            break
        if not ln.startswith("#SBATCH"):
            lines.append(ln[1:].strip())
    return " ".join(" ".join(lines).split())


def _body():
    return "\n".join(ln for ln in SBATCH.read_text().splitlines() if not ln.lstrip().startswith("#"))


def test_launcher_header_holds_the_frozen_rule_verbatim():
    h = _header()
    assert FROZEN_RULE in h
    assert " ".join(sel.READING.split()) in h
    assert "SLURM_JOBS.md, 'User approval recorded 2026-10-02 14:15', item (B)" in h
    assert "src/bhl_robust/mission/layout.py:152-153" in h and "pitch 1.2 m" in h and "8.4 m tiles" in h
    assert "COORDINATOR's later submission" in h and "drifts on straight walks" in h
    # disclosed: R1's unchanged feet_swing_height targets absolute world z (flat-plane assumption)
    assert " ".join(sel.SWING_DISCLOSURE.split()) in h and sel.SWING_DISCLOSURE in sel.DISCLOSED
    assert "(gates with the qualify verdict; never overwritten)" in h
    gc_src = (REPO / "src/bhl_robust/tasks/gait_clock_mdp.py").read_text()
    assert "flat plane, ground z = 0" in gc_src and "z_origin = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]" in gc_src


def test_launcher_resources_and_array_table():
    sb = SBATCH.read_text()
    r12 = R12.read_text()
    for key in ("--account=", "--partition=", "--constraint=", "--cpus-per-task=", "--mem=", "--gres="):
        mine = re.search(rf"^#SBATCH {key}.*$", sb, re.M).group(0)
        assert mine == re.search(rf"^#SBATCH {key}.*$", r12, re.M).group(0), key             # the turning launchers'
    assert "#SBATCH --array=0-2%3" in sb and "#SBATCH --time=12:00:00" in sb
    body = _body()
    for line in ("TASK_ID=Velocity-BHL-Arms-PlateCross-v0", "PREFIX=arms-platecross-clocks2",
                 "PARENT_RUN=arms-turngait-clock-s2", "PARENT_ITER=5999", "FT_ITERS=3000", "OBS_POLICY=77",
                 "OBS_CRITIC=80", "0|1|2) SEED=$IDX;;", 'FINAL_CKPT="model_$((PARENT_ITER + MAX_ITER - 1)).pt"',
                 'export MAX_ITER=$FT_ITERS RUN_NAME="$PREFIX-s${SEED}"',
                 'export MAX_ITER=3 NUM_ENVS=64 RUN_NAME="$PREFIX-s${SEED}-smoke"',
                 "RES=$REPO/results/repo-gpu-20260923/platecross-20261002",
                 "printf 'agent.resume=true\\nagent.load_run=%s\\nagent.load_checkpoint=model_%s.pt\\n' \"$PARENT_DIR\" \"$PARENT_ITER\"",
                 "dr_overrides_scale 1.0",
                 'TASK="$TASK_ID" bhl_exec "$REPO/slurm/inner/export_one_humanoid.sh"',
                 'TURNQ_RUN="$RUN_NAME" TURNQ_OUT_DIR="$RES/qualify" bash "$REPO/slurm/repo20260923/cpu_turn_qualify.sbatch"',
                 '"$PYH" "$SELECT" select --res "$RES" )'):
        assert line in body, line
    assert re.search(r"SMOKE_RES=\S+/campaign-m7-platecross/platecross-smoke\n", body) and "*-smoke) ;;" in body
    assert "TASK_ID" in body and sel.PREFIX == "arms-platecross-clocks2" and sel.FT_ITERS == 3000
    assert 5999 + 3000 - 1 == 8998


GOOD_LOG = """\
task=Velocity-BHL-Arms-PlateCross-v0 run=arms-platecross-clocks2-s0 seed=0 envs=cfg iters=3000
[platecross_terrain] plates tile 8.4 x 8.4 m: 24 round (r 0.24 m, 64-gon) + 24 square (0.48 m), height 0.03 m, pitch 1.2 m, centre clear
[INFO]: Loading model checkpoint from: /u/logs/rsl_rl/humanoid/2026-10-02_03-58-04_arms-turngait-clock-s2/model_5999.pt
|   0   | base_velocity | UniformVelocityCommand |
|   0   | push_robot |        (5.0, 9.0)       |
| Active Observation Terms in Group: 'policy' (shape: (77,)) |
| Active Observation Terms in Group: 'critic' (shape: (80,)) |
|   10  | feet_air_time              |    0.0 |
|   17  | feet_gait                  |    0.5 |
|   18  | feet_swing_height          |  -20.0 |
[gait_clock_mdp] feet_gait bodies ['leg_left_ankle_roll', 'leg_right_ankle_roll'] offsets [0.0, 0.5] period 0.8 threshold 0.55 step_dt 0.04
[gait_clock_mdp] feet_swing_height bodies ['leg_left_ankle_roll', 'leg_right_ankle_roll']: link-origin z median 0.0600 m (n=8)
train.sh: reached 3000 logged iterations
"""


def _guard(tmp_path, log_text):
    """The launcher's own guard() and evidence regexes (extracted from the sbatch), run in bash."""
    sb = SBATCH.read_text()
    fn = re.search(r"^guard\(\) \{.*?^\}$", sb, re.S | re.M).group(0)
    regs = "\n".join(re.search(rf"^{k}=.*$", sb, re.M).group(0)
                     for k in ("TASK_RE", "PARENT_RE", "PLATES_RE", "GAIT_RE", "PUSH_RE"))
    log = tmp_path / "train.log"
    log.write_text(log_text)
    script = ("set -euo pipefail\nTASK_ID=Velocity-BHL-Arms-PlateCross-v0\nRUN_NAME=arms-platecross-clocks2-s0\nSEED=0\n"
              "PARENT_DIR=2026-10-02_03-58-04_arms-turngait-clock-s2\nPARENT_ITER=5999\nOBS_POLICY=77\nOBS_CRITIC=80\n"
              f"{regs}\n{fn}\n"
              'why="$(guard "$1")"\n'
              'grep -qE "$TASK_RE" "$1" || why="$why task"\n'
              'grep -qE "$PARENT_RE" "$1" || why="$why parent"\n'
              'grep -qE "$PLATES_RE" "$1" || why="$why plates"\n'
              'grep -qE "$GAIT_RE" "$1" || why="$why gait"\n'
              'grep -qE "$PUSH_RE" "$1" || why="$why push"\n'
              'echo "WHY:$why"\n')
    out = subprocess.run(["bash", "-c", script, "guard", str(log)], capture_output=True, text=True, check=True)
    return out.stdout.strip().split("WHY:", 1)[1].split()


def test_launcher_log_guard_accepts_the_platecross_log(tmp_path):
    assert _guard(tmp_path, GOOD_LOG) == []
    assert pt.marker_line((8.4, 8.4), 0.24, 0.48, 0.03, 1.2, 64, True) in GOOD_LOG          # the real marker


@pytest.mark.parametrize("old, new, reason", [
    ("PlateCross-v0 run", "TurnGaitClock-v0 run", "task"),
    ("arms-turngait-clock-s2/model_5999.pt", "arms-turngait-clock-s1/model_5999.pt", "parent"),
    ("Loading model checkpoint", "Not loading", "parent"),
    ("pitch 1.2 m", "pitch 1.0 m", "plates"),
    ("24 round (r 0.24 m", "24 round (r 0.25 m", "plates"),
    ("height 0.03 m", "height 0.04 m", "plates"),
    ("(shape: (77,))", "(shape: (75,))", "policy-obs-not-77"),
    ("feet_gait                  |    0.5", "feet_gait | 1.0", "gait"),
    ("(5.0, 9.0)", "(10.0, 15.0)", "push"),
    ("|    0.0 |", "|    2.0 |", "feet_air_time-weight-not-0.0"),
    ("train.sh: reached 3000", "train.sh: none", "no-logged-iterations"),
])
def test_launcher_log_guard_rejects(tmp_path, old, new, reason):
    assert old in GOOD_LOG
    assert reason in _guard(tmp_path, GOOD_LOG.replace(old, new))


def test_launcher_log_guard_flags_extras(tmp_path):
    assert "unexpected-push_levels-curriculum" in _guard(tmp_path, GOOD_LOG + "|    0    | push_levels    |\n")
    bad = GOOD_LOG + "[bhl_robust.tasks] m7-platecross id NOT registered: ImportError()\n"
    assert "platecross-id-not-registered" in _guard(tmp_path, bad)


def test_launcher_smoke_touches_no_scored_seed_and_runs_no_gate():
    sb = SBATCH.read_text()
    smoke = sb[sb.index("# ---------------------------------------------------------------- smoke: plumbing"):
               sb.index("# ---------------------------------------------------------------- gates")]
    assert "--protocol v2" not in smoke and "cpu_turn_qualify" not in smoke
    assert smoke.count("--seed 100") == 2 and "--seed0 100" in smoke
    assert 'if [ "$SMOKE" = 1 ]; then' in smoke and 'exit 0; else echo "SMOKE: FAIL' in smoke
    # the smoke's selection checks: the real stand-ins and the SYNTHETIC joint check both must select stand-in s2
    assert smoke.count('"PLATECROSS SELECTION: SELECTED $PREFIX-s2-smoke "*) ;;') == 2
    assert "smoke-standins --joint-check" in smoke and 'why="$why selection-joint-check"' in smoke
    assert '("FAIL", "QUALIFIED", False, 0)' in smoke
    gates = sb[sb.index("# ---------------------------------------------------------------- gates"):]
    assert "probe-scene" not in gates and "smoke-standins" not in gates
    probe = sb[sb.index("# ---------------------------------------------------------------- smoke: the scene probe"):
               sb.index("# ---------------------------------------------------------------- export")]
    assert probe.count('if [ "$SMOKE" = 1 ]; then') == 1 and "probe-scene" in probe and "PROBE_ENVS=$PROBE_NUM_ENVS" in probe


def test_launcher_smoke_time_limit_parser():
    sb = SBATCH.read_text()
    fn = re.search(r"^secs_of_limit\(\) \{.*?^\}$", sb, re.S | re.M).group(0)
    for limit, want in (("1:00:00", "3600"), ("59:00", "3540"), ("1-00:00:00", "86400"), ("12:00:00", "43200"),
                        ("UNLIMITED", ""), ("", "")):
        out = subprocess.run(["bash", "-c", f"{fn}\nsecs_of_limit \"$1\"", "x", limit], capture_output=True, text=True,
                             check=True).stdout.strip()
        assert out == want, (limit, out)


def test_launcher_bash_syntax():
    subprocess.run(["bash", "-n", str(SBATCH)], check=True)


FAKE_TURN_TEST = """\
import argparse, json
p = argparse.ArgumentParser()
p.add_argument("--out")
a, _ = p.parse_known_args()
json.dump({"protocol": "v2", "verdict": "PASS", "turns": [{"name": "turn+_s0", "yaw_deg": 190.0}], "n_turn_ok": 1,
           "n_turn": 1, "walk": {"yaw_deg": 1.5, "fell_at_s": None}}, open(a.out, "w"))
"""


def test_launcher_v2_test_writes_once_and_rereads_on_a_resubmit(tmp_path):
    sb = SBATCH.read_text()
    fn = re.search(r"^v2_test\(\) \{.*?^\}$", sb, re.S | re.M).group(0)
    fake = tmp_path / "repo"
    (fake / "scripts/bench").mkdir(parents=True)
    (fake / "scripts/bench/turn_test.py").write_text(FAKE_TURN_TEST)
    out = tmp_path / "v2"
    out.mkdir()
    script = (f"set -euo pipefail\nPYH={sys.executable}\nREPO={fake}\nOUT_DIR={out}\nJOBTAG=7\nUPSTREAM=/u\nCACHE=/c\n"
              f"{fn}\nv2_test /d/deploy.yaml arms-platecross-clocks2-s0\n")
    run = lambda: subprocess.run(["bash", "-c", script], capture_output=True, text=True, check=True).stdout  # noqa: E731
    first = run()
    assert "TURN-V2 arms-platecross-clocks2-s0: PASS" in first and "have" not in first
    assert sorted(p.name for p in out.iterdir()) == ["arms-platecross-clocks2-s0.json"]
    recorded = (out / "arms-platecross-clocks2-s0.json").read_text()
    (fake / "scripts/bench/turn_test.py").write_text(FAKE_TURN_TEST.replace('"PASS"', '"FAIL"'))
    second = run()                                                                          # a resubmit
    assert "have " in second and "TURN-V2 arms-platecross-clocks2-s0: PASS" in second
    assert sorted(p.name for p in out.iterdir()) == ["arms-platecross-clocks2-s0.json"]
    assert (out / "arms-platecross-clocks2-s0.json").read_text() == recorded


def _real_mode_startup(tmp_path, make_runs):
    """Run the launcher's real-mode (PLATECROSS_SMOKE=0) prefix, everything before the GPU section, on a fake tree.
    The smoke never runs the run-dir block, so this is its only test."""
    import os
    head = SBATCH.read_text().split("# ---------------------------------------------------------------- GPU from here")[0]
    env_line = "source /nfs/hpc/share/$USER/Humanoid_Lite/bhl-robustness-ladder/slurm/_env.sh"
    assert head.count(env_line) == 1
    head = head.replace(env_line, "true")
    up, repo = tmp_path / "up", tmp_path / "repo"
    logroot = up / "logs/rsl_rl/humanoid"
    parent = logroot / "2026-10-02_03-58-04_arms-turngait-clock-s2"
    (parent / "params").mkdir(parents=True)
    (parent / "exported").mkdir()
    (parent / "model_5999.pt").write_bytes(b"x")
    (parent / "params/env.yaml").write_text("x")
    (parent / "exported/deploy.yaml").write_text("x")
    repo.mkdir()
    make_runs(logroot)
    env = {"PATH": os.environ["PATH"], "UPSTREAM": str(up), "REPO": str(repo), "WORKSPACE": str(tmp_path),
           "PLATECROSS_SMOKE": "0", "SLURM_ARRAY_TASK_ID": "0", "SLURM_JOB_ID": "1"}
    return subprocess.run(["bash", "-c", head + '\necho "REACHED-GPU-SECTION n_runs=$n_runs"\n'], env=env,
                          capture_output=True, text=True, timeout=60)


def test_real_mode_startup_survives_a_missing_run_dir(tmp_path):
    """Regression, array 21517362 (2026-10-02). Every real task died in 4 s with exit 2 and an empty log: under
    set -euo pipefail, `ls | wc -l` failed when no run dir existed yet."""
    r = _real_mode_startup(tmp_path, lambda logroot: None)
    assert r.returncode == 0, r.stdout + r.stderr
    assert "REACHED-GPU-SECTION n_runs=0" in r.stdout


def test_real_mode_startup_still_refuses_incomplete_and_duplicate_run_dirs(tmp_path):
    def incomplete(logroot):
        (logroot / "2026-10-02_16-00-00_arms-platecross-clocks2-s0").mkdir()
    r = _real_mode_startup(tmp_path / "a", incomplete)
    assert r.returncode == 1 and "an incomplete run dir exists" in r.stdout

    def duplicate(logroot):
        for ts in ("2026-10-02_16-00-00", "2026-10-02_17-00-00"):
            (logroot / f"{ts}_arms-platecross-clocks2-s0").mkdir()
    r = _real_mode_startup(tmp_path / "b", duplicate)
    assert r.returncode == 1 and "2 run dirs named arms-platecross-clocks2-s0" in r.stdout

    def complete(logroot):
        d = logroot / "2026-10-02_16-00-00_arms-platecross-clocks2-s0"
        d.mkdir()
        (d / "model_8998.pt").write_bytes(b"x")
    r = _real_mode_startup(tmp_path / "c", complete)
    assert r.returncode == 0 and "REACHED-GPU-SECTION n_runs=1" in r.stdout, r.stdout + r.stderr
