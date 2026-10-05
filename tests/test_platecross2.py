"""F1 "PlateCross v2" (2026-10-04): half of the tiles flat; v1 (PlateCross) unchanged. Isaac-free."""
import ast
import importlib.util
import os
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
SELECT = REPO / "scripts/bench/platecross_select.py"
TERRAIN = REPO / "src/bhl_robust/tasks/platecross_terrain.py"
ENV_CFG = REPO / "src/bhl_robust/tasks/platecross_env_cfg.py"
L1 = REPO / "slurm/repo20260923/gpu_platecross.sbatch"
L2 = REPO / "slurm/repo20260923/gpu_platecross2.sbatch"


def _load(path, name, variant=None, monkeypatch=None):
    if variant is None:
        monkeypatch.delenv("PLATECROSS_VARIANT", raising=False)
    else:
        monkeypatch.setenv("PLATECROSS_VARIANT", variant)
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def pt():
    spec = importlib.util.spec_from_file_location("pt_for_v2_tests", TERRAIN)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_curriculum_layout_puts_plates_in_columns_0_to_4(pt):
    assert pt.V2_SUB_TERRAINS == ("plates", "flat") and pt.V2_PROPORTIONS == (0.5, 0.5)
    assert pt.v2_plate_columns(10) == [0, 1, 2, 3, 4]
    # Isaac Lab's rule, restated: column c takes min{i : c / n + 0.001 < cumsum(p)[i]}
    assert pt.v2_plate_columns(10, (1.0,)) == list(range(10))


def test_v2_field_is_the_v1_tile_in_half_of_the_tiles(pt):
    v1 = pt.expected_world_plates()
    v2 = pt.expected_world_plates_v2()
    assert len(v1) == 100 * 48 and len(v2) == 50 * 48
    assert set(v2) <= set(v1)
    centres = pt.tile_centres()
    plate_tiles = {k for k, (cx, cy) in enumerate(centres)
                   if any(abs(x - cx) <= 3.61 and abs(y - cy) <= 3.61 for _, x, y in v2)}
    assert plate_tiles == {k for k in range(100) if k % 10 < 5}
    # columns 0-4 are the low-y half of the field (tile_centres: y = (c + 0.5) * 8.4 - 42)
    assert all(y < 0 for _, _, y in v2)


def _env(curriculum, subs):
    return {"scene": {"terrain": {"terrain_type": "generator", "max_init_terrain_level": None, "visual_material": None,
                                  "terrain_generator": {"class_type": "isaaclab.terrains.terrain_generator:TerrainGenerator",
                                                        "border_width": 20.0, "num_rows": 10, "num_cols": 10, "seed": 0,
                                                        "curriculum": curriculum, "use_cache": False,
                                                        "size": [8.4, 8.4], "sub_terrains": subs}}}}


def _plates(proportion):
    return {"function": "bhl_robust.tasks.platecross_terrain:plates_terrain", "proportion": proportion,
            "round_radius_m": 0.24, "square_side_m": 0.48, "height_m": 0.03, "pitch_m": 1.2, "disc_sections": 64,
            "clear_centre": True, "flat_patch_sampling": None}


FLAT = {"function": "isaaclab.terrains.trimesh.mesh_terrains:flat_terrain", "proportion": 0.5}


def test_v2_terrain_check_accepts_the_declared_field_and_rejects_others(monkeypatch):
    s2 = _load(SELECT, "pc_select_v2", "v2", monkeypatch)
    assert s2.TASK_ID == "Velocity-BHL-Arms-PlateCross2-v0" and s2.PREFIX == "arms-platecross2-clocks2"
    assert "half plates, half flat ground (F1)" in s2.FROZEN_LABELS
    assert s2.terrain_problems(_env(True, {"plates": _plates(0.5), "flat": FLAT})) == []
    assert s2.terrain_problems(_env(True, {"flat": FLAT, "plates": _plates(0.5)})) == []      # yaml order is not read
    assert any("curriculum" in p for p in s2.terrain_problems(_env(False, {"plates": _plates(0.5), "flat": FLAT})))
    assert any("plates.proportion" in p for p in s2.terrain_problems(_env(True, {"plates": _plates(1.0), "flat": FLAT})))
    assert any("flat.proportion" in p
               for p in s2.terrain_problems(_env(True, {"plates": _plates(0.5), "flat": {**FLAT, "proportion": 0.3}})))
    assert any("sub_terrains" in p for p in s2.terrain_problems(_env(True, {"plates": _plates(0.5)})))
    assert any("flat function" in p
               for p in s2.terrain_problems(_env(True, {"plates": _plates(0.5), "flat": {**FLAT, "function": "x:y"}})))


def test_v1_is_unchanged(monkeypatch):
    s1 = _load(SELECT, "pc_select_v1", None, monkeypatch)
    assert s1.VARIANT == "v1" and s1.TASK_ID == "Velocity-BHL-Arms-PlateCross-v0" and s1.PREFIX == "arms-platecross-clocks2"
    assert s1.FROZEN_LABELS == ("Labels: LEARNED gait (fine-tuned from clock-s2 on plates); MuJoCo gates; bench v2's "
                                "180-deg timing caveat applies.")
    assert s1.terrain_problems(_env(False, {"plates": _plates(1.0)})) == []
    assert any("curriculum" in p for p in s1.terrain_problems(_env(True, {"plates": _plates(1.0)})))
    assert any("sub_terrains" in p for p in s1.terrain_problems(_env(False, {"plates": _plates(0.5), "flat": FLAT})))


def test_unknown_variant_is_refused(monkeypatch):
    with pytest.raises(SystemExit):
        _load(SELECT, "pc_select_bad", "v9", monkeypatch)


def test_env_cfg_declares_the_v2_generator_and_task():
    src = ENV_CFG.read_text()
    tree = ast.parse(src)
    names = {n.targets[0].id for n in tree.body if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name)}
    assert {"PLATES_TERRAINS_CFG", "PLATES2_TERRAINS_CFG", "TASK_ID_V2"} <= names
    v2 = src[src.index("PLATES2_TERRAINS_CFG = TerrainGeneratorCfg("):]
    v2 = v2[:v2.index("\n)\n")]
    assert "curriculum=True" in v2 and "use_cache=False" in v2
    assert v2.index('"plates": PlatesTerrainCfg(proportion=_pt.V2_PROPORTIONS[0])') < v2.index(
        '"flat": MeshPlaneTerrainCfg(proportion=_pt.V2_PROPORTIONS[1])')
    v1 = src[src.index("PLATES_TERRAINS_CFG = TerrainGeneratorCfg("):src.index("@configclass\nclass HumanoidPlateCrossCfg")]
    assert "curriculum=False" in v1 and 'sub_terrains={"plates": PlatesTerrainCfg(proportion=1.0)}' in v1
    cls = [n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "HumanoidPlateCross2Cfg"]
    assert cls and [b.id for b in cls[0].bases] == ["HumanoidPlateCrossCfg"]
    assert 'TASK_ID_V2 = "Velocity-BHL-Arms-PlateCross2-v0"' in src
    init = (REPO / "src/bhl_robust/tasks/__init__.py").read_text()
    assert "_platecross2.HumanoidPlateCross2Cfg" in init and init.rstrip().endswith("# --- end m7-platecross2 ---")


def test_v2_launcher_differs_from_v1_only_where_declared():
    a, b = L1.read_text().splitlines(), L2.read_text().splitlines()
    body1, body2 = a[a.index("set -euo pipefail"):], b[b.index("set -euo pipefail"):]
    changed = [(x, y) for x, y in zip(body1, body2[:2] + body2[3:]) if x != y]
    assert body2[1] == "export BHL_STACK=v51" and body2[2].startswith("export PLATECROSS_VARIANT=v2")
    assert len(body2) == len(body1) + 1
    allowed = ("TASK_ID=", "PREFIX=", "SMOKE_RES=", "    RES=$REPO/results/", 'CACHE="${TMPDIR', 'TLOG="${TMPDIR',
               '    grep -q')
    assert changed and all(y.lstrip().startswith(tuple(s.strip() for s in allowed)) for _, y in changed), changed
    text = L2.read_text()
    for s in ("TASK_ID=Velocity-BHL-Arms-PlateCross2-v0", "PREFIX=arms-platecross2-clocks2",
              "RES=$REPO/results/repo-gpu-20260923/platecross2-20261004", "#SBATCH --job-name=platecross2",
              '2026-10-04: F1 "PlateCross v2"', "half of the terrain tiles are flat", "21544374",
              "71/76", "63/76", 'grep -qE "m7-platecross2? id NOT registered"'):
        assert s in text, s
    assert "platecross-20261002" not in text and "arms-platecross-clocks2-s" not in text
    # the predeclared rule is v1's, verbatim
    assert "PREDECLARED RULE (frozen 2026-10-02, before any PlateCross run exists; the task's text verbatim):" in text
