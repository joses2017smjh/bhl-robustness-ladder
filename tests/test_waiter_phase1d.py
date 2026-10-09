"""Waiter phase 1d (2026-10-08): the trained stance. The held phase, the constants shared by the task, the deploy
stamp, the env's stance hold and the Q3 gate, the launcher variant, and the Q3 clause of wbc_select.py (on the
committed phase 1c gate JSONs plus synthetic Q3 records). No simulator is started."""
import ast
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "waiter"))

MDP = REPO / "src/bhl_robust/tasks/gait_clock_mdp.py"
ENV_CFG = REPO / "src/bhl_robust/tasks/waiter_env_cfg.py"
TASKS_INIT = REPO / "src/bhl_robust/tasks/__init__.py"
TRAIN = REPO / "slurm/repo20260923/gpu_waiter_wbc.sbatch"
GATES = REPO / "slurm/repo20260923/cpu_waiter_gates.sbatch"
WBC1C = REPO / "results/waiter-20261005/wbc1c"


def _module_parts(path: Path, funcs=(), consts=()) -> dict:
    tree = ast.parse(path.read_text())
    keep = [n for n in tree.body if (isinstance(n, ast.FunctionDef) and n.name in funcs) or
            (isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name)
             and n.targets[0].id in consts)]
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns


M = _module_parts(MDP, funcs=("held_phase",), consts=("STANCE_PHASE", "GAIT_PERIOD_S", "GAIT_OFFSETS",
                                                       "STANCE_THRESHOLD"))
C = _module_parts(ENV_CFG, consts=("STANCE_TASK_ID", "STANCE_REL_STANDING_ENVS"))


def test_held_phase_holds_only_standing_envs():
    phase = np.array([0.1, 0.3, 0.7, 0.95])
    standing = np.array([True, False, True, False])
    out = M["held_phase"](phase, standing)
    assert np.allclose(out, [0.5, 0.3, 0.5, 0.95])


def test_stance_phase_is_double_support():
    """Phase 0.5 puts both feet in stance under the feet_gait schedule (stance iff (phase + offset) mod 1 < 0.55)."""
    ph = M["STANCE_PHASE"]
    assert all(((ph + o) % 1.0) < M["STANCE_THRESHOLD"] for o in M["GAIT_OFFSETS"])


def test_deploy_hold_and_gate_hold_the_same_phase():
    from bhl_robust.eval import waiter_env as E
    import stance_gate as G3
    period_steps = round(M["GAIT_PERIOD_S"] / 0.04)
    assert E.STANCE_PHASE_STEPS == (G3.HOLD_STEP,)
    assert G3.HOLD_STEP / period_steps == pytest.approx(M["STANCE_PHASE"])


def test_task_constants_agree():
    from bhl_robust.eval import waiter_wbc as WB
    assert C["STANCE_TASK_ID"] == WB.STANCE_TASK_ID and WB.STANCE_TASK_ID in WB.TASK_IDS
    assert C["STANCE_REL_STANDING_ENVS"] == WB.STANCE_REL_STANDING_ENVS == 0.25
    assert "waiter-wbc-stance" in TASKS_INIT.read_text() and "HumanoidWaiterWbcStanceCfg" in TASKS_INIT.read_text()


def test_check_stance():
    from bhl_robust.eval.waiter_wbc import check_stance
    good = {"commands": {"base_velocity": {"rel_standing_envs": 0.25}},
            "observations": {"policy": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}},
                             "critic": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}}},
            "rewards": {"feet_gait": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_gait_stance"}}}
    assert check_stance(good) == []
    bad = json.loads(json.dumps(good))
    bad["commands"]["base_velocity"]["rel_standing_envs"] = 0.02
    bad["observations"]["critic"]["gait_clock"]["func"] = "bhl_robust.tasks.gait_clock_mdp:gait_clock"
    bad["rewards"]["feet_gait"]["func"] = "bhl_robust.tasks.gait_clock_mdp:feet_gait"
    assert len(check_stance(bad)) == 3


def test_launcher_variant_1d():
    s = TRAIN.read_text()
    assert "1d) PREFIX=waiter-wbc1d; SEED0=9; RES_NAME=wbc1d; export BHL_STD_MAX=1.0" in s
    assert "TASK_ID=Velocity-BHL-Waiter-WBC-Stance-v0; FULL_ITERS=10000" in s
    assert "stance-check" in s and 'curriculum-check' in s
    g = GATES.read_text()
    assert "stance_gate.py" in g and "WAITER_GATES_STANCE" in g


def test_selection_needs_q3_when_asked(tmp_path):
    import wbc_select
    res = tmp_path / "wbc1d"
    shutil.copytree(WBC1C, res, ignore=shutil.ignore_patterns("*.log", "*.log.gz", "selection.json"))
    (res / "q3").mkdir()
    # rename the committed 1c records to a 1d prefix so the selection reads them
    for p in list(res.rglob("waiter-wbc1c-s*")):
        p.rename(p.with_name(p.name.replace("waiter-wbc1c", "waiter-wbc1d")))
    for s, verdict in ((6, "FAIL"), (7, "FAIL"), (8, "PASS")):
        (res / "q3" / f"waiter-wbc1d-s{s}.json").write_text(json.dumps({"verdict": verdict, "falls": 0, "n_ok": 10}))
    assert wbc_select.main(["--res", str(res), "--prefix", "waiter-wbc1d", "--seeds", "6", "7", "8",
                            "--min-walk-m", "1.5", "--stance"]) == 0
    sel = json.loads((res / "selection.json").read_text())
    # 1c: s7 and s8 qualified on Q1+Q2+walk; with Q3 only s8 passes
    assert sel["selected"] == "s8" and "Q3 stance PASS" in sel["rule"]
