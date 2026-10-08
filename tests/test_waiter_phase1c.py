"""Waiter phase 1c (2026-10-07): the disturbance curriculum's schedule, the task constants and launcher checks agree,
the curriculum check in the deploy stamp, and the predeclared walk-progress clause of scripts/waiter/wbc_select.py,
read on the committed phase 1b and clock-s2 gate JSONs. No simulator is started: the Isaac-dependent modules are read
as source (their pure functions are executed alone)."""
import ast
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "waiter"))

MDP = REPO / "src/bhl_robust/tasks/waiter_wbc_mdp.py"
ENV_CFG = REPO / "src/bhl_robust/tasks/waiter_env_cfg.py"
TASKS_INIT = REPO / "src/bhl_robust/tasks/__init__.py"
TRAIN = REPO / "slurm/repo20260923/gpu_waiter_wbc.sbatch"
GATES = REPO / "slurm/repo20260923/cpu_waiter_gates.sbatch"
WBC1B = REPO / "results/waiter-20261005/wbc1b"
CLOCK = REPO / "results/repo-gpu-20260923/turngait-r12-20261001"


def _pure_functions(path: Path, names) -> dict:
    """Execute only the named top-level functions of a module (they must not need its imports)."""
    tree = ast.parse(path.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert {n.name for n in keep} == set(names)
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns


def _constants(path: Path) -> dict:
    out = {}
    for n in ast.parse(path.read_text()).body:
        if isinstance(n, ast.Assign) and len(n.targets) == 1 and isinstance(n.targets[0], ast.Name):
            try:
                out[n.targets[0].id] = ast.literal_eval(n.value)
            except ValueError:
                pass
    return out


F = _pure_functions(MDP, ("disturbance_scale", "scaled_range"))
C = _constants(ENV_CFG)


def test_schedule_is_zero_then_linear_then_one():
    s = F["disturbance_scale"]
    assert s(0, 4000, 7000) == 0.0 and s(4000, 4000, 7000) == 0.0
    assert s(5500, 4000, 7000) == pytest.approx(0.5) and s(4300, 4000, 7000) == pytest.approx(0.1)
    assert s(7000, 4000, 7000) == 1.0 and s(9999, 4000, 7000) == 1.0
    with pytest.raises(ValueError):
        s(10, 5, 5)
    assert F["scaled_range"]((-6.0, 6.0), 0.25) == (-1.5, 1.5) and F["scaled_range"]((0.0, 0.8), 0.0) == (0.0, 0.0)


def test_task_constants_match_the_frozen_design_and_the_launcher_check():
    assert (C["CURRICULUM_START_ITER"], C["CURRICULUM_END_ITER"], C["CURRICULUM_ITERS"], C["STEPS_PER_ITER"]) == (4000, 7000, 10000, 24)
    assert C["HAND_FORCE_N"] == (-6.0, 6.0) and C["HAND_PAYLOAD_KG"] == (0.0, 0.8)     # phase 1's ranges, unchanged
    assert C["CURRICULUM_TASK_ID"] == "Velocity-BHL-Waiter-WBC-Curriculum-v0" and C["TASK_ID"] == "Velocity-BHL-Waiter-WBC-v0"
    body = TRAIN.read_text()
    assert "(4000, 7000, 24)" in body and "[-6.0, 6.0]" in body and "[0.0, 0.8]" in body
    assert "1c) PREFIX=waiter-wbc1c; SEED0=6; RES_NAME=wbc1c; export BHL_STD_MAX=1.0" in body
    assert "TASK_ID=Velocity-BHL-Waiter-WBC-Curriculum-v0; FULL_ITERS=10000" in body
    assert "env.curriculum.disturbance.params.start_iter=0 env.curriculum.disturbance.params.end_iter=2" in body
    assert '[ "$VARIANT" = 1c ] && [ "$SMOKE" = 1 ]' in body            # the compressed ramp is smoke-only
    assert '--task "$TASK_ID"' in body
    init = TASKS_INIT.read_text()
    assert "_waiter_c.CURRICULUM_TASK_ID" in init and "_waiter_c.HumanoidWaiterWbcCurriculumCfg" in init


def test_phase1_task_is_untouched_by_phase1c():
    src = ENV_CFG.read_text()
    head = src.split("# ---- phase 1c")[0]
    assert "CurrTerm(" not in head and 'mode="startup"' in head       # phase 1's payload stays a startup event
    mdp = MDP.read_text()
    assert "arm_scale: float = 1.0" in mdp and "if self.cfg.arm_scale != 1.0:" in mdp


def test_stamp_checks_the_curriculum_and_the_task_id(tmp_path):
    from bhl_robust.eval import waiter_wbc as ww
    ok = {"curriculum": {"disturbance": {"func": "bhl_robust.tasks.waiter_wbc_mdp:disturbance_curriculum"}},
          "events": {"hand_payload": {"mode": "reset"}}}
    assert ww.check_curriculum(ok) == []
    assert ww.check_curriculum({"events": {"hand_payload": {"mode": "startup"}}}) == [
        "curriculum term disturbance missing (None)", "hand_payload is not a reset event"]
    assert ww.TASK_IDS == ("Velocity-BHL-Waiter-WBC-v0", "Velocity-BHL-Waiter-WBC-Curriculum-v0")
    with pytest.raises(ValueError):
        ww.stamp(tmp_path / "deploy.yaml", tmp_path / "env.yaml", task="Velocity-BHL-Waiter-WBC-v9")


def test_walk_progress_fails_every_phase1b_seed_and_passes_clock_s2():
    import wbc_select as ws
    for s in (3, 4, 5):
        run = f"waiter-wbc1b-s{s}"
        v2 = json.loads((WBC1B / "turn-test-v2" / f"{run}.json").read_text())
        v2x = json.loads((WBC1B / "qualify" / f"{run}__v2x.json").read_text())
        wp = ws.walk_progress(v2, v2x, 1.5)
        assert wp["protocol_ok"] and wp["n_walks"] == 4 and not wp["pass"] and max(wp["displacement_m"]) < 0.5
    v2 = json.loads((CLOCK / "turn-test-v2" / "arms-turngait-clock-s2.json").read_text())
    v2x = json.loads((CLOCK / "qualify" / "arms-turngait-clock-s2__v2x.json").read_text())
    wp = ws.walk_progress(v2, v2x, 1.5)
    assert wp["pass"] and min(wp["displacement_m"]) >= 1.85


def test_selection_without_the_flag_reproduces_the_recorded_1b_selection(tmp_path):
    import wbc_select as ws
    res = tmp_path / "wbc1b"
    shutil.copytree(WBC1B, res, ignore=shutil.ignore_patterns("selection.json", "training"))
    assert ws.main(["--res", str(res), "--prefix", "waiter-wbc1b", "--seeds", "3", "4", "5"]) == 0
    new, old = (json.loads(p.read_text()) for p in (res / "selection.json", WBC1B / "selection.json"))
    assert new["line"] == old["line"] and new["verdict"] == old["verdict"] == "NEGATIVE" and new["rule"] == old["rule"]


def test_selection_with_the_flag_records_the_progress_clause(tmp_path):
    import wbc_select as ws
    res = tmp_path / "wbc1b"
    shutil.copytree(WBC1B, res, ignore=shutil.ignore_patterns("selection.json", "training"))
    # make s4 pass every phase 1b clause, so only the progress clause can stop it
    v2 = json.loads((res / "turn-test-v2" / "waiter-wbc1b-s4.json").read_text())
    v2["verdict"] = "PASS"
    (res / "turn-test-v2" / "waiter-wbc1b-s4.json").write_text(json.dumps(v2))
    assert ws.main(["--res", str(res), "--prefix", "waiter-wbc1b", "--seeds", "3", "4", "5", "--min-walk-m", "1.5"]) == 0
    rec = json.loads((res / "selection.json").read_text())
    assert rec["verdict"] == "NEGATIVE" and "walk progress" in rec["rule"]
    assert rec["per_seed"]["s4"]["walk_progress"]["pass"] is False and rec["per_seed"]["s4"]["qualified"] is False
    assert "walk 0." in rec["line"]


def test_gates_launcher_passes_the_flag_only_when_set():
    body = GATES.read_text()
    assert '[ -n "${WAITER_GATES_MIN_WALK_M:-}" ] && PROGRESS=(--min-walk-m "$WAITER_GATES_MIN_WALK_M")' in body
    assert '"${PROGRESS[@]}"' in body
