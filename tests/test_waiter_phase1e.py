"""Waiter phase 1e (2026-10-10): standing still under arm loads. The pure standing-penalty helpers, the constants
shared by the task and the deploy stamp, the stamp's stand check, the launcher variant, the Q4 gate's frozen
trajectories and the Q4 clause of wbc_select.py (on the committed phase 1c gate JSONs plus synthetic Q3/Q4 records).
No simulator is started."""
import ast
import importlib.util
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts" / "waiter"))
sys.path.insert(0, str(REPO / "scripts" / "bench"))

STAND_MDP = REPO / "src/bhl_robust/tasks/waiter_stand_mdp.py"
ENV_CFG = REPO / "src/bhl_robust/tasks/waiter_env_cfg.py"
TASKS_INIT = REPO / "src/bhl_robust/tasks/__init__.py"
TRAIN = REPO / "slurm/repo20260923/gpu_waiter_wbc.sbatch"
GATES = REPO / "slurm/repo20260923/cpu_waiter_gates.sbatch"
WBC1C = REPO / "results/waiter-20261005/wbc1c"


def _load(path: Path, name: str):
    """Import one file as a module without importing its package (bhl_robust.tasks registers Isaac Lab tasks)."""
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _consts(path: Path, names) -> dict:
    tree = ast.parse(path.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns


S = _load(STAND_MDP, "waiter_stand_mdp_under_test")


def test_anchor_error_is_normalised_and_clamped():
    dxy = np.array([[0.0, 0.0], [0.05, 0.0], [0.0, 0.10], [1.0, 1.0]])
    dyaw = np.radians([0.0, 0.0, 5.0, 90.0])
    e = S.anchor_error(dxy, dyaw)
    assert np.allclose(e, [0.0, 1.0, 4.0 + 1.0, S.ANCHOR_CLAMP + S.ANCHOR_CLAMP])
    assert (S.ANCHOR_POS_M, S.ANCHOR_YAW_DEG, S.ANCHOR_CLAMP) == (0.05, 5.0, 9.0)


def test_tilt_error_is_one_at_five_degrees():
    for deg, want in ((0.0, 0.0), (5.0, 1.0), (10.0, (math.sin(math.radians(10)) / math.sin(math.radians(5))) ** 2)):
        g = np.array([[math.sin(math.radians(deg)), 0.0]])
        assert S.tilt_error(g)[0] == pytest.approx(want, abs=1e-9)
    assert S.tilt_error(np.array([[1.0, 0.0]]))[0] == S.TILT_CLAMP           # 90 deg: clamped
    # the upstream posture term charges a 10 deg lean |g_xy|^2 = 0.03; this one ~4 before its weight
    assert S.tilt_error(np.array([[math.sin(math.radians(10)), 0.0]]))[0] > 3.9


def test_helpers_accept_torch_tensors():
    torch = pytest.importorskip("torch")
    e = S.anchor_error(torch.tensor([[0.05, 0.0], [1.0, 0.0]]), torch.tensor([0.0, 0.0]))
    assert torch.allclose(e, torch.tensor([1.0, S.ANCHOR_CLAMP]))
    assert torch.allclose(S.tilt_error(torch.tensor([[0.0, 0.0]])), torch.tensor([0.0]))
    ids = S.moving_ids(torch.tensor([0, 2, 3]), torch.tensor([True, False, False, True]))
    assert ids.tolist() == [2]


def test_moving_ids_and_wrap():
    assert S.moving_ids(np.array([0, 1, 2, 3]), np.array([True, False, True, False])).tolist() == [1, 3]
    assert S.wrap_to_pi(math.pi + 0.1) == pytest.approx(-math.pi + 0.1)
    assert S.wrap_to_pi(-0.2) == pytest.approx(-0.2)


def test_task_constants_agree():
    from bhl_robust.eval import waiter_wbc as WB
    c = _consts(ENV_CFG, ("STAND_TASK_ID", "STAND_ANCHOR_WEIGHT", "STAND_TILT_WEIGHT"))
    assert c["STAND_TASK_ID"] == WB.STAND_TASK_ID and WB.STAND_TASK_ID in WB.TASK_IDS
    assert (c["STAND_ANCHOR_WEIGHT"], c["STAND_TILT_WEIGHT"]) == (WB.STAND_ANCHOR_WEIGHT, WB.STAND_TILT_WEIGHT) == (-1.0, -0.5)
    init = TASKS_INIT.read_text()
    assert "waiter-wbc-stand" in init and "HumanoidWaiterWbcStandCfg" in init
    cfg = ENV_CFG.read_text()
    for needle in ("class HumanoidWaiterWbcStandCfg(HumanoidWaiterWbcStanceCfg)", "push_moving_only",
                   "hand_forces_moving_only", "waiter_stand_mdp.StandAnchor", "waiter_stand_mdp.stand_tilt"):
        assert needle in cfg


def _good_env() -> dict:
    return {"commands": {"base_velocity": {"rel_standing_envs": 0.25}},
            "observations": {"policy": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}},
                             "critic": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}}},
            "rewards": {"feet_gait": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_gait_stance"},
                        "stand_anchor": {"func": "bhl_robust.tasks.waiter_stand_mdp:StandAnchor", "weight": -1.0},
                        "stand_tilt": {"func": "bhl_robust.tasks.waiter_stand_mdp:stand_tilt", "weight": -0.5}},
            "events": {"push_robot": {"func": "bhl_robust.tasks.waiter_stand_mdp:push_moving_only"},
                       "hand_forces": {"func": "bhl_robust.tasks.waiter_stand_mdp:hand_forces_moving_only"}}}


def test_check_stand():
    from bhl_robust.eval.waiter_wbc import check_stand
    assert check_stand(_good_env()) == []
    bad = _good_env()
    bad["events"]["push_robot"]["func"] = "isaaclab.envs.mdp.events:push_by_setting_velocity"
    bad["rewards"]["stand_tilt"]["weight"] = -1.0
    del bad["rewards"]["stand_anchor"]
    bad["commands"]["base_velocity"]["rel_standing_envs"] = 0.02
    p = check_stand(bad)
    assert len(p) == 5 and any("push_robot" in x for x in p) and any("rel_standing_envs" in x for x in p)


def test_launcher_variant_1e():
    s = TRAIN.read_text()
    assert "1e) PREFIX=waiter-wbc1e; SEED0=12; RES_NAME=wbc1e; export BHL_STD_MAX=1.0" in s
    assert "TASK_ID=Velocity-BHL-Waiter-WBC-Stand-v0; FULL_ITERS=10000" in s
    assert "stand-check" in s and "check_stand" in s and "waiter_stand_mdp.py" in s
    g = GATES.read_text()
    assert "stand_manip_gate.py" in g and "WAITER_GATES_STANDMANIP" in g and "--stand-manip" in g


def test_q4_gate_is_frozen_and_within_limits():
    import q2_trajectories as T
    import stand_manip_gate as G4
    assert (G4.MAX_DISP_M, G4.MAX_YAW_DEG, G4.MAX_PITCH_DEG, G4.MIN_OK, G4.T0_S, G4.SEEDS) == (0.03, 3.0, 3.0, 10, 1.0, (0, 1, 2))
    assert G4.HOLD_STEP == 10 and list(G4.TRAJECTORIES) == ["P1_tumbler_left", "P2_tumbler_right", "P3_tray_hook",
                                                           "P4_pick_then_hook"]
    lim = np.r_[T.LEFT_LIMITS, T.RIGHT_LIMITS]
    for name, (traj, dur) in G4.TRAJECTORIES.items():
        assert dur == (16.0 if name.startswith("P4") else 12.0)
        for t in np.arange(0.0, dur, 0.04):
            q = traj(float(t))
            assert q.shape == (12,) and q.dtype == np.float32
            assert np.all(q[:10] >= lim[:, 0] - 1e-6) and np.all(q[:10] <= lim[:, 1] + 1e-6), (name, t)
            assert np.all(q[10:] == 0.0)                           # fingers open: the expert hooks without closing
    assert np.allclose(G4.p1(0.0), 0.0) and np.allclose(G4.p3(20.0)[:5], G4.POSE["hook_lift"], atol=1e-6)
    assert np.allclose(G4.p3(20.0)[5:10], -np.asarray(G4.POSE["hook_lift"]), atol=1e-6)          # right = -left
    assert np.allclose(G4.p2(8.0)[5:10], -G4.p1(8.0)[:5]) and np.allclose(G4.p2(8.0)[:5], 0.0)
    assert G4.pitch_of([math.cos(0.05), 0.0, math.sin(0.05), 0.0]) == pytest.approx(0.1)


def test_selection_needs_q4_when_asked(tmp_path):
    import wbc_select
    res = tmp_path / "wbc1e"
    shutil.copytree(WBC1C, res, ignore=shutil.ignore_patterns("*.log", "*.log.gz", "selection.json"))
    (res / "q3").mkdir()
    (res / "q4").mkdir()
    for p in list(res.rglob("waiter-wbc1c-s*")):
        p.rename(p.with_name(p.name.replace("waiter-wbc1c", "waiter-wbc1e")))
    for s, v3, v4 in ((6, "PASS", "PASS"), (7, "PASS", "FAIL"), (8, "PASS", "PASS")):
        (res / "q3" / f"waiter-wbc1e-s{s}.json").write_text(json.dumps({"verdict": v3, "falls": 0, "n_ok": 12}))
        (res / "q4" / f"waiter-wbc1e-s{s}.json").write_text(json.dumps({"verdict": v4, "falls": 0, "n_ok": 11}))
    assert wbc_select.main(["--res", str(res), "--prefix", "waiter-wbc1e", "--seeds", "6", "7", "8",
                            "--min-walk-m", "1.5", "--stance", "--stand-manip"]) == 0
    sel = json.loads((res / "selection.json").read_text())
    # 1c: s7 and s8 qualified on Q1 + Q2 + walk; s7 fails the synthetic Q4, so s8 is selected
    assert sel["selected"] == "s8" and "Q4 standing manipulation PASS" in sel["rule"]
    assert sel["per_seed"]["s7"]["q4"] == "FAIL" and not sel["per_seed"]["s7"]["qualified"]
