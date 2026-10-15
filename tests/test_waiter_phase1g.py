"""Waiter phase 1g (2026-10-11): the standing terms as bounded positive rewards (phase 1f with the two penalties
replaced). The pure kernel, the task constants shared by the task and the deploy stamp, the stamp's stand-bonus
check, the launcher variant, and the reward arithmetic the design rests on. No simulator is started."""
import ast
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

STAND_MDP = REPO / "src/bhl_robust/tasks/waiter_stand_mdp.py"
ENV_CFG = REPO / "src/bhl_robust/tasks/waiter_env_cfg.py"
TASKS_INIT = REPO / "src/bhl_robust/tasks/__init__.py"
TRAIN = REPO / "slurm/repo20260923/gpu_waiter_wbc.sbatch"


def _load(path: Path, name: str):
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


S = _load(STAND_MDP, "waiter_stand_mdp_1g_under_test")


def test_bounded_bonus_is_positive_bounded_and_decreasing():
    err = np.array([0.0, 1.0, 4.0, 9.0, 1e6])
    b = S.bounded_bonus(err)
    assert b.tolist() == pytest.approx([1.0, 0.5, 0.2, 0.1, 1.0 / (1.0 + 1e6)])
    assert (b > 0).all() and (b <= 1).all() and (np.diff(b) < 0).all()
    # on the unclamped errors of the 1e measures: 1 with no error, 0.5 at 5 cm or 5 deg, still sloped far away
    inf = float("inf")
    d = np.array([[0.0, 0.0], [0.05, 0.0], [0.0, 0.0], [0.30, 0.0]])
    yaw = np.radians([0.0, 0.0, 5.0, 0.0])
    assert S.bounded_bonus(S.anchor_error(d, yaw, clamp=inf)).tolist() == pytest.approx([1.0, 0.5, 0.5, 1.0 / 37.0])
    g = np.array([[0.0, 0.0], [math.sin(math.radians(5.0)), 0.0], [math.sin(math.radians(10.0)), 0.0]])
    t = S.bounded_bonus(S.tilt_error(g, clamp=inf))
    assert t[:2].tolist() == pytest.approx([1.0, 0.5]) and 0.19 < t[2] < 0.21


def test_bounded_bonus_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    err = S.anchor_error(torch.tensor([[0.05, 0.0], [1.0, 1.0]]), torch.tensor([0.0, 3.0]), clamp=float("inf"))
    b = S.bounded_bonus(err)
    assert b[0].item() == pytest.approx(0.5) and 0.0 < b[1].item() < 0.01


def test_standing_can_only_add_reward():
    """The arithmetic behind the design: with 1e/1f's penalties a drifted standing env earns less per second than
    nothing at all (so ending the episode pays); with 1g's rewards every standing second is worth at least as much
    as in phase 1d, whatever the error."""
    c = _consts(ENV_CFG, ("STAND_ANCHOR_WEIGHT", "STAND_TILT_WEIGHT", "STAND_ANCHOR_BONUS_WEIGHT", "STAND_TILT_BONUS_WEIGHT"))
    best_other = 2.0 + 2.0 + 0.5                       # track_lin_vel + track_ang_vel + feet_gait, per second
    worst_penalty = c["STAND_ANCHOR_WEIGHT"] * 2 * S.ANCHOR_CLAMP + c["STAND_TILT_WEIGHT"] * S.TILT_CLAMP
    assert worst_penalty == -22.5 and best_other + worst_penalty < 0
    assert (c["STAND_ANCHOR_BONUS_WEIGHT"], c["STAND_TILT_BONUS_WEIGHT"]) == (2.0, 1.0)
    for err in (0.0, 1.0, 18.0, 1e9):
        gain = c["STAND_ANCHOR_BONUS_WEIGHT"] * S.bounded_bonus(err) + c["STAND_TILT_BONUS_WEIGHT"] * S.bounded_bonus(err)
        assert 0.0 < gain <= 3.0


def test_task_constants_agree():
    from bhl_robust.eval import waiter_wbc as WB
    c = _consts(ENV_CFG, ("STANDBONUS_TASK_ID", "STAND_ANCHOR_BONUS_WEIGHT", "STAND_TILT_BONUS_WEIGHT"))
    assert c["STANDBONUS_TASK_ID"] == WB.STANDBONUS_TASK_ID == "Velocity-BHL-Waiter-WBC-StandBonus-v0"
    assert WB.TASK_IDS[5] == WB.STANDBONUS_TASK_ID
    assert (c["STAND_ANCHOR_BONUS_WEIGHT"], c["STAND_TILT_BONUS_WEIGHT"]) == \
        (WB.STAND_ANCHOR_BONUS_WEIGHT, WB.STAND_TILT_BONUS_WEIGHT)
    init = TASKS_INIT.read_text()
    assert "waiter-wbc-standbonus" in init and "HumanoidWaiterWbcStandBonusCfg" in init
    cfg = ENV_CFG.read_text()
    rewards = cfg[cfg.index("class WaiterStandBonusRewardsCfg(TurnGaitRewardsCfg)"):
                  cfg.index("class HumanoidWaiterWbcStandBonusCfg")]
    assert "func=waiter_stand_mdp.StandAnchorBonus, weight=STAND_ANCHOR_BONUS_WEIGHT" in rewards
    assert "func=waiter_stand_mdp.stand_tilt_bonus, weight=STAND_TILT_BONUS_WEIGHT" in rewards
    assert "StandAnchor," not in rewards and "stand_tilt," not in rewards and "clamp" not in rewards   # no penalty term
    block = cfg[cfg.index("class HumanoidWaiterWbcStandBonusCfg(HumanoidWaiterWbcStanceCfg)"):]
    assert "rewards: WaiterStandBonusRewardsCfg = WaiterStandBonusRewardsCfg()" in block
    assert "self.events.push_robot.func = waiter_stand_mdp.push_and_reanchor" in block           # 1f's disturbances
    assert "moving_only" not in block
    mdp = STAND_MDP.read_text()
    bonus = mdp[mdp.index("class StandAnchorBonus(StandAnchor)"):]
    assert 'float("inf")' in bonus and "bounded_bonus(err) * standing_mask(env, command_name).float()" in bonus


def _good_env() -> dict:
    return {"commands": {"base_velocity": {"rel_standing_envs": 0.25}},
            "observations": {"policy": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}},
                             "critic": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}}},
            "rewards": {"feet_gait": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_gait_stance"},
                        "stand_anchor_bonus": {"func": "bhl_robust.tasks.waiter_stand_mdp:StandAnchorBonus", "weight": 2.0},
                        "stand_tilt_bonus": {"func": "bhl_robust.tasks.waiter_stand_mdp:stand_tilt_bonus", "weight": 1.0}},
            "events": {"push_robot": {"func": "bhl_robust.tasks.waiter_stand_mdp:push_and_reanchor"},
                       "hand_forces": {"func": "bhl_robust.tasks.waiter_wbc_mdp:hand_forces"}}}


def test_check_stand_bonus():
    from bhl_robust.eval.waiter_wbc import check_stand_bonus, check_stand_push
    assert check_stand_bonus(_good_env()) == []
    assert len(check_stand_push(_good_env())) == 4               # a 1g run is not a 1f run: no penalty terms
    one_f = _good_env()                                          # a 1f run is not a 1g run
    one_f["rewards"] = {"feet_gait": one_f["rewards"]["feet_gait"],
                        "stand_anchor": {"func": "bhl_robust.tasks.waiter_stand_mdp:StandAnchor", "weight": -1.0},
                        "stand_tilt": {"func": "bhl_robust.tasks.waiter_stand_mdp:stand_tilt", "weight": -0.5}}
    p = check_stand_bonus(one_f)
    assert len(p) == 6 and sum("penalty present" in x for x in p) == 2
    bad = _good_env()
    bad["rewards"]["stand_anchor_bonus"]["weight"] = -2.0        # a negative weight would be a penalty again
    bad["rewards"]["stand_tilt_bonus"]["func"] = "bhl_robust.tasks.waiter_stand_mdp:stand_tilt"
    bad["events"]["push_robot"]["func"] = "bhl_robust.tasks.waiter_stand_mdp:push_moving_only"
    bad["events"]["hand_forces"]["func"] = "bhl_robust.tasks.waiter_stand_mdp:hand_forces_moving_only"
    assert len(check_stand_bonus(bad)) == 4


def test_launcher_variant_1g():
    s = TRAIN.read_text()
    assert "1g) PREFIX=waiter-wbc1g; SEED0=18; RES_NAME=wbc1g; export BHL_STD_MAX=1.0" in s
    assert "TASK_ID=Velocity-BHL-Waiter-WBC-StandBonus-v0; FULL_ITERS=10000" in s
    assert "stand-bonus check" in s and "check_stand_bonus" in s and "standbonus-task-not-registered" in s
    assert s.count('[ "$VARIANT" = 1f ] || [ "$VARIANT" = 1g ]') == 3     # smoke ramp, std check, curriculum check
    assert s.count("std-clamp-after-update-missing") == 5       # 1f, 1g, 1h (2026-10-12), 1i (2026-10-13), 1l (10-15)
    assert '"${WAITER_STD_LOG_TOL:-1e-6}"' in s                # the std check is the predeclared one by default
