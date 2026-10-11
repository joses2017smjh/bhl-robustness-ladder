"""Waiter phase 1f (2026-10-11): standing envs keep their pushes; a new anchor after each push. The pure re-anchor
helpers, the task constants shared by the task and the deploy stamp, the stamp's stand-push check, the launcher
variant and the training overlay that makes the logged std the bounded one. No simulator is started."""
import ast
import importlib.util
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


S = _load(STAND_MDP, "waiter_stand_mdp_1f_under_test")


def test_reanchor_mask():
    due = np.array([-1, 5, 5, 9, 5])
    standing = np.array([True, True, False, True, True])
    assert S.reanchor_mask(due, 5, standing).tolist() == [False, True, False, False, True]
    assert S.reanchor_mask(due, 4, standing).tolist() == [False] * 5
    assert (S.REANCHOR_DELAY_S, S.REANCHOR_ATTR) == (1.0, "_waiter_stand_reanchor_step")


def test_new_anchor_mask_starts_and_push_reanchors():
    standing = np.array([True, True, True, False, True])
    prev = np.array([False, True, True, True, True])
    assert S.new_anchor_mask(standing, prev).tolist() == [True, False, False, False, False]       # 1e: starts only
    due = np.array([-1, 10, 12, 10, -1])
    new = S.new_anchor_mask(standing, prev, due, now=10)
    # env 0 just started standing; env 1 was pushed 1 s ago; env 2 is not due yet; env 3 is moving (stays pending)
    assert new.tolist() == [True, True, False, False, False]
    assert due.tolist() == [-1, -1, 12, 10, -1]                                                   # fired entry cleared
    assert S.new_anchor_mask(standing, standing.copy(), due, now=11).tolist() == [False] * 5     # fires once
    assert S.new_anchor_mask(standing, standing.copy(), due, now=12).tolist() == [False, False, True, False, False]


def test_helpers_accept_torch_tensors():
    torch = pytest.importorskip("torch")
    due = torch.tensor([3, -1, 7])
    standing = torch.tensor([True, True, True])
    new = S.new_anchor_mask(standing, torch.tensor([True, True, True]), due, now=3)
    assert new.tolist() == [True, False, False] and due.tolist() == [-1, -1, 7]


def test_task_constants_agree():
    from bhl_robust.eval import waiter_wbc as WB
    c = _consts(ENV_CFG, ("STANDPUSH_TASK_ID",))
    assert c["STANDPUSH_TASK_ID"] == WB.STANDPUSH_TASK_ID == "Velocity-BHL-Waiter-WBC-StandPush-v0"
    assert WB.TASK_IDS[4] == WB.STANDPUSH_TASK_ID and len(WB.TASK_IDS) >= 5      # later phases append (1g is [5])
    init = TASKS_INIT.read_text()
    assert "waiter-wbc-standpush" in init and "HumanoidWaiterWbcStandPushCfg" in init
    cfg = ENV_CFG.read_text()
    block = cfg[cfg.index("class HumanoidWaiterWbcStandPushCfg(HumanoidWaiterWbcStanceCfg)"):]
    assert "rewards: WaiterStandRewardsCfg = WaiterStandRewardsCfg()" in block
    assert "self.events.push_robot.func = waiter_stand_mdp.push_and_reanchor" in block
    assert "moving_only" not in block                           # 1f does not take the disturbances off standing envs


def _good_env() -> dict:
    return {"commands": {"base_velocity": {"rel_standing_envs": 0.25}},
            "observations": {"policy": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}},
                             "critic": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}}},
            "rewards": {"feet_gait": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_gait_stance"},
                        "stand_anchor": {"func": "bhl_robust.tasks.waiter_stand_mdp:StandAnchor", "weight": -1.0},
                        "stand_tilt": {"func": "bhl_robust.tasks.waiter_stand_mdp:stand_tilt", "weight": -0.5}},
            "events": {"push_robot": {"func": "bhl_robust.tasks.waiter_stand_mdp:push_and_reanchor"},
                       "hand_forces": {"func": "bhl_robust.tasks.waiter_wbc_mdp:hand_forces"}}}


def test_check_stand_push():
    from bhl_robust.eval.waiter_wbc import check_stand, check_stand_push
    assert check_stand_push(_good_env()) == []
    assert len(check_stand(_good_env())) == 2                   # a 1f run is not a 1e run (and the reverse, below)
    one_e = _good_env()
    one_e["events"]["push_robot"]["func"] = "bhl_robust.tasks.waiter_stand_mdp:push_moving_only"
    one_e["events"]["hand_forces"]["func"] = "bhl_robust.tasks.waiter_stand_mdp:hand_forces_moving_only"
    p = check_stand_push(one_e)
    assert len(p) == 2 and any("push_robot" in x for x in p) and any("hand_forces" in x for x in p)
    bad = _good_env()
    bad["events"]["push_robot"]["func"] = "isaaclab.envs.mdp.events:push_by_setting_velocity"   # no re-anchor
    bad["rewards"]["stand_tilt"]["weight"] = -1.0
    del bad["rewards"]["stand_anchor"]
    assert len(check_stand_push(bad)) == 4


def test_launcher_variant_1f():
    s = TRAIN.read_text()
    assert "1f) PREFIX=waiter-wbc1f; SEED0=15; RES_NAME=wbc1f; export BHL_STD_MAX=1.0" in s
    assert "TASK_ID=Velocity-BHL-Waiter-WBC-StandPush-v0; FULL_ITERS=10000" in s
    assert "stand-push check" in s and "check_stand_push" in s and "std-clamp-after-update-missing" in s
    assert s.count('[ "$VARIANT" = 1e ] || [ "$VARIANT" = 1f ]') == 3   # smoke ramp, std check, curriculum check
    assert '"${WAITER_STD_LOG_TOL:-1e-6}"' in s                # the std check itself is the predeclared one by default


def test_training_overlay_logs_the_bounded_std():
    t = (REPO / "scripts/train.py").read_text()
    assert "_ppo.PPO.update = _update_then_clamp" in t
    assert "[overlay] action-noise std also bounded after each PPO update" in t
    assert t.index("_ac.ActorCritic.update_distribution = _bounded_update") < t.index("_ppo.PPO.update = _update_then_clamp")
