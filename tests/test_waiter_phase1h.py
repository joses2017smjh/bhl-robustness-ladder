"""Waiter phase 1h (2026-10-12): the standing terms published humanoid policies use. Isaac Lab's stand-still term
(gated on the whole command), the walk-first standing-share curriculum, the task constants shared by the task and the
deploy stamp, the stamp's stand-still check, and the launcher variant. No simulator is started."""
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


S = _load(STAND_MDP, "waiter_stand_mdp_1h_under_test")


def test_still_gate_uses_the_whole_command():
    cmd = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 0.6], [0.05, 0.0, 0.0], [0.05, 0.05, 0.0], [0.35, 0.0, 0.0],
                    [0.0, 0.0, 0.05]])
    # turning in place (0, 0, 0.6) is NOT still, unlike Isaac Lab's xy-only gate
    assert S.command_is_still(cmd).tolist() == [True, False, True, False, False, True]
    assert S.STILL_COMMAND_THRESHOLD == 0.06


def test_still_gate_accepts_torch_tensors():
    torch = pytest.importorskip("torch")
    assert S.command_is_still(torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, -0.6]])).tolist() == [True, False]


def test_standing_share_ramp():
    f = S.standing_share
    assert (S.STAND_SHARE_START, S.STAND_SHARE_END) == (0.02, 0.25)
    assert f(0, 4000, 7000) == pytest.approx(0.02) and f(4000, 4000, 7000) == pytest.approx(0.02)
    assert f(5500, 4000, 7000) == pytest.approx(0.135) and f(7000, 4000, 7000) == pytest.approx(0.25)
    assert f(9999, 4000, 7000) == pytest.approx(0.25)
    with pytest.raises(ValueError):
        f(10, 5, 5)


def test_falling_cannot_pay():
    """A still standing env earns about 2.0 + 2.0 per second from the zero-velocity tracking kernels; the stand-still
    charge at a deep crouch (0.5 rad off default on all 12 leg joints) stays below that, so staying up is worth more
    than ending the episode (the 1e/1f failure)."""
    c = _consts(ENV_CFG, ("STAND_STILL_WEIGHT",))
    assert c["STAND_STILL_WEIGHT"] == -0.4
    assert abs(c["STAND_STILL_WEIGHT"]) * 12 * 0.5 < 2.0 + 2.0


def test_task_constants_agree():
    from bhl_robust.eval import waiter_wbc as WB
    c = _consts(ENV_CFG, ("STANDSTILL_TASK_ID", "STAND_STILL_WEIGHT"))
    assert c["STANDSTILL_TASK_ID"] == WB.STANDSTILL_TASK_ID == "Velocity-BHL-Waiter-WBC-StandStill-v0"
    assert WB.TASK_IDS[6] == WB.STANDSTILL_TASK_ID and c["STAND_STILL_WEIGHT"] == WB.STAND_STILL_WEIGHT
    assert WB.STILL_COMMAND_THRESHOLD == S.STILL_COMMAND_THRESHOLD
    assert WB.STAND_SHARE == (S.STAND_SHARE_START, S.STAND_SHARE_END)
    init = TASKS_INIT.read_text()
    assert "waiter-wbc-standstill" in init and "HumanoidWaiterWbcStandStillCfg" in init
    cfg = ENV_CFG.read_text()
    block = cfg[cfg.index("# ---- phase 1h"):]
    assert "func=waiter_stand_mdp.stand_still_joint_deviation_l1, weight=STAND_STILL_WEIGHT" in block
    assert 'SceneEntityCfg("robot", joint_names=W.LEG_JOINTS, preserve_order=True)' in block
    assert "func=waiter_stand_mdp.standing_share_curriculum" in block and "class WaiterStandStillCurriculumCfg(WaiterCurriculumCfg)" in block
    assert "self.commands.base_velocity.rel_standing_envs = waiter_stand_mdp.STAND_SHARE_START" in block
    assert "class HumanoidWaiterWbcStandStillCfg(HumanoidWaiterWbcStanceCfg)" in block
    for other in ("moving_only", "reanchor", "Bonus", "StandAnchor", "stand_tilt"):
        assert other not in block, other
    mdp = STAND_MDP.read_text()
    term = mdp[mdp.index("def stand_still_joint_deviation_l1"):mdp.index("def standing_share_curriculum")]
    assert "il_mdp.joint_deviation_l1(" in term and "command_is_still(command, command_threshold)" in term


def _good_env() -> dict:
    from bhl_robust import waiter_asset as W
    return {"commands": {"base_velocity": {"rel_standing_envs": 0.02}},
            "observations": {"policy": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}},
                             "critic": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}}},
            "rewards": {"feet_gait": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_gait_stance"},
                        "stand_still": {"func": "bhl_robust.tasks.waiter_stand_mdp:stand_still_joint_deviation_l1",
                                        "weight": -0.4, "params": {"command_threshold": 0.06,
                                                                   "asset_cfg": {"joint_names": list(W.LEG_JOINTS)}}}},
            "curriculum": {"disturbance": {"func": "bhl_robust.tasks.waiter_wbc_mdp:disturbance_curriculum"},
                           "standing_share": {"func": "bhl_robust.tasks.waiter_stand_mdp:standing_share_curriculum",
                                              "params": {"start_iter": 4000, "end_iter": 7000, "steps_per_iter": 24,
                                                         "lo": 0.02, "hi": 0.25}}},
            "events": {"push_robot": {"func": "isaaclab.envs.mdp.events:push_by_setting_velocity"},
                       "hand_forces": {"func": "bhl_robust.tasks.waiter_wbc_mdp:hand_forces"}}}


def test_check_stand_still():
    from bhl_robust.eval.waiter_wbc import check_stand_still
    assert check_stand_still(_good_env()) == []
    one_d = _good_env()                                          # a 1d run: share 0.25, no stand-still term or curriculum
    one_d["commands"]["base_velocity"]["rel_standing_envs"] = 0.25
    del one_d["rewards"]["stand_still"], one_d["curriculum"]["standing_share"]
    p = check_stand_still(one_d)
    assert p[:2] == ["rel_standing_envs 0.25", "standing_share curriculum {}"] and len(p) == 6   # + 4 for the term
    one_g = _good_env()                                          # 1g's bonuses or 1f's re-anchoring push are not 1h
    one_g["rewards"]["stand_anchor_bonus"] = {"func": "x:StandAnchorBonus", "weight": 2.0}
    one_g["events"]["push_robot"]["func"] = "bhl_robust.tasks.waiter_stand_mdp:push_and_reanchor"
    assert len(check_stand_still(one_g)) == 2
    bad = _good_env()
    bad["rewards"]["stand_still"]["params"]["command_threshold"] = 0.1
    bad["rewards"]["stand_still"]["params"]["asset_cfg"]["joint_names"] = ["leg_left_hip_roll_joint"]
    bad["curriculum"]["standing_share"]["params"]["hi"] = 0.5
    assert len(check_stand_still(bad)) == 3
    smoke = _good_env()                                          # a smoke's compressed window passes only with window=False
    smoke["curriculum"]["standing_share"]["params"].update(start_iter=0, end_iter=2)
    assert len(check_stand_still(smoke)) == 1 and check_stand_still(smoke, window=False) == []


def test_launcher_variant_1h():
    s = TRAIN.read_text()
    assert "1h) PREFIX=waiter-wbc1h; SEED0=21; RES_NAME=wbc1h; export BHL_STD_MAX=1.0" in s
    assert "TASK_ID=Velocity-BHL-Waiter-WBC-StandStill-v0; FULL_ITERS=10000" in s
    assert "stand-still check" in s and "check_stand_still" in s and "standstill-task-not-registered" in s
    assert "window=sys.argv[2] != '1'" in s and '"$run/params/env.yaml" "$SMOKE"' in s
    assert s.count('[ "$VARIANT" = 1g ] || [ "$VARIANT" = 1h ]; }') == 3   # smoke ramp, std check, curriculum check
    assert "env.curriculum.standing_share.params.start_iter=0 env.curriculum.standing_share.params.end_iter=2" in s
    assert "Curriculum/standing_share" in s and "standing-share-check" in s
    assert '"${WAITER_STD_LOG_TOL:-1e-6}"' in s                # the std check is the predeclared one by default
