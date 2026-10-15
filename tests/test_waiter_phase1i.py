"""Waiter phase 1i (2026-10-13): a standing-only WBC deployed next to the walking WBC s7 behind a switch. The task
constants, the stamp's stand-only check, the pair (dual) deploy config and its compatibility checks, the switching
controller (which policy acts, what the standing policy sees), and the launchers. No simulator, no ONNX."""
import ast
import math
import sys
from pathlib import Path

import numpy as np
import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

ENV_CFG = REPO / "src/bhl_robust/tasks/waiter_env_cfg.py"
TASKS_INIT = REPO / "src/bhl_robust/tasks/__init__.py"
TRAIN = REPO / "slurm/repo20260923/gpu_waiter_wbc.sbatch"
GATES = REPO / "slurm/repo20260923/cpu_waiter_gates.sbatch"
QUALIFY = REPO / "slurm/repo20260923/cpu_turn_qualify.sbatch"
WALK = REPO / "results/waiter-20261005/wbc1c/training/waiter-wbc1c-s7.deploy.yaml"
OTHER = REPO / "results/waiter-20261005/wbc1h/training/waiter-wbc1h-s21.deploy.yaml"


def _consts(path: Path, names) -> dict:
    tree = ast.parse(path.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.Assign) and len(n.targets) == 1
            and isinstance(n.targets[0], ast.Name) and n.targets[0].id in names]
    ns: dict = {}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(path), "exec"), ns)
    return ns


def test_task_constants_agree():
    from bhl_robust.eval import waiter_wbc as WB
    c = _consts(ENV_CFG, ("STANDONLY_TASK_ID", "STANDONLY_REL_STANDING_ENVS"))
    assert c["STANDONLY_TASK_ID"] == WB.STANDONLY_TASK_ID == "Velocity-BHL-Waiter-WBC-StandOnly-v0"
    assert WB.TASK_IDS[7] == WB.STANDONLY_TASK_ID and c["STANDONLY_REL_STANDING_ENVS"] == WB.STANDONLY_REL_STANDING_ENVS == 1.0
    init = TASKS_INIT.read_text()
    assert "waiter-wbc-standonly" in init and "HumanoidWaiterWbcStandOnlyCfg" in init
    cfg = ENV_CFG.read_text()
    block = cfg[cfg.index("# ---- phase 1i"):]
    assert "class WaiterStandOnlyRewardsCfg(WaiterStandBonusRewardsCfg)" in block       # 1g's two bonuses
    assert "func=waiter_stand_mdp.stand_still_joint_deviation_l1, weight=STAND_STILL_WEIGHT" in block   # + 1h's term
    assert "self.commands.base_velocity.rel_standing_envs = STANDONLY_REL_STANDING_ENVS" in block
    assert "self.events.push_robot.func = waiter_stand_mdp.push_and_reanchor" in block
    assert "class HumanoidWaiterWbcStandOnlyCfg(HumanoidWaiterWbcStanceCfg)" in block and "moving_only" not in block


def _good_env() -> dict:
    from bhl_robust import waiter_asset as W
    return {"commands": {"base_velocity": {"rel_standing_envs": 1.0}},
            "observations": {"policy": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}},
                             "critic": {"gait_clock": {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}}},
            "rewards": {"feet_gait": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_gait_stance"},
                        "stand_anchor_bonus": {"func": "bhl_robust.tasks.waiter_stand_mdp:StandAnchorBonus", "weight": 2.0},
                        "stand_tilt_bonus": {"func": "bhl_robust.tasks.waiter_stand_mdp:stand_tilt_bonus", "weight": 1.0},
                        "stand_still": {"func": "bhl_robust.tasks.waiter_stand_mdp:stand_still_joint_deviation_l1",
                                        "weight": -0.4, "params": {"asset_cfg": {"joint_names": list(W.LEG_JOINTS)}}}},
            "curriculum": {"disturbance": {"func": "bhl_robust.tasks.waiter_wbc_mdp:disturbance_curriculum"}},
            "events": {"push_robot": {"func": "bhl_robust.tasks.waiter_stand_mdp:push_and_reanchor"},
                       "hand_forces": {"func": "bhl_robust.tasks.waiter_wbc_mdp:hand_forces"}}}


def test_check_stand_only():
    from bhl_robust.eval.waiter_wbc import check_stand_only
    assert check_stand_only(_good_env()) == []
    mixed = _good_env()                                          # a walk-and-stand run (1g/1h) is not 1i
    mixed["commands"]["base_velocity"]["rel_standing_envs"] = 0.25
    mixed["curriculum"]["standing_share"] = {"func": "x:standing_share_curriculum"}
    assert check_stand_only(mixed) == ["rel_standing_envs 0.25", "standing_share curriculum present"]
    bad = _good_env()
    del bad["rewards"]["stand_still"]
    bad["rewards"]["stand_anchor"] = {"func": "x:StandAnchor", "weight": -1.0}
    bad["events"]["push_robot"]["func"] = "isaaclab.envs.mdp.events:push_by_setting_velocity"
    p = check_stand_only(bad)
    assert len(p) == 5 and "stand_anchor penalty present" in p


def _stand_copy(tmp_path, edit=None) -> Path:
    """A standing-half deploy config: a real stamped Waiter export with its task line set to the stand-only task."""
    text = OTHER.read_text().replace("task: Velocity-BHL-Waiter-WBC-StandStill-v0", "task: Velocity-BHL-Waiter-WBC-StandOnly-v0")
    assert "task: Velocity-BHL-Waiter-WBC-StandOnly-v0" in text
    if edit:
        text = edit(text)
    p = tmp_path / "stand.deploy.yaml"
    p.write_text(text)
    return p


def test_stamp_dual_writes_once_and_refuses_mismatches(tmp_path):
    from omegaconf import OmegaConf
    from bhl_robust.eval import waiter_wbc as WB
    stand = _stand_copy(tmp_path)
    out = tmp_path / "pair.deploy.yaml"
    assert WB.stamp_dual(WALK, stand, out).startswith("dual deploy written")
    text = out.read_text()
    assert text.startswith(WALK.read_text())                     # the walking config is unchanged
    cfg = OmegaConf.load(out)
    assert cfg.policy_checkpoint_path == OmegaConf.load(WALK).policy_checkpoint_path
    assert cfg[WB.STAND_KEY]["checkpoint"] == OmegaConf.load(stand).policy_checkpoint_path
    assert cfg[WB.STAND_KEY]["switch"] == "clock_held" and cfg[WB.STAND_KEY]["stance_phase"] == 0.5
    assert WB.stamp_dual(WALK, stand, out).startswith("already written")
    with pytest.raises(ValueError):                              # same out, different content: refused
        WB.stamp_dual(stand, stand, out)
    assert out.read_text() == text
    with pytest.raises(ValueError, match="not task Velocity-BHL-Waiter-WBC-StandOnly-v0|is task"):
        WB.stamp_dual(WALK, OTHER, tmp_path / "o2.yaml")          # a walk-and-stand export is not a standing half
    with pytest.raises(ValueError, match="num_observations differs"):
        d = tmp_path / "m"
        d.mkdir()
        WB.stamp_dual(WALK, _stand_copy(d, lambda t: t.replace("num_observations: 83", "num_observations: 86")),
                      tmp_path / "o3.yaml")
    with pytest.raises(ValueError, match="one half of another"):
        WB.stamp_dual(out, stand, tmp_path / "o4.yaml")
    assert not (tmp_path / "o2.yaml").exists() and not (tmp_path / "o3.yaml").exists()


class _Fake:
    def __init__(self, value):
        self.value, self.seen = value, []

    def forward(self, obs):
        self.seen.append(np.array(obs, copy=True))
        return np.full((1, 12), self.value, np.float32)


def _robot_obs(cmd=(0.3, 0.0, 0.0)):
    return np.r_[[1.0, 0.0, 0.0, 0.0], np.zeros(3), np.zeros(24), np.zeros(24), [0.0], cmd].astype(np.float64)


def test_switch_rule_and_what_the_standing_policy_sees(tmp_path):
    from omegaconf import OmegaConf
    from bhl_robust.eval import waiter_wbc as WB
    from bhl_robust.eval.gait_clock import make_controller
    out = tmp_path / "pair.deploy.yaml"
    WB.stamp_dual(WALK, _stand_copy(tmp_path), out)
    ctrl = make_controller(OmegaConf.load(out))
    assert type(ctrl).__name__ == "DualWaiterController"
    assert type(make_controller(OmegaConf.load(WALK))).__name__ == "WaiterWbcController"
    walk, stand = _Fake(0.0), _Fake(0.1)
    ctrl.policy, ctrl.stand_policy = walk, stand
    for _ in range(5):                                           # the clock runs: the walking policy acts
        ctrl.update(_robot_obs())
    assert len(walk.seen) == 5 and not stand.seen and ctrl.clock_step == 5 and not ctrl.standing
    for held in (0, 0, 0, 3):                                    # a caller holds the clock (at step 0, then 3)
        ctrl.clock_step = held
        ctrl.update(_robot_obs())
        assert ctrl.clock_step == held + 1                       # the clock is the caller's: +1 as for one policy
    # the first update at a new held step still goes to the walking policy (it looks like a running clock)
    assert len(walk.seen) == 7 and len(stand.seen) == 2 and ctrl.standing_steps == 2
    o = stand.seen[-1][0]
    assert np.allclose(o[:3], 0.0)                               # zero command, though the caller sent 0.3 m/s
    assert np.allclose(o[-2:], [math.sin(math.pi), math.cos(math.pi)], atol=1e-6)   # phase 0.5, though step 0 is held
    assert np.allclose(walk.seen[0][0][:3], [0.3, 0.0, 0.0])     # the walking policy gets the command as given
    ctrl.reset_phase()
    ctrl.clock_step = 0
    ctrl.update(_robot_obs())
    assert len(stand.seen) == 2 and not ctrl.standing            # a reset forgets the previous clock


def test_launchers():
    s = TRAIN.read_text()
    assert "1i) PREFIX=waiter-wbc1i; SEED0=24; RES_NAME=wbc1i; export BHL_STD_MAX=1.0" in s
    assert "TASK_ID=Velocity-BHL-Waiter-WBC-StandOnly-v0; FULL_ITERS=10000" in s
    assert "stand-only check" in s and "check_stand_only" in s and "standonly-task-not-registered" in s
    assert s.count('[ "$VARIANT" = 1h ] || [ "$VARIANT" = 1i ]') == 3       # later variants append (1l, 10-15)
    g = GATES.read_text()
    assert 'if [ -n "${WAITER_GATES_DUAL_WALK:-}" ]; then' in g and "-m bhl_robust.eval.waiter_wbc dual --walk" in g
    assert 'TURNQ_DEPLOY="${WAITER_GATES_DUAL_WALK:+$DEPLOY}"' in g
    q = QUALIFY.read_text()
    assert 'if [ -n "${TURNQ_DEPLOY:-}" ]; then' in q and "DEPLOY=$TURNQ_DEPLOY" in q
