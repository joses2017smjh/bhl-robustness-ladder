"""Waiter phase 1l (2026-10-15): the standing-only WBC with its drift since the anchor as an input. The drift features
(frame, wrap, clip; arrays and tensors), the task constants shared by the task, the stamp and the controller, the
stamp's checks, the pair config, the switching controller's drift input (anchor at the first standing update, noise,
insertion before the gait clock, the simulator hook), and the launcher. No simulator, no ONNX."""
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
TRAIN = REPO / "slurm/repo20260923/gpu_waiter_wbc.sbatch"
WALK = REPO / "results/waiter-20261005/wbc1c/training/waiter-wbc1c-s7.deploy.yaml"
OTHER = REPO / "results/waiter-20261005/wbc1h/training/waiter-wbc1h-s21.deploy.yaml"


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


S = _load(STAND_MDP, "waiter_stand_mdp_1l_under_test")


def test_drift_features_frame_wrap_and_clip():
    a = np.array([1.0, 2.0])
    assert np.allclose(S.drift_features(a, 0.3, a, 0.3), 0.0)                       # at the anchor: zero
    # 5 cm forward along the base's own heading (yaw 90 deg: +y in the world) -> dx = 1 unit, dy = 0
    f = S.drift_features(a + [0.0, 0.05], math.pi / 2, a, math.pi / 2)
    assert np.allclose(f, [1.0, 0.0, 0.0], atol=1e-9)
    f = S.drift_features(a + [0.05, 0.0], math.pi / 2, a, math.pi / 2)               # world +x = the base's right
    assert np.allclose(f, [0.0, -1.0, 0.0], atol=1e-9)
    f = S.drift_features(a, math.radians(-175.0), a, math.radians(175.0))            # wraps through +-180
    assert f[2] == pytest.approx(2.0)                                                 # +10 deg = 2 units
    assert np.allclose(S.drift_features(a + [5.0, 0.0], 0.0, a, 0.0), [10.0, 0.0, 0.0])   # clipped at DRIFT_CLIP
    assert (S.DRIFT_CLIP, S.DRIFT_NOISE) == (10.0, 0.2)


def test_drift_features_on_tensors():
    torch = pytest.importorskip("torch")
    xy = torch.tensor([[0.0, 0.05], [0.0, 0.0]])
    f = S.drift_features(xy, torch.tensor([math.pi / 2, 0.0]), torch.zeros(2, 2), torch.tensor([math.pi / 2, 0.0]))
    assert torch.allclose(f, torch.tensor([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0]]), atol=1e-6)


def test_task_constants_agree():
    from bhl_robust.eval import waiter_wbc as WB
    c = _consts(ENV_CFG, ("STANDDRIFT_TASK_ID", "DRIFT_OBS"))
    assert c["STANDDRIFT_TASK_ID"] == WB.STANDDRIFT_TASK_ID == "Velocity-BHL-Waiter-WBC-StandDrift-v0"
    assert WB.TASK_IDS[8] == WB.STANDDRIFT_TASK_ID and c["DRIFT_OBS"] == WB.DRIFT_OBS == 3
    assert WB.DRIFT_NOISE == S.DRIFT_NOISE
    cfg = ENV_CFG.read_text()
    block = cfg[cfg.index("# ---- phase 1l"):]
    pol = block[block.index("class _WaiterDriftPolicyObsCfg"):block.index("class _WaiterDriftCriticObsCfg")]
    assert pol.index("upper_body = ObsTerm") < pol.index("stand_drift = ObsTerm") < pol.index("gait_clock = ObsTerm")
    assert "noise=_Unoise(n_min=-waiter_stand_mdp.DRIFT_NOISE, n_max=waiter_stand_mdp.DRIFT_NOISE)" in pol
    cri = block[block.index("class _WaiterDriftCriticObsCfg"):block.index("class WaiterDriftObservationsCfg")]
    assert "stand_drift = ObsTerm" in cri and "noise=" not in cri
    assert "class HumanoidWaiterWbcStandDriftCfg(HumanoidWaiterWbcStandOnlyCfg)" in block
    mdp = STAND_MDP.read_text()
    term = mdp[mdp.index("def stand_drift("):mdp.index("def standing_share_curriculum")]
    assert 'getattr(env, "reward_manager", None)' in term and "term.prev.unsqueeze(-1)" in term


def _env(policy_terms, critic_terms, drift_noise=True):
    from bhl_robust import waiter_asset as W
    obs = {"policy": {t: {"func": f"x:{t}"} for t in policy_terms}, "critic": {t: {"func": f"x:{t}"} for t in critic_terms}}
    for g in ("policy", "critic"):
        obs[g]["gait_clock"] = {"func": "bhl_robust.tasks.gait_clock_mdp:gait_clock_stance"}
        if "stand_drift" in obs[g]:
            obs[g]["stand_drift"] = {"func": "bhl_robust.tasks.waiter_stand_mdp:stand_drift",
                                     "params": {"reward_term": "stand_anchor_bonus"}}
    if drift_noise and "stand_drift" in obs["policy"]:
        obs["policy"]["stand_drift"]["noise"] = {"n_min": -0.2, "n_max": 0.2}
    return {"commands": {"base_velocity": {"rel_standing_envs": 1.0}}, "observations": obs,
            "rewards": {"feet_gait": {"func": "bhl_robust.tasks.gait_clock_mdp:feet_gait_stance"},
                        "stand_anchor_bonus": {"func": "x:StandAnchorBonus", "weight": 2.0},
                        "stand_tilt_bonus": {"func": "x:stand_tilt_bonus", "weight": 1.0},
                        "stand_still": {"func": "x:stand_still_joint_deviation_l1", "weight": -0.4,
                                        "params": {"asset_cfg": {"joint_names": list(W.LEG_JOINTS)}}}},
            "curriculum": {"disturbance": {"func": "x:disturbance_curriculum"}},
            "events": {"push_robot": {"func": "x:push_and_reanchor"}, "hand_forces": {"func": "x:hand_forces"}}}


def test_check_stand_drift():
    from bhl_robust.eval import waiter_wbc as WB
    pol = WB.POLICY_TERMS[:-1] + ["stand_drift"]
    cri = WB.CRITIC_TERMS[:-1] + ["stand_drift"]
    assert WB.check_stand_drift(_env(pol, cri)) == []
    assert WB.check_stand_drift(_env(pol, cri, drift_noise=False)) == ["policy stand_drift noise {}"]
    p = WB.check_stand_drift(_env(WB.POLICY_TERMS[:-1], WB.CRITIC_TERMS[:-1]))        # a 1i run: no drift input
    assert len(p) == 2 and all("stand_drift func" in x for x in p)


def _stand_copy(tmp_path, drift=True, n_obs=86) -> Path:
    t = OTHER.read_text().replace("task: Velocity-BHL-Waiter-WBC-StandStill-v0",
                                  "task: " + ("Velocity-BHL-Waiter-WBC-StandDrift-v0" if drift else "Velocity-BHL-Waiter-WBC-StandOnly-v0"))
    t = t.replace("num_observations: 83", f"num_observations: {n_obs}")
    p = tmp_path / "stand.deploy.yaml"
    p.write_text(t)
    return p


def test_pair_config_allows_the_drift_inputs_only_for_the_drift_task(tmp_path):
    from bhl_robust.eval import waiter_wbc as WB
    assert WB.stamp_dual(WALK, _stand_copy(tmp_path), tmp_path / "pair.yaml").startswith("dual deploy written")
    d = tmp_path / "x"
    d.mkdir()
    with pytest.raises(ValueError, match="num_observations differs"):              # a stand-only half with 86 inputs
        WB.stamp_dual(WALK, _stand_copy(d, drift=False, n_obs=86), tmp_path / "bad.yaml")
    with pytest.raises(ValueError, match="num_observations differs"):              # a drift half without them
        WB.stamp_dual(WALK, _stand_copy(d, drift=True, n_obs=83), tmp_path / "bad2.yaml")


class _Fake:
    def __init__(self):
        self.seen = []

    def forward(self, obs):
        self.seen.append(np.array(obs, copy=True))
        return np.zeros((1, 12), np.float32)


class _Data:
    def __init__(self):
        self.qpos = np.zeros(40)
        self.qpos[3] = 1.0                                                          # identity quaternion


class _Slot:
    qpos_adr = 0


def _robot_obs():
    return np.r_[[1.0, 0.0, 0.0, 0.0], np.zeros(3), np.zeros(24), np.zeros(24), [0.0], [0.0, 0.0, 0.0]].astype(np.float64)


def test_controller_feeds_the_drift_before_the_clock(tmp_path):
    from omegaconf import OmegaConf
    from bhl_robust.eval import waiter_wbc as WB
    from bhl_robust.eval.gait_clock import make_controller
    WB.stamp_dual(WALK, _stand_copy(tmp_path), tmp_path / "pair.yaml")
    ctrl = make_controller(OmegaConf.load(tmp_path / "pair.yaml"))
    assert type(ctrl).__name__ == "DualWaiterController" and ctrl.drift
    walk, stand = _Fake(), _Fake()
    ctrl.policy, ctrl.stand_policy = walk, stand
    ctrl.update(_robot_obs())                                                       # clock runs: walking policy
    ctrl.clock_step = 10
    ctrl.update(_robot_obs())                                                       # first held update: still walking
    with pytest.raises(RuntimeError, match="attach_sim"):                           # the drift needs the simulator
        ctrl.clock_step = 10
        ctrl.update(_robot_obs())
    data = _Data()
    ctrl.attach_sim(data, _Slot())
    for k in range(4):
        ctrl.clock_step = 10
        data.qpos[0] = 0.01 * k                                                     # the base slides 1 cm per update
        ctrl.update(_robot_obs())
    assert len(stand.seen) == 4 and all(o.shape == (1, 86) for o in stand.seen) and len(walk.seen) == 2
    log = ctrl.drift_log
    assert log[0]["starting"] and np.allclose(log[0]["clean"], 0.0)                 # anchor at the first standing update
    assert log[3]["clean"][0] == pytest.approx(0.6)                                 # 3 cm forward = 0.6 units
    fed = stand.seen[3][0, 81:84]
    assert np.allclose(fed, log[3]["fed"], atol=1e-6) and np.all(np.abs(fed - log[3]["clean"]) <= 0.2 + 1e-6)
    assert np.allclose(stand.seen[3][0, 84:], [math.sin(math.pi), math.cos(math.pi)], atol=1e-6)   # clock still last
    ctrl.policy_observations[:] = 0.0                                                # a runner reset (Q2, turn test)
    ctrl.clock_step = 10
    data.qpos[0] = 0.5
    ctrl.update(_robot_obs())
    assert ctrl.drift_log[-1]["starting"] and np.allclose(ctrl.drift_log[-1]["clean"], 0.0)   # a new anchor
    ctrl.reset_phase()
    assert ctrl.drift_log == [] and ctrl._anchor is None


def test_runner_hook_and_launcher():
    src = (REPO / "src/bhl_robust/eval/multi_robot.py").read_text()
    assert 'if callable(getattr(ctrl, "attach_sim", None)):' in src and "ctrl.attach_sim(self.d, slot)" in src
    s = TRAIN.read_text()
    assert "1l) PREFIX=waiter-wbc1l; SEED0=33; RES_NAME=wbc1l; export BHL_STD_MAX=1.0" in s
    assert "TASK_ID=Velocity-BHL-Waiter-WBC-StandDrift-v0; FULL_ITERS=10000" in s
    assert '[ "$VARIANT" = 1l ] && OBS_POLICY=86' in s and "check_stand_drift" in s and "standdrift-task-not-registered" in s
    assert s.count('[ "$VARIANT" = 1i ] || [ "$VARIANT" = 1l ]; }') == 3
    assert 'if [ "$VARIANT" = 1l ]; then SEED0=${WAITER_1L_SEED0:-$SEED0}; RES_NAME=${WAITER_1L_RES:-$RES_NAME}; fi' in s


def test_deploy_side_loads_the_drift_helper_without_the_isaac_tasks():
    import subprocess
    code = ("import sys; from bhl_robust.eval import waiter_wbc as WB; f = WB._stand_mdp().drift_features; "
            "import numpy as np; assert np.allclose(f(np.zeros(2), 0.0, np.zeros(2), 0.0), 0.0); "
            "bad = [m for m in sys.modules if m == 'isaaclab' or m.startswith('isaacsim') or m == 'bhl_robust.tasks']; "
            "assert not bad, bad")
    r = subprocess.run([sys.executable, "-c", code], cwd=REPO, env={**__import__("os").environ, "PYTHONPATH": str(REPO / "src")},
                       stdin=subprocess.DEVNULL, capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-2000:]
