import numpy as np
import pytest
from bhl_robust.mission.state import MissionState
from bhl_robust.mission.approach_debug import DebugEnv,command_action,classify_fall
from bhl_robust.mission.overnight import StudyEnv
from pathlib import Path


def tick(state,**changes):
    values=dict(now=state.last_s+.04,dt=.04,upright=True,collision=False,button_contacts=[],
        activate=False,crossing=[],xy_delta=0.,goal_inside=True,slow=True,object_inside=False,object_slow=False)
    values.update(changes)
    return state.update(**values)


@pytest.mark.parametrize('fixture,success,failure',[
    ('inside',True,None),('outside',False,None),('fallen',False,'fall'),
    ('brief',False,None),('fast',False,None),('collision',False,'excessive_collision')])
def test_approach_scoring_fixtures(fixture,success,failure):
    state=MissionState('approach')
    for i in range(30):
        tick(state,goal_inside=fixture!='outside' and (fixture!='brief' or i<10),
            upright=fixture!='fallen' or i<20,slow=fixture!='fast',collision=fixture=='collision')
    assert (state.completed_s is not None)==success
    assert state.failure==failure
    if success:assert state.completed_s==pytest.approx(1.)


def test_command_scaling_preserves_requested_supported_velocity():
    command=np.array([.3,-.2,.2]);action=command_action(command)
    np.testing.assert_allclose(np.tanh(action[:3])*[.4,.35,.4],command)


def test_diagnostic_preserves_original_observations_and_sparse_reward(tmp_path):
    root=Path(__file__).resolve().parents[1]
    original=StudyEnv(root,tmp_path/'original',stage='approach',seed=1000,approach_distance=.4)
    debug=DebugEnv(root,tmp_path/'debug',stage='approach',seed=1000,approach_distance=.4,dense=True)
    np.testing.assert_array_equal(original.reset(1),debug.reset(1))
    base_return=dense_return=0.
    for _ in range(4):
        action=command_action([.3,0,0]);a,r,d,_=original.step(action);b,s,e,_=debug.step(action)
        np.testing.assert_array_equal(a,b);assert d==e;base_return+=r;dense_return+=s
        if d:break
    assert dense_return-base_return==pytest.approx(10*(debug.initial_distance-debug.distance()))
    assert debug.metrics()['minimum_goal_distance_m']<=debug.metrics()['final_goal_distance_m']
    assert debug.metrics()['hold_s']==debug.state.hold_s
    assert debug.metrics()['action_hold_s']==pytest.approx(.2)


def test_ppo_telemetry_does_not_change_updates_or_random_stream(monkeypatch):
    import copy,torch
    from tensordict import TensorDict
    from rsl_rl.algorithms import PPO
    from bhl_robust.mission.overnight import OraclePolicy
    from bhl_robust.mission.sensors import FRAME,HISTORY
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1]/'scripts'))
    from mission7_debug_train import UpdateTelemetry
    torch.manual_seed(19)
    obs=TensorDict({'policy':torch.zeros(1,HISTORY*FRAME),'oracle':torch.tensor([[.2,.1,.3]])},batch_size=[1])
    policy=OraclePolicy(obs);other=copy.deepcopy(policy)
    first=PPO(policy,num_learning_epochs=1,num_mini_batches=2,schedule='fixed')
    second=PPO(other,num_learning_epochs=1,num_mini_batches=2,schedule='fixed')
    first.init_storage('rl',1,8,obs,[5])
    with torch.no_grad():
        for i in range(8):
            first.act(obs);first.process_env_step(obs,torch.tensor([float(i)/10]),torch.tensor([False]),{})
        first.compute_returns(obs)
    second.storage=copy.deepcopy(first.storage)
    audit=UpdateTelemetry(second)
    torch.manual_seed(51);first.update();rng=torch.random.get_rng_state()
    torch.manual_seed(51);metrics=audit.update()
    assert torch.equal(rng,torch.random.get_rng_state())
    assert all(torch.equal(a,b) for a,b in zip(policy.parameters(),other.parameters()))
    assert metrics['analytic_kl_post_update']>=0
    assert 0<=metrics['clip_fraction_post_update']<=1
    assert metrics['gradient_norm_after_clip_mean']<=1.0001


def test_stall_recovery_pulse_is_bounded_and_only_happens_once():
    from types import SimpleNamespace
    from bhl_robust.mission.approach_debug import RecoveryTranslationController
    data=SimpleNamespace(time=2.2,xpos=np.zeros((1,3)),qpos=np.array([0,0,0,1,0,0,0.]))
    env=SimpleNamespace(runner=SimpleNamespace(d=data),slot=SimpleNamespace(body_id=0,qpos_adr=0),
        initial_distance=.65,distance=lambda:.65,phase='policy')
    ctrl=RecoveryTranslationController(env);target=np.array([0,.65])
    np.testing.assert_allclose(np.tanh(ctrl.action(target)[:3])*[.4,.35,.4],[.3,0,0])
    data.time=2.4
    np.testing.assert_allclose(np.tanh(ctrl.action(target)[:3])*[.4,.35,.4],[.3,0,0])
    data.time=3.
    assert abs(ctrl.action(target)[0])<1e-10
    data.time=5.
    assert abs(ctrl.action(target)[0])<1e-10


def test_route_recovery_requires_sustained_stall_and_does_not_repeat(monkeypatch):
    from types import SimpleNamespace
    from bhl_robust.mission.approach_debug import MeasuredRouteController,RecoveryRouteController
    env=SimpleNamespace(runner=SimpleNamespace(d=SimpleNamespace(time=3.)),
        state=SimpleNamespace(open=[False,False]),phase='advance',diagnostic_trace=[])
    monkeypatch.setattr(MeasuredRouteController,'action',lambda self:command_action([0,.35,0]))
    controller=RecoveryRouteController(env)
    assert np.tanh(controller.action()[1])>.9
    env.diagnostic_trace=[dict(time_s=1.8+i*.04,command=[0,.35,0],xy=[0,0]) for i in range(31)]
    np.testing.assert_allclose(np.tanh(controller.action()[:3])*[.4,.35,.4],[.3,0,0])
    env.runner.d.time=3.2
    np.testing.assert_allclose(np.tanh(controller.action()[:3])*[.4,.35,.4],[.3,0,0])
    env.runner.d.time=3.6;env.phase='advance'
    env.diagnostic_trace=[dict(time_s=2.4+i*.04,command=[0,.35,0],xy=[0,0]) for i in range(31)]
    assert abs(controller.action()[0])<1e-10
