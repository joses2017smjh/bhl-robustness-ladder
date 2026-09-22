import numpy as np
import torch
from tensordict import TensorDict
from bhl_robust.mission.overnight import AuditedState, OraclePolicy, study_matrix
from bhl_robust.mission.state import MissionState
from bhl_robust.mission.sensors import FRAME,HISTORY


def test_instrumentation_preserves_every_reward_transition():
    original=MissionState('transport'); audited=AuditedState('transport')
    assert original.acquire()==audited.acquire()
    original.release(); audited.release()
    for i in range(1,40):
        args=dict(now=i*.04,dt=.04,upright=True,collision=i==3,
                  button_contacts=[(0,True),(1,False)] if i==1 else [(1,True)] if i==2 else [],
                  activate=True,crossing=[0,1] if i==4 else [],xy_delta=.01,
                  goal_inside=True,slow=True,object_inside=i>=5,object_slow=True)
        assert original.update(**args)==audited.update(**args)
    assert audited.completed_s is not None
    assert sum(audited.reward_totals.values())>10
    assert audited.reward_totals['correct_button']==4
    assert audited.reward_totals['wrong_button']==-1
    assert audited.reward_totals['acquisition']==2
    assert audited.reward_totals['success']==10


def test_oracle_observation_separate_and_has_gradient_path():
    obs=TensorDict({'policy':torch.zeros(2,HISTORY*FRAME),'oracle':torch.tensor([[.3,.2,.4],[-.3,.2,.4]])},batch_size=[2])
    policy=OraclePolicy(obs)
    values=policy.evaluate(obs)
    assert values.shape==(2,1)
    assert not torch.equal(policy.act_inference(obs)[0],policy.act_inference(obs)[1])
    values.sum().backward()
    assert all(torch.isfinite(p.grad).all() for p in policy.parameters() if p.grad is not None)


def test_study_matrix_is_bounded_matched_and_independent():
    matrix=study_matrix()
    assert len(matrix)==14
    assert len([r for r in matrix if r.get('mode')=='train'])==12
    assert {r.get('stage') for r in matrix[:5]}=={'approach','branches','navigation','doors','transport'}
    assert all(r.get('requires') is None for r in matrix[:9])
    shared=matrix[8]
    for r in matrix[9:12]:
        assert r['requires']=='C4'
        for k in ('updates','horizon','lr','entropy','noise','schedule','approach_distance','seed'):
            assert r[k]==shared[k]
    assert sum(r['wall_hours']*2 for r in matrix)==56
