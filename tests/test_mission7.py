"""Mission7 gates independent of Isaac and GPU availability."""
import json
import numpy as np
import pytest
import torch
from tensordict import TensorDict
from bhl_robust.mission.layout import generate, world_xml, edge, SPLITS
from bhl_robust.mission.state import MissionState
from bhl_robust.mission.sensors import pack, FRAME, HISTORY, PROPRIO, LIDAR, DEPTH
from bhl_robust.mission.policy import MissionPolicy


def sample(m, dt=.04, **overrides):
    values = dict(now=m.last_s+dt, dt=dt, upright=True, collision=False,
                  button_contacts=[], activate=False, crossing=[], xy_delta=0.,
                  goal_inside=False, slow=True, object_inside=False, object_slow=True)
    values.update(overrides)
    return m.update(**values)


def test_generation_deterministic_and_all_split_geometries_disjoint():
    hashes = set()
    for split, (_, count) in SPLITS.items():
        for i in range(count):
            layout = generate(split, i)
            assert layout == generate(split, i)
            assert layout.decisions == 7 and layout.turns >= 4
            assert 19.5 <= (len(layout.route)-1)*layout.cell_m <= 34
            assert layout.fingerprint not in hashes
            hashes.add(layout.fingerprint)
            assert all(edge(a, b) in layout.edges for a, b in zip(layout.route, layout.route[1:]))


def test_gates_are_required_cuts_and_plates_are_upstream():
    for i in range(30):
        layout = generate("train", i)
        for k in layout.door_indices:
            links = set(layout.edges)-{edge(layout.route[k], layout.route[k+1])}
            reachable = {layout.route[0]}
            for _ in range(36):
                for a, b in links:
                    if a in reachable or b in reachable:
                        reachable.update((a, b))
            assert layout.route[k] in reachable
            assert layout.route[-1] not in reachable


def test_contact_and_intent_both_required_correct_button_only():
    m = MissionState("doors")
    sample(m, activate=True)  # proximity/intent without contact is insufficient
    sample(m, button_contacts=[(0, True)])  # passive touch is insufficient
    sample(m, activate=True, button_contacts=[(0, False)])
    assert m.open == [False, False] and m.wrong_buttons == 1
    sample(m, activate=True, button_contacts=[(0, True)])
    assert m.open == [True, False] and m.activation_s[0] == pytest.approx(.16)
    assert sample(m, activate=True, button_contacts=[(0, True)]) < 0  # no repeated bonus
    sample(m, crossing=[0, 1])
    assert m.crossed == [True, False]


def test_placement_needs_acquisition_release_dwell_and_all_gates():
    m = MissionState("transport")
    for _ in range(30):
        sample(m, goal_inside=True)
    assert m.completed_s is None
    m.acquire()
    m.crossed[:] = [True, True]
    for _ in range(30):
        sample(m, object_inside=True)
    assert m.completed_s is None  # attached does not count
    m.release()
    for _ in range(24):
        sample(m, object_inside=True)
    assert m.completed_s is None
    sample(m, object_inside=False)
    for _ in range(25):
        sample(m, object_inside=True)
    assert m.completed_s is not None
    assert sample(m, object_inside=True) == 0


def test_dwell_idempotence_dt_and_final_fall_guard():
    for dt in (.02, .04, .1):
        m = MissionState("navigation")
        for _ in range(round(.8/dt)):
            sample(m, dt=dt, goal_inside=True)
        hold = m.hold_s
        sample(m, now=m.last_s, goal_inside=True)
        assert m.hold_s == hold
        sample(m, dt=dt, upright=False, goal_inside=True)
        assert m.completed_s is None and m.hold_s == 0
    with pytest.raises(ValueError):
        sample(MissionState("navigation"), now=10.)


def test_invalid_stale_future_and_dropout_not_free_space():
    np.testing.assert_array_equal(pack([np.nan, np.inf, -1], 0, 0, 12)[:6], 0)
    for stamp, now in ((0, .3), (1, 0), (None, 0)):
        assert not pack([2, 3], stamp, now, 12)[2:4].any()
    a = pack([3, np.nan], 1, 1.1, 12)
    np.testing.assert_array_equal(a[:4], [.25, 0, 1, 0])


def observations(n=4):
    x = torch.zeros(n, HISTORY, FRAME)
    x[..., PROPRIO+LIDAR:PROPRIO+2*LIDAR] = 1
    x[..., PROPRIO+2*LIDAR+1+DEPTH:PROPRIO+2*LIDAR+1+2*DEPTH] = 1
    return TensorDict({"policy": x.flatten(1)}, batch_size=[n])


def test_fusion_dimensions_gradients_and_one_or_both_missing():
    obs = observations()
    model = MissionPolicy(obs)
    a, v = model.act(obs), model.evaluate(obs)
    assert a.shape == (4, 5) and v.shape == (4, 1)
    (a.square().sum()+v.square().sum()+model.entropy.sum()).backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    for missing in ("lidar", "stereo", "both"):
        x = obs["policy"].clone().reshape(-1, HISTORY, FRAME)
        if missing in ("lidar", "both"):
            x[..., PROPRIO+LIDAR:PROPRIO+2*LIDAR] = 0
        if missing in ("stereo", "both"):
            x[..., PROPRIO+2*LIDAR+1+DEPTH:PROPRIO+2*LIDAR+1+2*DEPTH] = 0
        _, weights = model.actor.encode(x.flatten(1))
        if missing == "both":
            assert not weights.any()
        else:
            assert torch.all(weights[..., 0 if missing == "lidar" else 1] == 0)


def test_actor_and_critic_ignore_extra_privileged_fields():
    obs = observations()
    model = MissionPolicy(obs)
    a, v = model.act_inference(obs), model.evaluate(obs)
    obs["global_pose"] = torch.randn(4, 3)*100
    obs["oracle_route"] = torch.randn(4, 16)*100
    assert torch.equal(model.act_inference(obs), a)
    assert torch.equal(model.evaluate(obs), v)


def test_scene_compiles_without_robot_and_has_physical_gates_and_free_object():
    import mujoco
    m = mujoco.MjModel.from_xml_string(world_xml(generate("train", 0)))
    assert m.nmocap == 2 and m.nq == 7
    for name in ("door_0", "door_1", "plate_0_1", "plate_0_-1"):
        idx = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, name)
        assert m.geom_contype[idx] > 0


def test_ppo_update_uses_same_observations_for_actor_and_critic():
    from rsl_rl.algorithms import PPO
    obs = observations(2)
    model = MissionPolicy(obs)
    ppo = PPO(model, num_learning_epochs=1, num_mini_batches=2, schedule="fixed")
    ppo.init_storage("rl", 2, 4, obs, [5])
    old = model.actor.head[-1].weight.detach().clone()
    with torch.no_grad():
        for _ in range(4):
            ppo.act(obs)
            ppo.process_env_step(obs, torch.tensor([1., -.1]), torch.zeros(2, dtype=torch.bool), {})
        ppo.compute_returns(obs)
    losses = ppo.update()
    assert np.isfinite(list(losses.values())).all()
    assert not torch.equal(old, model.actor.head[-1].weight)
