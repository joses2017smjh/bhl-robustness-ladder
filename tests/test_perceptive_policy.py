import pytest
import torch

from bhl_robust.research.perceptive_policy import (
    ARMS, CELLS, HISTORY, RAYS, CausalPointHistory, TerrainStudent,
    TerrainTeacher, edge_reconstruction_loss, generalized_advantage,
    privileged_height, trajectory_penalty, trunk,
    velocity_tracking_qualified,
)


def test_privileged_height_uses_base_not_overhead_ray_origin():
    height = privileged_height(torch.tensor([.3]), torch.tensor([[0., .01, .03]]))
    assert torch.allclose(height, torch.tensor([[0., -.01, -.03]]), atol=1e-7)
    with pytest.raises(ValueError, match="missed"):
        privileged_height(torch.tensor([.3]), torch.tensor([[float("inf")]]))


def test_tracking_qualification_rejects_opposite_and_stationary_motion():
    # All three travel requests cover 2m. Correct, opposite and stopped motion
    # have integrated velocity errors 0m, 4m and 2m respectively.
    actual = velocity_tracking_qualified(torch.ones(3, dtype=torch.bool),
        torch.full((3,), 2.), torch.tensor([0., 4., 2.]))
    assert actual.tolist() == [True, False, False]


def test_tracking_qualification_accepts_returning_path_without_net_displacement():
    commands = torch.tensor([[.2, 0.], [0., .2], [-.2, 0.], [0., -.2]])
    assert torch.allclose(commands.sum(0), torch.zeros(2))
    distance = torch.linalg.vector_norm(commands, dim=-1).sum()[None]
    error = torch.linalg.vector_norm(commands-commands, dim=-1).sum()[None]
    assert velocity_tracking_qualified(torch.tensor([True]), distance, error).item()


def test_tracking_qualification_requires_survival_and_movement_with_exact_error_boundary():
    result = velocity_tracking_qualified(torch.tensor([True, True, False, True]),
        torch.tensor([1., 1., 1., .1]), torch.tensor([.5, .5001, 0., 0.]))
    assert result.tolist() == [True, False, False, False]
    with pytest.raises(ValueError, match="finite nonnegative"):
        velocity_tracking_qualified(torch.tensor([True]), torch.tensor([1.]), torch.tensor([float("nan")]))


def test_delayed_points_never_invent_pre_reset_measurements():
    state = CausalPointHistory(2, rays=2, history=2, delay_steps=1)
    p = torch.ones(2, 2, 3)
    v = torch.ones(2, 2, dtype=torch.bool)
    assert not state.update(p, v, 0)[..., 3].any()
    first = state.update(p*2, v, 1)
    assert torch.all(first[:, 0, :, :3] == 1)
    assert torch.allclose(first[:, 0, :, 4], torch.full((2, 2), .04))
    assert torch.equal(first, state.update(p*9, v, 1))
    state.reset(torch.tensor([0]))
    next_packet = state.update(p*3, v, 2)
    assert not next_packet[0, ..., 3].any()
    assert torch.all(next_packet[1, 0, :, :3] == 2)


def test_time_skips_and_nonfinite_returns():
    state = CausalPointHistory(1, rays=1)
    p = torch.full((1, 1, 3), float("nan"))
    packet = state.update(p, torch.ones(1, 1, dtype=torch.bool), 0)
    assert torch.isfinite(packet).all() and not packet[..., 3].any()
    with pytest.raises(ValueError, match="missing sensor tick"):
        state.update(p, torch.zeros(1, 1, dtype=torch.bool), 2)


def test_teacher_expansion_preserves_gait_and_critic():
    actor, critic = trunk(45, 12), trunk(48, 1)
    state = {"actor."+k: v for k, v in actor.state_dict().items()}
    state.update({"critic."+k: v for k, v in critic.state_dict().items()})
    teacher = TerrainTeacher()
    teacher.warm_start(state)
    p, h, vel = torch.randn(3, 45), torch.randn(3, CELLS), torch.randn(3, 3)
    assert torch.allclose(teacher(p, h), actor(p), atol=1e-6)
    assert torch.allclose(teacher.value(torch.cat((p, h, vel), -1)), critic(torch.cat((p, vel), -1)).flatten(), atol=1e-6)


@pytest.mark.parametrize("arm", ARMS)
def test_student_all_missing_cloud_is_finite_and_trainable(arm):
    model = TerrainStudent(arm)
    p = torch.randn(2, 45)
    history = torch.zeros(2, HISTORY, RAYS, 5)
    action = model(p, history)
    assert action.shape == (2, 12) and torch.isfinite(action).all()
    loss = action.square().mean()+edge_reconstruction_loss(model.reconstruct(p, history), torch.zeros(2, CELLS))
    loss.backward()
    assert any(x.grad is not None and torch.isfinite(x.grad).all() for x in model.parameters())


def test_memory_arm_has_causal_dependence_on_observed_past():
    torch.manual_seed(6)
    p = torch.randn(1, 45)
    a = torch.zeros(1, HISTORY, RAYS, 5)
    b = a.clone()
    b[:, 1, :, 2:4] = 1.
    b[:, 1, :, 4] = .04
    for arm in ("dense", "query"):
        model = TerrainStudent(arm).eval()
        assert torch.equal(model(p, a), model(p, b))
    model = TerrainStudent("query_memory").eval()
    assert not torch.allclose(model.reconstruct(p, a), model.reconstruct(p, b))


def test_trajectory_penalty_does_not_cross_episode_reset():
    s, t = torch.ones(2, 12), torch.zeros(2, 12)
    result = trajectory_penalty(s, t, torch.tensor([False, True]), .1)
    assert torch.allclose(result, torch.tensor([-.1, 0.]))


def test_gae_propagates_future_penalty_and_stops_at_terminal():
    rewards = torch.tensor([[0.], [-1.], [100.]])
    values = torch.zeros_like(rewards)
    done = torch.tensor([[False], [True], [True]])
    advantage, _ = generalized_advantage(rewards, values, done, torch.zeros(1), gamma=1., lam=1.)
    assert torch.equal(advantage, torch.tensor([[-1.], [-1.], [100.]]))
@pytest.mark.parametrize("arm", ["dense", "query", "query_memory", "query_memory_trajectory"])
def test_export_has_measured_inputs_only_and_variable_batch_parity(tmp_path, arm):
    from bhl_robust.research.perceptive_policy import TerrainStudent, export_student, PROPRIO, HISTORY, RAYS
    model = TerrainStudent(arm)
    receipt = export_student(model, tmp_path/"actor.jit")
    loaded = torch.jit.load(str(tmp_path/"actor.jit"))
    assert not any("critic" in name or "log_std" in name for name in loaded.state_dict())
    assert torch.isfinite(loaded(torch.zeros(3, PROPRIO), torch.zeros(3, HISTORY, RAYS, 5))).all()
    assert receipt["maximum_parity_error"] < 1e-5
