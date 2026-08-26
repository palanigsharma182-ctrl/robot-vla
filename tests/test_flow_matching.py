import pytest

torch = pytest.importorskip("torch")

from robot_vla.training.flow_matching import (
    build_critical_event_mask,
    euler_integrate_actions,
    masked_flow_mse,
    sample_flow_training_target,
)


def test_flow_training_target_matches_fixed_interpolation_and_is_reproducible() -> None:
    action = torch.tensor([[[0.25, -0.5], [0.0, 0.0]]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])

    first = sample_flow_training_target(
        action,
        mask,
        generator=torch.Generator().manual_seed(17),
    )
    second = sample_flow_training_target(
        action,
        mask,
        generator=torch.Generator().manual_seed(17),
    )

    torch.testing.assert_close(first.flow_time, second.flow_time)
    torch.testing.assert_close(first.noise, second.noise)
    time = first.flow_time.view(1, 1, 1)
    expected_noisy = time * first.noise + (1.0 - time) * action
    torch.testing.assert_close(first.noisy_action[:, :1], expected_noisy[:, :1])
    torch.testing.assert_close(first.target_velocity[:, :1], (first.noise - action)[:, :1])
    assert torch.count_nonzero(first.noisy_action[:, 1:]).item() == 0
    assert 0.001 <= first.flow_time.item() <= 1.0


def test_masked_flow_mse_ignores_padding_and_reduces_in_fp32() -> None:
    prediction = torch.zeros(1, 3, 2, dtype=torch.bfloat16)
    target = torch.zeros_like(prediction)
    target[:, 0] = 1.0
    target[:, 1:] = 100.0
    mask = torch.tensor([[True, False, False]])

    loss = masked_flow_mse(prediction, target, mask)

    assert loss.dtype == torch.float32
    assert loss.item() == pytest.approx(1.0)


def test_empty_event_mask_returns_differentiable_zero_when_allowed() -> None:
    prediction = torch.ones(1, 4, 2, requires_grad=True)
    target = torch.zeros_like(prediction)
    empty = torch.zeros(1, 4, dtype=torch.bool)

    loss = masked_flow_mse(prediction, target, empty, allow_empty=True)
    loss.backward()

    assert loss.item() == 0.0
    assert torch.count_nonzero(prediction.grad).item() == 0


def test_critical_event_mask_intersects_event_validity_and_execution_prefix() -> None:
    event = torch.tensor([[True, True, True, True, True]])
    valid = torch.tensor([[True, True, False, True, True]])

    critical = build_critical_event_mask(event, valid, executed_action_steps=4)

    assert critical.tolist() == [[True, True, False, True, False]]


def test_euler_integration_uses_noise_to_action_direction() -> None:
    target_action = torch.tensor([[[0.4, -0.25], [0.0, 0.0]]], dtype=torch.float32)
    initial_noise = torch.tensor([[[0.9, 0.5], [3.0, 3.0]]], dtype=torch.float32)
    mask = torch.tensor([[True, False]])
    constant_velocity = initial_noise - target_action
    seen_times = []

    def velocity_fn(_state, flow_time):
        seen_times.append(flow_time.clone())
        return constant_velocity

    result = euler_integrate_actions(velocity_fn, initial_noise, mask, num_steps=10)

    torch.testing.assert_close(result[:, :1], target_action[:, :1], atol=1e-6, rtol=0.0)
    assert torch.count_nonzero(result[:, 1:]).item() == 0
    assert [round(value.item(), 1) for value in seen_times] == [
        1.0,
        0.9,
        0.8,
        0.7,
        0.6,
        0.5,
        0.4,
        0.3,
        0.2,
        0.1,
    ]


def test_euler_clamps_only_final_state() -> None:
    initial_noise = torch.full((1, 1, 1), 2.0)
    mask = torch.tensor([[True]])
    seen_states = []

    def zero_velocity(state, _flow_time):
        seen_states.append(state.clone())
        return torch.zeros_like(state)

    result = euler_integrate_actions(zero_velocity, initial_noise, mask, num_steps=2)

    assert all(state.item() == pytest.approx(2.0) for state in seen_states)
    assert result.item() == pytest.approx(1.0)
