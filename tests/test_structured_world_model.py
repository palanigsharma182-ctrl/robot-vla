import pytest

torch = pytest.importorskip("torch")

from robot_vla.model.structured_world_model import (
    TINY_STRUCTURED_WORLD_MODEL_ARCH,
    TinyStructuredWorldModel,
    TinyStructuredWorldModelConfig,
)
from robot_vla.training.structured_world_model import structured_world_model_loss


def _inputs(batch_size: int = 2):
    initial_state = torch.zeros(batch_size, 15)
    action_prefix = torch.zeros(batch_size, 4, 8)
    command_target_prefix = torch.zeros(batch_size, 4, 7)
    transition_mask = torch.ones(batch_size, 4, dtype=torch.bool)
    return initial_state, action_prefix, command_target_prefix, transition_mask


def _command_sensitive_model() -> TinyStructuredWorldModel:
    """构造只响应 commanded target 第一个维度的确定性反例模型。"""

    model = TinyStructuredWorldModel()
    with torch.no_grad():
        for parameter in model.parameters():
            parameter.zero_()
        model.transition[0].weight[0, 15 + 8] = 1.0
        model.transition[2].weight[0, 0] = 1.0
        model.transition[4].weight.fill_(1.0)
        model.delta_head.weight[0, 0] = 1.0
    return model


def test_default_config_and_parameter_budget_are_frozen() -> None:
    config = TinyStructuredWorldModelConfig()
    model = TinyStructuredWorldModel(config)

    assert TINY_STRUCTURED_WORLD_MODEL_ARCH == "tiny-structured-world-model/v0"
    assert (
        config.state_dim,
        config.command_dim,
        config.action_dim,
        config.rollout_horizon,
    ) == (15, 7, 8, 4)
    assert config.hidden_dim == 128
    assert sum(parameter.numel() for parameter in model.parameters()) == 22_543


def test_config_digest_binds_every_numerical_variant() -> None:
    default = TinyStructuredWorldModelConfig()
    narrower = TinyStructuredWorldModelConfig(hidden_dim=64)

    assert default.sha256() == (
        "dd15873a6cad15282da903c2e53fc46d530f906f8a78b9344ff0c458a60b0bf2"
    )
    assert default.sha256() != narrower.sha256()
    assert default.to_dict()["command_dim"] == 7


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"state_dim": 14}, "state_dim"),
        ({"state_dim": 15.0}, "state_dim"),
        ({"command_dim": 6}, "command_dim"),
        ({"action_dim": 7}, "action_dim"),
        ({"rollout_horizon": 16}, "rollout_horizon"),
        ({"hidden_dim": 0}, "hidden_dim"),
        ({"hidden_dim": True}, "hidden_dim"),
        ({"rms_norm_eps": 0.0}, "rms_norm_eps"),
        ({"normalized_state_abs_limit": float("inf")}, "state_abs_limit"),
        ({"normalized_action_abs_limit": -1.0}, "action_abs_limit"),
    ],
)
def test_config_rejects_contract_drift(kwargs, message) -> None:
    with pytest.raises((TypeError, ValueError), match=message):
        TinyStructuredWorldModelConfig(**kwargs)


def test_forward_returns_fp32_four_step_state_hold_baseline() -> None:
    model = TinyStructuredWorldModel()
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs()
    initial_state[0, 0] = 0.5

    output = model(
        initial_state,
        action_prefix,
        command_target_prefix,
        transition_mask,
    )

    assert output.predicted_state.shape == (2, 4, 15)
    assert output.predicted_delta.shape == (2, 4, 15)
    assert output.predicted_state.dtype == torch.float32
    assert output.predicted_delta.dtype == torch.float32
    torch.testing.assert_close(
        output.predicted_state,
        initial_state.unsqueeze(1).expand(-1, 4, -1),
    )
    assert torch.count_nonzero(output.predicted_delta).item() == 0


def test_invalid_tail_is_zero_and_masked_inputs_do_not_affect_output() -> None:
    torch.manual_seed(5)
    model = TinyStructuredWorldModel()
    torch.nn.init.normal_(model.delta_head.weight, std=0.02)
    initial_state, first_actions, first_commands, transition_mask = _inputs()
    transition_mask[0] = torch.tensor([True, True, False, False])
    second_actions = first_actions.clone()
    second_commands = first_commands.clone()
    second_actions[0, 2:] = 100.0
    second_commands[0, 2:] = 100.0

    first = model(initial_state, first_actions, first_commands, transition_mask)
    second = model(initial_state, second_actions, second_commands, transition_mask)

    torch.testing.assert_close(first.predicted_state, second.predicted_state)
    torch.testing.assert_close(first.predicted_delta, second.predicted_delta)
    assert torch.count_nonzero(first.predicted_state[0, 2:]).item() == 0
    assert torch.count_nonzero(first.predicted_delta[0, 2:]).item() == 0


def test_forward_keeps_fp32_under_outer_cpu_autocast() -> None:
    torch.manual_seed(6)
    model = TinyStructuredWorldModel().eval()
    torch.nn.init.normal_(model.delta_head.weight, std=0.02)
    inputs = _inputs()

    with torch.no_grad():
        baseline = model(*inputs)
        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            autocast_output = model(*inputs)

    assert autocast_output.predicted_state.dtype == torch.float32
    torch.testing.assert_close(autocast_output.predicted_state, baseline.predicted_state)


def test_forward_rejects_non_fp32_model_parameters() -> None:
    model = TinyStructuredWorldModel().to(dtype=torch.bfloat16)

    with pytest.raises(TypeError, match="参数必须保持 FP32"):
        model(*_inputs())


def test_transition_mask_must_be_boolean_nonempty_true_prefix() -> None:
    model = TinyStructuredWorldModel()
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs()

    with pytest.raises(TypeError, match="bool"):
        model(
            initial_state,
            action_prefix,
            command_target_prefix,
            transition_mask.float(),
        )
    with pytest.raises(ValueError, match="没有有效 transition"):
        model(
            initial_state,
            action_prefix,
            command_target_prefix,
            torch.zeros_like(transition_mask),
        )
    hole = transition_mask.clone()
    hole[0] = torch.tensor([True, False, True, False])
    with pytest.raises(ValueError, match="True-prefix"):
        model(initial_state, action_prefix, command_target_prefix, hole)


@pytest.mark.parametrize(
    "field",
    [
        "state_nan",
        "state_range",
        "action_nan",
        "action_range",
        "command_nan",
        "command_range",
    ],
)
def test_forward_rejects_nonfinite_or_out_of_contract_values(field: str) -> None:
    model = TinyStructuredWorldModel()
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs()
    if field == "state_nan":
        initial_state[0, 0] = torch.nan
    elif field == "state_range":
        initial_state[0, 0] = 5.1
    elif field == "action_nan":
        action_prefix[0, 0, 0] = torch.nan
    elif field == "action_range":
        action_prefix[0, 0, 0] = 1.1
    elif field == "command_nan":
        command_target_prefix[0, 0, 0] = torch.nan
    else:
        command_target_prefix[0, 0, 0] = 5.1

    with pytest.raises(ValueError):
        model(
            initial_state,
            action_prefix,
            command_target_prefix,
            transition_mask,
        )


def test_forward_rejects_invalid_command_shape_and_dtype() -> None:
    model = TinyStructuredWorldModel()
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs()

    with pytest.raises(ValueError, match="command_target_prefix"):
        model(
            initial_state,
            action_prefix,
            torch.zeros(2, 4, 6),
            transition_mask,
        )
    with pytest.raises(TypeError, match="浮点"):
        model(
            initial_state,
            action_prefix,
            command_target_prefix.to(dtype=torch.int64),
            transition_mask,
        )


def test_forward_rejects_full_sixteen_step_chunk() -> None:
    model = TinyStructuredWorldModel()
    initial_state, _, command_target_prefix, transition_mask = _inputs()

    with pytest.raises(ValueError, match="action_prefix"):
        model(
            initial_state,
            torch.zeros(2, 16, 8),
            command_target_prefix,
            transition_mask,
        )


def test_changing_later_action_cannot_change_earlier_predictions() -> None:
    torch.manual_seed(7)
    model = TinyStructuredWorldModel()
    torch.nn.init.normal_(model.delta_head.weight, std=0.02)
    initial_state, first_actions, command_target_prefix, transition_mask = _inputs()
    second_actions = first_actions.clone()
    second_actions[:, 2, 0] = 0.75

    first = model(
        initial_state,
        first_actions,
        command_target_prefix,
        transition_mask,
    )
    second = model(
        initial_state,
        second_actions,
        command_target_prefix,
        transition_mask,
    )

    torch.testing.assert_close(first.predicted_state[:, :2], second.predicted_state[:, :2])
    assert not torch.equal(first.predicted_state[:, 2:], second.predicted_state[:, 2:])


def test_command_target_changes_prediction_with_same_actual_state_and_action() -> None:
    model = _command_sensitive_model()
    initial_state, action_prefix, first_commands, transition_mask = _inputs()
    second_commands = first_commands.clone()
    second_commands[:, 0, 0] = 0.5

    first = model(initial_state, action_prefix, first_commands, transition_mask)
    second = model(initial_state, action_prefix, second_commands, transition_mask)

    assert not torch.equal(first.predicted_state[:, 0], second.predicted_state[:, 0])


def test_changing_later_command_cannot_change_earlier_predictions() -> None:
    model = _command_sensitive_model()
    initial_state, action_prefix, first_commands, transition_mask = _inputs()
    second_commands = first_commands.clone()
    second_commands[:, 2, 0] = 0.5

    first = model(initial_state, action_prefix, first_commands, transition_mask)
    second = model(initial_state, action_prefix, second_commands, transition_mask)

    torch.testing.assert_close(first.predicted_state[:, :2], second.predicted_state[:, :2])
    assert not torch.equal(first.predicted_state[:, 2:], second.predicted_state[:, 2:])


def test_rollout_is_autoregressive() -> None:
    model = TinyStructuredWorldModel()
    # 用 forward hook 构造每一步 delta=当前 state，使状态依次翻倍。

    def transition_hook(_module, inputs, _output):
        state = inputs[0][:, :15]
        return torch.cat(
            (
                state,
                torch.zeros(
                    state.shape[0],
                    113,
                    dtype=state.dtype,
                    device=state.device,
                ),
            ),
            dim=-1,
        )

    handles = [
        model.transition.register_forward_hook(transition_hook),
        model.delta_head.register_forward_hook(
            lambda _module, inputs, _output: inputs[0][:, :15]
        ),
    ]
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs(
        batch_size=1
    )
    initial_state.fill_(0.25)
    try:
        output = model(
            initial_state,
            action_prefix,
            command_target_prefix,
            transition_mask,
        )
    finally:
        for handle in handles:
            handle.remove()

    expected_scale = torch.tensor([2.0, 4.0, 8.0, 16.0]).view(1, 4, 1)
    torch.testing.assert_close(
        output.predicted_state,
        initial_state.unsqueeze(1) * expected_scale,
    )


def test_masked_state_loss_matches_manual_float32_mean() -> None:
    model = TinyStructuredWorldModel()
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs()
    transition_mask[0] = torch.tensor([True, True, False, False])
    target_state = torch.ones(2, 4, 15)

    output = structured_world_model_loss(
        model,
        initial_state,
        action_prefix,
        command_target_prefix,
        target_state,
        transition_mask,
    )

    assert output.loss.dtype == torch.float32
    assert output.loss.item() == pytest.approx(1.0)
    assert output.state_loss.item() == pytest.approx(1.0)
    torch.testing.assert_close(output.per_step_state_mse, torch.ones(4))
    assert output.valid_transitions_per_step.tolist() == [2, 2, 1, 1]


def test_padding_target_does_not_change_loss() -> None:
    model = TinyStructuredWorldModel()
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs()
    transition_mask[0] = torch.tensor([True, False, False, False])
    first_target = torch.zeros(2, 4, 15)
    second_target = first_target.clone()
    second_target[0, 1:] = torch.finfo(second_target.dtype).max

    first = structured_world_model_loss(
        model,
        initial_state,
        action_prefix,
        command_target_prefix,
        first_target,
        transition_mask,
    )
    second = structured_world_model_loss(
        model,
        initial_state,
        action_prefix,
        command_target_prefix,
        second_target,
        transition_mask,
    )

    torch.testing.assert_close(first.loss, second.loss)
    torch.testing.assert_close(first.per_step_state_mse, second.per_step_state_mse)
    assert torch.isfinite(second.loss)


def test_per_step_loss_is_zero_when_the_whole_tail_step_is_invalid() -> None:
    model = TinyStructuredWorldModel()
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs()
    transition_mask[:] = torch.tensor([True, True, False, False])
    target_state = torch.ones(2, 4, 15)

    output = structured_world_model_loss(
        model,
        initial_state,
        action_prefix,
        command_target_prefix,
        target_state,
        transition_mask,
    )

    assert output.valid_transitions_per_step.tolist() == [2, 2, 0, 0]
    assert output.per_step_state_mse.tolist() == pytest.approx([1.0, 1.0, 0.0, 0.0])


def test_loss_rejects_invalid_target() -> None:
    model = TinyStructuredWorldModel()
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs()

    with pytest.raises(ValueError, match="target_state"):
        structured_world_model_loss(
            model,
            initial_state,
            action_prefix,
            command_target_prefix,
            torch.zeros(2, 3, 15),
            transition_mask,
        )
    target_state = torch.zeros(2, 4, 15)
    target_state[0, 0, 0] = 5.1
    with pytest.raises(ValueError, match="归一化状态范围"):
        structured_world_model_loss(
            model,
            initial_state,
            action_prefix,
            command_target_prefix,
            target_state,
            transition_mask,
        )


def test_loss_backpropagates_after_delta_head_is_nonzero() -> None:
    torch.manual_seed(11)
    model = TinyStructuredWorldModel()
    torch.nn.init.normal_(model.delta_head.weight, std=0.02)
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs()
    initial_state.normal_(std=0.1)
    action_prefix.uniform_(-0.5, 0.5)
    command_target_prefix.normal_(std=0.1)
    target_state = torch.zeros(2, 4, 15)

    output = structured_world_model_loss(
        model,
        initial_state,
        action_prefix,
        command_target_prefix,
        target_state,
        transition_mask,
    )
    output.loss.backward()

    assert model.transition[0].weight.grad is not None
    assert torch.count_nonzero(model.transition[0].weight.grad).item() > 0
    assert model.delta_head.weight.grad is not None
    assert torch.count_nonzero(model.delta_head.weight.grad).item() > 0


def test_eval_is_deterministic_and_batch_permutation_equivariant() -> None:
    torch.manual_seed(13)
    model = TinyStructuredWorldModel().eval()
    torch.nn.init.normal_(model.delta_head.weight, std=0.02)
    initial_state, action_prefix, command_target_prefix, transition_mask = _inputs(
        batch_size=3
    )
    initial_state.normal_(std=0.1)
    action_prefix.uniform_(-0.5, 0.5)
    command_target_prefix.normal_(std=0.1)
    permutation = torch.tensor([2, 0, 1])

    with torch.no_grad():
        first = model(
            initial_state,
            action_prefix,
            command_target_prefix,
            transition_mask,
        )
        repeated = model(
            initial_state,
            action_prefix,
            command_target_prefix,
            transition_mask,
        )
        permuted = model(
            initial_state[permutation],
            action_prefix[permutation],
            command_target_prefix[permutation],
            transition_mask[permutation],
        )

    torch.testing.assert_close(first.predicted_state, repeated.predicted_state)
    torch.testing.assert_close(
        first.predicted_state[permutation], permuted.predicted_state
    )


def test_state_dict_round_trip_is_numerically_identical() -> None:
    torch.manual_seed(17)
    first_model = TinyStructuredWorldModel().eval()
    torch.nn.init.normal_(first_model.delta_head.weight, std=0.02)
    second_model = TinyStructuredWorldModel().eval()
    second_model.load_state_dict(first_model.state_dict())
    inputs = _inputs()

    with torch.no_grad():
        first = first_model(*inputs)
        second = second_model(*inputs)

    torch.testing.assert_close(first.predicted_state, second.predicted_state)
    torch.testing.assert_close(first.predicted_delta, second.predicted_delta)
