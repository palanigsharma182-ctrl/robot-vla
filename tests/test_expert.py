import pytest

torch = pytest.importorskip("torch")

from robot_vla.model.expert import (
    ExpertConfig,
    StandaloneActionExpert,
    sinusoidal_time_embedding,
)
from robot_vla.model.qwen_context import QwenContext


def _tiny_config() -> ExpertConfig:
    return ExpertConfig(
        proprio_dim=5,
        action_dim=3,
        action_horizon=4,
        context_dim=32,
        hidden_size=32,
        state_hidden_size=16,
        num_layers=4,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )


def _inputs(config: ExpertConfig):
    context = QwenContext(
        tokens=torch.randn(2, 6, config.context_dim),
        mask=torch.tensor(
            [[True, True, True, False, False, False], [True, True, False, False, False, False]]
        ),
    )
    proprio = torch.randn(2, config.proprio_dim)
    noisy_action = torch.randn(2, config.action_horizon, config.action_dim)
    flow_time = torch.tensor([1.0, 0.25])
    action_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    return context, proprio, noisy_action, flow_time, action_mask


def test_default_expert_config_matches_qwen_vla_v01_identity() -> None:
    config = ExpertConfig()

    assert config.is_qwen_vla_v01()
    assert config.hidden_size == 720
    assert config.num_layers == 16
    assert config.num_attention_heads == 15
    assert config.num_key_value_heads == 5
    assert config.head_dim == 64
    assert config.intermediate_size == 2048


def test_time_embedding_keeps_flow_time_separate_from_action_slot() -> None:
    config = _tiny_config()
    embedded = sinusoidal_time_embedding(torch.tensor([0.0, 1.0]), config)

    assert embedded.shape == (2, config.hidden_size)
    torch.testing.assert_close(embedded[0, : config.hidden_size // 2], torch.zeros(16))
    torch.testing.assert_close(embedded[0, config.hidden_size // 2 :], torch.ones(16))


def test_expert_outputs_masked_fp32_velocity_and_backpropagates() -> None:
    config = _tiny_config()
    expert = StandaloneActionExpert(config)
    inputs = _inputs(config)

    velocity = expert(*inputs)
    velocity.square().mean().backward()

    action_mask = inputs[-1]
    assert velocity.shape == (2, config.action_horizon, config.action_dim)
    assert velocity.dtype == torch.float32
    assert torch.count_nonzero(velocity[~action_mask]).item() == 0
    assert expert.state_encoder.projection[0].weight.grad is not None
    assert expert.blocks[1].attention.k_projection.weight.grad is not None
    assert expert.velocity_head.weight.grad is not None


def test_context_kv_cache_is_numerically_equivalent() -> None:
    config = _tiny_config()
    expert = StandaloneActionExpert(config).eval()
    inputs = _inputs(config)

    with torch.no_grad():
        uncached = expert(*inputs)
        context_kv = expert.prepare_context_kv(inputs[0])
        cached = expert(*inputs, context_kv=context_kv)

    assert len(context_kv) == config.num_layers // 2
    torch.testing.assert_close(cached, uncached)
