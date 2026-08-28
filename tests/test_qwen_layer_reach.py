from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from robot_vla.diagnostics.qwen_layer_reach import (
    FrozenQwenLayerContextEncoder,
    FrozenQwenLayerPairContextEncoder,
    QwenLayerFusionAdapter,
    QwenLayerPairContext,
    QwenSemanticKeyGeometryValueAdapter,
    QwenSemanticKeyGeometryValueContext,
    SemanticKeyGeometryValueActionExpert,
)
from robot_vla.model.expert import ExpertConfig, StandaloneActionExpert
from robot_vla.model.qwen_context import QwenContext, QwenVLAAdapter


class FakeBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.last_kwargs = None

    def forward(self, input_ids, **kwargs):
        self.last_kwargs = kwargs
        base = input_ids.float().unsqueeze(-1).expand(-1, -1, 2048)
        return SimpleNamespace(hidden_states=tuple(base + layer for layer in range(25)))


class FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            text_config=SimpleNamespace(hidden_size=2048, num_hidden_layers=24)
        )
        self.model = FakeBaseModel()


def _model_inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0]], dtype=torch.bool),
        "pixel_values": torch.zeros(2, 1536),
        "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
        "mm_token_type_ids": torch.zeros(1, 3, dtype=torch.long),
    }


def test_layer12_encoder_uses_state_after_layer12_and_freezes_qwen() -> None:
    qwen = FakeQwen()
    encoder = FrozenQwenLayerContextEncoder(qwen, 12)

    context = encoder(_model_inputs())

    assert context.tokens.shape == (1, 3, 2048)
    assert torch.all(context.tokens[0, :, 0] == torch.tensor([13.0, 14.0, 12.0]))
    assert context.tokens.requires_grad is False
    assert torch.equal(context.mask, torch.tensor([[True, True, False]]))
    assert qwen.model.last_kwargs["output_hidden_states"] is True
    assert all(parameter.requires_grad is False for parameter in qwen.parameters())


def test_layer_encoder_rejects_embedding_and_out_of_range_layer() -> None:
    qwen = FakeQwen()

    with pytest.raises(ValueError, match=r"\[1,24\]"):
        FrozenQwenLayerContextEncoder(qwen, 0)
    with pytest.raises(ValueError, match=r"\[1,24\]"):
        FrozenQwenLayerContextEncoder(qwen, 25)


def test_layer_pair_encoder_returns_layer12_and_layer24_from_one_forward() -> None:
    qwen = FakeQwen()
    encoder = FrozenQwenLayerPairContextEncoder(qwen)

    context = encoder(_model_inputs())

    assert torch.all(context.layer12_tokens[0, :, 0] == torch.tensor([13.0, 14.0, 12.0]))
    assert torch.all(context.layer24_tokens[0, :, 0] == torch.tensor([25.0, 26.0, 24.0]))
    assert torch.equal(context.mask, torch.tensor([[True, True, False]]))


def test_fusion_projects_both_layers_and_learns_normalized_weights() -> None:
    qwen = FakeQwen()
    context = FrozenQwenLayerPairContextEncoder(qwen)(_model_inputs())
    adapter = QwenLayerFusionAdapter()

    fused = adapter(context)
    fused.tokens.sum().backward()

    assert fused.tokens.shape == (1, 3, 720)
    assert torch.equal(adapter.normalized_weights(), torch.tensor([0.5, 0.5]))
    assert torch.count_nonzero(fused.tokens[~context.mask]).item() == 0
    assert adapter.layer12_projection.skip_projection.weight.grad is not None
    assert adapter.layer24_projection.skip_projection.weight.grad is not None
    assert adapter.fusion_logits.grad is not None


def test_fusion_layer12_projection_matches_layer12_only_initialization() -> None:
    torch.manual_seed(42)
    layer12_only = QwenVLAAdapter()
    torch.manual_seed(42)
    fusion = QwenLayerFusionAdapter()

    for name, value in layer12_only.state_dict().items():
        assert torch.equal(value, fusion.layer12_projection.state_dict()[name])


def test_semantic_kv_maps_layer24_to_key_and_same_position_layer12_to_value() -> None:
    pair = FrozenQwenLayerPairContextEncoder(FakeQwen())(_model_inputs())
    adapter = QwenSemanticKeyGeometryValueAdapter()

    context = adapter(pair)
    expected_key = adapter.key_projection(
        QwenContext(tokens=pair.layer24_tokens, mask=pair.mask)
    )
    expected_value = adapter.value_projection(
        QwenContext(tokens=pair.layer12_tokens, mask=pair.mask)
    )
    (context.key_tokens.sum() + context.value_tokens.sum()).backward()

    assert context.tokens is context.key_tokens
    assert context.key_tokens.shape == context.value_tokens.shape == (1, 3, 720)
    torch.testing.assert_close(context.key_tokens, expected_key.tokens)
    torch.testing.assert_close(context.value_tokens, expected_value.tokens)
    assert torch.count_nonzero(context.key_tokens[~pair.mask]).item() == 0
    assert torch.count_nonzero(context.value_tokens[~pair.mask]).item() == 0
    assert adapter.key_projection.skip_projection.weight.grad is not None
    assert adapter.value_projection.skip_projection.weight.grad is not None


def test_semantic_kv_adapter_rejects_unaligned_token_sequences() -> None:
    adapter = QwenSemanticKeyGeometryValueAdapter()
    context = QwenLayerPairContext(
        layer12_tokens=torch.zeros(1, 2, 2048),
        layer24_tokens=torch.zeros(1, 3, 2048),
        mask=torch.ones(1, 2, dtype=torch.bool),
    )

    with pytest.raises(ValueError, match="token 位置一一对齐"):
        adapter(context)


def _tiny_semantic_kv_config() -> ExpertConfig:
    return ExpertConfig(
        proprio_dim=5,
        action_dim=3,
        action_horizon=4,
        context_dim=720,
        hidden_size=32,
        state_hidden_size=16,
        num_layers=4,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )


def _semantic_kv_expert_inputs(config: ExpertConfig):
    context = QwenSemanticKeyGeometryValueContext(
        key_tokens=torch.randn(2, 3, config.context_dim, requires_grad=True),
        value_tokens=torch.randn(2, 3, config.context_dim, requires_grad=True),
        mask=torch.tensor([[True, True, False], [True, True, True]]),
    )
    proprio = torch.randn(2, config.proprio_dim)
    noisy_action = torch.randn(2, config.action_horizon, config.action_dim)
    flow_time = torch.tensor([0.25, 0.75])
    action_mask = torch.tensor([[True, True, False, False], [True, True, True, False]])
    return context, proprio, noisy_action, flow_time, action_mask


def test_semantic_kv_expert_projects_key_and_value_from_their_declared_layers() -> None:
    config = _tiny_semantic_kv_config()
    expert = SemanticKeyGeometryValueActionExpert(config).eval()
    inputs = _semantic_kv_expert_inputs(config)
    context = inputs[0]

    kv_cache = expert.prepare_context_kv(context)
    first_attention = expert.blocks[1].attention
    expected_key = first_attention.k_projection(context.key_tokens).view(
        2, 3, config.num_key_value_heads, config.head_dim
    )
    expected_value = first_attention.v_projection(context.value_tokens).view_as(expected_key)

    assert len(kv_cache) == config.num_layers // 2
    torch.testing.assert_close(kv_cache[0].key, expected_key.transpose(1, 2))
    torch.testing.assert_close(kv_cache[0].value, expected_value.transpose(1, 2))
    assert torch.equal(kv_cache[0].mask, context.mask)


def test_semantic_kv_cached_and_internal_projection_match_and_backpropagate() -> None:
    config = _tiny_semantic_kv_config()
    expert = SemanticKeyGeometryValueActionExpert(config).eval()
    inputs = _semantic_kv_expert_inputs(config)

    with torch.no_grad():
        internal = expert(*inputs)
        cached = expert(*inputs, context_kv=expert.prepare_context_kv(inputs[0]))
    torch.testing.assert_close(cached, internal)

    expert(*inputs).square().mean().backward()
    first_attention = expert.blocks[1].attention
    assert inputs[0].key_tokens.grad is not None
    assert inputs[0].value_tokens.grad is not None
    assert first_attention.q_projection.weight.grad is not None
    assert first_attention.k_projection.weight.grad is not None
    assert first_attention.v_projection.weight.grad is not None


def test_semantic_kv_preserves_expert_and_layer12_value_initialization() -> None:
    config = _tiny_semantic_kv_config()
    torch.manual_seed(42)
    control_expert = StandaloneActionExpert(config)
    torch.manual_seed(42)
    semantic_expert = SemanticKeyGeometryValueActionExpert(config)

    assert control_expert.state_dict().keys() == semantic_expert.state_dict().keys()
    for name, value in control_expert.state_dict().items():
        assert torch.equal(value, semantic_expert.state_dict()[name])

    torch.manual_seed(42)
    layer12_only = QwenVLAAdapter()
    torch.manual_seed(42)
    semantic_adapter = QwenSemanticKeyGeometryValueAdapter()
    for name, value in layer12_only.state_dict().items():
        assert torch.equal(value, semantic_adapter.value_projection.state_dict()[name])
