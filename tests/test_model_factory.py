from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from robot_vla.contracts import QWEN_MODEL_ID, QWEN_REVISION
from robot_vla.model.expert import ExpertConfig
from robot_vla.model.factory import (
    QWEN_TEXT_LAYER_COUNT,
    build_qwen_vla_policy,
    load_frozen_qwen_v01,
    validate_qwen_v01_architecture,
)
from robot_vla.model.qwen_context import (
    FrozenQwenContextEncoder,
    FrozenQwenLayerContextEncoder,
)


class ExactFakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(
            model_type="qwen3_5",
            architectures=["Qwen3_5ForConditionalGeneration"],
            text_config=SimpleNamespace(
                hidden_size=2048,
                num_hidden_layers=24,
                full_attention_interval=4,
            ),
            vision_config=SimpleNamespace(
                hidden_size=1024,
                depth=24,
                out_hidden_size=2048,
            ),
        )
        self.model = nn.Linear(1, 1)
        self.lm_head = nn.Linear(1, 1)


def _tiny_expert_config() -> ExpertConfig:
    return ExpertConfig(
        hidden_size=32,
        state_hidden_size=16,
        num_layers=2,
        intermediate_size=64,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
    )


def test_factory_validates_full_qwen_architecture_and_freezes_it() -> None:
    qwen = ExactFakeQwen()

    policy = build_qwen_vla_policy(qwen, expert_config=_tiny_expert_config())

    assert all(not parameter.requires_grad for parameter in qwen.parameters())
    assert qwen.training is False
    assert policy.expert.config == _tiny_expert_config()
    assert isinstance(policy.context_encoder, FrozenQwenContextEncoder)
    assert not isinstance(policy.context_encoder, FrozenQwenLayerContextEncoder)


def test_factory_builds_explicit_intermediate_qwen_context_layer() -> None:
    qwen = ExactFakeQwen()

    policy = build_qwen_vla_policy(
        qwen,
        expert_config=_tiny_expert_config(),
        context_layer=12,
    )

    assert isinstance(policy.context_encoder, FrozenQwenLayerContextEncoder)
    assert policy.context_encoder.layer == 12
    assert QWEN_TEXT_LAYER_COUNT == 24

    with pytest.raises(ValueError, match=r"\[1,24\]"):
        build_qwen_vla_policy(qwen, context_layer=0)


def test_factory_rejects_qwen_architecture_drift() -> None:
    qwen = ExactFakeQwen()
    qwen.config.text_config.num_hidden_layers = 23

    with pytest.raises(ValueError, match="Qwen 架构"):
        validate_qwen_v01_architecture(qwen)


def test_weight_loader_uses_exact_revision_bf16_and_no_remote_code(monkeypatch) -> None:
    transformers = pytest.importorskip("transformers")
    captured = {}
    qwen = ExactFakeQwen()

    def fake_from_pretrained(model_id, **kwargs):
        captured["model_id"] = model_id
        captured.update(kwargs)
        return qwen

    monkeypatch.setattr(
        transformers.Qwen3_5ForConditionalGeneration,
        "from_pretrained",
        fake_from_pretrained,
    )

    loaded = load_frozen_qwen_v01(local_files_only=True)

    assert loaded is qwen
    assert captured["model_id"] == QWEN_MODEL_ID
    assert captured["revision"] == QWEN_REVISION
    assert captured["dtype"] == torch.bfloat16
    assert captured["trust_remote_code"] is False
    assert captured["use_safetensors"] is True
    assert captured["weights_only"] is True
    assert captured["local_files_only"] is True
