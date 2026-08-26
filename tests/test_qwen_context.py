from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from robot_vla.model.qwen_context import (
    FrozenQwenContextEncoder,
    QwenContext,
    QwenVLAAdapter,
)


class FakeBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(2.0))
        self.last_kwargs = None

    def forward(self, input_ids, **kwargs):
        self.last_kwargs = kwargs
        tokens = input_ids.float().unsqueeze(-1).expand(-1, -1, 2048) * self.scale
        return SimpleNamespace(hidden_states=(tokens - 1.0, tokens))


class FailingLMHead(nn.Module):
    def forward(self, _value):
        raise AssertionError("LM Head 不应被调用")


class FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048))
        self.model = FakeBaseModel()
        self.lm_head = FailingLMHead()


def _model_inputs():
    return {
        "input_ids": torch.tensor([[1, 2, 0], [3, 0, 0]], dtype=torch.long),
        "attention_mask": torch.tensor([[1, 1, 0], [1, 0, 0]], dtype=torch.bool),
        "mm_token_type_ids": torch.zeros(2, 3, dtype=torch.long),
        "pixel_values": torch.zeros(4, 1536),
        "image_grid_thw": torch.ones(4, 3, dtype=torch.long),
    }


def test_frozen_qwen_encoder_uses_final_hidden_state_without_lm_head() -> None:
    qwen = FakeQwen()
    encoder = FrozenQwenContextEncoder(qwen)
    encoder.train()

    context = encoder(_model_inputs())

    assert encoder.training is False
    assert qwen.training is False
    assert all(parameter.requires_grad is False for parameter in qwen.parameters())
    assert context.tokens.shape == (2, 3, 2048)
    assert context.tokens.requires_grad is False
    assert context.mask.dtype == torch.bool
    assert qwen.model.last_kwargs["use_cache"] is False
    assert qwen.model.last_kwargs["output_hidden_states"] is True
    assert qwen.model.last_kwargs["return_dict"] is True


def test_qwen_vla_adapter_preserves_tokens_mask_and_trainable_gradient() -> None:
    adapter = QwenVLAAdapter()
    tokens = torch.randn(2, 4, 2048)
    mask = torch.tensor([[True, True, False, False], [True, False, False, False]])

    adapted = adapter(QwenContext(tokens=tokens, mask=mask))
    adapted.tokens.sum().backward()

    assert adapted.tokens.shape == (2, 4, 720)
    assert torch.equal(adapted.mask, mask)
    assert torch.count_nonzero(adapted.tokens[~mask]).item() == 0
    assert adapter.skip_projection.weight.grad is not None
    assert adapter.main_projection[0].weight.grad is not None
