"""冻结 Qwen3.5 Backbone，并把最终多模态 hidden states 投影到 Expert 宽度。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from robot_vla.model.layers import FP32RMSNorm


@dataclass(frozen=True)
class QwenContext:
    tokens: torch.Tensor
    mask: torch.Tensor


class FrozenQwenContextEncoder(nn.Module):
    """只调用 Qwen base model，永远绕过 LM Head、冻结参数并保持 eval。"""

    context_dim = 2048

    def __init__(self, qwen_for_conditional_generation: nn.Module) -> None:
        super().__init__()
        self.qwen = qwen_for_conditional_generation
        if not hasattr(self.qwen, "model"):
            raise ValueError("Qwen conditional model 必须暴露 base model 属性 model")
        config = getattr(self.qwen, "config", None)
        text_config = getattr(config, "text_config", None)
        hidden_size = int(getattr(text_config, "hidden_size", -1))
        if hidden_size != self.context_dim:
            raise ValueError(
                f"Qwen text hidden size 应为 {self.context_dim}，实际为 {hidden_size}"
            )
        self.qwen.requires_grad_(False)
        self.qwen.eval()
        super().train(False)

    def train(self, mode: bool = True) -> FrozenQwenContextEncoder:
        del mode
        super().train(False)
        self.qwen.eval()
        return self

    @torch.no_grad()
    def forward(self, model_inputs: dict[str, Any]) -> QwenContext:
        if "labels" in model_inputs:
            raise ValueError("Context Encoder 不接受 labels，也不能调用语言建模损失")
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        missing = required.difference(model_inputs)
        if missing:
            raise ValueError(f"Qwen model input 缺少字段: {sorted(missing)}")
        self.qwen.eval()
        outputs = self.qwen.model(
            **model_inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        if not hidden_states:
            raise RuntimeError("Qwen base model 没有返回 hidden_states")
        tokens = hidden_states[-1]
        input_ids = model_inputs["input_ids"]
        if tokens.shape != (*input_ids.shape, self.context_dim):
            raise RuntimeError(
                f"Qwen context 应为 [B,N,{self.context_dim}]，实际为 {tuple(tokens.shape)}"
            )
        mask = model_inputs["attention_mask"].bool()
        if mask.shape != input_ids.shape:
            raise RuntimeError("Qwen context mask shape 必须与 input_ids 相同")
        return QwenContext(tokens=tokens, mask=mask)


class QwenVLAAdapter(nn.Module):
    """D010 固定的逐 token 2048→720 residual MLP，不改变 token 顺序和数量。"""

    input_dim = 2048
    hidden_dim = 1440
    output_dim = 720

    def __init__(self) -> None:
        super().__init__()
        self.input_norm = FP32RMSNorm(self.input_dim, eps=1e-5)
        self.skip_projection = nn.Linear(self.input_dim, self.output_dim)
        self.main_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.output_norm = FP32RMSNorm(self.output_dim, eps=1e-5)

    def forward(self, context: QwenContext) -> QwenContext:
        tokens = context.tokens
        mask = context.mask
        if tokens.ndim != 3 or tokens.shape[-1] != self.input_dim:
            raise ValueError(f"Qwen tokens 应为 [B,N,{self.input_dim}]，实际为 {tokens.shape}")
        if mask.shape != tokens.shape[:2] or mask.dtype != torch.bool:
            raise ValueError("Qwen context mask 必须是与 [B,N] 对齐的 bool Tensor")
        normalized = self.input_norm(tokens)
        adapted = self.output_norm(
            self.skip_projection(normalized) + self.main_projection(normalized)
        )
        adapted = adapted * mask.unsqueeze(-1).to(dtype=adapted.dtype)
        return QwenContext(tokens=adapted, mask=mask)
