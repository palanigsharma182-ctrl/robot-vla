"""冻结 Qwen3.5 Backbone，并把最终多模态 hidden states 投影到 Expert 宽度。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from robot_vla.model.layers import FP32RMSNorm
from robot_vla.model.qwen_processor import (
    VLA_CONTEXT_VALID_MASK,
    VLA_IMAGE_TIME_INDICES,
)


@dataclass(frozen=True)
class QwenContext:
    tokens: torch.Tensor
    mask: torch.Tensor
    image_time_indices: torch.Tensor | None = None


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
        image_time_indices = model_inputs.get(VLA_IMAGE_TIME_INDICES)
        context_valid_mask = model_inputs.get(VLA_CONTEXT_VALID_MASK)
        qwen_inputs = {
            key: value
            for key, value in model_inputs.items()
            if key not in {VLA_IMAGE_TIME_INDICES, VLA_CONTEXT_VALID_MASK}
        }
        self.qwen.eval()
        outputs = self.qwen.model(
            **qwen_inputs,
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
        if context_valid_mask is not None:
            if context_valid_mask.shape != mask.shape:
                raise ValueError("V2 context_valid_mask shape 必须与 attention_mask 相同")
            mask = mask & context_valid_mask.bool()
        if image_time_indices is not None:
            if image_time_indices.shape != mask.shape:
                raise ValueError("V2 image_time_indices shape 必须与 attention_mask 相同")
            if torch.any(image_time_indices < -1):
                raise ValueError("V2 image_time_indices 不能小于 -1")
        return QwenContext(
            tokens=tokens,
            mask=mask,
            image_time_indices=image_time_indices,
        )


class FrozenQwenLayerContextEncoder(FrozenQwenContextEncoder):
    """冻结完整 Qwen 前向，并选择指定文本层之后的 hidden state。"""

    def __init__(self, qwen_for_conditional_generation: nn.Module, layer: int) -> None:
        super().__init__(qwen_for_conditional_generation)
        text_config = self.qwen.config.text_config
        layer_count = int(getattr(text_config, "num_hidden_layers", -1))
        if layer_count <= 0:
            raise ValueError("Qwen text config 必须提供有效 num_hidden_layers")
        if not 1 <= layer <= layer_count:
            raise ValueError(f"Qwen layer 应位于 [1,{layer_count}]，实际为 {layer}")
        self.layer = int(layer)
        self.layer_count = layer_count

    @torch.no_grad()
    def forward(self, model_inputs: dict[str, Any]) -> QwenContext:
        if "labels" in model_inputs:
            raise ValueError("Context Encoder 不接受 labels，也不能调用语言建模损失")
        required = {"input_ids", "attention_mask", "pixel_values", "image_grid_thw"}
        missing = required.difference(model_inputs)
        if missing:
            raise ValueError(f"Qwen model input 缺少字段: {sorted(missing)}")
        image_time_indices = model_inputs.get(VLA_IMAGE_TIME_INDICES)
        context_valid_mask = model_inputs.get(VLA_CONTEXT_VALID_MASK)
        qwen_inputs = {
            key: value
            for key, value in model_inputs.items()
            if key not in {VLA_IMAGE_TIME_INDICES, VLA_CONTEXT_VALID_MASK}
        }
        self.qwen.eval()
        outputs = self.qwen.model(
            **qwen_inputs,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        hidden_states = getattr(outputs, "hidden_states", None)
        expected_states = self.layer_count + 1
        if hidden_states is None or len(hidden_states) != expected_states:
            actual = None if hidden_states is None else len(hidden_states)
            raise RuntimeError(
                f"Qwen 应返回 embedding 加 {self.layer_count} 层，共 {expected_states} 个 "
                f"hidden states，实际为 {actual}"
            )
        # Hugging Face hidden_states[0] 是 embedding；hidden_states[k] 是第 k 层之后。
        tokens = hidden_states[self.layer]
        input_ids = model_inputs["input_ids"]
        if tokens.shape != (*input_ids.shape, self.context_dim):
            raise RuntimeError(
                f"Qwen Layer{self.layer} context 应为 [B,N,{self.context_dim}]，"
                f"实际为 {tuple(tokens.shape)}"
            )
        mask = model_inputs["attention_mask"].bool()
        if mask.shape != input_ids.shape:
            raise RuntimeError("Qwen context mask shape 必须与 input_ids 相同")
        if context_valid_mask is not None:
            if context_valid_mask.shape != mask.shape:
                raise ValueError("V2 context_valid_mask shape 必须与 attention_mask 相同")
            mask = mask & context_valid_mask.bool()
        if image_time_indices is not None:
            if image_time_indices.shape != mask.shape:
                raise ValueError("V2 image_time_indices shape 必须与 attention_mask 相同")
            if torch.any(image_time_indices < -1):
                raise ValueError("V2 image_time_indices 不能小于 -1")
        return QwenContext(
            tokens=tokens,
            mask=mask,
            image_time_indices=image_time_indices,
        )


class QwenVLAAdapter(nn.Module):
    """D010 固定的逐 token 2048→720 residual MLP，不改变 token 顺序和数量。"""

    input_dim = 2048
    hidden_dim = 1440
    output_dim = 720

    def __init__(self, *, history_length: int = 1) -> None:
        super().__init__()
        if history_length <= 0:
            raise ValueError("history_length 必须为正整数")
        self.history_length = history_length
        self.input_norm = FP32RMSNorm(self.input_dim, eps=1e-5)
        self.skip_projection = nn.Linear(self.input_dim, self.output_dim)
        self.main_projection = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )
        self.output_norm = FP32RMSNorm(self.output_dim, eps=1e-5)
        if history_length == 1:
            self.register_parameter("visual_time_embedding", None)
        else:
            self.visual_time_embedding = nn.Parameter(
                torch.empty(history_length, self.output_dim)
            )
            nn.init.normal_(self.visual_time_embedding, std=0.02)

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
        time_indices = context.image_time_indices
        if time_indices is not None:
            if self.visual_time_embedding is None:
                raise ValueError("V1 Adapter 不能消费带时间索引的 Observation V2 Context")
            if time_indices.shape != mask.shape:
                raise ValueError("image_time_indices 必须与 context mask 对齐")
            if torch.any(time_indices < -1):
                raise ValueError("image_time_indices 不能小于 -1")
            if torch.any(time_indices >= self.history_length):
                raise ValueError("image_time_indices 超出 Adapter history_length")
            visual = time_indices >= 0
            safe_indices = time_indices.clamp_min(0)
            time_embedding = self.visual_time_embedding[safe_indices]
            adapted = adapted + time_embedding * visual.unsqueeze(-1).to(
                dtype=adapted.dtype
            )
        adapted = adapted * mask.unsqueeze(-1).to(dtype=adapted.dtype)
        return QwenContext(
            tokens=adapted,
            mask=mask,
            image_time_indices=time_indices,
        )
