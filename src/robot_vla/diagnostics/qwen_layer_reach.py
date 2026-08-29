"""Reach 诊断使用的冻结 Qwen 指定文本层上下文。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from robot_vla.model.expert import AttentionKV, StandaloneActionExpert
from robot_vla.model.qwen_context import (
    FrozenQwenContextEncoder,
    FrozenQwenLayerContextEncoder,
    QwenContext,
    QwenVLAAdapter,
)

QWEN_LAYER_REACH_EXPERIMENT_FORMAT = "robot-vla-qwen-layer-reach/v1"
QWEN_LAYER_REACH_CHECKPOINT_FORMAT = "robot-vla-qwen-layer-reach-checkpoint/v1"
QWEN_TEXT_LAYER_COUNT = 24
QWEN_LAYER12 = 12
QWEN_LAYER24 = 24


@dataclass(frozen=True)
class QwenLayerPairContext:
    layer12_tokens: torch.Tensor
    layer24_tokens: torch.Tensor
    mask: torch.Tensor


@dataclass(frozen=True)
class QwenSemanticKeyGeometryValueContext:
    """Layer24 语义 Key 与同 token 位置 Layer12 几何 Value。"""

    key_tokens: torch.Tensor
    value_tokens: torch.Tensor
    mask: torch.Tensor

    @property
    def tokens(self) -> torch.Tensor:
        """兼容 Policy 的 batch/token 接口；真正的 K/V 由诊断 Expert 分别读取。"""

        return self.key_tokens


class FrozenQwenLayerPairContextEncoder(FrozenQwenContextEncoder):
    """一次冻结 Qwen 前向同时返回 Layer12 与 Layer24 token。"""

    def __init__(self, qwen_for_conditional_generation: nn.Module) -> None:
        super().__init__(qwen_for_conditional_generation)
        text_config = self.qwen.config.text_config
        layer_count = int(getattr(text_config, "num_hidden_layers", -1))
        if layer_count != QWEN_TEXT_LAYER_COUNT:
            raise ValueError(
                f"Qwen 文本层数应为 {QWEN_TEXT_LAYER_COUNT}，实际为 {layer_count}"
            )
        self.layer_count = layer_count

    @torch.no_grad()
    def forward(self, model_inputs: dict[str, Any]) -> QwenLayerPairContext:
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
        expected_states = self.layer_count + 1
        if hidden_states is None or len(hidden_states) != expected_states:
            actual = None if hidden_states is None else len(hidden_states)
            raise RuntimeError(
                f"Qwen 应返回 embedding 加 {self.layer_count} 层，共 {expected_states} 个 "
                f"hidden states，实际为 {actual}"
            )
        input_ids = model_inputs["input_ids"]
        expected_shape = (*input_ids.shape, self.context_dim)
        layer12_tokens = hidden_states[QWEN_LAYER12]
        layer24_tokens = hidden_states[QWEN_LAYER24]
        if layer12_tokens.shape != expected_shape or layer24_tokens.shape != expected_shape:
            raise RuntimeError(
                "Qwen Layer12/24 context shape 无效："
                f"期望 {expected_shape}，实际 "
                f"{tuple(layer12_tokens.shape)}/{tuple(layer24_tokens.shape)}"
            )
        mask = model_inputs["attention_mask"].bool()
        if mask.shape != input_ids.shape:
            raise RuntimeError("Qwen context mask shape 必须与 input_ids 相同")
        return QwenLayerPairContext(
            layer12_tokens=layer12_tokens,
            layer24_tokens=layer24_tokens,
            mask=mask,
        )


class QwenLayerFusionAdapter(nn.Module):
    """Layer12/24 独立投影到 720 维，再用可学习 softmax 标量权重融合。"""

    output_dim = QwenVLAAdapter.output_dim

    def __init__(self) -> None:
        super().__init__()
        # Layer12 分支先构造，保证同 seed 下与 Layer12-only Adapter 初始化一致。
        self.layer12_projection = QwenVLAAdapter()
        self.layer24_projection = QwenVLAAdapter()
        self.fusion_logits = nn.Parameter(torch.zeros(2, dtype=torch.float32))

    def normalized_weights(self) -> torch.Tensor:
        return torch.softmax(self.fusion_logits.float(), dim=0)

    def forward(self, context: QwenLayerPairContext) -> QwenContext:
        layer12 = self.layer12_projection(
            QwenContext(tokens=context.layer12_tokens, mask=context.mask)
        )
        layer24 = self.layer24_projection(
            QwenContext(tokens=context.layer24_tokens, mask=context.mask)
        )
        weights = self.normalized_weights().to(dtype=layer12.tokens.dtype)
        tokens = weights[0] * layer12.tokens + weights[1] * layer24.tokens
        tokens = tokens * context.mask.unsqueeze(-1).to(dtype=tokens.dtype)
        return QwenContext(tokens=tokens, mask=context.mask)


class QwenSemanticKeyGeometryValueAdapter(nn.Module):
    """Layer24 投影为语义 Key，Layer12 同位置投影为精细几何 Value。"""

    output_dim = QwenVLAAdapter.output_dim

    def __init__(self) -> None:
        super().__init__()
        # Value 分支先构造，保证同 seed 下与 Layer12-only Adapter 初始化一致。
        self.value_projection = QwenVLAAdapter()
        self.key_projection = QwenVLAAdapter()

    def forward(
        self, context: QwenLayerPairContext
    ) -> QwenSemanticKeyGeometryValueContext:
        if context.layer12_tokens.shape != context.layer24_tokens.shape:
            raise ValueError("Layer12 Value 与 Layer24 Key 必须保持 token 位置一一对齐")
        if context.mask.shape != context.layer12_tokens.shape[:2]:
            raise ValueError("Layer12/24 mask 必须与 token 序列对齐")
        value = self.value_projection(
            QwenContext(tokens=context.layer12_tokens, mask=context.mask)
        )
        key = self.key_projection(
            QwenContext(tokens=context.layer24_tokens, mask=context.mask)
        )
        return QwenSemanticKeyGeometryValueContext(
            key_tokens=key.tokens,
            value_tokens=value.tokens,
            mask=context.mask,
        )


class SemanticKeyGeometryValueActionExpert(StandaloneActionExpert):
    """用 Layer24 寻址，并从相同 token 索引读取 Layer12 Value 的诊断 Expert。"""

    def prepare_context_kv(
        self, context: QwenSemanticKeyGeometryValueContext
    ) -> tuple[AttentionKV, ...]:
        self._validate_key_value_context(context)
        batch_size, source_length, _ = context.key_tokens.shape
        projected = []
        for block in self.blocks:
            if not block.cross_attention:
                continue
            attention = block.attention
            key = attention.k_projection(context.key_tokens).view(
                batch_size,
                source_length,
                self.config.num_key_value_heads,
                self.config.head_dim,
            )
            value = attention.v_projection(context.value_tokens).view_as(key)
            projected.append(
                AttentionKV(
                    key=key.transpose(1, 2),
                    value=value.transpose(1, 2),
                    mask=context.mask,
                )
            )
        return tuple(projected)

    def forward(
        self,
        context: QwenSemanticKeyGeometryValueContext,
        normalized_proprio: torch.Tensor,
        noisy_action: torch.Tensor,
        flow_time: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        context_kv: tuple[AttentionKV, ...] | None = None,
    ) -> torch.Tensor:
        if context_kv is None:
            context_kv = self.prepare_context_kv(context)
        else:
            self._validate_key_value_context(context)
        return super().forward(
            context,
            normalized_proprio,
            noisy_action,
            flow_time,
            action_mask,
            context_kv=context_kv,
        )

    def _validate_key_value_context(
        self, context: QwenSemanticKeyGeometryValueContext
    ) -> None:
        if not isinstance(context, QwenSemanticKeyGeometryValueContext):
            raise TypeError("语义 Key/几何 Value Expert 需要分离的 Layer24/12 Context")
        expected_width = self.config.context_dim
        if (
            context.key_tokens.ndim != 3
            or context.value_tokens.shape != context.key_tokens.shape
            or context.key_tokens.shape[-1] != expected_width
        ):
            raise ValueError(
                f"Layer24 Key 与 Layer12 Value 应为相同的 [B,N,{expected_width}] shape"
            )
        if (
            context.mask.shape != context.key_tokens.shape[:2]
            or context.mask.dtype != torch.bool
        ):
            raise ValueError("语义 Key/几何 Value mask 必须是与 [B,N] 对齐的 bool Tensor")
        if not torch.all(context.mask.any(dim=1)):
            raise ValueError("每个样本必须至少包含一个有效语义 Key/几何 Value Token")


__all__ = [
    "QWEN_LAYER12",
    "QWEN_LAYER24",
    "QWEN_LAYER_REACH_CHECKPOINT_FORMAT",
    "QWEN_LAYER_REACH_EXPERIMENT_FORMAT",
    "QWEN_TEXT_LAYER_COUNT",
    "FrozenQwenLayerContextEncoder",
    "FrozenQwenLayerPairContextEncoder",
    "QwenLayerFusionAdapter",
    "QwenLayerPairContext",
    "QwenSemanticKeyGeometryValueAdapter",
    "QwenSemanticKeyGeometryValueContext",
    "SemanticKeyGeometryValueActionExpert",
]
