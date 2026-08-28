"""严格锁定 Qwen revision 和 qwen-vla-v0.1 架构的模型工厂。"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import nn

from robot_vla.contracts import QWEN_MODEL_ID, QWEN_REVISION
from robot_vla.model.expert import ExpertConfig, StandaloneActionExpert
from robot_vla.model.policy import QwenVLAPolicy
from robot_vla.model.qwen_context import (
    FrozenQwenContextEncoder,
    FrozenQwenLayerContextEncoder,
    QwenVLAAdapter,
)

QWEN_TEXT_LAYER_COUNT = 24


def validate_qwen_v01_architecture(qwen: nn.Module) -> None:
    """核对 D009 依赖的 Qwen3.5 文本/视觉结构，而不只检查 hidden size。"""

    config = getattr(qwen, "config", None)
    text = getattr(config, "text_config", None)
    vision = getattr(config, "vision_config", None)
    actual = {
        "model_type": getattr(config, "model_type", None),
        "text_hidden_size": getattr(text, "hidden_size", None),
        "text_layers": getattr(text, "num_hidden_layers", None),
        "full_attention_interval": getattr(text, "full_attention_interval", None),
        "vision_hidden_size": getattr(vision, "hidden_size", None),
        "vision_depth": getattr(vision, "depth", None),
        "vision_out_hidden_size": getattr(vision, "out_hidden_size", None),
    }
    expected = {
        "model_type": "qwen3_5",
        "text_hidden_size": 2048,
        "text_layers": 24,
        "full_attention_interval": 4,
        "vision_hidden_size": 1024,
        "vision_depth": 24,
        "vision_out_hidden_size": 2048,
    }
    if actual != expected:
        raise ValueError(f"Qwen 架构与 qwen-vla-v0.1 不兼容：期望 {expected}，实际 {actual}")
    architectures = getattr(config, "architectures", None)
    if architectures is not None and "Qwen3_5ForConditionalGeneration" not in architectures:
        raise ValueError("Qwen config architectures 不包含 Qwen3_5ForConditionalGeneration")
    if not hasattr(qwen, "model") or not hasattr(qwen, "lm_head"):
        raise ValueError("Qwen conditional model 必须暴露 model 和 lm_head")


def load_frozen_qwen_v01(
    *,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    device: str | torch.device | None = None,
    hf_endpoint: str = "https://hf-mirror.com",
) -> nn.Module:
    """显式调用时才加载权重；默认使用 BF16、安全 Tensor 和固定 revision。"""

    os.environ.setdefault("HF_ENDPOINT", hf_endpoint)
    try:
        from transformers import Qwen3_5ForConditionalGeneration
    except ImportError as exc:
        raise ImportError("加载 qwen-vla-v0.1 需要 transformers qwen-vla extra") from exc

    kwargs: dict[str, Any] = {
        "revision": QWEN_REVISION,
        "trust_remote_code": False,
        "cache_dir": cache_dir,
        "local_files_only": local_files_only,
        "use_safetensors": True,
        "weights_only": True,
        "low_cpu_mem_usage": True,
        "dtype": torch.bfloat16,
    }
    if device is not None:
        resolved_device = torch.device(device)
        if resolved_device.type not in {"cpu", "cuda"}:
            raise ValueError("Qwen 工厂只支持 cpu/cuda device")
        if resolved_device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求在 CUDA 加载 Qwen，但当前 PyTorch 无可用 CUDA")
        kwargs["device_map"] = {"": str(resolved_device)}
    qwen = Qwen3_5ForConditionalGeneration.from_pretrained(QWEN_MODEL_ID, **kwargs)
    validate_qwen_v01_architecture(qwen)
    qwen.requires_grad_(False)
    qwen.eval()
    return qwen


def build_qwen_vla_policy(
    qwen: nn.Module,
    *,
    expert_config: ExpertConfig | None = None,
    context_layer: int = QWEN_TEXT_LAYER_COUNT,
) -> QwenVLAPolicy:
    validate_qwen_v01_architecture(qwen)
    if not 1 <= context_layer <= QWEN_TEXT_LAYER_COUNT:
        raise ValueError(
            f"Qwen context layer 应位于 [1,{QWEN_TEXT_LAYER_COUNT}]，实际为 {context_layer}"
        )
    context_encoder = (
        FrozenQwenContextEncoder(qwen)
        if context_layer == QWEN_TEXT_LAYER_COUNT
        else FrozenQwenLayerContextEncoder(qwen, context_layer)
    )
    return QwenVLAPolicy(
        context_encoder,
        StandaloneActionExpert(expert_config),
        QwenVLAAdapter(),
    )


def load_qwen_vla_policy(
    *,
    cache_dir: str | None = None,
    local_files_only: bool = False,
    device: str | torch.device | None = None,
    hf_endpoint: str = "https://hf-mirror.com",
    context_layer: int = QWEN_TEXT_LAYER_COUNT,
) -> QwenVLAPolicy:
    qwen = load_frozen_qwen_v01(
        cache_dir=cache_dir,
        local_files_only=local_files_only,
        device=device,
        hf_endpoint=hf_endpoint,
    )
    policy = build_qwen_vla_policy(qwen, context_layer=context_layer)
    if device is not None:
        policy.to(torch.device(device))
    return policy
