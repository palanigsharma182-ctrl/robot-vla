"""单物体 Memory 条件候选；复用 V1 Flow 路径，不改变 canonical 模型。

这里仅提供规划输入，既不授权重观察，也不授权 contact/close/lift。
"""
from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch
from torch import nn

from robot_vla.model.policy import QwenVLAPolicy
from robot_vla.model.qwen_context import QwenContext
from robot_vla.precision.object_memory import (
    ObjectMemoryConfig, ObjectMemoryMode, ObjectMemorySafetyContext, ObjectState,
)

MEMORY_INPUT_KEY = "experiment_object_memory"
MEMORY_SCHEMA = "pregrasp-object-conditioning/12d/v1"


@dataclass(frozen=True)
class MemorySnapshot:
    """一次规划的不可变输入；物理位置不被误写成新测量。

features 顺序：xyz、xx/xy/xz/yy/yz/zz 协方差、age、confidence、observable。
位置除以 1 m，协方差除以配置最大标准差的平方，年龄除以最大未观测时间。
这是固定物理尺度，不从 validation/test 拟合统计。
"""

    episode_id: str
    timestamp_s: float
    last_observed_timestamp_s: float | None
    features: tuple[float, ...]
    available: bool
    reasons: tuple[str, ...]
    source_camera: str | None
    source_model_identity: str | None
    schema: str = MEMORY_SCHEMA

    def __post_init__(self):
        if not isinstance(self.features, tuple) or len(self.features) != 12:
            raise ValueError("Memory features 必须是不可变 12 维 tuple")
        if not all(math.isfinite(v) for v in self.features):
            raise ValueError("Memory features 必须有限")
        if self.available == bool(self.reasons):
            raise ValueError("Memory 可用性和拒绝原因冲突")
        if not self.available and any(self.features):
            raise ValueError("不可用 Memory 必须清零，并由独立 mask 屏蔽")


def snapshot_memory(
    state: ObjectState, config: ObjectMemoryConfig, safety: ObjectMemorySafetyContext,
    *, episode_id: str, timestamp_s: float,
) -> MemorySnapshot:
    """按实际规划时间重验状态；不修改 live Memory，不复活失效记录。"""
    if episode_id != state.episode_id:
        raise ValueError("Memory Episode 不匹配")
    if not math.isfinite(timestamp_s) or timestamp_s < state.state_timestamp_s:
        raise ValueError("规划时间非法或早于 Memory 状态")
    reasons = list(state.invalid_reasons) + list(safety.invalidation_reasons)
    if not state.valid or state.mode is not ObjectMemoryMode.FREE_STATIC:
        reasons.append("memory_not_valid_free_static")
    if state.source_camera != config.expected_source_camera:
        reasons.append("source_camera_mismatch")
    if state.source_model_identity != config.expected_source_model_identity:
        reasons.append("source_model_identity_mismatch")
    age = None
    if state.last_observed_timestamp_s is None:
        reasons.append("memory_uninitialized")
    else:
        age = timestamp_s - state.last_observed_timestamp_s
        if age < 0:
            raise ValueError("Memory 来自未来观测")
        if age > config.max_unobserved_age_s + 1e-12:
            reasons.append("memory_stale")
    covariance = None
    if state.position_base_m is None or state.covariance_base_m2 is None:
        reasons.append("geometry_missing")
    else:
        covariance = np.asarray(state.covariance_base_m2, dtype=np.float64).copy()
        covariance += np.eye(3) * config.covariance_growth_m2_per_s * (
            timestamp_s - state.state_timestamp_s
        )
        if np.sqrt(np.diag(covariance).max()) > config.max_position_std_m + 1e-12:
            reasons.append("memory_uncertain")
    reasons = tuple(dict.fromkeys(reasons))
    features = (0.0,) * 12
    if not reasons:
        covariance_values = covariance[np.triu_indices(3)] / config.max_position_std_m**2
        # observable_now 只能沿用同一时刻的证据，时间流逝不伪造当前可见。
        observable = state.observable_now and timestamp_s == state.state_timestamp_s
        features = tuple(float(x) for x in state.position_base_m) + tuple(covariance_values) + (
            age / config.max_unobserved_age_s, state.measurement_confidence, float(observable),
        )
    return MemorySnapshot(
        episode_id, timestamp_s, state.last_observed_timestamp_s, features, not reasons,
        reasons, state.source_camera, state.source_model_identity,
    )


@dataclass(frozen=True)
class MemoryBatch:
    """仅存在于一次显式模型调用的候选输入，不保存在 policy 的跨调用状态中。"""

    features: torch.Tensor
    available: torch.Tensor

    @classmethod
    def from_snapshots(cls, snapshots, *, device):
        if not snapshots:
            raise ValueError("Memory batch 不能为空")
        return cls(
            torch.tensor([s.features for s in snapshots], dtype=torch.float32, device=device),
            torch.tensor([[s.available] for s in snapshots], dtype=torch.bool, device=device),
        )


class ObjectMemoryEncoder(nn.Module):
    """将显式数值状态编码到已有 Expert context 维度；不访问 Qwen 或执行器。"""

    def __init__(self, output_dim=720):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(12, 128), nn.SiLU(), nn.Linear(128, output_dim))

    def forward(self, batch: MemoryBatch):
        features, available = batch.features, batch.available
        if features.ndim != 2 or features.shape[1] != 12:
            raise ValueError("Memory features 应为 [B,12]")
        if available.shape != (features.shape[0], 1) or available.dtype != torch.bool:
            raise ValueError("Memory available 应为 bool [B,1]")
        if available.device != features.device or not features.is_floating_point():
            raise ValueError("Memory dtype/device 不匹配")
        if not torch.isfinite(features).all():
            raise ValueError("Memory 输入不能有 NaN/Inf")
        clean = torch.where(available, features, torch.zeros_like(features))
        return self.net(clean.to(dtype=self.net[0].weight.dtype)).unsqueeze(1)


class MemoryConditionedPolicy(QwenVLAPolicy):
    """实验 V1 子类：训练、普通 Flow 和 RTC 共用同一 encode_context 插入点。

调用者显式传入 MemoryBatch；省略该键走原始路径，不保留上次 Memory。
候选必须使用独立 checkpoint 身份，不能用原 Stage1 loader 吞掉新增参数。
"""

    def __init__(self, context_encoder, expert, adapter=None):
        super().__init__(context_encoder, expert, adapter)
        self.memory_encoder = ObjectMemoryEncoder(self.expert.config.context_dim)
        self.adapter.requires_grad_(False)

    def encode_context(self, model_inputs):
        inputs = dict(model_inputs)
        memory = inputs.pop(MEMORY_INPUT_KEY, None)
        context = super().encode_context(inputs)
        return self.condition_context(context, memory)

    def condition_context(self, context, memory):
        """冻结 Qwen/Adapter 缓存与在线编码共用同一 Memory 拼接路径。"""
        if memory is None:
            return context
        if not isinstance(memory, MemoryBatch):
            raise TypeError("候选 Memory 必须是 MemoryBatch")
        token = self.memory_encoder(memory)
        if token.shape[0] != context.tokens.shape[0] or token.device != context.tokens.device:
            raise ValueError("Memory 与 Qwen context batch/device 不一致")
        if context.image_time_indices is not None:
            raise ValueError("首轮候选只支持 V1；不得混用 V2 时间索引")
        if not memory.available.any():
            return context
        return QwenContext(
            torch.cat((context.tokens, token.to(context.tokens.dtype)), dim=1),
            torch.cat((context.mask, memory.available), dim=1),
        )
