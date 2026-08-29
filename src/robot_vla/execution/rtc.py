"""RTC 推理策略、Eq.(5) soft mask 与可审计诊断契约。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

RTC_SCHEDULE = "rtc-eq5-soft-mask"


class ChunkInferenceStrategy(str, Enum):
    NEWEST_ONLY = "newest-only"
    TEMPORAL_ENSEMBLE = "temporal-ensemble"
    RTC = "rtc"


def resolve_inference_strategy(
    value: str | ChunkInferenceStrategy | None,
    *,
    legacy_temporal_ensemble_enabled: bool | None = None,
) -> ChunkInferenceStrategy:
    """兼容旧 bool 配置，同时把新实验统一成三种显式策略。"""

    if value is None:
        if legacy_temporal_ensemble_enabled is None:
            return ChunkInferenceStrategy.TEMPORAL_ENSEMBLE
        return (
            ChunkInferenceStrategy.TEMPORAL_ENSEMBLE
            if legacy_temporal_ensemble_enabled
            else ChunkInferenceStrategy.NEWEST_ONLY
        )
    strategy = ChunkInferenceStrategy(value)
    if legacy_temporal_ensemble_enabled is not None:
        legacy_strategy = (
            ChunkInferenceStrategy.TEMPORAL_ENSEMBLE
            if legacy_temporal_ensemble_enabled
            else ChunkInferenceStrategy.NEWEST_ONLY
        )
        if strategy != legacy_strategy:
            raise ValueError("inference_strategy 与旧 temporal_ensemble_enabled 配置冲突")
    return strategy


@dataclass(frozen=True)
class RTCConfig:
    """同步第一版 RTC：不模拟推理延迟 ``d=0``，执行 horizon ``s=4``。"""

    execution_horizon: int = 4
    max_guidance_weight: float = 10.0
    schedule: str = RTC_SCHEDULE

    def __post_init__(self) -> None:
        if self.execution_horizon <= 0:
            raise ValueError("rtc_execution_horizon 必须为正整数")
        if not math.isfinite(self.max_guidance_weight) or self.max_guidance_weight <= 0:
            raise ValueError("rtc_max_guidance_weight 必须是有限正数")
        if self.schedule != RTC_SCHEDULE:
            raise ValueError(f"首版 RTC schedule 必须为 {RTC_SCHEDULE}")

    def slot_weights(self, action_horizon: int) -> np.ndarray:
        """返回论文 Eq.(5) 权重；本同步版本固定 ``d=0``、``s=execution_horizon``。"""

        horizon = int(action_horizon)
        d = 0
        s = self.execution_horizon
        if horizon <= 0 or s >= horizon:
            raise ValueError("RTC 要求 action_horizon > execution_horizon")
        overlap_end = horizon - s
        denominator = overlap_end - d + 1
        weights = np.zeros(horizon, dtype=np.float32)
        for index in range(d, overlap_end):
            c_i = (overlap_end - index) / denominator
            weights[index] = c_i * math.expm1(c_i) / math.expm1(1.0)
        return weights


@dataclass(frozen=True)
class RTCTrace:
    rtc_enabled: bool
    rtc_guidance_weight: float
    rtc_execution_horizon: int
    rtc_schedule: str
    previous_chunk_available: bool
    overlap_length: int
    slot_weights: tuple[float, ...]
    denoising_guidance_coefficients: tuple[float, ...]
    raw_mean_abs_disagreement: float | None = None
    raw_max_abs_disagreement: float | None = None
    prefix_mean_abs_disagreement: float | None = None
    prefix_max_abs_disagreement: float | None = None
    prefix_mean_abs_correction: float | None = None
    prefix_max_abs_correction: float | None = None
    future_mean_abs_correction: float | None = None
    future_max_abs_correction: float | None = None

    def __post_init__(self) -> None:
        if not self.rtc_enabled:
            raise ValueError("RTCTrace 只记录启用 RTC 的 Replan")
        if self.overlap_length < 0:
            raise ValueError("RTC overlap_length 不能为负数")
        if self.previous_chunk_available != (self.overlap_length > 0):
            raise ValueError("RTC previous_chunk_available 与 overlap_length 不一致")
        if any(
            not math.isfinite(value) or value < 0
            for value in (*self.slot_weights, *self.denoising_guidance_coefficients)
        ):
            raise ValueError("RTC slot/guidance 权重必须是有限非负数")
        values = (
            self.raw_mean_abs_disagreement,
            self.raw_max_abs_disagreement,
            self.prefix_mean_abs_disagreement,
            self.prefix_max_abs_disagreement,
            self.prefix_mean_abs_correction,
            self.prefix_max_abs_correction,
            self.future_mean_abs_correction,
            self.future_max_abs_correction,
        )
        if any(value is not None and (not math.isfinite(value) or value < 0) for value in values):
            raise ValueError("RTC disagreement/correction 必须是有限非负数")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def build_rtc_trace(
    config: RTCConfig,
    *,
    action_horizon: int,
    previous_overlap: np.ndarray | None,
    raw_action: np.ndarray,
    guided_action: np.ndarray,
    denoising_guidance_coefficients: tuple[float, ...] = (),
) -> RTCTrace:
    """在 normalized model action space 计算 paired raw/RTC 诊断。"""

    raw = np.asarray(raw_action, dtype=np.float32)
    guided = np.asarray(guided_action, dtype=np.float32)
    if raw.ndim != 2 or guided.ndim != 2:
        raise ValueError("RTC raw/guided Action Chunk 应为 [H,A]")
    expected = (action_horizon, raw.shape[-1])
    if raw.shape != expected or guided.shape != expected:
        raise ValueError("RTC raw/guided Action Chunk shape 无效")
    if not np.isfinite(raw).all() or not np.isfinite(guided).all():
        raise ValueError("RTC raw/guided Action Chunk 必须有限")
    weights = config.slot_weights(action_horizon)
    if previous_overlap is None:
        return RTCTrace(
            rtc_enabled=True,
            rtc_guidance_weight=config.max_guidance_weight,
            rtc_execution_horizon=config.execution_horizon,
            rtc_schedule=config.schedule,
            previous_chunk_available=False,
            overlap_length=0,
            slot_weights=tuple(float(value) for value in weights),
            denoising_guidance_coefficients=denoising_guidance_coefficients,
        )

    previous = np.asarray(previous_overlap, dtype=np.float32)
    if previous.ndim != 2 or previous.shape[1] != raw.shape[1]:
        raise ValueError("RTC previous overlap 应为 [L,A]")
    overlap_length = previous.shape[0]
    expected_overlap = action_horizon - config.execution_horizon
    if overlap_length != expected_overlap or not np.isfinite(previous).all():
        raise ValueError(
            f"RTC previous overlap 应为 [{expected_overlap},{raw.shape[1]}] 有限数组"
        )
    prefix_length = min(config.execution_horizon, overlap_length)
    raw_disagreement = np.abs(raw[:overlap_length] - previous)
    guided_disagreement = np.abs(guided[:prefix_length] - previous[:prefix_length])
    correction = np.abs(guided - raw)
    future = correction[prefix_length:]
    return RTCTrace(
        rtc_enabled=True,
        rtc_guidance_weight=config.max_guidance_weight,
        rtc_execution_horizon=config.execution_horizon,
        rtc_schedule=config.schedule,
        previous_chunk_available=True,
        overlap_length=overlap_length,
        slot_weights=tuple(float(value) for value in weights),
        denoising_guidance_coefficients=denoising_guidance_coefficients,
        raw_mean_abs_disagreement=float(np.mean(raw_disagreement)),
        raw_max_abs_disagreement=float(np.max(raw_disagreement)),
        prefix_mean_abs_disagreement=float(np.mean(guided_disagreement)),
        prefix_max_abs_disagreement=float(np.max(guided_disagreement)),
        prefix_mean_abs_correction=float(np.mean(correction[:prefix_length])),
        prefix_max_abs_correction=float(np.max(correction[:prefix_length])),
        future_mean_abs_correction=float(np.mean(future)),
        future_max_abs_correction=float(np.max(future)),
    )


__all__ = [
    "RTC_SCHEDULE",
    "ChunkInferenceStrategy",
    "RTCConfig",
    "RTCTrace",
    "build_rtc_trace",
    "resolve_inference_strategy",
]
