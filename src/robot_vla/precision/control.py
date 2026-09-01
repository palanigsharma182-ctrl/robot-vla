"""三头 U-Net 与显式几何之间的 fail-closed 控制仲裁。"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from robot_vla.precision.contracts import (
    PrecisionControlMode,
    PrecisionMotionSpec,
)


@dataclass(frozen=True)
class PrecisionControlConfig:
    """第一版高频层的保守门禁；正式阈值必须在独立 calibration split 冻结。"""

    mode: PrecisionControlMode = PrecisionControlMode.SHADOW
    motion_spec: PrecisionMotionSpec = field(default_factory=PrecisionMotionSpec)
    min_visibility_probability: float = 0.95
    min_projection_validity_probability: float = 0.99
    max_heatmap_entropy: float = 0.65
    max_keypoint_sigma_px: float = 0.75
    max_motion_sigma: tuple[float, ...] = (
        0.50e-3,
        0.50e-3,
        0.40e-3,
        math.radians(0.10),
    )

    def __post_init__(self) -> None:
        if not isinstance(self.mode, PrecisionControlMode):
            raise ValueError("mode 必须是 PrecisionControlMode")
        for name, value in (
            ("min_visibility_probability", self.min_visibility_probability),
            (
                "min_projection_validity_probability",
                self.min_projection_validity_probability,
            ),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须位于 [0,1]")
        if (
            not math.isfinite(self.max_heatmap_entropy)
            or not 0.0 <= self.max_heatmap_entropy <= 1.0
        ):
            raise ValueError("max_heatmap_entropy 必须位于 [0,1]")
        if not math.isfinite(self.max_keypoint_sigma_px) or self.max_keypoint_sigma_px <= 0.0:
            raise ValueError("max_keypoint_sigma_px 必须是有限正数")
        if len(self.max_motion_sigma) != self.motion_spec.motion_dim:
            raise ValueError("max_motion_sigma 与 motion_dim 不一致")
        if any(not math.isfinite(value) or value <= 0.0 for value in self.max_motion_sigma):
            raise ValueError("max_motion_sigma 必须全部为有限正数")


@dataclass(frozen=True)
class PrecisionConfidenceEvidence:
    """已经从网络输出和几何残差换算到可审计单位的 confidence 证据。"""

    visibility_probability: np.ndarray
    projection_validity_probability: float
    heatmap_entropy: np.ndarray
    keypoint_sigma_px: np.ndarray
    motion_sigma: np.ndarray
    required_keypoints: np.ndarray


@dataclass(frozen=True)
class PrecisionControlDecision:
    should_execute: bool
    mode: PrecisionControlMode
    command_delta: np.ndarray
    geometric_delta: np.ndarray
    learned_residual: np.ndarray
    bounded_residual: np.ndarray
    gate_failures: tuple[str, ...]
    diagnostic_warnings: tuple[str, ...]
    geometry_was_clipped: bool
    residual_was_clipped: bool


def _float32_vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.float32:
        raise ValueError(f"{name} 必须是 float32 {shape}")
    return array


def _probability_vector(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = _float32_vector(value, shape, name)
    if not np.isfinite(array).all() or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} 必须是 [0,1] 内的有限概率")
    return array


def decide_precision_command(
    geometric_delta: np.ndarray,
    learned_residual: np.ndarray,
    confidence: PrecisionConfidenceEvidence,
    config: PrecisionControlConfig,
) -> PrecisionControlDecision:
    """根据显式门禁生成一个 Tick 的命令。

    ``shadow`` 模式下 learned residual 永远不影响 ``command_delta``；它仍会进入返回值，
    供离线比较和校准。任何 confidence/几何非有限值都返回零命令，而不是尝试修补。
    """

    motion_dim = config.motion_spec.motion_dim
    geometry = _float32_vector(geometric_delta, (motion_dim,), "geometric_delta")
    residual = _float32_vector(learned_residual, (motion_dim,), "learned_residual")

    visibility = np.asarray(confidence.visibility_probability)
    entropy = np.asarray(confidence.heatmap_entropy)
    keypoint_sigma = np.asarray(confidence.keypoint_sigma_px)
    required = np.asarray(confidence.required_keypoints)
    if visibility.ndim != 1:
        raise ValueError("visibility_probability 必须是一维")
    keypoint_count = int(visibility.shape[0])
    visibility = _probability_vector(
        visibility,
        (keypoint_count,),
        "visibility_probability",
    )
    entropy = _float32_vector(entropy, (keypoint_count,), "heatmap_entropy")
    keypoint_sigma = _float32_vector(
        keypoint_sigma,
        (keypoint_count, 2),
        "keypoint_sigma_px",
    )
    if required.shape != (keypoint_count,) or required.dtype != np.bool_:
        raise ValueError("required_keypoints 必须是 bool [K]")
    if not bool(required.any()):
        raise ValueError("至少需要一个 required keypoint")
    motion_sigma = _float32_vector(
        confidence.motion_sigma,
        (motion_dim,),
        "motion_sigma",
    )
    projection_probability = float(confidence.projection_validity_probability)
    if not math.isfinite(projection_probability) or not 0.0 <= projection_probability <= 1.0:
        raise ValueError("projection_validity_probability 必须位于 [0,1]")

    failures: list[str] = []
    warnings: list[str] = []
    if not np.isfinite(geometry).all():
        failures.append("nonfinite_geometry")
    if not np.isfinite(residual).all():
        if config.mode == PrecisionControlMode.SHADOW:
            warnings.append("nonfinite_shadow_motion_head")
        else:
            failures.append("nonfinite_motion_head")
    if not np.isfinite(entropy).all() or np.any(entropy[required] < 0.0) or np.any(
        entropy[required] > 1.0
    ):
        failures.append("invalid_heatmap_entropy")
    elif np.any(entropy[required] > config.max_heatmap_entropy):
        failures.append("heatmap_entropy")
    if np.any(visibility[required] < config.min_visibility_probability):
        failures.append("keypoint_visibility")
    if projection_probability < config.min_projection_validity_probability:
        failures.append("projection_validity")
    if not np.isfinite(keypoint_sigma).all() or np.any(keypoint_sigma[required] < 0.0):
        failures.append("invalid_keypoint_uncertainty")
    elif np.any(keypoint_sigma[required] > config.max_keypoint_sigma_px):
        failures.append("keypoint_uncertainty")
    max_motion_sigma = np.asarray(config.max_motion_sigma, dtype=np.float32)
    if not np.isfinite(motion_sigma).all() or np.any(motion_sigma < 0.0):
        if config.mode == PrecisionControlMode.SHADOW:
            warnings.append("invalid_shadow_motion_uncertainty")
        else:
            failures.append("invalid_motion_uncertainty")
    elif np.any(motion_sigma > max_motion_sigma):
        if config.mode == PrecisionControlMode.SHADOW:
            warnings.append("shadow_motion_uncertainty")
        else:
            failures.append("motion_uncertainty")

    zero = np.zeros(motion_dim, dtype=np.float32)
    if failures:
        return PrecisionControlDecision(
            should_execute=False,
            mode=config.mode,
            command_delta=zero,
            geometric_delta=geometry.copy(),
            learned_residual=residual.copy(),
            bounded_residual=zero.copy(),
            gate_failures=tuple(failures),
            diagnostic_warnings=tuple(warnings),
            geometry_was_clipped=False,
            residual_was_clipped=False,
        )

    bounded_geometry = config.motion_spec.clip_step(geometry)
    geometry_was_clipped = not np.array_equal(bounded_geometry, geometry)
    if config.mode == PrecisionControlMode.SHADOW:
        if np.isfinite(residual).all():
            bounded_residual = config.motion_spec.clip_residual(residual)
            residual_was_clipped = not np.array_equal(bounded_residual, residual)
        else:
            bounded_residual = zero.copy()
            residual_was_clipped = False
        command = bounded_geometry
    else:
        bounded_residual = config.motion_spec.clip_residual(residual)
        residual_was_clipped = not np.array_equal(bounded_residual, residual)
        command = config.motion_spec.clip_step(bounded_geometry + bounded_residual)
    return PrecisionControlDecision(
        should_execute=True,
        mode=config.mode,
        command_delta=command.astype(np.float32, copy=True),
        geometric_delta=geometry.copy(),
        learned_residual=residual.copy(),
        bounded_residual=bounded_residual.astype(np.float32, copy=True),
        gate_failures=(),
        diagnostic_warnings=tuple(warnings),
        geometry_was_clipped=geometry_was_clipped,
        residual_was_clipped=residual_was_clipped,
    )


__all__ = [
    "PrecisionConfidenceEvidence",
    "PrecisionControlConfig",
    "PrecisionControlDecision",
    "decide_precision_command",
]
