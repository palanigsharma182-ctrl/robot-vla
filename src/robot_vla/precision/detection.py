"""Precision U-Net 解码结果到部署侧 wrist keypoint 契约的适配器。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from robot_vla.executive.estimation import WristKeypointDetection

if TYPE_CHECKING:
    from robot_vla.precision.model import DecodedPrecisionPrediction


PRECISION_TRACK_CONFIDENCE_SEMANTICS = (
    "min-keypoint-visibility-and-projection-validity/v1"
)


def _numpy(value: Any, name: str) -> np.ndarray:
    candidate = value
    if hasattr(candidate, "detach"):
        candidate = candidate.detach()
    if hasattr(candidate, "cpu"):
        candidate = candidate.cpu()
    if hasattr(candidate, "numpy"):
        candidate = candidate.numpy()
    array = np.asarray(candidate)
    if (
        not np.issubdtype(array.dtype, np.number)
        or np.iscomplexobj(array)
        or not np.isfinite(array).all()
    ):
        raise ValueError(f"{name} 必须是有限数值 array/tensor")
    return array.astype(np.float64, copy=False)


def _probability_array(value: Any, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = _numpy(value, name)
    if array.shape != shape or np.any(array < 0.0) or np.any(array > 1.0):
        raise ValueError(f"{name} 必须是 [0,1] 内的有限 {shape}")
    return array


@dataclass(frozen=True)
class PrecisionKeypointEvidence:
    """保留合并 track confidence 前后的可审计网络证据。"""

    name: str
    normalized_uv: tuple[float, float]
    visibility_probability: float
    projection_validity_probability: float
    peak_probability: float
    normalized_entropy: float
    sigma_px: tuple[float, float]
    track_confidence: float

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("Precision keypoint evidence name 不能为空")
        if len(self.normalized_uv) != 2 or any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in self.normalized_uv
        ):
            raise ValueError("Precision normalized_uv 必须位于 [0,1]")
        for value, name in (
            (self.visibility_probability, "visibility_probability"),
            (self.projection_validity_probability, "projection_validity_probability"),
            (self.peak_probability, "peak_probability"),
            (self.normalized_entropy, "normalized_entropy"),
            (self.track_confidence, "track_confidence"),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} 必须是 [0,1] 内的有限概率")
        if len(self.sigma_px) != 2 or any(
            not math.isfinite(value) or value < 0.0 for value in self.sigma_px
        ):
            raise ValueError("sigma_px 必须是两个有限非负数")
        expected_confidence = min(
            self.visibility_probability,
            self.projection_validity_probability,
        )
        if not math.isclose(
            self.track_confidence,
            expected_confidence,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("track_confidence 与冻结 confidence semantics 不一致")

    def to_dict(self) -> dict[str, object]:
        return {
            "name": self.name,
            "normalized_uv": list(self.normalized_uv),
            "visibility_probability": self.visibility_probability,
            "projection_validity_probability": self.projection_validity_probability,
            "peak_probability": self.peak_probability,
            "normalized_entropy": self.normalized_entropy,
            "sigma_px": list(self.sigma_px),
            "track_confidence": self.track_confidence,
        }


@dataclass(frozen=True)
class PrecisionDetectionAdapterResult:
    detection: WristKeypointDetection
    object_evidence: PrecisionKeypointEvidence
    goal_evidence: PrecisionKeypointEvidence
    confidence_semantics: str = PRECISION_TRACK_CONFIDENCE_SEMANTICS

    def __post_init__(self) -> None:
        if self.confidence_semantics != PRECISION_TRACK_CONFIDENCE_SEMANTICS:
            raise ValueError("Precision track confidence semantics 漂移")
        if self.object_evidence.name == self.goal_evidence.name:
            raise ValueError("object/goal evidence name 不能相同")
        if not math.isclose(
            self.detection.object_confidence,
            self.object_evidence.track_confidence,
            rel_tol=0.0,
            abs_tol=1e-9,
        ) or not math.isclose(
            self.detection.goal_confidence,
            self.goal_evidence.track_confidence,
            rel_tol=0.0,
            abs_tol=1e-9,
        ):
            raise ValueError("Wrist detection confidence 与 Precision evidence 不一致")

    def to_dict(self) -> dict[str, object]:
        return {
            "detection_timestamp_s": self.detection.timestamp_s,
            "object_evidence": self.object_evidence.to_dict(),
            "goal_evidence": self.goal_evidence.to_dict(),
            "confidence_semantics": self.confidence_semantics,
        }


def precision_prediction_to_wrist_detection(
    prediction: DecodedPrecisionPrediction,
    *,
    keypoint_names: tuple[str, ...],
    timestamp_s: float,
    batch_index: int = 0,
    object_keypoint_name: str = "object_center",
    goal_keypoint_name: str = "goal_center",
) -> PrecisionDetectionAdapterResult:
    """提取 object/goal center，并保留 visibility/entropy/sigma 诊断。

    Track confidence 固定为 keypoint visibility 与全局 projection validity 的最小值。Heatmap
    peak 依赖输出分辨率，sigma/entropy 也需要独立标定，因此三者只进入诊断，不在这里用经验权重
    混成一个不可解释分数。
    """

    if not keypoint_names or any(
        not isinstance(name, str) or not name.strip() for name in keypoint_names
    ):
        raise ValueError("keypoint_names 必须包含非空名称")
    if len(set(keypoint_names)) != len(keypoint_names):
        raise ValueError("keypoint_names 不能重复")
    if object_keypoint_name == goal_keypoint_name:
        raise ValueError("object/goal keypoint name 不能相同")
    try:
        object_index = keypoint_names.index(object_keypoint_name)
        goal_index = keypoint_names.index(goal_keypoint_name)
    except ValueError as error:
        raise ValueError("Precision prediction 缺少 object_center 或 goal_center") from error
    if (
        not isinstance(batch_index, int)
        or isinstance(batch_index, bool)
        or batch_index < 0
    ):
        raise ValueError("batch_index 必须是非负整数")
    if isinstance(timestamp_s, bool) or not math.isfinite(timestamp_s) or timestamp_s < 0.0:
        raise ValueError("timestamp_s 必须是有限非负数")

    normalized_uv = _numpy(prediction.keypoints.normalized_uv, "normalized_uv")
    if normalized_uv.ndim != 3 or normalized_uv.shape[1:] != (len(keypoint_names), 2):
        raise ValueError("normalized_uv 必须是与 keypoint_names 对齐的 [B,K,2]")
    batch_size = int(normalized_uv.shape[0])
    if batch_index >= batch_size:
        raise ValueError("batch_index 超出 Precision prediction batch")
    if np.any(normalized_uv < 0.0) or np.any(normalized_uv > 1.0):
        raise ValueError("normalized_uv 必须位于 [0,1]")
    keypoint_shape = (batch_size, len(keypoint_names))
    visibility = _probability_array(
        prediction.visibility_probability,
        keypoint_shape,
        "visibility_probability",
    )
    peak = _probability_array(
        prediction.keypoints.peak_probability,
        keypoint_shape,
        "peak_probability",
    )
    entropy = _probability_array(
        prediction.keypoints.normalized_entropy,
        keypoint_shape,
        "normalized_entropy",
    )
    projection = _probability_array(
        prediction.projection_validity_probability,
        (batch_size,),
        "projection_validity_probability",
    )
    sigma = _numpy(prediction.keypoint_sigma_px, "keypoint_sigma_px")
    if sigma.shape != (batch_size, len(keypoint_names), 2) or np.any(sigma < 0.0):
        raise ValueError("keypoint_sigma_px 必须是有限非负 [B,K,2]")

    def evidence(index: int) -> PrecisionKeypointEvidence:
        projection_probability = float(projection[batch_index])
        visibility_probability = float(visibility[batch_index, index])
        return PrecisionKeypointEvidence(
            name=keypoint_names[index],
            normalized_uv=tuple(float(value) for value in normalized_uv[batch_index, index]),
            visibility_probability=visibility_probability,
            projection_validity_probability=projection_probability,
            peak_probability=float(peak[batch_index, index]),
            normalized_entropy=float(entropy[batch_index, index]),
            sigma_px=tuple(float(value) for value in sigma[batch_index, index]),
            track_confidence=min(visibility_probability, projection_probability),
        )

    object_evidence = evidence(object_index)
    goal_evidence = evidence(goal_index)
    return PrecisionDetectionAdapterResult(
        detection=WristKeypointDetection(
            timestamp_s=float(timestamp_s),
            object_normalized_uv=object_evidence.normalized_uv,
            goal_normalized_uv=goal_evidence.normalized_uv,
            object_confidence=object_evidence.track_confidence,
            goal_confidence=goal_evidence.track_confidence,
        ),
        object_evidence=object_evidence,
        goal_evidence=goal_evidence,
    )


__all__ = [
    "PRECISION_TRACK_CONFIDENCE_SEMANTICS",
    "PrecisionDetectionAdapterResult",
    "PrecisionKeypointEvidence",
    "precision_prediction_to_wrist_detection",
]
