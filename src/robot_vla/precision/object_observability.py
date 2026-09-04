"""E018-P0 object observability 标签与部署侧单帧 write evidence。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

from robot_vla.precision.observability import normalized_uv_to_mask_index

E018_OBJECT_OBSERVABILITY_VERSION = "e018-p0-object-observability/v1"
OBJECT_OBSERVABILITY_SEMANTICS = "projected-object-center-ray-hits-own-mask/v1"
OBJECT_WRITE_SCORE_SEMANTICS = (
    "min-visibility-projection-object-mask-not-goal-not-entropy-inverse-sigma/v1"
)


def _probability(value: float, name: str) -> float:
    candidate = float(value)
    if not math.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
        raise ValueError(f"{name} 必须是 [0,1] 内有限概率")
    return candidate


def _mask(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype != np.bool_ or min(array.shape) <= 0:
        raise ValueError(f"{name} 必须是非空 bool [H,W]")
    return array


@dataclass(frozen=True)
class ObjectObservabilityLabel:
    """只用于离线评估的 privileged object-center 可观察性标签。"""

    object_exists: bool
    projection_valid: bool
    in_fov: bool
    observable: bool
    legacy_visible: bool
    center_inside_object_mask: bool
    center_inside_goal_mask: bool
    local_object_visible_fraction: float
    object_mask_area_fraction: float
    occlusion_type: str
    semantics: str = OBJECT_OBSERVABILITY_SEMANTICS
    version: str = E018_OBJECT_OBSERVABILITY_VERSION

    def __post_init__(self) -> None:
        for name in (
            "object_exists",
            "projection_valid",
            "in_fov",
            "observable",
            "legacy_visible",
            "center_inside_object_mask",
            "center_inside_goal_mask",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须为 bool")
        if self.in_fov and not self.projection_valid:
            raise ValueError("in_fov=true 要求 projection_valid=true")
        if self.observable and not (
            self.object_exists
            and self.projection_valid
            and self.in_fov
            and self.center_inside_object_mask
        ):
            raise ValueError("observable 与 exists/projected/in_fov/own-mask 语义冲突")
        _probability(self.local_object_visible_fraction, "local_object_visible_fraction")
        _probability(self.object_mask_area_fraction, "object_mask_area_fraction")
        allowed = {
            "observable",
            "object_absent",
            "projection_invalid",
            "out_of_frame",
            "goal_occlusion",
            "other_occlusion_or_background",
        }
        if self.occlusion_type not in allowed:
            raise ValueError(f"未知 object occlusion_type: {self.occlusion_type}")
        if self.semantics != OBJECT_OBSERVABILITY_SEMANTICS:
            raise ValueError("object observability semantics 漂移")
        if self.version != E018_OBJECT_OBSERVABILITY_VERSION:
            raise ValueError("object observability version 漂移")

    @property
    def legacy_contract_mismatch(self) -> bool:
        return self.legacy_visible and not self.observable

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["legacy_contract_mismatch"] = self.legacy_contract_mismatch
        return payload


def derive_object_observability(
    *,
    object_exists: bool,
    projection_valid: bool,
    projected_normalized_uv: np.ndarray | tuple[float, float] | None,
    object_mask: np.ndarray,
    goal_mask: np.ndarray,
    legacy_visible: bool,
    support_radius_px: int = 2,
) -> ObjectObservabilityLabel:
    """判断 object center 的投影射线是否命中当前可见 object mask。"""

    if not isinstance(object_exists, bool) or not isinstance(projection_valid, bool):
        raise TypeError("object_exists/projection_valid 必须为 bool")
    if not isinstance(legacy_visible, bool):
        raise TypeError("legacy_visible 必须为 bool")
    if (
        not isinstance(support_radius_px, int)
        or isinstance(support_radius_px, bool)
        or support_radius_px < 0
    ):
        raise ValueError("support_radius_px 必须是非负整数")
    own = _mask(object_mask, "object_mask")
    other = _mask(goal_mask, "goal_mask")
    if own.shape != other.shape:
        raise ValueError("object/goal mask shape 必须一致")

    center: tuple[int, int] | None = None
    if projection_valid:
        if projected_normalized_uv is None:
            raise ValueError("有效 projection 必须提供 projected_normalized_uv")
        center = normalized_uv_to_mask_index(projected_normalized_uv, own.shape)
    elif projected_normalized_uv is not None:
        value = np.asarray(projected_normalized_uv, dtype=np.float64)
        if value.shape != (2,) or not np.isfinite(value).all():
            raise ValueError("projected_normalized_uv 必须是有限 [2]")

    in_fov = center is not None
    center_inside_object = bool(center is not None and own[center[1], center[0]])
    center_inside_goal = bool(center is not None and other[center[1], center[0]])
    if center is None:
        local_fraction = 0.0
    else:
        x, y = center
        y0 = max(0, y - support_radius_px)
        y1 = min(own.shape[0], y + support_radius_px + 1)
        x0 = max(0, x - support_radius_px)
        x1 = min(own.shape[1], x + support_radius_px + 1)
        local_fraction = float(np.mean(own[y0:y1, x0:x1]))
    observable = bool(
        object_exists and projection_valid and in_fov and center_inside_object
    )
    if observable:
        occlusion = "observable"
    elif not object_exists:
        occlusion = "object_absent"
    elif not projection_valid:
        occlusion = "projection_invalid"
    elif not in_fov:
        occlusion = "out_of_frame"
    elif center_inside_goal:
        occlusion = "goal_occlusion"
    else:
        occlusion = "other_occlusion_or_background"
    return ObjectObservabilityLabel(
        object_exists=object_exists,
        projection_valid=projection_valid,
        in_fov=in_fov,
        observable=observable,
        legacy_visible=legacy_visible,
        center_inside_object_mask=center_inside_object,
        center_inside_goal_mask=center_inside_goal,
        local_object_visible_fraction=local_fraction,
        object_mask_area_fraction=float(np.mean(own)),
        occlusion_type=occlusion,
    )


@dataclass(frozen=True)
class ObjectWriteEvidence:
    """只由部署可得预测构成的 object-memory 单帧 write evidence。"""

    visibility_probability: float
    projection_validity_probability: float
    object_mask_probability: float
    goal_mask_probability: float
    normalized_entropy: float
    radial_sigma_px: float
    geometry_valid: bool
    min_object_mask_probability: float = 0.5
    max_goal_mask_probability: float = 0.5
    score_semantics: str = OBJECT_WRITE_SCORE_SEMANTICS

    def __post_init__(self) -> None:
        for value, name in (
            (self.visibility_probability, "visibility_probability"),
            (self.projection_validity_probability, "projection_validity_probability"),
            (self.object_mask_probability, "object_mask_probability"),
            (self.goal_mask_probability, "goal_mask_probability"),
            (self.normalized_entropy, "normalized_entropy"),
            (self.min_object_mask_probability, "min_object_mask_probability"),
            (self.max_goal_mask_probability, "max_goal_mask_probability"),
        ):
            _probability(value, name)
        if not math.isfinite(self.radial_sigma_px) or self.radial_sigma_px < 0.0:
            raise ValueError("radial_sigma_px 必须是有限非负数")
        if not isinstance(self.geometry_valid, bool):
            raise TypeError("geometry_valid 必须为 bool")
        if self.score_semantics != OBJECT_WRITE_SCORE_SEMANTICS:
            raise ValueError("object write score semantics 漂移")

    @property
    def observable(self) -> bool:
        return bool(
            self.object_mask_probability >= self.min_object_mask_probability
            and self.goal_mask_probability <= self.max_goal_mask_probability
        )

    @property
    def structurally_eligible(self) -> bool:
        return self.observable and self.geometry_valid

    @property
    def score(self) -> float:
        inverse_sigma = 1.0 / (1.0 + self.radial_sigma_px)
        return float(
            min(
                self.visibility_probability,
                self.projection_validity_probability,
                self.object_mask_probability,
                1.0 - self.goal_mask_probability,
                1.0 - self.normalized_entropy,
                inverse_sigma,
            )
        )

    def accepted(self, *, threshold: float, enabled: bool = True) -> bool:
        if not math.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
            raise ValueError("write threshold 必须是 [0,1] 内有限数值")
        if not isinstance(enabled, bool):
            raise TypeError("write enabled 必须为 bool")
        return bool(enabled and self.structurally_eligible and self.score >= threshold)

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload.update(
            {
                "observable": self.observable,
                "structurally_eligible": self.structurally_eligible,
                "score": self.score,
            }
        )
        return payload


__all__ = [
    "E018_OBJECT_OBSERVABILITY_VERSION",
    "OBJECT_OBSERVABILITY_SEMANTICS",
    "OBJECT_WRITE_SCORE_SEMANTICS",
    "ObjectObservabilityLabel",
    "ObjectWriteEvidence",
    "derive_object_observability",
]
