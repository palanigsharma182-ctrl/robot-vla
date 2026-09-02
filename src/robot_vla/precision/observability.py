"""E015 goal observability 标签与部署侧 write-gate 证据。

本模块只处理当前帧 evidence；跨帧状态由 :mod:`state_memory` 维护。Privileged
mask 只用于离线标签，部署 write score 只读取 frozen U-Net 的预测量。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

import numpy as np

E015_OBSERVABILITY_VERSION = "e015-goal-observability/v1"
GOAL_OBSERVABILITY_SEMANTICS = "projected-center-ray-hits-own-goal-mask/v1"
GOAL_WRITE_SCORE_SEMANTICS = (
    "min-visibility-projection-goal-mask-not-object-not-entropy-inverse-sigma/v1"
)


def _probability(value: float, name: str) -> float:
    candidate = float(value)
    if not math.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
        raise ValueError(f"{name} 必须是 [0,1] 内的有限概率")
    return candidate


def _mask(value: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 2 or array.dtype != np.bool_ or min(array.shape) <= 0:
        raise ValueError(f"{name} 必须是非空 bool [H,W]")
    return array


def _normalized_uv(value: np.ndarray | tuple[float, float]) -> np.ndarray:
    uv = np.asarray(value, dtype=np.float64)
    if uv.shape != (2,) or not np.isfinite(uv).all():
        raise ValueError("normalized_uv 必须是有限 [2]")
    return uv


def normalized_uv_to_mask_index(
    normalized_uv: np.ndarray | tuple[float, float],
    image_size_hw: tuple[int, int],
) -> tuple[int, int] | None:
    """返回投影中心所在像素 ``(x,y)``；出画返回 ``None``。

    项目坐标为 ``normalized=(pixel+0.5)/size``，因此 ``floor(normalized*size)``
    精确恢复像素中心所属 cell。
    """

    uv = _normalized_uv(normalized_uv)
    if len(image_size_hw) != 2:
        raise ValueError("image_size_hw 必须是 (height,width)")
    height, width = image_size_hw
    if (
        not isinstance(height, int)
        or isinstance(height, bool)
        or not isinstance(width, int)
        or isinstance(width, bool)
        or height <= 0
        or width <= 0
    ):
        raise ValueError("image_size_hw 必须包含两个正整数")
    x = math.floor(float(uv[0]) * width)
    y = math.floor(float(uv[1]) * height)
    if not 0 <= x < width or not 0 <= y < height:
        return None
    return x, y


def mask_probability_at_normalized_uv(
    probability: np.ndarray,
    normalized_uv: np.ndarray | tuple[float, float],
) -> float:
    """按项目 pixel-center 语义双线性采样预测 mask probability。"""

    values = np.asarray(probability, dtype=np.float64)
    if (
        values.ndim != 2
        or min(values.shape) <= 0
        or not np.isfinite(values).all()
        or np.any(values < 0.0)
        or np.any(values > 1.0)
    ):
        raise ValueError("mask probability 必须是有限 [0,1] [H,W]")
    uv = _normalized_uv(normalized_uv)
    if np.any(uv < 0.0) or np.any(uv > 1.0):
        raise ValueError("部署 mask sample normalized_uv 必须位于 [0,1]")
    height, width = values.shape
    x = float(np.clip(uv[0] * width - 0.5, 0.0, width - 1.0))
    y = float(np.clip(uv[1] * height - 0.5, 0.0, height - 1.0))
    x0 = math.floor(x)
    y0 = math.floor(y)
    x1 = min(x0 + 1, width - 1)
    y1 = min(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    value = (
        (1.0 - wx) * (1.0 - wy) * values[y0, x0]
        + wx * (1.0 - wy) * values[y0, x1]
        + (1.0 - wx) * wy * values[y1, x0]
        + wx * wy * values[y1, x1]
    )
    return float(value)


@dataclass(frozen=True)
class GoalObservabilityLabel:
    """Privileged E015-A label；不会进入模型输入或部署 Provider。"""

    goal_exists: bool
    projection_valid: bool
    in_fov: bool
    observable: bool
    legacy_visible: bool
    center_inside_goal_mask: bool
    center_inside_object_mask: bool
    local_goal_visible_fraction: float
    goal_mask_area_fraction: float
    occlusion_type: str
    semantics: str = GOAL_OBSERVABILITY_SEMANTICS
    version: str = E015_OBSERVABILITY_VERSION

    def __post_init__(self) -> None:
        for name in (
            "goal_exists",
            "projection_valid",
            "in_fov",
            "observable",
            "legacy_visible",
            "center_inside_goal_mask",
            "center_inside_object_mask",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须为 bool")
        if self.in_fov and not self.projection_valid:
            raise ValueError("in_fov=true 要求 projection_valid=true")
        if self.observable and not (
            self.goal_exists
            and self.projection_valid
            and self.in_fov
            and self.center_inside_goal_mask
        ):
            raise ValueError("observable 与 exists/projected/in_fov/own-mask 语义冲突")
        for value, name in (
            (self.local_goal_visible_fraction, "local_goal_visible_fraction"),
            (self.goal_mask_area_fraction, "goal_mask_area_fraction"),
        ):
            _probability(value, name)
        allowed = {
            "observable",
            "goal_absent",
            "projection_invalid",
            "out_of_frame",
            "object_occlusion",
            "other_occlusion_or_background",
        }
        if self.occlusion_type not in allowed:
            raise ValueError(f"未知 goal occlusion_type: {self.occlusion_type}")
        if self.semantics != GOAL_OBSERVABILITY_SEMANTICS:
            raise ValueError("goal observability semantics 漂移")
        if self.version != E015_OBSERVABILITY_VERSION:
            raise ValueError("goal observability version 漂移")

    @property
    def legacy_contract_mismatch(self) -> bool:
        return self.legacy_visible and not self.observable

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["legacy_contract_mismatch"] = self.legacy_contract_mismatch
        return payload


def derive_goal_observability(
    *,
    goal_exists: bool,
    projection_valid: bool,
    projected_normalized_uv: np.ndarray | tuple[float, float] | None,
    goal_mask: np.ndarray,
    object_mask: np.ndarray,
    legacy_visible: bool,
    support_radius_px: int = 2,
) -> GoalObservabilityLabel:
    """由 simulator segmentation 判断投影中心射线是否仍命中 goal actor。"""

    if not isinstance(goal_exists, bool) or not isinstance(projection_valid, bool):
        raise TypeError("goal_exists/projection_valid 必须为 bool")
    if not isinstance(legacy_visible, bool):
        raise TypeError("legacy_visible 必须为 bool")
    if (
        not isinstance(support_radius_px, int)
        or isinstance(support_radius_px, bool)
        or support_radius_px < 0
    ):
        raise ValueError("support_radius_px 必须是非负整数")
    own = _mask(goal_mask, "goal_mask")
    other = _mask(object_mask, "object_mask")
    if own.shape != other.shape:
        raise ValueError("goal/object mask shape 必须一致")
    center: tuple[int, int] | None = None
    if projection_valid:
        if projected_normalized_uv is None:
            raise ValueError("有效 projection 必须提供 projected_normalized_uv")
        center = normalized_uv_to_mask_index(projected_normalized_uv, own.shape)
    elif projected_normalized_uv is not None:
        _normalized_uv(projected_normalized_uv)
    in_fov = center is not None
    center_inside_goal = bool(center is not None and own[center[1], center[0]])
    center_inside_object = bool(center is not None and other[center[1], center[0]])
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
        goal_exists and projection_valid and in_fov and center_inside_goal
    )
    if observable:
        occlusion = "observable"
    elif not goal_exists:
        occlusion = "goal_absent"
    elif not projection_valid:
        occlusion = "projection_invalid"
    elif not in_fov:
        occlusion = "out_of_frame"
    elif center_inside_object:
        occlusion = "object_occlusion"
    else:
        occlusion = "other_occlusion_or_background"
    return GoalObservabilityLabel(
        goal_exists=goal_exists,
        projection_valid=projection_valid,
        in_fov=in_fov,
        observable=observable,
        legacy_visible=legacy_visible,
        center_inside_goal_mask=center_inside_goal,
        center_inside_object_mask=center_inside_object,
        local_goal_visible_fraction=local_fraction,
        goal_mask_area_fraction=float(np.mean(own)),
        occlusion_type=occlusion,
    )


@dataclass(frozen=True)
class GoalWriteEvidence:
    """只由部署可得量构成的 goal-memory write evidence。"""

    visibility_probability: float
    projection_validity_probability: float
    goal_mask_probability: float
    object_mask_probability: float
    normalized_entropy: float
    radial_sigma_px: float
    geometry_valid: bool
    min_goal_mask_probability: float = 0.5
    max_object_mask_probability: float = 0.5
    score_semantics: str = GOAL_WRITE_SCORE_SEMANTICS

    def __post_init__(self) -> None:
        for value, name in (
            (self.visibility_probability, "visibility_probability"),
            (self.projection_validity_probability, "projection_validity_probability"),
            (self.goal_mask_probability, "goal_mask_probability"),
            (self.object_mask_probability, "object_mask_probability"),
            (self.normalized_entropy, "normalized_entropy"),
            (self.min_goal_mask_probability, "min_goal_mask_probability"),
            (self.max_object_mask_probability, "max_object_mask_probability"),
        ):
            _probability(value, name)
        if not math.isfinite(self.radial_sigma_px) or self.radial_sigma_px < 0.0:
            raise ValueError("radial_sigma_px 必须是有限非负数")
        if not isinstance(self.geometry_valid, bool):
            raise TypeError("geometry_valid 必须为 bool")
        if self.score_semantics != GOAL_WRITE_SCORE_SEMANTICS:
            raise ValueError("goal write score semantics 漂移")

    @property
    def observable(self) -> bool:
        return bool(
            self.goal_mask_probability >= self.min_goal_mask_probability
            and self.object_mask_probability <= self.max_object_mask_probability
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
                self.goal_mask_probability,
                1.0 - self.object_mask_probability,
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
    "E015_OBSERVABILITY_VERSION",
    "GOAL_OBSERVABILITY_SEMANTICS",
    "GOAL_WRITE_SCORE_SEMANTICS",
    "GoalObservabilityLabel",
    "GoalWriteEvidence",
    "derive_goal_observability",
    "mask_probability_at_normalized_uv",
    "normalized_uv_to_mask_index",
]
