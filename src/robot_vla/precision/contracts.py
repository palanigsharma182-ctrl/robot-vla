"""高频精密执行层的坐标、单位和控制权限契约。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any

import numpy as np

PRECISION_MODEL_ARCH = "precision_unet_three_head_v1"
PRECISION_MOTION_SEMANTICS = "commanded-tcp-target-delta/base-frame/m-rad/v1"
PRECISION_MOTION_COMPONENTS = (
    "delta_x_base_m",
    "delta_y_base_m",
    "delta_z_base_m",
    "delta_yaw_base_rad",
)


class PrecisionControlMode(str, Enum):
    """学习 Motion Head 对正式命令的权限。"""

    SHADOW = "shadow"
    BOUNDED_RESIDUAL = "bounded_residual"


def _finite_positive_tuple(
    values: tuple[float, ...],
    *,
    expected: int,
    name: str,
) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} 必须包含 {expected} 个分量")
    if any(not math.isfinite(value) or value <= 0.0 for value in values):
        raise ValueError(f"{name} 必须全部为有限正数")


@dataclass(frozen=True)
class PrecisionMotionSpec:
    """单个高频 Tick 的 commanded TCP target delta。

    该语义与现有 VLA 的 commanded joint-target delta 明确隔离。这里固定使用机器人
    base frame、米和弧度；从笛卡尔增量到关节控制命令的 IK/控制器适配不属于本契约。
    """

    semantics: str = PRECISION_MOTION_SEMANTICS
    frame: str = "robot_base"
    translation_unit: str = "meter"
    rotation_unit: str = "radian"
    components: tuple[str, ...] = PRECISION_MOTION_COMPONENTS
    # 这些只是 v1 工程安全上限，不是已经验证的 2 mm 性能结论。
    step_limits: tuple[float, ...] = (
        1.0e-3,
        1.0e-3,
        0.5e-3,
        math.radians(0.2),
    )
    residual_limits: tuple[float, ...] = (
        0.25e-3,
        0.25e-3,
        0.20e-3,
        math.radians(0.05),
    )

    def __post_init__(self) -> None:
        if self.semantics != PRECISION_MOTION_SEMANTICS:
            raise ValueError(f"precision motion semantics 必须为 {PRECISION_MOTION_SEMANTICS}")
        if self.frame != "robot_base":
            raise ValueError("precision motion frame 必须为 robot_base")
        if self.translation_unit != "meter" or self.rotation_unit != "radian":
            raise ValueError("precision motion 单位必须为 meter/radian")
        if self.components != PRECISION_MOTION_COMPONENTS:
            raise ValueError(f"precision motion components 必须为 {PRECISION_MOTION_COMPONENTS}")
        _finite_positive_tuple(
            self.step_limits,
            expected=self.motion_dim,
            name="step_limits",
        )
        _finite_positive_tuple(
            self.residual_limits,
            expected=self.motion_dim,
            name="residual_limits",
        )
        if any(
            residual > step
            for residual, step in zip(self.residual_limits, self.step_limits, strict=True)
        ):
            raise ValueError("residual_limits 不能大于 step_limits")

    @property
    def motion_dim(self) -> int:
        return len(self.components)

    def clip_step(self, value: np.ndarray) -> np.ndarray:
        return self._clip(value, self.step_limits, "precision step")

    def clip_residual(self, value: np.ndarray) -> np.ndarray:
        return self._clip(value, self.residual_limits, "precision residual")

    def _clip(
        self,
        value: np.ndarray,
        limits: tuple[float, ...],
        name: str,
    ) -> np.ndarray:
        array = np.asarray(value)
        if (
            array.shape != (self.motion_dim,)
            or array.dtype != np.float32
            or not np.isfinite(array).all()
        ):
            raise ValueError(f"{name} 必须是有限 float32 [{self.motion_dim}]")
        bound = np.asarray(limits, dtype=np.float32)
        return np.clip(array, -bound, bound).astype(np.float32, copy=False)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
