"""E018-P1 外部相机离散视角与时序运动的纯数据契约。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np


class ExternalCameraMotionState(str, Enum):
    """G0/G1 共用的外部相机运动状态；不拥有机械臂执行权。"""

    HOME_ANCHOR = "home_anchor"
    MOVE_TO_VIEW = "move_to_view"
    SETTLE_AT_VIEW = "settle_at_view"
    COLLECT = "collect"
    RETURN_HOME = "return_home"
    VERIFY_HOME_AND_ARM_HOLD = "verify_home_and_arm_hold"


@dataclass(frozen=True)
class FrontCameraViewpoint:
    """世界坐标下的一个预注册离散 front-camera 视角。"""

    viewpoint_id: str
    lateral_anchor: str
    vertical_anchor: str
    position_world_m: tuple[float, float, float]
    look_at_world_m: tuple[float, float, float]
    yaw_rad: float
    pitch_rad: float
    roll_rad: float

    def validate(self, *, angle_atol_rad: float = 1e-6) -> None:
        if not self.viewpoint_id or not self.viewpoint_id.replace("_", "").isalnum():
            raise ValueError("viewpoint_id 必须是非空字母数字/下划线标识")
        if self.lateral_anchor not in {"CENTER", "LEFT", "RIGHT"}:
            raise ValueError("lateral_anchor 必须是 CENTER/LEFT/RIGHT")
        if self.vertical_anchor not in {"CENTER", "LOW", "HIGH"}:
            raise ValueError("vertical_anchor 必须是 CENTER/LOW/HIGH")
        position = _finite_vector(self.position_world_m, 3, "position_world_m")
        target = _finite_vector(self.look_at_world_m, 3, "look_at_world_m")
        direction = target - position
        if float(np.linalg.norm(direction)) <= 1e-6:
            raise ValueError("相机位置不能与 look-at target 重合")
        expected_yaw = math.atan2(float(direction[1]), float(direction[0]))
        expected_pitch = math.atan2(
            float(direction[2]),
            float(np.linalg.norm(direction[:2])),
        )
        for value, expected, name in (
            (self.yaw_rad, expected_yaw, "yaw_rad"),
            (self.pitch_rad, expected_pitch, "pitch_rad"),
        ):
            if not math.isfinite(value) or abs(_wrapped_angle(value - expected)) > angle_atol_rad:
                raise ValueError(f"{name} 与固定 look-at 几何不一致")
        if not math.isfinite(self.roll_rad) or abs(self.roll_rad) > angle_atol_rad:
            raise ValueError("G0 首版 roll_rad 必须固定为 0")


@dataclass(frozen=True)
class FrontCameraOrientationMode:
    """相对于锚点 nominal look-at 的离散局部旋转。"""

    orientation_id: str
    yaw_offset_rad: float
    pitch_offset_rad: float
    roll_offset_rad: float = 0.0

    def validate(self, *, angle_atol_rad: float = 1e-12) -> None:
        if not self.orientation_id or not self.orientation_id.replace("_", "").isalnum():
            raise ValueError("orientation_id 必须是非空字母数字/下划线标识")
        values = np.asarray(
            (self.yaw_offset_rad, self.pitch_offset_rad, self.roll_offset_rad),
            dtype=np.float64,
        )
        if not np.isfinite(values).all():
            raise ValueError("orientation offsets 必须有限")
        if abs(self.roll_offset_rad) > angle_atol_rad:
            raise ValueError("E018-P1 首版 roll_offset_rad 必须固定为 0")
        if abs(self.yaw_offset_rad) > math.pi / 2 or abs(self.pitch_offset_rad) > math.pi / 2:
            raise ValueError("development 离散 yaw/pitch offset 不得超过 90 度")
        if (
            abs(self.yaw_offset_rad) > angle_atol_rad
            and abs(self.pitch_offset_rad) > angle_atol_rad
        ):
            raise ValueError("首轮 cross orientation lattice 不允许 diagonal yaw+pitch")


def _finite_vector(value: object, size: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (size,) or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是有限 [{size}] 向量")
    return array


def _wrapped_angle(value: float) -> float:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def smootherstep(value: float) -> float:
    """端点一、二阶导数为零的五次插值，避免把视角切换建模成零时延跳变。"""

    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError("smootherstep 输入必须位于 [0,1]")
    return value**3 * (value * (value * 6.0 - 15.0) + 10.0)


def sample_translation_path(
    start_position_m: tuple[float, float, float] | np.ndarray,
    end_position_m: tuple[float, float, float] | np.ndarray,
    *,
    steps: int,
) -> np.ndarray:
    """返回不含起点、包含终点的逐 Tick 五次平移轨迹。"""

    if not isinstance(steps, int) or isinstance(steps, bool) or steps <= 1:
        raise ValueError("steps 必须是大于 1 的整数")
    start = _finite_vector(start_position_m, 3, "start_position_m")
    end = _finite_vector(end_position_m, 3, "end_position_m")
    if np.array_equal(start, end):
        raise ValueError("相机运动起点与终点不能相同")
    fractions = np.asarray(
        [smootherstep(index / steps) for index in range(1, steps + 1)],
        dtype=np.float64,
    )
    return start[None, :] + fractions[:, None] * (end - start)[None, :]


def _unit_quaternion(value: np.ndarray, name: str) -> np.ndarray:
    quaternion = _finite_vector(value, 4, name)
    norm = float(np.linalg.norm(quaternion))
    if not math.isfinite(norm) or norm <= 1e-12:
        raise ValueError("四元数范数必须有限且为正")
    return quaternion / norm


def quaternion_angular_distance_rad(
    first: tuple[float, float, float, float] | np.ndarray,
    second: tuple[float, float, float, float] | np.ndarray,
) -> float:
    """返回两个单位四元数之间的最短转角；对 q/-q 等价。"""

    one = _unit_quaternion(first, "first quaternion")
    two = _unit_quaternion(second, "second quaternion")
    cosine_half_angle = float(abs(np.dot(one, two)))
    return 2.0 * math.acos(float(np.clip(cosine_half_angle, -1.0, 1.0)))


def quaternion_multiply_wxyz(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Hamilton product，输入输出均使用 SAPIEN/ManiSkill 的 ``wxyz`` 顺序。"""

    one = _unit_quaternion(first, "first quaternion")
    two = _unit_quaternion(second, "second quaternion")
    w1, x1, y1, z1 = one
    w2, x2, y2, z2 = two
    result = np.asarray(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dtype=np.float64,
    )
    return _unit_quaternion(result, "result quaternion")


def compose_camera_orientation_wxyz(
    nominal_quaternion_wxyz: np.ndarray,
    orientation: FrontCameraOrientationMode,
) -> np.ndarray:
    """给 nominal camera pose 施加局部 yaw/pitch，返回完整 SAPIEN 四元数。

    SAPIEN camera mount frame 使用 ``+X`` forward、``+Y`` left、``+Z`` up。
    因此正 yaw 绕局部 ``+Z`` 朝左，正 pitch 绕局部 ``-Y`` 朝上。首轮
    cross lattice 不组合非零 yaw 与 pitch，roll 始终为零。
    """

    orientation.validate()
    nominal = _unit_quaternion(nominal_quaternion_wxyz, "nominal quaternion")
    half_yaw = orientation.yaw_offset_rad * 0.5
    half_pitch = orientation.pitch_offset_rad * 0.5
    local_yaw = np.asarray(
        (math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)),
        dtype=np.float64,
    )
    # 正 pitch 定义为 optical forward 朝 local +Z（up）转动，因此绕 -Y。
    local_pitch = np.asarray(
        (math.cos(half_pitch), 0.0, -math.sin(half_pitch), 0.0),
        dtype=np.float64,
    )
    local_offset = quaternion_multiply_wxyz(local_yaw, local_pitch)
    return quaternion_multiply_wxyz(nominal, local_offset)


def rotation_angular_distance_rad(first: np.ndarray, second: np.ndarray) -> float:
    """返回两个 3x3 旋转矩阵之间的最短转角。

    仿真器输出是 float32，矩阵可能在 ``validate_se3`` 允许范围内轻微偏离 SO(3)。
    先用极分解投影到最近的正规旋转，避免把同一矩阵的正交舍入误差经 ``acos``
    放大成虚假的姿态漂移。
    """

    one = np.asarray(first, dtype=np.float64)
    two = np.asarray(second, dtype=np.float64)
    if one.shape != (3, 3) or two.shape != (3, 3):
        raise ValueError("旋转矩阵必须是 [3,3]")
    if not np.isfinite(one).all() or not np.isfinite(two).all():
        raise ValueError("旋转矩阵必须有限")

    def closest_rotation(value: np.ndarray) -> np.ndarray:
        left, _, right = np.linalg.svd(value)
        projected = left @ right
        if np.linalg.det(projected) < 0.0:
            left[:, -1] *= -1.0
            projected = left @ right
        if np.linalg.norm(value - projected, ord="fro") > 1e-5:
            raise ValueError("旋转矩阵偏离 SO(3)，不是舍入误差")
        return projected

    one = closest_rotation(one)
    two = closest_rotation(two)
    cosine = float((np.trace(one.T @ two) - 1.0) * 0.5)
    return math.acos(float(np.clip(cosine, -1.0, 1.0)))


def measurement_write_eligible(
    state: ExternalCameraMotionState,
    *,
    settled: bool,
) -> bool:
    """运动/settle/return 帧永远不能成为 Memory 写入候选。"""

    if not isinstance(state, ExternalCameraMotionState):
        raise TypeError("state 必须是 ExternalCameraMotionState")
    if not isinstance(settled, bool):
        raise TypeError("settled 必须是 bool")
    return state is ExternalCameraMotionState.COLLECT and settled


__all__ = [
    "ExternalCameraMotionState",
    "FrontCameraOrientationMode",
    "FrontCameraViewpoint",
    "compose_camera_orientation_wxyz",
    "measurement_write_eligible",
    "quaternion_angular_distance_rad",
    "quaternion_multiply_wxyz",
    "rotation_angular_distance_rad",
    "sample_translation_path",
    "smootherstep",
]
