"""E018-P1 动态 external camera 观测 sidecar 契约。

该模块不修改冻结的 :class:`ObservationV2Frame`，也不持有 provider、Memory 或
机械臂执行权。actual pose 只能由与 RGB 同一次返回的 ManiSkill observation 中
提取，避免把 commanded pose 冒充传感器实际外参。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

from robot_vla.observation import invert_se3, opengl_camera_to_opencv, validate_se3
from robot_vla.precision.active_front_camera import (
    ExternalCameraMotionState,
    measurement_write_eligible,
)

ACTIVE_EXTERNAL_OBSERVATION_VERSION = "e018-p1-active-external-observation/v2"
ACTUAL_EXTERNAL_POSE_SOURCE = (
    "same-observation.sensor_param.base_camera.cam2world_gl/v1"
)


def _numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _single_matrix(value: Any, name: str) -> np.ndarray:
    if callable(getattr(value, "to_transformation_matrix", None)):
        value = value.to_transformation_matrix()
    matrix = _numpy(value)
    if matrix.shape == (1, 4, 4):
        matrix = matrix[0]
    return validate_se3(matrix, name)


def _intrinsic_matrix(value: Any) -> np.ndarray:
    intrinsic = _numpy(value)
    if intrinsic.shape == (1, 3, 3):
        intrinsic = intrinsic[0]
    intrinsic = np.asarray(intrinsic, dtype=np.float64)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("external intrinsic_cv 必须是有限 [3,3]")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("external intrinsic_cv 的 fx/fy 必须为正")
    if not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-8):
        raise ValueError("external intrinsic_cv 最后一行必须是 [0,0,1]")
    return intrinsic


def _timestamp(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result < 0.0:
        raise ValueError(f"{name} 必须是有限非负秒数")
    return result


@dataclass(frozen=True)
class RotationProjectionAudit:
    """float32 sensor rotation 到最近 SO(3) 的可审计数值修正。"""

    correction_frobenius: float
    determinant_before: float
    orthogonality_error_before_frobenius: float
    determinant_after: float
    orthogonality_error_after_frobenius: float
    maximum_correction_frobenius: float

    def __post_init__(self) -> None:
        for name in (
            "correction_frobenius",
            "determinant_before",
            "orthogonality_error_before_frobenius",
            "determinant_after",
            "orthogonality_error_after_frobenius",
            "maximum_correction_frobenius",
        ):
            if not math.isfinite(float(getattr(self, name))):
                raise ValueError(f"rotation projection audit.{name} 必须有限")
        if self.correction_frobenius < 0.0:
            raise ValueError("rotation projection correction 必须非负")
        if self.orthogonality_error_before_frobenius < 0.0:
            raise ValueError("rotation projection before orthogonality error 必须非负")
        if self.orthogonality_error_after_frobenius < 0.0:
            raise ValueError("rotation projection after orthogonality error 必须非负")
        if self.maximum_correction_frobenius <= 0.0:
            raise ValueError("rotation projection maximum correction 必须为正")

    def ledger_record(self) -> dict[str, float]:
        return {
            "correction_frobenius": self.correction_frobenius,
            "determinant_before": self.determinant_before,
            "orthogonality_error_before_frobenius": (
                self.orthogonality_error_before_frobenius
            ),
            "determinant_after": self.determinant_after,
            "orthogonality_error_after_frobenius": (
                self.orthogonality_error_after_frobenius
            ),
            "maximum_correction_frobenius": self.maximum_correction_frobenius,
        }


def _closest_rigid_transform(
    value: np.ndarray,
    name: str,
    *,
    maximum_rotation_projection_error_frobenius: float,
) -> tuple[np.ndarray, RotationProjectionAudit]:
    """把传感器 float32 旋转投影到最近 SO(3)，平移保持逐字不变。

    ManiSkill 的 ``cam2world_gl`` 以 float32 返回，满足项目 ``validate_se3``
    容差但不一定逐位正交。项目刚体逆使用 ``R.T``；若不在输入边界正规化，
    base→camera→base 会产生约 float32 epsilon 量级的伪 round-trip 误差。
    """

    maximum_correction = float(maximum_rotation_projection_error_frobenius)
    if not math.isfinite(maximum_correction) or maximum_correction <= 0.0:
        raise ValueError("maximum rotation projection error 必须是有限正数")
    transform = validate_se3(value, name)
    rotation = transform[:3, :3]
    determinant_before = float(np.linalg.det(rotation))
    orthogonality_before = float(
        np.linalg.norm(rotation.T @ rotation - np.eye(3), ord="fro")
    )
    left, _, right = np.linalg.svd(rotation)
    projected = left @ right
    if np.linalg.det(projected) < 0.0:
        left[:, -1] *= -1.0
        projected = left @ right
    result = transform.copy()
    result[:3, :3] = projected
    correction = float(np.linalg.norm(projected - rotation, ord="fro"))
    audit = RotationProjectionAudit(
        correction_frobenius=correction,
        determinant_before=determinant_before,
        orthogonality_error_before_frobenius=orthogonality_before,
        determinant_after=float(np.linalg.det(projected)),
        orthogonality_error_after_frobenius=float(
            np.linalg.norm(projected.T @ projected - np.eye(3), ord="fro")
        ),
        maximum_correction_frobenius=maximum_correction,
    )
    if correction > maximum_correction:
        raise ValueError(
            f"{name} rotation projection correction {correction} 超过冻结容差 "
            f"{maximum_correction}"
        )
    return validate_se3(result, f"canonical_{name}"), audit


@dataclass(frozen=True)
class ActiveExternalObservation:
    """一个控制 Tick 的动态 external-camera sidecar。

    RGB 与 actual pose 来自同一个 ``observation_sequence_id``。时间字段仍分开
    保存，仿真器可将二者映射到同一控制 Tick；真实硬件接入时不能沿用零 skew
    假设。
    """

    episode_id: str
    request_id: str
    observation_sequence_id: str
    camera_command_sequence_id: str
    control_tick: int
    control_timestamp_s: float
    rgb_timestamp_s: float
    camera_pose_timestamp_s: float
    camera_motion_state: ExternalCameraMotionState
    viewpoint_primitive_id: str
    rgb_external: np.ndarray
    intrinsic_cv: np.ndarray
    commanded_world_from_external_camera_gl: np.ndarray
    actual_world_from_external_camera_gl: np.ndarray
    base_from_external_camera_cv: np.ndarray
    settled: bool
    actual_rotation_projection_audit: RotationProjectionAudit
    base_rotation_projection_audit: RotationProjectionAudit
    version: str = ACTIVE_EXTERNAL_OBSERVATION_VERSION
    actual_pose_source: str | None = None
    camera_uid: str = "base_camera"

    def __post_init__(self) -> None:
        if self.version != ACTIVE_EXTERNAL_OBSERVATION_VERSION:
            raise ValueError("active external observation version 漂移")
        for name in (
            "episode_id",
            "request_id",
            "observation_sequence_id",
            "camera_command_sequence_id",
            "viewpoint_primitive_id",
        ):
            if not isinstance(getattr(self, name), str) or not getattr(self, name):
                raise ValueError(f"{name} 必须是非空字符串")
        if not isinstance(self.camera_uid, str) or not self.camera_uid.strip():
            raise ValueError("camera_uid 必须是非空字符串")
        expected_pose_source = f"same-observation.sensor_param.{self.camera_uid}.cam2world_gl/v1"
        if self.actual_pose_source is None:
            object.__setattr__(self, "actual_pose_source", expected_pose_source)
        if self.actual_pose_source != expected_pose_source:
            raise ValueError("actual pose source 漂移")
        if (
            not isinstance(self.control_tick, int)
            or isinstance(self.control_tick, bool)
            or self.control_tick < 0
        ):
            raise ValueError("control_tick 必须是非负整数")
        control_timestamp = _timestamp(self.control_timestamp_s, "control_timestamp_s")
        rgb_timestamp = _timestamp(self.rgb_timestamp_s, "rgb_timestamp_s")
        pose_timestamp = _timestamp(
            self.camera_pose_timestamp_s,
            "camera_pose_timestamp_s",
        )
        if rgb_timestamp > control_timestamp + 1e-9:
            raise ValueError("RGB timestamp 不能晚于控制 Tick")
        if pose_timestamp > control_timestamp + 1e-9:
            raise ValueError("camera pose timestamp 不能晚于控制 Tick")
        if not isinstance(self.camera_motion_state, ExternalCameraMotionState):
            raise TypeError("camera_motion_state 必须是 ExternalCameraMotionState")
        if not isinstance(self.settled, bool):
            raise TypeError("settled 必须是 bool")
        if not isinstance(self.actual_rotation_projection_audit, RotationProjectionAudit):
            raise TypeError("actual_rotation_projection_audit 类型错误")
        if not isinstance(self.base_rotation_projection_audit, RotationProjectionAudit):
            raise TypeError("base_rotation_projection_audit 类型错误")

        rgb = _numpy(self.rgb_external)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError("external RGB 必须是 uint8 [H,W,3]")
        if min(rgb.shape[:2]) <= 0:
            raise ValueError("external RGB 空间尺寸必须为正")
        object.__setattr__(self, "rgb_external", np.ascontiguousarray(rgb).copy())
        object.__setattr__(self, "intrinsic_cv", _intrinsic_matrix(self.intrinsic_cv).copy())
        commanded_world_from_gl = validate_se3(
            self.commanded_world_from_external_camera_gl,
            "commanded_world_from_external_camera_gl",
        )
        actual_world_from_gl = validate_se3(
            self.actual_world_from_external_camera_gl,
            "actual_world_from_external_camera_gl",
        )
        base_from_camera_cv = validate_se3(
            self.base_from_external_camera_cv,
            "base_from_external_camera_cv",
        )
        object.__setattr__(
            self,
            "commanded_world_from_external_camera_gl",
            commanded_world_from_gl.copy(),
        )
        object.__setattr__(
            self,
            "actual_world_from_external_camera_gl",
            actual_world_from_gl.copy(),
        )
        object.__setattr__(
            self,
            "base_from_external_camera_cv",
            base_from_camera_cv.copy(),
        )

    @property
    def rgb_pose_skew_s(self) -> float:
        return abs(self.rgb_timestamp_s - self.camera_pose_timestamp_s)

    @property
    def rgb_sha256(self) -> str:
        return hashlib.sha256(self.rgb_external.tobytes()).hexdigest()

    @property
    def memory_write_eligible(self) -> bool:
        return measurement_write_eligible(
            self.camera_motion_state,
            settled=self.settled,
        )

    def ledger_record(self) -> dict[str, Any]:
        """返回不含 RGB payload、GT、provider 输出或 Memory 状态的审计记录。"""

        return {
            "version": self.version,
            "camera_uid": self.camera_uid,
            "episode_id": self.episode_id,
            "request_id": self.request_id,
            "observation_sequence_id": self.observation_sequence_id,
            "camera_command_sequence_id": self.camera_command_sequence_id,
            "control_tick": self.control_tick,
            "control_timestamp_s": self.control_timestamp_s,
            "rgb_timestamp_s": self.rgb_timestamp_s,
            "camera_pose_timestamp_s": self.camera_pose_timestamp_s,
            "rgb_pose_skew_s": self.rgb_pose_skew_s,
            "camera_motion_state": self.camera_motion_state.value,
            "viewpoint_primitive_id": self.viewpoint_primitive_id,
            "rgb_shape": list(self.rgb_external.shape),
            "rgb_sha256": self.rgb_sha256,
            "intrinsic_cv": self.intrinsic_cv.tolist(),
            "commanded_world_from_external_camera_gl": (
                self.commanded_world_from_external_camera_gl.tolist()
            ),
            "actual_world_from_external_camera_gl": (
                self.actual_world_from_external_camera_gl.tolist()
            ),
            "base_from_external_camera_cv": self.base_from_external_camera_cv.tolist(),
            "actual_pose_source": self.actual_pose_source,
            "actual_rotation_projection_audit": (
                self.actual_rotation_projection_audit.ledger_record()
            ),
            "base_rotation_projection_audit": (
                self.base_rotation_projection_audit.ledger_record()
            ),
            "settled": self.settled,
            "memory_write_eligible": self.memory_write_eligible,
            "contains_gt": False,
            "provider_inference_executed": False,
            "memory_read_executed": False,
            "memory_write_executed": False,
            "test_data_read": False,
        }

    def audit_digest(self) -> str:
        payload = json.dumps(
            self.ledger_record(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


def extract_active_external_observation(
    observation: dict[str, Any],
    *,
    camera_uid: str,
    world_from_robot_base: np.ndarray,
    commanded_world_from_external_camera_gl: np.ndarray,
    episode_id: str,
    request_id: str,
    observation_sequence_id: str,
    camera_command_sequence_id: str,
    control_tick: int,
    control_timestamp_s: float,
    rgb_timestamp_s: float,
    camera_pose_timestamp_s: float,
    camera_motion_state: ExternalCameraMotionState,
    viewpoint_primitive_id: str,
    settled: bool,
    maximum_rotation_projection_error_frobenius: float,
) -> ActiveExternalObservation:
    """从同一个 ManiSkill observation 原子提取 RGB、intrinsic 与 actual pose。"""

    if not isinstance(observation, dict):
        raise TypeError("observation 必须是 dict")
    try:
        sensor_data = observation["sensor_data"][camera_uid]
        sensor_param = observation["sensor_param"][camera_uid]
        rgb = sensor_data["rgb"]
        intrinsic = sensor_param["intrinsic_cv"]
        actual_world_from_gl = sensor_param["cam2world_gl"]
    except (KeyError, TypeError) as error:
        raise ValueError("同次 observation 缺少 external RGB/intrinsic/actual pose") from error

    rgb_array = _numpy(rgb)
    if rgb_array.ndim == 4 and rgb_array.shape[0] == 1:
        rgb_array = rgb_array[0]
    raw_actual_world_from_gl = _single_matrix(
        actual_world_from_gl,
        "actual_world_from_external_camera_gl",
    )
    canonical_actual_world_from_gl, actual_projection_audit = _closest_rigid_transform(
        raw_actual_world_from_gl,
        "actual_world_from_external_camera_gl",
        maximum_rotation_projection_error_frobenius=(
            maximum_rotation_projection_error_frobenius
        ),
    )
    raw_world_from_base = _single_matrix(world_from_robot_base, "world_from_robot_base")
    world_from_base, base_projection_audit = _closest_rigid_transform(
        raw_world_from_base,
        "world_from_robot_base",
        maximum_rotation_projection_error_frobenius=(
            maximum_rotation_projection_error_frobenius
        ),
    )
    base_from_world = invert_se3(world_from_base, "world_from_robot_base")
    base_from_camera_cv = validate_se3(
        base_from_world @ opengl_camera_to_opencv(canonical_actual_world_from_gl),
        "base_from_external_camera_cv",
    )
    return ActiveExternalObservation(
        camera_uid=camera_uid,
        episode_id=episode_id,
        request_id=request_id,
        observation_sequence_id=observation_sequence_id,
        camera_command_sequence_id=camera_command_sequence_id,
        control_tick=control_tick,
        control_timestamp_s=control_timestamp_s,
        rgb_timestamp_s=rgb_timestamp_s,
        camera_pose_timestamp_s=camera_pose_timestamp_s,
        camera_motion_state=camera_motion_state,
        viewpoint_primitive_id=viewpoint_primitive_id,
        rgb_external=rgb_array,
        intrinsic_cv=intrinsic,
        commanded_world_from_external_camera_gl=commanded_world_from_external_camera_gl,
        # 原始 actual sensor matrix 逐值保留；只有派生的刚体变换使用最近 SO(3)。
        actual_world_from_external_camera_gl=raw_actual_world_from_gl,
        base_from_external_camera_cv=base_from_camera_cv,
        settled=settled,
        actual_rotation_projection_audit=actual_projection_audit,
        base_rotation_projection_audit=base_projection_audit,
    )


def project_base_point(
    sidecar: ActiveExternalObservation,
    position_base_m: np.ndarray,
) -> dict[str, Any]:
    """把 base-frame 点投影到当前动态 OpenCV 相机，用于只读诊断。"""

    point = np.asarray(position_base_m, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("position_base_m 必须是有限 [3]")
    camera_from_base = invert_se3(
        sidecar.base_from_external_camera_cv,
        "base_from_external_camera_cv",
    )
    camera_xyz = (camera_from_base @ np.concatenate((point, np.ones(1))))[:3]
    depth_m = float(camera_xyz[2])
    if depth_m <= 1e-8:
        return {
            "projection_valid": False,
            "in_frame": False,
            "uv_px": None,
            "depth_m": depth_m,
            "camera_xyz_m": camera_xyz.tolist(),
        }
    projected = sidecar.intrinsic_cv @ camera_xyz
    uv = projected[:2] / projected[2]
    height, width = sidecar.rgb_external.shape[:2]
    return {
        "projection_valid": True,
        "in_frame": bool(0.0 <= uv[0] <= width - 1 and 0.0 <= uv[1] <= height - 1),
        "uv_px": uv.tolist(),
        "depth_m": depth_m,
        "camera_xyz_m": camera_xyz.tolist(),
    }


def base_camera_round_trip_error_m(
    sidecar: ActiveExternalObservation,
    position_base_m: np.ndarray,
) -> float:
    """验证 base -> current camera -> base 的 SE(3) 方向和数值一致性。"""

    point = np.asarray(position_base_m, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError("position_base_m 必须是有限 [3]")
    homogeneous = np.concatenate((point, np.ones(1, dtype=np.float64)))
    camera_from_base = invert_se3(
        sidecar.base_from_external_camera_cv,
        "base_from_external_camera_cv",
    )
    recovered = sidecar.base_from_external_camera_cv @ (camera_from_base @ homogeneous)
    return float(np.linalg.norm(recovered[:3] - point))


__all__ = [
    "ACTIVE_EXTERNAL_OBSERVATION_VERSION",
    "ACTUAL_EXTERNAL_POSE_SOURCE",
    "ActiveExternalObservation",
    "RotationProjectionAudit",
    "base_camera_round_trip_error_m",
    "extract_active_external_observation",
    "project_base_point",
]
