"""最小可部署 Observation V2 的坐标、同步 Frame 与四步历史契约。"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Final

import numpy as np

from robot_vla.contracts import (
    OBSERVATION_HISTORY_LENGTH,
    OBSERVATION_HISTORY_STRIDE_CONTROL_STEPS,
    OBSERVATION_V2_VERSION,
    RobotSpec,
)

CAMERA_FRAME_CONVENTION: Final = "opencv-optical-x-right-y-down-z-forward"
OBSERVATION_MODALITIES: Final = (
    "rgb_external",
    "rgb_wrist",
    "proprio",
    "tcp_pose",
    "wrist_camera_pose",
    "finger_force",
)
GL_CAMERA_FROM_CV_CAMERA: Final = np.diag((1.0, -1.0, -1.0, 1.0)).astype(
    np.float64
)

# proprio(15) + TCP position/rotation6d(9) + wrist position/rotation6d(9)
# + F_L/F_R(2) + frame age(1) + per-modality validity(6)
OBSERVATION_V2_FRAME_STATE_DIM: Final = 42
# previous_command_q(7) + tracking_error(7) + previous_action(8) + 两个 valid bit
OBSERVATION_V2_CONTROLLER_STATE_DIM: Final = 24


def validate_se3(transform: np.ndarray, name: str = "transform") -> np.ndarray:
    """验证并返回 float64 SE(3)，拒绝缩放、反射和非齐次矩阵。"""

    value = np.asarray(transform, dtype=np.float64)
    if value.shape != (4, 4) or not np.isfinite(value).all():
        raise ValueError(f"{name} 必须是有限 [4,4] 矩阵")
    if not np.allclose(value[3], (0.0, 0.0, 0.0, 1.0), rtol=0.0, atol=1e-6):
        raise ValueError(f"{name} 最后一行必须是 [0,0,0,1]")
    rotation = value[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-5):
        raise ValueError(f"{name} rotation 必须正交")
    if not math.isclose(float(np.linalg.det(rotation)), 1.0, rel_tol=0.0, abs_tol=1e-5):
        raise ValueError(f"{name} rotation determinant 必须为 +1")
    return value


def invert_se3(transform: np.ndarray, name: str = "transform") -> np.ndarray:
    """利用刚体结构求逆，避免通用矩阵逆隐藏非刚体输入。"""

    value = validate_se3(transform, name)
    rotation = value[:3, :3]
    translation = value[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -(rotation.T @ translation)
    return inverse


def opengl_camera_to_opencv(world_from_camera_gl: np.ndarray) -> np.ndarray:
    """把 ManiSkill ``cam2world_gl`` 转为同一光心的 OpenCV optical frame。

    记号 ``A_from_B`` 表示把 B-frame 坐标变换到 A-frame：

    ``world_from_camera_cv = world_from_camera_gl @ camera_gl_from_camera_cv``。
    """

    world_from_gl = validate_se3(world_from_camera_gl, "world_from_camera_gl")
    return validate_se3(
        world_from_gl @ GL_CAMERA_FROM_CV_CAMERA,
        "world_from_camera_cv",
    )


def rotation_matrix_to_6d(rotation: np.ndarray) -> np.ndarray:
    """按前两列顺序 ``[r00,r10,r20,r01,r11,r21]`` 编码 Rotation-6D。"""

    value = np.asarray(rotation, dtype=np.float64)
    if value.shape != (3, 3):
        raise ValueError("rotation 必须是 [3,3]")
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, :3] = value
    validate_se3(candidate, "rotation")
    return value[:, :2].T.reshape(-1).astype(np.float32)


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """用 Gram-Schmidt 解码项目固定的列优先 Rotation-6D。"""

    value = np.asarray(rotation_6d, dtype=np.float64)
    if value.shape != (6,) or not np.isfinite(value).all():
        raise ValueError("rotation_6d 必须是有限 [6] 向量")
    first = value[:3]
    second = value[3:]
    first_norm = float(np.linalg.norm(first))
    if first_norm <= 1e-8:
        raise ValueError("rotation_6d 第一轴退化")
    first = first / first_norm
    second = second - float(np.dot(first, second)) * first
    second_norm = float(np.linalg.norm(second))
    if second_norm <= 1e-8:
        raise ValueError("rotation_6d 第二轴与第一轴共线")
    second = second / second_norm
    third = np.cross(first, second)
    rotation = np.column_stack((first, second, third))
    candidate = np.eye(4, dtype=np.float64)
    candidate[:3, :3] = rotation
    validate_se3(candidate, "decoded_rotation")
    return rotation.astype(np.float32)


def transform_to_position_rotation_6d(transform: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    value = validate_se3(transform)
    return (
        value[:3, 3].astype(np.float32),
        rotation_matrix_to_6d(value[:3, :3]),
    )


def _validated_rgb(value: np.ndarray, name: str) -> np.ndarray:
    image = np.asarray(value)
    if image.ndim != 3 or image.shape[-1] != 3 or min(image.shape[:2]) <= 0:
        raise ValueError(f"{name} 必须是 [H,W,3]")
    if image.dtype != np.uint8:
        raise ValueError(f"{name} dtype 必须是 uint8")
    return np.ascontiguousarray(image).copy()


def _validated_float32(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是有限 float32 {shape} 数组")
    return np.ascontiguousarray(array).copy()


@dataclass(frozen=True)
class ObservationV2Frame:
    """一个控制 Tick 的同步可部署状态；不包含 object GT 或 predicate truth。"""

    rgb_external: np.ndarray
    rgb_wrist: np.ndarray
    physical_proprio: np.ndarray
    base_from_tcp: np.ndarray
    base_from_wrist_camera: np.ndarray
    finger_force_n: np.ndarray
    timestamp_s: float
    modality_timestamp_s: np.ndarray
    modality_valid: np.ndarray
    version: str = OBSERVATION_V2_VERSION

    def __post_init__(self) -> None:
        if self.version != OBSERVATION_V2_VERSION:
            raise ValueError(f"Observation version 必须是 {OBSERVATION_V2_VERSION}")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s 必须是有限非负数")
        object.__setattr__(
            self,
            "rgb_external",
            _validated_rgb(self.rgb_external, "rgb_external"),
        )
        object.__setattr__(self, "rgb_wrist", _validated_rgb(self.rgb_wrist, "rgb_wrist"))
        proprio = np.asarray(self.physical_proprio)
        if proprio.ndim != 1 or proprio.dtype != np.float32 or not np.isfinite(proprio).all():
            raise ValueError("physical_proprio 必须是一维有限 float32")
        object.__setattr__(self, "physical_proprio", proprio.copy())
        object.__setattr__(
            self,
            "base_from_tcp",
            validate_se3(self.base_from_tcp, "base_from_tcp").astype(np.float32),
        )
        object.__setattr__(
            self,
            "base_from_wrist_camera",
            validate_se3(
                self.base_from_wrist_camera,
                "base_from_wrist_camera",
            ).astype(np.float32),
        )
        force = _validated_float32(self.finger_force_n, (2,), "finger_force_n")
        if np.any(force < 0.0):
            raise ValueError("finger_force_n 必须非负")
        object.__setattr__(self, "finger_force_n", force)
        timestamps = np.asarray(self.modality_timestamp_s)
        expected = (len(OBSERVATION_MODALITIES),)
        if (
            timestamps.shape != expected
            or timestamps.dtype != np.float64
            or not np.isfinite(timestamps).all()
            or np.any(timestamps < 0.0)
        ):
            raise ValueError(f"modality_timestamp_s 必须是有限非负 float64 {expected}")
        valid = np.asarray(self.modality_valid)
        if valid.shape != expected or valid.dtype != np.bool_:
            raise ValueError(f"modality_valid 必须是 bool {expected}")
        if np.any(timestamps[valid] > self.timestamp_s + 1e-9):
            raise ValueError("有效 modality 不能使用晚于控制 Tick 的未来观测")
        object.__setattr__(self, "modality_timestamp_s", timestamps.copy())
        object.__setattr__(self, "modality_valid", valid.copy())


@dataclass(frozen=True)
class ObservationV2Window:
    """严格 oldest→newest 的 4 个连续控制步，以及当前 controller reference。"""

    rgb_external: np.ndarray
    rgb_wrist: np.ndarray
    physical_proprio: np.ndarray
    tcp_position: np.ndarray
    tcp_rotation_6d: np.ndarray
    wrist_position: np.ndarray
    wrist_rotation_6d: np.ndarray
    finger_force_n: np.ndarray
    frame_age_s: np.ndarray
    modality_age_s: np.ndarray
    frame_timestamp_s: np.ndarray
    modality_timestamp_s: np.ndarray
    history_valid: np.ndarray
    modality_valid: np.ndarray
    previous_command_q: np.ndarray
    tracking_error: np.ndarray
    previous_action: np.ndarray
    controller_valid: np.ndarray
    instruction: str
    timestamp_s: float
    version: str = OBSERVATION_V2_VERSION

    @property
    def history_length(self) -> int:
        return int(self.history_valid.shape[0])

    def validate(
        self,
        spec: RobotSpec,
        *,
        require_current_complete: bool = False,
    ) -> None:
        """验证在线 Window 与训练 Dataset 使用同一零填充和有效性语义。"""

        if self.version != OBSERVATION_V2_VERSION:
            raise ValueError(f"Observation version 必须是 {OBSERVATION_V2_VERSION}")
        if not isinstance(self.instruction, str) or not self.instruction.strip():
            raise ValueError("instruction 必须是非空字符串")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("Observation V2 timestamp_s 必须是有限非负数")
        if self.history_length != OBSERVATION_HISTORY_LENGTH:
            raise ValueError(
                f"Observation V2 history_length 必须为 {OBSERVATION_HISTORY_LENGTH}"
            )
        for value, name in (
            (self.rgb_external, "rgb_external"),
            (self.rgb_wrist, "rgb_wrist"),
        ):
            images = np.asarray(value)
            if (
                images.ndim != 4
                or images.shape[0] != self.history_length
                or images.shape[-1] != 3
                or min(images.shape[1:3]) <= 0
                or images.dtype != np.uint8
            ):
                raise ValueError(f"{name} 必须是 uint8 [4,H,W,3]")

        expected_float32 = (
            (self.physical_proprio, (self.history_length, spec.proprio_dim), "physical_proprio"),
            (self.tcp_position, (self.history_length, 3), "tcp_position"),
            (self.tcp_rotation_6d, (self.history_length, 6), "tcp_rotation_6d"),
            (self.wrist_position, (self.history_length, 3), "wrist_position"),
            (self.wrist_rotation_6d, (self.history_length, 6), "wrist_rotation_6d"),
            (self.finger_force_n, (self.history_length, 2), "finger_force_n"),
            (self.frame_age_s, (self.history_length,), "frame_age_s"),
            (
                self.modality_age_s,
                (self.history_length, len(OBSERVATION_MODALITIES)),
                "modality_age_s",
            ),
            (self.previous_command_q, (spec.arm_dof,), "previous_command_q"),
            (self.tracking_error, (spec.arm_dof,), "tracking_error"),
            (self.previous_action, (spec.action_dim,), "previous_action"),
        )
        for value, shape, name in expected_float32:
            array = np.asarray(value)
            if array.shape != shape or array.dtype != np.float32 or not np.isfinite(array).all():
                raise ValueError(f"{name} 必须是有限 float32 {shape}")
        if np.any(self.finger_force_n < 0.0):
            raise ValueError("finger_force_n 必须非负")
        if np.any(self.frame_age_s < 0.0) or np.any(self.modality_age_s < 0.0):
            raise ValueError("Observation V2 age 必须非负")
        frame_timestamp = np.asarray(self.frame_timestamp_s)
        modality_timestamp = np.asarray(self.modality_timestamp_s)
        if (
            frame_timestamp.shape != (self.history_length,)
            or frame_timestamp.dtype != np.float64
            or not np.isfinite(frame_timestamp).all()
            or np.any(frame_timestamp < 0.0)
        ):
            raise ValueError("frame_timestamp_s 必须是有限非负 float64 [4]")
        if (
            modality_timestamp.shape
            != (self.history_length, len(OBSERVATION_MODALITIES))
            or modality_timestamp.dtype != np.float64
            or not np.isfinite(modality_timestamp).all()
            or np.any(modality_timestamp < 0.0)
        ):
            raise ValueError("modality_timestamp_s 必须是有限非负 float64 [4,6]")

        history_valid = np.asarray(self.history_valid)
        modality_valid = np.asarray(self.modality_valid)
        controller_valid = np.asarray(self.controller_valid)
        if history_valid.shape != (self.history_length,) or history_valid.dtype != np.bool_:
            raise ValueError("history_valid 必须是 bool [4]")
        if not bool(history_valid[-1]):
            raise ValueError("Observation V2 当前控制步必须有效")
        first_valid = int(np.flatnonzero(history_valid)[0])
        expected_history = np.arange(self.history_length) >= first_valid
        if not np.array_equal(history_valid, expected_history):
            raise ValueError("Observation V2 padding 只能是 history 的连续前缀")
        if (
            modality_valid.shape
            != (self.history_length, len(OBSERVATION_MODALITIES))
            or modality_valid.dtype != np.bool_
        ):
            raise ValueError("modality_valid 必须是 bool [4,6]")
        if np.any(modality_valid[~history_valid]):
            raise ValueError("padding history 不能声明有效 modality")
        if require_current_complete and not bool(modality_valid[-1].all()):
            raise ValueError("Observation V2 当前控制步必须六模态完整有效")
        if controller_valid.shape != (2,) or controller_valid.dtype != np.bool_:
            raise ValueError("controller_valid 必须是 bool [2]")

        # Dataset 与 Runtime 都必须对 padding/无效模态使用物理零，防止陈旧值泄漏。
        padding_rows = ~history_valid
        for value, name in (
            (self.rgb_external, "rgb_external"),
            (self.rgb_wrist, "rgb_wrist"),
            (self.physical_proprio, "physical_proprio"),
            (self.tcp_position, "tcp_position"),
            (self.tcp_rotation_6d, "tcp_rotation_6d"),
            (self.wrist_position, "wrist_position"),
            (self.wrist_rotation_6d, "wrist_rotation_6d"),
            (self.finger_force_n, "finger_force_n"),
            (self.frame_age_s, "frame_age_s"),
            (self.modality_age_s, "modality_age_s"),
            (self.frame_timestamp_s, "frame_timestamp_s"),
            (self.modality_timestamp_s, "modality_timestamp_s"),
        ):
            if np.any(np.asarray(value)[padding_rows] != 0):
                raise ValueError(f"{name} 的 padding history 必须为零")
        modality_values = (
            (self.rgb_external, 0, "rgb_external"),
            (self.rgb_wrist, 1, "rgb_wrist"),
            (self.physical_proprio, 2, "physical_proprio"),
            ((self.tcp_position, self.tcp_rotation_6d), 3, "tcp_pose"),
            ((self.wrist_position, self.wrist_rotation_6d), 4, "wrist_camera_pose"),
            (self.finger_force_n, 5, "finger_force_n"),
        )
        for value, modality_index, name in modality_values:
            invalid = history_valid & ~modality_valid[:, modality_index]
            values = value if isinstance(value, tuple) else (value,)
            if any(np.any(np.asarray(item)[invalid] != 0) for item in values):
                raise ValueError(f"无效 {name} 必须使用零值")

        valid_indices = np.flatnonzero(history_valid)
        expected_age = (
            (self.history_length - 1 - valid_indices)
            * OBSERVATION_HISTORY_STRIDE_CONTROL_STEPS
            / spec.control_hz
        )
        tolerance = max(1e-6, 0.2 / spec.control_hz)
        if not np.allclose(
            self.frame_age_s[valid_indices],
            expected_age,
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("frame_age_s 与连续控制步历史不一致")
        if np.any(
            self.modality_age_s[modality_valid]
            + tolerance
            < np.broadcast_to(self.frame_age_s[:, None], self.modality_age_s.shape)[
                modality_valid
            ]
        ):
            raise ValueError("有效 modality 的时间戳不能晚于所属控制步")
        if np.any(self.modality_age_s[~modality_valid] != 0):
            raise ValueError("无效 modality 的 age 必须使用零值")
        if np.any(self.modality_timestamp_s[~modality_valid] != 0):
            raise ValueError("无效 modality 的 timestamp 必须使用零值")
        if np.any(self.frame_timestamp_s[~history_valid] != 0):
            raise ValueError("padding history 的 frame timestamp 必须使用零值")
        if np.any(self.frame_timestamp_s[valid_indices] > self.timestamp_s + 1e-9):
            raise ValueError("frame timestamp 不能晚于 Window timestamp")
        if np.any(
            self.modality_timestamp_s[modality_valid] > self.timestamp_s + 1e-9
        ):
            raise ValueError("modality timestamp 不能晚于 Window timestamp")
        frame_timestamp_grid = np.broadcast_to(
            self.frame_timestamp_s[:, None],
            self.modality_timestamp_s.shape,
        )
        if np.any(
            self.modality_timestamp_s[modality_valid]
            > frame_timestamp_grid[modality_valid] + 1e-9
        ):
            raise ValueError("modality timestamp 不能晚于所属 frame timestamp")
        if not np.allclose(
            self.timestamp_s - self.frame_timestamp_s[valid_indices],
            self.frame_age_s[valid_indices],
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("frame timestamp 与 age 不一致")
        if not np.allclose(
            self.timestamp_s - self.modality_timestamp_s[modality_valid],
            self.modality_age_s[modality_valid],
            rtol=0.0,
            atol=tolerance,
        ):
            raise ValueError("modality timestamp 与 age 不一致")
        for index in valid_indices:
            if modality_valid[index, 3]:
                rotation_6d_to_matrix(self.tcp_rotation_6d[index])
            if modality_valid[index, 4]:
                rotation_6d_to_matrix(self.wrist_rotation_6d[index])

        if controller_valid[0]:
            limits = np.asarray(spec.joint_position_limits_rad, dtype=np.float32)
            if np.any(self.previous_command_q < limits[:, 0]) or np.any(
                self.previous_command_q > limits[:, 1]
            ):
                raise ValueError("previous_command_q 超出 Franka 关节位置限制")
            expected_tracking = (
                self.previous_command_q
                - self.physical_proprio[-1, : spec.arm_dof]
            )
            if not np.allclose(
                self.tracking_error,
                expected_tracking,
                rtol=0.0,
                atol=1e-6,
            ):
                raise ValueError("tracking_error 与 command reference/actual q 不一致")
        elif np.any(self.previous_command_q != 0) or np.any(self.tracking_error != 0):
            raise ValueError("无效 command reference 必须使用零值")
        if controller_valid[1]:
            arm_delta = self.previous_action[: spec.arm_dof]
            limits = np.asarray(spec.effective_joint_delta_limits_rad, dtype=np.float32)
            if np.any(np.abs(arm_delta) > limits + 1e-6) or not (
                0.0 <= float(self.previous_action[-1]) <= 1.0
            ):
                raise ValueError("previous_action 超出物理 Action 契约")
        elif np.any(self.previous_action != 0):
            raise ValueError("无效 previous_action 必须使用零值")

    def frame_state(
        self,
        normalized_proprio: np.ndarray,
        normalized_finger_force: np.ndarray,
    ) -> np.ndarray:
        """组装模型每帧数值状态；无效模态由 validity 明确标记。"""

        proprio = np.asarray(normalized_proprio)
        expected = (self.history_length, self.physical_proprio.shape[-1])
        if proprio.shape != expected or proprio.dtype != np.float32:
            raise ValueError(f"normalized_proprio 必须是 float32 {expected}")
        force = np.asarray(normalized_finger_force)
        force_expected = (self.history_length, 2)
        if (
            force.shape != force_expected
            or force.dtype != np.float32
            or not np.isfinite(force).all()
            or np.any(force < 0.0)
        ):
            raise ValueError(
                f"normalized_finger_force 必须是有限非负 float32 {force_expected}"
            )
        state = np.concatenate(
            (
                proprio,
                self.tcp_position,
                self.tcp_rotation_6d,
                self.wrist_position,
                self.wrist_rotation_6d,
                force,
                self.frame_age_s[:, None],
                self.modality_valid.astype(np.float32),
            ),
            axis=-1,
        ).astype(np.float32, copy=False)
        if state.shape != (self.history_length, OBSERVATION_V2_FRAME_STATE_DIM):
            raise RuntimeError(f"Observation V2 frame state 维度漂移: {state.shape}")
        return state

    def controller_state(self) -> np.ndarray:
        state = np.concatenate(
            (
                self.previous_command_q,
                self.tracking_error,
                self.previous_action,
                self.controller_valid.astype(np.float32),
            )
        ).astype(np.float32, copy=False)
        if state.shape != (OBSERVATION_V2_CONTROLLER_STATE_DIM,):
            raise RuntimeError(f"Observation V2 controller state 维度漂移: {state.shape}")
        return state


class ObservationV2History:
    """Episode-local ring buffer；reset 是唯一允许的跨 Episode 边界。"""

    def __init__(
        self,
        spec: RobotSpec,
        *,
        history_length: int = OBSERVATION_HISTORY_LENGTH,
    ) -> None:
        if history_length != OBSERVATION_HISTORY_LENGTH:
            raise ValueError(f"Observation V2 history_length 必须固定为 {OBSERVATION_HISTORY_LENGTH}")
        self.spec = spec
        self.history_length = history_length
        self._frames: deque[ObservationV2Frame] = deque(maxlen=history_length)

    def reset(self) -> None:
        self._frames.clear()

    def append(self, frame: ObservationV2Frame) -> None:
        if frame.physical_proprio.shape != (self.spec.proprio_dim,):
            raise ValueError(
                f"physical_proprio 应为 [{self.spec.proprio_dim}]，"
                f"实际为 {frame.physical_proprio.shape}"
            )
        if self._frames:
            expected_dt = OBSERVATION_HISTORY_STRIDE_CONTROL_STEPS / self.spec.control_hz
            actual_dt = frame.timestamp_s - self._frames[-1].timestamp_s
            if actual_dt <= 0.0:
                raise ValueError("Observation V2 timestamp 必须严格递增")
            if abs(actual_dt - expected_dt) > max(1e-6, expected_dt * 0.2):
                raise ValueError("Observation V2 history 必须来自连续控制步")
            if frame.rgb_external.shape != self._frames[-1].rgb_external.shape:
                raise ValueError("Episode 内 external RGB shape 不能变化")
            if frame.rgb_wrist.shape != self._frames[-1].rgb_wrist.shape:
                raise ValueError("Episode 内 wrist RGB shape 不能变化")
        self._frames.append(frame)

    def snapshot(
        self,
        instruction: str,
        *,
        previous_command_q: np.ndarray | None = None,
        previous_action: np.ndarray | None = None,
    ) -> ObservationV2Window:
        if not isinstance(instruction, str) or not instruction.strip():
            raise ValueError("instruction 必须是非空字符串")
        if not self._frames:
            raise ValueError("Observation V2 history 为空")
        newest = self._frames[-1]
        pad_count = self.history_length - len(self._frames)
        frames: list[ObservationV2Frame | None] = [None] * pad_count + list(self._frames)
        external = np.zeros(
            (self.history_length, *newest.rgb_external.shape),
            dtype=np.uint8,
        )
        wrist = np.zeros(
            (self.history_length, *newest.rgb_wrist.shape),
            dtype=np.uint8,
        )
        proprio = np.zeros((self.history_length, self.spec.proprio_dim), dtype=np.float32)
        tcp_position = np.zeros((self.history_length, 3), dtype=np.float32)
        tcp_rotation = np.zeros((self.history_length, 6), dtype=np.float32)
        wrist_position = np.zeros((self.history_length, 3), dtype=np.float32)
        wrist_rotation = np.zeros((self.history_length, 6), dtype=np.float32)
        forces = np.zeros((self.history_length, 2), dtype=np.float32)
        frame_age = np.zeros(self.history_length, dtype=np.float32)
        modality_age = np.zeros(
            (self.history_length, len(OBSERVATION_MODALITIES)),
            dtype=np.float32,
        )
        frame_timestamp = np.zeros(self.history_length, dtype=np.float64)
        modality_timestamp = np.zeros(
            (self.history_length, len(OBSERVATION_MODALITIES)),
            dtype=np.float64,
        )
        history_valid = np.zeros(self.history_length, dtype=np.bool_)
        modality_valid = np.zeros_like(modality_age, dtype=np.bool_)
        for index, frame in enumerate(frames):
            if frame is None:
                continue
            if frame.modality_valid[0]:
                external[index] = frame.rgb_external
            if frame.modality_valid[1]:
                wrist[index] = frame.rgb_wrist
            if frame.modality_valid[2]:
                proprio[index] = frame.physical_proprio
            if frame.modality_valid[3]:
                tcp_position[index], tcp_rotation[index] = (
                    transform_to_position_rotation_6d(frame.base_from_tcp)
                )
            if frame.modality_valid[4]:
                wrist_position[index], wrist_rotation[index] = (
                    transform_to_position_rotation_6d(frame.base_from_wrist_camera)
                )
            if frame.modality_valid[5]:
                forces[index] = frame.finger_force_n
            frame_age[index] = np.float32(newest.timestamp_s - frame.timestamp_s)
            modality_age[index, frame.modality_valid] = (
                newest.timestamp_s
                - frame.modality_timestamp_s[frame.modality_valid]
            ).astype(np.float32)
            frame_timestamp[index] = frame.timestamp_s
            modality_timestamp[index, frame.modality_valid] = (
                frame.modality_timestamp_s[frame.modality_valid]
            )
            history_valid[index] = True
            modality_valid[index] = frame.modality_valid

        command = np.zeros(self.spec.arm_dof, dtype=np.float32)
        tracking = np.zeros(self.spec.arm_dof, dtype=np.float32)
        action = np.zeros(self.spec.action_dim, dtype=np.float32)
        controller_valid = np.zeros(2, dtype=np.bool_)
        if previous_command_q is not None:
            command = _validated_float32(
                previous_command_q,
                (self.spec.arm_dof,),
                "previous_command_q",
            )
            tracking = (command - newest.physical_proprio[: self.spec.arm_dof]).astype(
                np.float32,
                copy=False,
            )
            controller_valid[0] = True
        if previous_action is not None:
            action = _validated_float32(
                previous_action,
                (self.spec.action_dim,),
                "previous_action",
            )
            controller_valid[1] = True
        window = ObservationV2Window(
            rgb_external=external,
            rgb_wrist=wrist,
            physical_proprio=proprio,
            tcp_position=tcp_position,
            tcp_rotation_6d=tcp_rotation,
            wrist_position=wrist_position,
            wrist_rotation_6d=wrist_rotation,
            finger_force_n=forces,
            frame_age_s=frame_age,
            modality_age_s=modality_age,
            frame_timestamp_s=frame_timestamp,
            modality_timestamp_s=modality_timestamp,
            history_valid=history_valid,
            modality_valid=modality_valid,
            previous_command_q=command,
            tracking_error=tracking,
            previous_action=action,
            controller_valid=controller_valid,
            instruction=instruction,
            timestamp_s=newest.timestamp_s,
        )
        window.validate(self.spec)
        return window


__all__ = [
    "CAMERA_FRAME_CONVENTION",
    "GL_CAMERA_FROM_CV_CAMERA",
    "OBSERVATION_MODALITIES",
    "OBSERVATION_V2_CONTROLLER_STATE_DIM",
    "OBSERVATION_V2_FRAME_STATE_DIM",
    "ObservationV2Frame",
    "ObservationV2History",
    "ObservationV2Window",
    "invert_se3",
    "opengl_camera_to_opencv",
    "rotation_6d_to_matrix",
    "rotation_matrix_to_6d",
    "transform_to_position_rotation_6d",
    "validate_se3",
]
