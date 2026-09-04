"""E018-P1 front-camera provider 的资格验证专用输入适配器。

该模块不把既有 wrist provider 改名为 front provider。它只定义一个显式、可审计的
camera-role substitution：使用 ``base_camera`` RGB，并把同一观测中的 actual external
camera pose 写入冻结模型原有的 camera-pose 数值槽。输出仍是 qualification-only，不能
进入 Object Memory、Executive 或 actuator。
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass

import numpy as np

from robot_vla.adapters import FingerForceNormalizer, ProprioNormalizer
from robot_vla.contracts import RobotSpec
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    OBSERVATION_V2_FRAME_STATE_DIM,
    transform_to_position_rotation_6d,
    validate_se3,
)
from robot_vla.precision.active_external_observation import ACTUAL_EXTERNAL_POSE_SOURCE
from robot_vla.precision.active_front_camera import ExternalCameraMotionState
from robot_vla.precision.provider import PrecisionGeometricMotionInput

ACTIVE_FRONT_PROVIDER_ADAPTER_VERSION = "e018-p1-front-provider-role-adapter/v1"
ACTIVE_FRONT_PROVIDER_IDENTITY_VERSION = "e018-p1-front-provider-identity/v1"
ACTIVE_FRONT_MODEL_INPUT_VERSION = "e018-p1-front-provider-model-input/v1"
FRONT_PROVIDER_ROLE_SUBSTITUTION_SEMANTICS = (
    "base-camera-rgb-and-actual-external-pose-in-model-camera-slots/v1"
)
FRONT_PROVIDER_EXECUTION_MODE = "qualification-only/no-memory/no-actuation/v1"
FRONT_PROVIDER_FRAME_CONVENTION = "robot-base-from-opencv-optical-camera/v1"
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _array_sha256(value: np.ndarray) -> str:
    array = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    dtype = str(array.dtype).encode("ascii")
    digest.update(len(dtype).to_bytes(2, "big"))
    digest.update(dtype)
    digest.update(len(array.shape).to_bytes(2, "big"))
    for dimension in array.shape:
        digest.update(int(dimension).to_bytes(8, "big", signed=False))
    digest.update(array.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class ActiveFrontProviderAdapterConfig:
    """冻结 camera-role substitution；首版只能用于资格验证。"""

    source_training_camera: str = "hand_camera"
    target_camera: str = "base_camera"
    role_substitution_semantics: str = FRONT_PROVIDER_ROLE_SUBSTITUTION_SEMANTICS
    maximum_rgb_pose_skew_s: float = 0.01
    require_collect_state: bool = True
    require_settled: bool = True
    qualification_only: bool = True
    memory_write_allowed: bool = False
    actuation_allowed: bool = False
    version: str = ACTIVE_FRONT_PROVIDER_ADAPTER_VERSION

    def __post_init__(self) -> None:
        if self.source_training_camera != "hand_camera":
            raise ValueError("当前冻结 checkpoint 的训练相机必须为 hand_camera")
        if self.target_camera != "base_camera":
            raise ValueError("E018-P1 首版 active front target 必须为 base_camera")
        if self.role_substitution_semantics != FRONT_PROVIDER_ROLE_SUBSTITUTION_SEMANTICS:
            raise ValueError("front provider camera-role substitution 语义漂移")
        if (
            not math.isfinite(self.maximum_rgb_pose_skew_s)
            or self.maximum_rgb_pose_skew_s < 0.0
        ):
            raise ValueError("maximum_rgb_pose_skew_s 必须是有限非负数")
        for name in (
            "require_collect_state",
            "require_settled",
            "qualification_only",
            "memory_write_allowed",
            "actuation_allowed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须是 bool")
        if not self.require_collect_state or not self.require_settled:
            raise ValueError("首版 front provider 输入必须固定为 COLLECT + settled")
        if not self.qualification_only or self.memory_write_allowed or self.actuation_allowed:
            raise ValueError("未资格化 front provider 必须禁止 Memory write 与 actuation")
        if self.version != ACTIVE_FRONT_PROVIDER_ADAPTER_VERSION:
            raise ValueError("front provider adapter version 漂移")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ActiveFrontProviderIdentity:
    """把冻结权重、训练域和显式 front adapter 绑定成新 identity。"""

    checkpoint_sha256: str
    checkpoint_parameter_sha256: str
    checkpoint_provenance_sha256: str
    model_config_sha256: str
    proprio_stats_sha256: str
    proprio_normalizer_sha256: str
    finger_force_stats_sha256: str
    finger_force_normalizer_sha256: str
    adapter_config_sha256: str
    primitive_id: str
    calibration_identity_sha256: str
    geometric_motion_provider_id: str
    source_training_camera: str
    target_camera: str
    frame_convention: str = FRONT_PROVIDER_FRAME_CONVENTION
    execution_mode: str = FRONT_PROVIDER_EXECUTION_MODE
    version: str = ACTIVE_FRONT_PROVIDER_IDENTITY_VERSION

    def __post_init__(self) -> None:
        for name in (
            "checkpoint_sha256",
            "checkpoint_parameter_sha256",
            "checkpoint_provenance_sha256",
            "model_config_sha256",
            "proprio_stats_sha256",
            "proprio_normalizer_sha256",
            "finger_force_stats_sha256",
            "finger_force_normalizer_sha256",
            "adapter_config_sha256",
            "calibration_identity_sha256",
        ):
            _require_sha256(getattr(self, name), name)
        for name in ("primitive_id", "geometric_motion_provider_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} 必须是非空字符串")
        if self.source_training_camera != "hand_camera":
            raise ValueError("front provider identity 必须保留真实 hand_camera 训练来源")
        if self.target_camera != "base_camera":
            raise ValueError("front provider identity target_camera 必须为 base_camera")
        if self.frame_convention != FRONT_PROVIDER_FRAME_CONVENTION:
            raise ValueError("front provider frame convention 漂移")
        if self.execution_mode != FRONT_PROVIDER_EXECUTION_MODE:
            raise ValueError("未资格化 front provider execution mode 漂移")
        if self.version != ACTIVE_FRONT_PROVIDER_IDENTITY_VERSION:
            raise ValueError("front provider identity version 漂移")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class ActiveFrontModelInput:
    """不含 GT 的单帧 front qualification 输入与同源几何审计。"""

    episode_id: str
    request_id: str
    observation_sequence_id: str
    primitive_id: str
    source_camera: str
    actual_pose_source: str
    camera_motion_state: ExternalCameraMotionState
    settled: bool
    control_timestamp_s: float
    rgb_timestamp_s: float
    camera_pose_timestamp_s: float
    tcp_pose_timestamp_s: float
    rgb_external: np.ndarray
    structured_state: np.ndarray
    geometric_motion: np.ndarray
    geometric_motion_timestamp_s: float
    geometric_motion_provider_id: str
    base_from_external_camera_cv: np.ndarray
    intrinsic_cv: np.ndarray
    qualification_only: bool = True
    memory_write_eligible: bool = False
    version: str = ACTIVE_FRONT_MODEL_INPUT_VERSION

    def __post_init__(self) -> None:
        for name in ("episode_id", "request_id", "observation_sequence_id", "primitive_id"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"{name} 必须是非空字符串")
        if self.source_camera != "base_camera":
            raise ValueError("front provider model input source_camera 必须是 base_camera")
        if self.actual_pose_source != ACTUAL_EXTERNAL_POSE_SOURCE:
            raise ValueError("front provider 必须使用同一 observation 的 actual external pose")
        if self.camera_motion_state is not ExternalCameraMotionState.COLLECT or not self.settled:
            raise ValueError("front provider model input 只接受 settled COLLECT frame")
        for name in (
            "control_timestamp_s",
            "rgb_timestamp_s",
            "camera_pose_timestamp_s",
            "tcp_pose_timestamp_s",
            "geometric_motion_timestamp_s",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须是有限非负数")
        image = np.asarray(self.rgb_external)
        if (
            image.ndim != 3
            or image.shape[-1] != 3
            or min(image.shape[:2]) <= 0
            or image.dtype != np.uint8
        ):
            raise ValueError("rgb_external 必须是 uint8 [H,W,3]")
        state = np.asarray(self.structured_state)
        if (
            state.shape != (OBSERVATION_V2_FRAME_STATE_DIM,)
            or state.dtype != np.float32
            or not np.isfinite(state).all()
        ):
            raise ValueError("structured_state 与冻结 Precision model 不一致")
        motion = np.asarray(self.geometric_motion)
        if (
            motion.shape != (4,)
            or motion.dtype != np.float32
            or not np.isfinite(motion).all()
        ):
            raise ValueError("qualification geometric_motion 必须是有限 float32 [4]")
        if (
            not isinstance(self.geometric_motion_provider_id, str)
            or not self.geometric_motion_provider_id.strip()
        ):
            raise ValueError("geometric_motion_provider_id 必须是非空字符串")
        transform = validate_se3(
            self.base_from_external_camera_cv,
            "base_from_external_camera_cv",
        )
        intrinsic = np.asarray(self.intrinsic_cv, dtype=np.float64)
        if (
            intrinsic.shape != (3, 3)
            or not np.isfinite(intrinsic).all()
            or intrinsic[0, 0] <= 0.0
            or intrinsic[1, 1] <= 0.0
            or not np.allclose(intrinsic[2], (0.0, 0.0, 1.0), rtol=0.0, atol=1e-7)
        ):
            raise ValueError("intrinsic_cv 必须是有效 OpenCV [3,3]")
        if not self.qualification_only or self.memory_write_eligible:
            raise ValueError("未资格化 front input 不得成为 Memory write 候选")
        if self.version != ACTIVE_FRONT_MODEL_INPUT_VERSION:
            raise ValueError("front provider model input version 漂移")
        object.__setattr__(self, "rgb_external", np.ascontiguousarray(image).copy())
        object.__setattr__(self, "structured_state", np.ascontiguousarray(state).copy())
        object.__setattr__(self, "geometric_motion", np.ascontiguousarray(motion).copy())
        object.__setattr__(
            self,
            "base_from_external_camera_cv",
            transform.astype(np.float64, copy=True),
        )
        object.__setattr__(self, "intrinsic_cv", intrinsic.copy())

    @property
    def timestamp_skew_s(self) -> float:
        values = (
            self.control_timestamp_s,
            self.rgb_timestamp_s,
            self.camera_pose_timestamp_s,
            self.tcp_pose_timestamp_s,
            self.geometric_motion_timestamp_s,
        )
        return float(max(values) - min(values))

    @property
    def input_digest(self) -> str:
        return _canonical_sha256(
            {
                "version": self.version,
                "episode_id": self.episode_id,
                "request_id": self.request_id,
                "observation_sequence_id": self.observation_sequence_id,
                "primitive_id": self.primitive_id,
                "source_camera": self.source_camera,
                "actual_pose_source": self.actual_pose_source,
                "camera_motion_state": self.camera_motion_state.value,
                "settled": self.settled,
                "timestamps": {
                    "control": self.control_timestamp_s,
                    "rgb": self.rgb_timestamp_s,
                    "camera_pose": self.camera_pose_timestamp_s,
                    "tcp_pose": self.tcp_pose_timestamp_s,
                    "geometric_motion": self.geometric_motion_timestamp_s,
                },
                "geometric_motion_provider_id": self.geometric_motion_provider_id,
                "rgb_sha256": _array_sha256(self.rgb_external),
                "structured_state_sha256": _array_sha256(self.structured_state),
                "geometric_motion_sha256": _array_sha256(self.geometric_motion),
                "base_from_external_camera_cv_sha256": _array_sha256(
                    self.base_from_external_camera_cv
                ),
                "intrinsic_cv_sha256": _array_sha256(self.intrinsic_cv),
                "qualification_only": self.qualification_only,
                "memory_write_eligible": self.memory_write_eligible,
            }
        )


def build_active_front_model_input(
    *,
    spec: RobotSpec,
    proprio_normalizer: ProprioNormalizer,
    finger_force_normalizer: FingerForceNormalizer,
    config: ActiveFrontProviderAdapterConfig,
    episode_id: str,
    request_id: str,
    observation_sequence_id: str,
    primitive_id: str,
    rgb_external: np.ndarray,
    physical_proprio: np.ndarray,
    base_from_tcp: np.ndarray,
    base_from_external_camera_cv: np.ndarray,
    finger_force_n: np.ndarray,
    intrinsic_cv: np.ndarray,
    control_timestamp_s: float,
    rgb_timestamp_s: float,
    camera_pose_timestamp_s: float,
    tcp_pose_timestamp_s: float,
    geometric_motion: PrecisionGeometricMotionInput,
    geometric_motion_provider_id: str,
    camera_motion_state: ExternalCameraMotionState,
    settled: bool,
    actual_pose_source: str = ACTUAL_EXTERNAL_POSE_SOURCE,
) -> ActiveFrontModelInput:
    """构造明确 OOD 的 front 输入；不接触 GT，也不生成 Memory measurement。"""

    if not isinstance(spec, RobotSpec):
        raise TypeError("spec 必须是 RobotSpec")
    if not isinstance(proprio_normalizer, ProprioNormalizer):
        raise TypeError("proprio_normalizer 类型错误")
    if not isinstance(finger_force_normalizer, FingerForceNormalizer):
        raise TypeError("finger_force_normalizer 类型错误")
    if not isinstance(config, ActiveFrontProviderAdapterConfig):
        raise TypeError("config 类型错误")
    if camera_motion_state is not ExternalCameraMotionState.COLLECT:
        raise ValueError("front provider 只接受 COLLECT state")
    if not settled:
        raise ValueError("front provider 只接受 settled frame")
    if not isinstance(geometric_motion, PrecisionGeometricMotionInput):
        raise TypeError("geometric_motion 必须是 PrecisionGeometricMotionInput")
    if (
        not isinstance(geometric_motion_provider_id, str)
        or not geometric_motion_provider_id.strip()
    ):
        raise ValueError("geometric_motion_provider_id 必须是非空字符串")
    timestamps = np.asarray(
        (
            control_timestamp_s,
            rgb_timestamp_s,
            camera_pose_timestamp_s,
            tcp_pose_timestamp_s,
            geometric_motion.timestamp_s,
        ),
        dtype=np.float64,
    )
    if not np.isfinite(timestamps).all() or np.any(timestamps < 0.0):
        raise ValueError("front provider timestamps 必须有限非负")
    if float(timestamps.max() - timestamps.min()) > config.maximum_rgb_pose_skew_s + 1e-12:
        raise ValueError("front provider RGB/pose timestamp skew 超出预算")

    structured_state = build_precision_camera_role_state(
        spec=spec,
        proprio_normalizer=proprio_normalizer,
        finger_force_normalizer=finger_force_normalizer,
        physical_proprio=physical_proprio,
        base_from_tcp=base_from_tcp,
        base_from_camera_cv=base_from_external_camera_cv,
        finger_force_n=finger_force_n,
    )
    return ActiveFrontModelInput(
        episode_id=episode_id,
        request_id=request_id,
        observation_sequence_id=observation_sequence_id,
        primitive_id=primitive_id,
        source_camera=config.target_camera,
        actual_pose_source=actual_pose_source,
        camera_motion_state=camera_motion_state,
        settled=settled,
        control_timestamp_s=float(control_timestamp_s),
        rgb_timestamp_s=float(rgb_timestamp_s),
        camera_pose_timestamp_s=float(camera_pose_timestamp_s),
        tcp_pose_timestamp_s=float(tcp_pose_timestamp_s),
        geometric_motion_timestamp_s=float(geometric_motion.timestamp_s),
        geometric_motion_provider_id=geometric_motion_provider_id,
        rgb_external=rgb_external,
        structured_state=structured_state,
        geometric_motion=geometric_motion.as_array(),
        base_from_external_camera_cv=base_from_external_camera_cv,
        intrinsic_cv=intrinsic_cv,
    )


def build_precision_camera_role_state(
    *,
    spec: RobotSpec,
    proprio_normalizer: ProprioNormalizer,
    finger_force_normalizer: FingerForceNormalizer,
    physical_proprio: np.ndarray,
    base_from_tcp: np.ndarray,
    base_from_camera_cv: np.ndarray,
    finger_force_n: np.ndarray,
) -> np.ndarray:
    """构造冻结单帧状态；camera role 由调用方和外围 identity 显式声明。"""

    if not isinstance(spec, RobotSpec):
        raise TypeError("spec 必须是 RobotSpec")
    if not isinstance(proprio_normalizer, ProprioNormalizer):
        raise TypeError("proprio_normalizer 类型错误")
    if not isinstance(finger_force_normalizer, FingerForceNormalizer):
        raise TypeError("finger_force_normalizer 类型错误")
    proprio = np.asarray(physical_proprio)
    if (
        proprio.shape != (spec.proprio_dim,)
        or proprio.dtype != np.float32
        or not np.isfinite(proprio).all()
    ):
        raise ValueError(f"physical_proprio 必须是有限 float32 [{spec.proprio_dim}]")
    force = np.asarray(finger_force_n)
    if (
        force.shape != (2,)
        or force.dtype != np.float32
        or not np.isfinite(force).all()
        or np.any(force < 0.0)
    ):
        raise ValueError("finger_force_n 必须是有限非负 float32 [2]")
    tcp_position, tcp_rotation = transform_to_position_rotation_6d(base_from_tcp)
    camera_position, camera_rotation = transform_to_position_rotation_6d(
        base_from_camera_cv
    )
    normalized_proprio = proprio_normalizer.normalize(proprio).astype(
        np.float32,
        copy=False,
    )
    normalized_force = finger_force_normalizer.normalize(force).astype(
        np.float32,
        copy=False,
    )
    structured_state = np.concatenate(
        (
            normalized_proprio,
            tcp_position,
            tcp_rotation,
            camera_position,
            camera_rotation,
            normalized_force,
            np.zeros(1, dtype=np.float32),
            np.ones(len(OBSERVATION_MODALITIES), dtype=np.float32),
        )
    ).astype(np.float32, copy=False)
    if structured_state.shape != (OBSERVATION_V2_FRAME_STATE_DIM,):
        raise RuntimeError("front provider structured_state 维度漂移")
    return structured_state


__all__ = [
    "ACTIVE_FRONT_MODEL_INPUT_VERSION",
    "ACTIVE_FRONT_PROVIDER_ADAPTER_VERSION",
    "ACTIVE_FRONT_PROVIDER_IDENTITY_VERSION",
    "FRONT_PROVIDER_EXECUTION_MODE",
    "FRONT_PROVIDER_FRAME_CONVENTION",
    "FRONT_PROVIDER_ROLE_SUBSTITUTION_SEMANTICS",
    "ActiveFrontModelInput",
    "ActiveFrontProviderAdapterConfig",
    "ActiveFrontProviderIdentity",
    "build_active_front_model_input",
    "build_precision_camera_role_state",
]
