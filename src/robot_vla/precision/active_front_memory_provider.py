"""E018-P1 Stage 2 的 provider identity、部署侧证据与 PRIMARY adapter。"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import asdict, dataclass, replace

import numpy as np

from robot_vla.executive.contracts import PhaseId
from robot_vla.observation import validate_se3
from robot_vla.precision.active_external_observation import ACTUAL_EXTERNAL_POSE_SOURCE
from robot_vla.precision.active_front_camera import (
    ExternalCameraMotionState,
    rotation_angular_distance_rad,
)
from robot_vla.precision.active_front_provider import (
    ACTIVE_FRONT_MODEL_INPUT_VERSION,
    FRONT_PROVIDER_ROLE_SUBSTITUTION_SEMANTICS,
    ActiveFrontModelInput,
)
from robot_vla.precision.object_memory import (
    ObjectMeasurement,
    ObjectMemoryConfig,
    ObjectMemorySafetyContext,
    ObjectState,
)
from robot_vla.precision.object_observability import (
    OBJECT_WRITE_SCORE_SEMANTICS,
    ObjectWriteEvidence,
)


ACTIVE_FRONT_STAGE2_VERSION = "e018-p1-stage2a-primary-memory-commit/v1"
ACTIVE_FRONT_STAGE2_PROVIDER_IDENTITY_VERSION = (
    "e018-p1-stage2a-primary-provider-identity/v1"
)
ACTIVE_FRONT_STAGE2_PROVIDER_ADAPTER_VERSION = (
    "e018-p1-stage2a-primary-provider-adapter/v1"
)
ACTIVE_FRONT_STAGE2_FRAME_VERSION = "e018-p1-stage2a-provider-frame/v1"
ACTIVE_FRONT_HOME_SCORE_FRAME_VERSION = "e018-p1-stage2a-home-score-frame/v1"
ACTIVE_FRONT_STAGE2_EXECUTION_MODE = (
    "development-primary-memory-candidate/no-canonical-runtime/no-actuation/v1"
)
ACTIVE_FRONT_SCORE_SEMANTICS = OBJECT_WRITE_SCORE_SEMANTICS
ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID = "LEFT_LOW__PITCH_UP"
ACTIVE_FRONT_HOME_PRIMITIVE_ID = "HOME__CENTER"
ACTIVE_FRONT_QUALIFIED_SHADOW_PRIMITIVE_IDS = (
    "LEFT_LOW__CENTER",
    "LEFT_LOW__YAW_RIGHT",
    "LEFT_LOW__PITCH_UP",
    "LEFT_LOW__PITCH_DOWN",
    "RIGHT_LOW__CENTER",
    "RIGHT_LOW__YAW_RIGHT",
    "RIGHT_LOW__PITCH_DOWN",
)
ACTIVE_FRONT_PROHIBITED_MEMORY_WRITE_PRIMITIVE_IDS = (
    "HOME__CENTER",
    "LEFT_LOW__YAW_LEFT",
    "RIGHT_LOW__YAW_LEFT",
    "RIGHT_LOW__PITCH_UP",
)
ACTIVE_FRONT_INFORMATION_GAIN_CANDIDATES = (0.02, 0.05, 0.10)

_PRIMARY_WRITE_THRESHOLD = 0.6127982139587402
_PRIMARY_CALIBRATION_SCALE = 1.0
_PRIMARY_CALIBRATION_IDENTITY_SHA256 = (
    "fcc5531ad989172a124c2cb16ee60283fdac36334c905ae9604f814bb323ca97"
)
_HOME_WRITE_THRESHOLD = 0.6123920381069183
_HOME_CALIBRATION_SCALE = 1.0
_HOME_CALIBRATION_IDENTITY_SHA256 = (
    "7091db69c2bdc65dc904c5ebb3fcd048beb51dc37e13acf484c072f5bece2d8e"
)
# D048/D046 的 HOME actual-pose witness（robot base <- OpenCV optical camera）。
# 这里只用它判定 HOME baseline 是否可比较，不授予 HOME measurement/write 权限。
ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV = (
    (0.0, 0.7808687686920166, -0.6246950626373291, 0.9150000214576721),
    (1.0, 0.0, 0.0, -7.275957614183426e-11),
    (0.0, -0.6246950626373291, -0.7808687686920166, 0.6000000387430191),
    (0.0, 0.0, 0.0, 1.0),
)
ACTIVE_FRONT_HOME_POSITION_TOLERANCE_M = 0.00001
ACTIVE_FRONT_HOME_ORIENTATION_TOLERANCE_RAD = 0.0001
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _sha256(value: str, name: str) -> str:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")
    return value


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")
    return value


def _timestamp(value: float, name: str) -> float:
    candidate = float(value)
    if not math.isfinite(candidate) or candidate < 0.0:
        raise ValueError(f"{name} 必须是有限非负数")
    return candidate


def _probability(value: float, name: str) -> float:
    candidate = float(value)
    if not math.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
        raise ValueError(f"{name} 必须在 [0,1]")
    return candidate


def _position(
    value: tuple[float, float, float] | np.ndarray | None,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError("position_base_m 必须是有限 [3]")
    return tuple(float(item) for item in array)


def _covariance(
    value: tuple[tuple[float, float, float], ...] | np.ndarray | None,
) -> tuple[tuple[float, float, float], ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError("covariance_base_m2 必须是有限 [3,3]")
    if not np.allclose(array, array.T, rtol=0.0, atol=1e-12):
        raise ValueError("covariance_base_m2 必须对称")
    if float(np.linalg.eigvalsh(array).min()) < -1e-12:
        raise ValueError("covariance_base_m2 必须半正定")
    return tuple(tuple(float(item) for item in row) for row in array)


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


def _immutable_float64_array(value: np.ndarray) -> np.ndarray:
    """复制到 immutable bytes backing，避免 frozen dataclass 的 ndarray TOCTOU。"""

    array = np.ascontiguousarray(value, dtype=np.float64)
    return np.frombuffer(array.tobytes(), dtype=np.float64).reshape(array.shape)


def _state_dict(state: ObjectState) -> dict[str, object]:
    return {
        "version": state.version,
        "episode_id": state.episode_id,
        "mode": state.mode.value,
        "position_base_m": state.position_base_m,
        "covariance_base_m2": state.covariance_base_m2,
        "measurement_confidence": state.measurement_confidence,
        "last_observed_timestamp_s": state.last_observed_timestamp_s,
        "state_timestamp_s": state.state_timestamp_s,
        "observable_now": state.observable_now,
        "valid": state.valid,
        "accepted_update_count": state.accepted_update_count,
        "source_camera": state.source_camera,
        "source_model_identity": state.source_model_identity,
        "invalid_reasons": state.invalid_reasons,
    }


def _measurement_dict(measurement: ObjectMeasurement) -> dict[str, object]:
    return {
        "version": measurement.version,
        "timestamp_s": measurement.timestamp_s,
        "rgb_timestamp_s": measurement.rgb_timestamp_s,
        "camera_pose_timestamp_s": measurement.camera_pose_timestamp_s,
        "tcp_pose_timestamp_s": measurement.tcp_pose_timestamp_s,
        "position_base_m": measurement.position_base_m,
        "covariance_base_m2": measurement.covariance_base_m2,
        "confidence": measurement.confidence,
        "projection_valid": measurement.projection_valid,
        "in_fov": measurement.in_fov,
        "observable": measurement.observable,
        "geometry_valid": measurement.geometry_valid,
        "write_gate_passed": measurement.write_gate_passed,
        "source_camera": measurement.source_camera,
        "source_model_identity": measurement.source_model_identity,
        "frame_semantics": measurement.frame_semantics,
    }


@dataclass(frozen=True)
class ActiveFrontStage2ProviderIdentity:
    """绑定 D048 PRIMARY 的完整 provider/checkpoint/calibration/schema 来源。"""

    qualification_artifact_id: str
    qualification_source_identity_sha256: str
    qualification_config_raw_sha256: str
    qualification_config_internal_sha256: str
    qualification_result_receipt_raw_sha256: str
    qualification_result_receipt_internal_sha256: str
    qualification_result_verification_sha256: str
    candidate_id: str
    checkpoint_epoch: int
    checkpoint_sha256: str
    checkpoint_parameter_sha256: str
    checkpoint_provenance_sha256: str
    model_config_sha256: str
    proprio_stats_sha256: str
    proprio_normalizer_sha256: str
    finger_force_stats_sha256: str
    finger_force_normalizer_sha256: str
    qualification_adapter_config_sha256: str
    calibration_config_raw_sha256: str
    calibration_config_internal_sha256: str
    calibration_result_receipt_raw_sha256: str
    calibration_result_receipt_internal_sha256: str
    calibration_viewpoints_raw_sha256: str
    calibration_identity_sha256: str
    calibration_scale_factor: float
    write_threshold: float
    primitive_id: str
    geometric_motion_provider_id: str
    source_training_camera: str = "hand_camera"
    source_camera: str = "base_camera"
    actual_pose_source: str = ACTUAL_EXTERNAL_POSE_SOURCE
    role_substitution_semantics: str = FRONT_PROVIDER_ROLE_SUBSTITUTION_SEMANTICS
    qualification_input_schema_version: str = ACTIVE_FRONT_MODEL_INPUT_VERSION
    stage2_frame_schema_version: str = ACTIVE_FRONT_STAGE2_FRAME_VERSION
    execution_mode: str = ACTIVE_FRONT_STAGE2_EXECUTION_MODE
    score_semantics: str = ACTIVE_FRONT_SCORE_SEMANTICS
    version: str = ACTIVE_FRONT_STAGE2_PROVIDER_IDENTITY_VERSION

    def __post_init__(self) -> None:
        for name in (
            "qualification_artifact_id",
            "candidate_id",
            "primitive_id",
            "geometric_motion_provider_id",
        ):
            _identity(getattr(self, name), name)
        for name in (
            "qualification_source_identity_sha256",
            "qualification_config_raw_sha256",
            "qualification_config_internal_sha256",
            "qualification_result_receipt_raw_sha256",
            "qualification_result_receipt_internal_sha256",
            "qualification_result_verification_sha256",
            "checkpoint_sha256",
            "checkpoint_parameter_sha256",
            "checkpoint_provenance_sha256",
            "model_config_sha256",
            "proprio_stats_sha256",
            "proprio_normalizer_sha256",
            "finger_force_stats_sha256",
            "finger_force_normalizer_sha256",
            "qualification_adapter_config_sha256",
            "calibration_config_raw_sha256",
            "calibration_config_internal_sha256",
            "calibration_result_receipt_raw_sha256",
            "calibration_result_receipt_internal_sha256",
            "calibration_viewpoints_raw_sha256",
            "calibration_identity_sha256",
        ):
            _sha256(getattr(self, name), name)
        if (
            not isinstance(self.checkpoint_epoch, int)
            or isinstance(self.checkpoint_epoch, bool)
            or self.checkpoint_epoch < 0
        ):
            raise ValueError("checkpoint_epoch 必须是非负整数")
        _probability(self.write_threshold, "write_threshold")
        if not math.isfinite(self.calibration_scale_factor) or self.calibration_scale_factor <= 0.0:
            raise ValueError("calibration_scale_factor 必须是有限正数")
        if self.source_training_camera != "hand_camera" or self.source_camera != "base_camera":
            raise ValueError("Stage 2 provider camera role 漂移")
        if self.actual_pose_source != ACTUAL_EXTERNAL_POSE_SOURCE:
            raise ValueError("Stage 2 provider 必须绑定 actual external pose")
        if self.role_substitution_semantics != FRONT_PROVIDER_ROLE_SUBSTITUTION_SEMANTICS:
            raise ValueError("Stage 2 provider role substitution 漂移")
        if self.qualification_input_schema_version != ACTIVE_FRONT_MODEL_INPUT_VERSION:
            raise ValueError("qualification input schema 漂移")
        if self.stage2_frame_schema_version != ACTIVE_FRONT_STAGE2_FRAME_VERSION:
            raise ValueError("Stage 2 frame schema 漂移")
        if self.execution_mode != ACTIVE_FRONT_STAGE2_EXECUTION_MODE:
            raise ValueError("Stage 2 execution mode 漂移")
        if self.score_semantics != ACTIVE_FRONT_SCORE_SEMANTICS:
            raise ValueError("Stage 2 score semantics 漂移")
        if self.version != ACTIVE_FRONT_STAGE2_PROVIDER_IDENTITY_VERSION:
            raise ValueError("Stage 2 provider identity version 漂移")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return _canonical_sha256(self.to_dict())

    @property
    def provider_family_sha256(self) -> str:
        value = self.to_dict()
        for key in (
            "primitive_id",
            "calibration_identity_sha256",
            "calibration_scale_factor",
            "write_threshold",
        ):
            value.pop(key)
        return _canonical_sha256(value)

    @property
    def object_memory_source_identity(self) -> str:
        return f"e018-p1-stage2a-primary-provider/{self.sha256}"


def d049_primary_provider_identity() -> ActiveFrontStage2ProviderIdentity:
    """返回 D049 冻结的唯一 live Memory write provider identity。"""

    return ActiveFrontStage2ProviderIdentity(
        qualification_artifact_id=(
            "g2c-formal-dynamic-qualification-d048-c2eb57a-20260905-v1"
        ),
        qualification_source_identity_sha256=(
            "adab2e92efec9153cf9d118df95e8fed4cf76b5000196c6eeb818e5e9e0c3b99"
        ),
        qualification_config_raw_sha256=(
            "bfe5cbefeed8903a610ccab9ecff4d4f0e1cfd9fd4c92ec5dc1af03428f145b8"
        ),
        qualification_config_internal_sha256=(
            "0ade177a588f3cfe2acb61634537f4d6ed3d92bb72daf52dcfb756e287864715"
        ),
        qualification_result_receipt_raw_sha256=(
            "a3edafd7cb31476bf19ec79544fa92d6ee4471e7cee61673e5cd93a24b987770"
        ),
        qualification_result_receipt_internal_sha256=(
            "dbf167a0a487102cb967d15b5bc206db1f515f56e1dcce6eec95265735a86953"
        ),
        qualification_result_verification_sha256=(
            "def73934b80de33acc63a8419463d7c10ae2b994f3a72f7b37cd1cb852ffae79"
        ),
        candidate_id="W-KV0",
        checkpoint_epoch=15,
        checkpoint_sha256=(
            "97e3b7289911bc73f67755a8d9c3598c50b6c80ef01e1af13cec698ec59d3d77"
        ),
        checkpoint_parameter_sha256=(
            "1ba14a9009829c1d354555e9b788a8e3627e33ccffaddee24e66d9696121cb24"
        ),
        checkpoint_provenance_sha256=(
            "8116f273c5f7339813a260bc919e25ee84cb3493a0e76eb454e6fbdeae83252c"
        ),
        model_config_sha256=(
            "4a284a59c8c6d1865910d333597b565183f4d455af520e5edad5a79d5c67d053"
        ),
        proprio_stats_sha256=(
            "2a1061b3a56edfcfeb6e955a1910dc309ff9b776dc4eb355192661fe628de01e"
        ),
        proprio_normalizer_sha256=(
            "eb39fa6750a80d4781559e465d694fca411e3a4c11f74cbd09423886735a219e"
        ),
        finger_force_stats_sha256=(
            "fcc5b4b87aa13919ec261fc5e71a24e1b6446f47abdbc87d4b1bf4f93fe7a9e8"
        ),
        finger_force_normalizer_sha256=(
            "6de38cb2a3d74c7da581a96712c1eafc974ccf52f4cefb5a339708c30b3e79d5"
        ),
        qualification_adapter_config_sha256=(
            "cb30f6c2c75e57648681fc1f746693e201ffca33c01e6bdad36902ed0ea61d48"
        ),
        calibration_config_raw_sha256=(
            "b2ff7ee79a87a65bc080c5b5411a8989971fd262ad8226f8f51b1f055937f75f"
        ),
        calibration_config_internal_sha256=(
            "98a5727766cfe46f133bc4945d154be58da52be8c7d341e0d857c84a65aeaa74"
        ),
        calibration_result_receipt_raw_sha256=(
            "16b377e8da6a3539883101c47f53a766d1efc60da4ff4b44b302e0db827698a9"
        ),
        calibration_result_receipt_internal_sha256=(
            "6e67503d1894670c8b5a8ea5f0139453eaa85f8114025b0172a112345926c837"
        ),
        calibration_viewpoints_raw_sha256=(
            "a51c879de45208d540bb3c5db8d6389ab3e2f8dace5e424ecef9c142578659b6"
        ),
        calibration_identity_sha256=_PRIMARY_CALIBRATION_IDENTITY_SHA256,
        calibration_scale_factor=_PRIMARY_CALIBRATION_SCALE,
        write_threshold=_PRIMARY_WRITE_THRESHOLD,
        primitive_id=ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
        geometric_motion_provider_id=(
            "safe-hold-commanded-tcp-target-delta/simulator-static-zero-measured/v1"
        ),
    )


def d049_home_baseline_provider_identity() -> ActiveFrontStage2ProviderIdentity:
    """返回与 PRIMARY 同族、但绑定 D048 HOME calibration 的完整身份。"""

    return replace(
        d049_primary_provider_identity(),
        calibration_identity_sha256=_HOME_CALIBRATION_IDENTITY_SHA256,
        calibration_scale_factor=_HOME_CALIBRATION_SCALE,
        write_threshold=_HOME_WRITE_THRESHOLD,
        primitive_id=ACTIVE_FRONT_HOME_PRIMITIVE_ID,
    )


@dataclass(frozen=True)
class ActiveFrontStage2Config:
    """默认关闭；只有 D049 development config 才显式打开 Memory candidate。"""

    enabled: bool = False
    memory_write_allowed: bool = False
    min_information_gain: float = 0.05
    information_gain_comparison_tolerance: float = 1e-12
    min_candidate_frames: int = 3
    max_candidate_gap_s: float = 0.075
    max_candidate_position_spread_m: float = 0.005
    max_innovation_m: float = 0.010
    max_position_std_m: float = 0.020
    max_sensor_skew_s: float = 0.010
    max_pending_age_s: float = 2.5
    max_memory_unobserved_age_s: float = 2.5
    home_v2_barrier_frames: int = 4
    maximum_attempts_per_episode: int = 1
    require_covariance: bool = True
    physical_camera_actuation_allowed: bool = False
    arm_gripper_actuation_allowed: bool = False
    fresh_test_allowed: bool = False
    version: str = ACTIVE_FRONT_STAGE2_PROVIDER_ADAPTER_VERSION

    def __post_init__(self) -> None:
        for name in (
            "enabled",
            "memory_write_allowed",
            "require_covariance",
            "physical_camera_actuation_allowed",
            "arm_gripper_actuation_allowed",
            "fresh_test_allowed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须为 bool")
        if self.memory_write_allowed and not self.enabled:
            raise ValueError("Memory write 只能在显式 enabled 后打开")
        if (
            self.physical_camera_actuation_allowed
            or self.arm_gripper_actuation_allowed
            or self.fresh_test_allowed
        ):
            raise ValueError("D049 禁止 physical/manipulation actuator 与 fresh test")
        if self.min_information_gain not in ACTIVE_FRONT_INFORMATION_GAIN_CANDIDATES:
            raise ValueError("min_information_gain 必须来自 D049 冻结候选")
        if self.information_gain_comparison_tolerance not in {0.0, 1e-12}:
            raise ValueError("information gain comparison tolerance 只能是 0 或 1e-12")
        fixed = {
            "min_candidate_frames": (self.min_candidate_frames, 3),
            "max_candidate_gap_s": (self.max_candidate_gap_s, 0.075),
            "max_candidate_position_spread_m": (
                self.max_candidate_position_spread_m,
                0.005,
            ),
            "max_innovation_m": (self.max_innovation_m, 0.010),
            "max_position_std_m": (self.max_position_std_m, 0.020),
            "max_sensor_skew_s": (self.max_sensor_skew_s, 0.010),
            "max_pending_age_s": (self.max_pending_age_s, 2.5),
            "max_memory_unobserved_age_s": (
                self.max_memory_unobserved_age_s,
                2.5,
            ),
            "home_v2_barrier_frames": (self.home_v2_barrier_frames, 4),
            "maximum_attempts_per_episode": (
                self.maximum_attempts_per_episode,
                1,
            ),
        }
        if any(actual != expected for actual, expected in fixed.values()):
            drifted = [name for name, values in fixed.items() if values[0] != values[1]]
            raise ValueError(f"D049 Stage 2 参数漂移: {','.join(drifted)}")
        if not self.require_covariance:
            raise ValueError("D049 Stage 2 必须要求 covariance")
        if self.version != ACTIVE_FRONT_STAGE2_PROVIDER_ADAPTER_VERSION:
            raise ValueError("Stage 2 adapter config version 漂移")

    @classmethod
    def development(
        cls,
        *,
        min_information_gain: float = 0.05,
        information_gain_comparison_tolerance: float = 1e-12,
    ) -> "ActiveFrontStage2Config":
        return cls(
            enabled=True,
            memory_write_allowed=True,
            min_information_gain=min_information_gain,
            information_gain_comparison_tolerance=(
                information_gain_comparison_tolerance
            ),
        )

    def information_gain_is_sufficient(self, information_gain: float) -> bool:
        """按显式冻结 tolerance 比较，selection 使用 0.0。"""

        value = float(information_gain)
        if not math.isfinite(value):
            raise ValueError("information gain 必须有限")
        return bool(
            value + self.information_gain_comparison_tolerance
            >= self.min_information_gain
        )


def build_stage2_object_memory_config(
    provider_identity: ActiveFrontStage2ProviderIdentity | None = None,
) -> ObjectMemoryConfig:
    identity = provider_identity or d049_primary_provider_identity()
    if identity.sha256 != d049_primary_provider_identity().sha256:
        raise ValueError("Stage 2 Object Memory 只接受 D049 PRIMARY identity")
    return ObjectMemoryConfig(
        max_unobserved_age_s=2.5,
        max_innovation_m=0.010,
        max_position_std_m=0.020,
        min_candidate_frames=3,
        max_candidate_gap_s=0.075,
        max_candidate_position_spread_m=0.005,
        max_sensor_skew_s=0.010,
        expected_source_camera="base_camera",
        expected_source_model_identity=identity.object_memory_source_identity,
        require_covariance=True,
        covariance_growth_m2_per_s=0.0,
    )


@dataclass(frozen=True)
class ActiveFrontScoreComponents:
    """来自同一个 deployable provider output 的原始 write-score 分量。"""

    object_visibility_probability: float
    projection_validity_probability: float
    object_mask_probability: float
    goal_mask_probability: float
    object_normalized_entropy: float
    object_sigma_xy_px: tuple[float, float] | np.ndarray

    def __post_init__(self) -> None:
        for name in (
            "object_visibility_probability",
            "projection_validity_probability",
            "object_mask_probability",
            "goal_mask_probability",
            "object_normalized_entropy",
        ):
            object.__setattr__(self, name, _probability(getattr(self, name), name))
        sigma = np.asarray(self.object_sigma_xy_px, dtype=np.float64)
        if sigma.shape != (2,) or not np.isfinite(sigma).all() or np.any(sigma < 0.0):
            raise ValueError("object_sigma_xy_px 必须是有限非负 [2]")
        object.__setattr__(
            self,
            "object_sigma_xy_px",
            tuple(float(value) for value in sigma),
        )

    @property
    def radial_sigma_px(self) -> float:
        """与 D048/ObjectWriteEvidence 完全相同的 L2 radial sigma。"""

        return float(np.linalg.norm(np.asarray(self.object_sigma_xy_px, dtype=np.float64)))

    def to_object_write_evidence(self, *, geometry_valid: bool) -> ObjectWriteEvidence:
        """只用部署可得分量机械构造唯一 write evidence。"""

        return ObjectWriteEvidence(
            visibility_probability=self.object_visibility_probability,
            projection_validity_probability=self.projection_validity_probability,
            object_mask_probability=self.object_mask_probability,
            goal_mask_probability=self.goal_mask_probability,
            normalized_entropy=self.object_normalized_entropy,
            radial_sigma_px=self.radial_sigma_px,
            geometry_valid=geometry_valid,
        )


@dataclass(frozen=True)
class ActiveFrontStage2FrameEvidence:
    """单个 deployable front-provider frame；不允许任何 privileged label。"""

    episode_id: str
    episode_generation: int
    request_id: str
    source_phase: PhaseId
    observation_sequence_id: str
    model_input_digest: str
    provider_output_digest: str
    provider_identity: ActiveFrontStage2ProviderIdentity
    camera_motion_state: ExternalCameraMotionState
    settled: bool
    control_timestamp_s: float
    rgb_timestamp_s: float
    camera_pose_timestamp_s: float
    tcp_pose_timestamp_s: float
    base_from_external_camera_cv: np.ndarray
    position_base_m: tuple[float, float, float] | np.ndarray | None
    covariance_base_m2: tuple[tuple[float, float, float], ...] | np.ndarray | None
    measurement_confidence: float
    write_score: float
    score_components: ActiveFrontScoreComponents
    projection_valid: bool
    in_fov: bool
    observable: bool
    geometry_valid: bool
    structurally_eligible: bool
    deployable_free_static_safe: bool
    source_camera: str = "base_camera"
    actual_pose_source: str = ACTUAL_EXTERNAL_POSE_SOURCE
    input_schema_version: str = ACTIVE_FRONT_MODEL_INPUT_VERSION
    score_semantics: str = ACTIVE_FRONT_SCORE_SEMANTICS
    execution_mode: str = ACTIVE_FRONT_STAGE2_EXECUTION_MODE
    qualification_only: bool = False
    version: str = ACTIVE_FRONT_STAGE2_FRAME_VERSION

    def __post_init__(self) -> None:
        for name in ("episode_id", "request_id", "observation_sequence_id"):
            _identity(getattr(self, name), name)
        _sha256(self.model_input_digest, "model_input_digest")
        _sha256(self.provider_output_digest, "provider_output_digest")
        if (
            not isinstance(self.episode_generation, int)
            or isinstance(self.episode_generation, bool)
            or self.episode_generation <= 0
        ):
            raise ValueError("episode_generation 必须是正整数")
        if not isinstance(self.source_phase, PhaseId):
            raise TypeError("source_phase 必须是 PhaseId")
        if not isinstance(self.provider_identity, ActiveFrontStage2ProviderIdentity):
            raise TypeError("provider_identity 类型错误")
        if not isinstance(self.camera_motion_state, ExternalCameraMotionState):
            raise TypeError("camera_motion_state 类型错误")
        for name in (
            "settled",
            "projection_valid",
            "in_fov",
            "observable",
            "geometry_valid",
            "structurally_eligible",
            "deployable_free_static_safe",
            "qualification_only",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须为 bool")
        for name in (
            "control_timestamp_s",
            "rgb_timestamp_s",
            "camera_pose_timestamp_s",
            "tcp_pose_timestamp_s",
        ):
            object.__setattr__(self, name, _timestamp(getattr(self, name), name))
        transform = validate_se3(
            self.base_from_external_camera_cv,
            "base_from_external_camera_cv",
        )
        object.__setattr__(
            self,
            "base_from_external_camera_cv",
            _immutable_float64_array(transform),
        )
        position = _position(self.position_base_m)
        covariance = _covariance(self.covariance_base_m2)
        if (position is None) != (covariance is None):
            raise ValueError("position/covariance 必须同时存在或同时缺失")
        object.__setattr__(self, "position_base_m", position)
        object.__setattr__(self, "covariance_base_m2", covariance)
        object.__setattr__(
            self,
            "measurement_confidence",
            _probability(self.measurement_confidence, "measurement_confidence"),
        )
        object.__setattr__(self, "write_score", _probability(self.write_score, "write_score"))
        if not isinstance(self.score_components, ActiveFrontScoreComponents):
            raise TypeError("score_components 类型错误")
        for name in (
            "source_camera",
            "actual_pose_source",
            "input_schema_version",
            "score_semantics",
            "execution_mode",
        ):
            _identity(getattr(self, name), name)
        write_evidence = self.score_components.to_object_write_evidence(
            geometry_valid=self.geometry_valid
        )
        if self.score_semantics != write_evidence.score_semantics:
            raise ValueError("frame score_semantics 与 ObjectWriteEvidence 漂移")
        if not math.isclose(
            self.write_score,
            write_evidence.score,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("frame stored write_score 与 deployable components 漂移")
        if not math.isclose(
            self.measurement_confidence,
            write_evidence.score,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("measurement_confidence 必须绑定机械重算 write_score")
        final_observable = bool(
            write_evidence.observable and self.projection_valid and self.in_fov
        )
        final_structurally_eligible = bool(
            final_observable and write_evidence.geometry_valid
        )
        if self.observable != final_observable:
            raise ValueError("frame stored observable 与完整 deployable evidence 漂移")
        if self.structurally_eligible != final_structurally_eligible:
            raise ValueError(
                "frame stored structurally_eligible 与完整 deployable evidence 漂移"
            )
        state_complete = self.position_base_m is not None
        if self.geometry_valid != state_complete:
            raise ValueError("geometry_valid 必须与 position/covariance 完整性一致")
        if self.in_fov and not self.projection_valid:
            raise ValueError("in_fov=true 要求 projection_valid=true")
        if self.observable and not (self.projection_valid and self.in_fov):
            raise ValueError("observable 与 projected/in_fov 语义冲突")
        if self.version != ACTIVE_FRONT_STAGE2_FRAME_VERSION:
            raise ValueError("Stage 2 frame version 漂移")

    @property
    def write_evidence(self) -> ObjectWriteEvidence:
        """返回由 score components 重建的证据，不信任 stored derived fields。"""

        return self.score_components.to_object_write_evidence(
            geometry_valid=self.geometry_valid
        )

    @property
    def final_observable(self) -> bool:
        return bool(
            self.write_evidence.observable and self.projection_valid and self.in_fov
        )

    @property
    def final_structurally_eligible(self) -> bool:
        return bool(self.final_observable and self.write_evidence.geometry_valid)

    @property
    def timestamp_skew_s(self) -> float:
        values = (
            self.control_timestamp_s,
            self.rgb_timestamp_s,
            self.camera_pose_timestamp_s,
            self.tcp_pose_timestamp_s,
        )
        return float(max(values) - min(values))

    @property
    def actual_pose_sha256(self) -> str:
        return _array_sha256(self.base_from_external_camera_cv)

    @property
    def frame_digest(self) -> str:
        return _canonical_sha256(
            {
                "version": self.version,
                "episode_id": self.episode_id,
                "episode_generation": self.episode_generation,
                "request_id": self.request_id,
                "source_phase": self.source_phase.value,
                "observation_sequence_id": self.observation_sequence_id,
                "model_input_digest": self.model_input_digest,
                "provider_output_digest": self.provider_output_digest,
                "provider_identity_sha256": self.provider_identity.sha256,
                "camera_motion_state": self.camera_motion_state.value,
                "settled": self.settled,
                "timestamps": {
                    "control": self.control_timestamp_s,
                    "rgb": self.rgb_timestamp_s,
                    "camera_pose": self.camera_pose_timestamp_s,
                    "tcp_pose": self.tcp_pose_timestamp_s,
                },
                "actual_pose_sha256": self.actual_pose_sha256,
                "position_base_m": self.position_base_m,
                "covariance_base_m2": self.covariance_base_m2,
                "measurement_confidence": self.measurement_confidence,
                "write_score": self.write_score,
                "score_components": asdict(self.score_components),
                "flags": {
                    "projection_valid": self.projection_valid,
                    "in_fov": self.in_fov,
                    "observable": self.observable,
                    "geometry_valid": self.geometry_valid,
                    "structurally_eligible": self.structurally_eligible,
                    "deployable_free_static_safe": self.deployable_free_static_safe,
                    "qualification_only": self.qualification_only,
                },
                "source_camera": self.source_camera,
                "actual_pose_source": self.actual_pose_source,
                "input_schema_version": self.input_schema_version,
                "score_semantics": self.score_semantics,
                "execution_mode": self.execution_mode,
            }
        )


@dataclass(frozen=True)
class PassiveHomeScoreEvidence:
    """HOME 的 raw score 证据；只可比较，不可形成 ObjectMeasurement。"""

    episode_id: str
    episode_generation: int
    request_id: str
    observation_sequence_id: str
    model_input_digest: str
    provider_output_digest: str
    provider_identity: ActiveFrontStage2ProviderIdentity
    viewpoint_primitive_id: str
    camera_motion_state: ExternalCameraMotionState
    settled: bool
    score_components: ActiveFrontScoreComponents
    stored_write_score: float
    geometry_valid: bool
    control_timestamp_s: float | None
    rgb_timestamp_s: float | None
    camera_pose_timestamp_s: float | None
    tcp_pose_timestamp_s: float | None
    base_from_external_camera_cv: np.ndarray | None
    actual_pose_source: str = ACTUAL_EXTERNAL_POSE_SOURCE
    score_semantics: str = ACTIVE_FRONT_SCORE_SEMANTICS
    object_measurement_usable: bool = False
    version: str = ACTIVE_FRONT_HOME_SCORE_FRAME_VERSION

    def __post_init__(self) -> None:
        for name in ("episode_id", "request_id", "observation_sequence_id"):
            _identity(getattr(self, name), name)
        _sha256(self.model_input_digest, "HOME model_input_digest")
        _sha256(self.provider_output_digest, "HOME provider_output_digest")
        if (
            not isinstance(self.episode_generation, int)
            or isinstance(self.episode_generation, bool)
            or self.episode_generation <= 0
        ):
            raise ValueError("HOME baseline episode_generation 必须是正整数")
        if not isinstance(self.provider_identity, ActiveFrontStage2ProviderIdentity):
            raise TypeError("HOME provider_identity 类型错误")
        _identity(self.viewpoint_primitive_id, "HOME viewpoint_primitive_id")
        if not isinstance(self.camera_motion_state, ExternalCameraMotionState):
            raise TypeError("HOME camera_motion_state 类型错误")
        if not isinstance(self.settled, bool):
            raise TypeError("HOME settled 必须为 bool")
        if not isinstance(self.score_components, ActiveFrontScoreComponents):
            raise TypeError("HOME score_components 类型错误")
        if not isinstance(self.geometry_valid, bool):
            raise TypeError("HOME geometry_valid 必须为 bool")
        for name in (
            "control_timestamp_s",
            "rgb_timestamp_s",
            "camera_pose_timestamp_s",
            "tcp_pose_timestamp_s",
        ):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _timestamp(value, f"HOME {name}"))
        if self.base_from_external_camera_cv is not None:
            transform = validate_se3(
                self.base_from_external_camera_cv,
                "HOME base_from_external_camera_cv",
            )
            object.__setattr__(
                self,
                "base_from_external_camera_cv",
                _immutable_float64_array(transform),
            )
        _identity(self.actual_pose_source, "HOME actual_pose_source")
        _identity(self.score_semantics, "HOME score_semantics")
        if not isinstance(self.object_measurement_usable, bool):
            raise TypeError("HOME object_measurement_usable 必须为 bool")
        if self.object_measurement_usable:
            raise ValueError(
                "D048 HOME raw score 只能作 baseline，不能成为 usable measurement"
            )
        object.__setattr__(
            self,
            "stored_write_score",
            _probability(self.stored_write_score, "HOME stored_write_score"),
        )
        if not math.isclose(
            self.stored_write_score,
            self.write_evidence.score,
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise ValueError("HOME stored write_score 与 deployable components 漂移")
        if self.version != ACTIVE_FRONT_HOME_SCORE_FRAME_VERSION:
            raise ValueError("PassiveHomeScoreEvidence version 漂移")

    @property
    def write_evidence(self) -> ObjectWriteEvidence:
        return self.score_components.to_object_write_evidence(
            geometry_valid=self.geometry_valid
        )

    @property
    def write_score(self) -> float:
        return self.write_evidence.score

    @property
    def provider_identity_sha256(self) -> str:
        return self.provider_identity.sha256

    @property
    def provider_family_sha256(self) -> str:
        return self.provider_identity.provider_family_sha256

    @property
    def actual_pose_sha256(self) -> str | None:
        if self.base_from_external_camera_cv is None:
            return None
        return _array_sha256(self.base_from_external_camera_cv)

    @property
    def pose_valid(self) -> bool:
        return bool(
            self.base_from_external_camera_cv is not None
            and self.actual_pose_source == ACTUAL_EXTERNAL_POSE_SOURCE
            and self.home_position_error_m is not None
            and self.home_position_error_m
            <= ACTIVE_FRONT_HOME_POSITION_TOLERANCE_M + 1e-12
            and self.home_orientation_error_rad is not None
            and self.home_orientation_error_rad
            <= ACTIVE_FRONT_HOME_ORIENTATION_TOLERANCE_RAD + 1e-12
        )

    @property
    def home_position_error_m(self) -> float | None:
        if self.base_from_external_camera_cv is None:
            return None
        expected = np.asarray(
            ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV,
            dtype=np.float64,
        )
        return float(
            np.linalg.norm(self.base_from_external_camera_cv[:3, 3] - expected[:3, 3])
        )

    @property
    def home_orientation_error_rad(self) -> float | None:
        if self.base_from_external_camera_cv is None:
            return None
        expected = np.asarray(
            ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV,
            dtype=np.float64,
        )
        return rotation_angular_distance_rad(
            expected[:3, :3],
            self.base_from_external_camera_cv[:3, :3],
        )

    @property
    def home_capture_valid(self) -> bool:
        return bool(
            self.viewpoint_primitive_id == ACTIVE_FRONT_HOME_PRIMITIVE_ID
            and self.camera_motion_state is ExternalCameraMotionState.HOME_ANCHOR
            and self.settled
        )

    @property
    def timestamp_skew_s(self) -> float | None:
        values = (
            self.control_timestamp_s,
            self.rgb_timestamp_s,
            self.camera_pose_timestamp_s,
            self.tcp_pose_timestamp_s,
        )
        if any(value is None for value in values):
            return None
        numeric = tuple(float(value) for value in values if value is not None)
        return float(max(numeric) - min(numeric))

    @property
    def timestamp_valid(self) -> bool:
        sensor_timestamps = (
            self.rgb_timestamp_s,
            self.camera_pose_timestamp_s,
            self.tcp_pose_timestamp_s,
        )
        if self.control_timestamp_s is None or any(
            value is None for value in sensor_timestamps
        ):
            return False
        if any(
            float(value) > self.control_timestamp_s + 1e-12
            for value in sensor_timestamps
            if value is not None
        ):
            return False
        return bool(
            self.timestamp_skew_s is not None
            and self.timestamp_skew_s <= 0.010 + 1e-12
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "episode_generation": self.episode_generation,
            "request_id": self.request_id,
            "observation_sequence_id": self.observation_sequence_id,
            "model_input_digest": self.model_input_digest,
            "provider_output_digest": self.provider_output_digest,
            "provider_identity": self.provider_identity.to_dict(),
            "provider_identity_sha256": self.provider_identity_sha256,
            "provider_family_sha256": self.provider_family_sha256,
            "viewpoint_primitive_id": self.viewpoint_primitive_id,
            "camera_motion_state": self.camera_motion_state.value,
            "settled": self.settled,
            "home_capture_valid": self.home_capture_valid,
            "score_components": asdict(self.score_components),
            "stored_write_score": self.stored_write_score,
            "recomputed_write_score": self.write_score,
            "geometry_valid": self.geometry_valid,
            "timestamps": {
                "control": self.control_timestamp_s,
                "rgb": self.rgb_timestamp_s,
                "camera_pose": self.camera_pose_timestamp_s,
                "tcp_pose": self.tcp_pose_timestamp_s,
            },
            "timestamp_skew_s": self.timestamp_skew_s,
            "timestamp_valid": self.timestamp_valid,
            "actual_pose_sha256": self.actual_pose_sha256,
            "actual_pose_source": self.actual_pose_source,
            "pose_valid": self.pose_valid,
            "home_position_error_m": self.home_position_error_m,
            "home_orientation_error_rad": self.home_orientation_error_rad,
            "score_semantics": self.score_semantics,
            "object_measurement_usable": self.object_measurement_usable,
        }

    @property
    def frame_digest(self) -> str:
        return _canonical_sha256(self.to_dict())


@dataclass(frozen=True)
class PassiveBaselineEvidence:
    episode_id: str
    episode_generation: int
    request_id: str
    timestamp_s: float
    wrist_object_measurement_usable: bool
    wrist_evidence_identity_sha256: str | None
    home_front: PassiveHomeScoreEvidence | None
    object_memory_navigation_state_available: bool
    object_memory_age_s: float | None
    object_memory_source_identity: str | None
    version: str = ACTIVE_FRONT_STAGE2_VERSION

    def __post_init__(self) -> None:
        for name in ("episode_id", "request_id"):
            _identity(getattr(self, name), name)
        if (
            not isinstance(self.episode_generation, int)
            or isinstance(self.episode_generation, bool)
            or self.episode_generation <= 0
        ):
            raise ValueError("baseline episode_generation 必须是正整数")
        object.__setattr__(
            self,
            "timestamp_s",
            _timestamp(self.timestamp_s, "baseline timestamp"),
        )
        for name in (
            "wrist_object_measurement_usable",
            "object_memory_navigation_state_available",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"baseline {name} 必须为 bool")
        if self.wrist_evidence_identity_sha256 is not None:
            _sha256(
                self.wrist_evidence_identity_sha256,
                "wrist_evidence_identity_sha256",
            )
        if self.home_front is not None:
            if not isinstance(self.home_front, PassiveHomeScoreEvidence):
                raise TypeError("home_front 必须是 PassiveHomeScoreEvidence 或 None")
            if (
                self.home_front.episode_id != self.episode_id
                or self.home_front.episode_generation != self.episode_generation
                or self.home_front.request_id != self.request_id
            ):
                raise ValueError("HOME baseline frame 与 trigger identity 漂移")
        if self.object_memory_age_s is not None:
            age = float(self.object_memory_age_s)
            if not math.isfinite(age) or age < 0.0:
                raise ValueError("object_memory_age_s 必须是有限非负数")
            object.__setattr__(self, "object_memory_age_s", age)
        if self.object_memory_source_identity is not None:
            _identity(self.object_memory_source_identity, "object_memory_source_identity")
        if self.version != ACTIVE_FRONT_STAGE2_VERSION:
            raise ValueError("PassiveBaselineEvidence version 漂移")

    @property
    def home_front_object_measurement_usable(self) -> bool:
        return bool(
            self.home_front is not None
            and self.home_front.object_measurement_usable
        )

    @property
    def home_front_write_score(self) -> float | None:
        return None if self.home_front is None else self.home_front.write_score

    @property
    def home_front_frame_digest(self) -> str | None:
        return None if self.home_front is None else self.home_front.frame_digest

    def to_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "episode_generation": self.episode_generation,
            "request_id": self.request_id,
            "timestamp_s": self.timestamp_s,
            "wrist_object_measurement_usable": self.wrist_object_measurement_usable,
            "wrist_evidence_identity_sha256": self.wrist_evidence_identity_sha256,
            "home_front": None if self.home_front is None else self.home_front.to_dict(),
            "home_front_frame_digest": self.home_front_frame_digest,
            "home_front_object_measurement_usable": (
                self.home_front_object_measurement_usable
            ),
            "object_memory_navigation_state_available": (
                self.object_memory_navigation_state_available
            ),
            "object_memory_age_s": self.object_memory_age_s,
            "object_memory_source_identity": self.object_memory_source_identity,
        }

    @property
    def digest(self) -> str:
        return _canonical_sha256(self.to_dict())

    def gain_unavailable_reasons(
        self,
        provider_identity: ActiveFrontStage2ProviderIdentity,
    ) -> tuple[str, ...]:
        reasons: list[str] = []
        home = self.home_front
        if home is None:
            return ("baseline_unavailable_shadow_only",)
        expected_home = d049_home_baseline_provider_identity()
        if home.provider_identity.sha256 != expected_home.sha256:
            reasons.append("baseline_home_provider_identity_mismatch")
        if home.provider_family_sha256 != provider_identity.provider_family_sha256:
            reasons.append("baseline_provider_identity_mismatch")
        if not home.home_capture_valid:
            reasons.append("baseline_home_capture_invalid")
        if home.score_semantics != provider_identity.score_semantics:
            reasons.append("baseline_score_semantics_mismatch")
        if not home.pose_valid:
            reasons.append("baseline_pose_invalid")
        if not home.timestamp_valid or (
            home.control_timestamp_s is not None
            and home.control_timestamp_s > self.timestamp_s + 1e-12
        ):
            reasons.append("baseline_timestamp_invalid")
        return tuple(reasons)


@dataclass(frozen=True)
class ActiveFrontFrameAdaptation:
    observation_sequence_id: str
    frame_digest: str
    provider_identity_sha256: str | None
    measurement: ObjectMeasurement | None
    write_score: float | None
    eligible: bool
    rejection_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _identity(self.observation_sequence_id, "adaptation observation_sequence_id")
        _sha256(self.frame_digest, "adaptation frame_digest")
        if self.provider_identity_sha256 is not None:
            _sha256(self.provider_identity_sha256, "adaptation provider identity")
        if self.write_score is not None:
            _probability(self.write_score, "adaptation write_score")
        if not isinstance(self.eligible, bool):
            raise TypeError("adaptation eligible 必须为 bool")
        if self.eligible != (self.measurement is not None and not self.rejection_reasons):
            raise ValueError("adaptation eligible/measurement/reasons 语义不一致")
        if not self.eligible and not self.rejection_reasons:
            raise ValueError("被拒绝 adaptation 必须有 rejection reason")


class ActiveFrontStage2ProviderAdapter:
    """只把 D049 PRIMARY 的 P1 frame 转为 ObjectMeasurement。"""

    def __init__(
        self,
        config: ActiveFrontStage2Config | None = None,
        *,
        provider_identity: ActiveFrontStage2ProviderIdentity | None = None,
    ) -> None:
        self.config = config or ActiveFrontStage2Config()
        self.provider_identity = provider_identity or d049_primary_provider_identity()
        if self.provider_identity.sha256 != d049_primary_provider_identity().sha256:
            raise ValueError("adapter expected identity 必须是 D049 PRIMARY")

    @staticmethod
    def _max_std(
        covariance: tuple[tuple[float, float, float], ...] | None,
    ) -> float | None:
        if covariance is None:
            return None
        maximum_eigenvalue = float(
            np.linalg.eigvalsh(np.asarray(covariance, dtype=np.float64)).max()
        )
        return float(math.sqrt(max(0.0, maximum_eigenvalue)))

    def adapt(
        self,
        frame: ActiveFrontStage2FrameEvidence | ActiveFrontModelInput,
        *,
        safety: ObjectMemorySafetyContext,
    ) -> ActiveFrontFrameAdaptation:
        if not isinstance(safety, ObjectMemorySafetyContext):
            raise TypeError("safety 必须是 ObjectMemorySafetyContext")
        if isinstance(frame, ActiveFrontModelInput):
            return ActiveFrontFrameAdaptation(
                observation_sequence_id=frame.observation_sequence_id,
                frame_digest=frame.input_digest,
                provider_identity_sha256=None,
                measurement=None,
                write_score=None,
                eligible=False,
                rejection_reasons=("qualification_only_adapter_forbidden",),
            )
        if not isinstance(frame, ActiveFrontStage2FrameEvidence):
            raise TypeError("frame 必须是 Stage 2 frame 或 qualification-only input")

        reasons: list[str] = []
        identity = frame.provider_identity
        expected = self.provider_identity
        write_evidence = frame.write_evidence
        if not self.config.enabled:
            reasons.append("stage2_feature_disabled")
        if not self.config.memory_write_allowed:
            reasons.append("stage2_memory_write_disabled")
        if frame.qualification_only:
            reasons.append("qualification_only_adapter_forbidden")
        if identity.primitive_id != ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID:
            reasons.append("primitive_not_primary")
        if identity.primitive_id in ACTIVE_FRONT_PROHIBITED_MEMORY_WRITE_PRIMITIVE_IDS:
            reasons.append("prohibited_memory_write_primitive")
        if identity.calibration_identity_sha256 != expected.calibration_identity_sha256:
            reasons.append("calibration_identity_mismatch")
        if identity.provider_family_sha256 != expected.provider_family_sha256:
            reasons.append("provider_family_identity_mismatch")
        if identity.sha256 != expected.sha256:
            reasons.append("provider_identity_mismatch")
        if frame.source_camera != expected.source_camera:
            reasons.append("source_camera_mismatch")
        if frame.actual_pose_source != expected.actual_pose_source:
            reasons.append("actual_pose_source_mismatch")
        if frame.input_schema_version != expected.qualification_input_schema_version:
            reasons.append("input_schema_identity_mismatch")
        if frame.score_semantics != expected.score_semantics:
            reasons.append("score_semantics_mismatch")
        if frame.execution_mode != ACTIVE_FRONT_STAGE2_EXECUTION_MODE:
            reasons.append("execution_mode_mismatch")
        if frame.camera_motion_state is not ExternalCameraMotionState.COLLECT:
            reasons.append("motion_or_non_collect_frame")
        if not frame.settled:
            reasons.append("frame_not_settled")
        timestamps = (
            frame.control_timestamp_s,
            frame.rgb_timestamp_s,
            frame.camera_pose_timestamp_s,
            frame.tcp_pose_timestamp_s,
        )
        if any(value > frame.control_timestamp_s + 1e-12 for value in timestamps[1:]):
            reasons.append("sensor_timestamp_after_control")
        if frame.timestamp_skew_s > self.config.max_sensor_skew_s + 1e-12:
            reasons.append("sensor_timestamp_unsynchronized")
        if not frame.projection_valid:
            reasons.append("projection_invalid")
        elif not frame.in_fov:
            reasons.append("out_of_fov")
        if not frame.final_observable:
            reasons.append("not_observable")
        if not write_evidence.geometry_valid or frame.position_base_m is None:
            reasons.append("geometry_invalid")
        if not frame.final_structurally_eligible:
            reasons.append("structurally_ineligible")
        if not frame.deployable_free_static_safe:
            reasons.append("deployable_safety_failed")
        if write_evidence.score + 1e-12 < expected.write_threshold:
            reasons.append("write_score_below_primary_threshold")
        if self.config.require_covariance and frame.covariance_base_m2 is None:
            reasons.append("measurement_covariance_missing")
        maximum_std = self._max_std(frame.covariance_base_m2)
        if maximum_std is not None and maximum_std > self.config.max_position_std_m + 1e-12:
            reasons.append("measurement_uncertain")
        reasons.extend(safety.invalidation_reasons)
        rejected = tuple(dict.fromkeys(reasons))
        if rejected:
            return ActiveFrontFrameAdaptation(
                observation_sequence_id=frame.observation_sequence_id,
                frame_digest=frame.frame_digest,
                provider_identity_sha256=identity.sha256,
                measurement=None,
                write_score=write_evidence.score,
                eligible=False,
                rejection_reasons=rejected,
            )

        measurement = ObjectMeasurement(
            timestamp_s=frame.control_timestamp_s,
            rgb_timestamp_s=frame.rgb_timestamp_s,
            camera_pose_timestamp_s=frame.camera_pose_timestamp_s,
            tcp_pose_timestamp_s=frame.tcp_pose_timestamp_s,
            position_base_m=frame.position_base_m,
            covariance_base_m2=frame.covariance_base_m2,
            confidence=write_evidence.score,
            projection_valid=frame.projection_valid,
            in_fov=frame.in_fov,
            observable=frame.final_observable,
            geometry_valid=write_evidence.geometry_valid,
            write_gate_passed=True,
            source_camera=frame.source_camera,
            source_model_identity=identity.object_memory_source_identity,
        )
        return ActiveFrontFrameAdaptation(
            observation_sequence_id=frame.observation_sequence_id,
            frame_digest=frame.frame_digest,
            provider_identity_sha256=identity.sha256,
            measurement=measurement,
            write_score=write_evidence.score,
            eligible=True,
            rejection_reasons=(),
        )


__all__ = [
    "ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV",
    "ACTIVE_FRONT_HOME_ORIENTATION_TOLERANCE_RAD",
    "ACTIVE_FRONT_HOME_POSITION_TOLERANCE_M",
    "ACTIVE_FRONT_HOME_PRIMITIVE_ID",
    "ACTIVE_FRONT_INFORMATION_GAIN_CANDIDATES",
    "ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID",
    "ACTIVE_FRONT_PROHIBITED_MEMORY_WRITE_PRIMITIVE_IDS",
    "ACTIVE_FRONT_QUALIFIED_SHADOW_PRIMITIVE_IDS",
    "ACTIVE_FRONT_SCORE_SEMANTICS",
    "ACTIVE_FRONT_STAGE2_EXECUTION_MODE",
    "ACTIVE_FRONT_STAGE2_FRAME_VERSION",
    "ACTIVE_FRONT_STAGE2_PROVIDER_ADAPTER_VERSION",
    "ACTIVE_FRONT_STAGE2_PROVIDER_IDENTITY_VERSION",
    "ACTIVE_FRONT_STAGE2_VERSION",
    "ActiveFrontFrameAdaptation",
    "ActiveFrontScoreComponents",
    "ActiveFrontStage2Config",
    "ActiveFrontStage2FrameEvidence",
    "ActiveFrontStage2ProviderAdapter",
    "ActiveFrontStage2ProviderIdentity",
    "PassiveBaselineEvidence",
    "PassiveHomeScoreEvidence",
    "build_stage2_object_memory_config",
    "d049_home_baseline_provider_identity",
    "d049_primary_provider_identity",
]
