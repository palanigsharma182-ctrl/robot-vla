"""E018-P0 抓取前 position-only object state memory。"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from robot_vla.precision.state_memory import GoalState

E018_OBJECT_MEMORY_VERSION = "e018-p0-pregrasp-object-memory/v1"
E018_DUAL_MEMORY_VERSION = "e018-p0-dual-precision-world-state/v1"
OBJECT_POSITION_FRAME_SEMANTICS = "position/robot-base/m/v1"
OBJECT_MEMORY_UPDATE_POLICY = "verified-replace-free-static-hold-pregrasp/v1"


class ObjectMemoryMode(str, Enum):
    """P0 只允许未初始化、抓取前静态和失效三种模式。"""

    UNINITIALIZED = "uninitialized"
    FREE_STATIC = "free_static"
    INVALID = "invalid"


class ObjectStateRequirement(str, Enum):
    """显式区分粗定位使用与接触授权使用。"""

    NAVIGATION = "navigation"
    CONTACT_READY = "contact_ready"


_IRREVERSIBLE_INVALID_REASONS = frozenset(
    {
        "pregrasp_window_closed",
        "object_contact_detected",
        "gripper_close_commanded",
        "grasp_candidate",
        "grasp_verified",
        "object_maybe_moved",
        "source_camera_mismatch",
        "source_model_identity_mismatch",
    }
)


def _timestamp(value: float, name: str) -> float:
    candidate = float(value)
    if not math.isfinite(candidate) or candidate < 0.0:
        raise ValueError(f"{name} 必须是有限非负数")
    return candidate


def _probability(value: float, name: str) -> float:
    candidate = float(value)
    if not math.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
        raise ValueError(f"{name} 必须是 [0,1] 内有限数值")
    return candidate


def _position(
    value: tuple[float, float, float] | np.ndarray | None,
    name: str,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是有限 base-frame XYZ")
    return tuple(float(item) for item in array)


def _covariance(
    value: tuple[tuple[float, float, float], ...] | np.ndarray | None,
    name: str,
) -> tuple[tuple[float, float, float], ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是有限 [3,3]")
    if not np.allclose(array, array.T, rtol=0.0, atol=1e-12):
        raise ValueError(f"{name} 必须对称")
    if float(np.linalg.eigvalsh(array).min()) < -1e-12:
        raise ValueError(f"{name} 必须半正定")
    return tuple(tuple(float(item) for item in row) for row in array)


def _max_std(
    covariance: tuple[tuple[float, float, float], ...] | None,
) -> float | None:
    if covariance is None:
        return None
    diagonal = np.diag(np.asarray(covariance, dtype=np.float64))
    return float(math.sqrt(max(0.0, float(diagonal.max()))))


def _identity(value: str | None, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串或 None")
    return value


def _required_identity(value: str | None, name: str) -> str:
    identity = _identity(value, name)
    if identity is None:
        raise ValueError(f"{name} 不能为空")
    return identity


def _unique_reasons(values: tuple[str, ...]) -> tuple[str, ...]:
    if len(set(values)) != len(values) or any(
        not isinstance(reason, str) or not reason for reason in values
    ):
        raise ValueError("invalid/rejection reasons 必须是互异非空字符串")
    return values


@dataclass(frozen=True)
class ObjectMeasurement:
    """单个控制 Tick 的当前 object measurement；不携带候选窗口结论。"""

    timestamp_s: float
    rgb_timestamp_s: float
    camera_pose_timestamp_s: float
    tcp_pose_timestamp_s: float
    position_base_m: tuple[float, float, float] | np.ndarray | None
    covariance_base_m2: tuple[tuple[float, float, float], ...] | np.ndarray | None
    confidence: float
    projection_valid: bool
    in_fov: bool
    observable: bool
    geometry_valid: bool
    write_gate_passed: bool
    source_camera: str
    source_model_identity: str
    frame_semantics: str = OBJECT_POSITION_FRAME_SEMANTICS
    version: str = E018_OBJECT_MEMORY_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s, "measurement timestamp"))
        for name in (
            "rgb_timestamp_s",
            "camera_pose_timestamp_s",
            "tcp_pose_timestamp_s",
        ):
            value = _timestamp(getattr(self, name), name)
            if value > self.timestamp_s + 1e-12:
                raise ValueError(f"{name} 不能晚于控制 Tick")
            object.__setattr__(self, name, value)
        position = _position(self.position_base_m, "measurement position_base_m")
        covariance = _covariance(self.covariance_base_m2, "measurement covariance_base_m2")
        object.__setattr__(self, "position_base_m", position)
        object.__setattr__(self, "covariance_base_m2", covariance)
        object.__setattr__(self, "confidence", _probability(self.confidence, "measurement confidence"))
        for name in (
            "projection_valid",
            "in_fov",
            "observable",
            "geometry_valid",
            "write_gate_passed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须为 bool")
        if self.in_fov and not self.projection_valid:
            raise ValueError("in_fov=true 要求 projection_valid=true")
        if self.observable and not (self.projection_valid and self.in_fov):
            raise ValueError("observable 与 projected/in_fov 语义冲突")
        if (position is None) != (covariance is None):
            raise ValueError("Object measurement position 与 covariance 必须同时存在或同时缺失")
        if self.write_gate_passed and not (
            self.observable and self.geometry_valid and position is not None
        ):
            raise ValueError("write_gate_passed 要求可观察、几何有效且状态完整")
        object.__setattr__(
            self,
            "source_camera",
            _required_identity(self.source_camera, "source_camera"),
        )
        object.__setattr__(
            self,
            "source_model_identity",
            _required_identity(self.source_model_identity, "source_model_identity"),
        )
        if self.frame_semantics != OBJECT_POSITION_FRAME_SEMANTICS:
            raise ValueError("ObjectMeasurement 只接受 robot-base position")
        if self.version != E018_OBJECT_MEMORY_VERSION:
            raise ValueError("ObjectMeasurement version 漂移")


@dataclass(frozen=True)
class ObjectMemorySafetyContext:
    """部署可得、每 Tick 显式提供的抓取前安全上下文。"""

    pregrasp_window_open: bool
    gripper_open: bool
    controller_tracking_valid: bool
    object_contact_detected: bool
    gripper_close_commanded: bool
    grasp_candidate: bool
    grasp_verified: bool
    object_maybe_moved: bool

    def __post_init__(self) -> None:
        for name in (
            "pregrasp_window_open",
            "gripper_open",
            "controller_tracking_valid",
            "object_contact_detected",
            "gripper_close_commanded",
            "grasp_candidate",
            "grasp_verified",
            "object_maybe_moved",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须为 bool")

    @property
    def invalidation_reasons(self) -> tuple[str, ...]:
        reasons: list[str] = []
        if not self.pregrasp_window_open:
            reasons.append("pregrasp_window_closed")
        if not self.gripper_open:
            reasons.append("gripper_not_open")
        if not self.controller_tracking_valid:
            reasons.append("controller_tracking_invalid")
        if self.object_contact_detected:
            reasons.append("object_contact_detected")
        if self.gripper_close_commanded:
            reasons.append("gripper_close_commanded")
        if self.grasp_candidate:
            reasons.append("grasp_candidate")
        if self.grasp_verified:
            reasons.append("grasp_verified")
        if self.object_maybe_moved:
            reasons.append("object_maybe_moved")
        return tuple(reasons)


@dataclass(frozen=True)
class ObjectState:
    episode_id: str
    mode: ObjectMemoryMode
    position_base_m: tuple[float, float, float] | np.ndarray | None
    covariance_base_m2: tuple[tuple[float, float, float], ...] | np.ndarray | None
    measurement_confidence: float
    last_observed_timestamp_s: float | None
    state_timestamp_s: float
    observable_now: bool
    valid: bool
    accepted_update_count: int
    source_camera: str | None
    source_model_identity: str | None
    invalid_reasons: tuple[str, ...]
    frame_semantics: str = OBJECT_POSITION_FRAME_SEMANTICS
    version: str = E018_OBJECT_MEMORY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("ObjectState episode_id 不能为空")
        try:
            mode = ObjectMemoryMode(self.mode)
        except ValueError as error:
            raise ValueError("ObjectState mode 无效") from error
        object.__setattr__(self, "mode", mode)
        position = _position(self.position_base_m, "state position_base_m")
        covariance = _covariance(self.covariance_base_m2, "state covariance_base_m2")
        object.__setattr__(self, "position_base_m", position)
        object.__setattr__(self, "covariance_base_m2", covariance)
        object.__setattr__(
            self,
            "measurement_confidence",
            _probability(self.measurement_confidence, "state measurement_confidence"),
        )
        state_timestamp = _timestamp(self.state_timestamp_s, "state timestamp")
        object.__setattr__(self, "state_timestamp_s", state_timestamp)
        if self.last_observed_timestamp_s is not None:
            last_observed = _timestamp(
                self.last_observed_timestamp_s,
                "last observed timestamp",
            )
            if last_observed > state_timestamp + 1e-12:
                raise ValueError("last observed 不能晚于 state timestamp")
            object.__setattr__(self, "last_observed_timestamp_s", last_observed)
        for value, name in ((self.observable_now, "observable_now"), (self.valid, "valid")):
            if not isinstance(value, bool):
                raise TypeError(f"{name} 必须为 bool")
        if (
            not isinstance(self.accepted_update_count, int)
            or isinstance(self.accepted_update_count, bool)
            or self.accepted_update_count < 0
        ):
            raise ValueError("accepted_update_count 必须是非负整数")
        source_camera = _identity(self.source_camera, "source_camera")
        source_model = _identity(self.source_model_identity, "source_model_identity")
        object.__setattr__(self, "source_camera", source_camera)
        object.__setattr__(self, "source_model_identity", source_model)
        if (source_camera is None) != (source_model is None):
            raise ValueError("ObjectState camera/model provenance 必须同时存在或同时缺失")
        reasons = _unique_reasons(self.invalid_reasons)
        object.__setattr__(self, "invalid_reasons", reasons)
        if position is None and (
            covariance is not None
            or self.last_observed_timestamp_s is not None
            or source_camera is not None
            or self.accepted_update_count != 0
            or self.measurement_confidence != 0.0
        ):
            raise ValueError("无历史位置的 ObjectState 不得携带 measurement 历史")
        if position is not None and (
            covariance is None
            or self.last_observed_timestamp_s is None
            or source_camera is None
            or self.accepted_update_count <= 0
        ):
            raise ValueError("有历史位置的 ObjectState 必须携带完整 covariance/provenance")
        if mode == ObjectMemoryMode.UNINITIALIZED:
            if position is not None or self.valid or reasons != ("memory_uninitialized",):
                raise ValueError("UNINITIALIZED ObjectState 语义不完整")
        elif mode == ObjectMemoryMode.FREE_STATIC:
            if position is None or not self.valid or reasons:
                raise ValueError("FREE_STATIC ObjectState 必须完整、有效且无 invalid reason")
        elif self.valid or not reasons:
            raise ValueError("INVALID ObjectState 必须无效且给出 invalid reason")
        if self.frame_semantics != OBJECT_POSITION_FRAME_SEMANTICS:
            raise ValueError("ObjectState 只允许 robot-base position")
        if self.version != E018_OBJECT_MEMORY_VERSION:
            raise ValueError("ObjectState version 漂移")

    @property
    def age_s(self) -> float | None:
        if self.last_observed_timestamp_s is None:
            return None
        return float(self.state_timestamp_s - self.last_observed_timestamp_s)

    @property
    def max_position_std_m(self) -> float | None:
        return _max_std(self.covariance_base_m2)


@dataclass(frozen=True)
class ObjectMemoryConfig:
    max_unobserved_age_s: float
    max_innovation_m: float
    max_position_std_m: float
    min_candidate_frames: int
    max_candidate_gap_s: float
    max_candidate_position_spread_m: float
    max_sensor_skew_s: float
    expected_source_camera: str
    expected_source_model_identity: str
    require_covariance: bool = True
    covariance_growth_m2_per_s: float = 0.0
    update_policy: str = OBJECT_MEMORY_UPDATE_POLICY
    frame_semantics: str = OBJECT_POSITION_FRAME_SEMANTICS
    version: str = E018_OBJECT_MEMORY_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.max_unobserved_age_s, "max_unobserved_age_s"),
            (self.max_innovation_m, "max_innovation_m"),
            (self.max_position_std_m, "max_position_std_m"),
            (self.max_candidate_gap_s, "max_candidate_gap_s"),
            (
                self.max_candidate_position_spread_m,
                "max_candidate_position_spread_m",
            ),
            (self.max_sensor_skew_s, "max_sensor_skew_s"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是有限正数")
        if (
            not isinstance(self.min_candidate_frames, int)
            or isinstance(self.min_candidate_frames, bool)
            or self.min_candidate_frames <= 0
        ):
            raise ValueError("min_candidate_frames 必须是正整数")
        if not isinstance(self.require_covariance, bool):
            raise TypeError("require_covariance 必须为 bool")
        object.__setattr__(
            self,
            "expected_source_camera",
            _required_identity(self.expected_source_camera, "expected_source_camera"),
        )
        object.__setattr__(
            self,
            "expected_source_model_identity",
            _required_identity(
                self.expected_source_model_identity,
                "expected_source_model_identity",
            ),
        )
        if (
            not math.isfinite(self.covariance_growth_m2_per_s)
            or self.covariance_growth_m2_per_s < 0.0
        ):
            raise ValueError("covariance_growth_m2_per_s 必须有限非负")
        if self.update_policy != OBJECT_MEMORY_UPDATE_POLICY:
            raise ValueError("Object memory P0 update policy 漂移")
        if self.frame_semantics != OBJECT_POSITION_FRAME_SEMANTICS:
            raise ValueError("Object memory config 只允许 robot-base position")
        if self.version != E018_OBJECT_MEMORY_VERSION:
            raise ValueError("Object memory config version 漂移")


@dataclass(frozen=True)
class ObjectCandidateDecision:
    """由 :class:`ObjectCandidateWindowVerifier` 生成的连续稳定窗口证明。"""

    measurement: ObjectMeasurement
    episode_id: str
    window_id: str | None
    window_start_timestamp_s: float | None
    frame_count: int
    max_position_spread_m: float | None
    verified: bool
    rejection_reasons: tuple[str, ...]
    version: str = E018_OBJECT_MEMORY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.measurement, ObjectMeasurement):
            raise TypeError("candidate measurement 必须是 ObjectMeasurement")
        _required_identity(self.episode_id, "candidate episode_id")
        window_id = _identity(self.window_id, "candidate window_id")
        object.__setattr__(self, "window_id", window_id)
        if self.window_start_timestamp_s is not None:
            start = _timestamp(self.window_start_timestamp_s, "candidate window start")
            if start > self.measurement.timestamp_s + 1e-12:
                raise ValueError("candidate window start 不能晚于 measurement")
            object.__setattr__(self, "window_start_timestamp_s", start)
        if (
            not isinstance(self.frame_count, int)
            or isinstance(self.frame_count, bool)
            or self.frame_count < 0
        ):
            raise ValueError("candidate frame_count 必须是非负整数")
        if self.max_position_spread_m is not None and (
            not math.isfinite(self.max_position_spread_m)
            or self.max_position_spread_m < 0.0
        ):
            raise ValueError("candidate max_position_spread_m 必须有限非负或 None")
        if not isinstance(self.verified, bool):
            raise TypeError("candidate verified 必须为 bool")
        reasons = _unique_reasons(self.rejection_reasons)
        object.__setattr__(self, "rejection_reasons", reasons)
        empty = self.frame_count == 0
        if empty != (window_id is None):
            raise ValueError("candidate 空窗口与 window_id 不一致")
        if empty != (self.window_start_timestamp_s is None):
            raise ValueError("candidate 空窗口与 start timestamp 不一致")
        if empty != (self.max_position_spread_m is None):
            raise ValueError("candidate 空窗口与 position spread 不一致")
        if self.verified and (empty or reasons):
            raise ValueError("verified candidate 必须是无拒绝原因的非空窗口")
        if not self.verified and not reasons:
            raise ValueError("未验证 candidate 必须给出拒绝原因")
        if self.version != E018_OBJECT_MEMORY_VERSION:
            raise ValueError("ObjectCandidateDecision version 漂移")


class ObjectCandidateWindowVerifier:
    """以部署侧证据构造滑动 candidate window，调用方不能直接传入 verified 布尔值。"""

    def __init__(self, config: ObjectMemoryConfig) -> None:
        if not isinstance(config, ObjectMemoryConfig):
            raise TypeError("config 必须是 ObjectMemoryConfig")
        self.config = config
        self._episode_id: str | None = None
        self._window_sequence = 0
        self._window_id: str | None = None
        self._window_start_timestamp_s: float | None = None
        self._measurements: list[ObjectMeasurement] = []
        self._last_timestamp_s: float | None = None

    def reset(self, episode_id: str) -> None:
        self._episode_id = _required_identity(episode_id, "candidate episode_id")
        self._window_sequence = 0
        self._window_id = None
        self._window_start_timestamp_s = None
        self._measurements = []
        self._last_timestamp_s = None

    def _clear_window(self) -> None:
        self._window_id = None
        self._window_start_timestamp_s = None
        self._measurements = []

    def _start_window(self, measurement: ObjectMeasurement) -> None:
        self._window_sequence += 1
        self._window_id = f"{self._episode_id}:object-candidate:{self._window_sequence}"
        self._window_start_timestamp_s = measurement.timestamp_s
        self._measurements = [measurement]

    def _sensor_skew_s(self, measurement: ObjectMeasurement) -> float:
        timestamps = (
            measurement.timestamp_s,
            measurement.rgb_timestamp_s,
            measurement.camera_pose_timestamp_s,
            measurement.tcp_pose_timestamp_s,
        )
        return float(max(timestamps) - min(timestamps))

    def _input_rejection_reasons(
        self,
        measurement: ObjectMeasurement,
        safety: ObjectMemorySafetyContext,
    ) -> tuple[str, ...]:
        reasons: list[str] = list(safety.invalidation_reasons)
        if not measurement.projection_valid:
            reasons.append("projection_invalid")
        elif not measurement.in_fov:
            reasons.append("out_of_fov")
        if not measurement.observable:
            reasons.append("not_observable")
        if not measurement.geometry_valid or measurement.position_base_m is None:
            reasons.append("geometry_invalid")
        if not measurement.write_gate_passed:
            reasons.append("write_gate_rejected")
        if self.config.require_covariance and measurement.covariance_base_m2 is None:
            reasons.append("measurement_covariance_missing")
        std = _max_std(measurement.covariance_base_m2)
        if std is not None and std > self.config.max_position_std_m + 1e-12:
            reasons.append("measurement_uncertain")
        if measurement.source_camera != self.config.expected_source_camera:
            reasons.append("source_camera_mismatch")
        if measurement.source_model_identity != self.config.expected_source_model_identity:
            reasons.append("source_model_identity_mismatch")
        if self._sensor_skew_s(measurement) > self.config.max_sensor_skew_s + 1e-12:
            reasons.append("sensor_timestamp_unsynchronized")
        return tuple(dict.fromkeys(reasons))

    @staticmethod
    def _position_spread(measurements: list[ObjectMeasurement]) -> float:
        positions = np.asarray(
            [measurement.position_base_m for measurement in measurements],
            dtype=np.float64,
        )
        if positions.ndim != 2 or positions.shape[1] != 3:
            raise RuntimeError("candidate window position contract 漂移")
        differences = positions[:, None, :] - positions[None, :, :]
        return float(np.linalg.norm(differences, axis=-1).max())

    def observe(
        self,
        measurement: ObjectMeasurement,
        *,
        episode_id: str,
        safety: ObjectMemorySafetyContext,
    ) -> ObjectCandidateDecision:
        if not isinstance(measurement, ObjectMeasurement):
            raise TypeError("measurement 必须是 ObjectMeasurement")
        if not isinstance(safety, ObjectMemorySafetyContext):
            raise TypeError("safety 必须是 ObjectMemorySafetyContext")
        if self._episode_id is None:
            raise RuntimeError("Object candidate verifier 必须先 reset(episode_id)")
        if episode_id != self._episode_id:
            raise ValueError("candidate episode identity 漂移；必须显式 reset")
        if (
            self._last_timestamp_s is not None
            and measurement.timestamp_s <= self._last_timestamp_s + 1e-12
        ):
            raise ValueError("Object candidate measurement timestamp 必须严格递增")
        previous_timestamp = self._last_timestamp_s
        self._last_timestamp_s = measurement.timestamp_s

        rejected = self._input_rejection_reasons(measurement, safety)
        if rejected:
            self._clear_window()
            return ObjectCandidateDecision(
                measurement=measurement,
                episode_id=episode_id,
                window_id=None,
                window_start_timestamp_s=None,
                frame_count=0,
                max_position_spread_m=None,
                verified=False,
                rejection_reasons=rejected,
            )

        restart_reason: str | None = None
        if (
            previous_timestamp is not None
            and measurement.timestamp_s - previous_timestamp
            > self.config.max_candidate_gap_s + 1e-12
        ):
            restart_reason = "candidate_time_gap"
            self._clear_window()
        if not self._measurements:
            self._start_window(measurement)
        else:
            trial = [*self._measurements, measurement]
            if (
                self._position_spread(trial)
                > self.config.max_candidate_position_spread_m + 1e-12
            ):
                restart_reason = "candidate_position_inconsistent"
                self._start_window(measurement)
            else:
                self._measurements.append(measurement)
                self._measurements = self._measurements[-self.config.min_candidate_frames :]
                self._window_start_timestamp_s = self._measurements[0].timestamp_s

        spread = self._position_spread(self._measurements)
        verified = len(self._measurements) >= self.config.min_candidate_frames
        reasons: tuple[str, ...]
        if verified:
            reasons = ()
        elif restart_reason is not None:
            reasons = (restart_reason, "candidate_too_short")
        else:
            reasons = ("candidate_too_short",)
        return ObjectCandidateDecision(
            measurement=measurement,
            episode_id=episode_id,
            window_id=self._window_id,
            window_start_timestamp_s=self._window_start_timestamp_s,
            frame_count=len(self._measurements),
            max_position_spread_m=spread,
            verified=verified,
            rejection_reasons=reasons,
        )


@dataclass(frozen=True)
class ObjectMemoryUpdate:
    state: ObjectState
    measurement_accepted: bool
    rejection_reasons: tuple[str, ...]
    innovation_m: float | None

    def __post_init__(self) -> None:
        if not isinstance(self.state, ObjectState):
            raise TypeError("ObjectMemoryUpdate.state 必须是 ObjectState")
        if not isinstance(self.measurement_accepted, bool):
            raise TypeError("measurement_accepted 必须为 bool")
        object.__setattr__(
            self,
            "rejection_reasons",
            _unique_reasons(self.rejection_reasons),
        )
        if self.measurement_accepted and self.rejection_reasons:
            raise ValueError("接受的 measurement 不得携带 rejection reason")
        if not self.measurement_accepted and not self.rejection_reasons:
            raise ValueError("拒绝的 measurement 必须携带 rejection reason")
        if self.innovation_m is not None and (
            not math.isfinite(self.innovation_m) or self.innovation_m < 0.0
        ):
            raise ValueError("innovation_m 必须是有限非负数或 None")


class ExplicitObjectStateMemory:
    """只在无接触 pregrasp 窗口保持已验证的静态 object position。"""

    def __init__(self, config: ObjectMemoryConfig) -> None:
        if not isinstance(config, ObjectMemoryConfig):
            raise TypeError("config 必须是 ObjectMemoryConfig")
        self.config = config
        self._state: ObjectState | None = None

    @property
    def state(self) -> ObjectState:
        if self._state is None:
            raise RuntimeError("Object memory 必须先 reset(episode_id)")
        return self._state

    def reset(self, episode_id: str, *, timestamp_s: float = 0.0) -> ObjectState:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id 不能为空")
        timestamp = _timestamp(timestamp_s, "reset timestamp")
        self._state = ObjectState(
            episode_id=episode_id,
            mode=ObjectMemoryMode.UNINITIALIZED,
            position_base_m=None,
            covariance_base_m2=None,
            measurement_confidence=0.0,
            last_observed_timestamp_s=None,
            state_timestamp_s=timestamp,
            observable_now=False,
            valid=False,
            accepted_update_count=0,
            source_camera=None,
            source_model_identity=None,
            invalid_reasons=("memory_uninitialized",),
        )
        return self._state

    def _aged_covariance(
        self,
        covariance: tuple[tuple[float, float, float], ...] | None,
        delta_s: float,
    ) -> tuple[tuple[float, float, float], ...] | None:
        if covariance is None:
            return None
        array = np.asarray(covariance, dtype=np.float64).copy()
        array += np.eye(3, dtype=np.float64) * (
            self.config.covariance_growth_m2_per_s * max(0.0, delta_s)
        )
        return _covariance(array, "aged covariance")

    def _retained_state(
        self,
        measurement: ObjectMeasurement,
        *,
        forced_reasons: tuple[str, ...] = (),
    ) -> ObjectState:
        previous = self.state
        covariance = self._aged_covariance(
            previous.covariance_base_m2,
            measurement.timestamp_s - previous.state_timestamp_s,
        )
        reasons: list[str] = list(forced_reasons)
        if previous.position_base_m is None:
            if not reasons:
                reasons.append("memory_uninitialized")
        else:
            if "measurement_conflict" in previous.invalid_reasons:
                reasons.append("measurement_conflict")
            if previous.last_observed_timestamp_s is None:
                raise RuntimeError("已初始化 ObjectState 缺少 last_observed timestamp")
            age = measurement.timestamp_s - previous.last_observed_timestamp_s
            if age > self.config.max_unobserved_age_s + 1e-12:
                reasons.append("memory_stale")
            std = _max_std(covariance)
            if self.config.require_covariance and covariance is None:
                reasons.append("memory_covariance_missing")
            elif std is not None and std > self.config.max_position_std_m + 1e-12:
                reasons.append("memory_uncertain")
        reasons = list(dict.fromkeys(reasons))
        if previous.position_base_m is None and reasons == ["memory_uninitialized"]:
            mode = ObjectMemoryMode.UNINITIALIZED
            valid = False
        elif reasons:
            mode = ObjectMemoryMode.INVALID
            valid = False
        else:
            mode = ObjectMemoryMode.FREE_STATIC
            valid = True
        state = ObjectState(
            episode_id=previous.episode_id,
            mode=mode,
            position_base_m=previous.position_base_m,
            covariance_base_m2=covariance,
            measurement_confidence=previous.measurement_confidence,
            last_observed_timestamp_s=previous.last_observed_timestamp_s,
            state_timestamp_s=measurement.timestamp_s,
            observable_now=measurement.observable,
            valid=valid,
            accepted_update_count=previous.accepted_update_count,
            source_camera=previous.source_camera,
            source_model_identity=previous.source_model_identity,
            invalid_reasons=tuple(reasons),
        )
        self._state = state
        return state

    def invalidate_for_safety(
        self,
        *,
        episode_id: str,
        timestamp_s: float,
        reasons: tuple[str, ...],
    ) -> ObjectState:
        """独立于 candidate 写入，按当前时刻单调失效既有 Memory。"""

        previous = self.state
        state = self._build_safety_invalidated_state(
            previous,
            episode_id=episode_id,
            timestamp_s=timestamp_s,
            reasons=reasons,
        )
        self._state = state
        return state

    def _build_safety_invalidated_state(
        self,
        previous: ObjectState,
        *,
        episode_id: str,
        timestamp_s: float,
        reasons: tuple[str, ...],
    ) -> ObjectState:
        """纯构造安全失效状态，供 delayed commit 在 mutation 前预演。"""

        if not isinstance(previous, ObjectState):
            raise TypeError("previous 必须是 ObjectState")
        if episode_id != previous.episode_id:
            raise ValueError("episode identity 漂移；必须显式 reset")
        timestamp = _timestamp(timestamp_s, "safety invalidation timestamp")
        if timestamp + 1e-12 < previous.state_timestamp_s:
            raise ValueError("safety invalidation timestamp 不能倒退")
        resolved_reasons = _unique_reasons(tuple(dict.fromkeys(reasons)))
        if not resolved_reasons:
            raise ValueError("safety invalidation 必须提供原因")
        covariance = self._aged_covariance(
            previous.covariance_base_m2,
            timestamp - previous.state_timestamp_s,
        )
        return ObjectState(
            episode_id=previous.episode_id,
            mode=ObjectMemoryMode.INVALID,
            position_base_m=previous.position_base_m,
            covariance_base_m2=covariance,
            measurement_confidence=previous.measurement_confidence,
            last_observed_timestamp_s=previous.last_observed_timestamp_s,
            state_timestamp_s=timestamp,
            observable_now=False,
            valid=False,
            accepted_update_count=previous.accepted_update_count,
            source_camera=previous.source_camera,
            source_model_identity=previous.source_model_identity,
            invalid_reasons=resolved_reasons,
        )

    def _measurement_rejection_reasons(
        self,
        candidate: ObjectCandidateDecision,
    ) -> tuple[str, ...]:
        measurement = candidate.measurement
        reasons: list[str] = []
        if not measurement.projection_valid:
            reasons.append("projection_invalid")
        elif not measurement.in_fov:
            reasons.append("out_of_fov")
        if not measurement.observable:
            reasons.append("not_observable")
        if not measurement.geometry_valid or measurement.position_base_m is None:
            reasons.append("geometry_invalid")
        if not measurement.write_gate_passed:
            reasons.append("write_gate_rejected")
        if self.config.require_covariance and measurement.covariance_base_m2 is None:
            reasons.append("measurement_covariance_missing")
        measurement_std = _max_std(measurement.covariance_base_m2)
        if (
            measurement_std is not None
            and measurement_std > self.config.max_position_std_m + 1e-12
        ):
            reasons.append("measurement_uncertain")
        if measurement.source_camera != self.config.expected_source_camera:
            reasons.append("source_camera_mismatch")
        if measurement.source_model_identity != self.config.expected_source_model_identity:
            reasons.append("source_model_identity_mismatch")
        timestamps = (
            measurement.timestamp_s,
            measurement.rgb_timestamp_s,
            measurement.camera_pose_timestamp_s,
            measurement.tcp_pose_timestamp_s,
        )
        if max(timestamps) - min(timestamps) > self.config.max_sensor_skew_s + 1e-12:
            reasons.append("sensor_timestamp_unsynchronized")
        if not candidate.verified:
            reasons.extend(candidate.rejection_reasons)
        elif candidate.frame_count < self.config.min_candidate_frames:
            raise RuntimeError("verified candidate frame_count 小于配置下限")
        return tuple(dict.fromkeys(reasons))

    def update(
        self,
        candidate: ObjectCandidateDecision,
        *,
        episode_id: str,
        safety: ObjectMemorySafetyContext,
    ) -> ObjectMemoryUpdate:
        if not isinstance(candidate, ObjectCandidateDecision):
            raise TypeError("candidate 必须是 ObjectCandidateDecision")
        if not isinstance(safety, ObjectMemorySafetyContext):
            raise TypeError("safety 必须是 ObjectMemorySafetyContext")
        measurement = candidate.measurement
        previous = self.state
        if episode_id != previous.episode_id:
            raise ValueError("episode identity 漂移；必须显式 reset")
        if candidate.episode_id != episode_id:
            raise ValueError("candidate 与 memory episode identity 不一致")
        if measurement.timestamp_s + 1e-12 < previous.state_timestamp_s:
            raise ValueError("Object measurement timestamp 不能倒退")

        forced_reasons = list(safety.invalidation_reasons)
        forced_reasons.extend(
            reason
            for reason in previous.invalid_reasons
            if reason in _IRREVERSIBLE_INVALID_REASONS
        )
        forced = tuple(dict.fromkeys(forced_reasons))
        if forced:
            state = self._retained_state(measurement, forced_reasons=forced)
            return ObjectMemoryUpdate(
                state=state,
                measurement_accepted=False,
                rejection_reasons=forced,
                innovation_m=None,
            )

        rejected = self._measurement_rejection_reasons(candidate)
        if rejected:
            provenance_failure = tuple(
                reason
                for reason in rejected
                if reason
                in {
                    "source_camera_mismatch",
                    "source_model_identity_mismatch",
                    "sensor_timestamp_unsynchronized",
                }
            )
            state = self._retained_state(
                measurement,
                forced_reasons=provenance_failure,
            )
            return ObjectMemoryUpdate(
                state=state,
                measurement_accepted=False,
                rejection_reasons=rejected,
                innovation_m=None,
            )

        if measurement.position_base_m is None or measurement.covariance_base_m2 is None:
            raise RuntimeError("通过 Object write gate 的 measurement 状态不完整")
        innovation = None
        if previous.position_base_m is not None:
            innovation = float(
                np.linalg.norm(
                    np.asarray(measurement.position_base_m, dtype=np.float64)
                    - np.asarray(previous.position_base_m, dtype=np.float64)
                )
            )
            if innovation > self.config.max_innovation_m + 1e-12:
                state = self._retained_state(
                    measurement,
                    forced_reasons=("measurement_conflict",),
                )
                return ObjectMemoryUpdate(
                    state=state,
                    measurement_accepted=False,
                    rejection_reasons=("measurement_conflict",),
                    innovation_m=innovation,
                )

        state = ObjectState(
            episode_id=previous.episode_id,
            mode=ObjectMemoryMode.FREE_STATIC,
            position_base_m=measurement.position_base_m,
            covariance_base_m2=measurement.covariance_base_m2,
            measurement_confidence=measurement.confidence,
            last_observed_timestamp_s=measurement.timestamp_s,
            state_timestamp_s=measurement.timestamp_s,
            observable_now=True,
            valid=True,
            accepted_update_count=previous.accepted_update_count + 1,
            source_camera=measurement.source_camera,
            source_model_identity=measurement.source_model_identity,
            invalid_reasons=(),
        )
        self._state = state
        return ObjectMemoryUpdate(
            state=state,
            measurement_accepted=True,
            rejection_reasons=(),
            innovation_m=innovation,
        )

    def preview_delayed_candidate(
        self,
        candidate: ObjectCandidateDecision,
        *,
        episode_id: str,
        safety: ObjectMemorySafetyContext,
        commit_timestamp_s: float,
        max_pending_age_s: float,
    ) -> ObjectMemoryUpdate:
        """纯预演离开观测视角后的 candidate 结果，不修改 Memory。

        成功状态保留 candidate 最后一帧的观测时间，以 HOME commit 时刻作为
        state time，并明确标记 ``observable_now=False``。调用方必须在全部
        postcondition 与 receipt 构造成功后再调用
        :meth:`apply_delayed_candidate_preview`。
        """

        if not isinstance(candidate, ObjectCandidateDecision):
            raise TypeError("candidate 必须是 ObjectCandidateDecision")
        if not isinstance(safety, ObjectMemorySafetyContext):
            raise TypeError("safety 必须是 ObjectMemorySafetyContext")
        commit_timestamp = _timestamp(commit_timestamp_s, "delayed commit timestamp")
        if not math.isfinite(max_pending_age_s) or max_pending_age_s <= 0.0:
            raise ValueError("max_pending_age_s 必须是有限正数")

        previous = self.state
        measurement = candidate.measurement
        if episode_id != previous.episode_id:
            raise ValueError("episode identity 漂移；必须显式 reset")
        if candidate.episode_id != episode_id:
            raise ValueError("candidate 与 memory episode identity 不一致")
        if commit_timestamp + 1e-12 < previous.state_timestamp_s:
            raise ValueError("delayed commit timestamp 不能早于当前 state")
        if commit_timestamp + 1e-12 < measurement.timestamp_s:
            raise ValueError("delayed commit timestamp 不能早于 candidate observation")

        pending_age_s = commit_timestamp - measurement.timestamp_s
        safety_reasons: list[str] = list(safety.invalidation_reasons)
        safety_reasons.extend(
            reason
            for reason in previous.invalid_reasons
            if reason in _IRREVERSIBLE_INVALID_REASONS
        )
        safety_reasons = list(dict.fromkeys(safety_reasons))
        reasons: list[str] = []
        reasons.extend(self._measurement_rejection_reasons(candidate))
        if pending_age_s > max_pending_age_s + 1e-12:
            reasons.append("pending_candidate_expired")
        if pending_age_s > self.config.max_unobserved_age_s + 1e-12:
            reasons.append("memory_stale_at_commit")

        aged_covariance = self._aged_covariance(
            measurement.covariance_base_m2,
            pending_age_s,
        )
        aged_std = _max_std(aged_covariance)
        if self.config.require_covariance and aged_covariance is None:
            reasons.append("measurement_covariance_missing")
        elif aged_std is not None and aged_std > self.config.max_position_std_m + 1e-12:
            reasons.append("memory_uncertain_at_commit")

        innovation = None
        if previous.position_base_m is not None and measurement.position_base_m is not None:
            innovation = float(
                np.linalg.norm(
                    np.asarray(measurement.position_base_m, dtype=np.float64)
                    - np.asarray(previous.position_base_m, dtype=np.float64)
                )
            )
            if innovation > self.config.max_innovation_m + 1e-12:
                reasons.append("measurement_conflict")

        rejected = tuple(dict.fromkeys((*safety_reasons, *reasons)))
        if rejected:
            state = previous
            if safety_reasons:
                state = self._build_safety_invalidated_state(
                    previous,
                    episode_id=episode_id,
                    timestamp_s=commit_timestamp,
                    reasons=tuple(safety_reasons),
                )
            return ObjectMemoryUpdate(
                state=state,
                measurement_accepted=False,
                rejection_reasons=rejected,
                innovation_m=innovation,
            )

        if measurement.position_base_m is None or aged_covariance is None:
            raise RuntimeError("通过 delayed Object write gate 的 measurement 状态不完整")
        state = ObjectState(
            episode_id=previous.episode_id,
            mode=ObjectMemoryMode.FREE_STATIC,
            position_base_m=measurement.position_base_m,
            covariance_base_m2=aged_covariance,
            measurement_confidence=measurement.confidence,
            last_observed_timestamp_s=measurement.timestamp_s,
            state_timestamp_s=commit_timestamp,
            observable_now=False,
            valid=True,
            accepted_update_count=previous.accepted_update_count + 1,
            source_camera=measurement.source_camera,
            source_model_identity=measurement.source_model_identity,
            invalid_reasons=(),
        )
        return ObjectMemoryUpdate(
            state=state,
            measurement_accepted=True,
            rejection_reasons=(),
            innovation_m=innovation,
        )

    def apply_delayed_candidate_preview(
        self,
        preview: ObjectMemoryUpdate,
        candidate: ObjectCandidateDecision,
        *,
        episode_id: str,
        safety: ObjectMemorySafetyContext,
        commit_timestamp_s: float,
        max_pending_age_s: float,
        expected_previous_state: ObjectState,
    ) -> ObjectMemoryUpdate:
        """复算并以单个无失败 assignment 应用已验证 preview。"""

        if not isinstance(preview, ObjectMemoryUpdate):
            raise TypeError("preview 必须是 ObjectMemoryUpdate")
        if not isinstance(expected_previous_state, ObjectState):
            raise TypeError("expected_previous_state 必须是 ObjectState")
        if self.state != expected_previous_state:
            raise RuntimeError("Object Memory 在 preview/apply 间发生变化")
        recomputed = self.preview_delayed_candidate(
            candidate,
            episode_id=episode_id,
            safety=safety,
            commit_timestamp_s=commit_timestamp_s,
            max_pending_age_s=max_pending_age_s,
        )
        if recomputed != preview:
            raise RuntimeError("delayed candidate preview identity 漂移")
        self._state = preview.state
        return preview

    def commit_delayed_candidate(
        self,
        candidate: ObjectCandidateDecision,
        *,
        episode_id: str,
        safety: ObjectMemorySafetyContext,
        commit_timestamp_s: float,
        max_pending_age_s: float,
    ) -> ObjectMemoryUpdate:
        """兼容入口：先纯预演，再原子应用；P0 ``update`` 语义不变。"""

        previous = self.state
        preview = self.preview_delayed_candidate(
            candidate,
            episode_id=episode_id,
            safety=safety,
            commit_timestamp_s=commit_timestamp_s,
            max_pending_age_s=max_pending_age_s,
        )
        return self.apply_delayed_candidate_preview(
            preview,
            candidate,
            episode_id=episode_id,
            safety=safety,
            commit_timestamp_s=commit_timestamp_s,
            max_pending_age_s=max_pending_age_s,
            expected_previous_state=previous,
        )


@dataclass(frozen=True)
class ObjectStateResolution:
    requirement: ObjectStateRequirement
    position_base_m: tuple[float, float, float] | None
    available: bool
    source: str | None
    memory_only: bool
    contact_authorized: bool
    version: str = E018_OBJECT_MEMORY_VERSION

    def __post_init__(self) -> None:
        try:
            requirement = ObjectStateRequirement(self.requirement)
        except ValueError as error:
            raise ValueError("ObjectStateResolution requirement 无效") from error
        object.__setattr__(self, "requirement", requirement)
        position = _position(self.position_base_m, "resolved position_base_m")
        object.__setattr__(self, "position_base_m", position)
        for value, name in (
            (self.available, "available"),
            (self.memory_only, "memory_only"),
            (self.contact_authorized, "contact_authorized"),
        ):
            if not isinstance(value, bool):
                raise TypeError(f"{name} 必须为 bool")
        source = _identity(self.source, "resolution source")
        object.__setattr__(self, "source", source)
        if self.available != (position is not None and source is not None):
            raise ValueError("resolution availability 与 position/source 不一致")
        if self.memory_only and (not self.available or source != "object_memory"):
            raise ValueError("memory_only resolution 语义冲突")
        if self.contact_authorized and (
            not self.available
            or source != "current_measurement"
            or requirement != ObjectStateRequirement.CONTACT_READY
        ):
            raise ValueError("接触授权只允许 CONTACT_READY 的当前 measurement")
        if requirement == ObjectStateRequirement.CONTACT_READY and self.memory_only:
            raise ValueError("CONTACT_READY 不得使用 memory-only state")
        if self.version != E018_OBJECT_MEMORY_VERSION:
            raise ValueError("ObjectStateResolution version 漂移")


def resolve_object_state(
    update: ObjectMemoryUpdate,
    *,
    requirement: ObjectStateRequirement,
) -> ObjectStateResolution:
    """按用途解析 object state；接触用途永远不接受 memory-only fallback。"""

    if not isinstance(update, ObjectMemoryUpdate):
        raise TypeError("update 必须是 ObjectMemoryUpdate")
    try:
        required = ObjectStateRequirement(requirement)
    except ValueError as error:
        raise ValueError("requirement 无效") from error
    state = update.state
    if update.measurement_accepted and state.observable_now:
        return ObjectStateResolution(
            requirement=required,
            position_base_m=state.position_base_m,
            available=True,
            source="current_measurement",
            memory_only=False,
            contact_authorized=required == ObjectStateRequirement.CONTACT_READY,
        )
    if required == ObjectStateRequirement.NAVIGATION and state.valid:
        return ObjectStateResolution(
            requirement=required,
            position_base_m=state.position_base_m,
            available=True,
            source="object_memory",
            memory_only=True,
            contact_authorized=False,
        )
    return ObjectStateResolution(
        requirement=required,
        position_base_m=None,
        available=False,
        source=None,
        memory_only=False,
        contact_authorized=False,
    )


@dataclass(frozen=True)
class DualPrecisionWorldState:
    """组合冻结 GoalState 与 E018 ObjectState，不改变 E015 Goal Memory。"""

    goal: GoalState
    object: ObjectState
    version: str = E018_DUAL_MEMORY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.goal, GoalState):
            raise TypeError("DualPrecisionWorldState.goal 必须是 GoalState")
        if not isinstance(self.object, ObjectState):
            raise TypeError("DualPrecisionWorldState.object 必须是 ObjectState")
        if self.goal.episode_id != self.object.episode_id:
            raise ValueError("Goal/Object memory episode identity 不一致")
        if self.version != E018_DUAL_MEMORY_VERSION:
            raise ValueError("DualPrecisionWorldState version 漂移")

    @property
    def estimated_goal_position_base_m(self) -> tuple[float, float, float] | None:
        return self.goal.position_base_m if self.goal.valid else None

    @property
    def estimated_object_position_for_navigation_base_m(
        self,
    ) -> tuple[float, float, float] | None:
        return self.object.position_base_m if self.object.valid else None


__all__ = [
    "E018_DUAL_MEMORY_VERSION",
    "E018_OBJECT_MEMORY_VERSION",
    "OBJECT_MEMORY_UPDATE_POLICY",
    "OBJECT_POSITION_FRAME_SEMANTICS",
    "DualPrecisionWorldState",
    "ExplicitObjectStateMemory",
    "ObjectCandidateDecision",
    "ObjectCandidateWindowVerifier",
    "ObjectMeasurement",
    "ObjectMemoryConfig",
    "ObjectMemoryMode",
    "ObjectMemorySafetyContext",
    "ObjectMemoryUpdate",
    "ObjectState",
    "ObjectStateRequirement",
    "ObjectStateResolution",
    "resolve_object_state",
]
