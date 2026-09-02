"""E015 episode-scoped 显式 base-frame goal state memory。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

E015_STATE_MEMORY_VERSION = "e015-explicit-goal-state-memory/v1"
GOAL_POSITION_FRAME_SEMANTICS = "position/robot-base/m/v1"
GOAL_MEMORY_UPDATE_POLICY = "direct-replace-reliable-hold-occluded/v1"


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


@dataclass(frozen=True)
class GoalMeasurement:
    """当前帧 measurement；它不等于可以供 controller 使用的 state。"""

    timestamp_s: float
    position_base_m: tuple[float, float, float] | np.ndarray | None
    covariance_base_m2: tuple[tuple[float, float, float], ...] | np.ndarray | None
    confidence: float
    goal_exists: bool
    projection_valid: bool
    in_fov: bool
    observable: bool
    geometry_valid: bool
    write_gate_passed: bool
    source: str = "precision-wrist-unet/v1"
    frame_semantics: str = GOAL_POSITION_FRAME_SEMANTICS

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s, "measurement timestamp"))
        position = _position(self.position_base_m, "measurement position_base_m")
        covariance = _covariance(self.covariance_base_m2, "measurement covariance_base_m2")
        object.__setattr__(self, "position_base_m", position)
        object.__setattr__(self, "covariance_base_m2", covariance)
        object.__setattr__(self, "confidence", _probability(self.confidence, "measurement confidence"))
        for name in (
            "goal_exists",
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
        if self.observable and not (self.goal_exists and self.projection_valid and self.in_fov):
            raise ValueError("observable 与 exists/projected/in_fov 语义冲突")
        if self.write_gate_passed and not (
            self.observable
            and self.geometry_valid
            and position is not None
        ):
            raise ValueError("write_gate_passed 要求可观察、几何有效且有 base-frame position")
        if covariance is not None and position is None:
            raise ValueError("measurement covariance 不能脱离 position")
        if not isinstance(self.source, str) or not self.source.strip():
            raise ValueError("measurement source 不能为空")
        if self.frame_semantics != GOAL_POSITION_FRAME_SEMANTICS:
            raise ValueError("GoalMeasurement 只接受 robot-base frame")


@dataclass(frozen=True)
class GoalState:
    episode_id: str
    position_base_m: tuple[float, float, float] | None
    covariance_base_m2: tuple[tuple[float, float, float], ...] | None
    measurement_confidence: float
    last_observed_timestamp_s: float | None
    state_timestamp_s: float
    observable_now: bool
    valid: bool
    accepted_update_count: int
    source: str | None
    invalid_reasons: tuple[str, ...]
    frame_semantics: str = GOAL_POSITION_FRAME_SEMANTICS
    version: str = E015_STATE_MEMORY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("GoalState episode_id 不能为空")
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
            last = _timestamp(self.last_observed_timestamp_s, "last observed timestamp")
            if last > state_timestamp + 1e-12:
                raise ValueError("last observed 不能晚于 state timestamp")
            object.__setattr__(self, "last_observed_timestamp_s", last)
        for value, name in ((self.observable_now, "observable_now"), (self.valid, "valid")):
            if not isinstance(value, bool):
                raise TypeError(f"{name} 必须为 bool")
        if (
            not isinstance(self.accepted_update_count, int)
            or isinstance(self.accepted_update_count, bool)
            or self.accepted_update_count < 0
        ):
            raise ValueError("accepted_update_count 必须是非负整数")
        if position is None and (
            any(
                value is not None
                for value in (covariance, self.last_observed_timestamp_s, self.source)
            )
            or self.accepted_update_count != 0
            or self.measurement_confidence != 0.0
        ):
            raise ValueError("未初始化 GoalState 不得携带历史 measurement")
        if self.valid and (
            position is None
            or self.last_observed_timestamp_s is None
            or self.accepted_update_count <= 0
            or self.invalid_reasons
        ):
            raise ValueError("有效 GoalState 必须完整且不能含 invalid_reasons")
        if not self.valid and not self.invalid_reasons:
            raise ValueError("无效 GoalState 必须给出 invalid_reasons")
        if self.source is not None and (not isinstance(self.source, str) or not self.source.strip()):
            raise ValueError("GoalState source 必须为非空字符串或 None")
        if len(set(self.invalid_reasons)) != len(self.invalid_reasons) or any(
            not isinstance(reason, str) or not reason for reason in self.invalid_reasons
        ):
            raise ValueError("invalid_reasons 必须是互异非空字符串")
        if self.frame_semantics != GOAL_POSITION_FRAME_SEMANTICS:
            raise ValueError("GoalState 只允许 robot-base frame")
        if self.version != E015_STATE_MEMORY_VERSION:
            raise ValueError("GoalState version 漂移")

    @property
    def age_s(self) -> float | None:
        if self.last_observed_timestamp_s is None:
            return None
        return float(self.state_timestamp_s - self.last_observed_timestamp_s)

    @property
    def max_position_std_m(self) -> float | None:
        return _max_std(self.covariance_base_m2)


@dataclass(frozen=True)
class ObjectState:
    """为后续 grasp-relative object memory 保留的显式接口；E015 不更新它。"""

    position_base_m: tuple[float, float, float] | None = None
    pose_tcp: tuple[float, ...] | None = None
    covariance_base_m2: tuple[tuple[float, float, float], ...] | None = None
    observable_now: bool = False
    grasped: bool = False
    valid: bool = False


@dataclass(frozen=True)
class PrecisionWorldState:
    goal: GoalState
    object: ObjectState | None = None
    version: str = E015_STATE_MEMORY_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.goal, GoalState):
            raise TypeError("PrecisionWorldState.goal 必须是 GoalState")
        if self.object is not None and not isinstance(self.object, ObjectState):
            raise TypeError("PrecisionWorldState.object 必须是 ObjectState 或 None")
        if self.version != E015_STATE_MEMORY_VERSION:
            raise ValueError("PrecisionWorldState version 漂移")

    @property
    def estimated_goal_position_base_m(self) -> tuple[float, float, float] | None:
        return self.goal.position_base_m if self.goal.valid else None


@dataclass(frozen=True)
class GoalMemoryConfig:
    max_unobserved_age_s: float
    max_innovation_m: float
    max_position_std_m: float
    require_covariance: bool = True
    covariance_growth_m2_per_s: float = 0.0
    update_policy: str = GOAL_MEMORY_UPDATE_POLICY
    frame_semantics: str = GOAL_POSITION_FRAME_SEMANTICS
    version: str = E015_STATE_MEMORY_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.max_unobserved_age_s, "max_unobserved_age_s"),
            (self.max_innovation_m, "max_innovation_m"),
            (self.max_position_std_m, "max_position_std_m"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} 必须是有限正数")
        if not isinstance(self.require_covariance, bool):
            raise TypeError("require_covariance 必须为 bool")
        if (
            not math.isfinite(self.covariance_growth_m2_per_s)
            or self.covariance_growth_m2_per_s < 0.0
        ):
            raise ValueError("covariance_growth_m2_per_s 必须有限非负")
        if self.update_policy != GOAL_MEMORY_UPDATE_POLICY:
            raise ValueError("Goal memory v1 只允许 direct-replace/hold policy")
        if self.frame_semantics != GOAL_POSITION_FRAME_SEMANTICS:
            raise ValueError("Goal memory config 只允许 robot-base frame")
        if self.version != E015_STATE_MEMORY_VERSION:
            raise ValueError("Goal memory config version 漂移")


@dataclass(frozen=True)
class GoalMemoryUpdate:
    state: GoalState
    measurement_accepted: bool
    rejection_reasons: tuple[str, ...]
    innovation_m: float | None


class ExplicitGoalStateMemory:
    """只接受可靠 base-frame measurement，并在短暂不可见时保留状态。"""

    def __init__(self, config: GoalMemoryConfig) -> None:
        if not isinstance(config, GoalMemoryConfig):
            raise TypeError("config 必须是 GoalMemoryConfig")
        self.config = config
        self._state: GoalState | None = None

    @property
    def state(self) -> GoalState:
        if self._state is None:
            raise RuntimeError("Goal memory 必须先 reset(episode_id)")
        return self._state

    @property
    def world_state(self) -> PrecisionWorldState:
        return PrecisionWorldState(goal=self.state)

    def reset(self, episode_id: str, *, timestamp_s: float = 0.0) -> GoalState:
        if not isinstance(episode_id, str) or not episode_id.strip():
            raise ValueError("episode_id 不能为空")
        timestamp = _timestamp(timestamp_s, "reset timestamp")
        self._state = GoalState(
            episode_id=episode_id,
            position_base_m=None,
            covariance_base_m2=None,
            measurement_confidence=0.0,
            last_observed_timestamp_s=None,
            state_timestamp_s=timestamp,
            observable_now=False,
            valid=False,
            accepted_update_count=0,
            source=None,
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

    def _hold(
        self,
        measurement: GoalMeasurement,
        *,
        rejection_reasons: tuple[str, ...],
        force_invalid_reason: str | None = None,
    ) -> GoalState:
        previous = self.state
        covariance = self._aged_covariance(
            previous.covariance_base_m2,
            measurement.timestamp_s - previous.state_timestamp_s,
        )
        reasons: list[str] = []
        if previous.position_base_m is None:
            reasons.append("memory_uninitialized")
        if "measurement_conflict" in previous.invalid_reasons:
            reasons.append("measurement_conflict")
        age = (
            None
            if previous.last_observed_timestamp_s is None
            else measurement.timestamp_s - previous.last_observed_timestamp_s
        )
        if age is not None and age > self.config.max_unobserved_age_s + 1e-12:
            reasons.append("memory_stale")
        std = _max_std(covariance)
        if self.config.require_covariance and covariance is None:
            reasons.append("memory_covariance_missing")
        elif std is not None and std > self.config.max_position_std_m + 1e-12:
            reasons.append("memory_uncertain")
        if force_invalid_reason is not None and force_invalid_reason not in reasons:
            reasons.append(force_invalid_reason)
        valid = previous.position_base_m is not None and not reasons
        state = GoalState(
            episode_id=previous.episode_id,
            position_base_m=previous.position_base_m,
            covariance_base_m2=covariance,
            measurement_confidence=previous.measurement_confidence,
            last_observed_timestamp_s=previous.last_observed_timestamp_s,
            state_timestamp_s=measurement.timestamp_s,
            observable_now=measurement.observable,
            valid=valid,
            accepted_update_count=previous.accepted_update_count,
            source=previous.source,
            invalid_reasons=tuple(reasons),
        )
        self._state = state
        return state

    def update(self, measurement: GoalMeasurement, *, episode_id: str) -> GoalMemoryUpdate:
        if not isinstance(measurement, GoalMeasurement):
            raise TypeError("measurement 必须是 GoalMeasurement")
        previous = self.state
        if episode_id != previous.episode_id:
            raise ValueError("episode identity 漂移；必须显式 reset")
        if measurement.timestamp_s + 1e-12 < previous.state_timestamp_s:
            raise ValueError("Goal measurement timestamp 不能倒退")
        if not measurement.goal_exists:
            self._state = GoalState(
                episode_id=previous.episode_id,
                position_base_m=None,
                covariance_base_m2=None,
                measurement_confidence=0.0,
                last_observed_timestamp_s=None,
                state_timestamp_s=measurement.timestamp_s,
                observable_now=False,
                valid=False,
                accepted_update_count=0,
                source=None,
                invalid_reasons=("goal_absent",),
            )
            return GoalMemoryUpdate(
                state=self._state,
                measurement_accepted=False,
                rejection_reasons=("goal_absent",),
                innovation_m=None,
            )

        rejected: list[str] = []
        if not measurement.projection_valid:
            rejected.append("projection_invalid")
        elif not measurement.in_fov:
            rejected.append("out_of_fov")
        if not measurement.observable:
            rejected.append("not_observable")
        if not measurement.geometry_valid or measurement.position_base_m is None:
            rejected.append("geometry_invalid")
        if not measurement.write_gate_passed:
            rejected.append("write_gate_rejected")
        if self.config.require_covariance and measurement.covariance_base_m2 is None:
            rejected.append("measurement_covariance_missing")
        measurement_std = _max_std(measurement.covariance_base_m2)
        if (
            measurement_std is not None
            and measurement_std > self.config.max_position_std_m + 1e-12
        ):
            rejected.append("measurement_uncertain")
        rejected = list(dict.fromkeys(rejected))
        if rejected:
            state = self._hold(
                measurement,
                rejection_reasons=tuple(rejected),
            )
            return GoalMemoryUpdate(
                state=state,
                measurement_accepted=False,
                rejection_reasons=tuple(rejected),
                innovation_m=None,
            )

        if measurement.position_base_m is None:
            raise RuntimeError("通过 write gate 的 measurement 缺少 position")
        innovation = None
        if previous.position_base_m is not None:
            innovation = float(
                np.linalg.norm(
                    np.asarray(measurement.position_base_m, dtype=np.float64)
                    - np.asarray(previous.position_base_m, dtype=np.float64)
                )
            )
            if innovation > self.config.max_innovation_m + 1e-12:
                state = self._hold(
                    measurement,
                    rejection_reasons=("measurement_conflict",),
                    force_invalid_reason="measurement_conflict",
                )
                return GoalMemoryUpdate(
                    state=state,
                    measurement_accepted=False,
                    rejection_reasons=("measurement_conflict",),
                    innovation_m=innovation,
                )

        state = GoalState(
            episode_id=previous.episode_id,
            position_base_m=measurement.position_base_m,
            covariance_base_m2=measurement.covariance_base_m2,
            measurement_confidence=measurement.confidence,
            last_observed_timestamp_s=measurement.timestamp_s,
            state_timestamp_s=measurement.timestamp_s,
            observable_now=True,
            valid=True,
            accepted_update_count=previous.accepted_update_count + 1,
            source=measurement.source,
            invalid_reasons=(),
        )
        self._state = state
        return GoalMemoryUpdate(
            state=state,
            measurement_accepted=True,
            rejection_reasons=(),
            innovation_m=innovation,
        )


__all__ = [
    "E015_STATE_MEMORY_VERSION",
    "GOAL_MEMORY_UPDATE_POLICY",
    "GOAL_POSITION_FRAME_SEMANTICS",
    "ExplicitGoalStateMemory",
    "GoalMeasurement",
    "GoalMemoryConfig",
    "GoalMemoryUpdate",
    "GoalState",
    "ObjectState",
    "PrecisionWorldState",
]
