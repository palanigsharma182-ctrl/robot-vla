"""E018-P0 validation-only object write calibration 与 deterministic replay。"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from robot_vla.precision.object_memory import (
    E018_OBJECT_MEMORY_VERSION,
    ExplicitObjectStateMemory,
    ObjectCandidateWindowVerifier,
    ObjectMeasurement,
    ObjectMemoryConfig,
    ObjectMemorySafetyContext,
    ObjectStateRequirement,
    resolve_object_state,
)
from robot_vla.precision.object_observability import OBJECT_WRITE_SCORE_SEMANTICS

E018_OBJECT_WRITE_CALIBRATION_VERSION = "e018-p0-object-write-calibration/v1"
E018_OBJECT_MEMORY_REPLAY_VERSION = "e018-p0-object-memory-replay/v1"
OBJECT_WRITE_THRESHOLD_POLICY = "maximum-coverage-zero-unsafe-on-validation/v1"


def _probability(value: float, name: str) -> float:
    candidate = float(value)
    if not math.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
        raise ValueError(f"{name} 必须是 [0,1] 内有限数值")
    return candidate


def _xyz(
    value: Sequence[float] | np.ndarray | None,
    name: str,
) -> tuple[float, float, float] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是有限 XYZ 或 None")
    return tuple(float(item) for item in array)


def _covariance(
    value: Sequence[Sequence[float]] | np.ndarray | None,
) -> tuple[tuple[float, float, float], ...] | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3, 3) or not np.isfinite(array).all():
        raise ValueError("measurement_covariance_base_m2 必须是有限 [3,3]")
    return tuple(tuple(float(item) for item in row) for row in array)


@dataclass(frozen=True)
class ObjectWriteCalibration:
    enabled: bool
    threshold: float
    validation_frame_count: int
    structurally_eligible_count: int
    oracle_safe_count: int
    accepted_count: int
    accepted_unsafe_count: int
    safe_coverage: float
    threshold_policy: str = OBJECT_WRITE_THRESHOLD_POLICY
    score_semantics: str = OBJECT_WRITE_SCORE_SEMANTICS
    version: str = E018_OBJECT_WRITE_CALIBRATION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("object write calibration enabled 必须为 bool")
        _probability(self.threshold, "object write threshold")
        for name in (
            "validation_frame_count",
            "structurally_eligible_count",
            "oracle_safe_count",
            "accepted_count",
            "accepted_unsafe_count",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必须是非负整数")
        _probability(self.safe_coverage, "safe_coverage")
        if self.accepted_unsafe_count != 0:
            raise ValueError("冻结 object write calibration 不允许 validation unsafe acceptance")
        if self.threshold_policy != OBJECT_WRITE_THRESHOLD_POLICY:
            raise ValueError("object write threshold policy 漂移")
        if self.score_semantics != OBJECT_WRITE_SCORE_SEMANTICS:
            raise ValueError("object write score semantics 漂移")
        if self.version != E018_OBJECT_WRITE_CALIBRATION_VERSION:
            raise ValueError("object write calibration version 漂移")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrate_object_write_threshold(
    *,
    scores: Sequence[float],
    structurally_eligible: Sequence[bool],
    oracle_safe: Sequence[bool],
) -> ObjectWriteCalibration:
    """选择 validation 上零 unsafe acceptance 时覆盖率最大的 score threshold。"""

    score = np.asarray(scores, dtype=np.float64)
    eligible = np.asarray(structurally_eligible)
    safe = np.asarray(oracle_safe)
    if score.ndim != 1 or score.size == 0 or not np.isfinite(score).all():
        raise ValueError("object write calibration scores 必须是非空有限一维")
    if np.any(score < 0.0) or np.any(score > 1.0):
        raise ValueError("object write calibration scores 必须位于 [0,1]")
    if eligible.shape != score.shape or eligible.dtype != np.bool_:
        raise ValueError("structurally_eligible 必须是 bool [N]")
    if safe.shape != score.shape or safe.dtype != np.bool_:
        raise ValueError("oracle_safe 必须是 bool [N]")

    best: tuple[int, float] | None = None
    for threshold in sorted({float(value) for value in score[eligible]}, reverse=True):
        accepted = eligible & (score >= threshold)
        if np.any(accepted & ~safe):
            continue
        candidate = (int(accepted.sum()), threshold)
        if candidate[0] > 0 and (best is None or candidate > best):
            best = candidate
    safe_count = int(safe.sum())
    if best is None:
        return ObjectWriteCalibration(
            enabled=False,
            threshold=1.0,
            validation_frame_count=int(score.size),
            structurally_eligible_count=int(eligible.sum()),
            oracle_safe_count=safe_count,
            accepted_count=0,
            accepted_unsafe_count=0,
            safe_coverage=0.0,
        )
    accepted_count, threshold = best
    accepted = eligible & (score >= threshold)
    return ObjectWriteCalibration(
        enabled=True,
        threshold=threshold,
        validation_frame_count=int(score.size),
        structurally_eligible_count=int(eligible.sum()),
        oracle_safe_count=safe_count,
        accepted_count=accepted_count,
        accepted_unsafe_count=int(np.sum(accepted & ~safe)),
        safe_coverage=float(accepted_count / safe_count) if safe_count else 0.0,
    )


@dataclass(frozen=True)
class ObjectReplayFrame:
    episode_id: str
    timestep: int
    timestamp_s: float
    rgb_timestamp_s: float
    camera_pose_timestamp_s: float
    tcp_pose_timestamp_s: float
    predicted_position_base_m: tuple[float, float, float] | np.ndarray | None
    measurement_covariance_base_m2: tuple[tuple[float, float, float], ...] | np.ndarray | None
    write_score: float
    structurally_eligible: bool
    predicted_observable: bool
    geometry_valid: bool
    gt_position_base_m: tuple[float, float, float] | np.ndarray
    gt_observable: bool
    oracle_safe_measurement: bool
    safety: ObjectMemorySafetyContext
    source_camera: str
    source_model_identity: str

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("object replay episode_id 不能为空")
        if not isinstance(self.timestep, int) or isinstance(self.timestep, bool) or self.timestep < 0:
            raise ValueError("object replay timestep 必须是非负整数")
        for name in (
            "timestamp_s",
            "rgb_timestamp_s",
            "camera_pose_timestamp_s",
            "tcp_pose_timestamp_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} 必须有限非负")
            if name != "timestamp_s" and value > self.timestamp_s + 1e-12:
                raise ValueError(f"{name} 不能晚于 replay Tick")
            object.__setattr__(self, name, value)
        object.__setattr__(
            self,
            "predicted_position_base_m",
            _xyz(self.predicted_position_base_m, "predicted_position_base_m"),
        )
        object.__setattr__(
            self,
            "measurement_covariance_base_m2",
            _covariance(self.measurement_covariance_base_m2),
        )
        gt_position = _xyz(self.gt_position_base_m, "gt_position_base_m")
        if gt_position is None:
            raise ValueError("gt_position_base_m 不能为空")
        object.__setattr__(self, "gt_position_base_m", gt_position)
        object.__setattr__(self, "write_score", _probability(self.write_score, "write_score"))
        for name in (
            "structurally_eligible",
            "predicted_observable",
            "geometry_valid",
            "gt_observable",
            "oracle_safe_measurement",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须为 bool")
        if not isinstance(self.safety, ObjectMemorySafetyContext):
            raise TypeError("replay safety 必须是 ObjectMemorySafetyContext")
        if not isinstance(self.source_camera, str) or not self.source_camera.strip():
            raise ValueError("source_camera 不能为空")
        if not isinstance(self.source_model_identity, str) or not self.source_model_identity.strip():
            raise ValueError("source_model_identity 不能为空")
        if (self.predicted_position_base_m is None) != (
            self.measurement_covariance_base_m2 is None
        ):
            raise ValueError("object replay position/covariance 必须同时存在或同时缺失")


@dataclass(frozen=True)
class ObjectMemoryReplayRecord:
    episode_id: str
    timestep: int
    timestamp_s: float
    pregrasp_window_open: bool
    gt_observable: bool
    oracle_safe_measurement: bool
    direct_write_gate_passed: bool
    candidate_verified: bool
    candidate_frame_count: int
    candidate_rejection_reasons: tuple[str, ...]
    current_measurement_accepted: bool
    current_world_xyz_error_m: float | None
    memory_valid: bool
    memory_observable_now: bool
    memory_only: bool
    navigation_available: bool
    contact_authorized: bool
    memory_world_xyz_error_m: float | None
    memory_age_s: float | None
    measurement_rejection_reasons: tuple[str, ...]
    innovation_m: float | None
    memory_mode: str

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["candidate_rejection_reasons"] = list(self.candidate_rejection_reasons)
        payload["measurement_rejection_reasons"] = list(
            self.measurement_rejection_reasons
        )
        return payload


def replay_object_memory(
    frames: Sequence[ObjectReplayFrame],
    *,
    calibration: ObjectWriteCalibration,
    memory_config: ObjectMemoryConfig,
) -> tuple[list[ObjectMemoryReplayRecord], int]:
    """按 Episode/Tick replay，并返回记录及显式 reset leakage count。"""

    if not frames:
        raise ValueError("object memory replay frames 不能为空")
    ordered = sorted(frames, key=lambda row: (row.episode_id, row.timestamp_s, row.timestep))
    seen: set[tuple[str, int]] = set()
    records: list[ObjectMemoryReplayRecord] = []
    memory = ExplicitObjectStateMemory(memory_config)
    verifier = ObjectCandidateWindowVerifier(memory_config)
    active_episode: str | None = None
    reset_leakage_count = 0
    for frame in ordered:
        identity = (frame.episode_id, frame.timestep)
        if identity in seen:
            raise ValueError("object memory replay episode/timestep 重复")
        seen.add(identity)
        if frame.episode_id != active_episode:
            reset = memory.reset(frame.episode_id, timestamp_s=frame.timestamp_s)
            verifier.reset(frame.episode_id)
            if reset.position_base_m is not None or reset.valid or reset.accepted_update_count != 0:
                reset_leakage_count += 1
            active_episode = frame.episode_id

        write_accepted = bool(
            calibration.enabled
            and frame.structurally_eligible
            and frame.write_score >= calibration.threshold
        )
        measurement = ObjectMeasurement(
            timestamp_s=frame.timestamp_s,
            rgb_timestamp_s=frame.rgb_timestamp_s,
            camera_pose_timestamp_s=frame.camera_pose_timestamp_s,
            tcp_pose_timestamp_s=frame.tcp_pose_timestamp_s,
            position_base_m=frame.predicted_position_base_m,
            covariance_base_m2=frame.measurement_covariance_base_m2,
            confidence=frame.write_score,
            projection_valid=frame.geometry_valid,
            in_fov=frame.geometry_valid,
            observable=frame.predicted_observable and frame.geometry_valid,
            geometry_valid=frame.geometry_valid,
            write_gate_passed=write_accepted,
            source_camera=frame.source_camera,
            source_model_identity=frame.source_model_identity,
        )
        candidate = verifier.observe(
            measurement,
            episode_id=frame.episode_id,
            safety=frame.safety,
        )
        update = memory.update(
            candidate,
            episode_id=frame.episode_id,
            safety=frame.safety,
        )
        navigation = resolve_object_state(
            update,
            requirement=ObjectStateRequirement.NAVIGATION,
        )
        contact = resolve_object_state(
            update,
            requirement=ObjectStateRequirement.CONTACT_READY,
        )
        current_error = (
            None
            if frame.predicted_position_base_m is None
            else float(
                np.linalg.norm(
                    np.asarray(frame.predicted_position_base_m)
                    - np.asarray(frame.gt_position_base_m)
                )
            )
        )
        memory_error = (
            None
            if update.state.position_base_m is None
            else float(
                np.linalg.norm(
                    np.asarray(update.state.position_base_m)
                    - np.asarray(frame.gt_position_base_m)
                )
            )
        )
        records.append(
            ObjectMemoryReplayRecord(
                episode_id=frame.episode_id,
                timestep=frame.timestep,
                timestamp_s=frame.timestamp_s,
                pregrasp_window_open=frame.safety.pregrasp_window_open,
                gt_observable=frame.gt_observable,
                oracle_safe_measurement=frame.oracle_safe_measurement,
                direct_write_gate_passed=write_accepted,
                candidate_verified=candidate.verified,
                candidate_frame_count=candidate.frame_count,
                candidate_rejection_reasons=candidate.rejection_reasons,
                current_measurement_accepted=update.measurement_accepted,
                current_world_xyz_error_m=current_error,
                memory_valid=update.state.valid,
                memory_observable_now=update.state.observable_now,
                memory_only=navigation.memory_only,
                navigation_available=navigation.available,
                contact_authorized=contact.contact_authorized,
                memory_world_xyz_error_m=memory_error,
                memory_age_s=update.state.age_s,
                measurement_rejection_reasons=update.rejection_reasons,
                innovation_m=update.innovation_m,
                memory_mode=update.state.mode.value,
            )
        )
    return records, reset_leakage_count


def summarize_object_memory_replay(
    records: Sequence[ObjectMemoryReplayRecord],
    *,
    catastrophic_world_xyz_error_m: float,
    reset_leakage_count: int = 0,
) -> dict[str, object]:
    if not records:
        raise ValueError("object memory replay records 不能为空")
    if (
        not math.isfinite(catastrophic_world_xyz_error_m)
        or catastrophic_world_xyz_error_m <= 0.0
    ):
        raise ValueError("catastrophic threshold 必须是有限正数")
    if reset_leakage_count < 0:
        raise ValueError("reset_leakage_count 必须非负")

    pregrasp = [row for row in records if row.pregrasp_window_open]
    after_pregrasp = [row for row in records if not row.pregrasp_window_open]
    current = [row for row in pregrasp if row.current_measurement_accepted]
    direct_gate = [row for row in pregrasp if row.direct_write_gate_passed]
    memory_valid = [row for row in pregrasp if row.memory_valid]
    held = [row for row in pregrasp if row.memory_valid and row.memory_only]
    gt_unobservable = [row for row in pregrasp if not row.gt_observable]
    direct_unavailable = [row for row in pregrasp if not row.current_measurement_accepted]
    memory_gt_unobservable = [row for row in gt_unobservable if row.memory_valid]
    memory_direct_unavailable = [row for row in direct_unavailable if row.memory_valid]
    episodes_with_memory = {row.episode_id for row in pregrasp if row.memory_valid}
    all_episodes = {row.episode_id for row in pregrasp}
    initialized_episodes: set[str] = set()
    cold_start_unobservable: list[ObjectMemoryReplayRecord] = []
    unobservable_after_initialization: list[ObjectMemoryReplayRecord] = []
    for row in pregrasp:
        if row.memory_valid or row.memory_age_s is not None:
            initialized_episodes.add(row.episode_id)
        if not row.gt_observable:
            target = (
                unobservable_after_initialization
                if row.episode_id in initialized_episodes
                else cold_start_unobservable
            )
            target.append(row)

    def errors(
        rows: Sequence[ObjectMemoryReplayRecord],
        *,
        memory: bool,
    ) -> np.ndarray:
        values = [
            row.memory_world_xyz_error_m if memory else row.current_world_xyz_error_m
            for row in rows
        ]
        return np.asarray(
            [float(value) for value in values if value is not None],
            dtype=np.float64,
        )

    def distribution(values: np.ndarray) -> dict[str, float | int | None]:
        if values.size == 0:
            return {"count": 0, "p50_mm": None, "p90_mm": None, "max_mm": None}
        return {
            "count": int(values.size),
            "p50_mm": float(np.quantile(values, 0.50) * 1000.0),
            "p90_mm": float(np.quantile(values, 0.90) * 1000.0),
            "max_mm": float(values.max() * 1000.0),
        }

    current_errors = errors(current, memory=False)
    memory_errors = errors(memory_valid, memory=True)
    memory_gt_unobservable_errors = errors(memory_gt_unobservable, memory=True)
    rejection_counts = Counter(
        reason for row in pregrasp for reason in row.measurement_rejection_reasons
    )
    unsafe_updates = [
        row
        for row in pregrasp
        if row.current_measurement_accepted and not row.oracle_safe_measurement
    ]
    return {
        "version": E018_OBJECT_MEMORY_REPLAY_VERSION,
        "memory_contract_version": E018_OBJECT_MEMORY_VERSION,
        "frame_count": len(records),
        "pregrasp_frame_count": len(pregrasp),
        "post_pregrasp_frame_count": len(after_pregrasp),
        "episode_count": len({row.episode_id for row in records}),
        "episodes_with_initialized_memory_count": len(episodes_with_memory),
        "episodes_without_initialized_memory_count": len(all_episodes - episodes_with_memory),
        "direct_write_gate_count": len(direct_gate),
        "direct_write_gate_coverage": len(direct_gate) / len(pregrasp),
        "current_candidate_valid_count": len(current),
        "current_candidate_coverage": len(current) / len(pregrasp),
        "object_memory_valid_count": len(memory_valid),
        "object_memory_coverage": len(memory_valid) / len(pregrasp),
        "paired_navigation_availability_gain_count": len(memory_valid) - len(current),
        "paired_navigation_availability_gain": (
            (len(memory_valid) - len(current)) / len(pregrasp)
        ),
        "held_memory_count": len(held),
        "gt_unobservable_count": len(gt_unobservable),
        "memory_valid_while_gt_unobservable_count": len(memory_gt_unobservable),
        "memory_unavailable_while_gt_unobservable_count": (
            len(gt_unobservable) - len(memory_gt_unobservable)
        ),
        "cold_start_gt_unobservable_count": len(cold_start_unobservable),
        "cold_start_gt_unobservable_episode_count": len(
            {row.episode_id for row in cold_start_unobservable}
        ),
        "gt_unobservable_after_memory_initialization_count": len(
            unobservable_after_initialization
        ),
        "memory_gt_unobservable_coverage": (
            len(memory_gt_unobservable) / len(gt_unobservable)
            if gt_unobservable
            else 0.0
        ),
        "current_unavailable_count": len(direct_unavailable),
        "memory_valid_while_current_unavailable_count": len(memory_direct_unavailable),
        "memory_current_unavailable_coverage": (
            len(memory_direct_unavailable) / len(direct_unavailable)
            if direct_unavailable
            else 0.0
        ),
        "accepted_update_unsafe_count": len(unsafe_updates),
        "current_measurement_catastrophic_count": int(
            np.sum(current_errors > catastrophic_world_xyz_error_m)
        ),
        "memory_catastrophic_count": int(
            np.sum(memory_errors > catastrophic_world_xyz_error_m)
        ),
        "post_pregrasp_memory_valid_count": sum(row.memory_valid for row in after_pregrasp),
        "memory_only_contact_authorization_count": sum(
            row.memory_only and row.contact_authorized for row in records
        ),
        "post_pregrasp_contact_authorization_count": sum(
            row.contact_authorized for row in after_pregrasp
        ),
        "episode_reset_leakage_count": reset_leakage_count,
        "measurement_rejection_reason_counts": dict(sorted(rejection_counts.items())),
        "current_measurement_error": distribution(current_errors),
        "memory_error": distribution(memory_errors),
        "memory_error_while_gt_unobservable": distribution(
            memory_gt_unobservable_errors
        ),
        "actuation_allowed": False,
        "camera_motion_allowed": False,
    }


__all__ = [
    "E018_OBJECT_MEMORY_REPLAY_VERSION",
    "E018_OBJECT_WRITE_CALIBRATION_VERSION",
    "OBJECT_WRITE_THRESHOLD_POLICY",
    "ObjectMemoryReplayRecord",
    "ObjectReplayFrame",
    "ObjectWriteCalibration",
    "calibrate_object_write_threshold",
    "replay_object_memory",
    "summarize_object_memory_replay",
]
