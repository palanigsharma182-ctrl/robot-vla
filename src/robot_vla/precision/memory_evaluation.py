"""E015 write-gate calibration 与 deterministic goal-memory replay。"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass

import numpy as np

from robot_vla.precision.observability import GOAL_WRITE_SCORE_SEMANTICS
from robot_vla.precision.state_memory import (
    ExplicitGoalStateMemory,
    GoalMeasurement,
    GoalMemoryConfig,
)

E015_WRITE_CALIBRATION_VERSION = "e015-goal-write-calibration/v1"
E015_MEMORY_REPLAY_VERSION = "e015-goal-memory-replay/v1"
WRITE_THRESHOLD_POLICY = "maximum-coverage-zero-unsafe-on-fresh-validation/v1"
MEMORY_AGE_POLICY = (
    "maximum-occluded-coverage-zero-catastrophic-on-fresh-validation/v1"
)


def _probability(value: float, name: str) -> float:
    candidate = float(value)
    if not math.isfinite(candidate) or not 0.0 <= candidate <= 1.0:
        raise ValueError(f"{name} 必须是 [0,1] 内有限数值")
    return candidate


def _xyz(value: Sequence[float] | np.ndarray, name: str) -> tuple[float, float, float]:
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError(f"{name} 必须是有限 XYZ")
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
class GoalWriteCalibration:
    enabled: bool
    threshold: float
    validation_frame_count: int
    structurally_eligible_count: int
    oracle_safe_count: int
    accepted_count: int
    accepted_unsafe_count: int
    safe_coverage: float
    threshold_policy: str = WRITE_THRESHOLD_POLICY
    score_semantics: str = GOAL_WRITE_SCORE_SEMANTICS
    version: str = E015_WRITE_CALIBRATION_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.enabled, bool):
            raise TypeError("write calibration enabled 必须为 bool")
        _probability(self.threshold, "write threshold")
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
            raise ValueError("frozen write calibration 不允许 validation unsafe acceptance")
        if self.threshold_policy != WRITE_THRESHOLD_POLICY:
            raise ValueError("write threshold policy 漂移")
        if self.score_semantics != GOAL_WRITE_SCORE_SEMANTICS:
            raise ValueError("write score semantics 漂移")
        if self.version != E015_WRITE_CALIBRATION_VERSION:
            raise ValueError("write calibration version 漂移")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def calibrate_goal_write_threshold(
    *,
    scores: Sequence[float],
    structurally_eligible: Sequence[bool],
    oracle_safe: Sequence[bool],
) -> GoalWriteCalibration:
    """只在 validation 上选择零 unsafe acceptance 下覆盖率最大的阈值。"""

    score = np.asarray(scores, dtype=np.float64)
    eligible = np.asarray(structurally_eligible)
    safe = np.asarray(oracle_safe)
    if score.ndim != 1 or score.size == 0 or not np.isfinite(score).all():
        raise ValueError("write calibration scores 必须是非空有限一维")
    if np.any(score < 0.0) or np.any(score > 1.0):
        raise ValueError("write calibration scores 必须位于 [0,1]")
    if eligible.shape != score.shape or eligible.dtype != np.bool_:
        raise ValueError("structurally_eligible 必须是 bool [N]")
    if safe.shape != score.shape or safe.dtype != np.bool_:
        raise ValueError("oracle_safe 必须是 bool [N]")

    candidates = sorted({float(value) for value in score[eligible]}, reverse=True)
    best: tuple[int, float, np.ndarray] | None = None
    for threshold in candidates:
        accepted = eligible & (score >= threshold)
        if np.any(accepted & ~safe):
            continue
        accepted_count = int(accepted.sum())
        if accepted_count <= 0:
            continue
        candidate = (accepted_count, threshold, accepted)
        if best is None or candidate[:2] > best[:2]:
            best = candidate

    safe_count = int(safe.sum())
    if best is None:
        return GoalWriteCalibration(
            enabled=False,
            threshold=1.0,
            validation_frame_count=int(score.size),
            structurally_eligible_count=int(eligible.sum()),
            oracle_safe_count=safe_count,
            accepted_count=0,
            accepted_unsafe_count=0,
            safe_coverage=0.0,
        )
    accepted_count, threshold, accepted = best
    return GoalWriteCalibration(
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
class GoalReplayFrame:
    episode_id: str
    timestep: int
    timestamp_s: float
    predicted_position_base_m: tuple[float, float, float] | None
    measurement_covariance_base_m2: tuple[tuple[float, float, float], ...] | None
    write_score: float
    structurally_eligible: bool
    predicted_observable: bool
    geometry_valid: bool
    gt_position_base_m: tuple[float, float, float]
    gt_observable: bool

    def __post_init__(self) -> None:
        if not isinstance(self.episode_id, str) or not self.episode_id.strip():
            raise ValueError("replay episode_id 不能为空")
        if not isinstance(self.timestep, int) or isinstance(self.timestep, bool) or self.timestep < 0:
            raise ValueError("replay timestep 必须是非负整数")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("replay timestamp_s 必须有限非负")
        if self.predicted_position_base_m is not None:
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
        object.__setattr__(
            self,
            "gt_position_base_m",
            _xyz(self.gt_position_base_m, "gt_position_base_m"),
        )
        object.__setattr__(self, "write_score", _probability(self.write_score, "write_score"))
        for name in (
            "structurally_eligible",
            "predicted_observable",
            "geometry_valid",
            "gt_observable",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须为 bool")


@dataclass(frozen=True)
class GoalMemoryReplayRecord:
    episode_id: str
    timestep: int
    timestamp_s: float
    gt_observable: bool
    current_measurement_accepted: bool
    current_world_xy_error_m: float | None
    memory_valid: bool
    memory_observable_now: bool
    memory_world_xy_error_m: float | None
    memory_age_s: float | None
    memory_measurement_accepted: bool
    measurement_rejection_reasons: tuple[str, ...]
    innovation_m: float | None

    def to_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["measurement_rejection_reasons"] = list(
            self.measurement_rejection_reasons
        )
        return payload


def replay_goal_memory(
    frames: Sequence[GoalReplayFrame],
    *,
    calibration: GoalWriteCalibration,
    memory_config: GoalMemoryConfig,
) -> list[GoalMemoryReplayRecord]:
    """按 Episode/timestamp 顺序 replay；每个 Episode 强制 reset。"""

    if not frames:
        raise ValueError("goal memory replay frames 不能为空")
    ordered = sorted(frames, key=lambda row: (row.episode_id, row.timestamp_s, row.timestep))
    seen_timesteps: set[tuple[str, int]] = set()
    records: list[GoalMemoryReplayRecord] = []
    memory = ExplicitGoalStateMemory(memory_config)
    active_episode: str | None = None
    for frame in ordered:
        identity = (frame.episode_id, frame.timestep)
        if identity in seen_timesteps:
            raise ValueError("goal memory replay episode/timestep 重复")
        seen_timesteps.add(identity)
        if frame.episode_id != active_episode:
            reset = memory.reset(frame.episode_id, timestamp_s=frame.timestamp_s)
            if reset.position_base_m is not None or reset.valid:
                raise RuntimeError("Episode reset 后 goal memory 未清空")
            active_episode = frame.episode_id
        write_accepted = bool(
            calibration.enabled
            and frame.structurally_eligible
            and frame.write_score >= calibration.threshold
        )
        measurement = GoalMeasurement(
            timestamp_s=frame.timestamp_s,
            position_base_m=frame.predicted_position_base_m,
            covariance_base_m2=frame.measurement_covariance_base_m2,
            confidence=frame.write_score,
            goal_exists=True,
            projection_valid=frame.geometry_valid,
            in_fov=frame.geometry_valid,
            observable=frame.predicted_observable and frame.geometry_valid,
            geometry_valid=frame.geometry_valid,
            write_gate_passed=write_accepted,
        )
        update = memory.update(measurement, episode_id=frame.episode_id)
        current_error = (
            None
            if frame.predicted_position_base_m is None
            else float(
                np.linalg.norm(
                    np.asarray(frame.predicted_position_base_m[:2])
                    - np.asarray(frame.gt_position_base_m[:2])
                )
            )
        )
        memory_error = (
            None
            if update.state.position_base_m is None
            else float(
                np.linalg.norm(
                    np.asarray(update.state.position_base_m[:2])
                    - np.asarray(frame.gt_position_base_m[:2])
                )
            )
        )
        records.append(
            GoalMemoryReplayRecord(
                episode_id=frame.episode_id,
                timestep=frame.timestep,
                timestamp_s=frame.timestamp_s,
                gt_observable=frame.gt_observable,
                current_measurement_accepted=write_accepted,
                current_world_xy_error_m=current_error,
                memory_valid=update.state.valid,
                memory_observable_now=update.state.observable_now,
                memory_world_xy_error_m=memory_error,
                memory_age_s=update.state.age_s,
                memory_measurement_accepted=update.measurement_accepted,
                measurement_rejection_reasons=update.rejection_reasons,
                innovation_m=update.innovation_m,
            )
        )
    return records


def summarize_goal_memory_replay(
    records: Sequence[GoalMemoryReplayRecord],
    *,
    catastrophic_world_xy_error_m: float,
) -> dict[str, object]:
    if not records:
        raise ValueError("goal memory replay records 不能为空")
    if (
        not math.isfinite(catastrophic_world_xy_error_m)
        or catastrophic_world_xy_error_m <= 0.0
    ):
        raise ValueError("catastrophic threshold 必须是有限正数")
    frame_count = len(records)
    current_valid = [row for row in records if row.current_measurement_accepted]
    memory_valid = [row for row in records if row.memory_valid]
    gt_occluded = [row for row in records if not row.gt_observable]
    memory_valid_occluded = [row for row in gt_occluded if row.memory_valid]
    current_valid_occluded = [
        row for row in gt_occluded if row.current_measurement_accepted
    ]
    accepted_updates = [row for row in records if row.memory_measurement_accepted]
    held = [
        row
        for row in records
        if row.memory_valid and not row.memory_measurement_accepted
    ]
    conflicts = [
        row
        for row in records
        if "measurement_conflict" in row.measurement_rejection_reasons
    ]
    stale = [
        row
        for row in records
        if not row.memory_valid
        and row.memory_age_s is not None
        and "not_observable" in row.measurement_rejection_reasons
    ]

    def errors(rows: Sequence[GoalMemoryReplayRecord], *, memory: bool) -> np.ndarray:
        values = [
            row.memory_world_xy_error_m if memory else row.current_world_xy_error_m
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

    current_errors = errors(current_valid, memory=False)
    memory_errors = errors(memory_valid, memory=True)
    memory_occluded_errors = errors(memory_valid_occluded, memory=True)
    current_catastrophic = int(
        np.sum(current_errors > catastrophic_world_xy_error_m)
    )
    memory_catastrophic = int(
        np.sum(memory_errors > catastrophic_world_xy_error_m)
    )
    return {
        "version": E015_MEMORY_REPLAY_VERSION,
        "frame_count": frame_count,
        "episode_count": len({row.episode_id for row in records}),
        "current_measurement_valid_count": len(current_valid),
        "current_measurement_coverage": len(current_valid) / frame_count,
        "current_measurement_catastrophic_count": current_catastrophic,
        "memory_valid_count": len(memory_valid),
        "memory_coverage": len(memory_valid) / frame_count,
        "memory_catastrophic_count": memory_catastrophic,
        "accepted_memory_update_count": len(accepted_updates),
        "held_valid_memory_count": len(held),
        "measurement_conflict_count": len(conflicts),
        "stale_or_uninitialized_occluded_count": len(stale),
        "gt_unobservable_count": len(gt_occluded),
        "current_valid_while_gt_unobservable_count": len(current_valid_occluded),
        "memory_valid_while_gt_unobservable_count": len(memory_valid_occluded),
        "memory_unobservable_coverage": (
            len(memory_valid_occluded) / len(gt_occluded) if gt_occluded else 0.0
        ),
        "current_measurement_error": distribution(current_errors),
        "memory_error": distribution(memory_errors),
        "memory_error_while_gt_unobservable": distribution(memory_occluded_errors),
        "episode_reset_leakage_count": 0,
        "actuation_allowed": False,
    }


def select_memory_max_age(
    frames: Sequence[GoalReplayFrame],
    *,
    calibration: GoalWriteCalibration,
    max_age_candidates_s: Sequence[float],
    max_innovation_m: float,
    max_position_std_m: float,
    require_covariance: bool,
    covariance_growth_m2_per_s: float,
    catastrophic_world_xy_error_m: float,
) -> tuple[float, list[dict[str, object]]]:
    """在 validation 上选零 catastrophic 下遮挡覆盖最多的 age。"""

    if not max_age_candidates_s:
        raise ValueError("max_age_candidates_s 不能为空")
    candidates = tuple(float(value) for value in max_age_candidates_s)
    if any(not math.isfinite(value) or value <= 0.0 for value in candidates):
        raise ValueError("max_age_candidates_s 必须全部为有限正数")
    if len(set(candidates)) != len(candidates):
        raise ValueError("max_age_candidates_s 不能重复")
    reports: list[dict[str, object]] = []
    best: tuple[int, int, float] | None = None
    selected: float | None = None
    for max_age in candidates:
        config = GoalMemoryConfig(
            max_unobserved_age_s=max_age,
            max_innovation_m=max_innovation_m,
            max_position_std_m=max_position_std_m,
            require_covariance=require_covariance,
            covariance_growth_m2_per_s=covariance_growth_m2_per_s,
        )
        records = replay_goal_memory(
            frames,
            calibration=calibration,
            memory_config=config,
        )
        summary = summarize_goal_memory_replay(
            records,
            catastrophic_world_xy_error_m=catastrophic_world_xy_error_m,
        )
        reports.append(
            {
                "max_unobserved_age_s": max_age,
                "memory_valid_count": summary["memory_valid_count"],
                "memory_valid_while_gt_unobservable_count": summary[
                    "memory_valid_while_gt_unobservable_count"
                ],
                "memory_catastrophic_count": summary["memory_catastrophic_count"],
            }
        )
        if int(summary["memory_catastrophic_count"]) != 0:
            continue
        # 覆盖率优先；完全相同时选择更短的 age。
        score = (
            int(summary["memory_valid_while_gt_unobservable_count"]),
            int(summary["memory_valid_count"]),
            -max_age,
        )
        if best is None or score > best:
            best = score
            selected = max_age
    if selected is None:
        raise RuntimeError("validation 上没有零 catastrophic 的 memory age 候选")
    return selected, reports


__all__ = [
    "E015_MEMORY_REPLAY_VERSION",
    "E015_WRITE_CALIBRATION_VERSION",
    "MEMORY_AGE_POLICY",
    "WRITE_THRESHOLD_POLICY",
    "GoalMemoryReplayRecord",
    "GoalReplayFrame",
    "GoalWriteCalibration",
    "calibrate_goal_write_threshold",
    "replay_goal_memory",
    "select_memory_max_age",
    "summarize_goal_memory_replay",
]
