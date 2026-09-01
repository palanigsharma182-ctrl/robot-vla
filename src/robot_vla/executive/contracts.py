"""v1.0 分层执行器的稳定、可回放契约。

本模块只描述语义计划、可部署证据、阶段状态和审计记录。它不读取仿真器对象状态，
也不发送机器人命令，因此可以在 E013 正式门禁完成前独立进行离线回放。
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import Any

from robot_vla.contracts import OBSERVATION_HISTORY_LENGTH, OBSERVATION_V2_VERSION

SEMANTIC_PLAN_SCHEMA_VERSION = "qwen-vla-semantic-plan/v1"
EXECUTIVE_PLAN_VERSION = "qwen-vla-executive-plan/v1"
TRANSITION_LEDGER_VERSION = "qwen-vla-transition-ledger/v1"

EXECUTIVE_MODALITIES = (
    "rgb_external",
    "rgb_wrist",
    "proprio",
    "tcp_pose",
    "wrist_camera_pose",
    "finger_force",
    "controller_state",
)

EXECUTIVE_PREDICATES = (
    "object_track_valid",
    "goal_track_valid",
    "coarse_reach_complete",
    "precision_target_valid",
    "fine_alignment_complete",
    "pregrasp_pose_valid",
    "pregrasp_stable",
    "close_ready",
    "close_authorized",
    "finger_contact_detected",
    "grasp_balanced",
    "grasp_candidate",
    "grasp_verified",
    "lift_authorized",
    "lift_clearance_reached",
    "goal_region_reached",
    "deposit_alignment_complete",
    "held_pose_stable",
    "support_contact_detected",
    "support_verified",
    "release_authorized",
    "object_released",
    "tcp_retracted",
    "placement_verified",
    "hold_confirmed",
    "modalities_recovered",
    "retry_authorized",
    "replan_authorized",
    "abort_required",
)


class SubtaskId(str, Enum):
    APPROACH_AND_ALIGN = "approach_and_align"
    ACQUIRE_AND_VERIFY = "acquire_and_verify"
    TRANSFER_HELD_OBJECT = "transfer_held_object"
    DEPOSIT_AND_VERIFY = "deposit_and_verify"
    RECOVER_OR_HOLD = "recover_or_hold"


class PhaseId(str, Enum):
    ACQUIRE_TRACK = "acquire_track"
    COARSE_APPROACH = "coarse_approach"
    FINE_ALIGN = "fine_align"
    STABILIZE_PREGRASP = "stabilize_pregrasp"

    FINAL_APPROACH = "final_approach"
    CLOSE_UNTIL_CONTACT = "close_until_contact"
    SEAT_AND_BALANCE = "seat_and_balance"
    VERIFY_GRASP = "verify_grasp"

    LIFT_CLEARANCE = "lift_clearance"
    MOVE_TO_GOAL = "move_to_goal"
    ALIGN_FOR_DEPOSIT = "align_for_deposit"
    STABILIZE_HELD = "stabilize_held"

    LOWER_TO_SUPPORT = "lower_to_support"
    CONFIRM_SUPPORT = "confirm_support"
    RELEASE = "release"
    RETRACT = "retract"
    VERIFY_SETTLED = "verify_settled"

    SAFE_HOLD = "safe_hold"
    REOBSERVE = "reobserve"
    DIAGNOSE = "diagnose"


class ControllerOwner(str, Enum):
    ACTION_CHUNK = "action_chunk"
    PRECISION = "precision"
    FORCE_GUARD = "force_guard"
    SAFE_HOLD = "safe_hold"


class CriticalAction(str, Enum):
    NONE = "none"
    CLOSE_GRIPPER = "close_gripper"
    LIFT = "lift"
    RELEASE_GRIPPER = "release_gripper"


class ExecutiveStatus(str, Enum):
    RUNNING = "running"
    HOLDING = "holding"
    COMPLETED = "completed"
    ABORTED = "aborted"


class PredicateSource(str, Enum):
    DEPLOYABLE_ESTIMATOR = "deployable_estimator"
    SAFETY_MONITOR = "safety_monitor"
    OUTCOME_MONITOR = "outcome_monitor"
    EVALUATOR_GT = "evaluator_gt"


class TransitionOutcome(str, Enum):
    UNCHANGED = "unchanged"
    REJECTED = "rejected"
    COMMITTED = "committed"


class TransitionReason(str, Enum):
    EXIT_NOT_SATISFIED = "exit_not_satisfied"
    ENTRY_NOT_SATISFIED = "entry_not_satisfied"
    STABILITY_PENDING = "stability_pending"
    PHASE_COMPLETED = "phase_completed"
    TASK_COMPLETED = "task_completed"
    MISSING_MODALITY = "missing_modality"
    CRITICAL_ACTION_NOT_AUTHORIZED = "critical_action_not_authorized"
    STATE_INVARIANT_FAILED = "state_invariant_failed"
    UNSAFE_OR_ANOMALOUS = "unsafe_or_anomalous"
    HIDDEN_GT_REJECTED = "hidden_gt_rejected"
    MISSING_STATE_ESTIMATE = "missing_state_estimate"
    INCONSISTENT_STATE_EVIDENCE = "inconsistent_state_evidence"
    PHASE_TIMEOUT = "phase_timeout"
    RECOVERY_HOLD_PENDING = "recovery_hold_pending"
    RECOVERY_REOBSERVE_PENDING = "recovery_reobserve_pending"
    RECOVERY_DIAGNOSE_PENDING = "recovery_diagnose_pending"
    RECOVERY_PHASE_COMPLETED = "recovery_phase_completed"
    RECOVERY_RETRY = "recovery_retry"
    RECOVERY_REPLAN = "recovery_replan"
    RETRY_BUDGET_EXHAUSTED = "retry_budget_exhausted"
    ABORT_REQUESTED = "abort_requested"
    TERMINAL_STATE = "terminal_state"


def _nonempty_text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} 必须是非空字符串")


def _unique_nonempty(values: tuple[str, ...], name: str) -> None:
    if any(not isinstance(value, str) or not value.strip() for value in values):
        raise ValueError(f"{name} 必须全部是非空字符串")
    if len(set(values)) != len(values):
        raise ValueError(f"{name} 不能重复")


@dataclass(frozen=True)
class SemanticPlanProposal:
    """Qwen 只能提出这个受限 schema；它不能直接提交 Runtime transition。"""

    proposal_id: str
    task_id: str
    object_ref: str
    goal_ref: str
    requested_subtasks: tuple[SubtaskId, ...]
    schema_version: str = SEMANTIC_PLAN_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != SEMANTIC_PLAN_SCHEMA_VERSION:
            raise ValueError(
                f"semantic plan schema 必须为 {SEMANTIC_PLAN_SCHEMA_VERSION}"
            )
        for name in ("proposal_id", "task_id", "object_ref", "goal_ref"):
            _nonempty_text(getattr(self, name), name)
        if not self.requested_subtasks:
            raise ValueError("requested_subtasks 不能为空")
        if len(set(self.requested_subtasks)) != len(self.requested_subtasks):
            raise ValueError("requested_subtasks 不能重复")
        if any(not isinstance(item, SubtaskId) for item in self.requested_subtasks):
            raise TypeError("requested_subtasks 必须是 SubtaskId")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "proposal_id": self.proposal_id,
            "task_id": self.task_id,
            "object_ref": self.object_ref,
            "goal_ref": self.goal_ref,
            "requested_subtasks": [item.value for item in self.requested_subtasks],
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SemanticPlanProposal:
        expected = {
            "schema_version",
            "proposal_id",
            "task_id",
            "object_ref",
            "goal_ref",
            "requested_subtasks",
        }
        if set(value) != expected:
            missing = sorted(expected - set(value))
            extra = sorted(set(value) - expected)
            raise ValueError(f"semantic plan 字段不匹配；missing={missing}, extra={extra}")
        requested = value["requested_subtasks"]
        if not isinstance(requested, list):
            raise TypeError("requested_subtasks 必须是 JSON list")
        try:
            subtasks = tuple(SubtaskId(str(item)) for item in requested)
        except ValueError as error:
            raise ValueError("semantic plan 包含未知 subtask") from error
        return cls(
            schema_version=str(value["schema_version"]),
            proposal_id=str(value["proposal_id"]),
            task_id=str(value["task_id"]),
            object_ref=str(value["object_ref"]),
            goal_ref=str(value["goal_ref"]),
            requested_subtasks=subtasks,
        )


@dataclass(frozen=True)
class ModalityStatus:
    name: str
    valid: bool
    age_s: float | None

    def __post_init__(self) -> None:
        if self.name not in EXECUTIVE_MODALITIES:
            raise ValueError(f"未知 Executive modality: {self.name}")
        if not isinstance(self.valid, bool):
            raise TypeError("modality valid 必须为 bool")
        if self.valid and self.age_s is None:
            raise ValueError("有效 modality 必须提供 age_s")
        if self.age_s is not None and (
            not math.isfinite(self.age_s) or self.age_s < 0.0
        ):
            raise ValueError("modality age_s 必须是有限非负数或 None")

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "valid": self.valid, "age_s": self.age_s}

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ModalityStatus:
        if set(value) != {"name", "valid", "age_s"}:
            raise ValueError("modality status 字段不匹配")
        return cls(
            name=str(value["name"]),
            valid=value["valid"],
            age_s=None if value["age_s"] is None else float(value["age_s"]),
        )


@dataclass(frozen=True)
class PredicateEvidence:
    """只保存可审计结论与 provenance，不在 Executive 内部隐藏阈值。"""

    name: str
    satisfied: bool
    confidence: float
    source: PredicateSource

    def __post_init__(self) -> None:
        if self.name not in EXECUTIVE_PREDICATES:
            raise ValueError(f"未知 Executive predicate: {self.name}")
        if not isinstance(self.satisfied, bool):
            raise TypeError("predicate satisfied 必须为 bool")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("predicate confidence 必须位于 [0,1]")
        if not isinstance(self.source, PredicateSource):
            raise TypeError("predicate source 必须是 PredicateSource")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "satisfied": self.satisfied,
            "confidence": self.confidence,
            "source": self.source.value,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PredicateEvidence:
        if set(value) != {"name", "satisfied", "confidence", "source"}:
            raise ValueError("predicate evidence 字段不匹配")
        return cls(
            name=str(value["name"]),
            satisfied=value["satisfied"],
            confidence=float(value["confidence"]),
            source=PredicateSource(str(value["source"])),
        )


def _finite_xyz_or_none(
    value: tuple[float, float, float] | None,
    name: str,
) -> None:
    if value is None:
        return
    if len(value) != 3 or any(not math.isfinite(item) for item in value):
        raise ValueError(f"{name} 必须是有限 XYZ 或 None")


@dataclass(frozen=True)
class SpatialTrackEstimate:
    """由四帧检测、相机位姿和时间戳估计的 base-frame track。"""

    position_base_m: tuple[float, float, float] | None
    velocity_base_m_s: tuple[float, float, float] | None
    confidence: float
    valid: bool
    age_s: float | None
    source: PredicateSource = PredicateSource.DEPLOYABLE_ESTIMATOR

    def __post_init__(self) -> None:
        _finite_xyz_or_none(self.position_base_m, "position_base_m")
        _finite_xyz_or_none(self.velocity_base_m_s, "velocity_base_m_s")
        if not isinstance(self.valid, bool):
            raise TypeError("track valid 必须为 bool")
        if self.valid and (
            self.position_base_m is None
            or self.velocity_base_m_s is None
            or self.age_s is None
        ):
            raise ValueError("有效 track 必须包含 position、velocity 和 age_s")
        if self.age_s is not None and (
            not math.isfinite(self.age_s) or self.age_s < 0.0
        ):
            raise ValueError("track age_s 必须是有限非负数或 None")
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("track confidence 必须位于 [0,1]")
        if not isinstance(self.source, PredicateSource):
            raise TypeError("track source 必须是 PredicateSource")

    def to_dict(self) -> dict[str, Any]:
        return {
            "position_base_m": (
                None if self.position_base_m is None else list(self.position_base_m)
            ),
            "velocity_base_m_s": (
                None
                if self.velocity_base_m_s is None
                else list(self.velocity_base_m_s)
            ),
            "confidence": self.confidence,
            "valid": self.valid,
            "age_s": self.age_s,
            "source": self.source.value,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> SpatialTrackEstimate:
        expected = {
            "position_base_m",
            "velocity_base_m_s",
            "confidence",
            "valid",
            "age_s",
            "source",
        }
        if set(value) != expected:
            raise ValueError("spatial track 字段不匹配")

        def xyz(name: str) -> tuple[float, float, float] | None:
            raw = value[name]
            if raw is None:
                return None
            if not isinstance(raw, list) or len(raw) != 3:
                raise ValueError(f"{name} 必须是三元素 JSON list 或 null")
            return (float(raw[0]), float(raw[1]), float(raw[2]))

        return cls(
            position_base_m=xyz("position_base_m"),
            velocity_base_m_s=xyz("velocity_base_m_s"),
            confidence=float(value["confidence"]),
            valid=value["valid"],
            age_s=None if value["age_s"] is None else float(value["age_s"]),
            source=PredicateSource(str(value["source"])),
        )


@dataclass(frozen=True)
class ScalarStateEstimate:
    """抓取、支撑或稳定等可部署置信度，不等同于仿真 bool truth。"""

    confidence: float
    valid: bool
    age_s: float | None
    source: PredicateSource = PredicateSource.DEPLOYABLE_ESTIMATOR

    def __post_init__(self) -> None:
        if not math.isfinite(self.confidence) or not 0.0 <= self.confidence <= 1.0:
            raise ValueError("state confidence 必须位于 [0,1]")
        if not isinstance(self.valid, bool):
            raise TypeError("state valid 必须为 bool")
        if self.valid and self.age_s is None:
            raise ValueError("有效 state estimate 必须提供 age_s")
        if self.age_s is not None and (
            not math.isfinite(self.age_s) or self.age_s < 0.0
        ):
            raise ValueError("state age_s 必须是有限非负数或 None")
        if not isinstance(self.source, PredicateSource):
            raise TypeError("state source 必须是 PredicateSource")

    def to_dict(self) -> dict[str, Any]:
        return {
            "confidence": self.confidence,
            "valid": self.valid,
            "age_s": self.age_s,
            "source": self.source.value,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ScalarStateEstimate:
        if set(value) != {"confidence", "valid", "age_s", "source"}:
            raise ValueError("scalar state estimate 字段不匹配")
        return cls(
            confidence=float(value["confidence"]),
            valid=value["valid"],
            age_s=None if value["age_s"] is None else float(value["age_s"]),
            source=PredicateSource(str(value["source"])),
        )


@dataclass(frozen=True)
class DeployableStateEstimate:
    """Executive 所需的最小派生状态；原始 Observation V2 仍由上游保存。"""

    object_track: SpatialTrackEstimate
    goal_track: SpatialTrackEstimate
    grasp: ScalarStateEstimate
    support_contact: ScalarStateEstimate
    settled: ScalarStateEstimate
    finger_force_n: tuple[float, float]
    timestamp_s: float
    observation_version: str = OBSERVATION_V2_VERSION
    history_length: int = OBSERVATION_HISTORY_LENGTH

    def __post_init__(self) -> None:
        if self.observation_version != OBSERVATION_V2_VERSION:
            raise ValueError(f"state estimate observation 必须为 {OBSERVATION_V2_VERSION}")
        if self.history_length != OBSERVATION_HISTORY_LENGTH:
            raise ValueError(
                f"state estimate history_length 必须为 {OBSERVATION_HISTORY_LENGTH}"
            )
        if len(self.finger_force_n) != 2 or any(
            not math.isfinite(value) or value < 0.0 for value in self.finger_force_n
        ):
            raise ValueError("finger_force_n 必须是两个有限非负数 F_L/F_R")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("state estimate timestamp_s 必须是有限非负数")

    @property
    def uses_evaluator_gt(self) -> bool:
        return any(
            item.source == PredicateSource.EVALUATOR_GT
            for item in (
                self.object_track,
                self.goal_track,
                self.grasp,
                self.support_contact,
                self.settled,
            )
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "object_track": self.object_track.to_dict(),
            "goal_track": self.goal_track.to_dict(),
            "grasp": self.grasp.to_dict(),
            "support_contact": self.support_contact.to_dict(),
            "settled": self.settled.to_dict(),
            "finger_force_n": list(self.finger_force_n),
            "timestamp_s": self.timestamp_s,
            "observation_version": self.observation_version,
            "history_length": self.history_length,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> DeployableStateEstimate:
        expected = {
            "object_track",
            "goal_track",
            "grasp",
            "support_contact",
            "settled",
            "finger_force_n",
            "timestamp_s",
            "observation_version",
            "history_length",
        }
        if set(value) != expected:
            raise ValueError("deployable state estimate 字段不匹配")
        force = value["finger_force_n"]
        if not isinstance(force, list) or len(force) != 2:
            raise ValueError("finger_force_n 必须是二元素 JSON list")
        return cls(
            object_track=SpatialTrackEstimate.from_dict(value["object_track"]),
            goal_track=SpatialTrackEstimate.from_dict(value["goal_track"]),
            grasp=ScalarStateEstimate.from_dict(value["grasp"]),
            support_contact=ScalarStateEstimate.from_dict(value["support_contact"]),
            settled=ScalarStateEstimate.from_dict(value["settled"]),
            finger_force_n=(float(force[0]), float(force[1])),
            timestamp_s=float(value["timestamp_s"]),
            observation_version=str(value["observation_version"]),
            history_length=int(value["history_length"]),
        )


@dataclass(frozen=True)
class ExecutiveSnapshot:
    """一个 Executive Tick 的可部署证据；不包含 object pose 或 is_grasped GT。"""

    tick: int
    timestamp_s: float
    modalities: tuple[ModalityStatus, ...]
    predicates: tuple[PredicateEvidence, ...]
    state_estimate: DeployableStateEstimate | None = None
    unsafe_or_anomalous: bool = False
    anomaly_reason: str | None = None

    def __post_init__(self) -> None:
        if (
            not isinstance(self.tick, int)
            or isinstance(self.tick, bool)
            or self.tick < 0
        ):
            raise ValueError("tick 必须是非负整数")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s 必须是有限非负数")
        if (
            self.state_estimate is not None
            and self.state_estimate.timestamp_s > self.timestamp_s + 1e-9
        ):
            raise ValueError("state estimate 不能晚于 Executive snapshot")
        modality_names = tuple(item.name for item in self.modalities)
        predicate_names = tuple(item.name for item in self.predicates)
        _unique_nonempty(modality_names, "modalities")
        _unique_nonempty(predicate_names, "predicates")
        modality_order = {name: index for index, name in enumerate(EXECUTIVE_MODALITIES)}
        predicate_order = {name: index for index, name in enumerate(EXECUTIVE_PREDICATES)}
        object.__setattr__(
            self,
            "modalities",
            tuple(sorted(self.modalities, key=lambda item: modality_order[item.name])),
        )
        object.__setattr__(
            self,
            "predicates",
            tuple(sorted(self.predicates, key=lambda item: predicate_order[item.name])),
        )
        if not isinstance(self.unsafe_or_anomalous, bool):
            raise TypeError("unsafe_or_anomalous 必须为 bool")
        if self.unsafe_or_anomalous:
            _nonempty_text(self.anomaly_reason or "", "anomaly_reason")
        elif self.anomaly_reason is not None:
            raise ValueError("没有 anomaly 时 anomaly_reason 必须为 None")

    @property
    def valid_modalities(self) -> frozenset[str]:
        return frozenset(item.name for item in self.modalities if item.valid)

    @property
    def uses_evaluator_gt(self) -> bool:
        predicate_gt = any(
            item.source == PredicateSource.EVALUATOR_GT for item in self.predicates
        )
        state_gt = (
            self.state_estimate is not None and self.state_estimate.uses_evaluator_gt
        )
        return predicate_gt or state_gt

    def predicate(self, name: str) -> PredicateEvidence | None:
        return next((item for item in self.predicates if item.name == name), None)

    def predicates_satisfied(self, names: tuple[str, ...]) -> bool:
        return all(
            (evidence := self.predicate(name)) is not None and evidence.satisfied
            for name in names
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "tick": self.tick,
            "timestamp_s": self.timestamp_s,
            "modalities": [item.to_dict() for item in self.modalities],
            "predicates": [item.to_dict() for item in self.predicates],
            "state_estimate": (
                None if self.state_estimate is None else self.state_estimate.to_dict()
            ),
            "unsafe_or_anomalous": self.unsafe_or_anomalous,
            "anomaly_reason": self.anomaly_reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> ExecutiveSnapshot:
        expected = {
            "tick",
            "timestamp_s",
            "modalities",
            "predicates",
            "state_estimate",
            "unsafe_or_anomalous",
            "anomaly_reason",
        }
        if set(value) != expected:
            raise ValueError("Executive snapshot 字段不匹配")
        modalities = value["modalities"]
        predicates = value["predicates"]
        if not isinstance(modalities, list) or not isinstance(predicates, list):
            raise TypeError("snapshot modalities/predicates 必须是 JSON list")
        state = value["state_estimate"]
        return cls(
            tick=int(value["tick"]),
            timestamp_s=float(value["timestamp_s"]),
            modalities=tuple(ModalityStatus.from_dict(item) for item in modalities),
            predicates=tuple(PredicateEvidence.from_dict(item) for item in predicates),
            state_estimate=(
                None if state is None else DeployableStateEstimate.from_dict(state)
            ),
            unsafe_or_anomalous=value["unsafe_or_anomalous"],
            anomaly_reason=value["anomaly_reason"],
        )


@dataclass(frozen=True)
class PhaseSpec:
    phase: PhaseId
    subtask: SubtaskId
    controller_owner: ControllerOwner
    required_modalities: tuple[str, ...]
    entry_predicates: tuple[str, ...]
    exit_predicates: tuple[str, ...]
    invariant_predicates: tuple[str, ...]
    stable_ticks_required: int
    timeout_ticks: int | None = None
    critical_action: CriticalAction = CriticalAction.NONE

    def __post_init__(self) -> None:
        if not isinstance(self.phase, PhaseId) or not isinstance(self.subtask, SubtaskId):
            raise TypeError("phase/subtask 类型无效")
        if not isinstance(self.controller_owner, ControllerOwner):
            raise TypeError("controller_owner 类型无效")
        _unique_nonempty(self.required_modalities, "required_modalities")
        unknown_modalities = set(self.required_modalities) - set(EXECUTIVE_MODALITIES)
        if unknown_modalities:
            raise ValueError(f"PhaseSpec 包含未知 modality: {sorted(unknown_modalities)}")
        for values, name in (
            (self.entry_predicates, "entry_predicates"),
            (self.exit_predicates, "exit_predicates"),
            (self.invariant_predicates, "invariant_predicates"),
        ):
            _unique_nonempty(values, name)
            unknown = set(values) - set(EXECUTIVE_PREDICATES)
            if unknown:
                raise ValueError(f"PhaseSpec 包含未知 predicate: {sorted(unknown)}")
        if (
            not isinstance(self.stable_ticks_required, int)
            or isinstance(self.stable_ticks_required, bool)
            or self.stable_ticks_required <= 0
        ):
            raise ValueError("stable_ticks_required 必须为正整数")
        if self.timeout_ticks is not None and (
            not isinstance(self.timeout_ticks, int)
            or isinstance(self.timeout_ticks, bool)
            or self.timeout_ticks <= 0
        ):
            raise ValueError("timeout_ticks 必须为正整数或 None")
        if not isinstance(self.critical_action, CriticalAction):
            raise TypeError("critical_action 类型无效")

    def to_dict(self) -> dict[str, Any]:
        return {
            "phase": self.phase.value,
            "subtask": self.subtask.value,
            "controller_owner": self.controller_owner.value,
            "required_modalities": list(self.required_modalities),
            "entry_predicates": list(self.entry_predicates),
            "exit_predicates": list(self.exit_predicates),
            "invariant_predicates": list(self.invariant_predicates),
            "stable_ticks_required": self.stable_ticks_required,
            "timeout_ticks": self.timeout_ticks,
            "critical_action": self.critical_action.value,
        }


@dataclass(frozen=True)
class SubtaskSpec:
    subtask: SubtaskId
    phases: tuple[PhaseId, ...]
    allowed_next: tuple[SubtaskId, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.subtask, SubtaskId):
            raise TypeError("subtask 类型无效")
        if not self.phases or len(set(self.phases)) != len(self.phases):
            raise ValueError("SubtaskSpec phases 必须非空且不重复")
        if any(not isinstance(item, PhaseId) for item in self.phases):
            raise TypeError("SubtaskSpec phases 必须是 PhaseId")
        if len(set(self.allowed_next)) != len(self.allowed_next):
            raise ValueError("allowed_next 不能重复")
        if any(not isinstance(item, SubtaskId) for item in self.allowed_next):
            raise TypeError("allowed_next 必须是 SubtaskId")

    def to_dict(self) -> dict[str, Any]:
        return {
            "subtask": self.subtask.value,
            "phases": [item.value for item in self.phases],
            "allowed_next": [item.value for item in self.allowed_next],
        }


@dataclass(frozen=True)
class CompiledTaskPlan:
    plan_id: str
    proposal: SemanticPlanProposal
    subtasks: tuple[SubtaskSpec, ...]
    phases: tuple[PhaseSpec, ...]
    version: str = EXECUTIVE_PLAN_VERSION

    def __post_init__(self) -> None:
        if self.version != EXECUTIVE_PLAN_VERSION:
            raise ValueError(f"Executive plan version 必须为 {EXECUTIVE_PLAN_VERSION}")
        _nonempty_text(self.plan_id, "plan_id")
        if len(self.plan_id) != 64 or any(char not in "0123456789abcdef" for char in self.plan_id):
            raise ValueError("plan_id 必须是 64 位小写 SHA-256")
        subtask_ids = tuple(item.subtask for item in self.subtasks)
        phase_ids = tuple(item.phase for item in self.phases)
        if len(set(subtask_ids)) != len(subtask_ids):
            raise ValueError("CompiledTaskPlan subtask 重复")
        if len(set(phase_ids)) != len(phase_ids):
            raise ValueError("CompiledTaskPlan phase 重复")
        phase_lookup = {item.phase: item for item in self.phases}
        referenced_phases: set[PhaseId] = set()
        for subtask in self.subtasks:
            for phase in subtask.phases:
                if phase not in phase_lookup or phase_lookup[phase].subtask != subtask.subtask:
                    raise ValueError("SubtaskSpec 与 PhaseSpec 不一致")
                referenced_phases.add(phase)
        if referenced_phases != set(phase_ids):
            raise ValueError("CompiledTaskPlan 包含未归属 Subtask 的 phase")

    @property
    def normal_phases(self) -> tuple[PhaseId, ...]:
        return tuple(
            phase
            for subtask in self.subtasks
            if subtask.subtask != SubtaskId.RECOVER_OR_HOLD
            for phase in subtask.phases
        )

    def phase_spec(self, phase: PhaseId) -> PhaseSpec:
        try:
            return next(item for item in self.phases if item.phase == phase)
        except StopIteration as error:
            raise ValueError(f"plan 不包含 phase: {phase.value}") from error

    def next_normal_phase(self, phase: PhaseId) -> PhaseId | None:
        phases = self.normal_phases
        try:
            index = phases.index(phase)
        except ValueError as error:
            raise ValueError(f"phase 不在 normal graph: {phase.value}") from error
        return None if index + 1 == len(phases) else phases[index + 1]

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "plan_id": self.plan_id,
            "proposal": self.proposal.to_dict(),
            "subtasks": [item.to_dict() for item in self.subtasks],
            "phases": [item.to_dict() for item in self.phases],
        }


@dataclass(frozen=True)
class ExecutiveState:
    status: ExecutiveStatus
    subtask: SubtaskId
    phase: PhaseId
    controller_owner: ControllerOwner
    phase_age_ticks: int
    recovery_attempts: int

    def __post_init__(self) -> None:
        if not isinstance(self.status, ExecutiveStatus):
            raise TypeError("Executive status 类型无效")
        if not isinstance(self.subtask, SubtaskId) or not isinstance(self.phase, PhaseId):
            raise TypeError("Executive subtask/phase 类型无效")
        if not isinstance(self.controller_owner, ControllerOwner):
            raise TypeError("Executive controller_owner 类型无效")
        for value, name in (
            (self.phase_age_ticks, "phase_age_ticks"),
            (self.recovery_attempts, "recovery_attempts"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"{name} 必须为非负整数")

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status.value,
            "subtask": self.subtask.value,
            "phase": self.phase.value,
            "controller_owner": self.controller_owner.value,
            "phase_age_ticks": self.phase_age_ticks,
            "recovery_attempts": self.recovery_attempts,
        }


@dataclass(frozen=True)
class ExecutiveDecision:
    state: ExecutiveState
    outcome: TransitionOutcome
    reason: TransitionReason
    proposed_subtask: SubtaskId | None
    proposed_phase: PhaseId | None
    proposed_critical_action: CriticalAction
    shadow_only: bool
    actuation_allowed: bool
    requires_action_reset: bool
    missing_modalities: tuple[str, ...] = ()
    missing_predicates: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "state": self.state.to_dict(),
            "outcome": self.outcome.value,
            "reason": self.reason.value,
            "proposed_subtask": (
                None if self.proposed_subtask is None else self.proposed_subtask.value
            ),
            "proposed_phase": (
                None if self.proposed_phase is None else self.proposed_phase.value
            ),
            "proposed_critical_action": self.proposed_critical_action.value,
            "shadow_only": self.shadow_only,
            "actuation_allowed": self.actuation_allowed,
            "requires_action_reset": self.requires_action_reset,
            "missing_modalities": list(self.missing_modalities),
            "missing_predicates": list(self.missing_predicates),
        }


@dataclass(frozen=True)
class TransitionLedgerEntry:
    index: int
    plan_id: str
    snapshot: ExecutiveSnapshot
    before: ExecutiveState
    decision: ExecutiveDecision
    version: str = TRANSITION_LEDGER_VERSION

    def __post_init__(self) -> None:
        if self.version != TRANSITION_LEDGER_VERSION:
            raise ValueError(f"ledger version 必须为 {TRANSITION_LEDGER_VERSION}")
        if self.index < 0:
            raise ValueError("ledger index 不能为负数")
        if len(self.plan_id) != 64:
            raise ValueError("ledger plan_id 无效")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "index": self.index,
            "plan_id": self.plan_id,
            "snapshot": self.snapshot.to_dict(),
            "before": self.before.to_dict(),
            "decision": self.decision.to_dict(),
        }

    def canonical_json(self) -> str:
        return json.dumps(
            self.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )


__all__ = [
    "EXECUTIVE_MODALITIES",
    "EXECUTIVE_PLAN_VERSION",
    "EXECUTIVE_PREDICATES",
    "SEMANTIC_PLAN_SCHEMA_VERSION",
    "TRANSITION_LEDGER_VERSION",
    "CompiledTaskPlan",
    "ControllerOwner",
    "CriticalAction",
    "DeployableStateEstimate",
    "ExecutiveDecision",
    "ExecutiveSnapshot",
    "ExecutiveState",
    "ExecutiveStatus",
    "ModalityStatus",
    "PhaseId",
    "PhaseSpec",
    "PredicateEvidence",
    "PredicateSource",
    "ScalarStateEstimate",
    "SemanticPlanProposal",
    "SubtaskId",
    "SubtaskSpec",
    "SpatialTrackEstimate",
    "TransitionLedgerEntry",
    "TransitionOutcome",
    "TransitionReason",
]
