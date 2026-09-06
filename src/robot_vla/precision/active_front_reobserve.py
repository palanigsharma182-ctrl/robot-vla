"""E018-P1 受限 active-front-reobserve 的纯 supervisor 契约。

首版只编排 camera recovery，不修改全局 Executive、PlanCompiler、Observation V2
或 Object Memory。调用方必须显式提供 Action history reset/resume receipt；本模块
不会把接口存在误报为 Runtime 已经完成集成。
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from enum import Enum
from typing import ClassVar

from robot_vla.executive.contracts import ControllerOwner, PhaseId

ACTIVE_FRONT_REOBSERVE_VERSION = "e018-p1-active-front-reobserve-supervisor/v2"
ALLOWED_ACTIVE_SOURCE_PHASES = frozenset(
    {PhaseId.ACQUIRE_TRACK, PhaseId.STABILIZE_PREGRASP}
)
POST_ACTIVE_WINDOW_PHASES = frozenset(
    {
        PhaseId.FINAL_APPROACH,
        PhaseId.CLOSE_UNTIL_CONTACT,
        PhaseId.SEAT_AND_BALANCE,
        PhaseId.VERIFY_GRASP,
        PhaseId.LIFT_CLEARANCE,
        PhaseId.MOVE_TO_GOAL,
        PhaseId.ALIGN_FOR_DEPOSIT,
        PhaseId.STABILIZE_HELD,
        PhaseId.LOWER_TO_SUPPORT,
        PhaseId.CONFIRM_SUPPORT,
        PhaseId.RELEASE,
        PhaseId.RETRACT,
        PhaseId.VERIFY_SETTLED,
    }
)


class ExternalCameraControllerOwner(str, Enum):
    NONE = "none"
    HOME_HOLD = "home_hold"
    ACTIVE_REOBSERVE = "active_reobserve"
    FAILSAFE_RETURN = "failsafe_return"


class ActiveFrontReobserveState(str, Enum):
    IDLE = "idle"
    REQUESTED = "requested"
    ACQUIRE_CAMERA_LEASE_AND_HOLD_ARM = "acquire_camera_lease_and_hold_arm"
    SELECT_FROZEN_PRIMITIVE = "select_frozen_primitive"
    MOVE_TO_VIEW = "move_to_view"
    SETTLE_AT_VIEW = "settle_at_view"
    COLLECT = "collect"
    STAGE_CANDIDATE = "stage_candidate"
    RETURN_HOME = "return_home"
    VERIFY_HOME_AND_ARM_HOLD = "verify_home_and_arm_hold"
    RECHECK_SOURCE_INVARIANTS = "recheck_source_invariants"
    COMMIT_AND_RESUME = "commit_and_resume"
    COMPLETE_NO_WRITE = "complete_no_write"
    COMPLETE_STAGE2_MEMORY_WRITE = "complete_stage2_memory_write"
    FAILSAFE_RETURN = "failsafe_return"
    FAILED_SAFE_HOLD = "failed_safe_hold"


class ActiveFrontTriggerReason(str, Enum):
    OUT_OF_FOV = "out_of_fov"
    OBJECT_OCCLUSION = "object_occlusion"
    LOW_VISUAL_CONFIDENCE = "low_visual_confidence"
    HIGH_LOCALIZATION_UNCERTAINTY = "high_localization_uncertainty"
    HIGH_GEOMETRIC_SENSITIVITY = "high_geometric_sensitivity"
    NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT = (
        "no_qualified_wrist_provider_in_parent"
    )
    INVALID_SENSOR_OR_POSE = "invalid_sensor_or_pose"
    PROVIDER_IDENTITY_MISMATCH = "provider_identity_mismatch"
    UNSAFE_ARM_STATE = "unsafe_arm_state"
    UNSAFE_CAMERA_STATE = "unsafe_camera_state"
    UNKNOWN = "unknown"


VIEWPOINT_RESOLVABLE_REASONS = frozenset(
    {
        ActiveFrontTriggerReason.OUT_OF_FOV,
        ActiveFrontTriggerReason.OBJECT_OCCLUSION,
        ActiveFrontTriggerReason.LOW_VISUAL_CONFIDENCE,
        ActiveFrontTriggerReason.HIGH_LOCALIZATION_UNCERTAINTY,
        ActiveFrontTriggerReason.HIGH_GEOMETRIC_SENSITIVITY,
    }
)


class ActiveFrontDecisionReason(str, Enum):
    REQUESTED = "requested"
    FEATURE_DISABLED = "feature_disabled"
    DISALLOWED_SOURCE_PHASE = "disallowed_source_phase"
    ACTIVE_WINDOW_CLOSED = "active_window_closed"
    DIRECT_WRIST_EVIDENCE_AVAILABLE = "direct_wrist_evidence_available"
    HOME_FRONT_EVIDENCE_AVAILABLE = "home_front_evidence_available"
    OBJECT_MEMORY_AVAILABLE = "object_memory_available"
    FAILURE_NOT_VIEWPOINT_RESOLVABLE = "failure_not_viewpoint_resolvable"
    ARM_OR_CAMERA_PREREQUISITE_FAILED = "arm_or_camera_prerequisite_failed"
    CONSECUTIVE_EVIDENCE_PENDING = "consecutive_evidence_pending"
    ATTEMPT_BUDGET_EXHAUSTED = "attempt_budget_exhausted"
    COOLDOWN_ACTIVE = "cooldown_active"
    REQUEST_ALREADY_ACTIVE = "request_already_active"


class ActiveFrontFailure(str, Enum):
    CAMERA_LEASE_FAILED = "camera_lease_failed"
    PRIMITIVE_IDENTITY_MISMATCH = "primitive_identity_mismatch"
    STATE_TRANSITION_INVALID = "state_transition_invalid"
    ARM_HOLD_VIOLATION = "arm_hold_violation"
    TCP_HOLD_VIOLATION = "tcp_hold_violation"
    GRIPPER_HOLD_VIOLATION = "gripper_hold_violation"
    CONTACT_DETECTED = "contact_detected"
    ACTIVE_WINDOW_CLOSED = "active_window_closed"
    HOME_FRAME_INVALID = "home_frame_invalid"
    HOME_FRAME_DUPLICATE = "home_frame_duplicate"
    SOURCE_INVARIANT_FAILED = "source_invariant_failed"
    ACTION_HISTORY_RESET_INVALID = "action_history_reset_invalid"
    STALE_ACTION_HISTORY_RESUME = "stale_action_history_resume"
    CANDIDATE_REJECTED = "candidate_rejected"
    CAMERA_RETURN_FAILED = "camera_return_failed"
    TIMEOUT = "timeout"


class ActiveFrontSignal(str, Enum):
    CAMERA_LEASE_ACQUIRED = "camera_lease_acquired"
    FROZEN_PRIMITIVE_SELECTED = "frozen_primitive_selected"
    MOVE_COMPLETE = "move_complete"
    SETTLE_COMPLETE = "settle_complete"
    COLLECTION_COMPLETE = "collection_complete"
    SHADOW_CANDIDATE_STAGED = "shadow_candidate_staged"
    RETURN_HOME_COMPLETE = "return_home_complete"
    SOURCE_INVARIANTS_VERIFIED = "source_invariants_verified"


@dataclass(frozen=True)
class ActiveFrontReobserveConfig:
    enabled: bool = False
    selected_primitive_id: str = "LEFT_LOW__YAW_LEFT"
    consecutive_unusable_ticks: int = 3
    cooldown_ticks: int = 20
    maximum_attempts_per_episode: int = 1
    home_v2_barrier_frames: int = 4
    allow_capability_absent_trigger: bool = False
    version: str = ACTIVE_FRONT_REOBSERVE_VERSION

    def __post_init__(self) -> None:
        if self.version != ACTIVE_FRONT_REOBSERVE_VERSION:
            raise ValueError("active-front-reobserve config version 漂移")
        if not isinstance(self.enabled, bool):
            raise TypeError("enabled 必须是 bool")
        if not isinstance(self.allow_capability_absent_trigger, bool):
            raise TypeError("allow_capability_absent_trigger 必须是 bool")
        if not self.selected_primitive_id:
            raise ValueError("selected_primitive_id 必须是非空字符串")
        for name in (
            "consecutive_unusable_ticks",
            "cooldown_ticks",
            "maximum_attempts_per_episode",
            "home_v2_barrier_frames",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} 必须是整数")
        if self.consecutive_unusable_ticks <= 1:
            raise ValueError("首版禁止单帧失败触发")
        if self.cooldown_ticks < 0:
            raise ValueError("cooldown_ticks 不能为负数")
        if self.maximum_attempts_per_episode != 1:
            raise ValueError("首版每个 Episode 只能主动观察一次")
        if self.home_v2_barrier_frames != 4:
            raise ValueError("恢复前必须有四个全新 HOME Observation V2 frame")


@dataclass(frozen=True)
class ActiveFrontTriggerEvidence:
    episode_id: str
    episode_generation: int
    control_tick: int
    timestamp_s: float
    source_phase: PhaseId
    wrist_object_measurement_usable: bool
    front_home_object_measurement_usable: bool
    object_memory_navigation_state_available: bool
    arm_hold_prerequisites_pass: bool
    camera_home_prerequisites_pass: bool
    failure_reason: ActiveFrontTriggerReason
    object_contact: bool = False
    gripper_close_commanded: bool = False
    grasp_candidate: bool = False
    grasp_verified: bool = False
    object_motion_risk: bool = False

    def __post_init__(self) -> None:
        if not self.episode_id:
            raise ValueError("episode_id 必须是非空字符串")
        if (
            not isinstance(self.episode_generation, int)
            or isinstance(self.episode_generation, bool)
            or self.episode_generation <= 0
        ):
            raise ValueError("episode_generation 必须是正整数")
        if (
            not isinstance(self.control_tick, int)
            or isinstance(self.control_tick, bool)
            or self.control_tick < 0
        ):
            raise ValueError("control_tick 必须是非负整数")
        if not math.isfinite(self.timestamp_s) or self.timestamp_s < 0.0:
            raise ValueError("timestamp_s 必须是有限非负数")
        if not isinstance(self.source_phase, PhaseId):
            raise TypeError("source_phase 必须是 PhaseId")
        if not isinstance(self.failure_reason, ActiveFrontTriggerReason):
            raise TypeError("failure_reason 类型错误")
        for name in (
            "wrist_object_measurement_usable",
            "front_home_object_measurement_usable",
            "object_memory_navigation_state_available",
            "arm_hold_prerequisites_pass",
            "camera_home_prerequisites_pass",
            "object_contact",
            "gripper_close_commanded",
            "grasp_candidate",
            "grasp_verified",
            "object_motion_risk",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"{name} 必须是 bool")


@dataclass(frozen=True)
class ActiveFrontReobserveRequest:
    episode_id: str
    episode_generation: int
    request_id: str
    source_phase: PhaseId
    resume_phase: PhaseId
    trigger_tick: int
    trigger_timestamp_s: float
    trigger_reason: ActiveFrontTriggerReason
    attempt_index: int
    selected_primitive_id: str
    camera_command_sequence_id: str


@dataclass(frozen=True)
class ActiveFrontTriggerDecision:
    requestable: bool
    reason: ActiveFrontDecisionReason
    consecutive_unusable_ticks: int
    request: ActiveFrontReobserveRequest | None = None

    def __post_init__(self) -> None:
        if self.requestable != (self.request is not None):
            raise ValueError("requestable 必须与 request 是否存在一致")
        if self.requestable and self.reason is not ActiveFrontDecisionReason.REQUESTED:
            raise ValueError("成功 request 的 reason 必须是 REQUESTED")


@dataclass(frozen=True)
class ActionHistoryResetReceipt:
    episode_id: str
    request_id: str
    reset_control_tick: int
    generation_before: int
    generation_after: int
    action_chunk_cleared: bool
    temporal_ensemble_cleared: bool
    rtc_overlap_cleared: bool
    command_reference_invalidated: bool

    def __post_init__(self) -> None:
        if not self.episode_id or not self.request_id:
            raise ValueError("Action history reset identity 必须非空")
        for name in ("reset_control_tick", "generation_before", "generation_after"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"Action history reset {name} 必须是非负整数")
        for name in (
            "action_chunk_cleared",
            "temporal_ensemble_cleared",
            "rtc_overlap_cleared",
            "command_reference_invalidated",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Action history reset {name} 必须是 bool")

    def validate_for(self, request: ActiveFrontReobserveRequest) -> None:
        if self.episode_id != request.episode_id or self.request_id != request.request_id:
            raise ValueError("Action history reset receipt request identity 不匹配")
        if self.reset_control_tick < request.trigger_tick:
            raise ValueError("Action history reset 不能早于触发 Tick")
        if self.generation_before < 0 or self.generation_after != self.generation_before + 1:
            raise ValueError("Action history generation 必须严格递增一次")
        if not all(
            (
                self.action_chunk_cleared,
                self.temporal_ensemble_cleared,
                self.rtc_overlap_cleared,
                self.command_reference_invalidated,
            )
        ):
            raise ValueError("Action chunk/temporal/RTC/command reference 必须原子失效")


@dataclass(frozen=True)
class ActionHistoryResumeReceipt:
    episode_id: str
    request_id: str
    generation: int
    home_observation_sequence_ids: tuple[str, ...]
    generated_from_fresh_home_v2: bool
    stale_action_chunk_resumed: bool
    observation_v2_window_sha256: str | None = None

    def __post_init__(self) -> None:
        if not self.episode_id or not self.request_id:
            raise ValueError("Action history resume identity 必须非空")
        if (
            not isinstance(self.generation, int)
            or isinstance(self.generation, bool)
            or self.generation < 0
        ):
            raise ValueError("Action history resume generation 必须是非负整数")
        if not self.home_observation_sequence_ids or any(
            not isinstance(value, str) or not value
            for value in self.home_observation_sequence_ids
        ):
            raise ValueError("Action history resume HOME frame identity 必须非空")
        if len(set(self.home_observation_sequence_ids)) != len(
            self.home_observation_sequence_ids
        ):
            raise ValueError("Action history resume HOME frame identity 不能重复")
        if not isinstance(self.generated_from_fresh_home_v2, bool):
            raise TypeError("generated_from_fresh_home_v2 必须是 bool")
        if not isinstance(self.stale_action_chunk_resumed, bool):
            raise TypeError("stale_action_chunk_resumed 必须是 bool")
        if self.observation_v2_window_sha256 is not None and (
            len(self.observation_v2_window_sha256) != 64
            or any(
                value not in "0123456789abcdef"
                for value in self.observation_v2_window_sha256
            )
        ):
            raise ValueError("observation_v2_window_sha256 必须是 SHA-256 或 None")


@dataclass(frozen=True)
class Stage1ShadowCandidateReceipt:
    request_id: str
    candidate_digest: str
    shadow_only: bool
    live_memory_write_executed: bool
    provider_forward_count: int

    def __post_init__(self) -> None:
        if not self.request_id or not self.candidate_digest:
            raise ValueError("shadow candidate identity/digest 必须非空")
        if not isinstance(self.shadow_only, bool):
            raise TypeError("shadow_only 必须是 bool")
        if not isinstance(self.live_memory_write_executed, bool):
            raise TypeError("live_memory_write_executed 必须是 bool")
        if (
            not isinstance(self.provider_forward_count, int)
            or isinstance(self.provider_forward_count, bool)
            or self.provider_forward_count < 0
        ):
            raise ValueError("provider_forward_count 必须是非负整数")


@dataclass(frozen=True)
class Stage2MemoryCandidateReceipt:
    """Stage 2 的三帧 PRIMARY candidate；Memory write 仍延迟到 HOME 后。"""

    request_id: str
    candidate_digest: str
    commit_eligible: bool
    rejection_reasons: tuple[str, ...]
    memory_write_deferred: bool
    live_memory_write_executed: bool
    provider_forward_count: int
    collect_frame_digests: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.request_id or not self.candidate_digest:
            raise ValueError("Stage 2 candidate identity/digest 必须非空")
        if len(self.candidate_digest) != 64 or any(
            value not in "0123456789abcdef" for value in self.candidate_digest
        ):
            raise ValueError("Stage 2 candidate digest 必须是 SHA-256")
        for name in (
            "commit_eligible",
            "memory_write_deferred",
            "live_memory_write_executed",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"Stage 2 candidate {name} 必须是 bool")
        if self.commit_eligible != (not self.rejection_reasons):
            raise ValueError("Stage 2 candidate eligibility/reasons 语义冲突")
        if self.memory_write_deferred is not self.commit_eligible:
            raise ValueError("只有 eligible candidate 可以延迟 Memory write")
        if self.live_memory_write_executed:
            raise ValueError("candidate staged 阶段禁止 live Memory write")
        if self.provider_forward_count != 3:
            raise ValueError("Stage 2 PRIMARY candidate 必须精确绑定三次 forward")
        if len(self.collect_frame_digests) != 3 or len(
            set(self.collect_frame_digests)
        ) != 3:
            raise ValueError("Stage 2 candidate 必须绑定三个唯一 collect frame")
        for digest in self.collect_frame_digests:
            if len(digest) != 64 or any(
                value not in "0123456789abcdef" for value in digest
            ):
                raise ValueError("Stage 2 collect frame digest 必须是 SHA-256")


@dataclass(frozen=True)
class ActiveFrontSafetyEvidence:
    arm_hold_pass: bool = True
    tcp_hold_pass: bool = True
    gripper_open_hold_pass: bool = True
    contact_absent: bool = True
    active_window_open: bool = True

    def __post_init__(self) -> None:
        for name in (
            "arm_hold_pass",
            "tcp_hold_pass",
            "gripper_open_hold_pass",
            "contact_absent",
            "active_window_open",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"safety evidence.{name} 必须是 bool")

    def failure(self) -> ActiveFrontFailure | None:
        if not self.arm_hold_pass:
            return ActiveFrontFailure.ARM_HOLD_VIOLATION
        if not self.tcp_hold_pass:
            return ActiveFrontFailure.TCP_HOLD_VIOLATION
        if not self.gripper_open_hold_pass:
            return ActiveFrontFailure.GRIPPER_HOLD_VIOLATION
        if not self.contact_absent:
            return ActiveFrontFailure.CONTACT_DETECTED
        if not self.active_window_open:
            return ActiveFrontFailure.ACTIVE_WINDOW_CLOSED
        return None


@dataclass(frozen=True)
class HomeV2BarrierFrame:
    observation_sequence_id: str
    camera_at_home: bool
    fresh_observation_v2_frame: bool
    captured_after_return: bool
    contains_alternate_or_motion_rgb: bool

    def __post_init__(self) -> None:
        if not isinstance(self.observation_sequence_id, str):
            raise TypeError("observation_sequence_id 必须是字符串")
        for name in (
            "camera_at_home",
            "fresh_observation_v2_frame",
            "captured_after_return",
            "contains_alternate_or_motion_rgb",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"HOME barrier.{name} 必须是 bool")

    def valid(self) -> bool:
        return bool(
            self.observation_sequence_id
            and self.camera_at_home
            and self.fresh_observation_v2_frame
            and self.captured_after_return
            and not self.contains_alternate_or_motion_rgb
        )


@dataclass(frozen=True)
class ActiveFrontReobserveReceipt:
    episode_id: str
    request_id: str
    status: str
    source_phase: PhaseId
    resume_phase: PhaseId
    selected_primitive_id: str
    state_trace: tuple[str, ...]
    home_observation_sequence_ids: tuple[str, ...]
    action_history_generation_before: int
    action_history_generation_after_reset: int
    resumed_action_history_generation: int | None
    memory_read_count: int
    memory_write_count: int
    test_read_count: int
    provider_forward_count: int
    failure: ActiveFrontFailure | None
    version: str = ACTIVE_FRONT_REOBSERVE_VERSION

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "request_id": self.request_id,
            "status": self.status,
            "source_phase": self.source_phase.value,
            "resume_phase": self.resume_phase.value,
            "selected_primitive_id": self.selected_primitive_id,
            "state_trace": list(self.state_trace),
            "home_observation_sequence_ids": list(self.home_observation_sequence_ids),
            "action_history_generation_before": self.action_history_generation_before,
            "action_history_generation_after_reset": (
                self.action_history_generation_after_reset
            ),
            "resumed_action_history_generation": self.resumed_action_history_generation,
            "memory_read_count": self.memory_read_count,
            "memory_write_count": self.memory_write_count,
            "test_read_count": self.test_read_count,
            "provider_forward_count": self.provider_forward_count,
            "failure": None if self.failure is None else self.failure.value,
        }

    @property
    def audit_digest(self) -> str:
        payload = json.dumps(
            self.as_dict(),
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()


class ActiveFrontReobserveController:
    """单 Episode、单尝试、无 live Memory write 的确定性 supervisor。"""

    _TRANSITIONS: ClassVar[
        dict[
            tuple[ActiveFrontReobserveState, ActiveFrontSignal],
            ActiveFrontReobserveState,
        ]
    ] = {
        (
            ActiveFrontReobserveState.ACQUIRE_CAMERA_LEASE_AND_HOLD_ARM,
            ActiveFrontSignal.CAMERA_LEASE_ACQUIRED,
        ): ActiveFrontReobserveState.SELECT_FROZEN_PRIMITIVE,
        (
            ActiveFrontReobserveState.SELECT_FROZEN_PRIMITIVE,
            ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
        ): ActiveFrontReobserveState.MOVE_TO_VIEW,
        (
            ActiveFrontReobserveState.MOVE_TO_VIEW,
            ActiveFrontSignal.MOVE_COMPLETE,
        ): ActiveFrontReobserveState.SETTLE_AT_VIEW,
        (
            ActiveFrontReobserveState.SETTLE_AT_VIEW,
            ActiveFrontSignal.SETTLE_COMPLETE,
        ): ActiveFrontReobserveState.COLLECT,
        (
            ActiveFrontReobserveState.COLLECT,
            ActiveFrontSignal.COLLECTION_COMPLETE,
        ): ActiveFrontReobserveState.STAGE_CANDIDATE,
        (
            ActiveFrontReobserveState.STAGE_CANDIDATE,
            ActiveFrontSignal.SHADOW_CANDIDATE_STAGED,
        ): ActiveFrontReobserveState.RETURN_HOME,
        (
            ActiveFrontReobserveState.RETURN_HOME,
            ActiveFrontSignal.RETURN_HOME_COMPLETE,
        ): ActiveFrontReobserveState.VERIFY_HOME_AND_ARM_HOLD,
        (
            ActiveFrontReobserveState.RECHECK_SOURCE_INVARIANTS,
            ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED,
        ): ActiveFrontReobserveState.COMMIT_AND_RESUME,
    }

    def __init__(self, config: ActiveFrontReobserveConfig | None = None) -> None:
        self.config = config or ActiveFrontReobserveConfig()
        self._episode_id: str | None = None
        self._episode_generation = 0
        self._active_window_open = False
        self._latch_close_reason: str | None = None
        self._state = ActiveFrontReobserveState.IDLE
        self._state_trace = [self._state.value]
        self._consecutive_unusable = 0
        self._attempt_count = 0
        self._last_request_tick: int | None = None
        self._request: ActiveFrontReobserveRequest | None = None
        self._reset_receipt: ActionHistoryResetReceipt | None = None
        self._home_frame_ids: list[str] = []
        self._failure: ActiveFrontFailure | None = None
        self._stage2_candidate_commit_eligible: bool | None = None

    @property
    def state(self) -> ActiveFrontReobserveState:
        return self._state

    @property
    def request(self) -> ActiveFrontReobserveRequest | None:
        return self._request

    @property
    def active_window_open(self) -> bool:
        return self._active_window_open

    @property
    def latch_close_reason(self) -> str | None:
        return self._latch_close_reason

    @property
    def arm_owner(self) -> ControllerOwner:
        return ControllerOwner.SAFE_HOLD

    @property
    def external_camera_owner(self) -> ExternalCameraControllerOwner:
        if self._state is ActiveFrontReobserveState.FAILSAFE_RETURN:
            return ExternalCameraControllerOwner.FAILSAFE_RETURN
        if self._state in {
            ActiveFrontReobserveState.REQUESTED,
            ActiveFrontReobserveState.ACQUIRE_CAMERA_LEASE_AND_HOLD_ARM,
            ActiveFrontReobserveState.SELECT_FROZEN_PRIMITIVE,
            ActiveFrontReobserveState.MOVE_TO_VIEW,
            ActiveFrontReobserveState.SETTLE_AT_VIEW,
            ActiveFrontReobserveState.COLLECT,
            ActiveFrontReobserveState.STAGE_CANDIDATE,
            ActiveFrontReobserveState.RETURN_HOME,
            ActiveFrontReobserveState.VERIFY_HOME_AND_ARM_HOLD,
            ActiveFrontReobserveState.RECHECK_SOURCE_INVARIANTS,
            ActiveFrontReobserveState.COMMIT_AND_RESUME,
        }:
            return ExternalCameraControllerOwner.ACTIVE_REOBSERVE
        if self._state is ActiveFrontReobserveState.IDLE:
            return ExternalCameraControllerOwner.HOME_HOLD
        return ExternalCameraControllerOwner.NONE

    def reset_episode(self, episode_id: str, *, episode_generation: int) -> None:
        """只接受严格递增的新 Episode generation，不能伪装 mid-Episode reset。"""

        if not episode_id:
            raise ValueError("episode_id 必须是非空字符串")
        if (
            not isinstance(episode_generation, int)
            or isinstance(episode_generation, bool)
            or episode_generation != self._episode_generation + 1
        ):
            raise ValueError("episode_generation 必须相对上一 Episode 严格递增一次")
        if self._episode_id == episode_id:
            raise ValueError("不能用同一个 episode_id 重新打开 active latch")
        self._episode_id = episode_id
        self._episode_generation = episode_generation
        self._active_window_open = True
        self._latch_close_reason = None
        self._state = ActiveFrontReobserveState.IDLE
        self._state_trace = [self._state.value]
        self._consecutive_unusable = 0
        self._attempt_count = 0
        self._last_request_tick = None
        self._request = None
        self._reset_receipt = None
        self._home_frame_ids = []
        self._failure = None
        self._stage2_candidate_commit_eligible = None

    def _set_state(self, state: ActiveFrontReobserveState) -> None:
        self._state = state
        self._state_trace.append(state.value)

    def _close_latch(self, reason: str) -> None:
        if self._active_window_open:
            self._active_window_open = False
            self._latch_close_reason = reason

    def _observe_monotonic_latch(self, evidence: ActiveFrontTriggerEvidence) -> None:
        if evidence.source_phase in POST_ACTIVE_WINDOW_PHASES:
            self._close_latch(f"entered_post_active_window:{evidence.source_phase.value}")
        elif evidence.object_contact:
            self._close_latch("object_contact")
        elif evidence.gripper_close_commanded:
            self._close_latch("gripper_close_commanded")
        elif evidence.grasp_candidate:
            self._close_latch("grasp_candidate")
        elif evidence.grasp_verified:
            self._close_latch("grasp_verified")
        elif evidence.object_motion_risk:
            self._close_latch("object_motion_risk")

    def consider_trigger(
        self,
        evidence: ActiveFrontTriggerEvidence,
    ) -> ActiveFrontTriggerDecision:
        if (
            evidence.episode_id != self._episode_id
            or evidence.episode_generation != self._episode_generation
        ):
            raise ValueError("trigger evidence Episode identity 不匹配")
        self._observe_monotonic_latch(evidence)

        def reject(
            reason: ActiveFrontDecisionReason,
            *,
            reset_streak: bool = True,
        ) -> ActiveFrontTriggerDecision:
            if reset_streak:
                self._consecutive_unusable = 0
            return ActiveFrontTriggerDecision(
                requestable=False,
                reason=reason,
                consecutive_unusable_ticks=self._consecutive_unusable,
            )

        if not self.config.enabled:
            return reject(ActiveFrontDecisionReason.FEATURE_DISABLED)
        if self._state is not ActiveFrontReobserveState.IDLE:
            return reject(ActiveFrontDecisionReason.REQUEST_ALREADY_ACTIVE, reset_streak=False)
        if evidence.source_phase not in ALLOWED_ACTIVE_SOURCE_PHASES:
            return reject(ActiveFrontDecisionReason.DISALLOWED_SOURCE_PHASE)
        if not self._active_window_open:
            return reject(ActiveFrontDecisionReason.ACTIVE_WINDOW_CLOSED)
        if evidence.wrist_object_measurement_usable:
            return reject(ActiveFrontDecisionReason.DIRECT_WRIST_EVIDENCE_AVAILABLE)
        if evidence.front_home_object_measurement_usable:
            return reject(ActiveFrontDecisionReason.HOME_FRONT_EVIDENCE_AVAILABLE)
        if evidence.object_memory_navigation_state_available:
            return reject(ActiveFrontDecisionReason.OBJECT_MEMORY_AVAILABLE)
        capability_absent_allowed = bool(
            self.config.allow_capability_absent_trigger
            and evidence.failure_reason
            is ActiveFrontTriggerReason.NO_QUALIFIED_WRIST_PROVIDER_IN_PARENT
        )
        if (
            evidence.failure_reason not in VIEWPOINT_RESOLVABLE_REASONS
            and not capability_absent_allowed
        ):
            return reject(ActiveFrontDecisionReason.FAILURE_NOT_VIEWPOINT_RESOLVABLE)
        if not (
            evidence.arm_hold_prerequisites_pass
            and evidence.camera_home_prerequisites_pass
        ):
            return reject(ActiveFrontDecisionReason.ARM_OR_CAMERA_PREREQUISITE_FAILED)
        if self._attempt_count >= self.config.maximum_attempts_per_episode:
            return reject(ActiveFrontDecisionReason.ATTEMPT_BUDGET_EXHAUSTED)
        if (
            self._last_request_tick is not None
            and evidence.control_tick - self._last_request_tick < self.config.cooldown_ticks
        ):
            return reject(ActiveFrontDecisionReason.COOLDOWN_ACTIVE, reset_streak=False)

        self._consecutive_unusable += 1
        if self._consecutive_unusable < self.config.consecutive_unusable_ticks:
            return reject(
                ActiveFrontDecisionReason.CONSECUTIVE_EVIDENCE_PENDING,
                reset_streak=False,
            )
        self._attempt_count += 1
        self._last_request_tick = evidence.control_tick
        request_id = f"{evidence.episode_id}-active-front-{self._attempt_count:02d}"
        request = ActiveFrontReobserveRequest(
            episode_id=evidence.episode_id,
            episode_generation=evidence.episode_generation,
            request_id=request_id,
            source_phase=evidence.source_phase,
            resume_phase=evidence.source_phase,
            trigger_tick=evidence.control_tick,
            trigger_timestamp_s=evidence.timestamp_s,
            trigger_reason=evidence.failure_reason,
            attempt_index=self._attempt_count,
            selected_primitive_id=self.config.selected_primitive_id,
            camera_command_sequence_id=f"{request_id}-camera-command-00",
        )
        self._request = request
        self._consecutive_unusable = 0
        self._set_state(ActiveFrontReobserveState.REQUESTED)
        return ActiveFrontTriggerDecision(
            requestable=True,
            reason=ActiveFrontDecisionReason.REQUESTED,
            consecutive_unusable_ticks=0,
            request=request,
        )

    def begin(self, reset_receipt: ActionHistoryResetReceipt) -> None:
        if self._state is not ActiveFrontReobserveState.REQUESTED or self._request is None:
            raise RuntimeError("只有 REQUESTED 状态可以 begin")
        try:
            reset_receipt.validate_for(self._request)
        except (TypeError, ValueError):
            self._fail(ActiveFrontFailure.ACTION_HISTORY_RESET_INVALID, camera_at_home=True)
            raise
        self._reset_receipt = reset_receipt
        self._set_state(ActiveFrontReobserveState.ACQUIRE_CAMERA_LEASE_AND_HOLD_ARM)

    def _fail(self, failure: ActiveFrontFailure, *, camera_at_home: bool) -> None:
        if self._failure is None:
            self._failure = failure
        target = (
            ActiveFrontReobserveState.FAILED_SAFE_HOLD
            if camera_at_home
            else ActiveFrontReobserveState.FAILSAFE_RETURN
        )
        if self._state is not target:
            self._set_state(target)

    def observe_safety(
        self,
        safety: ActiveFrontSafetyEvidence,
        *,
        camera_at_home: bool,
    ) -> bool:
        """在没有离散状态转换的 motion Tick 上仍持续执行 fail-closed。"""

        if self._request is None:
            raise RuntimeError("没有 active request")
        if not isinstance(safety, ActiveFrontSafetyEvidence):
            raise TypeError("safety 必须是 ActiveFrontSafetyEvidence")
        failure = safety.failure()
        if failure is None:
            return True
        self._close_latch(failure.value)
        self._fail(failure, camera_at_home=camera_at_home)
        return False

    def advance(
        self,
        signal: ActiveFrontSignal,
        *,
        safety: ActiveFrontSafetyEvidence | None = None,
        selected_primitive_id: str | None = None,
        shadow_candidate_receipt: (
            Stage1ShadowCandidateReceipt | Stage2MemoryCandidateReceipt | None
        ) = None,
        source_phase: PhaseId | None = None,
        source_invariants_passed: bool | None = None,
    ) -> None:
        if self._request is None:
            raise RuntimeError("没有 active request")
        safety = safety or ActiveFrontSafetyEvidence()
        camera_at_home = self._state in {
            ActiveFrontReobserveState.REQUESTED,
            ActiveFrontReobserveState.ACQUIRE_CAMERA_LEASE_AND_HOLD_ARM,
            ActiveFrontReobserveState.SELECT_FROZEN_PRIMITIVE,
            ActiveFrontReobserveState.VERIFY_HOME_AND_ARM_HOLD,
            ActiveFrontReobserveState.RECHECK_SOURCE_INVARIANTS,
            ActiveFrontReobserveState.COMMIT_AND_RESUME,
        }
        if not self.observe_safety(safety, camera_at_home=camera_at_home):
            return
        target = self._TRANSITIONS.get((self._state, signal))
        if target is None:
            self._fail(
                ActiveFrontFailure.STATE_TRANSITION_INVALID,
                camera_at_home=self._state
                in {
                    ActiveFrontReobserveState.REQUESTED,
                    ActiveFrontReobserveState.ACQUIRE_CAMERA_LEASE_AND_HOLD_ARM,
                    ActiveFrontReobserveState.SELECT_FROZEN_PRIMITIVE,
                },
            )
            return
        if (
            signal is ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED
            and selected_primitive_id != self._request.selected_primitive_id
        ):
            self._fail(
                ActiveFrontFailure.PRIMITIVE_IDENTITY_MISMATCH,
                camera_at_home=True,
            )
            return
        if signal is ActiveFrontSignal.SHADOW_CANDIDATE_STAGED:
            if isinstance(shadow_candidate_receipt, Stage1ShadowCandidateReceipt):
                valid_shadow_candidate = bool(
                    shadow_candidate_receipt.request_id == self._request.request_id
                    and shadow_candidate_receipt.shadow_only
                    and not shadow_candidate_receipt.live_memory_write_executed
                    and shadow_candidate_receipt.provider_forward_count == 0
                )
                self._stage2_candidate_commit_eligible = None
            elif isinstance(shadow_candidate_receipt, Stage2MemoryCandidateReceipt):
                valid_shadow_candidate = bool(
                    shadow_candidate_receipt.request_id == self._request.request_id
                    and not shadow_candidate_receipt.live_memory_write_executed
                    and shadow_candidate_receipt.provider_forward_count == 3
                )
                self._stage2_candidate_commit_eligible = (
                    shadow_candidate_receipt.commit_eligible
                )
            else:
                valid_shadow_candidate = False
            if not valid_shadow_candidate:
                self._fail(
                    ActiveFrontFailure.STATE_TRANSITION_INVALID,
                    camera_at_home=False,
                )
                return
        if signal is ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED:
            if not self._active_window_open:
                self._fail(
                    ActiveFrontFailure.ACTIVE_WINDOW_CLOSED,
                    camera_at_home=True,
                )
                return
            if (
                source_invariants_passed is not True
                or source_phase is not self._request.resume_phase
            ):
                self._fail(
                    ActiveFrontFailure.SOURCE_INVARIANT_FAILED,
                    camera_at_home=True,
                )
                return
        self._set_state(target)

    def accept_home_v2_barrier_frame(self, frame: HomeV2BarrierFrame) -> None:
        if self._state is not ActiveFrontReobserveState.VERIFY_HOME_AND_ARM_HOLD:
            self._fail(ActiveFrontFailure.HOME_FRAME_INVALID, camera_at_home=False)
            return
        if not frame.valid():
            self._fail(
                ActiveFrontFailure.HOME_FRAME_INVALID,
                camera_at_home=frame.camera_at_home,
            )
            return
        if frame.observation_sequence_id in self._home_frame_ids:
            self._fail(ActiveFrontFailure.HOME_FRAME_DUPLICATE, camera_at_home=True)
            return
        self._home_frame_ids.append(frame.observation_sequence_id)
        if len(self._home_frame_ids) == self.config.home_v2_barrier_frames:
            if self._stage2_candidate_commit_eligible is False:
                self._fail(ActiveFrontFailure.CANDIDATE_REJECTED, camera_at_home=True)
            else:
                self._set_state(ActiveFrontReobserveState.RECHECK_SOURCE_INVARIANTS)

    def fail(
        self,
        failure: ActiveFrontFailure,
        *,
        camera_at_home: bool,
    ) -> None:
        self._fail(failure, camera_at_home=camera_at_home)

    def complete_failsafe_return(self, *, home_verified: bool) -> None:
        if self._state is not ActiveFrontReobserveState.FAILSAFE_RETURN:
            raise RuntimeError("当前不在 FAILSAFE_RETURN")
        if not home_verified:
            self._failure = ActiveFrontFailure.CAMERA_RETURN_FAILED
        self._set_state(ActiveFrontReobserveState.FAILED_SAFE_HOLD)

    def complete_no_write_resume(
        self,
        receipt: ActionHistoryResumeReceipt,
    ) -> ActiveFrontReobserveReceipt:
        if (
            self._state is not ActiveFrontReobserveState.COMMIT_AND_RESUME
            or self._request is None
            or self._reset_receipt is None
        ):
            raise RuntimeError("只有 COMMIT_AND_RESUME 可以完成 no-write resume")
        valid = bool(
            receipt.episode_id == self._request.episode_id
            and receipt.request_id == self._request.request_id
            and receipt.generation == self._reset_receipt.generation_after + 1
            and receipt.home_observation_sequence_ids == tuple(self._home_frame_ids)
            and receipt.generated_from_fresh_home_v2
            and not receipt.stale_action_chunk_resumed
        )
        if not valid:
            self._fail(ActiveFrontFailure.STALE_ACTION_HISTORY_RESUME, camera_at_home=True)
            return self.receipt(resume_receipt=receipt)
        self._set_state(ActiveFrontReobserveState.COMPLETE_NO_WRITE)
        return self.receipt(resume_receipt=receipt)

    def complete_stage2_memory_write(
        self,
        receipt: ActionHistoryResumeReceipt,
        *,
        memory_write_count: int,
        provider_forward_count: int,
    ) -> ActiveFrontReobserveReceipt:
        """HOME 后一次 Memory write 与 fresh shadow replan 的 supervisor 终态。"""

        if (
            self._state is not ActiveFrontReobserveState.COMMIT_AND_RESUME
            or self._request is None
            or self._reset_receipt is None
        ):
            raise RuntimeError("只有 COMMIT_AND_RESUME 可以完成 Stage 2 Memory write")
        valid = bool(
            receipt.episode_id == self._request.episode_id
            and receipt.request_id == self._request.request_id
            and receipt.generation == self._reset_receipt.generation_after + 1
            and receipt.home_observation_sequence_ids == tuple(self._home_frame_ids)
            and receipt.generated_from_fresh_home_v2
            and not receipt.stale_action_chunk_resumed
            and receipt.observation_v2_window_sha256 is not None
            and memory_write_count == 1
            and provider_forward_count == 4
            and self._stage2_candidate_commit_eligible is True
        )
        if not valid:
            self._fail(ActiveFrontFailure.STALE_ACTION_HISTORY_RESUME, camera_at_home=True)
            return self.receipt(resume_receipt=receipt)
        self._set_state(ActiveFrontReobserveState.COMPLETE_STAGE2_MEMORY_WRITE)
        return self.receipt(
            resume_receipt=receipt,
            memory_write_count=memory_write_count,
            provider_forward_count=provider_forward_count,
        )

    def receipt(
        self,
        *,
        resume_receipt: ActionHistoryResumeReceipt | None = None,
        memory_write_count: int = 0,
        provider_forward_count: int = 0,
    ) -> ActiveFrontReobserveReceipt:
        if self._request is None or self._reset_receipt is None:
            raise RuntimeError("receipt 需要 request 与 Action history reset evidence")
        if self._state is ActiveFrontReobserveState.COMPLETE_NO_WRITE:
            status = "complete-stage1-shadow-no-write"
        elif self._state is ActiveFrontReobserveState.COMPLETE_STAGE2_MEMORY_WRITE:
            status = "complete-stage2-memory-write-shadow-replan"
        elif self._state in {
            ActiveFrontReobserveState.FAILSAFE_RETURN,
            ActiveFrontReobserveState.FAILED_SAFE_HOLD,
        }:
            status = "failed-safe-hold-no-write"
        else:
            status = "in-progress-stage1-no-write"
        return ActiveFrontReobserveReceipt(
            episode_id=self._request.episode_id,
            request_id=self._request.request_id,
            status=status,
            source_phase=self._request.source_phase,
            resume_phase=self._request.resume_phase,
            selected_primitive_id=self._request.selected_primitive_id,
            state_trace=tuple(self._state_trace),
            home_observation_sequence_ids=tuple(self._home_frame_ids),
            action_history_generation_before=self._reset_receipt.generation_before,
            action_history_generation_after_reset=self._reset_receipt.generation_after,
            resumed_action_history_generation=(
                None if resume_receipt is None else resume_receipt.generation
            ),
            memory_read_count=0,
            memory_write_count=memory_write_count,
            test_read_count=0,
            provider_forward_count=provider_forward_count,
            failure=self._failure,
        )


__all__ = [
    "ACTIVE_FRONT_REOBSERVE_VERSION",
    "ALLOWED_ACTIVE_SOURCE_PHASES",
    "POST_ACTIVE_WINDOW_PHASES",
    "VIEWPOINT_RESOLVABLE_REASONS",
    "ActionHistoryResetReceipt",
    "ActionHistoryResumeReceipt",
    "ActiveFrontDecisionReason",
    "ActiveFrontFailure",
    "ActiveFrontReobserveConfig",
    "ActiveFrontReobserveController",
    "ActiveFrontReobserveReceipt",
    "ActiveFrontReobserveRequest",
    "ActiveFrontReobserveState",
    "ActiveFrontSafetyEvidence",
    "ActiveFrontSignal",
    "ActiveFrontTriggerDecision",
    "ActiveFrontTriggerEvidence",
    "ActiveFrontTriggerReason",
    "ExternalCameraControllerOwner",
    "HomeV2BarrierFrame",
    "Stage1ShadowCandidateReceipt",
    "Stage2MemoryCandidateReceipt",
]
