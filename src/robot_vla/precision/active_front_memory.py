"""E018-P1 Stage 2 的 PRIMARY-only Object Memory 两阶段提交事务机。

provider identity、不可变 frame/baseline evidence 与 adapter 位于
``active_front_memory_provider``；本模块只管理 pending candidate、返回 HOME
屏障、source recheck、原子提交与 shadow Action generation receipt。
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from enum import Enum

import numpy as np

from robot_vla.executive.contracts import PhaseId
from robot_vla.precision.active_front_memory_provider import (
    ACTIVE_FRONT_HOME_PRIMITIVE_ID,
    ACTIVE_FRONT_INFORMATION_GAIN_CANDIDATES,
    ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
    ACTIVE_FRONT_PROHIBITED_MEMORY_WRITE_PRIMITIVE_IDS,
    ACTIVE_FRONT_QUALIFIED_SHADOW_PRIMITIVE_IDS,
    ACTIVE_FRONT_SCORE_SEMANTICS,
    ACTIVE_FRONT_STAGE2_EXECUTION_MODE,
    ACTIVE_FRONT_STAGE2_PROVIDER_ADAPTER_VERSION,
    ACTIVE_FRONT_STAGE2_PROVIDER_IDENTITY_VERSION,
    ACTIVE_FRONT_STAGE2_VERSION,
    ActiveFrontFrameAdaptation,
    ActiveFrontScoreComponents,
    ActiveFrontStage2Config,
    ActiveFrontStage2FrameEvidence,
    ActiveFrontStage2ProviderAdapter,
    ActiveFrontStage2ProviderIdentity,
    PassiveBaselineEvidence,
    PassiveHomeScoreEvidence,
    _canonical_sha256,
    _identity,
    _measurement_dict,
    _sha256,
    _state_dict,
    _timestamp,
    build_stage2_object_memory_config,
    d049_primary_provider_identity,
)
from robot_vla.precision.active_front_provider import ActiveFrontModelInput
from robot_vla.precision.active_front_reobserve import (
    ALLOWED_ACTIVE_SOURCE_PHASES,
    ActionHistoryResetReceipt,
    ActionHistoryResumeReceipt,
    ActiveFrontReobserveRequest,
    ActiveFrontSafetyEvidence,
    HomeV2BarrierFrame,
)
from robot_vla.precision.object_memory import (
    ExplicitObjectStateMemory,
    ObjectCandidateDecision,
    ObjectCandidateWindowVerifier,
    ObjectMeasurement,
    ObjectMemoryMode,
    ObjectMemorySafetyContext,
    ObjectMemoryUpdate,
    ObjectState,
    ObjectStateRequirement,
    resolve_object_state,
)


@dataclass(frozen=True)
class PendingActiveViewCandidate:
    episode_id: str
    episode_generation: int
    request_id: str
    window_id: str
    source_phase: PhaseId
    resume_phase: PhaseId
    frame_sequence_ids: tuple[str, str, str]
    frame_timestamps_s: tuple[float, float, float]
    frame_digests: tuple[str, str, str]
    model_input_digests: tuple[str, str, str]
    provider_output_digests: tuple[str, str, str]
    actual_pose_sha256s: tuple[str, str, str]
    write_scores: tuple[float, float, float]
    score_components: tuple[ActiveFrontScoreComponents, ...]
    minimum_candidate_score: float
    information_gain: float | None
    position_spread_m: float
    innovation_m: float | None
    final_measurement: ObjectMeasurement
    final_measurement_digest: str
    provider_identity: ActiveFrontStage2ProviderIdentity
    baseline_digest: str
    created_timestamp_s: float
    maximum_age_s: float
    commit_eligible: bool
    rejection_reasons: tuple[str, ...]
    version: str = ACTIVE_FRONT_STAGE2_VERSION

    def __post_init__(self) -> None:
        if len(self.score_components) != 3:
            raise ValueError("pending candidate 必须保留三个 score component records")
        for values, name in (
            (self.frame_sequence_ids, "frame_sequence_ids"),
            (self.frame_timestamps_s, "frame_timestamps_s"),
            (self.frame_digests, "frame_digests"),
            (self.model_input_digests, "model_input_digests"),
            (self.provider_output_digests, "provider_output_digests"),
            (self.actual_pose_sha256s, "actual_pose_sha256s"),
            (self.write_scores, "write_scores"),
        ):
            if len(values) != 3:
                raise ValueError(f"pending candidate {name} 必须精确为 3")
        if len(set(self.frame_sequence_ids)) != 3:
            raise ValueError("pending candidate frame identity 不能重复")
        if len(set(self.model_input_digests)) != 3:
            raise ValueError("pending candidate model input digest 不能重复")
        if len(set(self.provider_output_digests)) != 3:
            raise ValueError("pending candidate provider output digest 不能重复")
        if not self.frame_timestamps_s[0] < self.frame_timestamps_s[1] < self.frame_timestamps_s[2]:
            raise ValueError("pending candidate frame timestamp 必须严格递增")
        for value in (
            self.frame_digests
            + self.model_input_digests
            + self.provider_output_digests
            + self.actual_pose_sha256s
        ):
            _sha256(value, "pending frame/pose digest")
        _sha256(self.final_measurement_digest, "final_measurement_digest")
        _sha256(self.baseline_digest, "baseline_digest")
        if self.final_measurement.timestamp_s != self.frame_timestamps_s[-1]:
            raise ValueError("pending candidate 必须以第三/最终 frame 作为唯一 measurement")
        if self.final_measurement_digest != _canonical_sha256(
            _measurement_dict(self.final_measurement)
        ):
            raise ValueError("pending final measurement digest 漂移")
        if self.minimum_candidate_score != min(self.write_scores):
            raise ValueError("minimum_candidate_score 必须由三个 frame 机械派生")
        if not math.isfinite(self.position_spread_m) or self.position_spread_m < 0.0:
            raise ValueError("position_spread_m 必须是有限非负数")
        if self.innovation_m is not None and (
            not math.isfinite(self.innovation_m) or self.innovation_m < 0.0
        ):
            raise ValueError("innovation_m 必须是有限非负数或 None")
        if self.information_gain is not None and not math.isfinite(self.information_gain):
            raise ValueError("information_gain 必须有限或 None")
        object.__setattr__(
            self,
            "created_timestamp_s",
            _timestamp(self.created_timestamp_s, "candidate created timestamp"),
        )
        if self.created_timestamp_s != self.final_measurement.timestamp_s:
            raise ValueError("candidate created time 必须等于 final measurement time")
        if not math.isfinite(self.maximum_age_s) or self.maximum_age_s <= 0.0:
            raise ValueError("maximum_age_s 必须是有限正数")
        if self.commit_eligible != (not self.rejection_reasons):
            raise ValueError("commit_eligible 与 rejection reasons 不一致")
        if self.version != ACTIVE_FRONT_STAGE2_VERSION:
            raise ValueError("PendingActiveViewCandidate version 漂移")

    def as_dict(self) -> dict[str, object]:
        return {
            "version": self.version,
            "episode_id": self.episode_id,
            "episode_generation": self.episode_generation,
            "request_id": self.request_id,
            "window_id": self.window_id,
            "source_phase": self.source_phase.value,
            "resume_phase": self.resume_phase.value,
            "frame_sequence_ids": self.frame_sequence_ids,
            "frame_timestamps_s": self.frame_timestamps_s,
            "frame_digests": self.frame_digests,
            "model_input_digests": self.model_input_digests,
            "provider_output_digests": self.provider_output_digests,
            "actual_pose_sha256s": self.actual_pose_sha256s,
            "write_scores": self.write_scores,
            "score_components": [asdict(value) for value in self.score_components],
            "minimum_candidate_score": self.minimum_candidate_score,
            "information_gain": self.information_gain,
            "position_spread_m": self.position_spread_m,
            "innovation_m": self.innovation_m,
            "final_measurement": _measurement_dict(self.final_measurement),
            "final_measurement_digest": self.final_measurement_digest,
            "provider_identity": self.provider_identity.to_dict(),
            "provider_identity_sha256": self.provider_identity.sha256,
            "baseline_digest": self.baseline_digest,
            "created_timestamp_s": self.created_timestamp_s,
            "maximum_age_s": self.maximum_age_s,
            "commit_eligible": self.commit_eligible,
            "rejection_reasons": self.rejection_reasons,
        }

    @property
    def digest(self) -> str:
        return _canonical_sha256(self.as_dict())


class PendingActiveViewState(str, Enum):
    EMPTY = "empty"
    COLLECTING = "collecting"
    VERIFIED_PENDING = "verified_pending"
    RETURN_HOME_REQUIRED_NO_COMMIT = "return_home_required_no_commit"
    RETURNING_HOME = "returning_home"
    RETURNING_HOME_NO_COMMIT = "returning_home_no_commit"
    HOME_BARRIER_PASSED = "home_barrier_passed"
    HOME_VERIFIED_FAILED_SAFE_HOLD = "home_verified_failed_safe_hold"
    SOURCE_RECHECK_PASSED = "source_recheck_passed"
    COMMITTED = "committed"
    SHADOW_REPLAN_FAILED_SAFE_HOLD = "shadow_replan_failed_safe_hold"
    REJECTED = "rejected"
    EXPIRED = "expired"
    RESET_CLEARED = "reset_cleared"


@dataclass(frozen=True)
class ActiveFrontSourceRecheckEvidence:
    episode_id: str
    episode_generation: int
    request_id: str
    candidate_digest: str
    timestamp_s: float
    source_phase: PhaseId
    camera_at_home: bool
    source_invariants_passed: bool
    active_window_open: bool
    qualified_direct_wrist_measurement_usable: bool
    qualified_direct_wrist_evidence_identity_sha256: str

    def __post_init__(self) -> None:
        for name in ("episode_id", "request_id"):
            _identity(getattr(self, name), name)
        _sha256(self.candidate_digest, "candidate_digest")
        object.__setattr__(self, "timestamp_s", _timestamp(self.timestamp_s, "recheck timestamp"))
        if not isinstance(self.source_phase, PhaseId):
            raise TypeError("recheck source_phase 必须是 PhaseId")
        for name in (
            "camera_at_home",
            "source_invariants_passed",
            "active_window_open",
            "qualified_direct_wrist_measurement_usable",
        ):
            if not isinstance(getattr(self, name), bool):
                raise TypeError(f"recheck {name} 必须为 bool")
        _sha256(
            self.qualified_direct_wrist_evidence_identity_sha256,
            "qualified_direct_wrist_evidence_identity_sha256",
        )


@dataclass(frozen=True)
class DelayedActiveMemoryCommitReceipt:
    episode_id: str
    episode_generation: int
    request_id: str
    candidate_digest: str
    status: str
    commit_timestamp_s: float
    pre_state_digest: str
    post_state_digest: str
    last_observed_timestamp_s: float
    state_timestamp_s: float
    observable_now: bool
    memory_only: bool
    contact_authorized: bool
    memory_write_count: int
    provider_identity: ActiveFrontStage2ProviderIdentity
    provider_identity_sha256: str
    final_measurement_digest: str
    source_recheck_wrist_evidence_identity_sha256: str
    home_observation_sequence_ids: tuple[str, ...]
    home_observation_timestamps_s: tuple[float, ...]
    home_frame_digests: tuple[str, ...]
    version: str = ACTIVE_FRONT_STAGE2_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.episode_id, "episode_id"),
            (self.request_id, "request_id"),
        ):
            _identity(value, name)
        if (
            not isinstance(self.episode_generation, int)
            or isinstance(self.episode_generation, bool)
            or self.episode_generation <= 0
        ):
            raise ValueError("delayed commit episode_generation 必须是正整数")
        for value, name in (
            (self.candidate_digest, "candidate_digest"),
            (self.pre_state_digest, "pre_state_digest"),
            (self.post_state_digest, "post_state_digest"),
            (self.provider_identity_sha256, "provider_identity_sha256"),
            (self.final_measurement_digest, "final_measurement_digest"),
            (
                self.source_recheck_wrist_evidence_identity_sha256,
                "source_recheck_wrist_evidence_identity_sha256",
            ),
        ):
            _sha256(value, name)
        object.__setattr__(
            self,
            "commit_timestamp_s",
            _timestamp(self.commit_timestamp_s, "commit_timestamp_s"),
        )
        object.__setattr__(
            self,
            "last_observed_timestamp_s",
            _timestamp(self.last_observed_timestamp_s, "last_observed_timestamp_s"),
        )
        object.__setattr__(
            self,
            "state_timestamp_s",
            _timestamp(self.state_timestamp_s, "state_timestamp_s"),
        )
        if self.last_observed_timestamp_s > self.commit_timestamp_s + 1e-12:
            raise ValueError("delayed commit observation timestamp 不能晚于 commit")
        if self.state_timestamp_s != self.commit_timestamp_s:
            raise ValueError("delayed commit state timestamp 必须等于 commit timestamp")
        if not isinstance(self.provider_identity, ActiveFrontStage2ProviderIdentity):
            raise TypeError("delayed commit provider_identity 类型错误")
        if self.provider_identity.sha256 != self.provider_identity_sha256:
            raise ValueError("delayed commit provider identity digest 漂移")
        if self.provider_identity.sha256 != d049_primary_provider_identity().sha256:
            raise ValueError("delayed commit provider 必须是 D049 PRIMARY")
        if self.status != "committed-primary-memory-only":
            raise ValueError("delayed commit receipt status 漂移")
        if self.memory_write_count != 1:
            raise ValueError("成功 delayed commit 必须 exactly-one write")
        if self.observable_now or not self.memory_only or self.contact_authorized:
            raise ValueError("delayed commit 必须是 non-observable memory-only/no-contact")
        if len(self.home_observation_sequence_ids) != 4:
            raise ValueError("delayed commit 必须绑定四个 HOME frames")
        if (
            len(set(self.home_observation_sequence_ids)) != 4
            or any(not value for value in self.home_observation_sequence_ids)
        ):
            raise ValueError("delayed commit HOME frame identity 必须互异非空")
        if len(self.home_observation_timestamps_s) != 4:
            raise ValueError("delayed commit 必须绑定四个 HOME timestamps")
        if len(self.home_frame_digests) != 4:
            raise ValueError("delayed commit 必须绑定四个 HOME frame digests")
        timestamps = tuple(
            _timestamp(value, "home_observation_timestamp_s")
            for value in self.home_observation_timestamps_s
        )
        object.__setattr__(self, "home_observation_timestamps_s", timestamps)
        if any(
            later <= earlier + 1e-12
            for earlier, later in zip(
                self.home_observation_timestamps_s,
                self.home_observation_timestamps_s[1:],
            )
        ):
            raise ValueError("HOME receipt timestamp 必须严格递增")
        for value in self.home_frame_digests:
            _sha256(value, "home_frame_digest")
        if len(set(self.home_frame_digests)) != 4:
            raise ValueError("delayed commit HOME frame digest 必须互异")
        if self.version != ACTIVE_FRONT_STAGE2_VERSION:
            raise ValueError("delayed commit receipt version 漂移")

    @property
    def digest(self) -> str:
        value = asdict(self)
        return _canonical_sha256(value)


@dataclass(frozen=True)
class DelayedActiveMemoryNoCommitReceipt:
    """commit-time 拒绝凭据；证明 candidate 零写入及旧 Memory 的安全失效。"""

    episode_id: str
    episode_generation: int
    request_id: str
    candidate_digest: str
    status: str
    commit_timestamp_s: float
    pre_state_digest: str
    post_state_digest: str
    rejection_reasons: tuple[str, ...]
    safety_reasons: tuple[str, ...]
    candidate_write_count: int
    prior_memory_safety_invalidated: bool
    accepted_update_count_before: int
    accepted_update_count_after: int
    version: str = ACTIVE_FRONT_STAGE2_VERSION

    def __post_init__(self) -> None:
        for value, name in (
            (self.episode_id, "episode_id"),
            (self.request_id, "request_id"),
        ):
            _identity(value, name)
        if (
            not isinstance(self.episode_generation, int)
            or isinstance(self.episode_generation, bool)
            or self.episode_generation <= 0
        ):
            raise ValueError("no-commit receipt episode_generation 必须是正整数")
        for value, name in (
            (self.candidate_digest, "candidate_digest"),
            (self.pre_state_digest, "pre_state_digest"),
            (self.post_state_digest, "post_state_digest"),
        ):
            _sha256(value, name)
        object.__setattr__(
            self,
            "commit_timestamp_s",
            _timestamp(self.commit_timestamp_s, "no-commit timestamp"),
        )
        if self.status != "rejected-no-candidate-write":
            raise ValueError("no-commit receipt status 漂移")
        if not self.rejection_reasons or len(set(self.rejection_reasons)) != len(
            self.rejection_reasons
        ):
            raise ValueError("no-commit receipt 必须有唯一 rejection reasons")
        if len(set(self.safety_reasons)) != len(self.safety_reasons):
            raise ValueError("no-commit safety reasons 不能重复")
        if not set(self.safety_reasons).issubset(self.rejection_reasons):
            raise ValueError("no-commit safety reasons 必须属于 rejection reasons")
        if self.candidate_write_count != 0:
            raise ValueError("被拒绝 candidate 的 write count 必须为零")
        if not isinstance(self.prior_memory_safety_invalidated, bool):
            raise TypeError("prior_memory_safety_invalidated 必须为 bool")
        for name in (
            "accepted_update_count_before",
            "accepted_update_count_after",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value < 0:
                raise ValueError(f"no-commit {name} 必须是非负整数")
        if self.accepted_update_count_after != self.accepted_update_count_before:
            raise ValueError("no-commit 不得增加 accepted update count")
        state_changed = self.pre_state_digest != self.post_state_digest
        if self.prior_memory_safety_invalidated != state_changed:
            raise ValueError("no-commit prior Memory invalidation 与 state digest 不一致")
        if self.prior_memory_safety_invalidated and not self.safety_reasons:
            raise ValueError("Memory 安全失效必须绑定 safety reason")
        if self.version != ACTIVE_FRONT_STAGE2_VERSION:
            raise ValueError("no-commit receipt version 漂移")

    @property
    def digest(self) -> str:
        return _canonical_sha256(asdict(self))


@dataclass(frozen=True)
class ShadowActionGenerationReceipt:
    episode_id: str
    request_id: str
    candidate_digest: str
    commit_receipt_digest: str
    source_phase: PhaseId
    resume_phase: PhaseId
    action_generation_before: int
    action_generation_after: int
    source_phase_stability_reset: bool
    source_phase_stability_ticks: int
    generated_from_fresh_home_v2: bool
    stale_action_chunk_resumed: bool
    memory_only: bool
    contact_authorized: bool
    shadow_only: bool
    arm_command_count: int = 0
    gripper_close_count: int = 0
    test_read_count: int = 0
    version: str = ACTIVE_FRONT_STAGE2_VERSION

    def __post_init__(self) -> None:
        _sha256(self.candidate_digest, "candidate_digest")
        _sha256(self.commit_receipt_digest, "commit_receipt_digest")
        if self.action_generation_after != self.action_generation_before + 1:
            raise ValueError("shadow Action generation 必须严格递增一次")
        if self.source_phase is not self.resume_phase:
            raise ValueError("shadow replan 必须恢复原 source phase")
        if not self.source_phase_stability_reset or self.source_phase_stability_ticks != 0:
            raise ValueError("source phase stability 必须从零重新累计")
        if (
            not self.generated_from_fresh_home_v2
            or self.stale_action_chunk_resumed
            or not self.memory_only
            or self.contact_authorized
            or not self.shadow_only
        ):
            raise ValueError("shadow replan safety semantics 漂移")
        if any(
            value != 0
            for value in (self.arm_command_count, self.gripper_close_count, self.test_read_count)
        ):
            raise ValueError("Stage 2 shadow replan 禁止 actuator/test read")
        if self.version != ACTIVE_FRONT_STAGE2_VERSION:
            raise ValueError("shadow Action receipt version 漂移")

    @property
    def digest(self) -> str:
        value = asdict(self)
        value["source_phase"] = self.source_phase.value
        value["resume_phase"] = self.resume_phase.value
        return _canonical_sha256(value)


class ActiveFrontStage2MemoryOrchestrator:
    """单 Episode、单尝试、PRIMARY-only 的 pending→HOME→commit 事务机。"""

    def __init__(
        self,
        memory: ExplicitObjectStateMemory,
        *,
        config: ActiveFrontStage2Config | None = None,
        adapter: ActiveFrontStage2ProviderAdapter | None = None,
    ) -> None:
        if not isinstance(memory, ExplicitObjectStateMemory):
            raise TypeError("memory 必须是 ExplicitObjectStateMemory")
        self.config = config or ActiveFrontStage2Config()
        self.adapter = adapter or ActiveFrontStage2ProviderAdapter(self.config)
        self.memory = memory
        expected_memory = build_stage2_object_memory_config(self.adapter.provider_identity)
        if self.memory.config != expected_memory:
            raise ValueError("Stage 2 memory config 必须精确匹配 D049 active policy")
        self._candidate_verifier = ObjectCandidateWindowVerifier(self.memory.config)
        self._episode_id: str | None = None
        self._episode_generation = 0
        self._state = PendingActiveViewState.EMPTY
        self._request: ActiveFrontReobserveRequest | None = None
        self._reset_receipt: ActionHistoryResetReceipt | None = None
        self._baseline: PassiveBaselineEvidence | None = None
        self._frames: list[ActiveFrontStage2FrameEvidence] = []
        self._adaptations: list[ActiveFrontFrameAdaptation] = []
        self._candidate_decision: ObjectCandidateDecision | None = None
        self._candidate: PendingActiveViewCandidate | None = None
        self._home_frame_ids: list[str] = []
        self._home_frame_timestamps_s: list[float] = []
        self._home_frame_digests: list[str] = []
        self._return_home_timestamp_s: float | None = None
        self._last_home_timestamp_s: float | None = None
        self._recheck: ActiveFrontSourceRecheckEvidence | None = None
        self._commit_update: ObjectMemoryUpdate | None = None
        self._prepared_commit_receipt: DelayedActiveMemoryCommitReceipt | None = None
        self._commit_receipt: DelayedActiveMemoryCommitReceipt | None = None
        self._prepared_no_commit_receipt: DelayedActiveMemoryNoCommitReceipt | None = None
        self._no_commit_receipt: DelayedActiveMemoryNoCommitReceipt | None = None
        self._shadow_action_receipt: ShadowActionGenerationReceipt | None = None
        self._pre_memory_state_digest: str | None = None
        self._terminal_reasons: tuple[str, ...] = ()
        self._attempt_count = 0
        self._camera_lease_held = False
        self._commit_forbidden = False
        self._committed_candidate_digests: set[str] = set()

    @property
    def state(self) -> PendingActiveViewState:
        return self._state

    @property
    def pending_candidate(self) -> PendingActiveViewCandidate | None:
        return self._candidate

    @property
    def terminal_reasons(self) -> tuple[str, ...]:
        return self._terminal_reasons

    @property
    def home_observation_sequence_ids(self) -> tuple[str, ...]:
        return tuple(self._home_frame_ids)

    @property
    def home_observation_timestamps_s(self) -> tuple[float, ...]:
        return tuple(self._home_frame_timestamps_s)

    @property
    def home_frame_digests(self) -> tuple[str, ...]:
        return tuple(self._home_frame_digests)

    @property
    def request_active(self) -> bool:
        return self._request is not None

    @property
    def camera_lease_held(self) -> bool:
        return self._camera_lease_held

    @property
    def attempt_count(self) -> int:
        return self._attempt_count

    @property
    def memory_write_count(self) -> int:
        return 0 if self._commit_receipt is None else self._commit_receipt.memory_write_count

    @property
    def commit_receipt(self) -> DelayedActiveMemoryCommitReceipt | None:
        return self._commit_receipt

    @property
    def prepared_commit_receipt(self) -> DelayedActiveMemoryCommitReceipt | None:
        return self._prepared_commit_receipt

    @property
    def no_commit_receipt(self) -> DelayedActiveMemoryNoCommitReceipt | None:
        return self._no_commit_receipt

    @property
    def prepared_no_commit_receipt(self) -> DelayedActiveMemoryNoCommitReceipt | None:
        return self._prepared_no_commit_receipt

    @property
    def shadow_action_receipt(self) -> ShadowActionGenerationReceipt | None:
        return self._shadow_action_receipt

    @property
    def commit_update(self) -> ObjectMemoryUpdate | None:
        return self._commit_update

    def reset_episode(
        self,
        episode_id: str,
        *,
        episode_generation: int,
        timestamp_s: float = 0.0,
    ) -> ObjectState:
        _identity(episode_id, "episode_id")
        if episode_id == self._episode_id:
            raise ValueError("不能用相同 episode_id 清空 Stage 2 latch")
        if episode_generation != self._episode_generation + 1:
            raise ValueError("episode_generation 必须严格递增一次")
        timestamp = _timestamp(timestamp_s, "episode reset timestamp")
        state = self.memory.reset(episode_id, timestamp_s=timestamp)
        self._candidate_verifier.reset(episode_id)
        self._episode_id = episode_id
        self._episode_generation = episode_generation
        self._state = PendingActiveViewState.RESET_CLEARED
        self._request = None
        self._reset_receipt = None
        self._baseline = None
        self._frames = []
        self._adaptations = []
        self._candidate_decision = None
        self._candidate = None
        self._home_frame_ids = []
        self._home_frame_timestamps_s = []
        self._home_frame_digests = []
        self._return_home_timestamp_s = None
        self._last_home_timestamp_s = None
        self._recheck = None
        self._commit_update = None
        self._prepared_commit_receipt = None
        self._commit_receipt = None
        self._prepared_no_commit_receipt = None
        self._no_commit_receipt = None
        self._shadow_action_receipt = None
        self._pre_memory_state_digest = None
        self._terminal_reasons = ()
        self._attempt_count = 0
        self._camera_lease_held = False
        self._commit_forbidden = False
        self._committed_candidate_digests = set()
        return state

    def begin_collection(
        self,
        request: ActiveFrontReobserveRequest,
        *,
        reset_receipt: ActionHistoryResetReceipt,
        baseline: PassiveBaselineEvidence,
    ) -> None:
        if self._state is not PendingActiveViewState.RESET_CLEARED:
            raise RuntimeError("Stage 2 collection 必须从 RESET_CLEARED 开始")
        if not self.config.enabled or not self.config.memory_write_allowed:
            raise RuntimeError("Stage 2 PRIMARY Memory 默认关闭")
        if not isinstance(request, ActiveFrontReobserveRequest):
            raise TypeError("request 类型错误")
        if not isinstance(baseline, PassiveBaselineEvidence):
            raise TypeError("baseline 类型错误")
        if (
            request.episode_id != self._episode_id
            or request.episode_generation != self._episode_generation
            or baseline.episode_id != self._episode_id
            or baseline.episode_generation != self._episode_generation
            or baseline.request_id != request.request_id
        ):
            raise ValueError("Stage 2 request/baseline Episode identity 漂移")
        if request.selected_primitive_id != ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID:
            raise ValueError("Stage 2A request 必须固定 PRIMARY primitive")
        if request.source_phase not in ALLOWED_ACTIVE_SOURCE_PHASES:
            raise ValueError("Stage 2A request source phase 不允许")
        if request.resume_phase is not request.source_phase:
            raise ValueError("Stage 2A 必须恢复原 source phase")
        if baseline.timestamp_s > request.trigger_timestamp_s + 1e-12:
            raise ValueError("baseline 必须在 trigger timestamp 之前冻结")
        memory_state = self.memory.state
        if memory_state.state_timestamp_s > request.trigger_timestamp_s + 1e-12:
            raise ValueError("Object Memory state timestamp 晚于 trigger")
        if memory_state.mode is ObjectMemoryMode.INVALID:
            raise ValueError("Stage 2 不得从 safety-invalid Object Memory 启动相机移动")
        actual_memory_age_s = None
        if memory_state.last_observed_timestamp_s is not None:
            actual_memory_age_s = (
                request.trigger_timestamp_s - memory_state.last_observed_timestamp_s
            )
            if actual_memory_age_s < -1e-12:
                raise ValueError("Object Memory observation timestamp 晚于 trigger")
        actual_navigation_available = bool(
            memory_state.mode is ObjectMemoryMode.FREE_STATIC
            and memory_state.valid
            and memory_state.position_base_m is not None
            and actual_memory_age_s is not None
            and actual_memory_age_s <= self.config.max_memory_unobserved_age_s + 1e-12
            and memory_state.max_position_std_m is not None
            and memory_state.max_position_std_m
            <= self.config.max_position_std_m + 1e-12
        )
        age_matches = (
            baseline.object_memory_age_s is None
            and actual_memory_age_s is None
        ) or (
            baseline.object_memory_age_s is not None
            and actual_memory_age_s is not None
            and math.isclose(
                baseline.object_memory_age_s,
                actual_memory_age_s,
                rel_tol=0.0,
                abs_tol=1e-12,
            )
        )
        if (
            baseline.object_memory_navigation_state_available
            != actual_navigation_available
            or not age_matches
            or baseline.object_memory_source_identity
            != memory_state.source_model_identity
        ):
            raise ValueError("baseline Object Memory snapshot 与实际 state 不一致")
        if (
            baseline.wrist_object_measurement_usable
            or baseline.home_front_object_measurement_usable
            or actual_navigation_available
        ):
            raise ValueError("Stage 2 trigger baseline 必须是 dual-direct+Memory 全部不可用")
        reset_receipt.validate_for(request)
        if self._attempt_count >= self.config.maximum_attempts_per_episode:
            raise RuntimeError("Stage 2 attempt budget 已耗尽")

        self._attempt_count += 1
        self._request = request
        self._reset_receipt = reset_receipt
        self._baseline = baseline
        self._pre_memory_state_digest = _canonical_sha256(_state_dict(self.memory.state))
        self._camera_lease_held = True
        self._commit_forbidden = False
        self._state = PendingActiveViewState.COLLECTING

    def _record_failure(self, *reasons: str) -> None:
        self._terminal_reasons = tuple(dict.fromkeys((*self._terminal_reasons, *reasons)))
        self._commit_forbidden = True

    def _require_return_home_without_commit(self, *reasons: str) -> None:
        """保留 camera lease，强制失败路径继续执行 HOME recovery。"""

        self._record_failure(*reasons)
        if self._state in {
            PendingActiveViewState.RETURNING_HOME,
            PendingActiveViewState.RETURNING_HOME_NO_COMMIT,
        }:
            self._state = PendingActiveViewState.RETURNING_HOME_NO_COMMIT
        else:
            self._state = PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT

    def defer_candidate_for_pure_logic_replay(self, *, candidate_digest: str) -> None:
        """冻结已构造 candidate，但把任何 Memory/Action 副作用延迟到隔离重放。"""

        if (
            self._state is not PendingActiveViewState.VERIFIED_PENDING
            or self._candidate is None
        ):
            raise RuntimeError("只有 VERIFIED_PENDING candidate 可以延迟到纯逻辑重放")
        _sha256(candidate_digest, "candidate_digest")
        if candidate_digest != self._candidate.digest:
            raise RuntimeError("纯逻辑重放 candidate digest 漂移")
        if (
            self.memory_write_count != 0
            or self._commit_receipt is not None
            or self._shadow_action_receipt is not None
        ):
            raise RuntimeError("纯逻辑重放冻结前禁止 Memory write/Action generation")
        self._require_return_home_without_commit(
            "selection_capture_only_branching_deferred"
        )

    def _finish_failed_safe_hold(self, *reasons: str) -> None:
        """只在四帧 HOME barrier 已成立后释放 camera lease。"""

        self._record_failure(*reasons)
        self._camera_lease_held = False
        self._state = PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD

    def _expire_if_needed(self, timestamp_s: float) -> bool:
        if self._candidate is None:
            return False
        age = timestamp_s - self._candidate.created_timestamp_s
        if age < -1e-12:
            self._record_failure("timestamp_before_candidate")
            return True
        if age > self._candidate.maximum_age_s + 1e-12:
            self._record_failure("pending_candidate_expired")
            return True
        return False

    def observe_collect_frame(
        self,
        frame: ActiveFrontStage2FrameEvidence | ActiveFrontModelInput,
        *,
        safety: ObjectMemorySafetyContext,
    ) -> ActiveFrontFrameAdaptation:
        if self._state is not PendingActiveViewState.COLLECTING or self._request is None:
            raise RuntimeError("只有 COLLECTING 可以接收 provider frame")
        adaptation = self.adapter.adapt(frame, safety=safety)
        if safety.invalidation_reasons:
            self.memory.invalidate_for_safety(
                episode_id=self._request.episode_id,
                timestamp_s=frame.control_timestamp_s,
                reasons=safety.invalidation_reasons,
            )
        if not isinstance(frame, ActiveFrontStage2FrameEvidence):
            self._require_return_home_without_commit(*adaptation.rejection_reasons)
            return adaptation
        if (
            frame.episode_id != self._request.episode_id
            or frame.episode_generation != self._request.episode_generation
            or frame.request_id != self._request.request_id
            or frame.source_phase is not self._request.source_phase
        ):
            self._require_return_home_without_commit(
                "collect_frame_request_or_source_identity_mismatch"
            )
            return ActiveFrontFrameAdaptation(
                observation_sequence_id=frame.observation_sequence_id,
                frame_digest=frame.frame_digest,
                provider_identity_sha256=frame.provider_identity.sha256,
                measurement=None,
                write_score=frame.write_score,
                eligible=False,
                rejection_reasons=("collect_frame_request_or_source_identity_mismatch",),
            )
        if frame.observation_sequence_id in {
            value.observation_sequence_id for value in self._frames
        }:
            self._require_return_home_without_commit("duplicate_collect_frame")
            return ActiveFrontFrameAdaptation(
                observation_sequence_id=frame.observation_sequence_id,
                frame_digest=frame.frame_digest,
                provider_identity_sha256=frame.provider_identity.sha256,
                measurement=None,
                write_score=frame.write_score,
                eligible=False,
                rejection_reasons=("duplicate_collect_frame",),
            )
        replay_reasons: list[str] = []
        if frame.model_input_digest in {
            value.model_input_digest for value in self._frames
        }:
            replay_reasons.append("duplicate_model_input_digest")
        if frame.provider_output_digest in {
            value.provider_output_digest for value in self._frames
        }:
            replay_reasons.append("duplicate_provider_output_digest")
        if replay_reasons:
            rejected = tuple(replay_reasons)
            self._require_return_home_without_commit(*rejected)
            return ActiveFrontFrameAdaptation(
                observation_sequence_id=frame.observation_sequence_id,
                frame_digest=frame.frame_digest,
                provider_identity_sha256=frame.provider_identity.sha256,
                measurement=None,
                write_score=frame.write_score,
                eligible=False,
                rejection_reasons=rejected,
            )
        if not adaptation.eligible or adaptation.measurement is None:
            self._require_return_home_without_commit(*adaptation.rejection_reasons)
            return adaptation
        if (
            self._frames
            and frame.control_timestamp_s
            <= self._frames[-1].control_timestamp_s + 1e-12
        ):
            reason = "collect_frame_timestamp_not_increasing"
            self._require_return_home_without_commit(reason)
            return ActiveFrontFrameAdaptation(
                observation_sequence_id=frame.observation_sequence_id,
                frame_digest=frame.frame_digest,
                provider_identity_sha256=frame.provider_identity.sha256,
                measurement=None,
                write_score=frame.write_evidence.score,
                eligible=False,
                rejection_reasons=(reason,),
            )
        try:
            decision = self._candidate_verifier.observe(
                adaptation.measurement,
                episode_id=self._request.episode_id,
                safety=safety,
            )
        except ValueError:
            reason = "candidate_verifier_contract_error"
            self._require_return_home_without_commit(reason)
            return ActiveFrontFrameAdaptation(
                observation_sequence_id=frame.observation_sequence_id,
                frame_digest=frame.frame_digest,
                provider_identity_sha256=frame.provider_identity.sha256,
                measurement=None,
                write_score=frame.write_evidence.score,
                eligible=False,
                rejection_reasons=(reason,),
            )
        self._frames.append(frame)
        self._adaptations.append(adaptation)
        self._candidate_decision = decision
        if len(self._frames) < self.config.min_candidate_frames:
            return adaptation
        if len(self._frames) > self.config.min_candidate_frames:
            self._require_return_home_without_commit("too_many_collect_frames")
            return adaptation

        positions = np.asarray(
            [value.measurement.position_base_m for value in self._adaptations],
            dtype=np.float64,
        )
        pairwise = positions[:, None, :] - positions[None, :, :]
        spread = float(np.linalg.norm(pairwise, axis=-1).max())
        scores = tuple(float(value.write_score) for value in self._frames)
        minimum_score = min(scores)
        reasons: list[str] = []
        if not decision.verified:
            reasons.extend(decision.rejection_reasons)
        if spread > self.config.max_candidate_position_spread_m + 1e-12:
            reasons.append("candidate_position_inconsistent")
        if any(score + 1e-12 < self.adapter.provider_identity.write_threshold for score in scores):
            reasons.append("candidate_score_below_primary_threshold")

        if self._baseline is None:
            raise RuntimeError("COLLECTING 缺少 frozen baseline")
        baseline_reasons = self._baseline.gain_unavailable_reasons(
            self.adapter.provider_identity
        )
        reasons.extend(baseline_reasons)
        information_gain = None
        if not baseline_reasons and self._baseline.home_front_write_score is not None:
            information_gain = minimum_score - self._baseline.home_front_write_score
            if not self.config.information_gain_is_sufficient(information_gain):
                reasons.append("information_gain_below_threshold")

        final_measurement = adaptation.measurement
        innovation = None
        if self.memory.state.position_base_m is not None:
            innovation = float(
                np.linalg.norm(
                    np.asarray(final_measurement.position_base_m, dtype=np.float64)
                    - np.asarray(self.memory.state.position_base_m, dtype=np.float64)
                )
            )
            if innovation > self.config.max_innovation_m + 1e-12:
                reasons.append("measurement_conflict")
        rejected = tuple(dict.fromkeys(reasons))
        if decision.window_id is None:
            window_id = f"{self._request.request_id}:rejected-window"
        else:
            window_id = decision.window_id
        self._candidate = PendingActiveViewCandidate(
            episode_id=self._request.episode_id,
            episode_generation=self._request.episode_generation,
            request_id=self._request.request_id,
            window_id=window_id,
            source_phase=self._request.source_phase,
            resume_phase=self._request.resume_phase,
            frame_sequence_ids=tuple(value.observation_sequence_id for value in self._frames),
            frame_timestamps_s=tuple(value.control_timestamp_s for value in self._frames),
            frame_digests=tuple(value.frame_digest for value in self._frames),
            model_input_digests=tuple(value.model_input_digest for value in self._frames),
            provider_output_digests=tuple(
                value.provider_output_digest for value in self._frames
            ),
            actual_pose_sha256s=tuple(value.actual_pose_sha256 for value in self._frames),
            write_scores=scores,
            score_components=tuple(value.score_components for value in self._frames),
            minimum_candidate_score=minimum_score,
            information_gain=information_gain,
            position_spread_m=spread,
            innovation_m=innovation,
            final_measurement=final_measurement,
            final_measurement_digest=_canonical_sha256(
                _measurement_dict(final_measurement)
            ),
            provider_identity=self.adapter.provider_identity,
            baseline_digest=self._baseline.digest,
            created_timestamp_s=final_measurement.timestamp_s,
            maximum_age_s=self.config.max_pending_age_s,
            commit_eligible=not rejected,
            rejection_reasons=rejected,
        )
        if rejected:
            self._require_return_home_without_commit(*rejected)
        else:
            self._state = PendingActiveViewState.VERIFIED_PENDING
        return adaptation

    def mark_returning_home(
        self,
        *,
        timestamp_s: float,
        candidate_digest: str | None,
    ) -> None:
        if self._state not in {
            PendingActiveViewState.VERIFIED_PENDING,
            PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT,
        }:
            raise RuntimeError("只有 candidate 结束后可以进入 RETURNING_HOME")
        if self._candidate is None:
            if candidate_digest is not None:
                _sha256(candidate_digest, "candidate_digest")
                self._record_failure("unexpected_candidate_digest")
        else:
            if candidate_digest is None:
                self._record_failure("pending_candidate_digest_missing")
            else:
                _sha256(candidate_digest, "candidate_digest")
                if candidate_digest != self._candidate.digest:
                    self._record_failure("pending_candidate_digest_mismatch")
        timestamp = _timestamp(timestamp_s, "return-home timestamp")
        last_collect_timestamp = (
            self._frames[-1].control_timestamp_s
            if self._frames
            else self._request.trigger_timestamp_s if self._request is not None else 0.0
        )
        if timestamp + 1e-12 < last_collect_timestamp:
            self._record_failure("return_home_before_last_collect_frame")
        self._expire_if_needed(timestamp)
        self._return_home_timestamp_s = timestamp
        self._state = (
            PendingActiveViewState.RETURNING_HOME_NO_COMMIT
            if self._commit_forbidden
            else PendingActiveViewState.RETURNING_HOME
        )

    def accept_home_v2_barrier_frame(
        self,
        frame: HomeV2BarrierFrame,
        *,
        timestamp_s: float,
        safety: ActiveFrontSafetyEvidence | None = None,
    ) -> None:
        if self._state not in {
            PendingActiveViewState.RETURNING_HOME,
            PendingActiveViewState.RETURNING_HOME_NO_COMMIT,
        }:
            raise RuntimeError("HOME barrier frame 只允许在 RETURNING_HOME")
        timestamp = _timestamp(timestamp_s, "HOME barrier timestamp")
        if (
            self._return_home_timestamp_s is None
            or timestamp + 1e-12 < self._return_home_timestamp_s
        ):
            self._require_return_home_without_commit(
                "home_frame_before_return_home_command"
            )
            return
        if self._last_home_timestamp_s is not None and timestamp <= self._last_home_timestamp_s + 1e-12:
            self._require_return_home_without_commit(
                "home_frame_timestamp_not_increasing"
            )
            return
        resolved_safety = safety or ActiveFrontSafetyEvidence()
        failure = resolved_safety.failure()
        if failure is not None:
            mapping = {
                "arm_hold_violation": "controller_tracking_invalid",
                "tcp_hold_violation": "controller_tracking_invalid",
                "gripper_hold_violation": "gripper_not_open",
                "contact_detected": "object_contact_detected",
                "active_window_closed": "pregrasp_window_closed",
            }
            invalidation_reason = mapping[failure.value]
            if self._request is None:
                raise RuntimeError("HOME safety failure 缺少 request")
            self.memory.invalidate_for_safety(
                episode_id=self._request.episode_id,
                timestamp_s=timestamp,
                reasons=(invalidation_reason,),
            )
            self._require_return_home_without_commit(failure.value)
            return
        if not frame.valid():
            self._require_return_home_without_commit("home_frame_invalid")
            return
        if frame.observation_sequence_id in self._home_frame_ids:
            self._require_return_home_without_commit("home_frame_duplicate")
            return
        self._expire_if_needed(timestamp)
        self._last_home_timestamp_s = timestamp
        self._home_frame_ids.append(frame.observation_sequence_id)
        self._home_frame_timestamps_s.append(timestamp)
        self._home_frame_digests.append(
            _canonical_sha256(
                {
                    "frame": asdict(frame),
                    "timestamp_s": timestamp,
                }
            )
        )
        if len(self._home_frame_ids) == self.config.home_v2_barrier_frames:
            if self._commit_forbidden:
                self._finish_failed_safe_hold()
            else:
                self._camera_lease_held = False
                self._state = PendingActiveViewState.HOME_BARRIER_PASSED

    def recheck_source(
        self,
        evidence: ActiveFrontSourceRecheckEvidence,
        *,
        safety: ActiveFrontSafetyEvidence | None = None,
    ) -> bool:
        if self._state is not PendingActiveViewState.HOME_BARRIER_PASSED:
            raise RuntimeError("source recheck 要求完整 HOME barrier")
        if self._request is None or self._candidate is None:
            raise RuntimeError("source recheck 缺少 request/candidate")
        reasons: list[str] = []
        if (
            evidence.episode_id != self._request.episode_id
            or evidence.episode_generation != self._request.episode_generation
            or evidence.request_id != self._request.request_id
            or evidence.candidate_digest != self._candidate.digest
        ):
            reasons.append("source_recheck_identity_mismatch")
        if evidence.source_phase is not self._request.resume_phase:
            reasons.append("source_phase_changed")
        if not evidence.camera_at_home:
            reasons.append("camera_not_home")
        if not evidence.source_invariants_passed:
            reasons.append("source_invariant_failed")
        if not evidence.active_window_open:
            reasons.append("active_window_closed")
        resolved_safety = safety or ActiveFrontSafetyEvidence()
        failure = resolved_safety.failure()
        if failure is not None:
            reasons.append(failure.value)
        if evidence.qualified_direct_wrist_measurement_usable:
            reasons.append("superseded_by_fresh_direct_evidence")
        if (
            self._last_home_timestamp_s is None
            or evidence.timestamp_s + 1e-12 < self._last_home_timestamp_s
        ):
            reasons.append("source_recheck_before_last_home_frame")
        if self._expire_if_needed(evidence.timestamp_s):
            self._finish_failed_safe_hold()
            return False
        independent_safety_reasons: list[str] = []
        if not evidence.active_window_open:
            independent_safety_reasons.append("pregrasp_window_closed")
        if failure is not None:
            mapping = {
                "arm_hold_violation": "controller_tracking_invalid",
                "tcp_hold_violation": "controller_tracking_invalid",
                "gripper_hold_violation": "gripper_not_open",
                "contact_detected": "object_contact_detected",
                "active_window_closed": "pregrasp_window_closed",
            }
            independent_safety_reasons.append(mapping[failure.value])
        independent_safety_reasons = list(dict.fromkeys(independent_safety_reasons))
        pre_invalidation_state = self.memory.state
        pre_invalidation_digest = _canonical_sha256(
            _state_dict(pre_invalidation_state)
        )
        if self._pre_memory_state_digest != pre_invalidation_digest:
            reasons.append("memory_state_changed_while_pending")
        if independent_safety_reasons:
            invalidated_state = self.memory.invalidate_for_safety(
                episode_id=self._request.episode_id,
                timestamp_s=evidence.timestamp_s,
                reasons=tuple(independent_safety_reasons),
            )
            if (
                invalidated_state.mode is not ObjectMemoryMode.INVALID
                or invalidated_state.valid
                or invalidated_state.observable_now
                or invalidated_state.accepted_update_count
                != pre_invalidation_state.accepted_update_count
                or invalidated_state.position_base_m
                != pre_invalidation_state.position_base_m
                or invalidated_state.source_camera
                != pre_invalidation_state.source_camera
                or invalidated_state.source_model_identity
                != pre_invalidation_state.source_model_identity
                or not set(independent_safety_reasons).issubset(
                    invalidated_state.invalid_reasons
                )
            ):
                raise RuntimeError("source recheck safety invalidation 不是单调失效")
        if reasons:
            self._finish_failed_safe_hold(*reasons)
            return False
        self._recheck = evidence
        self._state = PendingActiveViewState.SOURCE_RECHECK_PASSED
        return True

    def commit(
        self,
        *,
        candidate_digest: str,
        commit_timestamp_s: float,
        safety: ObjectMemorySafetyContext,
    ) -> DelayedActiveMemoryCommitReceipt:
        if self._state is PendingActiveViewState.COMMITTED:
            self._record_failure("duplicate_delayed_commit")
            self._state = PendingActiveViewState.SHADOW_REPLAN_FAILED_SAFE_HOLD
            raise RuntimeError("duplicate delayed commit 被拒绝")
        if self._state is not PendingActiveViewState.SOURCE_RECHECK_PASSED:
            raise RuntimeError("commit 要求 SOURCE_RECHECK_PASSED")
        if (
            self._request is None
            or self._candidate is None
            or self._candidate_decision is None
            or self._recheck is None
        ):
            raise RuntimeError("commit 缺少冻结事务证据")
        _sha256(candidate_digest, "candidate_digest")
        if candidate_digest != self._candidate.digest:
            self._finish_failed_safe_hold("pending_candidate_digest_mismatch")
            raise RuntimeError("pending candidate digest 漂移")
        if candidate_digest in self._committed_candidate_digests:
            self._record_failure("duplicate_delayed_commit")
            self._state = PendingActiveViewState.SHADOW_REPLAN_FAILED_SAFE_HOLD
            raise RuntimeError("duplicate delayed commit 被拒绝")
        commit_timestamp = _timestamp(commit_timestamp_s, "commit timestamp")
        if commit_timestamp + 1e-12 < self._recheck.timestamp_s:
            self._finish_failed_safe_hold("commit_before_source_recheck")
            raise RuntimeError("commit timestamp 早于 source recheck")
        if self._expire_if_needed(commit_timestamp):
            self._finish_failed_safe_hold()
            raise RuntimeError("pending candidate 已过期")
        pre_state = self.memory.state
        pre_digest = _canonical_sha256(_state_dict(pre_state))
        if pre_digest != self._pre_memory_state_digest:
            self._finish_failed_safe_hold("memory_state_changed_while_pending")
            raise RuntimeError("pending 期间 Object Memory 已变化")

        update = self.memory.preview_delayed_candidate(
            self._candidate_decision,
            episode_id=self._request.episode_id,
            safety=safety,
            commit_timestamp_s=commit_timestamp,
            max_pending_age_s=self.config.max_pending_age_s,
        )
        if not update.measurement_accepted:
            state_changed = update.state != pre_state
            safety_reasons = safety.invalidation_reasons
            if state_changed and not safety_reasons:
                raise RuntimeError("non-safety rejection 不得改变 preview Memory state")
            if state_changed:
                if (
                    update.state.mode is not ObjectMemoryMode.INVALID
                    or update.state.valid
                    or update.state.observable_now
                    or update.state.accepted_update_count
                    != pre_state.accepted_update_count
                    or update.state.position_base_m != pre_state.position_base_m
                    or update.state.source_camera != pre_state.source_camera
                    or update.state.source_model_identity
                    != pre_state.source_model_identity
                    or not set(safety_reasons).issubset(update.rejection_reasons)
                ):
                    raise RuntimeError("safety rejection preview 不是单调 Memory invalidation")
            no_commit_receipt = DelayedActiveMemoryNoCommitReceipt(
                episode_id=self._request.episode_id,
                episode_generation=self._request.episode_generation,
                request_id=self._request.request_id,
                candidate_digest=candidate_digest,
                status="rejected-no-candidate-write",
                commit_timestamp_s=commit_timestamp,
                pre_state_digest=pre_digest,
                post_state_digest=_canonical_sha256(_state_dict(update.state)),
                rejection_reasons=update.rejection_reasons,
                safety_reasons=safety_reasons,
                candidate_write_count=0,
                prior_memory_safety_invalidated=state_changed,
                accepted_update_count_before=pre_state.accepted_update_count,
                accepted_update_count_after=update.state.accepted_update_count,
            )
            self._prepared_no_commit_receipt = no_commit_receipt
            try:
                if state_changed:
                    self.memory.apply_delayed_candidate_preview(
                        update,
                        self._candidate_decision,
                        episode_id=self._request.episode_id,
                        safety=safety,
                        commit_timestamp_s=commit_timestamp,
                        max_pending_age_s=self.config.max_pending_age_s,
                        expected_previous_state=pre_state,
                    )
            except Exception:
                self._prepared_no_commit_receipt = None
                raise
            self._commit_update = update
            self._no_commit_receipt = no_commit_receipt
            self._finish_failed_safe_hold(*update.rejection_reasons)
            raise RuntimeError("delayed Object Memory commit 被拒绝")
        state = update.state
        navigation = resolve_object_state(
            update,
            requirement=ObjectStateRequirement.NAVIGATION,
        )
        contact = resolve_object_state(
            update,
            requirement=ObjectStateRequirement.CONTACT_READY,
        )
        if (
            state.mode is not ObjectMemoryMode.FREE_STATIC
            or state.last_observed_timestamp_s
            != self._candidate.final_measurement.timestamp_s
            or state.state_timestamp_s != commit_timestamp
            or state.observable_now
            or not navigation.available
            or not navigation.memory_only
            or navigation.contact_authorized
            or contact.available
            or contact.contact_authorized
        ):
            raise RuntimeError("delayed commit 后 Memory 时间/权限语义漂移")
        post_digest = _canonical_sha256(_state_dict(state))
        if len(self._home_frame_ids) != self.config.home_v2_barrier_frames:
            raise RuntimeError("delayed commit 缺少完整 HOME barrier receipts")
        receipt = DelayedActiveMemoryCommitReceipt(
            episode_id=self._request.episode_id,
            episode_generation=self._request.episode_generation,
            request_id=self._request.request_id,
            candidate_digest=candidate_digest,
            status="committed-primary-memory-only",
            commit_timestamp_s=commit_timestamp,
            pre_state_digest=pre_digest,
            post_state_digest=post_digest,
            last_observed_timestamp_s=state.last_observed_timestamp_s,
            state_timestamp_s=state.state_timestamp_s,
            observable_now=state.observable_now,
            memory_only=navigation.memory_only,
            contact_authorized=contact.contact_authorized,
            memory_write_count=1,
            provider_identity=self._candidate.provider_identity,
            provider_identity_sha256=self._candidate.provider_identity.sha256,
            final_measurement_digest=self._candidate.final_measurement_digest,
            source_recheck_wrist_evidence_identity_sha256=(
                self._recheck.qualified_direct_wrist_evidence_identity_sha256
            ),
            home_observation_sequence_ids=tuple(self._home_frame_ids),
            home_observation_timestamps_s=tuple(self._home_frame_timestamps_s),
            home_frame_digests=tuple(self._home_frame_digests),
        )
        self._prepared_commit_receipt = receipt
        try:
            self.memory.apply_delayed_candidate_preview(
                update,
                self._candidate_decision,
                episode_id=self._request.episode_id,
                safety=safety,
                commit_timestamp_s=commit_timestamp,
                max_pending_age_s=self.config.max_pending_age_s,
                expected_previous_state=pre_state,
            )
        except Exception:
            self._prepared_commit_receipt = None
            raise
        self._commit_update = update
        self._commit_receipt = receipt
        self._committed_candidate_digests.add(candidate_digest)
        self._state = PendingActiveViewState.COMMITTED
        return receipt

    def create_shadow_action_generation(
        self,
        resume_receipt: ActionHistoryResumeReceipt,
        *,
        source_phase: PhaseId,
        source_phase_stability_reset: bool,
        source_phase_stability_ticks: int,
    ) -> ShadowActionGenerationReceipt:
        if self._state is not PendingActiveViewState.COMMITTED:
            raise RuntimeError("shadow replan 要求已完成 Memory commit")
        if self._shadow_action_receipt is not None:
            raise RuntimeError("duplicate shadow Action generation 被拒绝")
        if (
            self._request is None
            or self._reset_receipt is None
            or self._candidate is None
            or self._commit_receipt is None
            or self._commit_update is None
        ):
            raise RuntimeError("shadow replan 缺少事务证据")
        valid_resume = bool(
            resume_receipt.episode_id == self._request.episode_id
            and resume_receipt.request_id == self._request.request_id
            and resume_receipt.generation == self._reset_receipt.generation_after + 1
            and resume_receipt.home_observation_sequence_ids
            == tuple(self._home_frame_ids)
            and resume_receipt.generated_from_fresh_home_v2
            and not resume_receipt.stale_action_chunk_resumed
            and source_phase is self._request.resume_phase
            and source_phase_stability_reset
            and source_phase_stability_ticks == 0
        )
        if not valid_resume:
            self._record_failure("invalid_shadow_action_generation")
            self._state = PendingActiveViewState.SHADOW_REPLAN_FAILED_SAFE_HOLD
            raise ValueError("stale/invalid shadow Action generation 被拒绝")
        navigation = resolve_object_state(
            self._commit_update,
            requirement=ObjectStateRequirement.NAVIGATION,
        )
        contact = resolve_object_state(
            self._commit_update,
            requirement=ObjectStateRequirement.CONTACT_READY,
        )
        receipt = ShadowActionGenerationReceipt(
            episode_id=self._request.episode_id,
            request_id=self._request.request_id,
            candidate_digest=self._candidate.digest,
            commit_receipt_digest=self._commit_receipt.digest,
            source_phase=self._request.source_phase,
            resume_phase=self._request.resume_phase,
            action_generation_before=self._reset_receipt.generation_after,
            action_generation_after=resume_receipt.generation,
            source_phase_stability_reset=source_phase_stability_reset,
            source_phase_stability_ticks=source_phase_stability_ticks,
            generated_from_fresh_home_v2=resume_receipt.generated_from_fresh_home_v2,
            stale_action_chunk_resumed=resume_receipt.stale_action_chunk_resumed,
            memory_only=navigation.memory_only,
            contact_authorized=contact.contact_authorized,
            shadow_only=True,
        )
        self._shadow_action_receipt = receipt
        return receipt


__all__ = [
    "ACTIVE_FRONT_HOME_PRIMITIVE_ID",
    "ACTIVE_FRONT_INFORMATION_GAIN_CANDIDATES",
    "ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID",
    "ACTIVE_FRONT_PROHIBITED_MEMORY_WRITE_PRIMITIVE_IDS",
    "ACTIVE_FRONT_QUALIFIED_SHADOW_PRIMITIVE_IDS",
    "ACTIVE_FRONT_SCORE_SEMANTICS",
    "ACTIVE_FRONT_STAGE2_EXECUTION_MODE",
    "ACTIVE_FRONT_STAGE2_PROVIDER_ADAPTER_VERSION",
    "ACTIVE_FRONT_STAGE2_PROVIDER_IDENTITY_VERSION",
    "ACTIVE_FRONT_STAGE2_VERSION",
    "ActiveFrontFrameAdaptation",
    "ActiveFrontScoreComponents",
    "ActiveFrontSourceRecheckEvidence",
    "ActiveFrontStage2Config",
    "ActiveFrontStage2FrameEvidence",
    "ActiveFrontStage2MemoryOrchestrator",
    "ActiveFrontStage2ProviderAdapter",
    "ActiveFrontStage2ProviderIdentity",
    "DelayedActiveMemoryCommitReceipt",
    "DelayedActiveMemoryNoCommitReceipt",
    "PassiveBaselineEvidence",
    "PassiveHomeScoreEvidence",
    "PendingActiveViewCandidate",
    "PendingActiveViewState",
    "ShadowActionGenerationReceipt",
    "build_stage2_object_memory_config",
    "d049_primary_provider_identity",
]
