"""确定性 Subtask Executive、恢复路径和逐 Tick transition ledger。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from typing import Iterable

from robot_vla.executive.contracts import (
    CompiledTaskPlan,
    ControllerOwner,
    CriticalAction,
    ExecutiveDecision,
    ExecutiveSnapshot,
    ExecutiveState,
    ExecutiveStatus,
    PhaseId,
    PhaseSpec,
    SubtaskId,
    TransitionLedgerEntry,
    TransitionOutcome,
    TransitionReason,
    TRANSITION_LEDGER_VERSION,
)


@dataclass(frozen=True)
class ExecutiveConfig:
    """P0 默认只做 shadow；正式 actuation 必须由后续 Gate 显式开启。"""

    shadow_only: bool = True
    max_recovery_attempts: int = 1
    require_state_estimate: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.shadow_only, bool):
            raise TypeError("shadow_only 必须为 bool")
        if not isinstance(self.require_state_estimate, bool):
            raise TypeError("require_state_estimate 必须为 bool")
        if (
            not isinstance(self.max_recovery_attempts, int)
            or isinstance(self.max_recovery_attempts, bool)
            or self.max_recovery_attempts < 0
        ):
            raise ValueError("max_recovery_attempts 必须为非负整数")


class TransitionLedger:
    """内存中的不可变条目序列；调用方可以写出 JSONL 和冻结 digest。"""

    def __init__(self, plan_id: str) -> None:
        self.plan_id = plan_id
        self._entries: list[TransitionLedgerEntry] = []

    @property
    def entries(self) -> tuple[TransitionLedgerEntry, ...]:
        return tuple(self._entries)

    def append(self, entry: TransitionLedgerEntry) -> None:
        if entry.plan_id != self.plan_id:
            raise ValueError("ledger entry plan_id 漂移")
        if entry.index != len(self._entries):
            raise ValueError("ledger entry index 必须连续")
        if self._entries and entry.snapshot.tick <= self._entries[-1].snapshot.tick:
            raise ValueError("ledger snapshot tick 必须严格递增")
        self._entries.append(entry)

    def to_jsonl(self) -> str:
        if not self._entries:
            return ""
        return "\n".join(item.canonical_json() for item in self._entries) + "\n"

    def sha256(self) -> str:
        return hashlib.sha256(self.to_jsonl().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class ExecutiveReplayResult:
    decisions: tuple[ExecutiveDecision, ...]
    final_state: ExecutiveState
    ledger_entries: tuple[TransitionLedgerEntry, ...]
    ledger_sha256: str


class HierarchicalExecutive:
    """根据冻结 task graph 和可部署证据提交或拒绝 phase transition。"""

    def __init__(
        self,
        plan: CompiledTaskPlan,
        config: ExecutiveConfig | None = None,
    ) -> None:
        self.plan = plan
        self.config = config or ExecutiveConfig()
        initial_phase = plan.normal_phases[0]
        initial_spec = plan.phase_spec(initial_phase)
        self._state = ExecutiveState(
            status=ExecutiveStatus.RUNNING,
            subtask=initial_spec.subtask,
            phase=initial_phase,
            controller_owner=initial_spec.controller_owner,
            phase_age_ticks=0,
            recovery_attempts=0,
        )
        self.ledger = TransitionLedger(plan.plan_id)
        self._last_tick = -1
        self._last_timestamp_s = -1.0
        self._candidate_key: str | None = None
        self._candidate_ticks = 0
        self._resume_phase: PhaseId | None = None

    @property
    def state(self) -> ExecutiveState:
        return self._state

    def _validate_time(self, snapshot: ExecutiveSnapshot) -> None:
        if self._last_tick >= 0 and snapshot.tick != self._last_tick + 1:
            raise ValueError("Executive snapshot tick 必须严格连续递增")
        if snapshot.timestamp_s < self._last_timestamp_s:
            raise ValueError("Executive snapshot timestamp_s 不能回退")
        self._last_tick = snapshot.tick
        self._last_timestamp_s = snapshot.timestamp_s

    @staticmethod
    def _missing_modalities(
        spec: PhaseSpec,
        snapshot: ExecutiveSnapshot,
    ) -> tuple[str, ...]:
        return tuple(
            name for name in spec.required_modalities if name not in snapshot.valid_modalities
        )

    @staticmethod
    def _missing_predicates(
        names: tuple[str, ...],
        snapshot: ExecutiveSnapshot,
    ) -> tuple[str, ...]:
        return tuple(
            name
            for name in names
            if (evidence := snapshot.predicate(name)) is None or not evidence.satisfied
        )

    def _reset_candidate(self) -> None:
        self._candidate_key = None
        self._candidate_ticks = 0

    def _stable_candidate(self, key: str, required_ticks: int) -> bool:
        if self._candidate_key == key:
            self._candidate_ticks += 1
        else:
            self._candidate_key = key
            self._candidate_ticks = 1
        return self._candidate_ticks >= required_ticks

    def _current_critical_action(self) -> CriticalAction:
        if self._state.status != ExecutiveStatus.RUNNING:
            return CriticalAction.NONE
        return self.plan.phase_spec(self._state.phase).critical_action

    @staticmethod
    def _state_evidence_inconsistencies(
        snapshot: ExecutiveSnapshot,
    ) -> tuple[str, ...]:
        state = snapshot.state_estimate
        if state is None:
            return ()
        requirements = {
            "object_track_valid": state.object_track.valid,
            "goal_track_valid": state.goal_track.valid,
            "grasp_candidate": state.grasp.valid,
            "grasp_verified": state.grasp.valid,
            "support_contact_detected": state.support_contact.valid,
            "support_verified": state.support_contact.valid,
            "placement_verified": state.settled.valid,
        }
        return tuple(
            name
            for name, estimate_valid in requirements.items()
            if (
                (evidence := snapshot.predicate(name)) is not None
                and evidence.satisfied
                and not estimate_valid
            )
        )

    def _record(
        self,
        *,
        before: ExecutiveState,
        snapshot: ExecutiveSnapshot,
        outcome: TransitionOutcome,
        reason: TransitionReason,
        proposed_phase: PhaseId | None,
        proposed_subtask: SubtaskId | None,
        requires_action_reset: bool,
        missing_modalities: tuple[str, ...] = (),
        missing_predicates: tuple[str, ...] = (),
    ) -> ExecutiveDecision:
        critical = self._current_critical_action()
        actuation_allowed = (
            not self.config.shadow_only
            and self._state.status == ExecutiveStatus.RUNNING
            and self._state.controller_owner != ControllerOwner.SAFE_HOLD
        )
        decision = ExecutiveDecision(
            state=self._state,
            outcome=outcome,
            reason=reason,
            proposed_subtask=proposed_subtask,
            proposed_phase=proposed_phase,
            proposed_critical_action=critical,
            shadow_only=self.config.shadow_only,
            actuation_allowed=actuation_allowed,
            requires_action_reset=requires_action_reset,
            missing_modalities=missing_modalities,
            missing_predicates=missing_predicates,
        )
        self.ledger.append(
            TransitionLedgerEntry(
                index=len(self.ledger.entries),
                plan_id=self.plan.plan_id,
                snapshot=snapshot,
                before=before,
                decision=decision,
            )
        )
        return decision

    def _enter_recovery(
        self,
        *,
        before: ExecutiveState,
        snapshot: ExecutiveSnapshot,
        reason: TransitionReason,
        missing_modalities: tuple[str, ...] = (),
        missing_predicates: tuple[str, ...] = (),
    ) -> ExecutiveDecision:
        first_entry = self._state.subtask != SubtaskId.RECOVER_OR_HOLD
        if first_entry:
            self._resume_phase = self._state.phase
        self._state = ExecutiveState(
            status=ExecutiveStatus.HOLDING,
            subtask=SubtaskId.RECOVER_OR_HOLD,
            phase=PhaseId.SAFE_HOLD,
            controller_owner=ControllerOwner.SAFE_HOLD,
            phase_age_ticks=0,
            recovery_attempts=self._state.recovery_attempts,
        )
        self._reset_candidate()
        return self._record(
            before=before,
            snapshot=snapshot,
            outcome=(
                TransitionOutcome.COMMITTED if first_entry else TransitionOutcome.UNCHANGED
            ),
            reason=reason,
            proposed_phase=PhaseId.SAFE_HOLD,
            proposed_subtask=SubtaskId.RECOVER_OR_HOLD,
            requires_action_reset=first_entry,
            missing_modalities=missing_modalities,
            missing_predicates=missing_predicates,
        )

    def _commit_phase(
        self,
        *,
        before: ExecutiveState,
        snapshot: ExecutiveSnapshot,
        target: PhaseId,
        reason: TransitionReason,
    ) -> ExecutiveDecision:
        target_spec = self.plan.phase_spec(target)
        self._state = ExecutiveState(
            status=(
                ExecutiveStatus.HOLDING
                if target_spec.subtask == SubtaskId.RECOVER_OR_HOLD
                else ExecutiveStatus.RUNNING
            ),
            subtask=target_spec.subtask,
            phase=target,
            controller_owner=target_spec.controller_owner,
            phase_age_ticks=0,
            recovery_attempts=self._state.recovery_attempts,
        )
        self._reset_candidate()
        return self._record(
            before=before,
            snapshot=snapshot,
            outcome=TransitionOutcome.COMMITTED,
            reason=reason,
            proposed_phase=target,
            proposed_subtask=target_spec.subtask,
            requires_action_reset=True,
        )

    def _complete_or_abort(
        self,
        *,
        before: ExecutiveState,
        snapshot: ExecutiveSnapshot,
        status: ExecutiveStatus,
        reason: TransitionReason,
    ) -> ExecutiveDecision:
        self._state = replace(
            self._state,
            status=status,
            controller_owner=ControllerOwner.SAFE_HOLD,
            phase_age_ticks=0,
        )
        self._reset_candidate()
        return self._record(
            before=before,
            snapshot=snapshot,
            outcome=TransitionOutcome.COMMITTED,
            reason=reason,
            proposed_phase=None,
            proposed_subtask=None,
            requires_action_reset=True,
        )

    def _step_recovery(
        self,
        before: ExecutiveState,
        snapshot: ExecutiveSnapshot,
    ) -> ExecutiveDecision:
        phase = self._state.phase
        spec = self.plan.phase_spec(phase)
        missing_modalities = self._missing_modalities(spec, snapshot)
        if missing_modalities:
            self._reset_candidate()
            reason = (
                TransitionReason.RECOVERY_REOBSERVE_PENDING
                if phase == PhaseId.REOBSERVE
                else TransitionReason.MISSING_MODALITY
            )
            return self._record(
                before=before,
                snapshot=snapshot,
                outcome=TransitionOutcome.REJECTED,
                reason=reason,
                proposed_phase=phase,
                proposed_subtask=SubtaskId.RECOVER_OR_HOLD,
                requires_action_reset=False,
                missing_modalities=missing_modalities,
            )

        if phase == PhaseId.SAFE_HOLD:
            missing = self._missing_predicates(spec.exit_predicates, snapshot)
            if missing:
                self._reset_candidate()
                return self._record(
                    before=before,
                    snapshot=snapshot,
                    outcome=TransitionOutcome.UNCHANGED,
                    reason=TransitionReason.RECOVERY_HOLD_PENDING,
                    proposed_phase=PhaseId.REOBSERVE,
                    proposed_subtask=SubtaskId.RECOVER_OR_HOLD,
                    requires_action_reset=False,
                    missing_predicates=missing,
                )
            if not self._stable_candidate(
                PhaseId.REOBSERVE.value,
                spec.stable_ticks_required,
            ):
                return self._record(
                    before=before,
                    snapshot=snapshot,
                    outcome=TransitionOutcome.REJECTED,
                    reason=TransitionReason.STABILITY_PENDING,
                    proposed_phase=PhaseId.REOBSERVE,
                    proposed_subtask=SubtaskId.RECOVER_OR_HOLD,
                    requires_action_reset=False,
                )
            return self._commit_phase(
                before=before,
                snapshot=snapshot,
                target=PhaseId.REOBSERVE,
                reason=TransitionReason.RECOVERY_PHASE_COMPLETED,
            )

        if phase == PhaseId.REOBSERVE:
            missing = self._missing_predicates(spec.exit_predicates, snapshot)
            if missing:
                self._reset_candidate()
                return self._record(
                    before=before,
                    snapshot=snapshot,
                    outcome=TransitionOutcome.UNCHANGED,
                    reason=TransitionReason.RECOVERY_REOBSERVE_PENDING,
                    proposed_phase=PhaseId.DIAGNOSE,
                    proposed_subtask=SubtaskId.RECOVER_OR_HOLD,
                    requires_action_reset=False,
                    missing_predicates=missing,
                )
            if not self._stable_candidate(
                PhaseId.DIAGNOSE.value,
                spec.stable_ticks_required,
            ):
                return self._record(
                    before=before,
                    snapshot=snapshot,
                    outcome=TransitionOutcome.REJECTED,
                    reason=TransitionReason.STABILITY_PENDING,
                    proposed_phase=PhaseId.DIAGNOSE,
                    proposed_subtask=SubtaskId.RECOVER_OR_HOLD,
                    requires_action_reset=False,
                )
            return self._commit_phase(
                before=before,
                snapshot=snapshot,
                target=PhaseId.DIAGNOSE,
                reason=TransitionReason.RECOVERY_PHASE_COMPLETED,
            )

        missing_invariants = self._missing_predicates(
            spec.invariant_predicates,
            snapshot,
        )
        if missing_invariants:
            self._state = replace(
                self._state,
                phase=PhaseId.REOBSERVE,
                phase_age_ticks=0,
            )
            self._reset_candidate()
            return self._record(
                before=before,
                snapshot=snapshot,
                outcome=TransitionOutcome.REJECTED,
                reason=TransitionReason.RECOVERY_REOBSERVE_PENDING,
                proposed_phase=PhaseId.REOBSERVE,
                proposed_subtask=SubtaskId.RECOVER_OR_HOLD,
                requires_action_reset=False,
                missing_predicates=missing_invariants,
            )

        abort = snapshot.predicate("abort_required")
        if abort is not None and abort.satisfied:
            return self._complete_or_abort(
                before=before,
                snapshot=snapshot,
                status=ExecutiveStatus.ABORTED,
                reason=TransitionReason.ABORT_REQUESTED,
            )
        retry = snapshot.predicate("retry_authorized")
        replan = snapshot.predicate("replan_authorized")
        retry_requested = retry is not None and retry.satisfied
        replan_requested = replan is not None and replan.satisfied
        if not retry_requested and not replan_requested:
            return self._record(
                before=before,
                snapshot=snapshot,
                outcome=TransitionOutcome.UNCHANGED,
                reason=TransitionReason.RECOVERY_DIAGNOSE_PENDING,
                proposed_phase=None,
                proposed_subtask=None,
                requires_action_reset=False,
                missing_predicates=(
                    "retry_authorized|replan_authorized|abort_required",
                ),
            )
        if self._state.recovery_attempts >= self.config.max_recovery_attempts:
            return self._complete_or_abort(
                before=before,
                snapshot=snapshot,
                status=ExecutiveStatus.ABORTED,
                reason=TransitionReason.RETRY_BUDGET_EXHAUSTED,
            )

        target = (
            self.plan.normal_phases[0]
            if replan_requested
            else self._resume_phase or self.plan.normal_phases[0]
        )
        target_spec = self.plan.phase_spec(target)
        target_missing = self._missing_modalities(target_spec, snapshot)
        if target_missing:
            self._state = replace(
                self._state,
                phase=PhaseId.REOBSERVE,
                phase_age_ticks=0,
            )
            self._reset_candidate()
            return self._record(
                before=before,
                snapshot=snapshot,
                outcome=TransitionOutcome.REJECTED,
                reason=TransitionReason.RECOVERY_REOBSERVE_PENDING,
                proposed_phase=target,
                proposed_subtask=target_spec.subtask,
                requires_action_reset=False,
                missing_modalities=target_missing,
            )
        target_predicates = tuple(
            dict.fromkeys(
                target_spec.entry_predicates + target_spec.invariant_predicates
            )
        )
        target_missing_predicates = self._missing_predicates(
            target_predicates,
            snapshot,
        )
        if target_missing_predicates:
            self._state = replace(
                self._state,
                phase=PhaseId.REOBSERVE,
                phase_age_ticks=0,
            )
            self._reset_candidate()
            return self._record(
                before=before,
                snapshot=snapshot,
                outcome=TransitionOutcome.REJECTED,
                reason=TransitionReason.RECOVERY_REOBSERVE_PENDING,
                proposed_phase=target,
                proposed_subtask=target_spec.subtask,
                requires_action_reset=False,
                missing_predicates=target_missing_predicates,
            )

        self._state = ExecutiveState(
            status=ExecutiveStatus.RUNNING,
            subtask=target_spec.subtask,
            phase=target,
            controller_owner=target_spec.controller_owner,
            phase_age_ticks=0,
            recovery_attempts=self._state.recovery_attempts + 1,
        )
        self._resume_phase = None
        self._reset_candidate()
        return self._record(
            before=before,
            snapshot=snapshot,
            outcome=TransitionOutcome.COMMITTED,
            reason=(
                TransitionReason.RECOVERY_REPLAN
                if replan_requested
                else TransitionReason.RECOVERY_RETRY
            ),
            proposed_phase=target,
            proposed_subtask=target_spec.subtask,
            requires_action_reset=True,
        )

    def _step_normal(
        self,
        before: ExecutiveState,
        snapshot: ExecutiveSnapshot,
    ) -> ExecutiveDecision:
        current = self.plan.phase_spec(self._state.phase)
        missing_modalities = self._missing_modalities(current, snapshot)
        if missing_modalities:
            return self._enter_recovery(
                before=before,
                snapshot=snapshot,
                reason=TransitionReason.MISSING_MODALITY,
                missing_modalities=missing_modalities,
            )
        if current.critical_action != CriticalAction.NONE:
            missing_authorization = self._missing_predicates(
                current.entry_predicates,
                snapshot,
            )
            if missing_authorization:
                return self._enter_recovery(
                    before=before,
                    snapshot=snapshot,
                    reason=TransitionReason.CRITICAL_ACTION_NOT_AUTHORIZED,
                    missing_predicates=missing_authorization,
                )
        missing_invariants = self._missing_predicates(
            current.invariant_predicates,
            snapshot,
        )
        if missing_invariants:
            return self._enter_recovery(
                before=before,
                snapshot=snapshot,
                reason=TransitionReason.STATE_INVARIANT_FAILED,
                missing_predicates=missing_invariants,
            )
        if (
            current.timeout_ticks is not None
            and self._state.phase_age_ticks >= current.timeout_ticks
        ):
            return self._enter_recovery(
                before=before,
                snapshot=snapshot,
                reason=TransitionReason.PHASE_TIMEOUT,
            )

        next_phase = self.plan.next_normal_phase(current.phase)
        proposed_subtask = (
            None if next_phase is None else self.plan.phase_spec(next_phase).subtask
        )
        missing_exit = self._missing_predicates(current.exit_predicates, snapshot)
        if missing_exit:
            self._reset_candidate()
            return self._record(
                before=before,
                snapshot=snapshot,
                outcome=TransitionOutcome.UNCHANGED,
                reason=TransitionReason.EXIT_NOT_SATISFIED,
                proposed_phase=next_phase,
                proposed_subtask=proposed_subtask,
                requires_action_reset=False,
                missing_predicates=missing_exit,
            )

        if next_phase is not None:
            target = self.plan.phase_spec(next_phase)
            target_missing = self._missing_modalities(target, snapshot)
            if target_missing:
                return self._enter_recovery(
                    before=before,
                    snapshot=snapshot,
                    reason=TransitionReason.MISSING_MODALITY,
                    missing_modalities=target_missing,
                )
            target_requirements = tuple(
                dict.fromkeys(
                    target.entry_predicates + target.invariant_predicates
                )
            )
            missing_entry = self._missing_predicates(target_requirements, snapshot)
            if missing_entry:
                self._reset_candidate()
                return self._record(
                    before=before,
                    snapshot=snapshot,
                    outcome=TransitionOutcome.REJECTED,
                    reason=TransitionReason.ENTRY_NOT_SATISFIED,
                    proposed_phase=next_phase,
                    proposed_subtask=target.subtask,
                    requires_action_reset=False,
                    missing_predicates=missing_entry,
                )
            stable_ticks = max(
                current.stable_ticks_required,
                target.stable_ticks_required,
            )
            if not self._stable_candidate(next_phase.value, stable_ticks):
                return self._record(
                    before=before,
                    snapshot=snapshot,
                    outcome=TransitionOutcome.REJECTED,
                    reason=TransitionReason.STABILITY_PENDING,
                    proposed_phase=next_phase,
                    proposed_subtask=target.subtask,
                    requires_action_reset=False,
                )
            return self._commit_phase(
                before=before,
                snapshot=snapshot,
                target=next_phase,
                reason=TransitionReason.PHASE_COMPLETED,
            )

        if not self._stable_candidate("__task_completed__", current.stable_ticks_required):
            return self._record(
                before=before,
                snapshot=snapshot,
                outcome=TransitionOutcome.REJECTED,
                reason=TransitionReason.STABILITY_PENDING,
                proposed_phase=None,
                proposed_subtask=None,
                requires_action_reset=False,
            )
        return self._complete_or_abort(
            before=before,
            snapshot=snapshot,
            status=ExecutiveStatus.COMPLETED,
            reason=TransitionReason.TASK_COMPLETED,
        )

    def step(self, snapshot: ExecutiveSnapshot) -> ExecutiveDecision:
        """处理一个严格递增的 Tick；任何污染/异常都 fail closed 到 SafeHold。"""

        self._validate_time(snapshot)
        before = self._state
        if self._state.status in {ExecutiveStatus.COMPLETED, ExecutiveStatus.ABORTED}:
            return self._record(
                before=before,
                snapshot=snapshot,
                outcome=TransitionOutcome.UNCHANGED,
                reason=TransitionReason.TERMINAL_STATE,
                proposed_phase=None,
                proposed_subtask=None,
                requires_action_reset=False,
            )

        self._state = replace(
            self._state,
            phase_age_ticks=self._state.phase_age_ticks + 1,
        )
        if snapshot.uses_evaluator_gt:
            return self._enter_recovery(
                before=before,
                snapshot=snapshot,
                reason=TransitionReason.HIDDEN_GT_REJECTED,
            )
        if self.config.require_state_estimate and snapshot.state_estimate is None:
            return self._enter_recovery(
                before=before,
                snapshot=snapshot,
                reason=TransitionReason.MISSING_STATE_ESTIMATE,
                missing_predicates=("state_estimate",),
            )
        inconsistencies = self._state_evidence_inconsistencies(snapshot)
        if inconsistencies:
            return self._enter_recovery(
                before=before,
                snapshot=snapshot,
                reason=TransitionReason.INCONSISTENT_STATE_EVIDENCE,
                missing_predicates=inconsistencies,
            )
        if snapshot.unsafe_or_anomalous:
            return self._enter_recovery(
                before=before,
                snapshot=snapshot,
                reason=TransitionReason.UNSAFE_OR_ANOMALOUS,
            )
        if self._state.subtask == SubtaskId.RECOVER_OR_HOLD:
            return self._step_recovery(before, snapshot)
        return self._step_normal(before, snapshot)


def replay_executive(
    plan: CompiledTaskPlan,
    snapshots: Iterable[ExecutiveSnapshot],
    config: ExecutiveConfig | None = None,
) -> ExecutiveReplayResult:
    """用同一 Runtime 状态机离线重放证据，不引入第二套分析语义。"""

    executive = HierarchicalExecutive(plan, config)
    decisions = tuple(executive.step(snapshot) for snapshot in snapshots)
    return ExecutiveReplayResult(
        decisions=decisions,
        final_state=executive.state,
        ledger_entries=executive.ledger.entries,
        ledger_sha256=executive.ledger.sha256(),
    )


def replay_ledger_jsonl(
    plan: CompiledTaskPlan,
    ledger_jsonl: str,
    config: ExecutiveConfig | None = None,
) -> ExecutiveReplayResult:
    """从已保存 ledger 重新执行 snapshots，并拒绝任何状态或决策漂移。"""

    if not isinstance(ledger_jsonl, str):
        raise TypeError("ledger_jsonl 必须是字符串")
    records: list[dict[str, object]] = []
    snapshots: list[ExecutiveSnapshot] = []
    for index, line in enumerate(ledger_jsonl.splitlines()):
        if not line.strip():
            raise ValueError("ledger JSONL 不能包含空行")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise ValueError("ledger 每行必须是 JSON object")
        expected = {"version", "index", "plan_id", "snapshot", "before", "decision"}
        if set(value) != expected:
            raise ValueError("ledger entry 字段不匹配")
        if value["version"] != TRANSITION_LEDGER_VERSION:
            raise ValueError("ledger version 漂移")
        if value["index"] != index:
            raise ValueError("ledger index 不连续")
        if value["plan_id"] != plan.plan_id:
            raise ValueError("ledger plan_id 漂移")
        snapshot_value = value["snapshot"]
        if not isinstance(snapshot_value, dict):
            raise ValueError("ledger snapshot 必须是 JSON object")
        records.append(value)
        snapshots.append(ExecutiveSnapshot.from_dict(snapshot_value))

    replay = replay_executive(plan, snapshots, config)
    expected_jsonl = ""
    if records:
        expected_jsonl = "\n".join(
            json.dumps(
                record,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            for record in records
        ) + "\n"
    actual_jsonl = "".join(
        entry.canonical_json() + "\n" for entry in replay.ledger_entries
    )
    if actual_jsonl != expected_jsonl:
        raise ValueError("ledger replay drift：重新执行结果与保存记录不一致")
    return replay


__all__ = [
    "ExecutiveConfig",
    "ExecutiveReplayResult",
    "HierarchicalExecutive",
    "TransitionLedger",
    "replay_executive",
    "replay_ledger_jsonl",
]
