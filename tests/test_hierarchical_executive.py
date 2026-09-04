from __future__ import annotations

from dataclasses import replace

import pytest

from robot_vla.executive import (
    CANONICAL_PICK_PLACE_SUBTASKS,
    EXECUTIVE_MODALITIES,
    EXECUTIVE_PREDICATES,
    PICK_PLACE_TASK_ID,
    ControllerOwner,
    CriticalAction,
    DeployableStateEstimate,
    ExecutiveConfig,
    ExecutiveSnapshot,
    ExecutiveStatus,
    HierarchicalExecutive,
    ModalityStatus,
    PhaseId,
    PickPlacePlanCompiler,
    PlanCompilationError,
    PlanCompilerConfig,
    PredicateEvidence,
    PredicateSource,
    ScalarStateEstimate,
    SemanticPlanProposal,
    SubtaskId,
    SpatialTrackEstimate,
    TransitionOutcome,
    TransitionReason,
    replay_executive,
    replay_ledger_jsonl,
)


def _proposal() -> SemanticPlanProposal:
    return SemanticPlanProposal(
        proposal_id="proposal-001",
        task_id=PICK_PLACE_TASK_ID,
        object_ref="red-cube",
        goal_ref="green-region",
        requested_subtasks=CANONICAL_PICK_PLACE_SUBTASKS,
    )


def _valid_state_estimate(tick: int) -> DeployableStateEstimate:
    return DeployableStateEstimate(
        object_track=SpatialTrackEstimate(
            position_base_m=(0.42, -0.03, 0.02),
            velocity_base_m_s=(0.01, 0.0, 0.0),
            confidence=0.98,
            valid=True,
            age_s=0.01,
        ),
        goal_track=SpatialTrackEstimate(
            position_base_m=(0.55, 0.12, 0.02),
            velocity_base_m_s=(0.0, 0.0, 0.0),
            confidence=0.97,
            valid=True,
            age_s=0.01,
        ),
        grasp=ScalarStateEstimate(confidence=0.9, valid=True, age_s=0.01),
        support_contact=ScalarStateEstimate(
            confidence=0.9,
            valid=True,
            age_s=0.01,
        ),
        settled=ScalarStateEstimate(confidence=0.9, valid=True, age_s=0.01),
        finger_force_n=(1.2, 1.1),
        timestamp_s=tick * 0.05,
    )


def _snapshot(
    tick: int,
    *,
    satisfied: tuple[str, ...] = (),
    invalid_modalities: tuple[str, ...] = (),
    evaluator_gt_predicates: tuple[str, ...] = (),
    state_estimate: DeployableStateEstimate | None = None,
    include_state_estimate: bool = True,
    unsafe: bool = False,
) -> ExecutiveSnapshot:
    invalid = set(invalid_modalities)
    gt = set(evaluator_gt_predicates)
    return ExecutiveSnapshot(
        tick=tick,
        timestamp_s=tick * 0.05,
        modalities=tuple(
            ModalityStatus(
                name=name,
                valid=name not in invalid,
                age_s=None if name in invalid else 0.01,
            )
            for name in EXECUTIVE_MODALITIES
        ),
        predicates=tuple(
            PredicateEvidence(
                name=name,
                satisfied=name in satisfied,
                confidence=1.0 if name in satisfied else 0.0,
                source=(
                    PredicateSource.EVALUATOR_GT
                    if name in gt
                    else PredicateSource.DEPLOYABLE_ESTIMATOR
                ),
            )
            for name in EXECUTIVE_PREDICATES
            if name in satisfied or name in gt
        ),
        state_estimate=(
            state_estimate
            if state_estimate is not None
            else (_valid_state_estimate(tick) if include_state_estimate else None)
        ),
        unsafe_or_anomalous=unsafe,
        anomaly_reason="tracking_saturation" if unsafe else None,
    )


def test_plan_compiler_is_strict_and_identity_is_stable() -> None:
    proposal = _proposal()
    parsed = SemanticPlanProposal.from_dict(proposal.to_dict())
    compiler = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=2))

    first = compiler.compile(parsed)
    second = compiler.compile(proposal)

    assert first.plan_id == second.plan_id
    assert len(first.plan_id) == 64
    assert len(first.subtasks) == 5
    assert len(first.phases) == 20
    assert len(first.normal_phases) == 17
    assert first.normal_phases[0] == PhaseId.ACQUIRE_TRACK
    assert first.normal_phases[-1] == PhaseId.VERIFY_SETTLED
    assert first.phase_spec(PhaseId.RELEASE).critical_action == CriticalAction.RELEASE_GRIPPER
    assert first.phase_spec(PhaseId.FINE_ALIGN).controller_owner == ControllerOwner.PRECISION

    bad = SemanticPlanProposal(
        proposal_id="proposal-bad",
        task_id=PICK_PLACE_TASK_ID,
        object_ref="red-cube",
        goal_ref="green-region",
        requested_subtasks=tuple(reversed(CANONICAL_PICK_PLACE_SUBTASKS)),
    )
    with pytest.raises(PlanCompilationError, match="严格等于冻结"):
        compiler.compile(bad)

    payload = proposal.to_dict()
    payload["free_form_transition"] = "release-now"
    with pytest.raises(ValueError, match="字段不匹配"):
        SemanticPlanProposal.from_dict(payload)


def test_transition_requires_consecutive_stable_ticks_and_stays_shadow_only() -> None:
    plan = PickPlacePlanCompiler(
        PlanCompilerConfig(stable_ticks_required=2)
    ).compile(_proposal())
    executive = HierarchicalExecutive(plan)
    evidence = ("object_track_valid", "goal_track_valid")

    pending = executive.step(_snapshot(0, satisfied=evidence))
    committed = executive.step(_snapshot(1, satisfied=evidence))

    assert pending.outcome == TransitionOutcome.REJECTED
    assert pending.reason == TransitionReason.STABILITY_PENDING
    assert pending.state.phase == PhaseId.ACQUIRE_TRACK
    assert committed.outcome == TransitionOutcome.COMMITTED
    assert committed.state.phase == PhaseId.COARSE_APPROACH
    assert committed.state.controller_owner == ControllerOwner.ACTION_CHUNK
    assert committed.requires_action_reset is True
    assert committed.shadow_only is True
    assert committed.actuation_allowed is False
    assert tuple(item.index for item in executive.ledger.entries) == (0, 1)


def test_hidden_gt_and_missing_modality_both_fail_closed_to_safe_hold() -> None:
    plan = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=1)).compile(
        _proposal()
    )
    hidden_gt = HierarchicalExecutive(plan)
    decision = hidden_gt.step(
        _snapshot(
            0,
            satisfied=("object_track_valid", "goal_track_valid"),
            evaluator_gt_predicates=("object_track_valid",),
        )
    )

    assert decision.reason == TransitionReason.HIDDEN_GT_REJECTED
    assert decision.state.status == ExecutiveStatus.HOLDING
    assert decision.state.subtask == SubtaskId.RECOVER_OR_HOLD
    assert decision.state.phase == PhaseId.SAFE_HOLD
    assert decision.state.controller_owner == ControllerOwner.SAFE_HOLD
    assert decision.actuation_allowed is False
    assert decision.requires_action_reset is True

    missing = HierarchicalExecutive(plan)
    missing_decision = missing.step(
        _snapshot(0, invalid_modalities=("tcp_pose",))
    )
    assert missing_decision.reason == TransitionReason.MISSING_MODALITY
    assert missing_decision.missing_modalities == ("tcp_pose",)
    assert missing_decision.state.phase == PhaseId.SAFE_HOLD


def test_minimum_deployable_state_track_is_audited_and_gt_track_is_rejected() -> None:
    state = DeployableStateEstimate(
        object_track=SpatialTrackEstimate(
            position_base_m=(0.42, -0.03, 0.02),
            velocity_base_m_s=(0.01, 0.0, 0.0),
            confidence=0.98,
            valid=True,
            age_s=0.01,
            source=PredicateSource.EVALUATOR_GT,
        ),
        goal_track=SpatialTrackEstimate(
            position_base_m=(0.55, 0.12, 0.02),
            velocity_base_m_s=(0.0, 0.0, 0.0),
            confidence=0.97,
            valid=True,
            age_s=0.01,
        ),
        grasp=ScalarStateEstimate(confidence=0.1, valid=True, age_s=0.01),
        support_contact=ScalarStateEstimate(
            confidence=0.9,
            valid=True,
            age_s=0.01,
        ),
        settled=ScalarStateEstimate(confidence=0.8, valid=True, age_s=0.01),
        finger_force_n=(1.2, 1.1),
        timestamp_s=0.0,
    )
    plan = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=1)).compile(
        _proposal()
    )
    decision = HierarchicalExecutive(plan).step(
        _snapshot(0, state_estimate=state)
    )

    assert state.object_track.velocity_base_m_s == (0.01, 0.0, 0.0)
    assert state.history_length == 4
    assert state.finger_force_n == (1.2, 1.1)
    assert state.uses_evaluator_gt is True
    restored = ExecutiveSnapshot.from_dict(
        _snapshot(0, state_estimate=state).to_dict()
    )
    assert restored.state_estimate == state
    assert decision.reason == TransitionReason.HIDDEN_GT_REJECTED
    assert decision.state.phase == PhaseId.SAFE_HOLD


def test_missing_or_inconsistent_state_estimate_fails_closed() -> None:
    plan = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=1)).compile(
        _proposal()
    )
    missing = HierarchicalExecutive(plan).step(
        _snapshot(0, include_state_estimate=False)
    )
    assert missing.reason == TransitionReason.MISSING_STATE_ESTIMATE
    assert missing.state.phase == PhaseId.SAFE_HOLD

    invalid_object_track = SpatialTrackEstimate(
        position_base_m=None,
        velocity_base_m_s=None,
        confidence=0.0,
        valid=False,
        age_s=None,
    )
    inconsistent_state = replace(
        _valid_state_estimate(0),
        object_track=invalid_object_track,
    )
    inconsistent = HierarchicalExecutive(plan).step(
        _snapshot(
            0,
            satisfied=("object_track_valid",),
            state_estimate=inconsistent_state,
        )
    )
    assert inconsistent.reason == TransitionReason.INCONSISTENT_STATE_EVIDENCE
    assert inconsistent.missing_predicates == ("object_track_valid",)
    assert inconsistent.state.phase == PhaseId.SAFE_HOLD


def test_full_replay_reaches_completion_and_freezes_a_deterministic_ledger() -> None:
    plan = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=1)).compile(
        _proposal()
    )
    all_true = tuple(EXECUTIVE_PREDICATES)
    snapshots = tuple(
        _snapshot(tick, satisfied=all_true)
        for tick in range(len(plan.normal_phases))
    )

    shadow = replay_executive(plan, snapshots)
    repeated = replay_executive(plan, snapshots)
    active = replay_executive(
        plan,
        snapshots,
        ExecutiveConfig(shadow_only=False),
    )

    assert shadow.final_state.status == ExecutiveStatus.COMPLETED
    assert shadow.final_state.controller_owner == ControllerOwner.SAFE_HOLD
    assert shadow.ledger_sha256 == repeated.ledger_sha256
    assert len(shadow.ledger_entries) == len(plan.normal_phases)
    assert [entry.index for entry in shadow.ledger_entries] == list(
        range(len(plan.normal_phases))
    )
    assert {decision.proposed_critical_action for decision in shadow.decisions} >= {
        CriticalAction.CLOSE_GRIPPER,
        CriticalAction.LIFT,
        CriticalAction.RELEASE_GRIPPER,
    }
    assert not any(decision.actuation_allowed for decision in shadow.decisions)
    assert any(
        decision.proposed_critical_action == CriticalAction.RELEASE_GRIPPER
        and decision.actuation_allowed
        for decision in active.decisions
    )


def test_recovery_resumes_once_then_aborts_when_retry_budget_is_exhausted() -> None:
    plan = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=1)).compile(
        _proposal()
    )
    executive = HierarchicalExecutive(
        plan,
        ExecutiveConfig(max_recovery_attempts=1),
    )

    executive.step(
        _snapshot(0, satisfied=("object_track_valid", "goal_track_valid"))
    )
    first_failure = executive.step(
        _snapshot(1, invalid_modalities=("tcp_pose",))
    )
    executive.step(_snapshot(2, satisfied=("hold_confirmed",)))
    executive.step(_snapshot(3, satisfied=("modalities_recovered",)))
    resumed = executive.step(
        _snapshot(
            4,
            satisfied=(
                "modalities_recovered",
                "retry_authorized",
                "object_track_valid",
                "goal_track_valid",
            ),
        )
    )

    assert first_failure.state.phase == PhaseId.SAFE_HOLD
    assert resumed.reason == TransitionReason.RECOVERY_RETRY
    assert resumed.state.status == ExecutiveStatus.RUNNING
    assert resumed.state.phase == PhaseId.COARSE_APPROACH
    assert resumed.state.recovery_attempts == 1
    assert resumed.requires_action_reset is True

    executive.step(_snapshot(5, invalid_modalities=("tcp_pose",)))
    executive.step(_snapshot(6, satisfied=("hold_confirmed",)))
    executive.step(_snapshot(7, satisfied=("modalities_recovered",)))
    exhausted = executive.step(
        _snapshot(
            8,
            satisfied=(
                "modalities_recovered",
                "retry_authorized",
                "object_track_valid",
                "goal_track_valid",
            ),
        )
    )

    assert exhausted.reason == TransitionReason.RETRY_BUDGET_EXHAUSTED
    assert exhausted.state.status == ExecutiveStatus.ABORTED
    assert exhausted.state.controller_owner == ControllerOwner.SAFE_HOLD
    assert exhausted.requires_action_reset is True


def test_critical_action_authorization_must_remain_true_on_every_tick() -> None:
    plan = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=1)).compile(
        _proposal()
    )
    executive = HierarchicalExecutive(
        plan,
        ExecutiveConfig(shadow_only=False),
    )
    all_true = tuple(EXECUTIVE_PREDICATES)
    decision = None
    for tick in range(5):
        decision = executive.step(_snapshot(tick, satisfied=all_true))

    assert decision is not None
    assert decision.state.phase == PhaseId.CLOSE_UNTIL_CONTACT
    assert decision.proposed_critical_action == CriticalAction.CLOSE_GRIPPER
    assert decision.actuation_allowed is True

    without_close_authorization = tuple(
        name for name in EXECUTIVE_PREDICATES if name != "close_authorized"
    )
    rejected = executive.step(
        _snapshot(5, satisfied=without_close_authorization)
    )
    assert rejected.reason == TransitionReason.CRITICAL_ACTION_NOT_AUTHORIZED
    assert rejected.state.phase == PhaseId.SAFE_HOLD
    assert rejected.actuation_allowed is False
    assert rejected.missing_predicates == ("close_authorized",)


def test_lost_tracking_invariant_enters_recovery_without_waiting_for_timeout() -> None:
    plan = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=1)).compile(
        _proposal()
    )
    executive = HierarchicalExecutive(plan)
    executive.step(
        _snapshot(0, satisfied=("object_track_valid", "goal_track_valid"))
    )

    lost = executive.step(_snapshot(1, satisfied=("goal_track_valid",)))

    assert lost.reason == TransitionReason.STATE_INVARIANT_FAILED
    assert lost.missing_predicates == ("object_track_valid",)
    assert lost.state.phase == PhaseId.SAFE_HOLD
    assert lost.requires_action_reset is True


def test_target_phase_invariants_are_required_before_owner_handoff() -> None:
    plan = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=1)).compile(
        _proposal()
    )
    executive = HierarchicalExecutive(plan)
    all_true = tuple(EXECUTIVE_PREDICATES)
    executive.step(_snapshot(0, satisfied=all_true))
    executive.step(_snapshot(1, satisfied=all_true))
    executive.step(_snapshot(2, satisfied=all_true))

    without_precision_target = (
        "pregrasp_pose_valid",
        "pregrasp_stable",
    )
    blocked = executive.step(
        _snapshot(3, satisfied=without_precision_target)
    )

    assert blocked.reason == TransitionReason.ENTRY_NOT_SATISFIED
    assert blocked.state.phase == PhaseId.STABILIZE_PREGRASP
    assert blocked.proposed_phase == PhaseId.FINAL_APPROACH
    assert blocked.missing_predicates == ("precision_target_valid",)
    assert blocked.requires_action_reset is False


def test_snapshot_time_and_tick_cannot_move_backward() -> None:
    plan = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=1)).compile(
        _proposal()
    )
    executive = HierarchicalExecutive(plan)
    executive.step(_snapshot(1))

    with pytest.raises(ValueError, match="tick 必须严格连续递增"):
        executive.step(_snapshot(1))


def test_saved_jsonl_replay_recomputes_and_rejects_decision_drift() -> None:
    plan = PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=1)).compile(
        _proposal()
    )
    executive = HierarchicalExecutive(plan)
    snapshots = tuple(
        _snapshot(tick, satisfied=tuple(EXECUTIVE_PREDICATES))
        for tick in range(len(plan.normal_phases))
    )
    for snapshot in snapshots:
        executive.step(snapshot)
    jsonl = executive.ledger.to_jsonl()

    replayed = replay_ledger_jsonl(plan, jsonl)
    assert replayed.ledger_sha256 == executive.ledger.sha256()

    tampered = jsonl.replace(
        '"reason":"task_completed"',
        '"reason":"phase_completed"',
    )
    with pytest.raises(ValueError, match="ledger replay drift"):
        replay_ledger_jsonl(plan, tampered)
