from __future__ import annotations

from dataclasses import replace

import pytest

from robot_vla.executive.contracts import ControllerOwner, PhaseId
from robot_vla.precision.active_front_reobserve import (
    ALLOWED_ACTIVE_SOURCE_PHASES,
    POST_ACTIVE_WINDOW_PHASES,
    ActionHistoryResetReceipt,
    ActionHistoryResumeReceipt,
    ActiveFrontDecisionReason,
    ActiveFrontFailure,
    ActiveFrontReobserveConfig,
    ActiveFrontReobserveController,
    ActiveFrontReobserveState,
    ActiveFrontSafetyEvidence,
    ActiveFrontSignal,
    ActiveFrontTriggerEvidence,
    ActiveFrontTriggerReason,
    ExternalCameraControllerOwner,
    HomeV2BarrierFrame,
    Stage1ShadowCandidateReceipt,
)


def _controller(*, enabled: bool = True) -> ActiveFrontReobserveController:
    controller = ActiveFrontReobserveController(
        ActiveFrontReobserveConfig(enabled=enabled)
    )
    controller.reset_episode("episode-001", episode_generation=1)
    return controller


def _evidence(
    *,
    tick: int,
    phase: PhaseId = PhaseId.ACQUIRE_TRACK,
    wrist_usable: bool = False,
    front_usable: bool = False,
    memory_available: bool = False,
    failure: ActiveFrontTriggerReason = ActiveFrontTriggerReason.OBJECT_OCCLUSION,
    **events: bool,
) -> ActiveFrontTriggerEvidence:
    return ActiveFrontTriggerEvidence(
        episode_id="episode-001",
        episode_generation=1,
        control_tick=tick,
        timestamp_s=tick * 0.05,
        source_phase=phase,
        wrist_object_measurement_usable=wrist_usable,
        front_home_object_measurement_usable=front_usable,
        object_memory_navigation_state_available=memory_available,
        arm_hold_prerequisites_pass=True,
        camera_home_prerequisites_pass=True,
        failure_reason=failure,
        **events,
    )


def _request(controller: ActiveFrontReobserveController, *, phase: PhaseId = PhaseId.ACQUIRE_TRACK):
    decisions = [
        controller.consider_trigger(_evidence(tick=tick, phase=phase))
        for tick in range(3)
    ]
    assert decisions[-1].request is not None
    return decisions[-1].request


def _reset_receipt(request, **updates: object) -> ActionHistoryResetReceipt:
    values: dict[str, object] = {
        "episode_id": request.episode_id,
        "request_id": request.request_id,
        "reset_control_tick": request.trigger_tick,
        "generation_before": 8,
        "generation_after": 9,
        "action_chunk_cleared": True,
        "temporal_ensemble_cleared": True,
        "rtc_overlap_cleared": True,
        "command_reference_invalidated": True,
    }
    values.update(updates)
    return ActionHistoryResetReceipt(**values)


def _advance_to_home_barrier(controller: ActiveFrontReobserveController) -> None:
    controller.advance(ActiveFrontSignal.CAMERA_LEASE_ACQUIRED)
    controller.advance(
        ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
        selected_primitive_id="LEFT_LOW__YAW_LEFT",
    )
    controller.advance(ActiveFrontSignal.MOVE_COMPLETE)
    controller.advance(ActiveFrontSignal.SETTLE_COMPLETE)
    controller.advance(ActiveFrontSignal.COLLECTION_COMPLETE)
    request = controller.request
    assert request is not None
    controller.advance(
        ActiveFrontSignal.SHADOW_CANDIDATE_STAGED,
        shadow_candidate_receipt=Stage1ShadowCandidateReceipt(
            request_id=request.request_id,
            candidate_digest="shadow-candidate-digest",
            shadow_only=True,
            live_memory_write_executed=False,
            provider_forward_count=0,
        ),
    )
    controller.advance(ActiveFrontSignal.RETURN_HOME_COMPLETE)
    assert controller.state is ActiveFrontReobserveState.VERIFY_HOME_AND_ARM_HOLD


def _home_frame(index: int, **updates: bool) -> HomeV2BarrierFrame:
    values: dict[str, object] = {
        "observation_sequence_id": f"home-observation-{index}",
        "camera_at_home": True,
        "fresh_observation_v2_frame": True,
        "captured_after_return": True,
        "contains_alternate_or_motion_rgb": False,
    }
    values.update(updates)
    return HomeV2BarrierFrame(**values)


def _complete_success(controller: ActiveFrontReobserveController):
    request = _request(controller)
    controller.begin(_reset_receipt(request))
    _advance_to_home_barrier(controller)
    for index in range(4):
        controller.accept_home_v2_barrier_frame(_home_frame(index))
    assert controller.state is ActiveFrontReobserveState.RECHECK_SOURCE_INVARIANTS
    controller.advance(
        ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED,
        source_phase=request.resume_phase,
        source_invariants_passed=True,
    )
    return controller.complete_no_write_resume(
        ActionHistoryResumeReceipt(
            episode_id=request.episode_id,
            request_id=request.request_id,
            generation=10,
            home_observation_sequence_ids=tuple(
                f"home-observation-{index}" for index in range(4)
            ),
            generated_from_fresh_home_v2=True,
            stale_action_chunk_resumed=False,
        )
    )


def test_feature_is_disabled_by_default() -> None:
    controller = _controller(enabled=False)

    decision = controller.consider_trigger(_evidence(tick=0))

    assert decision.requestable is False
    assert decision.reason is ActiveFrontDecisionReason.FEATURE_DISABLED
    assert controller.state is ActiveFrontReobserveState.IDLE


@pytest.mark.parametrize(
    ("wrist_usable", "front_usable", "memory_available", "expected"),
    [
        (False, False, False, True),
        (False, False, True, False),
        (False, True, False, False),
        (False, True, True, False),
        (True, False, False, False),
        (True, False, True, False),
        (True, True, False, False),
        (True, True, True, False),
    ],
)
def test_trigger_requires_wrist_home_front_and_memory_all_unusable(
    wrist_usable: bool,
    front_usable: bool,
    memory_available: bool,
    expected: bool,
) -> None:
    controller = _controller()
    decision = None
    for tick in range(3):
        decision = controller.consider_trigger(
            _evidence(
                tick=tick,
                wrist_usable=wrist_usable,
                front_usable=front_usable,
                memory_available=memory_available,
            )
        )

    assert decision is not None
    assert decision.requestable is expected


def test_single_frame_failure_cannot_trigger_and_good_frame_resets_streak() -> None:
    controller = _controller()

    first = controller.consider_trigger(_evidence(tick=0))
    cleared = controller.consider_trigger(_evidence(tick=1, wrist_usable=True))
    second = controller.consider_trigger(_evidence(tick=2))

    assert first.reason is ActiveFrontDecisionReason.CONSECUTIVE_EVIDENCE_PENDING
    assert cleared.reason is ActiveFrontDecisionReason.DIRECT_WRIST_EVIDENCE_AVAILABLE
    assert second.consecutive_unusable_ticks == 1
    assert second.requestable is False


@pytest.mark.parametrize("phase", list(PhaseId))
def test_only_two_source_phases_can_request(phase: PhaseId) -> None:
    controller = _controller()
    decisions = [
        controller.consider_trigger(_evidence(tick=tick, phase=phase))
        for tick in range(3)
    ]

    assert decisions[-1].requestable is (phase in ALLOWED_ACTIVE_SOURCE_PHASES)
    if phase not in ALLOWED_ACTIVE_SOURCE_PHASES:
        assert decisions[-1].reason is ActiveFrontDecisionReason.DISALLOWED_SOURCE_PHASE


@pytest.mark.parametrize(
    "failure",
    [
        ActiveFrontTriggerReason.INVALID_SENSOR_OR_POSE,
        ActiveFrontTriggerReason.PROVIDER_IDENTITY_MISMATCH,
        ActiveFrontTriggerReason.UNSAFE_ARM_STATE,
        ActiveFrontTriggerReason.UNSAFE_CAMERA_STATE,
        ActiveFrontTriggerReason.UNKNOWN,
    ],
)
def test_non_viewpoint_resolvable_failures_never_trigger(failure: ActiveFrontTriggerReason) -> None:
    controller = _controller()

    decisions = [
        controller.consider_trigger(_evidence(tick=tick, failure=failure))
        for tick in range(3)
    ]

    assert decisions[-1].reason is ActiveFrontDecisionReason.FAILURE_NOT_VIEWPOINT_RESOLVABLE
    assert decisions[-1].requestable is False


@pytest.mark.parametrize(
    "phase,events",
    [
        (PhaseId.FINAL_APPROACH, {}),
        (PhaseId.ACQUIRE_TRACK, {"object_contact": True}),
        (PhaseId.ACQUIRE_TRACK, {"gripper_close_commanded": True}),
        (PhaseId.ACQUIRE_TRACK, {"grasp_candidate": True}),
        (PhaseId.ACQUIRE_TRACK, {"grasp_verified": True}),
        (PhaseId.ACQUIRE_TRACK, {"object_motion_risk": True}),
    ],
)
def test_active_window_latch_is_monotonic(phase: PhaseId, events: dict[str, bool]) -> None:
    controller = _controller()
    controller.consider_trigger(_evidence(tick=0, phase=phase, **events))

    assert controller.active_window_open is False


@pytest.mark.parametrize("phase", sorted(POST_ACTIVE_WINDOW_PHASES, key=lambda item: item.value))
def test_every_post_active_window_phase_permanently_closes_latch(phase: PhaseId) -> None:
    controller = _controller()

    first = controller.consider_trigger(_evidence(tick=0, phase=phase))
    recovered = controller.consider_trigger(
        _evidence(tick=1, phase=PhaseId.ACQUIRE_TRACK)
    )

    assert first.requestable is False
    assert first.reason is ActiveFrontDecisionReason.DISALLOWED_SOURCE_PHASE
    assert recovered.requestable is False
    assert recovered.reason is ActiveFrontDecisionReason.ACTIVE_WINDOW_CLOSED
    assert controller.active_window_open is False
    blocked = controller.consider_trigger(_evidence(tick=1, phase=PhaseId.ACQUIRE_TRACK))
    assert blocked.reason is ActiveFrontDecisionReason.ACTIVE_WINDOW_CLOSED
    assert controller.active_window_open is False


def test_same_episode_cannot_fake_reset_to_reopen_latch() -> None:
    controller = _controller()
    controller.consider_trigger(
        _evidence(tick=0, phase=PhaseId.ACQUIRE_TRACK, object_contact=True)
    )

    with pytest.raises(ValueError, match="同一个 episode_id"):
        controller.reset_episode("episode-001", episode_generation=2)


@pytest.mark.parametrize(
    "field",
    [
        "action_chunk_cleared",
        "temporal_ensemble_cleared",
        "rtc_overlap_cleared",
        "command_reference_invalidated",
    ],
)
def test_begin_requires_atomic_action_history_invalidation(field: str) -> None:
    controller = _controller()
    request = _request(controller)

    with pytest.raises(ValueError, match="必须原子失效"):
        controller.begin(_reset_receipt(request, **{field: False}))
    assert controller.state is ActiveFrontReobserveState.FAILED_SAFE_HOLD


def test_success_path_requires_exact_order_and_preserves_safe_owners() -> None:
    controller = _controller()
    receipt = _complete_success(controller)

    assert controller.state is ActiveFrontReobserveState.COMPLETE_NO_WRITE
    assert controller.arm_owner is ControllerOwner.SAFE_HOLD
    assert controller.external_camera_owner is ExternalCameraControllerOwner.NONE
    assert receipt.status == "complete-stage1-shadow-no-write"
    assert receipt.source_phase is receipt.resume_phase
    assert receipt.memory_read_count == 0
    assert receipt.memory_write_count == 0
    assert receipt.provider_forward_count == 0
    assert receipt.test_read_count == 0
    assert receipt.action_history_generation_before == 8
    assert receipt.action_history_generation_after_reset == 9
    assert receipt.resumed_action_history_generation == 10
    assert len(receipt.home_observation_sequence_ids) == 4


def test_out_of_order_transition_fails_closed_without_resume() -> None:
    controller = _controller()
    request = _request(controller)
    controller.begin(_reset_receipt(request))

    controller.advance(ActiveFrontSignal.MOVE_COMPLETE)

    assert controller.state is ActiveFrontReobserveState.FAILED_SAFE_HOLD
    receipt = controller.receipt()
    assert receipt.failure is ActiveFrontFailure.STATE_TRANSITION_INVALID
    assert receipt.resumed_action_history_generation is None
    assert receipt.memory_write_count == 0


def test_primitive_identity_drift_fails_before_camera_motion() -> None:
    controller = _controller()
    request = _request(controller)
    controller.begin(_reset_receipt(request))
    controller.advance(ActiveFrontSignal.CAMERA_LEASE_ACQUIRED)

    controller.advance(
        ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
        selected_primitive_id="RIGHT_LOW__YAW_RIGHT",
    )

    assert controller.state is ActiveFrontReobserveState.FAILED_SAFE_HOLD
    assert controller.receipt().failure is ActiveFrontFailure.PRIMITIVE_IDENTITY_MISMATCH


@pytest.mark.parametrize(
    ("safety", "failure"),
    [
        (ActiveFrontSafetyEvidence(arm_hold_pass=False), ActiveFrontFailure.ARM_HOLD_VIOLATION),
        (ActiveFrontSafetyEvidence(tcp_hold_pass=False), ActiveFrontFailure.TCP_HOLD_VIOLATION),
        (
            ActiveFrontSafetyEvidence(gripper_open_hold_pass=False),
            ActiveFrontFailure.GRIPPER_HOLD_VIOLATION,
        ),
        (ActiveFrontSafetyEvidence(contact_absent=False), ActiveFrontFailure.CONTACT_DETECTED),
        (
            ActiveFrontSafetyEvidence(active_window_open=False),
            ActiveFrontFailure.ACTIVE_WINDOW_CLOSED,
        ),
    ],
)
def test_safety_violation_during_motion_requires_failsafe_return(
    safety: ActiveFrontSafetyEvidence,
    failure: ActiveFrontFailure,
) -> None:
    controller = _controller()
    request = _request(controller)
    controller.begin(_reset_receipt(request))
    controller.advance(ActiveFrontSignal.CAMERA_LEASE_ACQUIRED)
    controller.advance(
        ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
        selected_primitive_id=request.selected_primitive_id,
    )

    controller.advance(ActiveFrontSignal.MOVE_COMPLETE, safety=safety)

    assert controller.state is ActiveFrontReobserveState.FAILSAFE_RETURN
    assert controller.external_camera_owner is ExternalCameraControllerOwner.FAILSAFE_RETURN
    controller.complete_failsafe_return(home_verified=True)
    receipt = controller.receipt()
    assert receipt.status == "failed-safe-hold-no-write"
    assert receipt.failure is failure
    assert receipt.memory_write_count == 0


@pytest.mark.parametrize(
    "updates",
    [
        {"camera_at_home": False},
        {"fresh_observation_v2_frame": False},
        {"captured_after_return": False},
        {"contains_alternate_or_motion_rgb": True},
    ],
)
def test_alternate_motion_or_stale_frame_cannot_enter_home_v2_barrier(
    updates: dict[str, bool],
) -> None:
    controller = _controller()
    request = _request(controller)
    controller.begin(_reset_receipt(request))
    _advance_to_home_barrier(controller)

    controller.accept_home_v2_barrier_frame(_home_frame(0, **updates))

    expected_state = (
        ActiveFrontReobserveState.FAILED_SAFE_HOLD
        if updates.get("camera_at_home", True)
        else ActiveFrontReobserveState.FAILSAFE_RETURN
    )
    assert controller.state is expected_state
    assert controller.receipt().failure is ActiveFrontFailure.HOME_FRAME_INVALID


def test_home_barrier_requires_four_unique_fresh_frames() -> None:
    controller = _controller()
    request = _request(controller)
    controller.begin(_reset_receipt(request))
    _advance_to_home_barrier(controller)
    for index in range(3):
        controller.accept_home_v2_barrier_frame(_home_frame(index))
        assert controller.state is ActiveFrontReobserveState.VERIFY_HOME_AND_ARM_HOLD
    controller.accept_home_v2_barrier_frame(_home_frame(2))

    assert controller.state is ActiveFrontReobserveState.FAILED_SAFE_HOLD
    assert controller.receipt().failure is ActiveFrontFailure.HOME_FRAME_DUPLICATE


@pytest.mark.parametrize(
    "resume_update",
    [
        {"generation": 9},
        {"generated_from_fresh_home_v2": False},
        {"stale_action_chunk_resumed": True},
        {
            "home_observation_sequence_ids": (
                "wrong-0",
                "wrong-1",
                "wrong-2",
                "wrong-3",
            )
        },
    ],
)
def test_stale_action_history_cannot_resume(resume_update: dict[str, object]) -> None:
    controller = _controller()
    request = _request(controller)
    controller.begin(_reset_receipt(request))
    _advance_to_home_barrier(controller)
    for index in range(4):
        controller.accept_home_v2_barrier_frame(_home_frame(index))
    controller.advance(
        ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED,
        source_phase=request.resume_phase,
        source_invariants_passed=True,
    )
    valid = ActionHistoryResumeReceipt(
        episode_id=request.episode_id,
        request_id=request.request_id,
        generation=10,
        home_observation_sequence_ids=tuple(f"home-observation-{i}" for i in range(4)),
        generated_from_fresh_home_v2=True,
        stale_action_chunk_resumed=False,
    )

    receipt = controller.complete_no_write_resume(replace(valid, **resume_update))

    assert controller.state is ActiveFrontReobserveState.FAILED_SAFE_HOLD
    assert receipt.failure is ActiveFrontFailure.STALE_ACTION_HISTORY_RESUME
    assert receipt.memory_write_count == 0


def test_replay_produces_stable_receipt_digest() -> None:
    first = _complete_success(_controller())
    second = _complete_success(_controller())

    assert first.as_dict() == second.as_dict()
    assert first.audit_digest == second.audit_digest


@pytest.mark.parametrize(
    ("source_phase", "invariants_passed"),
    [
        (PhaseId.STABILIZE_PREGRASP, True),
        (PhaseId.ACQUIRE_TRACK, False),
        (None, True),
    ],
)
def test_resume_rechecks_original_source_phase_and_invariants(
    source_phase: PhaseId | None,
    invariants_passed: bool,
) -> None:
    controller = _controller()
    request = _request(controller)
    controller.begin(_reset_receipt(request))
    _advance_to_home_barrier(controller)
    for index in range(4):
        controller.accept_home_v2_barrier_frame(_home_frame(index))

    controller.advance(
        ActiveFrontSignal.SOURCE_INVARIANTS_VERIFIED,
        source_phase=source_phase,
        source_invariants_passed=invariants_passed,
    )

    assert controller.state is ActiveFrontReobserveState.FAILED_SAFE_HOLD
    assert controller.receipt().failure is ActiveFrontFailure.SOURCE_INVARIANT_FAILED


@pytest.mark.parametrize(
    "receipt",
    [
        None,
        Stage1ShadowCandidateReceipt(
            request_id="wrong-request",
            candidate_digest="digest",
            shadow_only=True,
            live_memory_write_executed=False,
            provider_forward_count=0,
        ),
        Stage1ShadowCandidateReceipt(
            request_id="episode-001-active-front-01",
            candidate_digest="digest",
            shadow_only=False,
            live_memory_write_executed=False,
            provider_forward_count=0,
        ),
        Stage1ShadowCandidateReceipt(
            request_id="episode-001-active-front-01",
            candidate_digest="digest",
            shadow_only=True,
            live_memory_write_executed=True,
            provider_forward_count=0,
        ),
        Stage1ShadowCandidateReceipt(
            request_id="episode-001-active-front-01",
            candidate_digest="digest",
            shadow_only=True,
            live_memory_write_executed=False,
            provider_forward_count=1,
        ),
    ],
)
def test_stage1_candidate_cannot_hide_provider_or_live_memory_write(
    receipt: Stage1ShadowCandidateReceipt | None,
) -> None:
    controller = _controller()
    request = _request(controller)
    controller.begin(_reset_receipt(request))
    controller.advance(ActiveFrontSignal.CAMERA_LEASE_ACQUIRED)
    controller.advance(
        ActiveFrontSignal.FROZEN_PRIMITIVE_SELECTED,
        selected_primitive_id=request.selected_primitive_id,
    )
    controller.advance(ActiveFrontSignal.MOVE_COMPLETE)
    controller.advance(ActiveFrontSignal.SETTLE_COMPLETE)
    controller.advance(ActiveFrontSignal.COLLECTION_COMPLETE)

    controller.advance(
        ActiveFrontSignal.SHADOW_CANDIDATE_STAGED,
        shadow_candidate_receipt=receipt,
    )

    assert controller.state is ActiveFrontReobserveState.FAILSAFE_RETURN
    assert controller.receipt().memory_write_count == 0
