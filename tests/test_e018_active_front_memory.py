from __future__ import annotations

import json
from dataclasses import fields, replace
from pathlib import Path

import numpy as np
import pytest

import robot_vla.precision.active_front_memory as active_front_memory_module
from robot_vla.executive.contracts import PhaseId
from robot_vla.precision.active_front_camera import ExternalCameraMotionState
from robot_vla.precision.active_front_memory import (
    ActiveFrontSourceRecheckEvidence,
    ActiveFrontStage2MemoryOrchestrator,
    PendingActiveViewState,
)
from robot_vla.precision.active_front_memory_provider import (
    ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV,
    ACTIVE_FRONT_HOME_ORIENTATION_TOLERANCE_RAD,
    ACTIVE_FRONT_HOME_POSITION_TOLERANCE_M,
    ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
    ACTIVE_FRONT_PROHIBITED_MEMORY_WRITE_PRIMITIVE_IDS,
    ACTIVE_FRONT_QUALIFIED_SHADOW_PRIMITIVE_IDS,
    ACTIVE_FRONT_SCORE_SEMANTICS,
    ActiveFrontScoreComponents,
    ActiveFrontStage2Config,
    ActiveFrontStage2FrameEvidence,
    ActiveFrontStage2ProviderAdapter,
    ActiveFrontStage2ProviderIdentity,
    PassiveBaselineEvidence,
    PassiveHomeScoreEvidence,
    build_stage2_object_memory_config,
    d049_home_baseline_provider_identity,
    d049_primary_provider_identity,
)
from robot_vla.precision.active_front_reobserve import (
    ActionHistoryResetReceipt,
    ActionHistoryResumeReceipt,
    ActiveFrontReobserveRequest,
    ActiveFrontSafetyEvidence,
    ActiveFrontTriggerReason,
    HomeV2BarrierFrame,
)
from robot_vla.precision.object_memory import (
    ExplicitObjectStateMemory,
    ObjectCandidateWindowVerifier,
    ObjectMeasurement,
    ObjectMemoryMode,
    ObjectMemorySafetyContext,
    ObjectStateRequirement,
    resolve_object_state,
)
from robot_vla.precision.object_observability import OBJECT_WRITE_SCORE_SEMANTICS


def _safety(**updates: bool) -> ObjectMemorySafetyContext:
    values = {
        "pregrasp_window_open": True,
        "gripper_open": True,
        "controller_tracking_valid": True,
        "object_contact_detected": False,
        "gripper_close_commanded": False,
        "grasp_candidate": False,
        "grasp_verified": False,
        "object_maybe_moved": False,
    }
    values.update(updates)
    return ObjectMemorySafetyContext(**values)


def _request(
    *,
    episode_id: str = "episode-a",
    generation: int = 1,
    primitive_id: str = ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID,
    trigger_timestamp_s: float = 0.0,
) -> ActiveFrontReobserveRequest:
    return ActiveFrontReobserveRequest(
        episode_id=episode_id,
        episode_generation=generation,
        request_id=f"{episode_id}-active-front-01",
        source_phase=PhaseId.ACQUIRE_TRACK,
        resume_phase=PhaseId.ACQUIRE_TRACK,
        trigger_tick=3,
        trigger_timestamp_s=trigger_timestamp_s,
        trigger_reason=ActiveFrontTriggerReason.OBJECT_OCCLUSION,
        attempt_index=1,
        selected_primitive_id=primitive_id,
        camera_command_sequence_id=f"{episode_id}-camera-00",
    )


def _reset_receipt(request: ActiveFrontReobserveRequest) -> ActionHistoryResetReceipt:
    return ActionHistoryResetReceipt(
        episode_id=request.episode_id,
        request_id=request.request_id,
        reset_control_tick=request.trigger_tick,
        generation_before=8,
        generation_after=9,
        action_chunk_cleared=True,
        temporal_ensemble_cleared=True,
        rtc_overlap_cleared=True,
        command_reference_invalidated=True,
    )


def _baseline(
    *,
    home_score: float | None = 0.50,
    home_identity_updates: dict[str, object] | None = None,
    score_semantics: str | None = ACTIVE_FRONT_SCORE_SEMANTICS,
    episode_id: str = "episode-a",
    generation: int = 1,
) -> PassiveBaselineEvidence:
    identity = d049_home_baseline_provider_identity()
    if home_identity_updates:
        identity = replace(identity, **home_identity_updates)
    home_front = None
    if home_score is not None:
        transform = np.asarray(
            ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV,
            dtype=np.float64,
        )
        home_front = PassiveHomeScoreEvidence(
            episode_id=episode_id,
            episode_generation=generation,
            request_id=f"{episode_id}-active-front-01",
            observation_sequence_id=f"{episode_id}-home-baseline-00",
            model_input_digest="4" * 64,
            provider_output_digest="5" * 64,
            provider_identity=identity,
            viewpoint_primitive_id="HOME__CENTER",
            camera_motion_state=ExternalCameraMotionState.HOME_ANCHOR,
            settled=True,
            score_components=_components(home_score),
            stored_write_score=home_score,
            geometry_valid=True,
            control_timestamp_s=0.0,
            rgb_timestamp_s=0.0,
            camera_pose_timestamp_s=0.0,
            tcp_pose_timestamp_s=0.0,
            base_from_external_camera_cv=transform,
            score_semantics=(
                ACTIVE_FRONT_SCORE_SEMANTICS
                if score_semantics is None
                else score_semantics
            ),
        )
    return PassiveBaselineEvidence(
        episode_id=episode_id,
        episode_generation=generation,
        request_id=f"{episode_id}-active-front-01",
        timestamp_s=0.0,
        wrist_object_measurement_usable=False,
        wrist_evidence_identity_sha256="1" * 64,
        home_front=home_front,
        object_memory_navigation_state_available=False,
        object_memory_age_s=None,
        object_memory_source_identity=None,
    )


def _components(score: float = 0.70) -> ActiveFrontScoreComponents:
    return ActiveFrontScoreComponents(
        object_visibility_probability=score,
        projection_validity_probability=1.0,
        object_mask_probability=0.98,
        goal_mask_probability=0.01,
        object_normalized_entropy=0.2,
        object_sigma_xy_px=(0.1, 0.1),
    )


def _frame(
    index: int,
    *,
    timestamp_s: float | None = None,
    position_x: float | None = None,
    variance: float | None = None,
    write_score: float = 0.70,
    identity_updates: dict[str, object] | None = None,
    camera_motion_state: ExternalCameraMotionState = ExternalCameraMotionState.COLLECT,
    settled: bool = True,
    qualification_only: bool = False,
    projection_valid: bool = True,
    in_fov: bool = True,
    observable: bool = True,
    structurally_eligible: bool = True,
    episode_id: str = "episode-a",
    generation: int = 1,
) -> ActiveFrontStage2FrameEvidence:
    timestamp = index * 0.05 if timestamp_s is None else timestamp_s
    position = 0.400 + index * 0.001 if position_x is None else position_x
    covariance_value = (index + 1) ** 2 * 1e-6 if variance is None else variance
    identity = d049_primary_provider_identity()
    if identity_updates:
        identity = replace(identity, **identity_updates)
    transform = np.eye(4, dtype=np.float64)
    transform[:3, 3] = (0.30, -0.16, 0.48)
    return ActiveFrontStage2FrameEvidence(
        episode_id=episode_id,
        episode_generation=generation,
        request_id=f"{episode_id}-active-front-01",
        source_phase=PhaseId.ACQUIRE_TRACK,
        observation_sequence_id=f"alternate-collect-{index}",
        model_input_digest=f"{(index + 6) % 16:x}" * 64,
        provider_output_digest=f"{(index + 10) % 16:x}" * 64,
        provider_identity=identity,
        camera_motion_state=camera_motion_state,
        settled=settled,
        control_timestamp_s=timestamp,
        rgb_timestamp_s=timestamp,
        camera_pose_timestamp_s=timestamp,
        tcp_pose_timestamp_s=timestamp,
        base_from_external_camera_cv=transform,
        position_base_m=(position, 0.1, 0.02),
        covariance_base_m2=np.eye(3, dtype=np.float64) * covariance_value,
        measurement_confidence=write_score,
        write_score=write_score,
        score_components=_components(write_score),
        projection_valid=projection_valid,
        in_fov=in_fov,
        observable=observable,
        geometry_valid=True,
        structurally_eligible=structurally_eligible,
        deployable_free_static_safe=True,
        qualification_only=qualification_only,
    )


def _orchestrator(
    *,
    enabled: bool = True,
    baseline: PassiveBaselineEvidence | None = None,
) -> ActiveFrontStage2MemoryOrchestrator:
    config = (
        ActiveFrontStage2Config.development()
        if enabled
        else ActiveFrontStage2Config()
    )
    memory = ExplicitObjectStateMemory(build_stage2_object_memory_config())
    orchestrator = ActiveFrontStage2MemoryOrchestrator(memory, config=config)
    orchestrator.reset_episode("episode-a", episode_generation=1)
    if enabled:
        request = _request()
        orchestrator.begin_collection(
            request,
            reset_receipt=_reset_receipt(request),
            baseline=baseline or _baseline(),
        )
    return orchestrator


def _collect_valid(orchestrator: ActiveFrontStage2MemoryOrchestrator) -> None:
    for index in range(3):
        adaptation = orchestrator.observe_collect_frame(_frame(index), safety=_safety())
        assert adaptation.eligible
    assert orchestrator.state is PendingActiveViewState.VERIFIED_PENDING


def _pass_home_barrier(orchestrator: ActiveFrontStage2MemoryOrchestrator) -> None:
    candidate = orchestrator.pending_candidate
    assert candidate is not None
    orchestrator.mark_returning_home(
        timestamp_s=0.15,
        candidate_digest=candidate.digest,
    )
    for index in range(4):
        orchestrator.accept_home_v2_barrier_frame(
            HomeV2BarrierFrame(
                observation_sequence_id=f"home-{index}",
                camera_at_home=True,
                fresh_observation_v2_frame=True,
                captured_after_return=True,
                contains_alternate_or_motion_rgb=False,
            ),
            timestamp_s=0.20 + index * 0.05,
        )
    assert orchestrator.state is PendingActiveViewState.HOME_BARRIER_PASSED


def _return_failed_candidate_home(
    orchestrator: ActiveFrontStage2MemoryOrchestrator,
    *,
    return_timestamp_s: float = 0.15,
) -> None:
    candidate = orchestrator.pending_candidate
    orchestrator.mark_returning_home(
        timestamp_s=return_timestamp_s,
        candidate_digest=None if candidate is None else candidate.digest,
    )
    assert orchestrator.state is PendingActiveViewState.RETURNING_HOME_NO_COMMIT
    assert orchestrator.camera_lease_held
    for index in range(4):
        orchestrator.accept_home_v2_barrier_frame(
            HomeV2BarrierFrame(
                observation_sequence_id=f"failed-home-{index}",
                camera_at_home=True,
                fresh_observation_v2_frame=True,
                captured_after_return=True,
                contains_alternate_or_motion_rgb=False,
            ),
            timestamp_s=return_timestamp_s + 0.05 + index * 0.05,
        )
    assert (
        orchestrator.state
        is PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD
    )
    assert not orchestrator.camera_lease_held
    assert orchestrator.memory_write_count == 0


def _recheck(
    orchestrator: ActiveFrontStage2MemoryOrchestrator,
    *,
    timestamp_s: float = 0.40,
    direct_wrist: bool = False,
    active_window_open: bool = True,
    safety: ActiveFrontSafetyEvidence | None = None,
) -> bool:
    candidate = orchestrator.pending_candidate
    assert candidate is not None
    return orchestrator.recheck_source(
        ActiveFrontSourceRecheckEvidence(
            episode_id="episode-a",
            episode_generation=1,
            request_id="episode-a-active-front-01",
            candidate_digest=candidate.digest,
            timestamp_s=timestamp_s,
            source_phase=PhaseId.ACQUIRE_TRACK,
            camera_at_home=True,
            source_invariants_passed=True,
            active_window_open=active_window_open,
            qualified_direct_wrist_measurement_usable=direct_wrist,
            qualified_direct_wrist_evidence_identity_sha256="9" * 64,
        ),
        safety=safety,
    )


def _commit(orchestrator: ActiveFrontStage2MemoryOrchestrator, *, timestamp_s: float = 0.40):
    candidate = orchestrator.pending_candidate
    assert candidate is not None
    return orchestrator.commit(
        candidate_digest=candidate.digest,
        commit_timestamp_s=timestamp_s,
        safety=_safety(),
    )


def _drive_to_commit() -> ActiveFrontStage2MemoryOrchestrator:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    _pass_home_barrier(orchestrator)
    assert _recheck(orchestrator)
    _commit(orchestrator)
    return orchestrator


def test_stage2_is_disabled_by_default_without_changing_p0_config() -> None:
    orchestrator = _orchestrator(enabled=False)
    request = _request()

    with pytest.raises(RuntimeError, match="默认关闭"):
        orchestrator.begin_collection(
            request,
            reset_receipt=_reset_receipt(request),
            baseline=_baseline(),
        )

    assert orchestrator.memory.state.mode is ObjectMemoryMode.UNINITIALIZED
    assert orchestrator.memory_write_count == 0


@pytest.mark.parametrize(
    ("field_name", "tampered_value", "message"),
    [
        ("write_score", 0.99, "stored write_score"),
        ("measurement_confidence", 0.99, "measurement_confidence"),
        ("observable", False, "stored observable"),
        ("structurally_eligible", False, "stored structurally_eligible"),
    ],
)
def test_frame_rejects_tampered_derived_provider_fields(
    field_name: str,
    tampered_value: object,
    message: str,
) -> None:
    frame = _frame(0)

    with pytest.raises(ValueError, match=message):
        replace(frame, **{field_name: tampered_value})


def test_frame_score_is_recomputed_from_exact_object_write_semantics() -> None:
    frame = _frame(0)
    assert ACTIVE_FRONT_SCORE_SEMANTICS == OBJECT_WRITE_SCORE_SEMANTICS
    assert frame.score_components.radial_sigma_px == pytest.approx(2**0.5 * 0.1)
    assert frame.write_evidence.score == pytest.approx(0.70)

    with pytest.raises(ValueError, match="stored write_score"):
        replace(frame, score_components=_components(0.65))
    with pytest.raises(ValueError, match="read-only"):
        frame.base_from_external_camera_cv[0, 3] = 9.0

    baseline = _baseline()
    assert baseline.home_front is not None
    with pytest.raises(ValueError, match="read-only"):
        baseline.home_front.base_from_external_camera_cv[0, 3] = 9.0


def test_high_mask_projection_invalid_frame_is_deterministically_rejected() -> None:
    orchestrator = _orchestrator()
    frame = _frame(
        0,
        projection_valid=False,
        in_fov=False,
        observable=False,
        structurally_eligible=False,
    )

    adaptation = orchestrator.observe_collect_frame(frame, safety=_safety())

    assert not adaptation.eligible
    assert "projection_invalid" in adaptation.rejection_reasons
    assert "not_observable" in adaptation.rejection_reasons
    assert "structurally_ineligible" in adaptation.rejection_reasons
    assert orchestrator.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    _return_failed_candidate_home(orchestrator)


def test_correlated_covariance_uses_spectral_maximum_std() -> None:
    covariance = np.asarray(
        (
            (3e-4, 2e-4, 0.0),
            (2e-4, 3e-4, 0.0),
            (0.0, 0.0, 1e-6),
        ),
        dtype=np.float64,
    )
    assert np.sqrt(np.diag(covariance).max()) < 0.02
    assert np.sqrt(np.linalg.eigvalsh(covariance).max()) > 0.02
    orchestrator = _orchestrator()

    adaptation = orchestrator.observe_collect_frame(
        replace(_frame(0), covariance_base_m2=covariance),
        safety=_safety(),
    )

    assert not adaptation.eligible
    assert "measurement_uncertain" in adaptation.rejection_reasons
    assert orchestrator.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    _return_failed_candidate_home(orchestrator)


def test_actual_safety_context_cannot_be_bypassed_by_frame_safe_flag() -> None:
    orchestrator = _orchestrator()

    adaptation = orchestrator.observe_collect_frame(
        _frame(0),
        safety=_safety(object_contact_detected=True),
    )

    assert not adaptation.eligible
    assert "object_contact_detected" in adaptation.rejection_reasons
    assert orchestrator.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    assert orchestrator.memory.state.mode is ObjectMemoryMode.INVALID
    assert orchestrator.memory.state.accepted_update_count == 0
    _return_failed_candidate_home(orchestrator)


def test_success_uses_third_frame_without_averaging_and_delays_atomic_commit() -> None:
    orchestrator = _orchestrator()
    frames = [_frame(index) for index in range(3)]
    for frame in frames:
        orchestrator.observe_collect_frame(frame, safety=_safety())

    candidate = orchestrator.pending_candidate
    assert candidate is not None and candidate.commit_eligible
    assert candidate.model_input_digests == tuple(
        frame.model_input_digest for frame in frames
    )
    assert candidate.provider_output_digests == tuple(
        frame.provider_output_digest for frame in frames
    )
    assert candidate.final_measurement.position_base_m == pytest.approx((0.402, 0.1, 0.02))
    assert candidate.final_measurement.position_base_m != pytest.approx((0.401, 0.1, 0.02))
    assert np.asarray(candidate.final_measurement.covariance_base_m2) == pytest.approx(
        np.eye(3) * 9e-6
    )
    assert orchestrator.memory.state.mode is ObjectMemoryMode.UNINITIALIZED
    assert orchestrator.memory_write_count == 0

    _pass_home_barrier(orchestrator)
    assert _recheck(orchestrator)
    receipt = _commit(orchestrator)

    state = orchestrator.memory.state
    assert receipt.memory_write_count == 1
    assert receipt.source_recheck_wrist_evidence_identity_sha256 == "9" * 64
    assert receipt.home_observation_sequence_ids == tuple(
        f"home-{index}" for index in range(4)
    )
    assert receipt.home_observation_timestamps_s == pytest.approx(
        (0.20, 0.25, 0.30, 0.35)
    )
    assert len(receipt.home_frame_digests) == len(set(receipt.home_frame_digests)) == 4
    assert state.position_base_m == pytest.approx((0.402, 0.1, 0.02))
    assert np.asarray(state.covariance_base_m2) == pytest.approx(np.eye(3) * 9e-6)
    assert state.last_observed_timestamp_s == pytest.approx(0.10)
    assert state.state_timestamp_s == pytest.approx(0.40)
    assert not state.observable_now
    navigation = resolve_object_state(
        orchestrator.commit_update,
        requirement=ObjectStateRequirement.NAVIGATION,
    )
    contact = resolve_object_state(
        orchestrator.commit_update,
        requirement=ObjectStateRequirement.CONTACT_READY,
    )
    assert navigation.available and navigation.memory_only
    assert not navigation.contact_authorized
    assert not contact.available and not contact.contact_authorized


def test_frame_digest_binds_model_input_and_provider_output() -> None:
    frame = _frame(0)

    assert replace(frame, model_input_digest="e" * 64).frame_digest != frame.frame_digest
    assert (
        replace(frame, provider_output_digest="f" * 64).frame_digest
        != frame.frame_digest
    )


@pytest.mark.parametrize(
    ("identity_updates", "reason"),
    [
        ({"primitive_id": "LEFT_LOW__CENTER"}, "primitive_not_primary"),
        ({"checkpoint_sha256": "0" * 64}, "provider_family_identity_mismatch"),
        ({"calibration_identity_sha256": "0" * 64}, "calibration_identity_mismatch"),
    ],
)
def test_wrong_primitive_provider_or_calibration_rejects_before_pending(
    identity_updates: dict[str, object],
    reason: str,
) -> None:
    orchestrator = _orchestrator()

    adaptation = orchestrator.observe_collect_frame(
        _frame(0, identity_updates=identity_updates),
        safety=_safety(),
    )

    assert not adaptation.eligible
    assert reason in adaptation.rejection_reasons
    assert orchestrator.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    assert orchestrator.camera_lease_held
    assert orchestrator.memory.state.mode is ObjectMemoryMode.UNINITIALIZED
    assert orchestrator.memory_write_count == 0
    _return_failed_candidate_home(orchestrator)


@pytest.mark.parametrize(
    ("updates", "reason"),
    [
        (
            {"camera_motion_state": ExternalCameraMotionState.MOVE_TO_VIEW},
            "motion_or_non_collect_frame",
        ),
        ({"settled": False}, "frame_not_settled"),
        ({"qualification_only": True}, "qualification_only_adapter_forbidden"),
    ],
)
def test_motion_unsettled_or_qualification_only_frame_cannot_enter_candidate(
    updates: dict[str, object],
    reason: str,
) -> None:
    orchestrator = _orchestrator()

    adaptation = orchestrator.observe_collect_frame(_frame(0, **updates), safety=_safety())

    assert not adaptation.eligible
    assert reason in adaptation.rejection_reasons
    assert orchestrator.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    assert orchestrator.camera_lease_held
    assert orchestrator.memory_write_count == 0
    _return_failed_candidate_home(orchestrator)


def test_home_raw_score_is_baseline_only_and_can_never_be_direct_measurement() -> None:
    baseline = _baseline()
    assert baseline.home_front is not None
    with pytest.raises(ValueError, match="只能作 baseline"):
        replace(baseline.home_front, object_measurement_usable=True)

    recheck_fields = {field.name for field in fields(ActiveFrontSourceRecheckEvidence)}
    assert "direct_home_front_measurement_usable" not in recheck_fields
    assert "qualified_direct_wrist_measurement_usable" in recheck_fields


def test_home_baseline_score_pose_time_and_identity_are_digest_bound() -> None:
    baseline = _baseline()
    home = baseline.home_front
    assert home is not None
    assert home.pose_valid and home.timestamp_valid
    assert home.write_score == pytest.approx(0.50)
    assert home.frame_digest == baseline.home_front_frame_digest

    with pytest.raises(ValueError, match="stored write_score"):
        replace(home, stored_write_score=0.10)

    moved_pose = np.asarray(home.base_from_external_camera_cv).copy()
    moved_pose[0, 3] += 0.001
    variants = (
        replace(home, observation_sequence_id="different-home-frame"),
        replace(home, model_input_digest="6" * 64),
        replace(home, provider_output_digest="7" * 64),
        replace(home, base_from_external_camera_cv=moved_pose),
        replace(
            home,
            score_components=_components(0.49),
            stored_write_score=0.49,
        ),
        replace(
            home,
            control_timestamp_s=0.001,
            rgb_timestamp_s=0.001,
            camera_pose_timestamp_s=0.001,
            tcp_pose_timestamp_s=0.001,
        ),
        replace(
            home,
            provider_identity=replace(
                home.provider_identity,
                checkpoint_sha256="3" * 64,
            ),
        ),
    )
    assert all(value.frame_digest != home.frame_digest for value in variants)


@pytest.mark.parametrize(
    ("home_update", "reason"),
    [
        (
            {"provider_identity": d049_primary_provider_identity()},
            "baseline_home_provider_identity_mismatch",
        ),
        (
            {"viewpoint_primitive_id": ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID},
            "baseline_home_capture_invalid",
        ),
        (
            {"camera_motion_state": ExternalCameraMotionState.COLLECT},
            "baseline_home_capture_invalid",
        ),
        ({"settled": False}, "baseline_home_capture_invalid"),
    ],
)
def test_home_baseline_requires_frozen_home_identity_and_capture_state(
    home_update: dict[str, object],
    reason: str,
) -> None:
    baseline = _baseline()
    assert baseline.home_front is not None
    bad_baseline = replace(
        baseline,
        home_front=replace(baseline.home_front, **home_update),
    )
    orchestrator = _orchestrator(baseline=bad_baseline)

    for index in range(3):
        orchestrator.observe_collect_frame(_frame(index), safety=_safety())

    assert reason in orchestrator.terminal_reasons
    assert orchestrator.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    _return_failed_candidate_home(orchestrator)


def test_home_identity_and_actual_pose_are_mechanically_frozen() -> None:
    baseline = _baseline()
    home = baseline.home_front
    assert home is not None
    identity = d049_home_baseline_provider_identity()
    primary = d049_primary_provider_identity()
    assert identity.sha256 == (
        "d2323bc30a64341e49b73576b26894702ec0478f316cdb138241d8d78187401e"
    )
    assert identity.provider_family_sha256 == primary.provider_family_sha256
    assert home.provider_identity == identity
    assert home.home_capture_valid and home.pose_valid

    moved_pose = np.asarray(home.base_from_external_camera_cv).copy()
    moved_pose[0, 3] += 0.001
    moved_home = replace(home, base_from_external_camera_cv=moved_pose)
    assert not moved_home.pose_valid
    bad = _orchestrator(baseline=replace(baseline, home_front=moved_home))
    for index in range(3):
        bad.observe_collect_frame(_frame(index), safety=_safety())
    assert "baseline_pose_invalid" in bad.terminal_reasons
    _return_failed_candidate_home(bad)


def test_provider_identity_has_exact_nonduplicated_field_set() -> None:
    assert tuple(field.name for field in fields(ActiveFrontStage2ProviderIdentity)) == (
        "qualification_artifact_id",
        "qualification_source_identity_sha256",
        "qualification_config_raw_sha256",
        "qualification_config_internal_sha256",
        "qualification_result_receipt_raw_sha256",
        "qualification_result_receipt_internal_sha256",
        "qualification_result_verification_sha256",
        "candidate_id",
        "checkpoint_epoch",
        "checkpoint_sha256",
        "checkpoint_parameter_sha256",
        "checkpoint_provenance_sha256",
        "model_config_sha256",
        "proprio_stats_sha256",
        "proprio_normalizer_sha256",
        "finger_force_stats_sha256",
        "finger_force_normalizer_sha256",
        "qualification_adapter_config_sha256",
        "calibration_config_raw_sha256",
        "calibration_config_internal_sha256",
        "calibration_result_receipt_raw_sha256",
        "calibration_result_receipt_internal_sha256",
        "calibration_viewpoints_raw_sha256",
        "calibration_identity_sha256",
        "calibration_scale_factor",
        "write_threshold",
        "primitive_id",
        "geometric_motion_provider_id",
        "source_training_camera",
        "source_camera",
        "actual_pose_source",
        "role_substitution_semantics",
        "qualification_input_schema_version",
        "stage2_frame_schema_version",
        "execution_mode",
        "score_semantics",
        "version",
    )


@pytest.mark.parametrize(
    "home_update",
    [
        {"base_from_external_camera_cv": None},
        {"rgb_timestamp_s": 0.02},
        {"actual_pose_source": "caller-asserted-pose/v1"},
    ],
)
def test_home_baseline_pose_and_timestamp_validity_are_recomputed(
    home_update: dict[str, object],
) -> None:
    baseline = _baseline()
    assert baseline.home_front is not None
    bad_home = replace(baseline.home_front, **home_update)
    bad_baseline = replace(baseline, home_front=bad_home)
    orchestrator = _orchestrator(baseline=bad_baseline)

    for index in range(3):
        orchestrator.observe_collect_frame(_frame(index), safety=_safety())

    assert orchestrator.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    assert any(
        reason in orchestrator.terminal_reasons
        for reason in ("baseline_pose_invalid", "baseline_timestamp_invalid")
    )
    _return_failed_candidate_home(orchestrator)


@pytest.mark.parametrize(
    "baseline",
    [
        _baseline(home_score=None),
        _baseline(home_identity_updates={"checkpoint_sha256": "0" * 64}),
        _baseline(score_semantics="different-score-semantics/v1"),
    ],
)
def test_unavailable_or_incomparable_home_baseline_is_shadow_only(
    baseline: PassiveBaselineEvidence,
) -> None:
    orchestrator = _orchestrator(baseline=baseline)

    for index in range(3):
        orchestrator.observe_collect_frame(_frame(index), safety=_safety())

    candidate = orchestrator.pending_candidate
    assert candidate is not None
    assert not candidate.commit_eligible
    assert orchestrator.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    assert orchestrator.camera_lease_held
    assert orchestrator.memory_write_count == 0
    assert orchestrator.memory.state.mode is ObjectMemoryMode.UNINITIALIZED
    _return_failed_candidate_home(orchestrator)


def test_begin_collection_rejects_future_or_false_memory_baseline() -> None:
    with pytest.raises(ValueError, match="trigger timestamp"):
        _orchestrator(baseline=replace(_baseline(), timestamp_s=0.001))

    config = ActiveFrontStage2Config.development()
    memory = ExplicitObjectStateMemory(build_stage2_object_memory_config())
    orchestrator = ActiveFrontStage2MemoryOrchestrator(memory, config=config)
    orchestrator.reset_episode("episode-a", episode_generation=1)
    verifier = ObjectCandidateWindowVerifier(memory.config)
    verifier.reset("episode-a")
    decision = None
    for index in range(3):
        timestamp = 0.10 + index * 0.05
        measurement = ObjectMeasurement(
            timestamp_s=timestamp,
            rgb_timestamp_s=timestamp,
            camera_pose_timestamp_s=timestamp,
            tcp_pose_timestamp_s=timestamp,
            position_base_m=(0.4, 0.1, 0.02),
            covariance_base_m2=np.eye(3) * 1e-6,
            confidence=0.9,
            projection_valid=True,
            in_fov=True,
            observable=True,
            geometry_valid=True,
            write_gate_passed=True,
            source_camera="base_camera",
            source_model_identity=(
                d049_primary_provider_identity().object_memory_source_identity
            ),
        )
        decision = verifier.observe(
            measurement,
            episode_id="episode-a",
            safety=_safety(),
        )
    assert decision is not None and decision.verified
    accepted = memory.update(decision, episode_id="episode-a", safety=_safety())
    assert accepted.measurement_accepted

    request = _request(trigger_timestamp_s=1.0)
    false_unavailable = replace(_baseline(), timestamp_s=1.0)
    with pytest.raises(ValueError, match="snapshot 与实际 state"):
        orchestrator.begin_collection(
            request,
            reset_receipt=_reset_receipt(request),
            baseline=false_unavailable,
        )


def test_begin_collection_rejects_safety_invalid_memory_before_camera_lease() -> None:
    config = ActiveFrontStage2Config.development()
    memory = ExplicitObjectStateMemory(build_stage2_object_memory_config())
    orchestrator = ActiveFrontStage2MemoryOrchestrator(memory, config=config)
    orchestrator.reset_episode("episode-a", episode_generation=1)
    memory.invalidate_for_safety(
        episode_id="episode-a",
        timestamp_s=0.0,
        reasons=("object_contact_detected",),
    )
    request = _request()

    with pytest.raises(ValueError, match="safety-invalid"):
        orchestrator.begin_collection(
            request,
            reset_receipt=_reset_receipt(request),
            baseline=_baseline(),
        )

    assert not orchestrator.camera_lease_held
    assert not orchestrator.request_active
    assert orchestrator.attempt_count == 0


def test_begin_collection_accepts_reset_uninitialized_memory() -> None:
    config = ActiveFrontStage2Config.development()
    memory = ExplicitObjectStateMemory(build_stage2_object_memory_config())
    orchestrator = ActiveFrontStage2MemoryOrchestrator(memory, config=config)
    state = orchestrator.reset_episode("episode-a", episode_generation=1)
    request = _request()

    assert state.mode is ObjectMemoryMode.UNINITIALIZED
    assert state.invalid_reasons == ("memory_uninitialized",)

    orchestrator.begin_collection(
        request,
        reset_receipt=_reset_receipt(request),
        baseline=_baseline(),
    )

    assert orchestrator.state is PendingActiveViewState.COLLECTING
    assert orchestrator.camera_lease_held
    assert orchestrator.request_active
    assert orchestrator.attempt_count == 1


def test_gain_gap_and_spread_thresholds_fail_closed() -> None:
    low_gain = _orchestrator(baseline=_baseline(home_score=0.66))
    for index in range(3):
        low_gain.observe_collect_frame(_frame(index), safety=_safety())
    assert "information_gain_below_threshold" in low_gain.terminal_reasons
    assert low_gain.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    _return_failed_candidate_home(low_gain)

    gap = _orchestrator()
    for index, timestamp in enumerate((0.0, 0.08, 0.13)):
        gap.observe_collect_frame(_frame(index, timestamp_s=timestamp), safety=_safety())
    assert gap.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    assert "candidate_too_short" in gap.terminal_reasons
    _return_failed_candidate_home(gap, return_timestamp_s=0.18)

    spread = _orchestrator()
    for index, position in enumerate((0.400, 0.401, 0.406)):
        spread.observe_collect_frame(_frame(index, position_x=position), safety=_safety())
    assert spread.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    assert "candidate_position_inconsistent" in spread.terminal_reasons
    _return_failed_candidate_home(spread)


@pytest.mark.parametrize("timestamps", [(0.05, 0.05), (0.05, 0.04)])
def test_duplicate_or_out_of_order_collect_timestamp_requires_home_recovery(
    timestamps: tuple[float, float],
) -> None:
    orchestrator = _orchestrator()
    first = orchestrator.observe_collect_frame(
        _frame(0, timestamp_s=timestamps[0]),
        safety=_safety(),
    )
    second = orchestrator.observe_collect_frame(
        _frame(1, timestamp_s=timestamps[1]),
        safety=_safety(),
    )

    assert first.eligible and not second.eligible
    assert "collect_frame_timestamp_not_increasing" in second.rejection_reasons
    assert orchestrator.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    assert orchestrator.camera_lease_held
    assert orchestrator.memory_write_count == 0
    _return_failed_candidate_home(orchestrator, return_timestamp_s=0.10)


def test_replayed_input_and_output_with_new_sequence_id_requires_home_recovery() -> None:
    orchestrator = _orchestrator()
    first_frame = _frame(0)
    assert orchestrator.observe_collect_frame(first_frame, safety=_safety()).eligible
    replay = replace(
        _frame(1),
        model_input_digest=first_frame.model_input_digest,
        provider_output_digest=first_frame.provider_output_digest,
    )

    adaptation = orchestrator.observe_collect_frame(replay, safety=_safety())

    assert not adaptation.eligible
    assert "duplicate_model_input_digest" in adaptation.rejection_reasons
    assert "duplicate_provider_output_digest" in adaptation.rejection_reasons
    assert orchestrator.state is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT
    assert orchestrator.memory_write_count == 0
    _return_failed_candidate_home(orchestrator)


def test_pending_older_than_2_5_seconds_expires_without_memory_write() -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    candidate = orchestrator.pending_candidate
    assert candidate is not None

    orchestrator.mark_returning_home(
        timestamp_s=candidate.created_timestamp_s + 2.500001,
        candidate_digest=candidate.digest,
    )

    assert orchestrator.state is PendingActiveViewState.RETURNING_HOME_NO_COMMIT
    assert "pending_candidate_expired" in orchestrator.terminal_reasons
    assert orchestrator.camera_lease_held
    assert orchestrator.memory.state.mode is ObjectMemoryMode.UNINITIALIZED
    assert orchestrator.memory_write_count == 0
    for index in range(4):
        orchestrator.accept_home_v2_barrier_frame(
            HomeV2BarrierFrame(
                observation_sequence_id=f"expired-home-{index}",
                camera_at_home=True,
                fresh_observation_v2_frame=True,
                captured_after_return=True,
                contains_alternate_or_motion_rgb=False,
            ),
            timestamp_s=candidate.created_timestamp_s + 2.55 + index * 0.05,
        )
    assert (
        orchestrator.state
        is PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD
    )
    assert not orchestrator.camera_lease_held


def test_source_recheck_cannot_run_before_four_fresh_home_frames() -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    candidate = orchestrator.pending_candidate
    assert candidate is not None
    orchestrator.mark_returning_home(timestamp_s=0.15, candidate_digest=candidate.digest)
    for index in range(3):
        orchestrator.accept_home_v2_barrier_frame(
            HomeV2BarrierFrame(
                observation_sequence_id=f"home-{index}",
                camera_at_home=True,
                fresh_observation_v2_frame=True,
                captured_after_return=True,
                contains_alternate_or_motion_rgb=False,
            ),
            timestamp_s=0.20 + index * 0.05,
        )

    with pytest.raises(RuntimeError, match="完整 HOME barrier"):
        _recheck(orchestrator)

    assert orchestrator.state is PendingActiveViewState.RETURNING_HOME
    assert orchestrator.memory_write_count == 0


def test_invalid_home_frame_is_not_counted_and_lease_waits_for_four_valid_frames() -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    candidate = orchestrator.pending_candidate
    assert candidate is not None
    orchestrator.mark_returning_home(timestamp_s=0.15, candidate_digest=candidate.digest)

    orchestrator.accept_home_v2_barrier_frame(
        HomeV2BarrierFrame(
            observation_sequence_id="invalid-home",
            camera_at_home=False,
            fresh_observation_v2_frame=True,
            captured_after_return=True,
            contains_alternate_or_motion_rgb=False,
        ),
        timestamp_s=0.20,
    )
    assert orchestrator.home_observation_sequence_ids == ()
    assert orchestrator.camera_lease_held

    for index in range(4):
        orchestrator.accept_home_v2_barrier_frame(
            HomeV2BarrierFrame(
                observation_sequence_id=f"valid-home-{index}",
                camera_at_home=True,
                fresh_observation_v2_frame=True,
                captured_after_return=True,
                contains_alternate_or_motion_rgb=False,
            ),
            timestamp_s=0.21 + index * 0.05,
        )
        if index < 3:
            assert orchestrator.camera_lease_held

    assert orchestrator.home_observation_sequence_ids == tuple(
        f"valid-home-{index}" for index in range(4)
    )
    assert "home_frame_invalid" in orchestrator.terminal_reasons
    assert orchestrator.state is PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD
    assert not orchestrator.camera_lease_held
    assert orchestrator.memory_write_count == 0


def test_duplicate_home_timestamp_is_not_counted_and_forbids_commit() -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    candidate = orchestrator.pending_candidate
    assert candidate is not None
    orchestrator.mark_returning_home(timestamp_s=0.15, candidate_digest=candidate.digest)

    def accept(identifier: str, timestamp_s: float) -> None:
        orchestrator.accept_home_v2_barrier_frame(
            HomeV2BarrierFrame(
                observation_sequence_id=identifier,
                camera_at_home=True,
                fresh_observation_v2_frame=True,
                captured_after_return=True,
                contains_alternate_or_motion_rgb=False,
            ),
            timestamp_s=timestamp_s,
        )

    accept("home-first", 0.20)
    accept("home-duplicate-time", 0.20)
    assert orchestrator.home_observation_sequence_ids == ("home-first",)
    assert orchestrator.camera_lease_held
    for index, timestamp_s in enumerate((0.25, 0.30, 0.35), start=1):
        accept(f"home-valid-{index}", timestamp_s)

    assert orchestrator.home_observation_timestamps_s == pytest.approx(
        (0.20, 0.25, 0.30, 0.35)
    )
    assert "home_frame_timestamp_not_increasing" in orchestrator.terminal_reasons
    assert orchestrator.state is PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD
    assert not orchestrator.camera_lease_held


def test_source_recheck_before_last_home_timestamp_fails_closed() -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    _pass_home_barrier(orchestrator)

    assert not _recheck(orchestrator, timestamp_s=0.34)

    assert "source_recheck_before_last_home_frame" in orchestrator.terminal_reasons
    assert orchestrator.state is PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD
    assert orchestrator.memory.state.mode is ObjectMemoryMode.UNINITIALIZED
    assert orchestrator.memory_write_count == 0


def test_qualified_fresh_wrist_evidence_supersedes_old_alternate_candidate() -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    _pass_home_barrier(orchestrator)

    assert not _recheck(orchestrator, direct_wrist=True)

    assert (
        orchestrator.state
        is PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD
    )
    assert "superseded_by_fresh_direct_evidence" in orchestrator.terminal_reasons
    assert orchestrator.memory.state.mode is ObjectMemoryMode.UNINITIALIZED
    assert orchestrator.memory_write_count == 0


def _valid_resume_receipt() -> ActionHistoryResumeReceipt:
    return ActionHistoryResumeReceipt(
        episode_id="episode-a",
        request_id="episode-a-active-front-01",
        generation=10,
        home_observation_sequence_ids=tuple(f"home-{index}" for index in range(4)),
        generated_from_fresh_home_v2=True,
        stale_action_chunk_resumed=False,
    )


def test_duplicate_commit_latches_safe_hold_and_blocks_replan() -> None:
    orchestrator = _drive_to_commit()
    committed_state = orchestrator.memory.state

    with pytest.raises(RuntimeError, match="duplicate delayed commit"):
        _commit(orchestrator)
    assert orchestrator.memory.state == committed_state
    assert orchestrator.memory_write_count == 1
    assert (
        orchestrator.state
        is PendingActiveViewState.SHADOW_REPLAN_FAILED_SAFE_HOLD
    )
    assert "duplicate_delayed_commit" in orchestrator.terminal_reasons

    with pytest.raises(RuntimeError, match="已完成 Memory commit"):
        orchestrator.create_shadow_action_generation(
            _valid_resume_receipt(),
            source_phase=PhaseId.ACQUIRE_TRACK,
            source_phase_stability_reset=True,
            source_phase_stability_ticks=0,
        )


def test_invalid_action_generation_permanently_latches_safe_hold() -> None:
    orchestrator = _drive_to_commit()
    valid_resume = _valid_resume_receipt()

    with pytest.raises(ValueError, match="stale/invalid"):
        orchestrator.create_shadow_action_generation(
            replace(valid_resume, stale_action_chunk_resumed=True),
            source_phase=PhaseId.ACQUIRE_TRACK,
            source_phase_stability_reset=True,
            source_phase_stability_ticks=0,
        )
    assert (
        orchestrator.state
        is PendingActiveViewState.SHADOW_REPLAN_FAILED_SAFE_HOLD
    )
    assert "invalid_shadow_action_generation" in orchestrator.terminal_reasons

    with pytest.raises(RuntimeError, match="已完成 Memory commit"):
        orchestrator.create_shadow_action_generation(
            valid_resume,
            source_phase=PhaseId.ACQUIRE_TRACK,
            source_phase_stability_reset=True,
            source_phase_stability_ticks=0,
        )


def test_first_valid_action_generation_uses_fresh_home_v2() -> None:
    orchestrator = _drive_to_commit()

    receipt = orchestrator.create_shadow_action_generation(
        _valid_resume_receipt(),
        source_phase=PhaseId.ACQUIRE_TRACK,
        source_phase_stability_reset=True,
        source_phase_stability_ticks=0,
    )
    assert receipt.action_generation_before == 9
    assert receipt.action_generation_after == 10
    assert receipt.source_phase_stability_ticks == 0
    assert receipt.memory_only and not receipt.contact_authorized
    assert receipt.arm_command_count == receipt.gripper_close_count == 0
    assert receipt.test_read_count == 0


@pytest.mark.parametrize(
    ("active_window_open", "safety", "expected_reason"),
    [
        (False, None, "active_window_closed"),
        (
            True,
            ActiveFrontSafetyEvidence(contact_absent=False),
            "contact_detected",
        ),
    ],
)
def test_contact_or_closed_latch_rejects_and_invalidates_memory(
    active_window_open: bool,
    safety: ActiveFrontSafetyEvidence | None,
    expected_reason: str,
) -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    _pass_home_barrier(orchestrator)

    assert not _recheck(
        orchestrator,
        active_window_open=active_window_open,
        safety=safety,
    )

    assert expected_reason in orchestrator.terminal_reasons
    assert (
        orchestrator.state
        is PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD
    )
    assert orchestrator.memory.state.mode is ObjectMemoryMode.INVALID
    assert not orchestrator.memory.state.valid
    assert orchestrator.memory_write_count == 0


def test_source_recheck_records_external_memory_drift_even_with_safety_failure() -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    _pass_home_barrier(orchestrator)
    orchestrator.memory.invalidate_for_safety(
        episode_id="episode-a",
        timestamp_s=0.36,
        reasons=("controller_tracking_invalid",),
    )

    assert not _recheck(
        orchestrator,
        safety=ActiveFrontSafetyEvidence(contact_absent=False),
    )

    assert "memory_state_changed_while_pending" in orchestrator.terminal_reasons
    assert "contact_detected" in orchestrator.terminal_reasons
    assert orchestrator.memory.state.mode is ObjectMemoryMode.INVALID
    assert orchestrator.memory.state.accepted_update_count == 0
    assert orchestrator.memory_write_count == 0


@pytest.mark.parametrize(
    ("safety_update", "expected_reason"),
    [
        ({"object_contact_detected": True}, "object_contact_detected"),
        ({"pregrasp_window_open": False}, "pregrasp_window_closed"),
    ],
)
def test_commit_time_safety_failure_has_atomic_no_commit_receipt(
    safety_update: dict[str, bool],
    expected_reason: str,
) -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    _pass_home_barrier(orchestrator)
    assert _recheck(orchestrator)
    candidate = orchestrator.pending_candidate
    assert candidate is not None
    before = orchestrator.memory.state

    with pytest.raises(RuntimeError, match="commit 被拒绝"):
        orchestrator.commit(
            candidate_digest=candidate.digest,
            commit_timestamp_s=0.40,
            safety=_safety(**safety_update),
        )

    receipt = orchestrator.no_commit_receipt
    assert receipt is not None
    assert orchestrator.prepared_no_commit_receipt == receipt
    assert receipt.candidate_digest == candidate.digest
    assert receipt.candidate_write_count == 0
    assert receipt.prior_memory_safety_invalidated
    assert expected_reason in receipt.rejection_reasons
    assert expected_reason in receipt.safety_reasons
    assert receipt.accepted_update_count_before == before.accepted_update_count
    assert receipt.accepted_update_count_after == before.accepted_update_count
    assert orchestrator.commit_receipt is None
    assert orchestrator.prepared_commit_receipt is None
    assert orchestrator.memory_write_count == 0
    assert orchestrator.memory.state.mode is ObjectMemoryMode.INVALID
    assert not orchestrator.memory.state.valid
    assert orchestrator.memory.state.accepted_update_count == before.accepted_update_count
    assert orchestrator.state is PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD


def test_success_receipt_constructor_failure_leaves_memory_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    _pass_home_barrier(orchestrator)
    assert _recheck(orchestrator)
    before = orchestrator.memory.state

    def fail_receipt(**_: object) -> None:
        raise RuntimeError("injected success receipt failure")

    monkeypatch.setattr(
        active_front_memory_module,
        "DelayedActiveMemoryCommitReceipt",
        fail_receipt,
    )
    with pytest.raises(RuntimeError, match="injected success receipt failure"):
        _commit(orchestrator)

    assert orchestrator.memory.state == before
    assert orchestrator.state is PendingActiveViewState.SOURCE_RECHECK_PASSED
    assert orchestrator.commit_update is None
    assert orchestrator.prepared_commit_receipt is None
    assert orchestrator.commit_receipt is None
    assert orchestrator.memory_write_count == 0


def test_no_commit_receipt_constructor_failure_leaves_memory_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    _pass_home_barrier(orchestrator)
    assert _recheck(orchestrator)
    candidate = orchestrator.pending_candidate
    assert candidate is not None
    before = orchestrator.memory.state

    def fail_receipt(**_: object) -> None:
        raise RuntimeError("injected no-commit receipt failure")

    monkeypatch.setattr(
        active_front_memory_module,
        "DelayedActiveMemoryNoCommitReceipt",
        fail_receipt,
    )
    with pytest.raises(RuntimeError, match="injected no-commit receipt failure"):
        orchestrator.commit(
            candidate_digest=candidate.digest,
            commit_timestamp_s=0.40,
            safety=_safety(object_contact_detected=True),
        )

    assert orchestrator.memory.state == before
    assert orchestrator.state is PendingActiveViewState.SOURCE_RECHECK_PASSED
    assert orchestrator.commit_update is None
    assert orchestrator.prepared_no_commit_receipt is None
    assert orchestrator.no_commit_receipt is None
    assert orchestrator.memory_write_count == 0


def test_episode_reset_clears_no_commit_receipt_and_invalidated_memory() -> None:
    orchestrator = _orchestrator()
    _collect_valid(orchestrator)
    _pass_home_barrier(orchestrator)
    assert _recheck(orchestrator)
    candidate = orchestrator.pending_candidate
    assert candidate is not None
    with pytest.raises(RuntimeError, match="commit 被拒绝"):
        orchestrator.commit(
            candidate_digest=candidate.digest,
            commit_timestamp_s=0.40,
            safety=_safety(object_contact_detected=True),
        )
    assert orchestrator.no_commit_receipt is not None
    assert orchestrator.memory.state.mode is ObjectMemoryMode.INVALID

    reset_state = orchestrator.reset_episode(
        "episode-b",
        episode_generation=2,
        timestamp_s=1.0,
    )

    assert reset_state.mode is ObjectMemoryMode.UNINITIALIZED
    assert orchestrator.prepared_no_commit_receipt is None
    assert orchestrator.no_commit_receipt is None
    assert orchestrator.commit_update is None
    assert orchestrator.pending_candidate is None
    assert not orchestrator.request_active
    assert not orchestrator.camera_lease_held


def test_existing_valid_memory_is_invalidated_by_contact_without_candidate_write() -> None:
    identity = d049_primary_provider_identity()
    config = build_stage2_object_memory_config(identity)
    memory = ExplicitObjectStateMemory(config)
    verifier = ObjectCandidateWindowVerifier(config)
    memory.reset("episode-a")
    verifier.reset("episode-a")

    def measurement(timestamp_s: float, x: float) -> ObjectMeasurement:
        return ObjectMeasurement(
            timestamp_s=timestamp_s,
            rgb_timestamp_s=timestamp_s,
            camera_pose_timestamp_s=timestamp_s,
            tcp_pose_timestamp_s=timestamp_s,
            position_base_m=(x, 0.1, 0.02),
            covariance_base_m2=np.eye(3) * 1e-6,
            confidence=0.99,
            projection_valid=True,
            in_fov=True,
            observable=True,
            geometry_valid=True,
            write_gate_passed=True,
            source_camera="base_camera",
            source_model_identity=identity.object_memory_source_identity,
        )

    initial_decision = None
    for index in range(3):
        initial_decision = verifier.observe(
            measurement(index * 0.05, 0.400),
            episode_id="episode-a",
            safety=_safety(),
        )
    assert initial_decision is not None and initial_decision.verified
    initial = memory.update(initial_decision, episode_id="episode-a", safety=_safety())
    assert initial.measurement_accepted and initial.state.valid

    pending_decision = None
    for index in range(3):
        pending_decision = verifier.observe(
            measurement(0.20 + index * 0.05, 0.401),
            episode_id="episode-a",
            safety=_safety(),
        )
    assert pending_decision is not None and pending_decision.verified
    before_position = memory.state.position_base_m
    before_count = memory.state.accepted_update_count

    rejected = memory.commit_delayed_candidate(
        pending_decision,
        episode_id="episode-a",
        safety=_safety(object_contact_detected=True),
        commit_timestamp_s=0.40,
        max_pending_age_s=2.5,
    )

    assert not rejected.measurement_accepted
    assert "object_contact_detected" in rejected.rejection_reasons
    assert memory.state.mode is ObjectMemoryMode.INVALID
    assert memory.state.position_base_m == before_position
    assert memory.state.accepted_update_count == before_count
    assert "object_contact_detected" in memory.state.invalid_reasons
    contact = resolve_object_state(
        rejected,
        requirement=ObjectStateRequirement.CONTACT_READY,
    )
    assert not contact.available and not contact.contact_authorized


@pytest.mark.parametrize(
    "target",
    [
        PendingActiveViewState.COLLECTING,
        PendingActiveViewState.VERIFIED_PENDING,
        PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT,
        PendingActiveViewState.RETURNING_HOME,
        PendingActiveViewState.RETURNING_HOME_NO_COMMIT,
        PendingActiveViewState.HOME_BARRIER_PASSED,
        PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD,
        PendingActiveViewState.SOURCE_RECHECK_PASSED,
        PendingActiveViewState.COMMITTED,
        PendingActiveViewState.SHADOW_REPLAN_FAILED_SAFE_HOLD,
    ],
)
def test_episode_reset_clears_every_stage2_substate(target: PendingActiveViewState) -> None:
    orchestrator = _orchestrator()
    if target is PendingActiveViewState.RETURN_HOME_REQUIRED_NO_COMMIT:
        orchestrator.observe_collect_frame(
            _frame(0, camera_motion_state=ExternalCameraMotionState.MOVE_TO_VIEW),
            safety=_safety(),
        )
    elif target is PendingActiveViewState.HOME_VERIFIED_FAILED_SAFE_HOLD:
        orchestrator = _orchestrator(baseline=_baseline(home_score=0.66))
        for index in range(3):
            orchestrator.observe_collect_frame(_frame(index), safety=_safety())
        _return_failed_candidate_home(orchestrator)
    elif target is not PendingActiveViewState.COLLECTING:
        _collect_valid(orchestrator)
        if target is PendingActiveViewState.RETURNING_HOME_NO_COMMIT:
            candidate = orchestrator.pending_candidate
            assert candidate is not None
            orchestrator.mark_returning_home(
                timestamp_s=2.700001,
                candidate_digest=candidate.digest,
            )
        elif target is not PendingActiveViewState.VERIFIED_PENDING:
            candidate = orchestrator.pending_candidate
            assert candidate is not None
            orchestrator.mark_returning_home(
                timestamp_s=0.15,
                candidate_digest=candidate.digest,
            )
            if target is not PendingActiveViewState.RETURNING_HOME:
                for index in range(4):
                    orchestrator.accept_home_v2_barrier_frame(
                        HomeV2BarrierFrame(
                            observation_sequence_id=f"home-{index}",
                            camera_at_home=True,
                            fresh_observation_v2_frame=True,
                            captured_after_return=True,
                            contains_alternate_or_motion_rgb=False,
                        ),
                        timestamp_s=0.20 + index * 0.05,
                    )
                if target is not PendingActiveViewState.HOME_BARRIER_PASSED:
                    assert _recheck(orchestrator)
                    if target in {
                        PendingActiveViewState.COMMITTED,
                        PendingActiveViewState.SHADOW_REPLAN_FAILED_SAFE_HOLD,
                    }:
                        _commit(orchestrator)
                    if target is PendingActiveViewState.SHADOW_REPLAN_FAILED_SAFE_HOLD:
                        with pytest.raises(ValueError, match="stale/invalid"):
                            orchestrator.create_shadow_action_generation(
                                replace(
                                    _valid_resume_receipt(),
                                    stale_action_chunk_resumed=True,
                                ),
                                source_phase=PhaseId.ACQUIRE_TRACK,
                                source_phase_stability_reset=True,
                                source_phase_stability_ticks=0,
                            )
    assert orchestrator.state is target

    reset_state = orchestrator.reset_episode(
        "episode-b",
        episode_generation=2,
        timestamp_s=3.0,
    )

    assert orchestrator.state is PendingActiveViewState.RESET_CLEARED
    assert reset_state.mode is ObjectMemoryMode.UNINITIALIZED
    assert reset_state.episode_id == "episode-b"
    assert orchestrator.pending_candidate is None
    assert not orchestrator.request_active
    assert not orchestrator.camera_lease_held
    assert orchestrator.attempt_count == 0
    assert orchestrator.home_observation_sequence_ids == ()
    assert orchestrator.home_observation_timestamps_s == ()
    assert orchestrator.home_frame_digests == ()
    assert orchestrator.memory_write_count == 0
    assert orchestrator.commit_update is None
    assert orchestrator.prepared_commit_receipt is None
    assert orchestrator.commit_receipt is None
    assert orchestrator.prepared_no_commit_receipt is None
    assert orchestrator.no_commit_receipt is None
    assert orchestrator.shadow_action_receipt is None


def test_d049_allowlists_and_configs_preserve_stage2_stage3_boundaries() -> None:
    assert ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID in ACTIVE_FRONT_QUALIFIED_SHADOW_PRIMITIVE_IDS
    assert not set(ACTIVE_FRONT_PROHIBITED_MEMORY_WRITE_PRIMITIVE_IDS).intersection(
        ACTIVE_FRONT_QUALIFIED_SHADOW_PRIMITIVE_IDS
    )

    repository = Path(__file__).resolve().parents[1]
    stage2a = json.loads(
        (repository / "configs/e018_p1_stage2a_primary_memory_development_v1.json").read_text()
    )
    stage2b = json.loads(
        (repository / "configs/e018_p1_stage2b_information_gain_shadow_development_v1.json").read_text()
    )
    assert stage2a["decision"]["gate"] == "D049"
    assert stage2a["decision"]["d049_gate_commit"] == (
        "de48f1305098c86d7d49ab4a487eb1f36aea544c"
    )
    assert stage2a["decision"]["d049_home_clarification_commit"] == (
        "22d2719c2614dee2b02ebf396a55817b644810aa"
    )
    assert stage2a["provider"]["primary_primitive_id"] == ACTIVE_FRONT_PRIMARY_PRIMITIVE_ID
    assert stage2a["provider"]["score_semantics"] == ACTIVE_FRONT_SCORE_SEMANTICS
    assert stage2a["information_gain"][
        "non_promotional_integration_smoke_provisional_value"
    ] == 0.05
    assert stage2a["information_gain"]["frozen_evaluation_value"] is None
    assert stage2a["information_gain"]["evaluation_config_sha256"] is None
    assert stage2a["splits"]["integration_smoke"] == [76901, 76910]
    assert stage2a["splits"]["selection"] == [77001, 77025]
    assert stage2a["splits"]["evaluation"] == [77026, 77050]
    assert stage2a["splits"]["stage3_reserved"] == [77201, 77250]
    assert stage2a["budgets"]["total_gpu_wall_seconds_max"] == 7200
    assert stage2a["budgets"]["combined_artifact_bytes_max"] == 8 * 1024**3
    assert stage2a["permissions"]["fresh_test_reads"] == 0
    assert stage2a["permissions"]["runtime_object_gt_reads"] == 0
    assert stage2a["permissions"]["offline_label_reads"] == (
        "only-after-prediction-and-decision-freeze"
    )
    assert stage2a["permissions"]["arm_gripper_actuation"] == 0

    home_identity = d049_home_baseline_provider_identity()
    for config in (stage2a, stage2b):
        home = config["home_baseline"]
        assert home["role"] == (
            "raw-score-comparison-only-no-measurement-no-memory-write-"
            "no-state-resolution/v1"
        )
        assert home["primitive_id"] == "HOME__CENTER"
        assert home["camera_motion_state"] == ExternalCameraMotionState.HOME_ANCHOR.value
        assert home["settled_required"] is True
        assert home["provider_identity_sha256"] == home_identity.sha256
        assert home["provider_family_sha256"] == home_identity.provider_family_sha256
        assert home["calibration_identity_sha256"] == (
            home_identity.calibration_identity_sha256
        )
        assert home["calibration_scale_factor"] == home_identity.calibration_scale_factor
        assert home["write_threshold"] == home_identity.write_threshold
        assert home["score_semantics"] == ACTIVE_FRONT_SCORE_SEMANTICS
        assert np.asarray(home["expected_base_from_external_camera_cv"]) == pytest.approx(
            np.asarray(ACTIVE_FRONT_HOME_BASE_FROM_EXTERNAL_CAMERA_CV)
        )
        assert home["maximum_position_error_m"] == (
            ACTIVE_FRONT_HOME_POSITION_TOLERANCE_M
        )
        assert home["maximum_orientation_error_rad"] == (
            ACTIVE_FRONT_HOME_ORIENTATION_TOLERANCE_RAD
        )
        assert home["model_input_digest_required"] is True
        assert home["provider_output_digest_required"] is True

    assert stage2b["execution"]["memory_write_allowed"] is False
    assert stage2b["decision"]["d049_home_clarification_commit"] == (
        "22d2719c2614dee2b02ebf396a55817b644810aa"
    )
    assert stage2b["evidence"]["score_semantics"] == ACTIVE_FRONT_SCORE_SEMANTICS
    assert tuple(stage2b["viewpoints"]["qualified_shadow_only"]) == (
        ACTIVE_FRONT_QUALIFIED_SHADOW_PRIMITIVE_IDS
    )
    assert stage2b["splits"]["shadow"] == [77101, 77150]
    assert stage2b["splits"]["stage3_reserved"] == [77201, 77250]
    assert stage2b["maximum_routes"] == 350
    assert stage2b["permissions"]["runtime_object_gt_reads"] == 0
    assert stage2b["permissions"]["offline_label_reads"] == (
        "only-after-prediction-and-decision-freeze"
    )
