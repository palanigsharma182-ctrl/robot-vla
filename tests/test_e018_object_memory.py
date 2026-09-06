from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_vla.precision.object_memory import (
    DualPrecisionWorldState,
    ExplicitObjectStateMemory,
    ObjectCandidateWindowVerifier,
    ObjectMeasurement,
    ObjectMemoryConfig,
    ObjectMemoryMode,
    ObjectMemorySafetyContext,
    ObjectStateRequirement,
    resolve_object_state,
)
from robot_vla.precision.state_memory import ExplicitGoalStateMemory, GoalMemoryConfig


def _config(*, max_age_s: float = 1.0, candidate_frames: int = 2) -> ObjectMemoryConfig:
    return ObjectMemoryConfig(
        max_unobserved_age_s=max_age_s,
        max_innovation_m=0.01,
        max_position_std_m=0.02,
        min_candidate_frames=candidate_frames,
        max_candidate_gap_s=0.1,
        max_candidate_position_spread_m=0.005,
        max_sensor_skew_s=0.01,
        expected_source_camera="hand_camera",
        expected_source_model_identity="e016-p1-selected-epoch-12",
        require_covariance=True,
    )


def _safe(**overrides: bool) -> ObjectMemorySafetyContext:
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
    values.update(overrides)
    return ObjectMemorySafetyContext(**values)


def _measurement(
    timestamp_s: float,
    *,
    position: tuple[float, float, float] | None = (0.4, 0.1, 0.02),
    observable: bool = True,
    gate: bool = True,
    source_camera: str = "hand_camera",
    source_model_identity: str = "e016-p1-selected-epoch-12",
    rgb_timestamp_s: float | None = None,
) -> ObjectMeasurement:
    covariance = None if position is None else np.eye(3, dtype=np.float64) * 1e-6
    return ObjectMeasurement(
        timestamp_s=timestamp_s,
        rgb_timestamp_s=timestamp_s if rgb_timestamp_s is None else rgb_timestamp_s,
        camera_pose_timestamp_s=timestamp_s,
        tcp_pose_timestamp_s=timestamp_s,
        position_base_m=position,
        covariance_base_m2=covariance,
        confidence=0.99 if observable else 0.0,
        projection_valid=True,
        in_fov=True,
        observable=observable,
        geometry_valid=position is not None,
        write_gate_passed=bool(gate and observable and position is not None),
        source_camera=source_camera,
        source_model_identity=source_model_identity,
    )


def _pipeline(
    config: ObjectMemoryConfig | None = None,
) -> tuple[ExplicitObjectStateMemory, ObjectCandidateWindowVerifier]:
    resolved = config or _config()
    memory = ExplicitObjectStateMemory(resolved)
    verifier = ObjectCandidateWindowVerifier(resolved)
    memory.reset("episode-a")
    verifier.reset("episode-a")
    return memory, verifier


def _update(
    memory: ExplicitObjectStateMemory,
    verifier: ObjectCandidateWindowVerifier,
    measurement: ObjectMeasurement,
    *,
    safety: ObjectMemorySafetyContext | None = None,
):
    resolved_safety = safety or _safe()
    candidate = verifier.observe(
        measurement,
        episode_id="episode-a",
        safety=resolved_safety,
    )
    return memory.update(
        candidate,
        episode_id="episode-a",
        safety=resolved_safety,
    )


def _initialize(
    memory: ExplicitObjectStateMemory,
    verifier: ObjectCandidateWindowVerifier,
    *,
    start_s: float = 0.0,
):
    warming = _update(memory, verifier, _measurement(start_s))
    assert not warming.measurement_accepted
    return _update(memory, verifier, _measurement(start_s + 0.05))


def test_verified_candidate_initializes_free_static_memory() -> None:
    memory, verifier = _pipeline()

    update = _initialize(memory, verifier)

    assert update.measurement_accepted
    assert update.state.mode == ObjectMemoryMode.FREE_STATIC
    assert update.state.valid
    assert update.state.observable_now
    assert update.state.position_base_m == pytest.approx((0.4, 0.1, 0.02))
    assert update.state.source_camera == "hand_camera"


@pytest.mark.parametrize("first_rgb,second_rgb", [(0.005, 0.005), (0.005, 0.004)])
def test_reused_or_reversed_rgb_cannot_complete_candidate(first_rgb, second_rgb) -> None:
    memory, verifier = _pipeline()
    _update(memory, verifier, _measurement(0.005, rgb_timestamp_s=first_rgb))
    result = _update(memory, verifier, _measurement(0.010, rgb_timestamp_s=second_rgb))
    assert not result.measurement_accepted
    assert "rgb_timestamp_not_increasing" in result.rejection_reasons
    assert not result.state.valid
    # 拒绝后不能降低图像时间水位；两个新的、有序图像才能重建窗口。
    result = _update(memory, verifier, _measurement(0.011, rgb_timestamp_s=first_rgb))
    assert not result.measurement_accepted
    _update(memory, verifier, _measurement(0.015))
    assert _update(memory, verifier, _measurement(0.020)).measurement_accepted


def test_memory_age_includes_image_acquisition_delay() -> None:
    memory, verifier = _pipeline(_config(max_age_s=0.5))
    _update(memory, verifier, _measurement(0.0))
    accepted = _update(memory, verifier, _measurement(0.05, rgb_timestamp_s=0.041))
    assert accepted.measurement_accepted
    assert accepted.state.age_s == pytest.approx(0.009)
    held = _update(memory, verifier, _measurement(0.55, position=None, observable=False, gate=False))
    assert held.state.age_s == pytest.approx(0.509)
    assert not resolve_object_state(held, requirement=ObjectStateRequirement.NAVIGATION).available


@pytest.mark.parametrize("delayed", [False, True])
def test_already_stale_image_cannot_write_even_inside_sensor_skew(delayed) -> None:
    config = replace(_config(max_age_s=0.001), min_candidate_frames=1)
    memory, verifier = _pipeline(config)
    candidate = verifier.observe(_measurement(0.009, rgb_timestamp_s=0.0),
                                 episode_id="episode-a", safety=_safe())
    if delayed:
        update = memory.commit_delayed_candidate(candidate, episode_id="episode-a", safety=_safe(),
                                                 commit_timestamp_s=0.009, max_pending_age_s=1.0)
    else:
        update = memory.update(candidate, episode_id="episode-a", safety=_safe())
    assert not update.measurement_accepted
    assert not update.state.valid


def test_delayed_preview_ages_from_rgb_and_cannot_reconsume_candidate() -> None:
    config = replace(_config(max_age_s=0.5), covariance_growth_m2_per_s=1e-6)
    memory, verifier = _pipeline(config)
    verifier.observe(_measurement(0.0), episode_id="episode-a", safety=_safe())
    candidate = verifier.observe(_measurement(0.05, rgb_timestamp_s=0.041),
                                 episode_id="episode-a", safety=_safe())
    previous = memory.state
    preview = memory.preview_delayed_candidate(candidate, episode_id="episode-a", safety=_safe(),
                                               commit_timestamp_s=0.1, max_pending_age_s=0.5)
    assert memory.state == previous
    assert preview.measurement_accepted
    assert preview.state.age_s == pytest.approx(0.059)
    assert np.diag(preview.state.covariance_base_m2) == pytest.approx([1.059e-6] * 3)
    assert not preview.state.observable_now
    expired = memory.preview_delayed_candidate(candidate, episode_id="episode-a", safety=_safe(),
                                               commit_timestamp_s=0.55, max_pending_age_s=1.0)
    assert not expired.measurement_accepted
    assert "memory_stale_at_commit" in expired.rejection_reasons
    applied = memory.apply_delayed_candidate_preview(
        preview, candidate, episode_id="episode-a", safety=_safe(), commit_timestamp_s=0.1,
        max_pending_age_s=0.5, expected_previous_state=previous,
    )
    assert applied.state == preview.state
    repeated = memory.commit_delayed_candidate(candidate, episode_id="episode-a", safety=_safe(),
                                                commit_timestamp_s=0.11, max_pending_age_s=0.5)
    assert not repeated.measurement_accepted
    assert repeated.state.accepted_update_count == 1


def test_direct_write_ages_covariance_and_rejects_duplicate_decision() -> None:
    config = replace(_config(), min_candidate_frames=1, covariance_growth_m2_per_s=1e-6)
    memory, verifier = _pipeline(config)
    candidate = verifier.observe(_measurement(0.009, rgb_timestamp_s=0.0),
                                 episode_id="episode-a", safety=_safe())
    update = memory.update(candidate, episode_id="episode-a", safety=_safe())
    assert np.diag(update.state.covariance_base_m2) == pytest.approx([1.009e-6] * 3)
    repeated = memory.update(candidate, episode_id="episode-a", safety=_safe())
    assert not repeated.measurement_accepted
    assert repeated.state.accepted_update_count == 1


def test_rgb_gap_and_stale_early_frame_cannot_supply_window_support() -> None:
    memory, verifier = _pipeline()
    _update(memory, verifier, _measurement(0.009, rgb_timestamp_s=0.0))
    result = _update(memory, verifier, _measurement(0.105))
    assert not result.measurement_accepted
    assert "candidate_time_gap" in result.rejection_reasons
    memory, verifier = _pipeline(_config(max_age_s=0.001))
    _update(memory, verifier, _measurement(0.009, rgb_timestamp_s=0.0))
    result = _update(memory, verifier, _measurement(0.010))
    assert not result.measurement_accepted
    assert "candidate_too_short" in result.rejection_reasons


def test_rejected_delayed_preview_ages_old_state_without_mutating_it() -> None:
    memory, verifier = _pipeline(_config(max_age_s=0.5))
    _update(memory, verifier, _measurement(0.0))
    candidate = verifier.observe(_measurement(0.05), episode_id="episode-a", safety=_safe())
    memory.update(candidate, episode_id="episode-a", safety=_safe())
    previous = memory.state
    preview = memory.preview_delayed_candidate(candidate, episode_id="episode-a", safety=_safe(),
                                               commit_timestamp_s=0.6, max_pending_age_s=1.0)
    assert memory.state == previous
    assert not preview.measurement_accepted and not preview.state.valid
    assert preview.state.age_s == pytest.approx(0.55)
    assert not resolve_object_state(preview, requirement=ObjectStateRequirement.NAVIGATION).available
    memory.apply_delayed_candidate_preview(preview, candidate, episode_id="episode-a", safety=_safe(),
        commit_timestamp_s=0.6, max_pending_age_s=1.0, expected_previous_state=previous)
    assert memory.state == preview.state


def test_delayed_candidate_cannot_cross_reset_time_boundary() -> None:
    memory, verifier = _pipeline()
    verifier.observe(_measurement(0.1), episode_id="episode-a", safety=_safe())
    candidate = verifier.observe(_measurement(0.15), episode_id="episode-a", safety=_safe())
    memory.reset("episode-a", timestamp_s=0.2)
    verifier.reset("episode-a")
    result = memory.commit_delayed_candidate(candidate, episode_id="episode-a", safety=_safe(),
                                              commit_timestamp_s=0.25, max_pending_age_s=1.0)
    assert not result.measurement_accepted and not result.state.valid
    assert "measurement_before_reset" in result.rejection_reasons


def test_single_frame_or_rejected_input_cannot_initialize_memory() -> None:
    memory, verifier = _pipeline()

    single = _update(memory, verifier, _measurement(0.0))

    assert not single.measurement_accepted
    assert single.rejection_reasons == ("candidate_too_short",)
    assert single.state.mode == ObjectMemoryMode.UNINITIALIZED

    unverified = _update(memory, verifier, _measurement(0.1, gate=False))
    assert "write_gate_rejected" in unverified.rejection_reasons
    assert unverified.state.mode == ObjectMemoryMode.UNINITIALIZED


def test_candidate_window_restarts_on_position_inconsistency() -> None:
    memory, verifier = _pipeline()

    _update(memory, verifier, _measurement(0.0))
    restarted = _update(
        memory,
        verifier,
        _measurement(0.05, position=(0.41, 0.1, 0.02)),
    )
    accepted = _update(
        memory,
        verifier,
        _measurement(0.1, position=(0.41, 0.1, 0.02)),
    )

    assert not restarted.measurement_accepted
    assert "candidate_position_inconsistent" in restarted.rejection_reasons
    assert accepted.measurement_accepted
    assert accepted.state.position_base_m == pytest.approx((0.41, 0.1, 0.02))


def test_candidate_receipt_describes_current_sliding_window() -> None:
    _, verifier = _pipeline()

    first = verifier.observe(_measurement(0.0), episode_id="episode-a", safety=_safe())
    second = verifier.observe(_measurement(0.05), episode_id="episode-a", safety=_safe())
    third = verifier.observe(_measurement(0.1), episode_id="episode-a", safety=_safe())

    assert not first.verified
    assert second.verified and second.frame_count == 2
    assert second.window_start_timestamp_s == 0.0
    assert third.verified and third.frame_count == 2
    assert third.window_id == second.window_id
    assert third.window_start_timestamp_s == 0.05


def test_source_identity_drift_invalidates_episode_memory() -> None:
    memory, verifier = _pipeline()
    _initialize(memory, verifier)

    drifted = _update(
        memory,
        verifier,
        _measurement(0.1, source_model_identity="different-checkpoint"),
    )
    attempted_recovery = _update(memory, verifier, _measurement(0.15))

    assert drifted.state.mode == ObjectMemoryMode.INVALID
    assert "source_model_identity_mismatch" in drifted.state.invalid_reasons
    assert "source_model_identity_mismatch" in attempted_recovery.rejection_reasons


def test_sensor_timestamp_skew_invalidates_current_state() -> None:
    memory, verifier = _pipeline()
    _initialize(memory, verifier)

    skewed = _update(
        memory,
        verifier,
        _measurement(0.1, rgb_timestamp_s=0.08),
    )

    assert skewed.state.mode == ObjectMemoryMode.INVALID
    assert "sensor_timestamp_unsynchronized" in skewed.state.invalid_reasons


def test_measurement_rejects_missing_provenance_at_contract_boundary() -> None:
    values = _measurement(0.0).__dict__
    with pytest.raises(ValueError, match="source_camera 不能为空"):
        ObjectMeasurement(**{**values, "source_camera": None})


def test_short_occlusion_holds_position_only_for_navigation() -> None:
    memory, verifier = _pipeline()
    _initialize(memory, verifier)

    held = _update(
        memory,
        verifier,
        _measurement(
            0.5,
            position=None,
            observable=False,
            gate=False,
        ),
    )

    assert not held.measurement_accepted
    assert held.state.mode == ObjectMemoryMode.FREE_STATIC
    assert held.state.valid
    assert not held.state.observable_now
    navigation = resolve_object_state(
        held,
        requirement=ObjectStateRequirement.NAVIGATION,
    )
    assert navigation.available
    assert navigation.memory_only
    assert not navigation.contact_authorized
    contact = resolve_object_state(
        held,
        requirement=ObjectStateRequirement.CONTACT_READY,
    )
    assert not contact.available
    assert not contact.contact_authorized


def test_fresh_direct_candidate_can_authorize_contact_requirement() -> None:
    memory, verifier = _pipeline()
    update = _initialize(memory, verifier)

    contact = resolve_object_state(
        update,
        requirement=ObjectStateRequirement.CONTACT_READY,
    )

    assert contact.available
    assert contact.source == "current_measurement"
    assert contact.contact_authorized
    assert not contact.memory_only


def test_stale_or_conflicting_state_is_invalid_but_retained_for_audit() -> None:
    memory, verifier = _pipeline(_config(max_age_s=0.5))
    _initialize(memory, verifier)

    stale = _update(
        memory,
        verifier,
        _measurement(
            0.6,
            position=None,
            observable=False,
            gate=False,
        ),
    )
    assert stale.state.mode == ObjectMemoryMode.INVALID
    assert stale.state.position_base_m == pytest.approx((0.4, 0.1, 0.02))
    assert stale.state.invalid_reasons == ("memory_stale",)

    warming = _update(
        memory,
        verifier,
        _measurement(0.7, position=(0.5, 0.1, 0.02)),
    )
    assert not warming.measurement_accepted
    conflict = _update(
        memory,
        verifier,
        _measurement(0.75, position=(0.5, 0.1, 0.02)),
    )
    assert not conflict.measurement_accepted
    assert conflict.innovation_m == pytest.approx(0.1)
    assert conflict.state.mode == ObjectMemoryMode.INVALID
    assert conflict.state.position_base_m == pytest.approx((0.4, 0.1, 0.02))
    assert "measurement_conflict" in conflict.state.invalid_reasons


@pytest.mark.parametrize(
    ("overrides", "reason"),
    (
        ({"pregrasp_window_open": False}, "pregrasp_window_closed"),
        ({"object_contact_detected": True}, "object_contact_detected"),
        ({"gripper_close_commanded": True}, "gripper_close_commanded"),
        ({"grasp_candidate": True}, "grasp_candidate"),
        ({"grasp_verified": True}, "grasp_verified"),
        ({"object_maybe_moved": True}, "object_maybe_moved"),
    ),
)
def test_contact_boundary_events_irreversibly_invalidate_memory(
    overrides: dict[str, bool],
    reason: str,
) -> None:
    memory, verifier = _pipeline()
    _initialize(memory, verifier)

    invalidated = _update(
        memory,
        verifier,
        _measurement(0.1),
        safety=_safe(**overrides),
    )
    attempted_recovery = _update(memory, verifier, _measurement(0.2))

    assert invalidated.state.mode == ObjectMemoryMode.INVALID
    assert reason in invalidated.state.invalid_reasons
    assert not attempted_recovery.measurement_accepted
    assert reason in attempted_recovery.rejection_reasons
    assert not resolve_object_state(
        attempted_recovery,
        requirement=ObjectStateRequirement.NAVIGATION,
    ).available


def test_reset_timestamp_and_episode_identity_fail_closed() -> None:
    memory, verifier = _pipeline()
    _update(memory, verifier, _measurement(1.0))
    with pytest.raises(ValueError, match="严格递增"):
        verifier.observe(_measurement(0.9), episode_id="episode-a", safety=_safe())
    with pytest.raises(ValueError, match="episode identity"):
        verifier.observe(_measurement(1.1), episode_id="episode-b", safety=_safe())

    reset = memory.reset("episode-b", timestamp_s=2.0)
    assert reset.mode == ObjectMemoryMode.UNINITIALIZED
    assert reset.position_base_m is None
    assert reset.accepted_update_count == 0


def test_dual_world_state_requires_matching_episode_identity() -> None:
    goal_memory = ExplicitGoalStateMemory(
        GoalMemoryConfig(
            max_unobserved_age_s=1.0,
            max_innovation_m=0.01,
            max_position_std_m=0.02,
        )
    )
    goal = goal_memory.reset("episode-a")
    object_state = _pipeline()[0].state

    world = DualPrecisionWorldState(goal=goal, object=object_state)

    assert world.estimated_goal_position_base_m is None
    assert world.estimated_object_position_for_navigation_base_m is None
    other_goal = goal_memory.reset("episode-b")
    with pytest.raises(ValueError, match="episode identity"):
        DualPrecisionWorldState(goal=other_goal, object=object_state)
