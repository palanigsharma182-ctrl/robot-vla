from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.executive import (
    DeployableOutcomeEvidence,
    DeployablePredicateThresholds,
    DeployableStateEstimatorConfig,
    FourFrameDeployableStateEstimator,
    PredicateEvidence,
    PredicateSource,
    ScalarStateEstimate,
    WristKeypointDetection,
    build_deployable_snapshot,
)
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    ObservationV2Frame,
    ObservationV2History,
)
from robot_vla.precision.geometry import project_base_point_to_normalized_uv

IMAGE_SIZE_HW = (80, 100)


def _intrinsic() -> np.ndarray:
    return np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _transform(translation: tuple[float, float, float]) -> np.ndarray:
    value = np.eye(4, dtype=np.float32)
    value[:3, 3] = np.asarray(translation, dtype=np.float32)
    return value


def _physical_proprio(spec: RobotSpec) -> np.ndarray:
    q = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)
    return np.concatenate((q, np.zeros(spec.arm_dof, dtype=np.float32), (0.04,))).astype(
        np.float32
    )


def _frame(
    spec: RobotSpec,
    step: int,
    *,
    invalid_modalities: tuple[str, ...] = (),
) -> ObservationV2Frame:
    timestamp = step / spec.control_hz
    valid = np.ones(len(OBSERVATION_MODALITIES), dtype=np.bool_)
    for name in invalid_modalities:
        valid[OBSERVATION_MODALITIES.index(name)] = False
    return ObservationV2Frame(
        rgb_external=np.full((*IMAGE_SIZE_HW, 3), step, dtype=np.uint8),
        rgb_wrist=np.full((*IMAGE_SIZE_HW, 3), step + 10, dtype=np.uint8),
        physical_proprio=_physical_proprio(spec),
        base_from_tcp=_transform((0.2, 0.0, 0.5)),
        # 相机随手腕沿 base-x 移动；静态物体的像素因此逐帧变化。
        base_from_wrist_camera=_transform((step * 0.01, 0.0, 0.0)),
        finger_force_n=np.asarray((1.0 + step * 0.1, 1.2 + step * 0.1), dtype=np.float32),
        timestamp_s=timestamp,
        modality_timestamp_s=np.full(
            len(OBSERVATION_MODALITIES),
            timestamp,
            dtype=np.float64,
        ),
        modality_valid=valid,
    )


def _window(
    spec: RobotSpec,
    *,
    invalid_last: tuple[str, ...] = (),
):
    history = ObservationV2History(spec)
    frames = []
    for step in range(4):
        frame = _frame(
            spec,
            step,
            invalid_modalities=invalid_last if step == 3 else (),
        )
        frames.append(frame)
        history.append(frame)
    q = _physical_proprio(spec)[: spec.arm_dof].copy()
    action = np.zeros(spec.action_dim, dtype=np.float32)
    return history.snapshot(
        "pick the cube",
        previous_command_q=q,
        previous_action=action,
    ), tuple(frames)


def _detections(
    frames: tuple[ObservationV2Frame, ...],
    *,
    object_velocity_x_m_s: float = 0.0,
    timestamp_offset_s: float = 0.0,
    source: PredicateSource = PredicateSource.DEPLOYABLE_ESTIMATOR,
) -> tuple[WristKeypointDetection, ...]:
    result = []
    for frame in frames:
        object_point = np.asarray(
            (
                0.2 + object_velocity_x_m_s * frame.timestamp_s,
                0.0,
                1.0,
            ),
            dtype=np.float32,
        )
        goal_point = np.asarray((0.1, 0.1, 1.0), dtype=np.float32)
        object_uv = project_base_point_to_normalized_uv(
            object_point,
            _intrinsic(),
            frame.base_from_wrist_camera,
            IMAGE_SIZE_HW,
        )
        goal_uv = project_base_point_to_normalized_uv(
            goal_point,
            _intrinsic(),
            frame.base_from_wrist_camera,
            IMAGE_SIZE_HW,
        )
        result.append(
            WristKeypointDetection(
                timestamp_s=frame.timestamp_s + timestamp_offset_s,
                object_normalized_uv=tuple(float(value) for value in object_uv),
                goal_normalized_uv=tuple(float(value) for value in goal_uv),
                object_confidence=0.98,
                goal_confidence=0.97,
                source=source,
            )
        )
    return tuple(result)


def _config() -> DeployableStateEstimatorConfig:
    return DeployableStateEstimatorConfig(
        object_plane_base_z_m=1.0,
        goal_plane_base_z_m=1.0,
        min_detection_confidence=0.9,
        max_detection_timestamp_error_s=0.002,
        max_camera_image_skew_s=0.002,
        max_track_age_s=0.06,
        max_track_innovation_m=0.02,
        max_track_speed_m_s=0.5,
    )


def _thresholds() -> DeployablePredicateThresholds:
    return DeployablePredicateThresholds(
        track_confidence_min=0.9,
        grasp_candidate_min=0.7,
        grasp_verified_min=0.9,
        support_contact_min=0.7,
        support_verified_min=0.85,
        settled_min=0.8,
        max_track_age_s=0.06,
        max_scalar_age_s=0.06,
    )


def _scalar(confidence: float) -> ScalarStateEstimate:
    return ScalarStateEstimate(
        confidence=confidence,
        valid=True,
        age_s=0.01,
        source=PredicateSource.OUTCOME_MONITOR,
    )


def test_four_frame_geometry_compensates_dynamic_wrist_camera_pose() -> None:
    spec = RobotSpec()
    window, frames = _window(spec)
    estimator = FourFrameDeployableStateEstimator(spec, _intrinsic(), _config())

    result = estimator.estimate(
        window,
        _detections(frames),
        current_timestamp_s=frames[-1].timestamp_s,
    )

    state = result.state_estimate
    assert state.object_track.valid
    assert state.goal_track.valid
    np.testing.assert_allclose(state.object_track.position_base_m, (0.2, 0.0, 1.0), atol=1e-6)
    np.testing.assert_allclose(state.object_track.velocity_base_m_s, 0.0, atol=1e-5)
    np.testing.assert_allclose(state.goal_track.position_base_m, (0.1, 0.1, 1.0), atol=1e-6)
    assert result.object_diagnostics.accepted_count == 4
    assert result.object_diagnostics.innovation_m == pytest.approx(0.0, abs=1e-5)
    assert result.object_diagnostics.rejection_reasons == ()


def test_four_frame_fit_recovers_base_frame_object_velocity() -> None:
    spec = RobotSpec()
    window, frames = _window(spec)
    estimator = FourFrameDeployableStateEstimator(spec, _intrinsic(), _config())

    result = estimator.estimate(
        window,
        _detections(frames, object_velocity_x_m_s=0.2),
        current_timestamp_s=frames[-1].timestamp_s,
    )

    assert result.state_estimate.object_track.valid
    np.testing.assert_allclose(
        result.state_estimate.object_track.velocity_base_m_s,
        (0.2, 0.0, 0.0),
        atol=1e-5,
    )
    assert result.object_diagnostics.speed_m_s == pytest.approx(0.2, abs=1e-5)


def test_state_to_snapshot_preserves_pressure_modality_and_direct_predicates() -> None:
    spec = RobotSpec()
    window, frames = _window(spec)
    estimator = FourFrameDeployableStateEstimator(spec, _intrinsic(), _config())
    outcome = DeployableOutcomeEvidence(
        grasp=_scalar(0.95),
        support_contact=_scalar(0.9),
        settled=_scalar(0.85),
    )
    state = estimator.estimate(
        window,
        _detections(frames),
        current_timestamp_s=frames[-1].timestamp_s,
        outcome_evidence=outcome,
    ).state_estimate

    snapshot = build_deployable_snapshot(
        spec,
        window,
        state,
        _thresholds(),
        tick=3,
        timestamp_s=frames[-1].timestamp_s,
    )

    assert state.finger_force_n == pytest.approx((1.3, 1.5))
    assert snapshot.valid_modalities == {
        "rgb_external",
        "rgb_wrist",
        "proprio",
        "tcp_pose",
        "wrist_camera_pose",
        "finger_force",
        "controller_state",
    }
    predicates = {item.name: item.satisfied for item in snapshot.predicates}
    assert predicates == {
        "object_track_valid": True,
        "goal_track_valid": True,
        "precision_target_valid": True,
        "grasp_candidate": True,
        "grasp_verified": True,
        "support_contact_detected": True,
        "support_verified": True,
        "placement_verified": True,
    }
    assert snapshot.predicate("grasp_verified").source == PredicateSource.OUTCOME_MONITOR


def test_missing_outcome_monitor_does_not_invent_grasp_or_support_state() -> None:
    spec = RobotSpec()
    window, frames = _window(spec, invalid_last=("finger_force",))
    estimator = FourFrameDeployableStateEstimator(spec, _intrinsic(), _config())
    state = estimator.estimate(
        window,
        _detections(frames),
        current_timestamp_s=frames[-1].timestamp_s,
    ).state_estimate

    snapshot = build_deployable_snapshot(
        spec,
        window,
        state,
        _thresholds(),
        tick=3,
        timestamp_s=frames[-1].timestamp_s,
    )

    assert state.finger_force_n == (0.0, 0.0)
    assert not state.grasp.valid
    assert not state.support_contact.valid
    assert not state.settled.valid
    assert "finger_force" not in snapshot.valid_modalities
    assert not snapshot.predicate("grasp_verified").satisfied
    assert not snapshot.predicate("support_verified").satisfied
    assert not snapshot.predicate("placement_verified").satisfied


def test_timestamp_mismatch_fails_track_closed_with_auditable_reason() -> None:
    spec = RobotSpec()
    window, frames = _window(spec)
    estimator = FourFrameDeployableStateEstimator(spec, _intrinsic(), _config())

    result = estimator.estimate(
        window,
        _detections(frames, timestamp_offset_s=0.01),
        current_timestamp_s=frames[-1].timestamp_s,
    )

    assert not result.state_estimate.object_track.valid
    assert result.object_diagnostics.accepted_count == 0
    assert result.object_diagnostics.rejection_reasons == (
        "detection_timestamp_mismatch",
        "track_insufficient_points",
    )


def test_latest_keypoint_jump_is_rejected_by_temporal_innovation_gate() -> None:
    spec = RobotSpec()
    window, frames = _window(spec)
    config = replace(_config(), max_track_speed_m_s=10.0)
    estimator = FourFrameDeployableStateEstimator(spec, _intrinsic(), config)
    detections = list(_detections(frames))
    latest_uv = detections[-1].object_normalized_uv
    assert latest_uv is not None
    detections[-1] = replace(
        detections[-1],
        object_normalized_uv=(latest_uv[0] + 0.1, latest_uv[1]),
    )

    result = estimator.estimate(
        window,
        tuple(detections),
        current_timestamp_s=frames[-1].timestamp_s,
    )

    assert not result.state_estimate.object_track.valid
    assert result.object_diagnostics.innovation_m == pytest.approx(0.1, abs=1e-5)
    assert "track_innovation_exceeded" in result.object_diagnostics.rejection_reasons


def test_episode_prefix_padding_requires_none_and_cannot_fabricate_velocity() -> None:
    spec = RobotSpec()
    history = ObservationV2History(spec)
    frame = _frame(spec, 0)
    history.append(frame)
    window = history.snapshot("pick the cube")
    estimator = FourFrameDeployableStateEstimator(spec, _intrinsic(), _config())
    detection = _detections((frame,))[0]

    result = estimator.estimate(
        window,
        (None, None, None, detection),
        current_timestamp_s=frame.timestamp_s,
    )
    assert not result.state_estimate.object_track.valid
    assert "track_insufficient_points" in result.object_diagnostics.rejection_reasons

    with pytest.raises(ValueError, match="padding history"):
        estimator.estimate(
            window,
            (detection, None, None, detection),
            current_timestamp_s=frame.timestamp_s,
        )


def test_runtime_estimator_and_snapshot_reject_hidden_gt_or_evidence_drift() -> None:
    spec = RobotSpec()
    window, frames = _window(spec)
    estimator = FourFrameDeployableStateEstimator(spec, _intrinsic(), _config())

    with pytest.raises(ValueError, match="evaluator GT keypoint"):
        estimator.estimate(
            window,
            _detections(frames, source=PredicateSource.EVALUATOR_GT),
            current_timestamp_s=frames[-1].timestamp_s,
        )

    state = estimator.estimate(
        window,
        _detections(frames),
        current_timestamp_s=frames[-1].timestamp_s,
    ).state_estimate
    with pytest.raises(ValueError, match="F_L/F_R"):
        build_deployable_snapshot(
            spec,
            window,
            replace(state, finger_force_n=(9.0, 9.0)),
            _thresholds(),
            tick=3,
            timestamp_s=frames[-1].timestamp_s,
        )
    with pytest.raises(ValueError, match="evaluator GT predicate"):
        build_deployable_snapshot(
            spec,
            window,
            state,
            _thresholds(),
            tick=3,
            timestamp_s=frames[-1].timestamp_s,
            additional_predicates=(
                PredicateEvidence(
                    name="coarse_reach_complete",
                    satisfied=True,
                    confidence=1.0,
                    source=PredicateSource.EVALUATOR_GT,
                ),
            ),
        )
