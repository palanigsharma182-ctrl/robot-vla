from __future__ import annotations

import json

import numpy as np
import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.executive import (
    CANONICAL_PICK_PLACE_SUBTASKS,
    PICK_PLACE_TASK_ID,
    DeployablePredicateThresholds,
    DeployableStateEstimatorConfig,
    ExecutiveConfig,
    FourFrameDeployableStateEstimator,
    PhaseId,
    PickPlacePlanCompiler,
    PlanCompilerConfig,
    PredicateSource,
    SemanticPlanProposal,
    ShadowExecutiveObserver,
    ShadowExecutiveObserverConfig,
    TransitionReason,
    WristKeypointDetection,
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


def _window(spec: RobotSpec):
    history = ObservationV2History(spec)
    q = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)
    for step in range(4):
        timestamp = step / spec.control_hz
        proprio = np.concatenate(
            (q, np.zeros(spec.arm_dof, dtype=np.float32), (0.04,))
        ).astype(np.float32)
        history.append(
            ObservationV2Frame(
                rgb_external=np.full((*IMAGE_SIZE_HW, 3), step, dtype=np.uint8),
                rgb_wrist=np.full((*IMAGE_SIZE_HW, 3), step + 10, dtype=np.uint8),
                physical_proprio=proprio,
                base_from_tcp=_transform((0.2, 0.0, 0.5)),
                base_from_wrist_camera=_transform((0.0, 0.0, 0.0)),
                finger_force_n=np.asarray((1.0, 1.1), dtype=np.float32),
                timestamp_s=timestamp,
                modality_timestamp_s=np.full(
                    len(OBSERVATION_MODALITIES),
                    timestamp,
                    dtype=np.float64,
                ),
                modality_valid=np.ones(len(OBSERVATION_MODALITIES), dtype=np.bool_),
            )
        )
    return history.snapshot(
        "pick the cube",
        previous_command_q=q,
        previous_action=np.zeros(spec.action_dim, dtype=np.float32),
    )


def _detections(window) -> tuple[WristKeypointDetection, ...]:
    object_uv = project_base_point_to_normalized_uv(
        np.asarray((0.2, 0.0, 1.0), dtype=np.float32),
        _intrinsic(),
        _transform((0.0, 0.0, 0.0)),
        IMAGE_SIZE_HW,
    )
    goal_uv = project_base_point_to_normalized_uv(
        np.asarray((0.1, 0.1, 1.0), dtype=np.float32),
        _intrinsic(),
        _transform((0.0, 0.0, 0.0)),
        IMAGE_SIZE_HW,
    )
    wrist_index = OBSERVATION_MODALITIES.index("rgb_wrist")
    return tuple(
        WristKeypointDetection(
            timestamp_s=float(window.modality_timestamp_s[index, wrist_index]),
            object_normalized_uv=tuple(float(value) for value in object_uv),
            goal_normalized_uv=tuple(float(value) for value in goal_uv),
            object_confidence=0.98,
            goal_confidence=0.97,
        )
        for index in range(4)
    )


def _plan():
    proposal = SemanticPlanProposal(
        proposal_id="shadow-proposal-001",
        task_id=PICK_PLACE_TASK_ID,
        object_ref="cube",
        goal_ref="goal-region",
        requested_subtasks=CANONICAL_PICK_PLACE_SUBTASKS,
    )
    return PickPlacePlanCompiler(PlanCompilerConfig(stable_ticks_required=2)).compile(
        proposal
    )


def _estimator(spec: RobotSpec) -> FourFrameDeployableStateEstimator:
    return FourFrameDeployableStateEstimator(
        spec,
        _intrinsic(),
        DeployableStateEstimatorConfig(
            object_plane_base_z_m=1.0,
            goal_plane_base_z_m=1.0,
            min_detection_confidence=0.9,
            max_detection_timestamp_error_s=0.002,
            max_camera_image_skew_s=0.002,
            max_track_age_s=0.06,
            max_track_innovation_m=0.02,
            max_track_speed_m_s=0.5,
        ),
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


def _observer(
    spec: RobotSpec,
    *,
    enabled: bool,
    detection_provider=_detections,
) -> ShadowExecutiveObserver:
    return ShadowExecutiveObserver(
        spec,
        _plan(),
        _estimator(spec),
        _thresholds(),
        detection_provider,
        config=ShadowExecutiveObserverConfig(enabled=enabled),
    )


def test_shadow_observer_is_disabled_by_default_and_does_not_call_provider() -> None:
    spec = RobotSpec()

    def forbidden_provider(_window):
        raise AssertionError("disabled observer 不能执行 detection")

    observer = ShadowExecutiveObserver(
        spec,
        _plan(),
        _estimator(spec),
        _thresholds(),
        forbidden_provider,
    )

    assert observer.observe(_window(spec), control_step=0) is None
    assert observer.records == ()
    assert observer.ledger_jsonl == ""


def test_enabled_shadow_observer_advances_ledger_without_actuation_authority() -> None:
    spec = RobotSpec()
    observer = _observer(spec, enabled=True)
    window = _window(spec)

    first = observer.observe(window, control_step=0)
    second = observer.observe(window, control_step=4)

    assert first is not None and first.success
    assert second is not None and second.success
    assert first.decision is not None
    assert second.decision is not None
    assert first.decision.reason == TransitionReason.STABILITY_PENDING
    assert second.decision.reason == TransitionReason.PHASE_COMPLETED
    assert not first.decision.actuation_allowed
    assert not second.decision.actuation_allowed
    assert observer.state.phase == PhaseId.COARSE_APPROACH
    assert len(observer.records) == 2
    assert len(observer.ledger_jsonl.strip().splitlines()) == 2
    assert second.ledger_sha256 == observer.ledger_sha256
    json.dumps(second.to_dict(), sort_keys=True)


def test_shadow_observer_records_gt_failure_without_advancing_executive() -> None:
    spec = RobotSpec()

    def gt_provider(window):
        return tuple(
            WristKeypointDetection(
                timestamp_s=item.timestamp_s,
                object_normalized_uv=item.object_normalized_uv,
                goal_normalized_uv=item.goal_normalized_uv,
                object_confidence=item.object_confidence,
                goal_confidence=item.goal_confidence,
                source=PredicateSource.EVALUATOR_GT,
            )
            for item in _detections(window)
        )

    observer = _observer(spec, enabled=True, detection_provider=gt_provider)

    record = observer.observe(_window(spec), control_step=0)

    assert record is not None and not record.success
    assert record.error_type == "ValueError"
    assert "evaluator GT" in (record.error_message or "")
    assert record.executive_tick is None
    assert observer.state.phase == PhaseId.ACQUIRE_TRACK
    assert observer.ledger_jsonl == ""


def test_shadow_observer_rolls_back_partially_committed_executive_step() -> None:
    spec = RobotSpec()
    observer = _observer(spec, enabled=True)
    window = _window(spec)
    first = observer.observe(window, control_step=0)
    assert first is not None and first.success
    committed_ledger = observer.ledger_jsonl

    original_step = observer._executive.step

    def commit_then_raise(snapshot):
        original_step(snapshot)
        raise RuntimeError("injected post-commit failure")

    observer._executive.step = commit_then_raise
    failed = observer.observe(window, control_step=4)

    assert failed is not None and not failed.success
    assert failed.error_type == "RuntimeError"
    assert observer.ledger_jsonl == committed_ledger
    assert observer.ledger_sha256 == first.ledger_sha256

    retried = observer.observe(window, control_step=4)
    assert retried is not None and retried.success
    assert retried.executive_tick == 1


def test_shadow_observer_reset_is_episode_local_and_rejects_control_step_regression() -> None:
    spec = RobotSpec()
    observer = _observer(spec, enabled=True)
    window = _window(spec)
    observer.observe(window, control_step=4)

    failed = observer.observe(window, control_step=3)
    assert failed is not None and not failed.success
    assert "不能回退" in (failed.error_message or "")
    assert len(observer.records) == 2

    observer.reset()
    assert observer.records == ()
    assert observer.ledger_jsonl == ""
    assert observer.state.phase == PhaseId.ACQUIRE_TRACK
    restarted = observer.observe(window, control_step=0)
    assert restarted is not None and restarted.success
    assert restarted.record_index == 0
    assert restarted.executive_tick == 0


def test_shadow_observer_reset_also_resets_stateful_detection_provider() -> None:
    spec = RobotSpec()

    class StatefulProvider:
        def __init__(self) -> None:
            self.reset_calls = 0

        def __call__(self, window):
            return _detections(window)

        def reset(self) -> None:
            self.reset_calls += 1

    provider = StatefulProvider()
    observer = _observer(
        spec,
        enabled=True,
        detection_provider=provider,
    )
    observer.observe(_window(spec), control_step=0)

    observer.reset()

    assert provider.reset_calls == 1
    assert observer.records == ()
    assert observer.ledger_jsonl == ""


def test_shadow_observer_rejects_non_shadow_executive_configuration() -> None:
    spec = RobotSpec()
    with pytest.raises(ValueError, match="shadow_only=False"):
        ShadowExecutiveObserver(
            spec,
            _plan(),
            _estimator(spec),
            _thresholds(),
            _detections,
            executive_config=ExecutiveConfig(shadow_only=False),
            config=ShadowExecutiveObserverConfig(enabled=True),
        )
