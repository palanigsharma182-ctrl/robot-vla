from __future__ import annotations

import json
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from robot_vla.adapters import (
    FingerForceNormalizer,
    FingerForceStats,
    ProprioNormalizer,
    ProprioStats,
)
from robot_vla.contracts import RobotSpec
from robot_vla.executive import PredicateSource
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    OBSERVATION_V2_FRAME_STATE_DIM,
    ObservationV2Frame,
    ObservationV2History,
)
from robot_vla.precision.provider import (
    PrecisionDetectionProvider,
    PrecisionDetectionProviderConfig,
    PrecisionDetectionProviderError,
    PrecisionFrameStatus,
    PrecisionGeometricMotionInput,
    PrecisionPredictorIdentity,
)

IMAGE_SIZE_HW = (16, 20)


def _transform(translation: tuple[float, float, float]) -> np.ndarray:
    value = np.eye(4, dtype=np.float32)
    value[:3, 3] = np.asarray(translation, dtype=np.float32)
    return value


def _window(
    spec: RobotSpec,
    *,
    frame_count: int = 4,
    missing_wrist_step: int | None = None,
):
    history = ObservationV2History(spec)
    q = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)
    for step in range(frame_count):
        timestamp = 1.0 + step / spec.control_hz
        valid = np.ones(len(OBSERVATION_MODALITIES), dtype=np.bool_)
        if step == missing_wrist_step:
            valid[OBSERVATION_MODALITIES.index("rgb_wrist")] = False
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
                finger_force_n=np.asarray((1.0 + step, 2.0 + step), dtype=np.float32),
                timestamp_s=timestamp,
                modality_timestamp_s=np.full(
                    len(OBSERVATION_MODALITIES),
                    timestamp,
                    dtype=np.float64,
                ),
                modality_valid=valid,
            )
        )
    return history.snapshot("pick the cube")


def _normalizers(spec: RobotSpec):
    proprio = ProprioNormalizer(
        ProprioStats(
            mean=(0.0,) * spec.proprio_dim,
            std=(1.0,) * spec.proprio_dim,
            count=4,
            embodiment=spec.embodiment,
        ),
        spec,
    )
    force = FingerForceNormalizer(
        FingerForceStats(
            scale_log1p_p95=(1.0, 1.0),
            count=4,
            positive_count=(4, 4),
            embodiment=spec.embodiment,
        ),
        spec,
    )
    return proprio, force


class FakePredictor:
    def __init__(self, *, fail_on_call: int | None = None) -> None:
        self._identity = PrecisionPredictorIdentity(
            checkpoint_sha256="a" * 64,
            parameter_state_sha256="b" * 64,
            model_config_sha256="c" * 64,
            keypoint_names=("object_center", "goal_center"),
            structured_state_dim=OBSERVATION_V2_FRAME_STATE_DIM,
            motion_dim=4,
            device_type="cpu",
            use_bf16=False,
            temperature=1.0,
        )
        self.fail_on_call = fail_on_call
        self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []

    @property
    def identity(self) -> PrecisionPredictorIdentity:
        return self._identity

    def predict(self, rgb_wrist, structured_state, geometric_motion):
        call_index = len(self.calls)
        self.calls.append(
            (
                np.asarray(rgb_wrist).copy(),
                np.asarray(structured_state).copy(),
                np.asarray(geometric_motion).copy(),
            )
        )
        if call_index == self.fail_on_call:
            raise RuntimeError("injected predictor failure")
        offset = call_index * 0.01
        return SimpleNamespace(
            keypoints=SimpleNamespace(
                normalized_uv=np.asarray(
                    [[[0.2 + offset, 0.3], [0.7, 0.8 - offset]]],
                    dtype=np.float32,
                ),
                peak_probability=np.asarray([[0.8, 0.85]], dtype=np.float32),
                normalized_entropy=np.asarray([[0.1, 0.15]], dtype=np.float32),
            ),
            visibility_probability=np.asarray([[0.95, 0.96]], dtype=np.float32),
            projection_validity_probability=np.asarray((0.9,), dtype=np.float32),
            keypoint_sigma_px=np.asarray(
                [[[0.2, 0.3], [0.25, 0.35]]],
                dtype=np.float32,
            ),
        )


def _geometry(window, frame_index: int) -> PrecisionGeometricMotionInput:
    wrist_index = OBSERVATION_MODALITIES.index("rgb_wrist")
    return PrecisionGeometricMotionInput(
        timestamp_s=float(window.modality_timestamp_s[frame_index, wrist_index]),
        motion=(frame_index * 1e-4, 0.0, 0.0, 0.0),
    )


def _provider(
    spec: RobotSpec,
    predictor: FakePredictor,
    *,
    enabled: bool,
    geometry_provider=_geometry,
) -> PrecisionDetectionProvider:
    proprio, force = _normalizers(spec)
    return PrecisionDetectionProvider(
        spec,
        predictor,
        proprio,
        force,
        geometry_provider,
        geometric_motion_provider_id="deployable-planar-motion/test-v1",
        proprio_stats_sha256="d" * 64,
        finger_force_stats_sha256="e" * 64,
        config=PrecisionDetectionProviderConfig(enabled=enabled),
    )


def test_precision_provider_is_disabled_by_default_without_running_model() -> None:
    spec = RobotSpec()
    predictor = FakePredictor()
    proprio, force = _normalizers(spec)
    provider = PrecisionDetectionProvider(
        spec,
        predictor,
        proprio,
        force,
        _geometry,
        geometric_motion_provider_id="deployable-planar-motion/test-v1",
        proprio_stats_sha256="d" * 64,
        finger_force_stats_sha256="e" * 64,
    )

    with pytest.raises(RuntimeError, match="默认关闭"):
        provider(_window(spec))

    assert predictor.calls == []
    assert provider.records == ()


def test_precision_provider_runs_valid_frames_oldest_to_newest_with_exact_time() -> None:
    spec = RobotSpec()
    predictor = FakePredictor()
    provider = _provider(spec, predictor, enabled=True)
    window = _window(spec)

    detections = provider(window)

    assert len(detections) == 4
    assert all(detection is not None for detection in detections)
    expected_timestamps = window.modality_timestamp_s[
        :, OBSERVATION_MODALITIES.index("rgb_wrist")
    ]
    assert [detection.timestamp_s for detection in detections] == pytest.approx(
        expected_timestamps
    )
    assert [int(call[0][0, 0, 0]) for call in predictor.calls] == [10, 11, 12, 13]
    assert all(
        call[1].shape == (OBSERVATION_V2_FRAME_STATE_DIM,)
        and call[1].dtype == np.float32
        for call in predictor.calls
    )
    frame_age_index = spec.proprio_dim + 3 + 6 + 3 + 6 + 2
    assert [float(call[1][frame_age_index]) for call in predictor.calls] == [
        0.0,
        0.0,
        0.0,
        0.0,
    ]
    np.testing.assert_allclose(
        [float(call[2][0]) for call in predictor.calls],
        (0.0, 1e-4, 2e-4, 3e-4),
    )

    record = provider.last_call
    assert record is not None and record.success
    assert record.detections_count == 4
    assert tuple(frame.status for frame in record.frame_records) == (
        PrecisionFrameStatus.PREDICTED,
    ) * 4
    assert all(
        frame.geometry_timestamp_s == pytest.approx(frame.wrist_timestamp_s)
        for frame in record.frame_records
    )
    assert record.provider_identity_sha256 == provider.identity.sha256
    serialized = json.dumps(record.to_dict(), sort_keys=True)
    assert "rgb_wrist" not in serialized
    assert "physical_proprio" not in serialized
    assert provider.records_jsonl == record.canonical_json() + "\n"
    assert len(provider.records_sha256) == 64


def test_precision_provider_preserves_padding_and_missing_wrist_as_none() -> None:
    spec = RobotSpec()
    prefix_predictor = FakePredictor()
    prefix_provider = _provider(spec, prefix_predictor, enabled=True)

    prefix = prefix_provider(_window(spec, frame_count=1))

    assert prefix[:3] == (None, None, None)
    assert prefix[3] is not None
    assert len(prefix_predictor.calls) == 1
    assert tuple(
        frame.status for frame in prefix_provider.last_call.frame_records
    ) == (
        PrecisionFrameStatus.PADDING,
        PrecisionFrameStatus.PADDING,
        PrecisionFrameStatus.PADDING,
        PrecisionFrameStatus.PREDICTED,
    )

    missing_predictor = FakePredictor()
    missing_provider = _provider(spec, missing_predictor, enabled=True)
    missing = missing_provider(_window(spec, missing_wrist_step=1))

    assert missing[1] is None
    assert len(missing_predictor.calls) == 3
    assert (
        missing_provider.last_call.frame_records[1].status
        == PrecisionFrameStatus.WRIST_RGB_MISSING
    )


def test_precision_provider_rejects_gt_or_timestamp_drift_and_records_failure() -> None:
    spec = RobotSpec()
    predictor = FakePredictor()

    def gt_geometry(window, frame_index):
        value = _geometry(window, frame_index)
        return PrecisionGeometricMotionInput(
            timestamp_s=value.timestamp_s,
            motion=value.motion,
            source=PredicateSource.EVALUATOR_GT,
        )

    provider = _provider(
        spec,
        predictor,
        enabled=True,
        geometry_provider=gt_geometry,
    )
    with pytest.raises(PrecisionDetectionProviderError, match="evaluator GT"):
        provider(_window(spec))

    failed = provider.last_call
    assert failed is not None and not failed.success
    assert failed.frame_records[0].status == PrecisionFrameStatus.ERROR
    assert all(
        frame.status == PrecisionFrameStatus.NOT_RUN_AFTER_ERROR
        for frame in failed.frame_records[1:]
    )
    assert predictor.calls == []

    def stale_geometry(window, frame_index):
        value = _geometry(window, frame_index)
        return PrecisionGeometricMotionInput(
            timestamp_s=value.timestamp_s + 0.01,
            motion=value.motion,
        )

    stale_provider = _provider(
        spec,
        FakePredictor(),
        enabled=True,
        geometry_provider=stale_geometry,
    )
    with pytest.raises(PrecisionDetectionProviderError, match="timestamp"):
        stale_provider(_window(spec))


def test_precision_provider_failure_is_transactional_and_reset_is_episode_local() -> None:
    spec = RobotSpec()
    predictor = FakePredictor(fail_on_call=1)
    provider = _provider(spec, predictor, enabled=True)

    with pytest.raises(PrecisionDetectionProviderError, match="injected"):
        provider(_window(spec))

    record = provider.last_call
    assert record is not None and not record.success
    assert tuple(frame.status for frame in record.frame_records) == (
        PrecisionFrameStatus.PREDICTED,
        PrecisionFrameStatus.ERROR,
        PrecisionFrameStatus.NOT_RUN_AFTER_ERROR,
        PrecisionFrameStatus.NOT_RUN_AFTER_ERROR,
    )
    assert record.detections_count == 1

    provider.reset()
    assert provider.records == ()
    assert provider.last_call is None
    assert provider.records_jsonl == ""


def test_precision_provider_records_bad_observation_or_predictor_identity_drift() -> None:
    spec = RobotSpec()
    predictor = FakePredictor()
    provider = _provider(spec, predictor, enabled=True)

    with pytest.raises(PrecisionDetectionProviderError, match="ObservationV2Window"):
        provider(object())

    invalid = provider.last_call
    assert invalid is not None and not invalid.success
    assert invalid.observation_timestamp_s is None
    assert all(
        frame.status == PrecisionFrameStatus.NOT_RUN_AFTER_ERROR
        for frame in invalid.frame_records
    )

    provider.reset()
    predictor._identity = replace(
        predictor.identity,
        checkpoint_sha256="f" * 64,
    )
    with pytest.raises(PrecisionDetectionProviderError, match="identity"):
        provider(_window(spec))

    drifted = provider.last_call
    assert drifted is not None and not drifted.success
    assert predictor.calls == []
