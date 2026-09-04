from types import SimpleNamespace

import numpy as np
import pytest

from robot_vla.precision.detection import (
    PRECISION_TRACK_CONFIDENCE_SEMANTICS,
    precision_prediction_to_wrist_detection,
)


def _prediction():
    return SimpleNamespace(
        keypoints=SimpleNamespace(
            normalized_uv=np.asarray(
                [
                    [[0.2, 0.3], [0.7, 0.8]],
                    [[0.25, 0.35], [0.75, 0.85]],
                ],
                dtype=np.float32,
            ),
            peak_probability=np.asarray(
                [[0.4, 0.5], [0.6, 0.7]],
                dtype=np.float32,
            ),
            normalized_entropy=np.asarray(
                [[0.2, 0.3], [0.1, 0.15]],
                dtype=np.float32,
            ),
        ),
        visibility_probability=np.asarray(
            [[0.95, 0.85], [0.9, 0.99]],
            dtype=np.float32,
        ),
        projection_validity_probability=np.asarray((0.8, 0.92), dtype=np.float32),
        keypoint_sigma_px=np.asarray(
            [
                [[0.3, 0.4], [0.5, 0.6]],
                [[0.2, 0.25], [0.1, 0.15]],
            ],
            dtype=np.float32,
        ),
    )


def test_precision_adapter_maps_named_keypoints_and_preserves_raw_evidence() -> None:
    result = precision_prediction_to_wrist_detection(
        _prediction(),
        keypoint_names=("goal_center", "object_center"),
        timestamp_s=1.25,
        batch_index=1,
    )

    detection = result.detection
    assert detection.timestamp_s == 1.25
    assert detection.object_normalized_uv == pytest.approx((0.75, 0.85))
    assert detection.goal_normalized_uv == pytest.approx((0.25, 0.35))
    # confidence 不混入依赖 heatmap 分辨率的 peak，而取 visibility/projection 的保守最小值。
    assert detection.object_confidence == pytest.approx(0.92)
    assert detection.goal_confidence == pytest.approx(0.9)
    assert result.object_evidence.peak_probability == pytest.approx(0.7)
    assert result.object_evidence.normalized_entropy == pytest.approx(0.15)
    assert result.object_evidence.sigma_px == pytest.approx((0.1, 0.15))
    assert result.confidence_semantics == PRECISION_TRACK_CONFIDENCE_SEMANTICS
    assert result.to_dict()["confidence_semantics"] == (
        "min-keypoint-visibility-and-projection-validity/v1"
    )


def test_precision_adapter_rejects_identity_shape_and_probability_drift() -> None:
    with pytest.raises(ValueError, match="缺少"):
        precision_prediction_to_wrist_detection(
            _prediction(),
            keypoint_names=("object_center", "other"),
            timestamp_s=0.0,
        )
    with pytest.raises(ValueError, match="batch_index"):
        precision_prediction_to_wrist_detection(
            _prediction(),
            keypoint_names=("goal_center", "object_center"),
            timestamp_s=0.0,
            batch_index=2,
        )

    invalid = _prediction()
    invalid.visibility_probability[0, 0] = 1.1
    with pytest.raises(ValueError, match="visibility_probability"):
        precision_prediction_to_wrist_detection(
            invalid,
            keypoint_names=("goal_center", "object_center"),
            timestamp_s=0.0,
        )


def test_precision_adapter_does_not_require_torch_for_replay_arrays() -> None:
    result = precision_prediction_to_wrist_detection(
        _prediction(),
        keypoint_names=("goal_center", "object_center"),
        timestamp_s=0.0,
    )

    assert isinstance(result.detection.object_normalized_uv, tuple)
    assert isinstance(result.object_evidence.sigma_px, tuple)


def test_precision_adapter_rejects_complex_arrays_or_non_string_keypoint_names() -> None:
    complex_prediction = _prediction()
    complex_prediction.keypoints.normalized_uv = (
        complex_prediction.keypoints.normalized_uv.astype(np.complex64) + 0.1j
    )
    with pytest.raises(ValueError, match="normalized_uv"):
        precision_prediction_to_wrist_detection(
            complex_prediction,
            keypoint_names=("goal_center", "object_center"),
            timestamp_s=0.0,
        )

    with pytest.raises(ValueError, match="keypoint_names"):
        precision_prediction_to_wrist_detection(
            _prediction(),
            keypoint_names=("goal_center", 1),
            timestamp_s=0.0,
        )
