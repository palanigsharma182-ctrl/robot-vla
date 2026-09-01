from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_vla.precision.detection import PRECISION_TRACK_CONFIDENCE_SEMANTICS
from robot_vla.precision.held_out import (
    _calibrate,
    _held_out_metrics,
    _PredictionRows,
)


def _rows(*, scores: tuple[float, float], invalid: int = 0) -> _PredictionRows:
    return _PredictionRows(
        sample_count=1,
        valid=np.asarray((True, True), dtype=np.bool_),
        score=np.asarray(scores, dtype=np.float32),
        sigma_px=np.asarray((1.0, 1.0), dtype=np.float32),
        pixel_error=np.asarray((0.5, 0.75), dtype=np.float32),
        world_error_m=np.asarray((0.005, 0.010), dtype=np.float64),
        keypoint_index=np.asarray((0, 1), dtype=np.int8),
        invalid_backprojection_count=invalid,
    )


def test_confidence_calibration_declares_runtime_provider_semantics() -> None:
    calibration = _calibrate(
        checkpoint_sha256="a" * 64,
        data_identity_sha256="b" * 64,
        training_config_sha256="c" * 64,
        rows=_rows(scores=(0.8, 0.9)),
        temperature=1.0,
        target_coverage=0.5,
    )

    assert calibration.confidence_semantics == PRECISION_TRACK_CONFIDENCE_SEMANTICS
    assert calibration.confidence_threshold == pytest.approx(0.85)


def test_held_out_zero_accepted_samples_is_reported_without_crashing() -> None:
    calibration = _calibrate(
        checkpoint_sha256="a" * 64,
        data_identity_sha256="b" * 64,
        training_config_sha256="c" * 64,
        rows=_rows(scores=(0.8, 0.9)),
        temperature=1.0,
        target_coverage=0.5,
    )
    held_out = _held_out_metrics(
        checkpoint_sha256="a" * 64,
        data_identity_sha256="b" * 64,
        calibration=calibration,
        rows=_rows(scores=(0.1, 0.2)),
        mask_iou=0.8,
    )

    assert held_out.confidence_coverage == 0.0
    assert held_out.accepted_world_xy_error_p90_m is None
    assert held_out.perception_gate_passed

    invalid = _held_out_metrics(
        checkpoint_sha256="a" * 64,
        data_identity_sha256="b" * 64,
        calibration=calibration,
        rows=replace(_rows(scores=(0.1, 0.2)), invalid_backprojection_count=1),
        mask_iou=0.8,
    )
    assert not invalid.perception_gate_passed
