from __future__ import annotations

import numpy as np
import pytest

from robot_vla.precision.calibrated_front_provider import (
    ScalarCovarianceCalibration,
    build_calibrated_object_write_evidence,
    build_calibrated_provider_identity,
    build_stable_camera_calibration_identity,
)

SHA_A = "a" * 64
SHA_B = "b" * 64
SHA_C = "c" * 64


def _calibration(scale: float = 4.0) -> ScalarCovarianceCalibration:
    return ScalarCovarianceCalibration(
        scale_factor=scale,
        support_count=40,
        order_statistic_k=39,
        quantile_score=scale * 5.991,
        scoring_ledger_sha256=SHA_A,
        config_sha256=SHA_B,
        validation_data_identity_sha256=SHA_C,
        source_identity_sha256="d" * 64,
    )


def test_dynamic_pose_is_not_part_of_stable_camera_identity() -> None:
    intrinsic = np.asarray(((100.0, 0.0, 64.0), (0.0, 100.0, 64.0), (0.0, 0.0, 1.0)))
    pose_a = np.eye(4, dtype=np.float64)
    pose_b = pose_a.copy()
    pose_b[:3, 3] = (0.04, -0.03, 0.02)
    stable_a, record_a = build_stable_camera_calibration_identity(
        camera_uid="hand_camera",
        primitive_id="WRIST_NATIVE",
        intrinsic_cv=intrinsic,
        actual_base_from_camera_cv=pose_a,
        covariance_calibration_identity_sha256=SHA_A,
        source_training_camera="hand_camera",
        target_camera="hand_camera",
        frame_convention="robot-base-from-opencv-optical-camera/v1",
    )
    stable_b, record_b = build_stable_camera_calibration_identity(
        camera_uid="hand_camera",
        primitive_id="WRIST_NATIVE",
        intrinsic_cv=intrinsic,
        actual_base_from_camera_cv=pose_b,
        covariance_calibration_identity_sha256=SHA_A,
        source_training_camera="hand_camera",
        target_camera="hand_camera",
        frame_convention="robot-base-from-opencv-optical-camera/v1",
    )
    assert stable_a == stable_b
    assert (
        record_a["actual_base_from_camera_cv_sha256"]
        != record_b["actual_base_from_camera_cv_sha256"]
    )


def test_static_camera_or_provider_provenance_changes_identity() -> None:
    intrinsic = np.asarray(((100.0, 0.0, 64.0), (0.0, 100.0, 64.0), (0.0, 0.0, 1.0)))
    pose = np.eye(4, dtype=np.float64)

    def camera_identity(camera_uid: str, focal: float) -> str:
        value = intrinsic.copy()
        value[0, 0] = focal
        stable, _ = build_stable_camera_calibration_identity(
            camera_uid=camera_uid,
            primitive_id="WRIST_NATIVE",
            intrinsic_cv=value,
            actual_base_from_camera_cv=pose,
            covariance_calibration_identity_sha256=SHA_A,
            source_training_camera="hand_camera",
            target_camera="hand_camera",
            frame_convention="robot-base-from-opencv-optical-camera/v1",
        )
        return str(stable["identity_sha256"])

    assert camera_identity("hand_camera", 100.0) != camera_identity(
        "replacement_hand_camera", 100.0
    )
    assert camera_identity("hand_camera", 100.0) != camera_identity("hand_camera", 101.0)

    common = {
        "checkpoint_parameter_sha256": SHA_B,
        "checkpoint_provenance_sha256": SHA_C,
        "model_config_sha256": "d" * 64,
        "proprio_stats_sha256": "e" * 64,
        "proprio_normalizer_sha256": "f" * 64,
        "finger_force_stats_sha256": "0" * 64,
        "finger_force_normalizer_sha256": "1" * 64,
        "adapter_config_sha256": "2" * 64,
        "stable_camera_calibration_identity_sha256": "3" * 64,
        "covariance_calibration_identity_sha256": "4" * 64,
        "primitive_id": "WRIST_NATIVE",
        "geometric_motion_provider_id": "safe-hold/v1",
        "source_training_camera": "hand_camera",
        "target_camera": "hand_camera",
        "frame_convention": "robot-base-from-opencv-optical-camera/v1",
    }
    first = build_calibrated_provider_identity(checkpoint_sha256=SHA_A, **common)
    second = build_calibrated_provider_identity(checkpoint_sha256="5" * 64, **common)
    assert first["identity_sha256"] != second["identity_sha256"]


def test_calibration_reaches_covariance_and_object_write_evidence() -> None:
    calibration = _calibration(scale=4.0)
    raw_sigma = np.asarray((0.2, 0.3), dtype=np.float64)
    raw_covariance = np.diag((1e-6, 4e-6, 0.0))
    calibrated_covariance = calibration.calibrate_covariance(raw_covariance)
    assert np.array_equal(calibrated_covariance, raw_covariance * 4.0)

    evidence, calibrated_sigma = build_calibrated_object_write_evidence(
        calibration=calibration,
        raw_sigma_xy_px=raw_sigma,
        visibility_probability=0.9,
        projection_validity_probability=0.9,
        object_mask_probability=0.9,
        goal_mask_probability=0.1,
        normalized_entropy=0.1,
        geometry_valid=True,
        min_object_mask_probability=0.5,
        max_goal_mask_probability=0.5,
    )
    assert np.allclose(calibrated_sigma, raw_sigma * 2.0)
    assert evidence.radial_sigma_px == pytest.approx(np.linalg.norm(raw_sigma * 2.0))
    assert evidence.score == pytest.approx(1.0 / (1.0 + np.linalg.norm(raw_sigma * 2.0)))


def test_calibration_rejects_non_psd_covariance() -> None:
    with pytest.raises(ValueError, match="PSD"):
        _calibration().calibrate_covariance(np.diag((1e-6, -1e-6, 0.0)))
