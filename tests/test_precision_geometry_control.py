import math

import numpy as np
import pytest

from robot_vla.precision.contracts import (
    PRECISION_MOTION_SEMANTICS,
    PrecisionControlMode,
    PrecisionMotionSpec,
)
from robot_vla.precision.control import (
    PrecisionConfidenceEvidence,
    PrecisionControlConfig,
    decide_precision_command,
)
from robot_vla.precision.geometry import (
    normalized_uv_to_base_z_plane,
    planar_tcp_delta_from_normalized_uv,
    project_base_point_to_normalized_uv,
)


def _transform(translation=(0.0, 0.0, 0.0), *, yaw_rad: float = 0.0) -> np.ndarray:
    value = np.eye(4, dtype=np.float32)
    cosine = math.cos(yaw_rad)
    sine = math.sin(yaw_rad)
    value[:2, :2] = np.asarray(((cosine, -sine), (sine, cosine)), dtype=np.float32)
    value[:3, 3] = np.asarray(translation, dtype=np.float32)
    return value


def _intrinsic() -> np.ndarray:
    return np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 40.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _center_uv() -> np.ndarray:
    return np.asarray(((50.0 + 0.5) / 100.0, (40.0 + 0.5) / 80.0), dtype=np.float32)


def _confidence(*, visibility=(1.0, 1.0)) -> PrecisionConfidenceEvidence:
    return PrecisionConfidenceEvidence(
        visibility_probability=np.asarray(visibility, dtype=np.float32),
        projection_validity_probability=1.0,
        heatmap_entropy=np.asarray((0.05, 0.05), dtype=np.float32),
        keypoint_sigma_px=np.full((2, 2), 0.2, dtype=np.float32),
        motion_sigma=np.asarray(
            (0.1e-3, 0.1e-3, 0.1e-3, math.radians(0.01)),
            dtype=np.float32,
        ),
        required_keypoints=np.asarray((True, False), dtype=np.bool_),
    )


def test_precision_motion_contract_is_not_the_vla_joint_delta_contract() -> None:
    spec = PrecisionMotionSpec()

    assert spec.semantics == PRECISION_MOTION_SEMANTICS
    assert spec.frame == "robot_base"
    assert spec.translation_unit == "meter"
    assert spec.rotation_unit == "radian"
    assert spec.components == (
        "delta_x_base_m",
        "delta_y_base_m",
        "delta_z_base_m",
        "delta_yaw_base_rad",
    )

    value = np.asarray((0.01, -0.01, 0.01, 1.0), dtype=np.float32)
    clipped = spec.clip_step(value)
    np.testing.assert_allclose(np.abs(clipped), spec.step_limits)


def test_opencv_ray_plane_round_trip_uses_base_frame_camera_pose() -> None:
    base_from_camera = _transform((1.0, 2.0, 0.0))

    intersection = normalized_uv_to_base_z_plane(
        _center_uv(),
        _intrinsic(),
        base_from_camera,
        (80, 100),
        plane_base_z_m=1.0,
    )

    np.testing.assert_allclose(intersection.point_base_m, (1.0, 2.0, 1.0), atol=1e-6)
    recovered_uv = project_base_point_to_normalized_uv(
        intersection.point_base_m,
        _intrinsic(),
        base_from_camera,
        (80, 100),
    )
    np.testing.assert_allclose(recovered_uv, _center_uv(), atol=1e-6)


def test_plane_behind_camera_fails_closed() -> None:
    with pytest.raises(ValueError, match="射线后方"):
        normalized_uv_to_base_z_plane(
            _center_uv(),
            _intrinsic(),
            _transform(),
            (80, 100),
            plane_base_z_m=-1.0,
        )


def test_camera_rotation_maps_pixel_ray_into_base_frame() -> None:
    right_of_center = np.asarray(
        ((60.0 + 0.5) / 100.0, (40.0 + 0.5) / 80.0),
        dtype=np.float32,
    )

    intersection = normalized_uv_to_base_z_plane(
        right_of_center,
        _intrinsic(),
        _transform(yaw_rad=math.pi / 2.0),
        (np.int64(80), np.int64(100)),
        plane_base_z_m=1.0,
    )

    np.testing.assert_allclose(intersection.point_base_m, (0.0, 0.1, 1.0), atol=1e-6)


def test_planar_geometry_outputs_explicit_commanded_tcp_target_delta() -> None:
    delta = planar_tcp_delta_from_normalized_uv(
        _center_uv(),
        _intrinsic(),
        _transform(),
        _transform((0.1, -0.2, 0.5)),
        (80, 100),
        target_plane_base_z_m=1.0,
        desired_tcp_offset_from_target_base_m=np.asarray(
            (0.0, 0.0, 0.2),
            dtype=np.float32,
        ),
        desired_delta_yaw_base_rad=0.1,
    )

    np.testing.assert_allclose(delta, (-0.1, 0.2, 0.7, 0.1), atol=1e-6)


def test_shadow_mode_records_but_never_applies_motion_head_residual() -> None:
    config = PrecisionControlConfig(mode=PrecisionControlMode.SHADOW)
    geometry = np.asarray((2.0e-3, -0.5e-3, 0.0, 0.0), dtype=np.float32)
    residual = np.asarray((0.2e-3, 0.1e-3, 0.0, 0.0), dtype=np.float32)

    decision = decide_precision_command(geometry, residual, _confidence(), config)

    assert decision.should_execute is True
    assert decision.mode == PrecisionControlMode.SHADOW
    assert decision.geometry_was_clipped is True
    np.testing.assert_allclose(
        decision.command_delta,
        config.motion_spec.clip_step(geometry),
    )
    assert not np.array_equal(decision.command_delta, decision.command_delta + residual)


def test_bounded_residual_mode_adds_only_clipped_learning_correction() -> None:
    config = PrecisionControlConfig(mode=PrecisionControlMode.BOUNDED_RESIDUAL)
    geometry = np.asarray((0.2e-3, 0.0, 0.0, 0.0), dtype=np.float32)
    residual = np.asarray((10.0e-3, -0.1e-3, 0.0, 0.0), dtype=np.float32)

    decision = decide_precision_command(geometry, residual, _confidence(), config)

    assert decision.should_execute is True
    assert decision.residual_was_clipped is True
    np.testing.assert_allclose(
        decision.bounded_residual,
        config.motion_spec.clip_residual(residual),
    )
    np.testing.assert_allclose(
        decision.command_delta,
        config.motion_spec.clip_step(geometry + decision.bounded_residual),
    )


def test_confidence_gate_checks_only_required_keypoints_and_returns_zero_on_failure() -> None:
    geometry = np.asarray((0.2e-3, 0.0, 0.0, 0.0), dtype=np.float32)
    residual = np.zeros(4, dtype=np.float32)

    optional_low = decide_precision_command(
        geometry,
        residual,
        _confidence(visibility=(1.0, 0.0)),
        PrecisionControlConfig(),
    )
    assert optional_low.should_execute is True

    required_low_confidence = _confidence(visibility=(0.5, 1.0))
    stopped = decide_precision_command(
        geometry,
        residual,
        required_low_confidence,
        PrecisionControlConfig(),
    )
    assert stopped.should_execute is False
    assert "keypoint_visibility" in stopped.gate_failures
    np.testing.assert_array_equal(stopped.command_delta, np.zeros(4, dtype=np.float32))


def test_nonfinite_geometry_stops_instead_of_being_clipped() -> None:
    geometry = np.asarray((np.nan, 0.0, 0.0, 0.0), dtype=np.float32)

    decision = decide_precision_command(
        geometry,
        np.zeros(4, dtype=np.float32),
        _confidence(),
        PrecisionControlConfig(),
    )

    assert decision.should_execute is False
    assert decision.gate_failures == ("nonfinite_geometry",)
    np.testing.assert_array_equal(decision.command_delta, np.zeros(4, dtype=np.float32))


def test_shadow_motion_diagnostics_cannot_block_geometry_command() -> None:
    confidence = _confidence()
    confidence = PrecisionConfidenceEvidence(
        visibility_probability=confidence.visibility_probability,
        projection_validity_probability=confidence.projection_validity_probability,
        heatmap_entropy=confidence.heatmap_entropy,
        keypoint_sigma_px=confidence.keypoint_sigma_px,
        motion_sigma=np.full(4, np.inf, dtype=np.float32),
        required_keypoints=confidence.required_keypoints,
    )
    geometry = np.asarray((0.2e-3, 0.0, 0.0, 0.0), dtype=np.float32)

    decision = decide_precision_command(
        geometry,
        np.asarray((np.nan, 0.0, 0.0, 0.0), dtype=np.float32),
        confidence,
        PrecisionControlConfig(mode=PrecisionControlMode.SHADOW),
    )

    assert decision.should_execute is True
    assert decision.gate_failures == ()
    assert "nonfinite_shadow_motion_head" in decision.diagnostic_warnings
    assert "invalid_shadow_motion_uncertainty" in decision.diagnostic_warnings
    np.testing.assert_allclose(decision.command_delta, geometry)


def test_bounded_residual_requires_calibrated_motion_uncertainty() -> None:
    confidence = _confidence()
    confidence = PrecisionConfidenceEvidence(
        visibility_probability=confidence.visibility_probability,
        projection_validity_probability=confidence.projection_validity_probability,
        heatmap_entropy=confidence.heatmap_entropy,
        keypoint_sigma_px=confidence.keypoint_sigma_px,
        motion_sigma=np.ones(4, dtype=np.float32),
        required_keypoints=confidence.required_keypoints,
    )

    decision = decide_precision_command(
        np.asarray((0.2e-3, 0.0, 0.0, 0.0), dtype=np.float32),
        np.zeros(4, dtype=np.float32),
        confidence,
        PrecisionControlConfig(mode=PrecisionControlMode.BOUNDED_RESIDUAL),
    )

    assert decision.should_execute is False
    assert "motion_uncertainty" in decision.gate_failures
