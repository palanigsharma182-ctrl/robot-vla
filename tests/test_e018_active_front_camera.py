from __future__ import annotations

import math

import numpy as np
import pytest

from robot_vla.precision.active_front_camera import (
    ExternalCameraMotionState,
    FrontCameraOrientationMode,
    FrontCameraViewpoint,
    compose_camera_orientation_wxyz,
    measurement_write_eligible,
    quaternion_angular_distance_rad,
    quaternion_multiply_wxyz,
    rotation_angular_distance_rad,
    sample_translation_path,
    smootherstep,
)


def _viewpoint() -> FrontCameraViewpoint:
    position = np.asarray((0.3, -0.16, 0.48), dtype=np.float64)
    target = np.asarray((-0.1, 0.0, 0.1), dtype=np.float64)
    direction = target - position
    return FrontCameraViewpoint(
        viewpoint_id="LEFT_LOW",
        lateral_anchor="LEFT",
        vertical_anchor="LOW",
        position_world_m=tuple(position),
        look_at_world_m=tuple(target),
        yaw_rad=math.atan2(float(direction[1]), float(direction[0])),
        pitch_rad=math.atan2(float(direction[2]), float(np.linalg.norm(direction[:2]))),
        roll_rad=0.0,
    )


def test_viewpoint_requires_registered_look_at_angles_and_fixed_roll() -> None:
    viewpoint = _viewpoint()
    viewpoint.validate()

    with pytest.raises(ValueError, match="yaw_rad"):
        FrontCameraViewpoint(**{**viewpoint.__dict__, "yaw_rad": viewpoint.yaw_rad + 0.1}).validate()
    with pytest.raises(ValueError, match="roll_rad"):
        FrontCameraViewpoint(**{**viewpoint.__dict__, "roll_rad": 0.1}).validate()


def test_smootherstep_translation_has_exact_endpoint_and_no_teleport() -> None:
    assert smootherstep(0.0) == 0.0
    assert smootherstep(1.0) == 1.0
    path = sample_translation_path((0.3, 0.0, 0.6), (0.3, -0.16, 0.48), steps=30)

    assert path.shape == (30, 3)
    np.testing.assert_allclose(path[-1], (0.3, -0.16, 0.48), atol=0.0, rtol=0.0)
    assert not np.array_equal(path[0], path[-1])
    assert np.all(np.diff(path[:, 1]) < 0.0)
    assert np.all(np.diff(path[:, 2]) < 0.0)


def test_quaternion_and_rotation_distance_are_sign_and_roundoff_safe() -> None:
    quaternion = np.asarray((0.1, -0.2, 0.3, 0.9), dtype=np.float64)
    assert quaternion_angular_distance_rad(quaternion, -quaternion) == pytest.approx(0.0)
    assert rotation_angular_distance_rad(np.eye(3), np.eye(3)) == pytest.approx(0.0)
    rounded_rotation = np.eye(3)
    rounded_rotation[0, 0] = 0.9999998
    assert rotation_angular_distance_rad(
        rounded_rotation,
        rounded_rotation,
    ) == pytest.approx(0.0)


def test_degenerate_rotations_and_zero_quaternions_fail_closed() -> None:
    for matrix in (np.zeros((3, 3)), np.diag([1., 1., -1.]), np.eye(3)*2):
        with pytest.raises(ValueError, match="SO"):
            rotation_angular_distance_rad(matrix, np.eye(3))
    with pytest.raises(ValueError, match="四元数"):
        quaternion_multiply_wxyz(np.zeros(4), np.array([1., 0., 0., 0.]))
    with pytest.raises(TypeError, match="settled"):
        measurement_write_eligible(ExternalCameraMotionState.COLLECT, settled="false")


def test_cross_orientation_modes_reject_roll_and_diagonal_offsets() -> None:
    FrontCameraOrientationMode("CENTER", 0.0, 0.0).validate()
    FrontCameraOrientationMode("YAW_LEFT", math.radians(12.0), 0.0).validate()
    FrontCameraOrientationMode("PITCH_UP", 0.0, math.radians(8.0)).validate()

    with pytest.raises(ValueError, match="diagonal"):
        FrontCameraOrientationMode(
            "DIAGONAL",
            math.radians(12.0),
            math.radians(8.0),
        ).validate()
    with pytest.raises(ValueError, match="roll"):
        FrontCameraOrientationMode("ROLL", 0.0, 0.0, math.radians(5.0)).validate()


def test_camera_orientation_composition_uses_local_cross_offsets() -> None:
    nominal = np.asarray((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    yaw = FrontCameraOrientationMode("YAW_LEFT", math.radians(12.0), 0.0)
    pitch = FrontCameraOrientationMode("PITCH_UP", 0.0, math.radians(8.0))

    yaw_result = compose_camera_orientation_wxyz(nominal, yaw)
    pitch_result = compose_camera_orientation_wxyz(nominal, pitch)
    assert quaternion_angular_distance_rad(nominal, yaw_result) == pytest.approx(
        math.radians(12.0)
    )
    assert quaternion_angular_distance_rad(nominal, pitch_result) == pytest.approx(
        math.radians(8.0)
    )
    np.testing.assert_allclose(
        quaternion_multiply_wxyz(nominal, yaw_result),
        yaw_result,
        atol=1e-12,
    )


@pytest.mark.parametrize("state", list(ExternalCameraMotionState))
def test_only_settled_collect_frames_are_measurement_write_eligible(
    state: ExternalCameraMotionState,
) -> None:
    assert measurement_write_eligible(state, settled=False) is False
    assert measurement_write_eligible(state, settled=True) is (
        state is ExternalCameraMotionState.COLLECT
    )
