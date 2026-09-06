from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_vla.observation import GL_CAMERA_FROM_CV_CAMERA
from robot_vla.precision.active_external_observation import (
    ACTUAL_EXTERNAL_POSE_SOURCE,
    ActiveExternalObservation,
    base_camera_round_trip_error_m,
    extract_active_external_observation,
    project_base_point,
)
from robot_vla.precision.active_front_camera import ExternalCameraMotionState


def _observation(*, actual_world_from_gl: np.ndarray | None = None) -> dict[str, object]:
    if actual_world_from_gl is None:
        actual_world_from_gl = np.eye(4, dtype=np.float64)
    return {
        "sensor_data": {
            "base_camera": {
                "rgb": np.zeros((1, 8, 10, 3), dtype=np.uint8),
            },
        },
        "sensor_param": {
            "base_camera": {
                "intrinsic_cv": np.asarray(
                    [[[4.0, 0.0, 4.5], [0.0, 4.0, 3.5], [0.0, 0.0, 1.0]]],
                    dtype=np.float64,
                ),
                "cam2world_gl": actual_world_from_gl[None],
            },
        },
    }


def _extract(
    *,
    observation: dict[str, object] | None = None,
    state: ExternalCameraMotionState = ExternalCameraMotionState.COLLECT,
    settled: bool = True,
) -> ActiveExternalObservation:
    return extract_active_external_observation(
        _observation() if observation is None else observation,
        camera_uid="base_camera",
        world_from_robot_base=np.eye(4, dtype=np.float64),
        commanded_world_from_external_camera_gl=np.eye(4, dtype=np.float64),
        episode_id="episode-001",
        request_id="request-001",
        observation_sequence_id="observation-007",
        camera_command_sequence_id="camera-command-001",
        control_tick=7,
        control_timestamp_s=0.35,
        rgb_timestamp_s=0.35,
        camera_pose_timestamp_s=0.35,
        camera_motion_state=state,
        viewpoint_primitive_id="LEFT_LOW__YAW_LEFT",
        settled=settled,
        maximum_rotation_projection_error_frobenius=1e-6,
    )


def test_extract_uses_actual_pose_from_same_observation_and_keeps_v2_separate() -> None:
    actual = np.eye(4, dtype=np.float64)
    actual[:3, 3] = (0.3, -0.16, 0.48)
    commanded = np.eye(4, dtype=np.float64)
    commanded[:3, 3] = (9.0, 9.0, 9.0)
    sidecar = extract_active_external_observation(
        _observation(actual_world_from_gl=actual),
        camera_uid="base_camera",
        world_from_robot_base=np.eye(4, dtype=np.float64),
        commanded_world_from_external_camera_gl=commanded,
        episode_id="episode-001",
        request_id="request-001",
        observation_sequence_id="observation-007",
        camera_command_sequence_id="camera-command-001",
        control_tick=7,
        control_timestamp_s=0.35,
        rgb_timestamp_s=0.34,
        camera_pose_timestamp_s=0.35,
        camera_motion_state=ExternalCameraMotionState.MOVE_TO_VIEW,
        viewpoint_primitive_id="LEFT_LOW__YAW_LEFT",
        settled=False,
        maximum_rotation_projection_error_frobenius=1e-6,
    )

    assert sidecar.actual_pose_source == ACTUAL_EXTERNAL_POSE_SOURCE
    assert np.array_equal(sidecar.actual_world_from_external_camera_gl, actual)
    assert not np.array_equal(
        sidecar.actual_world_from_external_camera_gl,
        sidecar.commanded_world_from_external_camera_gl,
    )
    assert np.allclose(sidecar.base_from_external_camera_cv, actual @ GL_CAMERA_FROM_CV_CAMERA)
    assert sidecar.rgb_pose_skew_s == pytest.approx(0.01)
    assert sidecar.memory_write_eligible is False
    assert sidecar.ledger_record()["contains_gt"] is False


def test_camera_uid_is_bound_to_pose_source_and_audit_identity() -> None:
    first = _extract()
    second = replace(first, camera_uid="hand_camera", actual_pose_source=None)
    assert first.camera_uid == "base_camera"
    assert second.actual_pose_source == "same-observation.sensor_param.hand_camera.cam2world_gl/v1"
    assert first.audit_digest() != second.audit_digest()
    assert second.ledger_record()["camera_uid"] == "hand_camera"
    with pytest.raises(ValueError, match="actual pose source"):
        replace(first, camera_uid="hand_camera")


@pytest.mark.parametrize(
    ("state", "settled", "eligible"),
    [
        (ExternalCameraMotionState.MOVE_TO_VIEW, False, False),
        (ExternalCameraMotionState.SETTLE_AT_VIEW, True, False),
        (ExternalCameraMotionState.COLLECT, False, False),
        (ExternalCameraMotionState.COLLECT, True, True),
        (ExternalCameraMotionState.RETURN_HOME, True, False),
    ],
)
def test_write_eligibility_remains_derived_from_motion_state(
    state: ExternalCameraMotionState,
    settled: bool,
    eligible: bool,
) -> None:
    assert _extract(state=state, settled=settled).memory_write_eligible is eligible


def test_base_camera_round_trip_and_opencv_projection() -> None:
    sidecar = _extract()
    # world_from_camera_gl=I 时，OpenCV forward(+Z) 对应 world -Z。
    point_base = np.asarray((0.0, 0.0, -2.0), dtype=np.float64)

    projection = project_base_point(sidecar, point_base)

    assert projection["projection_valid"] is True
    assert projection["in_frame"] is True
    assert projection["depth_m"] == pytest.approx(2.0)
    assert projection["uv_px"] == pytest.approx([4.5, 3.5])
    assert base_camera_round_trip_error_m(sidecar, point_base) <= 1e-12


def test_float32_sensor_rotation_is_canonicalized_only_for_derived_transform() -> None:
    actual = np.eye(4, dtype=np.float64)
    actual[0, 1] = 1e-7

    sidecar = _extract(observation=_observation(actual_world_from_gl=actual))

    assert np.array_equal(sidecar.actual_world_from_external_camera_gl, actual)
    rotation = sidecar.base_from_external_camera_cv[:3, :3]
    assert np.allclose(rotation.T @ rotation, np.eye(3), rtol=0.0, atol=1e-12)
    audit = sidecar.actual_rotation_projection_audit
    assert audit.correction_frobenius > 0.0
    assert audit.orthogonality_error_before_frobenius > 0.0
    assert audit.orthogonality_error_after_frobenius <= 1e-12
    assert audit.determinant_after == pytest.approx(1.0)
    assert base_camera_round_trip_error_m(
        sidecar,
        np.asarray((0.1, -0.2, -1.0)),
    ) <= 1e-12


def test_excessive_rotation_projection_correction_fails_closed() -> None:
    actual = np.eye(4, dtype=np.float64)
    actual[0, 1] = 2e-6

    with pytest.raises(ValueError, match="超过冻结容差"):
        _extract(observation=_observation(actual_world_from_gl=actual))


def test_audit_digest_is_stable_and_changes_with_actual_pose() -> None:
    first = _extract()
    second = _extract()
    moved = np.eye(4, dtype=np.float64)
    moved[0, 3] = 0.1
    third = _extract(observation=_observation(actual_world_from_gl=moved))

    assert first.audit_digest() == second.audit_digest()
    assert first.audit_digest() != third.audit_digest()


def test_missing_actual_pose_fails_closed() -> None:
    observation = _observation()
    del observation["sensor_param"]["base_camera"]["cam2world_gl"]

    with pytest.raises(ValueError, match="同次 observation"):
        _extract(observation=observation)


def test_commanded_pose_cannot_overwrite_actual_pose_after_construction() -> None:
    sidecar = _extract()
    replacement = np.eye(4, dtype=np.float64)
    replacement[1, 3] = 4.0

    changed = replace(
        sidecar,
        commanded_world_from_external_camera_gl=replacement,
    )

    assert np.array_equal(
        changed.actual_world_from_external_camera_gl,
        sidecar.actual_world_from_external_camera_gl,
    )
    assert not np.array_equal(
        changed.commanded_world_from_external_camera_gl,
        changed.actual_world_from_external_camera_gl,
    )


def test_future_or_unbound_timestamps_fail_closed() -> None:
    sidecar = _extract()

    with pytest.raises(ValueError, match="晚于控制 Tick"):
        replace(sidecar, rgb_timestamp_s=0.36)
    with pytest.raises(ValueError, match="非空字符串"):
        replace(sidecar, observation_sequence_id="")
