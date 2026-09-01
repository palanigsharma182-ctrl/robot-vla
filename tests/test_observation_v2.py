from dataclasses import replace

import numpy as np
import pytest

from robot_vla.contracts import OBSERVATION_V2_VERSION, RobotSpec
from robot_vla.observation import (
    GL_CAMERA_FROM_CV_CAMERA,
    OBSERVATION_MODALITIES,
    OBSERVATION_V2_CONTROLLER_STATE_DIM,
    OBSERVATION_V2_FRAME_STATE_DIM,
    ObservationV2Frame,
    ObservationV2History,
    invert_se3,
    opengl_camera_to_opencv,
    rotation_6d_to_matrix,
    rotation_matrix_to_6d,
)


def _transform(translation=(0.0, 0.0, 0.0), yaw_rad: float = 0.0) -> np.ndarray:
    cosine = np.cos(yaw_rad)
    sine = np.sin(yaw_rad)
    value = np.asarray(
        [
            [cosine, -sine, 0.0, translation[0]],
            [sine, cosine, 0.0, translation[1]],
            [0.0, 0.0, 1.0, translation[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return value


def _frame(spec: RobotSpec, step: int) -> ObservationV2Frame:
    timestamp = step / spec.control_hz
    return ObservationV2Frame(
        rgb_external=np.full((8, 10, 3), step, dtype=np.uint8),
        rgb_wrist=np.full((6, 7, 3), step + 10, dtype=np.uint8),
        physical_proprio=np.full(spec.proprio_dim, step / 100.0, dtype=np.float32),
        base_from_tcp=_transform((step / 100.0, 0.0, 0.4), yaw_rad=step / 10.0),
        base_from_wrist_camera=_transform((0.2, step / 100.0, 0.5)),
        finger_force_n=np.asarray((step, step + 0.5), dtype=np.float32),
        timestamp_s=timestamp,
        modality_timestamp_s=np.full(
            len(OBSERVATION_MODALITIES),
            timestamp,
            dtype=np.float64,
        ),
        modality_valid=np.ones(len(OBSERVATION_MODALITIES), dtype=np.bool_),
    )


def test_opengl_to_opencv_camera_transform_flips_y_and_z_axes() -> None:
    world_from_gl = _transform((1.0, 2.0, 3.0))

    world_from_cv = opengl_camera_to_opencv(world_from_gl)

    np.testing.assert_allclose(world_from_cv, world_from_gl @ GL_CAMERA_FROM_CV_CAMERA)
    np.testing.assert_allclose(world_from_cv[:3, 3], (1.0, 2.0, 3.0))
    np.testing.assert_allclose(invert_se3(world_from_cv) @ world_from_cv, np.eye(4), atol=1e-6)


def test_rotation_6d_round_trip_uses_documented_column_order() -> None:
    rotation = _transform(yaw_rad=0.7)[:3, :3]

    encoded = rotation_matrix_to_6d(rotation)
    decoded = rotation_6d_to_matrix(encoded)

    np.testing.assert_allclose(encoded, rotation[:, :2].T.reshape(-1), atol=1e-6)
    np.testing.assert_allclose(decoded, rotation, atol=1e-6)


def test_history_uses_zero_padding_and_explicit_invalid_mask_at_episode_start() -> None:
    spec = RobotSpec()
    history = ObservationV2History(spec)
    history.append(_frame(spec, 0))

    window = history.snapshot("pick", previous_command_q=None, previous_action=None)

    assert window.version == OBSERVATION_V2_VERSION
    np.testing.assert_array_equal(window.history_valid, (False, False, False, True))
    assert np.count_nonzero(window.rgb_external[:3]) == 0
    assert np.count_nonzero(window.physical_proprio[:3]) == 0
    assert not window.modality_valid[:3].any()
    np.testing.assert_allclose(window.frame_age_s, 0.0)
    np.testing.assert_array_equal(window.controller_valid, (False, False))
    assert window.timestamp_s == 0.0
    np.testing.assert_array_equal(window.frame_timestamp_s, 0.0)
    np.testing.assert_array_equal(window.modality_timestamp_s, 0.0)


def test_history_is_four_consecutive_control_steps_oldest_to_newest() -> None:
    spec = RobotSpec()
    history = ObservationV2History(spec)
    for step in range(5):
        history.append(_frame(spec, step))
    previous_command = np.asarray(
        (0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0),
        dtype=np.float32,
    )
    previous_action = np.asarray((0.01,) * spec.arm_dof + (0.25,), dtype=np.float32)

    window = history.snapshot(
        "pick",
        previous_command_q=previous_command,
        previous_action=previous_action,
    )

    np.testing.assert_array_equal(window.history_valid, np.ones(4, dtype=np.bool_))
    np.testing.assert_array_equal(window.rgb_external[:, 0, 0, 0], (1, 2, 3, 4))
    np.testing.assert_allclose(window.frame_age_s, (0.15, 0.10, 0.05, 0.0), atol=1e-6)
    np.testing.assert_allclose(window.previous_command_q, previous_command)
    np.testing.assert_allclose(
        window.tracking_error,
        previous_command - window.physical_proprio[-1, : spec.arm_dof],
    )
    np.testing.assert_array_equal(window.controller_valid, (True, True))
    assert window.timestamp_s == pytest.approx(4 / spec.control_hz)
    np.testing.assert_allclose(window.frame_timestamp_s, (0.05, 0.10, 0.15, 0.20))
    np.testing.assert_allclose(
        window.modality_timestamp_s[:, 0],
        (0.05, 0.10, 0.15, 0.20),
    )
    state = window.frame_state(
        window.physical_proprio.copy(),
        np.log1p(window.finger_force_n).astype(np.float32),
    )
    assert state.shape == (4, OBSERVATION_V2_FRAME_STATE_DIM)
    assert window.controller_state().shape == (OBSERVATION_V2_CONTROLLER_STATE_DIM,)


def test_history_rejects_nonconsecutive_or_future_frames() -> None:
    spec = RobotSpec()
    history = ObservationV2History(spec)
    history.append(_frame(spec, 0))

    with pytest.raises(ValueError, match="连续控制步"):
        history.append(_frame(spec, 2))

    future = _frame(spec, 1)
    with pytest.raises(ValueError, match="未来观测"):
        ObservationV2Frame(
            rgb_external=future.rgb_external,
            rgb_wrist=future.rgb_wrist,
            physical_proprio=future.physical_proprio,
            base_from_tcp=future.base_from_tcp,
            base_from_wrist_camera=future.base_from_wrist_camera,
            finger_force_n=future.finger_force_n,
            timestamp_s=future.timestamp_s,
            modality_timestamp_s=np.full(
                len(OBSERVATION_MODALITIES),
                future.timestamp_s + 0.01,
                dtype=np.float64,
            ),
            modality_valid=future.modality_valid,
        )


def test_window_preserves_exact_timestamps_and_rejects_cross_frame_skew() -> None:
    spec = RobotSpec()
    history = ObservationV2History(spec)
    for step in range(4):
        history.append(_frame(spec, step))
    window = history.snapshot("pick")
    drifted = window.modality_timestamp_s.copy()
    drifted[0, OBSERVATION_MODALITIES.index("rgb_wrist")] = (
        window.frame_timestamp_s[0] + 0.001
    )

    with pytest.raises(ValueError, match="所属 frame"):
        replace(window, modality_timestamp_s=drifted).validate(spec)


def test_history_zeroes_invalid_modalities_and_runtime_gate_requires_complete_current() -> None:
    spec = RobotSpec()
    source = _frame(spec, 0)
    validity = source.modality_valid.copy()
    validity[3] = False
    frame = ObservationV2Frame(
        rgb_external=source.rgb_external,
        rgb_wrist=source.rgb_wrist,
        physical_proprio=source.physical_proprio,
        base_from_tcp=source.base_from_tcp,
        base_from_wrist_camera=source.base_from_wrist_camera,
        finger_force_n=source.finger_force_n,
        timestamp_s=source.timestamp_s,
        modality_timestamp_s=source.modality_timestamp_s,
        modality_valid=validity,
    )
    history = ObservationV2History(spec)
    history.append(frame)

    window = history.snapshot("pick")

    np.testing.assert_array_equal(window.tcp_position[-1], 0.0)
    np.testing.assert_array_equal(window.tcp_rotation_6d[-1], 0.0)
    assert window.modality_age_s[-1, 3] == 0.0
    window.validate(spec)
    with pytest.raises(ValueError, match="当前控制步必须六模态完整有效"):
        window.validate(spec, require_current_complete=True)
