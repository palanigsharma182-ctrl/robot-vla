"""验证真实夹爪目标、actual/command差分、尾部mask和异常拒绝。"""
from types import SimpleNamespace
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from experiments.tcp_atomic_skills.data import command_chunk, verify_commands, action_spec
from experiments.tcp_memory_control.geometry import TCPActionSpec, apply_delta


def test_command_and_gripper_roundtrip_in_rotated_anchor():
    actual = np.eye(4); actual[:3, :3] = Rotation.from_euler('z', 90, degrees=True).as_matrix()
    targets = []
    for i in range(3):
        t = actual.copy(); t[0, 3] = .002*(i+1); targets.append(t)
    normalized, mask, _ = command_chunk(actual, targets, np.array([1., 0., .25]), 0)
    physical = action_spec().denormalize(normalized)
    assert mask.tolist() == [True]*3 + [False]*13
    np.testing.assert_allclose(physical[:3, 6], [1., 0., .25])
    reconstructed = actual
    for i in range(3):
        reconstructed = apply_delta(reconstructed, physical[i, :6], actual)
        np.testing.assert_allclose(reconstructed, targets[i], atol=1e-8)
    assert np.all(normalized[~mask] == 0)


def test_first_step_uses_actual_not_hidden_previous_command():
    actual = np.eye(4); target = np.eye(4); target[0, 3] = .003
    normalized, _, _ = command_chunk(actual, [target], [0.], 0)
    assert normalized[0, 0] == pytest.approx(.12)


def test_oversized_label_is_rejected():
    t = np.eye(4); t[0, 3] = .03
    with pytest.raises(ValueError, match='超过'):
        command_chunk(np.eye(4), [t], [1.], 0)


def test_missing_command_is_not_reconstructed_from_old_joint_labels():
    with pytest.raises(ValueError, match='缺少'):
        verify_commands(SimpleNamespace(commanded_joint_target_rad=None, previous_command_q_rad=None))
