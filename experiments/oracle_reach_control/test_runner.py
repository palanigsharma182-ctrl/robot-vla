"""显式位置控制的最小几何、动作基准和共同输入边界测试。"""

from types import SimpleNamespace

import numpy as np
import pytest

from experiments.oracle_reach_control.runner import (
    OpenGripperRuntime, PositionRuntime, SETTINGS, position_step,
)
from robot_vla.contracts import RobotSpec
from robot_vla.runtime.policy_runtime import RuntimeActionChunk, SamplingTrace


def test_dls_reduces_error_and_handles_rank_deficiency():
    matrix = np.asarray([[1, 0, 0, 0, 0, 0, 0], [0, 2, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0]], dtype=float)
    fk = lambda q: matrix @ q
    q, target = np.zeros(7), np.asarray([0.1, -0.1, 0.05])
    dq = position_step(fk, q, target, limits=np.full(7, 0.05), **SETTINGS)
    assert np.linalg.norm(target - fk(q + dq)) < np.linalg.norm(target - fk(q))
    assert np.linalg.norm(fk(q + dq) - fk(q)) <= SETTINGS['max_cartesian_step_m'] + 1e-8
    assert np.isfinite(dq).all() and np.max(np.abs(dq)) <= 0.05
    np.testing.assert_array_equal(dq[2:], 0)


def test_runtime_labels_rebase_on_commanded_target_not_actual_q():
    spec = RobotSpec()
    q = np.asarray([0, -0.5, 0, -1.5, 0, 1.5, 0], dtype=float)
    fk = lambda value: np.asarray(value[:3])
    delta = np.asarray([0.02, 0, 0])
    runtime = PositionRuntime(fk, lambda: np.r_[delta, np.linalg.norm(delta)], spec, 42)
    reference = q.copy()
    reference[0] += 0.01
    runtime.command_reference = lambda: reference
    observation = SimpleNamespace(physical_proprio=np.r_[q, np.zeros(7), 1].astype(np.float32))
    chunk = runtime.infer_action_chunk(observation)
    desired = q + position_step(fk, q, fk(q) + delta, limits=runtime.adapter.delta_limits, **SETTINGS)
    np.testing.assert_allclose(reference + chunk.physical_action[0, :7], desired, atol=1e-7)
    assert chunk.physical_action[0, 0] < 0  # actual q仍在目标后方，但旧command已更靠前。
    assert chunk.physical_action.shape == (16, 8)
    assert np.max(np.abs(chunk.normalized_action)) <= 1.0
    np.testing.assert_array_equal(chunk.physical_action[:, -1], 1)


def test_common_open_gripper_preserves_arm_and_original_chunk():
    spec = RobotSpec()
    physical = np.zeros((16, 8), dtype=np.float32)
    physical[:, 0] = 0.01
    original = RuntimeActionChunk(physical.copy(), physical, (0, 0), 1, SamplingTrace(42, 0))
    runtime = OpenGripperRuntime(SimpleNamespace(infer_action_chunk=lambda _: original), spec)
    result = runtime.infer_action_chunk(None)
    np.testing.assert_array_equal(result.physical_action[:, :7], original.physical_action[:, :7])
    np.testing.assert_array_equal(original.physical_action[:, -1], 0)
    np.testing.assert_array_equal(result.physical_action[:, -1], 1)
    np.testing.assert_array_equal(result.normalized_action[:, -1], 1)


def test_dls_rejects_nonfinite_geometry():
    with pytest.raises(ValueError, match="有限"):
        position_step(lambda q: q[:3], np.zeros(7), [np.nan, 0, 0], limits=np.full(7, .05), **SETTINGS)
