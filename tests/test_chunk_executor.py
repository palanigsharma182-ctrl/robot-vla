import numpy as np
import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.execution.chunk_executor import (
    FrankaControlState,
    RecedingHorizonChunkExecutor,
)


class FakeController:
    def __init__(self, spec: RobotSpec, *, fail_on_send: int | None = None) -> None:
        self.spec = spec
        self.q = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)
        self.gripper = 0.5
        self.fail_on_send = fail_on_send
        self.send_attempts = 0
        self.actions: list[np.ndarray] = []
        self.hold_calls = 0

    def read_state(self) -> FrankaControlState:
        return FrankaControlState(self.q.copy(), self.gripper)

    def send_action(self, controller_action: np.ndarray) -> None:
        attempt = self.send_attempts
        self.send_attempts += 1
        if attempt == self.fail_on_send:
            raise RuntimeError("simulated controller failure")
        action = np.asarray(controller_action, dtype=np.float32)
        self.actions.append(action.copy())
        self.q += action[: self.spec.arm_dof] * self.spec.maniskill_arm_delta_range_rad
        self.gripper = float((action[-1] + 1.0) * 0.5)

    def hold_current(self) -> None:
        self.hold_calls += 1


def _chunk(spec: RobotSpec, delta: float = 0.01, gripper: float = 0.25) -> np.ndarray:
    chunk = np.zeros((spec.action_horizon, spec.action_dim), dtype=np.float32)
    chunk[:, 0] = delta
    chunk[:, -1] = gripper
    return chunk


def test_executor_runs_only_prefix_and_tracks_cumulative_position_targets() -> None:
    spec = RobotSpec()
    controller = FakeController(spec)

    result = RecedingHorizonChunkExecutor(spec).execute(_chunk(spec), controller)

    assert result.success is True
    assert result.executed_steps == spec.execute_steps == 4
    assert len(controller.actions) == 4
    np.testing.assert_allclose([action[0] for action in controller.actions], 0.1, atol=1e-6)
    np.testing.assert_allclose([action[-1] for action in controller.actions], -0.5)
    assert controller.q[0] == pytest.approx(0.04)
    assert controller.gripper == pytest.approx(0.25)
    assert controller.hold_calls == 0


def test_executor_honors_controller_boundary_stop_after_current_step() -> None:
    spec = RobotSpec()
    controller = FakeController(spec)
    controller.chunk_stop_requested = False
    original_send = controller.send_action

    def stop_after_second_action(controller_action: np.ndarray) -> None:
        original_send(controller_action)
        if len(controller.actions) == 2:
            controller.chunk_stop_requested = True

    controller.send_action = stop_after_second_action

    result = RecedingHorizonChunkExecutor(spec).execute(_chunk(spec), controller)

    assert result.success is True
    assert result.executed_steps == 2
    assert len(controller.actions) == 2
    assert controller.hold_calls == 0


def test_executor_stops_chunk_and_holds_without_loosening_after_controller_failure() -> None:
    spec = RobotSpec()
    controller = FakeController(spec, fail_on_send=1)

    result = RecedingHorizonChunkExecutor(spec).execute(_chunk(spec), controller)

    assert result.success is False
    assert result.executed_steps == 1
    assert result.failure_stage == "controller_step"
    assert result.hold_succeeded is True
    assert controller.hold_calls == 1
    assert len(controller.actions) == 1
    assert controller.gripper == pytest.approx(0.25)


def test_executor_saturates_tracking_correction_at_step_limit_and_records_it() -> None:
    spec = RobotSpec()
    controller = FakeController(spec)

    def ignore_motion(controller_action: np.ndarray) -> None:
        controller.actions.append(controller_action.copy())
        controller.gripper = float((controller_action[-1] + 1.0) * 0.5)

    controller.send_action = ignore_motion
    result = RecedingHorizonChunkExecutor(spec).execute(
        _chunk(spec, delta=0.05),
        controller,
    )

    assert result.success is True
    assert result.executed_steps == 2
    assert result.failure_stage is None
    assert controller.hold_calls == 0
    assert result.correction_saturation_steps == 1
    assert result.requested_correction_abs_max_rad == pytest.approx(0.1)
    assert result.applied_correction_abs_max_rad == pytest.approx(0.05)
    assert result.replan_required is True
    assert result.anomaly_kind == "tracking_correction_saturation"
    assert len(controller.actions) == 2
    assert max(abs(float(action[0])) for action in controller.actions) == pytest.approx(0.5)


def test_executor_reports_chunk_target_position_limit_violation() -> None:
    spec = RobotSpec()
    controller = FakeController(spec)
    controller.q[0] = 2.88

    result = RecedingHorizonChunkExecutor(spec).execute(
        _chunk(spec, delta=0.01),
        controller,
    )

    assert result.success is False
    assert result.failure_stage == "chunk_safety"
    assert result.diagnostic is not None
    assert result.diagnostic["kind"] == "chunk_joint_position_contract"
    assert result.diagnostic["violation_indices"] == [[1, 0], [2, 0], [3, 0]]


def test_executor_holds_on_invalid_initial_state() -> None:
    spec = RobotSpec()
    controller = FakeController(spec)
    controller.q = controller.q.astype(np.float64)

    result = RecedingHorizonChunkExecutor(spec).execute(_chunk(spec), controller)

    assert result.success is False
    assert result.executed_steps == 0
    assert result.failure_stage == "initial_observation"
    assert controller.hold_calls == 1
