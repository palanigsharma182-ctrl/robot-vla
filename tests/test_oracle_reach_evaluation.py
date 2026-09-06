"""验证历史 Oracle 入口与当前控制器的接线，不启动仿真。"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from robot_vla.contracts import RobotSpec
from robot_vla.adapters import FrankaObservationAdapter
from robot_vla.diagnostics import oracle_reach_evaluation as diagnostic
from robot_vla.evaluation import maniskill
from robot_vla.execution.chunk_executor import (
    ChunkExecutionResult, FrankaControlState, RecedingHorizonChunkExecutor,
)
from robot_vla.runtime import OnlineObservation, QwenVLAReplanLoop
from robot_vla.runtime.policy_runtime import RuntimeActionChunk, SamplingTrace
from robot_vla.runtime.control_loop import ReplanResult
from robot_vla.tasks.pick_place import PickPlaceState, PickPlaceTaskTracker


def test_oracle_episode_initializes_real_tracking_controller(monkeypatch):
    spec = RobotSpec()
    state = PickPlaceState(
        tcp_position=(0.0, 0.0, 0.2), object_position=(0.0, 0.0, 0.02),
        goal_position=(0.2, 0.2, 0.02), object_linear_velocity=(0.0, 0.0, 0.0),
        object_angular_velocity=(0.0, 0.0, 0.0), support_center_z_m=0.02,
        is_grasped=False,
    )
    tracker = PickPlaceTaskTracker()
    preparation = SimpleNamespace(
        observation={}, tracker=tracker, progress=tracker.update(state), preparation_steps=0,
    )
    # 仅替换仿真状态读取，保留真实控制器初始化与 Oracle 评估循环。
    monkeypatch.setattr(maniskill, "_read_predicate_state", lambda _: state)
    online = OnlineObservation(
        np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4, 3), dtype=np.uint8),
        np.zeros(spec.proprio_dim, dtype=np.float32), "synthetic reach",
    )
    adapters = []

    def read_observation(observation, env, adapter, instruction):
        adapters.append(adapter)
        return online

    def stop_at_inference(loop, observation, controller):
        assert observation is online
        assert controller.observation_adapter is adapters[0]
        assert controller.observation_v2_history is None
        return ReplanResult(
            action_chunk=None, sampling=None,
            execution=ChunkExecutionResult(
                success=False, executed_steps=0, failure_stage="inference", error="test stop",
            ),
        )

    monkeypatch.setattr(diagnostic, "_read_online_observation", read_observation)
    monkeypatch.setattr(diagnostic.QwenVLAReplanLoop, "replan_and_execute", stop_at_inference)
    env = SimpleNamespace(unwrapped=None, _elapsed_steps=7)
    result = diagnostic.run_reach_diagnostic_episode(
        env, object(), spec, seed=0, instruction=online.instruction,
        sampling_seed_base=0, preparation=preparation, max_policy_steps=100,
    )
    assert env._elapsed_steps == 0
    assert result.episode.error == "test stop"
    assert result.episode.policy_environment_steps == 0
    assert result.distance_trace_m == pytest.approx((0.18,))
    assert len(adapters) == 1


@pytest.mark.parametrize("limit", [1, 3, 100])
def test_distance_controller_stops_inside_action_prefix(monkeypatch, limit):
    spec = RobotSpec()
    state = PickPlaceState(
        tcp_position=(0.0, 0.0, 0.2), object_position=(0.0, 0.0, 0.02),
        goal_position=(0.2, 0.2, 0.02), object_linear_velocity=(0.0, 0.0, 0.0),
        object_angular_velocity=(0.0, 0.0, 0.0), support_center_z_m=0.02,
        is_grasped=False,
    )
    monkeypatch.setattr(maniskill, "_read_predicate_state", lambda _: state)
    actual_steps = []
    q = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)

    def step(action):
        actual_steps.append(action.copy())
        q[:] += action[:7] * spec.maniskill_arm_delta_range_rad
        return {}, 0.0, False, False, {"success": False}

    env = SimpleNamespace(
        unwrapped=None, step=step,
        action_space=SimpleNamespace(low=np.full(8, -1.0), high=np.ones(8)),
    )
    tracker = PickPlaceTaskTracker()
    controller = diagnostic._DistanceTraceController(
        env, spec, {}, tracker, tracker.update(state), FrankaObservationAdapter(spec),
        max_policy_steps=limit,
    )
    remaining = min(limit, 3)
    for _ in range(limit - remaining):
        controller.send_action(np.zeros(8, dtype=np.float32))
    monkeypatch.setattr(controller, "read_state", lambda: FrankaControlState(q.copy(), 0.5))
    physical = np.zeros((16, 8), dtype=np.float32)
    physical[:, 0] = 0.001
    physical[:, -1] = 0.5
    chunk = RuntimeActionChunk(
        normalized_action=physical.copy(), physical_action=physical,
        visual_tokens_per_image=(0, 0), context_length=1, sampling=SamplingTrace(0, 0),
    )
    runtime = SimpleNamespace(infer_action_chunk=lambda _: chunk)
    executor = RecedingHorizonChunkExecutor(spec)
    loop = QwenVLAReplanLoop(runtime, executor, temporal_ensemble_enabled=False)
    loop.control_step = controller.environment_steps
    result = loop.replan_and_execute(object(), controller)
    assert result.execution.success and result.execution.interrupted
    assert result.execution.executed_steps == remaining
    assert loop.control_step == controller.environment_steps == limit
    assert result.execution.correction_saturation_steps == 0
    assert not result.execution.replan_required
    np.testing.assert_allclose(executor.previous_command_q, q, rtol=0, atol=1e-7)
    for _ in range(4):
        controller.send_action(np.zeros(8, dtype=np.float32))
    assert controller.done
    assert controller.environment_steps == len(actual_steps) == limit
    assert len(controller.distance_trace_m) == limit + 1
