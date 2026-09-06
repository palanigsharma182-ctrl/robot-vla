"""用合成观测验证真实评估入口，不启动仿真或加载模型。"""

from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from robot_vla.contracts import PICK_AND_PLACE_SKILLS, RobotSpec
from robot_vla.evaluation import maniskill
from robot_vla.evaluation.rollout import RolloutEpisodeSpec
from robot_vla.execution.chunk_executor import ChunkExecutionResult, FrankaControlState
from robot_vla.observation import OBSERVATION_MODALITIES, ObservationV2Frame, ObservationV2History
from robot_vla.runtime import OnlineObservation, QwenVLAObservationV2Runtime
from robot_vla.runtime.control_loop import ReplanResult
from robot_vla.tasks.pick_place import PickPlaceState


@pytest.mark.parametrize("history_frames", [0, 1, 4], ids=["v1", "v2-padded", "v2-full"])
def test_episode_reports_current_velocity_without_flattening_history(monkeypatch, history_frames):
    spec = RobotSpec()
    q = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)
    proprio = np.concatenate((q, np.asarray([-0.2] * 7 + [0.5], dtype=np.float32)))
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    history = ObservationV2History(spec) if history_frames else None
    for step in range(history_frames):
        frame_proprio = proprio.copy()
        # 旧帧速度更大，用来识别错误的跨时间取最大值。
        if step < history_frames - 1:
            frame_proprio[7:14] = 0.8
        history.append(ObservationV2Frame(
            rgb_external=rgb.copy(), rgb_wrist=rgb.copy(),
            physical_proprio=frame_proprio,
            base_from_tcp=np.eye(4, dtype=np.float32),
            base_from_wrist_camera=np.eye(4, dtype=np.float32),
            finger_force_n=np.zeros(2, dtype=np.float32),
            timestamp_s=step / spec.control_hz,
            modality_timestamp_s=np.full(len(OBSERVATION_MODALITIES), step / spec.control_hz),
            modality_valid=np.ones(len(OBSERVATION_MODALITIES), dtype=np.bool_),
        ))
    online_v1 = OnlineObservation(rgb, rgb, proprio, "synthetic interface check")
    predicate = PickPlaceState(
        tcp_position=(0.0, 0.0, 0.4), object_position=(0.0, 0.0, 0.02),
        goal_position=(0.2, 0.2, 0.02), object_linear_velocity=(0.0, 0.0, 0.0),
        object_angular_velocity=(0.0, 0.0, 0.0), support_center_z_m=0.02, is_grasped=False,
    )
    monkeypatch.setattr(maniskill, "_read_predicate_state", lambda _: predicate)
    monkeypatch.setattr(maniskill, "_read_online_observation", lambda *args: online_v1)

    def make_controller(env, spec, observation, tracker, progress, adapter, **kwargs):
        assert kwargs["observation_v2_enabled"] == bool(history_frames)
        return SimpleNamespace(
            done=False, observation=observation, observation_v2_history=history,
            read_state=lambda: FrankaControlState(q.copy(), 0.5), progress=progress,
            last_tcp_linear_speed_m_s=0.0, environment_success=False,
            environment_steps=0, terminated=False, truncated=False,
            skill_completion_environment_steps=[None] * len(PICK_AND_PLACE_SKILLS),
        )

    received = []

    def stop_at_inference(loop, observation, controller):
        received.append(observation)
        # 在真实评估循环中生成一条诊断 trace，动作执行不属于本测试。
        return ReplanResult(
            action_chunk=None, sampling=None,
            execution=ChunkExecutionResult(
                success=False, executed_steps=0, failure_stage="inference", error="test stop",
            ),
        )

    monkeypatch.setattr(maniskill, "_TrackingManiSkillController", make_controller)
    monkeypatch.setattr(maniskill.QwenVLAReplanLoop, "replan_and_execute", stop_at_inference)
    runtime = object.__new__(QwenVLAObservationV2Runtime) if history_frames else object()
    env = SimpleNamespace(reset=lambda **kwargs: ({}, {}), unwrapped=None, _max_episode_steps=1)
    result = maniskill.run_maniskill_episode(
        env, runtime, spec, RolloutEpisodeSpec("unseen", 0, online_v1.instruction),
        sampling_seed_base=0,
    )

    assert result.error == "test stop"
    assert len(result.replan_traces) == len(received) == 1
    assert result.replan_traces[0]["joint_velocity_abs_max_rad_s"] == pytest.approx(0.2)
    delivered = received[0]
    if history_frames:
        assert delivered.physical_proprio.shape == (4, 15)
        np.testing.assert_array_equal(delivered.history_valid, np.arange(4) >= 4 - history_frames)
        np.testing.assert_array_equal(delivered.physical_proprio[-1], proprio)
        if history_frames == 4:
            np.testing.assert_array_equal(delivered.physical_proprio[:-1, 7:14], 0.8)
    else:
        assert delivered is online_v1
