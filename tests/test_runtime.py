from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from robot_vla.adapters import (
    FingerForceNormalizer,
    FingerForceStats,
    ProprioNormalizer,
    ProprioStats,
)
from robot_vla.contracts import RobotSpec
from robot_vla.execution.chunk_executor import (
    ChunkExecutionResult,
    FrankaControlState,
    RecedingHorizonChunkExecutor,
)
from robot_vla.execution.rtc import RTCConfig
from robot_vla.model.expert import (
    ExpertConfig,
    StandaloneActionExpert,
    TemporalExpertConfig,
    TemporalStandaloneActionExpert,
)
from robot_vla.model.policy import QwenVLAObservationV2Policy, QwenVLAPolicy
from robot_vla.model.qwen_context import FrozenQwenContextEncoder, QwenVLAAdapter
from robot_vla.model.qwen_processor import VLA_CONTEXT_VALID_MASK, VLA_IMAGE_TIME_INDICES
from robot_vla.observation import (
    OBSERVATION_MODALITIES,
    ObservationV2Frame,
    ObservationV2History,
)
from robot_vla.runtime.control_loop import QwenVLAReplanLoop
from robot_vla.runtime.policy_runtime import (
    OnlineObservation,
    QwenVLAObservationV2Runtime,
    QwenVLARuntime,
    RuntimeActionChunk,
    RuntimeConfig,
    SamplingTrace,
)


class FakeBaseModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.calls = 0

    def forward(self, input_ids, **_kwargs):
        self.calls += 1
        tokens = input_ids.float().unsqueeze(-1).expand(-1, -1, 2048) * self.scale
        return SimpleNamespace(hidden_states=(tokens,))


class FakeQwen(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(text_config=SimpleNamespace(hidden_size=2048))
        self.model = FakeBaseModel()
        self.lm_head = nn.Linear(2048, 1, bias=False)


class FakeProcessorAdapter:
    def encode(self, rgb_external, rgb_wrist, instruction):
        assert rgb_external.shape == (8, 10, 3)
        assert rgb_wrist.shape == (6, 6, 3)
        assert instruction == "pick the cube"
        return SimpleNamespace(
            model_inputs={
                "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.bool),
                "mm_token_type_ids": torch.zeros(1, 4, dtype=torch.long),
                "pixel_values": torch.zeros(2, 1536),
                "image_grid_thw": torch.ones(2, 3, dtype=torch.long),
            },
            visual_tokens_per_image=((4, 3),),
            context_lengths=(4,),
        )


class FakeHistoryProcessorAdapter:
    def __init__(self) -> None:
        self.calls = 0
        self.last_history_valid = None

    def encode_history(
        self,
        rgb_external_history,
        rgb_wrist_history,
        history_valid,
        instruction,
    ):
        self.calls += 1
        assert rgb_external_history.shape == (4, 8, 10, 3)
        assert rgb_wrist_history.shape == (4, 6, 6, 3)
        assert instruction == "pick the cube"
        self.last_history_valid = np.asarray(history_valid).copy()
        return SimpleNamespace(
            model_inputs={
                "input_ids": torch.tensor([[1, 2, 3, 4]], dtype=torch.long),
                "attention_mask": torch.ones(1, 4, dtype=torch.bool),
                "mm_token_type_ids": torch.zeros(1, 4, dtype=torch.long),
                "pixel_values": torch.zeros(8, 1536),
                "image_grid_thw": torch.ones(8, 3, dtype=torch.long),
                VLA_IMAGE_TIME_INDICES: torch.tensor(
                    [[0, 1, 2, 3]],
                    dtype=torch.long,
                ),
                VLA_CONTEXT_VALID_MASK: torch.ones(1, 4, dtype=torch.bool),
            },
            visual_tokens_per_image=((1,) * 8,),
            context_lengths=(4,),
        )


class HoldOnlyController:
    def __init__(self) -> None:
        self.hold_calls = 0

    def read_state(self) -> FrankaControlState:
        raise AssertionError("推理失败后不能开始执行旧 Chunk")

    def send_action(self, _controller_action) -> None:
        raise AssertionError("推理失败后不能发送旧 Action")

    def hold_current(self) -> None:
        self.hold_calls += 1


class StableController:
    def __init__(self) -> None:
        self.q = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)

    def read_state(self) -> FrankaControlState:
        return FrankaControlState(self.q.copy(), 0.5)

    def send_action(self, _controller_action) -> None:
        pass

    def hold_current(self) -> None:
        pass


class RecordingRTCRuntime:
    def __init__(self, spec: RobotSpec) -> None:
        self.spec = spec
        self.calls = []
        self._last_sampling_trace = None

    @property
    def last_sampling_trace(self):
        return self._last_sampling_trace

    def infer_action_chunk(self, _observation, **kwargs):
        self.calls.append(kwargs)
        index = len(self.calls) - 1
        trace = SamplingTrace(seed=100 + index, sample_index=index)
        self._last_sampling_trace = trace
        normalized = np.full(
            (self.spec.action_horizon, self.spec.action_dim),
            0.1 * (index + 1),
            dtype=np.float32,
        )
        return RuntimeActionChunk(
            normalized_action=normalized,
            physical_action=np.zeros_like(normalized),
            visual_tokens_per_image=(1, 1),
            context_length=2,
            sampling=trace,
        )


def _policy() -> tuple[QwenVLAPolicy, FakeQwen]:
    qwen = FakeQwen()
    expert = StandaloneActionExpert(
        ExpertConfig(
            hidden_size=32,
            state_hidden_size=16,
            num_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
    )
    return QwenVLAPolicy(FrozenQwenContextEncoder(qwen), expert), qwen


class RecordingObservationV2Policy(QwenVLAObservationV2Policy):
    def __init__(self, qwen: FakeQwen) -> None:
        config = TemporalExpertConfig(
            hidden_size=32,
            state_hidden_size=16,
            num_layers=2,
            intermediate_size=64,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=8,
        )
        super().__init__(
            FrozenQwenContextEncoder(qwen),
            TemporalStandaloneActionExpert(config),
            QwenVLAAdapter(history_length=4),
        )
        self.last_state_history = None
        self.last_state_history_mask = None
        self.last_controller_state = None

    def _record_temporal_inputs(
        self,
        state_history,
        state_history_mask,
        controller_state,
    ) -> None:
        self.last_state_history = state_history.detach().cpu().clone()
        self.last_state_history_mask = state_history_mask.detach().cpu().clone()
        self.last_controller_state = controller_state.detach().cpu().clone()

    def sample_actions(
        self,
        model_inputs,
        state_history,
        *,
        state_history_mask,
        controller_state,
        **kwargs,
    ):
        self._record_temporal_inputs(
            state_history,
            state_history_mask,
            controller_state,
        )
        return super().sample_actions(
            model_inputs,
            state_history,
            state_history_mask=state_history_mask,
            controller_state=controller_state,
            **kwargs,
        )

    def sample_actions_rtc(
        self,
        model_inputs,
        state_history,
        previous_action_target,
        slot_weights,
        *,
        state_history_mask,
        controller_state,
        **kwargs,
    ):
        self._record_temporal_inputs(
            state_history,
            state_history_mask,
            controller_state,
        )
        return super().sample_actions_rtc(
            model_inputs,
            state_history,
            previous_action_target,
            slot_weights,
            state_history_mask=state_history_mask,
            controller_state=controller_state,
            **kwargs,
        )


def _normalizer(spec: RobotSpec) -> ProprioNormalizer:
    return ProprioNormalizer(
        ProprioStats(
            mean=(0.0,) * spec.proprio_dim,
            std=(1.0,) * spec.proprio_dim,
            count=100,
            embodiment=spec.embodiment,
        ),
        spec,
    )


def _force_normalizer(spec: RobotSpec) -> FingerForceNormalizer:
    return FingerForceNormalizer(
        FingerForceStats(
            scale_log1p_p95=(1.0, 2.0),
            count=100,
            positive_count=(50, 40),
            embodiment=spec.embodiment,
        ),
        spec,
    )


def _base_from_pose(x: float, y: float, z: float) -> np.ndarray:
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = (x, y, z)
    return pose


def _observation_v2(spec: RobotSpec):
    history = ObservationV2History(spec)
    base_q = np.asarray(
        (0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0),
        dtype=np.float32,
    )
    for step in range(4):
        proprio = np.zeros(spec.proprio_dim, dtype=np.float32)
        proprio[: spec.arm_dof] = base_q + np.float32(step * 0.001)
        proprio[-1] = 0.5
        timestamp = step / spec.control_hz
        history.append(
            ObservationV2Frame(
                rgb_external=np.full((8, 10, 3), step, dtype=np.uint8),
                rgb_wrist=np.full((6, 6, 3), step + 10, dtype=np.uint8),
                physical_proprio=proprio,
                base_from_tcp=_base_from_pose(0.4 + step * 0.001, 0.0, 0.3),
                base_from_wrist_camera=_base_from_pose(0.2, 0.0, 0.5),
                finger_force_n=np.asarray((step + 1.0, step + 2.0), dtype=np.float32),
                timestamp_s=timestamp,
                modality_timestamp_s=np.full(
                    len(OBSERVATION_MODALITIES),
                    timestamp,
                    dtype=np.float64,
                ),
                modality_valid=np.ones(
                    len(OBSERVATION_MODALITIES),
                    dtype=np.bool_,
                ),
            )
        )
    previous_command = base_q + np.float32(0.005)
    previous_action = np.asarray((0.001,) * spec.arm_dof + (0.5,), dtype=np.float32)
    return history.snapshot(
        "pick the cube",
        previous_command_q=previous_command,
        previous_action=previous_action,
    )


def _runtime_v2(seed: int = 123):
    spec = RobotSpec()
    torch.manual_seed(77)
    qwen = FakeQwen()
    policy = RecordingObservationV2Policy(qwen)
    processor = FakeHistoryProcessorAdapter()
    force_normalizer = _force_normalizer(spec)
    runtime = QwenVLAObservationV2Runtime(
        policy,
        processor,
        _normalizer(spec),
        force_normalizer,
        spec,
        "cpu",
        RuntimeConfig(num_flow_steps=2, use_bf16=False, sampling_seed=seed),
    )
    return runtime, policy, qwen, processor, force_normalizer


def _observation(spec: RobotSpec) -> OnlineObservation:
    proprio = np.zeros(spec.proprio_dim, dtype=np.float32)
    proprio[:7] = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)
    proprio[-1] = 0.5
    return OnlineObservation(
        rgb_external=np.zeros((8, 10, 3), dtype=np.uint8),
        rgb_wrist=np.zeros((6, 6, 3), dtype=np.uint8),
        physical_proprio=proprio,
        instruction="pick the cube",
    )


def _runtime(seed: int = 123) -> tuple[QwenVLARuntime, FakeQwen]:
    spec = RobotSpec()
    torch.manual_seed(77)
    policy, qwen = _policy()
    runtime = QwenVLARuntime(
        policy,
        FakeProcessorAdapter(),
        _normalizer(spec),
        spec,
        "cpu",
        RuntimeConfig(num_flow_steps=2, use_bf16=False, sampling_seed=seed),
    )
    return runtime, qwen


def test_runtime_runs_qwen_once_and_returns_reproducible_physical_chunk() -> None:
    spec = RobotSpec()
    first_runtime, first_qwen = _runtime()
    second_runtime, _second_qwen = _runtime()

    first = first_runtime.infer_action_chunk(_observation(spec))
    reproduced = second_runtime.infer_action_chunk(_observation(spec))
    next_chunk = first_runtime.infer_action_chunk(_observation(spec))

    assert first_qwen.model.calls == 2
    assert first.normalized_action.shape == (16, 8)
    assert first.physical_action.shape == (16, 8)
    assert first.sampling.seed == 123 and first.sampling.sample_index == 0
    assert next_chunk.sampling.seed == 124 and next_chunk.sampling.sample_index == 1
    np.testing.assert_allclose(first.normalized_action, reproduced.normalized_action)
    assert not np.allclose(first.normalized_action, next_chunk.normalized_action)
    assert np.max(np.abs(first.physical_action[:, :7])) <= 0.05 + 1e-6
    assert np.all((0.0 <= first.physical_action[:, -1]) & (first.physical_action[:, -1] <= 1.0))
    assert first.visual_tokens_per_image == (4, 3)
    assert first.context_length == 4
    assert first_qwen.model.scale.grad is None
    assert first_qwen.training is False


def test_first_rtc_replan_is_identical_to_plain_flow_with_the_same_seed() -> None:
    spec = RobotSpec()
    rtc_runtime, _rtc_qwen = _runtime(seed=321)
    plain_runtime, _plain_qwen = _runtime(seed=321)

    rtc_chunk = rtc_runtime.infer_action_chunk(
        _observation(spec),
        rtc_config=RTCConfig(),
    )
    plain_chunk = plain_runtime.infer_action_chunk(_observation(spec))

    np.testing.assert_allclose(rtc_chunk.normalized_action, plain_chunk.normalized_action)
    assert rtc_chunk.rtc_trace is not None
    assert rtc_chunk.rtc_trace.previous_chunk_available is False
    assert rtc_chunk.rtc_trace.denoising_guidance_coefficients == ()


def test_runtime_rtc_branch_runs_vjp_once_per_flow_and_returns_finite_trace() -> None:
    spec = RobotSpec()
    runtime, qwen = _runtime(seed=654)
    previous = np.full((12, 8), 0.25, dtype=np.float32)

    chunk = runtime.infer_action_chunk(
        _observation(spec),
        rtc_previous_overlap=previous,
        rtc_config=RTCConfig(),
    )

    assert qwen.model.calls == 1
    assert chunk.normalized_action.shape == (16, 8)
    assert np.isfinite(chunk.normalized_action).all()
    assert chunk.rtc_trace is not None
    assert chunk.rtc_trace.previous_chunk_available is True
    assert chunk.rtc_trace.overlap_length == 12
    assert len(chunk.rtc_trace.denoising_guidance_coefficients) == 2
    assert chunk.rtc_trace.raw_mean_abs_disagreement is not None
    assert all(parameter.grad is None for parameter in runtime.policy.parameters())


@pytest.mark.skipif(
    not torch.cuda.is_available() or not torch.cuda.is_bf16_supported(),
    reason="需要支持 BF16 的 CUDA GPU",
)
def test_runtime_rtc_vjp_supports_cuda_bf16_autocast() -> None:
    spec = RobotSpec()
    torch.manual_seed(77)
    policy, qwen = _policy()
    runtime = QwenVLARuntime(
        policy,
        FakeProcessorAdapter(),
        _normalizer(spec),
        spec,
        "cuda",
        RuntimeConfig(num_flow_steps=2, use_bf16=True, sampling_seed=987),
    )

    chunk = runtime.infer_action_chunk(
        _observation(spec),
        rtc_previous_overlap=np.full((12, 8), -0.25, dtype=np.float32),
        rtc_config=RTCConfig(),
    )

    assert qwen.model.calls == 1
    assert np.isfinite(chunk.normalized_action).all()
    assert chunk.rtc_trace is not None
    assert len(chunk.rtc_trace.denoising_guidance_coefficients) == 2
    assert all(parameter.grad is None for parameter in runtime.policy.parameters())


def test_replan_loop_holds_and_records_seed_when_online_observation_is_invalid() -> None:
    spec = RobotSpec()
    runtime, _qwen = _runtime(seed=500)
    invalid = _observation(spec)
    invalid.physical_proprio[0] = np.nan
    controller = HoldOnlyController()
    loop = QwenVLAReplanLoop(runtime, RecedingHorizonChunkExecutor(spec))

    result = loop.replan_and_execute(invalid, controller)

    assert result.action_chunk is None
    assert result.execution.success is True
    assert result.execution.failure_stage == "inference"
    assert result.execution.replan_required is True
    assert result.execution.anomaly_kind == "inference"
    assert result.execution.hold_succeeded is True
    assert result.sampling is not None
    assert result.sampling.seed == 500
    assert controller.hold_calls == 1


def test_replan_loop_can_disable_anomaly_replanning() -> None:
    spec = RobotSpec()
    runtime, _qwen = _runtime(seed=500)
    invalid = _observation(spec)
    invalid.physical_proprio[0] = np.nan
    controller = HoldOnlyController()
    loop = QwenVLAReplanLoop(
        runtime,
        RecedingHorizonChunkExecutor(spec),
        max_anomaly_replans=0,
    )

    result = loop.replan_and_execute(invalid, controller)

    assert result.execution.success is False
    assert result.execution.failure_stage == "inference"
    assert result.execution.replan_required is False
    assert result.anomaly_replan_count == 0


def test_disabled_anomaly_replanning_turns_tracking_replan_into_failure() -> None:
    spec = RobotSpec()
    runtime, _qwen = _runtime(seed=500)
    loop = QwenVLAReplanLoop(
        runtime,
        RecedingHorizonChunkExecutor(spec),
        max_anomaly_replans=0,
    )
    execution = ChunkExecutionResult(
        success=True,
        executed_steps=2,
        replan_required=True,
        anomaly_kind="tracking_correction_saturation",
    )

    handled = loop._handle_anomaly(execution)

    assert handled.success is False
    assert handled.failure_stage == "replan_anomaly_exhausted"
    assert handled.replan_required is False
    assert handled.anomaly_kind == "tracking_correction_saturation"


def test_rtc_replan_aligns_previous_guided_chunk_and_reset_clears_history() -> None:
    spec = RobotSpec()
    runtime = RecordingRTCRuntime(spec)
    loop = QwenVLAReplanLoop(
        runtime,
        RecedingHorizonChunkExecutor(spec),
        inference_strategy="rtc",
    )
    controller = StableController()

    loop.replan_and_execute(_observation(spec), controller)
    loop.replan_and_execute(_observation(spec), controller)

    assert runtime.calls[0]["rtc_previous_overlap"] is None
    previous = runtime.calls[1]["rtc_previous_overlap"]
    assert previous.shape == (12, 8)
    np.testing.assert_allclose(previous, 0.1)

    loop.reset()
    loop.replan_and_execute(_observation(spec), controller)
    assert runtime.calls[2]["rtc_previous_overlap"] is None


def test_rtc_anomaly_clears_previous_reference() -> None:
    spec = RobotSpec()
    runtime = RecordingRTCRuntime(spec)
    loop = QwenVLAReplanLoop(
        runtime,
        RecedingHorizonChunkExecutor(spec),
        inference_strategy="rtc",
    )
    controller = StableController()
    loop.replan_and_execute(_observation(spec), controller)

    loop._handle_anomaly(
        ChunkExecutionResult(
            success=True,
            executed_steps=1,
            replan_required=True,
            anomaly_kind="tracking_correction_saturation",
        )
    )
    loop.replan_and_execute(_observation(spec), controller)

    assert runtime.calls[1]["rtc_previous_overlap"] is None


def test_observation_v2_runtime_uses_shared_force_transform_and_runs_qwen_once() -> None:
    spec = RobotSpec()
    runtime, policy, qwen, processor, force_normalizer = _runtime_v2()
    observation = _observation_v2(spec)

    chunk = runtime.infer_action_chunk(observation)

    assert qwen.model.calls == 1
    assert processor.calls == 1
    np.testing.assert_array_equal(processor.last_history_valid, np.ones(4, dtype=np.bool_))
    assert chunk.normalized_action.shape == (spec.action_horizon, spec.action_dim)
    assert policy.last_state_history is not None
    captured = policy.last_state_history[0].numpy()
    normalized_proprio = runtime.proprio_normalizer.normalize(
        observation.physical_proprio
    )
    normalized_force = force_normalizer.normalize(observation.finger_force_n)
    expected = observation.frame_state(normalized_proprio, normalized_force)
    np.testing.assert_allclose(captured, expected, atol=1e-6)


def test_observation_v2_runtime_rtc_runs_qwen_once() -> None:
    spec = RobotSpec()
    runtime, _policy, qwen, _processor, _force_normalizer = _runtime_v2(seed=321)
    overlap = np.zeros(
        (spec.action_horizon - spec.execute_steps, spec.action_dim),
        dtype=np.float32,
    )

    chunk = runtime.infer_action_chunk(
        _observation_v2(spec),
        rtc_previous_overlap=overlap,
        rtc_config=RTCConfig(),
    )

    assert qwen.model.calls == 1
    assert chunk.rtc_trace is not None
    assert chunk.rtc_trace.previous_chunk_available is True


def test_observation_v2_runtime_rejects_incomplete_current_state_before_qwen() -> None:
    spec = RobotSpec()
    runtime, _policy, qwen, processor, _force_normalizer = _runtime_v2()
    observation = _observation_v2(spec)
    observation.modality_valid[-1, 5] = False
    observation.finger_force_n[-1] = 0.0

    with pytest.raises(ValueError, match="当前控制步必须六模态完整有效"):
        runtime.infer_action_chunk(observation)

    assert qwen.model.calls == 0
    assert processor.calls == 0
