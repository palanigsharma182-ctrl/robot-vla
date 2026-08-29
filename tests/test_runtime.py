from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from robot_vla.adapters import ProprioNormalizer, ProprioStats
from robot_vla.contracts import RobotSpec
from robot_vla.execution.chunk_executor import (
    ChunkExecutionResult,
    FrankaControlState,
    RecedingHorizonChunkExecutor,
)
from robot_vla.execution.rtc import RTCConfig
from robot_vla.model.expert import ExpertConfig, StandaloneActionExpert
from robot_vla.model.policy import QwenVLAPolicy
from robot_vla.model.qwen_context import FrozenQwenContextEncoder
from robot_vla.runtime.control_loop import QwenVLAReplanLoop
from robot_vla.runtime.policy_runtime import (
    OnlineObservation,
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
