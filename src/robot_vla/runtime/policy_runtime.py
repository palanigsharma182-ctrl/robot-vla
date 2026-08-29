"""固定 Processor、Qwen、Flow 与 Action Adapter 的在线推理链路。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch

from robot_vla.adapters import ActionAdapter, ProprioNormalizer
from robot_vla.contracts import RobotSpec
from robot_vla.execution.rtc import RTCConfig, RTCTrace, build_rtc_trace
from robot_vla.model.policy import QwenVLAPolicy
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter


@dataclass(frozen=True)
class RuntimeConfig:
    num_flow_steps: int = 10
    use_bf16: bool = True
    sampling_seed: int = 42
    starting_sample_index: int = 0

    def __post_init__(self) -> None:
        if self.num_flow_steps <= 0:
            raise ValueError("num_flow_steps 必须为正整数")
        if not isinstance(self.use_bf16, bool):
            raise TypeError("use_bf16 必须为 bool")
        if self.sampling_seed < 0 or self.starting_sample_index < 0:
            raise ValueError("sampling_seed 和 starting_sample_index 不能为负数")


@dataclass(frozen=True)
class OnlineObservation:
    rgb_external: np.ndarray
    rgb_wrist: np.ndarray
    physical_proprio: np.ndarray
    instruction: str


@dataclass(frozen=True)
class SamplingTrace:
    seed: int
    sample_index: int


@dataclass(frozen=True)
class RuntimeActionChunk:
    normalized_action: np.ndarray
    physical_action: np.ndarray
    visual_tokens_per_image: tuple[int, int]
    context_length: int
    sampling: SamplingTrace
    rtc_trace: RTCTrace | None = None


def _move_model_inputs(value: Any, device: torch.device) -> Any:
    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: _move_model_inputs(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [_move_model_inputs(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(_move_model_inputs(item, device) for item in value)
    return value


class QwenVLARuntime:
    """每次 Replan 使用独立可记录 seed，并且只编码一次 Qwen Context。"""

    def __init__(
        self,
        policy: QwenVLAPolicy,
        processor_adapter: QwenVLAProcessorAdapter,
        proprio_normalizer: ProprioNormalizer,
        spec: RobotSpec,
        device: str | torch.device,
        config: RuntimeConfig | None = None,
    ) -> None:
        self.policy = policy
        self.processor_adapter = processor_adapter
        self.proprio_normalizer = proprio_normalizer
        self.spec = spec
        self.device = torch.device(device)
        self.config = config or RuntimeConfig()
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("首版 Runtime 只支持 cpu/cuda device")
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA Runtime，但当前 PyTorch 无可用 CUDA")
        if (
            self.config.use_bf16
            and self.device.type == "cuda"
            and not torch.cuda.is_bf16_supported()
        ):
            raise RuntimeError("当前 CUDA 设备不支持 Runtime 所需的 BF16")
        expert_config = self.policy.expert.config
        expected = (spec.proprio_dim, spec.action_horizon, spec.action_dim)
        actual = (
            expert_config.proprio_dim,
            expert_config.action_horizon,
            expert_config.action_dim,
        )
        if actual != expected:
            raise ValueError(f"Policy/RobotSpec 维度不兼容：期望 {expected}，实际 {actual}")
        self.action_adapter = ActionAdapter(spec)
        self.policy.to(self.device)
        self.policy.eval()
        self._sample_index = self.config.starting_sample_index
        self._last_sampling_trace: SamplingTrace | None = None

    @property
    def next_sample_index(self) -> int:
        return self._sample_index

    @property
    def last_sampling_trace(self) -> SamplingTrace | None:
        return self._last_sampling_trace

    def _next_sampling_trace(self) -> SamplingTrace:
        sample_index = self._sample_index
        self._sample_index += 1
        max_seed = 2**63 - 1
        seed = (self.config.sampling_seed + sample_index) % max_seed
        trace = SamplingTrace(seed=seed, sample_index=sample_index)
        self._last_sampling_trace = trace
        return trace

    @torch.no_grad()
    def infer_action_chunk(
        self,
        observation: OnlineObservation,
        *,
        rtc_previous_overlap: np.ndarray | None = None,
        rtc_config: RTCConfig | None = None,
    ) -> RuntimeActionChunk:
        sampling = self._next_sampling_trace()
        physical_proprio = np.asarray(observation.physical_proprio)
        if physical_proprio.shape != (self.spec.proprio_dim,):
            raise ValueError(
                f"physical_proprio 应为 [{self.spec.proprio_dim}]，"
                f"实际为 {physical_proprio.shape}"
            )
        if physical_proprio.dtype != np.float32 or not np.isfinite(physical_proprio).all():
            raise ValueError("physical_proprio 必须是有限 float32 向量")
        arm_q = physical_proprio[: self.spec.arm_dof]
        arm_dq = physical_proprio[self.spec.arm_dof : self.spec.arm_dof * 2]
        gripper = physical_proprio[-1]
        position_limits = np.asarray(self.spec.joint_position_limits_rad, dtype=np.float32)
        velocity_limits = np.asarray(self.spec.joint_velocity_limits_rad_s, dtype=np.float32)
        if np.any(arm_q < position_limits[:, 0] - 1e-5) or np.any(
            arm_q > position_limits[:, 1] + 1e-5
        ):
            raise ValueError("physical_proprio q 超出 Franka 关节位置限制")
        if np.any(np.abs(arm_dq) > velocity_limits + 1e-5) or not 0.0 <= gripper <= 1.0:
            raise ValueError("physical_proprio dq 或 gripper 超出 Franka 契约限制")
        normalized_proprio = self.proprio_normalizer.normalize(physical_proprio)
        processed = self.processor_adapter.encode(
            observation.rgb_external,
            observation.rgb_wrist,
            observation.instruction,
        )
        model_inputs = _move_model_inputs(processed.model_inputs, self.device)
        proprio_tensor = torch.from_numpy(normalized_proprio).unsqueeze(0).to(self.device)
        generator = torch.Generator(device=self.device)
        generator.manual_seed(sampling.seed)

        self.policy.eval()
        rtc_trace: RTCTrace | None = None
        if rtc_previous_overlap is not None and rtc_config is None:
            raise ValueError("提供 rtc_previous_overlap 时必须同时提供 rtc_config")
        if (
            rtc_config is not None
            and rtc_config.execution_horizon != self.spec.execute_steps
        ):
            raise ValueError("首版 RTC execution_horizon 必须等于 RobotSpec.execute_steps")
        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_bf16,
        ):
            if rtc_config is not None and rtc_previous_overlap is not None:
                previous = np.asarray(rtc_previous_overlap, dtype=np.float32)
                expected_overlap = self.spec.action_horizon - self.spec.execute_steps
                if previous.shape != (expected_overlap, self.spec.action_dim):
                    raise ValueError(
                        "rtc_previous_overlap 应为 "
                        f"[{expected_overlap},{self.spec.action_dim}]"
                    )
                target = np.zeros(
                    (self.spec.action_horizon, self.spec.action_dim),
                    dtype=np.float32,
                )
                target[:expected_overlap] = previous
                weights = rtc_config.slot_weights(self.spec.action_horizon)
                rtc_sample = self.policy.sample_actions_rtc(
                    model_inputs,
                    proprio_tensor,
                    torch.from_numpy(target).unsqueeze(0).to(self.device),
                    torch.from_numpy(weights).unsqueeze(0).to(self.device),
                    generator=generator,
                    num_steps=self.config.num_flow_steps,
                    max_guidance_weight=rtc_config.max_guidance_weight,
                )
                normalized_tensor = rtc_sample.guided_action
                raw_tensor = rtc_sample.raw_action
                guidance_coefficients = rtc_sample.guidance_coefficients
            else:
                normalized_tensor = self.policy.sample_actions(
                    model_inputs,
                    proprio_tensor,
                    generator=generator,
                    num_steps=self.config.num_flow_steps,
                )
                raw_tensor = normalized_tensor
                guidance_coefficients = ()
        normalized_action = normalized_tensor[0].float().cpu().numpy()
        raw_action = raw_tensor[0].float().cpu().numpy()
        expected_action = (self.spec.action_horizon, self.spec.action_dim)
        if normalized_action.shape != expected_action or not np.isfinite(normalized_action).all():
            raise RuntimeError("Policy 返回的 normalized Action Chunk 无效")
        if rtc_config is not None:
            rtc_trace = build_rtc_trace(
                rtc_config,
                action_horizon=self.spec.action_horizon,
                previous_overlap=rtc_previous_overlap,
                raw_action=raw_action,
                guided_action=normalized_action,
                denoising_guidance_coefficients=guidance_coefficients,
            )
        physical_action = self.action_adapter.denormalize(normalized_action)
        if len(processed.visual_tokens_per_image) != 1 or len(processed.context_lengths) != 1:
            raise RuntimeError("Processor 返回的在线 batch metadata 无效")
        return RuntimeActionChunk(
            normalized_action=normalized_action.copy(),
            physical_action=physical_action.copy(),
            visual_tokens_per_image=processed.visual_tokens_per_image[0],
            context_length=int(processed.context_lengths[0]),
            sampling=sampling,
            rtc_trace=rtc_trace,
        )
