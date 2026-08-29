"""Frozen Qwen Context、Adapter、Action Expert 与 Rectified Flow 的策略集成。"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn

from robot_vla.model.expert import StandaloneActionExpert
from robot_vla.model.qwen_context import (
    FrozenQwenContextEncoder,
    QwenContext,
    QwenVLAAdapter,
)
from robot_vla.training.flow_matching import (
    FlowTrainingTarget,
    RTCFlowIntegrationOutput,
    build_critical_event_mask,
    euler_integrate_actions,
    euler_integrate_actions_with_rtc,
    masked_flow_mse,
    sample_flow_training_target,
)


@dataclass(frozen=True)
class FlowLossOutput:
    loss: torch.Tensor
    base_loss: torch.Tensor
    event_loss: torch.Tensor
    critical_mask: torch.Tensor
    prediction: torch.Tensor
    target: FlowTrainingTarget


@dataclass(frozen=True)
class RTCSamplingOutput:
    guided_action: torch.Tensor
    raw_action: torch.Tensor
    guidance_coefficients: tuple[float, ...]


class QwenVLAPolicy(nn.Module):
    """每个 Action Chunk 只编码一次 Qwen Context 的 qwen-vla-v0.1 策略。"""

    def __init__(
        self,
        context_encoder: FrozenQwenContextEncoder,
        expert: StandaloneActionExpert,
        adapter: QwenVLAAdapter | None = None,
    ) -> None:
        super().__init__()
        self.context_encoder = context_encoder
        self.adapter = adapter or QwenVLAAdapter()
        self.expert = expert
        if self.expert.config.context_dim != self.adapter.output_dim:
            raise ValueError(
                f"Expert context_dim 应为 {self.adapter.output_dim}，"
                f"实际为 {self.expert.config.context_dim}"
            )

    def train(self, mode: bool = True) -> QwenVLAPolicy:
        super().train(mode)
        self.context_encoder.train(False)
        return self

    def encode_context(self, model_inputs: dict[str, Any]) -> QwenContext:
        return self.adapter(self.context_encoder(model_inputs))

    def predict_velocity(
        self,
        context: QwenContext,
        normalized_proprio: torch.Tensor,
        noisy_action: torch.Tensor,
        flow_time: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.expert(
            context,
            normalized_proprio,
            noisy_action,
            flow_time,
            action_mask,
        )

    def flow_matching_loss(
        self,
        model_inputs: dict[str, Any],
        normalized_proprio: torch.Tensor,
        normalized_action: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        event_mask: torch.Tensor | None = None,
        event_loss_weight: float = 0.0,
        executed_action_steps: int = 4,
        generator: torch.Generator | None = None,
    ) -> FlowLossOutput:
        if not math.isfinite(event_loss_weight) or event_loss_weight < 0:
            raise ValueError("event_loss_weight 必须是有限非负数")
        context = self.encode_context(model_inputs)
        target = sample_flow_training_target(
            normalized_action,
            action_mask,
            generator=generator,
        )
        prediction = self.predict_velocity(
            context,
            normalized_proprio,
            target.noisy_action,
            target.flow_time,
            action_mask,
        )
        base_loss = masked_flow_mse(prediction, target.target_velocity, action_mask)
        if event_mask is None:
            event_mask = torch.zeros_like(action_mask)
        critical_mask = build_critical_event_mask(
            event_mask,
            action_mask,
            executed_action_steps,
        )
        event_loss = masked_flow_mse(
            prediction,
            target.target_velocity,
            critical_mask,
            allow_empty=True,
        )
        loss = base_loss + float(event_loss_weight) * event_loss
        return FlowLossOutput(
            loss=loss,
            base_loss=base_loss,
            event_loss=event_loss,
            critical_mask=critical_mask,
            prediction=prediction,
            target=target,
        )

    def _prepare_action_sampling(
        self,
        model_inputs: dict[str, Any],
        normalized_proprio: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    ]:
        context = self.encode_context(model_inputs)
        batch_size = context.tokens.shape[0]
        if normalized_proprio.shape != (batch_size, self.expert.config.proprio_dim):
            raise ValueError(
                "normalized_proprio shape 应为 "
                f"[{batch_size},{self.expert.config.proprio_dim}]"
            )
        if action_mask is None:
            action_mask = torch.ones(
                batch_size,
                self.expert.config.action_horizon,
                dtype=torch.bool,
                device=normalized_proprio.device,
            )
        expected_mask = (batch_size, self.expert.config.action_horizon)
        if action_mask.shape != expected_mask or action_mask.dtype != torch.bool:
            raise ValueError(f"action_mask 应为 {expected_mask} bool Tensor")
        initial_noise = torch.randn(
            batch_size,
            self.expert.config.action_horizon,
            self.expert.config.action_dim,
            dtype=torch.float32,
            device=normalized_proprio.device,
            generator=generator,
        )
        context_kv = self.expert.prepare_context_kv(context)

        def velocity_fn(state: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
            return self.expert(
                context,
                normalized_proprio,
                state,
                flow_time,
                action_mask,
                context_kv=context_kv,
            )

        return initial_noise, action_mask, velocity_fn

    @torch.no_grad()
    def sample_actions(
        self,
        model_inputs: dict[str, Any],
        normalized_proprio: torch.Tensor,
        *,
        action_mask: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        num_steps: int = 10,
    ) -> torch.Tensor:
        initial_noise, resolved_mask, velocity_fn = self._prepare_action_sampling(
            model_inputs,
            normalized_proprio,
            action_mask=action_mask,
            generator=generator,
        )

        return euler_integrate_actions(
            velocity_fn,
            initial_noise,
            resolved_mask,
            num_steps=num_steps,
        )

    @torch.no_grad()
    def sample_actions_rtc(
        self,
        model_inputs: dict[str, Any],
        normalized_proprio: torch.Tensor,
        previous_action_target: torch.Tensor,
        slot_weights: torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        num_steps: int = 10,
        max_guidance_weight: float = 10.0,
    ) -> RTCSamplingOutput:
        """以同一 Context/Noise 生成 paired raw 与 RTC Chunk，只有 guided Chunk 用于执行。"""

        initial_noise, action_mask, velocity_fn = self._prepare_action_sampling(
            model_inputs,
            normalized_proprio,
            generator=generator,
        )
        expected = initial_noise.shape
        if previous_action_target.shape != expected:
            raise ValueError(f"RTC previous_action_target 应为 {tuple(expected)}")
        if slot_weights.shape != action_mask.shape:
            raise ValueError(f"RTC slot_weights 应为 {tuple(action_mask.shape)}")
        raw_action = euler_integrate_actions(
            velocity_fn,
            initial_noise,
            action_mask,
            num_steps=num_steps,
        )
        rtc: RTCFlowIntegrationOutput = euler_integrate_actions_with_rtc(
            velocity_fn,
            initial_noise,
            action_mask,
            previous_action_target,
            slot_weights,
            max_guidance_weight=max_guidance_weight,
            num_steps=num_steps,
        )
        return RTCSamplingOutput(
            guided_action=rtc.action,
            raw_action=raw_action,
            guidance_coefficients=rtc.guidance_coefficients,
        )
