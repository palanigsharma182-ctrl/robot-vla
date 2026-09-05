"""TinyStructuredWorldModel V0 的严格 masked rollout loss。"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from robot_vla.model.structured_world_model import (
    StructuredWorldModelOutput,
    TinyStructuredWorldModel,
)


@dataclass(frozen=True)
class StructuredWorldModelLossOutput:
    loss: torch.Tensor
    state_loss: torch.Tensor
    per_step_state_mse: torch.Tensor
    valid_transitions_per_step: torch.Tensor
    rollout: StructuredWorldModelOutput


def structured_world_model_loss(
    model: TinyStructuredWorldModel,
    initial_state: torch.Tensor,
    action_prefix: torch.Tensor,
    command_target_prefix: torch.Tensor,
    target_state: torch.Tensor,
    transition_mask: torch.Tensor,
) -> StructuredWorldModelLossOutput:
    """按显式 commanded-q 条件计算四步 open-loop state MSE。"""

    if not isinstance(initial_state, torch.Tensor):
        raise TypeError("initial_state 必须为 Tensor")
    expected_target = (
        initial_state.shape[0] if initial_state.ndim > 0 else 0,
        model.config.rollout_horizon,
        model.config.state_dim,
    )
    if not isinstance(target_state, torch.Tensor):
        raise TypeError("target_state 必须为 Tensor")
    if tuple(target_state.shape) != expected_target:
        raise ValueError(f"target_state 必须是 {expected_target} Tensor")
    if not target_state.is_floating_point():
        raise TypeError("target_state 必须为浮点 Tensor")
    if target_state.device != initial_state.device:
        raise ValueError("target_state 与模型输入必须位于同一 device")
    if not torch.isfinite(target_state).all():
        raise ValueError("target_state 包含 NaN 或 Inf")

    rollout = model(
        initial_state,
        action_prefix,
        command_target_prefix,
        transition_mask,
    )
    valid_target = transition_mask.unsqueeze(-1).expand_as(target_state)
    target_limit = model.config.normalized_state_abs_limit + 1e-5
    if torch.any(target_state[valid_target].abs() > target_limit):
        raise ValueError("有效 target_state 超出归一化状态范围")

    error = rollout.predicted_state.float() - target_state.float()
    valid_mask = transition_mask.unsqueeze(-1)
    # 必须在平方前清零 padding；极大的有限 padding 也不能形成 inf * 0 -> NaN。
    masked_error = torch.where(valid_mask, error, torch.zeros_like(error))
    valid_per_step = transition_mask.sum(dim=0)
    valid_count = valid_per_step.sum()
    if valid_count.item() == 0:
        raise ValueError("batch 中没有有效 transition")
    denominator = valid_count.to(dtype=torch.float32) * model.config.state_dim
    state_loss = masked_error.square().sum(dtype=torch.float32) / denominator
    if not torch.isfinite(state_loss):
        raise FloatingPointError("结构化世界模型 loss 产生 NaN 或 Inf")

    per_step_losses: list[torch.Tensor] = []
    for step in range(model.config.rollout_horizon):
        step_count = valid_per_step[step]
        if step_count.item() == 0:
            per_step_losses.append(masked_error[:, step].sum(dtype=torch.float32) * 0.0)
            continue
        step_denominator = step_count.to(dtype=torch.float32) * model.config.state_dim
        per_step_losses.append(
            masked_error[:, step].square().sum(dtype=torch.float32)
            / step_denominator
        )

    per_step_state_mse = torch.stack(per_step_losses)
    if not torch.isfinite(per_step_state_mse).all():
        raise FloatingPointError("结构化世界模型逐步 loss 产生 NaN 或 Inf")

    return StructuredWorldModelLossOutput(
        loss=state_loss,
        state_loss=state_loss,
        per_step_state_mse=per_step_state_mse,
        valid_transitions_per_step=valid_per_step,
        rollout=rollout,
    )


__all__ = [
    "StructuredWorldModelLossOutput",
    "structured_world_model_loss",
]
