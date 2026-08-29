"""D016 固定的 Rectified Flow 训练目标、masked loss 和 Euler 采样。"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class FlowTrainingTarget:
    noisy_action: torch.Tensor
    target_velocity: torch.Tensor
    flow_time: torch.Tensor
    noise: torch.Tensor


@dataclass(frozen=True)
class RTCFlowIntegrationOutput:
    action: torch.Tensor
    guidance_coefficients: tuple[float, ...]


def _validate_action_and_mask(action: torch.Tensor, action_mask: torch.Tensor) -> None:
    if action.ndim != 3:
        raise ValueError(f"action 应为 [B,H,A]，实际为 {tuple(action.shape)}")
    if action_mask.shape != action.shape[:2] or action_mask.dtype != torch.bool:
        raise ValueError("action_mask 必须是与 [B,H] 对齐的 bool Tensor")
    if not torch.isfinite(action).all():
        raise ValueError("action 包含 NaN 或 Inf")


def sample_flow_training_target(
    normalized_action: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    generator: torch.Generator | None = None,
) -> FlowTrainingTarget:
    """采样 ``x_t=t*epsilon+(1-t)*a`` 和目标速度 ``epsilon-a``。"""

    _validate_action_and_mask(normalized_action, action_mask)
    action = normalized_action.float()
    if torch.any(action[action_mask].abs() > 1.0 + 1e-5):
        raise ValueError("有效 normalized action 必须位于 [-1,1]")
    batch_size = action.shape[0]
    uniform = torch.rand(
        batch_size,
        device=action.device,
        dtype=torch.float32,
        generator=generator,
    )
    # Beta(alpha, 1) 的逆 CDF 为 U ** (1 / alpha)。
    flow_time = uniform.pow(1.0 / 1.5) * 0.999 + 0.001
    noise = torch.randn(
        action.shape,
        device=action.device,
        dtype=torch.float32,
        generator=generator,
    )
    time = flow_time.view(batch_size, 1, 1)
    noisy_action = time * noise + (1.0 - time) * action
    target_velocity = noise - action
    valid = action_mask.unsqueeze(-1)
    noisy_action = torch.where(valid, noisy_action, torch.zeros_like(noisy_action))
    target_velocity = torch.where(valid, target_velocity, torch.zeros_like(target_velocity))
    noise = torch.where(valid, noise, torch.zeros_like(noise))
    return FlowTrainingTarget(
        noisy_action=noisy_action,
        target_velocity=target_velocity,
        flow_time=flow_time,
        noise=noise,
    )


def masked_flow_mse(
    prediction: torch.Tensor,
    target_velocity: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    allow_empty: bool = False,
) -> torch.Tensor:
    if prediction.shape != target_velocity.shape:
        raise ValueError("prediction 与 target_velocity shape 必须相同")
    _validate_action_and_mask(target_velocity, action_mask)
    valid_count = action_mask.sum()
    if valid_count.item() == 0:
        if allow_empty:
            return prediction.float().sum() * 0.0
        raise ValueError("batch 中没有有效 Action Token")
    error = prediction.float() - target_velocity.float()
    mask = action_mask.unsqueeze(-1).to(dtype=torch.float32)
    denominator = valid_count.to(dtype=torch.float32) * prediction.shape[-1]
    return (error.square() * mask).sum(dtype=torch.float32) / denominator


def build_critical_event_mask(
    event_mask: torch.Tensor,
    action_mask: torch.Tensor,
    executed_action_steps: int,
) -> torch.Tensor:
    """只保留真实执行前缀中的有效关键事件 Action Token。"""

    if event_mask.shape != action_mask.shape:
        raise ValueError("event_mask 必须与 action_mask shape 相同")
    if event_mask.dtype != torch.bool or action_mask.dtype != torch.bool:
        raise ValueError("event_mask/action_mask 必须为 bool Tensor")
    if not 1 <= executed_action_steps <= action_mask.shape[1]:
        raise ValueError("executed_action_steps 必须位于 [1,H]")
    exec_mask = torch.arange(
        action_mask.shape[1],
        device=action_mask.device,
    ) < executed_action_steps
    return event_mask & action_mask & exec_mask.unsqueeze(0)


@torch.no_grad()
def euler_integrate_actions(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_noise: torch.Tensor,
    action_mask: torch.Tensor,
    *,
    num_steps: int = 10,
) -> torch.Tensor:
    """从 t=1 噪声出发，以负 dt 积分到 t=0，最后只 clamp 一次。"""

    _validate_action_and_mask(initial_noise, action_mask)
    if num_steps <= 0:
        raise ValueError("num_steps 必须为正整数")
    state = initial_noise.float()
    valid = action_mask.unsqueeze(-1)
    state = torch.where(valid, state, torch.zeros_like(state))
    dt = -1.0 / num_steps
    for step in range(num_steps):
        flow_time = torch.full(
            (state.shape[0],),
            1.0 + step * dt,
            device=state.device,
            dtype=torch.float32,
        )
        velocity = velocity_fn(state, flow_time)
        if velocity.shape != state.shape or not torch.isfinite(velocity).all():
            raise RuntimeError("velocity_fn 必须返回与 Action state 同 shape 的有限 Tensor")
        velocity = torch.where(valid, velocity.float(), torch.zeros_like(state))
        state = state + dt * velocity
        state = torch.where(valid, state, torch.zeros_like(state))
    return state.clamp(-1.0, 1.0)


def rtc_guidance_coefficient(
    flow_time: torch.Tensor,
    *,
    max_guidance_weight: float,
) -> torch.Tensor:
    """把 RTC Eq.(2) 的 noise→action 时间换算到本项目 action←noise 时间。"""

    if flow_time.ndim != 1 or not torch.isfinite(flow_time).all():
        raise ValueError("flow_time 必须是一维有限 Tensor")
    if torch.any((flow_time < 0.0) | (flow_time > 1.0)):
        raise ValueError("flow_time 必须位于 [0,1]")
    if not math.isfinite(max_guidance_weight) or max_guidance_weight <= 0:
        raise ValueError("max_guidance_weight 必须为正数")
    # 论文 tau 从 0(noise) 积分到 1(action)；本项目 t=1-tau，从 1 积分到 0。
    paper_tau = 1.0 - flow_time.float()
    project_time = flow_time.float()
    denominator = paper_tau * project_time
    ratio = torch.where(
        denominator > torch.finfo(torch.float32).eps,
        (paper_tau.square() + project_time.square()) / denominator,
        torch.full_like(project_time, float(max_guidance_weight)),
    )
    return ratio.clamp(max=float(max_guidance_weight))


@torch.no_grad()
def euler_integrate_actions_with_rtc(
    velocity_fn: Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
    initial_noise: torch.Tensor,
    action_mask: torch.Tensor,
    previous_action_target: torch.Tensor,
    slot_weights: torch.Tensor,
    *,
    max_guidance_weight: float,
    num_steps: int = 10,
) -> RTCFlowIntegrationOutput:
    """以 RTC ΠGDM VJP 修正 Flow velocity；不对中间 action state 做 clamp。"""

    _validate_action_and_mask(initial_noise, action_mask)
    if num_steps <= 0:
        raise ValueError("num_steps 必须为正整数")
    if previous_action_target.shape != initial_noise.shape:
        raise ValueError("previous_action_target 必须与 Action state 同 shape")
    if slot_weights.shape != action_mask.shape:
        raise ValueError("slot_weights 必须与 [B,H] 对齐")
    if not torch.isfinite(previous_action_target).all() or not torch.isfinite(slot_weights).all():
        raise ValueError("RTC target/slot_weights 必须有限")
    if torch.any((slot_weights < 0.0) | (slot_weights > 1.0)):
        raise ValueError("RTC slot_weights 必须位于 [0,1]")

    state = initial_noise.float()
    valid = action_mask.unsqueeze(-1)
    state = torch.where(valid, state, torch.zeros_like(state))
    target = previous_action_target.detach().float()
    weights = slot_weights.detach().float() * action_mask.to(dtype=torch.float32)
    dt = -1.0 / num_steps
    coefficients: list[float] = []
    for step in range(num_steps):
        flow_time = torch.full(
            (state.shape[0],),
            1.0 + step * dt,
            device=state.device,
            dtype=torch.float32,
        )
        with torch.enable_grad():
            differentiable_state = state.detach().requires_grad_(True)
            velocity = velocity_fn(differentiable_state, flow_time)
            if velocity.shape != state.shape or not torch.isfinite(velocity).all():
                raise RuntimeError("velocity_fn 必须返回与 Action state 同 shape 的有限 Tensor")
            velocity = torch.where(valid, velocity.float(), torch.zeros_like(state))
            # 本项目 x_t=t*noise+(1-t)*action、v=noise-action，因此 clean endpoint=a=x_t-t*v。
            clean_endpoint = differentiable_state - flow_time.view(-1, 1, 1) * velocity
            weighted_error = (target - clean_endpoint) * weights.unsqueeze(-1)
            guidance = torch.autograd.grad(
                clean_endpoint,
                differentiable_state,
                grad_outputs=weighted_error,
                create_graph=False,
                retain_graph=False,
            )[0]
        if not torch.isfinite(guidance).all():
            raise RuntimeError("RTC guidance 包含 NaN 或 Inf")
        coefficient = rtc_guidance_coefficient(
            flow_time,
            max_guidance_weight=max_guidance_weight,
        )
        coefficients.append(float(coefficient[0].item()))
        # 论文速度沿 noise→action；本项目速度和积分方向均相反，所以 guidance 符号取反。
        guided_velocity = velocity.detach() - coefficient.view(-1, 1, 1) * guidance.detach()
        guided_velocity = torch.where(valid, guided_velocity, torch.zeros_like(state))
        state = state + dt * guided_velocity
        state = torch.where(valid, state, torch.zeros_like(state))
    return RTCFlowIntegrationOutput(
        action=state.clamp(-1.0, 1.0),
        guidance_coefficients=tuple(coefficients),
    )
