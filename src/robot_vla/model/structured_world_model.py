"""只预测四步结构化机器人状态的轻量确定性世界模型。"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass

import torch
from torch import nn

from robot_vla.contracts import RobotSpec
from robot_vla.model.layers import FP32RMSNorm

TINY_STRUCTURED_WORLD_MODEL_ARCH = "tiny-structured-world-model/v0"


@dataclass(frozen=True)
class TinyStructuredWorldModelConfig:
    """固定 Franka 结构化状态、动作和实际执行前缀的 V0 配置。"""

    state_dim: int = 15
    command_dim: int = 7
    action_dim: int = 8
    rollout_horizon: int = 4
    hidden_dim: int = 128
    rms_norm_eps: float = 1e-5
    normalized_state_abs_limit: float = 5.0
    normalized_action_abs_limit: float = 1.0

    def __post_init__(self) -> None:
        spec = RobotSpec()
        for name in (
            "state_dim",
            "command_dim",
            "action_dim",
            "rollout_horizon",
            "hidden_dim",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int):
                raise TypeError(f"{name} 必须为整数")
        if self.state_dim != spec.proprio_dim:
            raise ValueError(f"state_dim 必须等于 Franka proprio_dim={spec.proprio_dim}")
        if self.command_dim != spec.arm_dof:
            raise ValueError(f"command_dim 必须等于 Franka arm_dof={spec.arm_dof}")
        if self.action_dim != spec.action_dim:
            raise ValueError(f"action_dim 必须等于 Franka action_dim={spec.action_dim}")
        if self.rollout_horizon != spec.execute_steps:
            raise ValueError(
                f"rollout_horizon 必须等于实际执行前缀长度 {spec.execute_steps}"
            )
        if self.hidden_dim <= 0:
            raise ValueError("hidden_dim 必须为正整数")
        for name in (
            "rms_norm_eps",
            "normalized_state_abs_limit",
            "normalized_action_abs_limit",
        ):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} 必须是数值")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} 必须是有限正数")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    def sha256(self) -> str:
        encoded = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


@dataclass(frozen=True)
class StructuredWorldModelOutput:
    """四个实际执行步的结构化状态预测。"""

    predicted_state: torch.Tensor
    predicted_delta: torch.Tensor
    transition_mask: torch.Tensor


class TinyStructuredWorldModel(nn.Module):
    """共享单步 residual MLP，并自回归展开固定四个控制步。"""

    def __init__(self, config: TinyStructuredWorldModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or TinyStructuredWorldModelConfig()
        input_dim = (
            self.config.state_dim + self.config.action_dim + self.config.command_dim
        )
        self.transition = nn.Sequential(
            nn.Linear(input_dim, self.config.hidden_dim),
            nn.SiLU(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.SiLU(),
            FP32RMSNorm(
                self.config.hidden_dim,
                eps=self.config.rms_norm_eps,
            ),
        )
        self.delta_head = nn.Linear(self.config.hidden_dim, self.config.state_dim)
        # 未训练模型从“状态保持”基线开始，不产生任意大的虚假运动。
        nn.init.zeros_(self.delta_head.weight)
        nn.init.zeros_(self.delta_head.bias)

    def _validate_inputs(
        self,
        initial_state: torch.Tensor,
        action_prefix: torch.Tensor,
        command_target_prefix: torch.Tensor,
        transition_mask: torch.Tensor,
    ) -> None:
        if not all(
            isinstance(value, torch.Tensor)
            for value in (
                initial_state,
                action_prefix,
                command_target_prefix,
                transition_mask,
            )
        ):
            raise TypeError(
                "initial_state/action_prefix/command_target_prefix/transition_mask 必须为 Tensor"
            )
        batch_size = initial_state.shape[0] if initial_state.ndim > 0 else 0
        expected_state = (batch_size, self.config.state_dim)
        expected_action = (
            batch_size,
            self.config.rollout_horizon,
            self.config.action_dim,
        )
        expected_command = (
            batch_size,
            self.config.rollout_horizon,
            self.config.command_dim,
        )
        expected_mask = (batch_size, self.config.rollout_horizon)
        if batch_size <= 0 or tuple(initial_state.shape) != expected_state:
            raise ValueError(f"initial_state 必须是非空 {expected_state} Tensor")
        if tuple(action_prefix.shape) != expected_action:
            raise ValueError(f"action_prefix 必须是 {expected_action} Tensor")
        if tuple(command_target_prefix.shape) != expected_command:
            raise ValueError(f"command_target_prefix 必须是 {expected_command} Tensor")
        if tuple(transition_mask.shape) != expected_mask:
            raise ValueError(f"transition_mask 必须是 {expected_mask} Tensor")
        if not all(
            value.is_floating_point()
            for value in (initial_state, action_prefix, command_target_prefix)
        ):
            raise TypeError("state/action/command target 必须为浮点 Tensor")
        if transition_mask.dtype != torch.bool:
            raise TypeError("transition_mask 必须为 bool Tensor")
        if not (
            initial_state.device
            == action_prefix.device
            == command_target_prefix.device
            == transition_mask.device
        ):
            raise ValueError("模型输入必须位于同一 device")
        if initial_state.device.type not in {"cpu", "cuda"}:
            raise ValueError("V0 世界模型只支持 cpu/cuda device")
        if any(parameter.device != initial_state.device for parameter in self.parameters()):
            raise ValueError("V0 世界模型参数与输入必须位于同一 device")
        if any(parameter.dtype != torch.float32 for parameter in self.parameters()):
            raise TypeError("V0 世界模型参数必须保持 FP32")
        if not all(
            torch.isfinite(value).all()
            for value in (initial_state, action_prefix, command_target_prefix)
        ):
            raise ValueError("模型输入包含 NaN 或 Inf")
        state_limit = self.config.normalized_state_abs_limit + 1e-5
        if torch.any(initial_state.abs() > state_limit):
            raise ValueError("initial_state 超出归一化状态范围")
        if not torch.any(transition_mask):
            raise ValueError("batch 中没有有效 transition")
        if self.config.rollout_horizon > 1 and torch.any(
            transition_mask[:, 1:] & ~transition_mask[:, :-1]
        ):
            raise ValueError("transition_mask 必须是连续 True-prefix")
        valid_action = transition_mask.unsqueeze(-1).expand_as(action_prefix)
        action_limit = self.config.normalized_action_abs_limit + 1e-5
        if torch.any(action_prefix[valid_action].abs() > action_limit):
            raise ValueError("有效 action_prefix 超出归一化动作范围")
        valid_command = transition_mask.unsqueeze(-1).expand_as(command_target_prefix)
        if torch.any(command_target_prefix[valid_command].abs() > state_limit):
            raise ValueError("有效 command_target_prefix 超出归一化状态范围")

    def forward(
        self,
        initial_state: torch.Tensor,
        action_prefix: torch.Tensor,
        command_target_prefix: torch.Tensor,
        transition_mask: torch.Tensor,
    ) -> StructuredWorldModelOutput:
        """使用显式 commanded-q reference 预测 ``state[t+1:t+5]``。"""

        self._validate_inputs(
            initial_state,
            action_prefix,
            command_target_prefix,
            transition_mask,
        )
        state = initial_state.float()
        predicted_states: list[torch.Tensor] = []
        predicted_deltas: list[torch.Tensor] = []

        for step in range(self.config.rollout_horizon):
            valid = transition_mask[:, step].unsqueeze(-1)
            action = torch.where(
                valid,
                action_prefix[:, step].float(),
                torch.zeros_like(action_prefix[:, step], dtype=torch.float32),
            )
            command_target = torch.where(
                valid,
                command_target_prefix[:, step].float(),
                torch.zeros_like(
                    command_target_prefix[:, step], dtype=torch.float32
                ),
            )
            # 23K 以内的 V0 固定使用 FP32；外层 VLA autocast 不能静默改变动力学数值路径。
            with torch.autocast(device_type=state.device.type, enabled=False):
                features = self.transition(
                    torch.cat((state, action, command_target), dim=-1).float()
                )
                delta = self.delta_head(features).float()
            candidate_state = state + delta
            if not torch.isfinite(candidate_state).all():
                raise FloatingPointError("结构化状态 rollout 产生 NaN 或 Inf")
            state = torch.where(valid, candidate_state, state)
            predicted_deltas.append(torch.where(valid, delta, torch.zeros_like(delta)))
            predicted_states.append(torch.where(valid, state, torch.zeros_like(state)))

        return StructuredWorldModelOutput(
            predicted_state=torch.stack(predicted_states, dim=1),
            predicted_delta=torch.stack(predicted_deltas, dim=1),
            transition_mask=transition_mask,
        )


__all__ = [
    "TINY_STRUCTURED_WORLD_MODEL_ARCH",
    "StructuredWorldModelOutput",
    "TinyStructuredWorldModel",
    "TinyStructuredWorldModelConfig",
]
