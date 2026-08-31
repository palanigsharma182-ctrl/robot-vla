"""D015 固定的 SmolVLA-style standalone Action Expert。"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from robot_vla.contracts import OBSERVATION_HISTORY_LENGTH, OBSERVATION_V2_VERSION
from robot_vla.model.layers import FP32RMSNorm
from robot_vla.model.qwen_context import QwenContext
from robot_vla.observation import (
    OBSERVATION_V2_CONTROLLER_STATE_DIM,
    OBSERVATION_V2_FRAME_STATE_DIM,
)


@dataclass(frozen=True)
class ExpertConfig:
    proprio_dim: int = 15
    action_dim: int = 8
    action_horizon: int = 16
    context_dim: int = 720
    hidden_size: int = 720
    state_hidden_size: int = 256
    num_layers: int = 16
    intermediate_size: int = 2048
    num_attention_heads: int = 15
    num_key_value_heads: int = 5
    head_dim: int = 64
    rms_norm_eps: float = 1e-5
    time_min_period: float = 0.004
    time_max_period: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            "proprio_dim",
            "action_dim",
            "action_horizon",
            "context_dim",
            "hidden_size",
            "state_hidden_size",
            "num_layers",
            "intermediate_size",
            "num_attention_heads",
            "num_key_value_heads",
            "head_dim",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} 必须为正数")
        if self.num_layers % 2 != 0:
            raise ValueError("num_layers 必须为偶数，以便交替 Self/Cross Attention")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads 必须能被 num_key_value_heads 整除")
        if self.hidden_size % 2 != 0:
            raise ValueError("hidden_size 必须为偶数，以构造 Sin/Cos Time Embedding")
        if self.rms_norm_eps <= 0:
            raise ValueError("rms_norm_eps 必须为正数")
        if not 0 < self.time_min_period < self.time_max_period:
            raise ValueError("Flow Time period 范围无效")

    def is_qwen_vla_v01(self) -> bool:
        return self == ExpertConfig()


@dataclass(frozen=True)
class TemporalExpertConfig(ExpertConfig):
    """Observation V2 专用配置；与 V1 ExpertConfig/checkpoint 显式隔离。"""

    observation_version: str = OBSERVATION_V2_VERSION
    history_length: int = OBSERVATION_HISTORY_LENGTH
    frame_state_dim: int = OBSERVATION_V2_FRAME_STATE_DIM
    controller_state_dim: int = OBSERVATION_V2_CONTROLLER_STATE_DIM

    def __post_init__(self) -> None:
        super().__post_init__()
        if self.observation_version != OBSERVATION_V2_VERSION:
            raise ValueError(f"Temporal Expert 必须使用 {OBSERVATION_V2_VERSION}")
        if self.history_length != OBSERVATION_HISTORY_LENGTH:
            raise ValueError(f"Temporal Expert history_length 必须为 {OBSERVATION_HISTORY_LENGTH}")
        if self.frame_state_dim != OBSERVATION_V2_FRAME_STATE_DIM:
            raise ValueError("Temporal Expert frame_state_dim 与 Observation V2 不一致")
        if self.controller_state_dim != OBSERVATION_V2_CONTROLLER_STATE_DIM:
            raise ValueError("Temporal Expert controller_state_dim 与 Observation V2 不一致")


def sinusoidal_time_embedding(flow_time: torch.Tensor, config: ExpertConfig) -> torch.Tensor:
    if flow_time.ndim != 1 or not torch.isfinite(flow_time).all():
        raise ValueError("flow_time 必须是 [B] 有限 Tensor")
    if torch.any(flow_time < 0.0) or torch.any(flow_time > 1.0):
        raise ValueError("flow_time 必须位于 [0,1]")
    half_dim = config.hidden_size // 2
    periods = torch.exp(
        torch.linspace(
            math.log(config.time_min_period),
            math.log(config.time_max_period),
            half_dim,
            device=flow_time.device,
            dtype=torch.float32,
        )
    )
    phase = flow_time.float().unsqueeze(-1) * (2.0 * math.pi) / periods
    return torch.cat((phase.sin(), phase.cos()), dim=-1)


class StateTokenEncoder(nn.Module):
    def __init__(self, config: ExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.projection = nn.Sequential(
            nn.Linear(config.proprio_dim, config.state_hidden_size),
            nn.SiLU(),
            nn.Linear(config.state_hidden_size, config.hidden_size),
        )
        self.norm = FP32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.type_embedding = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        nn.init.normal_(self.type_embedding, std=0.02)

    def forward(self, normalized_proprio: torch.Tensor) -> torch.Tensor:
        if normalized_proprio.ndim != 2 or normalized_proprio.shape[-1] != self.config.proprio_dim:
            raise ValueError(
                f"normalized_proprio 应为 [B,{self.config.proprio_dim}]，"
                f"实际为 {tuple(normalized_proprio.shape)}"
            )
        if not torch.isfinite(normalized_proprio).all():
            raise ValueError("normalized_proprio 包含 NaN 或 Inf")
        state = self.norm(self.projection(normalized_proprio))
        return state.unsqueeze(1) + self.type_embedding


class TemporalStateTokenEncoder(nn.Module):
    """四个共享 MLP frame token + 一个当前 controller token。"""

    def __init__(self, config: TemporalExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.frame_projection = nn.Sequential(
            nn.Linear(config.frame_state_dim, config.state_hidden_size),
            nn.SiLU(),
            nn.Linear(config.state_hidden_size, config.hidden_size),
        )
        self.controller_projection = nn.Sequential(
            nn.Linear(config.controller_state_dim, config.state_hidden_size),
            nn.SiLU(),
            nn.Linear(config.state_hidden_size, config.hidden_size),
        )
        self.frame_norm = FP32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.controller_norm = FP32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.frame_type_embedding = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.controller_type_embedding = nn.Parameter(torch.empty(1, 1, config.hidden_size))
        self.time_embedding = nn.Parameter(
            torch.empty(1, config.history_length, config.hidden_size)
        )
        nn.init.normal_(self.frame_type_embedding, std=0.02)
        nn.init.normal_(self.controller_type_embedding, std=0.02)
        nn.init.normal_(self.time_embedding, std=0.02)

    def forward(
        self,
        state_history: torch.Tensor,
        state_history_mask: torch.Tensor,
        controller_state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = state_history.shape[0]
        expected_history = (
            batch_size,
            self.config.history_length,
            self.config.frame_state_dim,
        )
        if state_history.ndim != 3 or tuple(state_history.shape) != expected_history:
            raise ValueError(f"state_history 应为 {expected_history}")
        if (
            state_history_mask.shape != expected_history[:2]
            or state_history_mask.dtype != torch.bool
        ):
            raise ValueError("state_history_mask 必须是对齐的 bool [B,4]")
        if not torch.all(state_history_mask[:, -1]):
            raise ValueError("Temporal Expert 当前状态 token 必须有效")
        expected_controller = (batch_size, self.config.controller_state_dim)
        if tuple(controller_state.shape) != expected_controller:
            raise ValueError(f"controller_state 应为 {expected_controller}")
        if not torch.isfinite(state_history).all() or not torch.isfinite(
            controller_state
        ).all():
            raise ValueError("Observation V2 state 包含 NaN 或 Inf")

        frame = self.frame_norm(self.frame_projection(state_history))
        frame = frame + self.frame_type_embedding + self.time_embedding
        frame = frame * state_history_mask.unsqueeze(-1).to(dtype=frame.dtype)
        controller = self.controller_norm(
            self.controller_projection(controller_state)
        ).unsqueeze(1)
        controller = controller + self.controller_type_embedding
        controller_mask = torch.ones(
            batch_size,
            1,
            dtype=torch.bool,
            device=state_history_mask.device,
        )
        return (
            torch.cat((frame, controller), dim=1),
            torch.cat((state_history_mask, controller_mask), dim=1),
        )


class ActionTokenEncoder(nn.Module):
    def __init__(self, config: ExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.action_projection = nn.Linear(config.action_dim, config.hidden_size)
        self.action_time_projection = nn.Sequential(
            nn.Linear(config.hidden_size * 2, config.hidden_size),
            nn.SiLU(),
            nn.Linear(config.hidden_size, config.hidden_size),
        )
        self.slot_embedding = nn.Parameter(
            torch.empty(1, config.action_horizon, config.hidden_size)
        )
        nn.init.normal_(self.slot_embedding, std=0.02)

    def forward(self, noisy_action: torch.Tensor, flow_time: torch.Tensor) -> torch.Tensor:
        expected = (noisy_action.shape[0], self.config.action_horizon, self.config.action_dim)
        if noisy_action.ndim != 3 or tuple(noisy_action.shape) != expected:
            raise ValueError(f"noisy_action 应为 [B,{expected[1]},{expected[2]}]")
        if not torch.isfinite(noisy_action).all():
            raise ValueError("noisy_action 包含 NaN 或 Inf")
        if flow_time.shape != (noisy_action.shape[0],):
            raise ValueError("flow_time shape 必须为 [B]")
        action = self.action_projection(noisy_action)
        time = sinusoidal_time_embedding(flow_time, self.config)
        time = time.unsqueeze(1).expand(-1, self.config.action_horizon, -1)
        return self.action_time_projection(torch.cat((action, time), dim=-1)) + self.slot_embedding


@dataclass(frozen=True)
class AttentionKV:
    key: torch.Tensor
    value: torch.Tensor
    mask: torch.Tensor


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: ExpertConfig) -> None:
        super().__init__()
        self.config = config
        self.q_projection = nn.Linear(
            config.hidden_size,
            config.num_attention_heads * config.head_dim,
            bias=False,
        )
        kv_width = config.num_key_value_heads * config.head_dim
        self.k_projection = nn.Linear(config.context_dim, kv_width, bias=False)
        self.v_projection = nn.Linear(config.context_dim, kv_width, bias=False)
        self.output_projection = nn.Linear(
            config.num_attention_heads * config.head_dim,
            config.hidden_size,
            bias=False,
        )

    def project_kv(self, source: torch.Tensor, source_mask: torch.Tensor) -> AttentionKV:
        if source.ndim != 3 or source.shape[-1] != self.config.context_dim:
            raise ValueError("Attention K/V source shape 无效")
        if source_mask.shape != source.shape[:2] or source_mask.dtype != torch.bool:
            raise ValueError("Attention K/V mask 必须是对齐的 bool Tensor")
        batch_size, source_length, _ = source.shape
        key = self.k_projection(source).view(
            batch_size,
            source_length,
            self.config.num_key_value_heads,
            self.config.head_dim,
        )
        value = self.v_projection(source).view_as(key)
        return AttentionKV(
            key=key.transpose(1, 2),
            value=value.transpose(1, 2),
            mask=source_mask,
        )

    def forward(
        self,
        query: torch.Tensor,
        query_mask: torch.Tensor,
        *,
        source: torch.Tensor | None = None,
        source_mask: torch.Tensor | None = None,
        kv: AttentionKV | None = None,
    ) -> torch.Tensor:
        if query.ndim != 3 or query.shape[-1] != self.config.hidden_size:
            raise ValueError("Attention Query shape 无效")
        if query_mask.shape != query.shape[:2] or query_mask.dtype != torch.bool:
            raise ValueError("Attention Query mask 必须是对齐的 bool Tensor")
        if kv is None:
            if source is None or source_mask is None:
                raise ValueError("没有 K/V Cache 时必须提供 source 和 source_mask")
            kv = self.project_kv(source, source_mask)
        if kv.mask.shape[0] != query.shape[0] or not torch.all(kv.mask.any(dim=1)):
            raise ValueError("每个样本必须至少包含一个有效 Attention K/V Token")

        batch_size, query_length, _ = query.shape
        q = self.q_projection(query).view(
            batch_size,
            query_length,
            self.config.num_attention_heads,
            self.config.head_dim,
        )
        q = q.transpose(1, 2)
        repeat = self.config.num_attention_heads // self.config.num_key_value_heads
        key = kv.key.repeat_interleave(repeat, dim=1)
        value = kv.value.repeat_interleave(repeat, dim=1)
        attention_mask = kv.mask[:, None, None, :]
        attended = F.scaled_dot_product_attention(
            q,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
        )
        attended = attended.transpose(1, 2).reshape(batch_size, query_length, -1)
        output = self.output_projection(attended)
        return output * query_mask.unsqueeze(-1).to(dtype=output.dtype)


class GatedMLP(nn.Module):
    def __init__(self, config: ExpertConfig) -> None:
        super().__init__()
        self.gate_projection = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.up_projection = nn.Linear(
            config.hidden_size,
            config.intermediate_size,
            bias=False,
        )
        self.down_projection = nn.Linear(
            config.intermediate_size,
            config.hidden_size,
            bias=False,
        )

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_projection(F.silu(self.gate_projection(value)) * self.up_projection(value))


class ExpertBlock(nn.Module):
    def __init__(self, config: ExpertConfig, *, cross_attention: bool) -> None:
        super().__init__()
        self.cross_attention = cross_attention
        attention_config = config
        if not cross_attention and config.context_dim != config.hidden_size:
            attention_config = type(config)(
                **{**config.__dict__, "context_dim": config.hidden_size}
            )
        self.attention_norm = FP32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.attention = GroupedQueryAttention(attention_config)
        self.mlp_norm = FP32RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.mlp = GatedMLP(config)

    def forward(
        self,
        hidden: torch.Tensor,
        sequence_mask: torch.Tensor,
        *,
        context: QwenContext | None = None,
        kv: AttentionKV | None = None,
    ) -> torch.Tensor:
        normalized = self.attention_norm(hidden)
        if self.cross_attention:
            if context is None:
                raise ValueError("Cross-Attention Block 需要 Qwen Context")
            attention = self.attention(
                normalized,
                sequence_mask,
                source=context.tokens,
                source_mask=context.mask,
                kv=kv,
            )
        else:
            attention = self.attention(
                normalized,
                sequence_mask,
                source=normalized,
                source_mask=sequence_mask,
            )
        hidden = (hidden + attention) * sequence_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        hidden = hidden + self.mlp(self.mlp_norm(hidden))
        return hidden * sequence_mask.unsqueeze(-1).to(dtype=hidden.dtype)


class StandaloneActionExpert(nn.Module):
    """State + Noisy Action 序列交替执行非因果 Self/Cross Attention。"""

    def __init__(self, config: ExpertConfig | None = None) -> None:
        super().__init__()
        self.config = config or ExpertConfig()
        self.state_encoder = StateTokenEncoder(self.config)
        self.action_encoder = ActionTokenEncoder(self.config)
        self.blocks = nn.ModuleList(
            ExpertBlock(self.config, cross_attention=layer_index % 2 == 1)
            for layer_index in range(self.config.num_layers)
        )
        self.final_norm = FP32RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.velocity_head = nn.Linear(self.config.hidden_size, self.config.action_dim)

    def prepare_context_kv(self, context: QwenContext) -> tuple[AttentionKV, ...]:
        self._validate_context(context)
        return tuple(
            block.attention.project_kv(context.tokens, context.mask)
            for block in self.blocks
            if block.cross_attention
        )

    def forward(
        self,
        context: QwenContext,
        normalized_proprio: torch.Tensor,
        noisy_action: torch.Tensor,
        flow_time: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        context_kv: tuple[AttentionKV, ...] | None = None,
    ) -> torch.Tensor:
        self._validate_context(context)
        batch_size = context.tokens.shape[0]
        expected_action = (
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
        )
        if tuple(noisy_action.shape) != expected_action:
            raise ValueError(f"noisy_action 应为 {expected_action}，实际为 {tuple(noisy_action.shape)}")
        if action_mask.shape != expected_action[:2] or action_mask.dtype != torch.bool:
            raise ValueError("action_mask 必须是与 [B,H] 对齐的 bool Tensor")
        if context_kv is not None and len(context_kv) != self.config.num_layers // 2:
            raise ValueError("context_kv 数量必须等于 Cross-Attention 层数")

        state_token = self.state_encoder(normalized_proprio)
        action_tokens = self.action_encoder(noisy_action, flow_time)
        sequence_mask = torch.cat(
            (
                torch.ones(batch_size, 1, dtype=torch.bool, device=action_mask.device),
                action_mask,
            ),
            dim=1,
        )
        hidden = torch.cat((state_token, action_tokens), dim=1)
        hidden = hidden * sequence_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        cross_index = 0
        for block in self.blocks:
            cached = None
            if block.cross_attention and context_kv is not None:
                cached = context_kv[cross_index]
            hidden = block(
                hidden,
                sequence_mask,
                context=context if block.cross_attention else None,
                kv=cached,
            )
            if block.cross_attention:
                cross_index += 1

        action_hidden = self.final_norm(hidden[:, 1:])
        velocity = self.velocity_head(action_hidden).float()
        return velocity * action_mask.unsqueeze(-1).to(dtype=torch.float32)

    def _validate_context(self, context: QwenContext) -> None:
        if context.tokens.ndim != 3 or context.tokens.shape[-1] != self.config.context_dim:
            raise ValueError(
                f"Expert Context 应为 [B,N,{self.config.context_dim}]，"
                f"实际为 {tuple(context.tokens.shape)}"
            )
        if context.mask.shape != context.tokens.shape[:2] or context.mask.dtype != torch.bool:
            raise ValueError("Expert Context mask 必须是与 [B,N] 对齐的 bool Tensor")
        if not torch.all(context.mask.any(dim=1)):
            raise ValueError("每个样本必须至少包含一个有效 Context Token")


class TemporalStandaloneActionExpert(StandaloneActionExpert):
    """Observation V2 的四步 State/Controller token Action Expert。"""

    def __init__(self, config: TemporalExpertConfig | None = None) -> None:
        nn.Module.__init__(self)
        self.config = config or TemporalExpertConfig()
        self.state_encoder = TemporalStateTokenEncoder(self.config)
        self.action_encoder = ActionTokenEncoder(self.config)
        self.blocks = nn.ModuleList(
            ExpertBlock(self.config, cross_attention=layer_index % 2 == 1)
            for layer_index in range(self.config.num_layers)
        )
        self.final_norm = FP32RMSNorm(self.config.hidden_size, eps=self.config.rms_norm_eps)
        self.velocity_head = nn.Linear(self.config.hidden_size, self.config.action_dim)

    def forward(
        self,
        context: QwenContext,
        state_history: torch.Tensor,
        noisy_action: torch.Tensor,
        flow_time: torch.Tensor,
        action_mask: torch.Tensor,
        *,
        state_history_mask: torch.Tensor,
        controller_state: torch.Tensor,
        context_kv: tuple[AttentionKV, ...] | None = None,
    ) -> torch.Tensor:
        self._validate_context(context)
        batch_size = context.tokens.shape[0]
        expected_action = (
            batch_size,
            self.config.action_horizon,
            self.config.action_dim,
        )
        if tuple(noisy_action.shape) != expected_action:
            raise ValueError(f"noisy_action 应为 {expected_action}")
        if action_mask.shape != expected_action[:2] or action_mask.dtype != torch.bool:
            raise ValueError("action_mask 必须是与 [B,H] 对齐的 bool Tensor")
        if context_kv is not None and len(context_kv) != self.config.num_layers // 2:
            raise ValueError("context_kv 数量必须等于 Cross-Attention 层数")

        state_tokens, state_token_mask = self.state_encoder(
            state_history,
            state_history_mask,
            controller_state,
        )
        action_tokens = self.action_encoder(noisy_action, flow_time)
        sequence_mask = torch.cat((state_token_mask, action_mask), dim=1)
        hidden = torch.cat((state_tokens, action_tokens), dim=1)
        hidden = hidden * sequence_mask.unsqueeze(-1).to(dtype=hidden.dtype)
        cross_index = 0
        for block in self.blocks:
            cached = None
            if block.cross_attention and context_kv is not None:
                cached = context_kv[cross_index]
            hidden = block(
                hidden,
                sequence_mask,
                context=context if block.cross_attention else None,
                kv=cached,
            )
            if block.cross_attention:
                cross_index += 1
        action_hidden = self.final_norm(hidden[:, state_tokens.shape[1] :])
        velocity = self.velocity_head(action_hidden).float()
        return velocity * action_mask.unsqueeze(-1).to(dtype=torch.float32)
