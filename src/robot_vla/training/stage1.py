"""D018 固定的 Stage 1 优化配置、训练循环和确定性验证。"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

import torch
from torch import nn

from robot_vla.contracts import PICK_AND_PLACE_SKILLS, UNKNOWN_SKILL_ID

if TYPE_CHECKING:
    from robot_vla.model.policy import QwenVLAPolicy


@dataclass(frozen=True)
class Stage1TrainingConfig:
    """单张 4090 上冻结 Qwen、训练 Adapter/Expert 的默认配置。"""

    learning_rate: float = 1e-4
    betas: tuple[float, float] = (0.9, 0.95)
    eps: float = 1e-8
    weight_decay: float = 1e-10
    max_grad_norm: float = 10.0
    warmup_steps: int = 1_000
    cosine_decay_steps: int = 30_000
    decay_learning_rate: float = 2.5e-6
    micro_batch_size: int = 1
    gradient_accumulation_steps: int = 1
    use_bf16: bool = True
    seed: int = 42
    validation_seeds: tuple[int, ...] = (1_009,)
    checkpoint_interval_steps: int = 1_000
    samples_per_epoch: int | None = None
    skill_sampling_weights: tuple[tuple[int, float], ...] = ()
    source_sampling_weights: tuple[tuple[str, float], ...] = ()
    event_loss_weight: float = 2.0
    event_loss_warmup_steps: int = 0
    executed_action_steps: int = 4

    def __post_init__(self) -> None:
        if self.learning_rate <= 0:
            raise ValueError("learning_rate 必须为正数")
        if len(self.betas) != 2 or not all(0.0 <= beta < 1.0 for beta in self.betas):
            raise ValueError("betas 必须包含两个位于 [0,1) 的数值")
        if self.eps <= 0 or self.weight_decay < 0 or self.max_grad_norm <= 0:
            raise ValueError("eps/max_grad_norm 必须为正数，weight_decay 不能为负数")
        if self.warmup_steps <= 0 or self.cosine_decay_steps <= 0:
            raise ValueError("warmup_steps 和 cosine_decay_steps 必须为正整数")
        if not 0 <= self.decay_learning_rate <= self.learning_rate:
            raise ValueError("decay_learning_rate 必须位于 [0, learning_rate]")
        if self.micro_batch_size <= 0 or self.gradient_accumulation_steps <= 0:
            raise ValueError("batch size 和 gradient accumulation steps 必须为正整数")
        if not isinstance(self.use_bf16, bool):
            raise TypeError("use_bf16 必须为 bool")
        if self.seed < 0 or not self.validation_seeds or any(seed < 0 for seed in self.validation_seeds):
            raise ValueError("训练 seed 必须非负，且 validation_seeds 不能为空")
        if len(set(self.validation_seeds)) != len(self.validation_seeds):
            raise ValueError("validation_seeds 不能重复")
        if self.checkpoint_interval_steps <= 0:
            raise ValueError("checkpoint_interval_steps 必须为正整数")
        if self.samples_per_epoch is not None and self.samples_per_epoch <= 0:
            raise ValueError("samples_per_epoch 必须为正整数或 None")
        if not math.isfinite(self.event_loss_weight) or self.event_loss_weight < 0:
            raise ValueError("event_loss_weight 必须是有限非负数")
        if self.event_loss_warmup_steps < 0:
            raise ValueError("event_loss_warmup_steps 不能为负数")
        if self.executed_action_steps <= 0:
            raise ValueError("executed_action_steps 必须为正整数")
        skill_ids: set[int] = set()
        allowed_skill_ids = {UNKNOWN_SKILL_ID, *range(len(PICK_AND_PLACE_SKILLS))}
        for skill_id, weight in self.skill_sampling_weights:
            resolved_skill_id = int(skill_id)
            if resolved_skill_id not in allowed_skill_ids:
                raise ValueError(f"skill_sampling_weights 包含未知 skill_id: {resolved_skill_id}")
            if resolved_skill_id in skill_ids:
                raise ValueError("skill_sampling_weights 不能重复定义 skill_id")
            if not math.isfinite(float(weight)) or float(weight) <= 0:
                raise ValueError("skill_sampling_weights 的权重必须是有限正数")
            skill_ids.add(resolved_skill_id)
        source_ids: set[str] = set()
        for source, weight in self.source_sampling_weights:
            resolved_source = str(source)
            if not resolved_source or resolved_source.strip() != resolved_source:
                raise ValueError("source_sampling_weights 的 source 名称无效")
            if resolved_source in source_ids:
                raise ValueError("source_sampling_weights 不能重复定义 source")
            if not math.isfinite(float(weight)) or float(weight) <= 0:
                raise ValueError("source_sampling_weights 的权重必须是有限正数")
            source_ids.add(resolved_source)

    @property
    def effective_batch_size(self) -> int:
        return self.micro_batch_size * self.gradient_accumulation_steps

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> Stage1TrainingConfig:
        data = dict(value)
        if "betas" in data:
            data["betas"] = tuple(float(beta) for beta in data["betas"])
        if "validation_seeds" in data:
            data["validation_seeds"] = tuple(int(seed) for seed in data["validation_seeds"])
        if "skill_sampling_weights" in data:
            data["skill_sampling_weights"] = tuple(
                (int(skill_id), float(weight))
                for skill_id, weight in data["skill_sampling_weights"]
            )
        if "source_sampling_weights" in data:
            data["source_sampling_weights"] = tuple(
                (str(source), float(weight))
                for source, weight in data["source_sampling_weights"]
            )
        return cls(**data)


def learning_rate_at_step(config: Stage1TrainingConfig, optimizer_step: int) -> float:
    """返回给定优化更新所使用的 LR；step=0 表示第一次更新。"""

    if optimizer_step < 0:
        raise ValueError("optimizer_step 不能为负数")
    if optimizer_step < config.warmup_steps:
        return config.learning_rate * (optimizer_step + 1) / config.warmup_steps

    decay_index = optimizer_step - config.warmup_steps
    if config.cosine_decay_steps == 1:
        progress = 1.0
    else:
        progress = min(decay_index / (config.cosine_decay_steps - 1), 1.0)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return config.decay_learning_rate + (
        config.learning_rate - config.decay_learning_rate
    ) * cosine


def event_loss_weight_at_step(
    config: Stage1TrainingConfig,
    optimizer_step: int,
) -> float:
    """线性增加训练事件权重；验证始终使用固定目标权重以保持跨 epoch 可比。"""

    if optimizer_step < 0:
        raise ValueError("optimizer_step 不能为负数")
    if config.event_loss_warmup_steps == 0:
        return config.event_loss_weight
    progress = min(
        (optimizer_step + 1) / config.event_loss_warmup_steps,
        1.0,
    )
    return config.event_loss_weight * progress


class WarmupCosineScheduler:
    """以已完成 optimizer step 为状态、可精确恢复的轻量调度器。"""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        config: Stage1TrainingConfig,
        *,
        completed_steps: int = 0,
    ) -> None:
        if completed_steps < 0:
            raise ValueError("completed_steps 不能为负数")
        self.optimizer = optimizer
        self.config = config
        self.completed_steps = completed_steps
        self._set_next_learning_rate()

    def _set_next_learning_rate(self) -> None:
        learning_rate = learning_rate_at_step(self.config, self.completed_steps)
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate

    def step(self) -> None:
        self.completed_steps += 1
        self._set_next_learning_rate()

    def get_last_lr(self) -> list[float]:
        return [float(group["lr"]) for group in self.optimizer.param_groups]

    def state_dict(self) -> dict[str, int]:
        return {"completed_steps": self.completed_steps}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        if set(state_dict) != {"completed_steps"}:
            raise ValueError("Scheduler state 字段不兼容")
        completed_steps = int(state_dict["completed_steps"])
        if completed_steps < 0:
            raise ValueError("Scheduler completed_steps 不能为负数")
        self.completed_steps = completed_steps
        self._set_next_learning_rate()


@dataclass
class TrainerState:
    completed_epochs: int = 0
    optimizer_steps: int = 0
    microbatches_seen: int = 0
    examples_seen: int = 0
    best_validation_loss: float | None = None

    def __post_init__(self) -> None:
        if min(
            self.completed_epochs,
            self.optimizer_steps,
            self.microbatches_seen,
            self.examples_seen,
        ) < 0:
            raise ValueError("TrainerState 计数不能为负数")
        if self.best_validation_loss is not None and (
            not math.isfinite(self.best_validation_loss) or self.best_validation_loss < 0
        ):
            raise ValueError("best_validation_loss 必须是有限非负数或 None")

    def to_dict(self) -> dict[str, int | float | None]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrainerState:
        return cls(
            completed_epochs=int(value["completed_epochs"]),
            optimizer_steps=int(value["optimizer_steps"]),
            microbatches_seen=int(value["microbatches_seen"]),
            examples_seen=int(value["examples_seen"]),
            best_validation_loss=(
                None
                if value.get("best_validation_loss") is None
                else float(value["best_validation_loss"])
            ),
        )


@dataclass(frozen=True)
class EpochMetrics:
    loss: float
    base_loss: float
    event_loss: float
    critical_steps: int
    valid_steps: int
    batches_with_events: int
    optimizer_steps: int
    microbatches: int
    examples: int
    mean_grad_norm: float
    event_loss_weight_start: float
    event_loss_weight_end: float


@dataclass(frozen=True)
class ValidationMetrics:
    loss: float
    base_loss: float
    event_loss: float
    critical_steps: int
    valid_steps: int
    batches_with_events: int
    seeds: tuple[int, ...]
    batches_per_seed: int
    examples_per_seed: int
    improved: bool


def move_to_device(value: Any, device: torch.device) -> Any:
    """递归移动 nested Tensor，同时保留字符串和追踪元数据。"""

    if isinstance(value, torch.Tensor):
        return value.to(device=device, non_blocking=device.type == "cuda")
    if isinstance(value, dict):
        return {key: move_to_device(item, device) for key, item in value.items()}
    if isinstance(value, list):
        return [move_to_device(item, device) for item in value]
    if isinstance(value, tuple):
        return tuple(move_to_device(item, device) for item in value)
    return value


def _stage1_trainable_parameters(policy: QwenVLAPolicy) -> list[nn.Parameter]:
    frozen_ids = {id(parameter) for parameter in policy.context_encoder.parameters()}
    if any(parameter.requires_grad for parameter in policy.context_encoder.parameters()):
        raise ValueError("Stage 1 的 Qwen Context Encoder 必须完全冻结")

    trainable = [
        parameter
        for module in (policy.adapter, policy.expert)
        for parameter in module.parameters()
        if parameter.requires_grad
    ]
    if not trainable:
        raise ValueError("Adapter/Expert 没有可训练参数")
    trainable_ids = {id(parameter) for parameter in trainable}
    unexpected = [
        name
        for name, parameter in policy.named_parameters()
        if parameter.requires_grad and id(parameter) not in trainable_ids
    ]
    if unexpected or trainable_ids.intersection(frozen_ids):
        raise ValueError(f"Stage 1 发现边界外可训练参数: {unexpected}")
    return trainable


def build_stage1_optimizer(
    policy: QwenVLAPolicy,
    config: Stage1TrainingConfig,
) -> torch.optim.AdamW:
    return torch.optim.AdamW(
        _stage1_trainable_parameters(policy),
        lr=config.learning_rate,
        betas=config.betas,
        eps=config.eps,
        weight_decay=config.weight_decay,
    )


class Stage1Trainer:
    """只在 optimizer step 边界暴露可保存状态的单设备 Trainer。"""

    def __init__(
        self,
        policy: QwenVLAPolicy,
        config: Stage1TrainingConfig,
        device: str | torch.device,
        *,
        optimizer: torch.optim.Optimizer | None = None,
        scheduler: WarmupCosineScheduler | None = None,
    ) -> None:
        self.policy = policy
        self.config = config
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("请求 CUDA 训练，但当前 PyTorch 无可用 CUDA")
        if config.use_bf16 and self.device.type == "cuda" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("当前 CUDA 设备不支持 Stage 1 所需的 BF16")

        self.policy.to(self.device)
        trainable = _stage1_trainable_parameters(self.policy)
        self.optimizer = optimizer or build_stage1_optimizer(self.policy, config)
        optimizer_ids = {
            id(parameter)
            for group in self.optimizer.param_groups
            for parameter in group["params"]
        }
        if optimizer_ids != {id(parameter) for parameter in trainable}:
            raise ValueError("Optimizer 参数必须与 Stage 1 的 Adapter/Expert 可训练参数完全一致")
        self.scheduler = scheduler or WarmupCosineScheduler(self.optimizer, config)
        self.scaler = torch.amp.GradScaler(self.device.type, enabled=False)
        self.flow_generator = torch.Generator(device=self.device)
        self.flow_generator.manual_seed(config.seed)
        self.state = TrainerState()
        self.optimizer.zero_grad(set_to_none=True)

    def _autocast(self):
        return torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.config.use_bf16,
        )

    def _validate_batch(self, batch: dict[str, Any]) -> int:
        required = {
            "qwen_inputs",
            "proprio",
            "action",
            "action_mask",
            "supervision_mask",
            "event_mask",
        }
        missing = required.difference(batch)
        if missing:
            raise ValueError(f"训练 batch 缺少字段: {sorted(missing)}")
        action = batch["action"]
        action_mask = batch["action_mask"]
        supervision_mask = batch["supervision_mask"]
        event_mask = batch["event_mask"]
        if not all(
            isinstance(value, torch.Tensor)
            for value in (action, action_mask, supervision_mask, event_mask)
        ):
            raise TypeError("action 与监督相关 mask 必须为 Tensor")
        if (
            action_mask.shape != action.shape[:2]
            or supervision_mask.shape != action.shape[:2]
            or event_mask.shape != action.shape[:2]
        ):
            raise ValueError("Action 相关 mask 必须与 Action [B,H] 对齐")
        if (
            action_mask.dtype != torch.bool
            or supervision_mask.dtype != torch.bool
            or event_mask.dtype != torch.bool
        ):
            raise ValueError("Action 相关 mask 必须为 bool Tensor")
        if torch.any(action_mask & ~supervision_mask):
            raise ValueError("action_mask 不能监督 supervision_mask 外的 Action Token")
        if torch.any(event_mask & ~action_mask):
            raise ValueError("event_mask 不能标记无效 Action Token")
        if self.config.executed_action_steps > action.shape[1]:
            raise ValueError("executed_action_steps 不能超过 Action Horizon")
        temporal_fields = {
            "state_history",
            "state_history_mask",
            "controller_state",
        }
        temporal_present = temporal_fields.intersection(batch)
        if temporal_present and temporal_present != temporal_fields:
            raise ValueError("Observation V2 temporal batch 字段必须同时存在")
        expert_config = self.policy.expert.config
        temporal_policy = all(
            hasattr(expert_config, name)
            for name in (
                "history_length",
                "frame_state_dim",
                "controller_state_dim",
            )
        )
        if bool(temporal_present) != temporal_policy:
            raise ValueError("Policy 与训练 batch 的 Observation V1/V2 契约不一致")
        if temporal_present:
            state_history = batch["state_history"]
            state_history_mask = batch["state_history_mask"]
            controller_state = batch["controller_state"]
            if not all(
                isinstance(value, torch.Tensor)
                for value in (state_history, state_history_mask, controller_state)
            ):
                raise TypeError("Observation V2 temporal state 必须为 Tensor")
            if (
                state_history.ndim != 3
                or state_history_mask.shape != state_history.shape[:2]
                or state_history_mask.dtype != torch.bool
                or controller_state.ndim != 2
                or controller_state.shape[0] != state_history.shape[0]
            ):
                raise ValueError("Observation V2 temporal state shape/dtype 无效")
            expected_state = (
                int(expert_config.history_length),
                int(expert_config.frame_state_dim),
            )
            if tuple(state_history.shape[1:]) != expected_state:
                raise ValueError(
                    f"Observation V2 state_history 应为 [B,{expected_state[0]},"
                    f"{expected_state[1]}]"
                )
            if controller_state.shape[1] != int(expert_config.controller_state_dim):
                raise ValueError("Observation V2 controller_state 维度与 Policy 不一致")
        batch_size = action.shape[0]
        if batch_size <= 0 or batch_size > self.config.micro_batch_size:
            raise ValueError(
                f"物理 batch size 应位于 [1,{self.config.micro_batch_size}]，实际为 {batch_size}"
            )
        return batch_size

    def _finish_accumulation_group(
        self,
        valid_elements: int,
        microbatches: int,
        examples: int,
    ) -> float:
        if valid_elements <= 0:
            raise RuntimeError("累积组没有有效 Action 元素")
        self.scaler.unscale_(self.optimizer)
        for parameter in _stage1_trainable_parameters(self.policy):
            if parameter.grad is not None:
                parameter.grad.div_(valid_elements)
        grad_norm = nn.utils.clip_grad_norm_(
            _stage1_trainable_parameters(self.policy),
            self.config.max_grad_norm,
        )
        if not torch.isfinite(grad_norm):
            self.optimizer.zero_grad(set_to_none=True)
            raise FloatingPointError("Stage 1 gradient norm 为 NaN 或 Inf")
        self.scaler.step(self.optimizer)
        self.scaler.update()
        self.scheduler.step()
        self.optimizer.zero_grad(set_to_none=True)
        self.state.optimizer_steps += 1
        self.state.microbatches_seen += microbatches
        self.state.examples_seen += examples
        return float(grad_norm.item())

    def train_epoch(self, dataloader: Any) -> EpochMetrics:
        self.policy.train()
        total_loss_numerator = 0.0
        total_base_loss_numerator = 0.0
        total_event_loss_numerator = 0.0
        total_valid_elements = 0
        total_critical_elements = 0
        batches_with_events = 0
        group_valid_elements = 0
        group_microbatches = 0
        group_examples = 0
        optimizer_steps_before = self.state.optimizer_steps
        event_loss_weight_start = event_loss_weight_at_step(
            self.config,
            optimizer_steps_before,
        )
        event_loss_weight_end = event_loss_weight_start
        microbatches = 0
        examples = 0
        grad_norms: list[float] = []

        try:
            for raw_batch in dataloader:
                batch_size = self._validate_batch(raw_batch)
                batch = move_to_device(raw_batch, self.device)
                event_loss_weight_end = event_loss_weight_at_step(
                    self.config,
                    self.state.optimizer_steps,
                )
                with self._autocast():
                    effective_action_mask = (
                        batch["action_mask"] & batch["supervision_mask"]
                    )
                    if "state_history" in batch:
                        output = self.policy.flow_matching_loss(
                            batch["qwen_inputs"],
                            batch["state_history"],
                            batch["action"],
                            effective_action_mask,
                            state_history_mask=batch["state_history_mask"],
                            controller_state=batch["controller_state"],
                            event_mask=batch["event_mask"],
                            event_loss_weight=event_loss_weight_end,
                            executed_action_steps=self.config.executed_action_steps,
                            generator=self.flow_generator,
                        )
                    else:
                        output = self.policy.flow_matching_loss(
                            batch["qwen_inputs"],
                            batch["proprio"],
                            batch["action"],
                            effective_action_mask,
                            event_mask=batch["event_mask"],
                            event_loss_weight=event_loss_weight_end,
                            executed_action_steps=self.config.executed_action_steps,
                            generator=self.flow_generator,
                        )
                valid_elements = int(effective_action_mask.sum().item()) * int(
                    batch["action"].shape[-1]
                )
                if valid_elements <= 0:
                    raise ValueError("训练 batch 没有有效 Action 元素")
                loss_numerator = output.loss.float() * valid_elements
                self.scaler.scale(loss_numerator).backward()

                total_loss_numerator += float(loss_numerator.detach().item())
                total_base_loss_numerator += float(output.base_loss.float().item()) * (
                    valid_elements
                )
                critical_elements = int(output.critical_mask.sum().item()) * int(
                    batch["action"].shape[-1]
                )
                if critical_elements > 0:
                    batches_with_events += 1
                    total_event_loss_numerator += float(
                        output.event_loss.float().item()
                    ) * critical_elements
                    total_critical_elements += critical_elements
                total_valid_elements += valid_elements
                group_valid_elements += valid_elements
                group_microbatches += 1
                group_examples += batch_size
                microbatches += 1
                examples += batch_size

                if group_microbatches == self.config.gradient_accumulation_steps:
                    grad_norms.append(
                        self._finish_accumulation_group(
                            group_valid_elements,
                            group_microbatches,
                            group_examples,
                        )
                    )
                    group_valid_elements = 0
                    group_microbatches = 0
                    group_examples = 0

            if group_microbatches:
                grad_norms.append(
                    self._finish_accumulation_group(
                        group_valid_elements,
                        group_microbatches,
                        group_examples,
                    )
                )
        except Exception:
            self.optimizer.zero_grad(set_to_none=True)
            raise

        if microbatches == 0 or total_valid_elements == 0:
            raise ValueError("训练 dataloader 不能为空")
        self.state.completed_epochs += 1
        return EpochMetrics(
            loss=total_loss_numerator / total_valid_elements,
            base_loss=total_base_loss_numerator / total_valid_elements,
            event_loss=(
                total_event_loss_numerator / total_critical_elements
                if total_critical_elements > 0
                else 0.0
            ),
            critical_steps=total_critical_elements // int(batch["action"].shape[-1]),
            valid_steps=total_valid_elements // int(batch["action"].shape[-1]),
            batches_with_events=batches_with_events,
            optimizer_steps=self.state.optimizer_steps - optimizer_steps_before,
            microbatches=microbatches,
            examples=examples,
            mean_grad_norm=sum(grad_norms) / len(grad_norms),
            event_loss_weight_start=event_loss_weight_start,
            event_loss_weight_end=event_loss_weight_end,
        )

    @torch.no_grad()
    def validate(self, dataloader: Any) -> ValidationMetrics:
        was_training = self.policy.training
        self.policy.eval()
        total_loss_numerator = 0.0
        total_base_loss_numerator = 0.0
        total_event_loss_numerator = 0.0
        total_valid_elements = 0
        total_critical_elements = 0
        batches_with_events = 0
        batches_per_seed: int | None = None
        examples_per_seed: int | None = None
        try:
            for seed in self.config.validation_seeds:
                generator = torch.Generator(device=self.device)
                generator.manual_seed(seed)
                seed_batches = 0
                seed_examples = 0
                for raw_batch in dataloader:
                    batch_size = self._validate_batch(raw_batch)
                    batch = move_to_device(raw_batch, self.device)
                    with self._autocast():
                        effective_action_mask = (
                            batch["action_mask"] & batch["supervision_mask"]
                        )
                        if "state_history" in batch:
                            output = self.policy.flow_matching_loss(
                                batch["qwen_inputs"],
                                batch["state_history"],
                                batch["action"],
                                effective_action_mask,
                                state_history_mask=batch["state_history_mask"],
                                controller_state=batch["controller_state"],
                                event_mask=batch["event_mask"],
                                event_loss_weight=self.config.event_loss_weight,
                                executed_action_steps=self.config.executed_action_steps,
                                generator=generator,
                            )
                        else:
                            output = self.policy.flow_matching_loss(
                                batch["qwen_inputs"],
                                batch["proprio"],
                                batch["action"],
                                effective_action_mask,
                                event_mask=batch["event_mask"],
                                event_loss_weight=self.config.event_loss_weight,
                                executed_action_steps=self.config.executed_action_steps,
                                generator=generator,
                            )
                    valid_elements = int(effective_action_mask.sum().item()) * int(
                        batch["action"].shape[-1]
                    )
                    total_loss_numerator += float(output.loss.float().item()) * valid_elements
                    total_base_loss_numerator += float(
                        output.base_loss.float().item()
                    ) * valid_elements
                    critical_elements = int(output.critical_mask.sum().item()) * int(
                        batch["action"].shape[-1]
                    )
                    if critical_elements > 0:
                        batches_with_events += 1
                        total_event_loss_numerator += float(
                            output.event_loss.float().item()
                        ) * critical_elements
                        total_critical_elements += critical_elements
                    total_valid_elements += valid_elements
                    seed_batches += 1
                    seed_examples += batch_size
                if seed_batches == 0:
                    raise ValueError("验证 dataloader 不能为空")
                if batches_per_seed is None:
                    batches_per_seed = seed_batches
                    examples_per_seed = seed_examples
                elif (seed_batches, seed_examples) != (batches_per_seed, examples_per_seed):
                    raise RuntimeError("验证 dataloader 每次迭代必须返回相同数量的样本")
        finally:
            self.policy.train(was_training)

        loss = total_loss_numerator / total_valid_elements
        improved = self.state.best_validation_loss is None or loss < self.state.best_validation_loss
        if improved:
            self.state.best_validation_loss = loss
        return ValidationMetrics(
            loss=loss,
            base_loss=total_base_loss_numerator / total_valid_elements,
            event_loss=(
                total_event_loss_numerator / total_critical_elements
                if total_critical_elements > 0
                else 0.0
            ),
            critical_steps=total_critical_elements // int(batch["action"].shape[-1]),
            valid_steps=total_valid_elements // int(batch["action"].shape[-1]),
            batches_with_events=batches_with_events,
            seeds=self.config.validation_seeds,
            batches_per_seed=int(batches_per_seed),
            examples_per_seed=int(examples_per_seed),
            improved=improved,
        )
