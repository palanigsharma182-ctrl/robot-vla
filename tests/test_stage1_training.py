from types import SimpleNamespace

import pytest

torch = pytest.importorskip("torch")
from torch import nn

from robot_vla.training.stage1 import (
    Stage1Trainer,
    Stage1TrainingConfig,
    event_loss_weight_at_step,
    learning_rate_at_step,
    move_to_device,
)


class ToyExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = SimpleNamespace(action_dim=1)


class ToyPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_encoder = nn.Linear(1, 1, bias=False)
        self.context_encoder.requires_grad_(False)
        self.adapter = nn.Linear(1, 1, bias=False)
        self.expert = ToyExpert()
        nn.init.zeros_(self.adapter.weight)

    def train(self, mode: bool = True):
        super().train(mode)
        self.context_encoder.train(False)
        return self

    def flow_matching_loss(
        self,
        _model_inputs,
        normalized_proprio,
        normalized_action,
        action_mask,
        *,
        event_mask=None,
        event_loss_weight=0.0,
        executed_action_steps=4,
        generator=None,
    ):
        del event_mask, event_loss_weight, executed_action_steps, generator
        prediction = self.adapter(normalized_proprio).view(-1, 1, 1)
        target = normalized_action[..., :1]
        loss = ((prediction - target).square() * action_mask.unsqueeze(-1)).sum()
        loss = loss / action_mask.sum()
        return SimpleNamespace(
            loss=loss,
            base_loss=loss,
            event_loss=loss * 0.0,
            critical_mask=torch.zeros_like(action_mask),
        )


class RandomToyPolicy(ToyPolicy):
    def flow_matching_loss(
        self,
        _model_inputs,
        normalized_proprio,
        normalized_action,
        action_mask,
        *,
        event_mask=None,
        event_loss_weight=0.0,
        executed_action_steps=4,
        generator=None,
    ):
        del event_mask, event_loss_weight, executed_action_steps
        noise = torch.rand((), generator=generator, device=normalized_proprio.device)
        prediction = self.adapter(normalized_proprio).view(-1, 1, 1) + noise
        target = normalized_action[..., :1]
        loss = ((prediction - target).square() * action_mask.unsqueeze(-1)).sum()
        loss = loss / action_mask.sum()
        return SimpleNamespace(
            loss=loss,
            base_loss=loss,
            event_loss=loss * 0.0,
            critical_mask=torch.zeros_like(action_mask),
        )


def _toy_batch(target: float) -> dict:
    return {
        "qwen_inputs": {"input_ids": torch.ones(1, 1, dtype=torch.long)},
        "proprio": torch.ones(1, 1),
        "action": torch.tensor([[[target]]], dtype=torch.float32),
        "action_mask": torch.ones(1, 1, dtype=torch.bool),
        "event_mask": torch.zeros(1, 1, dtype=torch.bool),
        "trajectory_id": ["episode"],
    }


def test_default_learning_rate_schedule_has_fixed_warmup_and_decay_boundaries() -> None:
    config = Stage1TrainingConfig()

    assert learning_rate_at_step(config, 0) == pytest.approx(1e-7)
    assert learning_rate_at_step(config, 999) == pytest.approx(1e-4)
    assert learning_rate_at_step(config, 1_000) == pytest.approx(1e-4)
    assert learning_rate_at_step(config, 30_999) == pytest.approx(2.5e-6)
    assert learning_rate_at_step(config, 50_000) == pytest.approx(2.5e-6)


def test_event_loss_weight_warmup_is_linear_and_optional() -> None:
    immediate = Stage1TrainingConfig(event_loss_weight=2.0)
    warmup = Stage1TrainingConfig(
        event_loss_weight=1.0,
        event_loss_warmup_steps=4,
    )

    assert event_loss_weight_at_step(immediate, 0) == pytest.approx(2.0)
    assert event_loss_weight_at_step(warmup, 0) == pytest.approx(0.25)
    assert event_loss_weight_at_step(warmup, 2) == pytest.approx(0.75)
    assert event_loss_weight_at_step(warmup, 3) == pytest.approx(1.0)
    assert event_loss_weight_at_step(warmup, 100) == pytest.approx(1.0)
    with pytest.raises(ValueError, match="不能为负数"):
        event_loss_weight_at_step(warmup, -1)


def test_event_loss_weight_warmup_rejects_negative_steps() -> None:
    with pytest.raises(ValueError, match="event_loss_warmup_steps"):
        Stage1TrainingConfig(event_loss_warmup_steps=-1)


def test_recursive_move_preserves_metadata_and_container_types() -> None:
    value = {
        "tensor": torch.ones(2),
        "nested": [torch.zeros(1), ("trace", torch.tensor(3))],
    }

    moved = move_to_device(value, torch.device("cpu"))

    assert moved["tensor"].device.type == "cpu"
    assert isinstance(moved["nested"], list)
    assert isinstance(moved["nested"][1], tuple)
    assert moved["nested"][1][0] == "trace"


def test_partial_accumulation_group_is_normalized_by_its_actual_size() -> None:
    policy = ToyPolicy()
    config = Stage1TrainingConfig(
        learning_rate=0.1,
        decay_learning_rate=0.1,
        weight_decay=0.0,
        max_grad_norm=100.0,
        warmup_steps=1,
        cosine_decay_steps=10,
        gradient_accumulation_steps=2,
        use_bf16=False,
        executed_action_steps=1,
    )
    optimizer = torch.optim.SGD(policy.adapter.parameters(), lr=config.learning_rate)
    trainer = Stage1Trainer(policy, config, "cpu", optimizer=optimizer)

    metrics = trainer.train_epoch([_toy_batch(1.0), _toy_batch(3.0), _toy_batch(5.0)])

    assert policy.adapter.weight.item() == pytest.approx(1.32)
    assert metrics.optimizer_steps == 2
    assert metrics.microbatches == 3
    assert metrics.examples == 3
    assert trainer.state.optimizer_steps == 2
    assert trainer.scheduler.completed_steps == 2
    assert policy.context_encoder.training is False


def test_optimizer_rejects_parameters_outside_stage1_boundary() -> None:
    policy = ToyPolicy()
    policy.context_encoder.weight.requires_grad_(True)
    config = Stage1TrainingConfig(use_bf16=False)

    with pytest.raises(ValueError, match="必须完全冻结"):
        Stage1Trainer(policy, config, "cpu")


def test_validation_is_repeatable_and_does_not_advance_training_flow_rng() -> None:
    policy = RandomToyPolicy()
    config = Stage1TrainingConfig(
        use_bf16=False,
        validation_seeds=(7, 11),
        executed_action_steps=1,
    )
    trainer = Stage1Trainer(policy, config, "cpu")
    trainer.policy.train()
    flow_state_before = trainer.flow_generator.get_state().clone()
    dataloader = [_toy_batch(1.0), _toy_batch(2.0)]

    first = trainer.validate(dataloader)
    second = trainer.validate(dataloader)

    assert first.loss == pytest.approx(second.loss)
    assert first.improved is True
    assert second.improved is False
    assert first.batches_per_seed == 2
    assert first.examples_per_seed == 2
    assert trainer.policy.training is True
    assert torch.equal(trainer.flow_generator.get_state(), flow_state_before)
