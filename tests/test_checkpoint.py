import random

import numpy as np
import pytest

torch = pytest.importorskip("torch")
from torch import nn

from robot_vla.adapters import ProprioStats
from robot_vla.contracts import MODEL_ARCH, QWEN_REVISION, RobotSpec
from robot_vla.model.expert import ExpertConfig
from robot_vla.model.qwen_processor import QwenProcessorConfig
from robot_vla.training.checkpoint import (
    load_stage1_checkpoint,
    load_stage1_policy_checkpoint,
    save_stage1_checkpoint_set,
)
from robot_vla.training.stage1 import Stage1Trainer, Stage1TrainingConfig


class CheckpointExpert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.config = ExpertConfig()
        self.weight = nn.Parameter(torch.tensor([1.0]))


class CheckpointPolicy(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.context_encoder = nn.Linear(1, 1, bias=False)
        self.context_encoder.requires_grad_(False)
        self.adapter = nn.Linear(1, 1, bias=False)
        self.expert = CheckpointExpert()

    def train(self, mode: bool = True):
        super().train(mode)
        self.context_encoder.train(False)
        return self


def _stats(spec: RobotSpec, *, first_mean: float = 0.0) -> ProprioStats:
    mean = [0.0] * spec.proprio_dim
    mean[0] = first_mean
    return ProprioStats(
        mean=tuple(mean),
        std=(1.0,) * spec.proprio_dim,
        count=100,
        embodiment=spec.embodiment,
    )


def test_checkpoint_round_trip_restores_trainable_state_and_all_rng(tmp_path) -> None:
    spec = RobotSpec()
    processor_config = QwenProcessorConfig()
    config = Stage1TrainingConfig(use_bf16=False, checkpoint_interval_steps=10)
    policy = CheckpointPolicy()
    trainer = Stage1Trainer(policy, config, "cpu")
    with torch.no_grad():
        policy.adapter.weight.fill_(2.0)
        policy.expert.weight.fill_(3.0)

    random.seed(7)
    np.random.seed(7)
    torch.manual_seed(7)
    trainer.flow_generator.manual_seed(7)
    paths = save_stage1_checkpoint_set(
        tmp_path,
        policy,
        trainer,
        spec,
        processor_config,
        _stats(spec),
        code_revision="source-digest-001",
        is_best=True,
    )
    expected_python = random.random()
    expected_numpy = np.random.rand()
    expected_torch = torch.rand(1)
    expected_flow = torch.rand(1, generator=trainer.flow_generator)

    assert paths.latest.exists()
    assert paths.periodic is None
    assert paths.best is not None and paths.best.exists()
    assert paths.latest.stat().st_ino == paths.best.stat().st_ino

    random.seed(99)
    np.random.seed(99)
    torch.manual_seed(99)
    trainer.flow_generator.manual_seed(99)
    with torch.no_grad():
        policy.adapter.weight.zero_()
        policy.expert.weight.zero_()

    metadata = load_stage1_checkpoint(
        paths.latest,
        policy,
        trainer,
        spec,
        processor_config,
        _stats(spec),
        expected_code_revision="source-digest-001",
    )

    assert metadata["model_arch"] == MODEL_ARCH
    assert metadata["qwen"]["revision"] == QWEN_REVISION
    assert policy.adapter.weight.item() == pytest.approx(2.0)
    assert policy.expert.weight.item() == pytest.approx(3.0)
    assert random.random() == pytest.approx(expected_python)
    assert np.random.rand() == pytest.approx(expected_numpy)
    torch.testing.assert_close(torch.rand(1), expected_torch)
    torch.testing.assert_close(torch.rand(1, generator=trainer.flow_generator), expected_flow)


def test_replacing_latest_does_not_mutate_previous_best_alias(tmp_path) -> None:
    spec = RobotSpec()
    processor_config = QwenProcessorConfig()
    policy = CheckpointPolicy()
    trainer = Stage1Trainer(policy, Stage1TrainingConfig(use_bf16=False), "cpu")
    with torch.no_grad():
        policy.adapter.weight.fill_(2.0)
    first = save_stage1_checkpoint_set(
        tmp_path,
        policy,
        trainer,
        spec,
        processor_config,
        _stats(spec),
        code_revision="source-digest-001",
        is_best=True,
    )
    assert first.best is not None
    best_inode = first.best.stat().st_ino

    with torch.no_grad():
        policy.adapter.weight.fill_(9.0)
    save_stage1_checkpoint_set(
        tmp_path,
        policy,
        trainer,
        spec,
        processor_config,
        _stats(spec),
        code_revision="source-digest-001",
        is_best=False,
    )

    best_payload = torch.load(first.best, map_location="cpu", weights_only=True)
    latest_payload = torch.load(first.latest, map_location="cpu", weights_only=True)
    assert first.best.stat().st_ino == best_inode
    assert first.latest.stat().st_ino != best_inode
    assert best_payload["model"]["adapter"]["weight"].item() == pytest.approx(2.0)
    assert latest_payload["model"]["adapter"]["weight"].item() == pytest.approx(9.0)


def test_checkpoint_rejects_metadata_mismatch_before_loading_weights(tmp_path) -> None:
    spec = RobotSpec()
    processor_config = QwenProcessorConfig()
    config = Stage1TrainingConfig(use_bf16=False)
    policy = CheckpointPolicy()
    trainer = Stage1Trainer(policy, config, "cpu")
    with torch.no_grad():
        policy.adapter.weight.fill_(2.0)
    paths = save_stage1_checkpoint_set(
        tmp_path,
        policy,
        trainer,
        spec,
        processor_config,
        _stats(spec),
        code_revision="source-digest-001",
    )
    with torch.no_grad():
        policy.adapter.weight.fill_(9.0)

    with pytest.raises(ValueError, match="proprio_stats"):
        load_stage1_checkpoint(
            paths.latest,
            policy,
            trainer,
            spec,
            processor_config,
            _stats(spec, first_mean=1.0),
        )

    assert policy.adapter.weight.item() == pytest.approx(9.0)


def test_checkpoint_rejects_internal_step_mismatch_before_loading_weights(tmp_path) -> None:
    spec = RobotSpec()
    processor_config = QwenProcessorConfig()
    config = Stage1TrainingConfig(use_bf16=False)
    policy = CheckpointPolicy()
    trainer = Stage1Trainer(policy, config, "cpu")
    with torch.no_grad():
        policy.adapter.weight.fill_(2.0)
    paths = save_stage1_checkpoint_set(
        tmp_path,
        policy,
        trainer,
        spec,
        processor_config,
        _stats(spec),
        code_revision="source-digest-001",
    )
    payload = torch.load(paths.latest, map_location="cpu", weights_only=True)
    payload["scheduler"]["completed_steps"] = 1
    corrupted = tmp_path / "corrupted.pt"
    torch.save(payload, corrupted)
    with torch.no_grad():
        policy.adapter.weight.fill_(9.0)

    with pytest.raises(ValueError, match="Trainer/Scheduler"):
        load_stage1_checkpoint(
            corrupted,
            policy,
            trainer,
            spec,
            processor_config,
            _stats(spec),
        )

    assert policy.adapter.weight.item() == pytest.approx(9.0)


def test_inference_checkpoint_loads_only_policy_without_restoring_rng(tmp_path) -> None:
    spec = RobotSpec()
    processor_config = QwenProcessorConfig()
    config = Stage1TrainingConfig(use_bf16=False)
    policy = CheckpointPolicy()
    trainer = Stage1Trainer(policy, config, "cpu")
    with torch.no_grad():
        policy.adapter.weight.fill_(2.0)
        policy.expert.weight.fill_(3.0)
    paths = save_stage1_checkpoint_set(
        tmp_path,
        policy,
        trainer,
        spec,
        processor_config,
        _stats(spec),
        code_revision="source-digest-001",
    )

    torch.manual_seed(91)
    expected_next = torch.rand(1)
    torch.manual_seed(91)
    with torch.no_grad():
        policy.adapter.weight.zero_()
        policy.expert.weight.zero_()

    metadata = load_stage1_policy_checkpoint(
        paths.latest,
        policy,
        spec,
        processor_config,
        _stats(spec),
    )

    assert metadata["code"]["revision"] == "source-digest-001"
    assert policy.adapter.weight.item() == pytest.approx(2.0)
    assert policy.expert.weight.item() == pytest.approx(3.0)
    torch.testing.assert_close(torch.rand(1), expected_next)


def test_inference_checkpoint_rejects_contract_mismatch_before_loading_weights(tmp_path) -> None:
    spec = RobotSpec()
    processor_config = QwenProcessorConfig()
    policy = CheckpointPolicy()
    trainer = Stage1Trainer(policy, Stage1TrainingConfig(use_bf16=False), "cpu")
    with torch.no_grad():
        policy.adapter.weight.fill_(2.0)
    paths = save_stage1_checkpoint_set(
        tmp_path,
        policy,
        trainer,
        spec,
        processor_config,
        _stats(spec),
        code_revision="source-digest-001",
    )
    with torch.no_grad():
        policy.adapter.weight.fill_(9.0)

    with pytest.raises(ValueError, match="proprio_stats"):
        load_stage1_policy_checkpoint(
            paths.latest,
            policy,
            spec,
            processor_config,
            _stats(spec, first_mean=1.0),
        )

    assert policy.adapter.weight.item() == pytest.approx(9.0)
