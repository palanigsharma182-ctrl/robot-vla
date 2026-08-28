"""qwen-vla-v0.1 的轻量、可恢复和严格版本化 Checkpoint。"""

from __future__ import annotations

import os
import random
import shutil
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_vla import __version__
from robot_vla.adapters import ProprioStats
from robot_vla.contracts import (
    MODEL_ARCH,
    PROMPT_VERSION,
    QWEN_MODEL_ID,
    QWEN_REVISION,
    TRAJECTORY_SCHEMA_VERSION,
    RobotSpec,
)
from robot_vla.model.policy import QwenVLAPolicy
from robot_vla.model.qwen_processor import QwenProcessorConfig
from robot_vla.training.stage1 import (
    Stage1Trainer,
    TrainerState,
)

CHECKPOINT_FORMAT = "robot-vla-stage1-checkpoint/v1"
FINAL_QWEN_CONTEXT = "final"


@dataclass(frozen=True)
class SavedCheckpointPaths:
    latest: Path
    periodic: Path | None
    best: Path | None


def _qwen_context_hidden_state(policy: QwenVLAPolicy) -> int | str:
    layer = getattr(policy.context_encoder, "layer", None)
    return FINAL_QWEN_CONTEXT if layer is None else int(layer)


def _metadata_value(
    metadata: dict[str, Any], key: str, expected: Any
) -> Any:
    # v1 历史 Checkpoint 未显式记录层号，当且仅当目标仍是最终层时兼容读取。
    if (
        key == "qwen_context_hidden_state"
        and key not in metadata
        and expected == FINAL_QWEN_CONTEXT
    ):
        return FINAL_QWEN_CONTEXT
    return metadata.get(key)


def _capture_rng_state(flow_generator: torch.Generator) -> dict[str, Any]:
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": {
            "bit_generator": numpy_state[0],
            "keys": torch.from_numpy(numpy_state[1].copy()),
            "position": numpy_state[2],
            "has_gauss": numpy_state[3],
            "cached_gaussian": numpy_state[4],
        },
        "torch_cpu": torch.get_rng_state(),
        "torch_cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "flow_generator": flow_generator.get_state(),
    }


def _restore_rng_state(state: dict[str, Any], flow_generator: torch.Generator) -> None:
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state(
        (
            str(numpy_state["bit_generator"]),
            numpy_state["keys"].cpu().numpy(),
            int(numpy_state["position"]),
            int(numpy_state["has_gauss"]),
            float(numpy_state["cached_gaussian"]),
        )
    )
    torch.set_rng_state(state["torch_cpu"].cpu())
    cuda_states = state["torch_cuda"]
    if cuda_states:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint 包含 CUDA RNG state，但当前没有可用 CUDA")
        if len(cuda_states) != torch.cuda.device_count():
            raise RuntimeError("Checkpoint 的 CUDA RNG state 数量与当前设备数量不一致")
        torch.cuda.set_rng_state_all(cuda_states)
    flow_generator.set_state(state["flow_generator"].cpu())


def _metadata(
    policy: QwenVLAPolicy,
    trainer: Stage1Trainer,
    robot_spec: RobotSpec,
    processor_config: QwenProcessorConfig,
    proprio_stats: ProprioStats,
    code_revision: str,
) -> dict[str, Any]:
    if not code_revision.strip():
        raise ValueError("code_revision 不能为空")
    proprio_stats.validate(robot_spec)
    return {
        "format": CHECKPOINT_FORMAT,
        "model_arch": MODEL_ARCH,
        "dataset_schema": TRAJECTORY_SCHEMA_VERSION,
        "robot_spec": robot_spec.to_dict(),
        "qwen": {"model_id": QWEN_MODEL_ID, "revision": QWEN_REVISION},
        "qwen_context_hidden_state": _qwen_context_hidden_state(policy),
        "prompt_version": PROMPT_VERSION,
        "processor_config": asdict(processor_config),
        "proprio_stats": asdict(proprio_stats),
        "training_config": trainer.config.to_dict(),
        "expert_config": asdict(policy.expert.config),
        "code": {"package_version": __version__, "revision": code_revision},
    }


def build_stage1_checkpoint(
    policy: QwenVLAPolicy,
    trainer: Stage1Trainer,
    robot_spec: RobotSpec,
    processor_config: QwenProcessorConfig,
    proprio_stats: ProprioStats,
    *,
    code_revision: str,
) -> dict[str, Any]:
    if trainer.scheduler.completed_steps != trainer.state.optimizer_steps:
        raise RuntimeError("Trainer 与 Scheduler optimizer step 不一致，拒绝保存")
    return {
        "metadata": _metadata(
            policy,
            trainer,
            robot_spec,
            processor_config,
            proprio_stats,
            code_revision,
        ),
        "model": {
            "adapter": policy.adapter.state_dict(),
            "expert": policy.expert.state_dict(),
        },
        "optimizer": trainer.optimizer.state_dict(),
        "scheduler": trainer.scheduler.state_dict(),
        "scaler": trainer.scaler.state_dict(),
        "trainer": trainer.state.to_dict(),
        "rng": _capture_rng_state(trainer.flow_generator),
    }


def _atomic_torch_save(payload: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.",
            suffix=".tmp",
            dir=path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def _atomic_checkpoint_alias(source: Path, target: Path) -> None:
    """同一状态优先使用不可变硬链接；不支持时回退为原子文件复制。"""

    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{target.name}.",
            suffix=".tmp",
            dir=target.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
        temporary_path.unlink()
        try:
            os.link(source, temporary_path)
        except OSError:
            shutil.copy2(source, temporary_path)
            with temporary_path.open("rb") as handle:
                os.fsync(handle.fileno())
        os.replace(temporary_path, target)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def save_stage1_checkpoint_set(
    output_dir: str | Path,
    policy: QwenVLAPolicy,
    trainer: Stage1Trainer,
    robot_spec: RobotSpec,
    processor_config: QwenProcessorConfig,
    proprio_stats: ProprioStats,
    *,
    code_revision: str,
    is_best: bool = False,
) -> SavedCheckpointPaths:
    """保存 latest，并按配置选择 periodic/best；所有文件都原子替换。"""

    payload = build_stage1_checkpoint(
        policy,
        trainer,
        robot_spec,
        processor_config,
        proprio_stats,
        code_revision=code_revision,
    )
    directory = Path(output_dir)
    latest = directory / "latest.pt"
    _atomic_torch_save(payload, latest)

    periodic = None
    if (
        trainer.state.optimizer_steps > 0
        and trainer.state.optimizer_steps % trainer.config.checkpoint_interval_steps == 0
    ):
        periodic = directory / f"step-{trainer.state.optimizer_steps:08d}.pt"
        _atomic_checkpoint_alias(latest, periodic)

    best = None
    if is_best:
        best = directory / "best.pt"
        _atomic_checkpoint_alias(latest, best)
    return SavedCheckpointPaths(latest=latest, periodic=periodic, best=best)


def _expected_metadata(
    policy: QwenVLAPolicy,
    trainer: Stage1Trainer,
    robot_spec: RobotSpec,
    processor_config: QwenProcessorConfig,
    proprio_stats: ProprioStats,
) -> dict[str, Any]:
    metadata = _metadata(
        policy,
        trainer,
        robot_spec,
        processor_config,
        proprio_stats,
        code_revision="comparison-placeholder",
    )
    metadata.pop("code")
    return metadata


def _load_checkpoint_payload(path: str | Path) -> dict[str, Any]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    required = {"metadata", "model", "optimizer", "scheduler", "scaler", "trainer", "rng"}
    if not isinstance(payload, dict) or set(payload) != required:
        raise ValueError("Checkpoint 顶层字段不兼容")
    if not isinstance(payload["model"], dict) or set(payload["model"]) != {
        "adapter",
        "expert",
    }:
        raise ValueError("Checkpoint model state 字段不兼容")
    return payload


def _validate_inference_metadata(
    metadata: dict[str, Any],
    policy: QwenVLAPolicy,
    robot_spec: RobotSpec,
    processor_config: QwenProcessorConfig,
    proprio_stats: ProprioStats,
) -> None:
    """验证推理必需契约，不要求评估源码与训练源码具有相同哈希。"""

    proprio_stats.validate(robot_spec)
    expected = {
        "format": CHECKPOINT_FORMAT,
        "model_arch": MODEL_ARCH,
        "dataset_schema": TRAJECTORY_SCHEMA_VERSION,
        "robot_spec": robot_spec.to_dict(),
        "qwen": {"model_id": QWEN_MODEL_ID, "revision": QWEN_REVISION},
        "qwen_context_hidden_state": _qwen_context_hidden_state(policy),
        "prompt_version": PROMPT_VERSION,
        "processor_config": asdict(processor_config),
        "proprio_stats": asdict(proprio_stats),
        "expert_config": asdict(policy.expert.config),
    }
    for key, expected_value in expected.items():
        if _metadata_value(metadata, key, expected_value) != expected_value:
            raise ValueError(f"Checkpoint metadata 不兼容: {key}")
    code = metadata.get("code", {})
    if code.get("package_version") != __version__ or not str(code.get("revision", "")).strip():
        raise ValueError("Checkpoint code identity 不兼容")


def load_stage1_policy_checkpoint(
    path: str | Path,
    policy: QwenVLAPolicy,
    robot_spec: RobotSpec,
    processor_config: QwenProcessorConfig,
    proprio_stats: ProprioStats,
) -> dict[str, Any]:
    """只恢复在线推理所需权重，同时严格验证模型、Processor 和状态契约。"""

    payload = _load_checkpoint_payload(path)
    metadata = payload["metadata"]
    if not isinstance(metadata, dict):
        raise TypeError("Checkpoint metadata 必须为字典")
    _validate_inference_metadata(
        metadata,
        policy,
        robot_spec,
        processor_config,
        proprio_stats,
    )
    restored_trainer_state = TrainerState.from_dict(payload["trainer"])
    scheduler_state = payload["scheduler"]
    if not isinstance(scheduler_state, dict) or set(scheduler_state) != {"completed_steps"}:
        raise ValueError("Checkpoint Scheduler state 字段不兼容")
    if int(scheduler_state["completed_steps"]) != restored_trainer_state.optimizer_steps:
        raise ValueError("Checkpoint 的 Trainer/Scheduler step 不一致")

    policy.adapter.load_state_dict(payload["model"]["adapter"], strict=True)
    policy.expert.load_state_dict(payload["model"]["expert"], strict=True)
    return metadata


def load_stage1_checkpoint(
    path: str | Path,
    policy: QwenVLAPolicy,
    trainer: Stage1Trainer,
    robot_spec: RobotSpec,
    processor_config: QwenProcessorConfig,
    proprio_stats: ProprioStats,
    *,
    expected_code_revision: str | None = None,
    restore_rng: bool = True,
) -> dict[str, Any]:
    """先验证完整契约，再恢复模型、优化器、调度器和 RNG。"""

    payload = _load_checkpoint_payload(path)

    actual_metadata = payload["metadata"]
    expected_metadata = _expected_metadata(
        policy,
        trainer,
        robot_spec,
        processor_config,
        proprio_stats,
    )
    for key, expected in expected_metadata.items():
        if _metadata_value(actual_metadata, key, expected) != expected:
            raise ValueError(f"Checkpoint metadata 不兼容: {key}")
    code = actual_metadata.get("code", {})
    if code.get("package_version") != __version__:
        raise ValueError("Checkpoint package version 不兼容")
    if expected_code_revision is not None and code.get("revision") != expected_code_revision:
        raise ValueError("Checkpoint code revision 不兼容")

    restored_trainer_state = TrainerState.from_dict(payload["trainer"])
    scheduler_state = payload["scheduler"]
    if not isinstance(scheduler_state, dict) or set(scheduler_state) != {"completed_steps"}:
        raise ValueError("Checkpoint Scheduler state 字段不兼容")
    if int(scheduler_state["completed_steps"]) != restored_trainer_state.optimizer_steps:
        raise ValueError("Checkpoint 的 Trainer/Scheduler step 不一致")

    policy.adapter.load_state_dict(payload["model"]["adapter"], strict=True)
    policy.expert.load_state_dict(payload["model"]["expert"], strict=True)
    trainer.optimizer.load_state_dict(payload["optimizer"])
    trainer.scheduler.load_state_dict(scheduler_state)
    trainer.scaler.load_state_dict(payload["scaler"])
    trainer.state = restored_trainer_state
    if restore_rng:
        _restore_rng_state(payload["rng"], trainer.flow_generator)
    return actual_metadata
