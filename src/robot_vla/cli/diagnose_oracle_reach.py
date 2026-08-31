"""训练、评估并汇总 Oracle Geometry Reach A/B 诊断实验。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from robot_vla.adapters import ProprioNormalizer, ProprioStats
from robot_vla.cli.evaluate_maniskill import (
    _atomic_write_json,
    _load_audit_identity,
    _sha256_file,
)
from robot_vla.cli.train_stage1 import compute_source_revision
from robot_vla.contracts import RobotSpec
from robot_vla.data.collator import QwenVLACollator
from robot_vla.data.sampler import TaskEpisodeBalancedSampler
from robot_vla.data.trajectory import load_manifest
from robot_vla.diagnostics.oracle_reach import (
    ORACLE_GEOMETRY_FORMAT,
    ORACLE_REACH_CHECKPOINT_FORMAT,
    ORACLE_REACH_EXPERIMENT_FORMAT,
    FrankaTCPForwardKinematics,
    OracleGeometryPolicy,
    OracleGeometryRuntime,
    OracleReachCollator,
    OracleReachDataset,
    current_relative_geometry,
    find_maniskill_panda_urdf,
    oracle_case,
    parameter_state_sha256,
    validate_reach_training_budget,
)
from robot_vla.diagnostics.qwen_layer_reach import (
    QWEN_LAYER12,
    QWEN_LAYER24,
    QWEN_LAYER_REACH_CHECKPOINT_FORMAT,
    QWEN_LAYER_REACH_EXPERIMENT_FORMAT,
    FrozenQwenLayerContextEncoder,
    FrozenQwenLayerPairContextEncoder,
    QwenLayerFusionAdapter,
    QwenSemanticKeyGeometryValueAdapter,
    SemanticKeyGeometryValueActionExpert,
)
from robot_vla.model.expert import ExpertConfig, StandaloneActionExpert
from robot_vla.model.policy import QwenVLAPolicy
from robot_vla.model.qwen_context import FrozenQwenContextEncoder, QwenVLAAdapter
from robot_vla.observation import validate_se3
from robot_vla.training.stage1 import Stage1Trainer, Stage1TrainingConfig


def _common_data_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    train = subparsers.add_parser("train", help="训练 Control 或 Oracle Treatment")
    _common_data_args(train)
    train.add_argument(
        "--mode",
        choices=(
            "control",
            "oracle",
            "layer12",
            "fusion",
            "semantic_kv",
        ),
        required=True,
    )
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--urdf", type=Path)
    train.add_argument("--epochs", type=int, default=30)
    train.add_argument("--batch-size", type=int, default=64)
    train.add_argument("--samples-per-epoch", type=int, default=4096)
    train.add_argument("--validation-samples", type=int, default=1024)
    train.add_argument("--learning-rate", type=float, default=1e-4)
    train.add_argument("--warmup-steps", type=int, default=1000)
    train.add_argument("--cosine-decay-steps", type=int, default=30_000)
    train.add_argument("--seed", type=int, default=42)

    evaluate = subparsers.add_parser("evaluate", help="固定 5 seed Atomic Reach 评估")
    _common_data_args(evaluate)
    evaluate.add_argument(
        "--mode",
        choices=(
            "control",
            "oracle",
            "layer12",
            "fusion",
            "semantic_kv",
        ),
        required=True,
    )
    evaluate.add_argument("--checkpoint", type=Path, required=True)
    evaluate.add_argument("--output", type=Path, required=True)
    evaluate.add_argument("--seed-start", type=int, default=10_000)
    evaluate.add_argument("--episodes", type=int, default=5)
    evaluate.add_argument("--max-policy-steps", type=int, default=100)
    evaluate.add_argument("--sampling-seed", type=int, default=42_424)
    evaluate.add_argument("--num-flow-steps", type=int, default=10)
    evaluate.add_argument("--recency-decay", type=float, default=0.5)
    evaluate.add_argument("--max-anomaly-replans", type=int, default=3)

    check = subparsers.add_parser("fk-check", help="对照仿真 tcp_pose 验证离线 FK")
    check.add_argument("--output", type=Path, required=True)
    check.add_argument("--urdf", type=Path)
    check.add_argument("--seed-start", type=int, default=10_000)
    check.add_argument("--episodes", type=int, default=5)
    check.add_argument("--max-error-m", type=float, default=1e-5)
    check.add_argument("--max-orientation-error-rad", type=float, default=1e-5)

    compare = subparsers.add_parser("compare", help="验证 A/B 身份并生成最终判断")
    compare.add_argument("--control-train", type=Path, required=True)
    compare.add_argument("--oracle-train", type=Path, required=True)
    compare.add_argument("--control-eval", type=Path, required=True)
    compare.add_argument("--oracle-eval", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _build_policy(
    mode: str,
    *,
    seed: int,
    model_cache: Path | None,
    device: str,
) -> QwenVLAPolicy:
    """在构造 Expert 前重置 RNG，保证两组 Expert 逐参数一致。"""

    qwen: nn.Module | None = None
    if mode != "oracle":
        if model_cache is None:
            raise ValueError("Qwen 训练/评估必须提供 --model-cache")
        from robot_vla.model.factory import load_frozen_qwen_v01

        qwen = load_frozen_qwen_v01(
            cache_dir=str(model_cache),
            local_files_only=True,
            device=device,
        )
    _seed_all(seed)
    if mode == "semantic_kv":
        expert = SemanticKeyGeometryValueActionExpert(ExpertConfig())
    else:
        expert = StandaloneActionExpert(ExpertConfig())
    if mode != "oracle":
        assert qwen is not None
        if mode == "control":
            return QwenVLAPolicy(
                FrozenQwenContextEncoder(qwen), expert, QwenVLAAdapter()
            )
        if mode == "layer12":
            return QwenVLAPolicy(
                FrozenQwenLayerContextEncoder(qwen, QWEN_LAYER12),
                expert,
                QwenVLAAdapter(),
            )
        if mode == "fusion":
            return QwenVLAPolicy(
                FrozenQwenLayerPairContextEncoder(qwen),
                expert,
                QwenLayerFusionAdapter(),
            )
        if mode == "semantic_kv":
            if not isinstance(expert, SemanticKeyGeometryValueActionExpert):
                raise TypeError("Semantic KV mode 构造了错误 Expert 类型")
            return QwenVLAPolicy(
                FrozenQwenLayerPairContextEncoder(qwen),
                expert,
                QwenSemanticKeyGeometryValueAdapter(),
            )
        raise ValueError(f"未知 Qwen Reach mode: {mode}")
    return OracleGeometryPolicy(expert)


def _experiment_format(mode: str) -> str:
    return (
        QWEN_LAYER_REACH_EXPERIMENT_FORMAT
        if mode in {"layer12", "fusion", "semantic_kv"}
        else ORACLE_REACH_EXPERIMENT_FORMAT
    )


def _checkpoint_format(mode: str) -> str:
    return (
        QWEN_LAYER_REACH_CHECKPOINT_FORMAT
        if mode in {"layer12", "fusion", "semantic_kv"}
        else ORACLE_REACH_CHECKPOINT_FORMAT
    )


def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_torch_save(value: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
        torch.save(value, temporary)
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def _replace_hardlink(source: Path, target: Path) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.parent / f".{target.name}.{os.getpid()}.tmp"
    try:
        if temporary.exists():
            temporary.unlink()
        os.link(source, temporary)
        os.replace(temporary, target)
    finally:
        if temporary.exists():
            temporary.unlink()


def _append_jsonl(path: Path, value: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(value, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _checkpoint_payload(
    *,
    mode: str,
    policy: QwenVLAPolicy,
    trainer: Stage1Trainer,
    experiment: dict[str, Any],
) -> dict[str, Any]:
    return {
        "format": _checkpoint_format(mode),
        "mode": mode,
        "ab_identity_sha256": experiment["ab_identity_sha256"],
        "code_revision": experiment["code_revision"],
        "dataset_sha256": experiment["dataset"]["dataset_sha256"],
        "train_window_sha256": experiment["windows"]["train"]["sha256"],
        "validation_window_sha256": experiment["windows"]["validation"]["sha256"],
        "expert_initialization_sha256": experiment["expert_initialization_sha256"],
        "expert_config": asdict(policy.expert.config),
        "training_config": trainer.config.to_dict(),
        "trainer_state": trainer.state.to_dict(),
        "model": {
            "context_adapter": policy.adapter.state_dict(),
            "expert": policy.expert.state_dict(),
        },
    }


def _load_policy_checkpoint(
    path: Path,
    mode: str,
    policy: QwenVLAPolicy,
) -> dict[str, Any]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("format") != _checkpoint_format(mode):
        raise ValueError("Reach diagnostic checkpoint format 不兼容")
    if payload.get("mode") != mode:
        raise ValueError("Oracle Reach checkpoint mode 不兼容")
    if payload.get("expert_config") != asdict(policy.expert.config):
        raise ValueError("Oracle Reach checkpoint Expert contract 不兼容")
    model = payload.get("model")
    if not isinstance(model, dict) or set(model) != {"context_adapter", "expert"}:
        raise ValueError("Oracle Reach checkpoint model 字段不兼容")
    policy.adapter.load_state_dict(model["context_adapter"], strict=True)
    policy.expert.load_state_dict(model["expert"], strict=True)
    return {key: value for key, value in payload.items() if key != "model"}


def _dataset_pair(
    data: Path,
    spec: RobotSpec,
    normalizer: ProprioNormalizer,
    urdf: Path,
) -> tuple[OracleReachDataset, OracleReachDataset]:
    fk = FrankaTCPForwardKinematics(urdf, spec)
    train_entries = load_manifest(data, split="train")
    validation_entries = load_manifest(data, split="val")
    train_dataset = OracleReachDataset(
        str(data),
        train_entries,
        spec,
        normalizer,
        fk,
        cache_size=len(train_entries),
    )
    validation_dataset = OracleReachDataset(
        str(data),
        validation_entries,
        spec,
        normalizer,
        fk,
        cache_size=len(validation_entries),
    )
    return train_dataset, validation_dataset


def _run_train(args: argparse.Namespace) -> None:
    mode: str = args.mode
    validate_reach_training_budget(
        epochs=args.epochs,
        samples_per_epoch=args.samples_per_epoch,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        event_loss_weight=0.0,
    )
    if args.validation_samples <= 0 or args.seed < 0:
        raise ValueError("validation_samples 必须为正数且 seed 不能为负数")
    if not torch.cuda.is_available():
        raise RuntimeError("Oracle Reach 正式训练需要 CUDA")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Oracle Reach 训练输出目录非空，拒绝覆盖")
    args.output.mkdir(parents=True, exist_ok=True)
    _seed_all(args.seed)

    spec = RobotSpec()
    stats = ProprioStats.from_json(args.data / "proprio_stats.json")
    stats.validate(spec)
    normalizer = ProprioNormalizer(stats, spec)
    urdf = args.urdf or find_maniskill_panda_urdf()
    train_dataset, validation_dataset = _dataset_pair(
        args.data, spec, normalizer, urdf
    )
    processor = None
    qwen_collator = None
    if mode != "oracle":
        if args.model_cache is None:
            raise ValueError("Qwen 训练必须提供 --model-cache")
        from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter

        processor = QwenVLAProcessorAdapter.from_pretrained(
            cache_dir=str(args.model_cache),
            local_files_only=True,
        )
        qwen_collator = QwenVLACollator(processor, spec)
    collator = OracleReachCollator(
        "oracle" if mode == "oracle" else "control", spec, qwen_collator
    )
    train_sampler = TaskEpisodeBalancedSampler(
        train_dataset,
        num_samples=args.samples_per_epoch,
        seed=args.seed,
    )
    validation_sampler = TaskEpisodeBalancedSampler(
        validation_dataset,
        num_samples=args.validation_samples,
        seed=1_009,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        sampler=train_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=args.batch_size,
        sampler=validation_sampler,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    config = Stage1TrainingConfig(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        cosine_decay_steps=args.cosine_decay_steps,
        micro_batch_size=args.batch_size,
        gradient_accumulation_steps=1,
        seed=args.seed,
        checkpoint_interval_steps=max(1, math.ceil(args.samples_per_epoch / args.batch_size)),
        samples_per_epoch=args.samples_per_epoch,
        skill_sampling_weights=(),
        event_loss_weight=0.0,
        event_loss_warmup_steps=0,
        executed_action_steps=spec.execute_steps,
    )
    policy = _build_policy(
        mode,
        seed=args.seed,
        model_cache=args.model_cache,
        device="cuda",
    )
    initial_expert_sha256 = parameter_state_sha256(policy.expert)
    trainer = Stage1Trainer(policy, config, "cuda")
    project_root = Path(__file__).resolve().parents[3]
    code_revision = compute_source_revision(project_root)
    invariants = {
        "train_windows": {
            "count": len(train_dataset),
            "sha256": train_dataset.window_sha256,
        },
        "validation_windows": {
            "count": len(validation_dataset),
            "sha256": validation_dataset.window_sha256,
        },
        "training_config": config.to_dict(),
        "expert_config": asdict(policy.expert.config),
        "robot_spec": spec.to_dict(),
        "expert_initialization_sha256": initial_expert_sha256,
    }
    if mode in {"layer12", "fusion", "semantic_kv"}:
        invariants["context_experiment"] = QWEN_LAYER_REACH_EXPERIMENT_FORMAT
    else:
        # 保持已经完成的 Oracle/Control A/B identity contract 不变。
        invariants["geometry_format"] = ORACLE_GEOMETRY_FORMAT
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    experiment = {
        "format": _experiment_format(mode),
        "mode": mode,
        "arguments": arguments,
        "dataset": _load_audit_identity(args.data),
        "windows": {
            "train": {
                "count": len(train_dataset),
                "trajectory_count": len(train_dataset.entries),
                "sha256": train_dataset.window_sha256,
            },
            "validation": {
                "count": len(validation_dataset),
                "trajectory_count": len(validation_dataset.entries),
                "sha256": validation_dataset.window_sha256,
            },
        },
        "training_config": config.to_dict(),
        "expert_config": asdict(policy.expert.config),
        "expert_initialization_sha256": initial_expert_sha256,
        "ab_invariants": invariants,
        "ab_identity_sha256": _json_sha256(invariants),
        "code_revision": code_revision,
        "processor_config": None if processor is None else asdict(processor.config),
        "trainable_parameters": sum(
            parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
        ),
        "gpu": torch.cuda.get_device_name(0),
    }
    if mode == "layer12":
        experiment["context"] = {
            "source": "frozen_qwen_text_hidden_state",
            "layer": QWEN_LAYER12,
            "hidden_state_index": QWEN_LAYER12,
            "hidden_state_zero_is_embedding": True,
            "adapter": "QwenVLAAdapter 2048->720",
            "future_information": False,
        }
    elif mode == "fusion":
        if not isinstance(policy.adapter, QwenLayerFusionAdapter):
            raise TypeError("Fusion mode 构造了错误 Adapter 类型")
        experiment["context"] = {
            "source": "frozen_qwen_text_hidden_states",
            "layers": [QWEN_LAYER12, QWEN_LAYER24],
            "hidden_state_indices": [QWEN_LAYER12, QWEN_LAYER24],
            "hidden_state_zero_is_embedding": True,
            "projections": "independent QwenVLAAdapter 2048->720 per layer",
            "fusion": "softmax(global_trainable_scalar_logits) weighted sum",
            "initial_normalized_weights": (
                policy.adapter.normalized_weights().detach().cpu().tolist()
            ),
            "future_information": False,
        }
    elif mode == "semantic_kv":
        if not isinstance(policy.adapter, QwenSemanticKeyGeometryValueAdapter):
            raise TypeError("Semantic KV mode 构造了错误 Adapter 类型")
        if not isinstance(policy.expert, SemanticKeyGeometryValueActionExpert):
            raise TypeError("Semantic KV mode 构造了错误 Expert 类型")
        experiment["context"] = {
            "source": "frozen_qwen_text_hidden_states",
            "key_layer": QWEN_LAYER24,
            "value_layer": QWEN_LAYER12,
            "hidden_state_zero_is_embedding": True,
            "token_alignment": "same Qwen forward, same sequence index and mask",
            "projections": "independent QwenVLAAdapter 2048->720 for Key and Value",
            "attention": (
                "Action/proprio hidden Query attends Layer24 semantic Key and reads "
                "same-position Layer12 geometry Value"
            ),
            "kv_cache": "project once per action chunk and reuse for all flow steps",
            "future_information": False,
        }
    else:
        # 保留原 Oracle/Control 诊断产物 schema，避免追随实验影响旧路径。
        experiment["geometry"] = {
            "format": ORACLE_GEOMETRY_FORMAT,
            "source": "current object_position_m - current q7 SAPIEN/Pinocchio FK TCP",
            "urdf": str(urdf),
            "tcp_link": "panda_hand_tcp",
            "world_base_position_m": [-0.615, 0.0, 0.0],
            "future_information": False,
        }
    _atomic_write_json(args.output / "experiment.json", experiment)
    metrics_path = args.output / "metrics.jsonl"
    initial_validation = trainer.validate(validation_loader)
    _append_jsonl(
        metrics_path,
        {"event": "initial_validation", **asdict(initial_validation)},
    )
    trainer.state.best_validation_loss = None

    best_epoch: int | None = None
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        train_metrics = trainer.train_epoch(train_loader)
        validation_metrics = trainer.validate(validation_loader)
        payload = {
            "event": "epoch",
            "epoch": epoch + 1,
            "train": asdict(train_metrics),
            "validation": asdict(validation_metrics),
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        _append_jsonl(metrics_path, payload)
        checkpoint = _checkpoint_payload(
            mode=mode,
            policy=policy,
            trainer=trainer,
            experiment=experiment,
        )
        latest_path = args.output / "checkpoints" / "latest.pt"
        _atomic_torch_save(checkpoint, latest_path)
        if validation_metrics.improved:
            best_epoch = epoch + 1
            _replace_hardlink(latest_path, args.output / "checkpoints" / "best.pt")
        print(json.dumps(payload, sort_keys=True), flush=True)
    if best_epoch is None:
        raise RuntimeError("Oracle Reach 训练没有产生 best checkpoint")
    summary = {
        "complete": True,
        "mode": mode,
        "epochs": args.epochs,
        "best_epoch": best_epoch,
        "best_validation_loss": trainer.state.best_validation_loss,
        "initial_validation_loss": initial_validation.loss,
        "final_train_loss": train_metrics.loss,
        "final_validation_loss": validation_metrics.loss,
        "optimizer_steps": trainer.state.optimizer_steps,
        "examples_seen": trainer.state.examples_seen,
        "ab_identity_sha256": experiment["ab_identity_sha256"],
        "best_checkpoint_sha256": _sha256_file(args.output / "checkpoints" / "best.pt"),
    }
    if mode == "fusion":
        if not isinstance(policy.adapter, QwenLayerFusionAdapter):
            raise TypeError("Fusion mode 构造了错误 Adapter 类型")
        summary["final_normalized_fusion_weights"] = (
            policy.adapter.normalized_weights().detach().cpu().tolist()
        )
        best_payload = torch.load(
            args.output / "checkpoints" / "best.pt", map_location="cpu", weights_only=True
        )
        best_logits = best_payload["model"]["context_adapter"]["fusion_logits"]
        summary["best_normalized_fusion_weights"] = (
            torch.softmax(best_logits.float(), dim=0).tolist()
        )
    _atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def _run_evaluate(args: argparse.Namespace) -> None:
    from robot_vla.data.trajectory import load_manifest
    from robot_vla.diagnostics.oracle_reach_evaluation import (
        run_reach_diagnostic_episode,
        summarize_reach_diagnostics,
    )
    from robot_vla.evaluation.atomic import derive_atomic_sampling_seed
    from robot_vla.runtime.policy_runtime import QwenVLARuntime, RuntimeConfig
    from robot_vla.sim.collector import TrustedPickPlaceCollector
    from robot_vla.tasks.pick_place import build_pick_place_task

    mode: str = args.mode
    if args.episodes != 5:
        raise ValueError("首版 Oracle Reach 协议固定为 5 个 Episode")
    if (
        args.seed_start != 10_000
        or args.max_policy_steps != 100
        or args.sampling_seed < 0
        or args.num_flow_steps <= 0
        or not 0.0 < args.recency_decay < 1.0
        or args.max_anomaly_replans < 0
    ):
        raise ValueError("Oracle Reach 固定 seed/步数或 Runtime 参数无效")
    if not torch.cuda.is_available():
        raise RuntimeError("Oracle Reach 正式评估需要 CUDA")
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"找不到 Oracle Reach checkpoint: {args.checkpoint}")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError("Oracle Reach 评估输出目录非空，拒绝覆盖")
    args.output.mkdir(parents=True, exist_ok=True)
    raw_checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if raw_checkpoint.get("format") != _checkpoint_format(mode):
        raise ValueError("Reach diagnostic checkpoint format 不兼容")
    training_seed = int(raw_checkpoint["training_config"]["seed"])
    spec = RobotSpec()
    stats = ProprioStats.from_json(args.data / "proprio_stats.json")
    stats.validate(spec)
    normalizer = ProprioNormalizer(stats, spec)
    processor = None
    if mode != "oracle":
        if args.model_cache is None:
            raise ValueError("Qwen 评估必须提供 --model-cache")
        from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter

        processor = QwenVLAProcessorAdapter.from_pretrained(
            cache_dir=str(args.model_cache), local_files_only=True
        )
    policy = _build_policy(
        mode,
        seed=training_seed,
        model_cache=args.model_cache,
        device="cuda",
    )
    checkpoint_metadata = _load_policy_checkpoint(args.checkpoint, mode, policy)
    policy.to("cuda").eval()
    dataset_seeds = {
        int(entry.randomization["seed"]) for entry in load_manifest(args.data)
    }
    seeds = list(range(args.seed_start, args.seed_start + args.episodes))
    if any(seed in dataset_seeds for seed in seeds):
        raise ValueError("Oracle Reach 评估 seed 与训练数据重叠")
    project_root = Path(__file__).resolve().parents[3]
    experiment = {
        "format": _experiment_format(mode),
        "mode": mode,
        "phase": "atomic_reach_evaluation",
        "dataset": _load_audit_identity(args.data),
        "checkpoint": {
            "path": str(args.checkpoint),
            "sha256": _sha256_file(args.checkpoint),
            "metadata": checkpoint_metadata,
        },
        "config": {
            "seeds": seeds,
            "skill": "reach",
            "initial_completed_skill_count": 0,
            "max_policy_steps": 100,
            "success_threshold_m": 0.04,
            "sampling_seed": args.sampling_seed,
            "num_flow_steps": args.num_flow_steps,
            "temporal_ensemble_enabled": True,
            "recency_decay": args.recency_decay,
            "max_anomaly_replans": args.max_anomaly_replans,
            "controller": "pd_joint_delta_pos + existing ActionAdapter/safety",
        },
        "code_revision": compute_source_revision(project_root),
    }
    _atomic_write_json(args.output / "experiment.json", experiment)
    episodes_path = args.output / "episodes.jsonl"
    results = []
    with TrustedPickPlaceCollector(None, spec) as preparer:
        for seed in seeds:
            instruction = build_pick_place_task(seed % 3).instruction
            preparation = preparer.prepare_atomic(seed=seed, skill_name="reach")
            sampling_seed_base = derive_atomic_sampling_seed(
                args.sampling_seed, seed, "reach"
            )
            runtime_config = RuntimeConfig(
                num_flow_steps=args.num_flow_steps,
                use_bf16=True,
                sampling_seed=sampling_seed_base,
            )
            if mode != "oracle":
                assert processor is not None
                runtime = QwenVLARuntime(
                    policy,
                    processor,
                    normalizer,
                    spec,
                    "cuda",
                    runtime_config,
                )
            else:
                if not isinstance(policy, OracleGeometryPolicy):
                    raise TypeError("Oracle mode 构造了错误 Policy 类型")
                runtime = OracleGeometryRuntime(
                    policy,
                    normalizer,
                    spec,
                    "cuda",
                    lambda: current_relative_geometry(preparer.base_env),
                    runtime_config,
                )
            result = run_reach_diagnostic_episode(
                preparer.env,
                runtime,
                spec,
                seed=seed,
                instruction=instruction,
                sampling_seed_base=sampling_seed_base,
                preparation=preparation,
                max_policy_steps=args.max_policy_steps,
                temporal_ensemble_enabled=True,
                recency_decay=args.recency_decay,
                max_anomaly_replans=args.max_anomaly_replans,
            )
            results.append(result)
            _append_jsonl(episodes_path, result.to_dict())
            print(json.dumps(result.to_dict(), sort_keys=True), flush=True)
    summary = summarize_reach_diagnostics(results)
    summary.update(
        {
            "complete": True,
            "mode": mode,
            "ab_identity_sha256": checkpoint_metadata["ab_identity_sha256"],
            "checkpoint_sha256": experiment["checkpoint"]["sha256"],
        }
    )
    if mode == "fusion":
        if not isinstance(policy.adapter, QwenLayerFusionAdapter):
            raise TypeError("Fusion mode 构造了错误 Adapter 类型")
        summary["normalized_fusion_weights"] = (
            policy.adapter.normalized_weights().detach().cpu().tolist()
        )
    if mode == "oracle":
        summary["case"] = oracle_case(summary["successes"], summary["episodes"])
    _atomic_write_json(args.output / "summary.json", summary)
    print(json.dumps(summary, sort_keys=True), flush=True)


def _run_fk_check(args: argparse.Namespace) -> None:
    if (
        args.seed_start < 0
        or args.episodes <= 0
        or not math.isfinite(args.max_error_m)
        or args.max_error_m <= 0
        or not math.isfinite(args.max_orientation_error_rad)
        or args.max_orientation_error_rad <= 0
    ):
        raise ValueError("FK check 参数无效")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    import gymnasium as gym

    from robot_vla.sim import PICK_CUBE_TO_REGION_ENV_ID, register_robot_vla_maniskill_envs

    register_robot_vla_maniskill_envs()
    spec = RobotSpec()
    urdf = args.urdf or find_maniskill_panda_urdf()
    fk = FrankaTCPForwardKinematics(urdf, spec)
    env = gym.make(
        PICK_CUBE_TO_REGION_ENV_ID,
        obs_mode="rgb",
        control_mode="pd_joint_delta_pos",
        num_envs=1,
    )
    rows = []
    try:
        for seed in range(args.seed_start, args.seed_start + args.episodes):
            env.reset(seed=seed)
            base = env.unwrapped
            q = base.agent.robot.get_qpos()[0].detach().cpu().numpy()[: spec.arm_dof]
            predicted_pose = fk.pose_world(q).astype(np.float64)
            actual_pose = (
                base.agent.tcp_pose.to_transformation_matrix()[0]
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64)
            )
            validate_se3(actual_pose, "sim_world_from_tcp")
            position_error_m = float(
                np.linalg.norm(predicted_pose[:3, 3] - actual_pose[:3, 3])
            )
            relative_rotation = predicted_pose[:3, :3].T @ actual_pose[:3, :3]
            cosine = float(np.clip((np.trace(relative_rotation) - 1.0) * 0.5, -1.0, 1.0))
            orientation_error_rad = float(math.acos(cosine))
            rows.append(
                {
                    "seed": seed,
                    "fk_tcp_pose_world": predicted_pose.tolist(),
                    "sim_tcp_pose_world": actual_pose.tolist(),
                    "position_error_m": position_error_m,
                    "orientation_error_rad": orientation_error_rad,
                }
            )
    finally:
        env.close()
    maximum_position = max(row["position_error_m"] for row in rows)
    maximum_orientation = max(row["orientation_error_rad"] for row in rows)
    result = {
        "complete": (
            maximum_position <= args.max_error_m
            and maximum_orientation <= args.max_orientation_error_rad
        ),
        "urdf": str(urdf),
        "tcp_link": "panda_hand_tcp",
        "threshold_m": args.max_error_m,
        "orientation_threshold_rad": args.max_orientation_error_rad,
        "max_position_error_m": maximum_position,
        "max_orientation_error_rad": maximum_orientation,
        "episodes": rows,
    }
    _atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)
    if not result["complete"]:
        raise RuntimeError(
            "Franka FK 与仿真 TCP pose 超出阈值："
            f"position={maximum_position:.9g} m, "
            f"orientation={maximum_orientation:.9g} rad"
        )


def _read_json(path: Path, expected_name: str) -> dict[str, Any]:
    if path.is_dir():
        path = path / expected_name
    if not path.is_file():
        raise FileNotFoundError(path)
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{path} 必须是 JSON object")
    return value


def _run_compare(args: argparse.Namespace) -> None:
    args.output.parent.mkdir(parents=True, exist_ok=True)
    control_train = _read_json(args.control_train, "experiment.json")
    oracle_train = _read_json(args.oracle_train, "experiment.json")
    control_eval = _read_json(args.control_eval, "summary.json")
    oracle_eval = _read_json(args.oracle_eval, "summary.json")
    if control_train.get("mode") != "control" or oracle_train.get("mode") != "oracle":
        raise ValueError("A/B train experiment mode 错误")
    if control_eval.get("mode") != "control" or oracle_eval.get("mode") != "oracle":
        raise ValueError("A/B evaluation summary mode 错误")
    identities = {
        control_train.get("ab_identity_sha256"),
        oracle_train.get("ab_identity_sha256"),
        control_eval.get("ab_identity_sha256"),
        oracle_eval.get("ab_identity_sha256"),
    }
    if len(identities) != 1 or None in identities:
        raise ValueError("Control/Treatment 数据、Expert 初始化或训练预算不一致")
    if control_train.get("expert_initialization_sha256") != oracle_train.get(
        "expert_initialization_sha256"
    ):
        raise ValueError("Control/Treatment Expert 初始化哈希不一致")
    case = oracle_case(int(oracle_eval["successes"]), int(oracle_eval["episodes"]))
    result = {
        "complete": True,
        "ab_identity_sha256": identities.pop(),
        "expert_initialization_sha256": control_train[
            "expert_initialization_sha256"
        ],
        "control": control_eval,
        "oracle": oracle_eval,
        "case": case,
        "interpretation": {
            "case_1": "Oracle >=4/5：强烈支持 Frozen Qwen 精细几何表示是主要瓶颈",
            "case_2": "Oracle <=1/5：主要转向 geometry+proprio 到 joint action/隐式 IK",
            "case_3": "Oracle 2/5 或 3/5：视觉表征与 joint-space 动作学习两侧都有问题",
        }[case],
    }
    _atomic_write_json(args.output, result)
    print(json.dumps(result, sort_keys=True), flush=True)


def run(args: argparse.Namespace) -> None:
    if args.command == "train":
        _run_train(args)
    elif args.command == "evaluate":
        _run_evaluate(args)
    elif args.command == "fk-check":
        _run_fk_check(args)
    elif args.command == "compare":
        _run_compare(args)
    else:
        raise ValueError(f"未知命令: {args.command}")


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
