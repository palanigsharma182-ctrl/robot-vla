"""单卡 qwen-vla-v0.1 Stage 1 训练、显存测量和小数据过拟合入口。"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import stat
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from robot_vla.adapters import (
    FingerForceNormalizer,
    FingerForceStats,
    ProprioNormalizer,
    ProprioStats,
)
from robot_vla.contracts import (
    PICK_AND_PLACE_SKILLS,
    PROMPT_VERSION,
    PROMPT_VERSION_OBSERVATION_V2,
    RobotSpec,
)
from robot_vla.data.collator import QwenVLACollator, QwenVLAObservationV2Collator
from robot_vla.data.dataset import (
    ActionChunkDataset,
    CompositeActionChunkDataset,
    ObservationV2ActionChunkDataset,
)
from robot_vla.data.sampler import TaskEpisodeBalancedSampler
from robot_vla.data.trajectory import load_manifest
from robot_vla.model.factory import (
    load_qwen_vla_observation_v2_policy,
    load_qwen_vla_policy,
)
from robot_vla.model.qwen_processor import QwenProcessorConfig, QwenVLAProcessorAdapter
from robot_vla.training.checkpoint import (
    initialize_stage1_policy_checkpoint,
    load_stage1_checkpoint,
    save_stage1_checkpoint_set,
)
from robot_vla.training.stage1 import Stage1Trainer, Stage1TrainingConfig

DEFAULT_SKILL_WEIGHTS = (1.0, 3.0, 1.5, 1.0, 1.5)
E012_TRAINING_SOURCES = {
    "base_d0",
    "dagger_reach_grasp",
    "dagger_grasp_lift",
}


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument(
        "--dagger-data",
        type=Path,
        help="只含 Local DAgger train trajectory 的已审计 additions 数据根",
    )
    parser.add_argument("--model-cache", type=Path, required=True)
    parser.add_argument(
        "--observation-v2",
        action="store_true",
        help="启用 TCP/相机位姿、四步双图、F_L/F_R 与 controller state；拒绝缺失字段的旧数据",
    )
    parser.add_argument(
        "--qwen-context-layer",
        type=int,
        choices=(12, 24),
        default=24,
        help="Action Expert 使用的 Qwen hidden state 层；24 保持历史最终层行为",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--gradient-accumulation-steps", type=int, default=8)
    parser.add_argument("--samples-per-epoch", type=int, default=512)
    parser.add_argument("--validation-samples", type=int, default=64)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument(
        "--event-loss-weight",
        type=float,
        default=2.0,
        help="前4个实际执行步中关键事件的额外 loss 权重；0 恢复原始目标",
    )
    parser.add_argument(
        "--event-loss-warmup-steps",
        type=int,
        default=0,
        help="训练事件权重从接近 0 线性增加到目标值的 optimizer steps；0 表示立即使用目标值",
    )
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--cosine-decay-steps", type=int, default=10_000)
    parser.add_argument("--checkpoint-interval-steps", type=int, default=500)
    parser.add_argument("--checkpoint-every-epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--skill-weights",
        type=float,
        nargs=len(PICK_AND_PLACE_SKILLS),
        default=DEFAULT_SKILL_WEIGHTS,
        metavar=tuple(name.upper() for name in PICK_AND_PLACE_SKILLS),
        help="按 reach grasp lift transport place 顺序设置 Episode 内阶段采样权重",
    )
    parser.add_argument(
        "--source-weight",
        action="append",
        default=[],
        metavar="SOURCE=WEIGHT",
        help="启用 source-first 确定性配额；可重复，例如 base_d0=0.8",
    )
    parser.add_argument(
        "--proprio-stats-data",
        type=Path,
        help="显式指定冻结 ProprioStats 的数据根；E012 D1 必须指向 D0",
    )
    parser.add_argument(
        "--overfit-samples",
        type=int,
        default=0,
        help="大于 0 时固定选择少量样本，并在相同样本上验证过拟合",
    )
    parser.add_argument("--measure-only", action="store_true")
    checkpoint_group = parser.add_mutually_exclusive_group()
    checkpoint_group.add_argument("--resume", type=Path)
    checkpoint_group.add_argument(
        "--init-checkpoint",
        type=Path,
        help="只初始化 Adapter/Expert；optimizer/scheduler/scaler/trainer/RNG 全部重置",
    )
    return parser.parse_args()


def compute_source_revision(project_root: Path) -> str:
    """无 Git 工作区使用源文件内容生成真实、稳定的代码 revision。"""

    paths = [project_root / "pyproject.toml"]
    paths.extend(sorted((project_root / "src").rglob("*.py")))
    digest = hashlib.sha256()
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"代码 revision 缺少文件: {path}")
        relative = path.relative_to(project_root).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(4, "big"))
        digest.update(relative)
        content = path.read_bytes()
        digest.update(len(content).to_bytes(8, "big"))
        digest.update(content)
    return f"source-tree-sha256:{digest.hexdigest()}"


def _fixed_overfit_indices(dataset: ActionChunkDataset, count: int) -> list[int]:
    if count <= 0:
        raise ValueError("overfit sample count 必须为正数")
    by_skill: dict[int, int] = {}
    fallback: list[int] = []
    for sample_index, (entry_index, timestep) in enumerate(dataset.index):
        arrays = dataset.store.get(dataset.entries[entry_index])
        skill_id = int(arrays.skill_id[timestep])
        by_skill.setdefault(skill_id, sample_index)
        fallback.append(sample_index)
        if len(by_skill) >= count:
            break
    selected = [by_skill[key] for key in sorted(by_skill)][:count]
    if len(selected) < count:
        selected.extend(index for index in fallback if index not in selected)
    if len(selected) < count:
        raise ValueError(f"Dataset 只有 {len(selected)} 个可用于过拟合的样本")
    return selected[:count]


def _resolve_skill_sampling_weights(
    values: list[float] | tuple[float, ...],
) -> tuple[tuple[int, float], ...]:
    if len(values) != len(PICK_AND_PLACE_SKILLS):
        raise ValueError(f"skill_weights 必须恰好包含 {len(PICK_AND_PLACE_SKILLS)} 个值")
    resolved = tuple(float(value) for value in values)
    if any(not math.isfinite(value) or value <= 0 for value in resolved):
        raise ValueError("skill_weights 必须全部是有限正数")
    return tuple(enumerate(resolved))


def _resolve_source_sampling_weights(
    values: list[str] | tuple[str, ...],
) -> tuple[tuple[str, float], ...]:
    resolved: list[tuple[str, float]] = []
    seen: set[str] = set()
    for raw in values:
        source, separator, raw_weight = str(raw).partition("=")
        if separator != "=" or not source or not raw_weight:
            raise ValueError("source-weight 必须使用 SOURCE=WEIGHT 格式")
        if source not in E012_TRAINING_SOURCES:
            raise ValueError(f"source-weight 包含未知 source: {source}")
        if source in seen:
            raise ValueError(f"source-weight 重复定义 source: {source}")
        try:
            weight = float(raw_weight)
        except ValueError as exc:
            raise ValueError(f"source-weight 不是数值: {raw}") from exc
        if not math.isfinite(weight) or weight <= 0:
            raise ValueError("source-weight 必须是有限正数")
        seen.add(source)
        resolved.append((source, weight))
    return tuple(resolved)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_audit_identity(data_root: Path) -> dict[str, object]:
    path = data_root / "audit_report.json"
    if not path.is_file():
        raise FileNotFoundError("正式训练前必须存在 audit_report.json")
    report = json.loads(path.read_text(encoding="utf-8"))
    if float(report.get("success_rate", 0.0)) != 1.0:
        raise ValueError("正式训练只接受 success_rate=1.0 的审计数据")
    manifest_path = data_root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError("审计数据缺少 manifest.jsonl")
    actual_manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    if actual_manifest_sha256 != report.get("manifest_sha256"):
        raise ValueError("audit_report.json 已过期：manifest SHA256 不一致")
    if len(load_manifest(data_root)) != int(report["trajectory_count"]):
        raise ValueError("audit_report.json 已过期：trajectory_count 不一致")
    return {
        "dataset_sha256": report["dataset_sha256"],
        "manifest_sha256": report["manifest_sha256"],
        "trajectory_count": report["trajectory_count"],
        "step_count": report["step_count"],
        "event_detection_config": report.get("event_detection_config"),
        "event_state_trajectory_count": report.get("event_state_trajectory_count", 0),
        "event_counts": report.get("event_counts", {}),
    }


def _append_metric(path: Path, payload: dict[str, object]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")


def _validate_output_path(output: Path, *, resume: bool) -> None:
    if not os.path.lexists(output):
        if resume:
            raise FileNotFoundError("恢复训练要求输出目录已存在")
        return
    output_stat = output.lstat()
    if stat.S_ISLNK(output_stat.st_mode) or not stat.S_ISDIR(output_stat.st_mode):
        raise ValueError("训练输出必须是普通目录，禁止 symlink")
    if output_stat.st_uid != os.getuid():
        raise PermissionError("训练输出目录 owner 与当前进程不一致")
    if resume:
        if stat.S_IMODE(output_stat.st_mode) != 0o700:
            raise PermissionError("恢复训练要求输出目录 mode=0700")
    elif any(output.iterdir()):
        raise FileExistsError("新训练输出目录必须为空")


def _read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, line in enumerate(
        path.read_text(encoding="utf-8").splitlines(),
        start=1,
    ):
        if not line.strip():
            raise ValueError(f"{path.name} 第 {line_number} 行为空")
        value = json.loads(line)
        if not isinstance(value, dict):
            raise TypeError(f"{path.name} 第 {line_number} 行必须是 object")
        rows.append(value)
    return rows


def _validate_resume_artifacts(
    *,
    experiment_path: Path,
    metrics_path: Path,
    exposure_path: Path,
    expected_experiment: dict[str, object],
    completed_epochs: int,
    overfit: bool,
) -> None:
    observed_experiment = json.loads(experiment_path.read_text(encoding="utf-8"))
    normalized_expected = json.loads(
        json.dumps(expected_experiment, sort_keys=True, allow_nan=False)
    )
    for key in (
        "training_config",
        "dataset",
        "proprio_stats",
        "code_revision",
        "trainable_parameters",
        "frozen_parameters",
        "gpu",
    ):
        if observed_experiment.get(key) != normalized_expected.get(key):
            raise ValueError(f"恢复训练 experiment {key} 漂移")
    ignored_arguments = {"resume", "init_checkpoint"}
    observed_arguments = {
        key: value
        for key, value in observed_experiment.get("arguments", {}).items()
        if key not in ignored_arguments
    }
    expected_arguments = {
        key: value
        for key, value in normalized_expected.get("arguments", {}).items()
        if key not in ignored_arguments
    }
    if observed_arguments != expected_arguments:
        raise ValueError("恢复训练 experiment arguments 漂移")
    metrics = _read_jsonl_rows(metrics_path)
    epoch_rows = [row for row in metrics if row.get("event") == "epoch"]
    if [row.get("epoch") for row in epoch_rows] != list(
        range(1, completed_epochs + 1)
    ):
        raise ValueError("恢复训练 metrics epoch 与 checkpoint 不一致")
    if not overfit:
        exposure_rows = _read_jsonl_rows(exposure_path)
        if [row.get("epoch") for row in exposure_rows] != list(
            range(1, completed_epochs + 1)
        ):
            raise ValueError("恢复训练 exposure epoch 与 checkpoint 不一致")


def _should_save_checkpoint(
    *,
    completed_epoch: int,
    total_epochs: int,
    every_epochs: int,
    validation_improved: bool,
) -> bool:
    """周期/latest 与真正 best 使用同一原子保存入口，但 best 不受周期限制。"""

    return (
        validation_improved
        or completed_epoch % every_epochs == 0
        or completed_epoch == total_epochs
    )


def run(args: argparse.Namespace) -> None:
    if (
        args.epochs <= 0
        or args.samples_per_epoch <= 0
        or args.validation_samples <= 0
        or args.checkpoint_every_epochs <= 0
    ):
        raise ValueError("epochs/samples-per-epoch/validation-samples 必须为正数")
    if not math.isfinite(args.event_loss_weight) or args.event_loss_weight < 0:
        raise ValueError("event-loss-weight 必须是有限非负数")
    if args.event_loss_warmup_steps < 0:
        raise ValueError("event-loss-warmup-steps 不能为负数")
    if args.resume is not None and args.init_checkpoint is not None:
        raise ValueError("--resume 与 --init-checkpoint 互斥")
    if args.dagger_data is not None and args.proprio_stats_data is None:
        raise ValueError("使用 --dagger-data 时必须用 --proprio-stats-data 显式冻结 D0 stats")
    _validate_output_path(args.output, resume=args.resume is not None)
    if not torch.cuda.is_available():
        raise RuntimeError("Stage 1 正式入口需要 CUDA")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    skill_sampling_weights = _resolve_skill_sampling_weights(args.skill_weights)
    source_sampling_weights = _resolve_source_sampling_weights(args.source_weight)
    if args.overfit_samples > 0 and source_sampling_weights:
        raise ValueError("小数据过拟合模式不支持 source-first quota")

    spec = RobotSpec()
    stats_root = args.proprio_stats_data or args.data
    if args.dagger_data is not None and stats_root.resolve() != args.data.resolve():
        raise ValueError("E012 dagger training 的 ProprioStats 必须来自 --data 指定的 D0")
    stats_path = stats_root / "proprio_stats.json"
    stats = ProprioStats.from_json(stats_path)
    stats.validate(spec)
    normalizer = ProprioNormalizer(stats, spec)
    finger_force_stats_path = stats_root / "finger_force_stats.json"
    finger_force_stats = None
    finger_force_normalizer = None
    if args.observation_v2:
        finger_force_stats = FingerForceStats.from_json(finger_force_stats_path)
        finger_force_stats.validate(spec)
        finger_force_normalizer = FingerForceNormalizer(finger_force_stats, spec)
    processor = QwenVLAProcessorAdapter.from_pretrained(
        cache_dir=str(args.model_cache),
        local_files_only=True,
        config=QwenProcessorConfig(
            prompt_version=(
                PROMPT_VERSION_OBSERVATION_V2
                if args.observation_v2
                else PROMPT_VERSION
            )
        ),
    )
    collator = (
        QwenVLAObservationV2Collator(processor, spec)
        if args.observation_v2
        else QwenVLACollator(processor, spec)
    )
    dataset_type = (
        ObservationV2ActionChunkDataset
        if args.observation_v2
        else ActionChunkDataset
    )
    dataset_kwargs = (
        {"finger_force_normalizer": finger_force_normalizer}
        if finger_force_normalizer is not None
        else {}
    )
    train_entries = load_manifest(args.data, split="train")
    val_entries = load_manifest(args.data, split="val")
    base_train_dataset = dataset_type(
        str(args.data),
        train_entries,
        spec,
        normalizer,
        cache_size=len(train_entries),
        **dataset_kwargs,
    )
    train_dataset: ActionChunkDataset | CompositeActionChunkDataset
    dagger_dataset_identity = None
    if args.dagger_data is None:
        train_dataset = base_train_dataset
    else:
        dagger_entries = load_manifest(args.dagger_data, split="train")
        if any(entry.local_dagger is None for entry in dagger_entries):
            raise ValueError("--dagger-data 禁止包含 clean/base trajectory")
        dagger_dataset = dataset_type(
            str(args.dagger_data),
            dagger_entries,
            spec,
            normalizer,
            cache_size=len(dagger_entries),
            **dataset_kwargs,
        )
        train_dataset = CompositeActionChunkDataset(
            (base_train_dataset, dagger_dataset)
        )
        dagger_dataset_identity = _load_audit_identity(args.dagger_data)
    val_dataset = dataset_type(
        str(args.data),
        val_entries,
        spec,
        normalizer,
        cache_size=len(val_entries),
        **dataset_kwargs,
    )

    if args.overfit_samples > 0:
        indices = _fixed_overfit_indices(train_dataset, args.overfit_samples)
        subset = Subset(train_dataset, indices)
        train_loader = DataLoader(
            subset,
            batch_size=args.micro_batch_size,
            shuffle=False,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = train_loader
    else:
        train_sampler = TaskEpisodeBalancedSampler(
            train_dataset,
            num_samples=args.samples_per_epoch,
            seed=args.seed,
            skill_weights=skill_sampling_weights,
            source_weights=source_sampling_weights,
        )
        val_sampler = TaskEpisodeBalancedSampler(
            val_dataset,
            num_samples=args.validation_samples,
            seed=1_009,
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=args.micro_batch_size,
            sampler=train_sampler,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=args.micro_batch_size,
            sampler=val_sampler,
            collate_fn=collator,
            num_workers=0,
            pin_memory=True,
        )

    config = Stage1TrainingConfig(
        learning_rate=args.learning_rate,
        warmup_steps=args.warmup_steps,
        cosine_decay_steps=args.cosine_decay_steps,
        micro_batch_size=args.micro_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        seed=args.seed,
        checkpoint_interval_steps=args.checkpoint_interval_steps,
        samples_per_epoch=(
            args.overfit_samples if args.overfit_samples > 0 else args.samples_per_epoch
        ),
        skill_sampling_weights=(
            () if args.overfit_samples > 0 else skill_sampling_weights
        ),
        source_sampling_weights=(
            () if args.overfit_samples > 0 else source_sampling_weights
        ),
        event_loss_weight=args.event_loss_weight,
        event_loss_warmup_steps=args.event_loss_warmup_steps,
        executed_action_steps=spec.execute_steps,
    )

    policy_loader = (
        load_qwen_vla_observation_v2_policy
        if args.observation_v2
        else load_qwen_vla_policy
    )
    policy = policy_loader(
        cache_dir=str(args.model_cache),
        local_files_only=True,
        device="cuda",
        context_layer=args.qwen_context_layer,
    )
    trainer = Stage1Trainer(policy, config, "cuda")
    project_root = Path(__file__).resolve().parents[3]
    code_revision = compute_source_revision(project_root)
    checkpoint_metadata = None
    initialization: dict[str, object] = {"mode": "fresh"}
    if args.resume is not None:
        checkpoint_metadata = load_stage1_checkpoint(
            args.resume,
            policy,
            trainer,
            spec,
            processor.config,
            stats,
            expected_code_revision=code_revision,
            finger_force_stats=finger_force_stats,
        )
        initialization = {
            "mode": "resume",
            "checkpoint": {
                "path": str(args.resume.resolve()),
                "sha256": _sha256_file(args.resume),
                "metadata": checkpoint_metadata,
            },
        }
    elif args.init_checkpoint is not None:
        initialization_receipt = initialize_stage1_policy_checkpoint(
            args.init_checkpoint,
            policy,
            spec,
            processor.config,
            stats,
            finger_force_stats=finger_force_stats,
        )
        checkpoint_metadata = initialization_receipt["metadata"]
        if (
            trainer.state.optimizer_steps != 0
            or trainer.scheduler.completed_steps != 0
            or trainer.optimizer.state
        ):
            raise RuntimeError("init-checkpoint 污染了新训练器状态")
        initialization = {
            "mode": "init_checkpoint",
            "checkpoint": {
                "path": str(args.init_checkpoint.resolve()),
                "sha256": _sha256_file(args.init_checkpoint),
                "metadata": checkpoint_metadata,
                "policy_state_sha256": initialization_receipt[
                    "policy_state_sha256"
                ],
            },
            "restored_state": "adapter_expert_weights_only",
            "trainer_state_reset": True,
            "rng_restored": False,
        }
    trainable_parameters = sum(
        parameter.numel() for parameter in policy.parameters() if parameter.requires_grad
    )
    frozen_parameters = sum(
        parameter.numel() for parameter in policy.parameters() if not parameter.requires_grad
    )
    baseline_allocated = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()

    _validate_output_path(args.output, resume=args.resume is not None)
    args.output.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(args.output, 0o700)
    metrics_path = args.output / "metrics.jsonl"
    exposure_path = args.output / "sampler_exposure.jsonl"
    if metrics_path.exists() and args.resume is None:
        raise FileExistsError("输出目录已有 metrics.jsonl，拒绝覆盖或混合实验")
    if args.resume is not None and not metrics_path.is_file():
        raise FileNotFoundError("恢复训练时输出目录必须包含已有 metrics.jsonl")
    if args.overfit_samples == 0:
        if exposure_path.exists() and args.resume is None:
            raise FileExistsError("输出目录已有 sampler_exposure.jsonl，拒绝混合实验")
        if args.resume is not None and not exposure_path.is_file():
            raise FileNotFoundError("恢复训练时输出目录必须包含 sampler_exposure.jsonl")
    arguments = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in vars(args).items()
    }
    experiment = {
        "arguments": arguments,
        "training_config": config.to_dict(),
        "dataset": {
            "base_d0": _load_audit_identity(args.data),
            "dagger_additions": dagger_dataset_identity,
        },
        "proprio_stats": {
            "path": str(stats_path.resolve()),
            "sha256": _sha256_file(stats_path),
            "frozen_from_data": str(stats_root.resolve()),
        },
        "finger_force_stats": (
            {
                "path": str(finger_force_stats_path.resolve()),
                "sha256": _sha256_file(finger_force_stats_path),
                "frozen_from_data": str(stats_root.resolve()),
            }
            if finger_force_stats is not None
            else None
        ),
        "initialization": initialization,
        "code_revision": code_revision,
        "trainable_parameters": trainable_parameters,
        "frozen_parameters": frozen_parameters,
        "gpu": torch.cuda.get_device_name(0),
    }
    experiment_path = args.output / "experiment.json"
    if args.resume is None:
        experiment_path.write_text(
            json.dumps(experiment, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    elif not experiment_path.is_file():
        raise FileNotFoundError("恢复训练时缺少 experiment.json")
    else:
        _validate_resume_artifacts(
            experiment_path=experiment_path,
            metrics_path=metrics_path,
            exposure_path=exposure_path,
            expected_experiment=experiment,
            completed_epochs=trainer.state.completed_epochs,
            overfit=args.overfit_samples > 0,
        )

    initial_validation = None
    if args.overfit_samples > 0 and not args.measure_only:
        initial_validation = trainer.validate(val_loader)
        _append_metric(
            metrics_path,
            {"event": "initial_validation", **asdict(initial_validation)},
        )

    epochs = 1 if args.measure_only else args.epochs
    start_epoch = trainer.state.completed_epochs
    if start_epoch >= epochs:
        raise ValueError(f"Checkpoint 已完成 {start_epoch} epochs，不小于目标 {epochs}")
    for epoch in range(start_epoch, epochs):
        if args.overfit_samples == 0:
            train_sampler.set_epoch(epoch)
        train_metrics = trainer.train_epoch(train_loader)
        memory = {
            "baseline_allocated_bytes": baseline_allocated,
            "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
            "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        }
        payload: dict[str, object] = {
            "event": "epoch",
            "epoch": epoch + 1,
            "train": asdict(train_metrics),
            "memory": memory,
        }
        exposure_record = None
        if args.overfit_samples == 0:
            exposure_rows = train_sampler.exposure_rows()
            exposure_total = sum(int(row["samples"]) for row in exposure_rows)
            if exposure_total != train_metrics.examples:
                raise RuntimeError(
                    "sampler exposure 与 Trainer examples 不一致: "
                    f"{exposure_total} != {train_metrics.examples}"
                )
            exposure_record = {
                "format": "robot-vla-stage1-sampler-exposure/v1",
                "epoch": epoch + 1,
                "configured_source_weights": [list(item) for item in source_sampling_weights],
                "samples": exposure_total,
                "source_skill_boundary_offset": exposure_rows,
            }
            payload["source_exposure"] = exposure_record
        if not args.measure_only:
            validation = trainer.validate(val_loader)
            payload["validation"] = asdict(validation)
            should_checkpoint = _should_save_checkpoint(
                completed_epoch=epoch + 1,
                total_epochs=epochs,
                every_epochs=args.checkpoint_every_epochs,
                validation_improved=validation.improved,
            )
            if should_checkpoint:
                save_stage1_checkpoint_set(
                    args.output / "checkpoints",
                    policy,
                    trainer,
                    spec,
                    processor.config,
                    stats,
                    code_revision=code_revision,
                    is_best=validation.improved,
                    finger_force_stats=finger_force_stats,
                )
        _append_metric(metrics_path, payload)
        if exposure_record is not None:
            _append_metric(exposure_path, exposure_record)
        print(json.dumps(payload, sort_keys=True), flush=True)

    if initial_validation is not None:
        final_loss = float(payload["validation"]["loss"])
        ratio = final_loss / initial_validation.loss
        result = {"event": "overfit_result", "loss_ratio": ratio, "passed": ratio <= 0.5}
        _append_metric(metrics_path, result)
        print(json.dumps(result, sort_keys=True), flush=True)
        if ratio > 0.5:
            raise RuntimeError(f"小数据过拟合未通过：final/initial={ratio:.4f} > 0.5")


def main() -> None:
    run(_parse_args())


if __name__ == "__main__":
    main()
