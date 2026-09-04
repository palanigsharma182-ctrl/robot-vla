"""E017-P0 保守 goal observability 分类行微调。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from robot_vla.precision.checkpoint import (
    PrecisionCheckpointProvenance,
    PrecisionCheckpointRole,
    load_precision_checkpoint,
    precision_parameter_state_sha256,
    save_precision_checkpoint,
)
from robot_vla.precision.data import canonical_sha256
from robot_vla.precision.e016_pretraining import (
    E016CorrectedPrecisionDataset,
    E016P0Metrics,
    build_e016_loader,
    evaluate_e016_p0_model,
    move_e016_batch_to_device,
    seed_e016_everything,
)
from robot_vla.precision.e016_training import (
    audit_e016_p1_train_val_inputs,
    load_e016_p1_config,
    validate_e016_p0_prerequisite,
)
from robot_vla.precision.model import PrecisionThreeHeadUNet
from robot_vla.precision.training import source_tree_sha256

E017_P0_VERSION = "e017-p0-conservative-observability-finetune/v1"
E017_P0_SCHEDULER = "cosine-annealing-eta-min-10-percent/v1"
E017_P0_TRAINABLE_SCOPE = "uncertainty-final-goal-visibility-and-projection-rows-only/v1"
E017_P0_LOSS_POLICY = "weighted-goal-visibility-plus-projection-bce/v1"
E017_P0_SELECTION_POLICY = "strict-validation-safety-improvement/v1"
E017_P0_SELECTION_ORDER = "min-goal-unobservable-fpr-then-max-recall-then-earlier-epoch"


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys 漂移: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _require_sha256(value: str, name: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")


def _positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _finite_probability(value: float, name: str) -> None:
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} 必须位于 [0,1]")


@dataclass(frozen=True)
class E017P0ParentConfig:
    checkpoint_sha256: str
    provenance_sha256: str
    parameter_state_sha256: str
    training_config_sha256: str
    data_identity_sha256: str
    corrected_data_identity_sha256: str
    selected_epoch: int
    required_role: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.checkpoint_sha256, "checkpoint_sha256"),
            (self.provenance_sha256, "provenance_sha256"),
            (self.parameter_state_sha256, "parameter_state_sha256"),
            (self.training_config_sha256, "training_config_sha256"),
            (self.data_identity_sha256, "data_identity_sha256"),
            (self.corrected_data_identity_sha256, "corrected_data_identity_sha256"),
        ):
            _require_sha256(value, f"parent.{name}")
        _positive_int(self.selected_epoch, "parent.selected_epoch")
        if self.required_role != PrecisionCheckpointRole.FORMAL_TRAINING.value:
            raise ValueError("E017-P0 parent 必须是 formal-training checkpoint")


@dataclass(frozen=True)
class E017P0TrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    use_bf16: bool
    cache_size: int
    heatmap_sigma_px: float
    goal_negative_weight: float
    scheduler: str
    initialization: str
    trainable_scope: str
    loss_policy: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.seed, "seed"),
            (self.epochs, "epochs"),
            (self.batch_size, "batch_size"),
            (self.cache_size, "cache_size"),
        ):
            _positive_int(value, f"training.{name}")
        if self.epochs != 8:
            raise ValueError("E017-P0 epochs 固定为 8")
        for value, name in (
            (self.learning_rate, "learning_rate"),
            (self.gradient_clip_norm, "gradient_clip_norm"),
            (self.heatmap_sigma_px, "heatmap_sigma_px"),
            (self.goal_negative_weight, "goal_negative_weight"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"training.{name} 必须是有限正数")
        if self.weight_decay != 0.0:
            raise ValueError("E017-P0 weight_decay 必须为 0，避免冻结行发生 AdamW 漂移")
        if self.goal_negative_weight != 4.0:
            raise ValueError("E017-P0 goal negative weight 固定为 4.0")
        if self.use_bf16 is not True:
            raise ValueError("E017-P0 必须启用 BF16")
        if self.scheduler != E017_P0_SCHEDULER:
            raise ValueError("E017-P0 scheduler 漂移")
        if self.initialization != "warm-start-e016-p1-selected-epoch-12":
            raise ValueError("E017-P0 必须从冻结的 E016-P1 selected epoch 12 warm-start")
        if self.trainable_scope != E017_P0_TRAINABLE_SCOPE:
            raise ValueError("E017-P0 trainable scope 漂移")
        if self.loss_policy != E017_P0_LOSS_POLICY:
            raise ValueError("E017-P0 loss policy 漂移")


@dataclass(frozen=True)
class E017P0ValidationSelectionConfig:
    policy: str
    visibility_threshold: float
    projection_threshold: float
    goal_unobservable_false_positive_rate_max: float
    goal_visibility_precision_min: float
    goal_visibility_recall_min: float
    projection_accuracy_min: float
    localization_metric_absolute_drift_max: float
    goal_mask_iou_absolute_drift_max: float
    selection_order: str

    def __post_init__(self) -> None:
        if self.policy != E017_P0_SELECTION_POLICY:
            raise ValueError("E017-P0 validation selection policy 漂移")
        for value, name in (
            (self.visibility_threshold, "visibility_threshold"),
            (self.projection_threshold, "projection_threshold"),
            (
                self.goal_unobservable_false_positive_rate_max,
                "goal_unobservable_false_positive_rate_max",
            ),
            (self.goal_visibility_precision_min, "goal_visibility_precision_min"),
            (self.goal_visibility_recall_min, "goal_visibility_recall_min"),
            (self.projection_accuracy_min, "projection_accuracy_min"),
        ):
            _finite_probability(value, f"validation_selection.{name}")
        for value, name in (
            (
                self.localization_metric_absolute_drift_max,
                "localization_metric_absolute_drift_max",
            ),
            (
                self.goal_mask_iou_absolute_drift_max,
                "goal_mask_iou_absolute_drift_max",
            ),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"validation_selection.{name} 必须是有限非负数")
        if self.selection_order != E017_P0_SELECTION_ORDER:
            raise ValueError("E017-P0 selection order 漂移")


@dataclass(frozen=True)
class E017P0ExecutionConfig:
    device: str
    required_gpu: str
    persist_selected_checkpoint_only: bool
    test_split_read_allowed: bool
    actuation_allowed: bool
    mode: str

    def __post_init__(self) -> None:
        if self.device != "cuda":
            raise ValueError("E017-P0 必须使用 CUDA")
        if self.required_gpu != "NVIDIA GeForce RTX 5080":
            raise ValueError("E017-P0 canonical hardware 必须是 RTX 5080")
        if self.persist_selected_checkpoint_only is not True:
            raise ValueError("E017-P0 只允许持久化 selected checkpoint")
        if self.test_split_read_allowed is not False:
            raise ValueError("E017-P0 禁止读取 test")
        if self.actuation_allowed is not False:
            raise ValueError("E017-P0 禁止 actuation")
        if self.mode != "train-val-only-no-actuation":
            raise ValueError("E017-P0 execution mode 漂移")


@dataclass(frozen=True)
class E017P0Config:
    parent: E017P0ParentConfig
    training: E017P0TrainingConfig
    validation_selection: E017P0ValidationSelectionConfig
    execution: E017P0ExecutionConfig
    version: str = E017_P0_VERSION

    def __post_init__(self) -> None:
        if self.version != E017_P0_VERSION:
            raise ValueError("E017-P0 config version 漂移")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def load_e017_p0_config(path: str | Path) -> E017P0Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("E017-P0 config 必须是 JSON object")
    _require_exact_keys(
        payload,
        {"version", "parent", "training", "validation_selection", "execution"},
        "E017-P0 config",
    )

    def section(name: str, cls: type[Any]) -> dict[str, Any]:
        value = payload[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"E017-P0 {name} 必须是 object")
        _require_exact_keys(value, set(cls.__dataclass_fields__), f"E017-P0 {name}")
        return dict(value)

    return E017P0Config(
        version=str(payload["version"]),
        parent=E017P0ParentConfig(**section("parent", E017P0ParentConfig)),
        training=E017P0TrainingConfig(**section("training", E017P0TrainingConfig)),
        validation_selection=E017P0ValidationSelectionConfig(
            **section("validation_selection", E017P0ValidationSelectionConfig)
        ),
        execution=E017P0ExecutionConfig(**section("execution", E017P0ExecutionConfig)),
    )


def e017_trainable_output_rows(model: PrecisionThreeHeadUNet) -> tuple[int, int]:
    """返回 final uncertainty linear 中 goal visibility / projection 的行号。"""

    if not isinstance(model, PrecisionThreeHeadUNet):
        raise TypeError("model 必须是 PrecisionThreeHeadUNet")
    keypoint_count = model.config.keypoint_count
    if keypoint_count != 2 or model.config.keypoint_names[1] != "goal_center":
        raise ValueError("E017-P0 要求 object_center/goal_center 两关键点契约")
    motion_end = keypoint_count * 2 + model.config.motion_spec.motion_dim
    goal_visibility_row = motion_end + 1
    projection_row = motion_end + keypoint_count
    final = model.uncertainty_head[-1]
    if not isinstance(final, nn.Linear) or final.out_features != projection_row + 1:
        raise RuntimeError("E017-P0 uncertainty final linear shape 漂移")
    return goal_visibility_row, projection_row


def e017_observability_loss(
    goal_visibility_logit: torch.Tensor,
    goal_observable: torch.Tensor,
    projection_logit: torch.Tensor,
    projection_valid: torch.Tensor,
    *,
    goal_negative_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """只训练 goal visibility 与 projection；负类加权抑制 unsafe write。"""

    expected = tuple(goal_visibility_logit.shape)
    if expected != tuple(goal_observable.shape):
        raise ValueError("goal visibility logit/target shape 不一致")
    if tuple(projection_logit.shape) != tuple(projection_valid.shape):
        raise ValueError("projection logit/target shape 不一致")
    if goal_observable.dtype != torch.bool or projection_valid.dtype != torch.bool:
        raise TypeError("observability target 必须是 bool")
    if not math.isfinite(goal_negative_weight) or goal_negative_weight <= 0.0:
        raise ValueError("goal_negative_weight 必须是有限正数")
    goal_target = goal_observable.float()
    goal_raw = F.binary_cross_entropy_with_logits(
        goal_visibility_logit.float(), goal_target, reduction="none"
    )
    goal_weights = torch.where(
        goal_observable,
        torch.ones_like(goal_raw),
        torch.full_like(goal_raw, float(goal_negative_weight)),
    )
    goal_loss = torch.mean(goal_raw * goal_weights)
    projection_loss = F.binary_cross_entropy_with_logits(
        projection_logit.float(), projection_valid.float()
    )
    return goal_loss + projection_loss, goal_loss, projection_loss


def _metric_drift(
    metrics: E016P0Metrics,
    parent: E016P0Metrics,
) -> tuple[float, float]:
    return (
        abs(metrics.goal_observable_normalized_uv_mae - parent.goal_observable_normalized_uv_mae),
        abs(metrics.goal_mask_iou - parent.goal_mask_iou),
    )


def e017_validation_guardrails(
    metrics: E016P0Metrics,
    parent: E016P0Metrics,
    config: E017P0ValidationSelectionConfig,
) -> dict[str, bool]:
    localization_drift, mask_drift = _metric_drift(metrics, parent)
    strict_improvement = (
        metrics.goal_unobservable_false_positive_rate < parent.goal_unobservable_false_positive_rate
        or (
            metrics.goal_unobservable_false_positive_rate
            == parent.goal_unobservable_false_positive_rate
            and metrics.goal_visibility_recall > parent.goal_visibility_recall
        )
    )
    return {
        "goal_unobservable_false_positive_rate": (
            metrics.goal_unobservable_false_positive_rate
            <= config.goal_unobservable_false_positive_rate_max
        ),
        "goal_visibility_precision": (
            metrics.goal_visibility_precision >= config.goal_visibility_precision_min
        ),
        "goal_visibility_recall": (
            metrics.goal_visibility_recall >= config.goal_visibility_recall_min
        ),
        "projection_accuracy": (metrics.projection_accuracy >= config.projection_accuracy_min),
        "localization_metric_unchanged": (
            localization_drift <= config.localization_metric_absolute_drift_max
        ),
        "goal_mask_iou_unchanged": (mask_drift <= config.goal_mask_iou_absolute_drift_max),
        "strict_safety_improvement_over_parent": strict_improvement,
    }


def select_e017_p0_checkpoint_epoch(
    candidates: Sequence[tuple[int, E016P0Metrics]],
    parent: E016P0Metrics,
    config: E017P0ValidationSelectionConfig,
) -> int | None:
    """先过滤全部门禁，再按 FPR、recall、epoch 做固定排序。"""

    seen: set[int] = set()
    eligible: list[tuple[float, float, int]] = []
    for epoch, metrics in candidates:
        _positive_int(epoch, "candidate epoch")
        if epoch in seen:
            raise ValueError("E017-P0 candidate epoch 重复")
        seen.add(epoch)
        if all(e017_validation_guardrails(metrics, parent, config).values()):
            eligible.append(
                (
                    metrics.goal_unobservable_false_positive_rate,
                    -metrics.goal_visibility_recall,
                    epoch,
                )
            )
    return None if not eligible else min(eligible)[2]


def _atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
        delete=False,
    ) as handle:
        temporary = Path(handle.name)
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    try:
        os.chmod(temporary, 0o600)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _append_jsonl(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _read_checkpoint_receipt(path: Path, config: E017P0Config) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, Mapping):
        raise TypeError("E016-P1 checkpoint receipt 必须是 object")
    receipt = value.get("checkpoint")
    if not isinstance(receipt, Mapping):
        raise TypeError("E016-P1 checkpoint receipt.checkpoint 必须是 object")
    expected = config.parent
    for key, wanted in (
        ("checkpoint_sha256", expected.checkpoint_sha256),
        ("provenance_sha256", expected.provenance_sha256),
        ("parameter_state_sha256", expected.parameter_state_sha256),
    ):
        if receipt.get(key) != wanted:
            raise RuntimeError(f"E017-P0 parent {key} 漂移")
    if value.get("training_config_sha256") != expected.training_config_sha256:
        raise RuntimeError("E017-P0 parent training config identity 漂移")
    if value.get("data_identity_sha256") != expected.data_identity_sha256:
        raise RuntimeError("E017-P0 parent data identity 漂移")
    if value.get("corrected_data_identity_sha256") != expected.corrected_data_identity_sha256:
        raise RuntimeError("E017-P0 parent corrected data identity 漂移")
    if value.get("selected_epoch") != expected.selected_epoch:
        raise RuntimeError("E017-P0 parent selected epoch 漂移")
    if value.get("test_split_read") is not False:
        raise RuntimeError("E017-P0 parent training receipt 意外读取 test")
    return dict(value)


def _frozen_parameter_state(
    model: PrecisionThreeHeadUNet,
    trainable_rows: tuple[int, int],
) -> dict[str, torch.Tensor]:
    """对所有冻结 tensor 和 final linear 的冻结行建立稳定 identity。"""

    final = model.uncertainty_head[-1]
    assert isinstance(final, nn.Linear)
    final_weight_name = next(
        name for name, value in model.named_parameters() if value is final.weight
    )
    final_bias_name = next(name for name, value in model.named_parameters() if value is final.bias)
    row_mask = torch.ones(final.out_features, dtype=torch.bool)
    row_mask[list(trainable_rows)] = False
    frozen: dict[str, torch.Tensor] = {}
    for name, value in model.state_dict().items():
        if name == final_weight_name or name == final_bias_name:
            frozen[f"{name}[frozen_rows]"] = value.detach().cpu()[row_mask].contiguous()
        else:
            frozen[name] = value.detach().cpu().contiguous()
    return frozen


def run_e017_p0_training(
    *,
    deployable_root: str | Path,
    source_label_root: str | Path,
    p0_output_root: str | Path,
    e016_config_path: str | Path,
    parent_checkpoint_path: str | Path,
    parent_receipt_path: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """在 5080 上执行 train/validation-only 的保守分类行微调。"""

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E017-P0 output 已存在，拒绝覆盖: {output}")
    config = load_e017_p0_config(config_path)
    source_identity = source_tree_sha256(repository_root)
    e016_config = load_e016_p1_config(e016_config_path)
    if e016_config.sha256 != config.parent.training_config_sha256:
        raise RuntimeError("E017-P0 E016 parent config SHA-256 漂移")
    prerequisite = validate_e016_p0_prerequisite(p0_output_root, e016_config)
    corrected_root = Path(prerequisite["corrected_label_root"])
    input_audit = audit_e016_p1_train_val_inputs(
        deployable_root=deployable_root,
        source_label_root=source_label_root,
        corrected_label_root=corrected_root,
        config=e016_config,
    )
    if input_audit["data_identity_sha256"] != config.parent.data_identity_sha256:
        raise RuntimeError("E017-P0 train/val input identity 漂移")
    if (
        input_audit["corrected_data_identity_sha256"]
        != config.parent.corrected_data_identity_sha256
    ):
        raise RuntimeError("E017-P0 corrected input identity 漂移")
    if any(
        input_audit[key] != 0
        for key in (
            "test_trajectory_file_read_count",
            "test_label_file_read_count",
            "test_corrected_label_file_read_count",
        )
    ):
        raise RuntimeError("E017-P0 input audit 意外读取 test")
    parent_receipt = _read_checkpoint_receipt(Path(parent_receipt_path), config)
    loaded = load_precision_checkpoint(
        parent_checkpoint_path,
        expected_checkpoint_sha256=config.parent.checkpoint_sha256,
        expected_provenance_sha256=config.parent.provenance_sha256,
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    if loaded.receipt.parameter_state_sha256 != config.parent.parameter_state_sha256:
        raise RuntimeError("E017-P0 loaded parent parameter identity 漂移")
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("E017-P0 要求支持 BF16 的 CUDA GPU")
    device = torch.device(config.execution.device)
    device_name = torch.cuda.get_device_name(device)
    if config.execution.required_gpu not in device_name:
        raise RuntimeError(
            f"E017-P0 canonical training 要求 {config.execution.required_gpu}，实际为 {device_name}"
        )

    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(output / "config_snapshot.json", config.to_dict())
    _atomic_json(output / "train_val_input_audit.json", input_audit)
    _atomic_json(
        output / "parent_receipt.json",
        {
            "checkpoint_sha256": config.parent.checkpoint_sha256,
            "provenance_sha256": config.parent.provenance_sha256,
            "parameter_state_sha256": config.parent.parameter_state_sha256,
            "selected_epoch": config.parent.selected_epoch,
            "formal_checkpoint_written": parent_receipt.get("formal_checkpoint_written"),
            "test_split_read": parent_receipt.get("test_split_read"),
        },
    )
    _atomic_json(
        output / "runtime.json",
        {
            "device": str(device),
            "device_name": device_name,
            "cuda_version": torch.version.cuda,
            "torch_version": torch.__version__,
            "bf16_supported": torch.cuda.is_bf16_supported(),
        },
    )

    training = config.training
    train_dataset = E016CorrectedPrecisionDataset(
        deployable_root,
        source_label_root,
        corrected_root,
        "train",
        cache_size=training.cache_size,
    )
    val_dataset = E016CorrectedPrecisionDataset(
        deployable_root,
        source_label_root,
        corrected_root,
        "val",
        cache_size=training.cache_size,
    )
    seed_e016_everything(training.seed)
    model = loaded.model.to(device)
    trainable_rows = e017_trainable_output_rows(model)
    frozen_before = precision_parameter_state_sha256(_frozen_parameter_state(model, trainable_rows))
    parent_state = {
        name: value.detach().cpu().clone() for name, value in model.state_dict().items()
    }
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    final = model.uncertainty_head[-1]
    assert isinstance(final, nn.Linear)
    final.weight.requires_grad_(True)
    final.bias.requires_grad_(True)
    gradient_rows = torch.zeros(final.out_features, dtype=torch.bool, device=device)
    gradient_rows[list(trainable_rows)] = True
    final.weight.register_hook(lambda gradient: gradient * gradient_rows[:, None])
    final.bias.register_hook(lambda gradient: gradient * gradient_rows)
    frozen_rows = ~gradient_rows
    frozen_weight = final.weight.detach().clone()
    frozen_bias = final.bias.detach().clone()

    parent_metrics = evaluate_e016_p0_model(
        model,
        val_dataset,
        device=device,
        batch_size=training.batch_size,
        use_bf16=training.use_bf16,
        heatmap_sigma_px=training.heatmap_sigma_px,
        loss_config=e016_config.loss,
        visibility_threshold=config.validation_selection.visibility_threshold,
        projection_threshold=config.validation_selection.projection_threshold,
    )
    _atomic_json(output / "parent_validation.json", parent_metrics.to_dict())
    optimizer = torch.optim.AdamW(
        (final.weight, final.bias),
        lr=training.learning_rate,
        weight_decay=training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training.epochs,
        eta_min=training.learning_rate * 0.1,
    )
    loader = build_e016_loader(
        train_dataset,
        batch_size=training.batch_size,
        shuffle=True,
        seed=training.seed,
    )
    candidates: list[tuple[int, E016P0Metrics]] = []
    selected_rows: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}
    counters: dict[int, tuple[int, int]] = {}
    examples_seen = 0
    optimizer_steps = 0
    metrics_path = output / "metrics.jsonl"
    for epoch in range(1, training.epochs + 1):
        model.train()
        weighted_loss = 0.0
        weighted_goal_loss = 0.0
        weighted_projection_loss = 0.0
        epoch_examples = 0
        gradient_norms: list[float] = []
        for raw_batch in loader:
            batch = move_e016_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=training.use_bf16,
            ):
                prediction = model(
                    batch["image"],
                    batch["structured_state"],
                    batch["geometric_motion"],
                )
                total_loss, goal_loss, projection_loss = e017_observability_loss(
                    prediction.visibility_logits[:, 1],
                    batch["keypoint_observable"][:, 1],
                    prediction.projection_validity_logit,
                    batch["projection_valid"],
                    goal_negative_weight=training.goal_negative_weight,
                )
            if not bool(torch.isfinite(total_loss)):
                raise RuntimeError("E017-P0 loss 出现 NaN/Inf")
            total_loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                (final.weight, final.bias), training.gradient_clip_norm
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("E017-P0 gradient 出现 NaN/Inf")
            optimizer.step()
            with torch.no_grad():
                final.weight[frozen_rows].copy_(frozen_weight[frozen_rows])
                final.bias[frozen_rows].copy_(frozen_bias[frozen_rows])
            count = int(batch["image"].shape[0])
            weighted_loss += float(total_loss.detach()) * count
            weighted_goal_loss += float(goal_loss.detach()) * count
            weighted_projection_loss += float(projection_loss.detach()) * count
            epoch_examples += count
            examples_seen += count
            optimizer_steps += 1
            gradient_norms.append(float(gradient_norm.detach()))

        validation = evaluate_e016_p0_model(
            model,
            val_dataset,
            device=device,
            batch_size=training.batch_size,
            use_bf16=training.use_bf16,
            heatmap_sigma_px=training.heatmap_sigma_px,
            loss_config=e016_config.loss,
            visibility_threshold=config.validation_selection.visibility_threshold,
            projection_threshold=config.validation_selection.projection_threshold,
        )
        candidates.append((epoch, validation))
        guardrails = e017_validation_guardrails(
            validation, parent_metrics, config.validation_selection
        )
        eligible = all(guardrails.values())
        if eligible:
            selected_rows[epoch] = (
                final.weight.detach().cpu()[list(trainable_rows)].clone(),
                final.bias.detach().cpu()[list(trainable_rows)].clone(),
            )
            counters[epoch] = (examples_seen, optimizer_steps)
        selected_so_far = select_e017_p0_checkpoint_epoch(
            candidates, parent_metrics, config.validation_selection
        )
        localization_drift, mask_drift = _metric_drift(validation, parent_metrics)
        _append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "examples_seen": examples_seen,
                "optimizer_steps": optimizer_steps,
                "train_loss": weighted_loss / epoch_examples,
                "train_goal_visibility_loss": weighted_goal_loss / epoch_examples,
                "train_projection_loss": weighted_projection_loss / epoch_examples,
                "gradient_norm_mean": float(np.mean(gradient_norms)),
                "gradient_norm_max": float(np.max(gradient_norms)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validation": validation.to_dict(),
                "validation_guardrails": guardrails,
                "localization_metric_absolute_drift": localization_drift,
                "goal_mask_iou_absolute_drift": mask_drift,
                "checkpoint_eligible": eligible,
                "selected_so_far": epoch == selected_so_far,
            },
        )
        scheduler.step()

    selected_epoch = select_e017_p0_checkpoint_epoch(
        candidates, parent_metrics, config.validation_selection
    )
    if selected_epoch is None:
        failure = {
            "version": E017_P0_VERSION,
            "passed": False,
            "reason": "no-trained-epoch-strictly-improved-validation-safety",
            "formal_checkpoint_written": False,
            "total_epochs_run": training.epochs,
            "total_optimizer_steps_run": optimizer_steps,
            "test_split_read": False,
            "actuation_allowed": False,
            "safe_for_actuator_promotion": False,
        }
        _atomic_json(output / "failure_receipt.json", failure)
        return failure

    model.load_state_dict(parent_state, strict=True)
    selected_weight, selected_bias = selected_rows[selected_epoch]
    with torch.no_grad():
        final = model.uncertainty_head[-1]
        assert isinstance(final, nn.Linear)
        final.weight[list(trainable_rows)].copy_(selected_weight.to(device))
        final.bias[list(trainable_rows)].copy_(selected_bias.to(device))
    frozen_after = precision_parameter_state_sha256(_frozen_parameter_state(model, trainable_rows))
    if frozen_after != frozen_before:
        raise RuntimeError("E017-P0 冻结参数或非目标 final rows 发生漂移")
    selected_metrics = dict(candidates)[selected_epoch]
    selected_examples, selected_steps = counters[selected_epoch]
    provenance = PrecisionCheckpointProvenance(
        role=PrecisionCheckpointRole.FORMAL_TRAINING,
        data_identity_sha256=input_audit["data_identity_sha256"],
        training_config_sha256=config.sha256,
        source_tree_sha256=source_identity,
        seed=training.seed,
        examples_seen=selected_examples,
        optimizer_steps=selected_steps,
    )
    checkpoint_path = output / "precision-e017-p0.pt"
    checkpoint_receipt = save_precision_checkpoint(checkpoint_path, model, provenance)
    strict = load_precision_checkpoint(
        checkpoint_path,
        expected_checkpoint_sha256=checkpoint_receipt.checkpoint_sha256,
        expected_provenance_sha256=checkpoint_receipt.provenance_sha256,
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    if precision_parameter_state_sha256(strict.model.state_dict()) != (
        checkpoint_receipt.parameter_state_sha256
    ):
        raise RuntimeError("E017-P0 strict reload parameter identity 漂移")
    receipt = {
        "version": E017_P0_VERSION,
        "experiment": "E017-P0 conservative observability row fine-tuning",
        "passed": True,
        "parent_checkpoint_sha256": config.parent.checkpoint_sha256,
        "parent_parameter_state_sha256": config.parent.parameter_state_sha256,
        "training_config_sha256": config.sha256,
        "source_tree_sha256": source_identity,
        "data_identity_sha256": input_audit["data_identity_sha256"],
        "corrected_data_identity_sha256": input_audit["corrected_data_identity_sha256"],
        "selected_epoch": selected_epoch,
        "selected_examples_seen": selected_examples,
        "selected_optimizer_steps": selected_steps,
        "total_epochs_run": training.epochs,
        "total_optimizer_steps_run": optimizer_steps,
        "parent_validation": parent_metrics.to_dict(),
        "selected_validation": selected_metrics.to_dict(),
        "validation_guardrails": e017_validation_guardrails(
            selected_metrics, parent_metrics, config.validation_selection
        ),
        "trainable_output_rows": list(trainable_rows),
        "frozen_parameter_state_sha256_before": frozen_before,
        "frozen_parameter_state_sha256_after": frozen_after,
        "frozen_parameters_unchanged": True,
        "checkpoint": checkpoint_receipt.to_dict(),
        "strict_reload_passed": True,
        "formal_checkpoint_written": True,
        "test_split_read": False,
        "actuation_allowed": False,
        "safe_for_actuator_promotion": False,
    }
    _atomic_json(output / "checkpoint_receipt.json", receipt)
    return receipt


__all__ = [
    "E017_P0_VERSION",
    "E017P0Config",
    "E017P0ExecutionConfig",
    "E017P0ParentConfig",
    "E017P0TrainingConfig",
    "E017P0ValidationSelectionConfig",
    "e017_observability_loss",
    "e017_trainable_output_rows",
    "e017_validation_guardrails",
    "load_e017_p0_config",
    "run_e017_p0_training",
    "select_e017_p0_checkpoint_epoch",
]
