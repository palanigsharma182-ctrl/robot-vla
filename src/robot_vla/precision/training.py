"""E013 Precision U-Net 的真实 RGB overfit gate 与正式训练。"""

from __future__ import annotations

import json
import math
import os
import random
import subprocess
import tempfile
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from robot_vla.contracts import RobotSpec
from robot_vla.precision.checkpoint import (
    PrecisionCheckpointProvenance,
    PrecisionCheckpointReceipt,
    PrecisionCheckpointRole,
    load_precision_checkpoint,
    precision_parameter_state_sha256,
    save_precision_checkpoint,
)
from robot_vla.precision.data import (
    PrecisionDatasetAuditReport,
    PrecisionRGBDataset,
    audit_precision_dataset,
    canonical_sha256,
)
from robot_vla.precision.losses import (
    PrecisionLossConfig,
    PrecisionSupervision,
    build_gaussian_heatmaps,
    precision_unet_loss,
)
from robot_vla.precision.model import PrecisionThreeHeadUNet, PrecisionUNetConfig

PRECISION_TRAINING_VERSION = "e013-precision-training/v1"


@dataclass(frozen=True)
class PrecisionDatasetPlan:
    start_seed: int
    max_candidates: int
    train_trajectories: int
    val_trajectories: int
    test_trajectories: int
    history_length: int
    image_height: int
    image_width: int
    model_camera: str
    privileged_sidecar: bool

    def __post_init__(self) -> None:
        for name in (
            "start_seed",
            "max_candidates",
            "train_trajectories",
            "val_trajectories",
            "test_trajectories",
            "history_length",
            "image_height",
            "image_width",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                if name == "start_seed" and value == 0:
                    continue
                raise ValueError(f"dataset.{name} 必须是正整数")
        if self.model_camera != "hand_camera":
            raise ValueError("Precision v1 model_camera 必须为 hand_camera")
        if self.privileged_sidecar is not True:
            raise ValueError("Precision v1 必须启用独立 privileged sidecar")


@dataclass(frozen=True)
class PrecisionOverfitConfig:
    sample_count: int
    optimizer_steps: int
    normalized_uv_mae_max: float
    normalized_uv_improvement_min: float
    mask_iou_min: float

    def __post_init__(self) -> None:
        if not 32 <= self.sample_count <= 128:
            raise ValueError("debug sample_count 必须位于 [32,128]")
        if self.optimizer_steps <= 0:
            raise ValueError("debug optimizer_steps 必须为正整数")
        if not 0.0 < self.normalized_uv_mae_max < 1.0:
            raise ValueError("debug normalized_uv_mae_max 必须位于 (0,1)")
        if not 0.0 < self.normalized_uv_improvement_min < 1.0:
            raise ValueError("debug normalized_uv_improvement_min 必须位于 (0,1)")
        if not 0.0 < self.mask_iou_min < 1.0:
            raise ValueError("debug mask_iou_min 必须位于 (0,1)")


@dataclass(frozen=True)
class PrecisionFormalTrainingConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    use_bf16: bool
    num_workers: int
    cache_size: int
    heatmap_sigma_px: float
    selection_metric: str
    selection_tie_break: str
    motion_head_policy: str

    def __post_init__(self) -> None:
        for name in ("seed", "epochs", "batch_size", "cache_size"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"formal_training.{name} 必须为正整数")
        if self.num_workers != 0:
            raise ValueError("首版正式训练固定 num_workers=0 以保持读取顺序可审计")
        for name in (
            "learning_rate",
            "gradient_clip_norm",
            "heatmap_sigma_px",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"formal_training.{name} 必须是有限正数")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("formal_training.weight_decay 必须有限非负")
        if self.use_bf16 is not True:
            raise ValueError("RTX 4090 正式配置固定 use_bf16=true")
        if self.selection_metric != "val_normalized_uv_mae":
            raise ValueError("首版 selection_metric 必须为 val_normalized_uv_mae")
        if self.selection_tie_break != "earlier_epoch":
            raise ValueError("首版 selection tie-break 必须为 earlier_epoch")
        if self.motion_head_policy != "frozen-zero-shadow-only":
            raise ValueError("首版 Motion Head 必须保持 frozen-zero-shadow-only")


@dataclass(frozen=True)
class PrecisionHeldOutConfig:
    calibration_split: str
    evaluation_split: str
    temperature_grid: tuple[float, ...]
    confidence_target_coverage: float
    provider_latency_warmup_calls: int
    provider_latency_measurement_calls: int


@dataclass(frozen=True)
class PrecisionShadowRolloutConfig:
    start_seed: int
    episode_count: int
    required_control_hz: float
    p95_latency_max_s: float
    actuation_allowed: bool


@dataclass(frozen=True)
class PrecisionExperimentConfig:
    dataset: PrecisionDatasetPlan
    debug_overfit: PrecisionOverfitConfig
    formal_training: PrecisionFormalTrainingConfig
    held_out: PrecisionHeldOutConfig
    shadow_rollout: PrecisionShadowRolloutConfig
    version: str = PRECISION_TRAINING_VERSION

    def __post_init__(self) -> None:
        if self.version != PRECISION_TRAINING_VERSION:
            raise ValueError("Precision training version 漂移")
        if self.dataset.history_length != 4:
            raise ValueError("E013 history length 必须固定为 4")
        if self.held_out.calibration_split != "val":
            raise ValueError("confidence calibration 必须只使用 val split")
        if self.held_out.evaluation_split != "test":
            raise ValueError("held-out evaluation 必须只使用 test split")
        if not self.held_out.temperature_grid or any(
            not math.isfinite(value) or value <= 0.0 for value in self.held_out.temperature_grid
        ):
            raise ValueError("temperature_grid 必须包含有限正数")
        if not 0.0 < self.held_out.confidence_target_coverage < 1.0:
            raise ValueError("confidence_target_coverage 必须位于 (0,1)")
        if (
            not isinstance(self.held_out.provider_latency_warmup_calls, int)
            or isinstance(self.held_out.provider_latency_warmup_calls, bool)
            or self.held_out.provider_latency_warmup_calls <= 0
        ):
            raise ValueError("provider latency warmup 必须为正整数")
        if (
            not isinstance(self.held_out.provider_latency_measurement_calls, int)
            or isinstance(self.held_out.provider_latency_measurement_calls, bool)
            or self.held_out.provider_latency_measurement_calls <= 0
        ):
            raise ValueError("provider latency measurement 必须为正整数")
        if (
            not isinstance(self.shadow_rollout.start_seed, int)
            or isinstance(self.shadow_rollout.start_seed, bool)
            or self.shadow_rollout.start_seed < 0
            or not isinstance(self.shadow_rollout.episode_count, int)
            or isinstance(self.shadow_rollout.episode_count, bool)
            or self.shadow_rollout.episode_count <= 0
        ):
            raise ValueError("shadow rollout seed/count 无效")
        if (
            not math.isfinite(self.shadow_rollout.required_control_hz)
            or self.shadow_rollout.required_control_hz < 20.0
        ):
            raise ValueError("shadow rollout required control rate 不能低于 20 Hz")
        if (
            not math.isfinite(self.shadow_rollout.p95_latency_max_s)
            or self.shadow_rollout.p95_latency_max_s <= 0.0
            or self.shadow_rollout.p95_latency_max_s > 0.05
            or self.shadow_rollout.p95_latency_max_s > 1.0 / self.shadow_rollout.required_control_hz
        ):
            raise ValueError("shadow rollout p95 latency gate 不能放宽到 50 ms 以上")
        if self.shadow_rollout.actuation_allowed is not False:
            raise ValueError("首轮 Precision rollout 必须禁止 actuation")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def load_precision_experiment_config(path: str | Path) -> PrecisionExperimentConfig:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    expected = {
        "version",
        "dataset",
        "debug_overfit",
        "formal_training",
        "held_out",
        "shadow_rollout",
    }
    if set(payload) != expected:
        raise ValueError("Precision experiment config 顶层字段漂移")
    held_out = dict(payload["held_out"])
    held_out["temperature_grid"] = tuple(float(value) for value in held_out["temperature_grid"])
    return PrecisionExperimentConfig(
        version=str(payload["version"]),
        dataset=PrecisionDatasetPlan(**payload["dataset"]),
        debug_overfit=PrecisionOverfitConfig(**payload["debug_overfit"]),
        formal_training=PrecisionFormalTrainingConfig(**payload["formal_training"]),
        held_out=PrecisionHeldOutConfig(**held_out),
        shadow_rollout=PrecisionShadowRolloutConfig(**payload["shadow_rollout"]),
    )


@dataclass(frozen=True)
class PrecisionSplitMetrics:
    sample_count: int
    valid_keypoint_count: int
    mean_loss: float
    normalized_uv_mae: float
    pixel_error_p50: float
    pixel_error_p90: float
    mask_iou: float
    visibility_accuracy: float
    projection_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class PrecisionOverfitReceipt:
    passed: bool
    sample_count: int
    optimizer_steps: int
    initial: PrecisionSplitMetrics
    final: PrecisionSplitMetrics
    loss_reduction: float
    normalized_uv_improvement: float
    motion_head_unchanged: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["initial"] = self.initial.to_dict()
        payload["final"] = self.final.to_dict()
        return payload


@dataclass(frozen=True)
class PrecisionTrainingReceipt:
    selected_epoch: int
    selected_metric: float
    examples_seen: int
    optimizer_steps: int
    total_optimizer_steps_run: int
    checkpoint: PrecisionCheckpointReceipt
    data_identity_sha256: str
    training_config_sha256: str
    source_tree_sha256: str
    overfit_gate_passed: bool

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["checkpoint"] = self.checkpoint.to_dict()
        return payload


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
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
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def source_tree_sha256(repository_root: str | Path) -> str:
    """正式训练只接受 clean Git tree，并把 tree object 转为 SHA-256 identity。"""

    root = Path(repository_root)
    status = subprocess.run(
        ["git", "status", "--porcelain", "--untracked-files=all"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout
    if status.strip():
        raise RuntimeError("正式 Precision 训练要求 clean Git worktree")
    tree = subprocess.run(
        ["git", "rev-parse", "HEAD^{tree}"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return canonical_sha256({"git_tree": tree, "git_commit": commit})


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    model_inputs = [sample["model_inputs"] for sample in samples]
    supervision = [sample["supervision"] for sample in samples]
    image = np.stack([item["rgb_wrist"] for item in model_inputs])
    return {
        "image": torch.from_numpy(
            np.ascontiguousarray(image.transpose(0, 3, 1, 2), dtype=np.float32) / np.float32(255.0)
        ),
        "structured_state": torch.from_numpy(
            np.stack([item["structured_state"] for item in model_inputs])
        ),
        "geometric_motion": torch.from_numpy(
            np.stack([item["geometric_motion"] for item in model_inputs])
        ),
        "mask_targets": torch.from_numpy(np.stack([item["mask_targets"] for item in supervision])),
        "normalized_uv_targets": torch.from_numpy(
            np.stack([item["normalized_uv_targets"] for item in supervision])
        ),
        "keypoint_valid": torch.from_numpy(
            np.stack([item["keypoint_valid"] for item in supervision])
        ),
        "motion_residual_targets": torch.from_numpy(
            np.stack([item["motion_residual_targets"] for item in supervision])
        ),
        "motion_valid": torch.from_numpy(np.stack([item["motion_valid"] for item in supervision])),
        "projection_valid": torch.from_numpy(
            np.stack([item["projection_valid"] for item in supervision])
        ),
        "audit": [sample["audit"] for sample in samples],
    }


def _to_device(batch: dict[str, Any], device: torch.device) -> dict[str, Any]:
    return {
        key: value.to(device, non_blocking=False) if isinstance(value, torch.Tensor) else value
        for key, value in batch.items()
    }


def _supervision(
    batch: dict[str, Any],
    image_size_hw: tuple[int, int],
    sigma_px: float,
) -> PrecisionSupervision:
    heatmaps = build_gaussian_heatmaps(
        batch["normalized_uv_targets"],
        batch["keypoint_valid"],
        image_size_hw,
        sigma_px=sigma_px,
    )
    return PrecisionSupervision(
        heatmap_targets=heatmaps,
        mask_targets=batch["mask_targets"],
        normalized_uv_targets=batch["normalized_uv_targets"],
        keypoint_valid=batch["keypoint_valid"],
        motion_residual_targets=batch["motion_residual_targets"],
        motion_valid=batch["motion_valid"],
        projection_valid=batch["projection_valid"],
    )


def _build_loader(
    dataset: Dataset[Any],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
    num_workers: int,
) -> DataLoader[Any]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=_collate,
        generator=generator,
        drop_last=False,
        pin_memory=False,
    )


def evaluate_precision_model(
    model: PrecisionThreeHeadUNet,
    dataset: Dataset[Any],
    *,
    device: torch.device,
    batch_size: int,
    use_bf16: bool,
    heatmap_sigma_px: float,
    temperature: float = 1.0,
) -> PrecisionSplitMetrics:
    loader = _build_loader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        seed=0,
        num_workers=0,
    )
    was_training = model.training
    model.eval()
    losses: list[float] = []
    uv_absolute: list[np.ndarray] = []
    pixel_errors: list[np.ndarray] = []
    mask_intersection = 0
    mask_union = 0
    visibility_correct = 0
    visibility_total = 0
    projection_correct = 0
    projection_total = 0
    sample_count = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            height, width = batch["image"].shape[-2:]
            target = _supervision(
                batch,
                (height, width),
                heatmap_sigma_px,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_bf16,
            ):
                output = model(
                    batch["image"],
                    batch["structured_state"],
                    batch["geometric_motion"],
                )
                loss = precision_unet_loss(output, target, PrecisionLossConfig())
            decoded = output.decode_for_control(temperature=temperature)
            valid = batch["keypoint_valid"]
            error = (decoded.keypoints.normalized_uv - batch["normalized_uv_targets"].float()).abs()
            if bool(valid.any()):
                uv_absolute.append(error[valid].detach().cpu().numpy())
                scale = torch.tensor(
                    (float(width), float(height)),
                    device=device,
                    dtype=torch.float32,
                )
                pixel = torch.linalg.norm(error * scale, dim=-1)
                pixel_errors.append(pixel[valid].detach().cpu().numpy())
            predicted_mask = output.mask_logits > 0.0
            target_mask = batch["mask_targets"] > 0.5
            mask_intersection += int((predicted_mask & target_mask).sum().item())
            mask_union += int((predicted_mask | target_mask).sum().item())
            visibility_predicted = decoded.visibility_probability >= 0.5
            visibility_correct += int((visibility_predicted == valid).sum().item())
            visibility_total += int(valid.numel())
            projection_predicted = decoded.projection_validity_probability >= 0.5
            projection_correct += int(
                (projection_predicted == batch["projection_valid"]).sum().item()
            )
            projection_total += int(batch["projection_valid"].numel())
            losses.append(float(loss.loss.detach().float().item()) * int(valid.shape[0]))
            sample_count += int(valid.shape[0])
    if was_training:
        model.train()
    if not uv_absolute or not pixel_errors:
        raise RuntimeError("Precision evaluation 没有有效 keypoint")
    uv_values = np.concatenate(uv_absolute)
    pixel_values = np.concatenate(pixel_errors)
    return PrecisionSplitMetrics(
        sample_count=sample_count,
        valid_keypoint_count=int(pixel_values.size),
        mean_loss=float(sum(losses) / sample_count),
        normalized_uv_mae=float(np.mean(uv_values)),
        pixel_error_p50=float(np.quantile(pixel_values, 0.50)),
        pixel_error_p90=float(np.quantile(pixel_values, 0.90)),
        mask_iou=float(mask_intersection / mask_union) if mask_union else 1.0,
        visibility_accuracy=float(visibility_correct / visibility_total),
        projection_accuracy=float(projection_correct / projection_total),
    )


def _new_model(device: torch.device) -> PrecisionThreeHeadUNet:
    return PrecisionThreeHeadUNet(PrecisionUNetConfig()).to(device)


def _freeze_motion_head(model: PrecisionThreeHeadUNet) -> str:
    model.motion_head.requires_grad_(False)
    return precision_parameter_state_sha256(model.motion_head.state_dict())


def run_real_sample_overfit_gate(
    train_dataset: PrecisionRGBDataset,
    config: PrecisionExperimentConfig,
    *,
    device: torch.device,
) -> PrecisionOverfitReceipt:
    formal = config.formal_training
    debug = config.debug_overfit
    if len(train_dataset) < debug.sample_count:
        raise ValueError("train Dataset 不足冻结的 real-sample overfit sample_count")
    _seed_everything(formal.seed + 1)
    generator = np.random.default_rng(formal.seed + 1)
    indices = np.sort(
        generator.choice(len(train_dataset), size=debug.sample_count, replace=False)
    ).tolist()
    subset = Subset(train_dataset, indices)
    loader = _build_loader(
        subset,
        batch_size=debug.sample_count,
        shuffle=False,
        seed=formal.seed + 1,
        num_workers=0,
    )
    raw_batch = next(iter(loader))
    batch = _to_device(raw_batch, device)
    model = _new_model(device)
    motion_before = _freeze_motion_head(model)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=formal.learning_rate,
        weight_decay=formal.weight_decay,
    )
    initial = evaluate_precision_model(
        model,
        subset,
        device=device,
        batch_size=debug.sample_count,
        use_bf16=formal.use_bf16,
        heatmap_sigma_px=formal.heatmap_sigma_px,
    )
    model.train()
    for _ in range(debug.optimizer_steps):
        optimizer.zero_grad(set_to_none=True)
        height, width = batch["image"].shape[-2:]
        target = _supervision(batch, (height, width), formal.heatmap_sigma_px)
        with torch.autocast(
            device_type=device.type,
            dtype=torch.bfloat16,
            enabled=formal.use_bf16,
        ):
            output = model(
                batch["image"],
                batch["structured_state"],
                batch["geometric_motion"],
            )
            loss = precision_unet_loss(output, target, PrecisionLossConfig())
        loss.loss.backward()
        torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            formal.gradient_clip_norm,
        )
        optimizer.step()
    final = evaluate_precision_model(
        model,
        subset,
        device=device,
        batch_size=debug.sample_count,
        use_bf16=formal.use_bf16,
        heatmap_sigma_px=formal.heatmap_sigma_px,
    )
    motion_unchanged = (
        precision_parameter_state_sha256(model.motion_head.state_dict()) == motion_before
    )
    loss_reduction = 1.0 - final.mean_loss / initial.mean_loss
    normalized_uv_improvement = 1.0 - final.normalized_uv_mae / initial.normalized_uv_mae
    passed = (
        final.normalized_uv_mae <= debug.normalized_uv_mae_max
        and normalized_uv_improvement >= debug.normalized_uv_improvement_min
        and final.mask_iou >= debug.mask_iou_min
        and motion_unchanged
    )
    return PrecisionOverfitReceipt(
        passed=passed,
        sample_count=debug.sample_count,
        optimizer_steps=debug.optimizer_steps,
        initial=initial,
        final=final,
        loss_reduction=float(loss_reduction),
        normalized_uv_improvement=float(normalized_uv_improvement),
        motion_head_unchanged=motion_unchanged,
    )


def _validate_dataset_plan(
    audit: PrecisionDatasetAuditReport,
    config: PrecisionExperimentConfig,
) -> None:
    expected = {
        "train": config.dataset.train_trajectories,
        "val": config.dataset.val_trajectories,
        "test": config.dataset.test_trajectories,
    }
    if audit.split_trajectory_counts != expected:
        raise ValueError(
            "Precision Dataset split 数量与冻结配置不一致: "
            f"actual={audit.split_trajectory_counts}, expected={expected}"
        )
    if not audit.passed:
        raise RuntimeError("Precision Dataset audit 未通过，禁止训练")


def train_precision_formal(
    *,
    deployable_root: str | Path,
    label_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
) -> PrecisionTrainingReceipt:
    """先过 real-sample overfit gate，再从新初始化执行正式训练并冻结 checkpoint。"""

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"Precision formal output 已存在，拒绝覆盖: {output}")
    source_identity = source_tree_sha256(repository_root)
    config = load_precision_experiment_config(config_path)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("正式 Precision 训练要求可用且支持 BF16 的 CUDA GPU")
    device = torch.device("cuda")
    audit = audit_precision_dataset(
        deployable_root,
        label_root,
        RobotSpec(),
        write_artifact=True,
    )
    _validate_dataset_plan(audit, config)
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(output / "training_config.json", config.to_dict())
    _atomic_json(output / "dataset_audit.json", audit.to_dict())

    train_dataset = PrecisionRGBDataset(
        deployable_root,
        label_root,
        "train",
        cache_size=config.formal_training.cache_size,
    )
    val_dataset = PrecisionRGBDataset(
        deployable_root,
        label_root,
        "val",
        cache_size=config.formal_training.cache_size,
    )
    first_shape = train_dataset[0]["model_inputs"]["rgb_wrist"].shape[:2]
    expected_shape = (config.dataset.image_height, config.dataset.image_width)
    if first_shape != expected_shape:
        raise ValueError(
            f"Precision image shape 与冻结配置不一致: {first_shape} != {expected_shape}"
        )

    overfit = run_real_sample_overfit_gate(
        train_dataset,
        config,
        device=device,
    )
    _atomic_json(output / "real_sample_overfit.json", overfit.to_dict())
    if not overfit.passed:
        raise RuntimeError("32–128 real-sample overfit gate 未通过，禁止正式训练")

    formal = config.formal_training
    _seed_everything(formal.seed)
    model = _new_model(device)
    motion_before = _freeze_motion_head(model)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=formal.learning_rate,
        weight_decay=formal.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=formal.epochs,
        eta_min=formal.learning_rate * 0.05,
    )
    train_loader = _build_loader(
        train_dataset,
        batch_size=formal.batch_size,
        shuffle=True,
        seed=formal.seed,
        num_workers=formal.num_workers,
    )
    metrics_path = output / "metrics.jsonl"
    best_metric = float("inf")
    best_epoch = 0
    best_state: dict[str, torch.Tensor] | None = None
    examples_seen = 0
    optimizer_steps = 0
    best_examples_seen = 0
    best_optimizer_steps = 0

    for epoch in range(1, formal.epochs + 1):
        model.train()
        weighted_loss = 0.0
        epoch_examples = 0
        gradient_norms: list[float] = []
        for raw_batch in train_loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            height, width = batch["image"].shape[-2:]
            target = _supervision(batch, (height, width), formal.heatmap_sigma_px)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=formal.use_bf16,
            ):
                output_value = model(
                    batch["image"],
                    batch["structured_state"],
                    batch["geometric_motion"],
                )
                loss = precision_unet_loss(
                    output_value,
                    target,
                    PrecisionLossConfig(),
                )
            loss.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                formal.gradient_clip_norm,
            )
            optimizer.step()
            batch_examples = int(batch["image"].shape[0])
            weighted_loss += float(loss.loss.detach().float().item()) * batch_examples
            epoch_examples += batch_examples
            examples_seen += batch_examples
            optimizer_steps += 1
            gradient_norms.append(float(gradient_norm.detach().float().item()))
        val_metrics = evaluate_precision_model(
            model,
            val_dataset,
            device=device,
            batch_size=formal.batch_size,
            use_bf16=formal.use_bf16,
            heatmap_sigma_px=formal.heatmap_sigma_px,
        )
        metric = val_metrics.normalized_uv_mae
        if metric < best_metric:
            best_metric = metric
            best_epoch = epoch
            best_state = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            best_examples_seen = examples_seen
            best_optimizer_steps = optimizer_steps
        record = {
            "epoch": epoch,
            "examples_seen": examples_seen,
            "optimizer_steps": optimizer_steps,
            "train_loss": weighted_loss / epoch_examples,
            "gradient_norm_mean": float(np.mean(gradient_norms)),
            "gradient_norm_max": float(np.max(gradient_norms)),
            "learning_rate": float(optimizer.param_groups[0]["lr"]),
            "validation": val_metrics.to_dict(),
            "selected_so_far": epoch == best_epoch,
        }
        with metrics_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True, allow_nan=False) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        scheduler.step()

    if best_state is None or best_epoch <= 0:
        raise RuntimeError("Precision formal training 没有产生可选 checkpoint")
    model.load_state_dict(best_state, strict=True)
    if precision_parameter_state_sha256(model.motion_head.state_dict()) != motion_before:
        raise RuntimeError("冻结 Motion Head 在正式训练中发生漂移")
    provenance = PrecisionCheckpointProvenance(
        role=PrecisionCheckpointRole.FORMAL_TRAINING,
        data_identity_sha256=audit.dataset_identity_sha256,
        training_config_sha256=config.sha256,
        source_tree_sha256=source_identity,
        seed=formal.seed,
        examples_seen=best_examples_seen,
        optimizer_steps=best_optimizer_steps,
    )
    checkpoint_path = output / "precision-formal.pt"
    checkpoint_receipt = save_precision_checkpoint(checkpoint_path, model, provenance)
    load_precision_checkpoint(
        checkpoint_path,
        expected_checkpoint_sha256=checkpoint_receipt.checkpoint_sha256,
        expected_provenance_sha256=checkpoint_receipt.provenance_sha256,
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    receipt = PrecisionTrainingReceipt(
        selected_epoch=best_epoch,
        selected_metric=best_metric,
        examples_seen=best_examples_seen,
        optimizer_steps=best_optimizer_steps,
        total_optimizer_steps_run=optimizer_steps,
        checkpoint=checkpoint_receipt,
        data_identity_sha256=audit.dataset_identity_sha256,
        training_config_sha256=config.sha256,
        source_tree_sha256=source_identity,
        overfit_gate_passed=overfit.passed,
    )
    _atomic_json(output / "checkpoint_receipt.json", receipt.to_dict())
    return receipt


__all__ = [
    "PRECISION_TRAINING_VERSION",
    "PrecisionDatasetPlan",
    "PrecisionExperimentConfig",
    "PrecisionFormalTrainingConfig",
    "PrecisionHeldOutConfig",
    "PrecisionOverfitConfig",
    "PrecisionOverfitReceipt",
    "PrecisionShadowRolloutConfig",
    "PrecisionSplitMetrics",
    "PrecisionTrainingReceipt",
    "evaluate_precision_model",
    "load_precision_experiment_config",
    "run_real_sample_overfit_gate",
    "source_tree_sha256",
    "train_precision_formal",
]
