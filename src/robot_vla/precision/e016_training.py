"""E016-P1 corrected-observability 正式训练与安全优先 checkpoint 选择。"""

from __future__ import annotations

import json
import math
import os
import tempfile
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from robot_vla.data.trajectory import load_manifest, resolve_trajectory_path
from robot_vla.precision.checkpoint import (
    PrecisionCheckpointProvenance,
    PrecisionCheckpointRole,
    load_precision_checkpoint,
    precision_parameter_state_sha256,
    save_precision_checkpoint,
)
from robot_vla.precision.data import (
    PRECISION_LABEL_SCHEMA_VERSION,
    PRECISION_RGB_DATASET_VERSION,
    canonical_sha256,
    file_sha256,
    load_precision_label_manifest,
)
from robot_vla.precision.e016_pretraining import (
    E016_CORRECTED_LABEL_SCHEMA_VERSION,
    E016_P0_VERSION,
    E016CorrectedPrecisionDataset,
    E016P0Metrics,
    E016P0ObservabilityConfig,
    build_e016_loader,
    build_e016_supervision,
    evaluate_e016_p0_model,
    load_e016_corrected_manifest,
    load_e016_p0_config,
    move_e016_batch_to_device,
    new_e016_frozen_motion_model,
    read_e016_corrected_labels,
    seed_e016_everything,
)
from robot_vla.precision.losses import PrecisionLossConfig, precision_unet_loss
from robot_vla.precision.memory_evaluation import MEMORY_AGE_POLICY, WRITE_THRESHOLD_POLICY
from robot_vla.precision.observability import GOAL_WRITE_SCORE_SEMANTICS
from robot_vla.precision.state_memory import (
    GOAL_MEMORY_UPDATE_POLICY,
    GOAL_POSITION_FRAME_SEMANTICS,
)
from robot_vla.precision.training import source_tree_sha256

E016_P1_VERSION = "e016-p1-formal-precision/v1"
E016_P1_SELECTION_POLICY = "all-observability-guardrails-then-min-goal-uv-mae/v1"
E016_P1_SCHEDULER = "cosine-annealing-eta-min-5-percent/v1"
E016_P1_TEST_POLICY = "single-claim-before-test-label-or-model-read/v1"


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
class E016P1PrerequisiteConfig:
    p0_version: str
    p0_receipt_sha256: str
    p0_training_config_sha256: str
    corrected_sidecar_audit_sha256: str
    corrected_data_identity_sha256: str
    require_p0_passed: bool
    require_p0_checkpoint_absent: bool

    def __post_init__(self) -> None:
        if self.p0_version != E016_P0_VERSION:
            raise ValueError("E016-P1 prerequisite P0 version 漂移")
        for value, name in (
            (self.p0_receipt_sha256, "p0_receipt_sha256"),
            (self.p0_training_config_sha256, "p0_training_config_sha256"),
            (self.corrected_sidecar_audit_sha256, "corrected_sidecar_audit_sha256"),
            (self.corrected_data_identity_sha256, "corrected_data_identity_sha256"),
        ):
            _require_sha256(value, f"prerequisite.{name}")
        if self.require_p0_passed is not True:
            raise ValueError("E016-P1 必须要求 P0 通过")
        if self.require_p0_checkpoint_absent is not True:
            raise ValueError("E016-P1 必须确认 P0 未持久化 checkpoint")


@dataclass(frozen=True)
class E016P1SourceConfig:
    allowed_splits: tuple[str, ...]
    excluded_splits: tuple[str, ...]
    expected_trajectory_counts: dict[str, int]
    expected_sample_counts: dict[str, int]
    image_height: int
    image_width: int
    model_camera: str
    precision_dataset_version: str
    precision_label_schema_version: str
    corrected_label_schema_version: str

    def __post_init__(self) -> None:
        if self.allowed_splits != ("train", "val"):
            raise ValueError("E016-P1 source 只允许 train、val，且顺序必须冻结")
        if self.excluded_splits != ("test",):
            raise ValueError("E016-P1 source 必须显式排除 test")
        expected_splits = set(self.allowed_splits)
        if set(self.expected_trajectory_counts) != expected_splits:
            raise ValueError("source.expected_trajectory_counts 必须只包含 train/val")
        if set(self.expected_sample_counts) != expected_splits:
            raise ValueError("source.expected_sample_counts 必须只包含 train/val")
        for split, count in self.expected_trajectory_counts.items():
            _positive_int(count, f"source.expected_trajectory_counts.{split}")
        for split, count in self.expected_sample_counts.items():
            _positive_int(count, f"source.expected_sample_counts.{split}")
        _positive_int(self.image_height, "source.image_height")
        _positive_int(self.image_width, "source.image_width")
        if self.model_camera != "hand_camera":
            raise ValueError("E016-P1 model_camera 必须是 hand_camera")
        if self.precision_dataset_version != PRECISION_RGB_DATASET_VERSION:
            raise ValueError("E016-P1 Precision Dataset version 漂移")
        if self.precision_label_schema_version != PRECISION_LABEL_SCHEMA_VERSION:
            raise ValueError("E016-P1 source label schema 漂移")
        if self.corrected_label_schema_version != E016_CORRECTED_LABEL_SCHEMA_VERSION:
            raise ValueError("E016-P1 corrected label schema 漂移")


@dataclass(frozen=True)
class E016P1FormalTrainingConfig:
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
    scheduler: str
    initialization: str
    motion_head_policy: str
    selection_metric: str
    selection_tie_break: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.seed, "seed"),
            (self.epochs, "epochs"),
            (self.batch_size, "batch_size"),
            (self.cache_size, "cache_size"),
        ):
            _positive_int(value, f"formal_training.{name}")
        if self.epochs != 20:
            raise ValueError("E016-P1 canonical formal training 固定为 20 epochs")
        if self.num_workers != 0:
            raise ValueError("E016-P1 num_workers 固定为 0，保持读取顺序可审计")
        for value, name in (
            (self.learning_rate, "learning_rate"),
            (self.gradient_clip_norm, "gradient_clip_norm"),
            (self.heatmap_sigma_px, "heatmap_sigma_px"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"formal_training.{name} 必须是有限正数")
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("formal_training.weight_decay 必须有限非负")
        if self.use_bf16 is not True:
            raise ValueError("E016-P1 canonical training 必须启用 BF16")
        if self.scheduler != E016_P1_SCHEDULER:
            raise ValueError("E016-P1 scheduler 漂移")
        if self.initialization != "random-from-scratch":
            raise ValueError("E016-P1 canonical checkpoint 必须从随机初始化训练")
        if self.motion_head_policy != "frozen-zero-shadow-only":
            raise ValueError("E016-P1 Motion Head 必须冻结为 zero-shadow-only")
        if self.selection_metric != "val_goal_observable_normalized_uv_mae":
            raise ValueError("E016-P1 selection metric 漂移")
        if self.selection_tie_break != "earlier_epoch":
            raise ValueError("E016-P1 selection tie-break 必须选更早 epoch")


@dataclass(frozen=True)
class E016P1ValidationSelectionConfig:
    policy: str
    visibility_threshold: float
    projection_threshold: float
    goal_unobservable_false_positive_rate_max: float
    goal_visibility_precision_min: float
    goal_visibility_recall_min: float
    projection_accuracy_min: float
    goal_mask_iou_min: float
    require_all_guardrails: bool

    def __post_init__(self) -> None:
        if self.policy != E016_P1_SELECTION_POLICY:
            raise ValueError("E016-P1 validation selection policy 漂移")
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
            (self.goal_mask_iou_min, "goal_mask_iou_min"),
        ):
            _finite_probability(value, f"validation_selection.{name}")
        if self.require_all_guardrails is not True:
            raise ValueError("E016-P1 checkpoint selection 必须满足全部 guardrails")


@dataclass(frozen=True)
class E016P1FreshHeldOutConfig:
    start_seed: int
    max_candidates: int
    collector_train_trajectories: int
    validation_trajectories: int
    test_trajectories: int
    validation_split: str
    test_split: str
    collector_train_usage: str
    validation_usage: str
    test_policy: str
    legacy_e013_e015_test_reuse_allowed: bool

    def __post_init__(self) -> None:
        if self.start_seed != 134000 or self.max_candidates != 1000:
            raise ValueError("E016-P1 fresh held-out seed 区间漂移")
        for value, name in (
            (self.collector_train_trajectories, "collector_train_trajectories"),
            (self.validation_trajectories, "validation_trajectories"),
            (self.test_trajectories, "test_trajectories"),
        ):
            _positive_int(value, f"fresh_held_out.{name}")
        if (
            self.collector_train_trajectories != 1
            or self.validation_trajectories != 20
            or self.test_trajectories != 100
        ):
            raise ValueError("E016-P1 fresh held-out 轨迹数漂移")
        if self.validation_split != "val" or self.test_split != "test":
            raise ValueError("E016-P1 fresh held-out split 语义漂移")
        if self.collector_train_usage != "collector-contract-only-never-training/v1":
            raise ValueError("fresh collector train usage 漂移")
        if self.validation_usage != "write-threshold-and-memory-age-calibration-only/v1":
            raise ValueError("fresh validation usage 漂移")
        if self.test_policy != E016_P1_TEST_POLICY:
            raise ValueError("E016-P1 test-once policy 漂移")
        if self.legacy_e013_e015_test_reuse_allowed is not False:
            raise ValueError("E016-P1 禁止把 E013/E015 test 重新声明为 unseen test")


@dataclass(frozen=True)
class E016P1MemoryReplayConfig:
    safe_world_xy_error_m: float
    catastrophic_world_xy_error_m: float
    min_goal_mask_probability: float
    max_object_mask_probability: float
    score_semantics: str
    threshold_policy: str
    frame_semantics: str
    update_policy: str
    max_innovation_m: float
    max_position_std_m: float
    require_covariance: bool
    covariance_growth_m2_per_s: float
    max_unobserved_age_candidates_s: tuple[float, ...]
    age_policy: str

    def __post_init__(self) -> None:
        for value, name in (
            (self.safe_world_xy_error_m, "safe_world_xy_error_m"),
            (self.catastrophic_world_xy_error_m, "catastrophic_world_xy_error_m"),
            (self.max_innovation_m, "max_innovation_m"),
            (self.max_position_std_m, "max_position_std_m"),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"memory_replay.{name} 必须是有限正数")
        if self.catastrophic_world_xy_error_m <= self.safe_world_xy_error_m:
            raise ValueError("catastrophic error 必须大于 safe error")
        _finite_probability(
            self.min_goal_mask_probability,
            "memory_replay.min_goal_mask_probability",
        )
        _finite_probability(
            self.max_object_mask_probability,
            "memory_replay.max_object_mask_probability",
        )
        if self.score_semantics != GOAL_WRITE_SCORE_SEMANTICS:
            raise ValueError("E016-P1 goal write score semantics 漂移")
        if self.threshold_policy != WRITE_THRESHOLD_POLICY:
            raise ValueError("E016-P1 write threshold policy 漂移")
        if self.frame_semantics != GOAL_POSITION_FRAME_SEMANTICS:
            raise ValueError("E016-P1 memory 必须存 base-frame state")
        if self.update_policy != GOAL_MEMORY_UPDATE_POLICY:
            raise ValueError("E016-P1 memory update policy 漂移")
        if self.require_covariance is not True:
            raise ValueError("E016-P1 memory write 必须要求 covariance")
        if (
            not math.isfinite(self.covariance_growth_m2_per_s)
            or self.covariance_growth_m2_per_s < 0.0
        ):
            raise ValueError("memory covariance growth 必须有限非负")
        if not self.max_unobserved_age_candidates_s or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.max_unobserved_age_candidates_s
        ):
            raise ValueError("memory max-age candidates 必须是有限正数")
        if tuple(sorted(set(self.max_unobserved_age_candidates_s))) != (
            self.max_unobserved_age_candidates_s
        ):
            raise ValueError("memory max-age candidates 必须严格递增且唯一")
        if self.age_policy != MEMORY_AGE_POLICY:
            raise ValueError("E016-P1 memory age policy 漂移")


@dataclass(frozen=True)
class E016P1SuccessCriteria:
    validation_unsafe_write_count_max: int
    test_unsafe_write_count_max: int
    test_memory_catastrophic_count_max: int
    episode_reset_leakage_count_max: int
    require_memory_unobservable_coverage_improvement: bool
    actuator_promotion_allowed: bool

    def __post_init__(self) -> None:
        for value, name in (
            (self.validation_unsafe_write_count_max, "validation_unsafe_write_count_max"),
            (self.test_unsafe_write_count_max, "test_unsafe_write_count_max"),
            (
                self.test_memory_catastrophic_count_max,
                "test_memory_catastrophic_count_max",
            ),
            (self.episode_reset_leakage_count_max, "episode_reset_leakage_count_max"),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value != 0:
                raise ValueError(f"success_criteria.{name} 必须冻结为 0")
        if self.require_memory_unobservable_coverage_improvement is not True:
            raise ValueError("E016-P1 必须要求 memory 提高不可观察帧 coverage")
        if self.actuator_promotion_allowed is not False:
            raise ValueError("E016-P1 本阶段禁止 actuator promotion")


@dataclass(frozen=True)
class E016P1ExecutionConfig:
    device: str
    required_gpu: str
    use_bf16: bool
    persist_selected_checkpoint_only: bool
    actuation_allowed: bool
    mode: str

    def __post_init__(self) -> None:
        if self.device != "cuda" or self.use_bf16 is not True:
            raise ValueError("E016-P1 必须使用 CUDA/BF16")
        if self.required_gpu != "NVIDIA GeForce RTX 4090":
            raise ValueError("E016-P1 canonical hardware 必须冻结为云端 RTX 4090")
        if self.persist_selected_checkpoint_only is not True:
            raise ValueError("E016-P1 只允许持久化安全门禁后的 selected checkpoint")
        if self.actuation_allowed is not False:
            raise ValueError("E016-P1 禁止 actuation")
        if self.mode != "formal-train-then-shadow-replay":
            raise ValueError("E016-P1 execution mode 漂移")


@dataclass(frozen=True)
class E016P1Config:
    prerequisite: E016P1PrerequisiteConfig
    source: E016P1SourceConfig
    observability: E016P0ObservabilityConfig
    loss: PrecisionLossConfig
    formal_training: E016P1FormalTrainingConfig
    validation_selection: E016P1ValidationSelectionConfig
    fresh_held_out: E016P1FreshHeldOutConfig
    memory_replay: E016P1MemoryReplayConfig
    success_criteria: E016P1SuccessCriteria
    execution: E016P1ExecutionConfig
    version: str = E016_P1_VERSION

    def __post_init__(self) -> None:
        if self.version != E016_P1_VERSION:
            raise ValueError("E016-P1 config version 漂移")
        if self.loss.keypoint_temperature != 1.0:
            raise ValueError("E016-P1 training keypoint temperature 固定为 1.0")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def load_e016_p1_config(path: str | Path) -> E016P1Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("E016-P1 config 必须是 JSON object")
    expected_top = {
        "version",
        "prerequisite",
        "source",
        "observability",
        "loss",
        "formal_training",
        "validation_selection",
        "fresh_held_out",
        "memory_replay",
        "success_criteria",
        "execution",
    }
    _require_exact_keys(payload, expected_top, "E016-P1 config")

    def section(name: str, cls: type[Any]) -> Any:
        value = payload[name]
        if not isinstance(value, Mapping):
            raise TypeError(f"E016-P1 {name} 必须是 object")
        expected = set(cls.__dataclass_fields__)
        _require_exact_keys(value, expected, f"E016-P1 {name}")
        return dict(value)

    prerequisite = section("prerequisite", E016P1PrerequisiteConfig)
    source = section("source", E016P1SourceConfig)
    source["allowed_splits"] = tuple(str(item) for item in source["allowed_splits"])
    source["excluded_splits"] = tuple(str(item) for item in source["excluded_splits"])
    source["expected_trajectory_counts"] = {
        str(key): int(value) for key, value in source["expected_trajectory_counts"].items()
    }
    source["expected_sample_counts"] = {
        str(key): int(value) for key, value in source["expected_sample_counts"].items()
    }
    observability = section("observability", E016P0ObservabilityConfig)
    loss = section("loss", PrecisionLossConfig)
    formal = section("formal_training", E016P1FormalTrainingConfig)
    selection = section("validation_selection", E016P1ValidationSelectionConfig)
    held_out = section("fresh_held_out", E016P1FreshHeldOutConfig)
    memory = section("memory_replay", E016P1MemoryReplayConfig)
    memory["max_unobserved_age_candidates_s"] = tuple(
        float(value) for value in memory["max_unobserved_age_candidates_s"]
    )
    success = section("success_criteria", E016P1SuccessCriteria)
    execution = section("execution", E016P1ExecutionConfig)
    return E016P1Config(
        version=str(payload["version"]),
        prerequisite=E016P1PrerequisiteConfig(**prerequisite),
        source=E016P1SourceConfig(**source),
        observability=E016P0ObservabilityConfig(**observability),
        loss=PrecisionLossConfig(**loss),
        formal_training=E016P1FormalTrainingConfig(**formal),
        validation_selection=E016P1ValidationSelectionConfig(**selection),
        fresh_held_out=E016P1FreshHeldOutConfig(**held_out),
        memory_replay=E016P1MemoryReplayConfig(**memory),
        success_criteria=E016P1SuccessCriteria(**success),
        execution=E016P1ExecutionConfig(**execution),
    )


def _read_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{name} 不存在: {path}")
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise TypeError(f"{name} 必须是 JSON object")
    return value


def validate_e016_p0_prerequisite(
    p0_output_root: str | Path,
    config: E016P1Config,
) -> dict[str, Any]:
    """验证 P0 receipt/config/corrected sidecar 身份，不接受重建或替换。"""

    root = Path(p0_output_root)
    receipt_path = root / "receipt.json"
    audit_path = root / "corrected_sidecar_audit.json"
    config_path = root / "config_snapshot.json"
    if file_sha256(receipt_path) != config.prerequisite.p0_receipt_sha256:
        raise RuntimeError("E016-P0 receipt SHA-256 漂移")
    receipt = _read_json(receipt_path, "E016-P0 receipt")
    if receipt.get("version") != config.prerequisite.p0_version:
        raise RuntimeError("E016-P0 receipt version 漂移")
    if receipt.get("passed") is not True:
        raise RuntimeError("E016-P0 未通过，禁止 P1 正式训练")
    if receipt.get("formal_checkpoint_written") is not False:
        raise RuntimeError("E016-P0 意外持久化 checkpoint")
    if receipt.get("test_split_read") is not False:
        raise RuntimeError("E016-P0 读取过 test，禁止作为 P1 prerequisite")
    if receipt.get("training_config_sha256") != config.prerequisite.p0_training_config_sha256:
        raise RuntimeError("E016-P0 training config identity 漂移")
    if (
        receipt.get("corrected_data_identity_sha256")
        != config.prerequisite.corrected_data_identity_sha256
    ):
        raise RuntimeError("E016-P0 corrected data identity 漂移")
    p0_config = load_e016_p0_config(config_path)
    if p0_config.sha256 != config.prerequisite.p0_training_config_sha256:
        raise RuntimeError("E016-P0 config snapshot SHA-256 漂移")
    if file_sha256(audit_path) != config.prerequisite.corrected_sidecar_audit_sha256:
        raise RuntimeError("E016-P0 corrected sidecar audit SHA-256 漂移")
    audit = _read_json(audit_path, "E016-P0 corrected sidecar audit")
    if (
        audit.get("passed") is not True
        or audit.get("test_label_file_read_count") != 0
        or audit.get("corrected_data_identity_sha256")
        != config.prerequisite.corrected_data_identity_sha256
    ):
        raise RuntimeError("E016-P0 corrected sidecar audit 不满足 P1 前置条件")
    corrected_root = root / "corrected-labels"
    if not corrected_root.is_dir():
        raise FileNotFoundError("E016-P0 corrected-labels 目录不存在")
    if file_sha256(corrected_root / "audit.json") != file_sha256(audit_path):
        raise RuntimeError("E016-P0 corrected sidecar 内外 audit 不一致")
    return {
        "p0_receipt_sha256": file_sha256(receipt_path),
        "p0_training_config_sha256": p0_config.sha256,
        "corrected_sidecar_audit_sha256": file_sha256(audit_path),
        "corrected_data_identity_sha256": audit["corrected_data_identity_sha256"],
        "corrected_label_root": corrected_root,
    }


def _resolve_child_file(root: Path, relative: str) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if not path.is_relative_to(resolved_root) or path.suffix != ".npz":
        raise ValueError("E016-P1 data file 必须是 root 内的 .npz")
    if not path.is_file():
        raise FileNotFoundError(f"E016-P1 data file 不存在: {path}")
    return path


def audit_e016_p1_train_val_inputs(
    *,
    deployable_root: str | Path,
    source_label_root: str | Path,
    corrected_label_root: str | Path,
    config: E016P1Config,
) -> dict[str, Any]:
    """只 hash/验证 train+val NPZ；旧 test array 不会被打开。"""

    deployable = Path(deployable_root)
    source_labels = Path(source_label_root)
    corrected_labels = Path(corrected_label_root)
    allowed = set(config.source.allowed_splits)
    trajectories = [entry for entry in load_manifest(deployable) if entry.split in allowed]
    labels = [
        entry for entry in load_precision_label_manifest(source_labels) if entry.split in allowed
    ]
    corrected = load_e016_corrected_manifest(corrected_labels)
    if any(entry.split not in allowed for entry in corrected):
        raise RuntimeError("E016-P1 corrected manifest 包含被排除 split")
    trajectory_by_id = {entry.trajectory_id: entry for entry in trajectories}
    label_by_id = {entry.trajectory_id: entry for entry in labels}
    corrected_by_id = {entry.trajectory_id: entry for entry in corrected}
    if not (set(trajectory_by_id) == set(label_by_id) == set(corrected_by_id)):
        raise ValueError("E016-P1 train/val source、label、corrected 集合不一致")
    trajectory_counts = Counter(entry.split for entry in trajectories)
    if dict(trajectory_counts) != config.source.expected_trajectory_counts:
        raise ValueError(
            "E016-P1 train/val trajectory 数量漂移: "
            f"actual={dict(trajectory_counts)}, "
            f"expected={config.source.expected_trajectory_counts}"
        )
    scene_splits: dict[str, str] = {}
    sample_counts: Counter[str] = Counter()
    corrected_identity_files: list[dict[str, Any]] = []
    input_identity_files: list[dict[str, Any]] = []
    for trajectory_id in sorted(trajectory_by_id):
        trajectory = trajectory_by_id[trajectory_id]
        label = label_by_id[trajectory_id]
        delta = corrected_by_id[trajectory_id]
        if (
            trajectory.split != label.split
            or trajectory.split != delta.split
            or trajectory.scene_id != label.scene_id
            or trajectory.scene_id != delta.scene_id
            or trajectory.num_steps != label.num_steps
            or trajectory.num_steps != delta.num_steps
        ):
            raise ValueError("E016-P1 train/val metadata 不一致")
        previous = scene_splits.setdefault(trajectory.scene_id, trajectory.split)
        if previous != trajectory.split:
            raise ValueError("E016-P1 scene 跨 train/val split")
        source_path = resolve_trajectory_path(deployable, trajectory.file)
        label_path = _resolve_child_file(source_labels, label.file)
        corrected_path = _resolve_child_file(corrected_labels, delta.file)
        source_sha256 = file_sha256(source_path)
        label_sha256 = file_sha256(label_path)
        corrected_sha256 = file_sha256(corrected_path)
        if source_sha256 != label.source_trajectory_sha256:
            raise RuntimeError("E016-P1 source trajectory 与 label 绑定漂移")
        if label_sha256 != delta.source_label_sha256:
            raise RuntimeError("E016-P1 source label 与 corrected label 绑定漂移")
        arrays = read_e016_corrected_labels(corrected_labels, delta)
        sample_counts[trajectory.split] += arrays.num_steps
        corrected_identity_files.append(
            {
                "trajectory_id": trajectory_id,
                "split": trajectory.split,
                "source_label_sha256": label_sha256,
                "corrected_label_sha256": corrected_sha256,
            }
        )
        input_identity_files.append(
            {
                "trajectory_id": trajectory_id,
                "split": trajectory.split,
                "source_sha256": source_sha256,
                "source_meta": trajectory.to_dict(),
                "label_sha256": label_sha256,
                "label_meta": label.to_dict(),
                "corrected_label_sha256": corrected_sha256,
                "corrected_meta": delta.to_dict(),
            }
        )
    if dict(sample_counts) != config.source.expected_sample_counts:
        raise ValueError(
            "E016-P1 train/val sample 数量漂移: "
            f"actual={dict(sample_counts)}, expected={config.source.expected_sample_counts}"
        )
    corrected_identity = canonical_sha256(
        {
            "schema_version": E016_CORRECTED_LABEL_SCHEMA_VERSION,
            "files": corrected_identity_files,
            "included_splits": list(config.source.allowed_splits),
            "excluded_splits": list(config.source.excluded_splits),
            "observability": asdict(config.observability),
        }
    )
    if corrected_identity != config.prerequisite.corrected_data_identity_sha256:
        raise RuntimeError("E016-P1 corrected sidecar data identity 漂移")
    input_identity = canonical_sha256(
        {
            "version": E016_P1_VERSION,
            "precision_dataset_version": PRECISION_RGB_DATASET_VERSION,
            "precision_label_schema_version": PRECISION_LABEL_SCHEMA_VERSION,
            "corrected_label_schema_version": E016_CORRECTED_LABEL_SCHEMA_VERSION,
            "included_splits": list(config.source.allowed_splits),
            "excluded_splits": list(config.source.excluded_splits),
            "files": input_identity_files,
        }
    )
    return {
        "version": E016_P1_VERSION,
        "passed": True,
        "data_identity_sha256": input_identity,
        "corrected_data_identity_sha256": corrected_identity,
        "included_splits": list(config.source.allowed_splits),
        "excluded_splits": list(config.source.excluded_splits),
        "trajectory_counts": {
            split: trajectory_counts[split] for split in config.source.allowed_splits
        },
        "sample_counts": {split: sample_counts[split] for split in config.source.allowed_splits},
        "test_trajectory_file_read_count": 0,
        "test_label_file_read_count": 0,
        "test_corrected_label_file_read_count": 0,
    }


def e016_p1_validation_guardrails(
    metrics: E016P0Metrics,
    config: E016P1ValidationSelectionConfig,
) -> dict[str, bool]:
    """返回每个 validation safety guardrail 的独立判定。"""

    if not isinstance(metrics, E016P0Metrics):
        raise TypeError("metrics 必须是 E016P0Metrics")
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
        "goal_mask_iou": metrics.goal_mask_iou >= config.goal_mask_iou_min,
    }


def select_e016_p1_checkpoint_epoch(
    candidates: Sequence[tuple[int, E016P0Metrics]],
    config: E016P1ValidationSelectionConfig,
) -> int | None:
    """安全门禁优先；合格 checkpoint 中 UV MAE 最小，同分取更早 epoch。"""

    seen: set[int] = set()
    eligible: list[tuple[float, int]] = []
    for epoch, metrics in candidates:
        _positive_int(epoch, "checkpoint candidate epoch")
        if epoch in seen:
            raise ValueError("checkpoint candidate epoch 重复")
        seen.add(epoch)
        gates = e016_p1_validation_guardrails(metrics, config)
        if all(gates.values()):
            metric = metrics.goal_observable_normalized_uv_mae
            if not math.isfinite(metric) or metric < 0.0:
                raise ValueError("checkpoint selection metric 必须有限非负")
            eligible.append((metric, epoch))
    if not eligible:
        return None
    return min(eligible, key=lambda item: (item[0], item[1]))[1]


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


def run_e016_p1_formal_training(
    *,
    deployable_root: str | Path,
    source_label_root: str | Path,
    p0_output_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    """在 RTX 4090 上训练 20 epochs，只保存安全优先选中的正式 checkpoint。"""

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E016-P1 formal output 已存在，拒绝覆盖: {output}")
    source_identity = source_tree_sha256(repository_root)
    config = load_e016_p1_config(config_path)
    prerequisite = validate_e016_p0_prerequisite(p0_output_root, config)
    corrected_root = Path(prerequisite["corrected_label_root"])
    input_audit = audit_e016_p1_train_val_inputs(
        deployable_root=deployable_root,
        source_label_root=source_label_root,
        corrected_label_root=corrected_root,
        config=config,
    )
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("E016-P1 formal training 要求支持 BF16 的 CUDA GPU")
    device = torch.device(config.execution.device)
    device_name = torch.cuda.get_device_name(device)
    if "RTX 4090" not in device_name:
        raise RuntimeError(f"E016-P1 canonical training 要求 RTX 4090，实际为 {device_name}")

    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(output / "config_snapshot.json", config.to_dict())
    _atomic_json(
        output / "prerequisite_receipt.json",
        prerequisite | {"corrected_label_root": "<private>"},
    )
    _atomic_json(output / "train_val_input_audit.json", input_audit)
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
    train_dataset = E016CorrectedPrecisionDataset(
        deployable_root,
        source_label_root,
        corrected_root,
        "train",
        cache_size=config.formal_training.cache_size,
    )
    val_dataset = E016CorrectedPrecisionDataset(
        deployable_root,
        source_label_root,
        corrected_root,
        "val",
        cache_size=config.formal_training.cache_size,
    )
    first_shape = tuple(train_dataset[0]["model_inputs"]["rgb_wrist"].shape[:2])
    expected_shape = (config.source.image_height, config.source.image_width)
    if first_shape != expected_shape:
        raise ValueError(f"E016-P1 image shape 漂移: {first_shape} != {expected_shape}")

    formal = config.formal_training
    seed_e016_everything(formal.seed)
    model, motion_before = new_e016_frozen_motion_model(device)
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
    train_loader = build_e016_loader(
        train_dataset,
        batch_size=formal.batch_size,
        shuffle=True,
        seed=formal.seed,
    )
    candidates: list[tuple[int, E016P0Metrics]] = []
    states: dict[int, dict[str, torch.Tensor]] = {}
    epoch_counters: dict[int, tuple[int, int]] = {}
    examples_seen = 0
    optimizer_steps = 0
    metrics_path = output / "metrics.jsonl"
    for epoch in range(1, formal.epochs + 1):
        model.train()
        weighted_loss = 0.0
        epoch_examples = 0
        gradient_norms: list[float] = []
        for raw_batch in train_loader:
            batch = move_e016_batch_to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            height, width = batch["image"].shape[-2:]
            target = build_e016_supervision(
                batch,
                (height, width),
                formal.heatmap_sigma_px,
            )
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
                loss = precision_unet_loss(output_value, target, config.loss)
            if not bool(torch.isfinite(loss.loss)):
                raise RuntimeError("E016-P1 formal loss 出现 NaN/Inf")
            loss.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                formal.gradient_clip_norm,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("E016-P1 formal gradient 出现 NaN/Inf")
            optimizer.step()
            count = int(batch["image"].shape[0])
            weighted_loss += float(loss.loss.detach().float().item()) * count
            epoch_examples += count
            examples_seen += count
            optimizer_steps += 1
            gradient_norms.append(float(gradient_norm.detach().float().item()))
        validation = evaluate_e016_p0_model(
            model,
            val_dataset,
            device=device,
            batch_size=formal.batch_size,
            use_bf16=formal.use_bf16,
            heatmap_sigma_px=formal.heatmap_sigma_px,
            loss_config=config.loss,
            visibility_threshold=config.validation_selection.visibility_threshold,
            projection_threshold=config.validation_selection.projection_threshold,
        )
        candidates.append((epoch, validation))
        gates = e016_p1_validation_guardrails(validation, config.validation_selection)
        eligible = all(gates.values())
        if eligible:
            states[epoch] = {
                name: value.detach().cpu().clone() for name, value in model.state_dict().items()
            }
            epoch_counters[epoch] = (examples_seen, optimizer_steps)
        selected_so_far = select_e016_p1_checkpoint_epoch(
            candidates,
            config.validation_selection,
        )
        _append_jsonl(
            metrics_path,
            {
                "epoch": epoch,
                "examples_seen": examples_seen,
                "optimizer_steps": optimizer_steps,
                "train_loss": float(weighted_loss / epoch_examples),
                "gradient_norm_mean": float(np.mean(gradient_norms)),
                "gradient_norm_max": float(np.max(gradient_norms)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validation": validation.to_dict(),
                "validation_guardrails": gates,
                "checkpoint_eligible": eligible,
                "selected_so_far": epoch == selected_so_far,
            },
        )
        scheduler.step()

    selected_epoch = select_e016_p1_checkpoint_epoch(
        candidates,
        config.validation_selection,
    )
    if selected_epoch is None:
        failure = {
            "version": E016_P1_VERSION,
            "passed": False,
            "reason": "no-validation-checkpoint-passed-all-safety-guardrails",
            "formal_checkpoint_written": False,
            "total_epochs_run": formal.epochs,
            "total_optimizer_steps_run": optimizer_steps,
            "test_split_read": False,
            "actuation_allowed": False,
        }
        _atomic_json(output / "failure_receipt.json", failure)
        raise RuntimeError("E016-P1 没有 checkpoint 通过全部 validation safety guardrails")
    selected_metrics = dict(candidates)[selected_epoch]
    model.load_state_dict(states[selected_epoch], strict=True)
    motion_after = precision_parameter_state_sha256(model.motion_head.state_dict())
    if motion_after != motion_before:
        raise RuntimeError("E016-P1 frozen Motion Head 发生漂移")
    selected_examples, selected_steps = epoch_counters[selected_epoch]
    provenance = PrecisionCheckpointProvenance(
        role=PrecisionCheckpointRole.FORMAL_TRAINING,
        data_identity_sha256=input_audit["data_identity_sha256"],
        training_config_sha256=config.sha256,
        source_tree_sha256=source_identity,
        seed=formal.seed,
        examples_seen=selected_examples,
        optimizer_steps=selected_steps,
    )
    checkpoint_path = output / "precision-formal.pt"
    checkpoint_receipt = save_precision_checkpoint(checkpoint_path, model, provenance)
    strict = load_precision_checkpoint(
        checkpoint_path,
        expected_checkpoint_sha256=checkpoint_receipt.checkpoint_sha256,
        expected_provenance_sha256=checkpoint_receipt.provenance_sha256,
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    strict_reload = {
        "passed": True,
        "checkpoint": strict.receipt.to_dict(),
        "provenance": strict.provenance.to_dict(),
        "parameter_state_matches_selected": (
            strict.receipt.parameter_state_sha256
            == precision_parameter_state_sha256(model.state_dict())
        ),
    }
    _atomic_json(output / "strict_reload_receipt.json", strict_reload)
    receipt = {
        "version": E016_P1_VERSION,
        "experiment": "E016-P1 corrected-observability formal training",
        "passed": bool(strict_reload["parameter_state_matches_selected"]),
        "source_tree_sha256": source_identity,
        "training_config_sha256": config.sha256,
        "data_identity_sha256": input_audit["data_identity_sha256"],
        "corrected_data_identity_sha256": input_audit["corrected_data_identity_sha256"],
        "initialization": formal.initialization,
        "selected_epoch": selected_epoch,
        "selected_metric_name": formal.selection_metric,
        "selected_metric": selected_metrics.goal_observable_normalized_uv_mae,
        "selected_validation": selected_metrics.to_dict(),
        "selected_validation_guardrails": e016_p1_validation_guardrails(
            selected_metrics,
            config.validation_selection,
        ),
        "selection_policy": config.validation_selection.policy,
        "selection_tie_break": formal.selection_tie_break,
        "selected_examples_seen": selected_examples,
        "selected_optimizer_steps": selected_steps,
        "total_epochs_run": formal.epochs,
        "total_optimizer_steps_run": optimizer_steps,
        "checkpoint": checkpoint_receipt.to_dict(),
        "strict_reload_passed": bool(strict_reload["parameter_state_matches_selected"]),
        "motion_head_sha256_before": motion_before,
        "motion_head_sha256_after": motion_after,
        "motion_head_unchanged": motion_after == motion_before,
        "included_splits": list(config.source.allowed_splits),
        "excluded_splits": list(config.source.excluded_splits),
        "test_split_read": False,
        "fresh_held_out_read": False,
        "formal_checkpoint_written": True,
        "checkpoint_eligible_for_fresh_held_out": True,
        "actuation_allowed": False,
        "safe_for_actuator_promotion": False,
    }
    _atomic_json(output / "checkpoint_receipt.json", receipt)
    return receipt


__all__ = [
    "E016_P1_SELECTION_POLICY",
    "E016_P1_TEST_POLICY",
    "E016_P1_VERSION",
    "E016P1Config",
    "E016P1FormalTrainingConfig",
    "E016P1FreshHeldOutConfig",
    "E016P1MemoryReplayConfig",
    "E016P1PrerequisiteConfig",
    "E016P1SourceConfig",
    "E016P1ValidationSelectionConfig",
    "audit_e016_p1_train_val_inputs",
    "e016_p1_validation_guardrails",
    "load_e016_p1_config",
    "run_e016_p1_formal_training",
    "select_e016_p1_checkpoint_epoch",
    "validate_e016_p0_prerequisite",
]
