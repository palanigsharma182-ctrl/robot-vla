"""E016-P0 corrected-observability sidecar、分层 overfit 与短 preflight。

E016-P0 只验证训练数据和 loss contract。它不会读取 test split，不会保存可部署
checkpoint，也不会把 privileged mask/geometry 暴露给模型输入。
"""

from __future__ import annotations

import json
import math
import os
import random
import tempfile
from collections import Counter, OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Subset

from robot_vla.precision.checkpoint import precision_parameter_state_sha256
from robot_vla.precision.data import (
    PRECISION_LABEL_SCHEMA_VERSION,
    PRECISION_RGB_DATASET_VERSION,
    PrecisionLabelArrays,
    PrecisionRGBDataset,
    canonical_sha256,
    file_sha256,
    load_precision_label_manifest,
    read_precision_labels,
)
from robot_vla.precision.losses import (
    PrecisionLossConfig,
    PrecisionSupervision,
    build_gaussian_heatmaps,
    precision_unet_loss,
)
from robot_vla.precision.model import PrecisionThreeHeadUNet, PrecisionUNetConfig
from robot_vla.precision.observability import (
    GOAL_OBSERVABILITY_SEMANTICS,
    derive_goal_observability,
)
from robot_vla.precision.training import source_tree_sha256

E016_P0_VERSION = "e016-p0-corrected-observability/v1"
E016_CORRECTED_LABEL_SCHEMA_VERSION = "e016-corrected-observability-sidecar/v1"
E016_GOAL_EXISTS_POLICY = "static-goal-present-every-frame/v1"
E016_GOAL_LOCALIZATION_POLICY = "observable-only/v1"
E016_MASK_SUPERVISION_POLICY = "visible-instance-pixels-independent-of-center/v1"
E016_STRATA = (
    "observable",
    "object_occlusion",
    "other_occlusion_or_background",
    "projection_invalid_or_out_of_frame",
)
E016_OCCLUSION_TYPES = (
    "observable",
    "goal_absent",
    "projection_invalid",
    "out_of_frame",
    "object_occlusion",
    "other_occlusion_or_background",
)
_OCCLUSION_TO_CODE = {name: index for index, name in enumerate(E016_OCCLUSION_TYPES)}
E016_CORRECTED_LABEL_ARRAYS = (
    "source_timestep",
    "goal_exists",
    "goal_projection_valid",
    "goal_in_fov",
    "goal_observable",
    "goal_localization_valid",
    "legacy_goal_visible",
    "center_inside_goal_mask",
    "center_inside_object_mask",
    "local_goal_visible_fraction",
    "goal_mask_area_fraction",
    "occlusion_type_code",
)


def _require_exact_keys(value: Mapping[str, Any], expected: set[str], name: str) -> None:
    actual = set(value)
    if actual != expected:
        raise ValueError(
            f"{name} keys 漂移: missing={sorted(expected - actual)}, "
            f"extra={sorted(actual - expected)}"
        )


def _positive_int(value: int, name: str) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} 必须是正整数")


def _positive_finite(value: float, name: str) -> None:
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"{name} 必须是有限正数")


@dataclass(frozen=True)
class E016P0SourceConfig:
    allowed_splits: tuple[str, ...]
    excluded_splits: tuple[str, ...]
    expected_trajectory_counts: dict[str, int]
    precision_dataset_version: str
    precision_label_schema_version: str

    def __post_init__(self) -> None:
        if self.allowed_splits != ("train", "val"):
            raise ValueError("E016-P0 只允许按 train、val 顺序读取")
        if self.excluded_splits != ("test",):
            raise ValueError("E016-P0 必须显式排除 test split")
        if set(self.expected_trajectory_counts) != set(self.allowed_splits):
            raise ValueError("source.expected_trajectory_counts 必须只覆盖 train/val")
        for split, count in self.expected_trajectory_counts.items():
            _positive_int(count, f"source.expected_trajectory_counts.{split}")
        if self.precision_dataset_version != PRECISION_RGB_DATASET_VERSION:
            raise ValueError("E016-P0 source Precision Dataset version 漂移")
        if self.precision_label_schema_version != PRECISION_LABEL_SCHEMA_VERSION:
            raise ValueError("E016-P0 source Precision label schema 漂移")


@dataclass(frozen=True)
class E016P0ObservabilityConfig:
    goal_exists_policy: str
    semantics: str
    support_radius_px: int
    goal_localization_policy: str
    mask_supervision_policy: str

    def __post_init__(self) -> None:
        if self.goal_exists_policy != E016_GOAL_EXISTS_POLICY:
            raise ValueError("E016-P0 goal_exists policy 漂移")
        if self.semantics != GOAL_OBSERVABILITY_SEMANTICS:
            raise ValueError("E016-P0 observability semantics 漂移")
        if self.support_radius_px != 2:
            raise ValueError("E016-P0 support_radius_px 固定为 2")
        if self.goal_localization_policy != E016_GOAL_LOCALIZATION_POLICY:
            raise ValueError("E016-P0 goal localization policy 漂移")
        if self.mask_supervision_policy != E016_MASK_SUPERVISION_POLICY:
            raise ValueError("E016-P0 mask supervision policy 漂移")


@dataclass(frozen=True)
class E016P0OverfitConfig:
    seed: int
    strata_counts: dict[str, int]
    optimizer_steps: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    heatmap_sigma_px: float
    goal_normalized_uv_mae_max: float
    goal_normalized_uv_improvement_min: float
    goal_mask_iou_min: float
    goal_visibility_precision_min: float
    goal_visibility_recall_min: float
    goal_unobservable_false_positive_rate_max: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.seed, "stratified_overfit.seed"),
            (self.optimizer_steps, "stratified_overfit.optimizer_steps"),
            (self.batch_size, "stratified_overfit.batch_size"),
        ):
            _positive_int(value, name)
        if tuple(self.strata_counts) != E016_STRATA:
            raise ValueError("E016-P0 strata 及其顺序必须保持冻结")
        for stratum, count in self.strata_counts.items():
            _positive_int(count, f"stratified_overfit.strata_counts.{stratum}")
        sample_count = sum(self.strata_counts.values())
        if not 32 <= sample_count <= 128:
            raise ValueError("E016-P0 分层 overfit 样本数必须位于 [32,128]")
        if self.batch_size > sample_count:
            raise ValueError("E016-P0 overfit batch_size 不能超过样本数")
        for value, name in (
            (self.learning_rate, "stratified_overfit.learning_rate"),
            (self.gradient_clip_norm, "stratified_overfit.gradient_clip_norm"),
            (self.heatmap_sigma_px, "stratified_overfit.heatmap_sigma_px"),
        ):
            _positive_finite(value, name)
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("stratified_overfit.weight_decay 必须有限非负")
        for value, name in (
            (self.goal_normalized_uv_mae_max, "goal_normalized_uv_mae_max"),
            (self.goal_normalized_uv_improvement_min, "goal_normalized_uv_improvement_min"),
            (self.goal_mask_iou_min, "goal_mask_iou_min"),
            (self.goal_visibility_precision_min, "goal_visibility_precision_min"),
            (self.goal_visibility_recall_min, "goal_visibility_recall_min"),
            (
                self.goal_unobservable_false_positive_rate_max,
                "goal_unobservable_false_positive_rate_max",
            ),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"stratified_overfit.{name} 必须位于 [0,1]")

    @property
    def sample_count(self) -> int:
        return sum(self.strata_counts.values())


@dataclass(frozen=True)
class E016P0PreflightConfig:
    seed: int
    epochs: int
    batch_size: int
    learning_rate: float
    weight_decay: float
    gradient_clip_norm: float
    heatmap_sigma_px: float

    def __post_init__(self) -> None:
        for value, name in (
            (self.seed, "full_preflight.seed"),
            (self.epochs, "full_preflight.epochs"),
            (self.batch_size, "full_preflight.batch_size"),
        ):
            _positive_int(value, name)
        if not 1 <= self.epochs <= 3:
            raise ValueError("E016-P0 full preflight 只允许 1–3 epochs")
        for value, name in (
            (self.learning_rate, "full_preflight.learning_rate"),
            (self.gradient_clip_norm, "full_preflight.gradient_clip_norm"),
            (self.heatmap_sigma_px, "full_preflight.heatmap_sigma_px"),
        ):
            _positive_finite(value, name)
        if not math.isfinite(self.weight_decay) or self.weight_decay < 0.0:
            raise ValueError("full_preflight.weight_decay 必须有限非负")


@dataclass(frozen=True)
class E016P0ExecutionConfig:
    device: str
    use_bf16: bool
    motion_head_policy: str
    initialization: str
    persist_checkpoint: bool
    actuation_allowed: bool

    def __post_init__(self) -> None:
        if self.device != "cuda" or self.use_bf16 is not True:
            raise ValueError("E016-P0 必须在 CUDA/BF16 执行")
        if self.motion_head_policy != "frozen-zero-shadow-only":
            raise ValueError("E016-P0 Motion Head 必须冻结")
        if self.initialization != "random-from-scratch":
            raise ValueError("E016-P0 必须从随机初始化开始")
        if self.persist_checkpoint is not False:
            raise ValueError("E016-P0 禁止持久化 checkpoint")
        if self.actuation_allowed is not False:
            raise ValueError("E016-P0 禁止 actuation")


@dataclass(frozen=True)
class E016P0Config:
    source: E016P0SourceConfig
    observability: E016P0ObservabilityConfig
    loss: PrecisionLossConfig
    stratified_overfit: E016P0OverfitConfig
    full_preflight: E016P0PreflightConfig
    execution: E016P0ExecutionConfig
    version: str = E016_P0_VERSION

    def __post_init__(self) -> None:
        if self.version != E016_P0_VERSION:
            raise ValueError("E016-P0 config version 漂移")
        if self.stratified_overfit.seed == self.full_preflight.seed:
            raise ValueError("E016-P0 overfit/preflight 必须使用不同随机种子")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @property
    def sha256(self) -> str:
        return canonical_sha256(self.to_dict())


def load_e016_p0_config(path: str | Path) -> E016P0Config:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("E016-P0 config 必须是 JSON object")
    _require_exact_keys(
        payload,
        {
            "version",
            "source",
            "observability",
            "loss",
            "stratified_overfit",
            "full_preflight",
            "execution",
        },
        "E016-P0 config",
    )
    source = dict(payload["source"])
    _require_exact_keys(
        source,
        {
            "allowed_splits",
            "excluded_splits",
            "expected_trajectory_counts",
            "precision_dataset_version",
            "precision_label_schema_version",
        },
        "E016-P0 source",
    )
    source["allowed_splits"] = tuple(str(value) for value in source["allowed_splits"])
    source["excluded_splits"] = tuple(str(value) for value in source["excluded_splits"])
    source["expected_trajectory_counts"] = {
        str(key): int(value) for key, value in source["expected_trajectory_counts"].items()
    }
    observability = dict(payload["observability"])
    _require_exact_keys(
        observability,
        {
            "goal_exists_policy",
            "semantics",
            "support_radius_px",
            "goal_localization_policy",
            "mask_supervision_policy",
        },
        "E016-P0 observability",
    )
    loss = dict(payload["loss"])
    _require_exact_keys(
        loss,
        {
            "heatmap_weight",
            "mask_weight",
            "mask_dice_weight",
            "coordinate_weight",
            "motion_weight",
            "uncertainty_weight",
            "visibility_weight",
            "projection_weight",
            "keypoint_temperature",
        },
        "E016-P0 loss",
    )
    overfit = dict(payload["stratified_overfit"])
    _require_exact_keys(
        overfit,
        {
            "seed",
            "strata_counts",
            "optimizer_steps",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "heatmap_sigma_px",
            "goal_normalized_uv_mae_max",
            "goal_normalized_uv_improvement_min",
            "goal_mask_iou_min",
            "goal_visibility_precision_min",
            "goal_visibility_recall_min",
            "goal_unobservable_false_positive_rate_max",
        },
        "E016-P0 stratified_overfit",
    )
    overfit["strata_counts"] = {
        str(key): int(value) for key, value in overfit["strata_counts"].items()
    }
    preflight = dict(payload["full_preflight"])
    _require_exact_keys(
        preflight,
        {
            "seed",
            "epochs",
            "batch_size",
            "learning_rate",
            "weight_decay",
            "gradient_clip_norm",
            "heatmap_sigma_px",
        },
        "E016-P0 full_preflight",
    )
    execution = dict(payload["execution"])
    _require_exact_keys(
        execution,
        {
            "device",
            "use_bf16",
            "motion_head_policy",
            "initialization",
            "persist_checkpoint",
            "actuation_allowed",
        },
        "E016-P0 execution",
    )
    return E016P0Config(
        version=str(payload["version"]),
        source=E016P0SourceConfig(**source),
        observability=E016P0ObservabilityConfig(**observability),
        loss=PrecisionLossConfig(**loss),
        stratified_overfit=E016P0OverfitConfig(**overfit),
        full_preflight=E016P0PreflightConfig(**preflight),
        execution=E016P0ExecutionConfig(**execution),
    )


@dataclass(frozen=True)
class E016CorrectedLabelMeta:
    trajectory_id: str
    file: str
    split: str
    scene_id: str
    num_steps: int
    source_label_sha256: str
    source_label_schema_version: str = PRECISION_LABEL_SCHEMA_VERSION
    goal_exists_policy: str = E016_GOAL_EXISTS_POLICY
    observability_semantics: str = GOAL_OBSERVABILITY_SEMANTICS
    support_radius_px: int = 2
    schema_version: str = E016_CORRECTED_LABEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("trajectory_id", "file", "scene_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"E016 corrected label {name} 不能为空")
        if self.split not in {"train", "val"}:
            raise ValueError("E016 corrected label 只允许 train/val")
        _positive_int(self.num_steps, "E016 corrected label num_steps")
        if len(self.source_label_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.source_label_sha256
        ):
            raise ValueError("source_label_sha256 必须是 64 位小写 SHA-256")
        if self.source_label_schema_version != PRECISION_LABEL_SCHEMA_VERSION:
            raise ValueError("E016 corrected label source schema 漂移")
        if self.goal_exists_policy != E016_GOAL_EXISTS_POLICY:
            raise ValueError("E016 corrected label goal_exists policy 漂移")
        if self.observability_semantics != GOAL_OBSERVABILITY_SEMANTICS:
            raise ValueError("E016 corrected label observability semantics 漂移")
        if self.support_radius_px != 2:
            raise ValueError("E016 corrected label support radius 漂移")
        if self.schema_version != E016_CORRECTED_LABEL_SCHEMA_VERSION:
            raise ValueError("E016 corrected label schema 漂移")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class E016CorrectedLabelArrays:
    source_timestep: np.ndarray
    goal_exists: np.ndarray
    goal_projection_valid: np.ndarray
    goal_in_fov: np.ndarray
    goal_observable: np.ndarray
    goal_localization_valid: np.ndarray
    legacy_goal_visible: np.ndarray
    center_inside_goal_mask: np.ndarray
    center_inside_object_mask: np.ndarray
    local_goal_visible_fraction: np.ndarray
    goal_mask_area_fraction: np.ndarray
    occlusion_type_code: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.source_timestep.shape[0])

    def occlusion_type(self, timestep: int) -> str:
        return E016_OCCLUSION_TYPES[int(self.occlusion_type_code[timestep])]


def validate_e016_corrected_arrays(
    arrays: E016CorrectedLabelArrays,
    meta: E016CorrectedLabelMeta,
) -> None:
    steps = arrays.num_steps
    if steps != meta.num_steps:
        raise ValueError("E016 corrected num_steps 与 manifest 不一致")
    if arrays.source_timestep.shape != (steps,) or arrays.source_timestep.dtype != np.int64:
        raise ValueError("source_timestep 必须是 int64 [T]")
    if not np.array_equal(arrays.source_timestep, np.arange(steps, dtype=np.int64)):
        raise ValueError("source_timestep 必须完整覆盖 0..T-1")
    bool_names = (
        "goal_exists",
        "goal_projection_valid",
        "goal_in_fov",
        "goal_observable",
        "goal_localization_valid",
        "legacy_goal_visible",
        "center_inside_goal_mask",
        "center_inside_object_mask",
    )
    for name in bool_names:
        value = getattr(arrays, name)
        if value.shape != (steps,) or value.dtype != np.bool_:
            raise ValueError(f"{name} 必须是 bool [T]")
    for name in ("local_goal_visible_fraction", "goal_mask_area_fraction"):
        value = getattr(arrays, name)
        if (
            value.shape != (steps,)
            or value.dtype != np.float32
            or not np.isfinite(value).all()
            or np.any(value < 0.0)
            or np.any(value > 1.0)
        ):
            raise ValueError(f"{name} 必须是 [0,1] float32 [T]")
    if arrays.occlusion_type_code.shape != (steps,) or arrays.occlusion_type_code.dtype != np.uint8:
        raise ValueError("occlusion_type_code 必须是 uint8 [T]")
    if np.any(arrays.occlusion_type_code >= len(E016_OCCLUSION_TYPES)):
        raise ValueError("occlusion_type_code 超出冻结枚举")
    if not bool(arrays.goal_exists.all()):
        raise ValueError("当前 E016-P0 source task 要求静态 goal 每帧存在")
    if np.any(arrays.goal_in_fov & ~arrays.goal_projection_valid):
        raise ValueError("goal_in_fov=true 要求 goal_projection_valid=true")
    expected_observable = (
        arrays.goal_exists
        & arrays.goal_projection_valid
        & arrays.goal_in_fov
        & arrays.center_inside_goal_mask
    )
    if not np.array_equal(arrays.goal_observable, expected_observable):
        raise ValueError("goal_observable 与 corrected 几何语义不一致")
    if not np.array_equal(arrays.goal_localization_valid, arrays.goal_observable):
        raise ValueError("E016-P0 goal localization 必须严格由 observable gate")
    expected_codes = np.empty(steps, dtype=np.uint8)
    for timestep in range(steps):
        if arrays.goal_observable[timestep]:
            name = "observable"
        elif not arrays.goal_exists[timestep]:
            name = "goal_absent"
        elif not arrays.goal_projection_valid[timestep]:
            name = "projection_invalid"
        elif not arrays.goal_in_fov[timestep]:
            name = "out_of_frame"
        elif arrays.center_inside_object_mask[timestep]:
            name = "object_occlusion"
        else:
            name = "other_occlusion_or_background"
        expected_codes[timestep] = _OCCLUSION_TO_CODE[name]
    if not np.array_equal(arrays.occlusion_type_code, expected_codes):
        raise ValueError("occlusion_type_code 与 corrected observability 不一致")


def derive_e016_corrected_arrays(
    labels: PrecisionLabelArrays,
    *,
    support_radius_px: int = 2,
) -> E016CorrectedLabelArrays:
    """从 E013 v1 privileged labels 派生显式 goal observability；不读取 RGB。"""

    steps = labels.num_steps
    values: dict[str, list[Any]] = {
        name: [] for name in E016_CORRECTED_LABEL_ARRAYS if name != "source_timestep"
    }
    for timestep in range(steps):
        projection_valid = bool(labels.keypoint_projection_valid[timestep, 1])
        label = derive_goal_observability(
            goal_exists=True,
            projection_valid=projection_valid,
            projected_normalized_uv=(
                labels.normalized_uv[timestep, 1] if projection_valid else None
            ),
            goal_mask=labels.goal_mask[timestep],
            object_mask=labels.object_mask[timestep],
            legacy_visible=bool(labels.keypoint_visible[timestep, 1]),
            support_radius_px=support_radius_px,
        )
        values["goal_exists"].append(label.goal_exists)
        values["goal_projection_valid"].append(label.projection_valid)
        values["goal_in_fov"].append(label.in_fov)
        values["goal_observable"].append(label.observable)
        values["goal_localization_valid"].append(label.observable)
        values["legacy_goal_visible"].append(label.legacy_visible)
        values["center_inside_goal_mask"].append(label.center_inside_goal_mask)
        values["center_inside_object_mask"].append(label.center_inside_object_mask)
        values["local_goal_visible_fraction"].append(label.local_goal_visible_fraction)
        values["goal_mask_area_fraction"].append(label.goal_mask_area_fraction)
        values["occlusion_type_code"].append(_OCCLUSION_TO_CODE[label.occlusion_type])
    return E016CorrectedLabelArrays(
        source_timestep=np.arange(steps, dtype=np.int64),
        goal_exists=np.asarray(values["goal_exists"], dtype=np.bool_),
        goal_projection_valid=np.asarray(values["goal_projection_valid"], dtype=np.bool_),
        goal_in_fov=np.asarray(values["goal_in_fov"], dtype=np.bool_),
        goal_observable=np.asarray(values["goal_observable"], dtype=np.bool_),
        goal_localization_valid=np.asarray(
            values["goal_localization_valid"], dtype=np.bool_
        ),
        legacy_goal_visible=np.asarray(values["legacy_goal_visible"], dtype=np.bool_),
        center_inside_goal_mask=np.asarray(
            values["center_inside_goal_mask"], dtype=np.bool_
        ),
        center_inside_object_mask=np.asarray(
            values["center_inside_object_mask"], dtype=np.bool_
        ),
        local_goal_visible_fraction=np.asarray(
            values["local_goal_visible_fraction"], dtype=np.float32
        ),
        goal_mask_area_fraction=np.asarray(
            values["goal_mask_area_fraction"], dtype=np.float32
        ),
        occlusion_type_code=np.asarray(values["occlusion_type_code"], dtype=np.uint8),
    )


def _corrected_path(root: Path, relative: str, *, require_file: bool) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if not path.is_relative_to(resolved_root) or path.suffix != ".npz":
        raise ValueError("E016 corrected label 路径必须是 root 内的 .npz")
    if require_file and not path.is_file():
        raise FileNotFoundError(f"找不到 E016 corrected label: {path}")
    return path


def load_e016_corrected_manifest(
    root: str | Path,
    *,
    split: str | None = None,
) -> list[E016CorrectedLabelMeta]:
    path = Path(root) / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 E016 corrected manifest: {path}")
    entries: list[E016CorrectedLabelMeta] = []
    seen: set[str] = set()
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
            if not isinstance(payload, Mapping):
                raise TypeError("manifest row 必须是 object")
            _require_exact_keys(
                payload,
                {
                    "trajectory_id",
                    "file",
                    "split",
                    "scene_id",
                    "num_steps",
                    "source_label_sha256",
                    "source_label_schema_version",
                    "goal_exists_policy",
                    "observability_semantics",
                    "support_radius_px",
                    "schema_version",
                },
                "E016 corrected manifest row",
            )
            entry = E016CorrectedLabelMeta(**payload)
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"E016 corrected manifest 第 {line_number} 行无效: {error}"
            ) from error
        if entry.trajectory_id in seen:
            raise ValueError(f"E016 corrected trajectory_id 重复: {entry.trajectory_id}")
        seen.add(entry.trajectory_id)
        entries.append(entry)
    selected = entries if split is None else [entry for entry in entries if entry.split == split]
    if not selected:
        raise ValueError("E016 corrected manifest 没有匹配条目")
    return selected


def read_e016_corrected_labels(
    root: str | Path,
    meta: E016CorrectedLabelMeta,
) -> E016CorrectedLabelArrays:
    path = _corrected_path(Path(root), meta.file, require_file=True)
    try:
        with np.load(path, allow_pickle=False) as npz:
            missing = [name for name in E016_CORRECTED_LABEL_ARRAYS if name not in npz]
            extra = sorted(set(npz.files) - set(E016_CORRECTED_LABEL_ARRAYS))
            if missing or extra:
                raise ValueError(f"corrected arrays 漂移: missing={missing}, extra={extra}")
            values = {
                name: np.asarray(npz[name]).copy() for name in E016_CORRECTED_LABEL_ARRAYS
            }
    except (OSError, ValueError) as error:
        raise ValueError(f"读取 E016 corrected label {path} 失败: {error}") from error
    arrays = E016CorrectedLabelArrays(**values)
    validate_e016_corrected_arrays(arrays, meta)
    return arrays


class E016CorrectedLabelStore:
    def __init__(self, root: str | Path, *, cache_size: int = 2) -> None:
        if cache_size < 0:
            raise ValueError("E016 corrected cache_size 不能为负数")
        self.root = Path(root)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, E016CorrectedLabelArrays] = OrderedDict()

    def get(self, meta: E016CorrectedLabelMeta) -> E016CorrectedLabelArrays:
        if meta.trajectory_id in self._cache:
            value = self._cache.pop(meta.trajectory_id)
            self._cache[meta.trajectory_id] = value
            return value
        value = read_e016_corrected_labels(self.root, meta)
        if self.cache_size:
            self._cache[meta.trajectory_id] = value
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return value


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


def build_e016_corrected_sidecar(
    source_label_root: str | Path,
    output_root: str | Path,
    config: E016P0Config,
) -> dict[str, Any]:
    """仅从 train/val v1 label 生成 delta-sidecar；test NPZ 永不打开。"""

    source_root = Path(source_label_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E016 corrected sidecar 已存在: {output}")
    source_entries = load_precision_label_manifest(source_root)
    selected = [
        entry for entry in source_entries if entry.split in config.source.allowed_splits
    ]
    actual_counts = Counter(entry.split for entry in selected)
    if dict(actual_counts) != config.source.expected_trajectory_counts:
        raise ValueError(
            "E016 source train/val trajectory 数量漂移: "
            f"actual={dict(actual_counts)}, expected={config.source.expected_trajectory_counts}"
        )
    if any(entry.split in config.source.excluded_splits for entry in selected):
        raise RuntimeError("E016-P0 selected source 意外包含 test")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))
    try:
        labels_dir = temporary_root / "labels"
        labels_dir.mkdir(mode=0o700)
        corrected_meta: list[E016CorrectedLabelMeta] = []
        file_records: list[dict[str, Any]] = []
        split_samples: Counter[str] = Counter()
        occlusions: Counter[str] = Counter()
        observable_count = 0
        legacy_visible_count = 0
        mismatch_count = 0
        for source_meta in sorted(selected, key=lambda item: item.trajectory_id):
            source_arrays = read_precision_labels(source_root, source_meta)
            source_path = (source_root / source_meta.file).resolve()
            source_sha256 = file_sha256(source_path)
            meta = E016CorrectedLabelMeta(
                trajectory_id=source_meta.trajectory_id,
                file=f"labels/{source_meta.trajectory_id}.npz",
                split=source_meta.split,
                scene_id=source_meta.scene_id,
                num_steps=source_meta.num_steps,
                source_label_sha256=source_sha256,
                support_radius_px=config.observability.support_radius_px,
            )
            arrays = derive_e016_corrected_arrays(
                source_arrays,
                support_radius_px=config.observability.support_radius_px,
            )
            validate_e016_corrected_arrays(arrays, meta)
            target = _corrected_path(temporary_root, meta.file, require_file=False)
            with target.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    **{
                        name: getattr(arrays, name)
                        for name in E016_CORRECTED_LABEL_ARRAYS
                    },
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(target, 0o600)
            corrected_sha256 = file_sha256(target)
            corrected_meta.append(meta)
            file_records.append(
                {
                    "trajectory_id": meta.trajectory_id,
                    "split": meta.split,
                    "source_label_sha256": source_sha256,
                    "corrected_label_sha256": corrected_sha256,
                }
            )
            split_samples[meta.split] += arrays.num_steps
            observable_count += int(arrays.goal_observable.sum())
            legacy_visible_count += int(arrays.legacy_goal_visible.sum())
            mismatch_count += int(
                (arrays.legacy_goal_visible & ~arrays.goal_observable).sum()
            )
            occlusions.update(
                arrays.occlusion_type(timestep) for timestep in range(arrays.num_steps)
            )
        manifest_payload = "".join(
            json.dumps(meta.to_dict(), sort_keys=True, allow_nan=False) + "\n"
            for meta in corrected_meta
        )
        manifest_path = temporary_root / "manifest.jsonl"
        manifest_path.write_text(manifest_payload, encoding="utf-8")
        os.chmod(manifest_path, 0o600)
        total = sum(split_samples.values())
        identity = canonical_sha256(
            {
                "schema_version": E016_CORRECTED_LABEL_SCHEMA_VERSION,
                "files": file_records,
                "included_splits": list(config.source.allowed_splits),
                "excluded_splits": list(config.source.excluded_splits),
                "observability": asdict(config.observability),
            }
        )
        audit = {
            "version": E016_P0_VERSION,
            "schema_version": E016_CORRECTED_LABEL_SCHEMA_VERSION,
            "passed": True,
            "corrected_data_identity_sha256": identity,
            "manifest_sha256": file_sha256(manifest_path),
            "included_splits": list(config.source.allowed_splits),
            "excluded_splits": list(config.source.excluded_splits),
            "test_label_file_read_count": 0,
            "trajectory_counts": {
                split: actual_counts[split] for split in config.source.allowed_splits
            },
            "sample_counts": {
                split: split_samples[split] for split in config.source.allowed_splits
            },
            "sample_count": total,
            "goal_exists_count": total,
            "goal_projection_valid_count": sum(
                count
                for name, count in occlusions.items()
                if name not in {"projection_invalid", "goal_absent"}
            ),
            "goal_observable_count": observable_count,
            "goal_unobservable_count": total - observable_count,
            "legacy_goal_visible_count": legacy_visible_count,
            "legacy_visible_but_unobservable_count": mismatch_count,
            "occlusion_counts": {
                name: occlusions[name] for name in E016_OCCLUSION_TYPES
            },
            "goal_exists_policy": E016_GOAL_EXISTS_POLICY,
            "observability_semantics": GOAL_OBSERVABILITY_SEMANTICS,
            "goal_localization_policy": E016_GOAL_LOCALIZATION_POLICY,
            "mask_supervision_policy": E016_MASK_SUPERVISION_POLICY,
        }
        _atomic_json(temporary_root / "audit.json", audit)
        os.replace(temporary_root, output)
        temporary_root = Path()
        return audit
    finally:
        if temporary_root != Path() and temporary_root.exists():
            import shutil

            shutil.rmtree(temporary_root)


class E016CorrectedPrecisionDataset(Dataset[dict[str, Any]]):
    """E013 RGB Dataset + E016 corrected delta-sidecar；模型输入保持不变。"""

    def __init__(
        self,
        deployable_root: str | Path,
        source_label_root: str | Path,
        corrected_label_root: str | Path,
        split: str,
        *,
        cache_size: int = 32,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError("E016-P0 Dataset 明确禁止读取 test split")
        self.split = split
        self.source_label_root = Path(source_label_root)
        self.base = PrecisionRGBDataset(
            deployable_root,
            source_label_root,
            split,
            cache_size=cache_size,
        )
        corrected_entries = load_e016_corrected_manifest(
            corrected_label_root,
            split=split,
        )
        self.corrected_by_trajectory = {
            entry.trajectory_id: entry for entry in corrected_entries
        }
        source_ids = {entry.trajectory_id for entry in self.base.base.entries}
        if source_ids != set(self.corrected_by_trajectory):
            raise ValueError("E016 source/corrected trajectory 集合不一致")
        for trajectory_id in sorted(source_ids):
            source_meta = self.base.label_by_trajectory[trajectory_id]
            corrected_meta = self.corrected_by_trajectory[trajectory_id]
            if (
                source_meta.split != corrected_meta.split
                or source_meta.scene_id != corrected_meta.scene_id
                or source_meta.num_steps != corrected_meta.num_steps
            ):
                raise ValueError("E016 source/corrected metadata 不一致")
            source_path = (self.source_label_root / source_meta.file).resolve()
            if file_sha256(source_path) != corrected_meta.source_label_sha256:
                raise RuntimeError("E016 corrected sidecar 绑定的 source label 已漂移")
        self.corrected_store = E016CorrectedLabelStore(
            corrected_label_root,
            cache_size=cache_size,
        )

    def __len__(self) -> int:
        return len(self.base)

    def _identity(self, index: int) -> tuple[str, int]:
        entry_index, timestep = self.base.base.index[index]
        trajectory_id = self.base.base.entries[entry_index].trajectory_id
        return trajectory_id, int(timestep)

    def sampling_metadata(self, index: int) -> dict[str, Any]:
        trajectory_id, timestep = self._identity(index)
        meta = self.corrected_by_trajectory[trajectory_id]
        corrected = self.corrected_store.get(meta)
        return {
            "dataset_index": index,
            "trajectory_id": trajectory_id,
            "timestep": timestep,
            "split": self.split,
            "goal_observable": bool(corrected.goal_observable[timestep]),
            "legacy_goal_visible": bool(corrected.legacy_goal_visible[timestep]),
            "occlusion_type": corrected.occlusion_type(timestep),
        }

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        trajectory_id, timestep = self._identity(index)
        meta = self.corrected_by_trajectory[trajectory_id]
        corrected = self.corrected_store.get(meta)
        supervision = dict(sample["supervision"])
        localization_valid = supervision["keypoint_valid"].copy()
        observable = localization_valid.copy()
        localization_valid[1] = corrected.goal_localization_valid[timestep]
        observable[1] = corrected.goal_observable[timestep]
        normalized_uv = supervision["normalized_uv_targets"].copy()
        if not localization_valid[1]:
            # 不可观察帧不携带 goal-center 伪 target；loss mask 提供第二重保护。
            normalized_uv[1] = 0.0
        supervision["normalized_uv_targets"] = normalized_uv
        supervision["keypoint_valid"] = localization_valid
        supervision["keypoint_observable"] = observable
        audit = dict(sample["audit"])
        audit.update(
            {
                "goal_exists": bool(corrected.goal_exists[timestep]),
                "goal_projection_valid": bool(
                    corrected.goal_projection_valid[timestep]
                ),
                "goal_in_fov": bool(corrected.goal_in_fov[timestep]),
                "goal_observable": bool(corrected.goal_observable[timestep]),
                "legacy_goal_visible": bool(corrected.legacy_goal_visible[timestep]),
                "occlusion_type": corrected.occlusion_type(timestep),
            }
        )
        return {
            "model_inputs": sample["model_inputs"],
            "supervision": supervision,
            "audit": audit,
        }


def _stratum(occlusion_type: str) -> str:
    if occlusion_type in {"projection_invalid", "out_of_frame"}:
        return "projection_invalid_or_out_of_frame"
    if occlusion_type in E016_STRATA:
        return occlusion_type
    raise ValueError(f"E016-P0 不支持的训练 stratum: {occlusion_type}")


def select_stratified_overfit_indices(
    dataset: E016CorrectedPrecisionDataset,
    config: E016P0OverfitConfig,
) -> tuple[list[int], list[dict[str, Any]]]:
    if dataset.split != "train":
        raise ValueError("E016-P0 stratified overfit 只能从 train 取样")
    candidates: dict[str, list[int]] = defaultdict(list)
    metadata: dict[int, dict[str, Any]] = {}
    for index in range(len(dataset)):
        item = dataset.sampling_metadata(index)
        stratum = _stratum(str(item["occlusion_type"]))
        candidates[stratum].append(index)
        metadata[index] = item
    generator = np.random.default_rng(config.seed)
    selected: list[int] = []
    rows: list[dict[str, Any]] = []
    for stratum in E016_STRATA:
        required = config.strata_counts[stratum]
        available = candidates[stratum]
        if len(available) < required:
            raise RuntimeError(
                f"E016-P0 stratum {stratum} 样本不足: {len(available)} < {required}"
            )
        chosen = np.sort(
            generator.choice(available, size=required, replace=False)
        ).tolist()
        selected.extend(chosen)
        for index in chosen:
            rows.append({**metadata[index], "stratum": stratum})
    if len(selected) != config.sample_count or len(set(selected)) != len(selected):
        raise RuntimeError("E016-P0 stratified subset 数量或唯一性漂移")
    return selected, rows


def _collate(samples: Sequence[dict[str, Any]]) -> dict[str, Any]:
    model_inputs = [sample["model_inputs"] for sample in samples]
    supervision = [sample["supervision"] for sample in samples]
    image = np.stack([item["rgb_wrist"] for item in model_inputs])
    return {
        "image": torch.from_numpy(
            np.ascontiguousarray(
                image.transpose(0, 3, 1, 2), dtype=np.float32
            )
            / np.float32(255.0)
        ),
        "structured_state": torch.from_numpy(
            np.stack([item["structured_state"] for item in model_inputs])
        ),
        "geometric_motion": torch.from_numpy(
            np.stack([item["geometric_motion"] for item in model_inputs])
        ),
        "mask_targets": torch.from_numpy(
            np.stack([item["mask_targets"] for item in supervision])
        ),
        "normalized_uv_targets": torch.from_numpy(
            np.stack([item["normalized_uv_targets"] for item in supervision])
        ),
        "keypoint_valid": torch.from_numpy(
            np.stack([item["keypoint_valid"] for item in supervision])
        ),
        "keypoint_observable": torch.from_numpy(
            np.stack([item["keypoint_observable"] for item in supervision])
        ),
        "motion_residual_targets": torch.from_numpy(
            np.stack([item["motion_residual_targets"] for item in supervision])
        ),
        "motion_valid": torch.from_numpy(
            np.stack([item["motion_valid"] for item in supervision])
        ),
        "projection_valid": torch.from_numpy(
            np.stack([item["projection_valid"] for item in supervision])
        ),
        "audit": [sample["audit"] for sample in samples],
    }


def _loader(
    dataset: Dataset[dict[str, Any]],
    *,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader[dict[str, Any]]:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        collate_fn=_collate,
        generator=generator,
        drop_last=False,
        pin_memory=False,
    )


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
    return PrecisionSupervision(
        heatmap_targets=build_gaussian_heatmaps(
            batch["normalized_uv_targets"],
            batch["keypoint_valid"],
            image_size_hw,
            sigma_px=sigma_px,
        ),
        mask_targets=batch["mask_targets"],
        normalized_uv_targets=batch["normalized_uv_targets"],
        keypoint_valid=batch["keypoint_valid"],
        motion_residual_targets=batch["motion_residual_targets"],
        motion_valid=batch["motion_valid"],
        projection_valid=batch["projection_valid"],
        keypoint_observable=batch["keypoint_observable"],
    )


@dataclass(frozen=True)
class E016P0Metrics:
    sample_count: int
    object_localization_valid_count: int
    goal_observable_count: int
    goal_unobservable_count: int
    mean_loss: float
    mean_heatmap_loss: float
    mean_mask_loss: float
    mean_coordinate_loss: float
    mean_uncertainty_loss: float
    mean_visibility_loss: float
    mean_projection_loss: float
    object_normalized_uv_mae: float
    goal_observable_normalized_uv_mae: float
    goal_observable_pixel_error_p50: float
    goal_observable_pixel_error_p90: float
    object_mask_iou: float
    goal_mask_iou: float
    goal_visibility_true_positive: int
    goal_visibility_false_positive: int
    goal_visibility_true_negative: int
    goal_visibility_false_negative: int
    goal_visibility_precision: float
    goal_visibility_recall: float
    goal_visibility_f1: float
    goal_unobservable_false_positive_rate: float
    projection_accuracy: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio(numerator: int, denominator: int, *, empty: float = 0.0) -> float:
    return float(numerator / denominator) if denominator else float(empty)


def evaluate_e016_p0_model(
    model: PrecisionThreeHeadUNet,
    dataset: Dataset[dict[str, Any]],
    *,
    device: torch.device,
    batch_size: int,
    use_bf16: bool,
    heatmap_sigma_px: float,
    loss_config: PrecisionLossConfig,
) -> E016P0Metrics:
    loader = _loader(dataset, batch_size=batch_size, shuffle=False, seed=0)
    was_training = model.training
    model.eval()
    loss_sums = Counter()
    sample_count = 0
    object_errors: list[np.ndarray] = []
    goal_errors: list[np.ndarray] = []
    goal_pixel_errors: list[np.ndarray] = []
    object_valid_count = 0
    goal_observable_count = 0
    intersections = np.zeros(2, dtype=np.int64)
    unions = np.zeros(2, dtype=np.int64)
    true_positive = false_positive = true_negative = false_negative = 0
    projection_correct = projection_total = 0
    with torch.inference_mode():
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            height, width = batch["image"].shape[-2:]
            target = _supervision(batch, (height, width), heatmap_sigma_px)
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
                loss = precision_unet_loss(output, target, loss_config)
            decoded = output.decode_for_control()
            batch_size_actual = int(batch["image"].shape[0])
            sample_count += batch_size_actual
            for name in (
                "loss",
                "heatmap_loss",
                "mask_loss",
                "coordinate_loss",
                "uncertainty_loss",
                "visibility_loss",
                "projection_loss",
            ):
                loss_sums[name] += (
                    float(getattr(loss, name).detach().float().item()) * batch_size_actual
                )
            error = (
                decoded.keypoints.normalized_uv
                - batch["normalized_uv_targets"].float()
            ).abs()
            object_valid = batch["keypoint_valid"][:, 0]
            goal_valid = batch["keypoint_valid"][:, 1]
            object_valid_count += int(object_valid.sum().item())
            goal_observable_count += int(goal_valid.sum().item())
            if bool(object_valid.any()):
                object_errors.append(error[:, 0][object_valid].detach().cpu().numpy())
            if bool(goal_valid.any()):
                goal_value = error[:, 1][goal_valid]
                goal_errors.append(goal_value.detach().cpu().numpy())
                scale = torch.tensor(
                    (float(width), float(height)),
                    dtype=torch.float32,
                    device=device,
                )
                goal_pixel_errors.append(
                    torch.linalg.norm(goal_value * scale, dim=-1).detach().cpu().numpy()
                )
            predicted_mask = output.mask_logits > 0.0
            target_mask = batch["mask_targets"] > 0.5
            for index in range(2):
                intersections[index] += int(
                    (predicted_mask[:, index] & target_mask[:, index]).sum().item()
                )
                unions[index] += int(
                    (predicted_mask[:, index] | target_mask[:, index]).sum().item()
                )
            goal_target = batch["keypoint_observable"][:, 1]
            goal_predicted = decoded.visibility_probability[:, 1] >= 0.5
            true_positive += int((goal_predicted & goal_target).sum().item())
            false_positive += int((goal_predicted & ~goal_target).sum().item())
            true_negative += int((~goal_predicted & ~goal_target).sum().item())
            false_negative += int((~goal_predicted & goal_target).sum().item())
            projection_predicted = decoded.projection_validity_probability >= 0.5
            projection_correct += int(
                (projection_predicted == batch["projection_valid"]).sum().item()
            )
            projection_total += int(batch["projection_valid"].numel())
    if was_training:
        model.train()
    if not object_errors or not goal_errors or not goal_pixel_errors:
        raise RuntimeError("E016-P0 evaluation 缺少有效 object/goal localization 样本")
    object_values = np.concatenate(object_errors)
    goal_values = np.concatenate(goal_errors)
    goal_pixels = np.concatenate(goal_pixel_errors)
    precision = _ratio(true_positive, true_positive + false_positive)
    recall = _ratio(true_positive, true_positive + false_negative)
    return E016P0Metrics(
        sample_count=sample_count,
        object_localization_valid_count=object_valid_count,
        goal_observable_count=goal_observable_count,
        goal_unobservable_count=sample_count - goal_observable_count,
        mean_loss=float(loss_sums["loss"] / sample_count),
        mean_heatmap_loss=float(loss_sums["heatmap_loss"] / sample_count),
        mean_mask_loss=float(loss_sums["mask_loss"] / sample_count),
        mean_coordinate_loss=float(loss_sums["coordinate_loss"] / sample_count),
        mean_uncertainty_loss=float(loss_sums["uncertainty_loss"] / sample_count),
        mean_visibility_loss=float(loss_sums["visibility_loss"] / sample_count),
        mean_projection_loss=float(loss_sums["projection_loss"] / sample_count),
        object_normalized_uv_mae=float(np.mean(object_values)),
        goal_observable_normalized_uv_mae=float(np.mean(goal_values)),
        goal_observable_pixel_error_p50=float(np.quantile(goal_pixels, 0.50)),
        goal_observable_pixel_error_p90=float(np.quantile(goal_pixels, 0.90)),
        object_mask_iou=_ratio(int(intersections[0]), int(unions[0]), empty=1.0),
        goal_mask_iou=_ratio(int(intersections[1]), int(unions[1]), empty=1.0),
        goal_visibility_true_positive=true_positive,
        goal_visibility_false_positive=false_positive,
        goal_visibility_true_negative=true_negative,
        goal_visibility_false_negative=false_negative,
        goal_visibility_precision=precision,
        goal_visibility_recall=recall,
        goal_visibility_f1=(
            float(2.0 * precision * recall / (precision + recall))
            if precision + recall
            else 0.0
        ),
        goal_unobservable_false_positive_rate=_ratio(
            false_positive,
            false_positive + true_negative,
        ),
        projection_accuracy=_ratio(projection_correct, projection_total),
    )


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _new_frozen_motion_model(
    device: torch.device,
) -> tuple[PrecisionThreeHeadUNet, str]:
    model = PrecisionThreeHeadUNet(PrecisionUNetConfig()).to(device)
    model.motion_head.requires_grad_(False)
    motion_hash = precision_parameter_state_sha256(model.motion_head.state_dict())
    return model, motion_hash


def run_e016_loss_contract_probe(device: torch.device) -> dict[str, Any]:
    """证明 all-negative localization 不产生 loss/gradient/NaN。"""

    _seed_everything(16015)
    model, _ = _new_frozen_motion_model(device)
    model.train()
    config = model.config
    image = torch.zeros((2, 3, 16, 16), device=device)
    state = torch.zeros((2, config.structured_state_dim), device=device)
    geometry = torch.zeros((2, config.motion_spec.motion_dim), device=device)
    output = model(image, state, geometry)
    output.heatmap_logits.retain_grad()
    output.subpixel_offsets.retain_grad()
    output.keypoint_log_variance.retain_grad()
    valid = torch.zeros((2, config.keypoint_count), dtype=torch.bool, device=device)
    target = PrecisionSupervision(
        heatmap_targets=build_gaussian_heatmaps(
            torch.zeros((2, config.keypoint_count, 2), device=device),
            valid,
            (16, 16),
        ),
        mask_targets=torch.zeros_like(output.mask_logits),
        normalized_uv_targets=torch.zeros(
            (2, config.keypoint_count, 2), device=device
        ),
        keypoint_valid=valid,
        motion_residual_targets=torch.zeros_like(output.motion_residual),
        motion_valid=torch.zeros_like(output.motion_residual, dtype=torch.bool),
        projection_valid=torch.zeros(2, dtype=torch.bool, device=device),
        keypoint_observable=valid.clone(),
    )
    localization_only = PrecisionLossConfig(
        heatmap_weight=1.0,
        mask_weight=0.0,
        coordinate_weight=2.0,
        motion_weight=0.0,
        uncertainty_weight=0.1,
        visibility_weight=0.0,
        projection_weight=0.0,
    )
    loss = precision_unet_loss(output, target, localization_only)
    loss.loss.backward()

    def maximum_gradient(value: torch.Tensor) -> float:
        if value.grad is None:
            return 0.0
        return float(value.grad.detach().abs().max().item())

    gradients = {
        "heatmap_logits_abs_max": maximum_gradient(output.heatmap_logits),
        "subpixel_offsets_abs_max": maximum_gradient(output.subpixel_offsets),
        "keypoint_log_variance_abs_max": maximum_gradient(
            output.keypoint_log_variance
        ),
    }
    losses = {
        "total": float(loss.loss.detach().float().item()),
        "heatmap": float(loss.heatmap_loss.detach().float().item()),
        "coordinate": float(loss.coordinate_loss.detach().float().item()),
        "uncertainty": float(loss.uncertainty_loss.detach().float().item()),
    }
    passed = all(math.isfinite(value) and value == 0.0 for value in losses.values()) and all(
        value == 0.0 for value in gradients.values()
    )
    return {
        "version": E016_P0_VERSION,
        "passed": passed,
        "all_negative_batch_finite": all(math.isfinite(value) for value in losses.values()),
        "localization_losses": losses,
        "localization_output_gradient_abs_max": gradients,
    }


def _train_optimizer_steps(
    model: PrecisionThreeHeadUNet,
    dataset: Dataset[dict[str, Any]],
    *,
    device: torch.device,
    seed: int,
    optimizer_steps: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    heatmap_sigma_px: float,
    use_bf16: bool,
    loss_config: PrecisionLossConfig,
) -> tuple[list[dict[str, float | int]], int]:
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    loader = _loader(dataset, batch_size=batch_size, shuffle=True, seed=seed)
    iterator = iter(loader)
    trace: list[dict[str, float | int]] = []
    examples_seen = 0
    model.train()
    for step in range(1, optimizer_steps + 1):
        try:
            raw_batch = next(iterator)
        except StopIteration:
            iterator = iter(loader)
            raw_batch = next(iterator)
        batch = _to_device(raw_batch, device)
        optimizer.zero_grad(set_to_none=True)
        height, width = batch["image"].shape[-2:]
        target = _supervision(batch, (height, width), heatmap_sigma_px)
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
            loss = precision_unet_loss(output, target, loss_config)
        if not bool(torch.isfinite(loss.loss)):
            raise RuntimeError(f"E016-P0 step {step} loss 出现 NaN/Inf")
        loss.loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, gradient_clip_norm)
        if not bool(torch.isfinite(gradient_norm)):
            raise RuntimeError(f"E016-P0 step {step} gradient 出现 NaN/Inf")
        optimizer.step()
        batch_examples = int(batch["image"].shape[0])
        examples_seen += batch_examples
        if step == 1 or step % 25 == 0 or step == optimizer_steps:
            trace.append(
                {
                    "optimizer_step": step,
                    "examples_seen": examples_seen,
                    "loss": float(loss.loss.detach().float().item()),
                    "gradient_norm": float(gradient_norm.detach().float().item()),
                }
            )
    return trace, examples_seen


def run_e016_stratified_overfit(
    train_dataset: E016CorrectedPrecisionDataset,
    selected_indices: Sequence[int],
    config: E016P0Config,
    *,
    device: torch.device,
) -> dict[str, Any]:
    overfit = config.stratified_overfit
    _seed_everything(overfit.seed)
    subset = Subset(train_dataset, list(selected_indices))
    model, motion_before = _new_frozen_motion_model(device)
    initial = evaluate_e016_p0_model(
        model,
        subset,
        device=device,
        batch_size=overfit.batch_size,
        use_bf16=config.execution.use_bf16,
        heatmap_sigma_px=overfit.heatmap_sigma_px,
        loss_config=config.loss,
    )
    trace, examples_seen = _train_optimizer_steps(
        model,
        subset,
        device=device,
        seed=overfit.seed,
        optimizer_steps=overfit.optimizer_steps,
        batch_size=overfit.batch_size,
        learning_rate=overfit.learning_rate,
        weight_decay=overfit.weight_decay,
        gradient_clip_norm=overfit.gradient_clip_norm,
        heatmap_sigma_px=overfit.heatmap_sigma_px,
        use_bf16=config.execution.use_bf16,
        loss_config=config.loss,
    )
    final = evaluate_e016_p0_model(
        model,
        subset,
        device=device,
        batch_size=overfit.batch_size,
        use_bf16=config.execution.use_bf16,
        heatmap_sigma_px=overfit.heatmap_sigma_px,
        loss_config=config.loss,
    )
    motion_after = precision_parameter_state_sha256(model.motion_head.state_dict())
    improvement = 1.0 - (
        final.goal_observable_normalized_uv_mae
        / initial.goal_observable_normalized_uv_mae
    )
    gates = {
        "goal_normalized_uv_mae": (
            final.goal_observable_normalized_uv_mae
            <= overfit.goal_normalized_uv_mae_max
        ),
        "goal_normalized_uv_improvement": (
            improvement >= overfit.goal_normalized_uv_improvement_min
        ),
        "goal_mask_iou": final.goal_mask_iou >= overfit.goal_mask_iou_min,
        "goal_visibility_precision": (
            final.goal_visibility_precision >= overfit.goal_visibility_precision_min
        ),
        "goal_visibility_recall": (
            final.goal_visibility_recall >= overfit.goal_visibility_recall_min
        ),
        "goal_unobservable_false_positive_rate": (
            final.goal_unobservable_false_positive_rate
            <= overfit.goal_unobservable_false_positive_rate_max
        ),
        "motion_head_unchanged": motion_after == motion_before,
    }
    return {
        "version": E016_P0_VERSION,
        "passed": all(gates.values()),
        "sample_count": len(selected_indices),
        "optimizer_steps": overfit.optimizer_steps,
        "examples_seen": examples_seen,
        "initialization": config.execution.initialization,
        "initial": initial.to_dict(),
        "final": final.to_dict(),
        "loss_reduction": float(1.0 - final.mean_loss / initial.mean_loss),
        "goal_normalized_uv_improvement": float(improvement),
        "gates": gates,
        "motion_head_sha256_before": motion_before,
        "motion_head_sha256_after": motion_after,
        "final_parameter_state_sha256": precision_parameter_state_sha256(
            model.state_dict()
        ),
        "checkpoint_persisted": False,
        "training_trace": trace,
    }


def run_e016_full_preflight(
    train_dataset: E016CorrectedPrecisionDataset,
    val_dataset: E016CorrectedPrecisionDataset,
    config: E016P0Config,
    *,
    device: torch.device,
) -> dict[str, Any]:
    preflight = config.full_preflight
    _seed_everything(preflight.seed)
    model, motion_before = _new_frozen_motion_model(device)
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=preflight.learning_rate,
        weight_decay=preflight.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=preflight.epochs,
        eta_min=preflight.learning_rate * 0.05,
    )
    loader = _loader(
        train_dataset,
        batch_size=preflight.batch_size,
        shuffle=True,
        seed=preflight.seed,
    )
    initial_validation = evaluate_e016_p0_model(
        model,
        val_dataset,
        device=device,
        batch_size=preflight.batch_size,
        use_bf16=config.execution.use_bf16,
        heatmap_sigma_px=preflight.heatmap_sigma_px,
        loss_config=config.loss,
    )
    epochs: list[dict[str, Any]] = []
    optimizer_steps = 0
    examples_seen = 0
    for epoch in range(1, preflight.epochs + 1):
        model.train()
        weighted_loss = 0.0
        epoch_examples = 0
        gradient_norms: list[float] = []
        for raw_batch in loader:
            batch = _to_device(raw_batch, device)
            optimizer.zero_grad(set_to_none=True)
            height, width = batch["image"].shape[-2:]
            target = _supervision(batch, (height, width), preflight.heatmap_sigma_px)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=config.execution.use_bf16,
            ):
                output = model(
                    batch["image"],
                    batch["structured_state"],
                    batch["geometric_motion"],
                )
                loss = precision_unet_loss(output, target, config.loss)
            if not bool(torch.isfinite(loss.loss)):
                raise RuntimeError("E016-P0 full preflight loss 出现 NaN/Inf")
            loss.loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                parameters,
                preflight.gradient_clip_norm,
            )
            if not bool(torch.isfinite(gradient_norm)):
                raise RuntimeError("E016-P0 full preflight gradient 出现 NaN/Inf")
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
            batch_size=preflight.batch_size,
            use_bf16=config.execution.use_bf16,
            heatmap_sigma_px=preflight.heatmap_sigma_px,
            loss_config=config.loss,
        )
        epochs.append(
            {
                "epoch": epoch,
                "train_loss": float(weighted_loss / epoch_examples),
                "gradient_norm_mean": float(np.mean(gradient_norms)),
                "gradient_norm_max": float(np.max(gradient_norms)),
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
                "validation": validation.to_dict(),
            }
        )
        scheduler.step()
    motion_after = precision_parameter_state_sha256(model.motion_head.state_dict())
    return {
        "version": E016_P0_VERSION,
        "passed": motion_after == motion_before,
        "fresh_random_initialization": True,
        "epochs_completed": preflight.epochs,
        "optimizer_steps": optimizer_steps,
        "examples_seen": examples_seen,
        "initial_validation": initial_validation.to_dict(),
        "epochs": epochs,
        "motion_head_sha256_before": motion_before,
        "motion_head_sha256_after": motion_after,
        "final_parameter_state_sha256": precision_parameter_state_sha256(
            model.state_dict()
        ),
        "checkpoint_persisted": False,
    }


def run_e016_p0(
    *,
    deployable_root: str | Path,
    source_label_root: str | Path,
    config_path: str | Path,
    output_root: str | Path,
    repository_root: str | Path,
) -> dict[str, Any]:
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"E016-P0 output 已存在，拒绝覆盖: {output}")
    config = load_e016_p0_config(config_path)
    source_identity = source_tree_sha256(repository_root)
    if not torch.cuda.is_available() or not torch.cuda.is_bf16_supported():
        raise RuntimeError("E016-P0 要求云端 RTX 4090 CUDA/BF16")
    device = torch.device(config.execution.device)
    output.mkdir(mode=0o700, parents=True)
    _atomic_json(output / "config_snapshot.json", config.to_dict())
    sidecar_audit = build_e016_corrected_sidecar(
        source_label_root,
        output / "corrected-labels",
        config,
    )
    _atomic_json(output / "corrected_sidecar_audit.json", sidecar_audit)
    train_dataset = E016CorrectedPrecisionDataset(
        deployable_root,
        source_label_root,
        output / "corrected-labels",
        "train",
    )
    val_dataset = E016CorrectedPrecisionDataset(
        deployable_root,
        source_label_root,
        output / "corrected-labels",
        "val",
    )
    loss_contract = run_e016_loss_contract_probe(device)
    _atomic_json(output / "loss_contract.json", loss_contract)
    selected_indices, subset_rows = select_stratified_overfit_indices(
        train_dataset,
        config.stratified_overfit,
    )
    _atomic_json(
        output / "stratified_subset.json",
        {
            "version": E016_P0_VERSION,
            "seed": config.stratified_overfit.seed,
            "strata_counts": config.stratified_overfit.strata_counts,
            "sample_count": len(selected_indices),
            "samples": subset_rows,
        },
    )
    overfit = run_e016_stratified_overfit(
        train_dataset,
        selected_indices,
        config,
        device=device,
    )
    _atomic_json(output / "stratified_overfit.json", overfit)
    preflight: dict[str, Any] | None = None
    if loss_contract["passed"] and overfit["passed"]:
        torch.cuda.empty_cache()
        preflight = run_e016_full_preflight(
            train_dataset,
            val_dataset,
            config,
            device=device,
        )
        _atomic_json(output / "full_preflight.json", preflight)
    passed = bool(
        sidecar_audit["passed"]
        and loss_contract["passed"]
        and overfit["passed"]
        and preflight is not None
        and preflight["passed"]
    )
    receipt = {
        "version": E016_P0_VERSION,
        "experiment": "E016-P0 corrected-observability pretraining contract",
        "passed": passed,
        "source_tree_sha256": source_identity,
        "training_config_sha256": config.sha256,
        "corrected_data_identity_sha256": sidecar_audit[
            "corrected_data_identity_sha256"
        ],
        "included_splits": list(config.source.allowed_splits),
        "excluded_splits": list(config.source.excluded_splits),
        "test_split_read": False,
        "e015_test_used_for_tuning": False,
        "loss_contract_passed": bool(loss_contract["passed"]),
        "stratified_overfit_passed": bool(overfit["passed"]),
        "full_preflight_executed": preflight is not None,
        "full_preflight_passed": bool(preflight and preflight["passed"]),
        "formal_checkpoint_written": False,
        "checkpoint_eligible_for_e016_p1": False,
        "actuation_allowed": False,
        "safe_for_actuator_promotion": False,
    }
    _atomic_json(output / "receipt.json", receipt)
    return receipt


__all__ = [
    "E016_CORRECTED_LABEL_ARRAYS",
    "E016_CORRECTED_LABEL_SCHEMA_VERSION",
    "E016_GOAL_EXISTS_POLICY",
    "E016_GOAL_LOCALIZATION_POLICY",
    "E016_MASK_SUPERVISION_POLICY",
    "E016_OCCLUSION_TYPES",
    "E016_P0_VERSION",
    "E016_STRATA",
    "E016CorrectedLabelArrays",
    "E016CorrectedLabelMeta",
    "E016CorrectedPrecisionDataset",
    "E016P0Config",
    "E016P0Metrics",
    "build_e016_corrected_sidecar",
    "derive_e016_corrected_arrays",
    "evaluate_e016_p0_model",
    "load_e016_corrected_manifest",
    "load_e016_p0_config",
    "read_e016_corrected_labels",
    "run_e016_full_preflight",
    "run_e016_loss_contract_probe",
    "run_e016_p0",
    "run_e016_stratified_overfit",
    "select_stratified_overfit_indices",
    "validate_e016_corrected_arrays",
]
