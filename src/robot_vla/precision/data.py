"""E013 RGB-only Dataset 与 privileged label sidecar 的严格契约。"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
from collections import OrderedDict
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.adapters import (
    FingerForceNormalizer,
    FingerForceStats,
    ProprioNormalizer,
    ProprioStats,
)
from robot_vla.contracts import RobotSpec
from robot_vla.data.dataset import ObservationV2ActionChunkDataset
from robot_vla.data.trajectory import (
    TRAJECTORY_SCHEMA_VERSION,
    TrajectoryArrays,
    TrajectoryMeta,
    load_manifest,
    resolve_trajectory_path,
)
from robot_vla.observation import CAMERA_FRAME_CONVENTION
from robot_vla.precision.geometry import (
    normalized_uv_to_base_z_plane,
    project_base_point_to_normalized_uv,
)

PRECISION_LABEL_SCHEMA_VERSION = "robot-vla-precision-label-sidecar/v1"
PRECISION_LABEL_SOURCE = "maniskill-evaluator-gt/offline-supervision-only/v1"
PRECISION_RGB_DATASET_VERSION = "robot-vla-precision-rgb-dataset/v1"
PRECISION_IMAGE_SEMANTICS = "wrist-rgb/current-frame/uint8-hwc/v1"
PRECISION_GEOMETRY_TRAINING_INPUT = "deployable-safe-hold-zero/motion-head-frozen/v1"
PRECISION_KEYPOINT_NAMES = ("object_center", "goal_center")
PRECISION_MASK_NAMES = ("object", "goal")
PRECISION_MODEL_INPUT_KEYS = (
    "rgb_wrist",
    "structured_state",
    "geometric_motion",
)

PRECISION_LABEL_ARRAYS = (
    "source_timestep",
    "timestamp_s",
    "object_mask",
    "goal_mask",
    "normalized_uv",
    "keypoint_visible",
    "keypoint_projection_valid",
    "object_position_base_m",
    "goal_position_base_m",
)

_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or _SHA256_PATTERN.fullmatch(value) is None:
        raise ValueError(f"{name} 必须是 64 位小写 SHA-256")


@dataclass(frozen=True)
class PrecisionLabelMeta:
    """不含绝对路径的 privileged sidecar manifest 行。"""

    trajectory_id: str
    file: str
    split: str
    scene_id: str
    num_steps: int
    source_trajectory_sha256: str
    source_trajectory_schema: str = TRAJECTORY_SCHEMA_VERSION
    label_source: str = PRECISION_LABEL_SOURCE
    keypoint_names: tuple[str, ...] = PRECISION_KEYPOINT_NAMES
    mask_names: tuple[str, ...] = PRECISION_MASK_NAMES
    camera_name: str = "hand_camera"
    camera_frame: str = CAMERA_FRAME_CONVENTION
    coordinate_frame: str = "robot_base"
    image_semantics: str = PRECISION_IMAGE_SEMANTICS
    geometry_training_input: str = PRECISION_GEOMETRY_TRAINING_INPUT
    privileged_only: bool = True
    schema_version: str = PRECISION_LABEL_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("trajectory_id", self.trajectory_id),
            ("file", self.file),
            ("scene_id", self.scene_id),
        ):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Precision label {name} 不能为空")
        if self.split not in {"train", "val", "test"}:
            raise ValueError("Precision label split 必须是 train/val/test")
        if self.num_steps <= 0:
            raise ValueError("Precision label num_steps 必须为正整数")
        _require_sha256(self.source_trajectory_sha256, "source_trajectory_sha256")
        if self.source_trajectory_schema != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError("Precision label source trajectory schema 漂移")
        if self.label_source != PRECISION_LABEL_SOURCE:
            raise ValueError("Precision label source 漂移")
        if self.keypoint_names != PRECISION_KEYPOINT_NAMES:
            raise ValueError("Precision keypoint names 漂移")
        if self.mask_names != PRECISION_MASK_NAMES:
            raise ValueError("Precision mask names 漂移")
        if self.camera_name != "hand_camera" or self.camera_frame != CAMERA_FRAME_CONVENTION:
            raise ValueError("Precision label camera 契约漂移")
        if self.coordinate_frame != "robot_base":
            raise ValueError("Precision label coordinate frame 必须为 robot_base")
        if self.image_semantics != PRECISION_IMAGE_SEMANTICS:
            raise ValueError("Precision label image semantics 漂移")
        if self.geometry_training_input != PRECISION_GEOMETRY_TRAINING_INPUT:
            raise ValueError("Precision geometry training input 漂移")
        if self.privileged_only is not True:
            raise ValueError("Precision label sidecar 必须声明 privileged_only=true")
        if self.schema_version != PRECISION_LABEL_SCHEMA_VERSION:
            raise ValueError("Precision label schema 漂移")

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["keypoint_names"] = list(self.keypoint_names)
        payload["mask_names"] = list(self.mask_names)
        return payload

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> PrecisionLabelMeta:
        return cls(
            trajectory_id=str(value["trajectory_id"]),
            file=str(value["file"]),
            split=str(value["split"]),
            scene_id=str(value["scene_id"]),
            num_steps=int(value["num_steps"]),
            source_trajectory_sha256=str(value["source_trajectory_sha256"]),
            source_trajectory_schema=str(value["source_trajectory_schema"]),
            label_source=str(value["label_source"]),
            keypoint_names=tuple(str(item) for item in value["keypoint_names"]),
            mask_names=tuple(str(item) for item in value["mask_names"]),
            camera_name=str(value["camera_name"]),
            camera_frame=str(value["camera_frame"]),
            coordinate_frame=str(value["coordinate_frame"]),
            image_semantics=str(value["image_semantics"]),
            geometry_training_input=str(value["geometry_training_input"]),
            privileged_only=value["privileged_only"],
            schema_version=str(value["schema_version"]),
        )


@dataclass(frozen=True)
class PrecisionLabelArrays:
    source_timestep: np.ndarray
    timestamp_s: np.ndarray
    object_mask: np.ndarray
    goal_mask: np.ndarray
    normalized_uv: np.ndarray
    keypoint_visible: np.ndarray
    keypoint_projection_valid: np.ndarray
    object_position_base_m: np.ndarray
    goal_position_base_m: np.ndarray

    @property
    def num_steps(self) -> int:
        return int(self.timestamp_s.shape[0])

    @property
    def keypoint_valid(self) -> np.ndarray:
        return self.keypoint_visible & self.keypoint_projection_valid


def _validate_array(
    value: np.ndarray,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic],
    name: str,
) -> None:
    if value.shape != shape or value.dtype != dtype:
        raise ValueError(
            f"Precision label {name} 必须是 {np.dtype(dtype)} {shape}，"
            f"实际 {value.dtype} {value.shape}"
        )


def validate_precision_label_arrays(
    arrays: PrecisionLabelArrays,
    meta: PrecisionLabelMeta,
) -> None:
    steps = arrays.num_steps
    if steps != meta.num_steps:
        raise ValueError("Precision label num_steps 与 manifest 不一致")
    _validate_array(arrays.source_timestep, (steps,), np.int64, "source_timestep")
    if not np.array_equal(arrays.source_timestep, np.arange(steps, dtype=np.int64)):
        raise ValueError("Precision label source_timestep 必须完整连续覆盖 0..T-1")
    _validate_array(arrays.timestamp_s, (steps,), np.float64, "timestamp_s")
    if not np.isfinite(arrays.timestamp_s).all() or np.any(arrays.timestamp_s < 0.0):
        raise ValueError("Precision label timestamp_s 必须有限非负")
    if steps > 1 and np.any(np.diff(arrays.timestamp_s) <= 0.0):
        raise ValueError("Precision label timestamp_s 必须严格递增")

    if arrays.object_mask.ndim != 3 or arrays.object_mask.shape[0] != steps:
        raise ValueError("Precision object_mask 必须是 [T,H,W]")
    if arrays.goal_mask.shape != arrays.object_mask.shape:
        raise ValueError("Precision goal_mask 必须与 object_mask shape 一致")
    if arrays.object_mask.dtype != np.bool_ or arrays.goal_mask.dtype != np.bool_:
        raise ValueError("Precision mask dtype 必须为 bool")
    if min(arrays.object_mask.shape[1:]) <= 0:
        raise ValueError("Precision mask H/W 必须为正数")

    _validate_array(arrays.normalized_uv, (steps, 2, 2), np.float32, "normalized_uv")
    _validate_array(
        arrays.keypoint_visible,
        (steps, 2),
        np.bool_,
        "keypoint_visible",
    )
    _validate_array(
        arrays.keypoint_projection_valid,
        (steps, 2),
        np.bool_,
        "keypoint_projection_valid",
    )
    for name in ("object_position_base_m", "goal_position_base_m"):
        value = getattr(arrays, name)
        _validate_array(value, (steps, 3), np.float32, name)
        if not np.isfinite(value).all():
            raise ValueError(f"Precision label {name} 包含 NaN/Inf")
    if not np.isfinite(arrays.normalized_uv).all():
        raise ValueError("Precision normalized_uv 包含 NaN/Inf")
    valid_uv = arrays.normalized_uv[arrays.keypoint_projection_valid]
    if np.any(valid_uv < 0.0) or np.any(valid_uv > 1.0):
        raise ValueError("有效 Precision normalized_uv 必须位于 [0,1]")
    if np.any(arrays.normalized_uv[~arrays.keypoint_projection_valid] != 0.0):
        raise ValueError("无效 Precision projection 的 normalized_uv 必须为零")
    expected_visible = np.column_stack(
        (
            arrays.object_mask.reshape(steps, -1).any(axis=1),
            arrays.goal_mask.reshape(steps, -1).any(axis=1),
        )
    )
    if not np.array_equal(arrays.keypoint_visible, expected_visible):
        raise ValueError("Precision keypoint_visible 必须由对应 segmentation mask 导出")


def load_precision_label_manifest(
    root: str | Path,
    *,
    split: str | None = None,
) -> list[PrecisionLabelMeta]:
    path = Path(root) / "manifest.jsonl"
    if not path.is_file():
        raise FileNotFoundError(f"找不到 Precision label manifest: {path}")
    entries: list[PrecisionLabelMeta] = []
    seen: set[str] = set()
    scene_splits: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            entry = PrecisionLabelMeta.from_dict(json.loads(line))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            raise ValueError(
                f"Precision label manifest 第 {line_number} 行无效: {error}"
            ) from error
        if entry.trajectory_id in seen:
            raise ValueError(f"Precision label trajectory_id 重复: {entry.trajectory_id}")
        seen.add(entry.trajectory_id)
        previous = scene_splits.setdefault(entry.scene_id, entry.split)
        if previous != entry.split:
            raise ValueError(f"Precision label scene_id 跨 split: {entry.scene_id}")
        entries.append(entry)
    selected = entries if split is None else [entry for entry in entries if entry.split == split]
    if not selected:
        raise ValueError("Precision label manifest 没有匹配条目")
    return selected


def _resolve_label_path(root: Path, relative_path: str, *, require_file: bool) -> Path:
    resolved_root = root.resolve()
    path = (resolved_root / relative_path).resolve()
    if not path.is_relative_to(resolved_root):
        raise ValueError("Precision label 文件不能位于 sidecar root 之外")
    if path.suffix != ".npz":
        raise ValueError("Precision label 文件必须使用 .npz")
    if require_file and not path.is_file():
        raise FileNotFoundError(f"找不到 Precision label 文件: {path}")
    return path


def read_precision_labels(
    root: str | Path,
    meta: PrecisionLabelMeta,
) -> PrecisionLabelArrays:
    path = _resolve_label_path(Path(root), meta.file, require_file=True)
    try:
        with np.load(path, allow_pickle=False) as npz:
            missing = [name for name in PRECISION_LABEL_ARRAYS if name not in npz]
            extra = sorted(set(npz.files) - set(PRECISION_LABEL_ARRAYS))
            if missing or extra:
                raise ValueError(f"label arrays 漂移: missing={missing}, extra={extra}")
            values = {name: np.asarray(npz[name]).copy() for name in PRECISION_LABEL_ARRAYS}
    except (OSError, ValueError) as error:
        raise ValueError(f"读取 Precision label {path} 失败: {error}") from error
    arrays = PrecisionLabelArrays(**values)
    validate_precision_label_arrays(arrays, meta)
    return arrays


class PrecisionLabelDatasetWriter:
    """原子写入 privileged NPZ，再提交 sidecar manifest；拒绝覆盖。"""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def write(self, meta: PrecisionLabelMeta, arrays: PrecisionLabelArrays) -> Path:
        validate_precision_label_arrays(arrays, meta)
        existing = self._existing_entries()
        if any(entry.trajectory_id == meta.trajectory_id for entry in existing):
            raise ValueError(f"Precision label trajectory_id 已存在: {meta.trajectory_id}")
        if any(entry.file == meta.file for entry in existing):
            raise ValueError(f"Precision label file 已在 manifest 中: {meta.file}")
        self.root.mkdir(mode=0o700, parents=True, exist_ok=True)
        target = _resolve_label_path(self.root, meta.file, require_file=False)
        if target.exists():
            raise FileExistsError(f"拒绝覆盖已有 Precision label: {target}")
        target.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        temporary: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as handle:
                temporary = Path(handle.name)
                np.savez_compressed(
                    handle,
                    **{name: getattr(arrays, name) for name in PRECISION_LABEL_ARRAYS},
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.chmod(temporary, 0o600)
            os.replace(temporary, target)
            temporary = None
            self._replace_manifest([*existing, meta])
        except Exception:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if not any(entry.file == meta.file for entry in self._existing_entries()):
                target.unlink(missing_ok=True)
            raise
        return target

    def _existing_entries(self) -> list[PrecisionLabelMeta]:
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
            return []
        return load_precision_label_manifest(self.root)

    def _replace_manifest(self, entries: Sequence[PrecisionLabelMeta]) -> None:
        payload = "".join(
            json.dumps(entry.to_dict(), sort_keys=True, allow_nan=False) + "\n" for entry in entries
        )
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=".manifest.",
            suffix=".tmp",
            dir=self.root,
            delete=False,
        ) as handle:
            temporary = Path(handle.name)
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.replace(temporary, self.root / "manifest.jsonl")
        finally:
            temporary.unlink(missing_ok=True)


class PrecisionLabelStore:
    def __init__(self, root: str | Path, cache_size: int = 2) -> None:
        if cache_size < 0:
            raise ValueError("Precision label cache_size 不能为负数")
        self.root = Path(root)
        self.cache_size = cache_size
        self._cache: OrderedDict[str, PrecisionLabelArrays] = OrderedDict()

    def get(self, meta: PrecisionLabelMeta) -> PrecisionLabelArrays:
        if meta.trajectory_id in self._cache:
            arrays = self._cache.pop(meta.trajectory_id)
            self._cache[meta.trajectory_id] = arrays
            return arrays
        arrays = read_precision_labels(self.root, meta)
        if self.cache_size:
            self._cache[meta.trajectory_id] = arrays
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays


def validate_precision_pair(
    deployable_root: str | Path,
    label_root: str | Path,
    trajectory_meta: TrajectoryMeta,
    trajectory_arrays: TrajectoryArrays,
    label_meta: PrecisionLabelMeta,
    label_arrays: PrecisionLabelArrays,
) -> None:
    deployable = Path(deployable_root).resolve()
    privileged = Path(label_root).resolve()
    if deployable == privileged or privileged.is_relative_to(deployable):
        raise ValueError("privileged label root 必须位于 deployable Dataset root 之外")
    if deployable.is_relative_to(privileged):
        raise ValueError("deployable Dataset root 不能位于 privileged label root 内")
    if trajectory_meta.trajectory_id != label_meta.trajectory_id:
        raise ValueError("Precision source/label trajectory_id 不一致")
    if trajectory_meta.split != label_meta.split or trajectory_meta.scene_id != label_meta.scene_id:
        raise ValueError("Precision source/label split 或 scene_id 不一致")
    if trajectory_arrays.num_steps != label_arrays.num_steps:
        raise ValueError("Precision source/label step 数不一致")
    source_path = resolve_trajectory_path(deployable, trajectory_meta.file)
    if file_sha256(source_path) != label_meta.source_trajectory_sha256:
        raise RuntimeError("Precision label 绑定的 source trajectory SHA-256 漂移")
    if not trajectory_arrays.observation_v2_available:
        raise ValueError("Precision RGB Dataset 要求完整 Observation V2")
    if not np.array_equal(label_arrays.timestamp_s, trajectory_arrays.timestamp_wrist):
        raise ValueError("Precision label timestamp 与 wrist RGB 不逐 Tick 对齐")
    expected_hw = tuple(int(value) for value in trajectory_arrays.rgb_wrist.shape[1:3])
    if label_arrays.object_mask.shape[1:] != expected_hw:
        raise ValueError("Precision mask spatial shape 与 wrist RGB 不一致")


@dataclass(frozen=True)
class PrecisionDatasetAuditReport:
    version: str
    passed: bool
    dataset_identity_sha256: str
    deployable_manifest_sha256: str
    label_manifest_sha256: str
    trajectory_count: int
    sample_count: int
    split_trajectory_counts: dict[str, int]
    split_sample_counts: dict[str, int]
    split_scene_overlap: dict[str, list[str]]
    keypoint_visible_counts: dict[str, int]
    keypoint_projection_valid_counts: dict[str, int]
    keypoint_training_counts: dict[str, int]
    oracle_roundtrip_error_p50_m: float
    oracle_roundtrip_error_p90_m: float
    oracle_roundtrip_error_max_m: float
    projection_unavailable_count: int
    oracle_invalid_count: int
    model_input_keys: tuple[str, ...]
    privileged_arrays: tuple[str, ...]
    model_input_privileged_overlap: tuple[str, ...]
    oracle_geometry_gate_passed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def audit_precision_dataset(
    deployable_root: str | Path,
    label_root: str | Path,
    spec: RobotSpec,
    *,
    write_artifact: bool = True,
) -> PrecisionDatasetAuditReport:
    """核验 split、identity、时间/shape 对齐和 oracle 几何 lower bound。"""

    deployable_root = Path(deployable_root)
    label_root = Path(label_root)
    trajectory_entries = load_manifest(deployable_root)
    label_entries = load_precision_label_manifest(label_root)
    trajectory_by_id = {entry.trajectory_id: entry for entry in trajectory_entries}
    label_by_id = {entry.trajectory_id: entry for entry in label_entries}
    if set(trajectory_by_id) != set(label_by_id):
        raise ValueError(
            "Precision source/label trajectory 集合不一致: "
            f"missing_labels={sorted(set(trajectory_by_id) - set(label_by_id))[:5]}, "
            f"orphan_labels={sorted(set(label_by_id) - set(trajectory_by_id))[:5]}"
        )

    from robot_vla.data.trajectory import TrajectoryStore

    trajectory_store = TrajectoryStore(deployable_root, spec, cache_size=0)
    label_store = PrecisionLabelStore(label_root, cache_size=0)
    split_trajectory_counts = {split: 0 for split in ("train", "val", "test")}
    split_sample_counts = {split: 0 for split in ("train", "val", "test")}
    split_scenes = {split: set() for split in ("train", "val", "test")}
    visible = np.zeros(2, dtype=np.int64)
    projected = np.zeros(2, dtype=np.int64)
    training = np.zeros(2, dtype=np.int64)
    roundtrip_errors: list[float] = []
    projection_unavailable = 0
    oracle_invalid = 0
    canonical_files: list[dict[str, Any]] = []

    for trajectory_id in sorted(trajectory_by_id):
        trajectory_meta = trajectory_by_id[trajectory_id]
        label_meta = label_by_id[trajectory_id]
        trajectory_arrays = trajectory_store.get(trajectory_meta)
        label_arrays = label_store.get(label_meta)
        validate_precision_pair(
            deployable_root,
            label_root,
            trajectory_meta,
            trajectory_arrays,
            label_meta,
            label_arrays,
        )
        split = trajectory_meta.split
        split_trajectory_counts[split] += 1
        split_sample_counts[split] += trajectory_arrays.num_steps
        split_scenes[split].add(trajectory_meta.scene_id)
        visible += label_arrays.keypoint_visible.sum(axis=0)
        projected += label_arrays.keypoint_projection_valid.sum(axis=0)
        training += label_arrays.keypoint_valid.sum(axis=0)

        intrinsic = np.asarray(
            trajectory_meta.camera_calibration.intrinsic_wrist,
            dtype=np.float64,
        ).reshape(3, 3)
        image_size = tuple(int(value) for value in trajectory_arrays.rgb_wrist.shape[1:3])
        positions = np.stack(
            (
                label_arrays.object_position_base_m,
                label_arrays.goal_position_base_m,
            ),
            axis=1,
        )
        for timestep in range(trajectory_arrays.num_steps):
            base_from_camera = np.eye(4, dtype=np.float64)
            from robot_vla.observation import rotation_6d_to_matrix

            base_from_camera[:3, :3] = rotation_6d_to_matrix(
                trajectory_arrays.wrist_camera_rotation_6d_base[timestep]
            )
            base_from_camera[:3, 3] = trajectory_arrays.wrist_camera_position_base_m[timestep]
            for keypoint_index in range(2):
                if not label_arrays.keypoint_projection_valid[timestep, keypoint_index]:
                    # 中心点在视野外是数据事实；部分 mask 仍可用于可见性监督。
                    projection_unavailable += 1
                    continue
                try:
                    reconstructed = normalized_uv_to_base_z_plane(
                        label_arrays.normalized_uv[timestep, keypoint_index],
                        intrinsic,
                        base_from_camera,
                        image_size,
                        plane_base_z_m=float(positions[timestep, keypoint_index, 2]),
                    ).point_base_m
                except ValueError:
                    oracle_invalid += 1
                    continue
                error = float(
                    np.linalg.norm(reconstructed[:2] - positions[timestep, keypoint_index, :2])
                )
                roundtrip_errors.append(error)

        label_path = _resolve_label_path(label_root, label_meta.file, require_file=True)
        source_path = resolve_trajectory_path(deployable_root, trajectory_meta.file)
        canonical_files.append(
            {
                "trajectory_id": trajectory_id,
                "source_sha256": file_sha256(source_path),
                "label_sha256": file_sha256(label_path),
                "source_meta": trajectory_meta.to_dict(),
                "label_meta": label_meta.to_dict(),
            }
        )

    overlaps = {
        "train_val": sorted(split_scenes["train"] & split_scenes["val"]),
        "train_test": sorted(split_scenes["train"] & split_scenes["test"]),
        "val_test": sorted(split_scenes["val"] & split_scenes["test"]),
    }
    if any(overlaps.values()):
        raise ValueError(f"Precision scene split 泄漏: {overlaps}")
    if any(count == 0 for count in split_trajectory_counts.values()):
        raise ValueError(f"Precision train/val/test 不能为空: {split_trajectory_counts}")
    if not roundtrip_errors:
        raise ValueError("Precision Dataset 没有可用 oracle projection")
    error_array = np.asarray(roundtrip_errors, dtype=np.float64)
    p50 = float(np.quantile(error_array, 0.50))
    p90 = float(np.quantile(error_array, 0.90))
    maximum = float(np.max(error_array))
    input_overlap = tuple(sorted(set(PRECISION_MODEL_INPUT_KEYS) & set(PRECISION_LABEL_ARRAYS)))
    geometry_gate = oracle_invalid == 0 and p90 <= 0.005
    dataset_identity = canonical_sha256(
        {
            "version": PRECISION_RGB_DATASET_VERSION,
            "files": canonical_files,
            "model_input_keys": list(PRECISION_MODEL_INPUT_KEYS),
            "privileged_arrays": list(PRECISION_LABEL_ARRAYS),
        }
    )
    report = PrecisionDatasetAuditReport(
        version=PRECISION_RGB_DATASET_VERSION,
        passed=not input_overlap and not any(overlaps.values()) and geometry_gate,
        dataset_identity_sha256=dataset_identity,
        deployable_manifest_sha256=file_sha256(deployable_root / "manifest.jsonl"),
        label_manifest_sha256=file_sha256(label_root / "manifest.jsonl"),
        trajectory_count=len(trajectory_entries),
        sample_count=sum(split_sample_counts.values()),
        split_trajectory_counts=split_trajectory_counts,
        split_sample_counts=split_sample_counts,
        split_scene_overlap=overlaps,
        keypoint_visible_counts=dict(zip(PRECISION_KEYPOINT_NAMES, visible.tolist(), strict=True)),
        keypoint_projection_valid_counts=dict(
            zip(PRECISION_KEYPOINT_NAMES, projected.tolist(), strict=True)
        ),
        keypoint_training_counts=dict(
            zip(PRECISION_KEYPOINT_NAMES, training.tolist(), strict=True)
        ),
        oracle_roundtrip_error_p50_m=p50,
        oracle_roundtrip_error_p90_m=p90,
        oracle_roundtrip_error_max_m=maximum,
        projection_unavailable_count=projection_unavailable,
        oracle_invalid_count=oracle_invalid,
        model_input_keys=PRECISION_MODEL_INPUT_KEYS,
        privileged_arrays=PRECISION_LABEL_ARRAYS,
        model_input_privileged_overlap=input_overlap,
        oracle_geometry_gate_passed=geometry_gate,
    )
    if write_artifact:
        path = label_root / "audit_report.json"
        path.write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


class PrecisionRGBDataset:
    """单帧 U-Net Dataset；privileged labels 只进入 ``supervision`` 字段。"""

    def __init__(
        self,
        deployable_root: str | Path,
        label_root: str | Path,
        split: str,
        spec: RobotSpec | None = None,
        *,
        cache_size: int = 2,
    ) -> None:
        if split not in {"train", "val", "test"}:
            raise ValueError("PrecisionRGBDataset split 必须是 train/val/test")
        self.spec = spec or RobotSpec()
        self.deployable_root = Path(deployable_root)
        self.label_root = Path(label_root)
        proprio_stats = ProprioStats.from_json(self.deployable_root / "proprio_stats.json")
        finger_stats = FingerForceStats.from_json(self.deployable_root / "finger_force_stats.json")
        self.base = ObservationV2ActionChunkDataset(
            str(self.deployable_root),
            load_manifest(self.deployable_root, split=split),
            self.spec,
            ProprioNormalizer(proprio_stats, self.spec),
            finger_force_normalizer=FingerForceNormalizer(finger_stats, self.spec),
            cache_size=cache_size,
        )
        label_entries = load_precision_label_manifest(self.label_root, split=split)
        self.label_by_trajectory = {entry.trajectory_id: entry for entry in label_entries}
        self.source_by_trajectory = {entry.trajectory_id: entry for entry in self.base.entries}
        source_ids = {entry.trajectory_id for entry in self.base.entries}
        if source_ids != set(self.label_by_trajectory):
            raise ValueError("PrecisionRGBDataset 当前 split 的 source/label 集合不一致")
        self.label_store = PrecisionLabelStore(self.label_root, cache_size=cache_size)

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, index: int) -> dict[str, Any]:
        sample = self.base[index]
        label_meta = self.label_by_trajectory[sample["trajectory_id"]]
        labels = self.label_store.get(label_meta)
        timestep = int(sample["timestep"])
        source_meta = self.source_by_trajectory[sample["trajectory_id"]]
        source_arrays = self.base.store.get(source_meta)
        if not bool(sample["state_history_mask"][-1]):
            raise RuntimeError("Precision current history row 必须有效")
        structured_state = sample["state_history"][-1].copy()
        # Motion Head v1 保持零初始化和冻结；禁止把 GT geometry 填入模型 feature。
        geometric_motion = np.zeros(4, dtype=np.float32)
        return {
            "model_inputs": {
                "rgb_wrist": sample["rgb_wrist"].copy(),
                "structured_state": structured_state,
                "geometric_motion": geometric_motion,
            },
            "supervision": {
                "mask_targets": np.stack(
                    (
                        labels.object_mask[timestep],
                        labels.goal_mask[timestep],
                    )
                ).astype(np.float32),
                "normalized_uv_targets": labels.normalized_uv[timestep].copy(),
                "keypoint_valid": labels.keypoint_valid[timestep].copy(),
                "motion_residual_targets": np.zeros(4, dtype=np.float32),
                "motion_valid": np.zeros(4, dtype=np.bool_),
                "projection_valid": np.asarray(
                    bool(labels.keypoint_projection_valid[timestep].all()),
                    dtype=np.bool_,
                ),
            },
            "audit": {
                "trajectory_id": sample["trajectory_id"],
                "scene_id": label_meta.scene_id,
                "split": label_meta.split,
                "timestep": timestep,
                "timestamp_s": float(labels.timestamp_s[timestep]),
                "object_position_base_m": labels.object_position_base_m[timestep].copy(),
                "goal_position_base_m": labels.goal_position_base_m[timestep].copy(),
                "intrinsic_wrist_cv": np.asarray(
                    source_meta.camera_calibration.intrinsic_wrist,
                    dtype=np.float32,
                ).reshape(3, 3),
                "wrist_camera_position_base_m": source_arrays.wrist_camera_position_base_m[
                    timestep
                ].copy(),
                "wrist_camera_rotation_6d_base": source_arrays.wrist_camera_rotation_6d_base[
                    timestep
                ].copy(),
            },
        }


def build_precision_label_meta(
    trajectory_meta: TrajectoryMeta,
    trajectory_path: str | Path,
) -> PrecisionLabelMeta:
    return PrecisionLabelMeta(
        trajectory_id=trajectory_meta.trajectory_id,
        file=f"labels/{trajectory_meta.trajectory_id}.npz",
        split=trajectory_meta.split,
        scene_id=trajectory_meta.scene_id,
        num_steps=trajectory_meta.num_steps,
        source_trajectory_sha256=file_sha256(trajectory_path),
    )


def project_label_keypoints(
    object_position_base_m: np.ndarray,
    goal_position_base_m: np.ndarray,
    intrinsic_wrist_cv: np.ndarray,
    base_from_wrist_camera_cv: np.ndarray,
    image_size_hw: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    """投影两个 GT center；无效投影用零表示并显式返回 validity。"""

    positions = (object_position_base_m, goal_position_base_m)
    normalized_uv = np.zeros((2, 2), dtype=np.float32)
    valid = np.zeros(2, dtype=np.bool_)
    for index, position in enumerate(positions):
        try:
            uv = project_base_point_to_normalized_uv(
                np.asarray(position, dtype=np.float32),
                intrinsic_wrist_cv,
                base_from_wrist_camera_cv,
                image_size_hw,
            )
        except ValueError:
            continue
        if np.all((uv >= 0.0) & (uv <= 1.0)):
            normalized_uv[index] = uv
            valid[index] = True
    return normalized_uv, valid


__all__ = [
    "PRECISION_GEOMETRY_TRAINING_INPUT",
    "PRECISION_IMAGE_SEMANTICS",
    "PRECISION_KEYPOINT_NAMES",
    "PRECISION_LABEL_ARRAYS",
    "PRECISION_LABEL_SCHEMA_VERSION",
    "PRECISION_LABEL_SOURCE",
    "PRECISION_MASK_NAMES",
    "PRECISION_MODEL_INPUT_KEYS",
    "PRECISION_RGB_DATASET_VERSION",
    "PrecisionDatasetAuditReport",
    "PrecisionLabelArrays",
    "PrecisionLabelDatasetWriter",
    "PrecisionLabelMeta",
    "PrecisionLabelStore",
    "PrecisionRGBDataset",
    "audit_precision_dataset",
    "build_precision_label_meta",
    "canonical_sha256",
    "file_sha256",
    "load_precision_label_manifest",
    "project_label_keypoints",
    "read_precision_labels",
    "validate_precision_label_arrays",
    "validate_precision_pair",
]
