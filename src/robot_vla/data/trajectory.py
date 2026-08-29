"""robot-vla-trajectory/v2 的 manifest、NPZ 读取和严格校验。"""

from __future__ import annotations

import json
import re
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.contracts import (
    TRAJECTORY_SCHEMA_VERSION,
    UNKNOWN_SKILL_ID,
    RobotSpec,
    TaskSpec,
)
from robot_vla.data.events import (
    EVENT_STATE_ARRAYS,
    EVENT_STATE_CONTRACT_VERSION,
)

REQUIRED_ARRAYS = (
    "rgb_external",
    "rgb_wrist",
    "timestamp_external",
    "timestamp_wrist",
    "timestamp_proprio",
    "timestamp_action",
    "proprio",
    "action",
    "external_valid",
    "wrist_valid",
    "proprio_valid",
    "terminated",
    "truncated",
    "success",
    "skill_id",
)

LOCAL_DAGGER_CONTRACT_VERSION = "robot-vla-local-dagger/v1"
LOCAL_DAGGER_WINDOW_STEPS = 64
LOCAL_DAGGER_SOURCES = (
    "dagger_reach_grasp",
    "dagger_grasp_lift",
)
LOCAL_DAGGER_BOUNDARIES = (
    "reach_grasp",
    "grasp_lift",
)
ACTION_SOURCE_POLICY = 0
ACTION_SOURCE_EXPERT = 1
LOCAL_DAGGER_ARRAYS = (
    "action_source",
    "expert_supervision_mask",
)


@dataclass(frozen=True)
class LocalDaggerProvenance:
    """Policy roll-in 后由 Expert 连续接管的版本化监督边界。"""

    source: str
    rollin_seed: int
    rollin_policy_checkpoint_sha256: str
    boundary_type: str
    boundary_detection_step: int
    expert_takeover_step: int
    training_window_start: int
    training_window_end: int
    expert_recovery_success: bool
    version: str = LOCAL_DAGGER_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if self.version != LOCAL_DAGGER_CONTRACT_VERSION:
            raise ValueError("Local DAgger contract version 不兼容")
        if self.source not in LOCAL_DAGGER_SOURCES:
            raise ValueError(f"未知 Local DAgger source: {self.source}")
        if self.boundary_type not in LOCAL_DAGGER_BOUNDARIES:
            raise ValueError(f"未知 Local DAgger boundary: {self.boundary_type}")
        expected_source = f"dagger_{self.boundary_type}"
        if self.source != expected_source:
            raise ValueError("Local DAgger source 与 boundary_type 不一致")
        if self.rollin_seed < 0:
            raise ValueError("Local DAgger rollin_seed 不能为负数")
        if not re.fullmatch(r"[0-9a-f]{64}", self.rollin_policy_checkpoint_sha256):
            raise ValueError("Local DAgger policy checkpoint 必须是小写 SHA256")
        if self.boundary_detection_step != self.expert_takeover_step:
            raise ValueError("第一版 boundary detection 必须与 Expert takeover 同步")
        if self.expert_takeover_step <= 0:
            raise ValueError("Local DAgger 必须包含至少一条 Policy roll-in Action")
        if self.training_window_start != self.expert_takeover_step:
            raise ValueError("Local DAgger training window 必须从 Expert takeover 开始")
        if self.training_window_end <= self.training_window_start:
            raise ValueError("Local DAgger training window 必须非空")
        if not isinstance(self.expert_recovery_success, bool):
            raise TypeError("expert_recovery_success 必须为 bool")
        if not self.expert_recovery_success:
            raise ValueError("正式 Local DAgger trajectory 必须由 Expert 完整恢复")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "source": self.source,
            "rollin_seed": self.rollin_seed,
            "rollin_policy_checkpoint_sha256": self.rollin_policy_checkpoint_sha256,
            "boundary_type": self.boundary_type,
            "boundary_detection_step": self.boundary_detection_step,
            "expert_takeover_step": self.expert_takeover_step,
            "training_window_start": self.training_window_start,
            "training_window_end": self.training_window_end,
            "expert_recovery_success": self.expert_recovery_success,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> LocalDaggerProvenance:
        return cls(
            version=str(value.get("version", "")),
            source=str(value["source"]),
            rollin_seed=int(value["rollin_seed"]),
            rollin_policy_checkpoint_sha256=str(
                value["rollin_policy_checkpoint_sha256"]
            ),
            boundary_type=str(value["boundary_type"]),
            boundary_detection_step=int(value["boundary_detection_step"]),
            expert_takeover_step=int(value["expert_takeover_step"]),
            training_window_start=int(value["training_window_start"]),
            training_window_end=int(value["training_window_end"]),
            expert_recovery_success=value["expert_recovery_success"],
        )


def _finite_tuple(values: tuple[float, ...], expected: int, name: str) -> None:
    if len(values) != expected:
        raise ValueError(f"{name} 长度应为 {expected}，实际为 {len(values)}")
    if not np.isfinite(np.asarray(values, dtype=np.float64)).all():
        raise ValueError(f"{name} 包含 NaN 或 Inf")


@dataclass(frozen=True)
class CameraCalibration:
    """每个 Episode 固定的双相机标定，矩阵按行展开。"""

    version: str
    intrinsic_external: tuple[float, ...]
    intrinsic_wrist: tuple[float, ...]
    world_from_external: tuple[float, ...]
    tcp_from_wrist: tuple[float, ...]

    def __post_init__(self) -> None:
        if not self.version.strip():
            raise ValueError("camera calibration version 不能为空")
        _finite_tuple(self.intrinsic_external, 9, "intrinsic_external")
        _finite_tuple(self.intrinsic_wrist, 9, "intrinsic_wrist")
        _finite_tuple(self.world_from_external, 16, "world_from_external")
        _finite_tuple(self.tcp_from_wrist, 16, "tcp_from_wrist")
        for name, intrinsic in (
            ("intrinsic_external", self.intrinsic_external),
            ("intrinsic_wrist", self.intrinsic_wrist),
        ):
            if intrinsic[0] <= 0 or intrinsic[4] <= 0:
                raise ValueError(f"{name} 的 fx/fy 必须为正数")
        for name, transform in (
            ("world_from_external", self.world_from_external),
            ("tcp_from_wrist", self.tcp_from_wrist),
        ):
            if not np.allclose(transform[12:16], (0.0, 0.0, 0.0, 1.0), atol=1e-6):
                raise ValueError(f"{name} 必须是齐次 4x4 变换")

    def to_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "intrinsic_external": list(self.intrinsic_external),
            "intrinsic_wrist": list(self.intrinsic_wrist),
            "world_from_external": list(self.world_from_external),
            "tcp_from_wrist": list(self.tcp_from_wrist),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CameraCalibration:
        return cls(
            version=str(value["version"]),
            intrinsic_external=tuple(float(item) for item in value["intrinsic_external"]),
            intrinsic_wrist=tuple(float(item) for item in value["intrinsic_wrist"]),
            world_from_external=tuple(float(item) for item in value["world_from_external"]),
            tcp_from_wrist=tuple(float(item) for item in value["tcp_from_wrist"]),
        )


@dataclass(frozen=True)
class OutcomeEvidence:
    """成功 Episode 结束时保存的 Predicate 审计证据。"""

    predicate_version: str
    task_completed: bool
    final_is_released: bool
    stable_place_steps: int
    external_goal_visible_steps: int
    wrist_goal_visible_steps: int
    both_goal_visible_steps: int
    final_object_to_goal_distance_m: float
    final_object_linear_speed_m_s: float
    final_object_angular_speed_rad_s: float

    def __post_init__(self) -> None:
        if not self.predicate_version.strip():
            raise ValueError("predicate_version 不能为空")
        if not isinstance(self.task_completed, bool) or not isinstance(
            self.final_is_released,
            bool,
        ):
            raise TypeError("task_completed/final_is_released 必须为 bool")
        for name in (
            "stable_place_steps",
            "external_goal_visible_steps",
            "wrist_goal_visible_steps",
            "both_goal_visible_steps",
        ):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} 不能为负数")
        if self.both_goal_visible_steps > min(
            self.external_goal_visible_steps,
            self.wrist_goal_visible_steps,
        ):
            raise ValueError("both_goal_visible_steps 不能超过任一路相机可见帧数")
        for name in (
            "final_object_to_goal_distance_m",
            "final_object_linear_speed_m_s",
            "final_object_angular_speed_rad_s",
        ):
            value = getattr(self, name)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} 必须是有限非负数")

    def to_dict(self) -> dict[str, Any]:
        return {
            "predicate_version": self.predicate_version,
            "task_completed": self.task_completed,
            "final_is_released": self.final_is_released,
            "stable_place_steps": self.stable_place_steps,
            "external_goal_visible_steps": self.external_goal_visible_steps,
            "wrist_goal_visible_steps": self.wrist_goal_visible_steps,
            "both_goal_visible_steps": self.both_goal_visible_steps,
            "final_object_to_goal_distance_m": self.final_object_to_goal_distance_m,
            "final_object_linear_speed_m_s": self.final_object_linear_speed_m_s,
            "final_object_angular_speed_rad_s": self.final_object_angular_speed_rad_s,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> OutcomeEvidence:
        return cls(
            predicate_version=str(value["predicate_version"]),
            task_completed=bool(value["task_completed"]),
            final_is_released=bool(value["final_is_released"]),
            stable_place_steps=int(value["stable_place_steps"]),
            external_goal_visible_steps=int(value["external_goal_visible_steps"]),
            wrist_goal_visible_steps=int(value["wrist_goal_visible_steps"]),
            both_goal_visible_steps=int(value["both_goal_visible_steps"]),
            final_object_to_goal_distance_m=float(
                value["final_object_to_goal_distance_m"]
            ),
            final_object_linear_speed_m_s=float(value["final_object_linear_speed_m_s"]),
            final_object_angular_speed_rad_s=float(
                value["final_object_angular_speed_rad_s"]
            ),
        )


@dataclass(frozen=True)
class TrajectoryMeta:
    trajectory_id: str
    source_episode_id: str
    file: str
    split: str
    scene_id: str
    task: TaskSpec
    num_steps: int
    camera_calibration: CameraCalibration
    randomization: dict[str, Any]
    outcome_evidence: OutcomeEvidence | None = None
    local_dagger: LocalDaggerProvenance | None = None
    schema_version: str = TRAJECTORY_SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name, value in (
            ("trajectory_id", self.trajectory_id),
            ("source_episode_id", self.source_episode_id),
            ("file", self.file),
            ("scene_id", self.scene_id),
        ):
            if not value.strip():
                raise ValueError(f"{name} 不能为空")
        if self.split not in {"train", "val", "test"}:
            raise ValueError(f"不支持的数据 split: {self.split}")
        if self.num_steps <= 0:
            raise ValueError("num_steps 必须为正整数")
        if self.schema_version != TRAJECTORY_SCHEMA_VERSION:
            raise ValueError(
                f"轨迹 schema 不兼容：期望 {TRAJECTORY_SCHEMA_VERSION}，"
                f"实际 {self.schema_version}"
            )
        if not isinstance(self.randomization, dict):
            raise TypeError("randomization 必须是 JSON object")
        try:
            json.dumps(self.randomization, allow_nan=False)
        except (TypeError, ValueError) as exc:
            raise ValueError("randomization 必须是有限、可序列化的 JSON 数据") from exc

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "trajectory_id": self.trajectory_id,
            "source_episode_id": self.source_episode_id,
            "file": self.file,
            "split": self.split,
            "scene_id": self.scene_id,
            "task": self.task.to_dict(),
            "num_steps": self.num_steps,
            "camera_calibration": self.camera_calibration.to_dict(),
            "randomization": self.randomization,
            "outcome_evidence": (
                None if self.outcome_evidence is None else self.outcome_evidence.to_dict()
            ),
            "local_dagger": (
                None if self.local_dagger is None else self.local_dagger.to_dict()
            ),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> TrajectoryMeta:
        return cls(
            trajectory_id=str(value["trajectory_id"]),
            source_episode_id=str(value["source_episode_id"]),
            file=str(value["file"]),
            split=str(value["split"]),
            scene_id=str(value["scene_id"]),
            task=TaskSpec.from_dict(value["task"]),
            num_steps=int(value["num_steps"]),
            camera_calibration=CameraCalibration.from_dict(value["camera_calibration"]),
            randomization=dict(value["randomization"]),
            outcome_evidence=(
                None
                if value.get("outcome_evidence") is None
                else OutcomeEvidence.from_dict(value["outcome_evidence"])
            ),
            local_dagger=(
                None
                if value.get("local_dagger") is None
                else LocalDaggerProvenance.from_dict(value["local_dagger"])
            ),
            schema_version=str(value.get("schema_version", "")),
        )


@dataclass(frozen=True)
class TrajectoryArrays:
    rgb_external: np.ndarray
    rgb_wrist: np.ndarray
    timestamp_external: np.ndarray
    timestamp_wrist: np.ndarray
    timestamp_proprio: np.ndarray
    timestamp_action: np.ndarray
    proprio: np.ndarray
    action: np.ndarray
    external_valid: np.ndarray
    wrist_valid: np.ndarray
    proprio_valid: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    success: np.ndarray
    skill_id: np.ndarray
    robot_object_contact_force_n: np.ndarray | None = None
    support_contact_force_n: np.ndarray | None = None
    is_grasped: np.ndarray | None = None
    object_position_m: np.ndarray | None = None
    object_linear_velocity_m_s: np.ndarray | None = None
    object_angular_velocity_rad_s: np.ndarray | None = None
    commanded_joint_target_rad: np.ndarray | None = None
    applied_joint_correction_rad: np.ndarray | None = None
    action_source: np.ndarray | None = None
    expert_supervision_mask: np.ndarray | None = None

    @property
    def num_steps(self) -> int:
        return int(self.rgb_external.shape[0])

    @property
    def observation_valid(self) -> np.ndarray:
        """只有双相机和 proprio 同时有效的控制索引才能成为样本起点。"""

        return self.external_valid & self.wrist_valid & self.proprio_valid

    @property
    def event_state_available(self) -> bool:
        present = [getattr(self, name) is not None for name in EVENT_STATE_ARRAYS]
        if any(present) and not all(present):
            raise ValueError("event-state optional arrays 必须同时存在或同时缺失")
        return all(present)

    @property
    def local_dagger_available(self) -> bool:
        present = [getattr(self, name) is not None for name in LOCAL_DAGGER_ARRAYS]
        if any(present) and not all(present):
            raise ValueError("Local DAgger optional arrays 必须同时存在或同时缺失")
        return all(present)


def load_manifest(root: str | Path, *, split: str | None = None) -> list[TrajectoryMeta]:
    root = Path(root)
    manifest_path = root / "manifest.jsonl"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到数据 manifest: {manifest_path}")

    entries: list[TrajectoryMeta] = []
    seen_ids: set[str] = set()
    source_splits: dict[str, str] = {}
    scene_splits: dict[str, str] = {}
    for line_number, line in enumerate(
        manifest_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            entry = TrajectoryMeta.from_dict(json.loads(line))
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise ValueError(f"manifest 第 {line_number} 行无效: {exc}") from exc
        if entry.trajectory_id in seen_ids:
            raise ValueError(f"manifest 存在重复 trajectory_id: {entry.trajectory_id}")
        seen_ids.add(entry.trajectory_id)
        _record_split(source_splits, entry.source_episode_id, entry.split, "source_episode_id")
        _record_split(scene_splits, entry.scene_id, entry.split, "scene_id")
        entries.append(entry)

    selected = entries if split is None else [entry for entry in entries if entry.split == split]
    if not selected:
        suffix = f" split={split}" if split else ""
        raise ValueError(f"manifest 没有可用轨迹{suffix}")
    return selected


def _record_split(mapping: dict[str, str], key: str, split: str, field: str) -> None:
    previous = mapping.setdefault(key, split)
    if previous != split:
        raise ValueError(f"{field}={key!r} 跨越 split: {previous} 与 {split}")


def resolve_trajectory_path(root: Path, relative_path: str) -> Path:
    root = root.resolve()
    path = (root / relative_path).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"轨迹文件不能位于数据根目录之外: {relative_path}")
    if not path.is_file():
        raise FileNotFoundError(f"找不到轨迹文件: {path}")
    return path


def read_trajectory(root: str | Path, meta: TrajectoryMeta) -> TrajectoryArrays:
    path = resolve_trajectory_path(Path(root), meta.file)
    try:
        with np.load(path, allow_pickle=False) as npz:
            missing = [name for name in REQUIRED_ARRAYS if name not in npz]
            if missing:
                raise ValueError(f"缺少数组: {missing}")
            values = {name: np.asarray(npz[name]).copy() for name in REQUIRED_ARRAYS}
            values.update(
                {
                    name: np.asarray(npz[name]).copy()
                    for name in EVENT_STATE_ARRAYS
                    if name in npz
                }
            )
            values.update(
                {
                    name: np.asarray(npz[name]).copy()
                    for name in LOCAL_DAGGER_ARRAYS
                    if name in npz
                }
            )
            arrays = TrajectoryArrays(**values)
    except (OSError, ValueError) as exc:
        raise ValueError(f"读取轨迹 {path} 失败: {exc}") from exc
    return arrays


def _validate_rgb(value: np.ndarray, length: int, name: str) -> None:
    if value.ndim != 4 or value.shape[0] != length or value.shape[-1] != 3:
        raise ValueError(f"{name} 应为 [T,H,W,3]，实际 {value.shape}")
    if value.shape[1] <= 0 or value.shape[2] <= 0:
        raise ValueError(f"{name} 的 H/W 必须为正数")
    if value.dtype != np.uint8:
        raise ValueError(f"{name} dtype 应为 uint8，实际 {value.dtype}")


def _validate_array(
    value: np.ndarray,
    shape: tuple[int, ...],
    dtype: np.dtype[Any] | type[np.generic],
    name: str,
) -> None:
    if value.shape != shape:
        raise ValueError(f"{name} shape 应为 {shape}，实际 {value.shape}")
    if value.dtype != dtype:
        raise ValueError(f"{name} dtype 应为 {np.dtype(dtype)}，实际 {value.dtype}")


def _validate_timestamps(arrays: TrajectoryArrays, spec: RobotSpec) -> None:
    timestamps = {
        "timestamp_external": arrays.timestamp_external,
        "timestamp_wrist": arrays.timestamp_wrist,
        "timestamp_proprio": arrays.timestamp_proprio,
        "timestamp_action": arrays.timestamp_action,
    }
    for name, value in timestamps.items():
        if not np.isfinite(value).all():
            raise ValueError(f"{name} 包含 NaN 或 Inf")
        if len(value) > 1 and np.any(np.diff(value) <= 0):
            raise ValueError(f"{name} 必须严格递增")

    action_time = arrays.timestamp_action
    tolerance_s = 1e-6
    for sensor_name, sensor_time, valid in (
        ("external", arrays.timestamp_external, arrays.external_valid),
        ("wrist", arrays.timestamp_wrist, arrays.wrist_valid),
        ("proprio", arrays.timestamp_proprio, arrays.proprio_valid),
    ):
        delta = action_time[valid] - sensor_time[valid]
        if np.any(delta < -1e-9):
            raise ValueError(f"{sensor_name} 使用了晚于 action 的未来观测")
        if np.any(delta > tolerance_s):
            raise ValueError(f"ManiSkill {sensor_name} 与 action 不在同一 Simulator Tick")

    both_cameras = arrays.external_valid & arrays.wrist_valid
    camera_skew = np.abs(
        arrays.timestamp_external[both_cameras] - arrays.timestamp_wrist[both_cameras]
    )
    if np.any(camera_skew > tolerance_s):
        raise ValueError("ManiSkill 双相机不在同一 Simulator Tick")

    if len(action_time) > 1:
        expected_dt = 1.0 / spec.control_hz
        median_dt = float(np.median(np.diff(action_time)))
        if abs(median_dt - expected_dt) / expected_dt > 0.2:
            raise ValueError(
                f"timestamp_action 中位间隔 {median_dt:.6f}s 与契约 {expected_dt:.6f}s 偏差过大"
            )


def validate_trajectory(arrays: TrajectoryArrays, meta: TrajectoryMeta, spec: RobotSpec) -> None:
    length = arrays.num_steps
    if length != meta.num_steps:
        raise ValueError(
            f"{meta.trajectory_id}: num_steps 声明为 {meta.num_steps}，实际为 {length}"
        )
    _validate_rgb(arrays.rgb_external, length, "rgb_external")
    _validate_rgb(arrays.rgb_wrist, length, "rgb_wrist")

    for name in (
        "timestamp_external",
        "timestamp_wrist",
        "timestamp_proprio",
        "timestamp_action",
    ):
        _validate_array(getattr(arrays, name), (length,), np.float64, name)
    _validate_array(arrays.proprio, (length, spec.proprio_dim), np.float32, "proprio")
    _validate_array(arrays.action, (length, spec.action_dim), np.float32, "action")
    for name in (
        "external_valid",
        "wrist_valid",
        "proprio_valid",
        "terminated",
        "truncated",
        "success",
    ):
        _validate_array(getattr(arrays, name), (length,), np.bool_, name)
    _validate_array(arrays.skill_id, (length,), np.int16, "skill_id")

    if arrays.event_state_available:
        if (
            meta.randomization.get("event_state_contract_version")
            != EVENT_STATE_CONTRACT_VERSION
        ):
            raise ValueError("event-state contract version 不兼容")
        for name in (
            "robot_object_contact_force_n",
            "support_contact_force_n",
        ):
            value = getattr(arrays, name)
            _validate_array(value, (length,), np.float32, name)
            if not np.isfinite(value).all() or np.any(value < 0.0):
                raise ValueError(f"{name} 必须是有限非负数")
        _validate_array(arrays.is_grasped, (length,), np.bool_, "is_grasped")
        for name in (
            "object_position_m",
            "object_linear_velocity_m_s",
            "object_angular_velocity_rad_s",
        ):
            value = getattr(arrays, name)
            _validate_array(value, (length, 3), np.float32, name)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} 包含 NaN 或 Inf")
        for name in (
            "commanded_joint_target_rad",
            "applied_joint_correction_rad",
        ):
            value = getattr(arrays, name)
            _validate_array(value, (length, spec.arm_dof), np.float32, name)
            if not np.isfinite(value).all():
                raise ValueError(f"{name} 包含 NaN 或 Inf")
        correction_limits = np.asarray(
            spec.effective_joint_delta_limits_rad,
            dtype=np.float32,
        )
        if np.any(np.abs(arrays.applied_joint_correction_rad) > correction_limits + 1e-6):
            raise ValueError("applied_joint_correction_rad 超出单步安全限制")
    elif meta.randomization.get("event_state_contract_version") is not None:
        raise ValueError("声明 event-state contract 的轨迹缺少 optional arrays")

    local_dagger_available = arrays.local_dagger_available
    if meta.local_dagger is None:
        if local_dagger_available:
            raise ValueError("Clean trajectory 不能携带未声明的 Local DAgger arrays")
    else:
        if not local_dagger_available:
            raise ValueError("Local DAgger trajectory 缺少逐 Action source/supervision arrays")
        if meta.split != "train":
            raise ValueError("第一版 Local DAgger trajectory 只能进入 train split")
        provenance = meta.local_dagger
        if meta.randomization.get("seed") != provenance.rollin_seed:
            raise ValueError("Local DAgger rollin_seed 与 environment seed 不一致")
        takeover = provenance.expert_takeover_step
        if takeover >= length:
            raise ValueError("Local DAgger Expert takeover 超出 Episode")
        expected_window_end = min(takeover + LOCAL_DAGGER_WINDOW_STEPS, length)
        if provenance.training_window_end != expected_window_end:
            raise ValueError("Local DAgger training window 必须固定为 takeover 后 64 步")
        if provenance.training_window_end - provenance.training_window_start < spec.action_horizon:
            raise ValueError("Local DAgger training window 不足一个完整 Action Chunk")
        _validate_array(arrays.action_source, (length,), np.int8, "action_source")
        _validate_array(
            arrays.expert_supervision_mask,
            (length,),
            np.bool_,
            "expert_supervision_mask",
        )
        allowed_sources = np.isin(
            arrays.action_source,
            (ACTION_SOURCE_POLICY, ACTION_SOURCE_EXPERT),
        )
        if not np.all(allowed_sources):
            raise ValueError("action_source 只能是 policy=0 或 expert=1")
        expected_source = np.full(length, ACTION_SOURCE_EXPERT, dtype=np.int8)
        expected_source[:takeover] = ACTION_SOURCE_POLICY
        if not np.array_equal(arrays.action_source, expected_source):
            raise ValueError("Local DAgger action_source 必须在 takeover 处连续切换一次")
        expected_supervision = expected_source == ACTION_SOURCE_EXPERT
        if not np.array_equal(arrays.expert_supervision_mask, expected_supervision):
            raise ValueError("expert_supervision_mask 与 action_source 不一致")

    if not np.isfinite(arrays.proprio).all() or not np.isfinite(arrays.action).all():
        raise ValueError("proprio/action 包含 NaN 或 Inf")
    _validate_timestamps(arrays, spec)

    q = arrays.proprio[:, : spec.arm_dof]
    dq = arrays.proprio[:, spec.arm_dof : spec.arm_dof * 2]
    g = arrays.proprio[:, -1]
    position_limits = np.asarray(spec.joint_position_limits_rad, dtype=np.float32)
    velocity_limits = np.asarray(spec.joint_velocity_limits_rad_s, dtype=np.float32)
    if np.any(q < position_limits[:, 0] - 1e-5) or np.any(q > position_limits[:, 1] + 1e-5):
        raise ValueError("proprio q 超出 Franka 关节位置限制")
    if np.any(np.abs(dq) > velocity_limits + 1e-5):
        raise ValueError("proprio dq 超出 Franka 关节速度限制")
    if np.any(g < 0.0) or np.any(g > 1.0):
        raise ValueError("proprio g 必须位于 [0,1]")

    delta_q = arrays.action[:, : spec.arm_dof]
    gripper_target = arrays.action[:, -1]
    delta_limits = np.asarray(spec.effective_joint_delta_limits_rad, dtype=np.float32)
    if np.any(np.abs(delta_q) > delta_limits + 1e-6):
        raise ValueError("action delta_q 超出单控制周期限制")
    if np.any(gripper_target < 0.0) or np.any(gripper_target > 1.0):
        raise ValueError("action gripper_target 必须位于 [0,1]")

    if np.any(arrays.terminated & arrays.truncated):
        raise ValueError("同一 Transition 不能同时 terminated 和 truncated")
    terminal = arrays.terminated | arrays.truncated
    if not terminal[-1] or np.any(terminal[:-1]):
        raise ValueError("完整 Episode 必须且只能在最后一个 Transition 结束")

    valid_skill_ids = (arrays.skill_id == UNKNOWN_SKILL_ID) | (
        (arrays.skill_id >= 0) & (arrays.skill_id < len(meta.task.skill_names))
    )
    if not np.all(valid_skill_ids):
        raise ValueError("skill_id 不在 TaskSpec 映射或 unknown 保留值中")


class TrajectoryStore:
    """带小型 LRU 缓存的严格 trajectory/v2 读取器。"""

    def __init__(self, root: str | Path, spec: RobotSpec, cache_size: int = 2) -> None:
        if cache_size < 0:
            raise ValueError("cache_size 不能为负数")
        self.root = Path(root)
        self.spec = spec
        self.cache_size = cache_size
        self._cache: OrderedDict[str, TrajectoryArrays] = OrderedDict()

    def get(self, meta: TrajectoryMeta) -> TrajectoryArrays:
        if meta.trajectory_id in self._cache:
            arrays = self._cache.pop(meta.trajectory_id)
            self._cache[meta.trajectory_id] = arrays
            return arrays

        arrays = read_trajectory(self.root, meta)
        validate_trajectory(arrays, meta, self.spec)
        if self.cache_size > 0:
            self._cache[meta.trajectory_id] = arrays
            while len(self._cache) > self.cache_size:
                self._cache.popitem(last=False)
        return arrays

    def iter_validated(self, entries: list[TrajectoryMeta]) -> Iterator[TrajectoryArrays]:
        for entry in entries:
            yield self.get(entry)
