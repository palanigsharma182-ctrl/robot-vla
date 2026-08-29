"""可信训练数据的完整性、语义和 split 泄漏审计。"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

import numpy as np

from robot_vla.adapters import ProprioStats
from robot_vla.contracts import OUTCOME_PREDICATE_VERSION, RobotSpec
from robot_vla.data.events import (
    EVENT_TYPES,
    EventDetectionConfig,
    detect_trajectory_events,
)
from robot_vla.data.recovery import RECOVERY_CONTRACT_VERSION, RECOVERY_PROFILES
from robot_vla.data.trajectory import TrajectoryStore, load_manifest
from robot_vla.tasks.pick_place import PickPlacePredicateConfig


@dataclass(frozen=True)
class DatasetAuditReport:
    dataset_sha256: str
    manifest_sha256: str
    trajectory_count: int
    step_count: int
    success_rate: float
    split_trajectory_counts: dict[str, int]
    split_step_counts: dict[str, int]
    episode_length_min: int
    episode_length_mean: float
    episode_length_max: int
    skill_frame_counts: dict[str, int]
    arm_action_saturation_rate: float
    external_image_shape: tuple[int, int, int]
    wrist_image_shape: tuple[int, int, int]
    proprio_stats_count: int
    event_detection_config: dict[str, object]
    event_state_trajectory_count: int
    event_state_step_count: int
    event_state_coverage_rate: float
    event_counts: dict[str, int]
    event_counts_by_split: dict[str, dict[str, int]]
    multi_event_step_count: int

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def audit_dataset(
    root: str | Path,
    spec: RobotSpec,
    *,
    write_artifacts: bool = True,
) -> DatasetAuditReport:
    """审计完整成功 Episode，并只用 train split 固定 ProprioStats。"""

    root = Path(root)
    entries = load_manifest(root)
    split_counts = {split: 0 for split in ("train", "val", "test")}
    split_steps = {split: 0 for split in ("train", "val", "test")}
    skill_counts = {name: 0 for name in entries[0].task.skill_names}
    lengths: list[int] = []
    train_proprio: list[np.ndarray] = []
    external_shapes: set[tuple[int, int, int]] = set()
    wrist_shapes: set[tuple[int, int, int]] = set()
    saturation_count = 0
    arm_action_count = 0
    successful = 0
    canonical_files: list[dict[str, object]] = []
    store = TrajectoryStore(root, spec, cache_size=0)
    predicate_config = PickPlacePredicateConfig()
    event_config = EventDetectionConfig()
    event_state_trajectory_count = 0
    event_state_step_count = 0
    event_counts = {name: 0 for name in EVENT_TYPES}
    event_counts_by_split = {
        split: {name: 0 for name in EVENT_TYPES}
        for split in ("train", "val", "test")
    }
    multi_event_step_count = 0

    for meta in entries:
        arrays = store.get(meta)
        _audit_episode(arrays, meta, spec, predicate_config)
        successful += int(bool(arrays.success[-1]))
        split_counts[meta.split] += 1
        split_steps[meta.split] += arrays.num_steps
        lengths.append(arrays.num_steps)
        if meta.split == "train":
            train_proprio.append(arrays.proprio)

        external_shapes.add(tuple(int(value) for value in arrays.rgb_external.shape[1:]))
        wrist_shapes.add(tuple(int(value) for value in arrays.rgb_wrist.shape[1:]))
        for skill_id, name in enumerate(meta.task.skill_names):
            skill_counts[name] += int(np.count_nonzero(arrays.skill_id == skill_id))

        arm_actions = np.abs(arrays.action[:, : spec.arm_dof])
        limits = np.asarray(spec.effective_joint_delta_limits_rad, dtype=np.float32)
        saturation_count += int(np.count_nonzero(arm_actions >= limits * 0.999))
        arm_action_count += int(arm_actions.size)
        events = detect_trajectory_events(arrays, event_config)
        if events.event_state_available:
            event_state_trajectory_count += 1
            event_state_step_count += arrays.num_steps
        stacked_events = np.stack(list(events.by_type.values()), axis=0)
        multi_event_step_count += int(np.count_nonzero(stacked_events.sum(axis=0) > 1))
        for name, count in events.counts.items():
            event_counts[name] += count
            event_counts_by_split[meta.split][name] += count
        canonical_files.append(
            {
                "meta": meta.to_dict(),
                "npz_sha256": _sha256_file(root / meta.file),
            }
        )

    if any(count == 0 for count in split_counts.values()):
        raise ValueError(f"train/val/test 必须都有可信轨迹，实际为 {split_counts}")
    if len(external_shapes) != 1 or len(wrist_shapes) != 1:
        raise ValueError("同一数据集的 external/wrist 图像 shape 必须固定")
    if any(count == 0 for count in skill_counts.values()):
        raise ValueError(f"数据集存在零帧原子技能: {skill_counts}")

    stats = ProprioStats.fit(train_proprio, spec)
    canonical_payload = json.dumps(
        sorted(canonical_files, key=lambda item: str(item["meta"])),
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    manifest_path = root / "manifest.jsonl"
    report = DatasetAuditReport(
        dataset_sha256=hashlib.sha256(canonical_payload).hexdigest(),
        manifest_sha256=_sha256_file(manifest_path),
        trajectory_count=len(entries),
        step_count=sum(lengths),
        success_rate=successful / len(entries),
        split_trajectory_counts=split_counts,
        split_step_counts=split_steps,
        episode_length_min=min(lengths),
        episode_length_mean=mean(lengths),
        episode_length_max=max(lengths),
        skill_frame_counts=skill_counts,
        arm_action_saturation_rate=saturation_count / arm_action_count,
        external_image_shape=next(iter(external_shapes)),
        wrist_image_shape=next(iter(wrist_shapes)),
        proprio_stats_count=stats.count,
        event_detection_config=asdict(event_config),
        event_state_trajectory_count=event_state_trajectory_count,
        event_state_step_count=event_state_step_count,
        event_state_coverage_rate=event_state_step_count / sum(lengths),
        event_counts=event_counts,
        event_counts_by_split=event_counts_by_split,
        multi_event_step_count=multi_event_step_count,
    )
    if write_artifacts:
        stats.to_json(root / "proprio_stats.json")
        (root / "audit_report.json").write_text(
            json.dumps(report.to_dict(), indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
    return report


def audit_trajectory(arrays, meta, spec: RobotSpec) -> None:
    """对单条 smoke/候选轨迹复用正式 Episode 语义，不放宽 Dataset split 契约。"""

    _audit_episode(
        arrays,
        meta,
        spec,
        PickPlacePredicateConfig(),
    )


def _audit_episode(arrays, meta, spec: RobotSpec, config: PickPlacePredicateConfig) -> None:
    if np.count_nonzero(arrays.success) != 1 or not bool(arrays.success[-1]):
        raise ValueError(f"{meta.trajectory_id}: 必须且只能在最终 Transition 成功")
    expected_skill_ids = np.arange(len(meta.task.skill_names), dtype=np.int16)
    observed_skill_ids = np.unique(arrays.skill_id)
    if not np.array_equal(observed_skill_ids, expected_skill_ids):
        raise ValueError(
            f"{meta.trajectory_id}: 原子技能覆盖应为 {expected_skill_ids.tolist()}，"
            f"实际为 {observed_skill_ids.tolist()}"
        )
    if np.any(np.diff(arrays.skill_id) < 0) or np.any(np.diff(arrays.skill_id) > 1):
        raise ValueError(f"{meta.trajectory_id}: skill_id 必须单调、连续且不回退")
    _audit_recovery_metadata(meta, arrays.num_steps)

    expected_time = np.arange(arrays.num_steps, dtype=np.float64) / spec.control_hz
    for name in (
        "timestamp_external",
        "timestamp_wrist",
        "timestamp_proprio",
        "timestamp_action",
    ):
        if not np.allclose(getattr(arrays, name), expected_time, rtol=0.0, atol=1e-9):
            raise ValueError(f"{meta.trajectory_id}: {name} 必须精确来自 step_index/control_hz")

    evidence = meta.outcome_evidence
    if evidence is None:
        raise ValueError(f"{meta.trajectory_id}: 缺少 Outcome Predicate 审计证据")
    if evidence.predicate_version != OUTCOME_PREDICATE_VERSION:
        raise ValueError(f"{meta.trajectory_id}: Outcome Predicate version 不兼容")
    if not evidence.task_completed or not evidence.final_is_released:
        raise ValueError(f"{meta.trajectory_id}: 最终 place 未完成或方块仍被抓持")
    if evidence.stable_place_steps < config.stable_place_steps:
        raise ValueError(f"{meta.trajectory_id}: place 稳定帧数不足")
    if (
        evidence.external_goal_visible_steps <= 0
        or evidence.wrist_goal_visible_steps <= 0
        or evidence.both_goal_visible_steps <= 0
    ):
        raise ValueError(f"{meta.trajectory_id}: 目标没有在双相机同帧可见")
    if max(
        evidence.external_goal_visible_steps,
        evidence.wrist_goal_visible_steps,
        evidence.both_goal_visible_steps,
    ) > arrays.num_steps:
        raise ValueError(f"{meta.trajectory_id}: 目标可见帧数超过 Episode 长度")
    if evidence.final_object_to_goal_distance_m > config.place_distance_m:
        raise ValueError(f"{meta.trajectory_id}: 最终方块超出目标区域")
    if evidence.final_object_linear_speed_m_s > config.static_linear_speed_m_s:
        raise ValueError(f"{meta.trajectory_id}: 最终方块线速度过高")
    if evidence.final_object_angular_speed_rad_s > config.static_angular_speed_rad_s:
        raise ValueError(f"{meta.trajectory_id}: 最终方块角速度过高")


def _audit_recovery_metadata(meta, num_steps: int) -> None:
    profile = meta.randomization.get("recovery_profile")
    if profile is None:
        return
    if profile not in RECOVERY_PROFILES:
        raise ValueError(f"{meta.trajectory_id}: recovery_profile 无效")
    if meta.randomization.get("recovery_contract_version") != RECOVERY_CONTRACT_VERSION:
        raise ValueError(f"{meta.trajectory_id}: recovery contract version 不兼容")
    evidence = meta.randomization.get("recovery_evidence")
    if not isinstance(evidence, dict):
        raise TypeError(f"{meta.trajectory_id}: 缺少 recovery_evidence")
    disturbance_step = evidence.get("disturbance_end_step")
    recovery_step = evidence.get("successful_recovery_end_step")
    if (
        not isinstance(disturbance_step, int)
        or isinstance(disturbance_step, bool)
        or not isinstance(recovery_step, int)
        or isinstance(recovery_step, bool)
        or not 0 <= disturbance_step < recovery_step == num_steps - 1
    ):
        raise ValueError(f"{meta.trajectory_id}: recovery_evidence step 索引无效")


__all__ = ["DatasetAuditReport", "audit_dataset"]
