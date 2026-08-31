"""可信 trajectory/v2 数据集的原子写入和确定性 scene 切分。"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from robot_vla.contracts import RobotSpec
from robot_vla.data.events import EVENT_STATE_ARRAYS
from robot_vla.data.trajectory import (
    LOCAL_DAGGER_ARRAYS,
    OBSERVATION_V2_ARRAYS,
    REQUIRED_ARRAYS,
    TrajectoryArrays,
    TrajectoryMeta,
    load_manifest,
    validate_trajectory,
)


def plan_scene_splits(
    scene_ids: Sequence[str],
    *,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, str]:
    """按 scene 的稳定哈希分配 split，小数据时保证三个 split 均非空。"""

    unique = set(scene_ids)
    if len(unique) != len(scene_ids):
        raise ValueError("scene_ids 不能重复")
    if len(unique) < 3:
        raise ValueError("可信 train/val/test 数据至少需要 3 个不同 scene")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("train_fraction/val_fraction 必须位于 (0,1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction 必须小于 1")

    ranked = sorted(
        unique,
        key=lambda value: (hashlib.sha256(value.encode("utf-8")).digest(), value),
    )
    total = len(ranked)
    train_count = max(1, round(total * train_fraction))
    val_count = max(1, round(total * val_fraction))
    if train_count + val_count >= total:
        train_count = total - val_count - 1
    if train_count <= 0:
        raise ValueError("split 比例没有为 train 留出 scene")

    result: dict[str, str] = {}
    for index, scene_id in enumerate(ranked):
        if index < train_count:
            split = "train"
        elif index < train_count + val_count:
            split = "val"
        else:
            split = "test"
        result[scene_id] = split
    return result


class TrajectoryDatasetWriter:
    """先严格校验，再以 NPZ→manifest 顺序原子提交一条完整 Episode。"""

    def __init__(self, root: str | Path, spec: RobotSpec) -> None:
        self.root = Path(root)
        self.spec = spec

    def write(self, meta: TrajectoryMeta, arrays: TrajectoryArrays) -> Path:
        validate_trajectory(arrays, meta, self.spec)
        if np.count_nonzero(arrays.success) != 1 or not bool(arrays.success[-1]):
            raise ValueError("可信训练轨迹必须且只能在最后一步成功")

        existing = self._existing_entries()
        if any(item.trajectory_id == meta.trajectory_id for item in existing):
            raise ValueError(f"trajectory_id 已存在: {meta.trajectory_id}")
        if any(item.file == meta.file for item in existing):
            raise ValueError(f"轨迹文件已在 manifest 中: {meta.file}")

        target = self._resolve_target(meta.file)
        if target.exists():
            raise FileExistsError(f"拒绝覆盖已有轨迹文件: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        self.root.mkdir(parents=True, exist_ok=True)

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w+b",
                prefix=f".{target.name}.",
                suffix=".tmp",
                dir=target.parent,
                delete=False,
            ) as handle:
                temporary_path = Path(handle.name)
                payload = {name: getattr(arrays, name) for name in REQUIRED_ARRAYS}
                if arrays.event_state_available:
                    payload.update(
                        {name: getattr(arrays, name) for name in EVENT_STATE_ARRAYS}
                    )
                if arrays.local_dagger_available:
                    payload.update(
                        {name: getattr(arrays, name) for name in LOCAL_DAGGER_ARRAYS}
                    )
                if arrays.observation_v2_available:
                    payload.update(
                        {name: getattr(arrays, name) for name in OBSERVATION_V2_ARRAYS}
                    )
                np.savez_compressed(
                    handle,
                    **payload,
                )
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(temporary_path, target)
            temporary_path = None
            self._replace_manifest([*existing, meta])
        except Exception:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)
            # manifest 尚未提交时，删除本次刚创建的孤立 NPZ。
            if not any(item.file == meta.file for item in self._existing_entries()):
                target.unlink(missing_ok=True)
            raise
        return target

    def _existing_entries(self) -> list[TrajectoryMeta]:
        manifest = self.root / "manifest.jsonl"
        if not manifest.is_file() or not manifest.read_text(encoding="utf-8").strip():
            return []
        return load_manifest(self.root)

    def _resolve_target(self, relative_path: str) -> Path:
        if Path(relative_path).suffix != ".npz":
            raise ValueError("轨迹文件必须使用 .npz 后缀")
        root = self.root.resolve()
        target = (root / relative_path).resolve()
        if not target.is_relative_to(root):
            raise ValueError("轨迹文件不能位于数据根目录之外")
        return target

    def _replace_manifest(self, entries: Sequence[TrajectoryMeta]) -> None:
        manifest = self.root / "manifest.jsonl"
        payload = "".join(
            json.dumps(entry.to_dict(), sort_keys=True, allow_nan=False) + "\n"
            for entry in entries
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
            os.replace(temporary, manifest)
        finally:
            temporary.unlink(missing_ok=True)


__all__ = ["TrajectoryDatasetWriter", "plan_scene_splits"]
