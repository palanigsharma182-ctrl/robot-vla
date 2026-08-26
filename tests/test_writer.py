from __future__ import annotations

from dataclasses import replace

import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import load_manifest
from robot_vla.data.writer import TrajectoryDatasetWriter, plan_scene_splits


def test_writer_atomically_commits_valid_success_episode(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    writer = TrajectoryDatasetWriter(tmp_path, RobotSpec())
    meta = meta_factory()
    target = writer.write(meta, arrays_factory())

    assert target.is_file()
    assert load_manifest(tmp_path) == [meta]
    with pytest.raises(ValueError, match="trajectory_id 已存在"):
        writer.write(meta, arrays_factory())


def test_writer_rejects_non_success_episode(tmp_path, meta_factory, arrays_factory) -> None:
    success = arrays_factory().success.copy()
    success[-1] = False

    with pytest.raises(ValueError, match="最后一步成功"):
        TrajectoryDatasetWriter(tmp_path, RobotSpec()).write(
            meta_factory(),
            replace(arrays_factory(), success=success),
        )
    assert not (tmp_path / "manifest.jsonl").exists()


def test_scene_split_plan_is_reproducible_and_non_empty() -> None:
    scene_ids = [f"scene-{index:03d}" for index in range(10)]

    first = plan_scene_splits(scene_ids)
    second = plan_scene_splits(list(reversed(scene_ids)))

    assert first == second
    assert set(first.values()) == {"train", "val", "test"}
    assert sum(split == "train" for split in first.values()) == 8
    assert sum(split == "val" for split in first.values()) == 1
    assert sum(split == "test" for split in first.values()) == 1
