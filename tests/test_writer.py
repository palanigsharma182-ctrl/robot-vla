from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import LocalDaggerProvenance, TrajectoryStore, load_manifest
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


def test_writer_persists_local_dagger_optional_arrays(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    steps = 80
    takeover = 4
    source = np.ones(steps, dtype=np.int8)
    source[:takeover] = 0
    arrays = arrays_factory(
        steps=steps,
        action_source=source,
        expert_supervision_mask=source == 1,
    )
    meta = meta_factory(
        num_steps=steps,
        local_dagger=LocalDaggerProvenance(
            source="dagger_grasp_lift",
            rollin_seed=7,
            rollin_policy_checkpoint_sha256="b" * 64,
            boundary_type="grasp_lift",
            boundary_detection_step=takeover,
            expert_takeover_step=takeover,
            training_window_start=takeover,
            training_window_end=takeover + 64,
            expert_recovery_success=True,
        ),
    )

    TrajectoryDatasetWriter(tmp_path, RobotSpec()).write(meta, arrays)
    loaded = TrajectoryStore(tmp_path, RobotSpec()).get(meta)

    np.testing.assert_array_equal(loaded.action_source, source)
    np.testing.assert_array_equal(loaded.expert_supervision_mask, source == 1)
