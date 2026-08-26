from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from robot_vla.contracts import OUTCOME_PREDICATE_VERSION, RobotSpec
from robot_vla.data.audit import _audit_recovery_metadata, audit_dataset
from robot_vla.data.recovery import RECOVERY_CONTRACT_VERSION
from robot_vla.data.trajectory import OutcomeEvidence
from robot_vla.data.writer import TrajectoryDatasetWriter


def _evidence() -> OutcomeEvidence:
    return OutcomeEvidence(
        predicate_version=OUTCOME_PREDICATE_VERSION,
        task_completed=True,
        final_is_released=True,
        stable_place_steps=4,
        external_goal_visible_steps=5,
        wrist_goal_visible_steps=5,
        both_goal_visible_steps=5,
        final_object_to_goal_distance_m=0.01,
        final_object_linear_speed_m_s=0.005,
        final_object_angular_speed_rad_s=0.1,
    )


def _write_three_splits(root, meta_factory, arrays_factory, *, bad_skill=False) -> None:
    writer = TrajectoryDatasetWriter(root, RobotSpec())
    for index, split in enumerate(("train", "val", "test")):
        meta = replace(
            meta_factory(),
            trajectory_id=f"episode-{index:03d}",
            source_episode_id=f"source-{index:03d}",
            file=f"trajectories/episode-{index:03d}.npz",
            split=split,
            scene_id=f"scene-{index:03d}",
            outcome_evidence=_evidence(),
        )
        arrays = arrays_factory()
        if bad_skill and index == 0:
            arrays = replace(
                arrays,
                skill_id=np.asarray([0, 1, 1, 3, 4], dtype=np.int16),
            )
        writer.write(meta, arrays)


def test_audit_builds_train_only_stats_and_reproducible_hash(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    _write_three_splits(tmp_path, meta_factory, arrays_factory)

    first = audit_dataset(tmp_path, RobotSpec())
    second = audit_dataset(tmp_path, RobotSpec(), write_artifacts=False)

    assert first == second
    assert first.success_rate == 1.0
    assert first.split_trajectory_counts == {"train": 1, "val": 1, "test": 1}
    assert first.skill_frame_counts == {
        "reach": 3,
        "grasp": 3,
        "lift": 3,
        "transport": 3,
        "place": 3,
    }
    assert first.proprio_stats_count == 5
    stats = json.loads((tmp_path / "proprio_stats.json").read_text(encoding="utf-8"))
    assert stats["count"] == 5
    assert (tmp_path / "audit_report.json").is_file()


def test_audit_rejects_missing_or_skipped_atomic_skill(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    _write_three_splits(tmp_path, meta_factory, arrays_factory, bad_skill=True)

    with pytest.raises(ValueError, match="原子技能覆盖"):
        audit_dataset(tmp_path, RobotSpec(), write_artifacts=False)


def test_audit_requires_versioned_recovery_step_evidence(meta_factory) -> None:
    valid = replace(
        meta_factory(),
        num_steps=5,
        randomization={
            "seed": 7,
            "recovery_profile": "reach",
            "recovery_contract_version": RECOVERY_CONTRACT_VERSION,
            "recovery_evidence": {
                "disturbance_end_step": 1,
                "successful_recovery_end_step": 4,
            },
        },
    )
    _audit_recovery_metadata(valid, 5)

    invalid = replace(
        valid,
        randomization={**valid.randomization, "recovery_evidence": None},
    )
    with pytest.raises(TypeError, match="recovery_evidence"):
        _audit_recovery_metadata(invalid, 5)
