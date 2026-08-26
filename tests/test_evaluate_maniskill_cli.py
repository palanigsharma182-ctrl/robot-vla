from __future__ import annotations

import json
from dataclasses import replace

import pytest

from robot_vla.cli.evaluate_maniskill import _load_audit_identity, build_episode_specs


def test_episode_specs_include_test_split_and_non_overlapping_unseen_seeds(
    tmp_path,
    meta_factory,
) -> None:
    entries = [
        meta_factory(
            trajectory_id="train-000",
            source_episode_id="source-train-000",
            scene_id="scene-train-000",
            split="train",
            randomization={"seed": 10_000},
        ),
        meta_factory(
            trajectory_id="test-000",
            source_episode_id="source-test-000",
            scene_id="scene-test-000",
            split="test",
            randomization={"seed": 27},
        ),
        meta_factory(
            trajectory_id="test-001",
            source_episode_id="source-test-001",
            scene_id="scene-test-001",
            split="test",
            randomization={"seed": 28},
            task=replace(meta_factory().task, instruction="second instruction"),
        ),
    ]
    (tmp_path / "manifest.jsonl").write_text(
        "".join(json.dumps(entry.to_dict()) + "\n" for entry in entries),
        encoding="utf-8",
    )

    specs = build_episode_specs(
        tmp_path,
        test_episodes=None,
        unseen_seed_start=10_000,
        unseen_episodes=2,
    )

    assert [(spec.seed_group, spec.seed) for spec in specs] == [
        ("test", 27),
        ("test", 28),
        ("unseen", 10_001),
        ("unseen", 10_002),
    ]
    assert specs[1].instruction == "second instruction"


def test_episode_specs_reject_empty_selection(tmp_path, meta_factory) -> None:
    entry = meta_factory(randomization={"seed": 7})
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(entry.to_dict()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="至少需要"):
        build_episode_specs(
            tmp_path,
            test_episodes=0,
            unseen_seed_start=10_000,
            unseen_episodes=0,
        )


def test_evaluation_rejects_stale_audit_manifest_hash(tmp_path, meta_factory) -> None:
    entry = meta_factory(randomization={"seed": 7})
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(entry.to_dict()) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "audit_report.json").write_text(
        json.dumps(
            {
                "manifest_sha256": "stale",
                "dataset_sha256": "dataset",
                "trajectory_count": 1,
                "step_count": 5,
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="已过期"):
        _load_audit_identity(tmp_path)
