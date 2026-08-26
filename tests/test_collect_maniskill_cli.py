from __future__ import annotations

import json
from dataclasses import replace

import pytest

from robot_vla.cli.collect_maniskill import (
    _check_or_write_config,
    _is_compatible_extension,
    _next_recovery_profile,
    _validate_existing_split_plan,
)


def _config(train: int, val: int, test: int) -> dict[str, object]:
    return {
        "environment_id": "RobotVLAPickCubeToRegion-v1",
        "train": train,
        "val": val,
        "test": test,
        "start_seed": 0,
        "max_candidates": 300,
    }


def test_dataset_extension_requires_same_split_ratio_and_seed_range() -> None:
    existing = _config(24, 3, 3)
    assert _is_compatible_extension(existing, _config(80, 10, 10))
    assert not _is_compatible_extension(existing, _config(80, 15, 5))
    assert not _is_compatible_extension(
        existing,
        {**_config(80, 10, 10), "start_seed": 1},
    )
    assert not _is_compatible_extension(existing, _config(23, 3, 3))
    recovery = {**_config(80, 10, 10), "recovery_profiles": ["reach", "place"]}
    assert _is_compatible_extension(existing, recovery)
    assert not _is_compatible_extension(
        recovery,
        {**_config(96, 12, 12), "recovery_profiles": ["reach", "transport"]},
    )


def test_recovery_profile_selection_balances_successful_trajectories() -> None:
    profiles = ("reach", "grasp", "lift")
    counts = {"reach": 2, "grasp": 1, "lift": 1}
    assert _next_recovery_profile(profiles, counts) == "grasp"
    counts["grasp"] += 1
    assert _next_recovery_profile(profiles, counts) == "lift"
    assert _next_recovery_profile((), {}) is None


def test_recovery_profile_selection_respects_explicit_final_targets() -> None:
    profiles = ("reach", "grasp", "transport")
    counts = {"reach": 4, "grasp": 4, "transport": 4}
    targets = {"reach": 6, "grasp": 5, "transport": 8}

    assert _next_recovery_profile(profiles, counts, targets) == "transport"
    counts["transport"] = 8
    counts["reach"] = 6
    counts["grasp"] = 5
    assert _next_recovery_profile(profiles, counts, targets) is None


def test_collection_config_extension_is_explicit_and_persists_new_target(tmp_path) -> None:
    _check_or_write_config(tmp_path, _config(24, 3, 3))
    with pytest.raises(ValueError, match="不兼容"):
        _check_or_write_config(tmp_path, _config(80, 10, 10))

    _check_or_write_config(
        tmp_path,
        _config(80, 10, 10),
        allow_extension=True,
    )

    actual = json.loads((tmp_path / "collection_config.json").read_text(encoding="utf-8"))
    assert actual == _config(80, 10, 10)


def test_existing_manifest_must_keep_deterministic_scene_split(tmp_path, meta_factory) -> None:
    entry = meta_factory(split="test")
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(entry.to_dict()) + "\n",
        encoding="utf-8",
    )
    _validate_existing_split_plan(tmp_path, {entry.scene_id: "test"})

    mismatched = replace(entry, split="train")
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(mismatched.to_dict()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="split"):
        _validate_existing_split_plan(tmp_path, {entry.scene_id: "test"})
