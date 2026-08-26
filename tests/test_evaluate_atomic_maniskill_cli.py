from __future__ import annotations

import json

import pytest

from robot_vla.cli.evaluate_atomic_maniskill import build_atomic_specs


def test_atomic_specs_skip_dataset_seeds_and_cross_product_skills(tmp_path, meta_factory) -> None:
    entry = meta_factory(randomization={"seed": 10_000})
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(entry.to_dict()) + "\n",
        encoding="utf-8",
    )

    specs = build_atomic_specs(
        tmp_path,
        seed_start=10_000,
        episodes=2,
        skills=["reach", "place"],
    )

    assert [(item.seed, item.skill_name) for item in specs] == [
        (10_001, "reach"),
        (10_001, "place"),
        (10_002, "reach"),
        (10_002, "place"),
    ]


def test_atomic_specs_reject_duplicate_skills(tmp_path, meta_factory) -> None:
    entry = meta_factory(randomization={"seed": 7})
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(entry.to_dict()) + "\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="重复"):
        build_atomic_specs(
            tmp_path,
            seed_start=10_000,
            episodes=1,
            skills=["reach", "reach"],
        )
