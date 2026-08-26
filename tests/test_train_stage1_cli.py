import json

import pytest

from robot_vla.cli.train_stage1 import (
    _load_audit_identity,
    _resolve_skill_sampling_weights,
    _should_save_checkpoint,
)


def test_checkpoint_schedule_never_loses_non_periodic_best() -> None:
    assert _should_save_checkpoint(
        completed_epoch=56,
        total_epochs=100,
        every_epochs=5,
        validation_improved=True,
    )
    assert _should_save_checkpoint(
        completed_epoch=60,
        total_epochs=100,
        every_epochs=5,
        validation_improved=False,
    )
    assert _should_save_checkpoint(
        completed_epoch=100,
        total_epochs=100,
        every_epochs=7,
        validation_improved=False,
    )
    assert not _should_save_checkpoint(
        completed_epoch=56,
        total_epochs=100,
        every_epochs=5,
        validation_improved=False,
    )


def test_training_rejects_stale_audit_manifest_hash(tmp_path, meta_factory) -> None:
    entry = meta_factory(randomization={"seed": 7})
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(entry.to_dict()) + "\n",
        encoding="utf-8",
    )
    (tmp_path / "audit_report.json").write_text(
        json.dumps(
            {
                "success_rate": 1.0,
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


def test_skill_sampling_weights_are_explicit_and_validated() -> None:
    assert _resolve_skill_sampling_weights([1.5, 1.0, 1.0, 1.5, 2.0]) == (
        (0, 1.5),
        (1, 1.0),
        (2, 1.0),
        (3, 1.5),
        (4, 2.0),
    )
    with pytest.raises(ValueError, match="恰好包含"):
        _resolve_skill_sampling_weights([1.0])
    with pytest.raises(ValueError, match="有限正数"):
        _resolve_skill_sampling_weights([1.0, 1.0, 0.0, 1.0, 1.0])
