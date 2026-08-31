import json

import pytest

from robot_vla.cli.train_stage1 import (
    _load_audit_identity,
    _resolve_skill_sampling_weights,
    _resolve_source_sampling_weights,
    _should_save_checkpoint,
    _validate_output_path,
    _validate_resume_artifacts,
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


def test_source_sampling_weights_are_explicit_and_validated() -> None:
    assert _resolve_source_sampling_weights(
        [
            "base_d0=0.8",
            "dagger_reach_grasp=0.1",
            "dagger_grasp_lift=0.1",
        ]
    ) == (
        ("base_d0", 0.8),
        ("dagger_reach_grasp", 0.1),
        ("dagger_grasp_lift", 0.1),
    )
    with pytest.raises(ValueError, match="SOURCE=WEIGHT"):
        _resolve_source_sampling_weights(["base_d0"])
    with pytest.raises(ValueError, match="未知 source"):
        _resolve_source_sampling_weights(["unknown=1"])
    with pytest.raises(ValueError, match="重复定义"):
        _resolve_source_sampling_weights(["base_d0=1", "base_d0=1"])
    with pytest.raises(ValueError, match="有限正数"):
        _resolve_source_sampling_weights(["base_d0=0"])


def test_training_output_path_is_single_owner_empty_or_strict_resume(tmp_path) -> None:
    output = tmp_path / "run"
    _validate_output_path(output, resume=False)
    output.mkdir(mode=0o700)
    output.chmod(0o700)
    _validate_output_path(output, resume=False)
    _validate_output_path(output, resume=True)
    (output / "existing.json").write_text("{}\n", encoding="utf-8")
    with pytest.raises(FileExistsError, match="必须为空"):
        _validate_output_path(output, resume=False)
    output.chmod(0o750)
    with pytest.raises(PermissionError, match="mode=0700"):
        _validate_output_path(output, resume=True)
    symlink = tmp_path / "run-link"
    symlink.symlink_to(output, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        _validate_output_path(symlink, resume=True)


def test_resume_artifacts_must_match_checkpoint_epoch_and_experiment(tmp_path) -> None:
    experiment = {
        "arguments": {
            "epochs": 2,
            "skill_weights": (1.5, 1.0, 1.0, 1.5, 2.0),
            "resume": None,
            "init_checkpoint": "/pi0.pt",
        },
        "training_config": {"seed": 12_012, "source_sampling_weights": ()},
        "dataset": {"base_d0": {"dataset_sha256": "a" * 64}},
        "proprio_stats": {"sha256": "b" * 64},
        "code_revision": "source",
        "trainable_parameters": 10,
        "frozen_parameters": 20,
        "gpu": "gpu",
    }
    normalized = json.loads(json.dumps(experiment))
    experiment_path = tmp_path / "experiment.json"
    metrics_path = tmp_path / "metrics.jsonl"
    exposure_path = tmp_path / "sampler_exposure.jsonl"
    experiment_path.write_text(json.dumps(normalized) + "\n", encoding="utf-8")
    metrics_path.write_text(
        json.dumps({"event": "epoch", "epoch": 1}) + "\n",
        encoding="utf-8",
    )
    exposure_path.write_text(json.dumps({"epoch": 1}) + "\n", encoding="utf-8")

    resumed = json.loads(json.dumps(experiment))
    resumed["arguments"]["resume"] = "/latest.pt"
    resumed["arguments"]["init_checkpoint"] = None
    _validate_resume_artifacts(
        experiment_path=experiment_path,
        metrics_path=metrics_path,
        exposure_path=exposure_path,
        expected_experiment=resumed,
        completed_epochs=1,
        overfit=False,
    )

    resumed["dataset"] = {"base_d0": {"dataset_sha256": "c" * 64}}
    with pytest.raises(ValueError, match="dataset 漂移"):
        _validate_resume_artifacts(
            experiment_path=experiment_path,
            metrics_path=metrics_path,
            exposure_path=exposure_path,
            expected_experiment=resumed,
            completed_epochs=1,
            overfit=False,
        )
