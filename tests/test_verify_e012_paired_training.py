from __future__ import annotations

import json

import pytest

from robot_vla.cli.verify_e012_paired_training import verify_pair


def _write_json(path, value) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path, rows) -> None:
    path.write_text(
        "".join(
            json.dumps(row, sort_keys=True, allow_nan=False) + "\n"
            for row in rows
        ),
        encoding="utf-8",
    )


def _make_run(root, *, dagger: bool) -> None:
    root.mkdir()
    source_weights = (
        [
            ["base_d0", 0.8],
            ["dagger_reach_grasp", 0.1],
            ["dagger_grasp_lift", 0.1],
        ]
        if dagger
        else [["base_d0", 1.0]]
    )
    initialization = {
        "mode": "init_checkpoint",
        "checkpoint": {
            "path": "/checkpoint.pt",
            "sha256": "a" * 64,
            "policy_state_sha256": "b" * 64,
            "metadata": {"format": "checkpoint"},
        },
        "restored_state": "adapter_expert_weights_only",
        "trainer_state_reset": True,
        "rng_restored": False,
    }
    config = {
        "samples_per_epoch": 10,
        "seed": 12_012,
        "micro_batch_size": 1,
        "gradient_accumulation_steps": 10,
        "source_sampling_weights": source_weights,
    }
    experiment = {
        "training_config": config,
        "initialization": initialization,
        "code_revision": "source-tree-sha256:" + "c" * 64,
        "proprio_stats": {"sha256": "d" * 64, "frozen_from_data": "/d0"},
        "dataset": {
            "base_d0": {"dataset_sha256": "e" * 64},
            "dagger_additions": {"dataset_sha256": "f" * 64} if dagger else None,
        },
    }
    _write_json(root / "experiment.json", experiment)
    metrics = []
    exposures = []
    for epoch in (1, 2):
        if dagger:
            rows = [
                {
                    "source": "base_d0",
                    "skill_id": 0,
                    "boundary_offset": None,
                    "samples": 8,
                },
                {
                    "source": "dagger_reach_grasp",
                    "skill_id": 1,
                    "boundary_offset": epoch - 1,
                    "samples": 1,
                },
                {
                    "source": "dagger_grasp_lift",
                    "skill_id": 2,
                    "boundary_offset": epoch - 1,
                    "samples": 1,
                },
            ]
        else:
            rows = [
                {
                    "source": "base_d0",
                    "skill_id": 0,
                    "boundary_offset": None,
                    "samples": 10,
                }
            ]
        exposure = {
            "format": "robot-vla-stage1-sampler-exposure/v1",
            "epoch": epoch,
            "configured_source_weights": source_weights,
            "samples": 10,
            "source_skill_boundary_offset": rows,
        }
        exposures.append(exposure)
        metrics.append(
            {
                "event": "epoch",
                "epoch": epoch,
                "train": {"examples": 10, "optimizer_steps": 1},
                "source_exposure": exposure,
            }
        )
    _write_jsonl(root / "metrics.jsonl", metrics)
    _write_jsonl(root / "sampler_exposure.jsonl", exposures)


def test_paired_verifier_accepts_only_source_and_dataset_intervention(tmp_path) -> None:
    replay = tmp_path / "replay"
    dagger = tmp_path / "dagger"
    _make_run(replay, dagger=False)
    _make_run(dagger, dagger=True)

    result = verify_pair(
        replay,
        dagger,
        expected_epochs=2,
        expected_samples_per_epoch=10,
        expected_optimizer_steps=2,
    )

    assert result["passed"] is True
    assert result["pi_replay"]["aggregate_source_exposure"] == {"base_d0": 20}
    assert result["pi_dagger"]["aggregate_source_exposure"] == {
        "base_d0": 16,
        "dagger_grasp_lift": 2,
        "dagger_reach_grasp": 2,
    }


def test_paired_verifier_rejects_non_source_config_drift(tmp_path) -> None:
    replay = tmp_path / "replay"
    dagger = tmp_path / "dagger"
    _make_run(replay, dagger=False)
    _make_run(dagger, dagger=True)
    experiment = json.loads((dagger / "experiment.json").read_text(encoding="utf-8"))
    experiment["training_config"]["seed"] = 22_012
    _write_json(dagger / "experiment.json", experiment)

    with pytest.raises(ValueError, match="training config"):
        verify_pair(
            replay,
            dagger,
            expected_epochs=2,
            expected_samples_per_epoch=10,
            expected_optimizer_steps=2,
        )
