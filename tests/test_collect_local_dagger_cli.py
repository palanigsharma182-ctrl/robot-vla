import sys
from types import SimpleNamespace

import pytest

from robot_vla.cli.collect_local_dagger import (
    _CANDIDATE_STAGING_MARKER,
    _CandidateDatasetPublisher,
    _parse_args,
    _validate_candidate_publish_gates,
    derive_collection_sampling_seed,
)
from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import load_manifest
from robot_vla.data.writer import TrajectoryDatasetWriter


def test_collection_sampling_seed_is_paired_and_boundary_specific() -> None:
    first = derive_collection_sampling_seed(
        52_012,
        environment_seed=29_990,
        boundary_type="reach_grasp",
    )
    repeated = derive_collection_sampling_seed(
        52_012,
        environment_seed=29_990,
        boundary_type="reach_grasp",
    )
    other_boundary = derive_collection_sampling_seed(
        52_012,
        environment_seed=29_990,
        boundary_type="grasp_lift",
    )

    assert first == repeated
    assert first != other_boundary


def test_collection_sampling_seed_rejects_invalid_identity() -> None:
    with pytest.raises(ValueError, match="未知 boundary"):
        derive_collection_sampling_seed(
            52_012,
            environment_seed=29_990,
            boundary_type="lift_transport",
        )


def test_action_budget_protocol_cli_is_legacy_by_default_and_opt_in(
    monkeypatch,
) -> None:
    required = [
        "collect_local_dagger",
        "--data",
        "/d0",
        "--model-cache",
        "/model",
        "--checkpoint",
        "/checkpoint.pt",
        "--output",
        "/output",
        "--record",
        "/record.json",
        "--seed",
        "30200",
        "--boundary-type",
        "grasp_lift",
    ]
    monkeypatch.setattr(sys, "argv", required)
    assert _parse_args().action_budget_protocol == "legacy"

    monkeypatch.setattr(
        sys,
        "argv",
        required + ["--action-budget-protocol", "segmented-300-180-480"],
    )
    assert _parse_args().action_budget_protocol == "segmented-300-180-480"


def test_candidate_dataset_is_staged_then_atomically_published(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    canonical = tmp_path / "dataset"
    meta = meta_factory()

    with _CandidateDatasetPublisher(canonical) as publisher:
        TrajectoryDatasetWriter(publisher.root, RobotSpec()).write(
            meta,
            arrays_factory(),
        )
        assert not canonical.exists()
        assert (publisher.root / "manifest.jsonl").is_file()

        publisher.publish()
        assert (canonical / "manifest.jsonl").is_file()
        assert (canonical / _CANDIDATE_STAGING_MARKER).is_file()
        publisher.prepare_commit()
        assert not (canonical / _CANDIDATE_STAGING_MARKER).exists()
        publisher.commit()

    assert load_manifest(canonical) == [meta]
    assert not (canonical / _CANDIDATE_STAGING_MARKER).exists()
    assert not list(tmp_path.glob(".dataset.candidate-staging-*"))


def test_candidate_rejection_cleans_staging_without_canonical_artifacts(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    canonical = tmp_path / "dataset"

    with (
        pytest.raises(RuntimeError, match="paired gate rejected"),
        _CandidateDatasetPublisher(canonical) as publisher,
    ):
        TrajectoryDatasetWriter(publisher.root, RobotSpec()).write(
            meta_factory(),
            arrays_factory(),
        )
        raise RuntimeError("paired gate rejected")

    assert not canonical.exists()
    assert not list(tmp_path.glob(".dataset.candidate-staging-*"))


def test_candidate_crash_after_publish_before_prepare_rolls_back_canonical_artifacts(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    canonical = tmp_path / "dataset"

    with (
        pytest.raises(RuntimeError, match="post-publish audit crashed"),
        _CandidateDatasetPublisher(canonical) as publisher,
    ):
        TrajectoryDatasetWriter(publisher.root, RobotSpec()).write(
            meta_factory(),
            arrays_factory(),
        )
        publisher.publish()
        assert (canonical / _CANDIDATE_STAGING_MARKER).is_file()
        raise RuntimeError("post-publish audit crashed")

    assert not canonical.exists()
    assert not list(tmp_path.glob(".dataset.candidate-staging-*"))


def test_candidate_crash_after_prepare_rolls_back_canonical_artifacts(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    canonical = tmp_path / "dataset"

    with (
        pytest.raises(RuntimeError, match="record write crashed"),
        _CandidateDatasetPublisher(canonical) as publisher,
    ):
        TrajectoryDatasetWriter(publisher.root, RobotSpec()).write(
            meta_factory(),
            arrays_factory(),
        )
        publisher.publish()
        publisher.prepare_commit()
        raise RuntimeError("record write crashed")

    assert not canonical.exists()
    assert not list(tmp_path.glob(".dataset.candidate-staging-*"))


def test_candidate_staging_orphan_blocks_new_collection_fail_closed(tmp_path) -> None:
    stale = tmp_path / ".dataset.candidate-staging-crashed"
    stale.mkdir()

    with (
        pytest.raises(RuntimeError, match="未完成 candidate staging"),
        _CandidateDatasetPublisher(tmp_path / "dataset"),
    ):
        pytest.fail("stale staging 不应被静默复用")

    assert stale.is_dir()
    assert not (tmp_path / "dataset").exists()


def test_candidate_publish_gates_require_snapshot_and_paired_eligibility() -> None:
    passed_snapshot = SimpleNamespace(passed=True)
    completed_expert = SimpleNamespace(task_completed=True)

    assert _validate_candidate_publish_gates(
        snapshot_round_trip=passed_snapshot,
        snapshot_required=True,
        paired_clean_expert=completed_expert,
        paired_required=True,
        risk_components={"risk": 0.0},
    )
    with pytest.raises(RuntimeError, match="snapshot"):
        _validate_candidate_publish_gates(
            snapshot_round_trip=None,
            snapshot_required=True,
            paired_clean_expert=completed_expert,
            paired_required=True,
            risk_components={"risk": 0.0},
        )
    with pytest.raises(RuntimeError, match="paired clean Expert"):
        _validate_candidate_publish_gates(
            snapshot_round_trip=passed_snapshot,
            snapshot_required=True,
            paired_clean_expert=SimpleNamespace(task_completed=False),
            paired_required=True,
            risk_components={"risk": 0.0},
        )
