import copy
import hashlib
import json
from pathlib import Path

import pytest

from robot_vla.cli.replay_local_dagger_failures import (
    COLLECTION_FORMAT,
    POOL_FORMAT,
    REPLAY_FORMAT,
    TARGET_FAILURE_REASONS,
    build_replay_experiment,
    candidate_command,
    reconcile_replay_record,
    select_replay_rows,
    summarize_replay,
    validate_original_record,
)
from robot_vla.sim.local_dagger_diagnostics import LOCAL_DAGGER_DIAGNOSTIC_FORMAT


def _config(seed: int = 30_100) -> dict[str, object]:
    return {
        "environment_seed": seed,
        "boundary_type": "grasp_lift",
        "sampling_seed_base": 52_012,
        "episode_sampling_seed": seed + 900_000,
        "num_flow_steps": 10,
        "recency_decay": 0.5,
        "max_anomaly_replans": 3,
        "qwen_context_layer": 12,
        "snapshot_round_trip_required": True,
        "paired_clean_expert_required": True,
    }


def _pool_config() -> dict[str, object]:
    return {
        "inference_strategy": "temporal-ensemble",
        "sampling_seed_base": 52_012,
        "num_flow_steps": 10,
        "recency_decay": 0.5,
        "max_anomaly_replans": 3,
        "qwen_context_layer": 12,
        "snapshot_round_trip_required": True,
        "paired_clean_expert_required": True,
    }


def _checkpoint(path: Path) -> dict[str, object]:
    return {
        "path": str(path.resolve()),
        "sha256": "a" * 64,
        "metadata": {"step": 1_920},
    }


def _record(
    tmp_path: Path,
    *,
    seed: int = 30_100,
    reason: str = TARGET_FAILURE_REASONS[0],
    source_revision: str = "source-tree-sha256:formal",
) -> dict[str, object]:
    return {
        "format": COLLECTION_FORMAT,
        "source_revision": source_revision,
        "base_dataset": str((tmp_path / "d0").resolve()),
        "checkpoint": _checkpoint(tmp_path / "checkpoint.pt"),
        "config": _config(seed),
        "status": "rejected",
        "failure": {"type": "EpisodeRejected", "reason": reason},
    }


def _pool_experiment(tmp_path: Path, seeds: list[int]) -> dict[str, object]:
    checkpoint = _checkpoint(tmp_path / "checkpoint.pt")
    return {
        "format": POOL_FORMAT,
        "source_revision": "source-tree-sha256:formal",
        "checkpoint": {key: checkpoint[key] for key in ("path", "sha256")},
        "base_dataset": {
            "path": str((tmp_path / "d0").resolve()),
            "audit": {"dataset_sha256": "d0"},
        },
        "model_cache": str((tmp_path / "model-cache").resolve()),
        "boundary_type": "grasp_lift",
        "environment_seeds": seeds,
        "config": _pool_config(),
    }


def _row(
    formal_root: Path,
    *,
    seed: int,
    status: str = "rejected",
    reason: str | None = TARGET_FAILURE_REASONS[0],
) -> dict[str, object]:
    failure = None if reason is None else {"type": "EpisodeRejected", "reason": reason}
    value: dict[str, object] = {
        "environment_seed": seed,
        "boundary_type": "grasp_lift",
        "status": status,
        "record": str(
            (formal_root / "candidates" / f"seed-{seed:06d}" / "record.json").resolve()
        ),
        "episode_sampling_seed": seed + 900_000,
        "eligible_for_risk_selection": status == "accepted",
    }
    if failure is not None:
        value["failure"] = failure
    return value


def _target(tmp_path: Path) -> dict[str, object]:
    original = _record(tmp_path)
    return {
        "environment_seed": 30_100,
        "boundary_type": "grasp_lift",
        "model_cache": str((tmp_path / "model-cache").resolve()),
        "replay_source_revision": "source-tree-sha256:replay",
        "original_record": original,
        "original_record_path": str((tmp_path / "original-record.json").resolve()),
        "original_record_sha256": "b" * 64,
    }


def _replay_record(target: dict[str, object]) -> dict[str, object]:
    original = target["original_record"]
    assert isinstance(original, dict)
    replay = copy.deepcopy(original)
    replay["source_revision"] = target["replay_source_revision"]
    replay["failure_diagnostics"] = {
        "format": LOCAL_DAGGER_DIAGNOSTIC_FORMAT,
        "environment_seed": target["environment_seed"],
        "boundary_type": target["boundary_type"],
        "failure_reason": replay["failure"]["reason"],
    }
    return replay


def test_select_replay_rows_filters_two_reasons_and_sorts() -> None:
    formal = Path("/formal")
    rows = [
        _row(formal, seed=30_103, reason=TARGET_FAILURE_REASONS[1]),
        _row(formal, seed=30_102, status="accepted", reason=None),
        _row(formal, seed=30_101, reason="MPlib 无可信 screw path"),
        _row(formal, seed=30_100, reason=TARGET_FAILURE_REASONS[0]),
    ]

    selected = select_replay_rows(
        rows,
        expected_seeds=[30_100, 30_101, 30_102, 30_103],
    )

    assert [row["environment_seed"] for row in selected] == [30_100, 30_103]


def test_select_replay_rows_rejects_duplicate_or_incomplete_formal_index() -> None:
    row = _row(Path("/formal"), seed=30_100)
    with pytest.raises(ValueError, match="重复 seed"):
        select_replay_rows([row, row])
    with pytest.raises(ValueError, match="seed 集合不一致"):
        select_replay_rows([row], expected_seeds=[30_100, 30_101])


def test_validate_original_record_aligns_row_record_pool_identity(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    seed = 30_100
    row = _row(formal, seed=seed)
    record = _record(tmp_path, seed=seed)
    experiment = _pool_experiment(tmp_path, [seed])
    record_path = formal / "candidates" / f"seed-{seed:06d}" / "record.json"

    validate_original_record(
        row,
        record,
        formal_experiment=experiment,
        expected_record_path=record_path,
    )

    drifted = copy.deepcopy(record)
    drifted["config"]["recency_decay"] = 0.25
    with pytest.raises(ValueError, match="recency_decay"):
        validate_original_record(
            row,
            drifted,
            formal_experiment=experiment,
            expected_record_path=record_path,
        )


def test_build_replay_experiment_freezes_selected_records(tmp_path: Path) -> None:
    formal = tmp_path / "formal"
    seed = 30_100
    record_path = formal / "candidates" / f"seed-{seed:06d}" / "record.json"
    record_path.parent.mkdir(parents=True)
    record = _record(tmp_path, seed=seed)
    record_path.write_text(json.dumps(record), encoding="utf-8")
    formal.mkdir(exist_ok=True)
    (formal / "experiment.json").write_text(
        json.dumps(_pool_experiment(tmp_path, [seed])),
        encoding="utf-8",
    )
    row = _row(formal, seed=seed)
    (formal / "collection_candidates.jsonl").write_text(
        json.dumps(row) + "\n",
        encoding="utf-8",
    )

    experiment, targets = build_replay_experiment(
        formal,
        replay_source_revision="source-tree-sha256:replay",
    )

    expected_sha256 = hashlib.sha256(record_path.read_bytes()).hexdigest()
    assert experiment["format"] == REPLAY_FORMAT
    assert experiment["environment_seeds"] == [seed]
    assert experiment["selected_count"] == 1
    assert experiment["selected_candidates"][0]["record_sha256"] == expected_sha256
    assert experiment["execution"]["trajectory_usage"].endswith("training data")
    assert targets[0]["original_record"] == record


def test_candidate_command_uses_only_original_paths_and_config(tmp_path: Path) -> None:
    target = _target(tmp_path)
    command = candidate_command(
        target,
        candidate_dir=tmp_path / "candidate",
        python_executable="python-test",
    )

    assert command[:3] == [
        "python-test",
        "-m",
        "robot_vla.cli.collect_local_dagger",
    ]
    assert command[command.index("--seed") + 1] == "30100"
    assert command[command.index("--sampling-seed") + 1] == "52012"
    assert command[command.index("--checkpoint") + 1] == str(
        (tmp_path / "checkpoint.pt").resolve()
    )
    assert "--require-paired-clean-expert" in command
    assert "--skip-snapshot-round-trip" not in command
    assert "--action-budget-protocol" not in command


def test_reconcile_exact_failure_includes_diagnostics(tmp_path: Path) -> None:
    target = _target(tmp_path)
    replay = _replay_record(target)

    row = reconcile_replay_record(
        target,
        replay,
        replay_record_path=tmp_path / "replay-record.json",
        subprocess_returncode=1,
    )

    assert row["reconciled"] is True
    assert row["reconciliation"]["classification"] == "matched"
    assert row["reconciliation"]["identity_contract_matches"] is True
    assert row["failure_diagnostics"]["format"] == LOCAL_DAGGER_DIAGNOSTIC_FORMAT


def test_reconcile_reason_mismatch_is_recorded_but_not_engineering_error(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    replay = _replay_record(target)
    replay["failure"]["reason"] = TARGET_FAILURE_REASONS[1]
    replay["failure_diagnostics"]["failure_reason"] = TARGET_FAILURE_REASONS[1]

    row = reconcile_replay_record(
        target,
        replay,
        replay_record_path=tmp_path / "replay-record.json",
        subprocess_returncode=1,
    )

    assert row["reconciled"] is False
    assert row["reconciliation"]["classification"] == "outcome_mismatch"
    assert row["reconciliation"]["reason_matches"] is False


@pytest.mark.parametrize(
    ("mutation", "returncode"),
    [
        ("status_error", 1),
        ("config_drift", 1),
        ("checkpoint_drift", 1),
        ("missing_diagnostics", 1),
        ("subprocess_conflict", 0),
    ],
)
def test_reconcile_engineering_contract_errors_fail_closed(
    tmp_path: Path,
    mutation: str,
    returncode: int,
) -> None:
    target = _target(tmp_path)
    replay = _replay_record(target)
    if mutation == "status_error":
        replay["status"] = "error"
    elif mutation == "config_drift":
        replay["config"]["num_flow_steps"] = 9
    elif mutation == "checkpoint_drift":
        replay["checkpoint"]["sha256"] = "0" * 64
    elif mutation == "missing_diagnostics":
        replay.pop("failure_diagnostics")

    row = reconcile_replay_record(
        target,
        replay,
        replay_record_path=tmp_path / "replay-record.json",
        subprocess_returncode=returncode,
    )

    assert row["reconciled"] is False
    assert row["reconciliation"]["classification"] == "engineering_error"


def test_summary_separates_scan_completion_from_reconciliation(tmp_path: Path) -> None:
    target = _target(tmp_path)
    matched = reconcile_replay_record(
        target,
        _replay_record(target),
        replay_record_path=tmp_path / "matched.json",
        subprocess_returncode=1,
    )
    second_target = copy.deepcopy(target)
    second_target["environment_seed"] = 30_101
    second_target["original_record"]["config"]["environment_seed"] = 30_101
    mismatch_record = _replay_record(second_target)
    mismatch_record["status"] = "accepted"
    mismatch_record.pop("failure")
    mismatch_record.pop("failure_diagnostics")
    mismatched = reconcile_replay_record(
        second_target,
        mismatch_record,
        replay_record_path=tmp_path / "mismatched.json",
        subprocess_returncode=0,
    )

    summary = summarize_replay([matched, mismatched], expected_candidates=2)

    assert summary["scan_complete"] is True
    assert summary["all_reconciled"] is False
    assert summary["blocked"] is False
    assert summary["mismatched_seeds"] == [30_101]
