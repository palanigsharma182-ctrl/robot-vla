import copy
import json
from pathlib import Path

import pytest

import robot_vla.cli.replay_local_dagger_budget_counterfactual as counterfactual_module
from robot_vla.cli.replay_local_dagger_budget_counterfactual import (
    CLASSIFICATIONS,
    COUNTERFACTUAL_FORMAT,
    COUNTERFACTUAL_PROTOCOL,
    E012_COUNTERFACTUAL_TARGET_SEEDS,
    EXPECTED_COUNTERFACTUAL_COUNT,
    EXPECTED_DIAGNOSTIC_REPLAY_COUNT,
    TRAJECTORY_USAGE,
    CounterfactualMaterializationError,
    _finalize_candidate,
    _load_finalized_candidate,
    _materialization_error_row,
    _prepare_output,
    _validate_diagnostic_summary,
    candidate_command,
    compare_prefix,
    reconcile_counterfactual_record,
    select_counterfactual_rows,
    summarize_counterfactual,
)
from robot_vla.cli.replay_local_dagger_failures import (
    COLLECTION_FORMAT,
)
from robot_vla.cli.replay_local_dagger_failures import (
    candidate_command as legacy_candidate_command,
)
from robot_vla.local_dagger_protocol import (
    EXPERT_ACTION_BUDGET_EXHAUSTED_REASON,
    LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD,
    LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD,
    resolve_local_dagger_action_budget,
)
from robot_vla.sim.local_dagger_diagnostics import (
    EPISODE_TIME_LIMIT_REASON,
    LOCAL_DAGGER_DIAGNOSTIC_FORMAT,
    POLICY_BEFORE_BOUNDARY_REASON,
)


def _legacy_config(seed: int = 30_103) -> dict[str, object]:
    return {
        "environment_seed": seed,
        "boundary_type": "grasp_lift",
        "sampling_seed_base": 52_012,
        "episode_sampling_seed": 7_191_229_130_600_072_479 + seed - 30_103,
        "num_flow_steps": 10,
        "recency_decay": 0.5,
        "max_anomaly_replans": 3,
        "qwen_context_layer": 12,
        "snapshot_round_trip_required": True,
        "paired_clean_expert_required": True,
    }


def _checkpoint(tmp_path: Path) -> dict[str, object]:
    return {
        "path": str((tmp_path / "checkpoint.pt").resolve()),
        "sha256": "a" * 64,
        "metadata": {"step": 1_920},
    }


def _original_record(tmp_path: Path, *, seed: int = 30_103) -> dict[str, object]:
    return {
        "format": COLLECTION_FORMAT,
        "source_revision": "source-tree-sha256:formal",
        "base_dataset": str((tmp_path / "d0").resolve()),
        "checkpoint": _checkpoint(tmp_path),
        "config": _legacy_config(seed),
        "status": "rejected",
        "failure": {
            "type": "EpisodeRejected",
            "reason": EPISODE_TIME_LIMIT_REASON,
        },
    }


def _traces() -> list[dict[str, object]]:
    return [
        {
            "replan_index": 0,
            "control_step": 0,
            "sampling_seed": 101,
            "executed_steps": 4,
            "completed_skill_count_before": 0,
            "completed_skill_count_after": 1,
            "temporal_buffer_size": 1,
            "temporal_max_proposal_spread": 0.0,
            "replan_required": False,
        },
        {
            "replan_index": 1,
            "control_step": 4,
            "sampling_seed": 102,
            "executed_steps": 4,
            "completed_skill_count_before": 1,
            "completed_skill_count_after": 2,
            "temporal_buffer_size": 2,
            "temporal_max_proposal_spread": 0.25,
            "replan_required": False,
        },
    ]


def _prefix(*, takeover: int = 8) -> dict[str, object]:
    traces = _traces()
    return {
        "expert_takeover_step": takeover,
        "boundary_detection_step": takeover,
        "grasp_completion_step": takeover,
        "policy_replan_count": len(traces),
        "policy_replan_traces": traces,
        "policy_sampling_seeds": [101, 102],
    }


def _target(tmp_path: Path, *, seed: int = 30_103) -> dict[str, object]:
    original = _original_record(tmp_path, seed=seed)
    return {
        "environment_seed": seed,
        "boundary_type": "grasp_lift",
        "model_cache": str((tmp_path / "model-cache").resolve()),
        "replay_source_revision": "source-tree-sha256:counterfactual",
        "original_record": original,
        "original_record_path": str((tmp_path / "formal-record.json").resolve()),
        "original_record_sha256": "b" * 64,
        "diagnostic_record_path": str(
            (tmp_path / "diagnostic-record.json").resolve()
        ),
        "diagnostic_record_sha256": "c" * 64,
        "reference_prefix": _prefix(),
    }


def _usage(*, takeover: int = 8, expert: int = 120) -> dict[str, int]:
    return {
        "policy_actions": takeover,
        "expert_actions": expert,
        "total_actions": takeover + expert,
    }


def _diagnostics(
    *,
    seed: int = 30_103,
    takeover: int = 8,
    expert: int = 120,
    reason: str = "Local DAgger Expert 未完成完整 Pick-and-Place",
    completed: bool = False,
    budget_phase: str | None = None,
) -> dict[str, object]:
    traces = _traces()
    usage = _usage(takeover=takeover, expert=expert)
    planned = resolve_local_dagger_action_budget(
        COUNTERFACTUAL_PROTOCOL
    ).planned_metadata()
    return {
        "format": LOCAL_DAGGER_DIAGNOSTIC_FORMAT,
        "environment_seed": seed,
        "boundary_type": "grasp_lift",
        "failure_reason": reason,
        "action_count": usage["total_actions"],
        "failure_control_step": usage["total_actions"],
        "boundary_reached": True,
        "boundary_detection_step": takeover,
        "expert_takeover_step": takeover,
        "skill_completion_steps": {
            "reach": 4,
            "grasp": takeover,
            "lift": 20,
            "transport": 80,
            "place": 120 if completed else None,
        },
        "max_completed_skill_count": 5 if completed else 4,
        "final_progress": {
            "completed_skill_count": 5 if completed else 4,
            "task_completed": completed,
        },
        "final_transition": {
            "action_step": usage["total_actions"],
            "terminated": False,
            "truncated": False,
        },
        "policy_replan_count": len(traces),
        "policy_replan_traces": traces,
        "budget_exhaustion_phase": budget_phase,
        LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: planned,
        LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
    }


def _base_counterfactual_record(
    target: dict[str, object],
    *,
    status: str,
    usage: dict[str, int],
) -> dict[str, object]:
    original = target["original_record"]
    assert isinstance(original, dict)
    planned = resolve_local_dagger_action_budget(
        COUNTERFACTUAL_PROTOCOL
    ).planned_metadata()
    config = copy.deepcopy(original["config"])
    config[LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD] = planned
    return {
        "format": COLLECTION_FORMAT,
        "source_revision": target["replay_source_revision"],
        "base_dataset": original["base_dataset"],
        "checkpoint": copy.deepcopy(original["checkpoint"]),
        "config": config,
        "status": status,
        LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
    }


def _accepted_record(target: dict[str, object]) -> dict[str, object]:
    seed = int(target["environment_seed"])
    usage = _usage()
    planned = resolve_local_dagger_action_budget(
        COUNTERFACTUAL_PROTOCOL
    ).planned_metadata()
    record = _base_counterfactual_record(target, status="accepted", usage=usage)
    record.update(
        {
            "result": {
                "trajectory": {
                    "file": "trajectories/control.npz",
                    "num_steps": usage["total_actions"],
                    "local_dagger": {
                        "rollin_seed": seed,
                        "boundary_type": "grasp_lift",
                        "boundary_detection_step": 8,
                        "expert_takeover_step": 8,
                        "expert_recovery_success": True,
                    },
                    "outcome_evidence": {"task_completed": True},
                    "randomization": {
                        LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: planned,
                        LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
                    },
                },
                "policy_replans": 2,
                "policy_replan_traces": _traces(),
                "policy_sampling_seeds": [101, 102],
                "snapshot_round_trip": {"passed": True},
                LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
            },
            "paired_clean_expert": {"task_completed": True},
            "risk_components": {"score": 0.2},
            "eligible_for_risk_selection": True,
            "audit": {"trajectory_contract": "passed"},
        }
    )
    return record


def _rejected_record(
    target: dict[str, object],
    *,
    reason: str = "Local DAgger Expert 未完成完整 Pick-and-Place",
    takeover: int = 8,
    expert: int = 120,
    completed: bool = False,
    budget_phase: str | None = None,
) -> dict[str, object]:
    usage = _usage(takeover=takeover, expert=expert)
    record = _base_counterfactual_record(target, status="rejected", usage=usage)
    record.update(
        {
            "failure": {"type": "EpisodeRejected", "reason": reason},
            "failure_diagnostics": _diagnostics(
                seed=int(target["environment_seed"]),
                takeover=takeover,
                expert=expert,
                reason=reason,
                completed=completed,
                budget_phase=budget_phase,
            ),
        }
    )
    return record


def _reconcile(
    tmp_path: Path,
    target: dict[str, object],
    record: dict[str, object],
    *,
    returncode: int,
) -> dict[str, object]:
    path = tmp_path / f"record-{record['status']}-{len(list(tmp_path.glob('record-*')))}.json"
    if record["status"] == "accepted":
        import numpy as np

        trajectory = record["result"]["trajectory"]
        dataset = path.parent / "dataset"
        npz_path = dataset / trajectory["file"]
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        steps = int(trajectory["num_steps"])
        np.savez_compressed(
            npz_path,
            action=np.zeros((steps, 8), dtype=np.float32),
            action_source=np.zeros(steps, dtype=np.uint8),
            expert_supervision_mask=np.zeros(steps, dtype=np.bool_),
            success=np.zeros(steps, dtype=np.bool_),
            skill_id=np.zeros(steps, dtype=np.int16),
        )
        (dataset / "manifest.jsonl").write_text(
            json.dumps(trajectory) + "\n",
            encoding="utf-8",
        )
    path.write_text(json.dumps(record), encoding="utf-8")
    command = candidate_command(
        target,
        candidate_dir=tmp_path / "candidate",
        python_executable="python-test",
    )
    return reconcile_counterfactual_record(
        target,
        record,
        record_path=path,
        subprocess_returncode=returncode,
        subprocess_command=command,
        expected_command=command,
    )


def _formal_row(
    formal: Path,
    *,
    seed: int,
    status: str,
    reason: str | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "environment_seed": seed,
        "boundary_type": "grasp_lift",
        "status": status,
        "record": str(
            (formal / "candidates" / f"seed-{seed:06d}" / "record.json").resolve()
        ),
        "episode_sampling_seed": seed + 900_000,
    }
    if reason is not None:
        row["failure"] = {"type": "EpisodeRejected", "reason": reason}
    return row


def test_selects_exact_formal_16_legacy_timeouts() -> None:
    formal = Path("/formal")
    rows: list[dict[str, object]] = []
    policy_seeds = iter(
        seed
        for seed in range(30_100, 30_200)
        if seed not in E012_COUNTERFACTUAL_TARGET_SEEDS
    )
    policy_seed_set = {next(policy_seeds) for _ in range(71)}
    for index in range(100):
        seed = 30_100 + index
        if seed in E012_COUNTERFACTUAL_TARGET_SEEDS:
            status, reason = "rejected", EPISODE_TIME_LIMIT_REASON
        elif seed in policy_seed_set:
            status, reason = "rejected", POLICY_BEFORE_BOUNDARY_REASON
        else:
            status, reason = "accepted", None
        rows.append(
            _formal_row(formal, seed=seed, status=status, reason=reason)
        )

    selected = select_counterfactual_rows(
        list(reversed(rows)),
        expected_seeds=[30_100 + index for index in range(100)],
    )

    assert [row["environment_seed"] for row in selected] == list(
        E012_COUNTERFACTUAL_TARGET_SEEDS
    )
    drifted = copy.deepcopy(rows)
    drifted[3]["failure"]["reason"] = POLICY_BEFORE_BOUNDARY_REASON
    with pytest.raises(ValueError, match="精确的 16"):
        select_counterfactual_rows(
            drifted,
            expected_seeds=[30_100 + index for index in range(100)],
        )


def test_diagnostic_precondition_requires_exact_87_of_87_matched() -> None:
    seeds = list(range(30_100, 30_100 + EXPECTED_DIAGNOSTIC_REPLAY_COUNT))
    summary = {
        "format": "robot-vla-local-dagger-failure-replay/v1",
        "scan_complete": True,
        "all_reconciled": True,
        "blocked": False,
        "expected_candidates": 87,
        "completed_candidates": 87,
        "reconciliation_counts": {"matched": 87},
        "matched_seeds": seeds,
        "mismatched_seeds": [],
        "engineering_error_seeds": [],
    }

    assert _validate_diagnostic_summary(summary) == seeds

    for key, value in (
        ("scan_complete", False),
        ("all_reconciled", False),
        ("blocked", True),
        ("completed_candidates", 86),
    ):
        drifted = {**summary, key: value}
        with pytest.raises(RuntimeError, match="前置未满足"):
            _validate_diagnostic_summary(drifted)


def test_command_only_adds_fixed_segmented_protocol(tmp_path: Path) -> None:
    target = _target(tmp_path)
    candidate = tmp_path / "candidate"
    legacy = legacy_candidate_command(
        target,
        candidate_dir=candidate,
        python_executable="python-test",
    )
    amended = candidate_command(
        target,
        candidate_dir=candidate,
        python_executable="python-test",
    )

    assert amended[:-2] == legacy
    assert amended[-2:] == [
        "--action-budget-protocol",
        "segmented-300-180-480",
    ]
    assert amended.count("--action-budget-protocol") == 1


@pytest.mark.parametrize(
    ("record_factory", "returncode", "expected"),
    [
        (
            lambda target: _accepted_record(target),
            0,
            "recovered_full_eligible",
        ),
        (
            lambda target: _rejected_record(
                target,
                reason="Boundary snapshot round-trip 未通过：test",
                completed=True,
            ),
            1,
            "expert_completed_but_snapshot_or_paired_gate_failed",
        ),
        (
            lambda target: _rejected_record(
                target,
                reason=EXPERT_ACTION_BUDGET_EXHAUSTED_REASON,
                expert=180,
                budget_phase="expert",
            ),
            1,
            "expert_recovery_budget_exhausted",
        ),
        (
            lambda target: _rejected_record(target),
            1,
            "other_behavioral_rejection",
        ),
    ],
)
def test_behavioral_classifications_are_mutually_exclusive(
    tmp_path: Path,
    record_factory,
    returncode: int,
    expected: str,
) -> None:
    target = _target(tmp_path)
    row = _reconcile(
        tmp_path,
        target,
        record_factory(target),
        returncode=returncode,
    )

    assert row["classification"] == expected
    assert row["engineering_errors"] == []
    assert row["prefix_alignment"]["aligned"] is True
    assert row["prefix_alignment"]["bitwise_action_identity_proven"] is False
    assert row["successful_npz_may_enter_d1"] is False


def test_prefix_mismatch_overrides_behavior_and_is_not_bitwise_claim(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    record = _rejected_record(target, takeover=9)

    row = _reconcile(tmp_path, target, record, returncode=1)

    assert row["classification"] == "prefix_mismatch"
    assert row["engineering_errors"] == []
    assert row["prefix_alignment"]["field_matches"] == {
        "expert_takeover_step": False,
        "boundary_detection_step": False,
        "grasp_completion_step": False,
        "policy_replan_count": True,
        "policy_replan_traces": True,
        "policy_sampling_seeds": True,
    }
    assert "does not prove bitwise" in row["prefix_alignment"]["non_proof"]

    exact = compare_prefix(_prefix(), _prefix())
    assert exact["aligned"] is True
    assert exact["bitwise_action_identity_proven"] is False


def test_environment_hard_deadline_remains_other_behavioral_rejection(
    tmp_path: Path,
) -> None:
    """300 + 180 == 480 时 truncation 保留环境 deadline 语义。"""

    target = _target(tmp_path)
    target["reference_prefix"] = _prefix(takeover=300)
    record = _rejected_record(
        target,
        reason=EPISODE_TIME_LIMIT_REASON,
        takeover=300,
        expert=180,
    )
    record["failure_diagnostics"]["final_transition"] = {
        "action_step": 480,
        "terminated": False,
        "truncated": True,
    }

    row = _reconcile(tmp_path, target, record, returncode=1)

    assert row["classification"] == "other_behavioral_rejection"
    assert row["action_budget_usage"] == {
        "policy_actions": 300,
        "expert_actions": 180,
        "total_actions": 480,
    }
    assert row["engineering_errors"] == []


def test_step_480_truncation_precedes_completed_progress_gate_classification(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    target["reference_prefix"] = _prefix(takeover=300)
    record = _rejected_record(
        target,
        reason=EPISODE_TIME_LIMIT_REASON,
        takeover=300,
        expert=180,
        completed=True,
    )
    record["failure_diagnostics"]["final_transition"] = {
        "action_step": 480,
        "terminated": False,
        "truncated": True,
    }

    row = _reconcile(tmp_path, target, record, returncode=1)

    assert row["classification"] == "other_behavioral_rejection"
    assert row["engineering_errors"] == []


def test_step_480_truncation_with_non_time_limit_reason_fails_closed(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    target["reference_prefix"] = _prefix(takeover=300)
    record = _rejected_record(
        target,
        reason="snapshot round-trip failed",
        takeover=300,
        expert=180,
        completed=True,
    )
    record["failure_diagnostics"]["final_transition"] = {
        "action_step": 480,
        "terminated": False,
        "truncated": True,
    }

    row = _reconcile(tmp_path, target, record, returncode=1)

    assert row["classification"] == "engineering_error"
    assert (
        "hard-deadline reason/usage/truncation evidence is not closed"
        in row["engineering_errors"]
    )


def test_time_limit_and_expert_budget_record_consistency_fail_closed(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    target["reference_prefix"] = _prefix(takeover=299)
    early_time_limit = _rejected_record(
        target,
        reason=EPISODE_TIME_LIMIT_REASON,
        takeover=299,
        expert=180,
    )
    early_time_limit["failure_diagnostics"]["final_transition"]["truncated"] = True
    early = _reconcile(tmp_path, target, early_time_limit, returncode=1)
    assert early["classification"] == "engineering_error"
    assert "time-limit rejection must occur at total action 480" in early[
        "engineering_errors"
    ]

    target = _target(tmp_path, seed=30_104)
    truncated_budget = _rejected_record(
        target,
        reason=EXPERT_ACTION_BUDGET_EXHAUSTED_REASON,
        expert=180,
        budget_phase="expert",
    )
    truncated_budget["failure_diagnostics"]["final_transition"]["truncated"] = True
    truncated = _reconcile(tmp_path, target, truncated_budget, returncode=1)
    assert truncated["classification"] == "engineering_error"
    assert "expert-budget rejection must be nontruncated" in truncated[
        "engineering_errors"
    ]

    target = _target(tmp_path, seed=30_105)
    mismatched_count = _rejected_record(target)
    mismatched_count["failure_diagnostics"]["action_count"] -= 1
    mismatch = _reconcile(tmp_path, target, mismatched_count, returncode=1)
    assert mismatch["classification"] == "engineering_error"
    assert "rejected diagnostics.action_count/usage total mismatch" in mismatch[
        "engineering_errors"
    ]


def test_accepted_success_must_strictly_precede_environment_hard_deadline(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    target["reference_prefix"] = _prefix(takeover=300)
    record = _accepted_record(target)
    usage = {
        "policy_actions": 300,
        "expert_actions": 180,
        "total_actions": 480,
    }
    record[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = usage
    record["result"][LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD] = usage
    record["result"]["trajectory"]["num_steps"] = 480
    record["result"]["trajectory"]["randomization"][
        LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD
    ] = usage
    provenance = record["result"]["trajectory"]["local_dagger"]
    provenance["expert_takeover_step"] = 300
    provenance["boundary_detection_step"] = 300

    row = _reconcile(tmp_path, target, record, returncode=0)

    assert row["classification"] == "engineering_error"
    assert "accepted total action usage must be strictly below 480" in row[
        "engineering_errors"
    ]


def test_accepted_requires_snapshot_and_materialized_manifest_npz(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    record = _accepted_record(target)
    record["result"]["snapshot_round_trip"]["passed"] = False
    row = _reconcile(tmp_path, target, record, returncode=0)
    assert row["classification"] == "engineering_error"
    assert "accepted snapshot round-trip did not pass" in row[
        "engineering_errors"
    ]

    target = _target(tmp_path, seed=30_104)
    record = _accepted_record(target)
    path = tmp_path / "missing-artifact" / "record.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(record), encoding="utf-8")
    command = candidate_command(
        target,
        candidate_dir=path.parent,
        python_executable="python-test",
    )
    missing = reconcile_counterfactual_record(
        target,
        record,
        record_path=path,
        subprocess_returncode=0,
        subprocess_command=command,
        expected_command=command,
    )
    assert missing["classification"] == "engineering_error"
    assert any("manifest" in error for error in missing["engineering_errors"])


def test_resume_requires_exact_frozen_experiment_identity(tmp_path: Path) -> None:
    output = tmp_path / "counterfactual"
    experiment = {
        "format": COUNTERFACTUAL_FORMAT,
        "purpose": "exploratory counterfactual",
        "trajectory_usage": TRAJECTORY_USAGE,
        "selected_count": EXPECTED_COUNTERFACTUAL_COUNT,
    }

    _prepare_output(output, experiment, resume=False)
    _prepare_output(output, experiment, resume=True)

    with pytest.raises(ValueError, match="identity 漂移"):
        _prepare_output(
            output,
            {**experiment, "selected_count": 15},
            resume=True,
        )
    with pytest.raises(FileExistsError, match="非空"):
        _prepare_output(output, experiment, resume=False)


def test_candidate_receipt_makes_record_subprocess_and_result_immutable(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "record.json").write_text("{}\n", encoding="utf-8")
    (candidate / "subprocess.json").write_text("{}\n", encoding="utf-8")
    row = {
        "format": "test-result/v1",
        "environment_seed": target["environment_seed"],
        "classification": "other_behavioral_rejection",
    }

    finalized = _finalize_candidate(
        candidate,
        target,
        row,
        result_filename="counterfactual.json",
    )
    loaded = _load_finalized_candidate(
        candidate,
        target,
        result_filename="counterfactual.json",
    )
    assert finalized == loaded == row

    (candidate / "record.json").write_text('{"drift": true}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="record receipt SHA 漂移"):
        _load_finalized_candidate(
            candidate,
            target,
            result_filename="counterfactual.json",
        )


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    (
        ("runner_error_sha256", "__delete__", "receipt schema 漂移"),
        ("immutable", "__delete__", "receipt schema 漂移"),
        ("immutable", False, "receipt immutable 标记漂移"),
    ),
)
def test_candidate_receipt_schema_and_immutable_flag_fail_closed(
    tmp_path: Path,
    field: str,
    replacement: object,
    message: str,
) -> None:
    target = _target(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "record.json").write_text("{}\n", encoding="utf-8")
    (candidate / "subprocess.json").write_text("{}\n", encoding="utf-8")
    row = {
        "format": "test-result/v1",
        "environment_seed": target["environment_seed"],
        "classification": "other_behavioral_rejection",
    }
    _finalize_candidate(
        candidate,
        target,
        row,
        result_filename="counterfactual.json",
    )
    receipt_path = candidate / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    if replacement == "__delete__":
        del receipt[field]
    else:
        receipt[field] = replacement
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match=message):
        _load_finalized_candidate(
            candidate,
            target,
            result_filename="counterfactual.json",
        )


def test_accepted_receipt_reaudits_manifest_and_npz_identity(
    tmp_path: Path,
    monkeypatch,
) -> None:
    target = _target(tmp_path)
    candidate = tmp_path / "accepted-candidate"
    candidate.mkdir()
    (candidate / "record.json").write_text(
        json.dumps({"status": "accepted"}) + "\n",
        encoding="utf-8",
    )
    (candidate / "subprocess.json").write_text("{}\n", encoding="utf-8")
    artifact_audit = {
        "passed": True,
        "trajectory_id": "test-trajectory",
        "num_steps": 10,
        "manifest_sha256": "a" * 64,
        "npz_sha256": "b" * 64,
    }
    row = {
        "format": "test-result/v1",
        "environment_seed": target["environment_seed"],
        "status": "accepted",
        "classification": "recovered_full_eligible",
        "artifact_audit": artifact_audit,
    }
    monkeypatch.setattr(
        counterfactual_module,
        "audit_accepted_candidate_artifact",
        lambda record, *, record_path: dict(artifact_audit),
    )

    _finalize_candidate(
        candidate,
        target,
        row,
        result_filename="counterfactual.json",
    )
    assert _load_finalized_candidate(
        candidate,
        target,
        result_filename="counterfactual.json",
    ) == row

    monkeypatch.setattr(
        counterfactual_module,
        "audit_accepted_candidate_artifact",
        lambda record, *, record_path: {
            **artifact_audit,
            "npz_sha256": "c" * 64,
        },
    )
    with pytest.raises(RuntimeError, match="artifact identity 漂移"):
        _load_finalized_candidate(
            candidate,
            target,
            result_filename="counterfactual.json",
        )


def test_missing_record_materialization_has_blocked_summary_and_stable_receipt(
    tmp_path: Path,
) -> None:
    target = _target(tmp_path)
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "subprocess.json").write_text("{}\n", encoding="utf-8")
    (candidate / "runner_error.json").write_text(
        json.dumps({"reason": "subprocess completed without record.json"}) + "\n",
        encoding="utf-8",
    )
    error = CounterfactualMaterializationError(
        int(target["environment_seed"]),
        candidate,
        "subprocess completed without record.json",
    )
    row = _materialization_error_row(target, error)
    _finalize_candidate(
        candidate,
        target,
        row,
        result_filename="counterfactual.json",
        allow_missing_record=True,
    )

    loaded = _load_finalized_candidate(
        candidate,
        target,
        result_filename="counterfactual.json",
        allow_missing_record=True,
    )
    summary = summarize_counterfactual([loaded], expected_candidates=1)
    assert loaded["classification"] == "engineering_error"
    assert summary["blocked"] is True
    assert summary["classification_additivity"]["holds"] is True

    receipt_path = candidate / "receipt.json"
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    del receipt["record_sha256"]
    receipt_path.write_text(json.dumps(receipt) + "\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="receipt schema 漂移"):
        _load_finalized_candidate(
            candidate,
            target,
            result_filename="counterfactual.json",
            allow_missing_record=True,
        )


def test_engineering_error_blocks_and_summary_is_additive(tmp_path: Path) -> None:
    target = _target(tmp_path)
    invalid = _accepted_record(target)
    invalid["config"]["num_flow_steps"] = 9

    engineering = _reconcile(tmp_path, target, invalid, returncode=0)
    assert engineering["classification"] == "engineering_error"
    assert any("config" in error for error in engineering["engineering_errors"])

    rows: list[dict[str, object]] = [engineering]
    for index, classification in enumerate(CLASSIFICATIONS[:-1], start=1):
        rows.append(
            {
                "environment_seed": 30_103 + index,
                "classification": classification,
            }
        )
    summary = summarize_counterfactual(rows, expected_candidates=len(rows))

    assert summary["blocked"] is True
    assert summary["classification_additivity"] == {
        "classified_candidates": len(rows),
        "completed_candidates": len(rows),
        "holds": True,
    }
    assert sum(summary["classification_counts"].values()) == len(rows)
    assert set(summary["classification_counts"]) == set(CLASSIFICATIONS)
    assert summary["successful_npz_may_enter_d1"] is False


def test_summary_rejects_unknown_or_incomplete_classification() -> None:
    with pytest.raises(ValueError, match="未知分类"):
        summarize_counterfactual(
            [{"environment_seed": 1, "classification": "ambiguous"}],
            expected_candidates=1,
        )

    summary = summarize_counterfactual(
        [
            {
                "environment_seed": 1,
                "classification": "recovered_full_eligible",
            }
        ],
        expected_candidates=EXPECTED_COUNTERFACTUAL_COUNT,
    )
    assert summary["scan_complete"] is False
    assert summary["classification_additivity"]["holds"] is True
