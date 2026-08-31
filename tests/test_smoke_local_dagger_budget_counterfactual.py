import copy
from pathlib import Path

import pytest

from robot_vla.cli.replay_local_dagger_budget_counterfactual import (
    COUNTERFACTUAL_PROTOCOL,
    E012_COUNTERFACTUAL_TARGET_SEEDS,
    candidate_command,
)
from robot_vla.cli.replay_local_dagger_failures import (
    COLLECTION_FORMAT,
)
from robot_vla.cli.replay_local_dagger_failures import (
    candidate_command as legacy_candidate_command,
)
from robot_vla.cli.smoke_local_dagger_budget_counterfactual import (
    CONTROL_EXPERT_ACTIONS,
    CONTROL_POLICY_REPLANS,
    CONTROL_SEED,
    CONTROL_TAKEOVER_STEP,
    CONTROL_TOTAL_ACTIONS,
    SMOKE_ORDER,
    summarize_smoke,
    validate_control_reference,
    validate_smoke_outcome,
)


def _traces() -> list[dict[str, object]]:
    return [
        {
            "replan_index": index,
            "control_step": index * 4,
            "sampling_seed": 10_000 + index,
            "executed_steps": 4,
        }
        for index in range(CONTROL_POLICY_REPLANS)
    ]


def _formal_control(tmp_path: Path) -> dict[str, object]:
    traces = _traces()
    return {
        "format": COLLECTION_FORMAT,
        "source_revision": "source-tree-sha256:formal",
        "base_dataset": str((tmp_path / "d0").resolve()),
        "checkpoint": {
            "path": str((tmp_path / "checkpoint.pt").resolve()),
            "sha256": "a" * 64,
        },
        "config": {
            "environment_seed": CONTROL_SEED,
            "boundary_type": "grasp_lift",
            "sampling_seed_base": 52_012,
            "episode_sampling_seed": 10_000,
            "num_flow_steps": 10,
            "recency_decay": 0.5,
            "max_anomaly_replans": 3,
            "qwen_context_layer": 12,
            "snapshot_round_trip_required": True,
            "paired_clean_expert_required": True,
        },
        "status": "accepted",
        "result": {
            "trajectory": {
                "num_steps": CONTROL_TOTAL_ACTIONS,
                "local_dagger": {
                    "rollin_seed": CONTROL_SEED,
                    "boundary_type": "grasp_lift",
                    "boundary_detection_step": CONTROL_TAKEOVER_STEP,
                    "expert_takeover_step": CONTROL_TAKEOVER_STEP,
                    "training_window_start": CONTROL_TAKEOVER_STEP,
                    "training_window_end": CONTROL_TAKEOVER_STEP + 64,
                    "expert_recovery_success": True,
                },
                "outcome_evidence": {"task_completed": True},
            },
            "boundary": {"control_step": CONTROL_TAKEOVER_STEP},
            "policy_replans": CONTROL_POLICY_REPLANS,
            "policy_replan_traces": traces,
            "policy_sampling_seeds": [10_000 + index for index in range(39)],
            "snapshot_round_trip": {"passed": True, "next_state_max_abs_error": 0.0},
        },
        "paired_clean_expert": {"task_completed": True, "num_steps": 189},
        "risk_components": {"score": 0.5},
        "eligible_for_risk_selection": True,
        "audit": {
            "trajectory_contract": "passed",
            "full_dataset_audit": "pending D0 union",
        },
    }


def _control_target(tmp_path: Path) -> tuple[dict[str, object], dict[str, object]]:
    formal = _formal_control(tmp_path)
    reference = validate_control_reference(formal)
    target = {
        "environment_seed": CONTROL_SEED,
        "boundary_type": "grasp_lift",
        "model_cache": str((tmp_path / "model-cache").resolve()),
        "replay_source_revision": "source-tree-sha256:smoke",
        "original_record": formal,
        "original_record_path": str((tmp_path / "formal-record.json").resolve()),
        "original_record_sha256": "b" * 64,
        "reference_prefix": reference["prefix"],
        "control_reference": reference,
    }
    return target, formal


def _counterfactual_control(
    formal: dict[str, object],
) -> dict[str, object]:
    record = copy.deepcopy(formal)
    usage = {
        "policy_actions": CONTROL_TAKEOVER_STEP,
        "expert_actions": CONTROL_EXPERT_ACTIONS,
        "total_actions": CONTROL_TOTAL_ACTIONS,
    }
    record["action_budget_usage"] = usage
    return record


def _row(classification: str, *, artifact_passed: bool = True) -> dict[str, object]:
    return {
        "classification": classification,
        "prefix_alignment": {"aligned": classification != "prefix_mismatch"},
        "artifact_audit": {"passed": artifact_passed},
    }


def test_fixed_smoke_order_does_not_change_formal_16_target_contract() -> None:
    assert E012_COUNTERFACTUAL_TARGET_SEEDS == (
        30_103,
        30_105,
        30_106,
        30_129,
        30_137,
        30_143,
        30_145,
        30_156,
        30_162,
        30_165,
        30_171,
        30_181,
        30_187,
        30_193,
        30_195,
        30_196,
    )
    assert SMOKE_ORDER == (
        ("accepted_control", 30_111),
        ("timeout_early", 30_193),
        ("timeout_late", 30_171),
    )
    assert CONTROL_SEED not in E012_COUNTERFACTUAL_TARGET_SEEDS


def test_control_reference_requires_exact_timing_audit_paired_snapshot_and_int_seeds(
    tmp_path: Path,
) -> None:
    record = _formal_control(tmp_path)

    reference = validate_control_reference(record)

    assert reference["timing"] == {
        "expert_takeover_step": 154,
        "policy_actions": 154,
        "expert_actions": 140,
        "total_actions": 294,
        "policy_replans": 39,
    }
    assert reference["audit"]["trajectory_contract"] == "passed"
    assert reference["paired_clean_expert"]["task_completed"] is True
    assert reference["snapshot_round_trip"]["passed"] is True

    drifted = copy.deepcopy(record)
    drifted["result"]["policy_sampling_seeds"][0] = "10000"
    drifted["result"]["policy_replan_traces"][0]["sampling_seed"] = "10000"
    with pytest.raises(ValueError, match="Python int"):
        validate_control_reference(drifted)


def test_each_smoke_command_only_adds_fixed_segmented_protocol(tmp_path: Path) -> None:
    control, _ = _control_target(tmp_path)
    timeout = copy.deepcopy(control)
    timeout["environment_seed"] = 30_193
    timeout["original_record"]["config"]["environment_seed"] = 30_193
    for target in (control, timeout):
        candidate = tmp_path / f"candidate-{target['environment_seed']}"
        legacy = legacy_candidate_command(
            target,
            candidate_dir=candidate,
            python_executable="python-test",
        )
        smoke = candidate_command(
            target,
            candidate_dir=candidate,
            python_executable="python-test",
        )
        assert smoke[:-2] == legacy
        assert smoke[-2:] == [
            "--action-budget-protocol",
            COUNTERFACTUAL_PROTOCOL,
        ]


def test_control_outcome_requires_accepted_exact_evidence_and_full_artifact_audit(
    tmp_path: Path,
) -> None:
    target, formal = _control_target(tmp_path)
    record = _counterfactual_control(formal)

    validation = validate_smoke_outcome(
        "accepted_control",
        target,
        record,
        _row("recovered_full_eligible"),
    )

    assert validation["passed"] is True
    assert validation["violations"] == []
    assert validation["successful_npz_may_enter_d1"] is False

    for mutation, expected in (
        ("usage", "154/140/294"),
        ("snapshot", "snapshot_round_trip differs"),
        ("artifact", "artifact audit"),
    ):
        changed_record = copy.deepcopy(record)
        changed_row = _row("recovered_full_eligible")
        if mutation == "usage":
            changed_record["action_budget_usage"]["total_actions"] = 295
        elif mutation == "snapshot":
            changed_record["result"]["snapshot_round_trip"][
                "next_state_max_abs_error"
            ] = 0.1
        else:
            changed_row["artifact_audit"] = {"passed": False}
        changed = validate_smoke_outcome(
            "accepted_control",
            target,
            changed_record,
            changed_row,
        )
        assert changed["passed"] is False
        assert any(expected in value for value in changed["violations"])


@pytest.mark.parametrize(
    ("classification", "passed"),
    [
        ("recovered_full_eligible", True),
        ("expert_recovery_budget_exhausted", True),
        ("other_behavioral_rejection", True),
        ("engineering_error", False),
        ("prefix_mismatch", False),
    ],
)
def test_timeout_smoke_allows_behavior_but_not_engineering_or_prefix_mismatch(
    tmp_path: Path,
    classification: str,
    passed: bool,
) -> None:
    target = {
        "environment_seed": 30_193,
    }
    validation = validate_smoke_outcome(
        "timeout_early",
        target,
        {},
        _row(classification),
    )
    assert validation["passed"] is passed


def test_smoke_summary_requires_fixed_order_and_is_additive() -> None:
    rows = []
    for role, seed in SMOKE_ORDER:
        validation = {
            "role": role,
            "environment_seed": seed,
            "passed": True,
        }
        rows.append(
            {
                "smoke_role": role,
                "environment_seed": seed,
                "classification": "recovered_full_eligible",
                "smoke_validation": validation,
            }
        )

    summary = summarize_smoke(rows)

    assert summary["passed"] is True
    assert summary["observed_order"] == [role for role, _ in SMOKE_ORDER]
    assert summary["classification_additivity"] == {
        "classified_candidates": 3,
        "completed_candidates": 3,
        "holds": True,
    }
    assert summary["successful_npz_may_enter_d1"] is False

    reordered = summarize_smoke([rows[1], rows[0], rows[2]])
    assert reordered["scan_complete"] is False
    assert reordered["passed"] is False
