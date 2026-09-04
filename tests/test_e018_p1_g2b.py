from __future__ import annotations

import inspect
import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from robot_vla.precision import e018_p1_g2b as g2b
from robot_vla.precision.e018_p1_g2b import (
    E018_P1_G2B_CONFIG_VERSION,
    assert_calibration_prediction_ledger_deployable_only,
    fit_covariance_scale,
    freeze_calibration_prediction_ledger,
    load_e018_p1_g2b_config,
    load_frozen_calibration_prediction_ledger,
    run_e018_p1_g2b_qualification,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/e018_p1_g2b_covariance_calibrated_provider_requalification_development_v2.json"
)
PARENT_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/e018_p1_g2a_front_provider_qualification_development_v1.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_calibration_receipt_fixture(
    root: Path,
    *,
    status: str,
    gate_evaluated: bool,
    gate_passed: bool | None,
    protocol_valid: bool,
) -> dict[str, object]:
    root.mkdir(parents=True, exist_ok=True)
    gate_state = {
        "protocol_gate_evaluated": True,
        "gate_evaluated": gate_evaluated,
        "gate_passed": gate_passed,
        "protocol_valid": protocol_valid,
    }
    json_artifacts = {
        "config_snapshot.json": _config(),
        "source_identity.json": {"identity_sha256": "f" * 64},
        "failed_v1_lineage_binding.json": {"binding_sha256": "1" * 64},
        "parent_g2a_receipt_binding.json": {},
        "manifest_audit.json": {},
        "calibration_inference_audit.json": {},
        "calibration_prediction_freeze.json": {},
        "calibration_cohort_audit.json": {"audit_sha256": "d" * 64},
        "calibration.json": {
            "version": g2b.E018_P1_G2B_RESULT_VERSION,
            "status": status.removeprefix("complete-"),
            "gate": g2b.E018_P1_G2B_CAL_GATE,
            **gate_state,
            "calibration": None,
        },
        "calibration_summary.json": {
            "version": g2b.E018_P1_G2B_RESULT_VERSION,
            "status": status.removeprefix("complete-"),
            "gate": g2b.E018_P1_G2B_CAL_GATE,
            **gate_state,
            "calibration_identity_sha256": None,
        },
    }
    for name, payload in json_artifacts.items():
        path = root / name
        if not path.exists():
            g2b._atomic_json(path, payload)
    for name in (
        "calibration_prediction_ledger.jsonl",
        "calibration_scoring_ledger.jsonl",
    ):
        path = root / name
        if not path.exists():
            g2b._atomic_jsonl(path, [])
    summary = json.loads((root / "calibration_summary.json").read_text(encoding="utf-8"))
    cohort = json.loads(
        (root / "calibration_cohort_audit.json").read_text(encoding="utf-8")
    )
    failed_v1 = json.loads(
        (root / "failed_v1_lineage_binding.json").read_text(encoding="utf-8")
    )
    receipt: dict[str, object] = {
        "version": g2b.E018_P1_G2B_RESULT_VERSION,
        "status": status,
        "gate": g2b.E018_P1_G2B_CAL_GATE,
        **gate_state,
        "config_sha256": g2b.canonical_sha256(_config()),
        "source_identity_sha256": "f" * 64,
        "failed_v1_lineage_binding_sha256": failed_v1["binding_sha256"],
        "parent_g2a_receipt_internal_sha256": "2" * 64,
        "validation_data_identity_sha256": "e" * 64,
        "prediction_ledger_sha256": g2b.file_sha256(
            root / "calibration_prediction_ledger.jsonl"
        ),
        "phase_a_applicability_audit_sha256": "c" * 64,
        "cohort_audit_sha256": cohort["audit_sha256"],
        "prediction_frozen_before_validation_label": True,
        "allowed_label_split": "val",
        "calibration_identity_sha256": summary.get("calibration_identity_sha256"),
        "failure_reasons": summary.get("failure_reasons", []),
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
        "live_memory_read_count": 0,
        "live_memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "arm_actuation_count": 0,
        "manipulation_progression_count": 0,
        "provider_training_count": 0,
        "files": g2b._artifact_hashes(root, list(g2b._CAL_ARTIFACT_NAMES)),
    }
    receipt["receipt_sha256"] = g2b.canonical_sha256(receipt)
    g2b._atomic_json(root / "calibration_receipt.json", receipt)
    return receipt


def test_g2b_config_is_strict_and_binds_frozen_g2a() -> None:
    config = load_e018_p1_g2b_config(
        CONFIG_PATH,
        parent_g2a_config_path=PARENT_CONFIG_PATH,
    )
    assert config["version"] == E018_P1_G2B_CONFIG_VERSION
    assert config["calibration_data"]["pregrasp_skill_id"] == 0
    assert config["calibration_data"]["expected_applicable_frame_count"] == 1135
    assert config["calibration"]["applicability_predicate"] == (
        "phase-a-deployable-skill-id-equals-0/v1"
    )
    assert "gt_object_z" in config["calibration"]["selection_must_not_use"]
    assert config["qualification"]["full_run_requires_decision_agent_exit_go"] is True


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("scope", "test_label_array_read_allowed"), True, "scope"),
        (("calibration", "minimum_support_count"), 29, "算法"),
        (("calibration", "applicability_predicate"), "gt-z", "算法"),
        (("calibration", "selection_must_not_use"), [], "算法"),
        (("calibration_data", "pregrasp_skill_id"), 1, "数据"),
        (("qualification", "same_seed_range"), [75002, 75051], "qualification"),
        (("execution", "memory_write_allowed"), True, "execution"),
    ],
)
def test_g2b_config_rejects_research_or_permission_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    match: str,
) -> None:
    config = _config()
    config[path[0]][path[1]] = value  # type: ignore[index]
    candidate = tmp_path / "g2b.json"
    candidate.write_text(json.dumps(config), encoding="utf-8")
    with pytest.raises(ValueError, match=match):
        load_e018_p1_g2b_config(
            candidate,
            parent_g2a_config_path=PARENT_CONFIG_PATH,
        )


def test_fixed_order_statistic_and_scale_rule() -> None:
    scores = [float(value) for value in range(1, 40)]
    fit = fit_covariance_scale(scores)
    assert fit["passed"] is True
    assert fit["support_count"] == 39
    assert fit["order_statistic_k"] == math.ceil((39 + 1) * 0.95) == 38
    assert fit["quantile_score"] == 38.0
    assert fit["scale_factor"] == pytest.approx(38.0 / 5.991)


def test_calibration_fail_closed_for_low_support_or_nonfinite_score() -> None:
    low_support = fit_covariance_scale([1.0] * 29)
    assert low_support["passed"] is False
    assert "insufficient_support" in low_support["failure_reasons"]

    nonfinite = fit_covariance_scale([1.0] * 30 + [math.inf])
    assert nonfinite["passed"] is False
    assert "nonfinite_calibration_score" in nonfinite["failure_reasons"]
    assert nonfinite["scale_factor"] is None


def test_calibration_prediction_freeze_reloads_only_disk_ledger(tmp_path: Path) -> None:
    config = _config()
    config["calibration_data"]["manifest_sample_count"] = 1
    config["calibration_data"]["expected_skill_counts"] = {
        "0": 1,
        "1": 0,
        "2": 0,
        "3": 0,
        "4": 0,
    }
    config["calibration_data"]["expected_applicable_frame_count"] = 1
    config["calibration_data"]["expected_nonapplicable_frame_count"] = 0
    config["calibration_data"]["expected_trajectories_with_applicable_frames"] = 1
    rows = [
        {
            "trajectory_id": "val-001",
            "timestep": 0,
            "skill_id": 0,
            "calibration_applicable": True,
            "task_spec_taxonomy_sha256": (
                g2b.calibration_protocol.G2B_TASK_TAXONOMY_SHA256
            ),
            "geometry": {"valid": True},
        }
    ]
    marker = freeze_calibration_prediction_ledger(
        tmp_path,
        rows=rows,
        config=config,
    )
    rows[0]["trajectory_id"] = "mutated-in-memory"
    loaded, loaded_marker = load_frozen_calibration_prediction_ledger(
        tmp_path,
        config=config,
        expected_prediction_count=1,
    )
    assert loaded[0]["trajectory_id"] == "val-001"
    assert loaded_marker == marker

    with (tmp_path / "calibration_prediction_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="identity"):
        load_frozen_calibration_prediction_ledger(
            tmp_path,
            config=config,
            expected_prediction_count=1,
        )


def test_phase_a_applicability_is_only_deployable_skill_id() -> None:
    config = _config()
    config["calibration_data"]["manifest_sample_count"] = 2
    config["calibration_data"]["expected_skill_counts"] = {
        "0": 1,
        "1": 1,
        "2": 0,
        "3": 0,
        "4": 0,
    }
    config["calibration_data"]["expected_applicable_frame_count"] = 1
    config["calibration_data"]["expected_nonapplicable_frame_count"] = 1
    config["calibration_data"]["expected_trajectories_with_applicable_frames"] = 1
    rows = [
        {
            "trajectory_id": "val-001",
            "skill_id": 0,
            "calibration_applicable": True,
            "task_spec_taxonomy_sha256": (
                g2b.calibration_protocol.G2B_TASK_TAXONOMY_SHA256
            ),
        },
        {
            "trajectory_id": "val-001",
            "skill_id": 1,
            "calibration_applicable": False,
            "task_spec_taxonomy_sha256": (
                g2b.calibration_protocol.G2B_TASK_TAXONOMY_SHA256
            ),
        },
    ]
    audit = g2b.audit_calibration_prediction_applicability(rows, config=config)
    assert audit["skill_counts"] == {"0": 1, "1": 1, "2": 0, "3": 0, "4": 0}
    assert audit["applicable_frame_count"] == 1

    rows[1]["calibration_applicable"] = True
    with pytest.raises(RuntimeError, match="applicability"):
        g2b.audit_calibration_prediction_applicability(rows, config=config)


class _FakeStore:
    def __init__(self, arrays: object) -> None:
        self.arrays = arrays

    def get(self, _meta: object) -> object:
        return self.arrays


def test_phase_b_cohort_invariants_fail_whole_without_row_filtering() -> None:
    config = _config()
    config["calibration_data"]["expected_applicable_frame_count"] = 1
    config["calibration_data"]["expected_trajectories_with_applicable_frames"] = 1
    prediction = {
        "trajectory_id": "val-001",
        "timestep": 0,
        "skill_id": 0,
        "calibration_applicable": True,
    }
    meta = SimpleNamespace(file="val-001.npz")
    labels = SimpleNamespace(
        num_steps=1,
        object_position_base_m=np.asarray([[0.0, 0.0, 0.02]], dtype=np.float32),
    )
    deployable = SimpleNamespace(
        num_steps=1,
        skill_id=np.asarray([0], dtype=np.int16),
        is_grasped=np.asarray([False], dtype=np.bool_),
        finger_force_valid=np.asarray([True], dtype=np.bool_),
        left_finger_force_n=np.asarray([0.0], dtype=np.float32),
        right_finger_force_n=np.asarray([0.0], dtype=np.float32),
        proprio=np.asarray([[0.0] * 14 + [1.0]], dtype=np.float32),
    )
    kwargs = {
        "predictions": [prediction],
        "config": config,
        "label_by_trajectory": {"val-001": meta},
        "label_store": _FakeStore(labels),
        "deployable_by_trajectory": {"val-001": meta},
        "deployable_store": _FakeStore(deployable),
        "phase_a_applicability_audit": {"audit_sha256": "a" * 64},
    }
    audit = g2b._audit_calibration_cohort_invariants(**kwargs)
    assert audit["cohort_passed"] is True
    assert set(audit["invariant_violation_counts"].values()) == {0}

    deployable.is_grasped[0] = True
    deployable.left_finger_force_n[0] = 0.02
    deployable.proprio[0, -1] = 0.9
    audit = g2b._audit_calibration_cohort_invariants(**kwargs)
    assert audit["cohort_passed"] is False
    assert audit["applicable_frame_count"] == 1
    assert audit["invariant_violation_counts"]["is_grasped"] == 1
    assert audit["invariant_violation_counts"]["left_finger_force"] == 1
    assert audit["invariant_violation_counts"]["raw_gripper_opening_ratio"] == 1


def test_calibration_prediction_ledger_rejects_privileged_fields() -> None:
    with pytest.raises(ValueError, match="privileged"):
        assert_calibration_prediction_ledger_deployable_only(
            [{"trajectory_id": "val-001", "gt_object_position_base_m": [0, 0, 0]}]
        )


@pytest.mark.parametrize(
    ("status", "gate_evaluated", "gate_passed", "protocol_valid"),
    [
        ("complete-calibration-pass", True, True, True),
        ("complete-calibration-no-go", True, False, True),
        ("complete-calibration-protocol-invalid", False, None, False),
    ],
)
def test_calibration_receipt_preserves_gate_tristate(
    tmp_path: Path,
    status: str,
    gate_evaluated: bool,
    gate_passed: bool | None,
    protocol_valid: bool,
) -> None:
    _write_calibration_receipt_fixture(
        tmp_path,
        status=status,
        gate_evaluated=gate_evaluated,
        gate_passed=gate_passed,
        protocol_valid=protocol_valid,
    )
    receipt = g2b.verify_g2b_calibration_receipt(tmp_path)
    assert receipt["protocol_gate_evaluated"] is True
    assert receipt["gate_evaluated"] is gate_evaluated
    assert receipt["gate_passed"] is gate_passed
    assert receipt["protocol_valid"] is protocol_valid


@pytest.mark.parametrize(
    ("status", "gate_evaluated", "gate_passed", "protocol_valid"),
    [
        ("complete-calibration-protocol-invalid", False, False, False),
        ("complete-calibration-no-go", True, None, True),
        ("complete-calibration-pass", True, False, True),
    ],
)
def test_calibration_receipt_rejects_invalid_gate_tristate(
    tmp_path: Path,
    status: str,
    gate_evaluated: bool,
    gate_passed: bool | None,
    protocol_valid: bool,
) -> None:
    _write_calibration_receipt_fixture(
        tmp_path,
        status=status,
        gate_evaluated=gate_evaluated,
        gate_passed=gate_passed,
        protocol_valid=protocol_valid,
    )
    with pytest.raises(RuntimeError, match="状态|三态"):
        g2b.verify_g2b_calibration_receipt(tmp_path)


def test_protocol_invalid_freezes_complete_negative_artifact(tmp_path: Path) -> None:
    config = _config()
    prediction_ledger = tmp_path / "calibration_prediction_ledger.jsonl"
    g2b._atomic_jsonl(prediction_ledger, [])
    marker = {
        "prediction_ledger_sha256": g2b.file_sha256(prediction_ledger),
        "freeze_marker_sha256": "b" * 64,
        "applicability_audit": {"audit_sha256": "c" * 64},
    }
    cohort_audit = {
        "frame_count": 4154,
        "validation_label_array_file_read_count": 20,
        "audit_sha256": "d" * 64,
    }
    g2b._atomic_json(tmp_path / "calibration_cohort_audit.json", cohort_audit)
    summary, calibration, scoring_rows = g2b._write_protocol_invalid_calibration_result(
        output=tmp_path,
        config=config,
        marker=marker,
        prediction_ledger=prediction_ledger,
        validation_data_identity="e" * 64,
        cohort_audit=cohort_audit,
    )
    assert summary["protocol_gate_evaluated"] is True
    assert summary["gate_evaluated"] is False
    assert summary["gate_passed"] is None
    assert summary["selection_evaluated"] is False
    assert summary["fit_evaluated"] is False
    assert calibration["gate_passed"] is None
    assert scoring_rows == []
    assert (tmp_path / "calibration_scoring_ledger.jsonl").read_bytes() == b""

    _write_calibration_receipt_fixture(
        tmp_path,
        status="complete-calibration-protocol-invalid",
        gate_evaluated=False,
        gate_passed=None,
        protocol_valid=False,
    )
    receipt = g2b.verify_g2b_calibration_receipt(tmp_path)
    assert receipt["gate_passed"] is None
    assert len(receipt["files"]) == 12
    with pytest.raises(RuntimeError, match="未通过"):
        g2b.load_passed_covariance_calibration(tmp_path)


def test_full_qualification_requires_explicit_decision_exit_go() -> None:
    with pytest.raises(RuntimeError, match="decision Agent"):
        run_e018_p1_g2b_qualification(
            config_path="unused",
            parent_g2a_config_path="unused",
            parent_g2a_receipt_path="unused",
            parent_g0c_config_path="unused",
            parent_g0c_receipt_path="unused",
            calibration_output="unused",
            e016_config_path="unused",
            e013_deployable_root="unused",
            e016_fresh_deployable_root="unused",
            training_output="unused",
            repository_root="unused",
            output_root="unused",
            preflight_only=False,
            decision_exit_go=False,
        )


def test_phase_b_apis_cannot_receive_model_or_provider_context() -> None:
    for function in (
        g2b._score_calibration_after_prediction_freeze,
        g2b._score_qualification_after_prediction_freeze,
    ):
        names = set(inspect.signature(function).parameters)
        assert "context" not in names
        assert "model" not in names
        assert "provider" not in names
