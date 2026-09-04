from __future__ import annotations

import inspect
import json
import os
import stat
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import robot_vla.precision.e018_p1_g2a as g2a
from robot_vla.precision.e018_p1_g2a import (
    E018_P1_G2A_CONFIG_VERSION,
    FRONT_ALTERNATE_IDS,
    FRONT_HOME_ID,
    NATIVE_WRIST_CONTROL_ID,
    PER_SCENE_CAPTURE_ORDER,
    assert_prediction_ledger_deployable_only,
    audit_qualification_seed_sets,
    camera_pose_ood_diagnostic,
    finalize_qualification_summaries,
    freeze_prediction_ledger,
    load_e018_p1_g2a_config,
    load_frozen_prediction_ledger,
    select_primary_front_viewpoint,
    summarize_qualification_rows,
    verify_g2a_receipt,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    ROOT
    / "configs"
    / "e018_p1_g2a_front_provider_qualification_development_v1.json"
)
PARENT_PATH = ROOT / "configs" / "e018_p1_g0c_rotated_motion_development_v1.json"


def _config() -> dict:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _qualified_rows(primitive_id: str) -> list[dict]:
    rows = []
    covariance = np.diag((1e-6, 1e-6, 0.0)).tolist()
    for index in range(50):
        observable = index < 45
        rows.append(
            {
                "seed": 75001 + index,
                "primitive_id": primitive_id,
                "calibration_identity_sha256": "1" * 64,
                "provider_identity_sha256": "2" * 64,
                "capture_integrity_passed": True,
                "physical_safety_passed": True,
                "predicted_observable": observable,
                "gt_observable": observable,
                "geometry_valid": observable,
                "world_xyz_error_m": 0.001 if observable else None,
                "world_xy_error_vector_m": [0.001, 0.001] if observable else None,
                "measurement_covariance_base_m2": covariance if observable else None,
                "write_accepted": observable,
                "oracle_safe_measurement": observable,
                "structurally_evaluable": observable,
                "camera_pose_ood": {
                    "outside_envelope": primitive_id != NATIVE_WRIST_CONTROL_ID,
                    "outside_dimension_count": (
                        0 if primitive_id == NATIVE_WRIST_CONTROL_ID else 3
                    ),
                    "maximum_component_excess": (
                        0.0 if primitive_id == NATIVE_WRIST_CONTROL_ID else 0.1
                    ),
                },
            }
        )
    return rows


def _passing_summary(primitive_id: str, *, coverage: float = 0.5) -> dict:
    summary = summarize_qualification_rows(
        _qualified_rows(primitive_id),
        config=_config(),
    )
    summary["accepted_safe_coverage"] = coverage
    return summary


def test_config_is_strict_pre_result_qualification_only() -> None:
    config = load_e018_p1_g2a_config(
        CONFIG_PATH,
        parent_g0c_config_path=PARENT_PATH,
    )

    assert config["version"] == E018_P1_G2A_CONFIG_VERSION
    assert config["sampling"]["seeds"] == list(range(75001, 75051))
    assert config["viewpoints"]["per_scene_capture_order"] == list(
        PER_SCENE_CAPTURE_ORDER
    )
    assert config["qualification"]["minimum_covariance_evaluable_count"] == 30
    assert config["scope"]["test_trajectory_array_read_allowed"] is False
    assert config["execution"]["memory_write_allowed"] is False
    assert config["execution"]["runtime_camera_actuation_allowed"] is False


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (
            ("qualification", "maximum_observable_world_xyz_p90_m"),
            0.006,
            "qualification",
        ),
        (("qualification", "minimum_covariance_evaluable_count"), 5, "qualification"),
        (("execution", "memory_write_allowed"), True, "execution"),
        (("scope", "test_trajectory_array_read_allowed"), True, "scope"),
        (
            ("adapter", "maximum_rotation_projection_error_frobenius"),
            1e-5,
            "adapter",
        ),
        (("geometry", "object_center_plane_tolerance_m"), 1e-4, "geometry"),
    ],
)
def test_config_rejects_post_result_gate_or_permission_drift(
    tmp_path: Path,
    path: tuple[str, str],
    value: object,
    message: str,
) -> None:
    config = _config()
    config[path[0]][path[1]] = value

    with pytest.raises(ValueError, match=message):
        load_e018_p1_g2a_config(
            _write_config(tmp_path, config),
            parent_g0c_config_path=PARENT_PATH,
        )


def test_seed_audit_checks_manifests_and_development_registry() -> None:
    audit = audit_qualification_seed_sets(
        candidate_seeds=[75001, 75002],
        manifest_seed_groups={"all_splits": [131000, 134000]},
        known_development_seed_groups={"g0": [71001], "g1a": [74101]},
    )
    assert audit["passed"] is True
    assert all(not overlap for overlap in audit["overlaps"].values())

    overlap = audit_qualification_seed_sets(
        candidate_seeds=[75001, 75002],
        manifest_seed_groups={"all_splits": [75002]},
        known_development_seed_groups={"g0": [71001]},
    )
    assert overlap["passed"] is False
    assert overlap["overlaps"]["manifest:all_splits"] == [75002]


def test_prediction_ledger_rejects_privileged_gt_at_any_depth() -> None:
    assert_prediction_ledger_deployable_only(
        [{"prediction": {"visibility": 0.8}, "target_camera": "base_camera"}]
    )
    with pytest.raises(ValueError, match="privileged"):
        assert_prediction_ledger_deployable_only(
            [{"prediction": {"gt_observable": True}}]
        )


def test_atomic_writes_fsync_file_and_parent_directory(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    observed: list[str] = []
    real_fsync = os.fsync

    def recording_fsync(file_descriptor: int) -> None:
        mode = os.fstat(file_descriptor).st_mode
        observed.append("directory" if stat.S_ISDIR(mode) else "file")
        real_fsync(file_descriptor)

    monkeypatch.setattr(g2a.os, "fsync", recording_fsync)
    g2a._atomic_json(tmp_path / "value.json", {"value": 1})
    g2a._atomic_jsonl(tmp_path / "value.jsonl", [{"value": 2}])

    assert observed == ["file", "directory", "file", "directory"]


def test_phase_b_can_only_reload_fsynced_frozen_prediction_rows(
    tmp_path: Path,
) -> None:
    config_sha = "a" * 64
    mutable_rows = [{"prediction": {"visibility": 0.8}, "seed": 75001}]
    freeze_prediction_ledger(
        tmp_path,
        rows=mutable_rows,
        config_sha256=config_sha,
    )
    mutable_rows[0]["prediction"]["visibility"] = 0.1

    reloaded, marker = load_frozen_prediction_ledger(
        tmp_path,
        config_sha256=config_sha,
        expected_prediction_count=1,
    )

    assert reloaded[0]["prediction"]["visibility"] == 0.8
    assert marker["status"] == "frozen-before-privileged-gt-read"
    phase_b_parameters = inspect.signature(
        g2a._score_after_prediction_freeze
    ).parameters
    assert "predictions" not in phase_b_parameters
    assert "prediction_freeze" not in phase_b_parameters
    assert "context" not in phase_b_parameters


def test_receipt_verifier_checks_internal_and_artifact_hashes(tmp_path: Path) -> None:
    config_snapshot: dict[str, object] = {"version": "fixture/v1"}
    source_identity_sha = "b" * 64
    prediction_ledger_sha = "c" * 64
    for name in g2a._ARTIFACT_NAMES:
        if name == "config_snapshot.json":
            g2a._atomic_json(tmp_path / name, config_snapshot)
        elif name == "source_identity.json":
            g2a._atomic_json(
                tmp_path / name,
                {"identity_sha256": source_identity_sha},
            )
        elif name == "prediction_freeze.json":
            g2a._atomic_json(
                tmp_path / name,
                {"prediction_ledger_sha256": prediction_ledger_sha},
            )
        else:
            g2a._atomic_text(tmp_path / name, f"fixture:{name}\n")
    receipt = {
        "version": g2a.E018_P1_G2A_RESULT_VERSION,
        "status": "complete-development-only",
        "gate": g2a.E018_P1_G2A_GATE,
        "gate_evaluated": True,
        "gate_passed": False,
        "config_sha256": g2a.canonical_sha256(config_snapshot),
        "source_identity_sha256": source_identity_sha,
        "prediction_ledger_sha256": prediction_ledger_sha,
        "prediction_frozen_before_gt": True,
        "test_split_status": "manifest-metadata-read-arrays-prohibited-unread",
        "test_trajectory_array_read_count": 0,
        "test_label_array_read_count": 0,
        "live_memory_read_count": 0,
        "live_memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "arm_actuation_count": 0,
        "manipulation_progression_count": 0,
        "provider_training_count": 0,
        "files": g2a._artifact_hashes(tmp_path, list(g2a._ARTIFACT_NAMES)),
    }
    receipt["receipt_sha256"] = g2a.canonical_sha256(receipt)
    g2a._atomic_json(tmp_path / "receipt.json", receipt)

    verified = verify_g2a_receipt(tmp_path)
    assert verified["receipt_sha256"] == receipt["receipt_sha256"]

    (tmp_path / "report.md").write_text("tampered\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact identity"):
        verify_g2a_receipt(tmp_path)


def test_camera_pose_ood_uses_actual_base_frame_pose_components() -> None:
    envelope = {
        "minimum": [-0.1] * 9,
        "maximum": [1.1] * 9,
        "envelope_sha256": "1" * 64,
    }
    inside = camera_pose_ood_diagnostic(np.eye(4), envelope)
    assert inside["outside_envelope"] is False

    moved = np.eye(4)
    moved[0, 3] = 2.0
    outside = camera_pose_ood_diagnostic(moved, envelope)
    assert outside["outside_envelope"] is True
    assert outside["outside_dimension_count"] == 1
    assert outside["maximum_component_excess"] == pytest.approx(0.9)


def test_summary_passes_all_frozen_absolute_gates() -> None:
    summary = summarize_qualification_rows(
        _qualified_rows("LEFT_LOW__YAW_LEFT"),
        config=_config(),
    )

    assert summary["absolute_gate_passed"] is True
    assert summary["visibility_precision"] == 1.0
    assert summary["visibility_recall"] == 1.0
    assert summary["observable_world_xyz_p90_m"] == pytest.approx(0.001)
    assert summary["unsafe_accepted_count"] == 0
    assert summary["covariance_evaluable_count"] == 45


def test_singular_psd_covariance_rejects_error_in_zero_variance_direction() -> None:
    covariance = np.diag((1e-6, 0.0))

    distance = g2a._mahalanobis_squared_psd(
        np.asarray((0.0, 0.001)),
        covariance,
    )

    assert np.isinf(distance)


def test_undefined_visibility_metric_and_low_covariance_support_fail() -> None:
    rows = _qualified_rows("LEFT_LOW__YAW_LEFT")
    for row in rows:
        row["gt_observable"] = False
        row["predicted_observable"] = False
        row["geometry_valid"] = False
        row["world_xyz_error_m"] = None
        row["world_xy_error_vector_m"] = None
        row["measurement_covariance_base_m2"] = None
        row["write_accepted"] = False
        row["oracle_safe_measurement"] = False
        row["structurally_evaluable"] = False

    summary = summarize_qualification_rows(rows, config=_config())

    assert summary["visibility_precision"] is None
    assert summary["visibility_recall"] is None
    assert summary["absolute_gate_passed"] is False
    assert "visibility_precision_defined" in summary["failure_reasons"]
    assert "covariance_support" in summary["failure_reasons"]


def test_summary_rejects_wrong_seed_identity_even_with_fifty_unique_rows() -> None:
    rows = _qualified_rows("LEFT_LOW__YAW_LEFT")
    rows[-1]["seed"] = 99999

    summary = summarize_qualification_rows(rows, config=_config())

    assert summary["absolute_gate_passed"] is False
    assert "complete_seed_coverage" in summary["failure_reasons"]


def test_native_failure_makes_every_front_result_inconclusive() -> None:
    summaries = [_passing_summary(item) for item in PER_SCENE_CAPTURE_ORDER]
    summaries[0] = deepcopy(summaries[0])
    summaries[0]["absolute_gate_passed"] = False
    summaries[0]["failure_reasons"] = ["visibility_recall"]

    final = finalize_qualification_summaries(summaries)

    assert final["status"] == "inconclusive_parent_health"
    assert final["primary"] is None
    assert final["qualified_front_alternate_ids"] == []
    assert all(
        summary["status"] == "inconclusive-native-wrist-control-failed"
        for summary in final["summaries"]
        if summary["primitive_id"] != NATIVE_WRIST_CONTROL_ID
    )


def test_finalization_rejects_duplicate_viewpoint_summary() -> None:
    summaries = [_passing_summary(item) for item in PER_SCENE_CAPTURE_ORDER]
    summaries.append(deepcopy(summaries[-1]))

    with pytest.raises(ValueError, match="viewpoint"):
        finalize_qualification_summaries(summaries)


def test_primary_tie_break_prefers_shortlist_then_higher_coverage() -> None:
    non_shortlist = _passing_summary("LEFT_LOW__CENTER", coverage=0.99)
    shortlist_lower = _passing_summary("LEFT_LOW__YAW_LEFT", coverage=0.30)
    shortlist_higher = _passing_summary("LEFT_LOW__PITCH_UP", coverage=0.40)
    for summary in (non_shortlist, shortlist_lower, shortlist_higher):
        summary["status"] = "pass"

    selected = select_primary_front_viewpoint(
        [non_shortlist, shortlist_lower, shortlist_higher]
    )

    assert selected is not None
    assert selected["primitive_id"] == "LEFT_LOW__PITCH_UP"


def test_primary_tie_break_uses_lexical_order_for_non_shortlist_final_tie() -> None:
    left = _passing_summary("LEFT_LOW__CENTER", coverage=0.5)
    right = _passing_summary("RIGHT_LOW__CENTER", coverage=0.5)
    left["status"] = "pass"
    right["status"] = "pass"

    selected = select_primary_front_viewpoint([right, left])

    assert selected is not None
    assert selected["primitive_id"] == "LEFT_LOW__CENTER"


def test_complete_native_pass_selects_only_qualified_alternate() -> None:
    summaries = [_passing_summary(item) for item in PER_SCENE_CAPTURE_ORDER]
    failed_id = FRONT_ALTERNATE_IDS[0]
    for summary in summaries:
        if summary["primitive_id"] == failed_id:
            summary["absolute_gate_passed"] = False
            summary["failure_reasons"] = ["unsafe_accepted"]

    final = finalize_qualification_summaries(summaries)

    assert final["status"] == "pass"
    assert failed_id not in final["qualified_front_alternate_ids"]
    assert final["primary"] is not None
    assert final["primary"]["primitive_id"] in FRONT_ALTERNATE_IDS
    assert FRONT_HOME_ID not in final["qualified_front_alternate_ids"]
