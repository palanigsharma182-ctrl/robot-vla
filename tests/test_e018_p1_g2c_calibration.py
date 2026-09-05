from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import numpy as np
import pytest

from robot_vla.precision.e018_p1_g2a import canonical_sha256, file_sha256
from robot_vla.precision.e018_p1_g2c import (
    _measurement_covariance,
    calibrate_g2c_viewpoint,
)
from robot_vla.precision.e018_p1_g2c_calibration import (
    _PHASE_A_CHECK_EVIDENCE_NAMES,
    _PREDICTION_ARTIFACTS,
    _RESULT_ARTIFACTS,
    E018_P1_G2C_CALIBRATION_CHECK_EVIDENCE_VERSION,
    E018_P1_G2C_CALIBRATION_COMPLETION_MARKER_VERSION,
    E018_P1_G2C_CALIBRATION_FREEZE_VERSION,
    E018_P1_G2C_CALIBRATION_INPUT_VIEW_VERSION,
    E018_P1_G2C_CALIBRATION_PERSISTENCE_VERSION,
    E018_P1_G2C_CALIBRATION_RESULT_VERSION,
    G2C_CALIBRATION_SELECTION_PARENT,
    _atomic_json,
    _atomic_jsonl,
    _calibration_summary,
    _deployable_free_static_safe,
    _input_manifest_rows,
    _load_calibration_labels,
    _prepare_calibration_input_view,
    _score_calibration_prediction,
    _validate_prediction_row_mechanics,
    build_g2c_calibration_config,
    build_g2c_calibration_phase_a_completion_marker,
    finalize_g2c_calibration_phase_a_persistence,
    load_g2c_calibration_config,
    prepare_g2c_calibration_deployable_view,
    prepare_g2c_calibration_privileged_view,
    record_g2c_calibration_phase_a_check_evidence,
    run_g2c_calibration_prediction_freeze,
    score_calibrate_g2c,
    verify_g2c_calibration_phase_a_persistence,
    verify_g2c_calibration_prediction_freeze,
    verify_g2c_calibration_result,
)
from robot_vla.precision.e018_p1_g2c_data import (
    _LABEL_ARRAYS,
    G2C_LABEL_SCHEMA_VERSION,
    G2C_MANIFEST_SCHEMA_VERSION,
    G2C_VIEW_ORDER,
)
from robot_vla.precision.object_observability import ObjectWriteEvidence
from robot_vla.precision.outliers import geometry_conditioning


def _source_identity() -> dict[str, str]:
    value = {
        "git_commit": "a" * 40,
        "source_tree_sha256": "b" * 64,
    }
    value["identity_sha256"] = canonical_sha256(value)
    return value


def _safety() -> dict[str, object]:
    return {
        "eligible_capture": True,
        "finger_force_n": [0.0, 0.0],
        "finger_force_valid": True,
        "raw_gripper_opening_ratio": 1.0,
        "arm_joint_drift_rad": 0.0,
        "tcp_position_drift_m": 0.0,
        "tcp_orientation_drift_rad": 0.0,
        "rgb_timestamp_s": 0.25,
        "pose_timestamp_s": 0.25,
        "camera_position_tracking_error_m": 0.0,
        "camera_orientation_tracking_error_rad": 0.0,
        "rotation_projection_error_frobenius": 0.0,
    }


def _prediction(
    *,
    config: dict[str, object],
    row_index: int = 0,
    seed: int = 990001,
    sample_index: int = 0,
    viewpoint_id: str = "HOME__CENTER",
    sigma_px: float = 1.0,
) -> dict[str, object]:
    intrinsic = np.asarray(
        [[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    base_from_camera = np.eye(4, dtype=np.float64)
    base_from_camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
    base_from_camera[:3, 3] = [0.0, 0.0, 1.0]
    uv = np.asarray([0.5, 0.5], dtype=np.float64)
    geometry = geometry_conditioning(
        normalized_uv=uv,
        intrinsic_cv=intrinsic,
        base_from_camera_cv=base_from_camera,
        image_size_hw=(128, 128),
        plane_base_z_m=0.02,
    )
    sigma = np.asarray([sigma_px, sigma_px], dtype=np.float64)
    covariance = _measurement_covariance(
        geometry["local_jacobian_xy_m_per_px"], sigma
    )
    evidence = ObjectWriteEvidence(
        visibility_probability=0.9,
        projection_validity_probability=0.9,
        object_mask_probability=0.9,
        goal_mask_probability=0.1,
        normalized_entropy=0.1,
        radial_sigma_px=float(np.linalg.norm(sigma)),
        geometry_valid=True,
    )
    parent = G2C_CALIBRATION_SELECTION_PARENT
    row = {
        "version": E018_P1_G2C_CALIBRATION_FREEZE_VERSION,
        "phase": "deployable-calibration-before-privileged-label-open/v1",
        "candidate_id": parent["candidate_id"],
        "epoch": parent["epoch"],
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "checkpoint_parameter_sha256": parent["parameter_state_sha256"],
        "checkpoint_provenance_sha256": parent["provenance_sha256"],
        "checkpoint_model_config_sha256": parent["model_config_sha256"],
        "row_index": row_index,
        "batch_index": row_index // 32,
        "batch_offset": row_index % 32,
        "seed": seed,
        "split": "calibration",
        "sample_index": sample_index,
        "viewpoint_id": viewpoint_id,
        "input_sha256": hashlib.sha256(f"synthetic-{row_index}".encode()).hexdigest(),
        "predicted_object_normalized_uv": uv.tolist(),
        "predicted_goal_normalized_uv": [0.5, 0.5],
        "object_visibility_probability": 0.9,
        "goal_visibility_probability": 0.9,
        "projection_validity_probability": 0.9,
        "object_normalized_entropy": 0.1,
        "object_sigma_xy_px": sigma.tolist(),
        "object_mask_probability_at_prediction": 0.9,
        "goal_mask_probability_at_prediction": 0.1,
        "predicted_observable": True,
        "geometry_valid": True,
        "predicted_object_position_base_m": np.asarray(
            geometry["predicted_world_point_base_m"], dtype=np.float64
        ).tolist(),
        "raw_covariance_base_m2": covariance.tolist(),
        "write_score": evidence.score,
        "external_intrinsic_cv": intrinsic.tolist(),
        "base_from_external_camera_cv": base_from_camera.tolist(),
        "deployable_safety": _safety(),
        "memory_write_allowed": False,
        "actuation_allowed": False,
    }
    _validate_prediction_row_mechanics(row, config=config)
    return row


def test_config_roundtrip_and_frozen_parent(tmp_path: Path) -> None:
    config = build_g2c_calibration_config()
    path = tmp_path / "config.json"
    _atomic_json(path, config)

    loaded = load_g2c_calibration_config(path)

    assert loaded == config
    assert loaded["selection_parent"]["candidate_id"] == "W-KV0"
    assert loaded["selection_parent"]["epoch"] == 15
    assert loaded["protocol"][
        "privileged_staging_requires_phase_a_drive_zero_difference"
    ] is True
    tracked = (
        Path(__file__).resolve().parents[1]
        / "configs"
        / "e018_p1_g2c_calibration_development_v1.json"
    )
    assert load_g2c_calibration_config(tracked) == config


def test_public_verifier_signatures_exclude_sensitive_inputs() -> None:
    freeze_parameters = set(
        inspect.signature(verify_g2c_calibration_prediction_freeze).parameters
    )
    result_parameters = set(
        inspect.signature(verify_g2c_calibration_result).parameters
    )

    assert freeze_parameters == {"calibration_config_path", "output_root"}
    assert result_parameters == {
        "calibration_config_path",
        "prediction_freeze_root",
        "output_root",
    }
    forbidden = {"label", "model", "checkpoint", "data_root", "rgb"}
    assert not any(
        token in name
        for name in freeze_parameters | result_parameters
        for token in forbidden
    )


def test_hold_fails_before_any_config_or_input_read(monkeypatch: pytest.MonkeyPatch) -> None:
    def unexpected(*args: object, **kwargs: object) -> object:
        raise AssertionError("HOLD 后不应读取 config/source/input")

    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.load_g2c_calibration_config",
        unexpected,
    )
    with pytest.raises(PermissionError):
        prepare_g2c_calibration_deployable_view(
            calibration_config_path="missing",
            training_config_path="missing",
            data_root="missing",
            output_root="missing",
            decision_exit_go=False,
        )
    with pytest.raises(PermissionError):
        prepare_g2c_calibration_privileged_view(
            calibration_config_path="missing",
            training_config_path="missing",
            data_root="missing",
            prediction_freeze_root="missing",
            repository_root="missing",
            phase_a_persistence_receipt_path="missing",
            expected_phase_a_persistence_receipt_raw_sha256="0" * 64,
            output_root="missing",
            decision_exit_go=False,
        )
    with pytest.raises(PermissionError):
        run_g2c_calibration_prediction_freeze(
            calibration_config_path="missing",
            training_config_path="missing",
            training_output_root="missing",
            model_val_prediction_freeze_root="missing",
            model_val_selection_root="missing",
            calibration_deployable_input_root="missing",
            repository_root="missing",
            output_root="missing",
            decision_exit_go=False,
        )
    with pytest.raises(PermissionError):
        score_calibrate_g2c(
            calibration_config_path="missing",
            training_config_path="missing",
            prediction_freeze_root="missing",
            calibration_privileged_input_root="missing",
            repository_root="missing",
            phase_a_persistence_receipt_path="missing",
            expected_phase_a_persistence_receipt_raw_sha256="0" * 64,
            output_root="missing",
            decision_exit_go=False,
        )


@pytest.mark.parametrize(
    ("include_deployable", "include_privileged", "allowed_manifest"),
    [
        (True, False, "deployable"),
        (False, True, "privileged_labels"),
    ],
)
def test_input_manifest_preflight_reads_only_requested_role(
    monkeypatch: pytest.MonkeyPatch,
    include_deployable: bool,
    include_privileged: bool,
    allowed_manifest: str,
) -> None:
    config = build_g2c_calibration_config()
    data_parent = {
        "data_receipt_raw_sha256": "1" * 64,
        "deployable_manifest_raw_sha256": "2" * 64,
        "privileged_manifest_raw_sha256": "3" * 64,
    }
    manifest = [
        {"split": "calibration", "seed": seed}
        for seed in range(76601, 76651)
    ]
    reads: list[str] = []

    def fake_file_sha(path: Path) -> str:
        name = path.as_posix()
        reads.append(name)
        if name.endswith("data_receipt.json"):
            return data_parent["data_receipt_raw_sha256"]
        if allowed_manifest in name:
            return data_parent[
                "deployable_manifest_raw_sha256"
                if include_deployable
                else "privileged_manifest_raw_sha256"
            ]
        raise AssertionError(f"跨 role 读取: {name}")

    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.file_sha256", fake_file_sha
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._read_jsonl",
        lambda path, name: manifest,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._manifest_inventory",
        lambda *args, **kwargs: {
            "deployable_inventory_sha256": (
                config["data_parent"]["deployable_inventory_sha256"]
                if include_deployable
                else None
            ),
            "privileged_inventory_sha256": (
                config["data_parent"]["privileged_inventory_sha256"]
                if include_privileged
                else None
            ),
            "paired_inventory_sha256": None,
        },
    )

    _input_manifest_rows(
        config=config,
        training_config={"data_parent": data_parent},
        data_root=Path("source"),
        include_deployable=include_deployable,
        include_privileged=include_privileged,
    )

    forbidden = "privileged_labels" if include_deployable else "deployable/"
    assert all(forbidden not in path for path in reads)


def test_privileged_staging_copies_50_bundles_without_np_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_g2c_calibration_config()
    manifest = [
        {
            "split": "calibration",
            "seed": seed,
            "file": f"bundles/calibration/seed-{seed:06d}.npz",
            "sha256": hashlib.sha256(str(seed).encode()).hexdigest(),
        }
        for seed in range(76601, 76651)
    ]
    inventory = {
        "deployable_inventory_sha256": None,
        "privileged_inventory_sha256": config["data_parent"][
            "privileged_inventory_sha256"
        ],
        "paired_inventory_sha256": None,
    }
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.load_g2c_calibration_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._verify_training_config_reference",
        lambda *args, **kwargs: {
            "raw_sha256": config["training_config"]["raw_sha256"],
            "internal_sha256": config["training_config"]["internal_sha256"],
            "verified": True,
        },
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.load_g2c_formal_training_config",
        lambda path: {"data_parent": {}},
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._input_manifest_rows",
        lambda **kwargs: (None, manifest, inventory),
    )
    copies: list[str] = []
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._copy_input_bundle",
        lambda source, target, expected_sha256: copies.append(target.as_posix()),
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.validate_g2c_calibration_input_view",
        lambda **kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        np,
        "load",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("privileged staging 禁止 np.load")
        ),
    )

    result = _prepare_calibration_input_view(
        calibration_config_path="config",
        training_config_path="training",
        data_root=tmp_path / "source",
        output_root=tmp_path / "staged",
        role="calibration-privileged",
        prediction_freeze_internal_sha256="1" * 64,
        source_identity_sha256="2" * 64,
        persistence_receipt_raw_sha256="3" * 64,
        persistence_receipt_internal_sha256="4" * 64,
        remote_identity_sha256="5" * 64,
    )

    assert result["verified"] is True
    assert len(copies) == 50
    receipt = json.loads(
        (tmp_path / "staged" / "input_view_receipt.json").read_text()
    )
    assert receipt["privileged_source_bundle_copy_count"] == 50
    assert receipt["privileged_label_array_open_count"] == 0


def test_label_loader_opens_exactly_50_bundles_once(tmp_path: Path) -> None:
    root = tmp_path / "labels"
    bundle_root = root / "privileged_labels" / "bundles" / "calibration"
    bundle_root.mkdir(parents=True)
    manifest = []
    for seed in range(76601, 76651):
        arrays = {
            "source_sample_index": np.arange(11, dtype=np.int64),
            "seed": np.full(11, seed, dtype=np.int64),
            "viewpoint_id": np.asarray(G2C_VIEW_ORDER),
            "object_position_base_m": np.tile(
                np.asarray([0.0, 0.0, 0.02], dtype=np.float32), (11, 1)
            ),
            "goal_position_base_m": np.zeros((11, 3), dtype=np.float32),
            "object_mask": np.zeros((11, 128, 128), dtype=np.bool_),
            "goal_mask": np.zeros((11, 128, 128), dtype=np.bool_),
            "normalized_uv": np.zeros((11, 2, 2), dtype=np.float32),
            "keypoint_projection_valid": np.ones((11, 2), dtype=np.bool_),
            "keypoint_observable": np.ones((11, 2), dtype=np.bool_),
            "object_exists": np.ones(11, dtype=np.bool_),
            "goal_exists": np.ones(11, dtype=np.bool_),
            "is_grasped": np.zeros(11, dtype=np.bool_),
            "robot_object_contact_force_n": np.zeros(11, dtype=np.float32),
            "geometry_roundtrip_error_m": np.zeros((11, 2), dtype=np.float64),
        }
        assert set(arrays) == set(_LABEL_ARRAYS)
        relative = f"bundles/calibration/seed-{seed:06d}.npz"
        path = root / "privileged_labels" / relative
        np.savez(path, **arrays)
        manifest.append(
            {
                "manifest_schema_version": G2C_MANIFEST_SCHEMA_VERSION,
                "split": "calibration",
                "seed": seed,
                "sample_count": 11,
                "view_order": list(G2C_VIEW_ORDER),
                "schema_version": G2C_LABEL_SCHEMA_VERSION,
                "file": relative,
                "sha256": file_sha256(path),
                "contains_model_input_rgb": False,
                "source_deployable_file": relative,
                "source_deployable_sha256": "0" * 64,
            }
        )
    manifest_path = root / "privileged_labels" / "manifest.jsonl"
    manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in manifest)
    )
    opens: list[int] = []

    labels = _load_calibration_labels(root, on_bundle_open=opens.append)

    assert len(labels) == 550
    assert opens == list(range(1, 51))


def test_scoring_recomputes_structural_oracle_and_catastrophic() -> None:
    config = build_g2c_calibration_config()
    prediction = _prediction(config=config)
    predicted = np.asarray(
        prediction["predicted_object_position_base_m"], dtype=np.float64
    )
    safe_label = {
        "gt_observable": True,
        "gt_object_position_base_m": predicted.tolist(),
        "gt_object_exists": True,
        "is_grasped": False,
        "robot_object_contact_force_n": 0.0,
    }
    safe = _score_calibration_prediction(
        prediction,
        safe_label,
        config=config,
        freeze_sha256="c" * 64,
    )
    unsafe_label = dict(safe_label)
    unsafe_position = predicted.copy()
    unsafe_position[0] += 0.03
    unsafe_label["gt_object_position_base_m"] = unsafe_position.tolist()
    unsafe = _score_calibration_prediction(
        prediction,
        unsafe_label,
        config=config,
        freeze_sha256="c" * 64,
    )

    assert safe["structurally_eligible"] is True
    assert safe["oracle_safe_measurement"] is True
    assert safe["catastrophic_measurement"] is False
    assert unsafe["structurally_eligible"] is True
    assert unsafe["oracle_safe_measurement"] is False
    assert unsafe["catastrophic_measurement"] is True


def test_deployable_safety_is_recomputed_not_trusted() -> None:
    config = build_g2c_calibration_config()
    value = _safety()
    assert _deployable_free_static_safe(value, config["protocol"]) is True
    value["finger_force_n"] = [0.0, 0.02]
    assert _deployable_free_static_safe(value, config["protocol"]) is False


def test_singular_psd_nullspace_error_is_protocol_valid_view_no_go() -> None:
    rows = [
        {
            "viewpoint_id": "LEFT_LOW__CENTER",
            "world_xy_error_vector_m": [0.0, 0.001],
            "raw_covariance_base_m2": np.diag([1e-6, 0.0, 0.0]).tolist(),
            "write_score": 0.8,
            "gt_observable": True,
            "geometry_valid": True,
            "structurally_eligible": True,
            "oracle_safe_measurement": True,
            "catastrophic_measurement": False,
        }
        for _ in range(30)
    ]

    result = calibrate_g2c_viewpoint(rows, viewpoint_id="LEFT_LOW__CENTER")

    assert result["status"] == "calibration-no-go"
    assert result["calibration"] is None
    assert result["conformity_score_count"] == 30
    assert result["finite_conformity_score_count"] == 0
    assert result["nonfinite_conformity_score_count"] == 30
    assert "nonfinite_conformity_score_present" in result["failure_reasons"]


def _build_persistence_control_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    config = build_g2c_calibration_config()
    source = _source_identity()
    freeze = {
        "freeze_raw_sha256": "d" * 64,
        "freeze_internal_sha256": "e" * 64,
        "source_git_commit": source["git_commit"],
        "source_identity_sha256": source["identity_sha256"],
        "artifact_bytes": 1234,
    }
    freeze_root = tmp_path / "freeze"
    freeze_root.mkdir()
    _atomic_json(
        freeze_root / "prediction_freeze.json",
        {"artifact_inventory_sha256": "f" * 64},
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.load_g2c_calibration_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.verify_g2c_calibration_prediction_freeze",
        lambda **kwargs: freeze,
    )
    control = tmp_path / "persistence-control"
    control.mkdir()
    remote = {
        "artifact_id": "synthetic-phase-a",
        "worker_id": "synthetic-worker",
        "remote_path": (
            "gdrive:VLA/experiments/e018-p1-g2c-calibration-phase-a/"
            "synthetic-phase-a/synthetic-worker"
        ),
    }

    def report(name: str, paths: list[str]) -> Path:
        path = tmp_path / name
        path.write_text("".join(f"= {item}\n" for item in paths))
        return path

    artifact_paths = [*_PREDICTION_ARTIFACTS, "prediction_freeze.json"]
    common = {
        "calibration_config_path": "unused",
        "prediction_freeze_root": freeze_root,
        **remote,
        "rclone_exit_code": 0,
        "decision_exit_go": True,
    }
    pre_path = control / _PHASE_A_CHECK_EVIDENCE_NAMES[
        "pre-marker-artifact-check"
    ]
    record_g2c_calibration_phase_a_check_evidence(
        **common,
        phase="pre-marker-artifact-check",
        combined_report_path=report("pre.combined", artifact_paths),
        output_path=pre_path,
    )
    marker_path = control / "DRIVE_BACKUP_COMPLETE.json"
    marker = build_g2c_calibration_phase_a_completion_marker(
        calibration_config_path="unused",
        prediction_freeze_root=freeze_root,
        pre_marker_check_evidence_path=pre_path,
        **remote,
        output_path=marker_path,
        decision_exit_go=True,
    )
    record_g2c_calibration_phase_a_check_evidence(
        **common,
        phase="post-marker-artifact-check",
        combined_report_path=report("post-artifact.combined", artifact_paths),
        output_path=control
        / _PHASE_A_CHECK_EVIDENCE_NAMES["post-marker-artifact-check"],
    )
    record_g2c_calibration_phase_a_check_evidence(
        **common,
        phase="post-marker-completion-marker-check",
        combined_report_path=report(
            "post-marker.combined", ["DRIVE_BACKUP_COMPLETE.json"]
        ),
        completion_marker_path=marker_path,
        output_path=control
        / _PHASE_A_CHECK_EVIDENCE_NAMES[
            "post-marker-completion-marker-check"
        ],
    )
    receipt_path = control / "phase_a_persistence_receipt.json"
    receipt = finalize_g2c_calibration_phase_a_persistence(
        calibration_config_path="unused",
        prediction_freeze_root=freeze_root,
        control_root=control,
        output_path=receipt_path,
        decision_exit_go=True,
    )
    return {
        "config": config,
        "freeze": freeze,
        "freeze_root": freeze_root,
        "control": control,
        "marker": marker,
        "marker_path": marker_path,
        "receipt": receipt,
        "receipt_path": receipt_path,
    }


def test_persistence_receipt_binds_three_checks_marker_and_freeze(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_persistence_control_fixture(tmp_path, monkeypatch)
    path = fixture["receipt_path"]
    assert isinstance(path, Path)

    verified = verify_g2c_calibration_phase_a_persistence(
        calibration_config_path="unused",
        prediction_freeze_root=fixture["freeze_root"],
        persistence_receipt_path=path,
        expected_receipt_raw_sha256=file_sha256(path),
    )

    assert verified["verified"] is True
    assert verified["pre_marker_difference_count"] == 0
    assert verified["post_marker_artifact_difference_count"] == 0
    assert verified["post_marker_completion_marker_difference_count"] == 0
    marker = fixture["marker"]
    assert isinstance(marker, dict)
    assert marker["version"] == E018_P1_G2C_CALIBRATION_COMPLETION_MARKER_VERSION
    receipt = fixture["receipt"]
    assert isinstance(receipt, dict)
    assert receipt["version"] == E018_P1_G2C_CALIBRATION_PERSISTENCE_VERSION
    pre = json.loads(
        (
            fixture["control"]
            / _PHASE_A_CHECK_EVIDENCE_NAMES["pre-marker-artifact-check"]
        ).read_text()
    )
    assert pre["version"] == E018_P1_G2C_CALIBRATION_CHECK_EVIDENCE_VERSION
    assert "receipt_sha256" not in marker
    assert not any("post_marker" in key for key in marker)


def test_persistence_finalize_requires_both_post_checks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_persistence_control_fixture(tmp_path, monkeypatch)
    control = fixture["control"]
    receipt_path = fixture["receipt_path"]
    assert isinstance(control, Path) and isinstance(receipt_path, Path)
    receipt_path.unlink()
    missing = control / _PHASE_A_CHECK_EVIDENCE_NAMES[
        "post-marker-completion-marker-check"
    ]
    missing.unlink()

    with pytest.raises((FileNotFoundError, RuntimeError)):
        finalize_g2c_calibration_phase_a_persistence(
            calibration_config_path="unused",
            prediction_freeze_root=fixture["freeze_root"],
            control_root=control,
            output_path=receipt_path,
            decision_exit_go=True,
        )
    assert not receipt_path.exists()


def test_persistence_rejects_rehashed_check_count_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_persistence_control_fixture(tmp_path, monkeypatch)
    control = fixture["control"]
    path = fixture["receipt_path"]
    assert isinstance(control, Path) and isinstance(path, Path)
    evidence_path = control / _PHASE_A_CHECK_EVIDENCE_NAMES[
        "post-marker-artifact-check"
    ]
    evidence = json.loads(evidence_path.read_text())
    evidence["matching_file_count"] -= 1
    evidence.pop("evidence_sha256")
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    _atomic_json(evidence_path, evidence)

    with pytest.raises(RuntimeError, match="evidence identity/count"):
        verify_g2c_calibration_phase_a_persistence(
            calibration_config_path="unused",
            prediction_freeze_root=fixture["freeze_root"],
            persistence_receipt_path=path,
            expected_receipt_raw_sha256=file_sha256(path),
        )


def test_persistence_rejects_completion_marker_self_reference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_persistence_control_fixture(tmp_path, monkeypatch)
    marker_path = fixture["marker_path"]
    receipt_path = fixture["receipt_path"]
    assert isinstance(marker_path, Path) and isinstance(receipt_path, Path)
    marker = json.loads(marker_path.read_text())
    marker["final_receipt_raw_sha256"] = file_sha256(receipt_path)
    marker.pop("marker_sha256")
    marker["marker_sha256"] = canonical_sha256(marker)
    _atomic_json(marker_path, marker)

    with pytest.raises(ValueError, match="keys 漂移"):
        verify_g2c_calibration_phase_a_persistence(
            calibration_config_path="unused",
            prediction_freeze_root=fixture["freeze_root"],
            persistence_receipt_path=receipt_path,
            expected_receipt_raw_sha256=file_sha256(receipt_path),
        )


def test_persistence_rejects_marker_inside_freeze_exact_tree(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_persistence_control_fixture(tmp_path, monkeypatch)
    freeze_root = fixture["freeze_root"]
    assert isinstance(freeze_root, Path)
    with pytest.raises(RuntimeError, match="禁止进入 Phase A exact-tree"):
        build_g2c_calibration_phase_a_completion_marker(
            calibration_config_path="unused",
            prediction_freeze_root=freeze_root,
            pre_marker_check_evidence_path=fixture["control"]
            / _PHASE_A_CHECK_EVIDENCE_NAMES["pre-marker-artifact-check"],
            artifact_id="synthetic-phase-a",
            worker_id="synthetic-worker",
            remote_path=(
                "gdrive:VLA/experiments/e018-p1-g2c-calibration-phase-a/"
                "synthetic-phase-a/synthetic-worker"
            ),
            output_path=freeze_root / "DRIVE_BACKUP_COMPLETE.json",
            decision_exit_go=True,
        )


@pytest.mark.parametrize("kind", ["extra", "symlink", "hardlink"])
def test_persistence_control_exact_tree_rejects_extra_and_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    fixture = _build_persistence_control_fixture(tmp_path, monkeypatch)
    control = fixture["control"]
    receipt_path = fixture["receipt_path"]
    assert isinstance(control, Path) and isinstance(receipt_path, Path)
    injected = control / f"injected-{kind}"
    if kind == "extra":
        injected.write_text("unexpected")
    elif kind == "symlink":
        injected.symlink_to(receipt_path)
    else:
        injected.hardlink_to(receipt_path)

    with pytest.raises(RuntimeError):
        verify_g2c_calibration_phase_a_persistence(
            calibration_config_path="unused",
            prediction_freeze_root=fixture["freeze_root"],
            persistence_receipt_path=receipt_path,
            expected_receipt_raw_sha256=file_sha256(receipt_path),
        )


def _build_complete_result_fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    safe_measurements: bool,
) -> dict[str, object]:
    config = build_g2c_calibration_config()
    source = _source_identity()
    checkpoint = {
        "candidate_id": G2C_CALIBRATION_SELECTION_PARENT["candidate_id"],
        "epoch": G2C_CALIBRATION_SELECTION_PARENT["epoch"],
        "relative_path": "precision-w-kv0-epoch-15.pt",
        "checkpoint_sha256": G2C_CALIBRATION_SELECTION_PARENT[
            "checkpoint_sha256"
        ],
        "parameter_state_sha256": G2C_CALIBRATION_SELECTION_PARENT[
            "parameter_state_sha256"
        ],
        "provenance_sha256": G2C_CALIBRATION_SELECTION_PARENT[
            "provenance_sha256"
        ],
        "model_config_sha256": G2C_CALIBRATION_SELECTION_PARENT[
            "model_config_sha256"
        ],
    }
    freeze = {
        "version": E018_P1_G2C_CALIBRATION_FREEZE_VERSION,
        "status": "complete-calibration-prediction-freeze-pass",
        "verified": True,
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": config["data_parent"]["data_identity_sha256"],
        "source_git_commit": source["git_commit"],
        "source_identity_sha256": source["identity_sha256"],
        "freeze_raw_sha256": "d" * 64,
        "freeze_internal_sha256": "e" * 64,
        "selected_checkpoint_identity": checkpoint,
        "selected_checkpoint_count": 1,
        "prediction_row_count": 550,
        "model_forward_batch_count": 18,
        "privileged_label_open_count_before_freeze": 0,
        "model_and_inference_context_destroyed": True,
        "artifact_bytes": 1234,
        "verification_sha256": "f" * 64,
    }
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.load_g2c_calibration_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.verify_g2c_calibration_prediction_freeze",
        lambda **kwargs: freeze,
    )
    freeze_root = tmp_path / "synthetic-freeze"
    freeze_root.mkdir()
    predictions = []
    scoring_rows = []
    for row_index in range(550):
        sample_index = row_index % len(G2C_VIEW_ORDER)
        prediction = _prediction(
            config=config,
            row_index=row_index,
            seed=76601 + row_index // len(G2C_VIEW_ORDER),
            sample_index=sample_index,
            viewpoint_id=G2C_VIEW_ORDER[sample_index],
        )
        predicted = np.asarray(
            prediction["predicted_object_position_base_m"], dtype=np.float64
        )
        gt = predicted.copy()
        if not safe_measurements:
            gt[0] += 0.01
        scoring_rows.append(
            _score_calibration_prediction(
                prediction,
                {
                    "gt_observable": True,
                    "gt_object_position_base_m": gt.tolist(),
                    "gt_object_exists": True,
                    "is_grasped": False,
                    "robot_object_contact_force_n": 0.0,
                },
                config=config,
                freeze_sha256=freeze["freeze_internal_sha256"],
            )
        )
        predictions.append(prediction)
    _atomic_jsonl(freeze_root / "prediction_ledger.jsonl", predictions)
    output = tmp_path / "synthetic-result"
    output.mkdir()
    _atomic_json(output / "config_snapshot.json", config)
    _atomic_json(output / "source_identity.json", source)
    _atomic_json(output / "prediction_freeze_verification.json", freeze)
    label_verification = {
        "version": E018_P1_G2C_CALIBRATION_INPUT_VIEW_VERSION,
        "verified": True,
        "role": "calibration-privileged",
        "split": "calibration",
        "seed_count": 50,
        "sample_count": 550,
        "bundle_bytes_verified": False,
        "prediction_freeze_internal_sha256": freeze[
            "freeze_internal_sha256"
        ],
        "source_identity_sha256": source["identity_sha256"],
        "phase_a_persistence_receipt_raw_sha256": "1" * 64,
        "phase_a_persistence_receipt_internal_sha256": "2" * 64,
        "phase_a_remote_identity_sha256": "3" * 64,
        "privileged_source_bundle_copy_count": 50,
        "privileged_label_array_open_count": 0,
        "receipt_raw_sha256": "4" * 64,
        "receipt_internal_sha256": "5" * 64,
        "inventory": {
            "deployable_inventory_sha256": None,
            "privileged_inventory_sha256": config["data_parent"][
                "privileged_inventory_sha256"
            ],
            "paired_inventory_sha256": None,
        },
    }
    label_verification["verification_sha256"] = canonical_sha256(
        label_verification
    )
    _atomic_json(output / "label_input_verification.json", label_verification)
    _atomic_jsonl(output / "calibration_scoring_ledger.jsonl", scoring_rows)
    calibrations = [
        calibrate_g2c_viewpoint(
            [row for row in scoring_rows if row["viewpoint_id"] == viewpoint],
            viewpoint_id=viewpoint,
        )
        for viewpoint in G2C_VIEW_ORDER
    ]
    _atomic_json(output / "viewpoint_calibrations.json", calibrations)
    summary = _calibration_summary(
        config=config,
        freeze_verification=freeze,
        source_identity=source,
        calibrations=calibrations,
        label_input_verification=label_verification,
    )
    _atomic_json(output / "calibration_summary.json", summary)
    receipt = {
        **summary,
        "artifact_sha256": {
            name: file_sha256(output / name) for name in _RESULT_ARTIFACTS
        },
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(output / "calibration_receipt.json", receipt)
    _atomic_json(
        output / "phase_state.json",
        {
            "version": E018_P1_G2C_CALIBRATION_RESULT_VERSION,
            "status": "complete-calibration-score",
            "config_sha256": config["config_sha256"],
            "prediction_freeze_internal_sha256": freeze[
                "freeze_internal_sha256"
            ],
            "source_identity_sha256": source["identity_sha256"],
            "label_array_consumed": True,
            "label_bundle_open_count": 50,
            "rerun_under_same_identity_allowed": False,
            "created_at_unix_ns": 1,
            "label_open_started_at_unix_ns": 2,
            "calibration_receipt_internal_sha256": receipt["receipt_sha256"],
            "precompletion_verification_sha256": "6" * 64,
        },
    )
    return {
        "config": config,
        "freeze_root": freeze_root,
        "output": output,
        "summary": summary,
        "scoring_rows": scoring_rows,
    }


@pytest.mark.parametrize("safe_measurements", [True, False])
def test_complete_result_public_verifier_accepts_protocol_positive_and_negative(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    safe_measurements: bool,
) -> None:
    fixture = _build_complete_result_fixture(
        tmp_path, monkeypatch, safe_measurements=safe_measurements
    )

    verified = verify_g2c_calibration_result(
        calibration_config_path="unused",
        prediction_freeze_root=fixture["freeze_root"],
        output_root=fixture["output"],
    )

    assert verified["verified"] is True
    assert verified["gate_passed"] is safe_measurements
    if safe_measurements:
        assert verified["qualified_non_home_viewpoint_count"] == 10
    else:
        assert verified["qualified_non_home_viewpoint_count"] == 0
        assert verified["status"] == "complete-calibration-protocol-valid-negative"


def test_complete_result_public_verifier_rejects_derived_bool_tamper(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fixture = _build_complete_result_fixture(
        tmp_path, monkeypatch, safe_measurements=True
    )
    rows = fixture["scoring_rows"]
    assert isinstance(rows, list)
    rows[0]["oracle_safe_measurement"] = False
    _atomic_jsonl(
        fixture["output"] / "calibration_scoring_ledger.jsonl", rows
    )

    with pytest.raises(RuntimeError, match="derived field"):
        verify_g2c_calibration_result(
            calibration_config_path="unused",
            prediction_freeze_root=fixture["freeze_root"],
            output_root=fixture["output"],
        )


@pytest.mark.parametrize("kind", ["extra", "symlink", "hardlink"])
def test_complete_result_exact_tree_rejects_extra_and_links(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    fixture = _build_complete_result_fixture(
        tmp_path, monkeypatch, safe_measurements=True
    )
    output = fixture["output"]
    assert isinstance(output, Path)
    receipt = output / "calibration_receipt.json"
    injected = output / f"injected-{kind}"
    if kind == "extra":
        injected.write_text("unexpected")
    elif kind == "symlink":
        injected.symlink_to(receipt)
    else:
        injected.hardlink_to(receipt)

    with pytest.raises(RuntimeError):
        verify_g2c_calibration_result(
            calibration_config_path="unused",
            prediction_freeze_root=fixture["freeze_root"],
            output_root=output,
        )


@pytest.mark.parametrize("kind", ["symlink", "hardlink"])
def test_prediction_freeze_verifier_rejects_links_before_artifact_reads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str
) -> None:
    config = build_g2c_calibration_config()
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.load_g2c_calibration_config",
        lambda path: config,
    )
    root = tmp_path / "freeze-links"
    root.mkdir()
    target = root / "target.json"
    target.write_text("{}")
    injected = root / "linked.json"
    if kind == "symlink":
        injected.symlink_to(target)
    else:
        injected.hardlink_to(target)

    with pytest.raises(RuntimeError):
        verify_g2c_calibration_prediction_freeze(
            calibration_config_path="unused", output_root=root
        )


def test_missing_persistence_fails_before_privileged_source_read(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    freeze = {
        "freeze_internal_sha256": "e" * 64,
    }
    source = _source_identity()
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.verify_g2c_calibration_prediction_freeze",
        lambda **kwargs: freeze,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._read_json",
        lambda *args, **kwargs: {
            "source_git_commit": source["git_commit"],
            "source_identity_sha256": source["identity_sha256"],
        },
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._git_source_identity",
        lambda path: source,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.verify_g2c_calibration_phase_a_persistence",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("missing marker")),
    )
    source_reads = {"count": 0}

    def unexpected_prepare(**kwargs: object) -> object:
        source_reads["count"] += 1
        raise AssertionError("persistence gate 前不得读取 privileged source")

    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._prepare_calibration_input_view",
        unexpected_prepare,
    )

    with pytest.raises(RuntimeError, match="missing marker"):
        prepare_g2c_calibration_privileged_view(
            calibration_config_path="config",
            training_config_path="training",
            data_root="privileged-source",
            prediction_freeze_root="freeze",
            repository_root="repo",
            phase_a_persistence_receipt_path="missing",
            expected_phase_a_persistence_receipt_raw_sha256="0" * 64,
            output_root="output",
            decision_exit_go=True,
        )
    assert source_reads["count"] == 0


def test_one_shot_failure_is_consumed_and_cannot_rerun(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config = build_g2c_calibration_config()
    source = _source_identity()
    freeze_root = tmp_path / "freeze"
    freeze_root.mkdir()
    _atomic_json(
        freeze_root / "prediction_freeze.json",
        {
            "source_git_commit": source["git_commit"],
            "source_identity_sha256": source["identity_sha256"],
        },
    )
    freeze = {
        "freeze_internal_sha256": "e" * 64,
        "freeze_raw_sha256": "d" * 64,
        "source_git_commit": source["git_commit"],
        "source_identity_sha256": source["identity_sha256"],
    }
    persistence = {
        "receipt_raw_sha256": "1" * 64,
        "receipt_internal_sha256": "2" * 64,
        "remote_identity_sha256": "3" * 64,
    }
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.load_g2c_calibration_config",
        lambda path: config,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._verify_training_config_reference",
        lambda *args, **kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.verify_g2c_calibration_prediction_freeze",
        lambda **kwargs: freeze,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._git_source_identity",
        lambda path: source,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.verify_g2c_calibration_phase_a_persistence",
        lambda **kwargs: persistence,
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration.validate_g2c_calibration_input_view",
        lambda **kwargs: {"verified": True},
    )
    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._read_jsonl",
        lambda *args, **kwargs: [{} for _ in range(550)],
    )

    def fail_after_first_open(root: Path, *, on_bundle_open: object) -> object:
        on_bundle_open(1)
        raise RuntimeError("injected label decode failure")

    monkeypatch.setattr(
        "robot_vla.precision.e018_p1_g2c_calibration._load_calibration_labels",
        fail_after_first_open,
    )
    output = tmp_path / "result"
    kwargs = {
        "calibration_config_path": "config",
        "training_config_path": "training",
        "prediction_freeze_root": freeze_root,
        "calibration_privileged_input_root": "labels",
        "repository_root": "repo",
        "phase_a_persistence_receipt_path": "persistence",
        "expected_phase_a_persistence_receipt_raw_sha256": "1" * 64,
        "output_root": output,
        "decision_exit_go": True,
    }
    with pytest.raises(RuntimeError, match="injected label decode failure"):
        score_calibrate_g2c(**kwargs)
    failure = json.loads((output / "consumed_failure.json").read_text())
    state = json.loads((output / "phase_state.json").read_text())
    assert failure["status"] == "consumed-calibration-failure"
    assert failure["label_bundle_open_count"] == 1
    assert failure["rerun_under_same_identity_allowed"] is False
    assert state["label_array_consumed"] is True
    assert state["status"] == "consumed-calibration-failure"
    with pytest.raises(FileExistsError):
        score_calibrate_g2c(**kwargs)
