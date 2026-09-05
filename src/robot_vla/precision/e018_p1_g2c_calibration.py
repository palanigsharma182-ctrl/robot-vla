"""E018-P1 G2C selected-provider 的两阶段 calibration 协议。

本模块把 deployable prediction freeze 与 privileged calibration 明确分开。
Phase A 只加载 D043 已冻结的 W-KV0 epoch-15 checkpoint；Phase B 在
``phase_state`` 持久化后一次性打开 50 个 calibration label bundles。公开
verifier 不接模型、checkpoint、DATA root 或 label path。
"""

from __future__ import annotations

import gc
import hashlib
import io
import math
import platform
import shutil
import time
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from robot_vla.precision.e018_p1_g2a import canonical_sha256, file_sha256
from robot_vla.precision.e018_p1_g2c import calibrate_g2c_viewpoint
from robot_vla.precision.e018_p1_g2c_data import (
    _LABEL_ARRAYS,
    G2C_LABEL_SCHEMA_VERSION,
    G2C_MANIFEST_SCHEMA_VERSION,
    G2C_VIEW_ORDER,
    G2CDeployableDataset,
    _atomic_json,
    _atomic_jsonl,
    _read_jsonl,
    _resolve_artifact_file,
)
from robot_vla.precision.e018_p1_g2c_model_val import (
    _prediction_rows_for_batch,
    verify_g2c_model_val_selection,
)
from robot_vla.precision.e018_p1_g2c_training import (
    D038_ACCEPTED_DATA,
    _assert_unlinked_regular_file_tree,
    _copy_input_bundle,
    _git_source_identity,
    _manifest_inventory,
    _read_json,
    _read_json_array,
    _require_exact_keys,
    _require_sha256,
    _verify_exact_regular_file_tree,
    load_g2c_formal_training_config,
    verify_g2c_formal_training,
)
from robot_vla.precision.object_observability import ObjectWriteEvidence
from robot_vla.precision.outliers import geometry_conditioning

E018_P1_G2C_CALIBRATION_CONFIG_VERSION = (
    "e018-p1-g2c-calibration-development/v1"
)
E018_P1_G2C_CALIBRATION_INPUT_VIEW_VERSION = (
    "e018-p1-g2c-calibration-input-view/v1"
)
E018_P1_G2C_CALIBRATION_FREEZE_VERSION = (
    "e018-p1-g2c-calibration-prediction-freeze/v1"
)
E018_P1_G2C_CALIBRATION_RESULT_VERSION = (
    "e018-p1-g2c-calibration-result/v1"
)
E018_P1_G2C_CALIBRATION_PERSISTENCE_VERSION = (
    "e018-p1-g2c-calibration-phase-a-drive-persistence/v1"
)
E018_P1_G2C_CALIBRATION_CHECK_EVIDENCE_VERSION = (
    "e018-p1-g2c-calibration-rclone-check-evidence/v1"
)
E018_P1_G2C_CALIBRATION_COMPLETION_MARKER_VERSION = (
    "e018-p1-g2c-calibration-phase-a-completion-marker/v1"
)

G2C_TRAINING_CONFIG_REFERENCE = {
    "raw_sha256": "e58bfd38ec27cde9c68af72a790474e137e0d5a7f6da8812b8f0156680ba7948",
    "internal_sha256": "6719acdfb95b1780bb6779ff48471bf78823ea062abb3f097d2564bcd0e203ab",
}

G2C_CALIBRATION_SELECTION_PARENT = {
    "selection_receipt_raw_sha256": (
        "a961d18d3bc02107385bd6d2feed461310fa7b39f93242ec0749ebaeee6f64bd"
    ),
    "selection_receipt_internal_sha256": (
        "f8874dc604ae8c727831ea21977b1913023a4df6944e76ec4c9f30c26aafb773"
    ),
    "selection_verification_sha256": (
        "1aee923b09fe62b1955dad5276b1a24fb19d2e122ddd638238d0f9830894e91a"
    ),
    "model_val_prediction_freeze_internal_sha256": (
        "e9afee15af889b769000c168c90b4f251d63e67da0e5ef2736ed50df38ea7ac2"
    ),
    "candidate_id": "W-KV0",
    "epoch": 15,
    "checkpoint_sha256": (
        "97e3b7289911bc73f67755a8d9c3598c50b6c80ef01e1af13cec698ec59d3d77"
    ),
    "parameter_state_sha256": (
        "1ba14a9009829c1d354555e9b788a8e3627e33ccffaddee24e66d9696121cb24"
    ),
    "provenance_sha256": (
        "8116f273c5f7339813a260bc919e25ee84cb3493a0e76eb454e6fbdeae83252c"
    ),
    "model_config_sha256": (
        "4a284a59c8c6d1865910d333597b565183f4d455af520e5edad5a79d5c67d053"
    ),
    "training_source_git_commit": "5bf05da5a22a07b8fabfc22b1f32da86fce40ba1",
    "training_source_identity_sha256": (
        "95b0fb26db8585decb9488ce0086ef1f9f6c8bc2a6496797e3d9681b89f2af05"
    ),
}

_PERMISSION_KEYS = {
    "test_array_reads",
    "memory_reads",
    "memory_writes",
    "runtime_camera_actuation",
    "physical_camera_actuation",
    "arm_motion_commands",
    "gripper_close_commands",
    "manipulation_progression",
    "checkpoint_writes",
}
_CALIBRATION_SEEDS = tuple(range(76601, 76651))
_INPUT_ROLES = {
    "calibration-deployable": (True, False),
    "calibration-privileged": (False, True),
}
_PREDICTION_COUNT_CONTRACT = {
    "selected_checkpoint_count": 1,
    "calibration_unique_deployable_bundle_count": 50,
    "calibration_deployable_bundle_open_count": 50,
    "calibration_prediction_row_count": 550,
    "model_forward_batch_count": 18,
    "privileged_label_open_count_before_freeze": 0,
}
_RESULT_COUNT_CONTRACT = {
    "calibration_prediction_row_count": 550,
    "calibration_scoring_row_count": 550,
    "calibration_viewpoint_count": 11,
    "calibration_privileged_label_bundle_open_count": 50,
}

_PREDICTION_ARTIFACTS = (
    "config_snapshot.json",
    "training_config_verification.json",
    "training_verification.json",
    "selection_verification.json",
    "source_identity.json",
    "deployable_input_verification.json",
    "selected_checkpoint_identity.json",
    "prediction_ledger.jsonl",
    "inference_audit.json",
)
_RESULT_ARTIFACTS = (
    "config_snapshot.json",
    "source_identity.json",
    "prediction_freeze_verification.json",
    "label_input_verification.json",
    "calibration_scoring_ledger.jsonl",
    "viewpoint_calibrations.json",
    "calibration_summary.json",
)

_PHASE_A_COMPLETION_MARKER_NAME = "DRIVE_BACKUP_COMPLETE.json"
_PHASE_A_PERSISTENCE_RECEIPT_NAME = "phase_a_persistence_receipt.json"
_PHASE_A_CHECK_EVIDENCE_NAMES = {
    "pre-marker-artifact-check": "pre_marker_artifact_check.json",
    "post-marker-artifact-check": "post_marker_artifact_check.json",
    "post-marker-completion-marker-check": "post_marker_completion_marker_check.json",
}
_PHASE_A_CHECK_REPORT_NAMES = {
    phase: name.removesuffix(".json") + ".combined"
    for phase, name in _PHASE_A_CHECK_EVIDENCE_NAMES.items()
}

_CALIBRATION_PREDICTION_ROW_KEYS = {
    "version",
    "phase",
    "candidate_id",
    "epoch",
    "checkpoint_sha256",
    "checkpoint_parameter_sha256",
    "checkpoint_provenance_sha256",
    "checkpoint_model_config_sha256",
    "row_index",
    "batch_index",
    "batch_offset",
    "seed",
    "split",
    "sample_index",
    "viewpoint_id",
    "input_sha256",
    "predicted_object_normalized_uv",
    "predicted_goal_normalized_uv",
    "object_visibility_probability",
    "goal_visibility_probability",
    "projection_validity_probability",
    "object_normalized_entropy",
    "object_sigma_xy_px",
    "object_mask_probability_at_prediction",
    "goal_mask_probability_at_prediction",
    "predicted_observable",
    "geometry_valid",
    "predicted_object_position_base_m",
    "raw_covariance_base_m2",
    "write_score",
    "external_intrinsic_cv",
    "base_from_external_camera_cv",
    "deployable_safety",
    "memory_write_allowed",
    "actuation_allowed",
}

_CALIBRATION_SCORING_ROW_KEYS = {
    "version",
    "phase",
    "prediction_freeze_sha256",
    "candidate_id",
    "epoch",
    "checkpoint_sha256",
    "row_index",
    "seed",
    "sample_index",
    "viewpoint_id",
    "gt_observable",
    "gt_object_position_base_m",
    "gt_object_exists",
    "is_grasped",
    "robot_object_contact_force_n",
    "predicted_observable",
    "object_write_structurally_eligible",
    "deployable_free_static_safe",
    "privileged_free_static_safe",
    "geometry_valid",
    "world_xyz_error_m",
    "world_xy_error_vector_m",
    "predicted_object_position_base_m",
    "raw_covariance_base_m2",
    "write_score",
    "structurally_eligible",
    "oracle_safe_measurement",
    "catastrophic_measurement",
    "test_data_read",
}


def _assert_exact_count_contract(
    value: Mapping[str, Any], *, expected: Mapping[str, int], name: str
) -> None:
    mismatches = {
        key: {"expected": expected_value, "actual": value.get(key)}
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise RuntimeError(f"{name} count contract 漂移: {mismatches}")


def g2c_calibration_protocol() -> dict[str, Any]:
    """返回 D044 冻结的 calibration-only 语义与资源边界。"""

    return {
        "version": E018_P1_G2C_CALIBRATION_CONFIG_VERSION,
        "split": "calibration",
        "seed_start": 76601,
        "seed_end": 76650,
        "seed_count": 50,
        "view_order": list(G2C_VIEW_ORDER),
        "sample_count": 550,
        "selected_checkpoint_count": 1,
        "batch_size": 32,
        "model_forward_batch_count": 18,
        "prediction_before_privileged_label": True,
        "privileged_staging_requires_phase_a_drive_zero_difference": True,
        "phase_a_persistence": {
            "artifact_copy_contract": "rclone-copy-immutable/v1",
            "completion_marker_copy_contract": "rclone-copyto-immutable/v1",
            "check_contract": "rclone-check-one-way-checksum-combined/v1",
            "pre_marker_artifact_matching_file_count": (
                len(_PREDICTION_ARTIFACTS) + 1
            ),
            "post_marker_artifact_matching_file_count": (
                len(_PREDICTION_ARTIFACTS) + 1
            ),
            "post_marker_completion_marker_matching_file_count": 1,
            "completion_marker_name": _PHASE_A_COMPLETION_MARKER_NAME,
            "final_receipt_name": _PHASE_A_PERSISTENCE_RECEIPT_NAME,
            "control_artifacts_outside_prediction_freeze_exact_tree": True,
            "final_receipt_after_both_post_checks": True,
        },
        "covariance": {
            "alpha": 0.05,
            "target_coverage": 0.95,
            "chi_square_threshold": 5.991,
            "minimum_support": 30,
            "maximum_calibrated_position_std_m": 0.020,
        },
        "write_threshold": {
            "maximum_oracle_safe_error_m": 0.005,
            "catastrophic_error_m": 0.020,
            "minimum_accepted_safe_coverage": 0.10,
            "unsafe_accepted_max": 0,
            "tie_break": "coverage-accepted-count-higher-threshold/v1",
            "coverage_denominator": "oracle_safe_count/v1",
        },
        "free_static_safety": {
            "maximum_finger_force_n": 0.01,
            "minimum_raw_gripper_opening_ratio": 0.95,
            "maximum_arm_joint_drift_rad": 1e-5,
            "maximum_tcp_position_drift_m": 1e-5,
            "maximum_tcp_orientation_drift_rad": 1e-4,
            "maximum_rgb_pose_skew_s": 0.01,
            "maximum_camera_position_tracking_error_m": 1e-5,
            "maximum_camera_orientation_tracking_error_rad": 1e-4,
            "maximum_rotation_projection_error_frobenius": 1e-6,
            "maximum_robot_object_contact_force_n": 0.01,
            "require_finger_force_valid": True,
            "require_not_grasped": True,
            "object_center_base_z_m": 0.02,
            "object_center_base_z_tolerance_m": 1e-5,
        },
        "zero_pass_policy": "protocol-valid-negative-no-best-effort-view/v1",
        "budgets": {
            "implementation_smoke_gpu_seconds_max": 600.0,
            "implementation_smoke_artifact_bytes_max": 1073741824,
            "formal_phase_a_gpu_seconds_max": 3600.0,
            "formal_artifact_bytes_max": 5368709120,
        },
    }


def build_g2c_calibration_config() -> dict[str, Any]:
    """构建不含执行 source commit 的 D044 calibration 配置。"""

    payload = {
        "version": E018_P1_G2C_CALIBRATION_CONFIG_VERSION,
        "status": "frozen-pre-formal-calibration-awaiting-source-r2-go/v1",
        "decision": {
            "model_selection": "D043",
            "runner_implementation": "D044",
            "formal_calibration_execution": "HOLD-until-exact-source-r2-go",
            "qualification_execution": "HOLD-until-separate-go",
        },
        "training_config": dict(G2C_TRAINING_CONFIG_REFERENCE),
        "data_parent": {
            **D038_ACCEPTED_DATA,
            "split": "calibration",
            "seed_count": 50,
            "sample_count": 550,
            "deployable_inventory_sha256": (
                "c2067d89e2cde7d57bada5723388eb29deb35757b62002f9dead2c1d2ed36516"
            ),
            "privileged_inventory_sha256": (
                "24205efa88c0bae2df6c51992696fbefc69ea4c24be8d3e82eda5591d2d653f2"
            ),
        },
        "selection_parent": dict(G2C_CALIBRATION_SELECTION_PARENT),
        "protocol": g2c_calibration_protocol(),
        "permissions": {name: 0 for name in sorted(_PERMISSION_KEYS)},
    }
    payload["config_sha256"] = canonical_sha256(payload)
    return payload


def load_g2c_calibration_config(path: str | Path) -> dict[str, Any]:
    config = _read_json(Path(path), "G2C calibration config")
    internal = config.get("config_sha256")
    unsigned = dict(config)
    unsigned.pop("config_sha256", None)
    _require_exact_keys(
        config,
        {
            "version",
            "status",
            "decision",
            "training_config",
            "data_parent",
            "selection_parent",
            "protocol",
            "permissions",
            "config_sha256",
        },
        "G2C calibration config",
    )
    expected_data = {
        **D038_ACCEPTED_DATA,
        "split": "calibration",
        "seed_count": 50,
        "sample_count": 550,
        "deployable_inventory_sha256": (
            "c2067d89e2cde7d57bada5723388eb29deb35757b62002f9dead2c1d2ed36516"
        ),
        "privileged_inventory_sha256": (
            "24205efa88c0bae2df6c51992696fbefc69ea4c24be8d3e82eda5591d2d653f2"
        ),
    }
    permissions = _require_exact_keys(
        config["permissions"], _PERMISSION_KEYS, "G2C calibration permissions"
    )
    if (
        internal != canonical_sha256(unsigned)
        or config["version"] != E018_P1_G2C_CALIBRATION_CONFIG_VERSION
        or config["status"]
        != "frozen-pre-formal-calibration-awaiting-source-r2-go/v1"
        or config["decision"]
        != {
            "model_selection": "D043",
            "runner_implementation": "D044",
            "formal_calibration_execution": "HOLD-until-exact-source-r2-go",
            "qualification_execution": "HOLD-until-separate-go",
        }
        or config["training_config"] != G2C_TRAINING_CONFIG_REFERENCE
        or config["data_parent"] != expected_data
        or config["selection_parent"] != G2C_CALIBRATION_SELECTION_PARENT
        or config["protocol"] != g2c_calibration_protocol()
        or any(type(value) is not int or value != 0 for value in permissions.values())
    ):
        raise RuntimeError("G2C calibration config protocol/identity 漂移")
    return config


def _verify_training_config_reference(
    calibration_config: Mapping[str, Any], training_config_path: str | Path
) -> dict[str, Any]:
    path = Path(training_config_path)
    training_config = load_g2c_formal_training_config(path)
    reference = calibration_config["training_config"]
    if (
        file_sha256(path) != reference["raw_sha256"]
        or training_config["config_sha256"] != reference["internal_sha256"]
    ):
        raise RuntimeError("G2C calibration TRAIN config parent 漂移")
    return {
        "raw_sha256": file_sha256(path),
        "internal_sha256": training_config["config_sha256"],
        "verified": True,
    }


def _input_manifest_rows(
    *,
    config: Mapping[str, Any],
    training_config: Mapping[str, Any],
    data_root: Path,
    include_deployable: bool,
    include_privileged: bool,
) -> tuple[list[dict[str, Any]] | None, list[dict[str, Any]] | None, dict[str, Any]]:
    data_parent = training_config["data_parent"]
    if file_sha256(data_root / "data_receipt.json") != data_parent[
        "data_receipt_raw_sha256"
    ]:
        raise RuntimeError("G2C calibration DATA receipt 漂移")
    deployable_manifest = data_root / "deployable" / "manifest.jsonl"
    privileged_manifest = data_root / "privileged_labels" / "manifest.jsonl"
    deployable = None
    privileged = None
    if include_deployable:
        if file_sha256(deployable_manifest) != data_parent[
            "deployable_manifest_raw_sha256"
        ]:
            raise RuntimeError("G2C calibration deployable source manifest 漂移")
        deployable_all = _read_jsonl(
            deployable_manifest, "G2C deployable manifest"
        )
        deployable = [
            row for row in deployable_all if row.get("split") == "calibration"
        ]
        if (
            len(deployable) != 50
            or [int(row["seed"]) for row in deployable]
            != list(_CALIBRATION_SEEDS)
        ):
            raise RuntimeError("G2C calibration deployable split identity 漂移")
    if include_privileged:
        if file_sha256(privileged_manifest) != data_parent[
            "privileged_manifest_raw_sha256"
        ]:
            raise RuntimeError("G2C calibration privileged source manifest 漂移")
        privileged_all = _read_jsonl(
            privileged_manifest, "G2C privileged manifest"
        )
        privileged = [
            row for row in privileged_all if row.get("split") == "calibration"
        ]
        if (
            len(privileged) != 50
            or [int(row["seed"]) for row in privileged]
            != list(_CALIBRATION_SEEDS)
        ):
            raise RuntimeError("G2C calibration privileged split identity 漂移")
    inventory = _manifest_inventory(
        deployable if include_deployable else None,
        privileged if include_privileged else None,
        split="calibration",
    )
    if (
        include_deployable
        and inventory["deployable_inventory_sha256"]
        != config["data_parent"]["deployable_inventory_sha256"]
    ) or (
        include_privileged
        and inventory["privileged_inventory_sha256"]
        != config["data_parent"]["privileged_inventory_sha256"]
    ):
        raise RuntimeError("G2C calibration split inventory 漂移")
    return (
        deployable if include_deployable else None,
        privileged if include_privileged else None,
        inventory,
    )


def _prepare_calibration_input_view(
    *,
    calibration_config_path: str | Path,
    training_config_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    role: str,
    prediction_freeze_internal_sha256: str | None = None,
    source_identity_sha256: str | None = None,
    persistence_receipt_raw_sha256: str | None = None,
    persistence_receipt_internal_sha256: str | None = None,
    remote_identity_sha256: str | None = None,
) -> dict[str, Any]:
    if role not in _INPUT_ROLES:
        raise ValueError("G2C calibration input role 未冻结")
    include_deployable, include_privileged = _INPUT_ROLES[role]
    config = load_g2c_calibration_config(calibration_config_path)
    training_verification = _verify_training_config_reference(
        config, training_config_path
    )
    training_config = load_g2c_formal_training_config(training_config_path)
    if include_privileged:
        _require_sha256(
            prediction_freeze_internal_sha256,
            "G2C calibration freeze binding",
        )
        _require_sha256(source_identity_sha256, "G2C calibration source binding")
        _require_sha256(
            persistence_receipt_raw_sha256,
            "G2C calibration persistence raw binding",
        )
        _require_sha256(
            persistence_receipt_internal_sha256,
            "G2C calibration persistence internal binding",
        )
        _require_sha256(remote_identity_sha256, "G2C calibration remote binding")
    elif any(
        value is not None
        for value in (
            prediction_freeze_internal_sha256,
            source_identity_sha256,
            persistence_receipt_raw_sha256,
            persistence_receipt_internal_sha256,
            remote_identity_sha256,
        )
    ):
        raise ValueError("G2C calibration deployable view 禁止 privileged binding")
    source = Path(data_root)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"G2C calibration input view 已存在: {output}")
    deployable, privileged, inventory = _input_manifest_rows(
        config=config,
        training_config=training_config,
        data_root=source,
        include_deployable=include_deployable,
        include_privileged=include_privileged,
    )
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    copied: list[dict[str, Any]] = []
    if deployable is not None:
        _atomic_jsonl(output / "deployable" / "manifest.jsonl", deployable)
        for row in deployable:
            target = output / "deployable" / str(row["file"])
            _copy_input_bundle(
                _resolve_artifact_file(source / "deployable", str(row["file"])),
                target,
                expected_sha256=str(row["sha256"]),
            )
            copied.append(
                {"role": "deployable", "file": str(row["file"]), "sha256": row["sha256"]}
            )
    if privileged is not None:
        _atomic_jsonl(
            output / "privileged_labels" / "manifest.jsonl", privileged
        )
        for row in privileged:
            target = output / "privileged_labels" / str(row["file"])
            _copy_input_bundle(
                _resolve_artifact_file(
                    source / "privileged_labels", str(row["file"])
                ),
                target,
                expected_sha256=str(row["sha256"]),
            )
            copied.append(
                {"role": "privileged", "file": str(row["file"]), "sha256": row["sha256"]}
            )
    receipt = {
        "version": E018_P1_G2C_CALIBRATION_INPUT_VIEW_VERSION,
        "status": "complete-calibration-input-view-pass",
        "role": role,
        "split": "calibration",
        "config_sha256": config["config_sha256"],
        "training_config_internal_sha256": training_verification["internal_sha256"],
        "data_identity_sha256": config["data_parent"]["data_identity_sha256"],
        "selected_checkpoint_sha256": config["selection_parent"]["checkpoint_sha256"],
        "seed_count": 50,
        "sample_count": 550,
        "deployable_included": include_deployable,
        "privileged_included": include_privileged,
        "inventory": inventory,
        "copied_file_inventory_sha256": canonical_sha256(copied),
        "prediction_freeze_internal_sha256": prediction_freeze_internal_sha256,
        "source_identity_sha256": source_identity_sha256,
        "phase_a_persistence_receipt_raw_sha256": persistence_receipt_raw_sha256,
        "phase_a_persistence_receipt_internal_sha256": (
            persistence_receipt_internal_sha256
        ),
        "phase_a_remote_identity_sha256": remote_identity_sha256,
        "privileged_source_bundle_copy_count": 50 if include_privileged else 0,
        "privileged_label_array_open_count": 0,
        "test_array_read_count": 0,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(output / "input_view_receipt.json", receipt)
    return validate_g2c_calibration_input_view(
        calibration_config_path=calibration_config_path,
        training_config_path=training_config_path,
        input_root=output,
        expected_role=role,
        expected_prediction_freeze_internal_sha256=prediction_freeze_internal_sha256,
        expected_source_identity_sha256=source_identity_sha256,
        expected_persistence_receipt_raw_sha256=persistence_receipt_raw_sha256,
        expected_persistence_receipt_internal_sha256=(
            persistence_receipt_internal_sha256
        ),
        expected_remote_identity_sha256=remote_identity_sha256,
    )


def validate_g2c_calibration_input_view(
    *,
    calibration_config_path: str | Path,
    training_config_path: str | Path,
    input_root: str | Path,
    expected_role: str,
    verify_bundle_bytes: bool = True,
    expected_prediction_freeze_internal_sha256: str | None = None,
    expected_source_identity_sha256: str | None = None,
    expected_persistence_receipt_raw_sha256: str | None = None,
    expected_persistence_receipt_internal_sha256: str | None = None,
    expected_remote_identity_sha256: str | None = None,
) -> dict[str, Any]:
    if expected_role not in _INPUT_ROLES:
        raise ValueError("G2C calibration expected role 未冻结")
    include_deployable, include_privileged = _INPUT_ROLES[expected_role]
    config = load_g2c_calibration_config(calibration_config_path)
    training_verification = _verify_training_config_reference(
        config, training_config_path
    )
    root = Path(input_root)
    _assert_unlinked_regular_file_tree(root, name="G2C calibration input view")
    receipt_path = root / "input_view_receipt.json"
    receipt = _read_json(receipt_path, "G2C calibration input receipt")
    internal = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    expected_freeze = (
        _require_sha256(
            expected_prediction_freeze_internal_sha256,
            "G2C calibration expected freeze",
        )
        if include_privileged
        else None
    )
    expected_source = (
        _require_sha256(
            expected_source_identity_sha256,
            "G2C calibration expected source",
        )
        if include_privileged
        else None
    )
    expected_persistence_raw = (
        _require_sha256(
            expected_persistence_receipt_raw_sha256,
            "G2C calibration expected persistence receipt raw",
        )
        if include_privileged
        else None
    )
    expected_persistence_internal = (
        _require_sha256(
            expected_persistence_receipt_internal_sha256,
            "G2C calibration expected persistence receipt internal",
        )
        if include_privileged
        else None
    )
    expected_remote_identity = (
        _require_sha256(
            expected_remote_identity_sha256,
            "G2C calibration expected remote identity",
        )
        if include_privileged
        else None
    )
    if (
        internal != canonical_sha256(unsigned)
        or receipt.get("version") != E018_P1_G2C_CALIBRATION_INPUT_VIEW_VERSION
        or receipt.get("status") != "complete-calibration-input-view-pass"
        or receipt.get("role") != expected_role
        or receipt.get("split") != "calibration"
        or receipt.get("config_sha256") != config["config_sha256"]
        or receipt.get("training_config_internal_sha256")
        != training_verification["internal_sha256"]
        or receipt.get("data_identity_sha256")
        != config["data_parent"]["data_identity_sha256"]
        or receipt.get("selected_checkpoint_sha256")
        != config["selection_parent"]["checkpoint_sha256"]
        or receipt.get("seed_count") != 50
        or receipt.get("sample_count") != 550
        or receipt.get("deployable_included") is not include_deployable
        or receipt.get("privileged_included") is not include_privileged
        or receipt.get("prediction_freeze_internal_sha256") != expected_freeze
        or receipt.get("source_identity_sha256") != expected_source
        or receipt.get("phase_a_persistence_receipt_raw_sha256")
        != expected_persistence_raw
        or receipt.get("phase_a_persistence_receipt_internal_sha256")
        != expected_persistence_internal
        or receipt.get("phase_a_remote_identity_sha256")
        != expected_remote_identity
        or receipt.get("privileged_source_bundle_copy_count")
        != (50 if include_privileged else 0)
        or receipt.get("privileged_label_array_open_count") != 0
        or receipt.get("test_array_read_count") != 0
    ):
        raise RuntimeError("G2C calibration input receipt identity/role 漂移")
    manifest_root = root / ("deployable" if include_deployable else "privileged_labels")
    manifest = _read_jsonl(
        manifest_root / "manifest.jsonl", "G2C calibration input manifest"
    )
    inventory = _manifest_inventory(
        manifest if include_deployable else None,
        manifest if include_privileged else None,
        split="calibration",
        deployable_root=manifest_root if include_deployable and verify_bundle_bytes else None,
        label_root=manifest_root if include_privileged and verify_bundle_bytes else None,
    )
    if inventory != receipt.get("inventory"):
        raise RuntimeError("G2C calibration input inventory/receipt 漂移")
    expected_inventory_name = (
        "deployable_inventory_sha256"
        if include_deployable
        else "privileged_inventory_sha256"
    )
    if inventory[expected_inventory_name] != config["data_parent"][expected_inventory_name]:
        raise RuntimeError("G2C calibration input inventory identity 漂移")
    copied = [
        {
            "role": "deployable" if include_deployable else "privileged",
            "file": str(row["file"]),
            "sha256": str(row["sha256"]),
        }
        for row in manifest
    ]
    if receipt.get("copied_file_inventory_sha256") != canonical_sha256(copied):
        raise RuntimeError("G2C calibration copied inventory 漂移")
    expected_files = {
        "input_view_receipt.json",
        f"{manifest_root.name}/manifest.jsonl",
        *(f"{manifest_root.name}/{row['file']}" for row in manifest),
    }
    _verify_exact_regular_file_tree(
        root, expected_files=expected_files, name="G2C calibration input view"
    )
    result = {
        "version": E018_P1_G2C_CALIBRATION_INPUT_VIEW_VERSION,
        "verified": True,
        "role": expected_role,
        "split": "calibration",
        "seed_count": 50,
        "sample_count": 550,
        "bundle_bytes_verified": verify_bundle_bytes,
        "prediction_freeze_internal_sha256": expected_freeze,
        "source_identity_sha256": expected_source,
        "phase_a_persistence_receipt_raw_sha256": expected_persistence_raw,
        "phase_a_persistence_receipt_internal_sha256": (
            expected_persistence_internal
        ),
        "phase_a_remote_identity_sha256": expected_remote_identity,
        "privileged_source_bundle_copy_count": 50 if include_privileged else 0,
        "privileged_label_array_open_count": 0,
        "receipt_raw_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": internal,
        "inventory": inventory,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def prepare_g2c_calibration_deployable_view(
    *,
    calibration_config_path: str | Path,
    training_config_path: str | Path,
    data_root: str | Path,
    output_root: str | Path,
    decision_exit_go: bool,
) -> dict[str, Any]:
    if decision_exit_go is not True:
        raise PermissionError("G2C calibration deployable staging 仍为 HOLD")
    return _prepare_calibration_input_view(
        calibration_config_path=calibration_config_path,
        training_config_path=training_config_path,
        data_root=data_root,
        output_root=output_root,
        role="calibration-deployable",
    )


def _phase_a_remote_identity(
    *, artifact_id: str, worker_id: str, remote_path: str
) -> dict[str, str]:
    for name, value in (("artifact_id", artifact_id), ("worker_id", worker_id)):
        if (
            not isinstance(value, str)
            or not value
            or "/" in value
            or "\\" in value
            or value in {".", ".."}
        ):
            raise ValueError(f"G2C calibration {name} 非法")
    prefix = "gdrive:VLA/experiments/e018-p1-g2c-calibration-phase-a/"
    expected_remote = f"{prefix}{artifact_id}/{worker_id}"
    if remote_path != expected_remote:
        raise ValueError("G2C calibration Phase A remote path/identity 漂移")
    return {
        "artifact_id": artifact_id,
        "worker_id": worker_id,
        "remote_path": remote_path,
    }


def _assert_control_path_outside_freeze(
    path: Path, *, prediction_freeze_root: Path
) -> None:
    freeze = prediction_freeze_root.resolve()
    candidate = path.resolve(strict=False)
    if candidate == freeze or freeze in candidate.parents:
        raise RuntimeError(
            "G2C calibration completion marker/receipt/check evidence 禁止进入 Phase A exact-tree"
        )


def _parse_rclone_combined_report(
    report_path: Path, *, expected_paths: Sequence[str]
) -> dict[str, Any]:
    if (
        report_path.is_symlink()
        or not report_path.is_file()
        or report_path.stat().st_nlink != 1
    ):
        raise RuntimeError("G2C calibration rclone combined report link/type 漂移")
    raw = report_path.read_bytes()
    try:
        lines = raw.decode("utf-8").splitlines()
    except UnicodeDecodeError as error:
        raise RuntimeError("G2C calibration rclone combined report 必须是 UTF-8") from error
    parsed: list[tuple[str, str]] = []
    for line in lines:
        if len(line) < 3 or line[0] not in "=+-*!" or line[1] != " ":
            raise RuntimeError("G2C calibration rclone combined report 行格式漂移")
        relative = line[2:]
        candidate = Path(relative)
        if (
            not relative
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.as_posix() != relative
        ):
            raise RuntimeError("G2C calibration rclone combined report 路径非法")
        parsed.append((line[0], relative))
    expected = list(expected_paths)
    parsed_paths = [relative for status, relative in parsed if status == "="]
    if (
        len(parsed) != len(expected)
        or len(parsed_paths) != len(parsed)
        or len(set(parsed_paths)) != len(parsed_paths)
        or set(parsed_paths) != set(expected)
    ):
        raise RuntimeError(
            "G2C calibration rclone combined report 必须逐文件全等且无缺失/重复"
        )
    canonical_paths = sorted(expected)
    return {
        "combined_report_raw_sha256": hashlib.sha256(raw).hexdigest(),
        "combined_report_size_bytes": len(raw),
        "matching_path_inventory_sha256": canonical_sha256(canonical_paths),
        "matching_file_count": len(expected),
        "difference_count": 0,
        "error_count": 0,
        "source_only_count": 0,
        "destination_only_count": 0,
    }


def _phase_a_check_source(
    *,
    phase: str,
    freeze: Mapping[str, Any],
    prediction_freeze_root: Path,
    completion_marker_path: Path | None,
) -> tuple[list[str], int, str, str]:
    if phase in {"pre-marker-artifact-check", "post-marker-artifact-check"}:
        if (prediction_freeze_root / _PHASE_A_COMPLETION_MARKER_NAME).exists():
            raise RuntimeError("G2C calibration completion marker 禁止进入 Phase A exact-tree")
        paths = [*_PREDICTION_ARTIFACTS, "prediction_freeze.json"]
        return (
            paths,
            int(freeze["artifact_bytes"]),
            str(freeze["freeze_internal_sha256"]),
            "phase-a-prediction-freeze-exact-tree/v1",
        )
    if phase != "post-marker-completion-marker-check":
        raise ValueError("G2C calibration rclone check phase 未冻结")
    if completion_marker_path is None:
        raise ValueError("G2C calibration marker-only check 缺 completion marker")
    marker_path = Path(completion_marker_path)
    _assert_control_path_outside_freeze(
        marker_path, prediction_freeze_root=prediction_freeze_root
    )
    if (
        marker_path.name != _PHASE_A_COMPLETION_MARKER_NAME
        or marker_path.is_symlink()
        or not marker_path.is_file()
        or marker_path.stat().st_nlink != 1
    ):
        raise RuntimeError("G2C calibration completion marker file 漂移")
    marker = _read_json(marker_path, "G2C calibration completion marker")
    return (
        [_PHASE_A_COMPLETION_MARKER_NAME],
        marker_path.stat().st_size,
        str(marker.get("marker_sha256")),
        "phase-a-completion-marker-include-only/v1",
    )


def record_g2c_calibration_phase_a_check_evidence(
    *,
    calibration_config_path: str | Path,
    prediction_freeze_root: str | Path,
    phase: str,
    artifact_id: str,
    worker_id: str,
    remote_path: str,
    combined_report_path: str | Path,
    rclone_exit_code: int,
    output_path: str | Path,
    completion_marker_path: str | Path | None = None,
    decision_exit_go: bool,
) -> dict[str, Any]:
    """把 ``rclone check --one-way --checksum --combined`` 原始报告冻结为证据。"""

    if decision_exit_go is not True:
        raise PermissionError("G2C formal calibration persistence 仍为 HOLD")
    if type(rclone_exit_code) is not int or rclone_exit_code != 0:
        raise RuntimeError("G2C calibration rclone check exit code 非零")
    output = Path(output_path)
    expected_name = _PHASE_A_CHECK_EVIDENCE_NAMES.get(phase)
    if expected_name is None or output.name != expected_name:
        raise ValueError("G2C calibration check evidence filename/phase 漂移")
    freeze_root = Path(prediction_freeze_root)
    _assert_control_path_outside_freeze(output, prediction_freeze_root=freeze_root)
    if output.exists():
        raise FileExistsError(f"G2C calibration check evidence 已存在: {output}")
    config = load_g2c_calibration_config(calibration_config_path)
    freeze = verify_g2c_calibration_prediction_freeze(
        calibration_config_path=calibration_config_path,
        output_root=freeze_root,
    )
    remote = _phase_a_remote_identity(
        artifact_id=artifact_id, worker_id=worker_id, remote_path=remote_path
    )
    marker_path = None if completion_marker_path is None else Path(completion_marker_path)
    paths, matching_bytes, source_sha, source_contract = _phase_a_check_source(
        phase=phase,
        freeze=freeze,
        prediction_freeze_root=freeze_root,
        completion_marker_path=marker_path,
    )
    parsed = _parse_rclone_combined_report(
        Path(combined_report_path), expected_paths=paths
    )
    report_target = output.parent / _PHASE_A_CHECK_REPORT_NAMES[phase]
    _assert_control_path_outside_freeze(
        report_target, prediction_freeze_root=freeze_root
    )
    if report_target.exists():
        raise FileExistsError(f"G2C calibration combined report 已存在: {report_target}")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    shutil.copyfile(combined_report_path, report_target)
    if file_sha256(report_target) != parsed["combined_report_raw_sha256"]:
        raise RuntimeError("G2C calibration combined report copy 漂移")
    evidence = {
        "version": E018_P1_G2C_CALIBRATION_CHECK_EVIDENCE_VERSION,
        "status": "complete-rclone-check-pass",
        "phase": phase,
        "command_contract": "rclone-check-one-way-checksum-combined/v1",
        "source_contract": source_contract,
        **remote,
        "remote_identity_sha256": canonical_sha256(remote),
        "config_sha256": config["config_sha256"],
        "prediction_freeze_internal_sha256": freeze["freeze_internal_sha256"],
        "checked_source_internal_sha256": source_sha,
        "combined_report_file": report_target.name,
        **parsed,
        "matching_bytes": matching_bytes,
        "rclone_exit_code": rclone_exit_code,
        "one_way": True,
        "checksum": True,
        "recorded_at_unix_ns": time.time_ns(),
    }
    evidence["evidence_sha256"] = canonical_sha256(evidence)
    _atomic_json(output, evidence)
    return evidence


def _verify_phase_a_check_evidence(
    *,
    evidence_path: Path,
    expected_phase: str,
    config: Mapping[str, Any],
    freeze: Mapping[str, Any],
    prediction_freeze_root: Path,
    remote: Mapping[str, str],
    completion_marker_path: Path | None = None,
) -> dict[str, Any]:
    expected_name = _PHASE_A_CHECK_EVIDENCE_NAMES[expected_phase]
    if (
        evidence_path.name != expected_name
        or evidence_path.is_symlink()
        or not evidence_path.is_file()
        or evidence_path.stat().st_nlink != 1
    ):
        raise RuntimeError("G2C calibration check evidence file/link 漂移")
    value = _read_json(evidence_path, "G2C calibration rclone check evidence")
    keys = {
        "version",
        "status",
        "phase",
        "command_contract",
        "source_contract",
        "artifact_id",
        "worker_id",
        "remote_path",
        "remote_identity_sha256",
        "config_sha256",
        "prediction_freeze_internal_sha256",
        "checked_source_internal_sha256",
        "combined_report_file",
        "combined_report_raw_sha256",
        "combined_report_size_bytes",
        "matching_path_inventory_sha256",
        "matching_file_count",
        "matching_bytes",
        "difference_count",
        "error_count",
        "source_only_count",
        "destination_only_count",
        "rclone_exit_code",
        "one_way",
        "checksum",
        "recorded_at_unix_ns",
        "evidence_sha256",
    }
    _require_exact_keys(value, keys, "G2C calibration check evidence")
    internal = value.get("evidence_sha256")
    unsigned = dict(value)
    unsigned.pop("evidence_sha256", None)
    paths, matching_bytes, source_sha, source_contract = _phase_a_check_source(
        phase=expected_phase,
        freeze=freeze,
        prediction_freeze_root=prediction_freeze_root,
        completion_marker_path=completion_marker_path,
    )
    report_path = evidence_path.parent / _PHASE_A_CHECK_REPORT_NAMES[expected_phase]
    parsed = _parse_rclone_combined_report(report_path, expected_paths=paths)
    if (
        internal != canonical_sha256(unsigned)
        or value.get("version") != E018_P1_G2C_CALIBRATION_CHECK_EVIDENCE_VERSION
        or value.get("status") != "complete-rclone-check-pass"
        or value.get("phase") != expected_phase
        or value.get("command_contract")
        != "rclone-check-one-way-checksum-combined/v1"
        or value.get("source_contract") != source_contract
        or {name: value.get(name) for name in remote} != dict(remote)
        or value.get("remote_identity_sha256") != canonical_sha256(remote)
        or value.get("config_sha256") != config["config_sha256"]
        or value.get("prediction_freeze_internal_sha256")
        != freeze["freeze_internal_sha256"]
        or value.get("checked_source_internal_sha256") != source_sha
        or value.get("combined_report_file") != report_path.name
        or any(value.get(name) != parsed[name] for name in parsed)
        or value.get("matching_bytes") != matching_bytes
        or value.get("rclone_exit_code") != 0
        or value.get("one_way") is not True
        or value.get("checksum") is not True
        or not isinstance(value.get("recorded_at_unix_ns"), int)
        or value["recorded_at_unix_ns"] <= 0
    ):
        raise RuntimeError("G2C calibration rclone check evidence identity/count 漂移")
    return {
        **value,
        "evidence_raw_sha256": file_sha256(evidence_path),
        "evidence_internal_sha256": internal,
        "combined_report_path": report_path,
    }


def build_g2c_calibration_phase_a_completion_marker(
    *,
    calibration_config_path: str | Path,
    prediction_freeze_root: str | Path,
    pre_marker_check_evidence_path: str | Path,
    artifact_id: str,
    worker_id: str,
    remote_path: str,
    output_path: str | Path,
    decision_exit_go: bool,
) -> dict[str, Any]:
    """在 artifact pre-check 通过后构建独立、非自指的 Drive marker。"""

    if decision_exit_go is not True:
        raise PermissionError("G2C formal calibration completion marker 仍为 HOLD")
    freeze_root = Path(prediction_freeze_root)
    output = Path(output_path)
    _assert_control_path_outside_freeze(output, prediction_freeze_root=freeze_root)
    if output.name != _PHASE_A_COMPLETION_MARKER_NAME:
        raise ValueError("G2C calibration completion marker filename 漂移")
    if output.exists():
        raise FileExistsError(f"G2C calibration completion marker 已存在: {output}")
    config = load_g2c_calibration_config(calibration_config_path)
    freeze = verify_g2c_calibration_prediction_freeze(
        calibration_config_path=calibration_config_path, output_root=freeze_root
    )
    freeze_marker = _read_json(
        freeze_root / "prediction_freeze.json", "G2C calibration prediction freeze"
    )
    remote = _phase_a_remote_identity(
        artifact_id=artifact_id, worker_id=worker_id, remote_path=remote_path
    )
    pre_path = Path(pre_marker_check_evidence_path)
    if pre_path.parent.resolve() != output.parent.resolve(strict=False):
        raise RuntimeError("G2C calibration pre-check evidence 与 control root 分离")
    pre = _verify_phase_a_check_evidence(
        evidence_path=pre_path,
        expected_phase="pre-marker-artifact-check",
        config=config,
        freeze=freeze,
        prediction_freeze_root=freeze_root,
        remote=remote,
    )
    created_at = time.time_ns()
    if created_at < pre["recorded_at_unix_ns"]:
        raise RuntimeError("G2C calibration completion marker 早于 pre-check")
    marker = {
        "version": E018_P1_G2C_CALIBRATION_COMPLETION_MARKER_VERSION,
        "status": "phase-a-completion-marker-ready-for-copy-check",
        "completion_marker_name": _PHASE_A_COMPLETION_MARKER_NAME,
        **remote,
        "remote_identity_sha256": canonical_sha256(remote),
        "config_sha256": config["config_sha256"],
        "prediction_freeze_raw_sha256": freeze["freeze_raw_sha256"],
        "prediction_freeze_internal_sha256": freeze["freeze_internal_sha256"],
        "source_git_commit": freeze["source_git_commit"],
        "source_identity_sha256": freeze["source_identity_sha256"],
        "artifact_inventory_sha256": freeze_marker["artifact_inventory_sha256"],
        "artifact_file_count": len(_PREDICTION_ARTIFACTS) + 1,
        "artifact_bytes": freeze["artifact_bytes"],
        "artifact_copy_contract": "rclone-copy-immutable/v1",
        "completion_marker_copy_contract": "rclone-copyto-immutable/v1",
        "pre_marker_check_evidence_raw_sha256": pre["evidence_raw_sha256"],
        "pre_marker_check_evidence_internal_sha256": pre[
            "evidence_internal_sha256"
        ],
        "created_at_unix_ns": created_at,
    }
    marker["marker_sha256"] = canonical_sha256(marker)
    _atomic_json(output, marker)
    return marker


def _verify_phase_a_completion_marker(
    *,
    marker_path: Path,
    config: Mapping[str, Any],
    freeze: Mapping[str, Any],
    prediction_freeze_root: Path,
    remote: Mapping[str, str],
    pre: Mapping[str, Any],
) -> dict[str, Any]:
    _assert_control_path_outside_freeze(
        marker_path, prediction_freeze_root=prediction_freeze_root
    )
    if (
        marker_path.name != _PHASE_A_COMPLETION_MARKER_NAME
        or marker_path.is_symlink()
        or not marker_path.is_file()
        or marker_path.stat().st_nlink != 1
    ):
        raise RuntimeError("G2C calibration completion marker file/link 漂移")
    freeze_marker = _read_json(
        prediction_freeze_root / "prediction_freeze.json",
        "G2C calibration prediction freeze",
    )
    marker = _read_json(marker_path, "G2C calibration completion marker")
    keys = {
        "version",
        "status",
        "completion_marker_name",
        "artifact_id",
        "worker_id",
        "remote_path",
        "remote_identity_sha256",
        "config_sha256",
        "prediction_freeze_raw_sha256",
        "prediction_freeze_internal_sha256",
        "source_git_commit",
        "source_identity_sha256",
        "artifact_inventory_sha256",
        "artifact_file_count",
        "artifact_bytes",
        "artifact_copy_contract",
        "completion_marker_copy_contract",
        "pre_marker_check_evidence_raw_sha256",
        "pre_marker_check_evidence_internal_sha256",
        "created_at_unix_ns",
        "marker_sha256",
    }
    _require_exact_keys(marker, keys, "G2C calibration completion marker")
    internal = marker.get("marker_sha256")
    unsigned = dict(marker)
    unsigned.pop("marker_sha256", None)
    if (
        internal != canonical_sha256(unsigned)
        or marker.get("version")
        != E018_P1_G2C_CALIBRATION_COMPLETION_MARKER_VERSION
        or marker.get("status")
        != "phase-a-completion-marker-ready-for-copy-check"
        or marker.get("completion_marker_name") != _PHASE_A_COMPLETION_MARKER_NAME
        or {name: marker.get(name) for name in remote} != dict(remote)
        or marker.get("remote_identity_sha256") != canonical_sha256(remote)
        or marker.get("config_sha256") != config["config_sha256"]
        or marker.get("prediction_freeze_raw_sha256")
        != freeze["freeze_raw_sha256"]
        or marker.get("prediction_freeze_internal_sha256")
        != freeze["freeze_internal_sha256"]
        or marker.get("source_git_commit") != freeze["source_git_commit"]
        or marker.get("source_identity_sha256") != freeze["source_identity_sha256"]
        or marker.get("artifact_inventory_sha256")
        != freeze_marker["artifact_inventory_sha256"]
        or marker.get("artifact_file_count") != len(_PREDICTION_ARTIFACTS) + 1
        or marker.get("artifact_bytes") != freeze["artifact_bytes"]
        or marker.get("artifact_copy_contract") != "rclone-copy-immutable/v1"
        or marker.get("completion_marker_copy_contract")
        != "rclone-copyto-immutable/v1"
        or marker.get("pre_marker_check_evidence_raw_sha256")
        != pre["evidence_raw_sha256"]
        or marker.get("pre_marker_check_evidence_internal_sha256")
        != pre["evidence_internal_sha256"]
        or not isinstance(marker.get("created_at_unix_ns"), int)
        or marker["created_at_unix_ns"] < pre["recorded_at_unix_ns"]
    ):
        raise RuntimeError("G2C calibration completion marker identity/order 漂移")
    return {
        **marker,
        "marker_raw_sha256": file_sha256(marker_path),
        "marker_internal_sha256": internal,
        "marker_size_bytes": marker_path.stat().st_size,
    }


def finalize_g2c_calibration_phase_a_persistence(
    *,
    calibration_config_path: str | Path,
    prediction_freeze_root: str | Path,
    control_root: str | Path,
    output_path: str | Path,
    decision_exit_go: bool,
) -> dict[str, Any]:
    """在 artifact/marker 两项 post-check 均落盘后生成最终 persistence receipt。"""

    if decision_exit_go is not True:
        raise PermissionError("G2C formal calibration persistence finalize 仍为 HOLD")
    freeze_root = Path(prediction_freeze_root)
    root = Path(control_root)
    output = Path(output_path)
    _assert_control_path_outside_freeze(root, prediction_freeze_root=freeze_root)
    if output.parent.resolve(strict=False) != root.resolve() or output.name != (
        _PHASE_A_PERSISTENCE_RECEIPT_NAME
    ):
        raise ValueError("G2C calibration final persistence receipt path 漂移")
    if output.exists():
        raise FileExistsError(f"G2C calibration persistence receipt 已存在: {output}")
    config = load_g2c_calibration_config(calibration_config_path)
    freeze = verify_g2c_calibration_prediction_freeze(
        calibration_config_path=calibration_config_path, output_root=freeze_root
    )
    marker_path = root / _PHASE_A_COMPLETION_MARKER_NAME
    marker_preview = _read_json(marker_path, "G2C calibration completion marker")
    remote = _phase_a_remote_identity(
        artifact_id=str(marker_preview.get("artifact_id")),
        worker_id=str(marker_preview.get("worker_id")),
        remote_path=str(marker_preview.get("remote_path")),
    )
    pre_path = root / _PHASE_A_CHECK_EVIDENCE_NAMES["pre-marker-artifact-check"]
    pre = _verify_phase_a_check_evidence(
        evidence_path=pre_path,
        expected_phase="pre-marker-artifact-check",
        config=config,
        freeze=freeze,
        prediction_freeze_root=freeze_root,
        remote=remote,
    )
    marker = _verify_phase_a_completion_marker(
        marker_path=marker_path,
        config=config,
        freeze=freeze,
        prediction_freeze_root=freeze_root,
        remote=remote,
        pre=pre,
    )
    post_artifact = _verify_phase_a_check_evidence(
        evidence_path=root
        / _PHASE_A_CHECK_EVIDENCE_NAMES["post-marker-artifact-check"],
        expected_phase="post-marker-artifact-check",
        config=config,
        freeze=freeze,
        prediction_freeze_root=freeze_root,
        remote=remote,
    )
    marker_check = _verify_phase_a_check_evidence(
        evidence_path=root
        / _PHASE_A_CHECK_EVIDENCE_NAMES[
            "post-marker-completion-marker-check"
        ],
        expected_phase="post-marker-completion-marker-check",
        config=config,
        freeze=freeze,
        prediction_freeze_root=freeze_root,
        remote=remote,
        completion_marker_path=marker_path,
    )
    created_at = time.time_ns()
    if (
        post_artifact["recorded_at_unix_ns"] < marker["created_at_unix_ns"]
        or marker_check["recorded_at_unix_ns"] < marker["created_at_unix_ns"]
        or created_at
        < max(
            post_artifact["recorded_at_unix_ns"],
            marker_check["recorded_at_unix_ns"],
        )
    ):
        raise RuntimeError("G2C calibration final receipt 早于 marker/post-check")
    expected_before_receipt = {
        _PHASE_A_COMPLETION_MARKER_NAME,
        *_PHASE_A_CHECK_EVIDENCE_NAMES.values(),
        *_PHASE_A_CHECK_REPORT_NAMES.values(),
    }
    _assert_unlinked_regular_file_tree(root, name="G2C Phase A persistence control")
    _verify_exact_regular_file_tree(
        root,
        expected_files=set(expected_before_receipt),
        name="G2C Phase A persistence pre-receipt control",
    )
    receipt = {
        "version": E018_P1_G2C_CALIBRATION_PERSISTENCE_VERSION,
        "status": "DRIVE_VERIFIED",
        "completion_marker_name": _PHASE_A_COMPLETION_MARKER_NAME,
        **remote,
        "remote_identity_sha256": canonical_sha256(remote),
        "config_sha256": config["config_sha256"],
        "prediction_freeze_raw_sha256": freeze["freeze_raw_sha256"],
        "prediction_freeze_internal_sha256": freeze["freeze_internal_sha256"],
        "source_git_commit": freeze["source_git_commit"],
        "source_identity_sha256": freeze["source_identity_sha256"],
        "artifact_file_count": len(_PREDICTION_ARTIFACTS) + 1,
        "artifact_bytes": freeze["artifact_bytes"],
        "completion_marker_raw_sha256": marker["marker_raw_sha256"],
        "completion_marker_internal_sha256": marker["marker_internal_sha256"],
        "completion_marker_size_bytes": marker["marker_size_bytes"],
        "pre_marker_check_evidence_raw_sha256": pre["evidence_raw_sha256"],
        "pre_marker_check_evidence_internal_sha256": pre[
            "evidence_internal_sha256"
        ],
        "post_marker_artifact_check_evidence_raw_sha256": post_artifact[
            "evidence_raw_sha256"
        ],
        "post_marker_artifact_check_evidence_internal_sha256": post_artifact[
            "evidence_internal_sha256"
        ],
        "post_marker_completion_marker_check_evidence_raw_sha256": marker_check[
            "evidence_raw_sha256"
        ],
        "post_marker_completion_marker_check_evidence_internal_sha256": (
            marker_check["evidence_internal_sha256"]
        ),
        "completed_at_unix_ns": created_at,
    }
    receipt["receipt_sha256"] = canonical_sha256(receipt)
    _atomic_json(output, receipt)
    return receipt


def verify_g2c_calibration_phase_a_persistence(
    *,
    calibration_config_path: str | Path,
    prediction_freeze_root: str | Path,
    persistence_receipt_path: str | Path,
    expected_receipt_raw_sha256: str,
) -> dict[str, Any]:
    """离线重算独立 marker、三份 rclone 证据与最终 Drive receipt。"""

    expected_raw = _require_sha256(
        expected_receipt_raw_sha256,
        "G2C calibration persistence receipt expected raw SHA",
    )
    freeze_root = Path(prediction_freeze_root)
    receipt_path = Path(persistence_receipt_path)
    _assert_control_path_outside_freeze(
        receipt_path, prediction_freeze_root=freeze_root
    )
    if (
        receipt_path.name != _PHASE_A_PERSISTENCE_RECEIPT_NAME
        or receipt_path.is_symlink()
        or not receipt_path.is_file()
        or receipt_path.stat().st_nlink != 1
        or file_sha256(receipt_path) != expected_raw
    ):
        raise RuntimeError("G2C calibration Phase A final persistence receipt 漂移")
    config = load_g2c_calibration_config(calibration_config_path)
    freeze = verify_g2c_calibration_prediction_freeze(
        calibration_config_path=calibration_config_path, output_root=freeze_root
    )
    receipt = _read_json(receipt_path, "G2C calibration Phase A persistence")
    keys = {
        "version",
        "status",
        "completion_marker_name",
        "artifact_id",
        "worker_id",
        "remote_path",
        "remote_identity_sha256",
        "config_sha256",
        "prediction_freeze_raw_sha256",
        "prediction_freeze_internal_sha256",
        "source_git_commit",
        "source_identity_sha256",
        "artifact_file_count",
        "artifact_bytes",
        "completion_marker_raw_sha256",
        "completion_marker_internal_sha256",
        "completion_marker_size_bytes",
        "pre_marker_check_evidence_raw_sha256",
        "pre_marker_check_evidence_internal_sha256",
        "post_marker_artifact_check_evidence_raw_sha256",
        "post_marker_artifact_check_evidence_internal_sha256",
        "post_marker_completion_marker_check_evidence_raw_sha256",
        "post_marker_completion_marker_check_evidence_internal_sha256",
        "completed_at_unix_ns",
        "receipt_sha256",
    }
    _require_exact_keys(receipt, keys, "G2C calibration final persistence receipt")
    internal = receipt.get("receipt_sha256")
    unsigned = dict(receipt)
    unsigned.pop("receipt_sha256", None)
    remote = _phase_a_remote_identity(
        artifact_id=str(receipt.get("artifact_id")),
        worker_id=str(receipt.get("worker_id")),
        remote_path=str(receipt.get("remote_path")),
    )
    root = receipt_path.parent
    pre = _verify_phase_a_check_evidence(
        evidence_path=root
        / _PHASE_A_CHECK_EVIDENCE_NAMES["pre-marker-artifact-check"],
        expected_phase="pre-marker-artifact-check",
        config=config,
        freeze=freeze,
        prediction_freeze_root=freeze_root,
        remote=remote,
    )
    marker_path = root / _PHASE_A_COMPLETION_MARKER_NAME
    marker = _verify_phase_a_completion_marker(
        marker_path=marker_path,
        config=config,
        freeze=freeze,
        prediction_freeze_root=freeze_root,
        remote=remote,
        pre=pre,
    )
    post_artifact = _verify_phase_a_check_evidence(
        evidence_path=root
        / _PHASE_A_CHECK_EVIDENCE_NAMES["post-marker-artifact-check"],
        expected_phase="post-marker-artifact-check",
        config=config,
        freeze=freeze,
        prediction_freeze_root=freeze_root,
        remote=remote,
    )
    marker_check = _verify_phase_a_check_evidence(
        evidence_path=root
        / _PHASE_A_CHECK_EVIDENCE_NAMES[
            "post-marker-completion-marker-check"
        ],
        expected_phase="post-marker-completion-marker-check",
        config=config,
        freeze=freeze,
        prediction_freeze_root=freeze_root,
        remote=remote,
        completion_marker_path=marker_path,
    )
    expected_bindings = {
        "completion_marker_raw_sha256": marker["marker_raw_sha256"],
        "completion_marker_internal_sha256": marker["marker_internal_sha256"],
        "completion_marker_size_bytes": marker["marker_size_bytes"],
        "pre_marker_check_evidence_raw_sha256": pre["evidence_raw_sha256"],
        "pre_marker_check_evidence_internal_sha256": pre[
            "evidence_internal_sha256"
        ],
        "post_marker_artifact_check_evidence_raw_sha256": post_artifact[
            "evidence_raw_sha256"
        ],
        "post_marker_artifact_check_evidence_internal_sha256": post_artifact[
            "evidence_internal_sha256"
        ],
        "post_marker_completion_marker_check_evidence_raw_sha256": marker_check[
            "evidence_raw_sha256"
        ],
        "post_marker_completion_marker_check_evidence_internal_sha256": (
            marker_check["evidence_internal_sha256"]
        ),
    }
    if (
        internal != canonical_sha256(unsigned)
        or receipt.get("version")
        != E018_P1_G2C_CALIBRATION_PERSISTENCE_VERSION
        or receipt.get("status") != "DRIVE_VERIFIED"
        or receipt.get("completion_marker_name") != _PHASE_A_COMPLETION_MARKER_NAME
        or {name: receipt.get(name) for name in remote} != dict(remote)
        or receipt.get("remote_identity_sha256") != canonical_sha256(remote)
        or receipt.get("config_sha256") != config["config_sha256"]
        or receipt.get("prediction_freeze_raw_sha256")
        != freeze["freeze_raw_sha256"]
        or receipt.get("prediction_freeze_internal_sha256")
        != freeze["freeze_internal_sha256"]
        or receipt.get("source_git_commit") != freeze["source_git_commit"]
        or receipt.get("source_identity_sha256") != freeze["source_identity_sha256"]
        or receipt.get("artifact_file_count") != len(_PREDICTION_ARTIFACTS) + 1
        or receipt.get("artifact_bytes") != freeze["artifact_bytes"]
        or any(receipt.get(name) != value for name, value in expected_bindings.items())
        or not isinstance(receipt.get("completed_at_unix_ns"), int)
        or receipt["completed_at_unix_ns"]
        < max(
            post_artifact["recorded_at_unix_ns"],
            marker_check["recorded_at_unix_ns"],
        )
    ):
        raise RuntimeError("G2C calibration Phase A Drive persistence identity/order 漂移")
    _assert_unlinked_regular_file_tree(root, name="G2C Phase A persistence control")
    _verify_exact_regular_file_tree(
        root,
        expected_files={
            _PHASE_A_COMPLETION_MARKER_NAME,
            _PHASE_A_PERSISTENCE_RECEIPT_NAME,
            *_PHASE_A_CHECK_EVIDENCE_NAMES.values(),
            *_PHASE_A_CHECK_REPORT_NAMES.values(),
        },
        name="G2C Phase A persistence control",
    )
    result = {
        "version": E018_P1_G2C_CALIBRATION_PERSISTENCE_VERSION,
        "verified": True,
        "receipt_raw_sha256": expected_raw,
        "receipt_internal_sha256": internal,
        "prediction_freeze_internal_sha256": freeze["freeze_internal_sha256"],
        "source_identity_sha256": freeze["source_identity_sha256"],
        "remote_identity_sha256": receipt["remote_identity_sha256"],
        "pre_marker_difference_count": 0,
        "post_marker_artifact_difference_count": 0,
        "post_marker_completion_marker_difference_count": 0,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def prepare_g2c_calibration_privileged_view(
    *,
    calibration_config_path: str | Path,
    training_config_path: str | Path,
    data_root: str | Path,
    prediction_freeze_root: str | Path,
    repository_root: str | Path,
    phase_a_persistence_receipt_path: str | Path,
    expected_phase_a_persistence_receipt_raw_sha256: str,
    output_root: str | Path,
    decision_exit_go: bool,
) -> dict[str, Any]:
    if decision_exit_go is not True:
        raise PermissionError("G2C calibration privileged staging 仍为 HOLD")
    freeze = verify_g2c_calibration_prediction_freeze(
        calibration_config_path=calibration_config_path,
        output_root=prediction_freeze_root,
    )
    marker = _read_json(
        Path(prediction_freeze_root) / "prediction_freeze.json",
        "G2C calibration prediction freeze",
    )
    source = _git_source_identity(Path(repository_root))
    if (
        source["git_commit"] != marker.get("source_git_commit")
        or source["identity_sha256"] != marker.get("source_identity_sha256")
    ):
        raise RuntimeError("G2C calibration privileged source 与 Phase A 漂移")
    persistence = verify_g2c_calibration_phase_a_persistence(
        calibration_config_path=calibration_config_path,
        prediction_freeze_root=prediction_freeze_root,
        persistence_receipt_path=phase_a_persistence_receipt_path,
        expected_receipt_raw_sha256=(
            expected_phase_a_persistence_receipt_raw_sha256
        ),
    )
    return _prepare_calibration_input_view(
        calibration_config_path=calibration_config_path,
        training_config_path=training_config_path,
        data_root=data_root,
        output_root=output_root,
        role="calibration-privileged",
        prediction_freeze_internal_sha256=freeze["freeze_internal_sha256"],
        source_identity_sha256=source["identity_sha256"],
        persistence_receipt_raw_sha256=persistence["receipt_raw_sha256"],
        persistence_receipt_internal_sha256=persistence[
            "receipt_internal_sha256"
        ],
        remote_identity_sha256=persistence["remote_identity_sha256"],
    )


_DEPLOYABLE_SAFETY_KEYS = {
    "eligible_capture",
    "finger_force_n",
    "finger_force_valid",
    "raw_gripper_opening_ratio",
    "arm_joint_drift_rad",
    "tcp_position_drift_m",
    "tcp_orientation_drift_rad",
    "rgb_timestamp_s",
    "pose_timestamp_s",
    "camera_position_tracking_error_m",
    "camera_orientation_tracking_error_rad",
    "rotation_projection_error_frobenius",
}


def _selected_checkpoint_identity(
    checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    identity = {
        "candidate_id": checkpoint.get("candidate_id"),
        "epoch": checkpoint.get("epoch"),
        "relative_path": checkpoint.get("relative_path"),
        "checkpoint_sha256": checkpoint.get("checkpoint_sha256"),
        "parameter_state_sha256": checkpoint.get("parameter_state_sha256"),
        "provenance_sha256": checkpoint.get("provenance_sha256"),
        "model_config_sha256": checkpoint.get("model_config_sha256"),
    }
    expected = G2C_CALIBRATION_SELECTION_PARENT
    if (
        identity["candidate_id"] != expected["candidate_id"]
        or identity["epoch"] != expected["epoch"]
        or identity["checkpoint_sha256"] != expected["checkpoint_sha256"]
        or identity["parameter_state_sha256"]
        != expected["parameter_state_sha256"]
        or identity["provenance_sha256"] != expected["provenance_sha256"]
        or identity["model_config_sha256"] != expected["model_config_sha256"]
        or not isinstance(identity["relative_path"], str)
        or not identity["relative_path"]
    ):
        raise RuntimeError("G2C calibration selected checkpoint identity 漂移")
    return identity


def _verify_selection_parent(
    *,
    training_config_path: str | Path,
    model_val_prediction_freeze_root: str | Path,
    model_val_selection_root: str | Path,
) -> dict[str, Any]:
    verification = verify_g2c_model_val_selection(
        config_path=training_config_path,
        prediction_freeze_root=model_val_prediction_freeze_root,
        output_root=model_val_selection_root,
    )
    expected = G2C_CALIBRATION_SELECTION_PARENT
    selected = verification.get("selected")
    identity = verification.get("selected_checkpoint_identity")
    if (
        verification.get("status") != "complete-model-val-pass"
        or verification.get("receipt_raw_sha256")
        != expected["selection_receipt_raw_sha256"]
        or verification.get("receipt_internal_sha256")
        != expected["selection_receipt_internal_sha256"]
        or verification.get("verification_sha256")
        != expected["selection_verification_sha256"]
        or verification.get("prediction_freeze_internal_sha256")
        != expected["model_val_prediction_freeze_internal_sha256"]
        or not isinstance(selected, Mapping)
        or selected.get("candidate_id") != expected["candidate_id"]
        or selected.get("epoch") != expected["epoch"]
        or not isinstance(identity, Mapping)
    ):
        raise RuntimeError("G2C calibration D043 selection parent 漂移")
    _selected_checkpoint_identity(identity)
    return verification


def _dataset_safety_primitives(
    dataset: G2CDeployableDataset, index: int
) -> dict[str, Any]:
    entry_index, sample_index = dataset.index[index]
    arrays = dataset._bundle(entry_index)
    return {
        "eligible_capture": bool(arrays["eligible_capture"][sample_index]),
        "finger_force_n": arrays["finger_force_n"][sample_index]
        .astype(float)
        .tolist(),
        "finger_force_valid": bool(arrays["finger_force_valid"][sample_index]),
        "raw_gripper_opening_ratio": float(
            arrays["raw_gripper_opening_ratio"][sample_index]
        ),
        "arm_joint_drift_rad": float(arrays["arm_joint_drift_rad"][sample_index]),
        "tcp_position_drift_m": float(
            arrays["tcp_position_drift_m"][sample_index]
        ),
        "tcp_orientation_drift_rad": float(
            arrays["tcp_orientation_drift_rad"][sample_index]
        ),
        "rgb_timestamp_s": float(arrays["rgb_timestamp_s"][sample_index]),
        "pose_timestamp_s": float(arrays["pose_timestamp_s"][sample_index]),
        "camera_position_tracking_error_m": float(
            arrays["camera_position_tracking_error_m"][sample_index]
        ),
        "camera_orientation_tracking_error_rad": float(
            arrays["camera_orientation_tracking_error_rad"][sample_index]
        ),
        "rotation_projection_error_frobenius": float(
            arrays["rotation_projection_error_frobenius"][sample_index]
        ),
    }


def _deployable_free_static_safe(
    value: Mapping[str, Any], protocol: Mapping[str, Any]
) -> bool:
    _require_exact_keys(value, _DEPLOYABLE_SAFETY_KEYS, "G2C deployable safety")
    limits = protocol["free_static_safety"]
    forces = np.asarray(value["finger_force_n"], dtype=np.float64)
    finite_names = _DEPLOYABLE_SAFETY_KEYS - {
        "eligible_capture",
        "finger_force_n",
        "finger_force_valid",
    }
    if (
        type(value["eligible_capture"]) is not bool
        or type(value["finger_force_valid"]) is not bool
        or forces.shape != (2,)
        or not np.isfinite(forces).all()
        or np.any(forces < 0.0)
        or any(
            not isinstance(value[name], (int, float))
            or isinstance(value[name], bool)
            or not math.isfinite(float(value[name]))
            for name in finite_names
        )
    ):
        raise RuntimeError("G2C deployable safety primitive 类型/有限性漂移")
    return bool(
        value["eligible_capture"]
        and value["finger_force_valid"]
        and float(forces.max()) <= limits["maximum_finger_force_n"]
        and float(value["raw_gripper_opening_ratio"])
        >= limits["minimum_raw_gripper_opening_ratio"]
        and float(value["arm_joint_drift_rad"])
        <= limits["maximum_arm_joint_drift_rad"]
        and float(value["tcp_position_drift_m"])
        <= limits["maximum_tcp_position_drift_m"]
        and float(value["tcp_orientation_drift_rad"])
        <= limits["maximum_tcp_orientation_drift_rad"]
        and abs(float(value["rgb_timestamp_s"]) - float(value["pose_timestamp_s"]))
        <= limits["maximum_rgb_pose_skew_s"]
        and float(value["camera_position_tracking_error_m"])
        <= limits["maximum_camera_position_tracking_error_m"]
        and float(value["camera_orientation_tracking_error_rad"])
        <= limits["maximum_camera_orientation_tracking_error_rad"]
        and float(value["rotation_projection_error_frobenius"])
        <= limits["maximum_rotation_projection_error_frobenius"]
    )


def _object_write_evidence_from_prediction(
    row: Mapping[str, Any], *, geometry_valid: bool
) -> ObjectWriteEvidence:
    sigma = np.asarray(row["object_sigma_xy_px"], dtype=np.float64)
    if sigma.shape != (2,) or not np.isfinite(sigma).all() or np.any(sigma < 0.0):
        raise RuntimeError("G2C calibration prediction sigma 漂移")
    return ObjectWriteEvidence(
        visibility_probability=float(row["object_visibility_probability"]),
        projection_validity_probability=float(
            row["projection_validity_probability"]
        ),
        object_mask_probability=float(
            row["object_mask_probability_at_prediction"]
        ),
        goal_mask_probability=float(row["goal_mask_probability_at_prediction"]),
        normalized_entropy=float(row["object_normalized_entropy"]),
        radial_sigma_px=float(np.linalg.norm(sigma)),
        geometry_valid=geometry_valid,
    )


def _validate_prediction_row_mechanics(
    row: Mapping[str, Any], *, config: Mapping[str, Any]
) -> None:
    _require_exact_keys(
        row, _CALIBRATION_PREDICTION_ROW_KEYS, "G2C calibration prediction row"
    )
    if (
        row.get("version") != E018_P1_G2C_CALIBRATION_FREEZE_VERSION
        or row.get("phase")
        != "deployable-calibration-before-privileged-label-open/v1"
        or row.get("candidate_id")
        != G2C_CALIBRATION_SELECTION_PARENT["candidate_id"]
        or row.get("epoch") != G2C_CALIBRATION_SELECTION_PARENT["epoch"]
        or row.get("checkpoint_sha256")
        != G2C_CALIBRATION_SELECTION_PARENT["checkpoint_sha256"]
        or row.get("checkpoint_parameter_sha256")
        != G2C_CALIBRATION_SELECTION_PARENT["parameter_state_sha256"]
        or row.get("checkpoint_provenance_sha256")
        != G2C_CALIBRATION_SELECTION_PARENT["provenance_sha256"]
        or row.get("checkpoint_model_config_sha256")
        != G2C_CALIBRATION_SELECTION_PARENT["model_config_sha256"]
        or row.get("split") != "calibration"
        or type(row.get("predicted_observable")) is not bool
        or type(row.get("geometry_valid")) is not bool
        or row.get("memory_write_allowed") is not False
        or row.get("actuation_allowed") is not False
    ):
        raise RuntimeError("G2C calibration prediction identity/permission 漂移")
    probability_names = (
        "object_visibility_probability",
        "goal_visibility_probability",
        "projection_validity_probability",
        "object_normalized_entropy",
        "object_mask_probability_at_prediction",
        "goal_mask_probability_at_prediction",
        "write_score",
    )
    probabilities = [float(row[name]) for name in probability_names]
    object_uv = np.asarray(row["predicted_object_normalized_uv"], dtype=np.float64)
    goal_uv = np.asarray(row["predicted_goal_normalized_uv"], dtype=np.float64)
    intrinsic = np.asarray(row["external_intrinsic_cv"], dtype=np.float64)
    base_from_camera = np.asarray(
        row["base_from_external_camera_cv"], dtype=np.float64
    )
    if (
        any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in probabilities)
        or object_uv.shape != (2,)
        or goal_uv.shape != (2,)
        or intrinsic.shape != (3, 3)
        or base_from_camera.shape != (4, 4)
        or not np.isfinite(object_uv).all()
        or not np.isfinite(goal_uv).all()
        or not np.isfinite(intrinsic).all()
        or not np.isfinite(base_from_camera).all()
        or np.any(object_uv < 0.0)
        or np.any(object_uv > 1.0)
        or np.any(goal_uv < 0.0)
        or np.any(goal_uv > 1.0)
    ):
        raise RuntimeError("G2C calibration prediction primitive 漂移")
    from robot_vla.precision.e018_p1_g2c import _measurement_covariance

    try:
        geometry = geometry_conditioning(
            normalized_uv=object_uv,
            intrinsic_cv=intrinsic,
            base_from_camera_cv=base_from_camera,
            image_size_hw=(128, 128),
            plane_base_z_m=0.02,
        )
        expected_position = np.asarray(
            geometry["predicted_world_point_base_m"], dtype=np.float64
        )
        expected_covariance = _measurement_covariance(
            geometry["local_jacobian_xy_m_per_px"],
            np.asarray(row["object_sigma_xy_px"], dtype=np.float64),
        )
        expected_geometry_valid = True
    except ValueError:
        expected_position = None
        expected_covariance = None
        expected_geometry_valid = False
    if row["geometry_valid"] is not expected_geometry_valid:
        raise RuntimeError("G2C calibration geometry_valid 重算漂移")
    if expected_geometry_valid:
        position = np.asarray(row["predicted_object_position_base_m"], dtype=np.float64)
        covariance = np.asarray(row["raw_covariance_base_m2"], dtype=np.float64)
        if (
            position.shape != (3,)
            or covariance.shape != (3, 3)
            or not np.isfinite(position).all()
            or not np.isfinite(covariance).all()
            or not np.allclose(position, expected_position, rtol=1e-9, atol=1e-12)
            or not np.allclose(
                covariance, expected_covariance, rtol=1e-9, atol=1e-15
            )
            or not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12)
            or float(np.linalg.eigvalsh(covariance).min()) < -1e-12
        ):
            raise RuntimeError("G2C calibration position/covariance 重算漂移")
    elif (
        row["predicted_object_position_base_m"] is not None
        or row["raw_covariance_base_m2"] is not None
    ):
        raise RuntimeError("G2C calibration invalid geometry 不得保留 position/covariance")
    evidence = _object_write_evidence_from_prediction(
        row, geometry_valid=expected_geometry_valid
    )
    if (
        row["predicted_observable"]
        is not (float(row["object_visibility_probability"]) >= 0.5)
        or not math.isclose(
            float(row["write_score"]), evidence.score, rel_tol=1e-12, abs_tol=1e-12
        )
    ):
        raise RuntimeError("G2C calibration observable/write-score 重算漂移")
    _deployable_free_static_safe(row["deployable_safety"], config["protocol"])


def run_g2c_calibration_prediction_freeze(
    *,
    calibration_config_path: str | Path,
    training_config_path: str | Path,
    training_output_root: str | Path,
    model_val_prediction_freeze_root: str | Path,
    model_val_selection_root: str | Path,
    calibration_deployable_input_root: str | Path,
    repository_root: str | Path,
    output_root: str | Path,
    decision_exit_go: bool,
) -> dict[str, Any]:
    """Phase A：selected checkpoint-only、deployable-only prediction freeze。"""

    if decision_exit_go is not True:
        raise PermissionError("G2C formal calibration Phase A 仍为 HOLD")
    import torch

    from robot_vla.precision.checkpoint import (
        PrecisionCheckpointRole,
        load_precision_checkpoint,
    )

    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"G2C calibration prediction output 已存在: {output}")
    config = load_g2c_calibration_config(calibration_config_path)
    training_config_verification = _verify_training_config_reference(
        config, training_config_path
    )
    training_verification = verify_g2c_formal_training(
        config_path=training_config_path, output_root=training_output_root
    )
    selection_verification = _verify_selection_parent(
        training_config_path=training_config_path,
        model_val_prediction_freeze_root=model_val_prediction_freeze_root,
        model_val_selection_root=model_val_selection_root,
    )
    source_identity = _git_source_identity(Path(repository_root))
    deployable_verification = validate_g2c_calibration_input_view(
        calibration_config_path=calibration_config_path,
        training_config_path=training_config_path,
        input_root=calibration_deployable_input_root,
        expected_role="calibration-deployable",
    )
    training_receipt = _read_json(
        Path(training_output_root) / "training_receipt.json",
        "G2C formal training receipt",
    )
    matches = [
        item
        for item in training_receipt.get("checkpoint_inventory", [])
        if item.get("candidate_id") == G2C_CALIBRATION_SELECTION_PARENT["candidate_id"]
        and item.get("epoch") == G2C_CALIBRATION_SELECTION_PARENT["epoch"]
    ]
    if len(matches) != 1:
        raise RuntimeError("G2C calibration selected checkpoint 必须唯一")
    checkpoint_identity = _selected_checkpoint_identity(matches[0])
    if _selected_checkpoint_identity(
        selection_verification["selected_checkpoint_identity"]
    ) != checkpoint_identity:
        raise RuntimeError("G2C calibration TRAIN/selection checkpoint lineage 漂移")
    if not torch.cuda.is_available():
        raise RuntimeError("G2C formal calibration Phase A 要求 CUDA")
    dataset = G2CDeployableDataset(
        calibration_deployable_input_root, "calibration"
    )
    if len(dataset) != 550 or len(dataset.entries) != 50:
        raise RuntimeError("G2C calibration deployable input 必须是 50 bundles/550 rows")
    expected_identity = [
        (seed, sample_index, viewpoint_id)
        for seed in _CALIBRATION_SEEDS
        for sample_index, viewpoint_id in enumerate(G2C_VIEW_ORDER)
    ]
    actual_identity = [
        (entry.seed, sample_index, entry.view_order[sample_index])
        for entry in dataset.entries
        for sample_index in range(entry.sample_count)
    ]
    if actual_identity != expected_identity:
        raise RuntimeError("G2C calibration deployable row order 漂移")
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(output / "config_snapshot.json", config)
    _atomic_json(
        output / "training_config_verification.json", training_config_verification
    )
    _atomic_json(output / "training_verification.json", training_verification)
    _atomic_json(output / "selection_verification.json", selection_verification)
    _atomic_json(output / "source_identity.json", source_identity)
    _atomic_json(
        output / "deployable_input_verification.json", deployable_verification
    )
    _atomic_json(
        output / "selected_checkpoint_identity.json", checkpoint_identity
    )
    checkpoint_path = (
        Path(training_output_root)
        / "candidates"
        / str(checkpoint_identity["candidate_id"])
        / str(checkpoint_identity["relative_path"])
    )
    loaded = load_precision_checkpoint(
        checkpoint_path,
        expected_checkpoint_sha256=checkpoint_identity["checkpoint_sha256"],
        expected_provenance_sha256=checkpoint_identity["provenance_sha256"],
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    if (
        loaded.receipt.parameter_state_sha256
        != checkpoint_identity["parameter_state_sha256"]
        or loaded.receipt.model_config_sha256
        != checkpoint_identity["model_config_sha256"]
    ):
        raise RuntimeError("G2C calibration loaded checkpoint identity 漂移")
    device = torch.device("cuda")
    model = loaded.model.to(device)
    model.eval()
    rows: list[dict[str, Any]] = []
    batch_sizes: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(dataset), 32)):
            samples = [
                dataset[index]
                for index in range(start, min(start + 32, len(dataset)))
            ]
            image = np.stack(
                [
                    sample["model_inputs"]["rgb_external"].transpose(2, 0, 1)
                    for sample in samples
                ]
            ).astype(np.float32)
            image /= np.float32(255.0)
            state = np.stack(
                [sample["model_inputs"]["structured_state"] for sample in samples]
            )
            motion = np.stack(
                [sample["model_inputs"]["geometric_motion"] for sample in samples]
            )
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                model_output = model(
                    torch.from_numpy(image).to(device),
                    torch.from_numpy(state).to(device),
                    torch.from_numpy(motion).to(device),
                )
            batch_rows = _prediction_rows_for_batch(
                samples=samples,
                output=model_output,
                candidate_id=str(checkpoint_identity["candidate_id"]),
                epoch=int(checkpoint_identity["epoch"]),
                checkpoint_identity=checkpoint_identity,
                global_start=start,
                batch_index=batch_index,
            )
            for offset, row in enumerate(batch_rows):
                sample = samples[offset]
                row.update(
                    {
                        "version": E018_P1_G2C_CALIBRATION_FREEZE_VERSION,
                        "phase": (
                            "deployable-calibration-before-privileged-label-open/v1"
                        ),
                        "external_intrinsic_cv": sample["capture"][
                            "external_intrinsic_cv"
                        ].astype(float).tolist(),
                        "base_from_external_camera_cv": sample["capture"][
                            "base_from_external_camera_cv"
                        ].astype(float).tolist(),
                        "deployable_safety": _dataset_safety_primitives(
                            dataset, start + offset
                        ),
                    }
                )
                _validate_prediction_row_mechanics(row, config=config)
            rows.extend(batch_rows)
            batch_sizes.append(len(samples))
    if len(rows) != 550 or batch_sizes != [32] * 17 + [6]:
        raise RuntimeError("G2C calibration Phase A 必须是 550 rows/18 batches")
    _atomic_jsonl(output / "prediction_ledger.jsonl", rows)
    del rows, model_output, model, loaded, dataset, samples, image, state, motion
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    if elapsed > config["protocol"]["budgets"]["formal_phase_a_gpu_seconds_max"]:
        raise RuntimeError("G2C calibration Phase A GPU 时间超过 1 hour 停止线")
    inference_audit = {
        "version": E018_P1_G2C_CALIBRATION_FREEZE_VERSION,
        **_PREDICTION_COUNT_CONTRACT,
        "batch_sizes": batch_sizes,
        "model_and_inference_context_destroyed": True,
        "calibration_privileged_label_bundle_open_count": 0,
        "test_array_read_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "physical_camera_actuation_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "manipulation_progression_count": 0,
        "checkpoint_write_count": 0,
    }
    _atomic_json(output / "inference_audit.json", inference_audit)
    artifact_inventory = [
        {
            "relative_path": name,
            "raw_sha256": file_sha256(output / name),
            "size_bytes": (output / name).stat().st_size,
        }
        for name in _PREDICTION_ARTIFACTS
    ]
    artifact_bytes = sum(item["size_bytes"] for item in artifact_inventory)
    if artifact_bytes > config["protocol"]["budgets"]["formal_artifact_bytes_max"]:
        raise RuntimeError("G2C calibration Phase A artifact 超过 5 GiB")
    marker = {
        "version": E018_P1_G2C_CALIBRATION_FREEZE_VERSION,
        "status": "complete-calibration-prediction-freeze-pass",
        "classification": "deployable-only-selected-checkpoint-no-test-no-actuation",
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": config["data_parent"]["data_identity_sha256"],
        "source_git_commit": source_identity["git_commit"],
        "source_identity_sha256": source_identity["identity_sha256"],
        "selection_receipt_internal_sha256": G2C_CALIBRATION_SELECTION_PARENT[
            "selection_receipt_internal_sha256"
        ],
        "selected_checkpoint_identity": checkpoint_identity,
        **_PREDICTION_COUNT_CONTRACT,
        "batch_sizes": batch_sizes,
        "model_and_inference_context_destroyed_before_freeze": True,
        "calibration_privileged_label_bundle_open_count": 0,
        "test_array_read_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "physical_camera_actuation_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "manipulation_progression_count": 0,
        "checkpoint_write_count": 0,
        "artifact_inventory": artifact_inventory,
        "artifact_inventory_sha256": canonical_sha256(artifact_inventory),
        "artifact_bytes_before_freeze_marker": artifact_bytes,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "torch": torch.__version__,
            "cuda_device": torch.cuda.get_device_name(device),
        },
        "elapsed_s": elapsed,
        "frozen_at_unix_ns": time.time_ns(),
    }
    marker["freeze_sha256"] = canonical_sha256(marker)
    _atomic_json(output / "prediction_freeze.json", marker)
    return {
        **marker,
        "verification": verify_g2c_calibration_prediction_freeze(
            calibration_config_path=calibration_config_path,
            output_root=output,
        ),
    }


def verify_g2c_calibration_prediction_freeze(
    *, calibration_config_path: str | Path, output_root: str | Path
) -> dict[str, Any]:
    """公开验证 Phase A；签名刻意不接 model/checkpoint/DATA/label path。"""

    config = load_g2c_calibration_config(calibration_config_path)
    root = Path(output_root)
    _assert_unlinked_regular_file_tree(
        root, name="G2C calibration prediction freeze"
    )
    marker_path = root / "prediction_freeze.json"
    marker = _read_json(marker_path, "G2C calibration prediction freeze")
    marker_keys = {
        "version",
        "status",
        "classification",
        "config_sha256",
        "data_identity_sha256",
        "source_git_commit",
        "source_identity_sha256",
        "selection_receipt_internal_sha256",
        "selected_checkpoint_identity",
        *_PREDICTION_COUNT_CONTRACT,
        "batch_sizes",
        "model_and_inference_context_destroyed_before_freeze",
        "calibration_privileged_label_bundle_open_count",
        "test_array_read_count",
        "memory_read_count",
        "memory_write_count",
        "runtime_camera_actuation_count",
        "physical_camera_actuation_count",
        "arm_motion_command_count",
        "gripper_close_command_count",
        "manipulation_progression_count",
        "checkpoint_write_count",
        "artifact_inventory",
        "artifact_inventory_sha256",
        "artifact_bytes_before_freeze_marker",
        "environment",
        "elapsed_s",
        "frozen_at_unix_ns",
        "freeze_sha256",
    }
    _require_exact_keys(marker, marker_keys, "G2C calibration freeze marker")
    internal = marker.get("freeze_sha256")
    unsigned = dict(marker)
    unsigned.pop("freeze_sha256", None)
    _assert_exact_count_contract(
        marker,
        expected=_PREDICTION_COUNT_CONTRACT,
        name="G2C calibration prediction freeze",
    )
    zero_fields = (
        "calibration_privileged_label_bundle_open_count",
        "test_array_read_count",
        "memory_read_count",
        "memory_write_count",
        "runtime_camera_actuation_count",
        "physical_camera_actuation_count",
        "arm_motion_command_count",
        "gripper_close_command_count",
        "manipulation_progression_count",
        "checkpoint_write_count",
    )
    if (
        internal != canonical_sha256(unsigned)
        or marker.get("version") != E018_P1_G2C_CALIBRATION_FREEZE_VERSION
        or marker.get("status")
        != "complete-calibration-prediction-freeze-pass"
        or marker.get("classification")
        != "deployable-only-selected-checkpoint-no-test-no-actuation"
        or marker.get("config_sha256") != config["config_sha256"]
        or marker.get("data_identity_sha256")
        != config["data_parent"]["data_identity_sha256"]
        or marker.get("selection_receipt_internal_sha256")
        != G2C_CALIBRATION_SELECTION_PARENT[
            "selection_receipt_internal_sha256"
        ]
        or marker.get("selected_checkpoint_identity")
        != _selected_checkpoint_identity(marker["selected_checkpoint_identity"])
        or marker.get("batch_sizes") != [32] * 17 + [6]
        or marker.get("model_and_inference_context_destroyed_before_freeze")
        is not True
        or any(marker.get(name) != 0 for name in zero_fields)
        or not isinstance(marker.get("elapsed_s"), (int, float))
        or isinstance(marker.get("elapsed_s"), bool)
        or not math.isfinite(float(marker["elapsed_s"]))
        or float(marker["elapsed_s"]) < 0.0
        or float(marker["elapsed_s"])
        > config["protocol"]["budgets"]["formal_phase_a_gpu_seconds_max"]
        or not isinstance(marker.get("frozen_at_unix_ns"), int)
    ):
        raise RuntimeError("G2C calibration freeze status/count/permission 漂移")
    if _read_json(root / "config_snapshot.json", "G2C calibration config snapshot") != config:
        raise RuntimeError("G2C calibration config snapshot 漂移")
    training_config_verification = _read_json(
        root / "training_config_verification.json",
        "G2C calibration training config verification",
    )
    if training_config_verification != {
        **G2C_TRAINING_CONFIG_REFERENCE,
        "verified": True,
    }:
        raise RuntimeError("G2C calibration training config reference 漂移")
    training_verification = _read_json(
        root / "training_verification.json", "G2C calibration training verification"
    )
    if (
        training_verification.get("status") != "complete-formal-training-pass"
        or training_verification.get("receipt_internal_sha256") is None
        or training_verification.get("source_git_commit")
        != G2C_CALIBRATION_SELECTION_PARENT["training_source_git_commit"]
        or training_verification.get("source_identity_sha256")
        != G2C_CALIBRATION_SELECTION_PARENT["training_source_identity_sha256"]
    ):
        raise RuntimeError("G2C calibration formal TRAIN verification 漂移")
    selection_verification = _read_json(
        root / "selection_verification.json",
        "G2C calibration D043 selection verification",
    )
    expected_selection = G2C_CALIBRATION_SELECTION_PARENT
    if (
        selection_verification.get("status") != "complete-model-val-pass"
        or selection_verification.get("receipt_raw_sha256")
        != expected_selection["selection_receipt_raw_sha256"]
        or selection_verification.get("receipt_internal_sha256")
        != expected_selection["selection_receipt_internal_sha256"]
        or selection_verification.get("verification_sha256")
        != expected_selection["selection_verification_sha256"]
        or selection_verification.get("prediction_freeze_internal_sha256")
        != expected_selection[
            "model_val_prediction_freeze_internal_sha256"
        ]
        or _selected_checkpoint_identity(
            selection_verification["selected_checkpoint_identity"]
        )
        != marker["selected_checkpoint_identity"]
    ):
        raise RuntimeError("G2C calibration selection verification 漂移")
    source = _read_json(root / "source_identity.json", "G2C calibration source")
    if (
        source.get("identity_sha256")
        != canonical_sha256(
            {
                "git_commit": source.get("git_commit"),
                "source_tree_sha256": source.get("source_tree_sha256"),
            }
        )
        or marker.get("source_git_commit") != source.get("git_commit")
        or marker.get("source_identity_sha256") != source.get("identity_sha256")
    ):
        raise RuntimeError("G2C calibration source identity 漂移")
    deployable_verification = _read_json(
        root / "deployable_input_verification.json",
        "G2C calibration deployable input verification",
    )
    inventory = deployable_verification.get("inventory")
    if (
        deployable_verification.get("verified") is not True
        or deployable_verification.get("role") != "calibration-deployable"
        or deployable_verification.get("split") != "calibration"
        or deployable_verification.get("seed_count") != 50
        or deployable_verification.get("sample_count") != 550
        or deployable_verification.get("prediction_freeze_internal_sha256")
        is not None
        or deployable_verification.get("source_identity_sha256") is not None
        or deployable_verification.get("privileged_label_array_open_count") != 0
        or not isinstance(inventory, Mapping)
        or inventory.get("deployable_inventory_sha256")
        != config["data_parent"]["deployable_inventory_sha256"]
        or inventory.get("privileged_inventory_sha256") is not None
    ):
        raise RuntimeError("G2C calibration deployable input verification 漂移")
    selected_file = _read_json(
        root / "selected_checkpoint_identity.json",
        "G2C calibration selected checkpoint",
    )
    if _selected_checkpoint_identity(selected_file) != marker[
        "selected_checkpoint_identity"
    ]:
        raise RuntimeError("G2C calibration selected checkpoint file 漂移")
    audit = _read_json(root / "inference_audit.json", "G2C calibration inference audit")
    _assert_exact_count_contract(
        audit,
        expected=_PREDICTION_COUNT_CONTRACT,
        name="G2C calibration inference audit",
    )
    expected_audit_keys = {
        "version",
        *_PREDICTION_COUNT_CONTRACT,
        "batch_sizes",
        "model_and_inference_context_destroyed",
        *zero_fields,
    }
    _require_exact_keys(audit, expected_audit_keys, "G2C calibration inference audit")
    if (
        audit["version"] != E018_P1_G2C_CALIBRATION_FREEZE_VERSION
        or audit["batch_sizes"] != [32] * 17 + [6]
        or audit["model_and_inference_context_destroyed"] is not True
        or any(audit.get(name) != 0 for name in zero_fields)
    ):
        raise RuntimeError("G2C calibration inference audit 漂移")
    rows = _read_jsonl(root / "prediction_ledger.jsonl", "G2C calibration ledger")
    if len(rows) != 550:
        raise RuntimeError("G2C calibration prediction ledger 必须是 550 rows")
    for index, row in enumerate(rows):
        _validate_prediction_row_mechanics(row, config=config)
        expected_seed = _CALIBRATION_SEEDS[index // len(G2C_VIEW_ORDER)]
        expected_sample_index = index % len(G2C_VIEW_ORDER)
        expected_viewpoint = G2C_VIEW_ORDER[expected_sample_index]
        if (
            row.get("row_index") != index
            or row.get("batch_index") != index // 32
            or row.get("batch_offset") != index % 32
            or row.get("seed") != expected_seed
            or row.get("sample_index") != expected_sample_index
            or row.get("viewpoint_id") != expected_viewpoint
        ):
            raise RuntimeError("G2C calibration prediction row order/identity 漂移")
        _require_sha256(row.get("input_sha256"), "G2C calibration prediction input")
    artifact_inventory = marker.get("artifact_inventory")
    if (
        not isinstance(artifact_inventory, list)
        or [item.get("relative_path") for item in artifact_inventory]
        != list(_PREDICTION_ARTIFACTS)
    ):
        raise RuntimeError("G2C calibration freeze artifact inventory order 漂移")
    for item in artifact_inventory:
        _require_exact_keys(
            item,
            {"relative_path", "raw_sha256", "size_bytes"},
            "G2C calibration freeze artifact row",
        )
        path = root / str(item["relative_path"])
        if (
            path.is_symlink()
            or file_sha256(path) != item["raw_sha256"]
            or path.stat().st_size != item["size_bytes"]
        ):
            raise RuntimeError("G2C calibration freeze artifact identity 漂移")
    if (
        marker.get("artifact_inventory_sha256")
        != canonical_sha256(artifact_inventory)
        or marker.get("artifact_bytes_before_freeze_marker")
        != sum(int(item["size_bytes"]) for item in artifact_inventory)
    ):
        raise RuntimeError("G2C calibration freeze aggregate artifact 漂移")
    total_bytes = _verify_exact_regular_file_tree(
        root,
        expected_files={*_PREDICTION_ARTIFACTS, "prediction_freeze.json"},
        name="G2C calibration prediction freeze",
    )
    if total_bytes > config["protocol"]["budgets"]["formal_artifact_bytes_max"]:
        raise RuntimeError("G2C calibration freeze artifact byte budget 漂移")
    result = {
        "version": E018_P1_G2C_CALIBRATION_FREEZE_VERSION,
        "status": marker["status"],
        "verified": True,
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": marker["data_identity_sha256"],
        "source_git_commit": marker["source_git_commit"],
        "source_identity_sha256": marker["source_identity_sha256"],
        "freeze_raw_sha256": file_sha256(marker_path),
        "freeze_internal_sha256": internal,
        "selected_checkpoint_identity": marker["selected_checkpoint_identity"],
        "selected_checkpoint_count": 1,
        "prediction_row_count": 550,
        "model_forward_batch_count": 18,
        "privileged_label_open_count_before_freeze": 0,
        "model_and_inference_context_destroyed": True,
        "artifact_bytes": total_bytes,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def _load_calibration_labels(
    label_input_root: Path,
    *,
    on_bundle_open: Any,
) -> dict[tuple[int, int, str], dict[str, Any]]:
    manifest = _read_jsonl(
        label_input_root / "privileged_labels" / "manifest.jsonl",
        "G2C calibration privileged manifest",
    )
    if (
        len(manifest) != 50
        or [int(row.get("seed", -1)) for row in manifest]
        != list(_CALIBRATION_SEEDS)
    ):
        raise RuntimeError("G2C calibration label manifest 必须是冻结的 50 seeds")
    result: dict[tuple[int, int, str], dict[str, Any]] = {}
    open_count = 0
    for manifest_row in manifest:
        _require_exact_keys(
            manifest_row,
            {
                "manifest_schema_version",
                "split",
                "seed",
                "sample_count",
                "view_order",
                "schema_version",
                "file",
                "sha256",
                "contains_model_input_rgb",
                "source_deployable_file",
                "source_deployable_sha256",
            },
            "G2C calibration label manifest row",
        )
        seed = int(manifest_row["seed"])
        if (
            manifest_row["manifest_schema_version"]
            != G2C_MANIFEST_SCHEMA_VERSION
            or manifest_row["schema_version"] != G2C_LABEL_SCHEMA_VERSION
            or manifest_row["split"] != "calibration"
            or manifest_row["sample_count"] != 11
            or tuple(manifest_row["view_order"]) != G2C_VIEW_ORDER
            or manifest_row["contains_model_input_rgb"] is not False
        ):
            raise RuntimeError("G2C calibration label manifest schema/role 漂移")
        path = _resolve_artifact_file(
            label_input_root / "privileged_labels", str(manifest_row["file"])
        )
        if path.is_symlink() or not path.is_file() or path.stat().st_nlink != 1:
            raise RuntimeError("G2C calibration label bundle link/type 漂移")
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest() != manifest_row["sha256"]:
            raise RuntimeError("G2C calibration label bundle SHA-256 漂移")
        open_count += 1
        on_bundle_open(open_count)
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            if set(archive.files) != set(_LABEL_ARRAYS):
                raise RuntimeError("G2C calibration label bundle arrays 漂移")
            arrays = {name: archive[name] for name in archive.files}
        if (
            arrays["source_sample_index"].shape != (11,)
            or arrays["source_sample_index"].dtype != np.int64
            or not np.array_equal(
                arrays["source_sample_index"], np.arange(11, dtype=np.int64)
            )
            or arrays["seed"].shape != (11,)
            or arrays["seed"].dtype != np.int64
            or not np.all(arrays["seed"] == seed)
            or tuple(str(value) for value in arrays["viewpoint_id"])
            != G2C_VIEW_ORDER
            or arrays["object_position_base_m"].shape != (11, 3)
            or arrays["object_position_base_m"].dtype != np.float32
            or arrays["keypoint_observable"].shape != (11, 2)
            or arrays["keypoint_observable"].dtype != np.bool_
            or arrays["object_exists"].shape != (11,)
            or arrays["object_exists"].dtype != np.bool_
            or arrays["is_grasped"].shape != (11,)
            or arrays["is_grasped"].dtype != np.bool_
            or arrays["robot_object_contact_force_n"].shape != (11,)
            or arrays["robot_object_contact_force_n"].dtype != np.float32
            or not np.isfinite(arrays["object_position_base_m"]).all()
            or not np.isfinite(arrays["robot_object_contact_force_n"]).all()
            or np.any(arrays["robot_object_contact_force_n"] < 0.0)
        ):
            raise RuntimeError("G2C calibration label shape/dtype/finite 漂移")
        for sample_index, viewpoint_id in enumerate(G2C_VIEW_ORDER):
            identity = (seed, sample_index, viewpoint_id)
            if identity in result:
                raise RuntimeError("G2C calibration label row identity 重复")
            result[identity] = {
                "gt_observable": bool(
                    arrays["keypoint_observable"][sample_index, 0]
                ),
                "gt_object_position_base_m": arrays[
                    "object_position_base_m"
                ][sample_index]
                .astype(float)
                .tolist(),
                "gt_object_exists": bool(arrays["object_exists"][sample_index]),
                "is_grasped": bool(arrays["is_grasped"][sample_index]),
                "robot_object_contact_force_n": float(
                    arrays["robot_object_contact_force_n"][sample_index]
                ),
            }
    if len(result) != 550 or open_count != 50:
        raise RuntimeError("G2C calibration labels 必须是 50 bundles/550 rows")
    return result


def _score_calibration_prediction(
    prediction: Mapping[str, Any],
    label: Mapping[str, Any],
    *,
    config: Mapping[str, Any],
    freeze_sha256: str,
) -> dict[str, Any]:
    _validate_prediction_row_mechanics(prediction, config=config)
    limits = config["protocol"]["free_static_safety"]
    gt_position = np.asarray(label["gt_object_position_base_m"], dtype=np.float64)
    contact = float(label["robot_object_contact_force_n"])
    for name in ("gt_observable", "gt_object_exists", "is_grasped"):
        if type(label[name]) is not bool:
            raise TypeError(f"G2C calibration {name} 必须是 bool")
    if (
        gt_position.shape != (3,)
        or not np.isfinite(gt_position).all()
        or not math.isfinite(contact)
        or contact < 0.0
    ):
        raise RuntimeError("G2C calibration GT primitive shape/finite 漂移")
    predicted_value = prediction["predicted_object_position_base_m"]
    predicted_position = (
        None
        if predicted_value is None
        else np.asarray(predicted_value, dtype=np.float64)
    )
    world_error = (
        None
        if predicted_position is None
        else float(np.linalg.norm(predicted_position - gt_position))
    )
    error_xy = (
        None
        if predicted_position is None
        else (predicted_position[:2] - gt_position[:2]).astype(float).tolist()
    )
    evidence = _object_write_evidence_from_prediction(
        prediction, geometry_valid=bool(prediction["geometry_valid"])
    )
    predicted_observable = bool(
        float(prediction["object_visibility_probability"]) >= 0.5
    )
    deployable_safe = _deployable_free_static_safe(
        prediction["deployable_safety"], config["protocol"]
    )
    privileged_safe = bool(
        label["gt_object_exists"]
        and not label["is_grasped"]
        and contact <= limits["maximum_robot_object_contact_force_n"]
        and abs(float(gt_position[2]) - limits["object_center_base_z_m"])
        <= limits["object_center_base_z_tolerance_m"]
    )
    structurally_eligible = bool(
        predicted_observable
        and evidence.structurally_eligible
        and deployable_safe
    )
    oracle_safe = bool(
        deployable_safe
        and privileged_safe
        and label["gt_observable"]
        and prediction["geometry_valid"]
        and world_error is not None
        and math.isfinite(world_error)
        and world_error
        <= config["protocol"]["write_threshold"][
            "maximum_oracle_safe_error_m"
        ]
    )
    catastrophic = bool(
        world_error is not None
        and math.isfinite(world_error)
        and world_error
        > config["protocol"]["write_threshold"]["catastrophic_error_m"]
    )
    return {
        "version": E018_P1_G2C_CALIBRATION_RESULT_VERSION,
        "phase": "privileged-calibration-after-complete-prediction-freeze/v1",
        "prediction_freeze_sha256": freeze_sha256,
        "candidate_id": prediction["candidate_id"],
        "epoch": prediction["epoch"],
        "checkpoint_sha256": prediction["checkpoint_sha256"],
        "row_index": prediction["row_index"],
        "seed": prediction["seed"],
        "sample_index": prediction["sample_index"],
        "viewpoint_id": prediction["viewpoint_id"],
        "gt_observable": label["gt_observable"],
        "gt_object_position_base_m": gt_position.astype(float).tolist(),
        "gt_object_exists": label["gt_object_exists"],
        "is_grasped": label["is_grasped"],
        "robot_object_contact_force_n": contact,
        "predicted_observable": predicted_observable,
        "object_write_structurally_eligible": evidence.structurally_eligible,
        "deployable_free_static_safe": deployable_safe,
        "privileged_free_static_safe": privileged_safe,
        "geometry_valid": prediction["geometry_valid"],
        "world_xyz_error_m": world_error,
        "world_xy_error_vector_m": error_xy,
        "predicted_object_position_base_m": predicted_value,
        "raw_covariance_base_m2": prediction["raw_covariance_base_m2"],
        "write_score": evidence.score,
        "structurally_eligible": structurally_eligible,
        "oracle_safe_measurement": oracle_safe,
        "catastrophic_measurement": catastrophic,
        "test_data_read": False,
    }


def _calibration_summary(
    *,
    config: Mapping[str, Any],
    freeze_verification: Mapping[str, Any],
    source_identity: Mapping[str, Any],
    calibrations: Sequence[Mapping[str, Any]],
    label_input_verification: Mapping[str, Any],
) -> dict[str, Any]:
    qualified_non_home = [
        str(item["viewpoint_id"])
        for item in calibrations
        if item["viewpoint_id"] != G2C_VIEW_ORDER[0] and item["passed"] is True
    ]
    passed = bool(qualified_non_home)
    return {
        "version": E018_P1_G2C_CALIBRATION_RESULT_VERSION,
        "status": (
            "complete-calibration-pass"
            if passed
            else "complete-calibration-protocol-valid-negative"
        ),
        "classification": "development-only-calibration-no-test-no-actuation",
        "protocol_valid": True,
        "gate_passed": passed,
        "config_sha256": config["config_sha256"],
        "data_identity_sha256": config["data_parent"]["data_identity_sha256"],
        "source_git_commit": source_identity["git_commit"],
        "source_identity_sha256": source_identity["identity_sha256"],
        "prediction_freeze_raw_sha256": freeze_verification[
            "freeze_raw_sha256"
        ],
        "prediction_freeze_internal_sha256": freeze_verification[
            "freeze_internal_sha256"
        ],
        "phase_a_persistence_receipt_raw_sha256": label_input_verification[
            "phase_a_persistence_receipt_raw_sha256"
        ],
        "phase_a_persistence_receipt_internal_sha256": label_input_verification[
            "phase_a_persistence_receipt_internal_sha256"
        ],
        "phase_a_remote_identity_sha256": label_input_verification[
            "phase_a_remote_identity_sha256"
        ],
        "selected_checkpoint_identity": freeze_verification[
            "selected_checkpoint_identity"
        ],
        **_RESULT_COUNT_CONTRACT,
        "qualified_non_home_viewpoint_ids": qualified_non_home,
        "qualified_non_home_viewpoint_count": len(qualified_non_home),
        "home_calibration_passed": bool(calibrations[0]["passed"]),
        "unsafe_accepted_count": sum(
            int(item["unsafe_accepted_count"] or 0) for item in calibrations
        ),
        "catastrophic_accepted_count": sum(
            int(item["catastrophic_accepted_count"]) for item in calibrations
        ),
        "label_bundle_sha_verified_count": 50,
        "label_bundle_reopen_count_for_verification": 0,
        "test_array_read_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "physical_camera_actuation_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "manipulation_progression_count": 0,
        "checkpoint_write_count": 0,
    }


def score_calibrate_g2c(
    *,
    calibration_config_path: str | Path,
    training_config_path: str | Path,
    prediction_freeze_root: str | Path,
    calibration_privileged_input_root: str | Path,
    repository_root: str | Path,
    phase_a_persistence_receipt_path: str | Path,
    expected_phase_a_persistence_receipt_raw_sha256: str,
    output_root: str | Path,
    decision_exit_go: bool,
) -> dict[str, Any]:
    """Phase B：只接 frozen prediction 与 privileged labels，不接模型或 DATA。"""

    if decision_exit_go is not True:
        raise PermissionError("G2C formal calibration Phase B 仍为 HOLD")
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"G2C calibration result output 已存在: {output}")
    config = load_g2c_calibration_config(calibration_config_path)
    _verify_training_config_reference(config, training_config_path)
    freeze_verification = verify_g2c_calibration_prediction_freeze(
        calibration_config_path=calibration_config_path,
        output_root=prediction_freeze_root,
    )
    freeze_marker = _read_json(
        Path(prediction_freeze_root) / "prediction_freeze.json",
        "G2C calibration freeze marker",
    )
    source_identity = _git_source_identity(Path(repository_root))
    if (
        source_identity["git_commit"] != freeze_marker["source_git_commit"]
        or source_identity["identity_sha256"]
        != freeze_marker["source_identity_sha256"]
    ):
        raise RuntimeError("G2C calibration Phase B source 与 Phase A 漂移")
    persistence = verify_g2c_calibration_phase_a_persistence(
        calibration_config_path=calibration_config_path,
        prediction_freeze_root=prediction_freeze_root,
        persistence_receipt_path=phase_a_persistence_receipt_path,
        expected_receipt_raw_sha256=(
            expected_phase_a_persistence_receipt_raw_sha256
        ),
    )
    label_verification = validate_g2c_calibration_input_view(
        calibration_config_path=calibration_config_path,
        training_config_path=training_config_path,
        input_root=calibration_privileged_input_root,
        expected_role="calibration-privileged",
        verify_bundle_bytes=False,
        expected_prediction_freeze_internal_sha256=freeze_verification[
            "freeze_internal_sha256"
        ],
        expected_source_identity_sha256=source_identity["identity_sha256"],
        expected_persistence_receipt_raw_sha256=persistence[
            "receipt_raw_sha256"
        ],
        expected_persistence_receipt_internal_sha256=persistence[
            "receipt_internal_sha256"
        ],
        expected_remote_identity_sha256=persistence["remote_identity_sha256"],
    )
    # 在首次 privileged array open 前把已经公开验证的 prediction ledger 固定到
    # 当前进程内，避免 label 消费后再从可变路径读取造成 TOCTOU。
    predictions = _read_jsonl(
        Path(prediction_freeze_root) / "prediction_ledger.jsonl",
        "G2C calibration frozen predictions",
    )
    if len(predictions) != 550:
        raise RuntimeError("G2C calibration frozen predictions 必须是 550 rows")
    output.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < 1024**3:
        raise RuntimeError("G2C calibration Phase B 可用磁盘不足 1 GiB")
    output.mkdir(mode=0o700, parents=False, exist_ok=False)
    _atomic_json(output / "config_snapshot.json", config)
    _atomic_json(output / "source_identity.json", source_identity)
    _atomic_json(
        output / "prediction_freeze_verification.json", freeze_verification
    )
    _atomic_json(output / "label_input_verification.json", label_verification)
    phase_state = {
        "version": E018_P1_G2C_CALIBRATION_RESULT_VERSION,
        "status": "pre-label-freeze-verified",
        "config_sha256": config["config_sha256"],
        "prediction_freeze_internal_sha256": freeze_verification[
            "freeze_internal_sha256"
        ],
        "source_identity_sha256": source_identity["identity_sha256"],
        "label_array_consumed": False,
        "label_bundle_open_count": 0,
        "rerun_under_same_identity_allowed": False,
        "created_at_unix_ns": time.time_ns(),
    }
    _atomic_json(output / "phase_state.json", phase_state)
    label_open_started_at = time.time_ns()
    phase_state.update(
        {
            "status": "label-consumption-started",
            "label_array_consumed": True,
            "label_open_started_at_unix_ns": label_open_started_at,
        }
    )
    _atomic_json(output / "phase_state.json", phase_state)

    def record_open(count: int) -> None:
        phase_state["label_bundle_open_count"] = count
        _atomic_json(output / "phase_state.json", phase_state)

    try:
        labels = _load_calibration_labels(
            Path(calibration_privileged_input_root), on_bundle_open=record_open
        )
        scoring_rows: list[dict[str, Any]] = []
        for prediction in predictions:
            identity = (
                int(prediction["seed"]),
                int(prediction["sample_index"]),
                str(prediction["viewpoint_id"]),
            )
            if identity not in labels:
                raise RuntimeError("G2C calibration prediction/label identity 漂移")
            scoring_rows.append(
                _score_calibration_prediction(
                    prediction,
                    labels[identity],
                    config=config,
                    freeze_sha256=freeze_verification["freeze_internal_sha256"],
                )
            )
        if len(scoring_rows) != 550:
            raise RuntimeError("G2C calibration scoring 必须产生 550 rows")
        _atomic_jsonl(
            output / "calibration_scoring_ledger.jsonl", scoring_rows
        )
        calibrations = [
            calibrate_g2c_viewpoint(
                [row for row in scoring_rows if row["viewpoint_id"] == viewpoint],
                viewpoint_id=viewpoint,
            )
            for viewpoint in G2C_VIEW_ORDER
        ]
        if any(item["row_count"] != 50 for item in calibrations):
            raise RuntimeError("G2C calibration 每 viewpoint 必须审计 50 rows")
        _atomic_json(output / "viewpoint_calibrations.json", calibrations)
        summary = _calibration_summary(
            config=config,
            freeze_verification=freeze_verification,
            source_identity=source_identity,
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
        phase_state.update(
            {
                "status": "calibration-artifacts-written-pending-verification",
                "label_bundle_open_count": 50,
                "calibration_receipt_internal_sha256": receipt["receipt_sha256"],
                "precompletion_verification_sha256": None,
            }
        )
        _atomic_json(output / "phase_state.json", phase_state)
        provisional = _verify_g2c_calibration_result(
            calibration_config_path=calibration_config_path,
            prediction_freeze_root=prediction_freeze_root,
            output_root=output,
            expected_phase_status=(
                "calibration-artifacts-written-pending-verification"
            ),
        )
        phase_state["status"] = "complete-calibration-score"
        phase_state["precompletion_verification_sha256"] = provisional[
            "verification_sha256"
        ]
        _atomic_json(output / "phase_state.json", phase_state)
        return verify_g2c_calibration_result(
            calibration_config_path=calibration_config_path,
            prediction_freeze_root=prediction_freeze_root,
            output_root=output,
        )
    except Exception as error:
        failure = {
            "version": E018_P1_G2C_CALIBRATION_RESULT_VERSION,
            "status": "consumed-calibration-failure",
            "config_sha256": config["config_sha256"],
            "prediction_freeze_internal_sha256": freeze_verification[
                "freeze_internal_sha256"
            ],
            "source_identity_sha256": source_identity["identity_sha256"],
            "label_array_consumed": True,
            "label_bundle_open_count": int(
                phase_state.get("label_bundle_open_count", 0)
            ),
            "rerun_under_same_identity_allowed": False,
            "error_type": type(error).__name__,
            "error_message": str(error),
            "failed_at_unix_ns": time.time_ns(),
        }
        failure["failure_sha256"] = canonical_sha256(failure)
        _atomic_json(output / "consumed_failure.json", failure)
        phase_state["status"] = failure["status"]
        phase_state["failure_sha256"] = failure["failure_sha256"]
        _atomic_json(output / "phase_state.json", phase_state)
        raise


def _verify_g2c_calibration_result(
    *,
    calibration_config_path: str | Path,
    prediction_freeze_root: str | Path,
    output_root: str | Path,
    expected_phase_status: str,
) -> dict[str, Any]:
    """公开重算已消费结果；绝不再次接收或打开 label bundle。"""

    config = load_g2c_calibration_config(calibration_config_path)
    freeze = verify_g2c_calibration_prediction_freeze(
        calibration_config_path=calibration_config_path,
        output_root=prediction_freeze_root,
    )
    root = Path(output_root)
    _assert_unlinked_regular_file_tree(root, name="G2C calibration result")
    source = _read_json(root / "source_identity.json", "G2C calibration source")
    if (
        source.get("identity_sha256")
        != canonical_sha256(
            {
                "git_commit": source.get("git_commit"),
                "source_tree_sha256": source.get("source_tree_sha256"),
            }
        )
        or source.get("git_commit") != freeze["source_git_commit"]
        or source.get("identity_sha256") != freeze["source_identity_sha256"]
    ):
        raise RuntimeError("G2C calibration Phase A/B source 漂移")
    if _read_json(root / "config_snapshot.json", "G2C calibration config snapshot") != config:
        raise RuntimeError("G2C calibration result config snapshot 漂移")
    stored_freeze = _read_json(
        root / "prediction_freeze_verification.json",
        "G2C calibration freeze verification",
    )
    if stored_freeze != freeze:
        raise RuntimeError("G2C calibration stored freeze verification 漂移")
    label_verification = _read_json(
        root / "label_input_verification.json",
        "G2C calibration label input verification",
    )
    label_verification_keys = {
        "version",
        "verified",
        "role",
        "split",
        "seed_count",
        "sample_count",
        "bundle_bytes_verified",
        "prediction_freeze_internal_sha256",
        "source_identity_sha256",
        "phase_a_persistence_receipt_raw_sha256",
        "phase_a_persistence_receipt_internal_sha256",
        "phase_a_remote_identity_sha256",
        "privileged_source_bundle_copy_count",
        "privileged_label_array_open_count",
        "receipt_raw_sha256",
        "receipt_internal_sha256",
        "inventory",
        "verification_sha256",
    }
    _require_exact_keys(
        label_verification,
        label_verification_keys,
        "G2C calibration label input verification",
    )
    label_unsigned = dict(label_verification)
    label_internal = label_unsigned.pop("verification_sha256", None)
    label_inventory = label_verification.get("inventory")
    for field in (
        "phase_a_persistence_receipt_raw_sha256",
        "phase_a_persistence_receipt_internal_sha256",
        "phase_a_remote_identity_sha256",
    ):
        _require_sha256(label_verification.get(field), f"G2C calibration {field}")
    if (
        label_internal != canonical_sha256(label_unsigned)
        or label_verification.get("verified") is not True
        or label_verification.get("role") != "calibration-privileged"
        or label_verification.get("split") != "calibration"
        or label_verification.get("seed_count") != 50
        or label_verification.get("sample_count") != 550
        or label_verification.get("bundle_bytes_verified") is not False
        or label_verification.get("prediction_freeze_internal_sha256")
        != freeze["freeze_internal_sha256"]
        or label_verification.get("source_identity_sha256")
        != source["identity_sha256"]
        or label_verification.get("privileged_source_bundle_copy_count") != 50
        or label_verification.get("privileged_label_array_open_count") != 0
        or not isinstance(label_inventory, Mapping)
        or label_inventory.get("deployable_inventory_sha256") is not None
        or label_inventory.get("privileged_inventory_sha256")
        != config["data_parent"]["privileged_inventory_sha256"]
    ):
        raise RuntimeError("G2C calibration label input binding/count 漂移")
    predictions = _read_jsonl(
        Path(prediction_freeze_root) / "prediction_ledger.jsonl",
        "G2C calibration frozen predictions",
    )
    scoring_rows = _read_jsonl(
        root / "calibration_scoring_ledger.jsonl",
        "G2C calibration scoring ledger",
    )
    if len(predictions) != 550 or len(scoring_rows) != 550:
        raise RuntimeError("G2C calibration prediction/scoring count 漂移")
    recomputed_rows: list[dict[str, Any]] = []
    for index, (prediction, scoring) in enumerate(
        zip(predictions, scoring_rows, strict=True)
    ):
        _require_exact_keys(
            scoring,
            _CALIBRATION_SCORING_ROW_KEYS,
            "G2C calibration scoring row",
        )
        label = {
            "gt_observable": scoring["gt_observable"],
            "gt_object_position_base_m": scoring["gt_object_position_base_m"],
            "gt_object_exists": scoring["gt_object_exists"],
            "is_grasped": scoring["is_grasped"],
            "robot_object_contact_force_n": scoring[
                "robot_object_contact_force_n"
            ],
        }
        expected = _score_calibration_prediction(
            prediction,
            label,
            config=config,
            freeze_sha256=freeze["freeze_internal_sha256"],
        )
        if scoring != expected or scoring.get("row_index") != index:
            raise RuntimeError("G2C calibration scoring primitive/derived field 重算漂移")
        recomputed_rows.append(expected)
    recomputed_calibrations = [
        calibrate_g2c_viewpoint(
            [row for row in recomputed_rows if row["viewpoint_id"] == viewpoint],
            viewpoint_id=viewpoint,
        )
        for viewpoint in G2C_VIEW_ORDER
    ]
    stored_calibrations = _read_json_array(
        root / "viewpoint_calibrations.json", "G2C viewpoint calibrations"
    )
    if stored_calibrations != recomputed_calibrations:
        raise RuntimeError("G2C viewpoint calibration 重算漂移")
    expected_summary = _calibration_summary(
        config=config,
        freeze_verification=freeze,
        source_identity=source,
        calibrations=recomputed_calibrations,
        label_input_verification=label_verification,
    )
    summary = _read_json(root / "calibration_summary.json", "G2C calibration summary")
    if summary != expected_summary:
        raise RuntimeError("G2C calibration summary 重算漂移")
    receipt_path = root / "calibration_receipt.json"
    receipt = _read_json(receipt_path, "G2C calibration receipt")
    expected_artifact_sha = {
        name: file_sha256(root / name) for name in _RESULT_ARTIFACTS
    }
    expected_receipt = {
        **expected_summary,
        "artifact_sha256": expected_artifact_sha,
    }
    expected_receipt["receipt_sha256"] = canonical_sha256(expected_receipt)
    if receipt != expected_receipt:
        raise RuntimeError("G2C calibration receipt 重算/identity 漂移")
    phase_state = _read_json(root / "phase_state.json", "G2C calibration phase state")
    phase_keys = {
        "version",
        "status",
        "config_sha256",
        "prediction_freeze_internal_sha256",
        "source_identity_sha256",
        "label_array_consumed",
        "label_bundle_open_count",
        "rerun_under_same_identity_allowed",
        "created_at_unix_ns",
        "label_open_started_at_unix_ns",
        "calibration_receipt_internal_sha256",
        "precompletion_verification_sha256",
    }
    _require_exact_keys(phase_state, phase_keys, "G2C calibration phase state")
    if (
        phase_state.get("version") != E018_P1_G2C_CALIBRATION_RESULT_VERSION
        or phase_state.get("status") != expected_phase_status
        or phase_state.get("config_sha256") != config["config_sha256"]
        or phase_state.get("prediction_freeze_internal_sha256")
        != freeze["freeze_internal_sha256"]
        or phase_state.get("source_identity_sha256")
        != source["identity_sha256"]
        or phase_state.get("label_array_consumed") is not True
        or phase_state.get("label_bundle_open_count") != 50
        or phase_state.get("rerun_under_same_identity_allowed") is not False
        or not isinstance(phase_state.get("created_at_unix_ns"), int)
        or not isinstance(phase_state.get("label_open_started_at_unix_ns"), int)
        or phase_state["label_open_started_at_unix_ns"]
        < phase_state["created_at_unix_ns"]
        or phase_state.get("calibration_receipt_internal_sha256")
        != receipt["receipt_sha256"]
    ):
        raise RuntimeError("G2C calibration consumed/completion state 漂移")
    precompletion_sha = phase_state["precompletion_verification_sha256"]
    if expected_phase_status == "calibration-artifacts-written-pending-verification":
        if precompletion_sha is not None:
            raise RuntimeError("G2C provisional calibration state 不得预写 verifier SHA")
    elif expected_phase_status == "complete-calibration-score":
        _require_sha256(precompletion_sha, "G2C calibration precompletion verifier")
    else:
        raise ValueError("G2C calibration expected phase status 未冻结")
    for name, sha in expected_artifact_sha.items():
        if receipt["artifact_sha256"].get(name) != sha:
            raise RuntimeError(f"G2C calibration artifact SHA 漂移: {name}")
    total_bytes = _verify_exact_regular_file_tree(
        root,
        expected_files={
            *_RESULT_ARTIFACTS,
            "calibration_receipt.json",
            "phase_state.json",
        },
        name="G2C calibration result",
    )
    if total_bytes > config["protocol"]["budgets"]["formal_artifact_bytes_max"]:
        raise RuntimeError("G2C calibration result artifact byte budget 漂移")
    result = {
        "version": E018_P1_G2C_CALIBRATION_RESULT_VERSION,
        "status": summary["status"],
        "verified": True,
        "protocol_valid": True,
        "gate_passed": summary["gate_passed"],
        "config_sha256": config["config_sha256"],
        "source_git_commit": source["git_commit"],
        "source_identity_sha256": source["identity_sha256"],
        "prediction_freeze_internal_sha256": freeze["freeze_internal_sha256"],
        "receipt_raw_sha256": file_sha256(receipt_path),
        "receipt_internal_sha256": receipt["receipt_sha256"],
        "qualified_non_home_viewpoint_ids": summary[
            "qualified_non_home_viewpoint_ids"
        ],
        "qualified_non_home_viewpoint_count": summary[
            "qualified_non_home_viewpoint_count"
        ],
        "label_bundle_reopen_count_for_verification": 0,
        "artifact_bytes": total_bytes,
    }
    result["verification_sha256"] = canonical_sha256(result)
    return result


def verify_g2c_calibration_result(
    *,
    calibration_config_path: str | Path,
    prediction_freeze_root: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """验证 complete Phase B；接口没有 label/model/checkpoint/DATA path。"""

    return _verify_g2c_calibration_result(
        calibration_config_path=calibration_config_path,
        prediction_freeze_root=prediction_freeze_root,
        output_root=output_root,
        expected_phase_status="complete-calibration-score",
    )


def run_g2c_calibration_synthetic_gpu_smoke(
    *,
    calibration_config_path: str | Path,
    selected_checkpoint_path: str | Path,
    output_root: str | Path,
) -> dict[str, Any]:
    """用非 canonical 合成输入验证 selected-only 550-row GPU 数据流。"""

    import torch

    from robot_vla.observation import OBSERVATION_V2_FRAME_STATE_DIM
    from robot_vla.precision.checkpoint import (
        PrecisionCheckpointRole,
        load_precision_checkpoint,
    )

    config = load_g2c_calibration_config(calibration_config_path)
    output = Path(output_root)
    if output.exists():
        raise FileExistsError(f"G2C calibration synthetic smoke 已存在: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("G2C calibration synthetic GPU smoke 要求 CUDA")
    parent = G2C_CALIBRATION_SELECTION_PARENT
    loaded = load_precision_checkpoint(
        selected_checkpoint_path,
        expected_checkpoint_sha256=parent["checkpoint_sha256"],
        expected_provenance_sha256=parent["provenance_sha256"],
        expected_role=PrecisionCheckpointRole.FORMAL_TRAINING,
    )
    if (
        loaded.receipt.parameter_state_sha256 != parent["parameter_state_sha256"]
        or loaded.receipt.model_config_sha256 != parent["model_config_sha256"]
    ):
        raise RuntimeError("G2C calibration smoke selected checkpoint identity 漂移")
    checkpoint_identity = {
        "candidate_id": parent["candidate_id"],
        "epoch": parent["epoch"],
        "checkpoint_sha256": parent["checkpoint_sha256"],
        "parameter_state_sha256": parent["parameter_state_sha256"],
        "provenance_sha256": parent["provenance_sha256"],
        "model_config_sha256": parent["model_config_sha256"],
    }
    output.mkdir(mode=0o700, parents=True, exist_ok=False)
    _atomic_json(output / "config_snapshot.json", config)
    _atomic_json(
        output / "selected_checkpoint_verification.json", checkpoint_identity
    )
    device = torch.device("cuda")
    model = loaded.model.to(device)
    model.eval()
    intrinsic = np.asarray(
        [[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    base_from_camera = np.eye(4, dtype=np.float64)
    base_from_camera[:3, :3] = np.diag([1.0, -1.0, -1.0])
    base_from_camera[:3, 3] = [0.0, 0.0, 1.0]
    smoke_seeds = tuple(range(910001, 910051))
    identities = [
        (seed, sample_index, viewpoint)
        for seed in smoke_seeds
        for sample_index, viewpoint in enumerate(G2C_VIEW_ORDER)
    ]
    rows: list[dict[str, Any]] = []
    batch_sizes: list[int] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for batch_index, start in enumerate(range(0, len(identities), 32)):
            batch_identity = identities[start : start + 32]
            samples = []
            for offset, (seed, sample_index, viewpoint) in enumerate(batch_identity):
                synthetic_id = {
                    "role": "noncanonical-synthetic-calibration-smoke/v1",
                    "seed": seed,
                    "sample_index": sample_index,
                    "viewpoint_id": viewpoint,
                    "row_index": start + offset,
                }
                samples.append(
                    {
                        "model_inputs": {
                            "rgb_external": np.zeros(
                                (128, 128, 3), dtype=np.uint8
                            ),
                            "structured_state": np.zeros(
                                OBSERVATION_V2_FRAME_STATE_DIM,
                                dtype=np.float32,
                            ),
                            "geometric_motion": np.zeros(4, dtype=np.float32),
                        },
                        "capture": {
                            "seed": seed,
                            "split": "calibration",
                            "sample_index": sample_index,
                            "viewpoint_id": viewpoint,
                            "input_sha256": canonical_sha256(synthetic_id),
                            "external_intrinsic_cv": intrinsic,
                            "base_from_external_camera_cv": base_from_camera,
                        },
                    }
                )
            image = np.zeros((len(samples), 3, 128, 128), dtype=np.float32)
            state = np.zeros(
                (len(samples), OBSERVATION_V2_FRAME_STATE_DIM), dtype=np.float32
            )
            motion = np.zeros((len(samples), 4), dtype=np.float32)
            with torch.autocast(
                device_type="cuda", dtype=torch.bfloat16, enabled=True
            ):
                model_output = model(
                    torch.from_numpy(image).to(device),
                    torch.from_numpy(state).to(device),
                    torch.from_numpy(motion).to(device),
                )
            batch_rows = _prediction_rows_for_batch(
                samples=samples,
                output=model_output,
                candidate_id=parent["candidate_id"],
                epoch=parent["epoch"],
                checkpoint_identity=checkpoint_identity,
                global_start=start,
                batch_index=batch_index,
            )
            for row in batch_rows:
                row.update(
                    {
                        "version": E018_P1_G2C_CALIBRATION_FREEZE_VERSION,
                        "phase": (
                            "deployable-calibration-before-privileged-label-open/v1"
                        ),
                        "external_intrinsic_cv": intrinsic.tolist(),
                        "base_from_external_camera_cv": base_from_camera.tolist(),
                        "deployable_safety": {
                            "eligible_capture": True,
                            "finger_force_n": [0.0, 0.0],
                            "finger_force_valid": True,
                            "raw_gripper_opening_ratio": 1.0,
                            "arm_joint_drift_rad": 0.0,
                            "tcp_position_drift_m": 0.0,
                            "tcp_orientation_drift_rad": 0.0,
                            "rgb_timestamp_s": 0.0,
                            "pose_timestamp_s": 0.0,
                            "camera_position_tracking_error_m": 0.0,
                            "camera_orientation_tracking_error_rad": 0.0,
                            "rotation_projection_error_frobenius": 0.0,
                        },
                    }
                )
                _validate_prediction_row_mechanics(row, config=config)
            rows.extend(batch_rows)
            batch_sizes.append(len(batch_rows))
    if len(rows) != 550 or batch_sizes != [32] * 17 + [6]:
        raise RuntimeError("G2C calibration smoke row/batch count 漂移")
    _atomic_jsonl(output / "synthetic_prediction_ledger.jsonl", rows)
    scoring_rows: list[dict[str, Any]] = []
    for row in rows:
        predicted = row["predicted_object_position_base_m"]
        gt_position = [0.0, 0.0, 0.02] if predicted is None else predicted
        scoring_rows.append(
            _score_calibration_prediction(
                row,
                {
                    "gt_observable": predicted is not None,
                    "gt_object_position_base_m": gt_position,
                    "gt_object_exists": True,
                    "is_grasped": False,
                    "robot_object_contact_force_n": 0.0,
                },
                config=config,
                freeze_sha256="0" * 64,
            )
        )
    _atomic_jsonl(output / "synthetic_scoring_ledger.jsonl", scoring_rows)
    calibrations = [
        calibrate_g2c_viewpoint(
            [row for row in scoring_rows if row["viewpoint_id"] == viewpoint],
            viewpoint_id=viewpoint,
        )
        for viewpoint in G2C_VIEW_ORDER
    ]
    _atomic_json(output / "synthetic_viewpoint_calibrations.json", calibrations)
    singular_rows = [
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
    singular_no_go = calibrate_g2c_viewpoint(
        singular_rows, viewpoint_id="LEFT_LOW__CENTER"
    )
    _atomic_json(output / "singular_psd_no_go.json", singular_no_go)
    del rows, scoring_rows, model_output, model, loaded, samples, image, state, motion
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - started
    files_before_summary = (
        "config_snapshot.json",
        "selected_checkpoint_verification.json",
        "synthetic_prediction_ledger.jsonl",
        "synthetic_scoring_ledger.jsonl",
        "synthetic_viewpoint_calibrations.json",
        "singular_psd_no_go.json",
    )
    bytes_before_summary = sum((output / name).stat().st_size for name in files_before_summary)
    summary = {
        "version": E018_P1_G2C_CALIBRATION_RESULT_VERSION,
        "status": "complete-synthetic-gpu-smoke-pass",
        "classification": "noncanonical-synthetic-no-promotion/v1",
        "config_sha256": config["config_sha256"],
        "synthetic_seed_start": smoke_seeds[0],
        "synthetic_seed_end": smoke_seeds[-1],
        "synthetic_seed_count": 50,
        "synthetic_prediction_row_count": 550,
        "synthetic_scoring_row_count": 550,
        "selected_checkpoint_load_count": 1,
        "selected_checkpoint_identity": checkpoint_identity,
        "model_forward_batch_count": 18,
        "batch_sizes": batch_sizes,
        "model_and_inference_context_destroyed": True,
        "singular_psd_protocol_valid_no_go": bool(
            singular_no_go["status"] == "calibration-no-go"
            and "nonfinite_conformal_quantile_or_scale"
            in singular_no_go["failure_reasons"]
        ),
        "formal_calibration_gate_evaluated": False,
        "canonical_calibration_deployable_bundle_open_count": 0,
        "canonical_calibration_privileged_label_bundle_open_count": 0,
        "test_array_read_count": 0,
        "memory_read_count": 0,
        "memory_write_count": 0,
        "runtime_camera_actuation_count": 0,
        "physical_camera_actuation_count": 0,
        "arm_motion_command_count": 0,
        "gripper_close_command_count": 0,
        "manipulation_progression_count": 0,
        "checkpoint_write_count": 0,
        "gpu_elapsed_s": elapsed,
        "artifact_bytes_before_summary": bytes_before_summary,
        "artifact_sha256": {
            name: file_sha256(output / name) for name in files_before_summary
        },
    }
    if (
        elapsed
        > config["protocol"]["budgets"]["implementation_smoke_gpu_seconds_max"]
        or bytes_before_summary
        > config["protocol"]["budgets"][
            "implementation_smoke_artifact_bytes_max"
        ]
        or summary["singular_psd_protocol_valid_no_go"] is not True
    ):
        raise RuntimeError("G2C calibration synthetic smoke budget/protocol 未通过")
    summary["summary_sha256"] = canonical_sha256(summary)
    _atomic_json(output / "smoke_summary.json", summary)
    total_bytes = _verify_exact_regular_file_tree(
        output,
        expected_files={*files_before_summary, "smoke_summary.json"},
        name="G2C calibration synthetic smoke",
    )
    if total_bytes > config["protocol"]["budgets"][
        "implementation_smoke_artifact_bytes_max"
    ]:
        raise RuntimeError("G2C calibration synthetic smoke 完整 artifact 超过 1 GiB")
    return {**summary, "total_artifact_bytes": total_bytes}


__all__ = [
    "E018_P1_G2C_CALIBRATION_CHECK_EVIDENCE_VERSION",
    "E018_P1_G2C_CALIBRATION_COMPLETION_MARKER_VERSION",
    "E018_P1_G2C_CALIBRATION_CONFIG_VERSION",
    "E018_P1_G2C_CALIBRATION_FREEZE_VERSION",
    "E018_P1_G2C_CALIBRATION_INPUT_VIEW_VERSION",
    "E018_P1_G2C_CALIBRATION_RESULT_VERSION",
    "G2C_CALIBRATION_SELECTION_PARENT",
    "build_g2c_calibration_config",
    "build_g2c_calibration_phase_a_completion_marker",
    "finalize_g2c_calibration_phase_a_persistence",
    "g2c_calibration_protocol",
    "load_g2c_calibration_config",
    "prepare_g2c_calibration_deployable_view",
    "prepare_g2c_calibration_privileged_view",
    "record_g2c_calibration_phase_a_check_evidence",
    "run_g2c_calibration_prediction_freeze",
    "run_g2c_calibration_synthetic_gpu_smoke",
    "score_calibrate_g2c",
    "validate_g2c_calibration_input_view",
    "verify_g2c_calibration_prediction_freeze",
    "verify_g2c_calibration_phase_a_persistence",
    "verify_g2c_calibration_result",
]
