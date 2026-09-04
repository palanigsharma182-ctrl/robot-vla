from __future__ import annotations

import inspect
import json
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest

import robot_vla.precision.e018_p1_g2c as g2c
import robot_vla.precision.e018_p1_g2c_data as g2c_data
from robot_vla.observation import OBSERVATION_V2_FRAME_STATE_DIM
from robot_vla.precision.e018_p1_g2c import (
    G2C_CANDIDATE_EPOCHS,
    G2C_CANDIDATE_IDS,
    assert_g2c_prediction_ledger_deployable_only,
    build_g2c_train_config_payload,
    calibrate_g2c_viewpoint,
    freeze_g2c_prediction_ledger,
    g2c_dynamic_qualification_plan,
    g2c_training_protocol,
    load_frozen_g2c_prediction_ledger,
    select_g2c_checkpoint,
    summarize_g2c_model_val_view,
    validate_g2c_dynamic_qualification_counters,
)
from robot_vla.precision.e018_p1_g2c_data import (
    G2C_SMOKE_SPLIT,
    G2C_VIEW_ORDER,
    G2CDeployableDataset,
    G2CFrontTrainingDataset,
    audit_g2c_lifecycle,
    g2c_split_seeds,
    load_e018_p1_g2c_data_config,
    validate_g2c_seed_bundle,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs/e018_p1_g2c_front_provider_data_development_v1.json"
G0C_CONFIG = ROOT / "configs/e018_p1_g0c_rotated_motion_development_v1.json"


def _config() -> dict:
    return json.loads(CONFIG.read_text(encoding="utf-8"))


def _write_config(tmp_path: Path, value: dict) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _bundles(seed: int = 76801) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    count = len(G2C_VIEW_ORDER)
    matrices = np.repeat(np.eye(4, dtype=np.float64)[None], count, axis=0)
    intrinsic = np.repeat(
        np.asarray([[100.0, 0.0, 64.0], [0.0, 100.0, 64.0], [0.0, 0.0, 1.0]])[
            None
        ],
        count,
        axis=0,
    )
    deployable = {
        "sample_index": np.arange(count, dtype=np.int64),
        "seed": np.full(count, seed, dtype=np.int64),
        "viewpoint_id": np.asarray(G2C_VIEW_ORDER, dtype="<U32"),
        "eligible_capture": np.ones(count, dtype=np.bool_),
        "rgb_external": np.zeros((count, 128, 128, 3), dtype=np.uint8),
        "physical_proprio": np.zeros((count, 15), dtype=np.float32),
        "structured_state": np.zeros(
            (count, OBSERVATION_V2_FRAME_STATE_DIM), dtype=np.float32
        ),
        "geometric_motion": np.zeros((count, 4), dtype=np.float32),
        "base_from_tcp": matrices.copy(),
        "base_from_external_camera_cv": matrices.copy(),
        "actual_world_from_external_camera_gl": matrices.copy(),
        "external_intrinsic_cv": intrinsic,
        "rgb_timestamp_s": np.arange(count, dtype=np.float64) * 1e-6 + 0.25,
        "pose_timestamp_s": np.arange(count, dtype=np.float64) * 1e-6 + 0.25,
        "finger_force_n": np.zeros((count, 2), dtype=np.float32),
        "finger_force_valid": np.ones(count, dtype=np.bool_),
        "raw_gripper_opening_ratio": np.ones(count, dtype=np.float32),
        "arm_joint_drift_rad": np.zeros(count, dtype=np.float64),
        "tcp_position_drift_m": np.zeros(count, dtype=np.float64),
        "tcp_orientation_drift_rad": np.zeros(count, dtype=np.float64),
        "camera_position_tracking_error_m": np.zeros(count, dtype=np.float64),
        "camera_orientation_tracking_error_rad": np.zeros(count, dtype=np.float64),
        "rotation_projection_error_frobenius": np.zeros(count, dtype=np.float64),
    }
    labels = {
        "source_sample_index": np.arange(count, dtype=np.int64),
        "seed": np.full(count, seed, dtype=np.int64),
        "viewpoint_id": np.asarray(G2C_VIEW_ORDER, dtype="<U32"),
        "object_position_base_m": np.repeat(
            np.asarray([[0.1, 0.1, 0.02]], dtype=np.float32), count, axis=0
        ),
        "goal_position_base_m": np.repeat(
            np.asarray([[0.2, 0.2, 0.02]], dtype=np.float32), count, axis=0
        ),
        "object_mask": np.zeros((count, 128, 128), dtype=np.bool_),
        "goal_mask": np.zeros((count, 128, 128), dtype=np.bool_),
        "normalized_uv": np.zeros((count, 2, 2), dtype=np.float32),
        "keypoint_projection_valid": np.zeros((count, 2), dtype=np.bool_),
        "keypoint_observable": np.zeros((count, 2), dtype=np.bool_),
        "object_exists": np.ones(count, dtype=np.bool_),
        "goal_exists": np.ones(count, dtype=np.bool_),
        "is_grasped": np.zeros(count, dtype=np.bool_),
        "robot_object_contact_force_n": np.zeros(count, dtype=np.float32),
        "geometry_roundtrip_error_m": np.zeros((count, 2), dtype=np.float64),
    }
    return deployable, labels


def _manifest_rows(
    root: Path,
    deployable: dict[str, np.ndarray],
    labels: dict[str, np.ndarray],
    *,
    seed: int,
) -> None:
    deployable_file = Path("bundles") / G2C_SMOKE_SPLIT / f"seed-{seed:06d}.npz"
    label_file = Path("bundles") / G2C_SMOKE_SPLIT / f"seed-{seed:06d}.npz"
    deployable_path = root / "deployable" / deployable_file
    label_path = root / "privileged_labels" / label_file
    g2c_data._atomic_npz(deployable_path, deployable)
    g2c_data._atomic_npz(label_path, labels)
    source_sha = g2c_data.file_sha256(deployable_path)
    common = {
        "manifest_schema_version": g2c_data.G2C_MANIFEST_SCHEMA_VERSION,
        "split": G2C_SMOKE_SPLIT,
        "seed": seed,
        "sample_count": len(G2C_VIEW_ORDER),
        "view_order": list(G2C_VIEW_ORDER),
    }
    g2c_data._atomic_jsonl(
        root / "deployable/manifest.jsonl",
        [
            {
                **common,
                "schema_version": g2c_data.G2C_DEPLOYABLE_SCHEMA_VERSION,
                "file": deployable_file.as_posix(),
                "sha256": source_sha,
                "contains_privileged_labels": False,
            }
        ],
    )
    g2c_data._atomic_jsonl(
        root / "privileged_labels/manifest.jsonl",
        [
            {
                **common,
                "schema_version": g2c_data.G2C_LABEL_SCHEMA_VERSION,
                "file": label_file.as_posix(),
                "sha256": g2c_data.file_sha256(label_path),
                "source_deployable_file": deployable_file.as_posix(),
                "source_deployable_sha256": source_sha,
                "contains_model_input_rgb": False,
            }
        ],
    )


def _metric_rows(
    viewpoint_id: str,
    *,
    count: int,
    error: float = 0.001,
    candidate_id: str = "W",
    epoch: int = 5,
) -> list[dict]:
    return [
        {
            "candidate_id": candidate_id,
            "epoch": epoch,
            "viewpoint_id": viewpoint_id,
            "gt_observable": True,
            "predicted_observable": True,
            "geometry_valid": True,
            "world_xyz_error_m": error,
        }
        for _ in range(count)
    ]


def test_config_freezes_four_disjoint_splits_and_smoke_only_seeds() -> None:
    config = load_e018_p1_g2c_data_config(
        CONFIG, parent_g0c_config_path=G0C_CONFIG
    )
    seeds = g2c_split_seeds(config)
    assert {name: len(values) for name, values in seeds.items()} == {
        "train": 400,
        "model_val": 100,
        "calibration": 50,
        "qualification": 50,
    }
    assert config["sampling"]["smoke_only_seeds"] == [76801, 76802, 76803, 76804]
    assert config["engineering_smoke"]["training_seeds"] == [76801, 76802, 76803]
    assert config["engineering_smoke"]["prediction_freeze_seed"] == 76804
    assert config["sampling"]["test_split"] is None


@pytest.mark.parametrize(
    ("section", "key", "value", "message"),
    [
        ("scope", "test_trajectory_array_read_allowed", True, "scope"),
        ("capture", "reset_warmup_ticks", 4, "capture"),
        ("execution", "formal_training_allowed", True, "execution"),
        ("engineering_smoke", "prediction_freeze_seed", 76803, "smoke"),
    ],
)
def test_config_rejects_permission_or_lifecycle_drift(
    tmp_path: Path, section: str, key: str, value: object, message: str
) -> None:
    config = _config()
    config[section][key] = value
    with pytest.raises(ValueError, match=message):
        load_e018_p1_g2c_data_config(
            _write_config(tmp_path, config), parent_g0c_config_path=G0C_CONFIG
        )


def test_seed_bundle_has_disjoint_model_and_privileged_arrays() -> None:
    deployable, labels = _bundles()
    validate_g2c_seed_bundle(deployable, labels, expected_seed=76801)
    assert set(deployable).isdisjoint(
        {"object_mask", "goal_mask", "object_position_base_m", "is_grasped"}
    )
    assert "rgb_external" not in labels
    audit = audit_g2c_lifecycle(deployable, labels, config=_config())
    assert audit["passed"] is True
    assert all(count == 0 for count in audit["violation_counts"].values())


def test_lifecycle_is_fail_whole_and_does_not_delete_bad_row() -> None:
    deployable, labels = _bundles()
    labels["is_grasped"][0] = True
    deployable["finger_force_n"][0] = [1.0, 1.0]
    audit = audit_g2c_lifecycle(deployable, labels, config=_config())
    assert audit["passed"] is False
    assert audit["eligible_capture_count"] == 11
    assert audit["violation_counts"]["is_grasped"] == 1
    assert audit["violation_counts"]["left_finger_force"] == 1
    assert audit["fail_whole_split_required"] is True


def test_deployable_dataset_does_not_need_to_open_label_bundle(tmp_path: Path) -> None:
    deployable, labels = _bundles()
    _manifest_rows(tmp_path, deployable, labels, seed=76801)
    label_path = tmp_path / "privileged_labels/bundles/engineering_smoke/seed-076801.npz"
    label_path.unlink()
    dataset = G2CDeployableDataset(tmp_path, G2C_SMOKE_SPLIT)
    assert len(dataset) == 11
    assert dataset[0]["model_inputs"]["rgb_external"].shape == (128, 128, 3)


def test_training_dataset_rejects_model_val_before_prediction_freeze(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="禁止"):
        G2CFrontTrainingDataset(tmp_path, "model_val")


def test_prediction_ledger_freezes_before_privileged_fields(tmp_path: Path) -> None:
    rows = [{"candidate_id": "W", "seed": 76804, "prediction": 0.8}]
    marker = freeze_g2c_prediction_ledger(
        tmp_path,
        rows=rows,
        config_sha256="a" * 64,
        data_identity_sha256="b" * 64,
    )
    rows[0]["prediction"] = 0.1
    loaded, loaded_marker = load_frozen_g2c_prediction_ledger(
        tmp_path,
        config_sha256="a" * 64,
        data_identity_sha256="b" * 64,
    )
    assert loaded[0]["prediction"] == 0.8
    assert loaded_marker["freeze_sha256"] == marker["freeze_sha256"]
    with pytest.raises(ValueError, match="privileged"):
        assert_g2c_prediction_ledger_deployable_only(
            [{"prediction": {"gt_object_position": [0.0, 0.0, 0.02]}}]
        )
    for key in (
        "object_position_base_m",
        "goal_position_base_m",
        "object_mask",
        "goal_mask",
        "keypoint_observable",
        "keypoint_projection_valid",
    ):
        with pytest.raises(ValueError, match="privileged"):
            assert_g2c_prediction_ledger_deployable_only(
                [{"prediction": {key: [0.0]}}]
            )
    assert_g2c_prediction_ledger_deployable_only(
        [
            {
                "predicted_object_position_base_m": [0.1, 0.2, 0.02],
                "object_mask_probability_at_prediction": 0.8,
                "predicted_observable": True,
            }
        ]
    )


def test_phase_b_signature_cannot_receive_model_or_in_memory_predictions() -> None:
    parameters = inspect.signature(g2c._score_smoke_after_prediction_freeze).parameters
    assert "model" not in parameters
    assert "predictions" not in parameters
    assert "dataset" not in parameters


def test_training_protocol_separates_init_rng_from_shared_sampler_rng() -> None:
    protocol = g2c_training_protocol()
    assert protocol["training"]["shared_sampler_seed"] == 18020
    assert protocol["candidates"]["W"]["initialization_seed"] == 18021
    assert protocol["candidates"]["S"]["initialization_seed"] == 18022
    assert protocol["training"]["require_identical_per_epoch_shuffle"] is True
    assert protocol["model_validation"][
        "minimum_observable_positive_support_per_viewpoint"
    ] == 30


def test_support_below_30_is_view_ineligible_not_protocol_invalid() -> None:
    summary = summarize_g2c_model_val_view(
        _metric_rows("LEFT_LOW__CENTER", count=29),
        viewpoint_id="LEFT_LOW__CENTER",
    )
    assert summary["eligible"] is False
    assert summary["protocol_invalid"] is False
    assert summary["ineligibility_reasons"] == [
        "observable_positive_support_below_30"
    ]


def test_no_eligible_checkpoint_is_protocol_valid_negative() -> None:
    rows = []
    losses = {}
    for candidate in G2C_CANDIDATE_IDS:
        for epoch in G2C_CANDIDATE_EPOCHS:
            losses[(candidate, epoch)] = 1.0
            for viewpoint in G2C_VIEW_ORDER:
                rows.extend(
                    _metric_rows(
                        viewpoint,
                        count=1,
                        candidate_id=candidate,
                        epoch=epoch,
                    )
                )
    result = select_g2c_checkpoint(rows, validation_losses=losses)
    assert result["protocol_valid"] is True
    assert result["selected"] is None
    assert result["status"] == "complete-model-val-protocol-valid-negative"


def test_checkpoint_selection_prefers_more_eligible_views_then_p90() -> None:
    rows = []
    losses = {}
    for candidate in G2C_CANDIDATE_IDS:
        for epoch in G2C_CANDIDATE_EPOCHS:
            losses[(candidate, epoch)] = 1.0
            for viewpoint in G2C_VIEW_ORDER:
                error = 0.03
                if candidate == "W" and epoch == 5 and viewpoint in {
                    "LEFT_LOW__CENTER",
                    "RIGHT_LOW__CENTER",
                }:
                    error = 0.001
                elif candidate == "S" and epoch == 5 and viewpoint == "LEFT_LOW__CENTER":
                    error = 0.0005
                rows.extend(
                    _metric_rows(
                        viewpoint,
                        count=30,
                        error=error,
                        candidate_id=candidate,
                        epoch=epoch,
                    )
                )
    result = select_g2c_checkpoint(rows, validation_losses=losses)
    assert result["selected"]["candidate_id"] == "W"
    assert result["selected"]["epoch"] == 5
    assert result["selected"]["eligible_non_home_view_count"] == 2


def test_per_view_calibration_fits_conformal_scale_and_zero_unsafe_threshold() -> None:
    rows = [
        {
            "viewpoint_id": "LEFT_LOW__CENTER",
            "world_xy_error_vector_m": [0.001, 0.0],
            "raw_covariance_base_m2": np.diag([1e-6, 1e-6, 0.0]).tolist(),
            "write_score": 0.8,
            "oracle_safe_measurement": True,
        }
        for _ in range(30)
    ]
    result = calibrate_g2c_viewpoint(rows, viewpoint_id="LEFT_LOW__CENTER")
    assert result["status"] == "calibration-pass"
    assert result["support_count"] == 30
    assert result["calibration"]["order_statistic_k"] == 30
    assert result["calibration"]["scale_factor"] == pytest.approx(1.0)
    assert result["write_threshold"] == pytest.approx(0.8)
    assert result["unsafe_accepted_count"] == 0


def test_dynamic_qualification_plan_has_exact_d036_counts() -> None:
    plan = g2c_dynamic_qualification_plan()
    assert plan["route_count"] == 500
    assert plan["totals"] == {
        "camera_pose_set_count": 48_500,
        "moving_interpolation_command_count": 40_000,
        "safe_hold_open_step_count": 48_000,
        "ledger_frame_count": 46_000,
        "provider_scored_home_frame_count": 50,
        "provider_scored_alternate_frame_count": 500,
        "provider_scored_frame_count": 550,
    }
    validate_g2c_dynamic_qualification_counters(plan["totals"])
    wrong = deepcopy(plan["totals"])
    wrong["camera_pose_set_count"] -= 1
    with pytest.raises(RuntimeError, match="精确计数"):
        validate_g2c_dynamic_qualification_counters(wrong)


def test_train_config_builder_rejects_noncanonical_smoke_receipt(tmp_path: Path) -> None:
    receipt = {
        "version": g2c_data.E018_P1_G2C_DATA_RESULT_VERSION,
        "status": "complete-engineering-smoke-pass",
        "gate_passed": True,
        "canonical_data_receipt": False,
        "checkpoint_write_count": 0,
    }
    path = tmp_path / "data_receipt.json"
    path.write_text(json.dumps(receipt), encoding="utf-8")
    with pytest.raises(RuntimeError, match="canonical DATA"):
        build_g2c_train_config_payload(path)


def test_full_data_runner_requires_new_r2_exit_before_opening_inputs(tmp_path: Path) -> None:
    with pytest.raises(PermissionError, match="R2 exit GO"):
        g2c_data.run_e018_p1_g2c_data(
            config_path=tmp_path / "missing.json",
            parent_g0c_config_path=tmp_path / "missing-parent.json",
            parent_g0c_receipt_path=tmp_path / "missing-receipt.json",
            e013_deployable_root=tmp_path,
            e016_fresh_deployable_root=tmp_path,
            stats_root=tmp_path,
            inventory_path=tmp_path / "inventory.json",
            repository_root=tmp_path,
            output_root=tmp_path / "output",
            mode="full",
            decision_exit_go=False,
        )
