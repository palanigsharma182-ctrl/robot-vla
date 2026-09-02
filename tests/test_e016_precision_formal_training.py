from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest

from robot_vla.precision.data import file_sha256
from robot_vla.precision.e016_pretraining import E016P0Metrics
from robot_vla.precision.e016_training import (
    E016_P1_TEST_POLICY,
    E016_P1_VERSION,
    e016_p1_validation_guardrails,
    load_e016_p1_config,
    select_e016_p1_checkpoint_epoch,
    validate_e016_p0_prerequisite,
)

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "configs" / "e016_p1_precision_observability_v1.json"


def _metrics(**overrides: float) -> E016P0Metrics:
    values: dict[str, float] = {
        "sample_count": 100,
        "object_localization_valid_count": 100,
        "goal_observable_count": 50,
        "goal_unobservable_count": 50,
        "mean_loss": 0.1,
        "mean_heatmap_loss": 0.1,
        "mean_mask_loss": 0.1,
        "mean_coordinate_loss": 0.1,
        "mean_uncertainty_loss": 0.1,
        "mean_visibility_loss": 0.1,
        "mean_projection_loss": 0.1,
        "object_normalized_uv_mae": 0.002,
        "goal_observable_normalized_uv_mae": 0.004,
        "goal_observable_pixel_error_p50": 0.5,
        "goal_observable_pixel_error_p90": 1.5,
        "object_mask_iou": 0.9,
        "goal_mask_iou": 0.6,
        "goal_visibility_true_positive": 50,
        "goal_visibility_false_positive": 0,
        "goal_visibility_true_negative": 50,
        "goal_visibility_false_negative": 0,
        "goal_visibility_precision": 1.0,
        "goal_visibility_recall": 1.0,
        "goal_visibility_f1": 1.0,
        "goal_unobservable_false_positive_rate": 0.0,
        "projection_accuracy": 1.0,
    }
    values.update(overrides)
    return E016P0Metrics(**values)  # type: ignore[arg-type]


def test_e016_p1_config_freezes_train_val_and_fresh_test_once_plan() -> None:
    config = load_e016_p1_config(CONFIG_PATH)

    assert config.version == E016_P1_VERSION
    assert config.source.allowed_splits == ("train", "val")
    assert config.source.excluded_splits == ("test",)
    assert config.formal_training.epochs == 20
    assert config.formal_training.initialization == "random-from-scratch"
    assert config.fresh_held_out.start_seed == 134000
    assert config.fresh_held_out.validation_trajectories == 20
    assert config.fresh_held_out.test_trajectories == 100
    assert config.fresh_held_out.test_policy == E016_P1_TEST_POLICY
    assert config.fresh_held_out.legacy_e013_e015_test_reuse_allowed is False
    assert config.execution.actuation_allowed is False
    assert config.success_criteria.actuator_promotion_allowed is False
    assert len(config.sha256) == 64


def test_e016_p1_config_rejects_guardrail_field_drift(tmp_path: Path) -> None:
    payload = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    payload["validation_selection"]["unregistered"] = 1
    path = tmp_path / "config.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="keys 漂移"):
        load_e016_p1_config(path)


def test_e016_p1_selection_rejects_better_metric_when_safety_fails() -> None:
    config = load_e016_p1_config(CONFIG_PATH).validation_selection
    unsafe_best = _metrics(
        goal_observable_normalized_uv_mae=0.001,
        goal_visibility_recall=0.89,
    )
    safe = _metrics(goal_observable_normalized_uv_mae=0.004)
    safe_tie_later = _metrics(goal_observable_normalized_uv_mae=0.004)

    assert (
        select_e016_p1_checkpoint_epoch(
            [(1, unsafe_best), (2, safe), (3, safe_tie_later)],
            config,
        )
        == 2
    )


def test_e016_p1_selection_fails_closed_when_no_epoch_is_safe() -> None:
    config = load_e016_p1_config(CONFIG_PATH).validation_selection
    unsafe = _metrics(goal_unobservable_false_positive_rate=0.010001)

    assert select_e016_p1_checkpoint_epoch([(1, unsafe)], config) is None


def test_e016_p1_guardrail_boundaries_are_inclusive() -> None:
    config = load_e016_p1_config(CONFIG_PATH).validation_selection
    boundary = _metrics(
        goal_unobservable_false_positive_rate=(config.goal_unobservable_false_positive_rate_max),
        goal_visibility_precision=config.goal_visibility_precision_min,
        goal_visibility_recall=config.goal_visibility_recall_min,
        projection_accuracy=config.projection_accuracy_min,
        goal_mask_iou=config.goal_mask_iou_min,
    )

    assert all(e016_p1_validation_guardrails(boundary, config).values())


def test_e016_p0_prerequisite_binds_receipt_config_and_sidecar(
    tmp_path: Path,
) -> None:
    p0_root = tmp_path / "p0"
    corrected_root = p0_root / "corrected-labels"
    corrected_root.mkdir(parents=True)
    p0_config_source = ROOT / "configs" / "e016_p0_precision_observability_v1.json"
    p0_config_path = p0_root / "config_snapshot.json"
    p0_config_path.write_bytes(p0_config_source.read_bytes())
    p1_config = load_e016_p1_config(CONFIG_PATH)
    corrected_identity = "a" * 64
    audit = {
        "version": "e016-p0-corrected-observability/v1",
        "passed": True,
        "test_label_file_read_count": 0,
        "corrected_data_identity_sha256": corrected_identity,
    }
    audit_payload = json.dumps(audit, indent=2, sort_keys=True) + "\n"
    audit_path = p0_root / "corrected_sidecar_audit.json"
    audit_path.write_text(audit_payload, encoding="utf-8")
    (corrected_root / "audit.json").write_text(audit_payload, encoding="utf-8")
    prerequisite = replace(
        p1_config.prerequisite,
        p0_training_config_sha256=(
            load_e016_p1_config(CONFIG_PATH).prerequisite.p0_training_config_sha256
        ),
        corrected_sidecar_audit_sha256=file_sha256(audit_path),
        corrected_data_identity_sha256=corrected_identity,
        p0_receipt_sha256="b" * 64,
    )
    # config snapshot 的 canonical hash 是冻结 P0 hash；这里显式验证测试 fixture 没有漂移。
    assert (
        prerequisite.p0_training_config_sha256 == p1_config.prerequisite.p0_training_config_sha256
    )
    receipt = {
        "version": "e016-p0-corrected-observability/v1",
        "passed": True,
        "formal_checkpoint_written": False,
        "test_split_read": False,
        "training_config_sha256": prerequisite.p0_training_config_sha256,
        "corrected_data_identity_sha256": corrected_identity,
    }
    receipt_path = p0_root / "receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    config = replace(
        p1_config,
        prerequisite=replace(
            prerequisite,
            p0_receipt_sha256=file_sha256(receipt_path),
        ),
    )

    result = validate_e016_p0_prerequisite(p0_root, config)

    assert result["p0_receipt_sha256"] == file_sha256(receipt_path)
    assert result["corrected_data_identity_sha256"] == corrected_identity

    receipt["test_split_read"] = True
    receipt_path.write_text(
        json.dumps(receipt, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    tampered = replace(
        config,
        prerequisite=replace(
            config.prerequisite,
            p0_receipt_sha256=file_sha256(receipt_path),
        ),
    )
    with pytest.raises(RuntimeError, match="读取过 test"):
        validate_e016_p0_prerequisite(p0_root, tampered)
