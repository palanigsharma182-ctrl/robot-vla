from __future__ import annotations

import json
import math
import inspect
from pathlib import Path

import pytest

from robot_vla.precision.e018_p1_g2b import (
    E018_P1_G2B_CONFIG_VERSION,
    assert_calibration_prediction_ledger_deployable_only,
    fit_covariance_scale,
    freeze_calibration_prediction_ledger,
    load_e018_p1_g2b_config,
    load_frozen_calibration_prediction_ledger,
    run_e018_p1_g2b_qualification,
)
from robot_vla.precision import e018_p1_g2b as g2b


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPOSITORY_ROOT
    / "configs/e018_p1_g2b_covariance_calibrated_provider_requalification_development_v1.json"
)
PARENT_CONFIG_PATH = (
    REPOSITORY_ROOT / "configs/e018_p1_g2a_front_provider_qualification_development_v1.json"
)


def _config() -> dict[str, object]:
    return json.loads(CONFIG_PATH.read_text(encoding="utf-8"))


def test_g2b_config_is_strict_and_binds_frozen_g2a() -> None:
    config = load_e018_p1_g2b_config(
        CONFIG_PATH,
        parent_g2a_config_path=PARENT_CONFIG_PATH,
    )
    assert config["version"] == E018_P1_G2B_CONFIG_VERSION
    assert config["calibration"]["selection_must_not_use"] == [
        "confidence",
        "write_accepted",
        "prediction_error_magnitude",
    ]
    assert config["qualification"]["full_run_requires_decision_agent_exit_go"] is True


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("scope", "test_label_array_read_allowed"), True, "scope"),
        (("calibration", "minimum_support_count"), 29, "算法"),
        (("calibration", "selection_must_not_use"), [], "算法"),
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
    rows = [
        {
            "trajectory_id": "val-001",
            "timestep": 0,
            "geometry": {"valid": True},
        }
    ]
    config_sha = "a" * 64
    marker = freeze_calibration_prediction_ledger(
        tmp_path,
        rows=rows,
        config_sha256=config_sha,
    )
    rows[0]["trajectory_id"] = "mutated-in-memory"
    loaded, loaded_marker = load_frozen_calibration_prediction_ledger(
        tmp_path,
        config_sha256=config_sha,
        expected_prediction_count=1,
    )
    assert loaded[0]["trajectory_id"] == "val-001"
    assert loaded_marker == marker

    with (tmp_path / "calibration_prediction_ledger.jsonl").open("a", encoding="utf-8") as handle:
        handle.write("{}\n")
    with pytest.raises(RuntimeError, match="identity"):
        load_frozen_calibration_prediction_ledger(
            tmp_path,
            config_sha256=config_sha,
            expected_prediction_count=1,
        )


def test_calibration_prediction_ledger_rejects_privileged_fields() -> None:
    with pytest.raises(ValueError, match="privileged"):
        assert_calibration_prediction_ledger_deployable_only(
            [{"trajectory_id": "val-001", "gt_object_position_base_m": [0, 0, 0]}]
        )


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
