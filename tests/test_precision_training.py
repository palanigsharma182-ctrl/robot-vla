from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_vla.precision.training import (
    PRECISION_TRAINING_VERSION,
    load_precision_experiment_config,
)


def test_frozen_precision_training_config_loads() -> None:
    config = load_precision_experiment_config("configs/e013_precision_v1.json")

    assert config.version == PRECISION_TRAINING_VERSION
    assert config.dataset.history_length == 4
    assert config.dataset.model_camera == "hand_camera"
    assert config.debug_overfit.sample_count == 64
    assert config.formal_training.motion_head_policy == "frozen-zero-shadow-only"
    assert config.held_out.calibration_split == "val"
    assert config.held_out.evaluation_split == "test"
    assert config.shadow_rollout.actuation_allowed is False
    assert len(config.sha256) == 64


def test_precision_training_config_refuses_control_or_split_relaxation(tmp_path) -> None:
    payload = json.loads(Path("configs/e013_precision_v1.json").read_text(encoding="utf-8"))
    payload["shadow_rollout"]["actuation_allowed"] = True
    path = tmp_path / "invalid.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="禁止 actuation"):
        load_precision_experiment_config(path)

    payload["shadow_rollout"]["actuation_allowed"] = False
    payload["held_out"]["calibration_split"] = "test"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="val split"):
        load_precision_experiment_config(path)


def test_precision_training_config_refuses_invalid_latency_budget(tmp_path) -> None:
    payload = json.loads(Path("configs/e013_precision_v1.json").read_text(encoding="utf-8"))
    path = tmp_path / "invalid-latency.json"

    payload["shadow_rollout"]["p95_latency_max_s"] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="p95 latency gate"):
        load_precision_experiment_config(path)

    payload["shadow_rollout"]["p95_latency_max_s"] = 0.05
    payload["shadow_rollout"]["required_control_hz"] = 40.0
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="p95 latency gate"):
        load_precision_experiment_config(path)
