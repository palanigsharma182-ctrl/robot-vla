from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_vla.precision.e018_p1_g0 import (
    E018_P1_G0_CONFIG_VERSION,
    load_e018_p1_g0_config,
)

CONFIG_PATH = Path(__file__).parents[1] / "configs/e018_p1_g0_camera_feasibility_v1.json"


def test_g0_config_is_strict_development_only_2x2_library() -> None:
    config = load_e018_p1_g0_config(CONFIG_PATH)

    assert config["version"] == E018_P1_G0_CONFIG_VERSION
    assert config["scope"]["test_split_allowed"] is False
    assert config["scope"]["formal_claim_allowed"] is False
    assert config["scope"]["provider_inference_allowed"] is False
    assert config["execution"]["arm_motion_command_allowed"] is False
    assert config["execution"]["memory_write_allowed"] is False
    assert {
        (item["lateral_anchor"], item["vertical_anchor"])
        for item in config["viewpoint_library"]["alternates"]
    } == {
        ("LEFT", "LOW"),
        ("LEFT", "HIGH"),
        ("RIGHT", "LOW"),
        ("RIGHT", "HIGH"),
    }


def test_g0_config_rejects_test_or_formal_scope(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["scope"]["test_split_allowed"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="scope"):
        load_e018_p1_g0_config(invalid)


def test_g0_config_rejects_missing_2x2_anchor(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["viewpoint_library"]["alternates"].pop()
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="2x2"):
        load_e018_p1_g0_config(invalid)
