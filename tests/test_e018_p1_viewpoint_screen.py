from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from robot_vla.precision.e018_p1_viewpoint_screen import (
    E018_P1_G0B_CONFIG_VERSION,
    load_e018_p1_g0b_config,
)

CONFIG_PATH = (
    Path(__file__).parents[1] / "configs/e018_p1_g0b_viewpoint_screen_development_v1.json"
)


def test_g0b_config_is_strict_5_by_5_development_lattice() -> None:
    config = load_e018_p1_g0b_config(CONFIG_PATH)
    lattice = config["viewpoint_lattice"]

    assert config["version"] == E018_P1_G0B_CONFIG_VERSION
    assert lattice["expected_anchor_count"] == 5
    assert lattice["expected_orientation_count"] == 5
    assert lattice["expected_primitive_count"] == 25
    assert len(lattice["anchors"]) * len(lattice["orientation_modes"]) == 25
    assert config["scope"]["test_split_allowed"] is False
    assert config["scope"]["provider_inference_allowed"] is False
    assert config["execution"]["environment_step_allowed"] is False


def test_g0b_orientation_modes_are_center_yaw12_pitch8_cross() -> None:
    config = load_e018_p1_g0b_config(CONFIG_PATH)
    modes = {
        item["orientation_id"]: item
        for item in config["viewpoint_lattice"]["orientation_modes"]
    }

    assert modes["YAW_LEFT"]["yaw_offset_rad"] == pytest.approx(math.radians(12.0))
    assert modes["YAW_RIGHT"]["yaw_offset_rad"] == pytest.approx(-math.radians(12.0))
    assert modes["PITCH_UP"]["pitch_offset_rad"] == pytest.approx(math.radians(8.0))
    assert modes["PITCH_DOWN"]["pitch_offset_rad"] == pytest.approx(-math.radians(8.0))
    assert all(
        not (item["yaw_offset_rad"] and item["pitch_offset_rad"])
        for item in modes.values()
    )


def test_g0b_config_rejects_test_scope(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["scope"]["test_split_allowed"] = True
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="scope"):
        load_e018_p1_g0b_config(invalid)


def test_g0b_config_rejects_diagonal_orientation(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    mode = config["viewpoint_lattice"]["orientation_modes"][1]
    mode["pitch_offset_rad"] = math.radians(8.0)
    invalid = tmp_path / "invalid.json"
    invalid.write_text(json.dumps(config), encoding="utf-8")

    with pytest.raises(ValueError, match="diagonal"):
        load_e018_p1_g0b_config(invalid)
