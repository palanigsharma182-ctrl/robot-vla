from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_vla.precision.e018_p1_g0c import (
    E018_P1_G0C_CONFIG_VERSION,
    _expand_primitives,
    _parse_library,
    load_e018_p1_g0c_config,
)

CONFIG_PATH = (
    Path(__file__).parents[1]
    / "configs"
    / "e018_p1_g0c_rotated_motion_development_v1.json"
)


def _write_config(tmp_path: Path, config: dict[str, object]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_g0c_config_is_strict_development_only_2x5_library() -> None:
    config = load_e018_p1_g0c_config(CONFIG_PATH)
    home, anchors, orientations = _parse_library(config)
    primitives = _expand_primitives(anchors, orientations)

    assert config["version"] == E018_P1_G0C_CONFIG_VERSION
    assert config["scope"]["test_split_allowed"] is False
    assert config["scope"]["formal_claim_allowed"] is False
    assert config["execution"]["provider_inference_allowed"] is False
    assert config["execution"]["memory_read_allowed"] is False
    assert config["execution"]["memory_write_allowed"] is False
    assert config["execution"]["arm_motion_command_allowed"] is False
    assert home.viewpoint_id == "HOME"
    assert {(item.lateral_anchor, item.vertical_anchor) for item in anchors} == {
        ("LEFT", "LOW"),
        ("RIGHT", "LOW"),
    }
    assert [item.orientation_id for item in orientations] == [
        "CENTER",
        "YAW_LEFT",
        "YAW_RIGHT",
        "PITCH_UP",
        "PITCH_DOWN",
    ]
    assert [item.viewpoint_id for item, _ in primitives] == config["parent_screen"][
        "eligible_primitive_ids"
    ]


def test_g0c_config_rejects_test_scope(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["scope"]["test_split_allowed"] = True

    with pytest.raises(ValueError, match="scope"):
        load_e018_p1_g0c_config(_write_config(tmp_path, config))


def test_g0c_config_rejects_parent_screen_identity_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["parent_screen"]["receipt_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="parent G0B"):
        load_e018_p1_g0c_config(_write_config(tmp_path, config))


def test_g0c_config_rejects_high_anchor(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    anchor = config["viewpoint_library"]["anchors"][0]
    anchor["viewpoint_id"] = "LEFT_HIGH"
    anchor["vertical_anchor"] = "HIGH"
    anchor["position_world_m"] = [0.3, -0.16, 0.72]
    anchor["pitch_rad"] = -0.9635283801277769

    with pytest.raises(ValueError, match="low-anchor pose"):
        load_e018_p1_g0c_config(_write_config(tmp_path, config))


def test_g0c_config_rejects_diagonal_orientation(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    orientation = config["viewpoint_library"]["orientation_modes"][1]
    orientation["pitch_offset_rad"] = 0.1

    with pytest.raises(ValueError, match="orientation offset"):
        load_e018_p1_g0c_config(_write_config(tmp_path, config))


def test_g0c_config_rejects_registered_anchor_numeric_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["viewpoint_library"]["anchors"][0]["position_world_m"][1] = -0.15

    with pytest.raises(ValueError, match="low-anchor pose"):
        load_e018_p1_g0c_config(_write_config(tmp_path, config))


def test_g0c_config_rejects_registered_orientation_numeric_drift(
    tmp_path: Path,
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["viewpoint_library"]["orientation_modes"][1][
        "yaw_offset_rad"
    ] = 0.20

    with pytest.raises(ValueError, match="orientation offset"):
        load_e018_p1_g0c_config(_write_config(tmp_path, config))


def test_g0c_config_requires_integer_motion_ticks(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["motion"]["move_duration_s"] = 2.01

    with pytest.raises(ValueError, match="整数"):
        load_e018_p1_g0c_config(_write_config(tmp_path, config))
