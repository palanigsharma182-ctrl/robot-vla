from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_vla.precision.e018_p1_g1a import (
    E018_P1_G1A_CONFIG_VERSION,
    load_e018_p1_g1a_config,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "e018_p1_g1a_dynamic_external_observation_probe_v2.json"
PARENT_PATH = ROOT / "configs" / "e018_p1_g0c_rotated_motion_development_v1.json"


def _write_config(tmp_path: Path, config: dict[str, object]) -> Path:
    path = tmp_path / "g1a.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_g1a_config_is_one_seed_one_fixed_primitive_and_no_live_consumers() -> None:
    config, parent = load_e018_p1_g1a_config(
        CONFIG_PATH,
        parent_g0c_config_path=PARENT_PATH,
    )

    assert config["version"] == E018_P1_G1A_CONFIG_VERSION
    assert config["probe"]["selected_primitive_id"] == "LEFT_LOW__YAW_LEFT"
    assert config["probe"]["home_barrier_ticks"] == 4
    assert config["scope"]["test_split_allowed"] is False
    assert config["scope"]["provider_inference_allowed"] is False
    assert config["scope"]["memory_read_allowed"] is False
    assert config["scope"]["memory_write_allowed"] is False
    assert config["scope"]["executive_mutation_allowed"] is False
    assert parent["viewpoint_library"]["expected_primitive_count"] == 10


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("scope", "test_split_allowed"),
        ("scope", "provider_inference_allowed"),
        ("scope", "memory_read_allowed"),
        ("scope", "memory_write_allowed"),
        ("scope", "executive_mutation_allowed"),
        ("execution", "physical_robot_actuation_allowed"),
        ("execution", "arm_motion_command_allowed"),
        ("execution", "provider_inference_allowed"),
        ("execution", "memory_read_allowed"),
        ("execution", "memory_write_allowed"),
        ("execution", "manipulation_progression_allowed"),
        ("execution", "test_data_read_allowed"),
    ],
)
def test_g1a_config_rejects_scope_expansion(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config[section][field] = True

    with pytest.raises(ValueError, match="scope|execution"):
        load_e018_p1_g1a_config(
            _write_config(tmp_path, config),
            parent_g0c_config_path=PARENT_PATH,
        )


def test_g1a_config_rejects_parent_receipt_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["parent_motion"]["receipt_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="parent G0C"):
        load_e018_p1_g1a_config(
            _write_config(tmp_path, config),
            parent_g0c_config_path=PARENT_PATH,
        )


def test_g1a_config_rejects_non_parent_primitive(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["probe"]["selected_primitive_id"] = "LEFT_HIGH__CENTER"

    with pytest.raises(ValueError, match="primitive"):
        load_e018_p1_g1a_config(
            _write_config(tmp_path, config),
            parent_g0c_config_path=PARENT_PATH,
        )


def test_g1a_config_requires_four_fresh_home_frames(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["probe"]["home_barrier_ticks"] = 3

    with pytest.raises(ValueError, match="四个全新 frame"):
        load_e018_p1_g1a_config(
            _write_config(tmp_path, config),
            parent_g0c_config_path=PARENT_PATH,
        )


def test_g1a_config_freezes_rotation_projection_correction_tolerance() -> None:
    config, _ = load_e018_p1_g1a_config(
        CONFIG_PATH,
        parent_g0c_config_path=PARENT_PATH,
    )

    assert config["gates"]["maximum_rotation_projection_error_frobenius"] == 1e-6
