from __future__ import annotations

import json
from pathlib import Path

import pytest

from robot_vla.precision.e018_p1_g1 import (
    E018_P1_G1_CONFIG_VERSION,
    load_e018_p1_g1_config,
    run_e018_p1_g1,
)

ROOT = Path(__file__).parents[1]
CONFIG_PATH = ROOT / "configs" / "e018_p1_g1_control_shadow_replay_development_v2.json"


def _write_config(tmp_path: Path, config: dict[str, object]) -> Path:
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    return path


def test_g1_config_is_strict_shadow_no_live_consumers() -> None:
    config = load_e018_p1_g1_config(CONFIG_PATH)

    assert config["version"] == E018_P1_G1_CONFIG_VERSION
    assert config["supervisor"]["allowed_source_phases"] == [
        "acquire_track",
        "stabilize_pregrasp",
    ]
    assert config["supervisor"]["home_v2_barrier_frames"] == 4
    assert len(config["supervisor"]["post_active_window_phases"]) == 13
    assert all(
        value is False
        for name, value in config["execution"].items()
        if name != "device"
    )


@pytest.mark.parametrize(
    ("section", "field"),
    [
        ("scope", "test_split_allowed"),
        ("scope", "formal_claim_allowed"),
        ("scope", "provider_inference_allowed"),
        ("scope", "memory_read_allowed"),
        ("scope", "memory_write_allowed"),
        ("scope", "executive_integration_allowed"),
        ("scope", "camera_actuation_allowed"),
        ("scope", "arm_actuation_allowed"),
        ("execution", "simulator_steps_allowed"),
        ("execution", "physical_robot_actuation_allowed"),
        ("execution", "simulated_camera_actuation_allowed"),
        ("execution", "provider_inference_allowed"),
        ("execution", "memory_read_allowed"),
        ("execution", "memory_write_allowed"),
        ("execution", "test_data_read_allowed"),
    ],
)
def test_g1_config_rejects_scope_expansion(
    tmp_path: Path,
    section: str,
    field: str,
) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config[section][field] = True

    with pytest.raises(ValueError, match="scope|execution"):
        load_e018_p1_g1_config(_write_config(tmp_path, config))


def test_g1_config_rejects_parent_receipt_drift(tmp_path: Path) -> None:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    config["parent_capability"]["receipt_sha256"] = "0" * 64

    with pytest.raises(ValueError, match="parent G1A"):
        load_e018_p1_g1_config(_write_config(tmp_path, config))


def test_g1_replay_writes_deterministic_no_write_receipts(tmp_path: Path) -> None:
    summary = run_e018_p1_g1(
        config_path=CONFIG_PATH,
        repository_root=ROOT,
        output_root=tmp_path / "result",
    )

    assert summary["gate_passed"] is True
    assert summary["success_replay_count"] == 2
    assert summary["failure_replay_count"] == 6
    assert summary["post_active_window_phase_count"] == 13
    assert summary["simulator_step_count"] == 0
    assert summary["camera_actuation_count"] == 0
    assert summary["arm_actuation_count"] == 0
    assert summary["provider_forward_count"] == 0
    assert summary["memory_read_count"] == 0
    assert summary["memory_write_count"] == 0
    assert summary["test_read_count"] == 0
    assert (tmp_path / "result" / "receipt.json").is_file()


def test_g1_replay_refuses_existing_output(tmp_path: Path) -> None:
    output = tmp_path / "existing"
    output.mkdir()

    with pytest.raises(FileExistsError):
        run_e018_p1_g1(
            config_path=CONFIG_PATH,
            repository_root=ROOT,
            output_root=output,
        )
