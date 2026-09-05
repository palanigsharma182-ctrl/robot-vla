from __future__ import annotations

import json
import inspect
from pathlib import Path

import numpy as np
import pytest

from robot_vla.precision import e018_p1_g0 as g0
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


class _PoisonAccessor:
    def __getattribute__(self, name: str) -> object:
        raise AssertionError(f"disabled 路径不应访问 actor: {name}")


class _PoisonMapping:
    def __getitem__(self, key: object) -> object:
        raise AssertionError(f"disabled 路径不应访问 segmentation: {key}")


def test_offline_diagnostics_false_path_never_reads_actor_or_segmentation() -> None:
    assert g0._offline_actor_ids(_PoisonAccessor(), enabled=False) == (None, None)
    assert (
        g0._offline_segmentation_diagnostics(
            _PoisonMapping(),
            enabled=False,
            object_actor_id=None,
            goal_actor_id=None,
        )
        is None
    )


def test_offline_diagnostics_true_path_preserves_actor_pixel_counts() -> None:
    segmentation = np.asarray([[[[3], [4]], [[3], [9]]]], dtype=np.int32)
    result = g0._offline_segmentation_diagnostics(
        {"segmentation": segmentation},
        enabled=True,
        object_actor_id=3,
        goal_actor_id=4,
    )

    assert result == {
        "oracle_only": True,
        "used_by_runtime_control": False,
        "object_visible_pixel_count": 2,
        "goal_visible_pixel_count": 1,
    }


def test_contact_force_safety_path_remains_explicitly_available() -> None:
    class Scene:
        def __init__(self) -> None:
            self.calls: list[tuple[object, object]] = []

        def get_pairwise_contact_forces(self, link: object, cube: object) -> np.ndarray:
            self.calls.append((link, cube))
            value = 3.0 if link == "left" else 4.0
            return np.asarray([[value, 0.0, 0.0]], dtype=np.float32)

    scene = Scene()
    base_env = type(
        "BaseEnv",
        (),
        {
            "scene": scene,
            "cube": "cube",
            "agent": type("Agent", (), {"finger1_link": "left", "finger2_link": "right"})(),
        },
    )()

    assert g0._read_finger_contact_force_n(base_env) == 4.0
    assert scene.calls == [("left", "cube"), ("right", "cube")]


def test_route_hook_is_backward_compatible_default_none() -> None:
    parameter = inspect.signature(g0._run_route).parameters["frame_hook"]

    assert parameter.default is None


def test_raw_safety_witnesses_are_qualification_only_and_default_byte_neutral() -> None:
    parameters = inspect.signature(g0._run_route).parameters
    assert parameters["include_raw_safety_witnesses"].default is False
    base = {"existing": [1, 2, 3]}
    before = json.dumps(base, sort_keys=True, separators=(",", ":"))
    kwargs = {
        "arm_anchor_q_rad": np.zeros(7),
        "arm_current_q_rad": np.zeros(7),
        "tcp_anchor_world": np.eye(4),
        "tcp_current_world": np.eye(4),
        "world_from_robot_base": np.eye(4),
        "finger_joint_positions_m": np.asarray([0.04, 0.04]),
    }

    disabled = g0._raw_safety_witness_fields(enabled=False, **kwargs)
    base.update(disabled)

    assert disabled == {}
    assert json.dumps(base, sort_keys=True, separators=(",", ":")) == before
    assert set(g0._raw_safety_witness_fields(enabled=True, **kwargs)) == {
        "arm_anchor_q_rad",
        "arm_current_q_rad",
        "tcp_anchor_world",
        "tcp_current_world",
        "world_from_robot_base",
        "finger_joint_positions_m",
    }
