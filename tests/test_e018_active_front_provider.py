from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_vla.adapters import (
    FingerForceNormalizer,
    FingerForceStats,
    ProprioNormalizer,
    ProprioStats,
)
from robot_vla.contracts import RobotSpec
from robot_vla.observation import OBSERVATION_MODALITIES
from robot_vla.precision.active_external_observation import ACTUAL_EXTERNAL_POSE_SOURCE
from robot_vla.precision.active_front_camera import ExternalCameraMotionState
from robot_vla.precision.active_front_provider import (
    ActiveFrontProviderAdapterConfig,
    ActiveFrontProviderIdentity,
    build_active_front_model_input,
)
from robot_vla.precision.provider import PrecisionGeometricMotionInput


def _normalizers(spec: RobotSpec) -> tuple[ProprioNormalizer, FingerForceNormalizer]:
    proprio = ProprioNormalizer(
        ProprioStats(
            mean=(0.0,) * spec.proprio_dim,
            std=(1.0,) * spec.proprio_dim,
            count=10,
        ),
        spec,
    )
    finger = FingerForceNormalizer(
        FingerForceStats(
            scale_log1p_p95=(1.0, 1.0),
            count=10,
            positive_count=(1, 1),
        ),
        spec,
    )
    return proprio, finger


def _input(**updates: object):
    spec = RobotSpec()
    proprio, finger = _normalizers(spec)
    camera = np.eye(4, dtype=np.float64)
    camera[:3, 3] = (0.3, -0.16, 0.48)
    values: dict[str, object] = {
        "spec": spec,
        "proprio_normalizer": proprio,
        "finger_force_normalizer": finger,
        "config": ActiveFrontProviderAdapterConfig(),
        "episode_id": "qual-seed-74201",
        "request_id": "qual-request-001",
        "observation_sequence_id": "qual-observation-001",
        "primitive_id": "LEFT_LOW__YAW_LEFT",
        "rgb_external": np.zeros((128, 128, 3), dtype=np.uint8),
        "physical_proprio": np.zeros(spec.proprio_dim, dtype=np.float32),
        "base_from_tcp": np.eye(4, dtype=np.float64),
        "base_from_external_camera_cv": camera,
        "finger_force_n": np.zeros(2, dtype=np.float32),
        "intrinsic_cv": np.asarray(
            ((100.0, 0.0, 63.5), (0.0, 100.0, 63.5), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        "control_timestamp_s": 1.0,
        "rgb_timestamp_s": 1.0,
        "camera_pose_timestamp_s": 1.0,
        "tcp_pose_timestamp_s": 1.0,
        "geometric_motion": PrecisionGeometricMotionInput(
            timestamp_s=1.0,
            motion=(0.0, 0.0, 0.0, 0.0),
        ),
        "geometric_motion_provider_id": "safe-hold-commanded-tcp-delta/test-v1",
        "camera_motion_state": ExternalCameraMotionState.COLLECT,
        "settled": True,
    }
    values.update(updates)
    return build_active_front_model_input(**values)


def test_adapter_explicitly_substitutes_actual_external_pose_in_camera_slots() -> None:
    model_input = _input()

    camera_state_start = RobotSpec().proprio_dim + 3 + 6
    assert np.array_equal(
        model_input.structured_state[camera_state_start : camera_state_start + 3],
        np.asarray((0.3, -0.16, 0.48), dtype=np.float32),
    )
    assert np.array_equal(
        model_input.structured_state[-len(OBSERVATION_MODALITIES) :],
        np.ones(len(OBSERVATION_MODALITIES), dtype=np.float32),
    )
    assert model_input.source_camera == "base_camera"
    assert model_input.actual_pose_source == ACTUAL_EXTERNAL_POSE_SOURCE
    assert model_input.qualification_only is True
    assert model_input.memory_write_eligible is False
    assert len(model_input.input_digest) == 64


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        (
            "camera_motion_state",
            ExternalCameraMotionState.MOVE_TO_VIEW,
            "COLLECT",
        ),
        ("settled", False, "settled"),
        ("rgb_timestamp_s", 1.02, "skew"),
        (
            "geometric_motion",
            PrecisionGeometricMotionInput(
                timestamp_s=1.02,
                motion=(0.0, 0.0, 0.0, 0.0),
            ),
            "skew",
        ),
    ],
)
def test_adapter_rejects_motion_unsettled_or_time_skew(
    field: str,
    value: object,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        _input(**{field: value})


def test_adapter_config_cannot_enable_memory_or_actuation() -> None:
    with pytest.raises(ValueError, match="禁止"):
        ActiveFrontProviderAdapterConfig(memory_write_allowed=True)
    with pytest.raises(ValueError, match="禁止"):
        ActiveFrontProviderAdapterConfig(actuation_allowed=True)


def test_identity_preserves_training_camera_mismatch_and_is_stable() -> None:
    config = ActiveFrontProviderAdapterConfig()
    identity = ActiveFrontProviderIdentity(
        checkpoint_sha256="1" * 64,
        checkpoint_parameter_sha256="2" * 64,
        checkpoint_provenance_sha256="3" * 64,
        model_config_sha256="4" * 64,
        proprio_stats_sha256="5" * 64,
        proprio_normalizer_sha256="6" * 64,
        finger_force_stats_sha256="7" * 64,
        finger_force_normalizer_sha256="8" * 64,
        adapter_config_sha256=config.sha256,
        primitive_id="LEFT_LOW__YAW_LEFT",
        calibration_identity_sha256="9" * 64,
        geometric_motion_provider_id="safe-hold-commanded-tcp-delta/test-v1",
        source_training_camera="hand_camera",
        target_camera="base_camera",
    )

    assert identity.to_dict()["source_training_camera"] == "hand_camera"
    assert identity.to_dict()["target_camera"] == "base_camera"
    assert identity.sha256 == identity.sha256
    with pytest.raises(ValueError, match="训练来源"):
        replace(identity, source_training_camera="base_camera")


def test_input_does_not_expose_privileged_fields() -> None:
    field_names = set(_input().__dataclass_fields__)

    assert not {
        "object_position_base_m",
        "goal_position_base_m",
        "object_mask",
        "goal_mask",
        "gt_observable",
    }.intersection(field_names)
