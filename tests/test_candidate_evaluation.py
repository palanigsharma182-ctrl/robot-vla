import json

import numpy as np
import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.runtime.candidate_evaluation import (
    CandidateEvaluationIdentity,
    CandidateEvaluationReceipt,
    CandidateEvaluationRequest,
    CandidateEvaluationStatus,
    NORMALIZED_ACTION_SCHEMA_ID,
    NORMALIZED_PROPRIO_COMMAND_SCHEMA_ID,
)


_WORLD_MODEL_CONFIG_DIGEST = "a" * 64


def _identity(**overrides) -> CandidateEvaluationIdentity:
    values = {
        "episode_id": "episode-0001",
        "candidate_id": "candidate-0001",
        "task_id": "pick-cube-to-region",
        "source_domain": "sim",
        "observation_schema_id": NORMALIZED_PROPRIO_COMMAND_SCHEMA_ID,
        "action_schema_id": NORMALIZED_ACTION_SCHEMA_ID,
        "policy_checkpoint_id": "sha256:policy",
        "proprio_stats_id": "sha256:stats",
    }
    values.update(overrides)
    return CandidateEvaluationIdentity(**values)


def _request(**overrides) -> CandidateEvaluationRequest:
    spec = RobotSpec()
    values = {
        "identity": _identity(),
        "origin_control_step": 8,
        "observation_timestamp_ns": 123456789,
        "normalized_proprio": np.zeros(spec.proprio_dim, dtype=np.float32),
        "observed_gripper_opening_ratio": 0.5,
        "raw_normalized_action": np.zeros(
            (spec.action_horizon, spec.action_dim), dtype=np.float32
        ),
        "effective_normalized_action": np.zeros(
            (spec.action_horizon, spec.action_dim), dtype=np.float32
        ),
        "normalized_arm_command_target_prefix": np.zeros(
            (spec.execute_steps, spec.arm_dof), dtype=np.float32
        ),
        "previous_executed_steps": spec.execute_steps,
    }
    values.update(overrides)
    return CandidateEvaluationRequest(**values)


def test_request_preserves_raw_and_effective_action_semantics() -> None:
    spec = RobotSpec()
    raw = np.full((spec.action_horizon, spec.action_dim), 0.25, dtype=np.float32)
    effective = np.full((spec.action_horizon, spec.action_dim), -0.5, dtype=np.float32)

    request = _request(raw_normalized_action=raw, effective_normalized_action=effective)

    np.testing.assert_array_equal(request.raw_normalized_action, 0.25)
    np.testing.assert_array_equal(request.effective_action_prefix, -0.5)
    assert request.effective_action_prefix.shape == (spec.execute_steps, spec.action_dim)
    assert request.raw_action_digest != request.effective_action_digest


def test_request_defensively_copies_and_freezes_arrays() -> None:
    spec = RobotSpec()
    raw = np.zeros((spec.action_horizon, spec.action_dim), dtype=np.float32)
    command = np.zeros((spec.execute_steps, spec.arm_dof), dtype=np.float32)
    request = _request(
        raw_normalized_action=raw,
        normalized_arm_command_target_prefix=command,
    )
    digest = request.raw_action_digest
    command_digest = request.normalized_arm_command_target_prefix_digest
    request_digest = request.request_digest

    raw[0, 0] = 1.0
    command[0, 0] = 1.0

    assert request.raw_normalized_action[0, 0] == 0.0
    assert request.normalized_arm_command_target_prefix[0, 0] == 0.0
    assert request.raw_action_digest == digest
    assert request.normalized_arm_command_target_prefix_digest == command_digest
    assert request.request_digest == request_digest
    with pytest.raises(ValueError, match="read-only"):
        request.raw_normalized_action[0, 0] = 0.5
    with pytest.raises(ValueError, match="read-only"):
        request.normalized_arm_command_target_prefix[0, 0] = 0.5


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("normalized_proprio", np.zeros(14, dtype=np.float32), "normalized_proprio"),
        (
            "raw_normalized_action",
            np.zeros((4, 8), dtype=np.float32),
            "raw_normalized_action",
        ),
        (
            "effective_normalized_action",
            np.full((16, 8), np.nan, dtype=np.float32),
            "NaN",
        ),
        (
            "raw_normalized_action",
            np.full((16, 8), 1.1, dtype=np.float32),
            "超出",
        ),
        (
            "normalized_arm_command_target_prefix",
            np.zeros((3, 7), dtype=np.float32),
            "normalized_arm_command_target_prefix",
        ),
        (
            "normalized_arm_command_target_prefix",
            np.zeros((4, 7), dtype=np.float64),
            "float32",
        ),
        (
            "normalized_arm_command_target_prefix",
            np.full((4, 7), np.nan, dtype=np.float32),
            "NaN",
        ),
        (
            "normalized_arm_command_target_prefix",
            np.full((4, 7), 5.1, dtype=np.float32),
            "超出",
        ),
    ],
)
def test_request_rejects_invalid_tensor_contract(field, value, error) -> None:
    with pytest.raises((TypeError, ValueError), match=error):
        _request(**{field: value})


def test_identity_rejects_unknown_domain_and_missing_checkpoint() -> None:
    with pytest.raises(ValueError, match="source_domain"):
        _identity(source_domain="privileged-sim")
    with pytest.raises(ValueError, match="policy_checkpoint_id"):
        _identity(policy_checkpoint_id="")


@pytest.mark.parametrize(
    ("field", "value", "expected"),
    [
        (
            "observation_schema_id",
            "online-observation/v0",
            NORMALIZED_PROPRIO_COMMAND_SCHEMA_ID,
        ),
        (
            "action_schema_id",
            "delta-q-gripper-target/v1",
            NORMALIZED_ACTION_SCHEMA_ID,
        ),
    ],
)
def test_identity_requires_exact_state_and_action_schema(
    field: str,
    value: str,
    expected: str,
) -> None:
    with pytest.raises(ValueError, match=expected):
        _identity(**{field: value})


def test_command_target_prefix_participates_in_request_identity() -> None:
    spec = RobotSpec()
    first_command = np.zeros((spec.execute_steps, spec.arm_dof), dtype=np.float32)
    second_command = first_command.copy()
    second_command[0, 0] = 0.25

    first = _request(normalized_arm_command_target_prefix=first_command)
    second = _request(normalized_arm_command_target_prefix=second_command)

    assert (
        first.normalized_arm_command_target_prefix_digest
        != second.normalized_arm_command_target_prefix_digest
    )
    assert first.request_digest != second.request_digest


def test_request_rejects_impossible_previous_execution_length() -> None:
    with pytest.raises(ValueError, match="previous_executed_steps"):
        _request(previous_executed_steps=RobotSpec().execute_steps + 1)


@pytest.mark.parametrize("opening", [-0.1, 1.1, float("nan"), True])
def test_request_rejects_invalid_physical_gripper_opening(opening) -> None:
    with pytest.raises(ValueError, match="observed_gripper_opening_ratio"):
        _request(observed_gripper_opening_ratio=opening)


def test_receipt_is_json_serializable_and_hard_disables_actuation() -> None:
    request = _request()
    payload_digest = "1" * 64
    world_model_config_digest = _WORLD_MODEL_CONFIG_DIGEST
    receipt = CandidateEvaluationReceipt.from_request(
        request,
        status=CandidateEvaluationStatus.WARMUP,
        reason_codes=("previous_chunk_unavailable",),
        evaluation_payload_digest=payload_digest,
        world_model_architecture="tiny-structured-world-model/v0",
        world_model_config_digest=world_model_config_digest,
        world_model_checkpoint_id="untrained:state-hold",
        critic_version="action-consistency-critic/v0",
        critic_checkpoint_id=None,
        calibration_id=None,
        latency_ms=0.25,
    )

    payload = receipt.to_dict()

    assert payload["actuation_allowed"] is False
    assert payload["action_parity_equal"] is True
    assert payload["status"] == "warmup"
    assert payload["request_digest"] == request.request_digest
    assert payload["normalized_arm_command_target_prefix_digest"] == (
        request.normalized_arm_command_target_prefix_digest
    )
    assert payload["evaluation_payload_digest"] == payload_digest
    assert payload["world_model_config_digest"] == world_model_config_digest
    receipt.validate_against(request)
    json.dumps(payload, allow_nan=False)


def test_receipt_rejects_actuation_and_records_failed_action_parity() -> None:
    request = _request()
    common = {
        "identity": request.identity,
        "status": CandidateEvaluationStatus.SCORED,
        "reason_codes": (),
        "request_digest": request.request_digest,
        "raw_action_digest": request.raw_action_digest,
        "effective_action_digest": request.effective_action_digest,
        "normalized_arm_command_target_prefix_digest": (
            request.normalized_arm_command_target_prefix_digest
        ),
        "evaluation_payload_digest": "2" * 64,
        "world_model_architecture": None,
        "world_model_config_digest": None,
        "world_model_checkpoint_id": None,
        "critic_version": "action-consistency-critic/v0",
        "critic_checkpoint_id": None,
        "calibration_id": None,
        "latency_ms": 0.1,
    }
    with pytest.raises(ValueError, match="actuator"):
        CandidateEvaluationReceipt(**common, actuation_allowed=True)
    with pytest.raises(ValueError, match="action parity"):
        CandidateEvaluationReceipt(**common, action_parity_equal=False)

    mismatch = CandidateEvaluationReceipt(
        **{
            **common,
            "status": CandidateEvaluationStatus.ERROR,
            "reason_codes": ("action_parity_mismatch",),
        },
        action_parity_equal=False,
    )
    assert mismatch.action_parity_equal is False
    assert mismatch.to_dict()["action_parity_equal"] is False


def test_scored_receipt_requires_evaluation_payload_digest() -> None:
    request = _request()

    with pytest.raises(ValueError, match="evaluation_payload_digest"):
        CandidateEvaluationReceipt.from_request(
            request,
            status=CandidateEvaluationStatus.SCORED,
            reason_codes=(),
            evaluation_payload_digest=None,
            world_model_architecture="tiny-structured-world-model/v0",
            world_model_config_digest=_WORLD_MODEL_CONFIG_DIGEST,
            world_model_checkpoint_id="untrained:state-hold",
            critic_version="action-consistency-critic/v0",
            critic_checkpoint_id=None,
            calibration_id=None,
            latency_ms=0.1,
        )


def test_receipt_rejects_a_different_replan_request() -> None:
    request = _request()
    receipt = CandidateEvaluationReceipt.from_request(
        request,
        status=CandidateEvaluationStatus.SCORED,
        reason_codes=(),
        evaluation_payload_digest="3" * 64,
        world_model_architecture="tiny-structured-world-model/v0",
        world_model_config_digest=_WORLD_MODEL_CONFIG_DIGEST,
        world_model_checkpoint_id="untrained:state-hold",
        critic_version="action-consistency-critic/v0",
        critic_checkpoint_id=None,
        calibration_id=None,
        latency_ms=0.1,
    )
    different_replan = _request(origin_control_step=request.origin_control_step + 4)

    with pytest.raises(ValueError, match="request_digest"):
        receipt.validate_against(different_replan)


def test_receipt_rejects_different_command_target_prefix() -> None:
    request = _request()
    receipt = CandidateEvaluationReceipt.from_request(
        request,
        status=CandidateEvaluationStatus.SCORED,
        reason_codes=(),
        evaluation_payload_digest="4" * 64,
        world_model_architecture="tiny-structured-world-model/v0",
        world_model_config_digest=_WORLD_MODEL_CONFIG_DIGEST,
        world_model_checkpoint_id="untrained:state-hold",
        critic_version="action-consistency-critic/v0",
        critic_checkpoint_id=None,
        calibration_id=None,
        latency_ms=0.1,
    )
    different_command = np.zeros(
        (RobotSpec().execute_steps, RobotSpec().arm_dof),
        dtype=np.float32,
    )
    different_command[0, 0] = 0.25

    with pytest.raises(ValueError, match="request_digest"):
        receipt.validate_against(
            _request(normalized_arm_command_target_prefix=different_command)
        )


def test_world_model_identity_requires_config_digest_and_checkpoint() -> None:
    request = _request()
    common = {
        "request": request,
        "status": CandidateEvaluationStatus.WARMUP,
        "reason_codes": ("previous_chunk_unavailable",),
        "evaluation_payload_digest": "5" * 64,
        "world_model_architecture": "tiny-structured-world-model/v0",
        "critic_version": None,
        "critic_checkpoint_id": None,
        "calibration_id": None,
        "latency_ms": 0.1,
    }

    with pytest.raises(ValueError, match="同时绑定"):
        CandidateEvaluationReceipt.from_request(
            **common,
            world_model_config_digest=None,
            world_model_checkpoint_id="untrained:state-hold",
        )
    with pytest.raises(ValueError, match="同时绑定"):
        CandidateEvaluationReceipt.from_request(
            **common,
            world_model_config_digest=_WORLD_MODEL_CONFIG_DIGEST,
            world_model_checkpoint_id=None,
        )
    with pytest.raises(ValueError, match="world_model_config_digest"):
        CandidateEvaluationReceipt.from_request(
            **common,
            world_model_config_digest="not-a-sha256",
            world_model_checkpoint_id="untrained:state-hold",
        )
