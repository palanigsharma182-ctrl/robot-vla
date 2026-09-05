import json

import numpy as np
import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.evaluation.action_consistency import (
    ACTION_CONSISTENCY_CRITIC_VERSION,
    ActionConsistencyCritic,
    ActionConsistencyStatus,
)


def _chunk(spec: RobotSpec, value: float = 0.0) -> np.ndarray:
    return np.full(
        (spec.action_horizon, spec.action_dim),
        value,
        dtype=np.float32,
    )


def test_critic_rejects_robot_spec_action_contract_drift() -> None:
    with pytest.raises(ValueError, match="V0 critic 合同"):
        ActionConsistencyCritic(RobotSpec(action_horizon=8, execute_steps=2))


def test_first_chunk_is_warmup_with_acm_but_without_tce() -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    chunk = _chunk(spec)
    chunk[: spec.execute_steps, : spec.arm_dof] = 0.5

    result = critic.evaluate(
        chunk,
        episode_id="episode-1",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )

    assert result.version == ACTION_CONSISTENCY_CRITIC_VERSION
    assert result.status is ActionConsistencyStatus.WARMUP
    assert result.reason == "no_previous_proposal"
    assert result.arm_tce_mse_normalized is None
    assert result.gripper_tce_mse_normalized is None
    assert result.arm_acm_rms_normalized == pytest.approx(0.5)
    assert result.gripper_transition_rms_opening_ratio == pytest.approx(0.0)


def test_reset_clears_history_and_preserves_reason_for_warmup() -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    critic.evaluate(
        _chunk(spec),
        episode_id="episode-1",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )
    critic.reset(episode_id="episode-2", reason="anomaly_reset")

    result = critic.evaluate(
        _chunk(spec),
        episode_id="episode-2",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )

    assert result.status is ActionConsistencyStatus.WARMUP
    assert result.reason == "anomaly_reset"
    assert result.previous_origin_control_step is None


def test_default_replan_compares_same_global_steps() -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    previous = _chunk(spec)
    previous[:, 0] = np.linspace(-0.75, 0.75, spec.action_horizon, dtype=np.float32)
    current = _chunk(spec, 0.75)
    current[:12] = previous[4:]
    critic.evaluate(
        previous,
        episode_id="episode",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )

    result = critic.evaluate(
        current,
        episode_id="episode",
        origin_control_step=4,
        observed_gripper_opening=0.5,
    )

    assert result.status is ActionConsistencyStatus.SCORED
    assert result.advance_steps == 4
    assert result.overlap_steps == 12
    assert result.arm_tce_mse_normalized == pytest.approx(0.0)
    assert result.gripper_tce_mse_normalized == pytest.approx(0.0)


def test_arm_and_gripper_tce_are_reported_separately() -> None:
    spec = RobotSpec()
    arm_critic = ActionConsistencyCritic(spec)
    previous = _chunk(spec)
    arm_critic.evaluate(
        previous,
        episode_id="episode",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )
    arm_current = _chunk(spec)
    arm_current[:12, 0] = 0.5

    arm_result = arm_critic.evaluate(
        arm_current,
        episode_id="episode",
        origin_control_step=4,
        observed_gripper_opening=0.5,
    )

    assert arm_result.arm_tce_mse_normalized == pytest.approx(0.25 / 7.0)
    assert arm_result.gripper_tce_mse_normalized == pytest.approx(0.0)

    gripper_critic = ActionConsistencyCritic(spec)
    gripper_critic.evaluate(
        previous,
        episode_id="episode",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )
    gripper_current = _chunk(spec)
    gripper_current[:12, 7] = 1.0
    gripper_result = gripper_critic.evaluate(
        gripper_current,
        episode_id="episode",
        origin_control_step=4,
        observed_gripper_opening=0.5,
    )

    assert gripper_result.arm_tce_mse_normalized == pytest.approx(0.0)
    assert gripper_result.gripper_tce_mse_normalized == pytest.approx(1.0)


def test_acm_only_uses_executed_prefix_and_gripper_target_transitions() -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    chunk = _chunk(spec)
    chunk[spec.execute_steps :, : spec.arm_dof] = 1.0
    chunk[: spec.execute_steps, 7] = 1.0

    stationary = critic.evaluate(
        chunk,
        episode_id="stationary",
        origin_control_step=0,
        observed_gripper_opening=1.0,
    )

    assert stationary.arm_acm_rms_normalized == pytest.approx(0.0)
    assert stationary.gripper_transition_rms_opening_ratio == pytest.approx(0.0)

    critic.reset(episode_id="transition")
    transitioned = critic.evaluate(
        chunk,
        episode_id="transition",
        origin_control_step=0,
        observed_gripper_opening=0.0,
    )
    assert transitioned.gripper_transition_rms_opening_ratio == pytest.approx(0.5)


@pytest.mark.parametrize(
    ("advance", "expected_overlap"),
    [(1, 15), (15, 1)],
)
def test_variable_control_step_advance_sets_exact_overlap(
    advance: int,
    expected_overlap: int,
) -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    critic.evaluate(
        _chunk(spec),
        episode_id="episode",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )

    result = critic.evaluate(
        _chunk(spec),
        episode_id="episode",
        origin_control_step=advance,
        observed_gripper_opening=0.5,
    )

    assert result.status is ActionConsistencyStatus.SCORED
    assert result.overlap_steps == expected_overlap


def test_no_overlap_abstains_and_current_chunk_becomes_recovery_baseline() -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    critic.evaluate(
        _chunk(spec, -0.5),
        episode_id="episode",
        origin_control_step=0,
        observed_gripper_opening=0.25,
    )

    abstained = critic.evaluate(
        _chunk(spec, 0.25),
        episode_id="episode",
        origin_control_step=16,
        observed_gripper_opening=0.25,
    )
    recovered = critic.evaluate(
        _chunk(spec, 0.25),
        episode_id="episode",
        origin_control_step=20,
        observed_gripper_opening=0.25,
    )

    assert abstained.status is ActionConsistencyStatus.ABSTAIN
    assert abstained.reason == "no_temporal_overlap"
    assert abstained.arm_tce_mse_normalized is None
    assert recovered.status is ActionConsistencyStatus.SCORED
    assert recovered.previous_origin_control_step == 16
    assert recovered.arm_tce_mse_normalized == pytest.approx(0.0)


@pytest.mark.parametrize("invalid_origin", [4, 3])
def test_non_monotonic_step_abstains_without_replacing_history(
    invalid_origin: int,
) -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    baseline = _chunk(spec, 0.25)
    critic.evaluate(
        baseline,
        episode_id="episode",
        origin_control_step=4,
        observed_gripper_opening=0.25,
    )

    abstained = critic.evaluate(
        _chunk(spec, -0.75),
        episode_id="episode",
        origin_control_step=invalid_origin,
        observed_gripper_opening=0.25,
    )
    recovered = critic.evaluate(
        baseline,
        episode_id="episode",
        origin_control_step=8,
        observed_gripper_opening=0.25,
    )

    assert abstained.status is ActionConsistencyStatus.ABSTAIN
    assert abstained.reason == "non_monotonic_control_step"
    assert recovered.status is ActionConsistencyStatus.SCORED
    assert recovered.previous_origin_control_step == 4
    assert recovered.arm_tce_mse_normalized == pytest.approx(0.0)


def test_episode_mismatch_abstains_until_explicit_reset() -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    critic.evaluate(
        _chunk(spec),
        episode_id="episode-1",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )

    mismatch = critic.evaluate(
        _chunk(spec),
        episode_id="episode-2",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )
    critic.reset(episode_id="episode-2")
    warmup = critic.evaluate(
        _chunk(spec),
        episode_id="episode-2",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )

    assert mismatch.status is ActionConsistencyStatus.ABSTAIN
    assert mismatch.reason == "episode_identity_mismatch"
    assert mismatch.advance_steps is None
    assert warmup.status is ActionConsistencyStatus.WARMUP


def test_mark_unavailable_clears_history_and_forces_warmup() -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    critic.evaluate(
        _chunk(spec),
        episode_id="episode",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )

    unavailable = critic.mark_unavailable(
        episode_id="episode",
        origin_control_step=4,
        reason="inference_unavailable",
    )
    warmup = critic.evaluate(
        _chunk(spec),
        episode_id="episode",
        origin_control_step=4,
        observed_gripper_opening=0.5,
    )

    assert unavailable.status is ActionConsistencyStatus.ABSTAIN
    assert unavailable.arm_acm_rms_normalized is None
    assert warmup.status is ActionConsistencyStatus.WARMUP
    assert warmup.reason == "inference_unavailable"


@pytest.mark.parametrize(
    "mutator",
    [
        lambda chunk: np.zeros((15, 8), dtype=np.float32),
        lambda chunk: np.full_like(chunk, np.nan),
        lambda chunk: np.full_like(chunk, np.inf),
        lambda chunk: np.full_like(chunk, 1.01),
    ],
)
def test_invalid_action_does_not_pollute_history(mutator) -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    baseline = _chunk(spec, 0.25)
    critic.evaluate(
        baseline,
        episode_id="episode",
        origin_control_step=0,
        observed_gripper_opening=0.25,
    )

    with pytest.raises(ValueError):
        critic.evaluate(
            mutator(baseline),
            episode_id="episode",
            origin_control_step=4,
            observed_gripper_opening=0.25,
        )

    recovered = critic.evaluate(
        baseline,
        episode_id="episode",
        origin_control_step=4,
        observed_gripper_opening=0.25,
    )
    assert recovered.status is ActionConsistencyStatus.SCORED
    assert recovered.previous_origin_control_step == 0
    assert recovered.arm_tce_mse_normalized == pytest.approx(0.0)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"episode_id": "", "origin_control_step": 0, "observed_gripper_opening": 0.5},
        {"episode_id": "episode", "origin_control_step": -1, "observed_gripper_opening": 0.5},
        {"episode_id": "episode", "origin_control_step": 0, "observed_gripper_opening": -0.1},
        {"episode_id": "episode", "origin_control_step": 0, "observed_gripper_opening": 1.1},
    ],
)
def test_metadata_validation_rejects_invalid_values(kwargs) -> None:
    with pytest.raises(ValueError):
        ActionConsistencyCritic(RobotSpec()).evaluate(_chunk(RobotSpec()), **kwargs)


def test_stored_chunk_is_independent_from_caller_mutation() -> None:
    spec = RobotSpec()
    critic = ActionConsistencyCritic(spec)
    previous = _chunk(spec)
    critic.evaluate(
        previous,
        episode_id="episode",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )
    previous[:] = 1.0

    result = critic.evaluate(
        _chunk(spec),
        episode_id="episode",
        origin_control_step=4,
        observed_gripper_opening=0.5,
    )

    assert result.arm_tce_mse_normalized == pytest.approx(0.0)
    assert result.gripper_tce_mse_normalized == pytest.approx(0.0)


def test_result_serializes_to_strict_json_without_nan() -> None:
    spec = RobotSpec()
    result = ActionConsistencyCritic(spec).evaluate(
        _chunk(spec),
        episode_id="episode",
        origin_control_step=0,
        observed_gripper_opening=0.5,
    )

    payload = result.to_dict()
    encoded = json.dumps(payload, allow_nan=False)

    assert '"status": "warmup"' in encoded
    assert payload["arm_tce_mse_normalized"] is None
    assert payload["gripper_tce_mse_normalized"] is None
