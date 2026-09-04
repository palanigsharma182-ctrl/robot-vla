import numpy as np
import pytest

from robot_vla.adapters import (
    ActionAdapter,
    ActionContractViolation,
    FingerForceNormalizer,
    FingerForceStats,
    FrankaObservationAdapter,
    ProprioNormalizer,
    ProprioStats,
)
from robot_vla.contracts import RobotSpec


def _maniskill_state() -> tuple[np.ndarray, np.ndarray]:
    qpos = np.asarray(
        (0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0, 0.02, 0.02),
        dtype=np.float32,
    )
    qvel = np.asarray((0.1, -0.1, 0.0, 0.0, 0.2, 0.0, -0.2, 0.0, 0.0), dtype=np.float32)
    return qpos, qvel


def test_observation_adapter_maps_two_fingers_to_one_opening_ratio() -> None:
    spec = RobotSpec()
    qpos, qvel = _maniskill_state()

    proprio = FrankaObservationAdapter(spec).from_maniskill(
        qpos,
        qvel,
        spec.active_joint_names,
    )

    assert proprio.shape == (15,)
    np.testing.assert_allclose(proprio[:7], qpos[:7])
    np.testing.assert_allclose(proprio[7:14], qvel[:7])
    assert proprio[-1] == pytest.approx(0.5)


def test_observation_adapter_rejects_joint_order_drift() -> None:
    spec = RobotSpec()
    qpos, qvel = _maniskill_state()
    wrong_order = list(spec.active_joint_names)
    wrong_order[0], wrong_order[1] = wrong_order[1], wrong_order[0]

    with pytest.raises(ValueError, match="joint 顺序"):
        FrankaObservationAdapter(spec).from_maniskill(qpos, qvel, wrong_order)


def test_gripper_opening_inverse_mapping() -> None:
    adapter = FrankaObservationAdapter(RobotSpec())

    finger_positions = adapter.gripper_joint_positions(
        np.asarray((0.0, 0.5, 1.0), dtype=np.float32)
    )

    np.testing.assert_allclose(
        finger_positions,
        np.asarray(((0.0, 0.0), (0.02, 0.02), (0.04, 0.04)), dtype=np.float32),
    )


def test_action_adapter_round_trip_uses_physical_gripper_opening() -> None:
    adapter = ActionAdapter(RobotSpec())
    physical = np.asarray([[0.025] * 7 + [0.25]], dtype=np.float32)

    normalized = adapter.normalize(physical)
    reconstructed = adapter.denormalize(normalized)

    np.testing.assert_allclose(normalized[0, :7], 0.5)
    assert normalized[0, -1] == pytest.approx(-0.5)
    np.testing.assert_allclose(reconstructed, physical)


def test_action_adapter_maps_project_limits_to_maniskill_controller() -> None:
    adapter = ActionAdapter(RobotSpec())
    physical = np.asarray([0.05] * 7 + [0.0], dtype=np.float32)

    controller_action = adapter.to_maniskill(physical)

    np.testing.assert_allclose(controller_action[:7], 0.5)
    assert controller_action[-1] == pytest.approx(-1.0)


def test_action_adapter_rejects_out_of_contract_action() -> None:
    adapter = ActionAdapter(RobotSpec())

    with pytest.raises(ValueError, match="物理 Action"):
        adapter.normalize(np.asarray([0.051] + [0.0] * 6 + [0.5], dtype=np.float32))

    with pytest.raises(ValueError, match="物理 Action"):
        adapter.normalize(np.asarray([0.0] * 7 + [1.01], dtype=np.float32))


def test_action_contract_violation_identifies_arm_and_gripper_dimensions() -> None:
    adapter = ActionAdapter(RobotSpec())
    physical = np.asarray([0.051] + [0.0] * 6 + [1.01], dtype=np.float32)

    with pytest.raises(ActionContractViolation) as caught:
        adapter.normalize(physical)

    diagnostic = caught.value.to_diagnostic()
    assert diagnostic["kind"] == "physical_action_contract"
    assert diagnostic["arm_violation_indices"] == [[0]]
    assert diagnostic["gripper_violation_indices"] == [[]]
    assert diagnostic["physical_action"] == pytest.approx(physical.tolist())


def test_receding_horizon_commands_integrate_from_supplied_command_reference() -> None:
    spec = RobotSpec()
    adapter = ActionAdapter(spec)
    chunk = np.zeros((spec.action_horizon, spec.action_dim), dtype=np.float32)
    chunk[:, 0] = 0.01
    chunk[:, -1] = 0.5
    initial_q = np.asarray((0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)

    first = adapter.build_receding_horizon_commands(initial_q, chunk)
    next_command_reference = initial_q.copy()
    next_command_reference[0] = 0.005
    second = adapter.build_receding_horizon_commands(next_command_reference, chunk)

    np.testing.assert_allclose(first.joint_position_targets[:, 0], (0.01, 0.02, 0.03, 0.04))
    np.testing.assert_allclose(second.joint_position_targets[:, 0], (0.015, 0.025, 0.035, 0.045))
    np.testing.assert_allclose(first.gripper_opening_targets, 0.5)


def test_receding_horizon_rejects_position_limit_violation() -> None:
    spec = RobotSpec()
    chunk = np.zeros((spec.action_horizon, spec.action_dim), dtype=np.float32)
    chunk[:, 0] = 0.01
    chunk[:, -1] = 0.5
    q_base = np.asarray((2.895, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32)

    with pytest.raises(ValueError, match="关节目标"):
        ActionAdapter(spec).build_receding_horizon_commands(q_base, chunk)


def test_proprio_stats_round_trip_and_clipping(tmp_path) -> None:
    spec = RobotSpec()
    first = np.zeros((2, spec.proprio_dim), dtype=np.float32)
    second = np.full((1, spec.proprio_dim), 3.0, dtype=np.float32)
    stats = ProprioStats.fit((first, second), spec)
    path = tmp_path / "proprio_stats.json"
    stats.to_json(path)
    restored = ProprioStats.from_json(path)
    normalizer = ProprioNormalizer(restored, spec)

    restored.validate(spec)
    assert restored.count == 3
    value = np.asarray(stats.mean, dtype=np.float32)
    np.testing.assert_allclose(normalizer.normalize(value), 0.0, atol=1e-6)
    extreme = value + np.asarray(stats.std, dtype=np.float32) * 100.0
    np.testing.assert_allclose(normalizer.normalize(extreme), 5.0)


def test_finger_force_stats_are_train_fitted_versioned_and_round_trip(tmp_path) -> None:
    spec = RobotSpec()
    train_force = np.asarray(
        ((0.0, 0.0), (1.0, 2.0), (3.0, 4.0), (1000.0, 2000.0)),
        dtype=np.float32,
    )
    stats = FingerForceStats.fit((train_force,), spec, quantile=0.75, clip=1.5)
    path = tmp_path / "finger_force_stats.json"
    stats.to_json(path)
    restored = FingerForceStats.from_json(path)
    restored.validate(spec)
    assert restored == stats
    assert restored.count == 4
    assert restored.positive_count == (3, 3)

    normalizer = FingerForceNormalizer(restored, spec)
    normalized = normalizer.normalize(train_force[:3])
    assert normalized.dtype == np.float32
    np.testing.assert_array_equal(normalized[0], (0.0, 0.0))
    assert np.all(normalized >= 0.0)
    assert np.all(normalized <= restored.clip)
    np.testing.assert_allclose(
        normalizer.denormalize(normalized[:2]),
        train_force[:2],
        atol=1e-6,
    )


def test_finger_force_stats_reject_missing_positive_finger() -> None:
    spec = RobotSpec()
    force = np.asarray(((0.0, 0.0), (1.0, 0.0)), dtype=np.float32)
    with pytest.raises(ValueError, match="F_R"):
        FingerForceStats.fit((force,), spec)
