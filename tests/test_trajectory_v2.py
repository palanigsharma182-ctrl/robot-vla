import json
from dataclasses import replace

import numpy as np
import pytest

from robot_vla.contracts import (
    FINGER_FORCE_SENSOR_VERSION,
    OBSERVATION_V2_VERSION,
    RobotSpec,
)
from robot_vla.data.events import EVENT_STATE_CONTRACT_VERSION
from robot_vla.data.trajectory import (
    ACTION_SOURCE_EXPERT,
    ACTION_SOURCE_POLICY,
    LocalDaggerProvenance,
    TrajectoryStore,
    load_manifest,
    validate_trajectory,
)
from robot_vla.local_dagger_protocol import (
    LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD,
    LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD,
    resolve_local_dagger_action_budget,
)


def _local_dagger_episode(meta_factory, arrays_factory, *, steps: int = 80, takeover: int = 4):
    source = np.full(steps, ACTION_SOURCE_EXPERT, dtype=np.int8)
    source[:takeover] = ACTION_SOURCE_POLICY
    supervision = source == ACTION_SOURCE_EXPERT
    arrays = arrays_factory(
        steps=steps,
        action_source=source,
        expert_supervision_mask=supervision,
    )
    meta = meta_factory(
        num_steps=steps,
        local_dagger=LocalDaggerProvenance(
            source="dagger_reach_grasp",
            rollin_seed=7,
            rollin_policy_checkpoint_sha256="a" * 64,
            boundary_type="reach_grasp",
            boundary_detection_step=takeover,
            expert_takeover_step=takeover,
            training_window_start=takeover,
            training_window_end=min(takeover + 64, steps),
            expert_recovery_success=True,
        ),
    )
    return meta, arrays


def _with_segmented_budget(meta, *, steps: int, takeover: int):
    plan = resolve_local_dagger_action_budget("segmented-300-180-480")
    planned = plan.planned_metadata()
    usage = plan.usage_metadata(
        total_actions=steps,
        expert_takeover_step=takeover,
    )
    assert planned is not None and usage is not None
    return replace(
        meta,
        randomization={
            **meta.randomization,
            LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: planned,
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: usage,
        },
    )


def _with_observation_v2(meta, arrays, spec: RobotSpec):
    steps = arrays.num_steps
    previous_command = np.empty((steps, spec.arm_dof), dtype=np.float32)
    commanded = np.empty_like(previous_command)
    proprio = arrays.proprio.copy()
    previous_command[0] = proprio[0, : spec.arm_dof]
    for step in range(steps):
        if step > 0:
            previous_command[step] = commanded[step - 1]
        proprio[step, : spec.arm_dof] = previous_command[step]
        commanded[step] = previous_command[step] + arrays.action[step, : spec.arm_dof]
    previous_action = np.zeros((steps, spec.action_dim), dtype=np.float32)
    previous_action[1:] = arrays.action[:-1]
    rotation = np.tile(
        np.asarray((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=np.float32),
        (steps, 1),
    )
    timestamp = arrays.timestamp_action.copy()
    left = np.linspace(0.0, 2.0, steps, dtype=np.float32)
    right = np.linspace(0.5, 1.0, steps, dtype=np.float32)
    valid = np.ones(steps, dtype=np.bool_)
    previous_action_valid = valid.copy()
    previous_action_valid[0] = False
    arrays = replace(
        arrays,
        proprio=proprio,
        robot_object_contact_force_n=np.maximum(left, right),
        support_contact_force_n=np.zeros(steps, dtype=np.float32),
        is_grasped=np.zeros(steps, dtype=np.bool_),
        object_position_m=np.zeros((steps, 3), dtype=np.float32),
        object_linear_velocity_m_s=np.zeros((steps, 3), dtype=np.float32),
        object_angular_velocity_rad_s=np.zeros((steps, 3), dtype=np.float32),
        commanded_joint_target_rad=commanded,
        applied_joint_correction_rad=arrays.action[:, : spec.arm_dof].copy(),
        timestamp_tcp_pose=timestamp.copy(),
        timestamp_camera_pose=timestamp.copy(),
        timestamp_finger_force=timestamp.copy(),
        tcp_position_base_m=np.zeros((steps, 3), dtype=np.float32),
        tcp_rotation_6d_base=rotation.copy(),
        wrist_camera_position_base_m=np.zeros((steps, 3), dtype=np.float32),
        wrist_camera_rotation_6d_base=rotation.copy(),
        left_finger_force_n=left,
        right_finger_force_n=right,
        tcp_pose_valid=valid.copy(),
        camera_pose_valid=valid.copy(),
        finger_force_valid=valid.copy(),
        previous_command_q_rad=previous_command,
        previous_action=previous_action,
        previous_command_valid=valid.copy(),
        previous_action_valid=previous_action_valid,
    )
    meta = replace(
        meta,
        randomization={
            **meta.randomization,
            "event_state_contract_version": EVENT_STATE_CONTRACT_VERSION,
            "observation_contract_version": OBSERVATION_V2_VERSION,
            "finger_force_sensor_version": FINGER_FORCE_SENSOR_VERSION,
        },
    )
    return meta, arrays


def test_manifest_and_trajectory_v2_round_trip(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    meta = meta_factory()
    arrays = arrays_factory()
    write_dataset(meta, arrays)

    loaded_meta = load_manifest(tmp_path, split="train")[0]
    loaded = TrajectoryStore(tmp_path, RobotSpec()).get(loaded_meta)

    assert loaded_meta == meta
    assert loaded.rgb_external.shape == (5, 16, 20, 3)
    assert loaded.rgb_wrist.shape == (5, 12, 12, 3)
    assert loaded.proprio.shape == (5, 15)
    assert loaded.action.shape == (5, 8)
    assert loaded.observation_valid.tolist() == [True] * 5


def test_observation_v2_round_trip_and_action_semantics(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    spec = RobotSpec()
    meta, arrays = _with_observation_v2(meta_factory(), arrays_factory(), spec)
    write_dataset(meta, arrays)

    loaded_meta = load_manifest(tmp_path, split="train")[0]
    loaded = TrajectoryStore(tmp_path, spec).get(loaded_meta)

    assert loaded.observation_v2_available is True
    assert loaded.observation_v2_valid.all()
    np.testing.assert_allclose(
        loaded.commanded_joint_target_rad,
        loaded.previous_command_q_rad + loaded.action[:, : spec.arm_dof],
    )
    np.testing.assert_allclose(loaded.previous_action[1:], loaded.action[:-1])
    assert loaded.previous_action_valid.tolist() == [False, True, True, True, True]


def test_observation_v2_rejects_action_label_execution_mismatch(
    meta_factory,
    arrays_factory,
) -> None:
    spec = RobotSpec()
    meta, arrays = _with_observation_v2(meta_factory(), arrays_factory(), spec)
    commanded = arrays.commanded_joint_target_rad.copy()
    commanded[2, 0] += 0.001
    arrays = replace(arrays, commanded_joint_target_rad=commanded)

    with pytest.raises(ValueError, match="Action 标签.*执行语义"):
        validate_trajectory(arrays, meta, spec)


def test_observation_v2_rejects_fabricated_symmetric_finger_force(
    meta_factory,
    arrays_factory,
) -> None:
    spec = RobotSpec()
    meta, arrays = _with_observation_v2(meta_factory(), arrays_factory(), spec)
    right = arrays.left_finger_force_n.copy()
    arrays = replace(arrays, right_finger_force_n=right)

    with pytest.raises(ValueError, match="aggregate contact force"):
        validate_trajectory(arrays, meta, spec)


def test_invalid_camera_marks_index_unavailable_without_fabricating_input(
    meta_factory,
    arrays_factory,
) -> None:
    wrist_valid = np.ones(5, dtype=np.bool_)
    wrist_valid[2] = False
    arrays = arrays_factory(wrist_valid=wrist_valid)

    validate_trajectory(arrays, meta_factory(), RobotSpec())

    assert arrays.observation_valid.tolist() == [True, True, False, True, True]


def test_future_sensor_observation_is_rejected(meta_factory, arrays_factory) -> None:
    timestamp_external = np.arange(5, dtype=np.float64) * 0.05
    timestamp_external[0] = 0.001
    arrays = arrays_factory(timestamp_external=timestamp_external)

    with pytest.raises(ValueError, match="未来观测"):
        validate_trajectory(arrays, meta_factory(), RobotSpec())


def test_wrong_dtype_is_rejected(meta_factory, arrays_factory) -> None:
    arrays = arrays_factory(proprio=np.zeros((5, 15), dtype=np.float64))

    with pytest.raises(ValueError, match="proprio dtype"):
        validate_trajectory(arrays, meta_factory(), RobotSpec())


def test_out_of_range_action_is_rejected(meta_factory, arrays_factory) -> None:
    action = np.zeros((5, 8), dtype=np.float32)
    action[:, 0] = 0.051
    action[:, -1] = 0.5

    with pytest.raises(ValueError, match="delta_q"):
        validate_trajectory(arrays_factory(action=action), meta_factory(), RobotSpec())


def test_unknown_skill_uses_only_reserved_value(meta_factory, arrays_factory) -> None:
    accepted = arrays_factory(skill_id=np.full(5, -1, dtype=np.int16))
    validate_trajectory(accepted, meta_factory(), RobotSpec())

    rejected = arrays_factory(skill_id=np.full(5, 5, dtype=np.int16))
    with pytest.raises(ValueError, match="skill_id"):
        validate_trajectory(rejected, meta_factory(), RobotSpec())


def test_manifest_rejects_scene_split_leakage(tmp_path, meta_factory) -> None:
    train = meta_factory()
    val = replace(
        train,
        trajectory_id="episode-001",
        source_episode_id="source-001",
        file="trajectories/episode-001.npz",
        split="val",
    )
    (tmp_path / "manifest.jsonl").write_text(
        json.dumps(train.to_dict()) + "\n" + json.dumps(val.to_dict()) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="scene_id=.*跨越 split"):
        load_manifest(tmp_path)


def test_complete_episode_must_end_once(meta_factory, arrays_factory) -> None:
    terminated = np.zeros(5, dtype=np.bool_)

    with pytest.raises(ValueError, match="最后一个 Transition"):
        validate_trajectory(
            arrays_factory(terminated=terminated),
            meta_factory(),
            RobotSpec(),
        )


def test_local_dagger_contract_round_trip_is_additive_to_trajectory_v2(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    meta, arrays = _local_dagger_episode(meta_factory, arrays_factory)
    write_dataset(meta, arrays)

    loaded_meta = load_manifest(tmp_path, split="train")[0]
    loaded = TrajectoryStore(tmp_path, RobotSpec()).get(loaded_meta)

    assert loaded_meta == meta
    assert loaded.action_source[:5].tolist() == [0, 0, 0, 0, 1]
    assert loaded.expert_supervision_mask[:5].tolist() == [False] * 4 + [True]


def test_local_dagger_contract_fails_closed_without_supervision_arrays(
    meta_factory,
    arrays_factory,
) -> None:
    meta, _ = _local_dagger_episode(meta_factory, arrays_factory)

    with pytest.raises(ValueError, match="缺少逐 Action"):
        validate_trajectory(arrays_factory(steps=80), meta, RobotSpec())


def test_local_dagger_contract_rejects_policy_action_marked_as_expert(
    meta_factory,
    arrays_factory,
) -> None:
    meta, arrays = _local_dagger_episode(meta_factory, arrays_factory)
    supervision = arrays.expert_supervision_mask.copy()
    supervision[0] = True

    with pytest.raises(ValueError, match="supervision_mask"):
        validate_trajectory(
            replace(arrays, expert_supervision_mask=supervision),
            meta,
            RobotSpec(),
        )


@pytest.mark.parametrize("takeover", (299, 300))
def test_segmented_local_dagger_budget_accepts_each_segment_cap_before_deadline(
    meta_factory,
    arrays_factory,
    takeover: int,
) -> None:
    meta, arrays = _local_dagger_episode(
        meta_factory,
        arrays_factory,
        steps=479,
        takeover=takeover,
    )
    meta = _with_segmented_budget(meta, steps=479, takeover=takeover)

    validate_trajectory(arrays, meta, RobotSpec())


@pytest.mark.parametrize(
    ("steps", "takeover", "message"),
    (
        (400, 301, "Policy action budget"),
        (185, 4, "Expert recovery"),
        (480, 300, "hard deadline"),
        (481, 300, "hard deadline"),
    ),
)
def test_segmented_local_dagger_budget_caps_fail_closed(
    meta_factory,
    arrays_factory,
    steps: int,
    takeover: int,
    message: str,
) -> None:
    meta, arrays = _local_dagger_episode(
        meta_factory,
        arrays_factory,
        steps=steps,
        takeover=takeover,
    )
    meta = _with_segmented_budget(meta, steps=steps, takeover=takeover)

    with pytest.raises(ValueError, match=message):
        validate_trajectory(arrays, meta, RobotSpec())


def test_segmented_local_dagger_budget_rejects_partial_or_drifted_usage(
    meta_factory,
    arrays_factory,
) -> None:
    meta, arrays = _local_dagger_episode(meta_factory, arrays_factory)
    plan = resolve_local_dagger_action_budget("segmented-300-180-480")
    planned = plan.planned_metadata()
    assert planned is not None
    partial = replace(
        meta,
        randomization={
            **meta.randomization,
            LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: planned,
        },
    )
    with pytest.raises(ValueError, match="必须同时声明"):
        validate_trajectory(arrays, partial, RobotSpec())

    explicit_null = replace(
        meta,
        randomization={
            **meta.randomization,
            LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: None,
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: None,
        },
    )
    with pytest.raises(ValueError, match="protocol 必须是对象"):
        validate_trajectory(arrays, explicit_null, RobotSpec())

    amended = _with_segmented_budget(meta, steps=80, takeover=4)
    drifted_usage = dict(
        amended.randomization[LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD]
    )
    drifted_usage["expert_actions"] -= 1
    drifted = replace(
        amended,
        randomization={
            **amended.randomization,
            LOCAL_DAGGER_ACTION_BUDGET_USAGE_FIELD: drifted_usage,
        },
    )
    with pytest.raises(ValueError, match="usage 与 trajectory"):
        validate_trajectory(arrays, drifted, RobotSpec())

    drifted_protocol = dict(
        amended.randomization[LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD]
    )
    drifted_protocol["deadline_semantics"] = "success_may_coincide_with_truncation"
    drifted = replace(
        amended,
        randomization={
            **amended.randomization,
            LOCAL_DAGGER_ACTION_BUDGET_PROTOCOL_FIELD: drifted_protocol,
        },
    )
    with pytest.raises(ValueError, match="冻结定义"):
        validate_trajectory(arrays, drifted, RobotSpec())
