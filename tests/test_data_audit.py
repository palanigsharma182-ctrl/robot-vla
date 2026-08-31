from __future__ import annotations

import json
from dataclasses import replace

import numpy as np
import pytest

from robot_vla.contracts import (
    FINGER_FORCE_SENSOR_VERSION,
    OBSERVATION_V2_VERSION,
    OUTCOME_PREDICATE_VERSION,
    RobotSpec,
)
from robot_vla.data.audit import _audit_recovery_metadata, audit_dataset
from robot_vla.data.events import EVENT_STATE_CONTRACT_VERSION
from robot_vla.data.recovery import RECOVERY_CONTRACT_VERSION
from robot_vla.data.trajectory import OutcomeEvidence
from robot_vla.data.writer import TrajectoryDatasetWriter


def _evidence() -> OutcomeEvidence:
    return OutcomeEvidence(
        predicate_version=OUTCOME_PREDICATE_VERSION,
        task_completed=True,
        final_is_released=True,
        stable_place_steps=4,
        external_goal_visible_steps=5,
        wrist_goal_visible_steps=5,
        both_goal_visible_steps=5,
        final_object_to_goal_distance_m=0.01,
        final_object_linear_speed_m_s=0.005,
        final_object_angular_speed_rad_s=0.1,
    )


def _as_observation_v2(meta, arrays, spec: RobotSpec, *, force_scale: float):
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
    previous_action_valid = np.ones(steps, dtype=np.bool_)
    previous_action_valid[0] = False
    timestamp = arrays.timestamp_action.copy()
    rotation = np.tile(
        np.asarray((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=np.float32),
        (steps, 1),
    )
    left = (np.arange(steps, dtype=np.float32) + 1.0) * force_scale
    right = left * 2.0
    valid = np.ones(steps, dtype=np.bool_)
    arrays = replace(
        arrays,
        proprio=proprio,
        robot_object_contact_force_n=np.maximum(left, right),
        support_contact_force_n=np.ones(steps, dtype=np.float32),
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


def _write_three_splits(
    root,
    meta_factory,
    arrays_factory,
    *,
    bad_skill=False,
    observation_v2=False,
) -> None:
    spec = RobotSpec()
    writer = TrajectoryDatasetWriter(root, RobotSpec())
    for index, split in enumerate(("train", "val", "test")):
        meta = replace(
            meta_factory(),
            trajectory_id=f"episode-{index:03d}",
            source_episode_id=f"source-{index:03d}",
            file=f"trajectories/episode-{index:03d}.npz",
            split=split,
            scene_id=f"scene-{index:03d}",
            outcome_evidence=_evidence(),
        )
        arrays = arrays_factory()
        if observation_v2:
            meta, arrays = _as_observation_v2(
                meta,
                arrays,
                spec,
                force_scale=1.0 if split == "train" else 1000.0,
            )
        if bad_skill and index == 0:
            arrays = replace(
                arrays,
                skill_id=np.asarray([0, 1, 1, 3, 4], dtype=np.int16),
            )
        writer.write(meta, arrays)


def test_audit_builds_train_only_stats_and_reproducible_hash(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    _write_three_splits(tmp_path, meta_factory, arrays_factory)

    first = audit_dataset(tmp_path, RobotSpec())
    second = audit_dataset(tmp_path, RobotSpec(), write_artifacts=False)

    assert first == second
    assert first.success_rate == 1.0
    assert first.split_trajectory_counts == {"train": 1, "val": 1, "test": 1}
    assert first.skill_frame_counts == {
        "reach": 3,
        "grasp": 3,
        "lift": 3,
        "transport": 3,
        "place": 3,
    }
    assert first.proprio_stats_count == 5
    stats = json.loads((tmp_path / "proprio_stats.json").read_text(encoding="utf-8"))
    assert stats["count"] == 5
    assert (tmp_path / "audit_report.json").is_file()


def test_audit_rejects_missing_or_skipped_atomic_skill(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    _write_three_splits(tmp_path, meta_factory, arrays_factory, bad_skill=True)

    with pytest.raises(ValueError, match="原子技能覆盖"):
        audit_dataset(tmp_path, RobotSpec(), write_artifacts=False)


def test_audit_requires_versioned_recovery_step_evidence(meta_factory) -> None:
    valid = replace(
        meta_factory(),
        num_steps=5,
        randomization={
            "seed": 7,
            "recovery_profile": "reach",
            "recovery_contract_version": RECOVERY_CONTRACT_VERSION,
            "recovery_evidence": {
                "disturbance_end_step": 1,
                "successful_recovery_end_step": 4,
            },
        },
    )
    _audit_recovery_metadata(valid, 5)

    invalid = replace(
        valid,
        randomization={**valid.randomization, "recovery_evidence": None},
    )
    with pytest.raises(TypeError, match="recovery_evidence"):
        _audit_recovery_metadata(invalid, 5)


def test_audit_freezes_observation_v2_force_stats_from_train_only(
    tmp_path,
    meta_factory,
    arrays_factory,
) -> None:
    _write_three_splits(
        tmp_path,
        meta_factory,
        arrays_factory,
        observation_v2=True,
    )

    report = audit_dataset(tmp_path, RobotSpec())
    force_stats = json.loads(
        (tmp_path / "finger_force_stats.json").read_text(encoding="utf-8")
    )

    assert report.observation_v2_trajectory_count == 3
    assert report.observation_v2_step_count == 15
    assert report.observation_v2_valid_step_count == 15
    assert report.observation_v2_coverage_rate == 1.0
    assert not any(report.observation_v2_modality_invalid_counts.values())
    assert set(report.observation_v2_modality_max_age_s.values()) == {0.0}
    assert report.finger_force_sensor_version == FINGER_FORCE_SENSOR_VERSION
    assert report.finger_force_stats_count == 5
    assert report.finger_force_positive_counts == {"left": 15, "right": 15}
    assert report.action_semantic_parity_step_count == 15
    assert force_stats["count"] == 5
    assert force_stats["positive_count"] == [5, 5]
    expected_left = float(np.quantile(np.log1p(np.arange(1.0, 6.0)), 0.95))
    expected_right = float(np.quantile(np.log1p(np.arange(1.0, 6.0) * 2.0), 0.95))
    np.testing.assert_allclose(
        force_stats["scale_log1p_p95"],
        (expected_left, expected_right),
    )
