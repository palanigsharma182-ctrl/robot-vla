from dataclasses import replace

import numpy as np
import pytest

from robot_vla.adapters import (
    FingerForceNormalizer,
    FingerForceStats,
    ProprioNormalizer,
    ProprioStats,
)
from robot_vla.contracts import (
    FINGER_FORCE_SENSOR_VERSION,
    OBSERVATION_V2_VERSION,
    RobotSpec,
)
from robot_vla.data.events import EVENT_STATE_CONTRACT_VERSION
from robot_vla.data.dataset import (
    ActionChunkDataset,
    CompositeActionChunkDataset,
    ObservationV2ActionChunkDataset,
)
from robot_vla.data.trajectory import LocalDaggerProvenance, load_manifest


def _normalizer(spec: RobotSpec, count: int) -> ProprioNormalizer:
    return ProprioNormalizer(
        ProprioStats(
            mean=(0.0,) * spec.proprio_dim,
            std=(1.0,) * spec.proprio_dim,
            count=count,
        ),
        spec,
    )


def _force_normalizer(spec: RobotSpec) -> FingerForceNormalizer:
    return FingerForceNormalizer(
        FingerForceStats(
            scale_log1p_p95=(1.0, 1.0),
            count=10,
            positive_count=(5, 5),
        ),
        spec,
    )


def _observation_v2(meta, arrays, spec: RobotSpec):
    steps = arrays.num_steps
    timestamp = arrays.timestamp_action.copy()
    rotation = np.tile(
        np.asarray((1.0, 0.0, 0.0, 0.0, 1.0, 0.0), dtype=np.float32),
        (steps, 1),
    )
    previous_action = np.zeros((steps, spec.action_dim), dtype=np.float32)
    previous_action[1:] = arrays.action[:-1]
    previous_action_valid = np.ones(steps, dtype=np.bool_)
    previous_action_valid[0] = False
    previous_command = np.tile(
        arrays.proprio[0, : spec.arm_dof],
        (steps, 1),
    ).astype(np.float32)
    valid = np.ones(steps, dtype=np.bool_)
    arrays = replace(
        arrays,
        robot_object_contact_force_n=np.maximum(
            np.arange(steps, dtype=np.float32),
            np.arange(steps, dtype=np.float32) + 0.5,
        ),
        support_contact_force_n=np.ones(steps, dtype=np.float32),
        is_grasped=np.zeros(steps, dtype=np.bool_),
        object_position_m=np.zeros((steps, 3), dtype=np.float32),
        object_linear_velocity_m_s=np.zeros((steps, 3), dtype=np.float32),
        object_angular_velocity_rad_s=np.zeros((steps, 3), dtype=np.float32),
        commanded_joint_target_rad=(
            previous_command + arrays.action[:, : spec.arm_dof]
        ),
        applied_joint_correction_rad=arrays.action[:, : spec.arm_dof].copy(),
        timestamp_tcp_pose=timestamp.copy(),
        timestamp_camera_pose=timestamp.copy(),
        timestamp_finger_force=timestamp.copy(),
        tcp_position_base_m=np.arange(steps, dtype=np.float32)[:, None]
        * np.ones((1, 3), dtype=np.float32),
        tcp_rotation_6d_base=rotation.copy(),
        wrist_camera_position_base_m=np.arange(steps, dtype=np.float32)[:, None]
        * np.ones((1, 3), dtype=np.float32),
        wrist_camera_rotation_6d_base=rotation.copy(),
        left_finger_force_n=np.arange(steps, dtype=np.float32),
        right_finger_force_n=np.arange(steps, dtype=np.float32) + 0.5,
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


def test_dataset_preserves_camera_roles_and_masks_episode_tail(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    spec = RobotSpec()
    meta = meta_factory()
    arrays = arrays_factory()
    arrays.rgb_external[:] = 32
    arrays.rgb_wrist[:] = 96
    write_dataset(meta, arrays)
    dataset = ActionChunkDataset(
        str(tmp_path),
        load_manifest(tmp_path, split="train"),
        spec,
        _normalizer(spec, arrays.num_steps),
    )

    sample = dataset[-1]

    assert sample["timestep"] == 4
    assert sample["rgb_external"].shape == (16, 20, 3)
    assert sample["rgb_wrist"].shape == (12, 12, 3)
    assert np.all(sample["rgb_external"] == 32)
    assert np.all(sample["rgb_wrist"] == 96)
    assert sample["action"].shape == (16, 8)
    assert sample["action_mask"].tolist() == [True] + [False] * 15
    assert sample["event_mask"].tolist() == [False] * 16
    np.testing.assert_allclose(sample["action"][0, :7], 0.5)
    assert sample["action"][0, -1] == 0.0


def test_dataset_excludes_invalid_observation_start(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    spec = RobotSpec()
    wrist_valid = np.ones(5, dtype=np.bool_)
    wrist_valid[2] = False
    arrays = arrays_factory(wrist_valid=wrist_valid)
    meta = meta_factory()
    write_dataset(meta, arrays)

    dataset = ActionChunkDataset(
        str(tmp_path),
        load_manifest(tmp_path, split="train"),
        spec,
        _normalizer(spec, arrays.num_steps),
    )

    assert len(dataset) == 4
    assert [dataset[index]["timestep"] for index in range(len(dataset))] == [0, 1, 3, 4]


def test_observation_v2_dataset_builds_contiguous_history_without_episode_leakage(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    spec = RobotSpec()
    arrays = arrays_factory()
    for timestep in range(arrays.num_steps):
        arrays.rgb_external[timestep] = timestep + 1
        arrays.rgb_wrist[timestep] = timestep + 11
    meta, arrays = _observation_v2(meta_factory(), arrays, spec)
    write_dataset(meta, arrays)
    dataset = ObservationV2ActionChunkDataset(
        str(tmp_path),
        load_manifest(tmp_path, split="train"),
        spec,
        _normalizer(spec, arrays.num_steps),
        finger_force_normalizer=_force_normalizer(spec),
    )

    early = dataset[1]
    full = dataset[4]

    np.testing.assert_array_equal(
        early["state_history_mask"],
        (False, False, True, True),
    )
    np.testing.assert_array_equal(early["rgb_external_history"][:, 0, 0, 0], (0, 0, 1, 2))
    np.testing.assert_allclose(early["frame_age_s"], (0.0, 0.0, 0.05, 0.0))
    np.testing.assert_array_equal(full["rgb_external_history"][:, 0, 0, 0], (2, 3, 4, 5))
    np.testing.assert_allclose(full["frame_age_s"], (0.15, 0.10, 0.05, 0.0), atol=1e-6)
    assert full["state_history"].shape == (4, 42)
    np.testing.assert_allclose(
        full["state_history"][:, 33:35],
        np.log1p(
            np.stack(
                (
                    np.arange(1, 5, dtype=np.float32),
                    np.arange(1, 5, dtype=np.float32) + 0.5,
                ),
                axis=-1,
            )
        ),
        atol=1e-6,
    )
    assert full["controller_state"].shape == (24,)
    assert full["controller_valid"].tolist() == [True, True]


def test_observation_v2_dataset_zeroes_invalid_historical_modality_age(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    spec = RobotSpec()
    arrays = arrays_factory()
    meta, arrays = _observation_v2(meta_factory(), arrays, spec)
    arrays.tcp_pose_valid[1] = False
    write_dataset(meta, arrays)
    dataset = ObservationV2ActionChunkDataset(
        str(tmp_path),
        load_manifest(tmp_path, split="train"),
        spec,
        _normalizer(spec, arrays.num_steps),
        finger_force_normalizer=_force_normalizer(spec),
    )

    sample = dataset[-1]

    assert not bool(sample["modality_valid"][0, 3])
    assert sample["modality_age_s"][0, 3] == 0.0
    np.testing.assert_array_equal(sample["state_history"][0, 15:24], 0.0)


def test_observation_v2_dataset_rejects_legacy_force_gap(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    spec = RobotSpec()
    write_dataset(meta_factory(), arrays_factory())

    with pytest.raises(ValueError, match="禁止伪造缺失状态"):
        ObservationV2ActionChunkDataset(
            str(tmp_path),
            load_manifest(tmp_path, split="train"),
            spec,
            _normalizer(spec, 5),
            finger_force_normalizer=_force_normalizer(spec),
        )


def test_local_dagger_dataset_only_indexes_complete_expert_window_chunks(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    spec = RobotSpec()
    steps = 80
    takeover = 4
    action_source = np.ones(steps, dtype=np.int8)
    action_source[:takeover] = 0
    supervision = action_source == 1
    actions = np.zeros((steps, spec.action_dim), dtype=np.float32)
    actions[:takeover, 0] = 0.05
    actions[:, -1] = 0.5
    arrays = arrays_factory(
        steps=steps,
        action=actions,
        action_source=action_source,
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
            training_window_end=takeover + 64,
            expert_recovery_success=True,
        ),
    )
    write_dataset(meta, arrays)

    dataset = ActionChunkDataset(
        str(tmp_path),
        load_manifest(tmp_path, split="train"),
        spec,
        _normalizer(spec, arrays.num_steps),
    )

    assert len(dataset) == 49
    assert dataset[0]["timestep"] == takeover
    assert dataset[-1]["timestep"] == takeover + 48
    assert dataset[0]["source"] == "dagger_reach_grasp"
    assert dataset[0]["boundary_offset"] == 0
    assert dataset[-1]["boundary_offset"] == 48
    assert dataset[0]["action_mask"].all()
    assert dataset[0]["supervision_mask"].all()
    assert not np.any(dataset[0]["action"][:, 0] == 1.0)


class _FakeChunkDataset:
    def __init__(self, label: str, size: int) -> None:
        self.label = label
        self.size = size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, index: int) -> dict[str, int | str]:
        return {"component": self.label, "index": index}

    def sampling_metadata(self, index: int) -> dict[str, int | str | None]:
        return {
            "episode_key": f"episode-{index // 2}",
            "task_id": "task",
            "source": "base_d0" if self.label == "base" else "dagger_reach_grasp",
            "skill_id": index,
            "boundary_offset": None if self.label == "base" else index,
        }


def test_composite_dataset_keeps_component_storage_separate() -> None:
    dataset = CompositeActionChunkDataset(
        (_FakeChunkDataset("base", 3), _FakeChunkDataset("dagger", 2))
    )

    assert len(dataset) == 5
    assert dataset[0] == {"component": "base", "index": 0}
    assert dataset[2] == {"component": "base", "index": 2}
    assert dataset[3] == {"component": "dagger", "index": 0}
    assert dataset[-1] == {"component": "dagger", "index": 1}
    assert dataset.sampling_metadata(1)["episode_key"] == (0, "episode-0")
    assert dataset.sampling_metadata(4)["episode_key"] == (1, "episode-0")
