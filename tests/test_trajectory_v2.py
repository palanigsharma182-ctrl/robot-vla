import json
from dataclasses import replace

import numpy as np
import pytest

from robot_vla.contracts import RobotSpec
from robot_vla.data.trajectory import (
    ACTION_SOURCE_EXPERT,
    ACTION_SOURCE_POLICY,
    LocalDaggerProvenance,
    TrajectoryStore,
    load_manifest,
    validate_trajectory,
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
