import numpy as np

from robot_vla.adapters import ProprioNormalizer, ProprioStats
from robot_vla.contracts import RobotSpec
from robot_vla.data.dataset import ActionChunkDataset, CompositeActionChunkDataset
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
