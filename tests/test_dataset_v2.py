import numpy as np

from robot_vla.adapters import ProprioNormalizer, ProprioStats
from robot_vla.contracts import RobotSpec
from robot_vla.data.dataset import ActionChunkDataset
from robot_vla.data.trajectory import load_manifest


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
