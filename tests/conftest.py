import json
from dataclasses import replace

import numpy as np
import pytest

from robot_vla.contracts import TRAJECTORY_SCHEMA_VERSION, TaskSpec
from robot_vla.data.trajectory import CameraCalibration, TrajectoryArrays, TrajectoryMeta
from robot_vla.model.qwen_processor import QwenVLAProcessorAdapter


@pytest.fixture
def calibration() -> CameraCalibration:
    intrinsic = (100.0, 0.0, 8.0, 0.0, 100.0, 8.0, 0.0, 0.0, 1.0)
    identity = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
    )
    return CameraCalibration(
        version="maniskill-pickcube-calibration/v1",
        intrinsic_external=intrinsic,
        intrinsic_wrist=intrinsic,
        world_from_external=identity,
        tcp_from_wrist=identity,
    )


@pytest.fixture
def meta_factory(calibration):
    def factory(**overrides) -> TrajectoryMeta:
        meta = TrajectoryMeta(
            schema_version=TRAJECTORY_SCHEMA_VERSION,
            trajectory_id="episode-000",
            source_episode_id="source-000",
            file="trajectories/episode-000.npz",
            split="train",
            scene_id="scene-seed-000",
            task=TaskSpec(
                task_id="pick-cube-to-region",
                task_group_id="pick-and-place",
                instruction="Pick up the red cube and place it in the target region.",
            ),
            num_steps=5,
            camera_calibration=calibration,
            randomization={"seed": 7, "cube_color": "red"},
        )
        return replace(meta, **overrides)

    return factory


@pytest.fixture
def arrays_factory():
    def factory(*, steps: int = 5, **overrides) -> TrajectoryArrays:
        timestamp = np.arange(steps, dtype=np.float64) * 0.05
        proprio = np.zeros((steps, 15), dtype=np.float32)
        proprio[:, :7] = np.asarray(
            (0.0, -0.5, 0.0, -1.5, 0.0, 1.5, 0.0), dtype=np.float32
        )
        proprio[:, -1] = 0.5
        action = np.zeros((steps, 8), dtype=np.float32)
        action[:, :7] = 0.025
        action[:, -1] = 0.5
        terminated = np.zeros(steps, dtype=np.bool_)
        terminated[-1] = True
        arrays = TrajectoryArrays(
            rgb_external=np.zeros((steps, 16, 20, 3), dtype=np.uint8),
            rgb_wrist=np.zeros((steps, 12, 12, 3), dtype=np.uint8),
            timestamp_external=timestamp.copy(),
            timestamp_wrist=timestamp.copy(),
            timestamp_proprio=timestamp.copy(),
            timestamp_action=timestamp.copy(),
            proprio=proprio,
            action=action,
            external_valid=np.ones(steps, dtype=np.bool_),
            wrist_valid=np.ones(steps, dtype=np.bool_),
            proprio_valid=np.ones(steps, dtype=np.bool_),
            terminated=terminated,
            truncated=np.zeros(steps, dtype=np.bool_),
            success=np.asarray([False] * (steps - 1) + [True], dtype=np.bool_),
            skill_id=np.arange(steps, dtype=np.int16) % 5,
        )
        return replace(arrays, **overrides)

    return factory


@pytest.fixture
def write_dataset(tmp_path):
    def write(meta: TrajectoryMeta, arrays: TrajectoryArrays) -> None:
        path = tmp_path / meta.file
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            path,
            **{
                name: getattr(arrays, name)
                for name in arrays.__dataclass_fields__
                if getattr(arrays, name) is not None
            },
        )
        (tmp_path / "manifest.jsonl").write_text(
            json.dumps(meta.to_dict()) + "\n",
            encoding="utf-8",
        )

    return write


@pytest.fixture(scope="session")
def qwen_processor_adapter(tmp_path_factory) -> QwenVLAProcessorAdapter:
    pytest.importorskip("transformers")
    cache_dir = tmp_path_factory.mktemp("qwen-processor-cache")
    return QwenVLAProcessorAdapter.from_pretrained(cache_dir=str(cache_dir))
