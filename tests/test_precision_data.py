from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from robot_vla.precision.collection import PrecisionLabelRecorder
from robot_vla.precision.data import (
    PRECISION_LABEL_ARRAYS,
    PrecisionLabelDatasetWriter,
    PrecisionLabelMeta,
    load_precision_label_manifest,
    read_precision_labels,
    validate_precision_label_arrays,
)


def _label_arrays():
    recorder = PrecisionLabelRecorder(object_actor_id=3, goal_actor_id=5)
    segmentation = np.zeros((1, 8, 10, 1), dtype=np.int16)
    segmentation[0, 3:5, 4:6, 0] = 3
    segmentation[0, 5:7, 7:9, 0] = 5
    observation = {
        "sensor_data": {
            "hand_camera": {
                "rgb": np.zeros((1, 8, 10, 3), dtype=np.uint8),
                "segmentation": segmentation,
            }
        },
        "sensor_param": {
            "hand_camera": {
                "intrinsic_cv": np.asarray(
                    [[[20.0, 0.0, 4.5], [0.0, 20.0, 3.5], [0.0, 0.0, 1.0]]],
                    dtype=np.float32,
                )
            }
        },
    }
    recorder.record(
        observation,
        timestep=0,
        timestamp_s=0.0,
        base_from_wrist_camera_cv=np.eye(4, dtype=np.float32),
        object_position_base_m=np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
        goal_position_base_m=np.asarray((0.1, 0.1, 1.0), dtype=np.float32),
    )
    return recorder.build()


def _meta() -> PrecisionLabelMeta:
    return PrecisionLabelMeta(
        trajectory_id="episode-001",
        file="labels/episode-001.npz",
        split="train",
        scene_id="scene-001",
        num_steps=1,
        source_trajectory_sha256="a" * 64,
    )


def test_precision_label_recorder_keeps_gt_outside_model_inputs() -> None:
    arrays = _label_arrays()
    validate_precision_label_arrays(arrays, _meta())

    assert arrays.object_mask.shape == (1, 8, 10)
    assert arrays.goal_mask.shape == arrays.object_mask.shape
    assert arrays.keypoint_visible.tolist() == [[True, True]]
    assert arrays.keypoint_projection_valid.tolist() == [[True, True]]
    np.testing.assert_allclose(arrays.normalized_uv[0, 0], (0.5, 0.5))


def test_precision_label_writer_roundtrip_is_strict_and_refuses_overwrite(tmp_path) -> None:
    writer = PrecisionLabelDatasetWriter(tmp_path)
    meta = _meta()
    arrays = _label_arrays()

    target = writer.write(meta, arrays)

    assert target.stat().st_mode & 0o777 == 0o600
    assert load_precision_label_manifest(tmp_path) == [meta]
    loaded = read_precision_labels(tmp_path, meta)
    for name in PRECISION_LABEL_ARRAYS:
        np.testing.assert_array_equal(getattr(loaded, name), getattr(arrays, name))
    with pytest.raises(ValueError, match="trajectory_id 已存在"):
        writer.write(meta, arrays)


def test_precision_labels_reject_visibility_or_invalid_uv_drift() -> None:
    arrays = _label_arrays()
    with pytest.raises(ValueError, match="segmentation mask"):
        validate_precision_label_arrays(
            replace(
                arrays,
                keypoint_visible=np.zeros((1, 2), dtype=np.bool_),
            ),
            _meta(),
        )

    invalid_projection = arrays.keypoint_projection_valid.copy()
    invalid_projection[0, 0] = False
    with pytest.raises(ValueError, match="normalized_uv 必须为零"):
        validate_precision_label_arrays(
            replace(arrays, keypoint_projection_valid=invalid_projection),
            _meta(),
        )
