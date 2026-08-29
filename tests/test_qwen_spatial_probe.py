from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from robot_vla.data.events import EVENT_STATE_CONTRACT_VERSION

from robot_vla.cli.probe_qwen_spatial import _extract_frozen_features
from robot_vla.contracts import RobotSpec
from robot_vla.diagnostics.qwen_spatial_probe import (
    LinearVisualTokenPositionProbe,
    QwenExternalSpatialProbeDataset,
    build_external_visual_token_layout,
    build_matched_linear_probes,
    interpret_layer12_probe,
    project_world_point_to_gl_camera,
    spatial_probe_loss,
    summarize_spatial_predictions,
    unproject_gl_camera_to_world_plane,
)

IMAGE_TOKEN_ID = 99
IDENTITY_4X4 = np.eye(4, dtype=np.float32)


def _intrinsic_100px() -> np.ndarray:
    return np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def _event_state_arrays(object_position_m: np.ndarray) -> dict[str, np.ndarray]:
    steps = object_position_m.shape[0]
    return {
        "robot_object_contact_force_n": np.zeros(steps, dtype=np.float32),
        "support_contact_force_n": np.ones(steps, dtype=np.float32),
        "is_grasped": np.zeros(steps, dtype=np.bool_),
        "object_position_m": object_position_m,
        "object_linear_velocity_m_s": np.zeros((steps, 3), dtype=np.float32),
        "object_angular_velocity_rad_s": np.zeros((steps, 3), dtype=np.float32),
        "commanded_joint_target_rad": np.zeros((steps, 7), dtype=np.float32),
        "applied_joint_correction_rad": np.zeros((steps, 7), dtype=np.float32),
    }


def test_projection_and_plane_unprojection_round_trip() -> None:
    intrinsic = _intrinsic_100px()
    point_world = np.asarray([0.2, -0.1, -2.0], dtype=np.float32)

    projection = project_world_point_to_gl_camera(
        point_world,
        intrinsic.ravel(),
        IDENTITY_4X4.ravel(),
        image_height=100,
        image_width=100,
    )

    np.testing.assert_allclose(projection.pixel_uv, [60.0, 55.0], atol=1e-6)
    assert projection.depth_m == pytest.approx(2.0)
    recovered = unproject_gl_camera_to_world_plane(
        projection.normalized_uv,
        plane_world_z_m=-2.0,
        intrinsic_cv=intrinsic.ravel(),
        world_from_camera_gl=IDENTITY_4X4.ravel(),
        image_height=100,
        image_width=100,
    )
    np.testing.assert_allclose(recovered, point_world, atol=1e-6)


def test_external_layout_uses_first_image_span_and_row_major_grid() -> None:
    input_ids = torch.tensor(
        [
            [0, 99, 99, 99, 99, 2, 99, 99, 3, 0],
            [0, 2, 99, 99, 3, 99, 99, 99, 99, 0],
        ],
        dtype=torch.long,
    )
    image_grid_thw = torch.tensor(
        [[1, 4, 4], [1, 2, 4], [1, 2, 4], [1, 4, 4]],
        dtype=torch.long,
    )

    layout = build_external_visual_token_layout(
        input_ids,
        image_grid_thw,
        image_token_id=IMAGE_TOKEN_ID,
        merge_size=2,
    )

    assert torch.nonzero(layout.mask[0]).flatten().tolist() == [1, 2, 3, 4]
    assert torch.nonzero(layout.mask[1]).flatten().tolist() == [2, 3]
    torch.testing.assert_close(
        layout.normalized_centers[0, 1:5],
        torch.tensor(
            [[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]
        ),
    )
    torch.testing.assert_close(
        layout.normalized_centers[1, 2:4],
        torch.tensor([[0.25, 0.5], [0.75, 0.5]]),
    )
    assert layout.grid_shapes.tolist() == [[2, 2], [1, 2]]


@pytest.mark.parametrize(
    ("bad_grids", "message"),
    [
        ([[1, 2, 4], [1, 2, 4]], "external"),
        ([[1, 4, 4], [1, 4, 4]], "wrist"),
    ],
)
def test_external_layout_rejects_image_span_grid_mismatch(
    bad_grids: list[list[int]],
    message: str,
) -> None:
    input_ids = torch.tensor(
        [[0, 99, 99, 99, 99, 2, 99, 99, 3]],
        dtype=torch.long,
    )
    with pytest.raises(ValueError, match=message):
        build_external_visual_token_layout(
            input_ids,
            torch.tensor(bad_grids, dtype=torch.long),
            image_token_id=IMAGE_TOKEN_ID,
            merge_size=2,
        )


def test_linear_probe_reads_continuous_uv_from_gt_selected_token() -> None:
    input_ids = torch.tensor(
        [[0, 99, 99, 99, 99, 2, 99, 99, 3]],
        dtype=torch.long,
    )
    layout = build_external_visual_token_layout(
        input_ids,
        torch.tensor([[1, 4, 4], [1, 2, 4]], dtype=torch.long),
        image_token_id=IMAGE_TOKEN_ID,
        merge_size=2,
    )
    tokens = torch.zeros(1, input_ids.shape[1], 2, dtype=torch.bfloat16)
    tokens[0, 1:5, 0] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    probe = LinearVisualTokenPositionProbe(hidden_size=2)
    with torch.no_grad():
        probe.position_decoder.weight.copy_(
            torch.tensor([[0.4, 0.0], [0.4, 0.0]])
        )
        probe.position_decoder.bias.zero_()

    output = probe(tokens, layout, torch.tensor([[0.7, 0.7]]))

    assert output.target_token_index.tolist() == [4]
    torch.testing.assert_close(
        output.predicted_uv,
        torch.tensor([[0.7, 0.7]]),
        atol=1e-6,
        rtol=0.0,
    )
    assert (
        spatial_probe_loss(
            output.predicted_uv,
            torch.tensor([[0.7, 0.7]]),
        ).item()
        < 1e-6
    )


def test_layer12_and_layer24_probes_have_equal_but_independent_initialization() -> None:
    probes = build_matched_linear_probes(seed=42, hidden_size=8)
    for first, second in zip(
        probes["layer12"].parameters(),
        probes["layer24"].parameters(),
        strict=True,
    ):
        assert torch.equal(first, second)
        assert first.data_ptr() != second.data_ptr()


def test_feature_cache_runs_encoder_once_and_keeps_only_gt_token() -> None:
    class FakeTokenizer:
        @staticmethod
        def convert_tokens_to_ids(token: str) -> int:
            assert token == "<|image_pad|>"
            return IMAGE_TOKEN_ID

    class FakeEncoder:
        def __init__(self) -> None:
            self.calls = 0

        def train(self, mode: bool):
            assert mode is False
            return self

        def __call__(self, model_inputs):
            self.calls += 1
            shape = (*model_inputs["input_ids"].shape, 2)
            layer12 = torch.zeros(shape)
            layer24 = torch.zeros(shape)
            layer12[0, 4] = torch.tensor([12.0, 13.0])
            layer24[0, 4] = torch.tensor([24.0, 25.0])
            return SimpleNamespace(
                layer12_tokens=layer12,
                layer24_tokens=layer24,
            )

    raw_batch = {
        "qwen_inputs": {
            "input_ids": torch.tensor(
                [[0, 99, 99, 99, 99, 2, 99, 99, 3]],
                dtype=torch.long,
            ),
            "image_grid_thw": torch.tensor(
                [[1, 4, 4], [1, 2, 4]],
                dtype=torch.long,
            ),
        },
        "target_uv_external": torch.tensor([[0.7, 0.7]]),
        "image_size_external": torch.tensor([[100, 100]]),
        "intrinsic_external": torch.from_numpy(_intrinsic_100px()[None]),
        "world_from_external": torch.from_numpy(IDENTITY_4X4[None]),
        "object_position_m": torch.tensor([[0.2, -0.1, -2.0]]),
        "trajectory_id": ["episode-000"],
        "timestep": torch.tensor([0]),
    }
    processor = SimpleNamespace(
        processor=SimpleNamespace(tokenizer=FakeTokenizer()),
        config=SimpleNamespace(merge_size=2),
    )
    encoder = FakeEncoder()

    cached = _extract_frozen_features(
        [raw_batch],
        encoder,
        processor=processor,
        device=torch.device("cpu"),
    )

    assert encoder.calls == 1
    torch.testing.assert_close(
        cached.layer12_features,
        torch.tensor([[12.0, 13.0]]),
    )
    torch.testing.assert_close(
        cached.layer24_features,
        torch.tensor([[24.0, 25.0]]),
    )
    torch.testing.assert_close(cached.nearest_token_uv, torch.tensor([[0.75, 0.75]]))


def test_perfect_prediction_has_zero_image_token_and_world_error() -> None:
    intrinsic = _intrinsic_100px()
    object_position = np.asarray([[0.2, -0.1, -2.0]], dtype=np.float32)
    projection = project_world_point_to_gl_camera(
        object_position[0],
        intrinsic.ravel(),
        IDENTITY_4X4.ravel(),
        image_height=100,
        image_width=100,
    )
    target_uv = projection.normalized_uv[None]

    metrics = summarize_spatial_predictions(
        predicted_uv=target_uv.copy(),
        target_uv=target_uv,
        image_sizes_hw=np.asarray([[100, 100]], dtype=np.int64),
        grid_shapes_hw=np.asarray([[8, 8]], dtype=np.int64),
        intrinsics=intrinsic[None],
        world_from_cameras=IDENTITY_4X4[None],
        object_positions_m=object_position,
    )

    assert metrics["median_pixel_error"] == pytest.approx(0.0)
    assert metrics["median_visual_token_error"] == pytest.approx(0.0)
    assert metrics["median_world_xy_error_m"] == pytest.approx(0.0, abs=1e-6)
    assert metrics["valid_world_unprojection_rate"] == pytest.approx(1.0)
    assert metrics["within_1_visual_token_rate"] == pytest.approx(1.0)


def test_screening_rejects_incomplete_world_unprojection() -> None:
    layer12 = {
        "position": {
            "median_world_xy_error_m": 0.01,
            "p90_world_xy_error_m": 0.02,
            "invalid_world_unprojections": 1,
        }
    }
    layer24 = {
        "position": {
            "median_world_xy_error_m": 0.02,
            "p90_world_xy_error_m": 0.03,
            "invalid_world_unprojections": 0,
        }
    }

    token_center = {
        "median_world_xy_error_m": 0.012,
        "invalid_world_unprojections": 0,
    }
    decision = interpret_layer12_probe(layer12, layer24, token_center)

    assert decision["layer12_has_reach_usable_position"] is False
    assert decision["layer12_clearly_better_than_layer24"] is False


def test_dataset_keeps_only_visible_valid_reach_windows(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    meta = meta_factory(
        randomization={
            "seed": 7,
            "event_state_contract_version": EVENT_STATE_CONTRACT_VERSION,
        }
    )
    object_positions = np.asarray(
        [
            [0.0, 0.0, -2.0],
            [0.0, 0.0, -2.0],
            [1.0, 0.0, -2.0],
            [0.0, 0.0, -2.0],
            [0.0, 0.0, -2.0],
        ],
        dtype=np.float32,
    )
    external_valid = np.ones(5, dtype=np.bool_)
    external_valid[3] = False
    arrays = arrays_factory(
        skill_id=np.asarray([0, 1, 0, 0, 3], dtype=np.int16),
        external_valid=external_valid,
        **_event_state_arrays(object_positions),
    )
    write_dataset(meta, arrays)

    dataset = QwenExternalSpatialProbeDataset(
        tmp_path,
        [meta],
        RobotSpec(),
    )

    assert dataset.index == [(0, 0)]
    assert dataset.rejected_missing_geometry == 0
    assert dataset.rejected_out_of_view == 1
    sample = dataset[0]
    assert sample["skill_id"] == 0
    assert sample["timestep"] == 0
    np.testing.assert_array_equal(sample["object_position_m"], object_positions[0])


def test_dataset_does_not_infer_missing_object_geometry(
    tmp_path,
    meta_factory,
    arrays_factory,
    write_dataset,
) -> None:
    meta = meta_factory()
    write_dataset(
        meta,
        arrays_factory(skill_id=np.zeros(5, dtype=np.int16)),
    )

    with pytest.raises(ValueError, match="没有可见方块"):
        QwenExternalSpatialProbeDataset(
            tmp_path,
            [meta],
            RobotSpec(),
        )
