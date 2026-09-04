from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_vla.diagnostics.qwen_spatial_probe import project_world_point_to_gl_camera
from robot_vla.diagnostics.v2_online_geometry_probe import (
    OnlineVisualTargetProbe,
    _complete_v2_history_source_indices,
    build_selected_visual_token_layout,
    compact_selected_visual_tokens,
    nearest_visual_token_indices,
    online_geometry_probe_loss,
    summarize_online_geometry_predictions,
)

IMAGE_TOKEN_ID = 99


def _eight_image_prompt() -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
    token_ids = [0]
    runs: list[list[int]] = []
    for image_index in range(8):
        run = list(range(len(token_ids), len(token_ids) + 4))
        runs.append(run)
        token_ids.extend([IMAGE_TOKEN_ID] * 4)
        token_ids.append(image_index + 1)
    grids = torch.tensor([[1, 2, 2]] * 8, dtype=torch.long)
    return torch.tensor([token_ids], dtype=torch.long), grids, runs


def _intrinsic_100px() -> np.ndarray:
    return np.asarray(
        [[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]],
        dtype=np.float32,
    )


def test_online_probe_requires_four_contiguous_fully_valid_v2_steps() -> None:
    spec = SimpleNamespace(control_hz=20.0)
    arrays = SimpleNamespace(
        timestamp_action=np.arange(6, dtype=np.float64) / spec.control_hz,
        observation_v2_valid=np.ones(6, dtype=np.bool_),
    )

    assert _complete_v2_history_source_indices(arrays, 2, spec) is None
    np.testing.assert_array_equal(
        _complete_v2_history_source_indices(arrays, 3, spec),
        np.asarray((0, 1, 2, 3), dtype=np.int64),
    )

    arrays.observation_v2_valid[1] = False
    assert _complete_v2_history_source_indices(arrays, 3, spec) is None


def test_v2_layout_selects_current_external_from_eight_image_prompt() -> None:
    input_ids, grids, runs = _eight_image_prompt()

    layout = build_selected_visual_token_layout(
        input_ids,
        grids,
        image_token_id=IMAGE_TOKEN_ID,
        merge_size=1,
    )

    assert torch.nonzero(layout.mask[0]).flatten().tolist() == runs[6]
    torch.testing.assert_close(
        layout.normalized_centers[0, runs[6]],
        torch.tensor([[0.25, 0.25], [0.75, 0.25], [0.25, 0.75], [0.75, 0.75]]),
    )
    assert layout.grid_shapes.tolist() == [[2, 2]]

    sequence_tokens = torch.arange(
        input_ids.shape[1] * 2,
        dtype=torch.float32,
    ).reshape(1, input_ids.shape[1], 2)
    compact, centers = compact_selected_visual_tokens(sequence_tokens, layout)
    torch.testing.assert_close(compact[0], sequence_tokens[0, runs[6]])
    torch.testing.assert_close(centers[0], layout.normalized_centers[0, runs[6]])


def test_online_probe_selects_tokens_without_receiving_gt_at_forward() -> None:
    tokens = torch.tensor([[[-2.0, 0.0], [-1.0, 0.0], [1.0, 0.0], [2.0, 0.0]]])
    centers = torch.tensor([[[0.1, 0.5], [0.3, 0.5], [0.7, 0.5], [0.9, 0.5]]])
    target_uv = torch.tensor([[[0.9, 0.5], [0.1, 0.5]]])
    probe = OnlineVisualTargetProbe(hidden_size=2, target_count=2)
    with torch.no_grad():
        probe.selector.weight.copy_(torch.tensor([[1.0, 0.0], [-1.0, 0.0]]))
        probe.selector.bias.zero_()

    output = probe(tokens)

    assert output.selected_token_indices.tolist() == [[3, 0]]
    nearest = nearest_visual_token_indices(centers, target_uv)
    assert nearest.tolist() == [[3, 0]]
    loss = online_geometry_probe_loss(
        output,
        target_uv,
        centers,
        selector_loss_weight=0.05,
    )
    assert torch.isfinite(loss.loss)
    assert loss.target_token_indices.tolist() == [[3, 0]]
    loss.loss.backward()
    assert probe.selector.weight.grad is not None


def test_perfect_online_geometry_prediction_passes_both_screening_levels() -> None:
    intrinsic = _intrinsic_100px()
    transform = np.eye(4, dtype=np.float32)
    positions = np.asarray(
        [[[0.2, -0.1, -2.0], [-0.2, 0.1, -2.0]]],
        dtype=np.float32,
    )
    target_uv = np.stack(
        [
            project_world_point_to_gl_camera(
                point,
                intrinsic.ravel(),
                transform.ravel(),
                100,
                100,
            ).normalized_uv
            for point in positions[0]
        ]
    )[None]

    summary = summarize_online_geometry_predictions(
        predicted_uv=target_uv.copy(),
        target_uv=target_uv,
        selected_token_indices=np.asarray([[3, 0]], dtype=np.int64),
        nearest_token_indices=np.asarray([[3, 0]], dtype=np.int64),
        image_sizes_hw=np.asarray([[100, 100]], dtype=np.int64),
        grid_shapes_hw=np.asarray([[8, 8]], dtype=np.int64),
        intrinsics=intrinsic[None],
        world_from_cameras=transform[None],
        target_positions_world_m=positions,
    )

    assert summary["coarse_reach_screen_passed"] is True
    assert summary["deployable_precision_candidate"] is True
    assert summary["by_target"]["object"]["p90_world_xy_error_m"] == pytest.approx(
        0.0,
        abs=1e-6,
    )
    assert summary["object_to_goal_relative_xy"]["p90_error_m"] == pytest.approx(
        0.0,
        abs=1e-6,
    )


def test_invalid_world_unprojection_fails_closed_without_metric_coercion() -> None:
    target_uv = np.asarray([[[0.5, 0.5], [0.6, 0.5]]], dtype=np.float32)
    # Identity OpenGL camera looks toward -Z；正 Z 平面在射线后方，必须计为 invalid。
    positions = np.asarray(
        [[[0.0, 0.0, 1.0], [0.1, 0.0, 1.0]]],
        dtype=np.float32,
    )

    summary = summarize_online_geometry_predictions(
        predicted_uv=target_uv.copy(),
        target_uv=target_uv,
        selected_token_indices=np.asarray([[0, 1]], dtype=np.int64),
        nearest_token_indices=np.asarray([[0, 1]], dtype=np.int64),
        image_sizes_hw=np.asarray([[100, 100]], dtype=np.int64),
        grid_shapes_hw=np.asarray([[8, 8]], dtype=np.int64),
        intrinsics=_intrinsic_100px()[None],
        world_from_cameras=np.eye(4, dtype=np.float32)[None],
        target_positions_world_m=positions,
    )

    assert summary["by_target"]["object"]["invalid_world_unprojections"] == 1
    assert summary["by_target"]["object"]["p90_world_xy_error_m"] is None
    assert summary["object_to_goal_relative_xy"]["invalid_samples"] == 1
    assert summary["coarse_reach_screen_passed"] is False
    assert summary["deployable_precision_candidate"] is False
