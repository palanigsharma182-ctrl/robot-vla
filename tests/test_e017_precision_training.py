from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from robot_vla.precision.e016_pretraining import E016P0Metrics
from robot_vla.precision.e017_training import (
    E017_P0_VERSION,
    e017_observability_loss,
    e017_trainable_output_rows,
    load_e017_p0_config,
    select_e017_p0_checkpoint_epoch,
)
from robot_vla.precision.model import PrecisionThreeHeadUNet

ROOT = Path(__file__).resolve().parents[1]


def _metrics(*, false_positive: int, recall_true_positive: int) -> E016P0Metrics:
    false_negative = 100 - recall_true_positive
    true_negative = 100 - false_positive
    precision = recall_true_positive / (recall_true_positive + false_positive)
    recall = recall_true_positive / 100
    return E016P0Metrics(
        sample_count=200,
        object_localization_valid_count=200,
        goal_observable_count=100,
        goal_unobservable_count=100,
        mean_loss=1.0,
        mean_heatmap_loss=1.0,
        mean_mask_loss=1.0,
        mean_coordinate_loss=1.0,
        mean_uncertainty_loss=1.0,
        mean_visibility_loss=1.0,
        mean_projection_loss=1.0,
        object_normalized_uv_mae=0.01,
        goal_observable_normalized_uv_mae=0.01,
        goal_observable_pixel_error_p50=1.0,
        goal_observable_pixel_error_p90=2.0,
        object_mask_iou=0.9,
        goal_mask_iou=0.9,
        goal_visibility_true_positive=recall_true_positive,
        goal_visibility_false_positive=false_positive,
        goal_visibility_true_negative=true_negative,
        goal_visibility_false_negative=false_negative,
        goal_visibility_precision=precision,
        goal_visibility_recall=recall,
        goal_visibility_f1=2 * precision * recall / (precision + recall),
        goal_unobservable_false_positive_rate=false_positive / 100,
        projection_accuracy=0.99,
    )


def test_e017_config_freezes_parent_and_no_test_execution() -> None:
    config = load_e017_p0_config(ROOT / "configs/e017_p0_conservative_observability_v1.json")

    assert config.version == E017_P0_VERSION
    assert config.parent.selected_epoch == 12
    assert config.training.goal_negative_weight == 4.0
    assert config.execution.required_gpu == "NVIDIA GeForce RTX 5080"
    assert config.execution.test_split_read_allowed is False
    assert config.execution.actuation_allowed is False


def test_e017_only_targets_goal_visibility_and_projection_rows() -> None:
    model = PrecisionThreeHeadUNet()

    goal_row, projection_row = e017_trainable_output_rows(model)

    assert (goal_row, projection_row) == (9, 10)


def test_e017_negative_goal_examples_receive_frozen_weight() -> None:
    goal_logits = torch.tensor([0.0, 0.0], requires_grad=True)
    goal_targets = torch.tensor([True, False])
    projection_logits = torch.tensor([0.0, 0.0], requires_grad=True)
    projection_targets = torch.tensor([True, False])

    total, goal, projection = e017_observability_loss(
        goal_logits,
        goal_targets,
        projection_logits,
        projection_targets,
        goal_negative_weight=4.0,
    )
    total.backward()

    assert float(goal.detach()) == pytest.approx(2.5 * 0.69314718056)
    assert float(projection.detach()) == pytest.approx(0.69314718056)
    assert goal_logits.grad is not None
    assert abs(float(goal_logits.grad[1])) == pytest.approx(4.0 * abs(float(goal_logits.grad[0])))


def test_e017_selection_requires_strict_parent_improvement() -> None:
    config = load_e017_p0_config(
        ROOT / "configs/e017_p0_conservative_observability_v1.json"
    ).validation_selection
    parent = _metrics(false_positive=1, recall_true_positive=92)
    same = replace(parent)
    lower_recall = _metrics(false_positive=0, recall_true_positive=89)
    improved_late = _metrics(false_positive=0, recall_true_positive=93)
    improved_early = _metrics(false_positive=0, recall_true_positive=94)

    selected = select_e017_p0_checkpoint_epoch(
        [(1, same), (2, lower_recall), (4, improved_late), (3, improved_early)],
        parent,
        config,
    )

    assert selected == 3
