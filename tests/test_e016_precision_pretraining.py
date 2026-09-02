from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
import torch

from robot_vla.precision.data import PrecisionLabelArrays
from robot_vla.precision.e016_pretraining import (
    E016_CORRECTED_LABEL_ARRAYS,
    E016_P0_VERSION,
    E016CorrectedLabelMeta,
    derive_e016_corrected_arrays,
    load_e016_p0_config,
    read_e016_corrected_labels,
    run_e016_loss_contract_probe,
    select_stratified_overfit_indices,
    validate_e016_corrected_arrays,
)
from robot_vla.precision.losses import (
    PrecisionSupervision,
    build_gaussian_heatmaps,
    precision_unet_loss,
)
from robot_vla.precision.model import PrecisionThreeHeadUNet, PrecisionUNetConfig


def _source_labels() -> PrecisionLabelArrays:
    steps = 4
    height = width = 7
    object_mask = np.zeros((steps, height, width), dtype=np.bool_)
    goal_mask = np.zeros_like(object_mask)
    # t0: projected center 命中 goal。
    goal_mask[0, 3, 3] = True
    # t1: goal 仍有边缘像素，但 projected center 被 object 遮挡。
    goal_mask[1, 0, 0] = True
    object_mask[1, 3, 3] = True
    # t2: goal 仍有边缘像素，projected center 命中背景。
    goal_mask[2, 0, 0] = True
    # t3: projection invalid。
    object_mask[:, 1, 1] = True
    normalized_uv = np.zeros((steps, 2, 2), dtype=np.float32)
    normalized_uv[:3] = np.asarray((0.5, 0.5), dtype=np.float32)
    projection = np.ones((steps, 2), dtype=np.bool_)
    projection[3] = False
    visible = np.column_stack(
        (
            object_mask.reshape(steps, -1).any(axis=1),
            goal_mask.reshape(steps, -1).any(axis=1),
        )
    )
    return PrecisionLabelArrays(
        source_timestep=np.arange(steps, dtype=np.int64),
        timestamp_s=np.arange(steps, dtype=np.float64) * 0.05,
        object_mask=object_mask,
        goal_mask=goal_mask,
        normalized_uv=normalized_uv,
        keypoint_visible=visible,
        keypoint_projection_valid=projection,
        object_position_base_m=np.zeros((steps, 3), dtype=np.float32),
        goal_position_base_m=np.ones((steps, 3), dtype=np.float32),
    )


def _corrected_meta() -> E016CorrectedLabelMeta:
    return E016CorrectedLabelMeta(
        trajectory_id="trajectory-a",
        file="labels/trajectory-a.npz",
        split="train",
        scene_id="scene-a",
        num_steps=4,
        source_label_sha256="a" * 64,
    )


def test_e016_config_freezes_train_val_only_and_disposable_checkpoint() -> None:
    config = load_e016_p0_config("configs/e016_p0_precision_observability_v1.json")

    assert config.version == E016_P0_VERSION
    assert config.source.allowed_splits == ("train", "val")
    assert config.source.excluded_splits == ("test",)
    assert config.stratified_overfit.sample_count == 128
    assert config.stratified_overfit.optimizer_steps == 600
    assert config.full_preflight.epochs == 3
    assert config.loss.mask_dice_weight == 1.0
    assert config.loss.visibility_weight == 1.0
    assert config.execution.persist_checkpoint is False
    assert config.execution.actuation_allowed is False
    assert len(config.sha256) == 64


def test_corrected_observability_separates_mask_any_from_center_evidence() -> None:
    arrays = derive_e016_corrected_arrays(_source_labels())
    validate_e016_corrected_arrays(arrays, _corrected_meta())

    assert arrays.goal_exists.tolist() == [True, True, True, True]
    assert arrays.goal_projection_valid.tolist() == [True, True, True, False]
    assert arrays.goal_observable.tolist() == [True, False, False, False]
    assert arrays.goal_localization_valid.tolist() == [True, False, False, False]
    assert arrays.legacy_goal_visible.tolist() == [True, True, True, False]
    assert [arrays.occlusion_type(index) for index in range(4)] == [
        "observable",
        "object_occlusion",
        "other_occlusion_or_background",
        "projection_invalid",
    ]


def test_corrected_sidecar_reader_is_strict_and_rejects_contract_drift(tmp_path) -> None:
    arrays = derive_e016_corrected_arrays(_source_labels())
    meta = _corrected_meta()
    target = tmp_path / meta.file
    target.parent.mkdir(parents=True)
    np.savez_compressed(
        target,
        **{name: getattr(arrays, name) for name in E016_CORRECTED_LABEL_ARRAYS},
    )

    loaded = read_e016_corrected_labels(tmp_path, meta)
    np.testing.assert_array_equal(loaded.goal_observable, arrays.goal_observable)

    invalid = arrays.goal_localization_valid.copy()
    invalid[1] = True
    with pytest.raises(ValueError, match="observable gate"):
        validate_e016_corrected_arrays(
            replace(arrays, goal_localization_valid=invalid),
            meta,
        )


def test_visibility_target_is_explicit_and_localization_must_be_observable() -> None:
    config = PrecisionUNetConfig(
        encoder_channels=(8, 16),
        structured_state_dim=6,
        state_hidden_size=8,
        head_hidden_size=16,
    )
    model = PrecisionThreeHeadUNet(config)
    output = model(
        torch.rand(1, 3, 8, 8),
        torch.zeros(1, 6),
        torch.zeros(1, config.motion_spec.motion_dim),
    )
    valid = torch.tensor([[False, False]])
    observable = torch.tensor([[False, True]])
    target_uv = torch.zeros((1, 2, 2))
    supervision = PrecisionSupervision(
        heatmap_targets=build_gaussian_heatmaps(target_uv, valid, (8, 8)),
        mask_targets=torch.zeros_like(output.mask_logits),
        normalized_uv_targets=target_uv,
        keypoint_valid=valid,
        motion_residual_targets=torch.zeros_like(output.motion_residual),
        motion_valid=torch.zeros_like(output.motion_residual, dtype=torch.bool),
        projection_valid=torch.zeros(1, dtype=torch.bool),
        keypoint_observable=observable,
    )

    loss = precision_unet_loss(output, supervision)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(
        output.visibility_logits,
        observable.float(),
    )
    torch.testing.assert_close(loss.visibility_loss, expected)

    with pytest.raises(ValueError, match="必须同时具有"):
        precision_unet_loss(
            output,
            replace(
                supervision,
                keypoint_valid=torch.tensor([[False, True]]),
                keypoint_observable=torch.tensor([[False, False]]),
                heatmap_targets=build_gaussian_heatmaps(
                    torch.full((1, 2, 2), 0.5),
                    torch.tensor([[False, True]]),
                    (8, 8),
                ),
                normalized_uv_targets=torch.full((1, 2, 2), 0.5),
            ),
        )


def test_all_negative_localization_contract_is_finite_and_zero_gradient() -> None:
    receipt = run_e016_loss_contract_probe(torch.device("cpu"))

    assert receipt["passed"]
    assert receipt["all_negative_batch_finite"]
    assert set(receipt["localization_losses"].values()) == {0.0}
    assert set(receipt["localization_output_gradient_abs_max"].values()) == {0.0}


class _SamplingDataset:
    split = "train"

    def __init__(self) -> None:
        names = (
            ["observable"] * 70
            + ["object_occlusion"] * 30
            + ["other_occlusion_or_background"] * 30
            + ["projection_invalid"] * 20
        )
        self.names = names

    def __len__(self) -> int:
        return len(self.names)

    def sampling_metadata(self, index: int) -> dict[str, object]:
        return {
            "dataset_index": index,
            "trajectory_id": f"trajectory-{index // 10}",
            "timestep": index,
            "split": "train",
            "goal_observable": self.names[index] == "observable",
            "legacy_goal_visible": True,
            "occlusion_type": self.names[index],
        }


def test_stratified_subset_is_deterministic_and_has_frozen_counts() -> None:
    config = load_e016_p0_config("configs/e016_p0_precision_observability_v1.json")
    dataset = _SamplingDataset()

    first, rows = select_stratified_overfit_indices(dataset, config.stratified_overfit)
    second, _ = select_stratified_overfit_indices(dataset, config.stratified_overfit)

    assert first == second
    assert len(first) == 128
    assert len(set(first)) == 128
    assert {
        name: sum(row["stratum"] == name for row in rows)
        for name in config.stratified_overfit.strata_counts
    } == config.stratified_overfit.strata_counts
