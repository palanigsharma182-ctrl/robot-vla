from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from robot_vla.precision.losses import (
    PrecisionSupervision,
    build_gaussian_heatmaps,
    precision_unet_loss,
)
from robot_vla.precision.model import (
    PrecisionThreeHeadUNet,
    PrecisionUNetConfig,
    decode_keypoints,
)
from robot_vla.precision.provider import (
    TorchPrecisionFramePredictor,
    TorchPrecisionFramePredictorConfig,
)


def _config() -> PrecisionUNetConfig:
    return PrecisionUNetConfig(
        encoder_channels=(8, 16, 32),
        structured_state_dim=6,
        state_hidden_size=8,
        head_hidden_size=16,
    )


def _inputs(config: PrecisionUNetConfig):
    image = torch.rand(2, config.input_channels, 32, 40)
    state = torch.randn(2, config.structured_state_dim)
    geometry = torch.randn(2, config.motion_spec.motion_dim) * 1e-3
    return image, state, geometry


def test_three_head_unet_preserves_dense_resolution_and_zero_initializes_residual() -> None:
    config = _config()
    model = PrecisionThreeHeadUNet(config)

    output = model(*_inputs(config))

    assert output.heatmap_logits.shape == (2, config.keypoint_count, 32, 40)
    assert output.mask_logits.shape == (2, config.mask_count, 32, 40)
    assert output.subpixel_offsets.shape == (2, config.keypoint_count, 2, 32, 40)
    assert output.motion_residual.shape == (2, config.motion_spec.motion_dim)
    assert output.keypoint_log_variance.shape == (2, config.keypoint_count, 2)
    assert output.motion_log_variance.shape == (2, config.motion_spec.motion_dim)
    assert output.visibility_logits.shape == (2, config.keypoint_count)
    assert output.projection_validity_logit.shape == (2,)
    torch.testing.assert_close(output.motion_residual, torch.zeros_like(output.motion_residual))
    assert torch.all(torch.abs(output.subpixel_offsets) <= 0.5)
    decoded = output.decode_for_control()
    assert decoded.keypoints.normalized_uv.shape == (2, config.keypoint_count, 2)
    assert decoded.keypoint_sigma_px.shape == (2, config.keypoint_count, 2)
    assert decoded.motion_sigma.shape == (2, config.motion_spec.motion_dim)
    assert torch.all(decoded.keypoint_sigma_px > 0.0)
    assert torch.all(decoded.motion_sigma > 0.0)


def test_localization_head_is_image_only_while_state_changes_other_heads() -> None:
    config = _config()
    model = PrecisionThreeHeadUNet(config).eval()
    image, state, geometry = _inputs(config)

    with torch.no_grad():
        first = model(image, state, geometry)
        second = model(image, state + 10.0, geometry - 0.01)

    torch.testing.assert_close(first.heatmap_logits, second.heatmap_logits)
    torch.testing.assert_close(first.mask_logits, second.mask_logits)
    assert not torch.equal(first.motion_log_variance, second.motion_log_variance)


def test_motion_head_is_hard_bounded_by_metric_residual_limits() -> None:
    config = _config()
    model = PrecisionThreeHeadUNet(config)
    with torch.no_grad():
        model.motion_head[-1].bias.fill_(100.0)

    output = model(*_inputs(config))
    limits = torch.tensor(config.motion_spec.residual_limits)

    assert torch.all(output.motion_residual <= limits + 1e-8)
    assert torch.all(output.motion_residual >= -limits - 1e-8)


def test_soft_argmax_uses_subpixel_offset_and_reports_low_entropy_for_sharp_peak() -> None:
    logits = torch.full((1, 1, 4, 5), -20.0)
    logits[0, 0, 2, 3] = 20.0
    offsets = torch.zeros((1, 1, 2, 4, 5))
    offsets[0, 0, 0, 2, 3] = 0.25
    offsets[0, 0, 1, 2, 3] = -0.25

    decoded = decode_keypoints(logits, offsets)

    torch.testing.assert_close(decoded.pixel_uv[0, 0], torch.tensor((3.25, 1.75)))
    torch.testing.assert_close(
        decoded.normalized_uv[0, 0],
        torch.tensor(((3.25 + 0.5) / 5.0, (1.75 + 0.5) / 4.0)),
    )
    assert decoded.peak_probability.item() > 0.999
    assert decoded.normalized_entropy.item() < 1e-5


def test_precision_loss_backpropagates_all_three_heads() -> None:
    config = _config()
    model = PrecisionThreeHeadUNet(config)
    image, state, geometry = _inputs(config)
    output = model(image, state, geometry)
    target_uv = torch.tensor(
        [
            [[0.30, 0.40], [0.70, 0.60]],
            [[0.35, 0.45], [0.65, 0.55]],
        ],
        dtype=torch.float32,
    )
    keypoint_valid = torch.tensor([[True, True], [True, False]])
    heatmaps = build_gaussian_heatmaps(target_uv, keypoint_valid, (32, 40))
    motion_target = torch.full_like(output.motion_residual, 0.1e-3)
    supervision = PrecisionSupervision(
        heatmap_targets=heatmaps,
        mask_targets=torch.zeros_like(output.mask_logits),
        normalized_uv_targets=target_uv,
        keypoint_valid=keypoint_valid,
        motion_residual_targets=motion_target,
        motion_valid=torch.ones_like(output.motion_residual, dtype=torch.bool),
        projection_valid=torch.tensor((True, False)),
    )

    loss = precision_unet_loss(output, supervision)
    loss.loss.backward()

    assert torch.isfinite(loss.loss)
    assert model.localization_head.heatmap.weight.grad is not None
    assert model.motion_head[-1].weight.grad is not None
    assert model.uncertainty_head[-1].weight.grad is not None


def test_model_rejects_image_or_state_contract_drift() -> None:
    config = _config()
    model = PrecisionThreeHeadUNet(config)
    image, state, geometry = _inputs(config)

    with pytest.raises(ValueError, match="image 必须"):
        model(image[:, :2], state, geometry)
    with pytest.raises(ValueError, match="structured_state"):
        model(image, state[:, :5], geometry)


def test_torch_frame_predictor_freezes_model_and_decodes_raw_rgb() -> None:
    config = PrecisionUNetConfig(
        encoder_channels=(8, 16, 32),
        state_hidden_size=8,
        head_hidden_size=16,
    )
    model = PrecisionThreeHeadUNet(config)
    predictor = TorchPrecisionFramePredictor(
        model,
        checkpoint_sha256="a" * 64,
        config=TorchPrecisionFramePredictorConfig(device="cpu"),
    )

    prediction = predictor.predict(
        np.full((32, 40, 3), 127, dtype=np.uint8),
        np.zeros(config.structured_state_dim, dtype=np.float32),
        np.zeros(config.motion_spec.motion_dim, dtype=np.float32),
    )

    assert prediction.keypoints.normalized_uv.shape == (1, config.keypoint_count, 2)
    assert predictor.identity.keypoint_names == config.keypoint_names
    assert len(predictor.identity.parameter_state_sha256) == 64
    assert not model.training
    assert all(not parameter.requires_grad for parameter in model.parameters())
    predictor.verify_identity()

    with torch.no_grad():
        model.motion_head[-1].bias.add_(1.0)
    with pytest.raises(RuntimeError, match="parameter state"):
        predictor.verify_identity()
