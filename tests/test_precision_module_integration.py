"""模型证据接入校准的回归；不消费真实 checkpoint 或评估数据。"""

from pathlib import Path
import runpy
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from robot_vla.precision.calibrated_front_provider import (
    ScalarCovarianceCalibration, build_calibrated_object_evidence_from_prediction,
)
from robot_vla.precision.model import PrecisionThreeHeadUNet, PrecisionUNetConfig
from robot_vla.precision.provider import TorchPrecisionFramePredictor, TorchPrecisionFramePredictorConfig


def test_model_evidence_is_opt_in_and_preserves_default_prediction():
    config = PrecisionUNetConfig(encoder_channels=(8, 16), state_hidden_size=8, head_hidden_size=16)
    predictor = TorchPrecisionFramePredictor(PrecisionThreeHeadUNet(config), checkpoint_sha256="a"*64,
                                            config=TorchPrecisionFramePredictorConfig(device="cpu"))
    inputs = (np.full((32, 32, 3), 127, dtype=np.uint8),
              np.zeros(config.structured_state_dim, dtype=np.float32), np.zeros(4, dtype=np.float32))
    default = predictor.predict(*inputs)
    detailed = predictor.predict(*inputs, include_mask_probability=True)
    assert default.mask_probability is None
    assert detailed.mask_probability.shape == (1, 2, 32, 32)
    assert not detailed.mask_probability.requires_grad
    for name in ("motion_residual", "visibility_probability", "projection_validity_probability",
                 "keypoint_sigma_px", "motion_sigma"):
        torch.testing.assert_close(getattr(default, name), getattr(detailed, name), rtol=0, atol=0)
    torch.testing.assert_close(default.keypoints.normalized_uv, detailed.keypoints.normalized_uv, rtol=0, atol=0)


def test_prediction_mask_sampling_tracks_uv_and_rejects_invalid_probability():
    prediction = SimpleNamespace(
        keypoints=SimpleNamespace(normalized_uv=np.array([[[0.25, 0.25], [0.5, 0.5]]]),
                                  peak_probability=np.array([[0.9, 0.9]]), normalized_entropy=np.array([[0.1, 0.1]])),
        visibility_probability=np.array([[0.95, 0.95]]), projection_validity_probability=np.array([0.95]),
        keypoint_sigma_px=np.full((1, 2, 2), 0.1),
        mask_probability=np.array([[[[0., 1.], [0., 1.]], [[0., 0.], [0., 0.]]]]),
    )
    kwargs = dict(keypoint_names=("object_center", "goal_center"), mask_names=("object", "goal"),
                  image_size_hw=(2, 2), timestamp_s=0., geometry_valid=True,
                  calibration=ScalarCovarianceCalibration(4., 40, 39, 4.*5.991, *("a"*64,)*4))
    first, _ = build_calibrated_object_evidence_from_prediction(prediction, **kwargs)
    prediction.keypoints.normalized_uv[0, 0] = [0.5, 0.5]
    second, _ = build_calibrated_object_evidence_from_prediction(prediction, **kwargs)
    assert first.object_mask_probability == pytest.approx(0.)
    assert second.object_mask_probability == pytest.approx(0.5)
    for value in (np.nan, -0.1, 1.1):
        prediction.mask_probability[0, 0, 0, 0] = value
        with pytest.raises(ValueError):
            build_calibrated_object_evidence_from_prediction(prediction, **kwargs)


def test_prediction_mask_channels_and_sigma_are_used_without_constant_substitution():
    calibration = ScalarCovarianceCalibration(4., 40, 39, 4.*5.991, *("a"*64,)*4)
    prediction = SimpleNamespace(
        keypoints=SimpleNamespace(normalized_uv=np.array([[[0.5, 0.5], [0.2, 0.2]]]),
                                  peak_probability=np.array([[0.9, 0.9]]), normalized_entropy=np.array([[0.1, 0.1]])),
        visibility_probability=np.array([[0.95, 0.95]]), projection_validity_probability=np.array([0.95]),
        keypoint_sigma_px=np.array([[[0.1, 0.2], [0.1, 0.2]]]),
        mask_probability=np.stack([np.full((4, 4), 0.2), np.full((4, 4), 0.8)])[None],
    )
    kwargs = dict(keypoint_names=("object_center", "goal_center"), mask_names=("goal", "object"),
                  image_size_hw=(4, 4), timestamp_s=0., calibration=calibration, geometry_valid=True)
    evidence, sigma = build_calibrated_object_evidence_from_prediction(prediction, **kwargs)
    assert evidence.object_mask_probability == pytest.approx(0.8)
    assert evidence.goal_mask_probability == pytest.approx(0.2)
    assert sigma == pytest.approx([0.2, 0.4])
    prediction.keypoints.normalized_uv = np.repeat(prediction.keypoints.normalized_uv, 2, axis=0)
    with pytest.raises(ValueError, match="normalized_uv"):
        build_calibrated_object_evidence_from_prediction(prediction, **kwargs)
    prediction.keypoints.normalized_uv = prediction.keypoints.normalized_uv[:1]
    prediction.mask_probability = None
    with pytest.raises(ValueError, match="mask_probability"):
        build_calibrated_object_evidence_from_prediction(prediction, **kwargs)


def test_existing_checkpoint_and_front_input_reach_real_frozen_model():
    runner = Path(__file__).resolve().parents[1]/"experiments/precision_module_integration/run.py"
    result = runpy.run_path(str(runner))["run_replay"]()
    assert [r["forward_count"] for r in result["rows"]] == [0, 1, 1]
    assert result["optimizer_steps"] == result["memory_write_count"] == result["actuation_count"] == 0
    assert not result["provider_qualified"]
    for row in result["rows"][1:]:
        assert row["source_camera"] == "base_camera"
        assert row["qualification_only"] and not row["memory_write_eligible"]
        assert row["mask_shape"] == [1, 2, 32, 32]
        assert not row["geometry_valid"]
